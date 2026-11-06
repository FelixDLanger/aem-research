"""SEC Insider Transactions Data Sets: Form 4 purchases and the routine/opportunistic split.

Each quarterly zip holds SUBMISSION (one row per filing: filing date, form type,
issuer CIK and trading symbol), REPORTINGOWNER (who filed and their relationship to
the issuer) and NONDERIV_TRANS (the transactions). The extract keeps Form 4 and 4/A
open-market purchases and sales (codes P and S) with the owner's roles, plus a
symbol-to-CIK bridge from every filing.

Classification follows Cohen, Malloy and Pomorski (2012): an insider who traded in the
same calendar month in each of the three years before the formation year is routine;
one who traded in each of those years but not in a common month is opportunistic;
anyone with a shorter record is unclassified. Classification is per insider and
issuer, made on January 1 from trades filed before it, so it never uses later data.
"""
from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..tickers import canonical_series
from .sec_datasets import INSIDER_URLS, parse_dates, process, quarter_items, read_member, to_int, valid_accession

FORM4 = ("4", "4/A")
OPEN_MARKET = ("P", "S")
_TRUE = {"1", "TRUE", "Y", "YES"}
_FALSE = {"0", "FALSE", "N", "NO"}


def _flag(values) -> pd.Series:
    s = pd.Series(values, dtype="object").astype(str).str.strip().str.upper()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[s.isin(_TRUE)] = True
    out[s.isin(_FALSE)] = False
    return out


def extract_quarter(zf: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    sub = read_member(zf, "SUBMISSION", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                         "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
    sub = sub[valid_accession(sub["accession_number"])].copy()
    sub["filing_date"] = parse_dates(sub["filing_date"])
    sub["issuer_cik"] = to_int(sub["issuercik"])
    raw_sym = sub["issuertradingsymbol"].fillna("").astype(str).str.upper().str.strip()
    raw_sym = raw_sym.str.replace(r"^[A-Z]+:\s*", "", regex=True).str.split(r"[\s,;]+").str[0].fillna("")
    raw_sym = raw_sym.where(~raw_sym.isin(["N/A", "NA", "NONE", "NULL", "-"]), "")
    sub["symbol"] = canonical_series(raw_sym)
    bridge = (sub[sub["symbol"].ne("") & sub["issuer_cik"].notna() & sub["filing_date"].notna()]
              .groupby(["symbol", "issuer_cik"], as_index=False)
              .agg(first_seen=("filing_date", "min"), last_seen=("filing_date", "max"),
                   filings=("accession_number", "size")))

    f4 = sub[sub["document_type"].astype(str).str.strip().isin(FORM4)]
    keep = set(f4["accession_number"])
    tr = read_member(zf, "NONDERIV_TRANS", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                            "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"],
                     chunk_filter=lambda c: c[c["accession_number"].isin(keep)
                                              & c["trans_code"].astype(str).str.strip().isin(OPEN_MARKET)])
    own = read_member(zf, "REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"],
                      chunk_filter=lambda c: c[c["accession_number"].isin(keep)])
    rel = own["rptowner_relationship"].fillna("").astype(str).str.upper()
    owners = pd.DataFrame({"accession_number": own["accession_number"],
                           "owner_cik": to_int(own["rptownercik"]),
                           "is_director": rel.str.contains("DIRECTOR"),
                           "is_officer": rel.str.contains("OFFICER"),
                           "is_ten_pct": rel.str.contains("TENPERCENT") | rel.str.contains("10%")})
    t = tr.merge(f4[["accession_number", "filing_date", "document_type", "issuer_cik", "symbol", "aff10b5one"]],
                 on="accession_number", how="inner").merge(owners, on="accession_number", how="inner")
    tx = pd.DataFrame({
        "accession": t["accession_number"].astype(str),
        "filing_date": t["filing_date"],
        "doc_type": t["document_type"].astype(str).str.strip(),
        "issuer_cik": t["issuer_cik"].astype("Int64"),
        "symbol": t["symbol"].astype(str),
        "owner_cik": t["owner_cik"].astype("Int64"),
        "is_officer": t["is_officer"].astype(bool),
        "is_director": t["is_director"].astype(bool),
        "is_ten_pct": t["is_ten_pct"].astype(bool),
        "trans_date": parse_dates(t["trans_date"]),
        "code": t["trans_code"].astype(str).str.strip(),
        "acq_disp": t["trans_acquired_disp_cd"].astype(str).str.strip().str.upper(),
        "shares": pd.to_numeric(t["trans_shares"], errors="coerce"),
        "price": pd.to_numeric(t["trans_pricepershare"], errors="coerce"),
        "plan_10b5_1": _flag(t["aff10b5one"]),
    })
    tx = tx.dropna(subset=["issuer_cik", "owner_cik", "trans_date", "filing_date"]).reset_index(drop=True)
    return {"tx": tx, "bridge": bridge.reset_index(drop=True)}


def dedupe_amendments(tx: pd.DataFrame) -> pd.DataFrame:
    """Drop 4/A rows that repeat a transaction already filed on the original Form 4."""
    if tx.empty:
        return tx
    key = ["issuer_cik", "owner_cik", "trans_date", "code", "shares", "price"]
    orig = tx[tx["doc_type"] == "4"]
    amend = tx[tx["doc_type"] != "4"]
    dup = amend.merge(orig[key].drop_duplicates(), on=key, how="left", indicator=True)["_merge"].eq("both").to_numpy()
    out = pd.concat([orig, amend[~dup]], ignore_index=True)
    return out.drop_duplicates(key + ["accession"]).reset_index(drop=True)


def merge_bridge(bridges: pd.DataFrame) -> pd.DataFrame:
    if bridges is None or bridges.empty:
        return pd.DataFrame(columns=["symbol", "issuer_cik", "first_seen", "last_seen", "filings"])
    return (bridges.groupby(["symbol", "issuer_cik"], as_index=False)
            .agg(first_seen=("first_seen", "min"), last_seen=("last_seen", "max"), filings=("filings", "sum")))


def load(sec_dir: str | Path, start_year: int, end_year: int, *, client=None, offline: bool = False,
         today: date | None = None, log=print, local_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(transactions, symbol bridge, stats), cached per quarter on Drive."""
    items = quarter_items(INSIDER_URLS, start_year, end_year, today)
    tables, stats = process(items, Path(sec_dir) / "insider", "insider", extract_quarter, client=client,
                            offline=offline, log=log, local_dir=local_dir)
    tx = tables.get("tx", pd.DataFrame())
    if len(tx):
        tx = dedupe_amendments(tx)
    return tx, merge_bridge(tables.get("bridge")), stats


def classify(tx: pd.DataFrame, years, *, purchases_only: bool = False) -> pd.DataFrame:
    """(year, owner_cik, issuer_cik, insider_class) for every insider-issuer pair with a full
    three-year record before `year`. Pairs not listed are unclassified."""
    cols = ["year", "owner_cik", "issuer_cik", "insider_class"]
    if tx is None or tx.empty:
        return pd.DataFrame(columns=cols)
    t = tx[tx["code"].isin(["P"] if purchases_only else list(OPEN_MARKET))]
    t = t.assign(ty=t["trans_date"].dt.year, tm=t["trans_date"].dt.month)
    out = []
    for y in sorted({int(v) for v in years}):
        h = t[(t["ty"] >= y - 3) & (t["ty"] <= y - 1) & (t["filing_date"] < pd.Timestamp(year=y, month=1, day=1))]
        if h.empty:
            continue
        months = h[["owner_cik", "issuer_cik", "ty", "tm"]].drop_duplicates()
        n_years = months.groupby(["owner_cik", "issuer_cik"])["ty"].nunique()
        full = n_years[n_years == 3].index
        if not len(full):
            continue
        same_month = months.groupby(["owner_cik", "issuer_cik", "tm"])["ty"].nunique()
        routine = set(same_month[same_month == 3].reset_index().set_index(["owner_cik", "issuer_cik"]).index)
        for pair in full:
            out.append((y, pair[0], pair[1], "routine" if pair in routine else "opportunistic"))
    df = pd.DataFrame(out, columns=cols)
    if len(df):
        df["owner_cik"] = df["owner_cik"].astype("Int64")
        df["issuer_cik"] = df["issuer_cik"].astype("Int64")
    return df


def purchases(tx: pd.DataFrame, roles=("officer", "director"), *, exclude_plan: bool = True) -> pd.DataFrame:
    """Open-market purchases (code P, acquired) by insiders in the given roles."""
    if tx is None or tx.empty:
        return pd.DataFrame(columns=list(tx.columns) if tx is not None else [])
    role = pd.Series(False, index=tx.index)
    if "officer" in roles:
        role |= tx["is_officer"]
    if "director" in roles:
        role |= tx["is_director"]
    p = tx[(tx["code"] == "P") & (tx["acq_disp"] == "A") & role]
    if exclude_plan:
        p = p[~p["plan_10b5_1"].fillna(False).astype(bool)]
    return p.reset_index(drop=True)


def sales(tx: pd.DataFrame, roles=("officer", "director"), *, exclude_plan: bool = True) -> pd.DataFrame:
    """Open-market sales (code S, disposed) by insiders in the given roles – the H2 placebo."""
    if tx is None or tx.empty:
        return pd.DataFrame(columns=list(tx.columns) if tx is not None else [])
    role = pd.Series(False, index=tx.index)
    if "officer" in roles:
        role |= tx["is_officer"]
    if "director" in roles:
        role |= tx["is_director"]
    s = tx[(tx["code"] == "S") & (tx["acq_disp"] == "D") & role]
    if exclude_plan:
        s = s[~s["plan_10b5_1"].fillna(False).astype(bool)]
    return s.reset_index(drop=True)


def signals(buys: pd.DataFrame, classes: pd.DataFrame, formation_dates, *, lookback_months: int = 6,
            cluster_min: int = 2, report_dates: pd.DataFrame | None = None,
            post_results_days: int = 14) -> pd.DataFrame:
    """Insider signals per issuer and formation date, from purchases filed in the window.

    `report_dates` (cik, filed) adds the post-results cluster (H6): purchases traded within
    `post_results_days` after a 10-Q or 10-K filing.
    """
    cols = ["issuer_cik", "start_date", "insider_any", "insider_cluster", "insider_opportunistic",
            "insider_routine_only", "insider_unclassified_only", "insider_n", "insider_post_results_cluster",
            "insider_available"]
    if buys is None or buys.empty:
        return pd.DataFrame(columns=cols)
    b = buys.copy()
    for col in ("trans_date", "filing_date"):
        b[col] = pd.to_datetime(b[col]).astype("datetime64[ns]")
    if report_dates is not None and len(report_dates):
        r = report_dates[["cik", "filed"]].dropna().copy()
        r["cik"] = r["cik"].astype("int64")
        r["filed"] = pd.to_datetime(r["filed"]).astype("datetime64[ns]")
        r = r.sort_values("filed")
        b["_k"] = b["issuer_cik"].astype("int64")
        b = b.sort_values("trans_date")
        m = pd.merge_asof(b, r.rename(columns={"cik": "_k", "filed": "_report"}), left_on="trans_date",
                          right_on="_report", by="_k", direction="backward")
        b = m
        b["post_results"] = (b["trans_date"] - b["_report"]).dt.days.between(0, post_results_days)
    else:
        b["post_results"] = False
    out = []
    for d in pd.to_datetime(list(formation_dates)):
        w = b[(b["filing_date"] >= d - pd.DateOffset(months=lookback_months)) & (b["filing_date"] < d)]
        if w.empty:
            continue
        c = classes[classes["year"] == d.year][["owner_cik", "issuer_cik", "insider_class"]] if len(classes) else None
        if c is not None and len(c):
            w = w.merge(c, on=["owner_cik", "issuer_cik"], how="left")
        else:
            w = w.assign(insider_class=np.nan)
        w["insider_class"] = w["insider_class"].fillna("unclassified")
        g = w.groupby("issuer_cik")
        agg = pd.DataFrame({
            "insider_n": g["owner_cik"].nunique(),
            "n_opp": g["insider_class"].apply(lambda s: int((s == "opportunistic").sum())),
            "n_rout": g["insider_class"].apply(lambda s: int((s == "routine").sum())),
            "n_uncl": g["insider_class"].apply(lambda s: int((s == "unclassified").sum())),
            "post_n": w[w["post_results"]].groupby("issuer_cik")["owner_cik"].nunique(),
            "insider_available": g["filing_date"].max(),
        }).reset_index()
        agg["post_n"] = agg["post_n"].fillna(0)
        agg["start_date"] = d
        out.append(agg)
    if not out:
        return pd.DataFrame(columns=cols)
    s = pd.concat(out, ignore_index=True)
    s["insider_any"] = s["insider_n"] >= 1
    s["insider_cluster"] = s["insider_n"] >= cluster_min
    s["insider_opportunistic"] = s["n_opp"] > 0
    s["insider_routine_only"] = (s["n_rout"] > 0) & (s["n_opp"] == 0)
    s["insider_unclassified_only"] = (s["n_uncl"] > 0) & (s["n_opp"] == 0) & (s["n_rout"] == 0)
    s["insider_post_results_cluster"] = s["post_n"] >= cluster_min
    s["issuer_cik"] = s["issuer_cik"].astype("int64")
    return s[cols]
