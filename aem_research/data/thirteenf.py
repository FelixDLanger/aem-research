"""SEC Form 13F Data Sets: institutional breadth, the attention proxy.

Breadth for a stock and quarter = the number of distinct 13F filers reporting a share
position (SSHPRNAMTTYPE "SH", no put or call) in its CUSIP. Only original holdings
reports (13F-HR) count, so each filer is counted once per quarter even though
amendments sit in later files. For point-in-time use, a quarter's breadth counts only
reports filed before the formation date that uses that quarter: the first day of the
fourth month after quarter-end (e.g. the Mar 31 quarter feeds the Jul 1 formation;
13F reports are due 45 days after quarter-end).

13F holdings are keyed by CUSIP. The bridge to the study's CIKs uses the CUSIP inside
a US ISIN (characters 3-11) from the SimFin companies table, then the issuer CUSIP
(first six characters), then the normalized issuer name.
"""
from __future__ import annotations

import re
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .sec_datasets import (F13_PREFIXES, cached_table, f13_items, parse_dates, process, read_member, to_int,
                           valid_accession)

__all__ = ["F13_PREFIXES", "f13_items", "extract_file", "load", "load_holders", "cusip_bridge", "breadth_at",
           "prices_at", "quarter_for_formation", "formation_for_quarter", "normalize_name", "normalize_cusip",
           "SCHEMA", "TOP_HOLDERS"]

VALUE_IN_DOLLARS_FROM = pd.Timestamp("2023-01-03")    # SEC Form 13F readme: VALUE in dollars from this date
MIN_PRICE_FILERS = 3
TOP_HOLDERS = 10
# Fingerprint of what extract_file keeps. A cache extracted under another schema is
# extracted again (Nov 8, 2026: share and value sums and the top holders were added).
SCHEMA = "f13-v2-sums-holders10"

_SUFFIX = re.compile(r"\b(INCORPORATED|INC|CORPORATION|CORP|CP|COMPANY|CO|LIMITED|LTD|PLC|LLC|LP|L P|HOLDINGS?|"
                     r"HLDGS?|GROUP|GRP|THE|COM|CL [A-Z]|CLASS [A-Z]|NEW|DEL|SHS|ORD|N V|NV|SA|AG|SE)\b")


_ABBREV = {r"\bINTL\b": "INTERNATIONAL", r"\bMFG\b": "MANUFACTURING", r"\bSVCS\b": "SERVICES",
           r"\bTECHNOLOGIES\b": "TECHNOLOGY", r"\bTECH\b": "TECHNOLOGY", r"\bPHARMACEUTICALS?\b": "PHARMA",
           r"\bLABORATORIES\b": "LABS", r"\bSYS\b": "SYSTEMS", r"\bFINL\b": "FINANCIAL", r"\bBANCORPORATION\b": "BANCORP",
           r"&": " AND "}


def normalize_name(values) -> pd.Series:
    s = pd.Series(values, dtype="object").fillna("").astype(str).str.upper()
    for pat, rep in _ABBREV.items():
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(_SUFFIX, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_cusip(values) -> pd.Series:
    s = pd.Series(values, dtype="object").fillna("").astype(str).str.upper().str.strip()
    s = s.where(~(s.str.fullmatch(r"\d{1,8}")), s.str.zfill(9))
    return s.where(s.str.fullmatch(r"[0-9A-Z]{9}"), "")


def formation_for_quarter(period: pd.Series) -> pd.Series:
    """First formation date allowed to use a quarter: quarter-end + 1 day + 3 months."""
    p = pd.to_datetime(period)
    return (p + pd.Timedelta(days=1) + pd.DateOffset(months=3)).astype("datetime64[ns]")


def extract_file(zf: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    sub = read_member(zf, "SUBMISSION", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK",
                                         "PERIODOFREPORT"])
    sub = sub[valid_accession(sub["accession_number"])
              & sub["submissiontype"].astype(str).str.strip().str.upper().eq("13F-HR")].copy()
    sub["filing_date"] = parse_dates(sub["filing_date"])
    sub["period"] = parse_dates(sub["periodofreport"])
    sub["filer_cik"] = to_int(sub["cik"])
    sub = sub.dropna(subset=["filing_date", "period", "filer_cik"])
    sub = sub[sub["filing_date"] < formation_for_quarter(sub["period"])]
    keep = set(sub["accession_number"])
    info = read_member(zf, "INFOTABLE", ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMTTYPE", "PUTCALL", "NAMEOFISSUER",
                                         "VALUE", "SSHPRNAMT"],
                       chunk_filter=lambda c: c[c["accession_number"].isin(keep)
                                                & c["sshprnamttype"].astype(str).str.strip().str.upper().eq("SH")
                                                & c["putcall"].isna()])
    info["cusip"] = normalize_cusip(info["cusip"])
    info = info[info["cusip"].ne("")]
    h = info.merge(sub[["accession_number", "filer_cik", "period", "filing_date"]], on="accession_number", how="inner")
    breadth = (h.drop_duplicates(["cusip", "period", "filer_cik"])
               .groupby(["cusip", "period"], as_index=False).agg(n_filers=("filer_cik", "size")))
    names = (h.assign(name=h["nameofissuer"].fillna("").astype(str).str.strip())
             .groupby(["cusip", "name"], as_index=False).size().rename(columns={"size": "n"}))
    # Implied unadjusted price at quarter-end: market value / shares held. VALUE is in
    # thousands of dollars on filings before Jan 3, 2023 and in dollars from then on.
    val = pd.to_numeric(h["value"], errors="coerce") * np.where(h["filing_date"] < VALUE_IN_DOLLARS_FROM, 1000.0, 1.0)
    shr = pd.to_numeric(h["sshprnamt"], errors="coerce")
    px = h.assign(px=val / shr, val=val, shr=shr)[(val > 0) & (shr > 0)]
    prices = (px.groupby(["cusip", "period"], as_index=False)
              .agg(price_13f=("px", "median"), n_price_filers=("px", "size"),
                   shares_13f=("shr", "sum"), value_13f=("val", "sum")))
    # The largest holders of each security: a split multiplies every holder's share count
    # by the same ratio, which is how the SEC price panel finds splits (aem_research.splits).
    per_filer = (h.assign(shr=shr)[shr > 0].groupby(["cusip", "period", "filer_cik"], as_index=False)["shr"].sum()
                 .rename(columns={"shr": "shares"}))
    holders = (per_filer.sort_values(["cusip", "period", "shares", "filer_cik"], ascending=[True, True, False, True],
                                     kind="mergesort")
               .groupby(["cusip", "period"], sort=False).head(TOP_HOLDERS).reset_index(drop=True))
    holders["filer_cik"] = holders["filer_cik"].astype("int64")
    return {"breadth": breadth, "names": names, "prices": prices, "holders": holders}


def load(sec_dir: str | Path, *, client=None, offline: bool = False, today: date | None = None, log=print,
         local_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(breadth per CUSIP and quarter, issuer names per CUSIP, stats), cached per file.

    The breadth table also carries the implied unadjusted quarter-end price (price_13f,
    median across filers) where at least one filer reported value and shares, and the
    shares and market value all filers reported (shares_13f, value_13f). The top holders
    stay on Drive; `load_holders` reads them for the companies that need them."""
    tables, stats = process(f13_items(today), Path(sec_dir) / "form13f", "f13", extract_file, client=client,
                            offline=offline, log=log, local_dir=local_dir, schema=SCHEMA, skip_load=("holders",))
    b = tables.get("breadth", pd.DataFrame(columns=["cusip", "period", "n_filers"]))
    if len(b):
        b = b.groupby(["cusip", "period"], as_index=False)["n_filers"].sum()
    p = tables.get("prices")
    if p is not None and len(p) and len(b):
        # A quarter's filings can sit in two files (late filers): the price comes from the
        # file with most filers, the share and value sums add up across files.
        sums = [c for c in ("shares_13f", "value_13f") if c in p.columns]
        tot = p.groupby(["cusip", "period"], as_index=False)[sums].sum() if sums else None
        p = p.sort_values("n_price_filers", ascending=False, kind="mergesort").drop_duplicates(["cusip", "period"])
        p = p.drop(columns=sums)
        if tot is not None:
            p = p.merge(tot, on=["cusip", "period"], how="left")
        b = b.merge(p, on=["cusip", "period"], how="left")
    n = tables.get("names", pd.DataFrame(columns=["cusip", "name", "n"]))
    if len(n):
        n = (n.groupby(["cusip", "name"], as_index=False)["n"].sum()
             .sort_values(["cusip", "n"], ascending=[True, False]).drop_duplicates("cusip"))
    return b, n[["cusip", "name"]] if len(n) else pd.DataFrame(columns=["cusip", "name"]), stats


def load_holders(sec_dir: str | Path, cusips=None) -> pd.DataFrame:
    """Top holders per CUSIP and quarter (cusip, period, filer_cik, shares) from the cached
    13F extracts, for the given CUSIPs only; a quarter split across two files keeps each
    filer once (its position summed), then the largest TOP_HOLDERS."""
    cols = ["cusip", "period", "filer_cik", "shares"]
    want = None if cusips is None else set(pd.Series(list(cusips), dtype="object").astype(str))

    def keep(df):
        df = df[cols]
        return df[df["cusip"].isin(want)] if want is not None else df

    h = cached_table(Path(sec_dir) / "form13f", "f13", "holders", keep, schema=SCHEMA)
    if h is None or h.empty:
        return pd.DataFrame(columns=cols)
    h["period"] = pd.to_datetime(h["period"]).astype("datetime64[ns]")
    h = h.groupby(["cusip", "period", "filer_cik"], as_index=False)["shares"].sum()
    h = (h.sort_values(["cusip", "period", "shares", "filer_cik"], ascending=[True, True, False, True], kind="mergesort")
         .groupby(["cusip", "period"], sort=False).head(TOP_HOLDERS).reset_index(drop=True))
    h["filer_cik"] = h["filer_cik"].astype("int64")
    h["shares"] = h["shares"].astype("float64")
    return h[cols]


def prices_at(breadth: pd.DataFrame, bridge: pd.DataFrame, formation_dates, *,
              min_filers: int = MIN_PRICE_FILERS) -> pd.DataFrame:
    """(cik, start_date, price_13f, price_13f_period): the implied unadjusted price at the
    latest quarter-end usable at each formation date, from the CUSIP with the most filers
    and only where at least `min_filers` filers reported it."""
    cols = ["cik", "start_date", "price_13f", "price_13f_period"]
    if breadth is None or breadth.empty or "price_13f" not in breadth.columns or bridge is None or bridge.empty:
        return pd.DataFrame(columns=cols)
    b = breadth[breadth["n_price_filers"] >= min_filers].merge(bridge[["cusip", "cik"]], on="cusip", how="inner")
    b["period"] = pd.to_datetime(b["period"]).astype("datetime64[ns]")
    out = []
    for d in pd.to_datetime(list(formation_dates)):
        q = quarter_for_formation(d)
        w = b[b["period"] == q].sort_values("n_filers", ascending=False, kind="mergesort").drop_duplicates("cik")
        if w.empty:
            continue
        out.append(pd.DataFrame({"cik": w["cik"].astype("int64").to_numpy(), "start_date": d,
                                 "price_13f": w["price_13f"].to_numpy(), "price_13f_period": q}))
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def cusip_bridge(companies: pd.DataFrame | None, names: pd.DataFrame, filer_names: pd.DataFrame | None = None
                 ) -> pd.DataFrame:
    """(cusip, cik, method). `companies` needs cik and isin (SimFin); `filer_names` (cik,
    name) are SEC registrant names for the name fallback."""
    cols = ["cusip", "cik", "method"]
    rows = []
    known = names[["cusip", "name"]].drop_duplicates("cusip") if len(names) else pd.DataFrame(columns=["cusip", "name"])
    known = known.assign(cusip6=known["cusip"].str[:6])
    if companies is not None and len(companies) and {"cik", "isin"} <= set(companies.columns):
        c = companies[["cik", "isin"]].dropna()
        c = c[c["isin"].astype(str).str.upper().str.match(r"^US[0-9A-Z]{9}\d$")]
        c = c.assign(cusip=c["isin"].astype(str).str.upper().str[2:11], cik=to_int(c["cik"]))
        c = c.dropna(subset=["cik"])
        exact = known.merge(c[["cusip", "cik"]], on="cusip", how="inner").assign(method="isin")
        rows.append(exact[cols])
        c6 = c.assign(cusip6=c["cusip"].str[:6])[["cusip6", "cik"]].drop_duplicates()
        c6 = c6[~c6.duplicated("cusip6", keep=False)]          # an issuer code shared by two CIKs is ambiguous
        rest = known[~known["cusip"].isin(exact["cusip"])]
        by6 = rest.merge(c6[["cusip6", "cik"]], on="cusip6", how="inner").assign(method="issuer cusip")
        rows.append(by6[cols])
    done = set(pd.concat(rows)["cusip"]) if rows else set()
    if filer_names is not None and len(filer_names):
        fn = filer_names[["cik", "name"]].dropna().assign(key=lambda d: normalize_name(d["name"]))
        fn = fn[fn["key"].ne("")].drop_duplicates(["key", "cik"])
        fn = fn[~fn.duplicated("key", keep=False)]          # a name shared by two CIKs is ambiguous
        rest = known[~known["cusip"].isin(done)].assign(key=lambda d: normalize_name(d["name"]))
        byname = rest.merge(fn[["key", "cik"]], on="key", how="inner").assign(method="name")
        rows.append(byname[cols])
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True).drop_duplicates("cusip")
    out["cik"] = out["cik"].astype("int64")
    return out.reset_index(drop=True)


def quarter_for_formation(d) -> pd.Timestamp:
    """Latest quarter-end whose breadth may be used at formation date d."""
    d = pd.Timestamp(d)
    q = (d - pd.DateOffset(months=3) - pd.Timedelta(days=1)).to_period("Q").end_time.normalize()
    while formation_for_quarter(pd.Series([q])).iloc[0] > d:
        q = (q - pd.DateOffset(months=3)).to_period("Q").end_time.normalize()
    return pd.Timestamp(q)


def breadth_at(breadth: pd.DataFrame, bridge: pd.DataFrame, formation_dates) -> pd.DataFrame:
    """(cik, start_date, breadth, breadth_period, breadth_available) per formation date. A CIK with
    several CUSIPs (share classes) takes the largest breadth."""
    cols = ["cik", "start_date", "breadth", "breadth_period", "breadth_available"]
    if breadth is None or breadth.empty or bridge is None or bridge.empty:
        return pd.DataFrame(columns=cols)
    b = breadth.merge(bridge[["cusip", "cik"]], on="cusip", how="inner")
    b["period"] = pd.to_datetime(b["period"]).astype("datetime64[ns]")
    out = []
    for d in pd.to_datetime(list(formation_dates)):
        q = quarter_for_formation(d)
        w = b[b["period"] == q]
        if w.empty:
            continue
        g = w.groupby("cik", as_index=False)["n_filers"].max().rename(columns={"n_filers": "breadth"})
        g["start_date"], g["breadth_period"] = d, q
        g["breadth_available"] = formation_for_quarter(pd.Series([q])).iloc[0]
        out.append(g)
    if not out:
        return pd.DataFrame(columns=cols)
    s = pd.concat(out, ignore_index=True)
    s["cik"] = s["cik"].astype("int64")
    return s[cols]
