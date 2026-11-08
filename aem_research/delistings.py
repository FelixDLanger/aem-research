"""Classify every price series that stops before the as-of date (pre-registered rules).

No free source records delisting returns, so each ended series gets a category and the
outcome engine applies the pre-registered return for it (config: delisting.returns):
  1. merger       – a merger filing about the issuer (DEFM14A, SC 14D9, SC TO-T, 425 and
                    others) in the 365 days before its last trading day or the 30 after
  2. performance  – no merger evidence, and the last close is below US$1, or at most
                    half the close a year earlier, or the exchange filed Form 25-NSE;
                    suffixed with the exchange (Form 25-NSE filer, else Stooq folder):
                    performance_nasdaq (-55%), performance_nyse_amex (-30%)
  3. unknown      – neither
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.sec import is_exchange_name
from .tickers import canonical_series

CATEGORIES = ("merger", "performance_nasdaq", "performance_nyse_amex", "performance", "unknown")


def exchange_label(name: str) -> str:
    n = str(name).upper()
    if "NASDAQ" in n:
        return "nasdaq"
    if "NYSE" in n or "NEW YORK STOCK EXCHANGE" in n or "AMERICAN STOCK EXCHANGE" in n or "AMEX" in n:
        return "nyse_amex"
    return "other"


def stooq_exchange(group: str) -> str:
    g = str(group).lower()
    if g.startswith("nasdaq"):
        return "nasdaq"
    if g.startswith("nyse") or g.startswith("amex"):
        return "nyse_amex"
    return "other"


def ended_segments(panel: pd.DataFrame, as_of, *, stale_gap_days: int = 30, max_gap_days: int = 120,
                   price_col: str = "price") -> pd.DataFrame:
    """One row per series segment that stops more than stale_gap_days before as_of."""
    cols = ["ticker", "first_date", "last_date", "last_price", "price_1y_before"]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=cols)
    as_of = np.datetime64(pd.Timestamp(as_of), "ns")
    gap, stale = np.timedelta64(max_gap_days, "D"), np.timedelta64(stale_gap_days, "D")
    year = np.timedelta64(365, "D")
    dates = pd.to_datetime(panel["date"]).astype("datetime64[ns]").to_numpy()
    prices = pd.to_numeric(panel[price_col], errors="coerce").to_numpy(dtype="float64")
    rows = []
    for t, pos in panel.groupby("ticker", observed=True, sort=False).indices.items():
        d, p = dates[pos], prices[pos]
        ok = ~np.isnat(d) & np.isfinite(p) & (p > 0)
        d, p = d[ok], p[ok]
        if d.size == 0:
            continue
        o = np.argsort(d, kind="mergesort")
        d, p = d[o], p[o]
        cuts = np.r_[0, np.flatnonzero(np.diff(d) > gap) + 1, d.size]
        for a, b in zip(cuts[:-1], cuts[1:]):
            sd, sp = d[a:b], p[a:b]
            if sd[-1] >= as_of - stale:
                continue
            j = int(np.searchsorted(sd, sd[-1] - year, side="right")) - 1
            rows.append((str(t), sd[0], sd[-1], float(sp[-1]), float(sp[j]) if j >= 0 else np.nan))
    out = pd.DataFrame(rows, columns=cols)
    for c in ("first_date", "last_date"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ns]")
    return out


def form25_with_exchange(form25: pd.DataFrame) -> pd.DataFrame:
    """Issuer rows of the Form 25 index with the filing exchange (from the exchange's row of
    the same filing). Columns: cik, date_filed, form, exchange."""
    cols = ["cik", "date_filed", "form", "exchange"]
    if form25 is None or form25.empty:
        return pd.DataFrame(columns=cols)
    f = form25.copy()
    f["is_exchange"] = f["company"].map(is_exchange_name)
    # The index lists a filing once per CIK, each under that CIK's own folder
    # (edgar/data/<cik>/<accession>.txt), so the issuer and the exchange share the
    # accession number, not the path.
    f["accession"] = f["filename"].astype(str).str.rsplit("/", n=1).str[-1]
    ex = f[f["is_exchange"]].assign(exchange=lambda d: d["company"].map(exchange_label))[["accession", "exchange"]]
    ex = ex.drop_duplicates("accession")
    iss = f[~f["is_exchange"]].merge(ex, on="accession", how="left")
    iss["exchange"] = iss["exchange"].fillna("")
    iss["date_filed"] = pd.to_datetime(iss["date_filed"]).astype("datetime64[ns]")
    return iss[cols].reset_index(drop=True)


def map_ciks(segments: pd.DataFrame, companies: pd.DataFrame | None = None,
             bridge: pd.DataFrame | None = None, *, margin_days: int = 400) -> pd.DataFrame:
    """Attach a CIK to each segment: SimFin companies first, then the symbol bridge (Form 4
    trading symbols and XBRL instance names) whose dates cover the segment's last day."""
    seg = segments.copy()
    seg["cik"] = pd.Series(pd.NA, index=seg.index, dtype="Int64")
    seg["cik_source"] = ""
    if companies is not None and len(companies) and {"ticker", "cik"} <= set(companies.columns):
        c = companies[["ticker", "cik"]].dropna().copy()
        c["ticker"] = c["ticker"].astype(str)
        c["cik"] = pd.to_numeric(c["cik"], errors="coerce").astype("Int64")
        c = c.dropna().drop_duplicates("ticker")
        m = seg[["ticker"]].merge(c, on="ticker", how="left")["cik"].to_numpy()
        hit = pd.notna(m)
        seg.loc[hit, "cik"] = m[hit]
        seg.loc[hit, "cik_source"] = "simfin"
    if bridge is not None and len(bridge):
        b = bridge.rename(columns={"issuer_cik": "cik"})[["symbol", "cik", "first_seen", "last_seen", "filings"]].copy()
        b["symbol"] = canonical_series(b["symbol"])
        need = seg[seg["cik"].isna()].reset_index()
        need["symbol"] = canonical_series(need["ticker"].astype(str).str.split("~").str[0])
        m = need.merge(b, on="symbol", how="inner", suffixes=("", "_b"))
        lo = pd.to_datetime(m["first_seen"]) - pd.Timedelta(days=margin_days)
        hi = pd.to_datetime(m["last_seen"]) + pd.Timedelta(days=margin_days)
        m = m[(m["last_date"] >= lo) & (m["last_date"] <= hi)]
        m = m.sort_values("filings", ascending=False).drop_duplicates("index")
        seg.loc[m["index"].to_numpy(), "cik"] = pd.to_numeric(m["cik_b"], errors="coerce").astype("Int64").to_numpy()
        seg.loc[m["index"].to_numpy(), "cik_source"] = "bridge"
    return seg


def classify(segments: pd.DataFrame, form25: pd.DataFrame | None = None, merger: pd.DataFrame | None = None,
             stooq_groups: pd.DataFrame | None = None, *, before_days: int = 365, after_days: int = 30,
             form25_days: int = 60, penny: float = 1.0, collapse: float = 0.5,
             exchange_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Delisting records for the outcome engine: ticker, delist_date, category, delist_return
    (always missing here), plus the evidence used. `segments` needs a cik column (map_ciks).

    The exchange comes from an exchange-filed Form 25 near the last trading day, else the
    issuer's latest exchange-filed Form 25 before it, else Stooq's group, else
    `exchange_map` (ticker → nasdaq / nyse_amex, e.g. EODHD's symbol lists)."""
    s = segments.copy().reset_index(drop=True)
    if s.empty:
        return pd.DataFrame(columns=["ticker", "delist_date", "category", "delist_return", "cik", "evidence"])
    s["merger"] = False
    s["f25_nse"] = False
    s["f25_exchange"] = ""
    has_cik = s["cik"].notna()
    if merger is not None and len(merger) and has_cik.any():
        mg = merger[["cik", "date_filed"]].dropna().copy()
        mg["cik"] = mg["cik"].astype("int64")
        mg["date_filed"] = pd.to_datetime(mg["date_filed"]).astype("datetime64[ns]")
        j = s[has_cik].reset_index().assign(cik=lambda d: d["cik"].astype("int64")).merge(mg, on="cik")
        lo = j["last_date"] - pd.Timedelta(days=before_days)
        hi = j["last_date"] + pd.Timedelta(days=after_days)
        hit = j[(j["date_filed"] >= lo) & (j["date_filed"] <= hi)]["index"].unique()
        s.loc[hit, "merger"] = True
    f25 = form25_with_exchange(form25) if form25 is not None else None
    if f25 is not None and len(f25) and has_cik.any():
        f = f25.copy()
        f["cik"] = f["cik"].astype("int64")
        j = s[has_cik].reset_index().assign(cik=lambda d: d["cik"].astype("int64")).merge(f, on="cik")
        near = j[(j["date_filed"] - j["last_date"]).abs() <= pd.Timedelta(days=form25_days)]
        nse = near[near["form"].astype(str).str.startswith("25-NSE")]
        s.loc[nse["index"].unique(), "f25_nse"] = True
        ex = near[near["exchange"].isin(["nasdaq", "nyse_amex"])].drop_duplicates("index")
        s.loc[ex["index"].to_numpy(), "f25_exchange"] = ex["exchange"].to_numpy()
        # else the latest exchange-filed Form 25 for the issuer up to the last trading day
        early = j[j["exchange"].isin(["nasdaq", "nyse_amex"])
                  & (j["date_filed"] <= j["last_date"] + pd.Timedelta(days=form25_days))]
        early = early.sort_values("date_filed").drop_duplicates("index", keep="last")
        empty = s.loc[early["index"].to_numpy(), "f25_exchange"].eq("").to_numpy()
        s.loc[early["index"].to_numpy()[empty], "f25_exchange"] = early["exchange"].to_numpy()[empty]
    s["stooq_exchange"] = ""
    if stooq_groups is not None and len(stooq_groups):
        g = stooq_groups.drop_duplicates("ticker").set_index("ticker")["group"].map(stooq_exchange)
        s["stooq_exchange"] = s["ticker"].map(g).fillna("")
    s["map_exchange"] = ""
    if exchange_map is not None and len(exchange_map):
        em = exchange_map.drop_duplicates("ticker").set_index("ticker")["exchange"]
        s["map_exchange"] = s["ticker"].astype(str).str.split("~").str[0].map(em).fillna("")
    collapsed = (s["last_price"] < penny) | (s["last_price"] <= collapse * s["price_1y_before"])
    performance = ~s["merger"] & (collapsed | s["f25_nse"])
    exch = s["f25_exchange"].where(s["f25_exchange"].ne(""), s["stooq_exchange"])
    exch = exch.where(exch.ne(""), s["map_exchange"])
    cat = np.where(s["merger"], "merger",
                   np.where(performance,
                            np.where(exch.eq("nasdaq"), "performance_nasdaq",
                                     np.where(exch.eq("nyse_amex"), "performance_nyse_amex", "performance")),
                            "unknown"))
    ev = np.where(s["merger"], "merger filing",
                  np.where(performance & s["f25_nse"], "Form 25-NSE",
                           np.where(performance & (s["last_price"] < penny), "last close below $1",
                                    np.where(performance, "price collapse", "none"))))
    return pd.DataFrame({"ticker": s["ticker"], "delist_date": s["last_date"], "category": cat,
                         "delist_return": np.nan, "cik": s["cik"], "evidence": ev,
                         "last_price": s["last_price"]})


def summary(classified: pd.DataFrame) -> dict:
    if classified is None or classified.empty:
        return {"ended_series": 0}
    n = len(classified)
    out = {"ended_series": n}
    for c in CATEGORIES:
        out[f"share_{c}"] = round(float((classified["category"] == c).mean()), 4)
    out["with_cik"] = round(float(classified["cik"].notna().mean()), 4)
    return out
