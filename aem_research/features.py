"""Point-in-time features and the study universe at each formation date.

Joins price features (entry price, 3-year drawdown, dollar volume, turnover) with the
fundamentals, insider signals and institutional breadth, applies the pre-registered
universe filters, and builds the H2 cohort and H4 signal flags. Everything a feature
uses was public before the formation date; `check_point_in_time` fails the run if not.

`coverage_report` summarises counts only – no returns – so it can run before the
pre-registration is locked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .guards import assert_point_in_time
from .tickers import canonical_series

H4_SIGNALS = ("s1_proxy", "insider_any", "insider_cluster", "insider_opportunistic", "net_cash_pos", "fcf_pos",
              "low_dilution", "low_attention")


def formation_dates(years, months=(1,)) -> list[pd.Timestamp]:
    """Calendar start dates (the 1st of each month); the entry is the first close on or after it."""
    y0, y1 = int(years[0]), int(years[1])
    return [pd.Timestamp(year=y, month=int(m), day=1) for y in range(y0, y1 + 1) for m in months]


def price_features(panel: pd.DataFrame, starts, *, tolerance_days: int = 7, lookback_years: int = 3,
                   max_gap_days: int = 120, shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per ticker and start date: entry date and price, 3-year high and drawdown, median 20-day
    dollar volume and median 126-day volume before entry, and the prices at the quarter-end
    whose 13F holdings are usable at the start date (to undo later split adjustments). The
    lookback stays inside the series segment that contains the entry (a gap longer than
    max_gap_days starts a new one)."""
    from .data.thirteenf import quarter_for_formation

    cols = ["ticker", "start_date", "entry_date", "entry_price", "entry_close_raw", "high_3y", "drawdown",
            "dollar_volume_20d", "volume_126d", "price_q", "close_raw_q", "source", "shares_vendor"]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=cols)
    start_list = list(pd.to_datetime(list(starts)))
    starts = np.array(pd.DatetimeIndex(start_list).astype("datetime64[ns]"))
    qends = np.array(pd.DatetimeIndex([quarter_for_formation(s) for s in start_list]).astype("datetime64[ns]"))
    dates = pd.to_datetime(panel["date"]).astype("datetime64[ns]").to_numpy()
    price = pd.to_numeric(panel["price"], errors="coerce").to_numpy(dtype="float64")
    raw = (pd.to_numeric(panel["close_raw"], errors="coerce").to_numpy(dtype="float64")
           if "close_raw" in panel.columns else price.copy())
    vol = (pd.to_numeric(panel["volume"], errors="coerce").to_numpy(dtype="float64")
           if "volume" in panel.columns else np.full(len(panel), np.nan))
    src = panel["source"].astype(str).to_numpy() if "source" in panel.columns else np.full(len(panel), "")
    shr = (pd.to_numeric(panel["shares_out"], errors="coerce").to_numpy(dtype="float64")
           if "shares_out" in panel.columns else np.full(len(panel), np.nan))
    tol = np.timedelta64(tolerance_days, "D")
    gap = np.timedelta64(max_gap_days, "D")
    q_tol = np.timedelta64(10, "D")
    rows = []
    for t, pos in panel.groupby("ticker", observed=True, sort=False).indices.items():
        d, p, r, v, sh = dates[pos], price[pos], raw[pos], vol[pos], shr[pos]
        ok = ~np.isnat(d) & np.isfinite(p) & (p > 0)
        d, p, r, v, sh, s_src = d[ok], p[ok], r[ok], v[ok], sh[ok], src[pos][ok]
        if d.size == 0:
            continue
        o = np.argsort(d, kind="mergesort")
        d, p, r, v, sh, s_src = d[o], p[o], r[o], v[o], sh[o], s_src[o]
        r = np.where(np.isfinite(r) & (r > 0), r, p)
        breaks = np.flatnonzero(np.diff(d) > gap) + 1
        j0 = np.searchsorted(d, starts, side="left")
        jq0 = np.searchsorted(d, qends, side="right") - 1
        for s, qe, j, jq in zip(starts, qends, j0, jq0):
            if j >= d.size or d[j] - s > tol:
                continue
            k = int(np.searchsorted(breaks, j, side="right"))
            seg_lo = int(breaks[k - 1]) if k > 0 else 0
            lo = max(seg_lo, int(np.searchsorted(d, d[j] - np.timedelta64(365 * lookback_years + 1, "D"), "left")))
            hi_price = float(p[lo:j + 1].max())
            pre_lo20, pre_lo126 = max(seg_lo, j - 20), max(seg_lo, j - 126)
            dv = r[pre_lo20:j] * v[pre_lo20:j]
            dv = dv[np.isfinite(dv)]
            v126 = v[pre_lo126:j]
            v126 = v126[np.isfinite(v126)]
            q_ok = bool(seg_lo <= jq < d.size and d[jq] >= qe - q_tol)   # a close within 10 days of quarter-end
            rows.append((str(t), s, d[j], float(p[j]), float(r[j]), hi_price, float(p[j] / hi_price - 1.0),
                         float(np.median(dv)) if dv.size >= 10 else np.nan,
                         float(np.median(v126)) if v126.size >= 60 else np.nan,
                         float(p[jq]) if q_ok else np.nan, float(r[jq]) if q_ok else np.nan,
                         s_src[j], float(sh[j]) if np.isfinite(sh[j]) else np.nan))
    out = pd.DataFrame(rows, columns=cols)
    for c in ("start_date", "entry_date"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ns]")
    return out


def instance_bridge(subs: pd.DataFrame) -> pd.DataFrame:
    """Symbol bridge from XBRL instance file names ('acme-20120930.xml' -> ACME)."""
    cols = ["symbol", "issuer_cik", "first_seen", "last_seen", "filings"]
    if subs is None or subs.empty or "instance" not in subs.columns:
        return pd.DataFrame(columns=cols)
    s = subs.dropna(subset=["instance", "cik", "filed"])
    sym = s["instance"].astype(str).str.extract(r"^([A-Za-z][A-Za-z0-9.]{0,9})-\d{8}", expand=False)
    s = s.assign(symbol=canonical_series(sym.fillna("")), issuer_cik=s["cik"].astype("int64"))
    s = s[s["symbol"].ne("")]
    return (s.groupby(["symbol", "issuer_cik"], as_index=False)
            .agg(first_seen=("filed", "min"), last_seen=("filed", "max"), filings=("adsh", "size")))[cols]


def ticker_ciks(pf: pd.DataFrame, companies: pd.DataFrame | None, bridges: list[pd.DataFrame], *,
                margin_days: int = 400) -> pd.DataFrame:
    """(ticker, start_date, cik, cik_source): SimFin's CIK for its tickers, else the symbol bridges
    (Form 4 trading symbols, XBRL instance names) whose dates cover the start date."""
    keys = pf[["ticker", "start_date"]].drop_duplicates().reset_index(drop=True)
    keys["cik"] = pd.Series(pd.NA, index=keys.index, dtype="Int64")
    keys["cik_source"] = ""
    if companies is not None and len(companies) and {"ticker", "cik"} <= set(companies.columns):
        c = companies[["ticker", "cik"]].copy()
        c["ticker"] = c["ticker"].astype(str)
        c["cik"] = pd.to_numeric(c["cik"], errors="coerce").astype("Int64")
        c = c.dropna().drop_duplicates("ticker")
        m = keys[["ticker"]].merge(c, on="ticker", how="left")["cik"].to_numpy()
        hit = pd.notna(m)
        keys.loc[hit, "cik"], keys.loc[hit, "cik_source"] = m[hit], "simfin"
    usable = [x for x in bridges if x is not None and len(x)]
    if usable:
        b = pd.concat(usable, ignore_index=True)[["symbol", "issuer_cik", "first_seen", "last_seen", "filings"]]
        b["symbol"] = canonical_series(b["symbol"])
        need = keys[keys["cik"].isna()].reset_index()
        need["symbol"] = canonical_series(need["ticker"].astype(str).str.split("~").str[0])
        m = need.drop(columns=["cik"]).merge(b, on="symbol", how="inner")
        lo = pd.to_datetime(m["first_seen"]) - pd.Timedelta(days=margin_days)
        hi = pd.to_datetime(m["last_seen"]) + pd.Timedelta(days=margin_days)
        m = m[(m["start_date"] >= lo) & (m["start_date"] <= hi)]
        m = m.sort_values("filings", ascending=False, kind="mergesort").drop_duplicates("index")
        keys.loc[m["index"].to_numpy(), "cik"] = pd.to_numeric(m["issuer_cik"]).astype("Int64").to_numpy()
        keys.loc[m["index"].to_numpy(), "cik_source"] = "bridge"
    return keys


def _ns(df: pd.DataFrame | None, *cols) -> pd.DataFrame | None:
    if df is None:
        return None
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    return df


def _tercile_low(x: pd.Series, size: pd.Series) -> pd.Series:
    """Bottom tercile of x net of log size (OLS residual), NA where x is unknown."""
    out = pd.Series(pd.NA, index=x.index, dtype="boolean")
    ok = x.notna() & size.notna() & np.isfinite(x) & (size > 0)
    if ok.sum() < 9:
        return out
    ls = np.log(size[ok].astype(float))
    slope, icpt = np.polyfit(ls, x[ok].astype(float), 1)
    resid = x[ok].astype(float) - (icpt + slope * ls)
    out[ok] = (resid.rank(pct=True) <= 1 / 3).astype(bool)
    return out


def assemble(pf: pd.DataFrame, keys: pd.DataFrame, fund: pd.DataFrame | None, ins: pd.DataFrame | None,
             breadth: pd.DataFrame | None, cfg_data: dict, prices13: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per ticker and start date with every feature and the universe flag.

    Size and price filters use the unadjusted entry price. Vendor histories are rescaled
    for every later split, so a stock that later reverse-splits 1:10 shows ten times its
    actual price; filtering on that would use the future. Where 13F holders report the
    stock, the unadjusted price is the vendor price times the ratio of the 13F implied
    quarter-end price to the vendor's quarter-end price; elsewhere the vendor's close."""
    u = cfg_data.get("universe", {})
    cap = u.get("market_cap_usd", {}) or {}
    cohort = cfg_data.get("turnaround_cohort", {}) or {}
    pf, keys = _ns(pf.copy(), "start_date", "entry_date"), _ns(keys.copy(), "start_date")
    fund = _ns(fund.copy(), "start_date", "fund_available", "shares_dei_filed") if fund is not None and len(fund) else None
    ins = _ns(ins.copy(), "start_date", "insider_available") if ins is not None and len(ins) else None
    breadth = (_ns(breadth.copy(), "start_date", "breadth_period", "breadth_available")
               if breadth is not None and len(breadth) else None)
    f = pf.merge(keys, on=["ticker", "start_date"], how="left")
    f["cik"] = f["cik"].astype("Int64")
    if fund is not None and len(fund):
        fd = fund.copy()
        fd["cik"] = fd["cik"].astype("Int64")
        f = f.merge(fd, on=["cik", "start_date"], how="left")
    for c in ("fund_available", "shares_dei", "shares_dei_filed", "sic", "fcf_pos", "net_cash_pos", "s1_proxy",
              "low_dilution", "net_cash", "public_float"):
        if c not in f.columns:
            f[c] = pd.NA
    if ins is not None and len(ins):
        i = ins.copy()
        i["issuer_cik"] = i["issuer_cik"].astype("Int64")
        f = f.merge(i.rename(columns={"issuer_cik": "cik"}), on=["cik", "start_date"], how="left")
    for c in ("insider_any", "insider_cluster", "insider_opportunistic", "insider_routine_only",
              "insider_unclassified_only", "insider_post_results_cluster"):
        if c not in f.columns:
            f[c] = False
        f[c] = f[c].astype("boolean").fillna(False)   # no Form 4 purchase on file = no purchase
    if "insider_available" not in f.columns:
        f["insider_available"] = pd.NaT
    if breadth is not None and len(breadth):
        b = breadth.copy()
        b["cik"] = b["cik"].astype("Int64")
        f = f.merge(b, on=["cik", "start_date"], how="left")
    for c in ("breadth", "breadth_period", "breadth_available"):
        if c not in f.columns:
            f[c] = pd.NA if c == "breadth" else pd.NaT
    if prices13 is not None and len(prices13):
        p13 = _ns(prices13.copy(), "start_date", "price_13f_period")
        p13["cik"] = p13["cik"].astype("Int64")
        f = f.merge(p13, on=["cik", "start_date"], how="left")
    for c in ("price_13f", "price_13f_period", "price_q", "close_raw_q", "entry_close_raw"):
        if c not in f.columns:
            f[c] = np.nan if c != "price_13f_period" else pd.NaT

    ratio = pd.to_numeric(f["price_13f"], errors="coerce") / pd.to_numeric(f["price_q"], errors="coerce")
    ratio = ratio.where(np.isfinite(ratio) & (ratio > 0))
    raw_close = pd.to_numeric(f["entry_close_raw"], errors="coerce").fillna(f["entry_price"])
    f["entry_price_unadjusted"] = (f["entry_price"] * ratio).fillna(raw_close)
    f["price_basis"] = np.where(ratio.notna(), "13F", "vendor close")
    f["vendor_close_vs_13f"] = pd.to_numeric(f["close_raw_q"], errors="coerce") / pd.to_numeric(f["price_13f"], errors="coerce")

    dei_fresh = pd.to_datetime(f["shares_dei_filed"]) >= f["start_date"] - pd.DateOffset(months=15)
    shares = pd.to_numeric(f["shares_dei"], errors="coerce").where(dei_fresh)
    f["shares_source"] = np.where(shares.notna(), "filing", np.where(f["shares_vendor"].notna(), "vendor", ""))
    f["shares"] = shares.fillna(pd.to_numeric(f["shares_vendor"], errors="coerce"))
    f["market_cap"] = f["entry_price_unadjusted"] * f["shares"]
    f["turnover_126d"] = f["volume_126d"] / f["shares"]

    sic = pd.to_numeric(f["sic"], errors="coerce")
    f["is_filer"] = f["fund_available"].notna()
    f["in_universe"] = (f["is_filer"] & (sic != 6770)
                        & (f["entry_price_unadjusted"] >= float(u.get("min_price_usd", 1.0)))
                        & f["market_cap"].between(float(cap.get("min", 1e8)), float(cap.get("max", 2e9)))
                        & (f["dollar_volume_20d"] >= float(u.get("min_median_dollar_volume_20d_usd", 1e5))))
    f["in_universe"] = f["in_universe"].fillna(False).astype(bool)

    # attention: bottom tercile net of size, within each start date's universe
    f["low_breadth"] = pd.Series(pd.NA, index=f.index, dtype="boolean")
    f["low_turnover"] = pd.Series(pd.NA, index=f.index, dtype="boolean")
    for _, idx in f[f["in_universe"]].groupby("start_date").groups.items():
        g = f.loc[idx]
        f.loc[idx, "low_breadth"] = _tercile_low(np.log1p(pd.to_numeric(g["breadth"], errors="coerce")), g["market_cap"])
        f.loc[idx, "low_turnover"] = _tercile_low(np.log(pd.to_numeric(g["turnover_126d"], errors="coerce")
                                                         .where(lambda s: s > 0)), g["market_cap"])
    f["low_attention"] = f["low_breadth"]

    dd = float(cohort.get("drawdown_from_high", -0.40))
    f["fallen"] = f["drawdown"] <= dd
    med = f[f["in_universe"]].groupby("start_date")["drawdown"].median().rename("drawdown_median")
    f = f.merge(med, on="start_date", how="left")
    f["fallen_relative"] = (f["drawdown"] - f["drawdown_median"]) <= dd
    quality = f["fcf_pos"].astype("boolean") & f["net_cash_pos"].astype("boolean")
    f["cohort_known"] = f["fcf_pos"].notna() & f["net_cash_pos"].notna()
    f["h2a"] = (f["fallen"] & quality & f["insider_any"]).astype("boolean")
    f["h2b"] = (f["fallen"] & quality & f["insider_opportunistic"]).astype("boolean")
    for c in ("h2a", "h2b"):
        f.loc[~f["cohort_known"], c] = pd.NA
    return f


def price_basis_check(features: pd.DataFrame) -> pd.DataFrame:
    """Is each vendor's close unadjusted? Median of vendor close / 13F implied price at the
    quarter-end, and the share of names off by more than 10% (later splits). Counts only."""
    v = features.dropna(subset=["vendor_close_vs_13f"])
    v = v[np.isfinite(v["vendor_close_vs_13f"])]
    if v.empty:
        return pd.DataFrame(columns=["source", "names", "median_ratio", "share_off_10pct"])
    g = v.groupby("source")["vendor_close_vs_13f"]
    return pd.DataFrame({"names": g.size(), "median_ratio": g.median().round(4),
                         "share_off_10pct": g.apply(lambda s: float(((s - 1).abs() > 0.10).mean())).round(4)}
                        ).reset_index()


def choose_attention(features: pd.DataFrame, h4_start, min_known: float = 0.8) -> tuple[str, float]:
    """Pre-registered rule: breadth if known for at least 80% of H4 windows, else turnover."""
    h4 = features[features["in_universe"] & (features["start_date"] >= pd.Timestamp(h4_start))]
    share = float(h4["low_breadth"].notna().mean()) if len(h4) else 0.0
    return ("breadth" if share >= min_known else "turnover"), share


def check_point_in_time(features: pd.DataFrame) -> None:
    """Fundamentals, insider purchases and 13F breadth must all predate the start date."""
    for col, strict in (("fund_available", True), ("insider_available", True), ("breadth_available", False)):
        if col not in features.columns:
            continue
        sub = features[features[col].notna()][["start_date", col]].copy()
        if sub.empty:
            continue
        sub[col] = pd.to_datetime(sub[col])
        if strict:   # filed before the start date
            sub[col] = sub[col] + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        assert_point_in_time(sub, start_col="start_date", available_col=col)


def coverage_report(features: pd.DataFrame, *, h4_start=None) -> pd.DataFrame:
    """Counts only, per start date: universe size, share of each feature known, cohort sizes."""
    u = features[features["in_universe"]]
    if u.empty:
        return pd.DataFrame()
    g = u.groupby("start_date")
    rep = pd.DataFrame({
        "universe": g.size(),
        "fcf_known": g["fcf_pos"].apply(lambda s: s.notna().mean()),
        "net_cash_known": g["net_cash_pos"].apply(lambda s: s.notna().mean()),
        "s1_known": g["s1_proxy"].apply(lambda s: s.notna().mean()),
        "dilution_known": g["low_dilution"].apply(lambda s: s.notna().mean()),
        "breadth_known": g["low_breadth"].apply(lambda s: s.notna().mean()),
        "turnover_known": g["low_turnover"].apply(lambda s: s.notna().mean()),
        "shares_from_filing": g["shares_source"].apply(lambda s: (s == "filing").mean()),
        "price_from_13f": g["price_basis"].apply(lambda s: (s == "13F").mean()),
        "fallen": g["fallen"].sum(),
        "insider_buy": g["insider_any"].sum(),
        "h2a_cohort": g["h2a"].apply(lambda s: int(s.fillna(False).sum())),
        "h2b_cohort": g["h2b"].apply(lambda s: int(s.fillna(False).sum())),
    })
    return rep.round(3).reset_index()


def universe_holes(fund: pd.DataFrame, features: pd.DataFrame, bridges: list[pd.DataFrame], *,
                   cap_min: float = 1e8, cap_max: float = 2e9) -> pd.DataFrame:
    """Filers that look like universe members (public float in range, a trading symbol on file)
    but never appear in the price data – the names a survivorship-free source would add.

    A company priced at any start date is not a hole: after it delists it may keep filing
    for a while, and that is attrition, which the delisting rules already handle."""
    if fund is None or fund.empty:
        return pd.DataFrame(columns=["start_date", "in_range_filers", "unpriced", "unpriced_share"])
    syms = pd.concat([b for b in bridges if b is not None and len(b)], ignore_index=True) if bridges else pd.DataFrame()
    listed = set(syms["issuer_cik"].astype("int64")) if len(syms) else set()
    priced = set(features["cik"].dropna().astype("int64"))
    f = fund[pd.to_numeric(fund["public_float"], errors="coerce").between(cap_min, cap_max)]
    f = f[f["cik"].astype("int64").isin(listed)] if listed else f
    rows = []
    for d, grp in f.groupby("start_date"):
        ciks = set(grp["cik"].astype("int64"))
        miss = len(ciks - priced)
        rows.append({"start_date": d, "in_range_filers": len(ciks), "unpriced": miss,
                     "unpriced_share": round(miss / len(ciks), 4) if ciks else np.nan})
    return pd.DataFrame(rows)
