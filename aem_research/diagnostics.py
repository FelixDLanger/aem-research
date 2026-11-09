"""Why a universe is the size it is – counts and ratios only, no returns (blind-safe).

`universe_funnel` walks the pre-registered universe filters in order and counts the
windows that pass each filter alone, every filter so far, and every filter but that
one. The binding filter is the one whose removal brings most windows back. "shares
known" is a step of the funnel, not a filter of its own: the market-cap filter needs
a share count, so leaving out the market-cap filter leaves out that need too.

`compare_universes` takes the vendor universe's windows (B) and looks up the same
company and start date in the SEC panel's features (A): is the company there, which A
filter does it fail first, and how do A's unadjusted price, share count, market cap and
13F breadth compare with the vendor row's. Nothing here reads a price after the start
date, so it can run before the pre-registration is locked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORDER = ("filer", "not a SPAC shell", "price >= min", "shares known", "market cap in range", "liquid", "listed")
FILTERS = tuple(k for k in ORDER if k != "shares known")
SHORT = {"filer": "filer", "not a SPAC shell": "not_spac", "price >= min": "price", "shares known": "shares",
         "market cap in range": "mcap", "liquid": "liquid", "listed": "listed"}
FRESH_MONTHS = 15          # features.assemble: cover-page shares filed within 15 months of the start date


def _col(f: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return f[name] if name in f.columns else pd.Series(default, index=f.index)


def _num(s: pd.Series) -> pd.Series:
    """Plain float64 (nullable integers and objects included), missing = NaN."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _flag(s: pd.Series) -> pd.Series:
    """A yes/no column as plain bool, missing = no."""
    return s.astype("boolean").fillna(False).astype(bool)


def _since(f: pd.DataFrame, start) -> pd.DataFrame:
    return f if start is None else f[pd.to_datetime(f["start_date"]) >= pd.Timestamp(start)]


def filter_flags(f: pd.DataFrame, cfg_data: dict) -> pd.DataFrame:
    """One bool column per funnel step (ORDER), as features.assemble applies the filters."""
    u = cfg_data.get("universe", {})
    cap = u.get("market_cap_usd", {}) or {}
    sic = _num(_col(f, "sic"))
    price = _num(_col(f, "entry_price_unadjusted"))
    mcap = _num(_col(f, "market_cap"))
    out = pd.DataFrame(index=f.index)
    out["filer"] = _col(f, "fund_available", pd.NaT).notna()
    out["not a SPAC shell"] = ~sic.eq(6770).fillna(False).astype(bool)
    out["price >= min"] = (price >= float(u.get("min_price_usd", 1.0))).fillna(False)
    out["shares known"] = _num(_col(f, "shares")).gt(0)
    out["market cap in range"] = mcap.between(float(cap.get("min", 1e8)), float(cap.get("max", 2e9))).fillna(False)
    out["liquid"] = _flag(_col(f, "liquid", True))
    out["listed"] = _flag(_col(f, "listed", True))
    return out.astype(bool)


def universe_funnel(f: pd.DataFrame, cfg_data: dict, *, start=None) -> pd.DataFrame:
    """Per step: windows passing it alone, every step up to it, and every filter but it
    (blank for "shares known", which is not a filter of its own). The last row is the
    universe flag itself; it equals the last cumulative count unless the flag was built
    from something these filters do not see."""
    f = _since(f, start)
    flags = filter_flags(f, cfg_data)
    n = len(f)
    arr = {k: flags[k].to_numpy() for k in ORDER}
    rows, cum = [], np.ones(n, dtype=bool)
    for k in ORDER:
        cum = cum & arr[k]
        without = np.nan
        if k in FILTERS:
            keep = np.ones(n, dtype=bool)
            for j in FILTERS:
                if j != k:
                    keep &= arr[j]
            without = int(keep.sum())
        rows.append({"filter": k, "alone": int(arr[k].sum()), "so_far": int(cum.sum()), "all_but_this": without})
    uni = _flag(_col(f, "in_universe", False)).to_numpy()
    rows.append({"filter": "universe flag", "alone": int(uni.sum()), "so_far": int(uni.sum()), "all_but_this": np.nan})
    out = pd.DataFrame(rows)
    out.attrs["rows"] = n
    return out


def binding_filter(funnel: pd.DataFrame) -> tuple[str, int]:
    """The filter whose removal brings back most windows, and how many it brings back."""
    body = funnel[funnel["filter"].isin(FILTERS)]
    if body.empty:
        return "", 0
    final = int(funnel.loc[funnel["filter"] == "listed", "so_far"].iloc[0]) if "listed" in set(funnel["filter"]) else 0
    gain = (body["all_but_this"].astype(float) - final)
    i = int(gain.to_numpy().argmax())
    return str(body["filter"].iloc[i]), int(gain.iloc[i])


def funnel_summary(funnel: pd.DataFrame) -> dict:
    """Paste-block form: rows, each step's cumulative count, the flag, the count without each
    filter, and the binding filter."""
    out = {"rows": int(funnel.attrs.get("rows", 0))}
    for r in funnel[funnel["filter"].isin(ORDER)].itertuples(index=False):
        out[SHORT[r.filter]] = int(r.so_far)
    flag = funnel.loc[funnel["filter"] == "universe flag", "so_far"]
    out["flag"] = int(flag.iloc[0]) if len(flag) else None
    for r in funnel[funnel["filter"].isin(FILTERS)].itertuples(index=False):
        out[f"without_{SHORT[r.filter]}"] = int(r.all_but_this)
    name, gain = binding_filter(funnel)
    out["binding"] = f"{SHORT.get(name, name)} (+{gain:,} without it)" if name else "none"
    return out


def _q(s: pd.Series, qs=(0.1, 0.5, 0.9)) -> str:
    x = _num(pd.Series(s)).to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "n/a"
    return "/".join(f"{v:.3g}" for v in np.quantile(x, list(qs)))


def _share(mask) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    return round(float(mask.mean()), 4) if mask.size else None


def dei_fresh(f: pd.DataFrame) -> np.ndarray:
    """Cover-page share count known and filed within FRESH_MONTHS of the start date."""
    if "shares_dei" not in f.columns or "shares_dei_filed" not in f.columns:
        return np.zeros(len(f), dtype=bool)
    known = _num(f["shares_dei"]).gt(0).to_numpy()
    filed = pd.to_datetime(f["shares_dei_filed"])
    start = pd.to_datetime(f["start_date"])
    return known & (filed >= start - pd.DateOffset(months=FRESH_MONTHS)).fillna(False).to_numpy()


def detail(f: pd.DataFrame, cfg_data: dict, *, start=None, min_breadth=None) -> dict:
    """What lies under the filters: share counts known and fresh, size, price and breadth
    levels (10th/50th/90th percentiles), on the rows that pass filer, SPAC and price
    (base) and on those also in the size range (sized)."""
    f = _since(f, start)
    if f.empty:
        return {}
    flags = filter_flags(f, cfg_data)
    base = (flags["filer"] & flags["not a SPAC shell"] & flags["price >= min"]).to_numpy()
    sized = base & flags["market cap in range"].to_numpy()
    out = {"rows": len(f), "base_rows": int(base.sum()), "sized_rows": int(sized.sum())}
    if base.any():
        if "shares_dei" in f.columns:
            out["dei_known_base"] = _share(_num(f["shares_dei"]).gt(0).to_numpy()[base])
            out["dei_fresh_base"] = _share(dei_fresh(f)[base])
        if "shares_dei_filed" in f.columns:
            age = (pd.to_datetime(f["start_date"]) - pd.to_datetime(f["shares_dei_filed"])).dt.days
            out["dei_age_days_base"] = _q(age[base])
        if "shares_source" in f.columns:
            src = f["shares_source"].astype(str).to_numpy()[base]
            out["shares_from_filing_base"] = _share(src == "filing")
            out["shares_from_vendor_base"] = _share(src == "vendor")
        if "wad_now" in f.columns:
            out["wad_known_base"] = _share(_num(f["wad_now"]).gt(0).to_numpy()[base])
        out["mcap_musd_base"] = _q(_num(_col(f, "market_cap"))[base] / 1e6)
        out["price_base"] = _q(_num(_col(f, "entry_price_unadjusted"))[base])
    if "breadth" in f.columns and sized.any():
        b = _num(f["breadth"])
        out["breadth_known_sized"] = _share(b.notna().to_numpy()[sized])
        out["breadth_sized"] = _q(b[sized])
        if min_breadth is not None:
            out[f"breadth_ge_{int(min_breadth)}_sized"] = _share((b >= float(min_breadth)).to_numpy()[sized])
    if "dollar_volume_20d" in f.columns and sized.any():
        dv = _num(f["dollar_volume_20d"])
        if dv[sized].notna().any():
            u = cfg_data.get("universe", {})
            out["dollar_volume_k_sized"] = _q(dv[sized] / 1e3)
            out["volume_pass_sized"] = _share((dv >= float(u.get("min_median_dollar_volume_20d_usd", 1e5)))
                                              .to_numpy()[sized])
    if "listed" in f.columns and sized.any():
        out["listed_sized"] = _share(_flag(f["listed"]).to_numpy()[sized])
    return out


_A_COLS = ["cik", "start_date", "ticker", "in_universe", "fund_available", "sic", "entry_price_unadjusted", "shares",
           "shares_dei", "shares_dei_filed", "market_cap", "breadth", "liquid", "listed"]
_V_COLS = ["cik", "start_date", "ticker", "in_universe", "entry_price_unadjusted", "shares", "shares_source",
           "market_cap", "breadth", "dollar_volume_20d"]
_BOTH = ("ticker", "in_universe", "entry_price_unadjusted", "shares", "market_cap", "breadth")


def _keyed(f: pd.DataFrame, cols) -> pd.DataFrame:
    x = f[[c for c in cols if c in f.columns]].copy()
    x["cik"] = pd.to_numeric(x["cik"], errors="coerce")
    x = x[x["cik"].notna()]
    x["cik"] = x["cik"].astype("int64")
    x["start_date"] = pd.to_datetime(x["start_date"]).astype("datetime64[ns]")
    return x


def compare_universes(a: pd.DataFrame, v: pd.DataFrame, cfg_data: dict, *, start=None, min_breadth=None) -> dict:
    """The vendor universe's windows, looked up in A by (cik, start date): found or not, the
    first A step each fails, and A over vendor ratios of price, shares, size and breadth
    (median; share within 10%)."""
    vu = _since(v[_flag(v["in_universe"])], start)
    vu = _keyed(vu, _V_COLS)
    if vu.empty:
        return {"vendor_windows": 0}
    ak = _keyed(a, _A_COLS)
    # a company with two SEC series on one date: the one in A's universe first
    ak = (ak.assign(_u=_flag(_col(ak, "in_universe", False)))
          .sort_values(["cik", "start_date", "_u"], ascending=[True, True, False], kind="mergesort")
          .drop_duplicates(["cik", "start_date"]).drop(columns="_u"))
    m = vu.merge(ak, on=["cik", "start_date"], how="left", suffixes=("_v", "_a"), indicator=True)
    found = (m["_merge"] == "both").to_numpy()
    out = {"vendor_windows": len(vu), "found_in_a": _share(found),
           "cik_anywhere_in_a": _share(vu["cik"].isin(set(ak["cik"])).to_numpy())}
    if not found.any():
        return out
    mf = m[found].reset_index(drop=True)
    a_side = mf.rename(columns={c + "_a": c for c in _BOTH})
    flags = filter_flags(a_side, cfg_data)
    out["a_in_universe"] = _share(_flag(a_side["in_universe"]).to_numpy())
    for k in ORDER:
        out[f"a_pass_{SHORT[k]}"] = _share(flags[k].to_numpy())
    first = np.full(len(flags), "none", dtype=object)
    for k in reversed(ORDER):
        first = np.where(~flags[k].to_numpy(), SHORT[k], first)
    vc = pd.Series(first).value_counts(normalize=True)
    out["a_first_fail"] = ", ".join(f"{k} {x:.1%}" for k, x in vc.items())

    def ratio(col):
        x = (_num(mf[f"{col}_a"]) / _num(mf[f"{col}_v"])).to_numpy()
        x = x[np.isfinite(x) & (x > 0)]
        if x.size == 0:
            return "n/a"
        within = float((np.abs(np.log(x)) <= np.log(1.1)).mean())
        return f"{np.median(x):.3g} (within 10% {within:.0%}, n {x.size:,})"

    out["price_a_over_v"] = ratio("entry_price_unadjusted")
    out["shares_a_over_v"] = ratio("shares")
    out["mcap_a_over_v"] = ratio("market_cap")
    ba, bv = _num(mf["breadth_a"]), _num(mf["breadth_v"])
    both = (ba.notna() & bv.notna()).to_numpy()
    out["breadth_known_a_v"] = f"{ba.notna().mean():.1%}/{bv.notna().mean():.1%}"
    out["breadth_equal"] = _share((ba.to_numpy()[both] == bv.to_numpy()[both])) if both.any() else None
    out["breadth_median_a_v"] = f"{ba.median():.4g}/{bv.median():.4g}"
    if min_breadth is not None:
        out[f"breadth_ge_{int(min_breadth)}_a_v"] = f"{(ba >= min_breadth).mean():.1%}/{(bv >= min_breadth).mean():.1%}"
    if "shares_source" in mf.columns:
        src = mf["shares_source"].astype(str).to_numpy()
        out["v_shares_from_vendor"] = _share(src == "vendor")
        no_a = ~flags["shares known"].to_numpy()
        out["v_vendor_shares_where_a_has_none"] = _share((src == "vendor")[no_a]) if no_a.any() else None
    if "shares_dei" in mf.columns:
        out["a_dei_known"] = _share(_num(mf["shares_dei"]).gt(0).to_numpy())
        out["a_dei_fresh"] = _share(dei_fresh(mf))
    return out


def what_if(f: pd.DataFrame, cfg_data: dict, *, start=None, ks=(3, 10, 20, 30, 40, 60)) -> dict:
    """Universe size under other choices – counts only: as run; the filters recomputed here
    (must match); share counts falling back to the filing's weighted-average diluted shares
    where no fresh cover-page count exists; other 13F-holder minimums; no listing filter."""
    f = _since(f, start)
    if f.empty:
        return {}
    u = cfg_data.get("universe", {})
    cap = u.get("market_cap_usd", {}) or {}
    flags = filter_flags(f, cfg_data)
    core = (flags["filer"] & flags["not a SPAC shell"] & flags["price >= min"]).to_numpy()
    mcap_ok = flags["market cap in range"].to_numpy()
    liq, lst = flags["liquid"].to_numpy(), flags["listed"].to_numpy()
    out = {"as_run": int(_flag(_col(f, "in_universe", False)).sum()),
           "recomputed": int((core & mcap_ok & liq & lst).sum())}
    if "wad_now" in f.columns:
        price, shares = _num(_col(f, "entry_price_unadjusted")), _num(_col(f, "shares"))
        alt = shares.where(shares > 0).fillna(_num(f["wad_now"]))
        m2 = (price * alt).between(float(cap.get("min", 1e8)), float(cap.get("max", 2e9))).to_numpy()
        out["shares_or_wad"] = int((core & m2 & liq & lst).sum())
    if "breadth" in f.columns:
        b = _num(f["breadth"])
        for k in ks:
            out[f"holders_{k}"] = int((core & mcap_ok & lst & (b >= k).to_numpy()).sum())
    out["no_listing"] = int((core & mcap_ok & liq).sum())
    out["no_liquidity"] = int((core & mcap_ok & lst).sum())
    return out


def examples(a: pd.DataFrame, v: pd.DataFrame, cfg_data: dict, *, start=None, n: int = 8, seed: int = 0) -> pd.DataFrame:
    """A few vendor-universe windows whose A row fails, with A's values (SEC data) and the
    A-over-vendor ratios – to see the failure, not to analyse it."""
    vu = _keyed(_since(v[_flag(v["in_universe"])], start), _V_COLS)
    ak = _keyed(a, _A_COLS)
    ak = ak[~_flag(_col(ak, "in_universe", False))].drop_duplicates(["cik", "start_date"])
    m = vu.merge(ak, on=["cik", "start_date"], how="inner", suffixes=("_v", "_a"))
    if m.empty:
        return pd.DataFrame()
    m = m.sample(min(n, len(m)), random_state=seed).reset_index(drop=True)
    a_side = m.rename(columns={c + "_a": c for c in _BOTH})
    flags = filter_flags(a_side, cfg_data)
    first = np.full(len(flags), "none", dtype=object)
    for k in reversed(ORDER):
        first = np.where(~flags[k].to_numpy(), SHORT[k], first)
    out = pd.DataFrame({"start_date": m["start_date"].dt.date, "cik": m["cik"], "a_ticker": m["ticker_a"],
                        "a_fails": first, "a_price": m["entry_price_unadjusted_a"].round(2),
                        "a_shares_m": (_num(m["shares_a"]) / 1e6).round(2),
                        "a_mcap_m": (_num(m["market_cap_a"]) / 1e6).round(1),
                        "a_breadth": m["breadth_a"], "v_breadth": m["breadth_v"],
                        "a_listed": _col(m, "listed", np.nan),
                        "price_a_over_v": (_num(m["entry_price_unadjusted_a"]) / _num(m["entry_price_unadjusted_v"])).round(3),
                        "shares_a_over_v": (_num(m["shares_a"]) / _num(m["shares_v"])).round(3)})
    if "shares_dei_filed" in m.columns:
        out["dei_age_days"] = (m["start_date"] - pd.to_datetime(m["shares_dei_filed"])).dt.days
    return out


def dtypes_line(f: pd.DataFrame, cols=("cik", "start_date", "breadth", "listed", "liquid", "shares_dei_filed",
                                       "market_cap", "in_universe")) -> dict:
    return {c: str(f[c].dtype) for c in cols if c in f.columns}


__all__ = ["ORDER", "FILTERS", "filter_flags", "universe_funnel", "binding_filter", "funnel_summary", "detail",
           "dei_fresh", "compare_universes", "what_if", "examples", "dtypes_line"]
