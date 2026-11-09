"""Point-in-time fundamentals at each formation date, from the FSDS extract.

For a formation date d, a company's anchor is its latest original 10-K or 10-Q filed
before d (and within `current_months`). Every value used is the latest one filed
before d for its period, so restatements count only once they are public.

Trailing twelve months for flows (operating cash flow, capex, revenue):
  * anchor is a 10-K (fiscal period FY):  TTM = annual value at the anchor period P
  * anchor is a 10-Q for quarter k (1-3): TTM = annual value at the prior fiscal year
    end (P - 3k months) + year-to-date at P - year-to-date at P - 12 months
The same formulas one year back give the prior TTM used for growth rates. FSDS rounds
every period end to a month-end, so periods are matched on a month index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.fsds import CLASS_SUMMED, PERIODIC_FORMS, TAGS

FP_QUARTERS = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}
FLOW_GROUPS = ("ocf", "capex", "revenue")
BALANCE_GROUPS = ("cash", "cash_sti", "sti", "debt_total", "debt_noncurrent", "debt_current", "short_borrowings")
LOW_DILUTION = 0.02


def month_index(ts) -> int:
    ts = pd.Timestamp(ts)
    return ts.year * 12 + ts.month - 1


def _ym_to_ts(ym) -> pd.Series:
    ym = pd.Series(ym, dtype="Int64")
    y, m = ym // 12, ym % 12 + 1
    out = pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": 1}), errors="coerce")
    return (out + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")


def prepare(facts: pd.DataFrame) -> dict:
    """Sort once by filing date and split off the per-class share counts."""
    f = facts.copy()
    f["filed"] = pd.to_datetime(f["filed"]).astype("datetime64[ns]")
    tag = f["tag"].astype(str).astype("category") if f["tag"].dtype != "category" else f["tag"]
    cats = [str(c) for c in tag.cat.categories]          # map once per tag, not once per row
    group_of = {c: TAGS.get(c, ("other", 99))[0] for c in cats}
    prio_of = {c: TAGS.get(c, ("other", 99))[1] for c in cats}
    f["tag"] = tag.cat.rename_categories(cats)
    f["group"] = f["tag"].map(group_of).astype("category")
    f["prio"] = f["tag"].map(prio_of).astype("int16")
    f = f.sort_values("filed", kind="mergesort").reset_index(drop=True)
    is_dim = f["dim"].astype(bool)
    is_dei = f["tag"].isin(list(CLASS_SUMMED))
    plain = f[~is_dim].reset_index(drop=True)
    dims = f[is_dim & is_dei].reset_index(drop=True)
    dei = f[~is_dim & is_dei].reset_index(drop=True)
    return {"plain": plain, "dims": dims, "dei": dei,
            "plain_filed": plain["filed"].to_numpy(dtype="datetime64[ns]"),
            "dims_filed": dims["filed"].to_numpy(dtype="datetime64[ns]"),
            "dei_filed": dei["filed"].to_numpy(dtype="datetime64[ns]")}


def _latest(prep: dict, cutoff: pd.Timestamp, min_ym: int) -> pd.DataFrame:
    n = int(np.searchsorted(prep["plain_filed"], np.datetime64(cutoff, "ns"), side="left"))
    f = prep["plain"].iloc[:n]
    f = f[f["ym"] >= min_ym]
    f = f.drop_duplicates(["cik", "tag", "ym", "qtrs"], keep="last")          # latest filed value
    f = f.sort_values("prio", kind="mergesort").drop_duplicates(["cik", "group", "ym", "qtrs"], keep="first")
    return f[["cik", "group", "tag", "ym", "qtrs", "value", "filed"]]


def _anchors(subs: pd.DataFrame, cutoff: pd.Timestamp, current_months: int) -> pd.DataFrame:
    s = subs[(subs["filed"] < cutoff) & subs["form"].isin(PERIODIC_FORMS)
             & ~subs["form"].astype(str).str.endswith("/A")]
    s = s.dropna(subset=["cik", "period", "filed"])
    s = s.sort_values(["filed", "period"], kind="mergesort").drop_duplicates("cik", keep="last")
    s = s[s["filed"] >= cutoff - pd.DateOffset(months=current_months)].copy()
    s["cik"] = s["cik"].astype("int64")
    s["P"] = (s["period"].dt.year * 12 + s["period"].dt.month - 1).astype("int64")
    k = s["fp"].astype(str).str.upper().map(FP_QUARTERS)
    k = k.where(k.notna(), np.where(s["form"].astype(str).str.startswith("10-K"), 4, np.nan))
    s["k"] = pd.to_numeric(k, errors="coerce")
    return s.reset_index(drop=True)


class _Lookup:
    def __init__(self, vals: pd.DataFrame):
        self.series = {g: d.set_index(["cik", "ym", "qtrs"])["value"] for g, d in vals.groupby("group", observed=True)}
        self.tags = {g: d.set_index(["cik", "ym", "qtrs"])["tag"].astype(str) for g, d in vals.groupby("group", observed=True)}

    def get(self, group: str, cik, ym, q, what: str = "series") -> np.ndarray:
        src = (self.series if what == "series" else self.tags).get(group)
        n = len(cik)
        if src is None:
            return np.full(n, np.nan) if what == "series" else np.full(n, None, dtype=object)
        ym = np.asarray(ym, dtype="float64")
        q = np.broadcast_to(np.asarray(q, dtype="float64"), (n,))
        ok = np.isfinite(ym) & np.isfinite(q)
        out = np.full(n, np.nan) if what == "series" else np.full(n, None, dtype=object)
        if ok.any():
            idx = pd.MultiIndex.from_arrays([np.asarray(cik)[ok].astype("int64"), ym[ok].astype("int64"),
                                             q[ok].astype("int64")])
            out[ok] = src.reindex(idx).to_numpy()
        return out


def _ttm(look: _Lookup, group: str, cik, P, k, back: int = 0) -> np.ndarray:
    """TTM ending `back` years before the anchor period."""
    P = np.asarray(P, dtype="float64") - 12 * back
    k = np.asarray(k, dtype="float64")
    annual = look.get(group, cik, P, 4)
    fy = look.get(group, cik, P - 3 * k, 4)
    ytd = look.get(group, cik, P, k)
    ytd_prev = look.get(group, cik, P - 12, k)
    return np.where(k == 4, annual, fy + ytd - ytd_prev)


def _typed_empty() -> pd.DataFrame:
    return pd.DataFrame({"cik": pd.Series(dtype="int64"), "value": pd.Series(dtype="float64"),
                         "filed": pd.Series(dtype="datetime64[ns]")})


def _dei_shares(prep: dict, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Latest cover-page share count filed before the cutoff; per-class values summed
    when the filing reports classes only."""
    cut = np.datetime64(cutoff, "ns")
    n = int(np.searchsorted(prep["dei_filed"], cut, side="left"))
    p = prep["dei"].iloc[:n]
    plain = p.drop_duplicates("cik", keep="last")[["cik", "value", "filed"]] if len(p) else _typed_empty()
    m = int(np.searchsorted(prep["dims_filed"], cut, side="left"))
    d = prep["dims"].iloc[:m]
    if len(d):
        d = d.groupby(["cik", "adsh"], as_index=False).agg(value=("value", "sum"), filed=("filed", "max"))
        d = d.sort_values("filed", kind="mergesort").drop_duplicates("cik", keep="last")[["cik", "value", "filed"]]
    else:
        d = _typed_empty()
    plain, d = plain.astype({"cik": "int64"}), d.astype({"cik": "int64"})
    both = plain.merge(d, on="cik", how="outer", suffixes=("_plain", "_dim"))
    use_dim = both["value_plain"].isna() | (both["filed_dim"] > both["filed_plain"])
    both["shares_dei"] = np.where(use_dim & both["value_dim"].notna(), both["value_dim"], both["value_plain"])
    both["shares_dei_filed"] = np.where(use_dim & both["value_dim"].notna(), both["filed_dim"], both["filed_plain"])
    out = both[["cik", "shares_dei", "shares_dei_filed"]].copy()
    out["shares_dei_filed"] = pd.to_datetime(out["shares_dei_filed"]).astype("datetime64[ns]")
    out["cik"] = out["cik"].astype("int64")
    return out


def restrict(prep: dict, ciks) -> dict:
    """The prepared facts of these companies only (order kept): a fast path for a few firms."""
    keep = pd.Index(pd.unique(np.asarray(list(ciks), dtype="int64")))
    out = {}
    for key in ("plain", "dims", "dei"):
        d = prep[key]
        d = d[d["cik"].astype("int64").isin(keep)].reset_index(drop=True)
        out[key] = d
        out[f"{key}_filed"] = d["filed"].to_numpy(dtype="datetime64[ns]")
    return out


def snapshot(prep: dict, subs: pd.DataFrame, cutoff, *, current_months: int = 15,
             extra_flows: tuple = ()) -> pd.DataFrame:
    """Fundamentals known before `cutoff` for every current filer.

    `extra_flows` adds trailing-twelve-month columns for more tag groups (Paper A:
    operating_income, dna, amortization, interest_expense), each with the tag used at
    the anchor period in `<group>_tag`; the default output is unchanged.
    """
    cutoff = pd.Timestamp(cutoff)
    a = _anchors(subs, cutoff, current_months)
    if a.empty:
        return pd.DataFrame()
    vals = _latest(prep, cutoff, month_index(cutoff) - 12 * 4)
    look = _Lookup(vals)
    cik, P, k = a["cik"].to_numpy(), a["P"].to_numpy(dtype="float64"), a["k"].to_numpy(dtype="float64")
    out = a.reindex(columns=["cik", "name", "sic", "form", "fp", "afs", "instance", "filed"]).copy()
    out["period_end"] = a["period"]
    for g in FLOW_GROUPS:
        out[g] = _ttm(look, g, cik, P, k)
        out[f"{g}_prev"] = _ttm(look, g, cik, P, k, back=1)
    for g in (x for x in extra_flows if x not in FLOW_GROUPS):
        out[g] = _ttm(look, g, cik, P, k)
        out[f"{g}_prev"] = _ttm(look, g, cik, P, k, back=1)
        out[f"{g}_tag"] = look.get(g, cik, P, k, what="tags")
    for g in BALANCE_GROUPS:
        out[g] = look.get(g, cik, P, 0)
    debt_current_tag = look.get("debt_current", cik, P, 0, what="tags")

    out["capex_missing"] = np.isnan(out["capex"].to_numpy())
    out["fcf"] = out["ocf"] - out["capex"].fillna(0.0)
    out["fcf_prev"] = out["ocf_prev"] - out["capex_prev"].fillna(0.0)
    cash = out["cash"].where(out["cash"].notna(), out["cash_sti"])
    sti = out["sti"].where(out["cash"].notna(), 0.0).fillna(0.0)
    out["cash_total"] = cash + sti
    debt_parts = out[["debt_total", "debt_noncurrent", "debt_current", "short_borrowings"]]
    out["debt_missing"] = debt_parts.isna().all(axis=1)
    base = out["debt_total"].where(out["debt_total"].notna(),
                                   out["debt_noncurrent"].fillna(0.0) + out["debt_current"].fillna(0.0))
    add_short = pd.Series(debt_current_tag, index=out.index).ne("DebtCurrent") | out["debt_total"].notna()
    out["debt"] = base.fillna(0.0) + out["short_borrowings"].fillna(0.0).where(add_short, 0.0)
    out["net_cash"] = out["cash_total"] - out["debt"]

    # Weighted-average diluted shares: latest quarter, else year-to-date, else annual – same basis both years.
    wad_now, wad_prev = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for q in (np.ones(len(a)), k, np.full(len(a), 4.0)):
        now, prev = look.get("shares_wad", cik, P, q), look.get("shares_wad", cik, P - 12, q)
        fill = np.isnan(wad_now) & np.isfinite(now) & np.isfinite(prev)
        wad_now[fill], wad_prev[fill] = now[fill], prev[fill]
    out["wad_now"], out["wad_prev"] = wad_now, wad_prev

    out = out.merge(_dei_shares(prep, cutoff), on="cik", how="left")
    pf = vals[vals["group"] == "public_float"].sort_values("filed").drop_duplicates("cik", keep="last")
    out = out.merge(pf[["cik", "value"]].rename(columns={"value": "public_float"}), on="cik", how="left")

    # No weighted-average pair: cover-page shares a year apart (point in time, same basis both years).
    prev = _dei_shares(prep, cutoff - pd.Timedelta(days=365))[["cik", "shares_dei"]].rename(
        columns={"shares_dei": "shares_dei_prev"})
    out = out.merge(prev, on="cik", how="left")
    have_wad = out["wad_now"].notna() & out["wad_prev"].notna()
    use_cover = ~have_wad & (out["shares_dei"] > 0) & (out["shares_dei_prev"] > 0)
    out["share_basis"] = np.where(have_wad, "weighted average", np.where(use_cover, "cover page", ""))
    out["wad_now"] = out["wad_now"].where(have_wad, out["shares_dei"].where(use_cover))
    out["wad_prev"] = out["wad_prev"].where(have_wad, out["shares_dei_prev"].where(use_cover))
    out["share_growth"] = out["wad_now"] / out["wad_prev"] - 1.0

    fcf_ps = out["fcf"] / out["wad_now"]
    fcf_ps_prev = out["fcf_prev"] / out["wad_prev"]
    g_rev = (out["revenue"] / out["revenue_prev"] - 1.0).where(out["revenue_prev"] > 0)
    g_fcf = fcf_ps / fcf_ps_prev - 1.0
    both_pos = (fcf_ps > 0) & (fcf_ps_prev > 0)
    known = fcf_ps.notna() & fcf_ps_prev.notna()
    s1 = pd.Series(pd.NA, index=out.index, dtype="boolean")
    s1[known & ~both_pos] = False
    cmp_ok = known & both_pos & g_rev.notna()
    s1[cmp_ok] = (g_fcf > g_rev)[cmp_ok]
    out["fcf_ps"], out["fcf_ps_prev"], out["revenue_growth"] = fcf_ps, fcf_ps_prev, g_rev
    out["s1_proxy"] = s1
    out["fcf_pos"] = (out["fcf"] > 0).astype("boolean").mask(out["fcf"].isna())
    out["net_cash_pos"] = (out["net_cash"] > 0).astype("boolean").mask(out["net_cash"].isna())
    out["low_dilution"] = (out["share_growth"] < LOW_DILUTION).astype("boolean").mask(out["share_growth"].isna())
    out["start_date"] = cutoff
    out = out.rename(columns={"filed": "fund_available"})
    return out


def dei_shares_at(facts: pd.DataFrame, cutoffs) -> pd.DataFrame:
    """(cik, start_date, shares_dei, shares_dei_filed): the latest cover-page share count filed
    before each cutoff (classes summed), from the share-count facts alone – a light version
    of the snapshot's shares for dates the snapshots do not cover (the pre-sample calibration)."""
    cols = ["cik", "start_date", "shares_dei", "shares_dei_filed"]
    if facts is None or not len(facts) or "tag" not in facts.columns:
        return pd.DataFrame(columns=cols)
    f = facts[facts["tag"].isin(list(CLASS_SUMMED))]
    if f.empty:
        return pd.DataFrame(columns=cols)
    prep = prepare(f)
    parts = []
    for c in cutoffs:
        s = _dei_shares(prep, pd.Timestamp(c))
        if len(s):
            parts.append(s.assign(start_date=pd.Timestamp(c)))
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out["start_date"] = pd.to_datetime(out["start_date"]).astype("datetime64[ns]")
    return out[cols]


def dei_share_history(facts: pd.DataFrame) -> pd.DataFrame:
    """(cik, asof, filed, shares): every cover-page share count on file, dated by the month
    it was counted in (month-end; FSDS rounds dates to month-ends), the class counts summed
    where a filing reports classes only, the latest filing kept per company and month.
    The SEC price panel reads splits from it (a split changes the count by its ratio)."""
    cols = ["cik", "asof", "filed", "shares"]
    if facts is None or not len(facts) or "tag" not in facts.columns:
        return pd.DataFrame(columns=cols)
    f = facts[facts["tag"].isin(list(CLASS_SUMMED))][["cik", "ym", "value", "dim", "filed", "adsh"]]
    f = f[pd.to_numeric(f["value"], errors="coerce") > 0]
    if f.empty:
        return pd.DataFrame(columns=cols)
    f = f.assign(dim=f["dim"].astype(bool), filed=pd.to_datetime(f["filed"]).astype("datetime64[ns]"))
    plain = (f[~f["dim"]].groupby(["cik", "adsh", "ym"], as_index=False)
             .agg(shares=("value", "max"), filed=("filed", "max")))
    dims = (f[f["dim"]].groupby(["cik", "adsh", "ym"], as_index=False)
            .agg(shares=("value", "sum"), filed=("filed", "max")))
    dims = dims.merge(plain[["cik", "adsh", "ym"]].assign(_p=True), on=["cik", "adsh", "ym"], how="left")
    both = pd.concat([plain, dims[dims["_p"].isna()].drop(columns="_p")], ignore_index=True)
    both["asof"] = _ym_to_ts(both["ym"]).to_numpy()
    both = (both.sort_values(["cik", "asof", "filed"], kind="mergesort")
            .drop_duplicates(["cik", "asof"], keep="last"))
    both["cik"] = both["cik"].astype("int64")
    both["shares"] = both["shares"].astype("float64")
    return both[cols].reset_index(drop=True)


def snapshots(facts: pd.DataFrame, subs: pd.DataFrame, cutoffs, *, current_months: int = 15,
              log=None, extra_flows: tuple = (), prep: dict | None = None) -> pd.DataFrame:
    prep = prep if prep is not None else prepare(facts)
    subs = subs.copy()
    subs["filed"] = pd.to_datetime(subs["filed"]).astype("datetime64[ns]")
    subs["period"] = pd.to_datetime(subs["period"]).astype("datetime64[ns]")
    parts = []
    for c in cutoffs:
        snap = snapshot(prep, subs, c, current_months=current_months, extra_flows=extra_flows)
        if len(snap):
            parts.append(snap)
        if log:
            log(f"  fundamentals {pd.Timestamp(c).date()}: {len(snap):,} filers")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
