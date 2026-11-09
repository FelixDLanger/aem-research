"""Splits from the filings' restated history (D10).

After a split, every 10-Q and 10-K a company files restates the share counts of earlier
periods by the split ratio: the weighted-average shares behind earnings per share are
adjusted retroactively for all periods presented (ASC 260-10-55-12), and the balance
sheet's and equity statement's share counts with them. The same period's count, as one
filing reported it and as a later filing reports it again, therefore differs by the
product of the splits between the two filing dates and by nothing else: the period is
history, so neither share issuance nor the stock price enters it. A count read at two
different dates (fundamentals.filing_share_history) does move with issuance, which
shrinks a reverse split – distressed companies issue shares soon after one – and prices
read across a CUSIP change drift with the stock.

The SEC Financial Statement Data Sets carry every comparative column, so the extract
(aem_research.data.fsds) already holds both reports. From consecutive reports of the same
fact (tag, period end, duration) by one company, amendments left out:
  * a ratio within 1% of 1 in the weighted-average shares says no split happened between
    the two filing dates – only these must be restated (ASC 260); a balance-sheet or
    equity-statement count may stay as first reported, with the split shown as a
    movement (the Nov 9 run: such counts denied about 60 splits the holders and the
    vendors both showed), so they never rule a split out;
  * a ratio within 1% of a standard split ratio of at least 1.2 either way, plus what the
    reported values' rounding allows, says one did, by that ratio;
  * anything else (a correction, the new history after a reverse merger) says nothing.
Observations of one split merge into an event whose window is the intersection of theirs,
widened by a few days for filings made just around the split date. Filing dates fall
between quarter-ends, so a window usually holds two of them; aem_research.secprices
places each event at one pair of quarter-ends with the 13F holders and the price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import splits as sp
from .data.fsds import TAG_GROUPS

# history, reported again as comparatives: not the cover page, a count at the filing date
TAGS = TAG_GROUPS["shares_wad"] + TAG_GROUPS["shares_bs"]
N_WAD = len(TAG_GROUPS["shares_wad"])     # TAGS[:N_WAD]: the weighted-average shares (tag code = key % 8)
SAME_TOL = 0.01          # a ratio within 1% of 1: no split between the two filings
RATIO_TOL = 0.01         # within 1% of a standard ratio (plus rounding): a split
ROUND_CAP = 0.02         # the rounding allowance of one reported value, at most 2%
MAX_DAYS = 800           # reports further apart than this are not paired
PRE_DAYS = 10            # an event's window opens this many days before the earlier filing (one filed days
POST_DAYS = 20           # after a split may not restate yet) and closes this many after the later one (one
#                          filed days before a split may already restate)
SAME_INSET = (20, 10)    # a no-split span leaves out this many days after the earlier filing and before the later
_YM0 = 2000 * 12

OBS_COLS = ["cik", "lo", "hi", "ratio", "kind", "split", "key"]
EVENT_COLS = ["cik", "lo", "hi", "ratio", "n_obs", "n_periods"]
SPAN_COLS = ["cik", "lo", "hi"]


def _empty(cols, types) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(cols, types)})


def _rounding(v) -> np.ndarray:
    """Relative rounding allowance of reported values: half the place value of the last
    nonzero digit over the value (12,300,000 may be 12.3 million rounded), at most ROUND_CAP."""
    a = np.abs(np.round(np.asarray(v, dtype="float64")))
    tz = np.zeros(a.shape, dtype="float64")
    for k in range(1, 10):
        tz += (a > 0) & (np.mod(a, 10.0 ** k) == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 0.5 * np.power(10.0, tz) / a
    return np.minimum(np.where(a > 0, rel, ROUND_CAP), ROUND_CAP)


def _tag_codes(tags: pd.Series) -> np.ndarray:
    code = {t: i for i, t in enumerate(TAGS)}
    if isinstance(tags.dtype, pd.CategoricalDtype):
        cat = np.array([code.get(str(c), -1) for c in tags.cat.categories], dtype="int64")
        k = tags.cat.codes.to_numpy()
        return np.where(k >= 0, cat[np.clip(k, 0, None)] if cat.size else -1, -1)
    return tags.astype(str).map(code).fillna(-1).to_numpy(dtype="int64")


def observations(facts: pd.DataFrame | None, subs: pd.DataFrame | None = None, *,
                 max_days: int = MAX_DAYS) -> pd.DataFrame:
    """(cik, lo, hi, ratio, kind, split, key): consecutive reports of the same share fact
    (tag, period end, duration) by one company, filed on `lo` and on `hi`; `ratio` is the
    later value over the earlier; `kind` is 'same', 'split' (`split` holds the standard
    ratio) or 'other'. Reports filed on one day are one report (their median); amendments
    (forms ending in /A, when `subs` is given) are left out – one may re-file old numbers."""
    empty = _empty(OBS_COLS, ["int64", "datetime64[ns]", "datetime64[ns]", "float64", "object", "float64", "int64"])
    if facts is None or not len(facts) or "tag" not in facts.columns:
        return empty
    # one subset of the (large) facts table, then plain arrays
    mask = facts["tag"].isin(list(TAGS)).to_numpy()
    if "dim" in facts.columns:
        mask = mask & ~facts["dim"].to_numpy(dtype=bool)          # (pandas may hand out read-only arrays)
    if not mask.any():
        return empty
    f = facts.loc[mask, [c for c in ("cik", "tag", "ym", "qtrs", "value", "filed", "adsh") if c in facts.columns]]
    tag = _tag_codes(f["tag"])
    cik = f["cik"].to_numpy(dtype="int64")
    ym = f["ym"].to_numpy(dtype="int64") - _YM0
    qtrs = np.clip(pd.to_numeric(f["qtrs"], errors="coerce").fillna(-1).to_numpy(dtype="int64"), -1, 6) + 1
    val = pd.to_numeric(f["value"], errors="coerce").to_numpy(dtype="float64")
    filed = pd.to_datetime(f["filed"]).to_numpy(dtype="datetime64[ns]")
    day = filed.astype("datetime64[D]").astype("int64")
    ok = (tag >= 0) & (ym >= 0) & (ym < 4096) & (cik >= 0) & np.isfinite(val) & (val > 0) & ~np.isnat(filed)
    if subs is not None and len(subs) and {"adsh", "form"} <= set(subs.columns) and "adsh" in f.columns:
        amended = set(subs.loc[subs["form"].astype(str).str.endswith("/A"), "adsh"].astype(str))
        if amended:
            ok = ok & ~f["adsh"].isin(amended).to_numpy()
    del f
    tag, cik, ym, qtrs, val, day = tag[ok], cik[ok], ym[ok], qtrs[ok], val[ok], day[ok]
    if not cik.size:
        return empty
    key = ((cik * 4096 + ym) * 8 + qtrs) * 8 + tag
    order = np.lexsort((val, day, key))
    key, day, val, cik = key[order], day[order], val[order], cik[order]
    # one report per fact and filing day: the (lower) median of that day's values
    first = np.r_[True, (key[1:] != key[:-1]) | (day[1:] != day[:-1])]
    st = np.flatnonzero(first)
    en = np.r_[st[1:], key.size]
    mid = (st + en - 1) // 2
    key, day, val, cik = key[mid], day[mid], val[mid], cik[mid]
    nxt = np.flatnonzero(key[1:] == key[:-1]) + 1
    prv = nxt - 1
    keep = (day[nxt] - day[prv]) <= max_days
    nxt, prv = nxt[keep], prv[keep]
    if not nxt.size:
        return empty
    ratio = val[nxt] / val[prv]
    std, dist = sp.snap(ratio)
    tol = np.log1p(RATIO_TOL + _rounding(val[prv]) + _rounding(val[nxt]))
    same = np.abs(np.log(ratio)) <= np.log1p(SAME_TOL)
    split = ~same & (dist <= tol) & (np.abs(np.log(std)) >= np.log(sp.MIN_MOVE))
    as_ns = lambda d: d.astype("datetime64[D]").astype("datetime64[ns]")           # noqa: E731
    return pd.DataFrame({"cik": cik[nxt], "lo": as_ns(day[prv]), "hi": as_ns(day[nxt]), "ratio": ratio,
                         "kind": np.where(same, "same", np.where(split, "split", "other")),
                         "split": np.where(split, std, np.nan), "key": key[nxt]})


def _greedy(cik, r, lo_, hi_, per) -> list:
    """One company's observations of one ratio, sorted by start: each joins the running
    event while its window overlaps the event's, else starts the next one."""
    rows, L, H, n, pers = [], None, None, 0, set()
    for a, b, p in zip(lo_, hi_, per):
        if L is not None and a < H and b > L:
            L, H = max(L, a), min(H, b)
            n += 1
            pers.add(int(p))
            continue
        if L is not None:
            rows.append((int(cik), L, H, float(r), n, len(pers)))
        L, H, n, pers = a, b, 1, {int(p)}
    rows.append((int(cik), L, H, float(r), n, len(pers)))
    return rows


def _events(obs: pd.DataFrame) -> pd.DataFrame:
    """Split observations of one company and ratio whose windows overlap, as one event: the
    window is the intersection of theirs, widened by PRE_DAYS / POST_DAYS."""
    s = obs[obs["kind"] == "split"]
    if s.empty:
        return _empty(EVENT_COLS, ["int64", "datetime64[ns]", "datetime64[ns]", "float64", "int64", "int64"])
    pre, post = np.timedelta64(PRE_DAYS, "D"), np.timedelta64(POST_DAYS, "D")
    s = s.sort_values(["cik", "split", "lo", "hi"], kind="mergesort")
    cik = s["cik"].to_numpy(dtype="int64")
    r = s["split"].to_numpy(dtype="float64")
    lo_ = s["lo"].to_numpy(dtype="datetime64[ns]") - pre
    hi_ = s["hi"].to_numpy(dtype="datetime64[ns]") + post
    per = s["key"].to_numpy(dtype="int64") // 8                             # (company, period end, duration)
    # clusters of overlapping windows per company and ratio; one split's windows all hold its date,
    # so their intersection is not empty – a cluster whose intersection is (two splits of one ratio
    # close together, noise) is split up one observation at a time
    grp = np.r_[True, (cik[1:] != cik[:-1]) | (r[1:] != r[:-1])]
    reach = pd.Series(hi_).groupby(np.cumsum(grp)).cummax().to_numpy(dtype="datetime64[ns]")
    new = grp.copy()
    new[1:] |= lo_[1:] >= reach[:-1]
    cl = np.cumsum(new) - 1
    d = pd.DataFrame({"cl": cl, "cik": cik, "ratio": r, "lo": lo_, "hi": hi_, "per": per})
    agg = d.groupby("cl", sort=True).agg(cik=("cik", "first"), ratio=("ratio", "first"), lo=("lo", "max"),
                                          hi=("hi", "min"), n_obs=("lo", "size"), n_periods=("per", "nunique"))
    good = agg["lo"] < agg["hi"]
    parts = [agg[good][EVENT_COLS]]
    bad = agg.index[~good.to_numpy()]
    if len(bad):
        rows = []
        for c_, g in d[d["cl"].isin(set(bad))].groupby("cl", sort=False):
            rows += _greedy(g["cik"].iloc[0], g["ratio"].iloc[0], g["lo"].to_numpy(), g["hi"].to_numpy(),
                            g["per"].to_numpy())
        parts.append(pd.DataFrame(rows, columns=EVENT_COLS))
    ev = pd.concat(parts, ignore_index=True)
    for c in ("lo", "hi"):
        ev[c] = pd.to_datetime(ev[c]).astype("datetime64[ns]")
    ev["cik"] = ev["cik"].astype("int64")
    ev[["n_obs", "n_periods"]] = ev[["n_obs", "n_periods"]].astype("int64")
    return ev.sort_values(["cik", "lo"], kind="mergesort").reset_index(drop=True)


def _spans(obs: pd.DataFrame) -> pd.DataFrame:
    """The no-split observations of each company's weighted-average shares, inset by
    SAME_INSET and merged where they overlap or touch: no split happened in (lo, hi]."""
    s = obs[(obs["kind"] == "same").to_numpy() & ((obs["key"].to_numpy(dtype="int64") % 8) < N_WAD)]
    if s.empty:
        return _empty(SPAN_COLS, ["int64", "datetime64[ns]", "datetime64[ns]"])
    a = s["lo"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(SAME_INSET[0], "D")
    b = s["hi"].to_numpy(dtype="datetime64[ns]") - np.timedelta64(SAME_INSET[1], "D")
    d = pd.DataFrame({"cik": s["cik"].to_numpy(dtype="int64"), "lo": a, "hi": b})
    d = d[d["lo"] < d["hi"]].sort_values(["cik", "lo"], kind="mergesort").reset_index(drop=True)
    if d.empty:
        return _empty(SPAN_COLS, ["int64", "datetime64[ns]", "datetime64[ns]"])
    reach = d.groupby("cik")["hi"].cummax()
    prev = reach.groupby(d["cik"]).shift(1)
    new = prev.isna().to_numpy() | (d["lo"].to_numpy() > prev.to_numpy(dtype="datetime64[ns]"))
    sid = np.cumsum(new)
    out = d.groupby(sid).agg(cik=("cik", "first"), lo=("lo", "min"), hi=("hi", "max")).reset_index(drop=True)
    return out[SPAN_COLS]


def evidence(facts: pd.DataFrame | None, subs: pd.DataFrame | None = None, *,
             obs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(events, spans, info). events: cik, lo, hi, ratio, n_obs, n_periods – a split of
    `ratio` happened in (lo, hi], seen in `n_obs` pairs of reports of `n_periods` periods.
    spans: cik, lo, hi – no split happened in (lo, hi]."""
    o = observations(facts, subs) if obs is None else obs
    ev, spans = _events(o), _spans(o)
    k = o["kind"].value_counts() if len(o) else pd.Series(dtype="int64")
    wad = (o["key"].to_numpy(dtype="int64") % 8) < N_WAD if len(o) else np.zeros(0, dtype=bool)
    info = {"observations": int(len(o)), "obs_same": int(k.get("same", 0)),
            "obs_same_wad": int(((o["kind"] == "same").to_numpy() & wad).sum()) if len(o) else 0,
            "obs_split": int(k.get("split", 0)),
            "obs_other": int(k.get("other", 0)), "events": int(len(ev)),
            "events_two_periods": int((ev["n_periods"] >= 2).sum()) if len(ev) else 0,
            "companies_with_events": int(ev["cik"].nunique()) if len(ev) else 0,
            "companies_with_history": int(o["cik"].nunique()) if len(o) else 0}
    return ev, spans, info


_NO_DATES = np.array([], dtype="datetime64[ns]")


def _by_cik(df: pd.DataFrame, cols: list, want) -> dict:
    """cik -> tuple of arrays (one per column), rows sorted by `lo`."""
    if df is None or not len(df):
        return {}
    d = df if want is None else df[df["cik"].isin(want)]
    if d.empty:
        return {}
    d = d.sort_values(["cik", "lo"], kind="mergesort")
    c = d["cik"].to_numpy(dtype="int64")
    arrs = [d[k].to_numpy(dtype="datetime64[ns]") if k in ("lo", "hi") else d[k].to_numpy() for k in cols]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ends = np.r_[starts[1:], c.size]
    return {int(c[a]): tuple(x[a:b] for x in arrs) for a, b in zip(starts, ends)}


def index(events: pd.DataFrame | None, spans: pd.DataFrame | None, ciks=None) -> dict:
    """cik -> (events: (lo, hi, ratio, n_obs, n_periods) arrays or None, span starts, span
    ends), for lookups."""
    want = None if ciks is None else {int(c) for c in ciks}
    ev = _by_cik(events, ["lo", "hi", "ratio", "n_obs", "n_periods"], want)
    sp_ = _by_cik(spans, ["lo", "hi"], want)
    return {c: (ev.get(c), *(sp_.get(c) or (_NO_DATES, _NO_DATES))) for c in set(ev) | set(sp_)}


def _ns(t) -> np.datetime64:
    return np.datetime64(pd.Timestamp(t), "ns")


def events_of(idx: dict, cik) -> list:
    """The company's events as (lo, hi, ratio, n_obs, n_periods) tuples, by start."""
    x = idx.get(int(cik))
    if x is None or x[0] is None:
        return []
    lo, hi, r, n, p = x[0]
    return [(lo[k], hi[k], float(r[k]), int(n[k]), int(p[k])) for k in range(lo.size)]


def events_between(idx: dict, cik, t0, t1) -> list:
    """The company's events whose window overlaps (t0, t1]."""
    x = idx.get(int(cik))
    if x is None or x[0] is None:
        return []
    a, b = _ns(t0), _ns(t1)
    lo, hi, r, n, p = x[0]
    k = np.flatnonzero((lo < b) & (hi > a))
    return [(lo[j], hi[j], float(r[j]), int(n[j]), int(p[j])) for j in k]


def no_split(idx: dict, cik, t0, t1) -> bool:
    """The restated history shows no split in (t0, t1]: the interval lies inside one
    no-split span, and no event's window overlaps it."""
    x = idx.get(int(cik))
    if x is None or not x[1].size:
        return False
    a, b = _ns(t0), _ns(t1)
    j = int(np.searchsorted(x[1], a, side="right")) - 1
    if j < 0 or x[2][j] < b:
        return False
    return x[0] is None or not bool(((x[0][0] < b) & (x[0][1] > a)).any())


__all__ = ["TAGS", "observations", "evidence", "index", "events_of", "events_between", "no_split"]
