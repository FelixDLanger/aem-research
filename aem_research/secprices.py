"""The SEC price panel: one quarter-end price series per company from public filings (D10).

Primary price basis of the study (option A, Nov 8, 2026). Each company (CIK) gets a
price at every quarter-end:
  * from Form 13F – the median of market value over shares held across the filers
    reporting the company's main CUSIP (the one most filers hold; kept while it has at
    least a quarter of the leader's filers, so share classes do not flip), at least three;
  * where 13F has none (before the 13F data sets begin in 2013, or a thin quarter), from
    the SEC fails-to-deliver files – the closing price dated nearest the quarter-end,
    within 10 days.
Splits are found from holders, CUSIP changes and the filings' cover-page share counts
(aem_research.splits; every split must move the price with it) and every price is
restated in the units after the company's last split. A one-quarter price error (far
from both neighbouring quarter-ends, which agree) gives way to the fails-to-deliver
close of that quarter-end, or is left out. A change of CUSIP that is neither
a split nor the same security (holders or prices show neither) starts a new series: old
equity cancelled in a reorganization is not the new equity. When a series stops before
the panel's as-of date, the last fails-to-deliver price within the following quarter is
added as the final price, so a takeover or a collapse between quarter-ends is not missed.

Quarter-end prices are dated the day after the quarter-end – the start date they serve
as entry for – so the outcome engine and the price features read them like daily closes.
The quarter-end close was public on its day; 13F only measures it later. Breadth, which
was not public until the 13F filing, keeps its point-in-time rule (thirteenf.breadth_at).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import splits as sp
from .data import ftd as ftd_mod
from .data.thirteenf import cusip_bridge, normalize_name
from .tickers import canonical_series

MIN_PRICE_FILERS = 3
KEEP_MAIN_SHARE = 0.25
FTD_TOLERANCE_DAYS = 10
FINAL_WINDOW_DAYS = 92
FINAL_MAX_RISE = 3.0
MAX_GAP_DAYS = 200            # one missing quarter does not end a series; two do
COMPLETE_SHARE = 0.9          # as-of: the last quarter whose 13F coverage is at least 90% of the prior four
SOURCE = "sec"
SPIKE_MOVE = 4.0              # a one-quarter price error: at least 4x away from both neighbours ...
SPIKE_BACK = 1.5              # ... which agree with each other within 50%
SANE_MOVE = 6.0               # a split must leave a split-adjusted quarterly move within 1/6 to 6x
FILING_MOVE = 2.5             # splits read from filings: the adjusted move within 1/2.5 to 2.5x
FILING_TOL = 0.08             # the filings' share-count ratio within 8% of a standard ratio
FILING_MIN = 1.9              # ... of at least 2-for-1 either way (issuance blurs smaller ones)
DEI_AGREE = 0.35              # at a CUSIP change, the filings' ratio within 35% of the price ratio
JUMP = 2.5                    # a quarter-to-quarter price move this large asks for evidence
FILINGS_FLAT = 1.15           # filings whose share count moved less than 15% show no split

PANEL_COLS = ["ticker", "cik", "date", "period", "price", "close_raw", "cusip", "price_source", "source", "n_filers"]


def _first_word(names) -> pd.Series:
    """First word of the normalized company name ('' when none)."""
    key = normalize_name(pd.Series(names, dtype="object").fillna("").astype(str))
    return key.str.split(" ").str[0].fillna("")


def series_id(cik, segment) -> str:
    return f"CIK{int(cik)}" + (f"~{int(segment)}" if int(segment) > 1 else "")


def quarter_ends(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="QE").normalize()


# ---------------------------------------------------------------------------
# CUSIP -> CIK
# ---------------------------------------------------------------------------

def bridge(f13_names: pd.DataFrame | None, ftd_names: pd.DataFrame | None, companies: pd.DataFrame | None,
           filer_names: pd.DataFrame | None, symbol_bridges: list | None, *, margin_days: int = 400) -> pd.DataFrame:
    """(cusip, cik, method) for every CUSIP in the 13F or fails files: SimFin's ISIN, then
    the issuer CUSIP, then the fails files' ticker matched to the SEC symbol bridges (Form 4
    trading symbols, XBRL instance names) whose dates overlap, then the issuer name."""
    cols = ["cusip", "cik", "method"]
    names = []
    if f13_names is not None and len(f13_names):
        names.append(f13_names[["cusip", "name"]])
    if ftd_names is not None and len(ftd_names):
        names.append(ftd_names[["cusip", "description"]].rename(columns={"description": "name"}))
    if not names:
        return pd.DataFrame(columns=cols)
    known = pd.concat(names, ignore_index=True).dropna()
    known = known[known["cusip"].astype(str).str.len() == 9].drop_duplicates("cusip")
    parts = [cusip_bridge(companies, known, None)]
    done = set(parts[0]["cusip"])
    usable = [b for b in (symbol_bridges or []) if b is not None and len(b)]
    if ftd_names is not None and len(ftd_names) and usable:
        sb = pd.concat(usable, ignore_index=True)[["symbol", "issuer_cik", "first_seen", "last_seen", "filings"]].copy()
        sb["symbol"] = canonical_series(sb["symbol"])
        f = ftd_names[~ftd_names["cusip"].isin(done)][["cusip", "symbol", "first_date", "last_date", "n"]].copy()
        f["symbol"] = canonical_series(f["symbol"])
        f = f[f["symbol"].ne("")]
        m = f.merge(sb, on="symbol", how="inner")
        lo = pd.to_datetime(m["first_seen"]) - pd.Timedelta(days=margin_days)
        hi = pd.to_datetime(m["last_seen"]) + pd.Timedelta(days=margin_days)
        m = m[(pd.to_datetime(m["last_date"]) >= lo) & (pd.to_datetime(m["first_date"]) <= hi)]
        if filer_names is not None and len(filer_names):
            # the fails file's company name must start like a name the company filed under, so a
            # symbol reused by an ETF or another company does not tie to the wrong filer
            fn = filer_names[["cik", "name"]].dropna().assign(first=lambda d: _first_word(d["name"]))
            fn = fn[fn["first"].ne("")].assign(cik=lambda d: pd.to_numeric(d["cik"], errors="coerce"))
            ok_pairs = set(zip(fn["cik"].dropna().astype("int64"), fn["first"]))
            have_name = set(fn["cik"].dropna().astype("int64"))
            desc = ftd_names.drop_duplicates("cusip").set_index("cusip")["description"]
            first_desc = _first_word(m["cusip"].map(desc).fillna(""))
            cik_i = m["issuer_cik"].astype("int64")
            keep = [(c not in have_name) or ((c, w) in ok_pairs) for c, w in zip(cik_i, first_desc)]
            m = m[np.asarray(keep, dtype=bool)]
        m = m.sort_values(["n", "filings"], ascending=False, kind="mergesort")
        # a CUSIP whose symbol ties to two companies over the same dates is left to the name step
        amb = m.groupby("cusip")["issuer_cik"].nunique()
        m = m[m["cusip"].isin(amb[amb == 1].index)].drop_duplicates("cusip")
        if len(m):
            parts.append(pd.DataFrame({"cusip": m["cusip"], "cik": m["issuer_cik"].astype("int64"), "method": "fails ticker"}))
            done |= set(m["cusip"])
    if filer_names is not None and len(filer_names):
        rest = known[~known["cusip"].isin(done)]
        if len(rest):
            parts.append(cusip_bridge(None, rest, filer_names))
    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame(columns=cols)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out.drop_duplicates("cusip")
    out["cik"] = out["cik"].astype("int64")
    return out[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main CUSIP per company and quarter
# ---------------------------------------------------------------------------

def main_cusips(f13: pd.DataFrame, bridge_tab: pd.DataFrame, *, min_filers: int = MIN_PRICE_FILERS,
                keep_share: float = KEEP_MAIN_SHARE) -> pd.DataFrame:
    """(cik, period, cusip, price_13f, n_price_filers, n_filers) – one CUSIP per company and
    quarter, the one most filers hold, kept from one quarter to the next while it has at
    least `keep_share` of the leader's filers."""
    cols = ["cik", "period", "cusip", "price_13f", "n_price_filers", "n_filers"]
    if f13 is None or f13.empty or "price_13f" not in f13.columns or bridge_tab is None or bridge_tab.empty:
        return pd.DataFrame(columns=cols)
    q = f13[f13["n_price_filers"] >= min_filers].merge(bridge_tab[["cusip", "cik"]], on="cusip", how="inner")
    q = q[q["price_13f"] > 0].copy()
    q["period"] = pd.to_datetime(q["period"]).astype("datetime64[ns]")
    q = q.sort_values(["cik", "period", "n_filers", "n_price_filers", "cusip"],
                      ascending=[True, True, False, False, True], kind="mergesort")
    keep_cols = ["cik", "period", "cusip", "price_13f", "n_price_filers", "n_filers"]
    many = q.groupby("cik")["cusip"].nunique()
    single = q[q["cik"].isin(many[many == 1].index)].drop_duplicates(["cik", "period"])[keep_cols]
    out = []
    for cik, g in q[q["cik"].isin(many[many > 1].index)].groupby("cik", sort=False):
        prev = None
        for period, rows in g.groupby("period", sort=True):
            lead = rows.iloc[0]
            pick = lead
            if prev is not None and prev in set(rows["cusip"]):
                cand = rows[rows["cusip"] == prev].iloc[0]
                if cand["n_filers"] >= keep_share * lead["n_filers"]:
                    pick = cand
            out.append((cik, period, pick["cusip"], float(pick["price_13f"]), int(pick["n_price_filers"]),
                        int(pick["n_filers"])))
            prev = pick["cusip"]
    res = pd.concat([single, pd.DataFrame(out, columns=cols)], ignore_index=True)
    res = res.sort_values(["cik", "period"], kind="mergesort").reset_index(drop=True)
    res["period"] = res["period"].astype("datetime64[ns]")
    return res


def complete_through(f13: pd.DataFrame, *, min_filers: int = MIN_PRICE_FILERS, share: float = COMPLETE_SHARE):
    """The last quarter-end whose priced CUSIPs number at least `share` of the median of the
    four before it – a quarter still being filed is not mistaken for a wave of delistings."""
    if f13 is None or f13.empty or "price_13f" not in f13.columns:
        return None
    n = (f13[f13["n_price_filers"] >= min_filers].assign(period=lambda d: pd.to_datetime(d["period"]))
         .groupby("period")["cusip"].nunique().sort_index())
    if n.empty:
        return None
    last = n.index[-1]
    for i in range(len(n) - 1, -1, -1):
        prior = n.iloc[max(0, i - 4):i]
        if len(prior) == 0 or n.iloc[i] >= share * prior.median():
            last = n.index[i]
            break
    return pd.Timestamp(last)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def _fill_choice(cp: pd.DataFrame, prefer: str | None, symbols: dict, main_symbols: set) -> pd.Series | None:
    """The fails-to-deliver price to use for a quarter without a 13F price: the main CUSIP's,
    else a CUSIP trading under the main CUSIP's symbol (an older CUSIP of the same stock,
    not a preferred share), nearest the quarter-end."""
    pick = cp[cp["cusip"] == prefer]
    if len(pick):
        return pick.iloc[0]
    if main_symbols:
        same = cp[cp["cusip"].map(lambda c: bool(symbols.get(c, set()) & main_symbols))]
        if len(same):
            same = same.assign(_gap=(same["price_date"] - same["period"]).abs())
            return same.sort_values("_gap", kind="mergesort").iloc[0]
    return None


def _dei_index(hist: pd.DataFrame | None, ciks) -> dict:
    """cik -> (filing dates, share counts), sorted, from fundamentals.filing_share_history."""
    if hist is None or not len(hist):
        return {}
    d = hist[hist["cik"].isin(ciks)].sort_values(["cik", "filed"], kind="mergesort")
    return {int(c): (g["filed"].to_numpy(dtype="datetime64[ns]"), g["shares"].to_numpy(dtype="float64"))
            for c, g in d.groupby("cik", sort=False)}


def _dei_ratio(idx: dict, cik, p0, p1, max_days: int = 200) -> float:
    """Shares in the first filing filed after p1 over shares in the last filing filed on or
    before p0. A filing's share counts reflect every split before its filing date, so a
    split between the two quarter-ends changes the ratio by its ratio. NaN without a filing
    on each side within `max_days`."""
    x = idx.get(int(cik))
    if x is None:
        return np.nan
    filed, sh = x
    t0, t1 = np.datetime64(pd.Timestamp(p0), "ns"), np.datetime64(pd.Timestamp(p1), "ns")
    j0 = int(np.searchsorted(filed, t0, side="right")) - 1
    j1 = int(np.searchsorted(filed, t1, side="right"))
    if j0 < 0 or j1 >= filed.size:
        return np.nan
    lim = np.timedelta64(max_days, "D")
    if filed[j0] < t0 - lim or filed[j1] > t1 + lim or sh[j0] <= 0 or sh[j1] <= 0:
        return np.nan
    return float(sh[j1] / sh[j0])


def _sane(r_raw: float, s: float, lim: float) -> bool:
    """A split of ratio s leaves the split-adjusted move r_raw * s within 1/lim to lim."""
    return bool(np.isfinite(r_raw) and r_raw > 0 and s > 0 and abs(np.log(r_raw * s)) <= np.log(lim))


def _few_holders(hrow) -> float | None:
    """One or two matched holders (too few for the holder test) whose median share ratio is
    within 3% of a standard ratio of at least 1.2 either way – the ratio, or None."""
    if hrow is None or not (1 <= int(hrow.get("matched", 0)) < sp.MIN_MATCHED):
        return None
    med, r = float(hrow.get("median_ratio", np.nan)), float(hrow.get("ratio", np.nan))
    if not (np.isfinite(med) and np.isfinite(r)) or abs(np.log(r)) < np.log(sp.MIN_MOVE):
        return None
    return r if abs(np.log(med / r)) <= np.log(1 + sp.MEDIAN_TOL) else None


def _filing_split(s_f: float, r_raw: float, *, reverse_only: bool = False) -> float | None:
    """A split read from the filings' share counts: their ratio within FILING_TOL of a standard
    ratio of at least FILING_MIN either way, and the price moved the other way by about as
    much (split-adjusted move within FILING_MOVE). `reverse_only` where nothing else points
    to a split: share issuance can double a count while the price halves, but no ordinary
    event halves it while the price doubles."""
    if not (np.isfinite(s_f) and s_f > 0):
        return None
    std, dist = sp.snap([s_f])
    s = float(std[0])
    if reverse_only and s >= 1:
        return None
    if dist[0] <= np.log(1 + FILING_TOL) and abs(np.log(s)) >= np.log(FILING_MIN) and _sane(r_raw, s, FILING_MOVE):
        return s
    return None


def _spikes(q: pd.DataFrame, gap_limit: pd.Timedelta) -> np.ndarray:
    """Rows whose price jumps at least SPIKE_MOVE-fold against both neighbours, in opposite
    directions, while the neighbours agree within SPIKE_BACK: one-quarter price errors (a
    13F median from filers who reported value in the wrong unit, say). A split moves the
    price once and for good, so it never looks like this."""
    n = len(q)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out
    p = q["price"].to_numpy(dtype="float64")
    t = q["period"].to_numpy(dtype="datetime64[ns]")
    lim = np.timedelta64(int(gap_limit / pd.Timedelta(days=1)), "D")
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(p[1:-1] / p[:-2])
        b = np.log(p[2:] / p[1:-1])
    near = ((t[1:-1] - t[:-2]) <= lim) & ((t[2:] - t[1:-1]) <= lim)
    hit = near & (np.abs(a) >= np.log(SPIKE_MOVE)) & (np.abs(b) >= np.log(SPIKE_MOVE)) & (a * b < 0) \
        & (np.abs(a + b) <= np.log(SPIKE_BACK))
    out[1:-1] = np.nan_to_num(hit, nan=0).astype(bool)
    return out


def build_panel(mains: pd.DataFrame, holders: pd.DataFrame | None, ftd_px: pd.DataFrame | None,
                bridge_tab: pd.DataFrame, *, as_of_period, ftd_names: pd.DataFrame | None = None,
                lookback_quarters: int = 13, max_gap_days: int = MAX_GAP_DAYS,
                dei: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(panel, events, info). panel: PANEL_COLS, one row per series and quarter-end (plus a
    final fails-to-deliver price where a series stops early). events: splits and new series
    (cik, period, kind, ratio, evidence, matched, agree).

    `dei` (cik, filed, shares – fundamentals.filing_share_history: each filing's cover-page
    count, else its weighted-average shares) adds the filings' share counts as split
    evidence: they confirm or refute a split read from prices at a CUSIP change, and find
    reverse splits the holders cannot show (too few of them) where the price moved by the
    same ratio."""
    info = {"companies": 0, "series": 0, "quarters_13f": 0, "quarters_fails": 0, "final_prices": 0,
            "splits_holders": 0, "splits_cusip": 0, "splits_filings": 0, "splits_few_holders": 0,
            "holder_splits_rejected": 0, "new_series": 0, "fails_dropped_at_split": 0,
            "spikes_replaced": 0, "spikes_dropped": 0, "unexplained_jumps": 0}
    ev_cols = ["cik", "period", "kind", "ratio", "evidence", "matched", "agree"]
    if mains is None or mains.empty or as_of_period is None:
        return pd.DataFrame(columns=PANEL_COLS), pd.DataFrame(columns=ev_cols), info
    as_of_period = pd.Timestamp(as_of_period)
    m = mains[mains["period"] <= as_of_period].copy()
    m["src"] = "13f"
    m["price"] = m["price_13f"]
    ciks = set(m["cik"].unique())
    symbols: dict = {}
    if ftd_names is not None and len(ftd_names):
        sy = ftd_names.assign(symbol=canonical_series(ftd_names["symbol"]))
        sy = sy[sy["symbol"].ne("")]
        symbols = sy.groupby("cusip")["symbol"].apply(set).to_dict()

    # fails-to-deliver: quarter-end prices to fill 13F gaps and the lookback before 2013
    fq = pd.DataFrame(columns=["cusip", "period", "price", "price_date", "cik"])
    pts = pd.DataFrame(columns=["cusip", "price_date", "price"])
    if ftd_px is not None and len(ftd_px):
        first13 = m["period"].min()
        q_all = quarter_ends(first13 - pd.DateOffset(months=3 * lookback_quarters), as_of_period)
        pts = ftd_mod._points(ftd_px)
        fq = ftd_mod.quarter_end_prices(None, q_all, FTD_TOLERANCE_DAYS, points=pts)
        fq = fq.merge(bridge_tab[["cusip", "cik"]], on="cusip", how="inner")
        fq = fq[fq["cik"].isin(ciks)]
    fq_by_cik = {c: g for c, g in fq.groupby("cik", sort=False)} if len(fq) else {}
    pts_by_cusip = {c: g.sort_values("price_date") for c, g in pts.groupby("cusip", sort=False)} if len(pts) else {}

    # holders: share ratios between consecutive 13F quarters of the same company
    pairs = m.sort_values(["cik", "period"])[["cik", "period"]].copy()
    pairs["prev_period"] = pairs.groupby("cik")["period"].shift(1)
    pairs = pairs.dropna(subset=["prev_period"])
    pairs = pairs[(pairs["period"] - pairs["prev_period"]) <= pd.Timedelta(days=max_gap_days)]
    hr = pd.DataFrame(columns=["cik", "period", "matched", "median_ratio", "ratio", "agree"])
    if holders is not None and len(holders):
        hm = holders.merge(m[["cik", "period", "cusip"]], on=["cusip", "period"], how="inner")
        hr = sp.holder_ratios(hm[["cik", "period", "filer_cik", "shares"]], pairs)
    hr_idx = {(int(r.cik), pd.Timestamp(r.period)): r._asdict() for r in hr.itertuples(index=False)} if len(hr) else {}
    dei_idx = _dei_index(dei, ciks)

    rows, events = [], []
    one_day = pd.Timedelta(days=1)
    gap_limit = pd.Timedelta(days=max_gap_days)
    for cik, g in m.groupby("cik", sort=False):
        g = g.sort_values("period")
        q = g[["period", "cusip", "price", "src", "n_filers"]].copy()
        q["n_filers"] = q["n_filers"].astype("float64")
        fqc = fq_by_cik.get(cik)
        if fqc is not None and len(fqc):
            have = set(q["period"])
            first_p, last_p = q["period"].min(), q["period"].max()
            grid = quarter_ends(first_p - pd.DateOffset(months=3 * lookback_quarters), last_p)
            need = [p for p in grid if p not in have]
            cand = fqc[fqc["period"].isin(need)] if need else fqc.iloc[0:0]
            if len(cand):
                mp = q["period"].to_numpy(dtype="datetime64[ns]")
                mc = q["cusip"].to_numpy()
                fill = []
                for p, cp in cand.groupby("period"):
                    j = int(np.argmin(np.abs(mp - np.datetime64(p, "ns"))))      # nearest 13F quarter
                    prefer = mc[j]
                    pick = _fill_choice(cp, prefer, symbols, symbols.get(prefer, set()))
                    if pick is not None:
                        fill.append((p, pick["cusip"], float(pick["price"]), "fails", np.nan))
                if fill:
                    q = pd.concat([q, pd.DataFrame(fill, columns=q.columns)], ignore_index=True)
        q = q.sort_values("period").reset_index(drop=True)

        # one-quarter price errors: the fails-to-deliver close of that quarter-end if it fits the
        # neighbours, else the quarter is left out (one missing quarter does not end a series)
        bad = np.flatnonzero(_spikes(q, gap_limit))
        if bad.size:
            gone = []
            for i in bad:
                rep = None
                if q.at[i, "src"] == "13f" and fqc is not None and len(fqc):
                    cp = fqc[fqc["period"] == q.at[i, "period"]]
                    pick = _fill_choice(cp, q.at[i, "cusip"], symbols, symbols.get(q.at[i, "cusip"], set())) \
                        if len(cp) else None
                    if pick is not None:
                        fp, lo_, hi_ = float(pick["price"]), float(q.at[i - 1, "price"]), float(q.at[i + 1, "price"])
                        if fp > 0 and abs(np.log(fp / lo_)) <= np.log(2) and abs(np.log(hi_ / fp)) <= np.log(2):
                            rep = (fp, pick["cusip"])
                if rep is not None:
                    q.at[i, "price"], q.at[i, "cusip"], q.at[i, "src"], q.at[i, "n_filers"] = rep[0], rep[1], "fails", np.nan
                    info["spikes_replaced"] += 1
                else:
                    gone.append(i)
                    info["spikes_dropped"] += 1
            if gone:
                q = q.drop(index=gone).reset_index(drop=True)

        # walk forward: splits, continuations and new series
        n = len(q)
        seg = np.ones(n, dtype=int)
        ratio_at = np.ones(n)
        drop = np.zeros(n, dtype=bool)
        prev13 = -1
        for i in range(n):
            if i > 0:
                seg[i] = seg[i - 1]
                p0, p1 = q.at[i - 1, "period"], q.at[i, "period"]
                c0, c1 = q.at[i - 1, "cusip"], q.at[i, "cusip"]
                px0, px1 = float(q.at[i - 1, "price"]), float(q.at[i, "price"])
                r_raw = px1 / px0 if px0 > 0 and px1 > 0 else np.nan
                both13 = q.at[i, "src"] == "13f" and q.at[i - 1, "src"] == "13f"
                done = False
                if p1 - p0 > gap_limit:
                    seg[i] += 1
                    events.append((cik, p1, "new series", np.nan, "gap", 0, np.nan))
                    done = True
                if not done and q.at[i, "src"] == "13f" and prev13 >= 0 and seg[prev13] == seg[i] \
                        and (cik, p1) in hr_idx:
                    hrow = hr_idx[(cik, p1)]
                    s = sp.holder_split(hrow)
                    if s is not None:
                        p13 = float(q.at[prev13, "price"])
                        # the filings must not say otherwise: a count unchanged or moved the other way
                        # means the holders' ratio came from something else (a share-class flip, say)
                        s_f = _dei_ratio(dei_idx, cik, q.at[prev13, "period"], p1)
                        refuted = np.isfinite(s_f) and (abs(np.log(s_f)) < np.log(FILINGS_FLAT)
                                                        or np.sign(np.log(s_f)) != np.sign(np.log(s)))
                        if p13 > 0 and _sane(px1 / p13, s, SANE_MOVE) and not refuted:
                            ratio_at[i] = s
                            events.append((cik, p1, "split", s, "holders", int(hrow["matched"]), float(hrow["agree"])))
                            between = np.arange(prev13 + 1, i)             # fails prices of unknown side of the split
                            drop[between] = True
                            info["fails_dropped_at_split"] += int(len(between))
                            done = True
                        else:
                            info["holder_splits_rejected"] += 1            # the price or the filings did not move
                if not done and c0 != c1:
                    done = True
                    hrow = hr_idx.get((cik, p1)) if both13 else None
                    if not (hrow is not None and sp.holders_same(hrow)):
                        verdict, r = "break", None
                        old, new = pts_by_cusip.get(c0), pts_by_cusip.get(c1)
                        if old is not None and new is not None:
                            o = old[old["price_date"] <= p1 + pd.Timedelta(days=sp.SWITCH_DAYS)]
                            if len(o):
                                lo = o.iloc[-1]
                                gap = (new["price_date"] - lo["price_date"]).abs()
                                if len(gap) and gap.min() <= pd.Timedelta(days=sp.SWITCH_DAYS):
                                    fn = new.loc[gap.idxmin()]
                                    verdict, r = sp.price_switch(float(lo["price"]), float(fn["price"]))
                        s_f = _dei_ratio(dei_idx, cik, p0, p1)
                        evid = "CUSIP change"
                        if verdict == "split":
                            if np.isfinite(s_f):
                                if abs(np.log(s_f / r)) > np.log(1 + DEI_AGREE):
                                    verdict = "break"                      # the share count did not move with it
                                else:
                                    evid = "CUSIP change + filings"
                            elif sp.snap_common([r])[1][0] > np.log(1 + sp.PRICE_TOL):
                                verdict = "break"                          # an uncommon ratio, from prices alone
                        if verdict == "break" and np.isfinite(s_f):
                            s = _filing_split(s_f, r_raw)
                            if s is not None:
                                verdict, r, evid = "split", s, "CUSIP change + filings"
                        if verdict == "split" and np.isfinite(r_raw) and not _sane(r_raw, r, SANE_MOVE):
                            verdict = "break"
                        if verdict == "split":
                            ratio_at[i] = r
                            events.append((cik, p1, "split", r, evid, 0, np.nan))
                        elif verdict == "break":
                            seg[i] += 1
                            events.append((cik, p1, "new series", np.nan, "CUSIP change", 0, np.nan))
                if not done and np.isfinite(r_raw) and abs(np.log(r_raw)) >= np.log(JUMP):
                    # a large move and no verdict yet: one or two holders, or the filings' share counts
                    hrow = hr_idx.get((cik, p1)) if both13 else None
                    s_h = _few_holders(hrow)
                    s_f = _dei_ratio(dei_idx, cik, p0, p1)
                    s, evid = None, None
                    if s_h is not None and _sane(r_raw, s_h, FILING_MOVE):
                        # one holder can move by a split-like ratio through trading: the filings
                        # must agree, or, without filings, both of two holders
                        if np.isfinite(s_f):
                            ok = abs(np.log(s_f / s_h)) <= np.log(1 + DEI_AGREE)
                        else:
                            ok = int(hrow["matched"]) >= 2 and float(hrow["agree"]) >= 1.0
                        if ok:
                            s, evid = s_h, "holders (few)"
                    if s is None and np.isfinite(s_f):
                        s = _filing_split(s_f, r_raw, reverse_only=True)
                        evid = "filings" if s is not None else None
                    if s is not None:
                        ratio_at[i] = s
                        events.append((cik, p1, "split", s, evid, int(hrow["matched"]) if hrow else 0,
                                       float(hrow["agree"]) if hrow else np.nan))
                    else:
                        info["unexplained_jumps"] += 1
            if q.at[i, "src"] == "13f":
                prev13 = i
        keep = ~drop
        q, seg, ratio_at = q[keep].reset_index(drop=True), seg[keep], ratio_at[keep]
        info["quarters_13f"] += int((q["src"] == "13f").sum())
        info["quarters_fails"] += int((q["src"] == "fails").sum())

        # restate every price in the units after the series' last split
        for s_id in np.unique(seg):
            idx = np.flatnonzero(seg == s_id)
            r_seg = ratio_at[idx]
            factor = np.ones(len(idx))                  # product of the ratios after each row
            for k in range(len(idx) - 2, -1, -1):
                factor[k] = factor[k + 1] * r_seg[k + 1]
            sub = q.iloc[idx]
            tid = series_id(cik, s_id)
            raw = sub["price"].to_numpy(dtype="float64")
            for (period, cusip, price_raw, src, nf), f in zip(sub[["period", "cusip", "price", "src", "n_filers"]]
                                                               .itertuples(index=False, name=None), factor):
                rows.append((tid, cik, period + one_day, period, price_raw / f, price_raw, cusip,
                             "13F" if src == "13f" else "fails", SOURCE, nf))
            info["series"] += 1
            # a series that stops before the as-of quarter: its last fails price within a quarter
            last_period, last_cusip, last_raw = sub["period"].iloc[-1], sub["cusip"].iloc[-1], raw[-1]
            if last_period < as_of_period:
                cand = pts_by_cusip.get(last_cusip)
                if cand is not None:
                    w = cand[(cand["price_date"] > last_period + one_day)        # after the quarter-end row's date
                             & (cand["price_date"] <= min(last_period + pd.Timedelta(days=FINAL_WINDOW_DAYS),
                                                          as_of_period))]
                    if len(w):
                        fin = w.iloc[-1]
                        ratio = float(fin["price"]) / last_raw if last_raw > 0 else np.nan
                        if np.isfinite(ratio) and 0.0 < ratio <= FINAL_MAX_RISE:
                            rows.append((tid, cik, pd.Timestamp(fin["price_date"]), last_period, float(fin["price"]),
                                         float(fin["price"]), last_cusip, "fails final", SOURCE, np.nan))
                            info["final_prices"] += 1
        info["companies"] += 1

    panel = pd.DataFrame(rows, columns=PANEL_COLS)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    panel["period"] = pd.to_datetime(panel["period"]).astype("datetime64[ns]")
    panel["cik"] = panel["cik"].astype("int64")
    panel = (panel.assign(_final=panel["price_source"].eq("fails final"))
             .sort_values(["ticker", "date", "_final"], kind="mergesort")
             .drop_duplicates(["ticker", "date"], keep="first").drop(columns="_final").reset_index(drop=True))
    ev = pd.DataFrame(events, columns=ev_cols)
    if len(ev):
        ev["period"] = pd.to_datetime(ev["period"]).astype("datetime64[ns]")
    if len(ev):
        spl = ev[ev["kind"] == "split"]["evidence"].astype(str)
        info["splits_holders"] = int((spl == "holders").sum())
        info["splits_cusip"] = int(spl.str.startswith("CUSIP change").sum())
        info["splits_filings"] = int((spl == "filings").sum())
        info["splits_few_holders"] = int((spl == "holders (few)").sum())
        info["new_series"] = int((ev["kind"] == "new series").sum())
    return panel, ev, info


def as_of_date(as_of_period) -> pd.Timestamp:
    """The panel's as-of date: the start date the last complete quarter-end serves."""
    return pd.Timestamp(as_of_period) + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Delistings of SEC series
# ---------------------------------------------------------------------------

def ended_segments(panel: pd.DataFrame, as_of, *, stale_gap_days: int = 30, max_gap_days: int = MAX_GAP_DAYS
                   ) -> pd.DataFrame:
    """Series that stop before the as-of date, in the layout delistings.classify expects:
    ticker, first_date, last_date, last_price (the last price of the day, unadjusted),
    price_1y_before (in the same units), cik. The collapse test compares split-adjusted
    prices, the US$1 test the price of the day."""
    from .delistings import ended_segments as _ended

    cols = ["ticker", "first_date", "last_date", "last_price", "price_1y_before", "cik"]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=cols)
    seg = _ended(panel, as_of, stale_gap_days=stale_gap_days, max_gap_days=max_gap_days, price_col="price")
    if seg.empty:
        return pd.DataFrame(columns=cols)
    last = panel.sort_values("date").groupby("ticker").tail(1)[["ticker", "date", "price", "close_raw", "cik"]]
    seg = seg.merge(last.rename(columns={"date": "last_date", "price": "_adj", "close_raw": "_raw"}),
                    on=["ticker", "last_date"], how="left")
    scale = seg["_raw"] / seg["_adj"]
    seg["price_1y_before"] = seg["price_1y_before"] * scale
    seg["last_price"] = seg["_raw"]
    seg["cik"] = seg["cik"].astype("Int64")
    return seg[cols]


# ---------------------------------------------------------------------------
# Listing, validation, coverage
# ---------------------------------------------------------------------------

def listed_flags(keys: pd.DataFrame, vendor_keys: pd.DataFrame | None, form25x: pd.DataFrame | None,
                 symbol_exchange: pd.DataFrame | None = None) -> pd.Series:
    """Exchange listing at the start date, from public evidence: a vendor series prices the
    company then (SimFin and Stooq hold exchange-listed stocks), an exchange files Form 25
    for it later (it was listed until then), or its symbol is in an exchange folder of the
    Stooq file. keys: (cik, start_date) rows; returns a boolean Series aligned with keys."""
    k = keys[["cik", "start_date"]].copy()
    k["cik"] = pd.to_numeric(k["cik"], errors="coerce").astype("Int64")
    k["start_date"] = pd.to_datetime(k["start_date"]).astype("datetime64[ns]")
    flag = pd.Series(False, index=keys.index)
    if vendor_keys is not None and len(vendor_keys):
        v = vendor_keys.dropna(subset=["cik"])[["cik", "start_date"]].drop_duplicates()
        v = v.assign(cik=pd.to_numeric(v["cik"]).astype("Int64"),
                     start_date=pd.to_datetime(v["start_date"]).astype("datetime64[ns]"), _v=True)
        hit = k.merge(v, on=["cik", "start_date"], how="left")["_v"].fillna(False).astype(bool).to_numpy()
        flag |= hit
    if form25x is not None and len(form25x):
        f = form25x[form25x["exchange"].isin(["nasdaq", "nyse_amex"])
                    | form25x["form"].astype(str).str.startswith("25-NSE")][["cik", "date_filed"]].dropna()
        last = f.assign(cik=f["cik"].astype("int64")).groupby("cik")["date_filed"].max()
        when = k["cik"].map(lambda c: last.get(int(c)) if pd.notna(c) else None)
        hit = (pd.to_datetime(when) > k["start_date"]).fillna(False).to_numpy()
        flag |= hit
    if symbol_exchange is not None and len(symbol_exchange):
        s_ = symbol_exchange[symbol_exchange["exchange"].isin(["nasdaq", "nyse_amex"])]["cik"].dropna().astype("int64")
        flag |= k["cik"].isin(set(s_)).fillna(False).to_numpy()
    return flag


CASE_COLS = ["category", "cik", "ticker", "prev_period", "period", "evidence", "found", "vendor_ratio",
             "filings_ratio", "raw_prev", "raw", "sec_return", "vendor_return", "cusip_change", "n_filers"]


def validate(panel: pd.DataFrame, events: pd.DataFrame, vendor_panel: pd.DataFrame,
             vendor_keys: pd.DataFrame, *, tol: float = 0.10, filings: pd.DataFrame | None = None,
             return_cases: bool = False):
    """Blind-safe check of the SEC panel against the vendors' split-adjusted prices (no
    outcomes): share of company-quarters whose quarterly return agrees within 10%, and the
    splits the vendors' adjustments imply versus those found (precision and recall).

    `filings` (fundamentals.filing_share_history) is a third opinion: a split moves the
    filings' share count by its ratio; a spin-off or special dividend, which the vendors
    also adjust their prices for, leaves it alone. A vendor move with an unchanged count is
    therefore no split; a split found that the filings confirm counts as right even where
    the vendor shows nothing (unless the filings were its evidence). With `return_cases`,
    also returns one row per questionable quarter (CASE_COLS): splits found but not
    confirmed, vendor splits missed, and 2x gaps with no split on either side."""
    out = {"quarters_compared": 0}
    empty = pd.DataFrame(columns=CASE_COLS)
    if panel is None or panel.empty or vendor_panel is None or vendor_panel.empty or vendor_keys is None:
        return (out, empty) if return_cases else out
    keep = ["ticker", "cik", "period", "price", "close_raw"] + [c for c in ("cusip", "n_filers") if c in panel.columns]
    sec = panel[panel["price_source"] == "13F"][keep].copy()
    sec = sec.sort_values(["ticker", "period"])
    sec["prev_period"] = sec.groupby("ticker")["period"].shift(1)
    if "cusip" in sec.columns:
        sec["cusip_prev"] = sec.groupby("ticker")["cusip"].shift(1)
    sec["r_sec"] = sec["price"] / sec.groupby("ticker")["price"].shift(1)
    sec["r_raw"] = sec["close_raw"] / sec.groupby("ticker")["close_raw"].shift(1)
    sec = sec.dropna(subset=["prev_period", "r_sec"])
    sec = sec[(sec["period"] - sec["prev_period"]) <= pd.Timedelta(days=100)]
    vk = vendor_keys.dropna(subset=["cik"])[["ticker", "start_date", "cik"]].copy()
    vk["cik"] = pd.to_numeric(vk["cik"]).astype("int64")
    vk["period"] = (pd.to_datetime(vk["start_date"]) - pd.Timedelta(days=1)).astype("datetime64[ns]")
    vk = vk.drop_duplicates(["cik", "period"])
    want = set(vk["ticker"])
    periods = sorted(set(sec["period"]) | set(sec["prev_period"]))
    qe = pd.DataFrame({"period": pd.to_datetime(periods).astype("datetime64[ns]")})
    vq = []
    # one ticker at a time: the vendor panel is tens of millions of rows and is never copied whole
    for t, g in vendor_panel.groupby("ticker", sort=False, observed=True):
        if t not in want:
            continue
        g = g[["date", "price"]].assign(date=lambda d: pd.to_datetime(d["date"]).astype("datetime64[ns]"))
        g = g.dropna().sort_values("date")
        x = pd.merge_asof(qe, g.rename(columns={"date": "vdate"}), left_on="period", right_on="vdate",
                          direction="backward", tolerance=pd.Timedelta(days=10))
        x["ticker"] = t
        vq.append(x.dropna(subset=["price"]))
    if not vq:
        return (out, empty) if return_cases else out
    vq = pd.concat(vq, ignore_index=True)[["ticker", "period", "price"]].rename(columns={"price": "v"})
    a = sec.merge(vk[["cik", "period", "ticker"]].rename(columns={"ticker": "vt"}), on=["cik", "period"], how="inner")
    a = a.merge(vq.rename(columns={"ticker": "vt"}), on=["vt", "period"], how="inner")
    a = a.merge(vq.rename(columns={"ticker": "vt", "period": "prev_period", "v": "v_prev"}),
                on=["vt", "prev_period"], how="inner")
    a = a[(a["v"] > 0) & (a["v_prev"] > 0)]
    if a.empty:
        return (out, empty) if return_cases else out
    a = a.sort_values(["ticker", "period"], kind="mergesort").reset_index(drop=True)
    r_v = (a["v"] / a["v_prev"]).to_numpy()
    gap = np.log(a["r_sec"].to_numpy() / r_v)                 # SEC over vendor, log: 0 when they agree
    agree = np.abs(gap) <= np.log(1 + tol)
    implied = r_v / a["r_raw"].to_numpy()                     # = the split ratio when one happened
    # a gap the next quarter undoes is a one-quarter price error (either source), not a split
    nxt = pd.Series(gap).groupby(a["ticker"].to_numpy()).shift(-1).to_numpy()
    nxt_prev = a.groupby("ticker")["prev_period"].shift(-1).to_numpy()
    linked = nxt_prev == a["period"].to_numpy()
    nxt = np.where(linked, nxt, np.nan)
    trip_start = (np.abs(gap) >= np.log(2)) & (np.abs(gap + nxt) <= np.log(1.25))
    trip_end = np.zeros(len(a), dtype=bool)                   # the quarter that undoes it
    trip_end[1:] = trip_start[:-1] & linked[:-1]
    round_trip = trip_start | trip_end
    # the vendor's moves: its implied ratio within 5% of a common ratio (the dense standard set
    # would call almost any large disagreement a split), and no one-quarter round trip
    cstd, cdist = sp.snap_common(implied)
    vend_move = (np.abs(np.log(cstd)) >= np.log(sp.MIN_MOVE)) & (cdist <= np.log(1.05)) & ~round_trip
    # the filings across the same two quarter-ends: a split moves their count, a distribution does not
    s_f = np.full(len(a), np.nan)
    if filings is not None and len(filings):
        idx = _dei_index(filings, set(a["cik"].astype("int64")))
        s_f = np.array([_dei_ratio(idx, c, p0, p1) for c, p0, p1 in
                        zip(a["cik"].to_numpy(), a["prev_period"].to_numpy(), a["period"].to_numpy())], dtype="float64")
    f_known = np.isfinite(s_f)
    lf = np.log(np.where(f_known, s_f, 1.0))
    f_flat = f_known & (np.abs(lf) < np.log(FILINGS_FLAT))

    def filings_agree(ls):                    # moved, the same way, within 1.5x of the ratio
        return f_known & ~f_flat & (np.sign(lf) == np.sign(ls)) & (np.abs(lf - ls) <= np.log(1.5))

    vend_split = vend_move & (~f_known | filings_agree(np.log(cstd)))
    ev = pd.DataFrame(columns=["cik", "period", "ratio", "evidence"])
    if events is not None and len(events):
        ev = events[events["kind"] == "split"].reindex(columns=["cik", "period", "ratio", "evidence"])
        ev["evidence"] = ev["evidence"].fillna("n/a")
    fm = a[["cik", "period"]].merge(ev.drop_duplicates(["cik", "period"]), on=["cik", "period"], how="left")
    found = fm["ratio"].to_numpy(dtype="float64")
    has = np.isfinite(found)
    lfound = np.log(np.where(has, found, 1.0))
    fit = has & (np.abs(lfound - np.log(implied)) <= np.log(1 + tol))                    # vendor moved by it
    flat = has & (np.abs(np.log(implied)) < np.log(1 + tol))                              # vendor: no split
    uses_filings = fm["evidence"].astype(str).str.contains("filings").to_numpy()
    by_filings = has & ~uses_filings & filings_agree(lfound)                              # the filings moved by it
    confirmed = fit | by_filings
    match = vend_split & fit
    out.update({"quarters_compared": int(len(a)), "returns_agree": round(float(agree.mean()), 4),
                "vendor_splits": int(vend_split.sum()), "found_splits": int(has.sum()),
                "split_recall": round(float(match.sum() / vend_split.sum()), 4) if vend_split.sum() else None,
                "split_precision": round(float(confirmed.sum() / has.sum()), 4) if has.sum() else None,
                "split_precision_vendor": round(float(fit.sum() / has.sum()), 4) if has.sum() else None,
                "found_vendor_flat": int(flat.sum()),
                "vendor_moves_shares_flat": int((vend_move & f_flat).sum()),
                "filings_known": round(float(f_known.mean()), 4),
                "jumps_up_2x": int((gap >= np.log(2)).sum()), "jumps_down_2x": int((gap <= -np.log(2)).sum()),
                "jumps_round_trip": int(trip_start.sum())})
    if has.any():
        by = pd.DataFrame({"evidence": fm["evidence"].astype(str).to_numpy()[has], "ok": confirmed[has]})
        out["found_by"] = " ".join(f"{e} {len(g)} ({g['ok'].mean():.0%})" for e, g in by.groupby("evidence"))
    miss = vend_split & ~fit
    cus_change = (a["cusip"].to_numpy() != a["cusip_prev"].to_numpy()) if "cusip_prev" in a.columns \
        else np.zeros(len(a), dtype=bool)
    if miss.any():
        out["missed_splits"] = int(miss.sum())
        out["missed_at_cusip_change"] = round(float(cus_change[miss].mean()), 4)
        if "n_filers" in a.columns:
            out["missed_filers_median"] = float(np.nanmedian(pd.to_numeric(a["n_filers"], errors="coerce")
                                                             .to_numpy(dtype="float64")[miss]))
        out["missed_reverse_share"] = round(float((cstd[miss] < 1).mean()), 4)
    # the 2x gaps nothing explains: no split found, none the vendor implies, no round trip
    lone = (np.abs(gap) >= np.log(2)) & ~round_trip & ~has & ~vend_split
    out["gaps_2x_unexplained"] = int(lone.sum())
    if not return_cases:
        return out
    cat = np.full(len(a), "", dtype=object)
    cat[lone] = "2x gap, no split"
    cat[miss] = "split missed"
    cat[has & ~confirmed] = "split unconfirmed"
    keep_rows = cat != ""
    raw = a["close_raw"].to_numpy(dtype="float64")
    cases = pd.DataFrame({
        "category": cat, "cik": a["cik"].to_numpy(), "ticker": a["ticker"].to_numpy(),
        "prev_period": a["prev_period"].to_numpy(), "period": a["period"].to_numpy(),
        "evidence": fm["evidence"].fillna("").astype(str).to_numpy(), "found": found,
        "vendor_ratio": implied, "filings_ratio": s_f, "raw_prev": raw / a["r_raw"].to_numpy(), "raw": raw,
        "sec_return": a["r_sec"].to_numpy(), "vendor_return": r_v, "cusip_change": cus_change,
        "n_filers": pd.to_numeric(a["n_filers"], errors="coerce").to_numpy() if "n_filers" in a.columns else np.nan,
    })[keep_rows].reset_index(drop=True)
    return out, cases[CASE_COLS]


def case_lines(cases: pd.DataFrame, per_category: int = 4, seed: int = 0) -> list[str]:
    """A few cases of each category for the paste block (prices at two quarter-ends, ratios)."""
    if cases is None or cases.empty:
        return []
    lines = []
    for cat_, g in cases.groupby("category", sort=True):
        for r in g.sample(min(per_category, len(g)), random_state=seed).itertuples(index=False):
            f = lambda x, d=3: "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}g}"  # noqa: E731
            lines.append(f"{cat_} · cik {int(r.cik)} · {pd.Timestamp(r.prev_period):%Y-%m} to "
                         f"{pd.Timestamp(r.period):%Y-%m} · found {f(r.found)} ({r.evidence or 'none'}) · "
                         f"vendor ratio {f(r.vendor_ratio)} · filings ratio {f(r.filings_ratio)} · "
                         f"raw {f(r.raw_prev)} to {f(r.raw)} · returns SEC {f(r.sec_return)} vendor "
                         f"{f(r.vendor_return)} · CUSIP change {bool(r.cusip_change)} · 13F filers {f(r.n_filers)}")
    return lines


def holes_priced(holes_keys: pd.DataFrame, sec_pf: pd.DataFrame, sec_keys: pd.DataFrame) -> float | None:
    """Share of the vendor panel's holes (cik, start_date) the SEC panel prices at entry."""
    if holes_keys is None or holes_keys.empty or sec_pf is None or sec_pf.empty:
        return None
    s = sec_pf[["ticker", "start_date"]].merge(sec_keys[["ticker", "start_date", "cik"]], on=["ticker", "start_date"])
    s = s.dropna(subset=["cik"]).assign(cik=lambda d: pd.to_numeric(d["cik"]).astype("int64"), _p=True)
    h = holes_keys.assign(cik=pd.to_numeric(holes_keys["cik"]).astype("int64"))
    hit = h.merge(s[["cik", "start_date", "_p"]].drop_duplicates(), on=["cik", "start_date"], how="left")["_p"]
    return round(float(hit.fillna(False).mean()), 4)


def keys_for(sec_pf: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """(ticker, start_date, cik, cik_source) for SEC series – the CIK is the series' own."""
    k = sec_pf[["ticker", "start_date"]].drop_duplicates().reset_index(drop=True)
    cmap = panel.drop_duplicates("ticker").set_index("ticker")["cik"]
    k["cik"] = k["ticker"].map(cmap).astype("Int64")
    k["cik_source"] = "sec"
    return k


def entry_source(sec_pf: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """13F or fails for each SEC entry (aligned with sec_pf)."""
    src = panel[["ticker", "date", "price_source"]].rename(columns={"date": "entry_date"})
    src = src.drop_duplicates(["ticker", "entry_date"])
    return sec_pf[["ticker", "entry_date"]].merge(src, on=["ticker", "entry_date"], how="left")["price_source"].to_numpy()


def symbol_table(ftd_names: pd.DataFrame | None, bridge_tab: pd.DataFrame, symbol_bridges: list | None) -> pd.DataFrame:
    """(cik, symbol, last_date): symbols the fails files and the SEC symbol bridges tie to each
    company, with the last date each was seen."""
    cols = ["cik", "symbol", "last_date"]
    parts = []
    if ftd_names is not None and len(ftd_names) and bridge_tab is not None and len(bridge_tab):
        f = ftd_names.merge(bridge_tab[["cusip", "cik"]], on="cusip", how="inner")[["cik", "symbol", "last_date"]]
        parts.append(f)
    for b in symbol_bridges or []:
        if b is not None and len(b):
            parts.append(b.rename(columns={"issuer_cik": "cik", "last_seen": "last_date"})[["cik", "symbol", "last_date"]])
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True).dropna(subset=["cik", "symbol"])
    out["symbol"] = canonical_series(out["symbol"])
    out = out[out["symbol"].ne("")]
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    out["last_date"] = pd.to_datetime(out["last_date"], errors="coerce").astype("datetime64[ns]")
    out = out.dropna(subset=["cik"]).groupby(["cik", "symbol"], as_index=False)["last_date"].max()
    return out[cols]


def symbol_exchange(sym_tab: pd.DataFrame, stooq_groups: pd.DataFrame | None = None,
                    eod_exchange: pd.DataFrame | None = None, *, recent_days: int = 365) -> pd.DataFrame:
    """(cik, exchange) for companies whose symbol, still in use in the last `recent_days` of the
    SEC files, sits in an exchange folder of the Stooq file (or EODHD's list). Older symbols are
    not used: a delisted company's symbol may belong to another company today."""
    from .delistings import stooq_exchange

    cols = ["cik", "exchange"]
    if sym_tab is None or sym_tab.empty:
        return pd.DataFrame(columns=cols)
    ex = []
    if stooq_groups is not None and len(stooq_groups):
        g = stooq_groups.assign(symbol=canonical_series(stooq_groups["ticker"]),
                                exchange=stooq_groups["group"].map(stooq_exchange))
        ex.append(g[["symbol", "exchange"]])
    if eod_exchange is not None and len(eod_exchange):
        ex.append(eod_exchange.assign(symbol=canonical_series(eod_exchange["ticker"]))[["symbol", "exchange"]])
    if not ex:
        return pd.DataFrame(columns=cols)
    ex = pd.concat(ex, ignore_index=True)
    ex = ex[ex["exchange"].isin(["nasdaq", "nyse_amex"])].drop_duplicates("symbol")
    latest = sym_tab["last_date"].max()
    recent = sym_tab[sym_tab["last_date"] >= latest - pd.Timedelta(days=recent_days)]
    out = recent.merge(ex, on="symbol", how="inner")[cols].drop_duplicates("cik")
    out["cik"] = out["cik"].astype("int64")
    return out.reset_index(drop=True)


def exchange_map_for(panel: pd.DataFrame, form25x: pd.DataFrame | None, sym_ex: pd.DataFrame | None) -> pd.DataFrame:
    """(ticker, exchange) for the SEC series, for the delisting returns: the exchange that
    filed the company's latest Form 25, else the symbol's exchange folder."""
    cols = ["ticker", "exchange"]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=cols)
    t = panel.drop_duplicates("ticker")[["ticker", "cik"]]
    parts = []
    if form25x is not None and len(form25x):
        f = form25x[form25x["exchange"].isin(["nasdaq", "nyse_amex"])].dropna(subset=["cik"])
        f = f.assign(cik=f["cik"].astype("int64")).sort_values("date_filed").drop_duplicates("cik", keep="last")
        parts.append(f[["cik", "exchange"]])
    if sym_ex is not None and len(sym_ex):
        parts.append(sym_ex[["cik", "exchange"]])
    if not parts:
        return pd.DataFrame(columns=cols)
    e = pd.concat(parts, ignore_index=True).drop_duplicates("cik")
    return t.merge(e, on="cik", how="inner")[cols].reset_index(drop=True)


def dense_panel(panel: pd.DataFrame, events: pd.DataFrame, ftd_px: pd.DataFrame | None,
                tickers) -> pd.DataFrame:
    """Robustness B: the SEC series of the given tickers plus the highest fails-to-deliver
    price of every half-month file inside them, restated in the series' split units.
    Quarters with a split are skipped (the split day is not known), and so are prices
    after the series' last quarter-end other than its final price."""
    base = panel[panel["ticker"].isin(set(tickers))].copy()
    if base.empty or ftd_px is None or ftd_px.empty:
        return base
    seg_rows = base[base["price_source"] != "fails final"].sort_values(["ticker", "period"])
    split_q = set()
    if events is not None and len(events):
        split_q = set(zip(events.loc[events["kind"] == "split", "cik"].astype("int64"),
                          pd.to_datetime(events.loc[events["kind"] == "split", "period"])))
    pts = ftd_px[["cusip", "max_date", "max_price"]].dropna().rename(columns={"max_date": "price_date",
                                                                             "max_price": "raw"})
    extra = []
    for t, g in seg_rows.groupby("ticker", sort=False):
        cusips = set(g["cusip"])
        p = pts[pts["cusip"].isin(cusips)]
        if p.empty:
            continue
        periods = g["period"].to_numpy(dtype="datetime64[ns]")
        factor = (g["close_raw"] / g["price"]).to_numpy(dtype="float64")       # raw / adjusted per quarter
        cik = int(g["cik"].iloc[0])
        d = p["price_date"].to_numpy(dtype="datetime64[ns]")
        j = np.searchsorted(periods, d, side="left")                           # the quarter-end on or after d
        ok = (j > 0) & (j < len(periods))
        for k in np.flatnonzero(ok):
            qe = pd.Timestamp(periods[j[k]])
            if (cik, qe) in split_q:
                continue
            extra.append((t, cik, pd.Timestamp(d[k]), qe, float(p["raw"].iloc[k]) / factor[j[k]],
                          float(p["raw"].iloc[k]), p["cusip"].iloc[k], "fails max", SOURCE, np.nan))
    if not extra:
        return base
    add = pd.DataFrame(extra, columns=PANEL_COLS)
    out = pd.concat([base, add], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out.sort_values(["ticker", "date"], kind="mergesort").drop_duplicates(["ticker", "date"]).reset_index(drop=True)


__all__ = ["bridge", "main_cusips", "complete_through", "build_panel", "as_of_date", "ended_segments",
           "listed_flags", "validate", "holes_priced", "keys_for", "entry_source", "symbol_table", "dense_panel",
           "symbol_exchange", "exchange_map_for",
           "series_id", "quarter_ends", "MAX_GAP_DAYS", "SOURCE", "normalize_name"]
