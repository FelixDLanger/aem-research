"""A signal scan over everything the feature table holds (exploratory; notebook 09).

The study tested the signals AEM named. This asks the other question: is there any 3x or
10x signal in the data that nobody named? Run on one sample, such a search finds patterns
by chance, so it is split in time: every candidate is measured on the start dates to 2017
(discovery), corrected for the number of candidates (Benjamini-Hochberg), and counted only
if it shows the same direction on the start dates from 2018 (holdout). Three targets: 3x
within three years, 10x within five, and the mirror – the window ended below half. The start dates
after the locked sample (two-year outcomes) are a third look. A flexible model on all
features, fitted on discovery and scored on holdout, bounds what any combination could add.

The candidate list was fixed on Nov 12, 2026, before any of these was set against an outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import estimators as est

DISCOVERY_TO = 2017
PHARMA_SIC = (2833, 2834, 2835, 2836, 8731)
CONTINUOUS = ("market_cap", "entry_price", "drawdown", "drawdown_vs_median", "momentum_6m", "momentum_12m", "fcf_yield",
              "fcf_change_to_cap", "fcf_margin", "ocf_change_to_cap", "revenue_growth", "sales_to_cap", "net_cash_to_cap",
              "cash_to_cap", "debt_to_cap", "share_growth", "capex_to_revenue", "breadth", "turnover_126d")
FLAGS = ("sale_any", "insider_routine_only", "several_buyers", "pharma_biotech", "turned_cash_positive", "burner_short_runway",
         "profitable_growth", "fallen_and_growing", "fallen_and_cash_positive", "shrinking_share_count")


# A second batch, fixed on Nov 13, 2026 after the first scan was read: patterns the finance
# literature documents that the first list left out and that these tables can measure.
# Long-term reversal (De Bondt and Thaler), nearness to the one-year high (George and Hwang),
# past volatility (Ang, Hodrick, Xing and Zhang; lottery-type stocks), investment growth
# (Titman, Wei and Xie), enterprise value to sales, change in the breadth of institutional
# ownership (Chen, Hong and Stein), industry momentum (Moskowitz and Grinblatt), a
# fundamentals score in the spirit of Piotroski, recent listing (Ritter), Graham's net cash.
LIT_CONTINUOUS = ("momentum_36m", "near_high_1y", "volatility_8q", "capex_growth", "ev_to_sales", "breadth_change_1y",
                  "industry_momentum_6m", "fundamentals_score")
LIT_FLAGS = ("score_high", "score_low", "listed_under_3_years", "net_cash_two_thirds")


def _num(f: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(f[c], errors="coerce") if c in f.columns else pd.Series(np.nan, index=f.index)


def momentum(features: pd.DataFrame, panel: pd.DataFrame | None) -> pd.DataFrame:
    """Price change over the six and twelve months before the start date, from the SEC panel's
    quarter-end prices (both prices in the series' split units; known on the start date)."""
    out = features[["ticker", "start_date"]].copy()
    out["start_date"] = pd.to_datetime(out["start_date"]).astype("datetime64[ns]")
    out["momentum_6m"] = out["momentum_12m"] = np.nan
    if panel is None or not len(panel):
        return out
    p = panel[["ticker", "date", "price"]].assign(date=pd.to_datetime(panel["date"]).astype("datetime64[ns]"))
    p = p.drop_duplicates(["ticker", "date"]).set_index(["ticker", "date"])["price"]
    idx = pd.MultiIndex.from_arrays([out["ticker"].astype(str), out["start_date"]])
    now = p.reindex(idx).to_numpy(dtype=float)
    for months, col in ((6, "momentum_6m"), (12, "momentum_12m")):
        back = pd.MultiIndex.from_arrays([out["ticker"].astype(str), out["start_date"] - pd.DateOffset(months=months)])
        then = p.reindex(back).to_numpy(dtype=float)
        out[col] = np.where((then > 0) & (now > 0), now / then - 1, np.nan)
    return out


def price_history(features: pd.DataFrame, panel: pd.DataFrame | None) -> pd.DataFrame:
    """From the SEC panel's quarter-end prices before the start date: the 36-month price
    change, the price against the highest of the last five quarter-ends, the volatility of
    the last eight quarterly returns, and whether the series is under three years old."""
    out = pd.DataFrame({"momentum_36m": np.nan, "near_high_1y": np.nan, "volatility_8q": np.nan, "listed_under_3_years": np.nan},
                       index=features.index)
    if panel is None or not len(panel) or not len(features):
        return out
    tick = features["ticker"].astype(str).to_numpy()
    start = pd.to_datetime(features["start_date"]).astype("datetime64[ns]")
    q = panel[["ticker", "date", "price"]].assign(date=pd.to_datetime(panel["date"]).astype("datetime64[ns]"), ticker=panel["ticker"].astype(str))
    first = q.groupby("ticker")["date"].min()
    p = q.drop_duplicates(["ticker", "date"]).set_index(["ticker", "date"])["price"]

    def at(months):
        v = p.reindex(pd.MultiIndex.from_arrays([tick, (start - pd.DateOffset(months=months)).to_numpy()])).to_numpy(dtype=float)
        return np.where(v > 0, v, np.nan)
    px = np.column_stack([at(m) for m in range(0, 27, 3)])               # now, 3, 6, ... 24 months back
    out["momentum_36m"] = px[:, 0] / at(36) - 1
    with np.errstate(all="ignore"):
        high = np.nanmax(px[:, :5], axis=1)
        out["near_high_1y"] = np.where(np.isfinite(px[:, :5]).sum(axis=1) >= 4, px[:, 0] / high, np.nan)
        rets = np.log(px[:, :-1] / px[:, 1:])
        out["volatility_8q"] = np.where(np.isfinite(rets).sum(axis=1) >= 6, np.nanstd(rets, axis=1, ddof=1), np.nan)
    born = pd.Series(tick).map(first).to_numpy(dtype="datetime64[ns]")
    age = (start.to_numpy() - born) / np.timedelta64(1, "D")
    out["listed_under_3_years"] = np.where(np.isnan(age), np.nan, (age < 3 * 365.25).astype(float))
    return out


def candidates(w: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """The windows with every candidate: continuous measures and yes/no flags, all from values
    known on the start date."""
    x = w.copy()
    cap = _num(x, "market_cap").where(lambda v: v > 0)
    rev = _num(x, "revenue").where(lambda v: v > 0)
    fcf, fcf_prev = _num(x, "fcf"), _num(x, "fcf_prev")
    x["entry_price"] = _num(x, "entry_price_unadjusted").fillna(_num(x, "entry_price"))
    x["drawdown_vs_median"] = _num(x, "drawdown") - _num(x, "drawdown_median")
    mom = momentum(x, panel)
    x["momentum_6m"], x["momentum_12m"] = mom["momentum_6m"].to_numpy(), mom["momentum_12m"].to_numpy()
    x["fcf_yield"], x["fcf_change_to_cap"], x["fcf_margin"] = fcf / cap, (fcf - fcf_prev) / cap, fcf / rev
    x["ocf_change_to_cap"] = (_num(x, "ocf") - _num(x, "ocf_prev")) / cap
    x["sales_to_cap"], x["net_cash_to_cap"] = rev / cap, _num(x, "net_cash") / cap
    x["cash_to_cap"], x["debt_to_cap"] = _num(x, "cash_total") / cap, _num(x, "debt") / cap
    x["capex_to_revenue"] = _num(x, "capex").abs() / rev
    growth, shares = _num(x, "revenue_growth"), _num(x, "share_growth")
    fallen = est._flag(x["fallen"]).eq(1) if "fallen" in x.columns else pd.Series(False, index=x.index)
    known = fcf.notna()
    x["several_buyers"] = (_num(x, "insider_n").fillna(0) >= 3).astype(float)
    x["pharma_biotech"] = _num(x, "sic").isin(PHARMA_SIC).astype(float)
    x["turned_cash_positive"] = ((fcf > 0) & (fcf_prev < 0)).astype(float).where(known & fcf_prev.notna())
    runway = _num(x, "cash_total") / (-fcf).where(fcf < 0)
    x["burner_short_runway"] = ((fcf < 0) & (runway < 2)).astype(float).where(known)
    x["profitable_growth"] = ((fcf > 0) & (growth > 0.2)).astype(float).where(known & growth.notna())
    x["fallen_and_growing"] = (fallen & (growth > 0.1)).astype(float).where(growth.notna())
    x["fallen_and_cash_positive"] = (fallen & (fcf > 0)).astype(float).where(known)
    x["shrinking_share_count"] = (shares < -0.02).astype(float).where(shares.notna())
    for c in ("sale_any", "insider_routine_only"):
        x[c] = est._flag(x[c]) if c in x.columns else np.nan
    # ---- the second batch
    hist = price_history(x, panel)
    for c in hist.columns:
        x[c] = hist[c].to_numpy()
    capex, capex_prev = _num(x, "capex").abs(), _num(x, "capex_prev").abs()
    x["capex_growth"] = capex / capex_prev.where(capex_prev > 0) - 1
    x["ev_to_sales"] = ((cap - _num(x, "net_cash")) / rev).where(lambda v: v > 0)
    b = _num(x, "breadth")
    if "firm" in x.columns and b.notna().any():
        lag = x[["firm", "start_date"]].assign(start_date=pd.to_datetime(x["start_date"]) + pd.DateOffset(years=1), _b=b.to_numpy())
        lag = lag.dropna(subset=["firm", "_b"]).drop_duplicates(["firm", "start_date"])
        prev = x[["firm", "start_date"]].assign(start_date=pd.to_datetime(x["start_date"])).merge(lag, on=["firm", "start_date"], how="left")["_b"]
        x["breadth_change_1y"] = b.to_numpy() / np.where(prev.to_numpy() > 0, prev.to_numpy(), np.nan) - 1
    else:
        x["breadth_change_1y"] = np.nan
    industry = (_num(x, "sic") // 100).where(_num(x, "sic").notna())
    grp = x.assign(_i=industry.to_numpy(), _m=x["momentum_6m"].to_numpy()).groupby(["start_date", "_i"])["_m"]
    x["industry_momentum_6m"] = grp.transform("mean").where(grp.transform("count") >= 5).to_numpy()
    ocf, ocf_prev = _num(x, "ocf"), _num(x, "ocf_prev")
    parts = [(ocf > 0, ocf.notna()), (ocf > ocf_prev, ocf.notna() & ocf_prev.notna()), (fcf > 0, fcf.notna()),
             (shares <= 0.01, shares.notna()), (growth > 0, growth.notna())]
    all_known = np.logical_and.reduce([k.to_numpy() for _, k in parts])
    x["fundamentals_score"] = np.where(all_known, sum(c.to_numpy().astype(float) for c, _ in parts), np.nan)
    x["score_high"] = np.where(all_known, (x["fundamentals_score"] >= 4).astype(float), np.nan)
    x["score_low"] = np.where(all_known, (x["fundamentals_score"] <= 1).astype(float), np.nan)
    x["net_cash_two_thirds"] = ((_num(x, "net_cash") / cap) >= 2 / 3).astype(float).where((_num(x, "net_cash") / cap).notna())
    # the mirror target: the window ended below half its entry value (five years; two on the new dates)
    for end, loss in (("end5", "loss5"), ("end2", "loss2")):
        if end in x.columns:
            e = _num(x, end)
            x[loss] = (e < 0.5).astype(float).where(e.notna())
    return x


TARGETS = (("3x within 3 years", "hit3_3y", "hit2_2y"), ("10x within 5 years", "hit10_5y", None),
           ("ended 5 years below half", "loss5", "loss2"))


def _quintile_flags(x: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Lowest and highest fifth of `col` within each start date (missing where the value is)."""
    v = pd.to_numeric(x[col], errors="coerce")
    r = v.groupby(x["start_date"]).rank(pct=True)
    return (r <= 0.2).astype(float).where(v.notna()), (r > 0.8).astype(float).where(v.notna())


def _lr(g: pd.DataFrame, sig: pd.Series, hit: str, resamples: int, seed: int) -> dict:
    y = g.assign(_s=sig.reindex(g.index))
    y = y[y[hit].notna() & y["_s"].notna()]
    if len(y) < 200 or y["_s"].sum() < 30 or y[hit].sum() < 10:
        return {"lr": np.nan, "p": np.nan, "n": int(len(y)), "with": int(y["_s"].sum()) if len(y) else 0}
    r = est.likelihood_ratio(y, "_s", hit, resamples=resamples, seed=seed)
    above = r["p_one_sided"]
    p = float(min(1.0, 2 * min(above, 1 - above))) if np.isfinite(above) else np.nan
    return {"lr": r["lr"], "p": p, "n": r["windows"], "with": r["with_signal"]}


def _bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (missing stay missing)."""
    out = pd.Series(np.nan, index=p.index)
    k = p.dropna().sort_values()
    if len(k):
        adj = (k * len(k) / np.arange(1, len(k) + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
        out.loc[adj.index] = adj
    return out


def scan(w: pd.DataFrame, w_new: pd.DataFrame | None = None, *, panel=None, resamples: int = 500, seed: int = 20261106,
         fdr: float = 0.10, alpha: float = 0.05, continuous=CONTINUOUS, flags=FLAGS) -> pd.DataFrame:
    """One row per candidate, side and target: the ratio on discovery, holdout and the new
    dates, and whether it held."""
    x = candidates(w, panel)
    xn = candidates(w_new, panel) if w_new is not None and len(w_new) else None
    disc, hold = x[x["start_year"] <= DISCOVERY_TO], x[x["start_year"] > DISCOVERY_TO]
    sides = []
    for c in continuous:
        if c in x.columns and x[c].notna().any():
            lo, hi = _quintile_flags(x, c)
            lo_n, hi_n = _quintile_flags(xn, c) if xn is not None and c in xn.columns else (None, None)
            sides += [(c, "lowest fifth", lo, lo_n), (c, "highest fifth", hi, hi_n)]
    for c in flags:
        if c in x.columns and x[c].notna().any():
            sides.append((c, "flag", x[c], xn[c] if xn is not None and c in xn.columns else None))
    rows = []
    for target, hit, hit_new in TARGETS:
        if hit not in x.columns:
            continue
        for name, side, sig, sig_new in sides:
            d, h = _lr(disc, sig, hit, resamples, seed), _lr(hold, sig, hit, resamples, seed)
            row = {"target": target, "candidate": name, "side": side, "discovery_windows": d["with"], "discovery_lr": d["lr"],
                   "discovery_p": d["p"], "holdout_windows": h["with"], "holdout_lr": h["lr"], "holdout_p": h["p"]}
            if xn is not None and sig_new is not None and hit_new is not None and hit_new in xn.columns:
                n = _lr(xn, sig_new, hit_new, resamples, seed)
                row.update(new_dates_windows=n["with"], new_dates_lr=n["lr"], new_dates_p=n["p"])
            rows.append(row)
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    t["discovery_q"] = _bh(t["discovery_p"])
    same = np.sign(np.log(t["discovery_lr"].where(t["discovery_lr"] > 0))) == np.sign(np.log(t["holdout_lr"].where(t["holdout_lr"] > 0)))
    held = (t["discovery_q"] <= fdr) & same & (t["holdout_p"] <= alpha)
    t["verdict"] = np.where(held, "held on holdout", np.where(t["discovery_q"] <= fdr, "discovery only", ""))
    if "new_dates_lr" in t.columns:
        again = held & (np.sign(np.log(t["new_dates_lr"].where(t["new_dates_lr"] > 0))) == np.sign(np.log(t["holdout_lr"].where(t["holdout_lr"] > 0)))) \
            & (t["new_dates_p"] <= alpha)
        t.loc[again, "verdict"] = "held on holdout and on the new dates"
    return t


def model_bound(w: pd.DataFrame, w_new: pd.DataFrame | None = None, *, panel=None, seed: int = 20261106) -> pd.DataFrame:
    """What any combination of the candidates could add: gradient-boosted trees fitted on
    discovery, scored on holdout (and on the new dates for the 3x model). AUC 0.5 and lift 1
    mean no information beyond the base rate."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    x = candidates(w, panel)
    xn = candidates(w_new, panel) if w_new is not None and len(w_new) else None
    base = ("fcf_pos", "net_cash_pos", "low_dilution", "s1_proxy", "insider_any", "insider_cluster", "insider_opportunistic",
            "insider_post_results_cluster", "low_attention", "fallen")
    cols = [c for c in (*CONTINUOUS, *FLAGS, *base) if c in x.columns]

    def matrix(f):
        m = pd.DataFrame({c: (est._flag(f[c]) if c in base else pd.to_numeric(f[c], errors="coerce")) for c in cols})
        # within-date ranks for the continuous measures: the model may not learn the calendar
        for c in CONTINUOUS:
            if c in m.columns:
                m[c] = m[c].groupby(f["start_date"].to_numpy()).rank(pct=True)
        return m

    def score(y, p):
        y = np.asarray(y, dtype=float)
        if len(y) < 200 or y.sum() < 10 or y.sum() == len(y):
            return {"windows": int(len(y)), "base_rate": float(y.mean()) if len(y) else np.nan}
        top = p >= np.quantile(p, 0.9)
        return {"windows": int(len(y)), "base_rate": float(y.mean()), "auc": float(roc_auc_score(y, p)),
                "top_decile_hit_rate": float(y[top].mean()), "top_decile_lift": float(y[top].mean() / y.mean())}
    rows = []
    for target, hit, hit_new in TARGETS:
        if hit not in x.columns:
            continue
        d = x[(x["start_year"] <= DISCOVERY_TO) & x[hit].notna()]
        h = x[(x["start_year"] > DISCOVERY_TO) & x[hit].notna()]
        if len(d) < 1000 or d[hit].sum() < 30 or not len(h):
            rows.append({"target": target, "sample": "discovery", "windows": int(len(d)), "note": "too few windows or hits to fit"})
            continue
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                               l2_regularization=1.0, random_state=int(seed) % (2 ** 31))
        md = matrix(d)
        use = [c for c in md.columns if md[c].nunique(dropna=True) >= 2]      # a measure with one value carries nothing
        prob = lambda f: model.predict_proba(matrix(f)[use].to_numpy(dtype=float))[:, 1]  # noqa: E731
        model.fit(md[use].to_numpy(dtype=float), d[hit].to_numpy(dtype=int))
        rows.append({"target": target, "sample": "discovery (fitted here, flattering)", "measures": len(use), **score(d[hit], prob(d))})
        rows.append({"target": target, "sample": "holdout (start dates from 2018)", "measures": len(use), **score(h[hit], prob(h))})
        if xn is not None and hit_new is not None and hit_new in xn.columns:
            n = xn[xn[hit_new].notna()]
            if len(n):
                rows.append({"target": f"{target}: the same model on the two-year outcome", "sample": "new dates", "measures": len(use),
                             **score(n[hit_new], prob(n))})
    return pd.DataFrame(rows)


def digest(t: pd.DataFrame, top: int = 12) -> dict:
    """The rows worth reading: every candidate that held, and the strongest holdout ratios
    either way for each target (so a miss is visible as a miss)."""
    if t is None or not len(t):
        return {"tested": 0, "held": [], "strongest": []}
    held = t[t["verdict"].str.startswith("held")]
    strongest = []
    for target, g in t.dropna(subset=["holdout_lr"]).groupby("target"):
        g = g.assign(_d=np.abs(np.log(g["holdout_lr"].where(g["holdout_lr"] > 0)))).sort_values("_d", ascending=False)
        strongest += g.head(top).drop(columns="_d").to_dict("records")
    return {"tested": int(t["discovery_p"].notna().sum()), "discovery_significant": int((t["verdict"] != "").sum()),
            "held": held.to_dict("records"), "strongest": strongest}


# ---------------------------------------------------------------------------
# The net question (notebook 10): does any combination shift the whole outcome – fewer
# wipe-outs and more multiples at once? Combinations fixed on Nov 13, 2026, before any of
# them was set against an end multiple. Outcomes are end multiples (what holding paid),
# not touches.
# ---------------------------------------------------------------------------

COMBOS = (
    ("positive free cash flow", "fcf_pos"),
    ("fallen", "fallen"),
    ("fallen and cash-positive", "c_fallen_fcf"),
    ("fallen, cash-positive, six-month momentum in the top fifth", "c_fallen_fcf_mom"),
    ("six-month momentum in the top fifth", "c_mom"),
    ("cash-positive with momentum in the top fifth", "c_fcf_mom"),
    ("cash to market cap in the top fifth", "c_cash"),
    ("cash to market cap in the top fifth and cash-positive", "c_cash_fcf"),
    ("floor: net cash at least 30% of market cap", "c_floor"),
    ("floor net of five years of burn at least 30%", "c_floor_net"),
    ("fallen with a floor net of burn", "c_fallen_floor"),
    ("fallen, cash-positive, floor", "c_fallen_fcf_floor"),
    ("cash-positive and shrinking share count", "c_fcf_buyback"),
    ("cash-flow margin in the top fifth and shrinking share count", "c_quality"),
    ("fallen, growing above 10%, cash-positive", "c_fallen_growing_fcf"),
    ("AEM cohort: fallen, cash-positive, net cash, insider purchase", "h2a"),
    ("the 10x corner: price in the lowest fifth and a cash-burner with a short runway", "c_lottery"),
    ("avoid list: short runway, or burning cash with share count in the top fifth", "c_avoid"),
    # the second batch: what the literature suggests for separating winners from losers among the cheap and the fallen
    ("fallen with a fundamentals score of 4 or 5 (Piotroski's idea)", "c_fallen_score_high"),
    ("fallen with a fundamentals score of 0 or 1", "c_fallen_score_low"),
    ("fallen, score of 4 or 5, floor net of burn", "c_fallen_score_floor"),
    ("cheap and good: cash-flow yield and margin both in the top fifth", "c_cheap_good"),
    ("calm and cash-positive: volatility in the lowest fifth, positive free cash flow", "c_calm_fcf"),
    ("volatility in the top fifth", "c_volatile"),
    ("net cash at least two-thirds of market cap (Graham's idea)", "net_cash_two_thirds"),
)


def with_combos(w: pd.DataFrame, panel=None) -> pd.DataFrame:
    x = candidates(w, panel)
    flag = lambda c: est._flag(x[c]).eq(1) if c in x.columns else pd.Series(False, index=x.index)  # noqa: E731
    fcf, cap = _num(x, "fcf"), _num(x, "market_cap").where(lambda v: v > 0)
    pos, fallen = fcf > 0, flag("fallen")
    _, mom = _quintile_flags(x, "momentum_6m")
    _, cash = _quintile_flags(x, "cash_to_cap")
    _, margin = _quintile_flags(x, "fcf_margin")
    low_price, _ = _quintile_flags(x, "entry_price")
    _, issuing = _quintile_flags(x, "share_growth")
    floor = _num(x, "net_cash") / cap
    floor_net = (_num(x, "net_cash") + 5 * fcf.clip(upper=0)) / cap
    known = fcf.notna()

    def put(name, cond, need):
        x[name] = cond.astype(float).where(need)
    put("c_fallen_fcf", fallen & pos, known)
    put("c_fallen_fcf_mom", fallen & pos & mom.eq(1), known & mom.notna())
    put("c_mom", mom.eq(1), mom.notna())
    put("c_fcf_mom", pos & mom.eq(1), known & mom.notna())
    put("c_cash", cash.eq(1), cash.notna())
    put("c_cash_fcf", cash.eq(1) & pos, known & cash.notna())
    put("c_floor", floor >= 0.30, floor.notna())
    put("c_floor_net", floor_net >= 0.30, floor_net.notna())
    put("c_fallen_floor", fallen & (floor_net >= 0.30), floor_net.notna())
    put("c_fallen_fcf_floor", fallen & pos & (floor >= 0.30), known & floor.notna())
    put("c_fcf_buyback", pos & x["shrinking_share_count"].eq(1), known & x["shrinking_share_count"].notna())
    put("c_quality", margin.eq(1) & x["shrinking_share_count"].eq(1), margin.notna() & x["shrinking_share_count"].notna())
    put("c_fallen_growing_fcf", fallen & pos & (_num(x, "revenue_growth") > 0.1), known & _num(x, "revenue_growth").notna())
    put("c_lottery", low_price.eq(1) & x["burner_short_runway"].eq(1), low_price.notna() & x["burner_short_runway"].notna())
    put("c_avoid", x["burner_short_runway"].eq(1) | ((fcf < 0) & issuing.eq(1)), known)
    calm, wild = _quintile_flags(x, "volatility_8q")
    _, cheap = _quintile_flags(x, "fcf_yield")
    put("c_fallen_score_high", fallen & x["score_high"].eq(1), x["score_high"].notna())
    put("c_fallen_score_low", fallen & x["score_low"].eq(1), x["score_low"].notna())
    put("c_fallen_score_floor", fallen & x["score_high"].eq(1) & (floor_net >= 0.30), x["score_high"].notna() & floor_net.notna())
    put("c_cheap_good", cheap.eq(1) & margin.eq(1), cheap.notna() & margin.notna())
    put("c_calm_fcf", calm.eq(1) & pos, calm.notna() & known)
    put("c_volatile", wild.eq(1), wild.notna())
    for c in ("fcf_pos", "fallen", "h2a"):
        x[c] = est._flag(x[c]) if c in x.columns else np.nan
    return x


def _side(g: pd.DataFrame, sig: str, end: str, resamples: int, seed: int) -> dict:
    """A group against the rest of the same sample: the end multiple's shape and the
    firm-cluster intervals of the two differences that define asymmetry."""
    y = g[g[end].notna() & g[sig].notna()]
    s, e = y[sig].to_numpy(dtype=float), y[end].to_numpy(dtype=float)
    n1 = int(s.sum())
    if len(y) < 300 or n1 < 50 or n1 > len(y) - 50:
        return {"windows": n1}
    loss, gain = (e < 0.5).astype(float), (e >= 2).astype(float)
    cl = est.Clusters(y["firm"], resamples, seed)
    _, b = cl.sums(s * loss, s, (1 - s) * loss, 1 - s, s * gain, (1 - s) * gain)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_loss = b[:, 0] / b[:, 1] - b[:, 2] / b[:, 3]
        d_gain = b[:, 4] / b[:, 1] - b[:, 5] / b[:, 3]
    in_, out = e[s == 1], e[s == 0]
    lo_l, hi_l = est._ci(d_loss)
    lo_g, hi_g = est._ci(d_gain)
    fewer, more_loss = hi_l < 0, lo_l > 0
    more, fewer_gain = lo_g > 0, hi_g < 0
    shape = ("asymmetric" if fewer and more else "defensive" if fewer and not fewer_gain else "defensive, costs upside" if fewer
             else "variance" if more_loss and more else "worse" if more_loss else "more upside, same losses" if more else "no difference")
    return {"windows": n1, "median_end": float(np.median(in_)), "median_rest": float(np.median(out)), "mean_end": float(in_.mean()),
            "mean_rest": float(out.mean()), "below_half": float((in_ < 0.5).mean()), "below_half_rest": float((out < 0.5).mean()),
            "at_least_2x": float((in_ >= 2).mean()), "at_least_2x_rest": float((out >= 2).mean()),
            "at_least_3x": float((in_ >= 3).mean()), "at_least_3x_rest": float((out >= 3).mean()), "shape": shape}


def asymmetry(w: pd.DataFrame, w_new: pd.DataFrame | None = None, *, panel=None, resamples: int = 500, seed: int = 20261106,
              combos=None, build=None) -> pd.DataFrame:
    """Every combination on three samples: five-year end multiples on start dates to 2017 and
    from 2018, two-year end multiples on the new dates. `verdict` is the shape that both
    five-year samples agree on (the new dates are shown beside it)."""
    build = build or with_combos
    x = build(w, panel)
    xn = build(w_new, panel) if w_new is not None and len(w_new) else None
    rows = []
    for name, col in (combos if combos is not None else COMBOS):
        if col not in x.columns:
            continue
        parts = [("to 2017, five years", x[x["start_year"] <= DISCOVERY_TO], "end5"),
                 ("from 2018, five years", x[x["start_year"] > DISCOVERY_TO], "end5")]
        if xn is not None and col in xn.columns and "end2" in xn.columns:
            parts.append(("new dates, two years", xn, "end2"))
        shapes = []
        for sample, g, end in parts:
            r = _side(g, col, end, resamples, seed)
            shapes.append(r.get("shape", "too few"))
            rows.append({"combination": name, "sample": sample, **r})
        verdict = shapes[0] if len(shapes) >= 2 and shapes[0] == shapes[1] and shapes[0] != "too few" else "not stable"
        for r in rows[-len(parts):]:
            r["verdict"] = verdict
    return pd.DataFrame(rows)


def outcome_model(w: pd.DataFrame, w_new: pd.DataFrame | None = None, *, panel=None, seed: int = 20261106,
                  extra_continuous=(), extra_flags=(), build=None) -> pd.DataFrame:
    """The flexible version: gradient-boosted trees predict the log end multiple from all
    measures of both batches (fitted on start dates to 2017). Scored out of time by what the best-ranked and
    the worst-ranked tenth of windows actually ended at."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    build = build or with_combos
    x = build(w, panel)
    xn = build(w_new, panel) if w_new is not None and len(w_new) else None
    ranked = (*CONTINUOUS, *LIT_CONTINUOUS, *extra_continuous)
    base = ("fcf_pos", "net_cash_pos", "low_dilution", "s1_proxy", "insider_any", "insider_cluster", "insider_opportunistic",
            "insider_post_results_cluster", "low_attention", "fallen")
    cols = [c for c in (*ranked, *FLAGS, *LIT_FLAGS, *extra_flags, *base) if c in x.columns]

    def matrix(f):
        m = pd.DataFrame({c: (est._flag(f[c]) if c in base else pd.to_numeric(f[c], errors="coerce")) for c in cols})
        for c in ranked:
            if c in m.columns:
                m[c] = m[c].groupby(f["start_date"].to_numpy()).rank(pct=True)
        return m
    d = x[(x["start_year"] <= DISCOVERY_TO) & x["end5"].notna()]
    if len(d) < 1000:
        return pd.DataFrame([{"sample": "to 2017", "windows": int(len(d)), "note": "too few windows to fit"}])
    md = matrix(d)
    use = [c for c in md.columns if md[c].nunique(dropna=True) >= 2]
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200, l2_regularization=1.0,
                                          loss="absolute_error", random_state=int(seed) % (2 ** 31))
    model.fit(md[use].to_numpy(dtype=float), np.log(d["end5"].clip(lower=0.01)).to_numpy())
    rows = []
    for sample, g, end in (("to 2017, five years (fitted here, flattering)", d, "end5"),
                           ("from 2018, five years", x[(x["start_year"] > DISCOVERY_TO) & x["end5"].notna()], "end5"),
                           ("new dates, two years", xn[xn["end2"].notna()] if xn is not None and "end2" in xn.columns else None, "end2")):
        if g is None or len(g) < 300:
            continue
        p = model.predict(matrix(g)[use].to_numpy(dtype=float))
        e = g[end].to_numpy(dtype=float)
        for label, pick in (("best-ranked tenth", p >= np.quantile(p, 0.9)), ("best-ranked fifth", p >= np.quantile(p, 0.8)),
                            ("all windows", np.ones(len(e), dtype=bool)), ("worst-ranked fifth", p <= np.quantile(p, 0.2)),
                            ("worst-ranked tenth", p <= np.quantile(p, 0.1))):
            v = e[pick]
            rows.append({"sample": sample, "group": label, "windows": int(len(v)), "median_end": float(np.median(v)),
                         "mean_end": float(v.mean()), "below_half": float((v < 0.5).mean()), "at_least_2x": float((v >= 2).mean()),
                         "at_least_3x": float((v >= 3).mean())})
    return pd.DataFrame(rows)
