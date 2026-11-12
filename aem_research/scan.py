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
         fdr: float = 0.10, alpha: float = 0.05) -> pd.DataFrame:
    """One row per candidate, side and target: the ratio on discovery, holdout and the new
    dates, and whether it held."""
    x = candidates(w, panel)
    xn = candidates(w_new, panel) if w_new is not None and len(w_new) else None
    disc, hold = x[x["start_year"] <= DISCOVERY_TO], x[x["start_year"] > DISCOVERY_TO]
    sides = []
    for c in CONTINUOUS:
        if c in x.columns and x[c].notna().any():
            lo, hi = _quintile_flags(x, c)
            lo_n, hi_n = _quintile_flags(xn, c) if xn is not None and c in xn.columns else (None, None)
            sides += [(c, "lowest fifth", lo, lo_n), (c, "highest fifth", hi, hi_n)]
    for c in FLAGS:
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
