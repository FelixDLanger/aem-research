"""Paper B's estimators for H2 to H6, as the locked pre-registration's estimation section sets
them out: likelihood ratios with firm-cluster bootstrap intervals and Holm correction, the
horizon conversion on paired windows, the out-of-sample split, the 10x logit against the
unit-weighted checklist and the base rate by Brier score, the ordered logit on the six
buckets, and the robustness versions that the saved tables allow.

Inputs are the tables of the locked data run: the features of every universe window and the
outcomes notebook 04 computed from them. Nothing here reads prices or filings.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import H4_SIGNALS
from .outcomes import COMPLETE, DELISTED, ENDED

RESOLVED = (COMPLETE, DELISTED, ENDED)
SIGNAL_NAMES = {"s1_proxy": "S1 (FCF per share outgrew revenue)", "insider_any": "S4 any purchase",
                "insider_cluster": "S4 cluster", "insider_opportunistic": "S4 opportunistic", "net_cash_pos": "floor (net cash)",
                "fcf_pos": "positive free cash flow", "low_dilution": "low dilution", "low_attention": "low attention"}


# ---------------------------------------------------------------------------
# Windows: features joined with outcomes
# ---------------------------------------------------------------------------

def windows(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per universe window with its features and, per horizon, the resolved outcome:
    hit3_3y (touched 3x within 3 years), hit10_5y, hit624_5y, max5, bucket5, end5 and bounds.
    A horizon that is not resolved (censored, no entry) leaves its columns missing."""
    f = features[features["in_universe"].fillna(False).astype(bool)].copy()
    f["start_date"] = pd.to_datetime(f["start_date"]).astype("datetime64[ns]")
    f["firm"] = pd.to_numeric(f["cik"], errors="coerce")
    o = outcomes[outcomes["status"].isin(RESOLVED)].copy()
    o["start_date"] = pd.to_datetime(o["start_date"]).astype("datetime64[ns]")
    num = lambda c: pd.to_numeric(o[c], errors="coerce") if c in o.columns else pd.Series(np.nan, index=o.index)  # noqa: E731
    o3 = o[o["horizon_years"] == 3].assign(hit3_3y=num("touch_3x"), days3=num("days_to_3x"))[
        ["ticker", "start_date", "hit3_3y", "days3"]]
    o5 = o[o["horizon_years"] == 5].assign(hit10_5y=num("touch_10x"), hit624_5y=num("touch_6.24x"), hit3_5y=num("touch_3x"),
                                           max5=num("max_multiple"), bucket5=num("bucket"), end5=num("end_multiple"),
                                           end5_lower=num("end_multiple_lower"), end5_upper=num("end_multiple_upper"),
                                           days10=num("days_to_10x"))
    o5 = o5[["ticker", "start_date", "hit10_5y", "hit624_5y", "hit3_5y", "max5", "bucket5", "end5", "end5_lower",
             "end5_upper", "days10", "delist_category"]]
    w = f.merge(o3, on=["ticker", "start_date"], how="left").merge(o5, on=["ticker", "start_date"], how="left")
    w["start_year"] = w["start_date"].dt.year
    w["regime"] = np.where(w["start_date"] < pd.Timestamp("2022-01-01"), "before 2022", "2022 and later")
    return w


def _flag(s: pd.Series) -> pd.Series:
    """A signal as float 1/0 with NaN where unknown."""
    if s.dtype == bool:
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s) and s.dtype != "boolean":
        v = pd.to_numeric(s, errors="coerce").astype(float)
        return v.where(v.isna(), (v != 0).astype(float))
    return s.astype("boolean").astype("Float64").astype(float)


# ---------------------------------------------------------------------------
# Firm-cluster bootstrap on sums
# ---------------------------------------------------------------------------

class Clusters:
    """Per-firm sums of any number of columns and bootstrap weights over the firms. Every
    statistic here is a function of sums, so a resample is one matrix product."""

    def __init__(self, firm, resamples: int = 2000, seed: int = 20261106):
        codes, self.firms = pd.factorize(pd.Series(firm).to_numpy(), sort=False)
        self.codes, self.n = codes, len(self.firms)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.n, size=(resamples, self.n))
        flat = (np.arange(resamples)[:, None] * self.n + draws).ravel()
        self.weights = np.bincount(flat, minlength=resamples * self.n).reshape(resamples, self.n).astype(np.float32)

    def sums(self, *columns):
        """(point sums, bootstrap sums) for each column: arrays of shape (k,) and (resamples, k)."""
        m = np.column_stack([np.nan_to_num(np.asarray(c, dtype=float)) for c in columns])
        per_firm = np.zeros((self.n, m.shape[1]))
        np.add.at(per_firm, self.codes, m)
        return per_firm.sum(axis=0), self.weights @ per_firm


def _ci(boot, level: float = 0.95):
    b = boot[np.isfinite(boot)]
    if len(b) < 50:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    return (float(np.quantile(b, a)), float(np.quantile(b, 1 - a)))


def _p_above(boot, null: float = 1.0) -> float:
    """One-sided bootstrap p-value for 'statistic > null': the share of resamples at or below it."""
    b = boot[np.isfinite(boot)]
    return float(((b <= null).sum() + 1) / (len(b) + 1)) if len(b) >= 50 else np.nan


def _p_two_sided(boot, null: float = 1.0) -> float:
    b = boot[np.isfinite(boot)]
    if len(b) < 50:
        return np.nan
    lo, hi = ((b <= null).sum() + 1) / (len(b) + 1), ((b >= null).sum() + 1) / (len(b) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvalues: dict) -> dict:
    known = sorted(((k, p) for k, p in pvalues.items() if p is not None and np.isfinite(p)), key=lambda kp: kp[1])
    out, running, m = {k: np.nan for k in pvalues}, 0.0, len(known)
    for i, (k, p) in enumerate(known):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(b) > 0, np.asarray(a, dtype=float) / np.asarray(b, dtype=float), np.nan)


def likelihood_ratio(w: pd.DataFrame, signal: str, hit: str, *, resamples: int = 2000, seed: int = 20261106,
                     clusters: Clusters | None = None) -> dict:
    """LR = P(signal | hit) / P(signal | miss) on the windows where both are known, with the
    hit rates behind it, a 95% firm-cluster bootstrap interval and the one-sided p for LR > 1."""
    s, h = _flag(w[signal]), pd.to_numeric(w[hit], errors="coerce")
    ok = (s.notna() & h.notna()).to_numpy()
    out = {"signal": signal, "windows": int(ok.sum()), "firms": int(w.loc[ok, "firm"].nunique()), "with_signal": int(s[ok].sum()),
           "left_out_share": float(1 - ok.sum() / max(int(h.notna().sum()), 1))}
    if not ok.any():
        return {**out, "base_rate": np.nan, "hit_rate_with_signal": np.nan, "lr": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_one_sided": np.nan}
    x = w.loc[ok]
    s, h = s[ok].to_numpy(), h[ok].to_numpy()
    cl = clusters if clusters is not None and len(clusters.codes) == len(x) else Clusters(x["firm"], resamples, seed)
    point, boot = cl.sums(s * h, h, s * (1 - h), 1 - h, s)

    def lr(v):
        return _div(_div(v[..., 0], v[..., 1]), _div(v[..., 2], v[..., 3]))
    b = lr(boot)
    lo, hi = _ci(b)
    return {**out, "base_rate": float(h.mean()), "hit_rate_with_signal": float(_div(point[0], point[4])),
            "lr": float(lr(point)), "ci_low": lo, "ci_high": hi, "p_one_sided": _p_above(b)}


# ---------------------------------------------------------------------------
# H2: the fallen-quality cohort
# ---------------------------------------------------------------------------

def h2(w: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106, hit: str = "hit3_3y", fallen: str = "fallen") -> dict:
    """H2a and H2b (Holm across the two), the placebo, H2c within the cohort, and the ablation."""
    x = w[w["cohort_known"].fillna(False).astype(bool) & w[hit].notna()].copy()
    quality = _flag(x["fcf_pos"]).eq(1) & _flag(x["net_cash_pos"]).eq(1)
    fall, ins, opp = _flag(x[fallen]).eq(1), _flag(x["insider_any"]).eq(1), _flag(x["insider_opportunistic"]).eq(1)
    x["_h2a"], x["_h2b"] = (fall & quality & ins).astype(float), (fall & quality & opp).astype(float)
    x["_placebo"] = (fall & quality & _flag(x["sale_any"]).eq(1) & ~ins).astype(float)
    cl = Clusters(x["firm"], resamples, seed)
    kw = {"resamples": resamples, "seed": seed, "clusters": cl}
    main = {k: likelihood_ratio(x, c, hit, **kw) for k, c in (("H2a", "_h2a"), ("H2b", "_h2b"))}
    adj = holm({k: v["p_one_sided"] for k, v in main.items()})
    for k in main:
        main[k]["holm_p"] = adj[k]
        main[k]["verdict"] = ("not testable" if not np.isfinite(adj[k]) else "supported" if adj[k] < 0.05 and main[k]["lr"] > 1
                              else "not supported")
    placebo = likelihood_ratio(x, "_placebo", hit, **kw)
    # ablation: the cohort without each condition, and each condition alone
    drops = {"full cohort (H2a)": fall & quality & ins, "without the fall": quality & ins,
             "without positive FCF": fall & _flag(x["net_cash_pos"]).eq(1) & ins,
             "without net cash": fall & _flag(x["fcf_pos"]).eq(1) & ins, "without the insider purchase": fall & quality,
             "the fall alone": fall, "positive FCF alone": _flag(x["fcf_pos"]).eq(1),
             "net cash alone": _flag(x["net_cash_pos"]).eq(1), "an insider purchase alone": ins}
    ablation = []
    for name, m in drops.items():
        x["_c"] = m.astype(float)
        ablation.append({"cohort": name, **{k: v for k, v in likelihood_ratio(x, "_c", hit, **kw).items() if k != "signal"}})
    # H2c: inside the cohort, by the kind of insider who bought
    inside = x[x["_h2a"].eq(1)]
    kinds = {"opportunistic": _flag(inside["insider_opportunistic"]).eq(1),
             "routine only": _flag(inside.get("insider_routine_only", pd.Series(np.nan, index=inside.index))).eq(1),
             "unclassified only": _flag(inside.get("insider_unclassified_only", pd.Series(np.nan, index=inside.index))).eq(1)}
    h2c = [{"insiders": k, "windows": int(m.sum()), "firms": int(inside.loc[m, "firm"].nunique()),
            "hit_rate": float(inside.loc[m, hit].mean()) if m.any() else np.nan} for k, m in kinds.items()]
    diff = {"difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_two_sided": np.nan}
    a, b = kinds["opportunistic"].to_numpy(), kinds["routine only"].to_numpy()
    if a.sum() >= 5 and b.sum() >= 5:
        ci = Clusters(inside["firm"], resamples, seed)
        hv = inside[hit].to_numpy(dtype=float)
        point, boot = ci.sums(a * hv, a, b * hv, b)
        d = _div(boot[:, 0], boot[:, 1]) - _div(boot[:, 2], boot[:, 3])
        lo, hi = _ci(d)
        diff = {"difference": float(point[0] / point[1] - point[2] / point[3]), "ci_low": lo, "ci_high": hi,
                "p_two_sided": _p_two_sided(d, 0.0)}
    return {"main": main, "placebo": placebo, "ablation": pd.DataFrame(ablation), "h2c": pd.DataFrame(h2c), "h2c_test": diff,
            "windows": int(len(x)), "firms": int(x["firm"].nunique()), "base_rate": float(x[hit].mean()) if len(x) else np.nan}


# ---------------------------------------------------------------------------
# H3: is 3x in three years the same event as 6.24x in five?
# ---------------------------------------------------------------------------

def h3(w: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106) -> dict:
    """On windows resolved at both horizons: P(3x within 3y), P(6.24x within 5y), their ratio
    with a firm-cluster interval and a two-sided test of ratio = 1, and the multiple that
    five years would have to reach as often as three years reach 3x."""
    x = w[w["hit3_3y"].notna() & w["hit624_5y"].notna()]
    if not len(x):
        return {"windows": 0}
    a, b = x["hit3_3y"].to_numpy(dtype=float), x["hit624_5y"].to_numpy(dtype=float)
    cl = Clusters(x["firm"], resamples, seed)
    point, boot = cl.sums(a, b)
    ratio = _div(boot[:, 1], boot[:, 0])
    lo, hi = _ci(ratio)
    p3 = float(a.mean())
    peak = pd.to_numeric(x["max5"], errors="coerce").dropna()
    equivalent = float(np.quantile(peak, 1 - p3)) if len(peak) and 0 < p3 < 1 else np.nan
    p = _p_two_sided(ratio)
    return {"windows": int(len(x)), "firms": int(x["firm"].nunique()), "p_3x_3y": p3, "p_624x_5y": float(b.mean()),
            "ratio": float(point[1] / point[0]) if point[0] else np.nan, "ci_low": lo, "ci_high": hi, "p_two_sided": p,
            "equivalent_multiple_5y": equivalent,
            "verdict": "not testable" if not np.isfinite(p) else "mapping rejected" if p < 0.05 else "mapping kept"}


# ---------------------------------------------------------------------------
# H4: signal likelihood ratios out of sample, and the model against the checklist
# ---------------------------------------------------------------------------

def split(w: pd.DataFrame, fit_to: int = 2016, test=(2017, 2021)):
    return w[w["start_year"] <= fit_to], w[w["start_year"].between(*test)]


def h4(w: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106, hit: str = "hit10_5y", first="2013-07-01") -> dict:
    """Each signal's likelihood ratio for 10x within five years: fitted period (start dates to
    2016) and test period (2017 to 2021); one-sided tests in the test period, Holm across the eight."""
    x = w[(w["start_date"] >= pd.Timestamp(first)) & w[hit].notna()]
    fit, test = split(x)
    rows = []
    for s in H4_SIGNALS:
        if s not in x.columns:
            continue
        a, b = likelihood_ratio(fit, s, hit, resamples=resamples, seed=seed), likelihood_ratio(test, s, hit, resamples=resamples, seed=seed)
        rows.append({"signal": s, "name": SIGNAL_NAMES.get(s, s), "fit_windows": a["windows"], "fit_lr": a["lr"],
                     "fit_ci_low": a["ci_low"], "fit_ci_high": a["ci_high"], "test_windows": b["windows"],
                     "test_firms": b["firms"], "test_with_signal": b["with_signal"], "test_base_rate": b["base_rate"],
                     "test_hit_rate_with_signal": b["hit_rate_with_signal"], "test_lr": b["lr"], "test_ci_low": b["ci_low"],
                     "test_ci_high": b["ci_high"], "p_one_sided": b["p_one_sided"], "left_out_share": b["left_out_share"]})
    t = pd.DataFrame(rows)
    if len(t):
        adj = holm(dict(zip(t["signal"], t["p_one_sided"])))
        t["holm_p"] = t["signal"].map(adj)
        t["verdict"] = np.where(~np.isfinite(t["holm_p"]), "not testable",
                                np.where((t["holm_p"] < 0.05) & (t["test_lr"] > 1), "supported", "not supported"))
    return {"table": t, "fit_windows": int(len(fit)), "test_windows": int(len(test)),
            "fit_base_rate": float(fit[hit].mean()) if len(fit) else np.nan,
            "test_base_rate": float(test[hit].mean()) if len(test) else np.nan}


def _design(x: pd.DataFrame, regressors, dates=None):
    """Regressor matrix plus start-date dummies (first date dropped)."""
    X = np.column_stack([pd.to_numeric(_flag(x[c]) if x[c].dtype != float else x[c], errors="coerce").to_numpy(dtype=float)
                         for c in regressors]) if regressors else np.zeros((len(x), 0))
    dates = sorted(pd.unique(x["start_date"])) if dates is None else dates
    D = np.column_stack([(x["start_date"].to_numpy() == d).astype(float) for d in dates[1:]]) if len(dates) > 1 else \
        np.zeros((len(x), 0))
    return X, D, dates


def _cluster_cov(scores: np.ndarray, hessian_inv: np.ndarray, firm) -> np.ndarray:
    codes, uniq = pd.factorize(np.asarray(firm))
    g = np.zeros((len(uniq), scores.shape[1]))
    np.add.at(g, codes, scores)
    meat = g.T @ g * (len(uniq) / max(len(uniq) - 1, 1))
    return hessian_inv @ meat @ hessian_inv


def logit(y, X, firm, *, ridge: float = 1e-8, max_iter: int = 100) -> dict:
    """Logit by Newton's method with an intercept; firm-clustered standard errors."""
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(A.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
        H = (A * (p * (1 - p))[:, None]).T @ A + ridge * np.eye(A.shape[1])
        step = np.linalg.solve(H, A.T @ (y - p))
        beta = beta + step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
    H_inv = np.linalg.pinv((A * (p * (1 - p))[:, None]).T @ A + ridge * np.eye(A.shape[1]))
    cov = _cluster_cov(A * (y - p)[:, None], H_inv, firm)
    return {"beta": beta, "se": np.sqrt(np.clip(np.diag(cov), 0, None)), "p_hat": p}


def _coef_table(names, beta, se) -> pd.DataFrame:
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return pd.DataFrame({"regressor": names, "coef": beta, "odds_ratio": np.exp(np.clip(beta, -30, 30)), "se": se, "z": z,
                         "p": [float(math.erfc(abs(v) / math.sqrt(2))) if np.isfinite(v) else np.nan for v in z]})


REGRESSORS = (*H4_SIGNALS, "h2a", "net_cash_to_cap")


def usable(x: pd.DataFrame, regs: list, y=None) -> tuple[list, dict]:
    """Regressors a model can estimate on this sample, and why the others cannot: a regressor
    that never varies, or (for a yes/no outcome) a flag with no hit on one of its sides –
    there the likelihood has no maximum and any coefficient printed would be an artefact."""
    keep, dropped = [], {}
    for c in regs:
        v = x[c].to_numpy(dtype=float)
        if np.nanmax(v) == np.nanmin(v):
            dropped[c] = "no variation in the sample"
            continue
        if y is not None and set(np.unique(v)) <= {0.0, 1.0}:
            hits = np.asarray(y, dtype=float)
            if hits[v == 1].sum() == 0 or hits[v == 0].sum() == 0:
                dropped[c] = "no hit with the signal" if hits[v == 1].sum() == 0 else "no hit without the signal"
                continue
        keep.append(c)
    return keep, dropped


def _with_dropped(table: pd.DataFrame, dropped: dict, order: list) -> pd.DataFrame:
    extra = pd.DataFrame([{"regressor": c, "note": why} for c, why in dropped.items()])
    t = pd.concat([table.assign(note=""), extra], ignore_index=True) if len(extra) else table.assign(note="")
    return t.set_index("regressor").reindex(order).reset_index()


def model_frame(w: pd.DataFrame, hit: str, first="2013-07-01") -> tuple[pd.DataFrame, list]:
    """Windows with the outcome and every regressor known (complete cases), and the regressor list."""
    x = w[(w["start_date"] >= pd.Timestamp(first)) & w[hit].notna()].copy()
    cap = pd.to_numeric(x["market_cap"], errors="coerce")
    x["net_cash_to_cap"] = (pd.to_numeric(x["net_cash"], errors="coerce") / cap.where(cap > 0)).clip(-2, 2)
    regs = [c for c in REGRESSORS if c in x.columns]
    for c in regs:
        if c != "net_cash_to_cap":
            x[c] = _flag(x[c])
    total = len(x)
    x = x.dropna(subset=regs)
    x.attrs["left_out_share"] = float(1 - len(x) / total) if total else np.nan
    return x, regs


def m2_and_benchmark(w: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106, hit: str = "hit10_5y") -> dict:
    """M2 (the 10x logit with start-date fixed effects, firm-clustered errors) on all complete
    windows; then, fitted on start dates to 2016 and scored on 2017 to 2021 by Brier score,
    against the unit-weighted checklist of the same signals and the base rate alone.
    Out of sample the fixed effects of unseen dates do not exist: the fitted model predicts
    with the average fitted fixed effect, the checklist is a one-variable logit on the number
    of signals present, the base rate is the fitted period's hit rate."""
    x, regs = model_frame(w, hit)
    out = {"windows": int(len(x)), "left_out_share": x.attrs.get("left_out_share"), "coefficients": pd.DataFrame(),
           "brier": pd.DataFrame()}
    if len(x) < 200 or x[hit].sum() < 10:
        return out
    all_regs = regs
    regs, dropped = usable(x, all_regs, x[hit])
    X, D, _ = _design(x, regs)
    full = logit(x[hit], np.column_stack([X, D]), x["firm"])
    k = len(regs)
    out["coefficients"] = _with_dropped(_coef_table(regs, full["beta"][1:k + 1], full["se"][1:k + 1]), dropped, all_regs)
    fit, test = split(x)
    if len(fit) < 200 or len(test) < 200 or fit[hit].sum() < 10:
        return out
    regs, dropped = usable(fit, all_regs, fit[hit])                      # what the fitted period alone can estimate
    k = len(regs)
    Xf, Df, dates = _design(fit, regs)
    m = logit(fit[hit], np.column_stack([Xf, Df]), fit["firm"])
    fe = np.r_[0.0, m["beta"][k + 1:]]
    weights = np.array([(fit["start_date"].to_numpy() == d).mean() for d in dates])
    intercept = m["beta"][0] + float(fe @ weights)                       # the average fitted fixed effect
    Xt = _design(test, regs, dates)[0]
    p_model = 1 / (1 + np.exp(-(intercept + Xt @ m["beta"][1:k + 1])))
    signals = [c for c in all_regs if c in H4_SIGNALS]              # the checklist counts all eight
    score_f, score_t = fit[signals].sum(axis=1).to_numpy(), test[signals].sum(axis=1).to_numpy()
    c = logit(fit[hit], score_f[:, None], fit["firm"])
    p_check = 1 / (1 + np.exp(-(c["beta"][0] + c["beta"][1] * score_t)))
    p_base = np.full(len(test), float(fit[hit].mean()))
    y = test[hit].to_numpy(dtype=float)
    cl = Clusters(test["firm"], resamples, seed)
    errs = {"M2 (fitted logit)": (p_model - y) ** 2, "unit-weighted checklist": (p_check - y) ** 2, "base rate": (p_base - y) ** 2}
    point, boot = cl.sums(*errs.values(), np.ones(len(y)))
    rows = []
    for i, name in enumerate(errs):
        rows.append({"model": name, "brier": float(point[i] / point[-1]), "test_windows": int(len(y))})
    table = pd.DataFrame(rows)
    b = boot[:, :3] / boot[:, [3]]
    for i, j, label in ((0, 1, "M2 minus checklist"), (0, 2, "M2 minus base rate"), (1, 2, "checklist minus base rate")):
        d = b[:, i] - b[:, j]
        lo, hi = _ci(d)
        rows.append({"model": label, "brier": float(point[i] / point[-1] - point[j] / point[-1]), "ci_low": lo, "ci_high": hi,
                     "p_two_sided": _p_two_sided(d, 0.0), "test_windows": int(len(y))})
    table = pd.DataFrame(rows)
    beats = table.loc[table["model"].eq("M2 minus checklist")].iloc[0]
    out.update({"brier": table, "fit_windows": int(len(fit)), "test_windows": int(len(test)),
                "terminal_uses": "the fitted model" if beats["brier"] < 0 and beats["ci_high"] < 0 else "the checklist",
                "oos_coefficients": _with_dropped(_coef_table(regs, m["beta"][1:k + 1], m["se"][1:k + 1]), dropped, all_regs)})
    return out


def ordered_logit(y, X, firm, *, max_iter: int = 500) -> dict:
    """Ordered logit (proportional odds) by maximum likelihood; firm-clustered standard errors.
    y holds ordered categories 0..J-1; thresholds are kept increasing by construction."""
    from scipy import optimize
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)
    n, k, J = len(y), X.shape[1], int(y.max()) + 1

    def unpack(theta):
        cuts = np.cumsum(np.r_[theta[k], np.exp(theta[k + 1:])])
        return theta[:k], cuts

    def parts(theta):
        beta, cuts = unpack(theta)
        eta = X @ beta
        upper = np.r_[cuts, np.inf][y] - eta
        lower = np.r_[-np.inf, cuts][y] - eta
        F = lambda v: 1 / (1 + np.exp(-np.clip(v, -35, 35)))  # noqa: E731
        Fu, Fl = np.where(np.isinf(upper), 1.0, F(upper)), np.where(np.isinf(lower), 0.0, F(lower))
        prob = np.clip(Fu - Fl, 1e-300, None)
        fu, fl = Fu * (1 - Fu), Fl * (1 - Fl)
        return beta, cuts, prob, fu, fl

    def scores(theta):
        beta, cuts, prob, fu, fl = parts(theta)
        s = np.zeros((n, len(theta)))
        s[:, :k] = X * ((fl - fu) / prob)[:, None]
        dcut = np.zeros((n, J - 1))                         # d loglik / d cut_j
        idx = np.arange(n)
        has_u, has_l = y < J - 1, y > 0
        dcut[idx[has_u], y[has_u]] += (fu / prob)[has_u]
        dcut[idx[has_l], y[has_l] - 1] -= (fl / prob)[has_l]
        # cut_j = theta_k + sum_{i<=j} exp(theta_{k+i}); chain rule
        s[:, k] = dcut.sum(axis=1)
        for j in range(1, J - 1):
            s[:, k + j] = dcut[:, j:].sum(axis=1) * np.exp(theta[k + j])
        return s

    def negll(theta):
        return -np.log(parts(theta)[2]).sum()

    def grad(theta):
        return -scores(theta).sum(axis=0)

    share = np.clip(np.cumsum(np.bincount(y, minlength=J))[:-1] / n, 1e-4, 1 - 1e-4)
    cuts0 = np.log(share / (1 - share))
    theta0 = np.r_[np.zeros(k), cuts0[0], np.log(np.clip(np.diff(cuts0), 1e-3, None))]
    res = optimize.minimize(negll, theta0, jac=grad, method="BFGS", options={"maxiter": max_iter, "gtol": 1e-6})
    theta = res.x
    eps = 1e-5
    H = np.zeros((len(theta), len(theta)))
    for i in range(len(theta)):
        e = np.zeros(len(theta))
        e[i] = eps
        H[:, i] = (grad(theta + e) - grad(theta - e)) / (2 * eps)
    H_inv = np.linalg.pinv((H + H.T) / 2)
    cov = _cluster_cov(scores(theta), H_inv, firm)
    beta, cuts = unpack(theta)
    return {"beta": beta, "se": np.sqrt(np.clip(np.diag(cov), 0, None))[:k], "cuts": cuts,
            "converged": bool(res.success or np.abs(grad(theta)).max() / n < 1e-4),   # BFGS may stop on precision at the optimum
            "loglik": float(-res.fun)}


def m1(w: pd.DataFrame, labels=(0.1, 0.4, 1.0, 2.5, 5.0, 10.0), column: str = "bucket5") -> dict:
    """M1: the five-year end multiple on the six-bucket grid, ordered logit with start-date
    fixed effects and firm-clustered errors, on the complete windows."""
    x, regs = model_frame(w, column)
    out = {"windows": int(len(x)), "left_out_share": x.attrs.get("left_out_share"), "coefficients": pd.DataFrame()}
    if len(x) < 300:
        return out
    y = pd.Series(np.searchsorted(np.asarray(labels), x[column].to_numpy(dtype=float)), index=x.index).clip(0, len(labels) - 1)
    all_regs = regs
    regs, dropped = usable(x, all_regs)
    X, D, _ = _design(x, regs)
    fit = ordered_logit(y, np.column_stack([X, D]), x["firm"])
    k = len(regs)
    out.update({"coefficients": _with_dropped(_coef_table(regs, fit["beta"][:k], fit["se"][:k]), dropped, all_regs),
                "converged": fit["converged"]})
    return out


# ---------------------------------------------------------------------------
# H5, H6, descriptives and the robustness versions the saved tables allow
# ---------------------------------------------------------------------------

def h5_regime(w: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106) -> pd.DataFrame:
    """Exploratory: the three-year base rate and likelihood ratios before 2022 against 2022 and later."""
    rows = []
    for regime, g in w[w["hit3_3y"].notna()].groupby("regime"):
        for s in ("h2a", *H4_SIGNALS):
            if s in g.columns:
                r = likelihood_ratio(g, s, "hit3_3y", resamples=resamples, seed=seed)
                rows.append({"regime": regime, "signal": s, "windows": r["windows"], "with_signal": r["with_signal"],
                             "base_rate": r["base_rate"], "hit_rate_with_signal": r["hit_rate_with_signal"], "lr": r["lr"],
                             "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def h6_post_results(w: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106) -> pd.DataFrame:
    """Exploratory: purchases in a cluster within 14 days after results against other purchases."""
    if "insider_post_results_cluster" not in w.columns:
        return pd.DataFrame()
    x = w[w["hit3_3y"].notna() & _flag(w["insider_any"]).eq(1)]
    rows = []
    for name, g in (("among windows with a purchase", x), ("among all windows", w[w["hit3_3y"].notna()])):
        r = likelihood_ratio(g, "insider_post_results_cluster", "hit3_3y", resamples=resamples, seed=seed)
        rows.append({"scope": name, **{k: r[k] for k in ("windows", "with_signal", "base_rate", "hit_rate_with_signal", "lr",
                                                         "ci_low", "ci_high")}})
    return pd.DataFrame(rows)


def touch_years(w: pd.DataFrame) -> pd.DataFrame:
    """The calendar year of each first 3x (three-year windows) and 10x (five-year windows) touch."""
    rows = []
    for col, days, label in (("hit3_3y", "days3", "first 3x touch"), ("hit10_5y", "days10", "first 10x touch")):
        x = w[w[col].eq(1) & w[days].notna()]
        year = (x["start_date"] + pd.to_timedelta(x[days], unit="D")).dt.year
        for y, n in year.value_counts().sort_index().items():
            rows.append({"event": label, "year": int(y), "touches": int(n), "share": float(n / len(x))})
    return pd.DataFrame(rows)


def delisting_classes(w: pd.DataFrame) -> pd.DataFrame:
    """Five-year windows by how the series ended, by start year."""
    x = w[w["end5"].notna()].copy()
    x["class"] = x["delist_category"].fillna("").astype(str).replace("", "survived")
    t = (x.groupby(["start_year", "class"]).size().unstack(fill_value=0))
    t = t.div(t.sum(axis=1), axis=0).reset_index()
    return t


def robustness(w: pd.DataFrame, *, resamples: int = 1000, seed: int = 20261106) -> pd.DataFrame:
    """H2a, H2b and the H4 test-period ratios again on the versions the saved tables allow."""
    versions = {
        "primary": w,
        "January formation dates only": w[w["start_date"].dt.month == 1],
        "market cap from US$300m": w[pd.to_numeric(w["market_cap"], errors="coerce") >= 3e8],
        "without the April and July 2020 start dates": w[~w["start_date"].isin(pd.to_datetime(["2020-04-01", "2020-07-01"]))],
        "H2 on start dates from 2016": w[w["start_date"] >= pd.Timestamp("2016-01-01")],
    }
    rows = []
    for name, g in versions.items():
        if len(g) < 100:
            continue
        r2 = h2(g, resamples=resamples, seed=seed)
        r4 = h4(g, resamples=resamples, seed=seed)["table"]
        row = {"version": name, "windows": int(len(g)), "h2a_lr": r2["main"]["H2a"]["lr"], "h2a_ci_low": r2["main"]["H2a"]["ci_low"],
               "h2a_ci_high": r2["main"]["H2a"]["ci_high"], "h2a_windows": r2["main"]["H2a"]["with_signal"],
               "h2b_lr": r2["main"]["H2b"]["lr"], "placebo_lr": r2["placebo"]["lr"]}
        for s in r4.itertuples():
            row[f"lr_{s.signal}"] = s.test_lr
        rows.append(row)
    if "fallen_relative" in w.columns:
        r2 = h2(w, resamples=resamples, seed=seed, fallen="fallen_relative")
        rows.append({"version": "drawdown relative to the universe median (H2)", "windows": int(len(w)),
                     "h2a_lr": r2["main"]["H2a"]["lr"], "h2a_ci_low": r2["main"]["H2a"]["ci_low"],
                     "h2a_ci_high": r2["main"]["H2a"]["ci_high"], "h2a_windows": r2["main"]["H2a"]["with_signal"],
                     "h2b_lr": r2["main"]["H2b"]["lr"], "placebo_lr": r2["placebo"]["lr"]})
    return pd.DataFrame(rows)


NOT_COMPUTED = ("market cap from US$50m", "no minimum price", "Nasdaq performance delistings at -30%",
                "insider classification on purchases only", "multiples in real terms",
                "without windows crossing a neutralized jump or a weak-evidence split",
                "likelihood-ratio intervals with start years resampled as well as firms")


def run_all(features: pd.DataFrame, outcomes: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106) -> dict:
    w = windows(features, outcomes)
    kw = {"resamples": resamples, "seed": seed}
    return {"windows": w, "h2": h2(w, **kw), "h3": h3(w, **kw), "h4": h4(w, **kw), "m2": m2_and_benchmark(w, **kw), "m1": m1(w),
            "h5": h5_regime(w, **kw), "h6": h6_post_results(w, **kw), "touch_years": touch_years(w),
            "delisting_classes": delisting_classes(w), "robustness": robustness(w, resamples=min(resamples, 1000), seed=seed)}


# ---------------------------------------------------------------------------
# Follow-up robustness (notebook 08): start years resampled with firms; other universes
# ---------------------------------------------------------------------------

def likelihood_ratio_two_way(w: pd.DataFrame, signal: str, hit: str, *, resamples: int = 2000, seed: int = 20261106) -> dict:
    """The likelihood ratio with firms and start years resampled together: a window's weight is
    its firm's draw count times its start year's. Market-wide shocks hit every firm of a year at
    once, so this interval is wider wherever one year carries the result."""
    s, h = _flag(w[signal]), pd.to_numeric(w[hit], errors="coerce")
    ok = (s.notna() & h.notna()).to_numpy()
    x = w.loc[ok]
    if not len(x):
        return {"signal": signal, "windows": 0, "years": 0, "lr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_one_sided": np.nan}
    s, h = s[ok].to_numpy(), h[ok].to_numpy()
    fc, firms = pd.factorize(x["firm"].to_numpy())
    yc, years = pd.factorize(x["start_year"].to_numpy())
    cols = np.column_stack([s * h, h, s * (1 - h), 1 - h])
    cell = np.zeros((len(firms), len(years), 4))
    np.add.at(cell, (fc, yc), cols)
    rng = np.random.default_rng(seed)

    def weights(n):
        draws = rng.integers(0, n, size=(resamples, n))
        flat = (np.arange(resamples)[:, None] * n + draws).ravel()
        return np.bincount(flat, minlength=resamples * n).reshape(resamples, n).astype(float)
    wf, wy = weights(len(firms)), weights(len(years))
    boot = np.einsum("bf,fyk,by->bk", wf, cell, wy, optimize=True)
    point = cols.sum(axis=0)

    def lr(v):
        return _div(_div(v[..., 0], v[..., 1]), _div(v[..., 2], v[..., 3]))
    b = lr(boot)
    lo, hi = _ci(b)
    return {"signal": signal, "windows": int(len(x)), "years": int(len(years)), "lr": float(lr(point)), "ci_low": lo,
            "ci_high": hi, "p_one_sided": _p_above(b)}


def two_way_table(w: pd.DataFrame, *, resamples: int = 2000, seed: int = 20261106) -> pd.DataFrame:
    """H2a, H2b (three-year 3x) and the eight H4 signals (five-year 10x, test period) with the
    firm-only interval next to the firm-and-year interval."""
    rows = []
    x2 = w[w["cohort_known"].fillna(False).astype(bool) & w["hit3_3y"].notna()].copy()
    cohort = _flag(x2["fallen"]).eq(1) & _flag(x2["fcf_pos"]).eq(1) & _flag(x2["net_cash_pos"]).eq(1)   # as in h2()
    x2["h2a"] = (cohort & _flag(x2["insider_any"]).eq(1)).astype(float)
    x2["h2b"] = (cohort & _flag(x2["insider_opportunistic"]).eq(1)).astype(float)
    x4 = split(w[(w["start_date"] >= pd.Timestamp("2013-07-01")) & w["hit10_5y"].notna()])[1]
    for label, g, sig, hit in ([("H2a", x2, "h2a", "hit3_3y"), ("H2b", x2, "h2b", "hit3_3y")]
                               + [(f"H4 {SIGNAL_NAMES.get(s, s)}", x4, s, "hit10_5y") for s in H4_SIGNALS if s in w.columns]):
        a = likelihood_ratio(g, sig, hit, resamples=resamples, seed=seed)
        b = likelihood_ratio_two_way(g, sig, hit, resamples=resamples, seed=seed)
        rows.append({"test": label, "windows": a["windows"], "years": b["years"], "lr": a["lr"], "firm_ci_low": a["ci_low"],
                     "firm_ci_high": a["ci_high"], "firm_p": a["p_one_sided"], "two_way_ci_low": b["ci_low"],
                     "two_way_ci_high": b["ci_high"], "two_way_p": b["p_one_sided"]})
    return pd.DataFrame(rows)


def universe_variant(features: pd.DataFrame, cfg_data: dict, *, min_price: float | None = None, cap_min: float | None = None) -> pd.DataFrame:
    """The features with the universe redrawn under another minimum price or market-cap floor
    (every other rule as locked), and the per-date attention tercile and relative drawdown
    recomputed inside the new universe. The cohort flags do not depend on the universe."""
    from .features import _tercile_low
    u = cfg_data["universe"]
    cap = u["market_cap_usd"]
    f = features.copy()
    not_spac = ~pd.to_numeric(f["sic"], errors="coerce").astype("float64").eq(6770)
    price = float(u.get("min_price_usd", 1.0)) if min_price is None else float(min_price)
    low = float(cap["min"]) if cap_min is None else float(cap_min)
    f["in_universe"] = (f["is_filer"].fillna(False).astype(bool) & not_spac
                        & (pd.to_numeric(f["entry_price_unadjusted"], errors="coerce") >= price)
                        & pd.to_numeric(f["market_cap"], errors="coerce").between(low, float(cap["max"]))
                        & f["liquid"].fillna(False).astype(bool) & f["listed"].fillna(False).astype(bool)).fillna(False).astype(bool)
    f["low_attention"] = pd.Series(pd.NA, index=f.index, dtype="boolean")
    for _, idx in f[f["in_universe"]].groupby("start_date").groups.items():
        g = f.loc[idx]
        f.loc[idx, "low_attention"] = _tercile_low(np.log1p(pd.to_numeric(g["breadth"], errors="coerce")), g["market_cap"])
    return f


def _period_end(period: pd.Series) -> pd.Series:
    """The last day of each split's quarter, whatever form the period was saved in."""
    if isinstance(period.dtype, pd.PeriodDtype):
        return period.dt.end_time.dt.normalize()
    if pd.api.types.is_datetime64_any_dtype(period):
        return pd.to_datetime(period)
    as_date = pd.to_datetime(period, errors="coerce")
    if as_date.notna().mean() > 0.5:
        return as_date
    try:
        return pd.Series(pd.PeriodIndex(period.astype(str), freq="Q").end_time.normalize(), index=period.index)
    except Exception:
        return as_date


def weak_split_windows(w: pd.DataFrame, splits: pd.DataFrame | None, horizon_years: int) -> pd.Series:
    """True for windows whose `horizon_years` cross a split that rests on weak evidence (the
    holders' median, one or two holders, the filings' share counts) or a neutralized jump."""
    if splits is None or not len(splits):
        return pd.Series(False, index=w.index)
    ev = splits["evidence"].fillna("").astype(str).str.lower() + " " + splits["kind"].fillna("").astype(str).str.lower()
    weak = splits[ev.str.contains("median|few|filings|implausible|neutral", regex=True)].copy()
    if not len(weak):
        return pd.Series(False, index=w.index)
    weak["cik"] = pd.to_numeric(weak["cik"], errors="coerce")
    weak["when"] = _period_end(weak["period"])
    weak = weak.dropna(subset=["when"])
    by = weak.dropna(subset=["cik"]).groupby("cik")["when"].apply(lambda s: np.sort(s.to_numpy(dtype="datetime64[ns]"))).to_dict()
    end = w["start_date"] + pd.DateOffset(years=horizon_years)
    out = np.zeros(len(w), dtype=bool)
    for i, (cik, a, b) in enumerate(zip(w["firm"].to_numpy(), w["start_date"].to_numpy(), end.to_numpy())):
        d = by.get(cik)
        if d is not None:
            k = np.searchsorted(d, a, side="right")
            out[i] = k < len(d) and d[k] <= b
    return pd.Series(out, index=w.index)


def headline(w: pd.DataFrame, name: str, *, resamples: int = 1000, seed: int = 20261106) -> dict:
    """One robustness row: H2a, H2b, the placebo and the eight test-period ratios on `w`."""
    r2, r4 = h2(w, resamples=resamples, seed=seed), h4(w, resamples=resamples, seed=seed)["table"]
    row = {"version": name, "windows": int(len(w)), "h2a_lr": r2["main"]["H2a"]["lr"], "h2a_ci_low": r2["main"]["H2a"]["ci_low"],
           "h2a_ci_high": r2["main"]["H2a"]["ci_high"], "h2a_windows": r2["main"]["H2a"]["with_signal"],
           "h2b_lr": r2["main"]["H2b"]["lr"], "placebo_lr": r2["placebo"]["lr"]}
    for s in r4.itertuples():
        row[f"lr_{s.signal}"] = s.test_lr
    return row
