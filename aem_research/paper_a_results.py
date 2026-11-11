"""Paper A results: the confirmatory tests and the exploratory tables, from the locked data run.

Everything here reads the tables notebook 02 saved (the deals, the public benchmark, the
WSIB fund table) and the cached documents; nothing is downloaded. The confirmatory part
follows config/paper_a.yaml's estimation section to the letter; the exploratory part is
labelled as such in every table (decision A3: all of it is reported, none of it is a finding
until confirmed on other data).
"""
from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .takeprivate import BUYER_TYPES, _read_ahead, _safe

RUN_LINE = re.compile(r"^v(\d+) · run (.+? UTC) · repo (\S+) · notebook (\S+) ")
STOCK_WORDS = re.compile(r"exchange ratio", re.I)
DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# The data run and the sample
# ---------------------------------------------------------------------------

def run_identity(summary_text: str | None) -> dict | None:
    """Release, start time, commit and notebook build of the notebook-02 run, from its saved block."""
    for line in (summary_text or "").splitlines():
        m = RUN_LINE.match(line)
        if m:
            return {"release": int(m.group(1)), "generated_utc": m.group(2), "repo_commit": m.group(3),
                    "notebook_build": m.group(4)}
    return None


def stock_mentions(deals: pd.DataFrame, docs_dir, *, batch: int = 80) -> pd.Series:
    """How often each deal's document speaks of an exchange ratio (cached texts; NaN where the
    text is not on Drive). A cash take-private has none; a stock or mixed merger has many."""
    docs_dir = Path(docs_dir)
    paths = {}
    for d in deals.itertuples():
        name = getattr(d, "main_file", None)
        name = name if isinstance(name, str) and name else getattr(d, "doc_file", None)
        if isinstance(name, str) and name:
            paths[d.deal_id] = docs_dir / f"{d.doc_adsh}_{_safe(name)}.txt.gz"
    out, items = {}, list(paths.items())
    from .takeprivate import _HELD
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        _read_ahead([p for _, p in chunk])
        for deal_id, p in chunk:
            raw = _HELD.pop(p, None)
            out[deal_id] = np.nan if raw is None else len(STOCK_WORDS.findall(gzip.decompress(raw).decode("utf-8", "ignore")))
    return deals["deal_id"].map(out).astype("float64")


def analysis_sample(deals: pd.DataFrame, bench: pd.DataFrame | None, cfg: dict, *, as_of) -> tuple[pd.DataFrame, dict]:
    """The deals of the analysis with every derived column, and the funnel that led to them.

    Rules (config: sample.analysis): completed; not a SPAC merger; not a stock or mixed
    merger (exchange-ratio mentions below the limit; unknown counts stay in). Columns added:
    group (core / no sponsor identified), entry_date, benchmark multiples at the latest
    quarterly start date before entry, log premiums, the multiples' validity flags, exit
    time and event, cohort flags.
    """
    rules = cfg["sample"]["analysis"]
    est = cfg["estimation"]
    d = deals.copy()
    for c in ("stock_mentions", "ev_ebitda", "ev_revenue", "sic", "reipo_date", "sale_date", "completion_date",
              "check_price", "check_value", "equity_value", "enterprise_value"):
        if c not in d.columns:
            d[c] = np.nan
    funnel = {"deals": int(len(d))}
    d = d[d["completed"].fillna(False).astype(bool)]
    funnel["completed"] = int(len(d))
    d = d[d["buyer_type"].astype(str).ne("SPAC merger")]
    funnel["not_spac"] = int(len(d))
    stock = pd.to_numeric(d["stock_mentions"], errors="coerce") >= rules["stock_merger_exchange_ratio_mentions"]
    funnel["stock_or_mixed_excluded"] = int(stock.sum())
    d = d[~stock].copy()
    funnel["analysis_sample"] = int(len(d))
    d["group"] = np.where(d["buyer_type"].isin(BUYER_TYPES[:2]), "core", "no sponsor identified")
    funnel["core"] = int(d["group"].eq("core").sum())
    funnel["no_sponsor_identified"] = int(len(d) - funnel["core"])

    d["entry_date"] = pd.to_datetime(d["first_filed"]).astype("datetime64[ns]")
    d["post_2022"] = d["entry_date"] >= pd.Timestamp(est["regime_split"])
    sic = pd.to_numeric(d["sic"], errors="coerce")
    lo, hi = rules["multiples_exclude_sic"]
    d["financial"] = sic.between(lo, hi)
    if bench is not None and len(bench):
        b = bench[["start_date", "median_ev_ebitda", "median_ev_revenue"]].copy()
        b["start_date"] = pd.to_datetime(b["start_date"]).astype("datetime64[ns]")
        b = b.sort_values("start_date").rename(columns={"start_date": "bench_date", "median_ev_ebitda": "bench_ev_ebitda",
                                                        "median_ev_revenue": "bench_ev_revenue"})
        d = pd.merge_asof(d.sort_values("entry_date"), b, left_on="entry_date", right_on="bench_date",
                          direction="backward", tolerance=pd.Timedelta(days=rules["benchmark_max_days_before"]))
    else:
        d["bench_date"], d["bench_ev_ebitda"], d["bench_ev_revenue"] = pd.NaT, np.nan, np.nan
    e, r = pd.to_numeric(d["ev_ebitda"], errors="coerce"), pd.to_numeric(d["ev_revenue"], errors="coerce")
    ok_e = (e > 0) & (e <= rules["ev_ebitda_max"]) & ~d["financial"] & (d["bench_ev_ebitda"] > 0)
    ok_r = (r > 0) & (r <= rules["ev_revenue_max"]) & ~d["financial"] & (d["bench_ev_revenue"] > 0)
    d["log_premium"] = np.log((e / d["bench_ev_ebitda"]).where(ok_e))
    d["log_premium_revenue"] = np.log((r / d["bench_ev_revenue"]).where(ok_r))
    funnel["with_ev_ebitda_premium"] = int(d["log_premium"].notna().sum())
    funnel["core_with_ev_ebitda_premium"] = int((d["log_premium"].notna() & d["group"].eq("core")).sum())
    funnel["with_ev_revenue_premium"] = int(d["log_premium_revenue"].notna().sum())

    as_of = pd.Timestamp(as_of)
    done = pd.to_datetime(d["completion_date"]).astype("datetime64[ns]")
    exit_date = pd.concat([pd.to_datetime(d["reipo_date"]), pd.to_datetime(d["sale_date"])], axis=1).min(axis=1)
    exit_date = exit_date.where(exit_date > done)
    d["exit_date"] = exit_date
    d["exit_kind"] = np.where(exit_date.isna(), "", np.where(pd.to_datetime(d["reipo_date"]).eq(exit_date), "reIPO",
                                                             "sold to a listed buyer"))
    d["exit_event"] = exit_date.notna() & (exit_date <= as_of)
    end = exit_date.where(d["exit_event"], as_of)
    d["exit_years"] = ((end - done).dt.days / DAYS_PER_YEAR).clip(lower=0)
    d["completion_year"] = done.dt.year
    early, late = est["h3"]["early_cohort"], est["h3"]["late_cohort"]
    d["cohort"] = np.select([d["completion_year"].between(*early), d["completion_year"].between(*late)],
                            [f"{early[0]}–{early[1]}", f"{late[0]}–{late[1]}"], default="")
    funnel["visible_exits"] = int(d["exit_event"].sum())
    return d.reset_index(drop=True), funnel


# ---------------------------------------------------------------------------
# Statistics (small, explicit, tested against scipy where scipy has the test)
# ---------------------------------------------------------------------------

def bootstrap_ci(values, stat=np.median, *, draws: int = 10_000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap interval of `stat` (NaN, NaN below three values)."""
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(draws, len(v)))
    s = stat(v[idx], axis=1)
    a = (1 - level) / 2
    return (float(np.quantile(s, a)), float(np.quantile(s, 1 - a)))


def bootstrap_diff_ci(a, b, stat=np.median, *, draws: int = 10_000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap interval of stat(a) − stat(b), the groups resampled separately."""
    a = np.asarray(pd.to_numeric(pd.Series(a), errors="coerce").dropna(), dtype=float)
    b = np.asarray(pd.to_numeric(pd.Series(b), errors="coerce").dropna(), dtype=float)
    if len(a) < 3 or len(b) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    sa = stat(a[rng.integers(0, len(a), size=(draws, len(a)))], axis=1)
    sb = stat(b[rng.integers(0, len(b), size=(draws, len(b)))], axis=1)
    q = (1 - level) / 2
    return (float(np.quantile(sa - sb, q)), float(np.quantile(sa - sb, 1 - q)))


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values (missing ones stay missing and do not count as tests)."""
    known = sorted(((k, p) for k, p in pvalues.items() if p is not None and np.isfinite(p)), key=lambda kp: kp[1])
    out, running, m = {k: np.nan for k in pvalues}, 0.0, len(known)
    for i, (k, p) in enumerate(known):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def kaplan_meier(time, event, at: float) -> float:
    """Probability of the event by time `at` (1 − Kaplan–Meier survival); NaN without data."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    keep = np.isfinite(t)
    t, e = t[keep], e[keep]
    if not len(t):
        return np.nan
    surv = 1.0
    for u in np.unique(t[e & (t <= at)]):
        at_risk = (t >= u).sum()
        surv *= 1.0 - ((t == u) & e).sum() / at_risk
    return float(1.0 - surv)


def logrank(time, event, group) -> tuple[float, float]:
    """Two-group log-rank test: (chi-square with one degree of freedom, p-value)."""
    t, e, g = np.asarray(time, dtype=float), np.asarray(event, dtype=bool), np.asarray(group)
    keep = np.isfinite(t)
    t, e, g = t[keep], e[keep], g[keep]
    labels = pd.unique(g)
    if len(labels) != 2:
        return (np.nan, np.nan)
    first = g == labels[0]
    o_minus_e, var = 0.0, 0.0
    for u in np.unique(t[e]):
        risk = t >= u
        n, n1 = risk.sum(), (risk & first).sum()
        dth = ((t == u) & e).sum()
        d1 = ((t == u) & e & first).sum()
        if n < 2:
            continue
        o_minus_e += d1 - dth * n1 / n
        var += dth * (n1 / n) * (1 - n1 / n) * (n - dth) / (n - 1)
    if var <= 0:
        return (np.nan, np.nan)
    chi = o_minus_e ** 2 / var
    return (float(chi), float(math.erfc(math.sqrt(chi / 2))))


def truncate(time, event, horizon: float):
    """Follow-up cut at `horizon` years: later events count as not yet seen."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    return np.minimum(t, horizon), e & (t <= horizon)


def ols(y, X: pd.DataFrame) -> pd.DataFrame:
    """Least squares with an intercept and heteroskedasticity-robust (HC1) standard errors."""
    cols = ["term", "coef", "se", "t", "p", "n"]
    if X.shape[1] == 0 or len(X) == 0:
        return pd.DataFrame(columns=cols)
    data = pd.concat([pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"),
                      X.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")], axis=1).dropna()
    k = X.shape[1] + 1
    if len(data) <= k + 2:
        return pd.DataFrame(columns=cols)
    yv = data.iloc[:, 0].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(data)), data.iloc[:, 1:].to_numpy(dtype=float)])
    xtx_inv = np.linalg.pinv(A.T @ A)
    beta = xtx_inv @ A.T @ yv
    resid = yv - A @ beta
    meat = (A * resid[:, None] ** 2).T @ A
    cov = xtx_inv @ meat @ xtx_inv * len(data) / (len(data) - k)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = [float(math.erfc(abs(v) / math.sqrt(2))) if np.isfinite(v) else np.nan for v in t]
    return pd.DataFrame({"term": ["intercept", *X.columns], "coef": beta, "se": se, "t": t, "p": p, "n": len(data)})


def _wilcoxon(v) -> float:
    from scipy import stats
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v != 0)]
    return float(stats.wilcoxon(v, alternative="two-sided").pvalue) if len(v) >= 6 else np.nan


def _mannwhitney(a, b) -> float:
    from scipy import stats
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) >= 3 and len(b) >= 3 else np.nan


# ---------------------------------------------------------------------------
# Confirmatory tests (config: estimation)
# ---------------------------------------------------------------------------

def premium_tests(d: pd.DataFrame, cfg: dict, *, column: str = "log_premium", group: str | None = "core",
                  where: pd.Series | None = None) -> dict:
    """A-H1 and A-H2 on one multiple and one group: the median log premium over the public
    benchmark with its interval and signed-rank test; the post-2022 difference with its
    interval and rank-sum test."""
    est = cfg["estimation"]
    x = d if group is None else d[d["group"].eq(group)]
    if where is not None:
        x = x[where.reindex(x.index).fillna(False)]
    v = x[column].dropna()
    pre, post = x.loc[~x["post_2022"], column].dropna(), x.loc[x["post_2022"], column].dropna()
    boot = {"draws": est["bootstrap_draws"], "seed": est["seed"]}
    ci, dci = bootstrap_ci(v, **boot), bootstrap_diff_ci(post, pre, **boot)
    return {"n": int(len(v)), "median_log_premium": float(v.median()) if len(v) else np.nan,
            "median_premium_pct": float(np.expm1(v.median())) if len(v) else np.nan,
            "ci_low_pct": float(np.expm1(ci[0])), "ci_high_pct": float(np.expm1(ci[1])),
            "share_above_benchmark": float((v > 0).mean()) if len(v) else np.nan, "h1_p": _wilcoxon(v),
            "n_before_2022": int(len(pre)), "n_2022_on": int(len(post)),
            "median_before_pct": float(np.expm1(pre.median())) if len(pre) else np.nan,
            "median_2022_on_pct": float(np.expm1(post.median())) if len(post) else np.nan,
            "diff_log": float(post.median() - pre.median()) if len(pre) and len(post) else np.nan,
            "diff_ci_low": dci[0], "diff_ci_high": dci[1], "h2_p": _mannwhitney(post, pre)}


def exit_tests(d: pd.DataFrame, cfg: dict, *, group: str | None = "core") -> dict:
    """A-H3: the probability of a visible exit within the horizon, early against late
    completion cohort (Kaplan–Meier, log-rank on follow-up cut at the horizon)."""
    h3 = cfg["estimation"]["h3"]
    horizon = float(h3["horizon_years"])
    x = d if group is None else d[d["group"].eq(group)]
    x = x[x["cohort"].ne("") & x["exit_years"].notna()]
    out = {"horizon_years": horizon}
    names = [f"{h3['early_cohort'][0]}–{h3['early_cohort'][1]}", f"{h3['late_cohort'][0]}–{h3['late_cohort'][1]}"]
    for key, name in zip(("early", "late"), names):
        g = x[x["cohort"].eq(name)]
        t, e = truncate(g["exit_years"], g["exit_event"], horizon)
        out[f"{key}_cohort"], out[f"{key}_n"], out[f"{key}_exits"] = name, int(len(g)), int(e.sum())
        out[f"{key}_rate"] = kaplan_meier(t, e, horizon)
    t, e = truncate(x["exit_years"], x["exit_event"], horizon)
    out["chi2"], out["h3_p"] = logrank(t, e, x["cohort"].to_numpy())
    out["diff_rate"] = out["late_rate"] - out["early_rate"]
    return out


def confirmatory(d: pd.DataFrame, cfg: dict) -> dict:
    """The three pre-registered tests on the core sample, Holm-adjusted, with their robustness versions."""
    prem, ex = premium_tests(d, cfg), exit_tests(d, cfg)
    adj = holm({"A-H1": prem["h1_p"], "A-H2": prem["h2_p"], "A-H3": ex["h3_p"]})
    alpha = cfg["estimation"]["alpha"]
    verdict = {k: ("not testable" if not np.isfinite(p) else "supported" if p < alpha else "not supported")
               for k, p in adj.items()}
    # direction must match the hypothesis for "supported"
    if verdict["A-H1"] == "supported" and not prem["median_log_premium"] > 0:
        verdict["A-H1"] = "significant, opposite direction"
    if verdict["A-H2"] == "supported" and not prem["diff_log"] < 0:
        verdict["A-H2"] = "significant, opposite direction"
    if verdict["A-H3"] == "supported" and not ex["diff_rate"] < 0:
        verdict["A-H3"] = "significant, opposite direction"
    confirmed = d["check_price"].astype(str).ne("disagree") & d["check_value"].astype(str).ne("disagree")
    robust = {"EV/revenue, core": premium_tests(d, cfg, column="log_premium_revenue"),
              "EV/EBITDA, all buyers": premium_tests(d, cfg, group=None),
              "EV/EBITDA, core, no two-source disagreement": premium_tests(d, cfg, where=confirmed),
              "exits, all buyers": exit_tests(d, cfg, group=None)}
    return {"premium": prem, "exits": ex, "holm": adj, "verdict": verdict, "robustness": robust}


# ---------------------------------------------------------------------------
# Exploratory tables (decision A3: all reported, none a finding yet)
# ---------------------------------------------------------------------------

def _by(d: pd.DataFrame, key: str, column: str = "log_premium", min_n: int = 5) -> pd.DataFrame:
    g = d.dropna(subset=[column]).groupby(key, observed=True)[column]
    t = g.agg(n="size", median_log="median").reset_index()
    t["median_premium_pct"] = np.expm1(t["median_log"])
    t["share_above_benchmark"] = g.apply(lambda v: float((v > 0).mean())).to_numpy()
    return t[t["n"] >= min_n].drop(columns="median_log").rename(columns={key: "group"}).assign(by=key)[
        ["by", "group", "n", "median_premium_pct", "share_above_benchmark"]]


def sic_division(sic) -> pd.Series:
    s = pd.to_numeric(pd.Series(sic), errors="coerce")
    bins = [0, 1000, 1500, 1800, 4000, 5000, 5200, 6000, 6800, 7000, 7370, 7380, 9000, 10000]
    names = ["agriculture", "mining and energy", "construction", "manufacturing", "transport and utilities", "wholesale",
             "retail", "finance and real estate", "hotels and personal services", "business services",
             "software and data", "other services", "public and other"]
    return pd.cut(s, bins=bins, labels=names, right=False).astype("object")


def premium_tables(d: pd.DataFrame) -> pd.DataFrame:
    """The entry premium cut by buyer group, entry year, size tercile, sector and deal status."""
    x = d.copy()
    x["entry_year"] = x["entry_date"].dt.year
    size = pd.to_numeric(x["enterprise_value"], errors="coerce")
    x["size_tercile"] = pd.qcut(size.rank(method="first"), 3, labels=["small", "medium", "large"]).astype("object") \
        if size.notna().sum() >= 6 else None
    x["sector"] = sic_division(x["sic"]).to_numpy()
    x["regime"] = np.where(x["post_2022"], "2022 or later", "before 2022")
    x["exit_seen"] = np.where(x["exit_event"], "visible exit", "no visible exit")
    parts = [_by(x, k) for k in ("group", "regime", "entry_year", "size_tercile", "sector", "buyer_type", "exit_seen")
             if x[k].notna().any()]
    parts.append(_by(x, "group", "log_premium_revenue").assign(by="group (EV/revenue)"))
    parts.append(_by(x, "regime", "log_premium_revenue").assign(by="regime (EV/revenue)"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def premium_regression(d: pd.DataFrame) -> pd.DataFrame:
    """Log premium on the entry regime (Paper B's D7 moderators), size and the buyer group."""
    x = d.dropna(subset=["log_premium"])
    X = pd.DataFrame({"real_yield_10y": x.get("entry_real_yield_10y"), "tightening_2y_12m": x.get("entry_tightening_2y_12m"),
                      "credit_spread_baa": x.get("entry_credit_spread_baa"),
                      "inflation_cpi_yoy": x.get("entry_inflation_cpi_yoy"),
                      "log_enterprise_value": np.log(pd.to_numeric(x["enterprise_value"], errors="coerce").where(lambda v: v > 0)),
                      "core": x["group"].eq("core").astype(float)})
    X = X.loc[:, X.notna().any()]
    return ols(x["log_premium"], X)


def exit_tables(d: pd.DataFrame, horizons=(3, 5, 7)) -> pd.DataFrame:
    """Probability of a visible exit by 3, 5 and 7 years (Kaplan–Meier), by cohort, buyer group,
    entry-premium tercile and entry regime; with the re-IPO share among the exits seen."""
    x = d[d["exit_years"].notna()].copy()
    prem = x["log_premium"]
    x["premium_tercile"] = pd.qcut(prem.rank(method="first"), 3, labels=["low premium", "middle", "high premium"]).astype(
        "object") if prem.notna().sum() >= 9 else None
    x["completion_period"] = pd.cut(x["completion_year"], [2011, 2014, 2017, 2020, 2023, 2100],
                                    labels=["2012–2014", "2015–2017", "2018–2020", "2021–2023", "2024 on"]).astype("object")
    rows = []
    for key in ("group", "cohort", "completion_period", "premium_tercile", "buyer_type"):
        if not x[key].notna().any():
            continue
        for name, g in x[x[key].notna() & x[key].astype(str).ne("")].groupby(key, observed=True):
            ev = g[g["exit_event"]]
            rows.append({"by": key, "group": name, "n": int(len(g)), "exits_seen": int(len(ev)),
                         **{f"exit_by_{h}y": kaplan_meier(g["exit_years"], g["exit_event"], h) for h in horizons},
                         "reipo_share_of_exits": float(ev["exit_kind"].eq("reIPO").mean()) if len(ev) else np.nan,
                         "median_years_to_exit_seen": float(ev["exit_years"].median()) if len(ev) else np.nan})
    return pd.DataFrame(rows)


def holding_by_exit_year(d: pd.DataFrame) -> pd.DataFrame:
    """The drafted A-H3, as exploration: time to the exits that are visible, by exit year.
    Later exit years can show longer holds only because old deals have had more time – the
    table says how long each year's exits were held, not how long deals are held."""
    ev = d[d["exit_event"]].copy()
    if not len(ev):
        return pd.DataFrame(columns=["exit_year", "exits", "median_years_held", "reipo", "sold_to_listed_buyer"])
    ev["exit_year"] = ev["exit_date"].dt.year
    return ev.groupby("exit_year").agg(exits=("deal_id", "size"), median_years_held=("exit_years", "median"),
                                       reipo=("exit_kind", lambda k: int(k.eq("reIPO").sum())),
                                       sold_to_listed_buyer=("exit_kind", lambda k: int(k.ne("reIPO").sum()))).reset_index()


def unvalidated_terms(d: pd.DataFrame, accuracy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leverage and the equity share from the commitments, and the sponsor league table. The
    terms missed the 90% gate; `accuracy` (share correct in the last reading) rides along."""
    x = d[d["group"].eq("core")].copy()
    x["regime"] = np.where(x["post_2022"], "2022 or later", "before 2022")
    x["debt_to_ebitda"] = pd.to_numeric(x.get("leverage_x"), errors="coerce").where(lambda v: (v > 0) & (v < 20))
    x["equity_share_of_ev"] = pd.to_numeric(x.get("equity_share"), errors="coerce").where(lambda v: (v > 0) & (v <= 1.5))
    both = pd.to_numeric(x.get("equity_commitment_usd"), errors="coerce") + pd.to_numeric(x.get("debt_commitment_usd"),
                                                                                         errors="coerce")
    x["equity_share_of_commitments"] = pd.to_numeric(x.get("equity_commitment_usd"), errors="coerce") / both.where(both > 0)
    rows = []
    for name, g in [("all core deals", x), *x.groupby("regime")]:
        for col, acc in (("debt_to_ebitda", "debt_commitment"), ("equity_share_of_ev", "equity_commitment"),
                         ("equity_share_of_commitments", "both commitments")):
            v = g[col].dropna()
            rows.append({"group": name, "measure": col, "n": int(len(v)), "median": float(v.median()) if len(v) else np.nan,
                         "p25": float(v.quantile(0.25)) if len(v) else np.nan,
                         "p75": float(v.quantile(0.75)) if len(v) else np.nan,
                         "term_accuracy_in_reading": accuracy.get(acc, min(accuracy.get("equity_commitment", np.nan),
                                                                           accuracy.get("debt_commitment", np.nan)))})
    named = d[d["sponsor"].fillna("").astype(str).ne("")]
    league = (named.groupby("sponsor").agg(deals=("deal_id", "size"), median_premium_pct=("log_premium", lambda v: float(
        np.expm1(v.median())) if v.notna().any() else np.nan), visible_exits=("exit_event", "sum")).reset_index()
        .sort_values(["deals", "sponsor"], ascending=[False, True]).head(25))
    league["sponsor_name_accuracy_in_reading"] = accuracy.get("sponsor_name", np.nan)
    return pd.DataFrame(rows), league


def measurement(d: pd.DataFrame) -> pd.DataFrame:
    """How the computed values compare with what the announcements state (second source)."""
    stated = pd.to_numeric(d.get("second_deal_value_usd"), errors="coerce")
    rows = []
    for name, col in (("stated value / computed equity value", "equity_value"),
                      ("stated value / computed enterprise value", "enterprise_value")):
        r = (stated / pd.to_numeric(d[col], errors="coerce").where(lambda v: v > 0)).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"measure": name, "n": int(len(r)), "median": float(r.median()) if len(r) else np.nan,
                     "p10": float(r.quantile(0.1)) if len(r) else np.nan, "p90": float(r.quantile(0.9)) if len(r) else np.nan,
                     "within_15pct": float(((r > 0.85) & (r < 1.15)).mean()) if len(r) else np.nan})
    return pd.DataFrame(rows)


def wsib_vintages(funds: pd.DataFrame | None, strategies=("Corporate Finance/Buyout",)) -> pd.DataFrame:
    """Net results of WSIB's buyout funds by vintage year, from the latest report: what the
    limited partner got, next to what the deal sample says sponsors paid."""
    cols = ["vintage_year", "funds", "paid_in", "pooled_tvpi", "pooled_dpi", "median_net_irr", "report_date"]
    if funds is None or not len(funds):
        return pd.DataFrame(columns=cols)
    f = funds[funds["strategy"].fillna("").astype(str).str.startswith(tuple(strategies))]
    f = f[f["report_date"].eq(f["report_date"].max()) & (pd.to_numeric(f["paid_in"], errors="coerce") > 0)]
    if not len(f):
        return pd.DataFrame(columns=cols)
    g = f.groupby("vintage_year")
    out = g.agg(funds=("name", "size"), paid_in=("paid_in", "sum"), total=("total_value", "sum"),
                distributed=("distributed", "sum"), median_net_irr=("net_irr", "median")).reset_index()
    out["pooled_tvpi"], out["pooled_dpi"] = out["total"] / out["paid_in"], out["distributed"] / out["paid_in"]
    out["report_date"] = f["report_date"].max()
    return out[cols]


def wsib_distribution_pace(funds: pd.DataFrame | None, strategies=("Corporate Finance/Buyout",)) -> pd.DataFrame:
    """Distributions over paid-in capital for all buyout funds together, report by report:
    whether money came back more slowly after 2021 (the drafted A-H3 from the investor's side)."""
    cols = ["report_date", "funds", "paid_in", "distributed", "market_value", "dpi", "distributed_since_last_report"]
    if funds is None or not len(funds):
        return pd.DataFrame(columns=cols)
    f = funds[funds["strategy"].fillna("").astype(str).str.startswith(tuple(strategies))]
    g = f.groupby("report_date").agg(funds=("name", "size"), paid_in=("paid_in", "sum"), distributed=("distributed", "sum"),
                                     market_value=("market_value", "sum")).reset_index().sort_values("report_date")
    g["dpi"] = g["distributed"] / g["paid_in"].where(g["paid_in"] > 0)
    g["distributed_since_last_report"] = g["distributed"].diff()
    return g[cols]


def premium_vs_fund_returns(d: pd.DataFrame, vintages: pd.DataFrame) -> dict:
    """Entry-year median premium against the same vintage's pooled fund multiple (rank correlation)."""
    x = d.dropna(subset=["log_premium"]).assign(year=lambda t: t["entry_date"].dt.year)
    p = x.groupby("year")["log_premium"].agg(["median", "size"]).reset_index()
    p = p[p["size"] >= 5]
    j = p.merge(vintages, left_on="year", right_on="vintage_year")
    j = j[j["funds"] >= 2]
    if len(j) < 4:
        return {"years": int(len(j)), "spearman": np.nan}
    return {"years": int(len(j)), "spearman": float(j["median"].rank().corr(j["pooled_tvpi"].rank()))}


def exploratory(d: pd.DataFrame, funds: pd.DataFrame | None, cfg: dict) -> dict:
    acc = cfg["validation"]["term_gates"]["measured_accuracy"]
    lev, league = unvalidated_terms(d, acc)
    vint = wsib_vintages(funds, tuple(cfg["wsib"]["strategies_of_interest"]))
    return {"premium by group": premium_tables(d), "premium regression": premium_regression(d),
            "exit probability": exit_tables(d), "holding by exit year": holding_by_exit_year(d),
            "leverage (unvalidated)": lev, "sponsors (unvalidated)": league, "measurement": measurement(d),
            "WSIB by vintage": vint,
            "WSIB distribution pace": wsib_distribution_pace(funds, tuple(cfg["wsib"]["strategies_of_interest"])),
            "premium vs fund returns": pd.DataFrame([premium_vs_fund_returns(d, vint)])}
