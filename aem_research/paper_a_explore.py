"""Paper A and B together: exploratory tables from what the two data runs left on Drive.

Nothing here is pre-registered and nothing is downloaded. Each function answers one question
the confirmatory results raised (decision A5: go as far as the data allows; every table is
reported, none is a finding until confirmed on other data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paper_a_results import (DAYS_PER_YEAR, _mannwhitney, _wilcoxon, bootstrap_ci, kaplan_meier, logrank, sic_division,
                              truncate)


def _med(v) -> float:
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    return float(v.median()) if len(v) else np.nan


# 1 ── why the premium looks larger after 2021: the deals' multiple or the market's? ──────────

def premium_decomposition(d: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """By entry year: the deals' median EV/EBITDA, the benchmark's median on their dates and the
    premium between them. A premium can rise because deals got dearer or because the public
    market got cheaper; the two columns say which."""
    x = d.dropna(subset=["log_premium"]).assign(entry_year=lambda t: t["entry_date"].dt.year)
    g = x.groupby("entry_year")
    out = g.agg(deals=("deal_id", "size"), deal_ev_ebitda=("ev_ebitda", "median"),
                benchmark_ev_ebitda=("bench_ev_ebitda", "median"), log_premium=("log_premium", "median")).reset_index()
    out["median_premium_pct"] = np.expm1(out.pop("log_premium"))
    return out[out["deals"] >= min_n]


# 2 ── what the EBITDA-but-not-revenue premium means: do sponsors buy thin margins? ───────────

def margin_test(d: pd.DataFrame, *, seed: int = 0, draws: int = 10_000) -> pd.DataFrame:
    """Target EBITDA margin against the margin the benchmark's two medians imply
    (median EV/revenue ÷ median EV/EBITDA – a ratio of medians, so an approximation of the
    typical public margin). One row per group; the test is a signed-rank on the log ratio."""
    x = d.copy()
    margin = pd.to_numeric(x["ebitda"], errors="coerce") / pd.to_numeric(x["revenue"], errors="coerce").where(lambda v: v > 0)
    implied = x["bench_ev_revenue"] / x["bench_ev_ebitda"].where(x["bench_ev_ebitda"] > 0)
    ok = (margin > 0) & (margin < 1) & (implied > 0) & ~x["financial"].astype(bool)
    x = x.assign(margin=margin.where(ok), implied=implied.where(ok), log_ratio=np.log((margin / implied).where(ok)))
    rows = []
    for name, g in [("core", x[x["group"].eq("core")]), ("no sponsor identified", x[x["group"].ne("core")]), ("all", x)]:
        v = g["log_ratio"].dropna()
        ci = bootstrap_ci(v, seed=seed, draws=draws)
        rows.append({"group": name, "n": int(len(v)), "target_margin": _med(g["margin"]), "benchmark_implied_margin": _med(g["implied"]),
                     "median_ratio": float(np.exp(v.median())) if len(v) else np.nan, "ci_low": float(np.exp(ci[0])),
                     "ci_high": float(np.exp(ci[1])), "share_below_benchmark": float((v < 0).mean()) if len(v) else np.nan,
                     "p": _wilcoxon(v)})
    return pd.DataFrame(rows)


# 3 ── coverage: the announcement's stated value where the filing tags no debt ─────────────────

def wider_coverage(d: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The premium again with the enterprise value taken from the announcement where the
    computed one is unknown (only values the announcement calls an enterprise or transaction
    value). Returns (the sample with log_premium_wide, a table: computed only against wide)."""
    rules = cfg["sample"]["analysis"]
    x = d.copy()
    stated = pd.to_numeric(x["second_deal_value_usd"], errors="coerce").where(
        x["second_deal_value_kind"].astype(str).isin(["enterprise", "transaction"]))
    ev = pd.to_numeric(x["enterprise_value"], errors="coerce")
    x["ev_wide"] = ev.where(ev > 0, stated)
    x["ev_source"] = np.where(ev > 0, "computed", np.where(stated > 0, "stated in the announcement", ""))
    mult = x["ev_wide"] / pd.to_numeric(x["ebitda"], errors="coerce").where(lambda v: v > 0)
    ok = (mult > 0) & (mult <= rules["ev_ebitda_max"]) & ~x["financial"].astype(bool) & (x["bench_ev_ebitda"] > 0)
    x["log_premium_wide"] = np.log((mult / x["bench_ev_ebitda"]).where(ok))
    rows = []
    for name, col in (("computed enterprise value only", "log_premium"), ("with stated values", "log_premium_wide")):
        for grp, g in (("core", x[x["group"].eq("core")]), ("all buyers", x)):
            v = g[col].dropna()
            pre, post = g.loc[~g["post_2022"], col].dropna(), g.loc[g["post_2022"], col].dropna()
            ci = bootstrap_ci(v, seed=cfg["estimation"]["seed"], draws=cfg["estimation"]["bootstrap_draws"])
            rows.append({"enterprise_value": name, "group": grp, "n": int(len(v)),
                         "median_premium_pct": float(np.expm1(v.median())) if len(v) else np.nan,
                         "ci_low_pct": float(np.expm1(ci[0])), "ci_high_pct": float(np.expm1(ci[1])), "p": _wilcoxon(v),
                         "n_before_2022": int(len(pre)), "before_2022_pct": float(np.expm1(pre.median())) if len(pre) else np.nan,
                         "n_2022_on": int(len(post)), "from_2022_pct": float(np.expm1(post.median())) if len(post) else np.nan,
                         "p_regime": _mannwhitney(post, pre)})
    return x, pd.DataFrame(rows)


# 4 ── each target against itself: the offer against its own price history ────────────────────

def own_history(d: pd.DataFrame, panel: pd.DataFrame | None, *, gap_days: int = 45, quarters: int = 12) -> pd.DataFrame:
    """Per deal, from Paper B's quarter-end price panel: the last price at least `gap_days`
    before the first merger filing (before the announcement moved it), the mean and the high
    of the `quarters` quarter-ends up to it, and the offer against each."""
    cols = ["deal_id", "pre_price", "pre_date", "quarters_seen", "premium_to_last", "offer_to_own_mean", "offer_to_own_high"]
    if panel is None or not len(panel) or not len(d):
        return pd.DataFrame(columns=cols)
    p = panel[["cik", "date", "price"]].dropna().copy()
    p["cik"] = pd.to_numeric(p["cik"], errors="coerce")
    p = p.dropna(subset=["cik"]).astype({"cik": "int64"})
    p["date"] = pd.to_datetime(p["date"]).astype("datetime64[ns]")
    p = p[p["price"] > 0].sort_values(["cik", "date"]).drop_duplicates(["cik", "date"], keep="last")
    by = {c: g for c, g in p.groupby("cik")}
    rows = []
    for r in d.itertuples():
        cik, offer = pd.to_numeric(getattr(r, "target_cik", np.nan), errors="coerce"), pd.to_numeric(r.offer_price, errors="coerce")
        g = by.get(int(cik)) if pd.notna(cik) else None
        if g is None or not (offer > 0):
            continue
        h = g[g["date"] <= r.entry_date - pd.Timedelta(days=gap_days)].tail(quarters)
        h = h[h["date"] >= r.entry_date - pd.Timedelta(days=gap_days + 100 * quarters)]
        if not len(h) or (r.entry_date - h["date"].iloc[-1]).days > gap_days + 135:
            continue                                        # no price shortly before the deal
        last = float(h["price"].iloc[-1])
        rows.append({"deal_id": r.deal_id, "pre_price": last, "pre_date": h["date"].iloc[-1], "quarters_seen": int(len(h)),
                     "premium_to_last": offer / last - 1, "offer_to_own_mean": offer / float(h["price"].mean()),
                     "offer_to_own_high": offer / float(h["price"].max())})
    return pd.DataFrame(rows, columns=cols)


def own_history_table(d: pd.DataFrame, own: pd.DataFrame, min_quarters: int = 8) -> pd.DataFrame:
    """The takeover premium and the offer against the target's own three-year mean and high,
    by buyer group and regime. A premium outside −50% to +300% is a broken price pair and left out."""
    cols = ["group", "n", "median_premium_to_last", "p25", "p75", "n_with_history", "median_offer_to_own_mean",
            "median_offer_to_own_high", "share_offer_below_own_high"]
    if not len(own):
        return pd.DataFrame(columns=cols)
    x = d.merge(own, on="deal_id")
    x = x[(x["premium_to_last"] > -0.5) & (x["premium_to_last"] < 3.0)]
    x["regime"] = np.where(x["post_2022"], "entered 2022 or later", "entered before 2022")
    rows = []
    for name, g in [("all", x), *x.groupby("group"), *x.groupby("regime")]:
        hist = g[g["quarters_seen"] >= min_quarters]
        v = g["premium_to_last"]
        rows.append({"group": name, "n": int(len(g)), "median_premium_to_last": _med(v),
                     "p25": float(v.quantile(0.25)) if len(v) else np.nan, "p75": float(v.quantile(0.75)) if len(v) else np.nan,
                     "n_with_history": int(len(hist)), "median_offer_to_own_mean": _med(hist["offer_to_own_mean"]),
                     "median_offer_to_own_high": _med(hist["offer_to_own_high"]),
                     "share_offer_below_own_high": float((hist["offer_to_own_high"] < 1).mean()) if len(hist) else np.nan})
    return pd.DataFrame(rows, columns=cols)


# 5 ── what sponsors select on: the AEM signals of targets before the deal ────────────────────

SIGNALS = ("in_universe", "fcf_pos", "net_cash_pos", "low_dilution", "insider_cluster", "low_attention", "fallen", "h2a", "h2b")


def target_signals(d: pd.DataFrame, features: pd.DataFrame | None, *, max_days: int = 200) -> pd.DataFrame:
    """Paper B's features of each target at its last start date before the deal, against every
    firm in the universe on the same dates. For the signals the comparison is made among
    firms in the universe (targets outside it have no cohort flags)."""
    cols = ["signal", "targets_n", "targets_share", "universe_share", "difference", "group"]
    if features is None or not len(features) or not len(d):
        return pd.DataFrame(columns=cols)
    have = [s for s in SIGNALS if s in features.columns]
    f = features[["cik", "start_date", *have]].copy()
    f["cik"] = pd.to_numeric(f["cik"], errors="coerce")
    f = f.dropna(subset=["cik"]).astype({"cik": "int64"})
    f["start_date"] = pd.to_datetime(f["start_date"]).astype("datetime64[ns]")
    t = d[["deal_id", "target_cik", "entry_date", "group"]].dropna(subset=["target_cik"]).astype({"target_cik": "int64"})
    j = pd.merge_asof(t.sort_values("entry_date"), f.sort_values("start_date"), left_on="entry_date", right_on="start_date",
                      left_by="target_cik", right_by="cik", direction="backward", tolerance=pd.Timedelta(days=max_days))
    j = j.dropna(subset=["start_date"])
    if not len(j):
        return pd.DataFrame(columns=cols)
    dates = j["start_date"].value_counts()
    rows = []
    for grp, g in [("all buyers", j), ("core", j[j["group"].eq("core")])]:
        for s in have:
            inside = g if s == "in_universe" else g[g["in_universe"].fillna(False).astype(bool)]
            base = f[f["start_date"].isin(dates.index)] if s == "in_universe" else \
                f[f["start_date"].isin(dates.index) & f["in_universe"].fillna(False).astype(bool)]
            v = pd.to_numeric(inside[s], errors="coerce").astype("float64").dropna()
            w = base.assign(w=base["start_date"].map(dates))            # the universe weighted by the targets' dates
            b = pd.to_numeric(w[s], errors="coerce").astype("float64")
            share = float((b * w["w"]).sum() / w.loc[b.notna(), "w"].sum()) if b.notna().any() else np.nan
            rows.append({"signal": s, "targets_n": int(len(v)), "targets_share": float(v.mean()) if len(v) else np.nan,
                         "universe_share": share, "difference": (float(v.mean()) - share) if len(v) else np.nan, "group": grp})
    return pd.DataFrame(rows, columns=cols)


# 6 ── a buyout from the public holder's side: Paper B windows that ended in a take-private ───

def public_holder_outcomes(d: pd.DataFrame, outcomes: pd.DataFrame | None, features: pd.DataFrame | None,
                           horizon: int = 5) -> pd.DataFrame:
    """Paper B's five-year windows on firms that were then taken private inside the window,
    against all other finished windows: what the public holder ended with."""
    cols = ["windows", "n", "firms", "mean_multiple", "median_multiple", "share_below_1x", "share_2x_or_more", "touch_3x"]
    if outcomes is None or features is None or not len(outcomes) or not len(d):
        return pd.DataFrame(columns=cols)
    o = outcomes[(outcomes["horizon_years"] == horizon) & outcomes["end_multiple"].notna()].copy()
    o["start_date"] = pd.to_datetime(o["start_date"]).astype("datetime64[ns]")
    ciks = features[["ticker", "cik"]].dropna().drop_duplicates("ticker")
    o = o.merge(ciks, on="ticker", how="left")
    o["cik"] = pd.to_numeric(o["cik"], errors="coerce")
    t = d[["target_cik", "entry_date", "group"]].dropna().drop_duplicates("target_cik").rename(
        columns={"target_cik": "cik", "entry_date": "deal_entry", "group": "deal_group"})
    t["cik"] = pd.to_numeric(t["cik"], errors="coerce")
    o = o.merge(t, on="cik", how="left")
    inside = o["deal_entry"].notna() & (o["deal_entry"] > o["start_date"]) & \
        (o["deal_entry"] <= o["start_date"] + pd.Timedelta(days=int(horizon * DAYS_PER_YEAR)))
    rows = []
    for name, g in (("ended in a take-private (all buyers)", o[inside]), ("ended in a take-private (core)", o[inside & o["deal_group"].eq("core")]),
                    ("all other finished windows", o[~inside])):
        m = g["end_multiple"]
        rows.append({"windows": name, "n": int(len(g)), "firms": int(g["cik"].nunique()),
                     "mean_multiple": float(m.mean()) if len(g) else np.nan, "median_multiple": _med(m),
                     "share_below_1x": float((m < 1).mean()) if len(g) else np.nan,
                     "share_2x_or_more": float((m >= 2).mean()) if len(g) else np.nan,
                     "touch_3x": float(g["touch_3x"].fillna(False).astype(bool).mean()) if len(g) and "touch_3x" in g.columns else np.nan})
    return pd.DataFrame(rows, columns=cols)


# 7 ── the exit slowdown, on windows that can actually be compared ────────────────────────────

def exit_slowdown(d: pd.DataFrame, *, as_of, horizon: float = 3.0, early=(2019, 2021), late=(2022, 2023)) -> pd.DataFrame:
    """Visible exit within `horizon` years by completion year, only for years with the full
    follow-up, and the comparison the locked A-H3 could not make: completions just before
    against just after the regime change (log-rank on follow-up cut at the horizon)."""
    as_of = pd.Timestamp(as_of)
    x = d[d["exit_years"].notna() & d["completion_date"].notna()].copy()
    full = x["completion_date"] <= as_of - pd.Timedelta(days=int(horizon * DAYS_PER_YEAR))
    x = x[full]
    rows = []
    for (grp, year), g in x.assign(scope="all buyers").groupby(["scope", "completion_year"]):
        t, e = truncate(g["exit_years"], g["exit_event"], horizon)
        rows.append({"completed": str(int(year)), "n": int(len(g)), "exits": int(e.sum()), "exit_rate": kaplan_meier(t, e, horizon),
                     "reipo": int((g["exit_kind"].eq("reIPO") & e).sum()), "p": np.nan})
    for grp, g0 in (("all buyers", x), ("core", x[x["group"].eq("core")])):
        g = g0[g0["completion_year"].between(early[0], late[1])].copy()
        g["period"] = np.where(g["completion_year"] <= early[1], "early", "late")
        t, e = truncate(g["exit_years"], g["exit_event"], horizon)
        for name, key in ((f"{early[0]}–{early[1]} ({grp})", "early"), (f"{late[0]}–{late[1]} ({grp})", "late")):
            m = (g["period"] == key).to_numpy()
            rows.append({"completed": name, "n": int(m.sum()), "exits": int(e[m].sum()), "exit_rate": kaplan_meier(t[m], e[m], horizon),
                         "reipo": int((g["exit_kind"].eq("reIPO").to_numpy() & e & m).sum()),
                         "p": logrank(t, e, g["period"].to_numpy())[1] if key == "late" else np.nan})
    return pd.DataFrame(rows, columns=["completed", "n", "exits", "exit_rate", "reipo", "p"])


# 8 ── sponsor styles without the sector mix ──────────────────────────────────────────────────

def sponsor_styles(d: pd.DataFrame, accuracy: float, min_deals: int = 5) -> pd.DataFrame:
    """Each sponsor's median premium over the benchmark and over the median deal of the same
    sector (so a software buyer is compared with software deals). Names are 79% accurate."""
    x = d.dropna(subset=["log_premium"]).copy()
    x["sector"] = sic_division(x["sic"]).to_numpy()
    x["vs_sector"] = x["log_premium"] - x.groupby("sector")["log_premium"].transform("median")
    x = x[x["sponsor"].fillna("").astype(str).ne("")]
    g = x.groupby("sponsor")
    out = g.agg(deals_with_premium=("deal_id", "size"), premium=("log_premium", "median"), vs_sector=("vs_sector", "median"),
                main_sector=("sector", lambda s: s.mode().iloc[0] if s.notna().any() else "")).reset_index()
    out["median_premium_pct"], out["vs_sector_deals_pct"] = np.expm1(out.pop("premium")), np.expm1(out.pop("vs_sector"))
    out["sponsor_name_accuracy_in_reading"] = accuracy
    return out[out["deals_with_premium"] >= min_deals].sort_values("deals_with_premium", ascending=False)[
        ["sponsor", "deals_with_premium", "median_premium_pct", "vs_sector_deals_pct", "main_sector",
         "sponsor_name_accuracy_in_reading"]]


# 9 ── public against private, year by year ───────────────────────────────────────────────────

def public_vs_private(outcomes: pd.DataFrame | None, vintages: pd.DataFrame | None, horizon: int = 5) -> pd.DataFrame:
    """Paper B's universe by start year (five-year end multiple of every finished window, its
    annual rate) next to WSIB's buyout funds of the same vintage (net multiple, cash returned,
    net IRR). Not like for like: the funds' figures are net of fees and partly unrealized marks,
    held over each fund's own life; the public figures are equal-weighted quarter-end prices."""
    cols = ["year", "public_windows", "public_mean_multiple", "public_median_multiple", "public_mean_annual",
            "public_mean_lower_bound", "funds", "fund_tvpi", "fund_dpi", "fund_median_net_irr"]
    pub = pd.DataFrame(columns=["year"])
    if outcomes is not None and len(outcomes):
        o = outcomes[(outcomes["horizon_years"] == horizon) & outcomes["end_multiple"].notna()].copy()
        o["year"] = pd.to_datetime(o["start_date"]).dt.year
        low = "end_multiple_lower" if "end_multiple_lower" in o.columns else "end_multiple"
        pub = o.groupby("year").agg(public_windows=("end_multiple", "size"), public_mean_multiple=("end_multiple", "mean"),
                                    public_median_multiple=("end_multiple", "median"),
                                    public_mean_lower_bound=(low, "mean")).reset_index()
        pub["public_mean_annual"] = pub["public_mean_multiple"] ** (1 / horizon) - 1
    fund = pd.DataFrame(columns=["year"])
    if vintages is not None and len(vintages):
        fund = vintages.rename(columns={"vintage_year": "year", "pooled_tvpi": "fund_tvpi", "pooled_dpi": "fund_dpi",
                                        "median_net_irr": "fund_median_net_irr"})[["year", "funds", "fund_tvpi", "fund_dpi",
                                                                                   "fund_median_net_irr"]]
    out = pub.merge(fund, on="year", how="outer").sort_values("year")
    out = out[out["year"] >= 2010] if len(out) else out
    return out.reindex(columns=cols)


def benchmark_path(bench: pd.DataFrame | None) -> pd.DataFrame:
    """The public benchmark itself, year by year: how far small-cap multiples moved."""
    cols = ["year", "dates", "median_ev_ebitda", "median_ev_revenue", "firms"]
    if bench is None or not len(bench):
        return pd.DataFrame(columns=cols)
    b = bench.assign(year=pd.to_datetime(bench["start_date"]).dt.year)
    return b.groupby("year").agg(dates=("start_date", "size"), median_ev_ebitda=("median_ev_ebitda", "median"),
                                 median_ev_revenue=("median_ev_revenue", "median"), firms=("firms", "median")).reset_index()[cols]


def run_all(d: pd.DataFrame, cfg: dict, *, bench=None, panel=None, features=None, outcomes=None, vintages=None, as_of) -> dict:
    """Every table, keyed by its name in the block and the workbook."""
    wide, coverage = wider_coverage(d, cfg)
    own = own_history(d, panel)
    acc = cfg["validation"]["term_gates"]["measured_accuracy"]["sponsor_name"]
    return {"1 premium by year": premium_decomposition(d), "1 benchmark by year": benchmark_path(bench),
            "2 margins": margin_test(d, seed=cfg["estimation"]["seed"], draws=cfg["estimation"]["bootstrap_draws"]),
            "3 wider coverage": coverage, "4 own history": own_history_table(d, own),
            "5 target signals": target_signals(d, features), "6 public holder": public_holder_outcomes(d, outcomes, features),
            "7 exit slowdown": exit_slowdown(d, as_of=as_of), "8 sponsor styles": sponsor_styles(d, acc),
            "9 public vs private": public_vs_private(outcomes, vintages), "_own_history_deals": own}
