"""Paper B's estimators on synthetic windows with built-in answers."""
import numpy as np
import pandas as pd
import pytest

from aem_research import estimators as est
from aem_research.features import H4_SIGNALS

DATES = pd.date_range("2013-07-01", "2023-07-01", freq="QS")


def _world(firms=700, seed=0, lr_h2=3.0, signal_effect=None, null=False):
    """Features and outcomes for `firms` firms on every quarterly date. Hits depend on the
    signals through known log-odds, plus a firm effect so that a firm's windows hang together."""
    rng = np.random.default_rng(seed)
    effect = {"s1_proxy": 0.9, "insider_any": 0.5, "insider_cluster": 0.0, "insider_opportunistic": 0.7, "net_cash_pos": 0.0,
              "fcf_pos": 0.4, "low_dilution": 0.0, "low_attention": -0.5} if signal_effect is None else signal_effect
    if null:
        effect = {k: 0.0 for k in effect}
    n = firms * len(DATES)
    f = pd.DataFrame({"cik": np.repeat(np.arange(1, firms + 1), len(DATES)), "start_date": np.tile(DATES, firms)})
    f["ticker"] = "CIK" + f["cik"].astype(str)
    f["in_universe"] = True
    firm_fx = np.repeat(rng.normal(0, 0.5, firms), len(DATES))
    for s, p_ in zip(H4_SIGNALS, (0.3, 0.2, 0.05, 0.1, 0.5, 0.6, 0.5, 0.33)):
        f[s] = rng.random(n) < p_
    f["insider_cluster"] = f["insider_cluster"] & f["insider_any"]
    f["insider_opportunistic"] = f["insider_opportunistic"] & f["insider_any"]
    f["insider_routine_only"] = f["insider_any"] & ~f["insider_opportunistic"] & (rng.random(n) < 0.5)
    f["insider_unclassified_only"] = f["insider_any"] & ~f["insider_opportunistic"] & ~f["insider_routine_only"]
    f["insider_post_results_cluster"] = f["insider_cluster"] & (rng.random(n) < 0.5)
    f["sale_any"] = rng.random(n) < 0.2
    f["fallen"] = rng.random(n) < 0.3
    f["fallen_relative"] = f["fallen"]
    f["cohort_known"] = rng.random(n) < 0.97
    f["h2a"] = f["fallen"] & f["fcf_pos"] & f["net_cash_pos"] & f["insider_any"]
    f["market_cap"] = np.exp(rng.normal(20.5, 1.0, n))
    f["net_cash"] = f["market_cap"] * rng.normal(0.05, 0.2, n)
    f["s1_proxy"] = f["s1_proxy"].astype("boolean").mask(rng.random(n) < 0.05)       # some signals unknown
    lin = sum(effect[s] * f[s].fillna(False).astype(float) for s in H4_SIGNALS) + firm_fx
    year_fx = f["start_date"].dt.year.map({2020: 0.8}).fillna(0.0)
    p10 = 1 / (1 + np.exp(-(-4.0 + lin + year_fx)))
    p3 = 1 / (1 + np.exp(-(-2.3 + 0.6 * lin + year_fx + (0.0 if null else np.log(lr_h2)) * f["h2a"])))
    hit3, hit10 = rng.random(n) < p3, rng.random(n) < p10
    peak = np.where(hit10, rng.uniform(10, 30, n), np.where(hit3, rng.uniform(3, 9.9, n), rng.uniform(0.5, 2.9, n)))
    end = np.clip(peak * rng.uniform(0.3, 1.0, n), 0.02, None)
    bucket = np.asarray([0.1, 0.4, 1.0, 2.5, 5.0, 10.0])[np.searchsorted([0.2, 0.632, 1.58, 3.54, 7.07], end, side="right")]
    rows = []
    for h in (3, 5):
        last = pd.Timestamp("2026-07-01") - pd.DateOffset(years=h)
        o = pd.DataFrame({"ticker": f["ticker"], "start_date": f["start_date"], "horizon_years": h,
                          "status": np.where(f["start_date"] <= last, "complete", "censored"),
                          "touch_3x": hit3 | hit10, "touch_10x": hit10 & (h == 5), "touch_6.24x": (peak >= 6.24) & (h == 5),
                          "max_multiple": peak, "end_multiple": end, "end_multiple_lower": end * 0.9, "end_multiple_upper": end,
                          "bucket": bucket, "days_to_3x": np.where(hit3 | hit10, 400.0, np.nan),
                          "days_to_10x": np.where(hit10, 900.0, np.nan), "delist_category": ""})
        rows.append(o)
    return f, pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def world():
    return _world()


@pytest.fixture(scope="module")
def w(world):
    return est.windows(*world)


def test_windows_join_features_and_resolved_outcomes_only(world, w):
    f, o = world
    assert len(w) == len(f) and w["firm"].nunique() == 700
    late = w["start_date"] > pd.Timestamp("2021-07-01")
    assert w.loc[late, "hit10_5y"].isna().all() and w.loc[~late, "hit10_5y"].notna().all()       # five years must be over
    assert w.loc[w["start_date"] <= "2023-07-01", "hit3_3y"].notna().all()
    assert set(w["regime"]) == {"before 2022", "2022 and later"}
    f2 = f.assign(in_universe=f["cik"] % 2 == 0)
    assert len(est.windows(f2, o)) == len(f) // 2


def test_a_likelihood_ratio_is_the_ratio_of_signal_shares_with_a_firm_cluster_interval(w):
    r = est.likelihood_ratio(w, "s1_proxy", "hit10_5y", resamples=500)
    x = w[w["s1_proxy"].notna() & w["hit10_5y"].notna()]
    s, h = x["s1_proxy"].astype(float), x["hit10_5y"]
    by_hand = (s[h == 1].mean()) / (s[h == 0].mean())
    assert r["lr"] == pytest.approx(by_hand) and r["windows"] == len(x) and 0.03 < r["left_out_share"] < 0.07
    assert r["ci_low"] < r["lr"] < r["ci_high"] and r["ci_low"] > 1.2 and r["p_one_sided"] < 0.01
    assert r["hit_rate_with_signal"] == pytest.approx(h[s == 1].mean()) and r["base_rate"] == pytest.approx(h.mean())
    # windows of one firm move together: the cluster interval is wider than one that ignores firms
    rng = np.random.default_rng(9)                                      # a signal and a hit that both persist within the firm
    sticky = x.assign(sig=(rng.random(701) < 0.3)[x["firm"].astype(int)],
                      hit=(rng.random(len(x)) < np.where((rng.random(701) < 0.2)[x["firm"].astype(int)], 0.5, 0.05)).astype(float))
    by_firm = est.likelihood_ratio(sticky, "sig", "hit", resamples=500)
    iid = est.likelihood_ratio(sticky.assign(firm=np.arange(len(sticky))), "sig", "hit", resamples=500)
    assert (by_firm["ci_high"] - by_firm["ci_low"]) > 2 * (iid["ci_high"] - iid["ci_low"]) and by_firm["lr"] == pytest.approx(iid["lr"])
    none = est.likelihood_ratio(w, "low_dilution", "hit10_5y", resamples=500)                       # no effect built in
    assert none["ci_low"] < 1 < none["ci_high"] and none["p_one_sided"] > 0.05
    assert np.isnan(est.likelihood_ratio(w.iloc[0:0], "s1_proxy", "hit10_5y")["lr"])


def test_h2_finds_the_cohort_effect_separates_the_placebo_and_shows_what_carries_it(w):
    r = est.h2(w, resamples=500)
    a, b = r["main"]["H2a"], r["main"]["H2b"]
    assert a["lr"] > 1.5 and a["verdict"] == "supported" and a["holm_p"] >= a["p_one_sided"] and b["with_signal"] < a["with_signal"]
    assert r["placebo"]["lr"] < a["lr"] and r["placebo"]["ci_low"] < 1.3
    ab = r["ablation"].set_index("cohort")
    assert ab.loc["full cohort (H2a)", "lr"] == pytest.approx(a["lr"])
    assert ab.loc["without the fall", "with_signal"] > ab.loc["full cohort (H2a)", "with_signal"]
    assert set(r["h2c"]["insiders"]) == {"opportunistic", "routine only", "unclassified only"}
    assert r["h2c"]["windows"].sum() == a["with_signal"] and np.isfinite(r["h2c_test"]["p_two_sided"])
    assert r["windows"] < len(w) and r["base_rate"] == pytest.approx(w[w["cohort_known"] & w["hit3_3y"].notna()]["hit3_3y"].mean())
    quiet = est.h2(est.windows(*_world(firms=500, seed=3, null=True)), resamples=500)
    assert quiet["main"]["H2a"]["verdict"] == "not supported" and quiet["main"]["H2a"]["ci_low"] < 1 < quiet["main"]["H2a"]["ci_high"]


def test_h3_compares_the_two_horizons_on_the_same_windows_and_names_the_equivalent_multiple(w):
    r = est.h3(w, resamples=500)
    x = w[w["hit3_3y"].notna() & w["hit624_5y"].notna()]
    assert r["windows"] == len(x) and r["p_3x_3y"] == pytest.approx(x["hit3_3y"].mean())
    assert r["ratio"] == pytest.approx(x["hit624_5y"].mean() / x["hit3_3y"].mean()) and r["ci_low"] < r["ratio"] < r["ci_high"]
    assert r["ratio"] < 0.9 and r["verdict"] == "mapping rejected"                # 6.24x is rarer here by construction
    # the equivalent multiple is touched as often in five years as 3x in three
    assert (x["max5"] >= r["equivalent_multiple_5y"]).mean() == pytest.approx(r["p_3x_3y"], abs=0.002)
    same = w.assign(hit624_5y=w["hit3_3y"].where(w["hit624_5y"].notna()))
    assert est.h3(same, resamples=500)["verdict"] == "mapping kept" and est.h3(w.iloc[0:0]) == {"windows": 0}


def test_h4_reports_fit_and_test_periods_and_corrects_across_the_eight_signals(w):
    r = est.h4(w, resamples=500)
    t = r["table"].set_index("signal")
    assert list(t.index) == list(H4_SIGNALS) and r["fit_windows"] > 0 and r["test_windows"] > 0
    fit, test = est.split(w[w["hit10_5y"].notna()])
    assert fit["start_year"].max() == 2016 and set(test["start_year"]) == {2017, 2018, 2019, 2020, 2021}
    assert t.loc["s1_proxy", "verdict"] == "supported" and t.loc["s1_proxy", "test_lr"] > 1.3
    assert t.loc["low_attention", "test_lr"] < 1 and t.loc["low_attention", "verdict"] == "not supported"
    assert t.loc["low_dilution", "verdict"] == "not supported" and (t["holm_p"] >= t["p_one_sided"] - 1e-12).all()
    quiet = est.h4(est.windows(*_world(firms=500, seed=4, null=True)), resamples=500)["table"]
    assert (quiet["verdict"] == "not supported").all()


def test_the_logit_recovers_known_coefficients_with_clustered_errors():
    rng = np.random.default_rng(5)
    n, firms = 40000, 2000
    firm = rng.integers(0, firms, n)
    X = np.column_stack([rng.random(n) < 0.3, rng.normal(0, 1, n)]).astype(float)
    y = rng.random(n) < 1 / (1 + np.exp(-(-2.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1])))
    m = est.logit(y, X, firm)
    assert m["beta"] == pytest.approx([-2.0, 0.8, -0.5], abs=0.08) and (m["se"] > 0).all() and (m["se"] < 0.06).all()
    # a regressor that is constant within firms: clustering must widen its error
    z = rng.normal(0, 1, firms)[firm]
    yz = rng.random(n) < 1 / (1 + np.exp(-(-2.0 + 0.5 * z + rng.normal(0, 1, firms)[firm])))
    clustered, naive = est.logit(yz, z[:, None], firm)["se"][1], est.logit(yz, z[:, None], np.arange(n))["se"][1]
    assert clustered > 1.5 * naive


def test_the_ordered_logit_recovers_known_coefficients_and_ordered_cuts():
    rng = np.random.default_rng(6)
    n = 30000
    X = np.column_stack([rng.random(n) < 0.4, rng.normal(0, 1, n)]).astype(float)
    latent = 0.7 * X[:, 0] - 0.4 * X[:, 1] + rng.logistic(0, 1, n)
    y = np.searchsorted([-1.5, -0.3, 0.8, 2.0, 3.2], latent)
    m = est.ordered_logit(y, X, rng.integers(0, 3000, n))
    assert m["converged"] and m["beta"] == pytest.approx([0.7, -0.4], abs=0.06)
    assert m["cuts"] == pytest.approx([-1.5, -0.3, 0.8, 2.0, 3.2], abs=0.08) and (np.diff(m["cuts"]) > 0).all()
    assert (m["se"] > 0).all() and (m["se"] < 0.05).all()


def test_the_model_is_scored_against_the_checklist_and_the_base_rate_out_of_sample(w):
    r = est.m2_and_benchmark(w, resamples=300)
    c = r["coefficients"].set_index("regressor")
    assert c.loc["s1_proxy", "coef"] == pytest.approx(0.9, abs=0.25) and c.loc["low_attention", "coef"] < 0
    assert c.loc["s1_proxy", "p"] < 0.001 and set(c.index) == {*H4_SIGNALS, "h2a", "net_cash_to_cap"}
    b = r["brier"].set_index("model")
    assert b.loc["M2 (fitted logit)", "brier"] < b.loc["base rate", "brier"]
    assert b.loc["M2 minus base rate", "ci_high"] < 0                       # the signals carry information out of sample
    assert b.loc["M2 minus checklist", "brier"] < 0 and r["terminal_uses"] in ("the fitted model", "the checklist")
    assert 0.03 < r["left_out_share"] < 0.07 and r["fit_windows"] + r["test_windows"] == r["windows"]
    # with no information in the signals the model does not beat the base rate
    quiet = est.m2_and_benchmark(est.windows(*_world(firms=500, seed=4, null=True)), resamples=300)
    assert quiet["brier"].set_index("model").loc["M2 minus base rate", "ci_high"] > 0 and quiet["terminal_uses"] == "the checklist"
    assert len(est.m2_and_benchmark(w.iloc[:50])["brier"]) == 0


def test_m1_the_exploratory_tables_and_the_robustness_versions_run_on_the_same_windows(world):
    f, o = _world(firms=300, seed=8)
    r = est.run_all(f, o, resamples=200)
    assert set(r) == {"windows", "h2", "h3", "h4", "m2", "m1", "h5", "h6", "touch_years", "delisting_classes", "robustness"}
    m1 = r["m1"]["coefficients"].set_index("regressor")
    assert r["m1"]["converged"] and m1.loc["s1_proxy", "coef"] > 0 and len(m1) == 10
    assert set(r["h5"]["regime"]) == {"before 2022", "2022 and later"} and len(r["h6"]) == 2
    ty = r["touch_years"]
    assert set(ty["event"]) == {"first 3x touch", "first 10x touch"} and ty.groupby("event")["share"].sum().round(6).eq(1).all()
    rb = r["robustness"].set_index("version")
    assert {"primary", "January formation dates only", "market cap from US$300m", "without the April and July 2020 start dates",
            "H2 on start dates from 2016", "drawdown relative to the universe median (H2)"} == set(rb.index)
    assert rb.loc["primary", "h2a_lr"] == pytest.approx(est.h2(r["windows"], resamples=200)["main"]["H2a"]["lr"])
    assert rb.loc["January formation dates only", "windows"] < rb.loc["primary", "windows"] / 3
    assert len(est.NOT_COMPUTED) == 7


def test_a_flag_that_never_varies_or_never_hits_is_reported_not_estimated(w):
    x, regs = est.model_frame(w, "hit10_5y")
    x = x.assign(insider_cluster=0.0, low_dilution=1.0)
    x.loc[x["insider_opportunistic"].eq(1), "hit10_5y"] = 0.0                  # nobody with the signal hits
    keep, dropped = est.usable(x, regs, x["hit10_5y"])
    assert dropped == {"insider_cluster": "no variation in the sample", "low_dilution": "no variation in the sample",
                       "insider_opportunistic": "no hit with the signal"} and len(keep) == len(regs) - 3
    r = est.m2_and_benchmark(x.assign(net_cash=x["net_cash"]), resamples=200)
    c = r["coefficients"].set_index("regressor")
    assert list(c.index) == regs and c.loc["insider_opportunistic", "note"] == "no hit with the signal"
    assert np.isnan(c.loc["insider_opportunistic", "coef"]) and c.loc["s1_proxy", "note"] == "" and abs(c.loc["s1_proxy", "coef"]) < 3
    assert (c["se"].dropna() < 1).all()                                        # no runaway coefficient is left in the table
    m = est.m1(x)["coefficients"].set_index("regressor")
    assert m.loc["low_dilution", "note"] == "no variation in the sample" and (m["se"].dropna() < 1).all()
    from aem_research.report import _coef_line
    assert _coef_line("M2        ", {"regressor": "insider_cluster", "coef": float("nan"), "note": "no variation in the sample"}) == \
        "M2        insider_cluster · not estimated: no variation in the sample"
