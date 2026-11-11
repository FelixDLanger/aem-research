"""Paper A's results: the sample, the three confirmatory tests and the exploratory tables."""
import gzip

import numpy as np
import pandas as pd
import pytest
import yaml

from aem_research import paper_a_results as par
from aem_research.takeprivate import _safe

CFG = yaml.safe_load(open("config/paper_a.yaml", encoding="utf-8"))
AS_OF = "2026-11-11"


def _deals(n=240, seed=3, premium_before=0.25, premium_after=0.0, exit_early=0.35, exit_late=0.10):
    """Deals with a known premium over a flat benchmark of 10x and known exit rates by cohort."""
    rng = np.random.default_rng(seed)
    entry = pd.to_datetime("2016-02-15") + pd.to_timedelta(rng.integers(0, 3700, n), unit="D")
    post = entry >= pd.Timestamp("2022-01-01")
    ev_ebitda = 10 * np.exp(np.where(post, premium_after, premium_before) + rng.normal(0, 0.3, n))
    done = entry + pd.to_timedelta(rng.integers(60, 200, n), unit="D")
    # completion years 2012–2021 for the exit test: shift a third of the deals back
    back = rng.random(n) < 0.4
    done = done.where(~back, pd.to_datetime("2012-03-01") + pd.to_timedelta(rng.integers(0, 1700, n), unit="D"))
    entry = entry.where(~back, done - pd.Timedelta(days=90))
    early = done.year <= 2016
    exits = rng.random(n) < np.where(early, exit_early, exit_late)
    years = rng.uniform(0.6, 4.5, n)
    exit_date = pd.Series(done + pd.to_timedelta((years * 365).astype(int), unit="D")).where(exits)
    reipo = exits & (rng.random(n) < 0.2)
    ebitda = rng.uniform(2e7, 3e8, n)
    d = pd.DataFrame({
        "deal_id": [f"{1000 + i}-x" for i in range(n)], "target_name": [f"T{i}" for i in range(n)],
        "doc_adsh": [f"A{i}" for i in range(n)], "main_file": "defm14a.htm", "sic": rng.choice([7372, 3674, 5812, 6500], n),
        "first_filed": entry, "completed": rng.random(n) < 0.92, "completion_date": done,
        "buyer_type": rng.choice(["sponsor", "sponsor not on the list", "strategic or unclear", "SPAC merger"], n,
                                 p=[0.55, 0.1, 0.3, 0.05]),
        "ev_ebitda": ev_ebitda, "ev_revenue": ev_ebitda / 5, "ebitda": ebitda, "enterprise_value": ev_ebitda * ebitda,
        "equity_value": ev_ebitda * ebitda * 0.8, "offer_price": 20.0,
        "reipo_date": exit_date.where(reipo), "sale_date": exit_date.where(~reipo),
        "check_price": rng.choice(["agree", "one source", "disagree"], n, p=[0.6, 0.35, 0.05]), "check_value": "agree",
        "second_deal_value_usd": ev_ebitda * ebitda * rng.normal(1.0, 0.05, n),
        "equity_commitment_usd": ev_ebitda * ebitda * 0.4, "debt_commitment_usd": ebitda * 5.5,
        "leverage_x": 5.5, "equity_share": 0.4, "sponsor": rng.choice(["KKR", "Apollo", "Thoma Bravo", ""], n),
        "entry_real_yield_10y": np.where(post, 1.8, 0.3) + rng.normal(0, 0.2, n),
        "entry_credit_spread_baa": rng.normal(2.2, 0.3, n), "entry_tightening_2y_12m": rng.normal(0, 1, n),
        "entry_inflation_cpi_yoy": rng.normal(0.03, 0.01, n)})
    d["stock_mentions"] = np.where(np.arange(n) % 40 == 0, 30.0, 0.0)
    return d


BENCH = pd.DataFrame({"start_date": pd.date_range("2016-01-01", "2026-07-01", freq="QS"), "firms": 1400,
                      "median_ev_ebitda": 10.0, "median_ev_revenue": 2.0})


def test_the_run_is_identified_from_its_saved_block():
    block = ("PAPER A RUN SUMMARY v11 – paste everything\n" + "=" * 72 + "\nv11 · run 2026-11-11 05:33 UTC · repo eca6f4a · "
             "notebook df1e6ca31b56 · config A 90c17f7af446 (unlocked) · {}\nSEC ID    colab secret\n")
    assert par.run_identity(block) == {"release": 11, "generated_utc": "2026-11-11 05:33 UTC", "repo_commit": "eca6f4a",
                                       "notebook_build": "df1e6ca31b56"}
    assert par.run_identity("nothing here") is None and par.run_identity(None) is None


def test_the_sample_follows_the_locked_rules_and_counts_every_step():
    d, f = par.analysis_sample(_deals(), BENCH, CFG, as_of=AS_OF)
    raw = _deals()
    assert f["deals"] == 240 and f["completed"] == int(raw["completed"].sum())
    assert f["not_spac"] == int((raw["completed"] & raw["buyer_type"].ne("SPAC merger")).sum())
    assert f["stock_or_mixed_excluded"] >= 1 and f["analysis_sample"] == len(d) == f["core"] + f["no_sponsor_identified"]
    assert not d["buyer_type"].eq("SPAC merger").any() and (d["stock_mentions"] < 5).all()
    assert set(d["group"]) == {"core", "no sponsor identified"}
    # financial firms and deals before the benchmark's first date have no premium
    assert d.loc[d["sic"].eq(6500), "log_premium"].isna().all()
    assert d.loc[d["entry_date"] < "2016-01-01", "log_premium"].isna().all() and d["log_premium"].notna().sum() > 60
    ok = d["log_premium"].notna()
    assert np.allclose(d.loc[ok, "log_premium"], np.log(d.loc[ok, "ev_ebitda"] / 10.0))
    # exits: time runs from completion to the exit, or to the data run's date
    ev = d[d["exit_event"]]
    assert np.allclose(ev["exit_years"], (ev["exit_date"] - ev["completion_date"]).dt.days / 365.25)
    no = d[~d["exit_event"]]
    assert np.allclose(no["exit_years"], ((pd.Timestamp(AS_OF) - no["completion_date"]).dt.days / 365.25).clip(lower=0))
    assert set(d["cohort"]) <= {"2012–2016", "2017–2021", ""} and set(ev["exit_kind"]) <= {"reIPO", "sold to a listed buyer"}
    # without a benchmark nothing has a premium and nothing breaks
    d0, f0 = par.analysis_sample(_deals(), None, CFG, as_of=AS_OF)
    assert f0["with_ev_ebitda_premium"] == 0 and len(d0) == len(d)
    e, fe = par.analysis_sample(_deals().iloc[0:0], BENCH, CFG, as_of=AS_OF)
    assert len(e) == 0 and fe["analysis_sample"] == 0


def test_the_three_tests_find_what_was_built_in_and_stay_quiet_on_nothing():
    d, _ = par.analysis_sample(_deals(n=900, seed=5), BENCH, CFG, as_of=AS_OF)
    c = par.confirmatory(d, CFG)
    p, x = c["premium"], c["exits"]
    assert 0.15 < p["median_log_premium"] and p["ci_low_pct"] > 0 and p["h1_p"] < 0.001      # a premium was built in
    assert p["diff_log"] < -0.1 and p["diff_ci_high"] < 0 and p["h2_p"] < 0.01                # smaller from 2022 on
    assert 0.25 < x["early_rate"] < 0.45 and 0.04 < x["late_rate"] < 0.17 and x["h3_p"] < 0.01
    assert c["verdict"] == {"A-H1": "supported", "A-H2": "supported", "A-H3": "supported"}
    assert all(c["holm"][k] >= v for k, v in (("A-H1", p["h1_p"]), ("A-H2", p["h2_p"]), ("A-H3", x["h3_p"])))
    assert set(c["robustness"]) == {"EV/revenue, core", "EV/EBITDA, all buyers",
                                    "EV/EBITDA, core, no two-source disagreement", "exits, all buyers"}
    assert c["robustness"]["EV/EBITDA, all buyers"]["n"] > p["n"] > c["robustness"]["EV/EBITDA, core, no two-source disagreement"]["n"]
    # no premium, no regime difference, equal exit rates: nothing is supported
    flat, _ = par.analysis_sample(_deals(n=900, seed=6, premium_before=0.0, premium_after=0.0, exit_early=0.2, exit_late=0.2),
                                  BENCH, CFG, as_of=AS_OF)
    assert set(par.confirmatory(flat, CFG)["verdict"].values()) == {"not supported"}
    # the opposite of the hypotheses is named as such
    opp, _ = par.analysis_sample(_deals(n=900, seed=7, premium_before=-0.3, premium_after=0.1, exit_early=0.05, exit_late=0.4),
                                 BENCH, CFG, as_of=AS_OF)
    assert set(par.confirmatory(opp, CFG)["verdict"].values()) == {"significant, opposite direction"}
    # too little data is not a result
    tiny = par.confirmatory(d.iloc[:4], CFG)
    assert tiny["verdict"]["A-H1"] == "not testable"


def test_the_statistics_agree_with_scipy_and_with_hand_counts():
    from scipy import stats
    rng = np.random.default_rng(1)
    # Kaplan–Meier by hand: 10 subjects, events at 1 and 2, one censored at 1.5
    t = np.array([1, 1.5, 2, 3, 3, 3, 3, 3, 3, 3.0]); e = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    assert par.kaplan_meier(t, e, 2.5) == pytest.approx(1 - (9 / 10) * (7 / 8))
    assert par.kaplan_meier(t, e, 0.5) == 0.0 and np.isnan(par.kaplan_meier([], [], 1))
    # log-rank against scipy's
    a, b = rng.exponential(2.0, 150), rng.exponential(4.0, 150)
    ea, eb = a < 5, b < 5
    time, event = np.minimum(np.r_[a, b], 5), np.r_[ea, eb]
    chi, p = par.logrank(time, event, np.r_[["a"] * 150, ["b"] * 150])
    ref = stats.logrank(stats.CensoredData(uncensored=time[:150][ea], right=time[:150][~ea]),
                        stats.CensoredData(uncensored=time[150:][eb], right=time[150:][~eb]))
    assert chi == pytest.approx(ref.statistic ** 2, rel=1e-6) and p == pytest.approx(ref.pvalue, rel=1e-5)
    assert np.isnan(par.logrank(time, event, ["a"] * 300)[0])
    tt, ee = par.truncate([1, 6, 7], [True, True, False], 5)
    assert tt.tolist() == [1, 5, 5] and ee.tolist() == [True, False, False]
    # Holm
    h = par.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": np.nan})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06) and np.isnan(h["d"])
    # bootstrap: reproducible, covers the median, empty below three values
    v = rng.normal(0.2, 0.5, 400)
    lo, hi = par.bootstrap_ci(v, seed=1)
    assert lo < np.median(v) < hi and (lo, hi) == par.bootstrap_ci(v, seed=1) and np.isnan(par.bootstrap_ci([1, 2])[0])
    dlo, dhi = par.bootstrap_diff_ci(v + 1, v, seed=1)
    assert 0.8 < dlo < 1 < dhi < 1.2
    # OLS with HC1 against a known line
    x = rng.normal(0, 1, 500)
    o = par.ols(2 + 3 * x + rng.normal(0, 0.1, 500), pd.DataFrame({"x": x})).set_index("term")
    assert o.loc["x", "coef"] == pytest.approx(3, abs=0.02) and o.loc["intercept", "coef"] == pytest.approx(2, abs=0.02)
    assert o.loc["x", "p"] < 1e-6 and o.loc["x", "n"] == 500 and len(par.ols([1, 2], pd.DataFrame({"x": [1, 2]}))) == 0


def test_every_exploratory_table_comes_out_and_carries_its_caveat():
    d, _ = par.analysis_sample(_deals(n=600, seed=8), BENCH, CFG, as_of=AS_OF)
    funds = pd.DataFrame({"report_date": pd.to_datetime(["2025-09-30"] * 6 + ["2026-03-31"] * 6),
                          "strategy": ["Corporate Finance/Buyout - Large"] * 4 + ["Growth Equity", "Corporate Finance/Buyout - Mid"]
                          + ["Corporate Finance/Buyout - Large"] * 4 + ["Growth Equity", "Corporate Finance/Buyout - Mid"],
                          "name": list("abcdef") * 2, "vintage_year": [2016, 2016, 2019, 2019, 2019, 2021] * 2,
                          "paid_in": [100.0] * 12, "distributed": [150, 90, 20, 30, 5, 0, 160, 100, 30, 40, 5, 10.0],
                          "market_value": [40, 60, 110, 100, 120, 105, 35, 55, 105, 95, 120, 100.0],
                          "total_value": [190, 150, 130, 130, 125, 105, 195, 155, 135, 135, 125, 110.0],
                          "net_irr": [0.15, 0.1, 0.12, 0.11, 0.2, 0.03] * 2})
    x = par.exploratory(d, funds, CFG)
    assert set(x) == {"premium by group", "premium regression", "exit probability", "holding by exit year",
                      "leverage (unvalidated)", "sponsors (unvalidated)", "measurement", "WSIB by vintage",
                      "WSIB distribution pace", "premium vs fund returns"}
    assert all(len(t) for k, t in x.items())
    by = x["premium by group"]
    assert {"group", "regime", "entry_year", "size_tercile", "sector", "group (EV/revenue)"} <= set(by["by"])
    assert by.loc[by["by"].eq("regime") & by["group"].eq("before 2022"), "median_premium_pct"].iloc[0] > \
        by.loc[by["by"].eq("regime") & by["group"].eq("2022 or later"), "median_premium_pct"].iloc[0]
    assert {"real_yield_10y", "core", "intercept"} <= set(x["premium regression"]["term"])
    ex = x["exit probability"]
    assert ex.loc[ex["group"].eq("2012–2016"), "exit_by_5y"].iloc[0] > ex.loc[ex["group"].eq("2017–2021"), "exit_by_5y"].iloc[0]
    assert (x["leverage (unvalidated)"]["term_accuracy_in_reading"] <= 0.84).all()
    assert (x["sponsors (unvalidated)"]["sponsor_name_accuracy_in_reading"] == 0.79).all()
    w = x["WSIB by vintage"].set_index("vintage_year")            # latest report, buyout only
    assert w.loc[2016, "funds"] == 2 and w.loc[2016, "pooled_tvpi"] == pytest.approx(1.75) and w.loc[2019, "funds"] == 2
    assert w.loc[2016, "pooled_dpi"] == pytest.approx(1.3) and 2021 in w.index
    pace = x["WSIB distribution pace"]
    assert len(pace) == 2 and pace["distributed_since_last_report"].iloc[1] == pytest.approx(50.0)
    # nothing to explore is not an error
    e, _ = par.analysis_sample(_deals().iloc[0:0], BENCH, CFG, as_of=AS_OF)
    assert all(isinstance(t, pd.DataFrame) for t in par.exploratory(e, None, CFG).values())


def test_stock_mergers_are_counted_from_the_cached_documents(tmp_path):
    d = pd.DataFrame({"deal_id": ["a", "b", "c"], "doc_adsh": ["A1", "B1", "C1"], "main_file": ["x.htm", "y.htm", None],
                      "doc_file": ["x.htm", "y.htm", "z.htm"]})
    (tmp_path / f"A1_{_safe('x.htm')}.txt.gz").write_bytes(gzip.compress(("the exchange ratio " * 12 + "cash").encode()))
    (tmp_path / f"B1_{_safe('y.htm')}.txt.gz").write_bytes(gzip.compress(b"each share converts into $10.00 in cash"))
    m = par.stock_mentions(d, tmp_path)
    assert m.iloc[0] == 12 and m.iloc[1] == 0 and np.isnan(m.iloc[2])
