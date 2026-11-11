"""The exploratory tables that join Paper A's deals with Paper B's caches."""
import numpy as np
import pandas as pd
import pytest
import yaml

from aem_research import paper_a_explore as px
from aem_research import paper_a_results as par
from test_paper_a_results import AS_OF, BENCH, _deals

CFG = yaml.safe_load(open("config/paper_a.yaml", encoding="utf-8"))


def _sample(n=600, seed=11, **kw):
    raw = _deals(n=n, seed=seed, **kw)
    raw["target_cik"] = 1000 + np.arange(n)
    raw["revenue"] = raw["ebitda"] / 0.12                       # targets earn 12% on revenue
    raw["second_deal_value_kind"] = np.where(np.arange(n) % 3 == 0, "equity", "transaction")
    d, _ = par.analysis_sample(raw, BENCH, CFG, as_of=AS_OF)
    return d


def test_the_premium_is_split_into_the_deals_multiple_and_the_markets():
    d = _sample()
    t = px.premium_decomposition(d)
    assert (t["benchmark_ev_ebitda"] == 10.0).all() and (t["deals"] >= 5).all()
    assert np.allclose(t["median_premium_pct"], t["deal_ev_ebitda"] / 10.0 - 1, atol=0.08)
    early, late = t[t["entry_year"] < 2022]["median_premium_pct"].mean(), t[t["entry_year"] >= 2022]["median_premium_pct"].mean()
    assert early > late + 0.1                                    # built in: a premium before 2022, none after
    b = px.benchmark_path(BENCH)
    assert b["median_ev_ebitda"].eq(10.0).all() and b["dates"].sum() == len(BENCH) and len(px.benchmark_path(None)) == 0


def test_thin_margins_show_against_the_benchmarks_implied_margin():
    d = _sample()
    m = px.margin_test(d).set_index("group")
    # benchmark medians 2.0 and 10.0 imply a 20% margin; the targets were given 12%
    assert m.loc["core", "benchmark_implied_margin"] == pytest.approx(0.2) and m.loc["core", "target_margin"] == pytest.approx(0.12)
    assert m.loc["core", "median_ratio"] == pytest.approx(0.6, abs=0.01) and m.loc["core", "p"] < 0.001
    assert m.loc["core", "share_below_benchmark"] == 1.0 and m.loc["all", "n"] > m.loc["core", "n"]


def test_stated_values_widen_the_coverage_only_where_the_computed_value_is_missing():
    d = _sample()
    gone = d.index[::2]
    known = d["log_premium"].copy()
    d.loc[gone, ["enterprise_value", "ev_ebitda", "log_premium"]] = np.nan          # no debt tag: no computed value
    wide, t = px.wider_coverage(d, CFG)
    t = t.set_index(["enterprise_value", "group"])
    assert t.loc[("with stated values", "all buyers"), "n"] > t.loc[("computed enterprise value only", "all buyers"), "n"]
    kept = wide.index.difference(gone)
    assert np.allclose(wide.loc[kept, "log_premium_wide"].dropna(), known.loc[kept].dropna())   # computed values stay
    filled = wide.loc[gone]
    assert filled.loc[filled["second_deal_value_kind"].eq("equity"), "log_premium_wide"].isna().all()   # an equity value is not an EV
    ok = filled["log_premium_wide"].notna()
    assert ok.sum() > 20 and np.allclose(filled.loc[ok, "log_premium_wide"], known.loc[gone][ok], atol=0.25)
    assert set(wide.loc[ok[ok].index, "ev_source"]) == {"stated in the announcement"}


def test_the_offer_is_set_against_the_targets_own_prices_before_the_deal():
    d = _sample(n=200)
    rows = []
    for r in d.itertuples():
        q = pd.date_range(end=r.entry_date - pd.Timedelta(days=50), periods=14, freq="QE")
        price = np.r_[np.full(6, 30.0), np.full(len(q) - 6, 16.0)]       # a fall from 30 to 16; the offer is 20
        rows += [{"cik": r.target_cik, "date": dt, "price": p_} for dt, p_ in zip(q, price)]
        rows.append({"cik": r.target_cik, "date": r.entry_date + pd.Timedelta(days=20), "price": 19.9})   # after the news: unused
    panel = pd.DataFrame(rows)
    own = px.own_history(d, panel)
    assert len(own) > 150 and np.allclose(own["premium_to_last"], 20 / 16 - 1) and (own["quarters_seen"] == 12).all()
    assert np.allclose(own["offer_to_own_high"], 20 / 30) and (own["pre_date"] < d.set_index("deal_id").loc[own["deal_id"], "entry_date"].to_numpy()).all()
    t = px.own_history_table(d, own).set_index("group")
    assert t.loc["all", "median_premium_to_last"] == pytest.approx(0.25) and t.loc["all", "share_offer_below_own_high"] == 1.0
    assert {"core", "no sponsor identified", "entered before 2022", "entered 2022 or later"} <= set(t.index)
    # a firm without prices shortly before the deal is left out; no panel is no table
    stale = panel.assign(date=panel["date"] - pd.Timedelta(days=900))
    assert len(px.own_history(d, stale)) == 0 and len(px.own_history_table(d, px.own_history(d, None))) == 0


def test_targets_are_compared_with_the_universe_on_their_own_dates():
    d = _sample(n=300)
    dates = pd.date_range("2013-07-01", "2026-07-01", freq="QS")
    rng = np.random.default_rng(0)
    uni = pd.DataFrame([(c, dt) for c in range(5000, 5300) for dt in dates], columns=["cik", "start_date"])
    uni["in_universe"], uni["fcf_pos"], uni["fallen"] = True, rng.random(len(uni)) < 0.5, rng.random(len(uni)) < 0.3
    tgt = pd.DataFrame([(c, dt) for c in d["target_cik"] for dt in dates], columns=["cik", "start_date"])
    tgt["in_universe"], tgt["fcf_pos"], tgt["fallen"] = tgt["cik"] % 4 != 0, True, False     # targets: cash-generative, not fallen
    f = pd.concat([uni, tgt], ignore_index=True)
    t = px.target_signals(d, f).set_index(["group", "signal"])
    assert t.loc[("all buyers", "in_universe"), "targets_share"] == pytest.approx(0.75, abs=0.06)
    assert t.loc[("all buyers", "fcf_pos"), "targets_share"] == 1.0 and t.loc[("all buyers", "fallen"), "targets_share"] == 0.0
    assert 0.5 < t.loc[("all buyers", "fcf_pos"), "universe_share"] < 0.72 and t.loc[("core", "fcf_pos"), "difference"] > 0.25
    assert len(px.target_signals(d, None)) == 0


def test_windows_that_ended_in_a_buyout_are_set_apart_for_the_public_holder():
    d = _sample(n=200)
    feats = pd.DataFrame({"ticker": [f"CIK{c}" for c in d["target_cik"]] + ["CIK9"], "cik": list(d["target_cik"]) + [9]})
    rows = [{"ticker": f"CIK{r.target_cik}", "start_date": r.entry_date - pd.Timedelta(days=400), "horizon_years": 5,
             "end_multiple": 1.6, "end_multiple_lower": 1.6, "touch_3x": False} for r in d.itertuples()]
    rows += [{"ticker": f"CIK{r.target_cik}", "start_date": r.entry_date - pd.Timedelta(days=3000), "horizon_years": 5,
              "end_multiple": 0.7, "end_multiple_lower": 0.5, "touch_3x": False} for r in d.itertuples()]      # ended before the deal
    rows += [{"ticker": "CIK9", "start_date": pd.Timestamp("2015-01-01"), "horizon_years": 5, "end_multiple": 3.5,
              "end_multiple_lower": 3.5, "touch_3x": True}, {"ticker": "CIK9", "start_date": pd.Timestamp("2015-01-01"),
                                                             "horizon_years": 3, "end_multiple": 9.0, "touch_3x": True}]
    out = pd.DataFrame(rows).assign(entry_date=lambda x: x["start_date"], group="x")     # Paper B's table has its own entry date
    t = px.public_holder_outcomes(d, out, feats).set_index("windows")
    assert t.loc["ended in a take-private (all buyers)", "n"] == len(d) and t.loc["ended in a take-private (all buyers)", "mean_multiple"] == pytest.approx(1.6)
    other = t.loc["all other finished windows"]
    assert other["n"] == len(d) + 1 and other["mean_multiple"] == pytest.approx((0.7 * len(d) + 3.5) / (len(d) + 1))
    assert t.loc["ended in a take-private (core)", "n"] == int(d["group"].eq("core").sum())
    pv = px.public_vs_private(pd.DataFrame(rows[-2:]), pd.DataFrame({"vintage_year": [2015, 2016], "funds": [3, 4], "pooled_tvpi": [1.9, 1.7],
                                                             "pooled_dpi": [1.5, 1.1], "median_net_irr": [0.15, 0.12]})).set_index("year")
    assert pv.loc[2015, "public_windows"] == 1 and pv.loc[2015, "public_mean_annual"] == pytest.approx(3.5 ** 0.2 - 1)
    assert pv.loc[2015, "fund_tvpi"] == 1.9 and pv.loc[2016, "funds"] == 4 and len(px.public_vs_private(None, None)) == 0


def test_the_slowdown_is_measured_only_where_the_follow_up_is_complete():
    raw = _deals(n=1500, seed=4)
    done = pd.to_datetime("2019-01-10") + pd.to_timedelta(np.random.default_rng(1).integers(0, 2700, len(raw)), unit="D")
    late = done >= pd.Timestamp("2022-01-01")
    rng = np.random.default_rng(2)
    exits = rng.random(len(raw)) < np.where(late, 0.05, 0.30)
    raw["completion_date"], raw["first_filed"] = done, done - pd.Timedelta(days=90)
    raw["sale_date"] = pd.Series(done + pd.to_timedelta(rng.integers(200, 1000, len(raw)), unit="D")).where(exits)
    raw["reipo_date"] = pd.NaT
    d, _ = par.analysis_sample(raw, BENCH, CFG, as_of=AS_OF)
    t = px.exit_slowdown(d, as_of=AS_OF).set_index("completed")
    assert "2024" not in t.index and {"2019", "2020", "2021", "2022", "2023"} <= set(t.index)     # 2024 lacks three years
    assert t.loc["2019–2021 (all buyers)", "exit_rate"] > 0.22 and t.loc["2022–2023 (all buyers)", "exit_rate"] < 0.1
    assert t.loc["2022–2023 (all buyers)", "p"] < 0.001 and np.isnan(t.loc["2019–2021 (all buyers)", "p"])
    assert t.loc["2019", "n"] + t.loc["2020", "n"] + t.loc["2021", "n"] == t.loc["2019–2021 (all buyers)", "n"]


def test_sponsors_are_also_measured_against_deals_of_their_own_sector_and_everything_runs_together():
    d = _sample()
    s = px.sponsor_styles(d, 0.79)
    assert set(s["sponsor"]) <= {"KKR", "Apollo", "Thoma Bravo"} and (s["deals_with_premium"] >= 5).all()
    assert (s["vs_sector_deals_pct"].abs() < 0.25).all() and (s["sponsor_name_accuracy_in_reading"] == 0.79).all()
    out = px.run_all(d, CFG, bench=BENCH, as_of=AS_OF)
    assert set(out) == {"1 premium by year", "1 benchmark by year", "2 margins", "3 wider coverage", "4 own history",
                        "5 target signals", "6 public holder", "7 exit slowdown", "8 sponsor styles", "9 public vs private",
                        "_own_history_deals"}
    assert all(isinstance(t, pd.DataFrame) for t in out.values())
    empty = px.run_all(d.iloc[0:0], CFG, as_of=AS_OF)                     # nothing to explore is not an error
    assert all(isinstance(t, pd.DataFrame) for t in empty.values())
