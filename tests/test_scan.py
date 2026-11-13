import numpy as np
import pandas as pd
import pytest

from aem_research import scan


@pytest.fixture(scope="module")
def w():
    rng = np.random.default_rng(0)
    rows = [{"ticker": f"T{i}", "cik": i, "start_date": pd.Timestamp(year=y, month=q, day=1)}
            for y in range(2013, 2022) for q in (1, 4, 7, 10) for i in range(300)]
    f = pd.DataFrame(rows)
    n = len(f)
    f["market_cap"], f["entry_price_unadjusted"] = np.exp(rng.normal(20, 1, n)), np.exp(rng.normal(2.5, 1, n))
    f["drawdown"], f["drawdown_median"] = -rng.random(n) * 0.9, -0.3
    for c in ("fcf", "fcf_prev", "ocf", "ocf_prev", "net_cash"):
        f[c] = rng.normal(0, 5e7, n)
    f["revenue"], f["cash_total"], f["debt"] = np.exp(rng.normal(19, 1, n)), np.exp(rng.normal(17, 1, n)), np.exp(rng.normal(17, 1, n))
    f["capex"], f["revenue_growth"], f["share_growth"] = np.exp(rng.normal(15, 1, n)), rng.normal(0.08, 0.3, n), rng.normal(0.03, 0.08, n)
    f["breadth"], f["turnover_126d"], f["insider_n"] = rng.integers(3, 300, n), rng.random(n), rng.integers(0, 5, n).astype(float)
    f["sic"] = np.where(rng.random(n) < 0.15, 2834, 7372)
    for c in ("sale_any", "insider_routine_only", "fcf_pos", "net_cash_pos", "low_dilution", "s1_proxy", "insider_any", "insider_cluster",
              "insider_opportunistic", "insider_post_results_cluster", "low_attention"):
        f[c] = rng.random(n) < 0.3
    f["fallen"] = f["drawdown"] <= -0.4
    f["firm"], f["start_year"] = f["cik"], f["start_date"].dt.year
    f["hit3_3y"] = (rng.random(n) < 0.05 + 0.10 * (f["revenue_growth"] > 0.3)).astype(float)       # one planted signal
    f["hit10_5y"] = (rng.random(n) < 0.012).astype(float)                                          # and a target with none
    f["end5"] = np.exp(rng.normal(0.1, 0.8, n) - 0.9 * (f["fcf"] < -6e7))                         # losses planted among heavy cash-burners
    return f


def test_the_scan_finds_the_planted_signal_on_both_periods_and_no_other(w):
    t = scan.scan(w, None, resamples=300)
    assert set(t["target"]) == {"3x within 3 years", "10x within 5 years", "ended 5 years below half"}
    assert t["discovery_p"].notna().sum() > 90
    held = t[t["verdict"].str.startswith("held")]
    assert ("revenue_growth", "highest fifth") in set(zip(held["candidate"], held["side"]))
    gains = held[held["target"].eq("3x within 3 years")]
    assert set(gains["candidate"]) <= {"revenue_growth", "profitable_growth", "fallen_and_growing"}  # the signal and flags built on it
    assert not len(held[held["target"].eq("10x within 5 years")])                                # noise does not survive the two periods
    losses = held[held["target"].eq("ended 5 years below half")]                                 # the mirror target finds the planted loss side
    assert ("fcf_yield", "lowest fifth") in set(zip(losses["candidate"], losses["side"]))
    assert losses.loc[losses["candidate"].eq("fcf_yield") & losses["side"].eq("lowest fifth"), "holdout_lr"].iloc[0] > 1.3
    assert (t["discovery_q"].dropna() >= t["discovery_p"].dropna() - 1e-12).all()
    d = scan.digest(t)
    assert d["tested"] == t["discovery_p"].notna().sum() and len(d["held"]) == len(held) and len(d["strongest"]) == 36
    assert scan.digest(None) == {"tested": 0, "held": [], "strongest": []}


def test_candidates_use_only_start_date_values_and_momentum_comes_from_the_panel(w):
    panel = pd.DataFrame({"ticker": ["T0"] * 3, "date": pd.to_datetime(["2014-01-01", "2014-07-01", "2015-01-01"]), "price": [10.0, 20.0, 30.0]})
    x = scan.candidates(w[(w["ticker"] == "T0") & (w["start_date"] == "2015-01-01")], panel)
    assert x["momentum_6m"].iloc[0] == pytest.approx(0.5) and x["momentum_12m"].iloc[0] == pytest.approx(2.0)
    y = scan.candidates(w[(w["ticker"] == "T1") & (w["start_date"] == "2015-01-01")], panel)
    assert y["momentum_12m"].isna().all()
    row = w.iloc[[0]].assign(fcf=5.0, fcf_prev=-1.0, revenue_growth=0.5, cash_total=10.0, share_growth=-0.05)
    c = scan.candidates(row, None).iloc[0]
    assert c["turned_cash_positive"] == 1 and c["profitable_growth"] == 1 and c["burner_short_runway"] == 0 and c["shrinking_share_count"] == 1
    c = scan.candidates(row.assign(fcf=-8.0), None).iloc[0]
    assert c["burner_short_runway"] == 1 and c["profitable_growth"] == 0          # cash for 1.25 years
    assert np.isnan(scan.candidates(row.assign(fcf=np.nan), None).iloc[0]["turned_cash_positive"])


def test_the_model_bound_scores_out_of_time_and_survives_an_empty_measure(w):
    m = scan.model_bound(w, None).set_index(["target", "sample"])
    hold = m.loc[("3x within 3 years", "holdout (start dates from 2018)")]
    assert 0.55 < hold["auc"] < 0.8 and hold["top_decile_lift"] > 1.3            # the planted signal is found
    none = m.loc[("10x within 5 years", "holdout (start dates from 2018)")]
    assert 0.4 < none["auc"] < 0.6                                                # and noise scores as noise
    assert m.loc[("3x within 3 years", "discovery (fitted here, flattering)"), "auc"] >= hold["auc"]
    assert m.loc[("ended 5 years below half", "holdout (start dates from 2018)"), "auc"] > 0.55


def test_combinations_are_judged_on_end_multiples_and_only_stable_shapes_get_a_verdict(w):
    t = scan.asymmetry(w, None, resamples=300)
    assert set(t["sample"]) == {"to 2017, five years", "from 2018, five years"} and t["combination"].nunique() >= 15
    burn = t[t["combination"].str.startswith("avoid list")]
    assert (burn["below_half"] > burn["below_half_rest"]).all() and set(burn["verdict"]) <= {"worse", "variance"}   # losses were planted there
    noise = t[t["combination"].eq("positive free cash flow")]                       # a coin flip in this fixture
    assert noise["verdict"].iloc[0] in ("no difference", "not stable") and (noise["below_half"] - noise["below_half_rest"]).abs().max() < 0.03
    known = t[t["shape"].notna()]
    assert set(known["shape"]) <= {"asymmetric", "defensive", "defensive, costs upside", "variance", "worse", "more upside, same losses",
                                   "no difference"}
    assert (t.groupby("combination")["verdict"].nunique() == 1).all()
    x = scan.with_combos(w.iloc[:50].assign(net_cash=4e8, market_cap=1e9, fcf=-5e7))
    assert x["c_floor"].eq(1).all() and x["c_floor_net"].eq(0).all()               # 40% of the price, 15% after five years of burn


def test_the_outcome_model_ranks_out_of_time(w):
    m = scan.outcome_model(w, None).set_index(["sample", "group"])
    hold = "from 2018, five years"
    assert m.loc[(hold, "worst-ranked tenth"), "below_half"] > m.loc[(hold, "all windows"), "below_half"]      # the planted loss side
    assert m.loc[(hold, "best-ranked tenth"), "windows"] < m.loc[(hold, "all windows"), "windows"]
    assert m.loc[(hold, "worst-ranked tenth"), "median_end"] < m.loc[(hold, "best-ranked tenth"), "median_end"]
