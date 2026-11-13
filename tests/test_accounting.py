import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from aem_research import accounting as acc
from sec_fixtures import fsds_zip, num_row, sub_row


def _filing(adsh, cik, form, period, fp, filed, values, prev):
    """A 10-K (four quarters) or 10-Q with the extra tags for the period and the comparative one."""
    q = 4 if fp == "FY" else int(fp[1])
    nums = []
    for P, v in ((period, values), (str(int(period[:4]) - 1) + period[4:], prev)):
        for tag, key in (("NetIncomeLoss", "ni"), ("CostOfRevenue", "cogs"), ("ResearchAndDevelopmentExpense", "rnd")):
            if key in v:
                nums.append(num_row(adsh, tag, P, q, v[key]))
        for tag, key in (("Assets", "assets"), ("StockholdersEquity", "equity"), ("Liabilities", "liab"), ("AssetsCurrent", "ca"),
                         ("LiabilitiesCurrent", "cl")):
            if key in v:
                nums.append(num_row(adsh, tag, P, 0, v[key]))
    nums.append(num_row(adsh, "Revenues", period, q, 999.0))                                 # not an extra tag: left out
    nums.append(num_row(adsh, "NetIncomeLoss", period, q, 5.0, segments="Segment=A;"))       # a segment value: left out
    return [sub_row(adsh, cik, form, period, fp, filed)], nums


@pytest.fixture(scope="module")
def tables():
    now = {"ni": 20.0, "cogs": 60.0, "rnd": 10.0, "assets": 200.0, "equity": 80.0, "liab": 120.0, "ca": 90.0, "cl": 30.0}
    old = {"ni": 10.0, "cogs": 55.0, "rnd": 8.0, "assets": 160.0, "equity": 50.0, "liab": 110.0, "ca": 60.0, "cl": 30.0}
    s1, n1 = _filing("0000000001-15-000001", 1, "10-K", "20141231", "FY", "20150301", now, old)
    s2, n2 = _filing("0000000002-15-000001", 2, "10-K", "20141231", "FY", "20150301",
                     {"ni": -30.0, "assets": 100.0, "equity": -20.0, "liab": 120.0}, {"ni": -10.0, "assets": 120.0, "equity": 5.0, "liab": 115.0})
    with zipfile.ZipFile(io.BytesIO(fsds_zip(s1 + s2, n1 + n2))) as zf:
        return acc.extract_extra(zf)


def test_only_the_extra_tags_are_kept_and_values_are_taken_point_in_time(tables):
    f, s = tables["facts"], tables["subs"]
    assert set(f["tag"].astype(str)) <= set(acc.EXTRA_TAGS) and "Revenues" not in set(f["tag"].astype(str))
    assert (f[f["tag"].astype(str).eq("NetIncomeLoss") & f["cik"].eq(1) & f["ym"].eq(2014 * 12 + 11)]["value"] == 20.0).all()   # no segment row
    snap = acc.snapshots(f, s, [pd.Timestamp("2015-04-01"), pd.Timestamp("2015-01-01")]).set_index(["firm", "start_date"])
    assert (1, pd.Timestamp("2015-01-01")) not in snap.index                      # filed in March: not known in January
    a = snap.loc[(1, pd.Timestamp("2015-04-01"))]
    assert a["net_income"] == 20 and a["net_income_prev"] == 10 and a["assets"] == 200 and a["assets_prev"] == 160 and a["rnd"] == 10
    assert np.isnan(a["gross_profit"]) and a["cogs"] == 60


def test_measures_follow_their_definitions_and_the_score_needs_seven_known_parts(tables):
    snap = acc.snapshots(tables["facts"], tables["subs"], [pd.Timestamp("2015-04-01")])
    w = pd.DataFrame({"firm": [1, 2, 3], "start_date": pd.Timestamp("2015-04-01"), "ticker": ["A", "B", "C"], "market_cap": [400.0, 50.0, 80.0],
                      "ocf": [30.0, -25.0, 5.0], "revenue": [100.0, 40.0, 60.0], "revenue_prev": [90.0, 50.0, 50.0],
                      "share_growth": [0.0, 0.30, 0.0], "fcf": [25.0, -30.0, 4.0], "fallen": [False, True, True]})
    x = acc.measures(w, snap).set_index("firm")
    a = x.loc[1]
    assert a["accruals"] == pytest.approx((20 - 30) / 200) and a["gross_profitability"] == pytest.approx((100 - 60) / 200)
    assert a["book_to_market"] == pytest.approx(0.2) and a["asset_growth"] == pytest.approx(0.25) and a["rnd_to_cap"] == pytest.approx(0.025)
    assert a["return_on_assets"] == pytest.approx(0.1) and a["negative_equity"] == 0 and a["reports_rnd"] == 1
    # nine parts: ROA>0, CFO>0, ROA up, CFO>NI, leverage down, current ratio up, no dilution, margin up, turnover down → 8
    assert a["f_parts_known"] == 9 and a["f_score"] == 8 and a["f_score_high"] == 1 and a["f_score_low"] == 0
    b = x.loc[2]
    assert b["negative_equity"] == 1 and np.isnan(b["book_to_market"]) and b["reports_rnd"] == 0
    # no working capital and no cost line: seven parts known, one of them met (cash flow above net income), scaled to nine
    assert b["f_parts_known"] == 7 and b["f_score"] == 1 and b["f_score_low"] == 1 and b["f_score_high"] == 0
    assert np.isnan(x.loc[3, "accruals"]) and np.isnan(x.loc[3, "f_score"])                           # no statement lines at all
    cov = acc.coverage(x.reset_index())
    assert cov["accruals"] == pytest.approx(2 / 3, abs=0.001) and cov["f_score"] == pytest.approx(2 / 3, abs=0.001)


def test_the_third_batch_runs_through_the_scan_the_combinations_and_the_model():
    rng = np.random.default_rng(5)
    rows = [{"ticker": f"T{i}", "firm": i, "start_date": pd.Timestamp(year=y, month=q, day=1)}
            for y in range(2013, 2022) for q in (1, 4, 7, 10) for i in range(250)]
    w = pd.DataFrame(rows)
    n = len(w)
    w["start_year"] = w["start_date"].dt.year
    w["market_cap"], w["revenue"], w["revenue_prev"] = np.exp(rng.normal(20, 1, n)), np.exp(rng.normal(19, 1, n)), np.exp(rng.normal(19, 1, n))
    w["ocf"], w["fcf"], w["fcf_prev"], w["net_cash"] = rng.normal(0, 5e7, n), rng.normal(0, 5e7, n), rng.normal(0, 5e7, n), rng.normal(0, 1e8, n)
    w["share_growth"], w["revenue_growth"], w["drawdown"] = rng.normal(0.02, 0.05, n), rng.normal(0.05, 0.2, n), -rng.random(n)
    w["fallen"] = w["drawdown"] < -0.4
    snaps = w[["firm", "start_date"]].copy()
    for c in ("net_income", "net_income_prev"):
        snaps[c] = rng.normal(0, 4e7, n)
    for c in ("assets", "assets_prev", "liabilities", "liabilities_prev", "assets_current", "assets_current_prev", "liabilities_current",
              "liabilities_current_prev", "cogs", "cogs_prev", "rnd"):
        snaps[c] = np.exp(rng.normal(18.5, 0.6, n))
    snaps["equity"], snaps["gross_profit"], snaps["gross_profit_prev"], snaps["rnd_prev"] = np.exp(rng.normal(19, 1, n)), np.nan, np.nan, np.nan
    x = acc.measures(w, snaps)
    weak = x["f_score"] <= 3
    x["end5"] = np.exp(rng.normal(0.1, 0.7, n) - 0.8 * weak)                    # a weak score is where the losses are planted
    x["hit3_3y"], x["hit10_5y"] = (rng.random(n) < 0.08).astype(float), (rng.random(n) < 0.012).astype(float)
    from aem_research import scan
    t = scan.scan(x, None, resamples=200, continuous=acc.ACC_CONTINUOUS, flags=acc.ACC_FLAGS)
    held = t[t["verdict"].str.startswith("held") & t["target"].eq("ended 5 years below half")]
    assert "f_score_low" in set(held["candidate"]) and held.loc[held["candidate"].eq("f_score_low"), "holdout_lr"].iloc[0] > 1.5
    a = scan.asymmetry(x, None, resamples=200, combos=acc.ACC_COMBOS, build=acc.with_combos)
    assert a["combination"].nunique() == len(acc.ACC_COMBOS)
    low = a[a["combination"].eq("fallen with an F-score of 0 to 3")]
    assert (low["below_half"] > low["below_half_rest"]).all() and low["verdict"].iloc[0] in ("worse", "variance")
    m = scan.outcome_model(x, None, extra_continuous=acc.ACC_CONTINUOUS, extra_flags=acc.ACC_FLAGS, build=acc.with_combos)
    m = m.set_index(["sample", "group"])
    assert m.loc[("from 2018, five years", "worst-ranked tenth"), "below_half"] > m.loc[("from 2018, five years", "best-ranked tenth"), "below_half"]
