import numpy as np
import pandas as pd
import pytest

from aem_research.config import PreregistrationError, load_config, lock_config
from aem_research.outcomes import (CENSORED, COMPLETE, DELISTED, ENDED, NO_ENTRY, OutcomeSpec,
                                   _compute_outcomes, base_rates, build_price_panel, build_starts,
                                   compute_outcomes, days_col, quarter_end_panel, touch_col,
                                   touch_sampling_ratios)
from conftest import bseries

SPEC = OutcomeSpec(horizons_years=(3,), touch_multiples=(3.0, 4.0, 6.24, 10.0))
START = pd.DataFrame({"ticker": ["A"], "start_date": [pd.Timestamp("2015-01-01")]})


def run(prices, starts=START, spec=SPEC, **kw):
    return _compute_outcomes(spec, prices, starts, **kw)


def one(df, h=3):
    return df[df["horizon_years"] == h].iloc[0]


def test_flat_price():
    r = one(run(bseries("A", "2015-01-01", [10.0] * 1200)))
    assert r["status"] == COMPLETE
    assert r["end_multiple"] == pytest.approx(1.0)
    assert r[touch_col(3)] == 0.0 and np.isnan(r[days_col(3)])
    assert r["bucket"] == 1.0


def test_ramp_touches_and_end_multiple():
    vals = list(np.linspace(10, 40, 500)) + [40.0] * 700   # 4x within ~2 years, then flat
    r = one(run(bseries("A", "2015-01-01", vals)))
    assert r[touch_col(3)] == 1.0 and r[touch_col(4)] == 1.0
    assert r[touch_col(6.24)] == 0.0 and r[touch_col(10)] == 0.0
    assert r["end_multiple"] == pytest.approx(4.0)
    assert r["bucket"] == 5.0                                # 4.0 is in [3.54, 7.07)
    assert 0 < r[days_col(3)] < r[days_col(4)] <= 3 * 366


def test_spike_then_crash_touch_vs_end():
    vals = [10.0] * 100 + [35.0] * 5 + [5.0] * 1000
    r = one(run(bseries("A", "2015-01-01", vals)))
    assert r[touch_col(3)] == 1.0
    assert r["end_multiple"] == pytest.approx(0.5)
    assert r["max_multiple"] == pytest.approx(3.5)


def test_known_delisting_return():
    px = bseries("A", "2015-01-01", [10.0] * 300 + [8.0])
    dl = pd.DataFrame({"ticker": ["A"], "delist_date": [px["date"].iloc[-1] + pd.Timedelta(days=3)],
                       "category": ["performance"], "delist_return": [-0.5]})
    r = one(run(px, delistings=dl, as_of=pd.Timestamp("2026-06-30")))
    assert r["status"] == DELISTED
    assert r["end_multiple"] == pytest.approx(8.0 * 0.5 / 10.0)
    assert r["end_multiple_lower"] == r["end_multiple_upper"] == pytest.approx(0.4)
    assert bool(r["delist_return_observed"])


def test_imputed_delisting_by_category_with_bounds():
    px = bseries("A", "2015-01-01", [10.0] * 300 + [12.0])
    dl = pd.DataFrame({"ticker": ["A"], "delist_date": [px["date"].iloc[-1]], "category": ["merger"]})
    r = one(run(px, delistings=dl, as_of=pd.Timestamp("2026-06-30")))
    assert r["status"] == DELISTED and r["delist_category"] == "merger"
    assert r["end_multiple"] == pytest.approx(1.2)                # merger return 0.0
    assert r["end_multiple_lower"] == 0.0 and r["end_multiple_upper"] == pytest.approx(1.2)


def test_series_ended_without_record_uses_unknown():
    px = bseries("A", "2015-01-01", [10.0] * 300)
    r = one(run(px, as_of=pd.Timestamp("2026-06-30")))
    assert r["status"] == ENDED
    assert r["end_multiple"] == pytest.approx(0.7)                # unknown: -30%
    assert r["end_multiple_lower"] == 0.0 and r["end_multiple_upper"] == pytest.approx(1.0)


def test_censored_window():
    px = bseries("A", "2015-01-01", [10.0] * 400)                 # data ends early 2016
    r = one(run(px))                                              # as_of = last date
    assert r["status"] == CENSORED and np.isnan(r["end_multiple"]) and np.isnan(r[touch_col(3)])


def test_no_entry_and_entry_tolerance():
    px = bseries("A", "2015-01-12", [10.0] * 1200)
    r = one(run(px))
    assert r["status"] == NO_ENTRY                                # first price 11 days after start
    starts = pd.DataFrame({"ticker": ["A"], "start_date": [pd.Timestamp("2015-01-10")]})  # Saturday
    r2 = one(run(px, starts=starts))
    assert r2["status"] == COMPLETE and r2["entry_date"] == pd.Timestamp("2015-01-12")


def test_unknown_ticker_is_no_entry():
    starts = pd.DataFrame({"ticker": ["ZZZ"], "start_date": [pd.Timestamp("2015-01-01")]})
    r = one(run(bseries("A", "2015-01-01", [10.0] * 1200), starts=starts))
    assert r["status"] == NO_ENTRY


def test_bucket_edges_are_left_closed():
    spec = OutcomeSpec()
    assert spec.bucket(0.1999) == 0.1 and spec.bucket(0.2) == 0.4
    assert spec.bucket(1.0) == 1.0 and spec.bucket(3.54) == 5.0
    assert spec.bucket(7.07) == 10.0 and spec.bucket(250.0) == 10.0
    assert np.isnan(spec.bucket(float("nan")))


def test_leap_day_window_end():
    px = bseries("A", "2012-02-29", [10.0] * 1000)
    starts = pd.DataFrame({"ticker": ["A"], "start_date": [pd.Timestamp("2012-02-29")]})
    r = one(run(px, starts=starts))
    assert r["window_end"] == pd.Timestamp("2015-02-28")


def test_mixed_datetime_units_are_handled():
    px = bseries("A", "2015-01-01", [10.0] * 1200)
    px["date"] = px["date"].astype("datetime64[s]")
    starts = START.assign(start_date=START["start_date"].astype("datetime64[s]"))
    assert one(run(px, starts=starts))["status"] == COMPLETE


def test_unsorted_categorical_input_with_duplicates_and_bad_prices():
    px = bseries("A", "2015-01-01", [10.0] * 1200)
    dup = px.iloc[[0]].assign(price=20.0)                        # same date, later row wins
    bad = px.iloc[[5]].assign(price=-1.0)
    mixed = pd.concat([px.iloc[::-1], dup, bad], ignore_index=True)
    mixed["ticker"] = mixed["ticker"].astype("category")
    r = one(run(mixed))
    assert r["status"] == COMPLETE and r["entry_price"] == pytest.approx(20.0)
    assert r["end_multiple"] == pytest.approx(0.5)


def test_symbol_reuse_gap_never_fabricates_a_jump():
    old = bseries("A", "2015-01-01", [1.0] * 250)                 # old company, ends ~Dec 2015
    new = bseries("A", "2018-01-01", [100.0] * 600)               # new company reuses the symbol
    r = one(run(pd.concat([old, new])))
    assert r["status"] == ENDED and r[touch_col(10)] == 0.0
    assert r["max_multiple"] == pytest.approx(1.0)
    starts = pd.DataFrame({"ticker": ["A"], "start_date": [pd.Timestamp("2018-01-01")]})
    r2 = one(run(pd.concat([old, new]), starts=starts, as_of=pd.Timestamp("2021-06-30")))
    assert r2["entry_price"] == pytest.approx(100.0) and r2["status"] in (COMPLETE, ENDED)


def test_base_rates_exclude_censored():
    a = bseries("A", "2015-01-01", list(np.linspace(10, 40, 600)) + [40.0] * 600)
    b = bseries("B", "2015-01-01", [10.0] * 1200)
    starts = pd.DataFrame({"ticker": ["A", "B", "A"],
                           "start_date": pd.to_datetime(["2015-01-01", "2015-01-01", "2019-01-01"])})
    out = run(pd.concat([a, b]), starts=starts)
    br = base_rates(out, multiples=(3.0,))
    row = br[br["horizon_years"] == 3].iloc[0]
    assert row["n_windows"] == 2 and row[touch_col(3)] == pytest.approx(0.5)


def test_compute_outcomes_requires_lock(final_config):
    px = bseries("A", "2015-01-01", [10.0] * 1200)
    with pytest.raises(PreregistrationError):
        compute_outcomes(load_config(final_config), px, START)
    lock_config(final_config)
    out = compute_outcomes(load_config(final_config), px, START)
    assert set(out["horizon_years"]) == {3, 5, 10}


def test_spec_from_config_matches_shipped_values(final_config):
    spec = OutcomeSpec.from_config(load_config(final_config).data)
    assert spec.touch_multiples == (2.0, 3.0, 4.0, 6.24, 10.0, 100.0)
    assert spec.delist_return("performance_nasdaq") == -0.55
    assert spec.delist_return("no-such-category") == -0.30


def test_build_price_panel_prefers_simfin_and_fills_gaps():
    dates = pd.bdate_range("2015-01-01", periods=3)
    sim = pd.DataFrame({"ticker": ["A"] * 3, "date": dates, "close": [10.0, 11, 12],
                        "adj_close": [9.0, np.nan, 11.0]})
    stq = pd.DataFrame({"ticker": ["A"] * 3 + ["B"] * 3, "date": list(dates) * 2,
                        "close": [1.0, 1, 1, 5.0, 6, 7]})
    panel = build_price_panel(sim, stq)
    a = panel[panel["ticker"] == "A"]
    assert list(a["source"].unique()) == ["simfin"]
    assert a["price"].tolist() == pytest.approx([9.0, 11.0, 11.0])   # adj_close, gap filled by close
    assert panel[panel["ticker"] == "B"]["source"].unique().tolist() == ["stooq"]


def test_build_starts_min_price():
    panel = pd.concat([bseries("A", "2015-01-02", [0.5] * 5), bseries("B", "2015-01-02", [5.0] * 5)])
    st = build_starts(panel, (2015, 2015), min_price=1.0)
    assert st["ticker"].tolist() == ["B"] and st["start_date"].iloc[0] == pd.Timestamp("2015-01-01")


def test_quarter_end_sampling_dates_prices_for_the_next_start():
    p = bseries("A", "2014-12-01", np.linspace(10, 20, 150))
    q = quarter_end_panel(p)
    assert list(q["date"].dt.strftime("%Y-%m-%d")) == ["2015-01-01", "2015-04-01", "2015-07-01"]
    assert q["price"].iloc[0] == pytest.approx(p[p["date"] <= "2014-12-31"]["price"].iloc[-1])


def test_quarter_end_touches_miss_spikes_and_stay_outside_the_sample():
    vals = [10.0] * 50 + [35.0] * 5 + [10.0] * 1000                  # from Dec 2011; a one-week spike to 3.5x in Feb 2012
    p = bseries("A", "2011-12-01", vals)
    p["source"] = "simfin"
    starts = pd.DataFrame({"ticker": ["A"], "start_date": [pd.Timestamp("2012-01-01")]})
    r = touch_sampling_ratios(SPEC, p, starts, sample_start="2013-01-01", horizons=(3,), multiples=(3.0,))
    assert r.iloc[0]["quarter_end_over_daily"] == 0.0 and r.iloc[0]["windows"] == 1
    with pytest.raises(ValueError, match="before the sample"):
        touch_sampling_ratios(SPEC, p, starts.assign(start_date=pd.Timestamp("2013-01-01")), sample_start="2013-01-01")


def test_h1_tables_pooled_by_regime_and_start_year():
    from aem_research.outcomes import h1_tables
    rows = []
    # four 5-year windows and four 3-year windows: two firms (one with two series), two regimes
    for t, s, h, st, t3, t10, em, b in [
            ("CIK1", "2015-01-02", 5, "complete", 1.0, 0.0, 3.0, 2.5),
            ("CIK1~2", "2016-01-04", 5, "delisted", 0.0, 0.0, 0.5, 0.4),
            ("CIK2", "2015-01-02", 5, "complete", 1.0, 1.0, 12.0, 10.0),
            ("CIK2", "2016-01-04", 5, "censored", np.nan, np.nan, np.nan, np.nan),
            ("CIK1", "2015-01-02", 3, "complete", 1.0, 0.0, 2.0, 1.0),
            ("CIK2", "2022-01-03", 3, "series_ended", 0.0, 0.0, 0.7, 1.0),
            ("CIK2", "2023-01-02", 3, "complete", 0.0, 0.0, 1.1, 1.0),
            ("CIK2", "2023-04-03", 3, "no_entry", np.nan, np.nan, np.nan, np.nan)]:
        rows.append({"ticker": t, "start_date": pd.Timestamp(s), "horizon_years": h, "status": st,
                     "touch_3x": t3, "touch_10x": t10, "end_multiple": em, "end_multiple_lower": em * 0.5,
                     "end_multiple_upper": em, "bucket": b})
    out = h1_tables(pd.DataFrame(rows))
    t, bk = out["touch"], out["buckets"]
    pooled = t[t["scope"] == "pooled"].set_index("horizon_years")
    assert pooled.loc[5, "windows"] == 3 and pooled.loc[5, "firms"] == 2           # censored left out, ~2 is CIK1
    assert pooled.loc[5, "touch_3x"] == pytest.approx(2 / 3, abs=1e-4) and pooled.loc[5, "touch_10x"] == pytest.approx(1 / 3, abs=1e-4)
    assert pooled.loc[3, "windows"] == 3 and pooled.loc[3, "touch_3x"] == pytest.approx(1 / 3, abs=1e-4)
    reg = t[(t["scope"] == "regime") & (t["horizon_years"] == 3)].set_index("group")
    assert reg.loc["before 2022", "windows"] == 1 and reg.loc["2022 and later", "windows"] == 2
    years = t[t["scope"] == "start year"]
    assert set(years["group"]) == {"2015", "2016", "2022", "2023"}
    b = bk[bk["scope"] == "pooled"].iloc[0]
    assert b["windows"] == 3 and b["share_2.5x"] == pytest.approx(1 / 3, abs=1e-4) and b["share_0.4x"] == pytest.approx(1 / 3, abs=1e-4)
    assert b["share_10x"] == pytest.approx(1 / 3, abs=1e-4) and b["share_0.1x"] == 0.0
    assert b["mean"] == pytest.approx((3.0 + 0.5 + 12.0) / 3, abs=1e-4) and b["mean_lower"] == pytest.approx(b["mean"] / 2, abs=1e-4)
