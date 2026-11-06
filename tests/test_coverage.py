import pandas as pd
import pytest

from aem_research import coverage as cv

AS_OF = pd.Timestamp("2026-10-30")


def sp(rows):
    return pd.DataFrame(rows, columns=["ticker", "first_date", "last_date", "n_obs"]).assign(
        first_date=lambda d: pd.to_datetime(d["first_date"]), last_date=lambda d: pd.to_datetime(d["last_date"]))


SIM = sp([("AAA", "2005-01-03", "2026-10-30", 5000), ("BBB", "2005-01-03", "2014-03-14", 2300),
          ("CCC", "2005-01-03", "2026-10-30", 5000), ("MOV", "2005-01-03", "2026-10-30", 5000)])
STQ = sp([("AAA", "2005-01-03", "2026-10-30", 5000), ("DDD", "2006-01-03", "2016-05-20", 2600),
          ("BBB", "2005-01-03", "2014-03-12", 2300)])


def test_spans_and_summary():
    prices = pd.DataFrame({"ticker": ["A", "A", "B"],
                           "date": pd.to_datetime(["2010-01-04", "2012-01-04", "2011-01-03"])})
    s = cv.spans(prices)
    assert s.set_index("ticker").loc["A", "n_obs"] == 2
    summ = cv.source_summary("x", SIM, AS_OF)
    assert summ["tickers"] == 4 and summ["stopped_trading"] == 1
    assert cv.stopped_by_year(SIM, AS_OF, (2013, 2015)).tolist() == [0, 1, 0]


def test_overlaps():
    ov = cv.stopped_overlap(SIM, STQ, AS_OF)
    assert ov == {"simfin_stopped": 1, "stooq_stopped": 2, "matched_both": 1, "stooq_only": 1, "simfin_only": 0}
    live = cv.live_overlap(SIM, STQ, "2015-06-01")
    assert live["stooq_live"] == 2 and live["stooq_live_in_simfin"] == 1


def test_form25_events_filters_and_dedups():
    f = pd.DataFrame({
        "cik": [1, 1, 2, 3, 4], "company": ["B Corp", "B Corp", "Gores Acquisition Corp",
                                            "The NASDAQ Stock Market LLC", "D Inc"],
        "form": ["25-NSE", "25-NSE", "25-NSE", "25-NSE", "10-K"],
        "date_filed": pd.to_datetime(["2014-03-10", "2014-06-01", "2015-01-05", "2015-01-05", "2015-01-05"])})
    ev = cv.form25_events(f, ["25-NSE"], (2009, 2025))
    assert ev["cik"].tolist() == [1, 2] and ev["spac"].tolist() == [False, True]


def test_benchmark_classifies_events():
    companies = pd.DataFrame({"ticker": ["BBB", "DDD", "MOV", "EEE"], "cik": [10, 20, 30, 40]})
    events = pd.DataFrame({
        "cik": [10, 20, 30, 40, 50], "company": ["B", "D", "Mover", "E", "Unknown"],
        "date_filed": pd.to_datetime(["2014-03-10", "2016-05-25", "2015-02-02", "2018-01-10", "2014-07-01"]),
        "year": [2014, 2016, 2015, 2018, 2014], "spac": [False] * 5})
    detail, table = cv.benchmark_against_form25(events, companies, SIM, STQ, window_days=45)
    status = dict(zip(detail["cik"].astype(int), detail["status"]))
    assert status == {10: "simfin", 20: "stooq", 30: "continued", 40: "missing", 50: "unmapped"}
    t = table.set_index("year")
    assert t.loc[2014, "events"] == 2 and t.loc[2014, "coverage_union"] == pytest.approx(0.5)
    assert t.loc[2014, "coverage_if_mapped"] == pytest.approx(1.0)
    assert t.loc[2015, "events"] == 0 and t.loc[2015, "continued_excluded"] == 1


def test_benchmark_needs_cik_bridge():
    assert cv.benchmark_against_form25(pd.DataFrame(), pd.DataFrame({"ticker": ["A"]}), SIM, STQ) == (None, None)


def test_split_simfin_tickers_match_stooq_by_base_symbol():
    sim = sp([("BBB~77", "2005-01-03", "2014-03-14", 2300), ("AAA", "2005-01-03", "2026-10-30", 5000)])
    assert cv.base_symbol("BBB~77") == "BBB"
    assert cv.stopped_overlap(sim, STQ, AS_OF)["matched_both"] == 1
    assert set(cv.stooq_only(sim, STQ)["ticker"]) == {"DDD"}


def test_coverage_range_credits_stooq_only_names():
    table = pd.DataFrame({"year": [2014, 2015], "events": [10, 10], "simfin": [6, 7],
                          "stooq_adds": [0, 1], "unmapped": [4, 2]})
    lo, hi = cv.coverage_range(table, pd.Series({2014: 3, 2015: 5}))
    assert lo == pytest.approx(14 / 20) and hi == pytest.approx((14 + 3 + 2) / 20)


def test_count_ratio():
    events = pd.DataFrame({"year": [2014, 2014, 2015], "spac": [False, False, True]})
    out = cv.count_ratio(events, pd.Series({2014: 1, 2015: 3}))
    assert out.to_dict("records") == [{"year": 2014, "form25_issuers": 2, "stopped_in_data": 1, "ratio": 0.5}]


def test_attrition():
    out = cv.attrition(SIM, [2011, 2024], 5, AS_OF)
    assert out.to_dict("records") == [{"start_year": 2011, "live": 4, "stopped_within_h": 1,
                                       "attrition_observed": 0.25}]


def test_missing_share_math():
    a, m = cv.missing_share(100, 20, 0.8)
    assert a == pytest.approx(25 / 105) and m == pytest.approx(25 * 0.2 / 105)
    assert cv.missing_share(100, 20, 1.0)[1] == 0.0
    assert all(x != x for x in cv.missing_share(100, 20, 0.0))


def test_adjustment_check_detects_basis():
    dates = pd.bdate_range("2015-01-02", periods=4)
    sim = pd.DataFrame({"ticker": pd.Categorical(["A"] * 4), "date": dates,
                        "close": [10.0, 10, 10, 10], "adj_close": [5.0, 5, 5, 5]})
    stq = pd.DataFrame({"ticker": pd.Categorical(["A"] * 4 + ["Z"] * 4), "date": list(dates) * 2,
                        "close": [5.0, 5, 5, 5, 1, 1, 1, 1]})
    out = cv.adjustment_check(sim, stq)
    assert out["pairs"] == 4
    assert out["within_1pct_of_simfin_adj_close"] == 1.0 and out["within_1pct_of_simfin_close"] == 0.0
    assert cv.adjustment_check(None, stq) == {}


def test_verdict_thresholds():
    th = {"not_needed_max_missing_share": 0.05, "optional_max_missing_share": 0.15}
    assert cv.eodhd_verdict(0.04, 0.9, th, "t")["verdict"] == "not needed"
    assert cv.eodhd_verdict(0.10, 0.7, th, "t")["verdict"] == "optional"
    v = cv.eodhd_verdict(0.20, 0.5, th, "t")
    assert v["verdict"] == "recommended" and "2.40%–3.00%" in v["reason"]
    assert cv.eodhd_verdict(None, None, th, "t")["verdict"] == "undetermined"
    assert cv.eodhd_verdict(None, 0.0, th, "t")["verdict"] == "recommended"
    assert cv.eodhd_verdict(None, 0.6, th, "t")["verdict"] == "undetermined"
