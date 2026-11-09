import numpy as np
import pandas as pd
import pytest

from aem_research import secprices as sp
from aem_research import splits

PERIODS = pd.date_range("2013-03-31", "2014-12-31", freq="QE")
FILERS = range(900001, 900011)


def _company(cusip_of, price_of, shares_of, filers=FILERS, n_filers=20):
    rows, hold = [], []
    for k, p in enumerate(PERIODS):
        c = cusip_of(k)
        rows.append((c, p, price_of(k), 12, n_filers))
        for j, f in enumerate(filers(k) if callable(filers) else filers):
            hold.append((c, p, f, float(shares_of(k, j))))
    return rows, hold


def _build(companies, ftd_px=None, ftd_names=None):
    rows, hold, br = [], [], []
    for cik, (r, h, cus) in companies.items():
        rows += r
        hold += h
        br += [(c, cik, "isin") for c in cus]
    f13 = pd.DataFrame(rows, columns=["cusip", "period", "price_13f", "n_price_filers", "n_filers"])
    holders = pd.DataFrame(hold, columns=["cusip", "period", "filer_cik", "shares"])
    bridge = pd.DataFrame(br, columns=["cusip", "cik", "method"]).drop_duplicates("cusip")
    mains = sp.main_cusips(f13, bridge)
    return sp.build_panel(mains, holders, ftd_px, bridge, as_of_period=PERIODS[-1], ftd_names=ftd_names)


def test_standard_ratios_snap_and_reject_odd_moves():
    r, d = splits.snap([2.01, 0.099, 1.49, 7.3])
    assert list(r[:3]) == [2.0, 0.1, 1.5] and d[3] > np.log(1.03)          # 7.3 is no ratio companies use
    row = {"matched": 8, "median_ratio": 2.0, "ratio": 2.0, "agree": 0.9}
    assert splits.holder_split(row) == 2.0
    assert splits.holder_split({**row, "agree": 0.3}) is None            # holders moved by different amounts
    assert splits.holder_split({**row, "matched": 2}) is None
    assert splits.holder_split({**row, "median_ratio": 1.1, "ratio": 1.1}) is None   # below 1.2: a stock dividend
    assert splits.price_switch(0.5, 5.0) == ("split", 0.1)
    assert splits.price_switch(10.0, 10.5)[0] == "same" and splits.price_switch(10.0, 3.7)[0] == "break"


def test_forward_and_reverse_splits_are_found_and_prices_restated():
    fwd = _company(lambda k: "AAA111111", lambda k: (10 + k) if k < 4 else (10 + k) / 2,
                   lambda k, j: (1000 + 10 * j) * (2 if k >= 4 else 1))
    rev = _company(lambda k: "BBB222222" if k < 5 else "CCC333333",
                   lambda k: (1.0 - 0.05 * k) * (10 if k >= 5 else 1),
                   lambda k, j: (5000 + 7 * j) * (0.1 if k >= 5 else 1))
    panel, ev, info = _build({1: (*fwd, ["AAA111111"]), 2: (*rev, ["BBB222222", "CCC333333"])})
    assert info["splits_holders"] == 2 and info["new_series"] == 0
    assert set(zip(ev["cik"], ev["ratio"])) == {(1, 2.0), (2, 0.1)}
    a = panel[panel["ticker"] == "CIK1"]
    assert a["price"].round(3).tolist() == [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]   # no fake halving
    assert a["close_raw"].iloc[0] == 10.0 and (a["date"] == a["period"] + pd.Timedelta(days=1)).all()
    b = panel[panel["ticker"] == "CIK2"]["price"].round(3).tolist()
    assert b == [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5]                     # no fake tenfold jump


def test_a_cusip_change_without_evidence_starts_a_new_series():
    other = lambda k: [f + (100 if k >= 3 else 0) for f in FILERS]      # all-new holders after the change
    reorg = _company(lambda k: "DDD444444" if k < 3 else "EEE555555", lambda k: 20.0 + k,
                     lambda k, j: 3000 + j, filers=other)
    panel, ev, info = _build({3: (*reorg, ["DDD444444", "EEE555555"])})
    assert info["new_series"] == 1 and set(panel["ticker"]) == {"CIK3", "CIK3~2"}
    assert ev.iloc[0]["evidence"] == "CUSIP change"


def test_fails_prices_fill_the_lookback_and_give_the_final_price():
    # 13F covers 2013-03-31 to 2014-06-30; the fails files price 2012 quarter-ends and a last day in Aug 2014
    rows, hold = _company(lambda k: "AAA111111", lambda k: 10.0 + k, lambda k, j: 1000 + j)
    keep = [i for i, r in enumerate(rows) if r[1] <= pd.Timestamp("2014-06-30")]
    rows = [rows[i] for i in keep]
    hold = [h for h in hold if h[1] <= pd.Timestamp("2014-06-30")]
    px = pd.DataFrame({"cusip": ["AAA111111"] * 3,
                       "first_date": pd.to_datetime(["2012-09-28", "2012-12-28", "2014-08-14"]),
                       "first_price": [7.0, 8.0, 4.0],
                       "last_date": pd.to_datetime(["2012-09-28", "2012-12-28", "2014-08-20"]),
                       "last_price": [7.0, 8.0, 3.5],
                       "max_date": pd.NaT, "max_price": np.nan, "min_date": pd.NaT, "min_price": np.nan, "n_obs": 1})
    names = pd.DataFrame({"cusip": ["AAA111111"], "symbol": ["AAA"], "description": ["A CORP"],
                          "first_date": pd.Timestamp("2012-01-01"), "last_date": pd.Timestamp("2014-08-20"), "n": 9})
    panel, _ev, info = _build({1: (rows, hold, ["AAA111111"])}, ftd_px=px, ftd_names=names)
    p = panel.set_index("date")
    assert info["quarters_fails"] == 2 and p.loc[pd.Timestamp("2013-01-01"), "price_source"] == "fails"
    assert p.loc[pd.Timestamp("2014-08-20"), "price_source"] == "fails final" and p.loc[pd.Timestamp("2014-08-20"), "price"] == 3.5


def test_main_cusip_does_not_flip_between_classes():
    f13 = pd.DataFrame({"cusip": ["A", "B", "A", "B", "A", "B"],
                        "period": pd.to_datetime(["2015-03-31"] * 2 + ["2015-06-30"] * 2 + ["2015-09-30"] * 2),
                        "price_13f": 10.0, "n_price_filers": 10, "n_filers": [50, 40, 45, 52, 44, 60]})
    bridge = pd.DataFrame({"cusip": ["A", "B"], "cik": [7, 7], "method": "issuer cusip"})
    assert sp.main_cusips(f13, bridge)["cusip"].tolist() == ["A", "A", "A"]


def test_as_of_skips_a_quarter_still_being_filed():
    rows = [(f"C{i:08d}", p, 10.0, 5, 5) for p in pd.date_range("2024-03-31", "2025-06-30", freq="QE")
            for i in range(100 if p < pd.Timestamp("2025-06-30") else 40)]
    f13 = pd.DataFrame(rows, columns=["cusip", "period", "price_13f", "n_price_filers", "n_filers"])
    assert sp.complete_through(f13) == pd.Timestamp("2025-03-31")


def test_ended_series_carry_raw_and_adjusted_prices_for_the_delisting_rules():
    panel = pd.DataFrame({"ticker": "CIK9", "cik": 9,
                          "date": pd.to_datetime(["2015-01-01", "2015-04-01", "2015-07-01", "2015-10-01", "2016-01-01"]),
                          "price": [10.0, 9.0, 4.0, 3.0, 2.0], "close_raw": [1.0, 0.9, 0.4, 3.0, 2.0],
                          "price_source": "13F", "source": "sec"})
    seg = sp.ended_segments(panel, pd.Timestamp("2017-01-01"))
    r = seg.iloc[0]
    # a 1-for-10 reverse split in Q3: the year-ago price in the last price's units is 10, not the raw 1.0,
    # so the collapse test (last <= half the year-ago price) still sees the collapse
    assert r["last_price"] == 2.0 and r["price_1y_before"] == pytest.approx(10.0)
    assert int(r["cik"]) == 9


def test_listing_from_vendor_prices_later_form25_or_a_current_symbol():
    keys = pd.DataFrame({"cik": [1, 2, 3, 4], "start_date": pd.to_datetime(["2015-01-01"] * 4)})
    vendor = pd.DataFrame({"ticker": ["X"], "start_date": pd.to_datetime(["2015-01-01"]), "cik": [1]})
    f25 = pd.DataFrame({"cik": [2, 4], "date_filed": pd.to_datetime(["2017-05-01", "2014-01-01"]),
                        "form": ["25-NSE", "25-NSE"], "exchange": ["nasdaq", "nasdaq"]})
    sym = pd.DataFrame({"cik": [3], "exchange": ["nyse_amex"]})
    assert sp.listed_flags(keys, vendor, f25, sym).tolist() == [True, True, True, False]   # 4 left before 2015


def test_validation_compares_returns_and_splits_with_the_vendor():
    q = pd.date_range("2015-03-31", "2016-06-30", freq="QE")
    raw = [10, 11, 12, 6.5, 7, 7.5]                                   # 2:1 split in Q4 2015
    adj = [5, 5.5, 6, 6.5, 7, 7.5]
    panel = pd.DataFrame({"ticker": "CIK5", "cik": 5, "period": q, "date": q + pd.Timedelta(days=1),
                          "price": adj, "close_raw": raw, "price_source": "13F", "source": "sec"})
    ev = pd.DataFrame({"cik": [5], "period": [q[3]], "kind": ["split"], "ratio": [2.0]})
    days = pd.bdate_range("2015-03-02", "2016-07-15")
    vendor = pd.DataFrame({"ticker": "FIVE", "date": days,
                           "price": np.interp(days.asi8, q.asi8, adj)})
    keys = pd.DataFrame({"ticker": "FIVE", "start_date": q + pd.Timedelta(days=1), "cik": 5})
    out = sp.validate(panel, ev, vendor, keys)
    assert out["quarters_compared"] == 5 and out["returns_agree"] == 1.0
    assert out["split_recall"] == 1.0 and out["split_precision"] == 1.0


def test_dense_panel_adds_half_month_highs_outside_split_quarters():
    q = pd.date_range("2015-03-31", "2015-12-31", freq="QE")
    panel = pd.DataFrame({"ticker": "CIK5", "cik": 5, "period": q, "date": q + pd.Timedelta(days=1),
                          "price": [5.0, 5.5, 6.0, 6.5], "close_raw": [10.0, 11.0, 12.0, 6.5], "cusip": "AAA111111",
                          "price_source": "13F", "source": "sec", "n_filers": 9})
    ev = pd.DataFrame({"cik": [5], "period": [q[3]], "kind": ["split"], "ratio": [2.0]})
    px = pd.DataFrame({"cusip": "AAA111111", "max_date": pd.to_datetime(["2015-05-12", "2015-11-03"]),
                       "max_price": [16.0, 14.0]})
    out = sp.dense_panel(panel, ev, px, ["CIK5"])
    extra = out[out["price_source"] == "fails max"]
    assert len(extra) == 1 and extra["price"].iloc[0] == pytest.approx(8.0)   # 16 in pre-split units: 8 after
