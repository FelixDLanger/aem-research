import numpy as np
import pandas as pd
import pytest

from aem_research import restatements as rs
from aem_research import secprices as sp
from aem_research import splits
from sec_fixtures import share_facts

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


def _build(companies, ftd_px=None, ftd_names=None, dei=None, restated=None):
    rows, hold, br = [], [], []
    for cik, (r, h, cus) in companies.items():
        rows += r
        hold += h
        br += [(c, cik, "isin") for c in cus]
    f13 = pd.DataFrame(rows, columns=["cusip", "period", "price_13f", "n_price_filers", "n_filers"])
    holders = pd.DataFrame(hold, columns=["cusip", "period", "filer_cik", "shares"])
    bridge = pd.DataFrame(br, columns=["cusip", "cik", "method"]).drop_duplicates("cusip")
    mains = sp.main_cusips(f13, bridge)
    return sp.build_panel(mains, holders, ftd_px, bridge, as_of_period=PERIODS[-1], ftd_names=ftd_names, dei=dei,
                          restated=restated)


def _px(points):
    """Fails-to-deliver price table from (cusip, date, price) points, one half-month row each."""
    rows = [(c, pd.Timestamp(d), float(v), pd.Timestamp(d), float(v)) for c, d, v in points]
    px = pd.DataFrame(rows, columns=["cusip", "first_date", "first_price", "last_date", "last_price"])
    return px.assign(max_date=px["first_date"], max_price=px["first_price"], min_date=px["first_date"],
                     min_price=px["first_price"], n_obs=1)


def _dei(cik, before, after, split_month="2014-02"):
    """Cover-page share counts a month after each quarter-end: `before` until the split month, then `after`."""
    asof = pd.date_range("2012-10-31", "2017-01-31", freq="3ME")
    sh = np.where(asof < pd.Timestamp(split_month), before, after)
    return pd.DataFrame({"cik": cik, "asof": asof, "filed": asof + pd.Timedelta(days=10), "shares": sh})


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


def test_validation_tells_split_misses_fakes_and_one_quarter_errors_apart():
    q = pd.date_range("2015-03-31", "2016-06-30", freq="QE")
    days = pd.bdate_range("2015-03-02", "2016-07-15")
    flat = [10.0] * 6

    def series(cik, adj, raw, cusips=None):
        return pd.DataFrame({"ticker": f"CIK{cik}", "cik": cik, "period": q, "date": q + pd.Timedelta(days=1),
                             "price": adj, "close_raw": raw, "cusip": cusips or ["C%08d" % cik] * 6,
                             "n_filers": 12, "price_source": "13F", "source": "sec"})

    panel = pd.concat([
        series(5, [5, 5.5, 6, 6.5, 7, 7.5], [10, 11, 12, 6.5, 7, 7.5]),           # 2:1 split, found
        series(6, flat[:2] + [100.0] + flat[3:], flat[:2] + [100.0] + flat[3:]),   # a one-quarter price error
        series(7, [1, 1, 10, 10, 10, 10], [1, 1, 10, 10, 10, 10],                  # 1:10 reverse split, missed
               ["OLD000001"] * 2 + ["NEW000001"] * 4),
        series(8, [5, 5, 10, 10, 10, 10], flat),                                   # a split found that never was
    ], ignore_index=True)
    ev = pd.DataFrame({"cik": [5, 8], "period": [q[3], q[2]], "kind": "split", "ratio": [2.0, 2.0],
                       "evidence": ["holders", "CUSIP change"]})
    vend = {5: [5, 5.5, 6, 6.5, 7, 7.5], 6: flat, 7: flat, 8: flat}
    vendor = pd.concat([pd.DataFrame({"ticker": f"V{c}", "date": days, "price": np.interp(days.asi8, q.asi8, p)})
                        for c, p in vend.items()], ignore_index=True)
    keys = pd.concat([pd.DataFrame({"ticker": f"V{c}", "start_date": q + pd.Timedelta(days=1), "cik": c})
                      for c in vend], ignore_index=True)
    out = sp.validate(panel, ev, vendor, keys)
    assert out["quarters_compared"] == 20 and out["found_splits"] == 2 and out["found_vendor_flat"] == 1
    assert out["split_precision"] == 0.5 and out["vendor_splits"] == 2 and out["split_recall"] == 0.5
    assert out["jumps_up_2x"] == 3 and out["jumps_down_2x"] == 1 and out["jumps_round_trip"] == 1
    assert out["missed_splits"] == 1 and out["missed_at_cusip_change"] == 1.0 and out["missed_reverse_share"] == 1.0
    assert out["found_by"] == "CUSIP change 1 (0%) holders 1 (100%)"


NOBODY = lambda k: []                                                    # noqa: E731 – no 13F holders to match


def test_a_one_quarter_price_error_is_replaced_by_the_fails_close_or_left_out():
    spike = lambda k: 10000.0 if k == 3 else 10.0 + k                    # noqa: E731 – value reported in the wrong unit
    a = _company(lambda k: "GGG000010", spike, lambda k, j: 1000 + j)
    b = _company(lambda k: "GGG000011", spike, lambda k, j: 1000 + j)
    px = _px([("GGG000010", PERIODS[3] - pd.Timedelta(days=1), 13.0)])
    panel, _ev, info = _build({10: (*a, ["GGG000010"]), 11: (*b, ["GGG000011"])}, ftd_px=px)
    assert info["spikes_replaced"] == 1 and info["spikes_dropped"] == 1
    p10 = panel[panel["ticker"] == "CIK10"].set_index("period")
    assert p10.loc[PERIODS[3], "price"] == 13.0 and p10.loc[PERIODS[3], "price_source"] == "fails"
    p11 = panel[panel["ticker"] == "CIK11"]
    assert len(p11) == 7 and p11["price"].max() < 100 and set(panel["ticker"]) == {"CIK10", "CIK11"}


def test_filings_show_a_reverse_split_the_holders_cannot():
    price = lambda k: (1.0 - 0.05 * k) * (10 if k >= 4 else 1)          # noqa: E731
    c = _company(lambda k: "FFF000009", price, lambda k, j: 1, filers=NOBODY)
    panel, ev, info = _build({9: (*c, ["FFF000009"])}, dei=_dei(9, 5e7, 5e6))
    assert info["splits_filings"] == 1 and ev.iloc[0]["evidence"] == "filings" and ev.iloc[0]["ratio"] == 0.1
    assert panel["price"].round(3).tolist() == [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5]    # no fake tenfold gain
    _p, _e, info0 = _build({9: (*c, ["FFF000009"])})                       # without filings: seen, not explained
    assert info0["unexplained_jumps"] == 1 and info0["splits_filings"] == 0


def test_filings_decide_a_cusip_change_the_prices_alone_cannot():
    def changed(cik, ratio):
        old, new = f"HHH{cik:06d}", f"III{cik:06d}"
        c = _company(lambda k: old if k < 4 else new, lambda k: 1.0 if k < 4 else 1.0 / ratio,
                     lambda k, j: 1, filers=NOBODY)
        pts = [(old, "2014-02-14", 1.0), (new, "2014-02-18", 1.0 / ratio)]
        return c, [old, new], pts

    c12, cus12, pts12 = changed(12, 1 / 9)          # 1-for-9: a standard ratio, not a common one
    c13, cus13, pts13 = changed(13, 0.1)            # 1-for-10 by price, but the share count did not move
    c18, cus18, pts18 = changed(18, 0.1)            # 1-for-10 by price, no counts known: the switch decides
    px = _px(pts12 + pts13 + pts18)
    # counts known, but no restated history: neither the switch nor the counts set the ratio (since Nov 9:
    # both were off for 86% of such splits) – old equity and new start separate series
    dei = pd.concat([_dei(12, 5.4e7, 6e6), _dei(13, 5e7, 5e7)], ignore_index=True)
    panel, ev, info = _build({12: (*c12, cus12), 13: (*c13, cus13), 18: (*c18, cus18)}, ftd_px=px, dei=dei)
    assert set(panel.loc[panel["cik"] == 12, "ticker"]) == {"CIK12", "CIK12~2"}
    assert set(panel.loc[panel["cik"] == 13, "ticker"]) == {"CIK13", "CIK13~2"}
    assert info["cusip_switch_counts_known"] == 2
    sp18 = ev[(ev["cik"] == 18) & (ev["kind"] == "split")]
    assert len(sp18) == 1 and sp18.iloc[0]["evidence"] == "CUSIP change" and sp18.iloc[0]["ratio"] == pytest.approx(0.1)
    _p, ev0, _i = _build({12: (*c12, cus12)}, ftd_px=_px(pts12))                  # prices alone: 1-for-9 is too rare
    assert (ev0["kind"] == "new series").sum() == 1 and (ev0["kind"] == "split").sum() == 0


def test_two_holders_with_the_price_show_a_split_and_holders_without_it_do_not():
    rev = _company(lambda k: "JJJ000014", lambda k: (1.0 - 0.05 * k) * (10 if k >= 4 else 1),
                   lambda k, j: 1000 * (0.1 if k >= 4 else 1), filers=lambda k: [900001, 900002])
    flip = _company(lambda k: "KKK000015", lambda k: 10.0, lambda k, j: (5000 + j) * (0.1 if k >= 4 else 1))
    panel, ev, info = _build({14: (*rev, ["JJJ000014"]), 15: (*flip, ["KKK000015"])})
    sp14 = ev[(ev["cik"] == 14) & (ev["kind"] == "split")]
    assert len(sp14) == 1 and sp14.iloc[0]["evidence"] == "holders (few)" and info["splits_few_holders"] == 1
    assert info["holder_splits_rejected"] == 1 and (ev["cik"] == 15).sum() == 0
    assert panel.loc[panel["cik"] == 15, "price"].tolist() == [10.0] * 8      # a share-class flip is no split


def test_one_holder_needs_the_filings_to_agree():
    rev = _company(lambda k: "LLL000016", lambda k: (1.0 - 0.05 * k) * (10 if k >= 4 else 1),
                   lambda k, j: 1000 * (0.1 if k >= 4 else 1), filers=lambda k: [900001])
    _p, ev, info = _build({16: (*rev, ["LLL000016"])})
    assert info["splits_few_holders"] == 0 and info["unexplained_jumps"] == 1         # one holder alone: no
    _p, ev, info = _build({16: (*rev, ["LLL000016"])}, dei=_dei(16, 5e7, 5e6))
    assert info["splits_few_holders"] == 1 and ev.iloc[0]["ratio"] == 0.1             # with the filings: yes


def test_holders_are_overruled_when_the_filings_show_no_split():
    # all ten holders' counts double, the price halves – a split, unless the filings' count stayed put
    c = _company(lambda k: "MMM000017", lambda k: (10 + k) if k < 4 else (10 + k) / 2,
                 lambda k, j: (1000 + 10 * j) * (2 if k >= 4 else 1))
    _p, ev, info = _build({17: (*c, ["MMM000017"])}, dei=_dei(17, 5e7, 5.2e7))
    assert info["splits_holders"] == 0 and info["holder_splits_rejected"] == 1
    _p, ev, info = _build({17: (*c, ["MMM000017"])}, dei=_dei(17, 5e7, 1.04e8))
    assert info["splits_holders"] == 1 and ev.iloc[0]["ratio"] == 2.0                  # the filings agree


def test_validation_uses_the_filings_as_a_third_opinion():
    q = pd.date_range("2015-03-31", "2016-06-30", freq="QE")
    days = pd.bdate_range("2015-03-02", "2016-07-15")
    panel = pd.concat([
        pd.DataFrame({"ticker": "CIK5", "cik": 5, "period": q, "date": q + pd.Timedelta(days=1),
                      "price": [5, 5.5, 6, 6.5, 7, 7.5], "close_raw": [10, 11, 12, 6.5, 7, 7.5],
                      "cusip": "C00000005", "n_filers": 12, "price_source": "13F", "source": "sec"}),
        pd.DataFrame({"ticker": "CIK6", "cik": 6, "period": q, "date": q + pd.Timedelta(days=1),
                      "price": [10, 10, 10, 6.6, 6.6, 6.6], "close_raw": [10, 10, 10, 6.6, 6.6, 6.6],
                      "cusip": "C00000006", "n_filers": 12, "price_source": "13F", "source": "sec"})],
        ignore_index=True)
    ev = pd.DataFrame({"cik": [5], "period": [q[3]], "kind": "split", "ratio": [2.0], "evidence": ["holders"]})
    # company 6: a spin-off worth a third of the price – the vendor adjusts for it (ratio 1.5), the share count stays
    vend = {5: [5, 5.5, 6, 6.5, 7, 7.5], 6: [6.6, 6.6, 6.6, 6.6, 6.6, 6.6]}
    vendor = pd.concat([pd.DataFrame({"ticker": f"V{c}", "date": days, "price": np.interp(days.asi8, q.asi8, p)})
                        for c, p in vend.items()], ignore_index=True)
    keys = pd.concat([pd.DataFrame({"ticker": f"V{c}", "start_date": q + pd.Timedelta(days=1), "cik": c})
                      for c in vend], ignore_index=True)
    filings = pd.concat([_dei(5, 5e7, 1e8, split_month="2015-11"), _dei(6, 5e7, 5e7, split_month="2015-11")],
                        ignore_index=True)
    plain = sp.validate(panel, ev, vendor, keys)
    assert plain["vendor_splits"] == 2 and plain["split_recall"] == 0.5               # the spin-off counts as a split
    out, cases = sp.validate(panel, ev, vendor, keys, filings=filings, return_cases=True)
    assert out["vendor_splits"] == 1 and out["split_recall"] == 1.0 and out["vendor_moves_shares_flat"] == 1
    assert out["split_precision"] == 1.0 and out["filings_known"] == 1.0
    assert cases.empty or "split missed" not in set(cases["category"])
    lines = sp.case_lines(pd.DataFrame([{**dict.fromkeys(sp.CASE_COLS, np.nan), "category": "split missed", "cik": 6,
                                         "ticker": "CIK6", "prev_period": q[2], "period": q[3], "evidence": "",
                                         "cusip_change": False}]))
    assert len(lines) == 1 and lines[0].startswith("split missed · cik 6 · 2015-09 to 2015-12")


FILING_QUARTERS = pd.date_range("2012-03-31", "2015-12-31", freq="QE")
SPLIT_DAY = pd.Timestamp("2014-02-20")                                  # between PERIODS[3] and PERIODS[4]


def _restated(*companies):
    """(events, spans) of the restated history: companies as (cik, ratio or None, issue or None)."""
    facts = pd.concat([share_facts(c, FILING_QUARTERS, splits=[(SPLIT_DAY, r)] if r else (), issue=iss)
                       for c, r, iss in companies], ignore_index=True)
    ev, spans, _info = rs.evidence(facts)
    return ev, spans


BLUR = [1, 1, 1, 1, 0.7, 0.8, 1.25, 1.4, 0.75, 1.3]                    # four holders agree, six traded


def test_restated_history_sets_the_ratio_where_prices_and_counts_both_shrink_it():
    # 1-for-10 at a CUSIP change; the stock halves within weeks and the company doubles its share
    # count, so the switch of daily prices and the counts both say 1-for-5
    old, new = "NNN000020", "OOO000020"
    c = _company(lambda k: old if k < 4 else new, lambda k: (1.0 - 0.05 * k) * (5 if k >= 4 else 1),
                 lambda k, j: 1, filers=NOBODY)
    px = _px([(old, "2014-02-14", 0.82), (new, "2014-03-10", 4.1)])
    dei = _dei(20, 5e7, 1e7)
    p0_, ev0, _i = _build({20: (*c, [old, new])}, ftd_px=px, dei=dei)
    # without the restated history: no ratio from the switch or the counts (both say 1-for-5) – a new series
    assert (ev0["kind"] == "split").sum() == 0 and set(p0_["ticker"]) == {"CIK20", "CIK20~2"}
    doubled = lambda d: 5e7 * (2.0 if pd.Timestamp(d) >= pd.Timestamp("2014-03-15") else 1.0)  # noqa: E731
    panel, ev, info = _build({20: (*c, [old, new])}, ftd_px=px, dei=dei, restated=_restated((20, 0.1, doubled)))
    s1 = ev[ev["kind"] == "split"]
    assert len(s1) == 1 and s1.iloc[0]["ratio"] == pytest.approx(0.1)
    assert s1.iloc[0]["evidence"] == "CUSIP change + restated" and info["splits_restated"] == 1
    p = panel.set_index("period")["price"]
    assert p[PERIODS[4]] / p[PERIODS[3]] == pytest.approx(4.0 / 8.5)       # the halving stays, no fake gain


def test_restated_history_and_the_holders_median_catch_a_reverse_split_the_holders_test_misses():
    c = _company(lambda k: "PPP000021", lambda k: (1.0 - 0.05 * k) * (10 if k >= 4 else 1),
                 lambda k, j: (5000 + 7 * j) * (0.1 * BLUR[j] if k >= 4 else 1))
    _p, ev0, info0 = _build({21: (*c, ["PPP000021"])})
    s0 = ev0[ev0["kind"] == "split"]
    assert info0["splits_holders"] == 0 and len(s0) == 1                   # the agreement share is 40%
    assert s0.iloc[0]["evidence"] == "holders (median)" and s0.iloc[0]["ratio"] == pytest.approx(0.1)
    panel, ev, info = _build({21: (*c, ["PPP000021"])}, restated=_restated((21, 0.1, None)))
    s1 = ev[ev["kind"] == "split"]
    assert len(s1) == 1 and s1.iloc[0]["evidence"] == "restated" and s1.iloc[0]["ratio"] == pytest.approx(0.1)
    assert panel["price"].round(3).tolist() == [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5]


def test_a_stock_dividend_is_placed_by_the_holders_median():
    price = lambda k: 20.0 * 1.02 ** k / (1.25 if k >= 4 else 1)          # noqa: E731
    noise = [1, 1, 1, 1, 0.85, 0.9, 1.1, 1.15, 0.88, 1.12]
    c = _company(lambda k: "QQQ000022", price, lambda k, j: (2000 + 3 * j) * (1.25 * noise[j] if k >= 4 else 1))
    _p, ev0, _i = _build({22: (*c, ["QQQ000022"])})
    assert (ev0["kind"] == "split").sum() == 0                              # holders alone: 40% agree, no split
    panel, ev, info = _build({22: (*c, ["QQQ000022"])}, restated=_restated((22, 1.25, None)))
    s1 = ev[ev["kind"] == "split"]
    assert len(s1) == 1 and s1.iloc[0]["period"] == PERIODS[4] and s1.iloc[0]["ratio"] == pytest.approx(1.25)
    assert info["restated_applied"] == 1 and info["restated_unplaced"] == 0
    assert panel["price"].pct_change().dropna().round(4).eq(0.02).all()     # a steady 2% a quarter


def test_restated_history_without_a_split_overrules_the_holders():
    c = _company(lambda k: "RRR000023", lambda k: (10 + k) if k < 4 else (10 + k) / 2,
                 lambda k, j: (1000 + 10 * j) * (2 if k >= 4 else 1))
    _p, ev0, _i = _build({23: (*c, ["RRR000023"])})
    assert (ev0["evidence"] == "holders").sum() == 1
    _p, ev, info = _build({23: (*c, ["RRR000023"])}, restated=_restated((23, None, None)))
    assert (ev["kind"] == "split").sum() == 0 and info["holder_splits_refuted_restated"] == 1


def test_a_jump_is_neutralized_only_on_two_records_or_when_no_stock_moves_like_it():
    jump = lambda k: 0.255 if k < 4 else 1.66                           # noqa: E731 – up 6.5x in one quarter
    # the filings' count fell 99%, nothing else: not enough (counts jump with unit errors and reverse mergers)
    alone = _company(lambda k: "SSS000024", jump, lambda k, j: 1, filers=NOBODY)
    _p, ev0, info0 = _build({24: (*alone, ["SSS000024"])}, dei=_dei(24, 5e7, 5.35e5))
    assert (ev0["kind"] == "split").sum() == 0 and info0["unexplained_jumps"] == 1
    # the holders' median fell to 0.3 as well (no common ratio within 8%): a reverse split of unknown ratio
    both = _company(lambda k: "SSS000027", jump, lambda k, j: (4000 + j) * (0.3 * BLUR[j] if k >= 4 else 1))
    panel, ev, info = _build({27: (*both, ["SSS000027"])}, dei=_dei(27, 5e7, 5.35e5))
    s = ev[ev["kind"] == "split"]
    assert len(s) == 1 and s.iloc[0]["evidence"] == "unresolved jump" and info["splits_neutralized"] == 1
    p = panel.set_index("period")["price"]
    assert p[PERIODS[4]] / p[PERIODS[3]] == pytest.approx(1.66 / 0.255 / 7)   # undone: about flat, never 6.5x
    # a real tripling; the holders' median halves (profit taking, 40% agree)
    real = _company(lambda k: "TTT000025", lambda k: 10.0 if k < 4 else 30.0,
                    lambda k, j: (3000 + j) * (0.5 * BLUR[j] if k >= 4 else 1))
    _p, ev1, _i = _build({25: (*real, ["TTT000025"])})
    assert ev1.iloc[0]["evidence"] == "holders (median)"                    # without the history: read as 1-for-2
    panel2, ev2, info2 = _build({25: (*real, ["TTT000025"])}, restated=_restated((25, None, None)))
    assert (ev2["kind"] == "split").sum() == 0 and info2["jumps_restated_no_split"] == 1
    assert panel2.set_index("period")["price"][PERIODS[4]] == 30.0
    # 624-fold with nothing restated and the holders unchanged: a 13F unit error, not a move
    error = _company(lambda k: "VVV000028", lambda k: 0.319 if k < 4 else 199.0, lambda k, j: 2000 + j)
    panel3, ev3, info3 = _build({28: (*error, ["VVV000028"])}, restated=_restated((28, None, None)))
    s3 = ev3[ev3["kind"] == "split"]
    assert len(s3) == 1 and s3.iloc[0]["evidence"] == "implausible jump" and info3["splits_implausible"] == 1
    p3 = panel3.set_index("period")["price"]
    assert p3[PERIODS[4]] / p3[PERIODS[3]] == pytest.approx(199.0 / 0.319 / 1000)   # the unit undone, the move kept
    forty = _company(lambda k: "VVV000032", lambda k: 1.0 if k < 4 else 40.0, lambda k, j: 2000 + j)
    panel4, ev4, _i = _build({32: (*forty, ["VVV000032"])})
    p4 = panel4.set_index("period")["price"]
    assert ev4.iloc[0]["evidence"] == "implausible jump" and p4[PERIODS[4]] == pytest.approx(p4[PERIODS[3]])


def test_a_forward_split_from_holders_alone_needs_the_price_or_the_restated_history():
    # every holder's count grows a quarter – a 5:4 split, or a 25% share issuance index funds follow
    grow = lambda k, j: (1000 + 10 * j) * (1.25 if k >= 4 else 1)        # noqa: E731
    flat = _company(lambda k: "WWW000029", lambda k: 20.0, grow)
    _p, ev, info = _build({29: (*flat, ["WWW000029"])})
    assert (ev["kind"] == "split").sum() == 0 and info["holder_splits_forward_no_price"] == 1
    _p, ev, info = _build({29: (*flat, ["WWW000029"])}, restated=_restated((29, 1.25, None)))
    assert ev.iloc[0]["evidence"] == "holders" and ev.iloc[0]["ratio"] == 1.25        # the filings restated it
    fell = _company(lambda k: "WWW000030", lambda k: 20.0 if k < 4 else 16.4, grow)
    _p, ev, info = _build({30: (*fell, ["WWW000030"])})
    assert ev.iloc[0]["evidence"] == "holders" and ev.iloc[0]["ratio"] == 1.25        # the price moved with it


def test_the_holder_table_is_shared_with_the_panel():
    c = _company(lambda k: "XXX000031", lambda k: (10 + k) if k < 4 else (10 + k) / 2,
                 lambda k, j: (1000 + 10 * j) * (2 if k >= 4 else 1))
    rows, hold = c
    f13 = pd.DataFrame(rows, columns=["cusip", "period", "price_13f", "n_price_filers", "n_filers"])
    holders = pd.DataFrame(hold, columns=["cusip", "period", "filer_cik", "shares"])
    bridge = pd.DataFrame({"cusip": ["XXX000031"], "cik": [31], "method": "isin"})
    mains = sp.main_cusips(f13, bridge)
    tab = sp.holder_table(mains, holders, as_of_period=PERIODS[-1])
    assert list(tab.columns) == sp.HOLDER_COLS and len(tab) == 7
    row = tab.set_index("period").loc[PERIODS[4]]
    assert row["matched"] == 10 and row["ratio"] == 2.0 and row["agree"] == 1.0
    a = sp.build_panel(mains, holders, None, bridge, as_of_period=PERIODS[-1])
    b = sp.build_panel(mains, None, None, bridge, as_of_period=PERIODS[-1], holder_tab=tab)
    pd.testing.assert_frame_equal(a[0], b[0])
    assert a[1].equals(b[1]) and a[2]["splits_holders"] == b[2]["splits_holders"] == 1


def test_validation_with_the_restated_history():
    q = pd.date_range("2015-03-31", "2016-06-30", freq="QE")
    days = pd.bdate_range("2015-03-02", "2016-07-15")

    def series(cik, adj, raw):
        return pd.DataFrame({"ticker": f"CIK{cik}", "cik": cik, "period": q, "date": q + pd.Timedelta(days=1),
                             "price": adj, "close_raw": raw, "cusip": "C%08d" % cik, "n_filers": 12,
                             "price_source": "13F", "source": "sec"})

    panel = pd.concat([
        series(5, [5, 5.5, 6, 6.5, 7, 7.5], [10, 11, 12, 6.5, 7, 7.5]),           # 2:1 split, found by holders
        series(6, [10, 10, 10, 6.6, 6.6, 6.6], [10, 10, 10, 6.6, 6.6, 6.6]),      # a spin-off: no split
        series(8, [5, 5, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10]),              # a split the vendor lacks
        series(9, [1, 1, 1, 0.93, 0.93, 0.93], [1, 1, 1, 6.5, 6.5, 6.5]),         # a neutralized jump
    ], ignore_index=True)
    ev = pd.DataFrame({"cik": [5, 8, 9], "period": [q[3], q[2], q[3]], "kind": "split", "ratio": [2.0, 2.0, 1 / 7],
                       "evidence": ["holders", "CUSIP change", "unresolved jump"]})
    vend = {5: [5, 5.5, 6, 6.5, 7, 7.5], 6: [6.6] * 6, 8: [10] * 6, 9: [10, 10, 10, 6.5, 6.5, 6.5]}
    vendor = pd.concat([pd.DataFrame({"ticker": f"V{c}", "date": days, "price": np.interp(days.asi8, q.asi8, p)})
                        for c, p in vend.items()], ignore_index=True)
    keys = pd.concat([pd.DataFrame({"ticker": f"V{c}", "start_date": q + pd.Timedelta(days=1), "cik": c})
                      for c in vend], ignore_index=True)
    fq = pd.date_range("2014-03-31", "2017-03-31", freq="QE")
    facts = pd.concat([share_facts(5, fq, splits=[(pd.Timestamp("2015-11-15"), 2.0)]), share_facts(6, fq),
                       share_facts(8, fq, splits=[(pd.Timestamp("2015-08-15"), 2.0)]),
                       share_facts(9, fq, splits=[(pd.Timestamp("2015-11-15"), 0.1)])], ignore_index=True)
    restated = rs.evidence(facts)[:2]
    plain = sp.validate(panel, ev, vendor, keys)
    assert plain["vendor_splits"] == 3 and plain["found_splits"] == 2 and plain["split_precision"] == 0.5
    out, cases = sp.validate(panel, ev, vendor, keys, restated=restated, return_cases=True)
    # the spin-off is no split; company 9's vendor shows the 1-for-10 the panel only neutralized
    assert out["vendor_splits"] == 2 and out["vendor_moves_restated_none"] == 1
    assert out["found_splits"] == 2 and out["split_precision"] == 1.0 and out["found_restated_confirms"] == 2
    assert out["split_precision_vendor"] == 0.5 and out["neutralized"] == 1 and out["neutralized_vendor_reverse"] == 1.0
    assert out["split_recall"] == 0.5 and out["missed_restated_known"] == 1.0 and out["restated_known"] == 1.0
    assert set(cases["category"]) == {"jump neutralized"} and out["neutralized_within_2x"] == 1.0
    line = sp.case_lines(cases[cases["category"] == "jump neutralized"])[0]
    assert "restated ratio 0.1 ·" in line and "found 0.143 (unresolved jump)" in line


def test_a_restated_split_placed_in_a_fails_quarter_moves_to_the_holders_quarter():
    # 13F has no price for 2014-03-31 (fails-to-deliver fills it); the holders date the 1-for-10
    # split between 2013-12-31 and 2014-06-30, the restated history only to a window around March
    price = lambda k: (1.0 - 0.05 * k) * (10 if k >= 4 else 1)          # noqa: E731
    rows, hold = _company(lambda k: "UUU000026", price, lambda k, j: (5000 + 7 * j) * (0.1 if k >= 4 else 1))
    rows = [r for r in rows if r[1] != PERIODS[4]]
    hold = [h for h in hold if h[1] != PERIODS[4]]
    px = _px([("UUU000026", "2014-03-28", 8.0)])
    panel, ev, info = _build({26: (rows, hold, ["UUU000026"])}, ftd_px=px, restated=_restated((26, 0.1, None)))
    s = ev[ev["kind"] == "split"]
    assert len(s) == 1 and s.iloc[0]["evidence"] == "holders" and s.iloc[0]["period"] == PERIODS[5]
    assert info["restated_moved"] == 1 and info["restated_applied"] == 0 and info["fails_dropped_at_split"] == 1
    assert panel["price"].round(3).tolist() == [10.0, 9.5, 9.0, 8.5, 7.5, 7.0, 6.5]


def test_validation_on_the_study_population_with_the_holders_as_arbiter():
    q = pd.date_range("2015-03-31", "2016-06-30", freq="QE")
    days = pd.bdate_range("2015-03-02", "2016-07-15")

    def series(cik, adj, raw):
        return pd.DataFrame({"ticker": f"CIK{cik}", "cik": cik, "period": q, "date": q + pd.Timedelta(days=1),
                             "price": adj, "close_raw": raw, "cusip": "C%08d" % cik, "n_filers": 12,
                             "price_source": "13F", "source": "sec"})

    panel = pd.concat([
        series(41, [5, 5.5, 6, 6.5, 7, 7.5], [10, 11, 12, 6.5, 7, 7.5]),           # 2:1, found
        series(42, [9, 9, 9, 6, 6, 6], [9, 9, 9, 6, 6, 6]),                          # a spin-off, foreign issuer
        series(43, [5, 5, 6, 6, 6, 6], [10, 10, 6, 6, 6, 6]),                       # 2:1, missed
    ], ignore_index=True)
    ev = pd.DataFrame({"cik": [41], "period": [q[3]], "kind": "split", "ratio": [2.0], "evidence": ["holders"]})
    # the vendor adjusts company 42's history by 4:3 for the spin-off
    vend = {41: [5, 5.5, 6, 6.5, 7, 7.5], 42: [6.75, 6.75, 6.75, 6, 6, 6], 43: [5, 5, 6, 6, 6, 6]}
    vendor = pd.concat([pd.DataFrame({"ticker": f"V{c}", "date": days, "price": np.interp(days.asi8, q.asi8, p)})
                        for c, p in vend.items()], ignore_index=True)
    keys = pd.concat([pd.DataFrame({"ticker": f"V{c}", "start_date": q + pd.Timedelta(days=1), "cik": c})
                      for c in vend], ignore_index=True)
    # ten holders: company 42's kept their counts through the spin-off
    hold = pd.DataFrame({"cik": [41, 42, 43], "period": [q[3], q[3], q[2]], "matched": [10, 10, 10],
                         "median_ratio": [2.0, 1.0, 2.0], "ratio": [2.0, 1.0, 2.0], "agree": [1.0, 1.0, 1.0]})
    pop = pd.DataFrame({"cik": [41, 43], "first": pd.to_datetime(["2015-04-01"] * 2),
                        "last": pd.to_datetime(["2015-10-01"] * 2)})
    plain = sp.validate(panel, ev, vendor, keys)
    assert plain["vendor_splits"] == 3 and plain["split_recall"] == round(1 / 3, 4)
    out, cases = sp.validate(panel, ev, vendor, keys, holders=hold, population=pop, return_cases=True)
    assert out["vendor_splits"] == 2 and out["vendor_moves_holders_same"] == 1 and out["split_recall"] == 0.5
    assert out["quarters_compared_universe"] == 10 and out["vendor_splits_universe"] == 2
    assert out["split_recall_universe"] == 0.5 and out["split_precision_universe"] == 1.0
    assert out["found_by_universe"] == "holders 1 (100%)" and out["missed_holders_known"] == 1.0
    assert set(cases["cik"]) == {43} and set(cases["category"]) == {"split missed"}
    line = sp.case_lines(cases)[0]
    assert line.startswith("split missed · cik 43 · 2015-06 to 2015-09") and line.endswith("holders 10 at 2")
