from datetime import date

import numpy as np
import pandas as pd
import pytest

from aem_research import delistings as dl
from aem_research import features as ft
from aem_research.config import load_config
from aem_research.data import sec
from aem_research.guards import LookAheadError
from aem_research.outcomes import (CENSORED, COMPLETE, OutcomeSpec, _compute_outcomes, build_price_panel,
                                   build_starts, panel_as_of, starts_from_features)
from conftest import CONFIG, bseries
from sec_fixtures import FakeClient


# ---------------------------------------------------------------- outcome engine additions

def two_source_panel():
    sim = bseries("A", "2023-06-01", [10.0] * 870)                 # ends mid-Oct 2026
    sim["source"] = "simfin"
    stq = bseries("B", "2023-06-01", [10.0] * 890)                 # runs ~3 weeks longer
    stq["source"] = "stooq"
    gap_fill = bseries("Z", "2015-01-01", [5.0] * 100)             # only an old delisted name
    gap_fill["source"] = "eodhd"
    return pd.concat([sim, stq, gap_fill], ignore_index=True)


def test_panel_as_of_is_the_earliest_live_source():
    p = two_source_panel()
    a_end = p.loc[p["ticker"] == "A", "date"].max()
    assert panel_as_of(p) == a_end
    # without the as-of rule a SimFin series ending a few weeks early would look delisted
    spec = OutcomeSpec(horizons_years=(3,), touch_multiples=(3.0,))
    starts = pd.DataFrame({"ticker": ["A"], "start_date": [pd.Timestamp("2023-10-02")]})
    out = _compute_outcomes(spec, p, starts)
    assert out["status"].iloc[0] == CENSORED                       # window ends after the as-of date
    early = pd.DataFrame({"ticker": ["A"], "start_date": [pd.Timestamp("2023-06-01")]})
    assert _compute_outcomes(spec, p, early)["status"].iloc[0] == COMPLETE


def test_build_starts_quarterly_and_volume_in_panel():
    panel = bseries("A", "2015-01-01", [5.0] * 300)
    st = build_starts(panel, (2015, 2015), months=(1, 4, 7, 10))
    assert st["start_date"].dt.month.tolist() == [1, 4, 7, 10] and st["start_date"].dtype == "datetime64[ns]"
    sim = pd.DataFrame({"ticker": ["A"] * 2, "date": pd.bdate_range("2015-01-01", periods=2), "close": [1.0, 2.0],
                        "adj_close": [1.0, 2.0], "volume": [100.0, 200.0], "shares_out": [1e6, 1e6]})
    pnl = build_price_panel(sim, None)
    assert {"volume", "shares_out"} <= set(pnl.columns) and pnl["volume"].tolist() == [100.0, 200.0]


# ---------------------------------------------------------------- delistings

def ended_panel():
    parts = [bseries("MRG", "2016-01-01", list(np.linspace(20, 25, 600))),          # merger target
             bseries("PNY", "2016-01-01", list(np.linspace(10, 0.5, 600))),         # penny, Form 25-NSE
             bseries("CLP", "2016-01-01", [30.0] * 350 + list(np.linspace(30, 9, 250))),  # collapse, NYSE folder
             bseries("QUI", "2016-01-01", [15.0] * 600),                            # quiet stop, no evidence
             bseries("NOC", "2016-01-01", list(np.linspace(5, 0.8, 600))),          # no CIK, penny
             bseries("LIVE", "2016-01-01", [12.0] * 2900)]
    return pd.concat(parts, ignore_index=True)


def test_ended_segments_and_classification():
    panel = ended_panel()
    as_of = panel["date"].max()
    seg = dl.ended_segments(panel, as_of)
    assert set(seg["ticker"]) == {"MRG", "PNY", "CLP", "QUI", "NOC"}
    row = seg.set_index("ticker").loc["CLP"]
    assert row["last_price"] == pytest.approx(9.0) and row["price_1y_before"] == pytest.approx(30.0)
    companies = pd.DataFrame({"ticker": ["MRG", "PNY", "CLP"], "cik": [11, 22, 33]})
    bridge = pd.DataFrame({"symbol": ["QUI"], "issuer_cik": [44], "first_seen": [pd.Timestamp("2015-01-01")],
                           "last_seen": [pd.Timestamp("2018-01-01")], "filings": [5]})
    seg = dl.map_ciks(seg, companies, bridge)
    assert seg.set_index("ticker")["cik_source"].to_dict() == {"MRG": "simfin", "PNY": "simfin", "CLP": "simfin",
                                                                "QUI": "bridge", "NOC": ""}
    last = seg.set_index("ticker")["last_date"]
    merger = pd.DataFrame({"cik": [11, 33], "form": ["DEFM14A", "DEFM14A"],
                           "date_filed": [last["MRG"] - pd.Timedelta(days=90), last["CLP"] - pd.Timedelta(days=800)]})
    form25 = pd.DataFrame({"cik": [22, 9999], "company": ["PENNY CORP", "NASDAQ STOCK MARKET LLC"],
                           "form": ["25-NSE", "25-NSE"], "date_filed": [last["PNY"]] * 2,
                           "filename": ["edgar/data/22/x.txt"] * 2})
    groups = pd.DataFrame({"ticker": ["CLP"], "group": ["nyse stocks"]})
    out = dl.classify(seg, form25, merger, groups).set_index("ticker")
    assert out["category"].to_dict() == {"MRG": "merger", "PNY": "performance_nasdaq",
                                         "CLP": "performance_nyse_amex", "QUI": "unknown", "NOC": "performance"}
    assert out.loc["PNY", "evidence"] == "Form 25-NSE" and out["delist_return"].isna().all()
    s = dl.summary(out.reset_index())
    assert s["ended_series"] == 5 and s["share_unknown"] == pytest.approx(0.2)
    # the engine picks the category up at the series end
    spec = OutcomeSpec.from_config(load_config(CONFIG).data)
    starts = pd.DataFrame({"ticker": ["PNY"], "start_date": [pd.Timestamp("2016-01-01")]})
    res = _compute_outcomes(spec, panel, starts, out.reset_index(), as_of=as_of)
    r3 = res[res["horizon_years"] == 3].iloc[0]
    assert r3["delist_category"] == "performance_nasdaq" and r3["delist_return_used"] == pytest.approx(-0.55)


def test_index_subsets_one_download_and_partial_cache(tmp_path):
    idx = ("Description: Master Index\n\nCIK|Company Name|Form Type|Date Filed|Filename\n"
           "--------------------------------------------------------------------------------\n"
           "22|PENNY CORP|25-NSE|2018-03-01|edgar/data/22/a.txt\n"
           "11|MERGED CO|DEFM14A|2018-02-01|edgar/data/11/b.txt\n"
           "33|OTHER INC|10-K|2018-03-01|edgar/data/33/c.txt\n").encode("latin-1")
    url = sec.INDEX_URL.format(year=2018, q=1, ext="gz")
    import gzip
    client = FakeClient({url: gzip.compress(idx)})
    tables, stats = sec.load_index_subsets(tmp_path, 2018, 2018, client=client, today=date(2018, 4, 2), log=None)
    assert stats == {"cached": 0, "downloaded": 1, "missing": 0}
    assert tables["form25"]["cik"].tolist() == [22] and tables["merger"]["form"].tolist() == ["DEFM14A"]
    # a later run reuses both caches; an older Form 25-only cache triggers one re-download
    t2, s2 = sec.load_index_subsets(tmp_path, 2018, 2018, client=FakeClient({}), today=date(2018, 4, 2), log=None)
    assert s2["cached"] == 1 and len(t2["merger"]) == 1
    (tmp_path / "merger_2018Q1.pkl").unlink(missing_ok=True)
    (tmp_path / "merger_2018Q1.parquet").unlink(missing_ok=True)
    t3, s3 = sec.load_index_subsets(tmp_path, 2018, 2018, client=client, today=date(2018, 4, 2), log=None)
    assert s3["downloaded"] == 1 and len(t3["merger"]) == 1
    t4, s4 = sec.load_index_subsets(tmp_path / "empty", 2018, 2018, offline=True, today=date(2018, 4, 2), log=None)
    assert s4["missing"] == 1 and t4["form25"].empty


# ---------------------------------------------------------------- features

def feature_inputs():
    s = pd.Timestamp("2016-07-01")
    n = 12
    tickers = [f"T{i:02d}" for i in range(n)]
    parts = []
    for i, t in enumerate(tickers):
        # T00 fell 60% from its high; the rest are flat at different prices
        vals = [50.0] * 400 + [20.0] * 420 if i == 0 else [10.0 + i] * 820   # runs past the start date
        b = bseries(t, "2013-07-01", vals)
        b["volume"] = 1e5 * (i + 1)
        b["source"] = "simfin"
        parts.append(b)
    panel = pd.concat(parts, ignore_index=True)
    companies = pd.DataFrame({"ticker": tickers, "cik": range(100, 100 + n)})
    fund = pd.DataFrame({"cik": range(100, 100 + n), "start_date": s, "fund_available": s - pd.Timedelta(days=30),
                         "shares_dei": 2e7, "shares_dei_filed": s - pd.Timedelta(days=30), "sic": 3571,
                         "fcf_pos": True, "net_cash_pos": True, "s1_proxy": [True, False] * 6,
                         "low_dilution": True, "net_cash": 1e7, "public_float": 3e8})
    fund.loc[11, "sic"] = 6770                                      # a SPAC shell
    ins = pd.DataFrame({"issuer_cik": [100, 101], "start_date": s, "insider_any": True, "insider_cluster": False,
                        "insider_opportunistic": [True, False], "insider_routine_only": [False, True],
                        "insider_unclassified_only": False, "insider_n": 1,
                        "insider_post_results_cluster": False, "insider_available": s - pd.Timedelta(days=10)})
    breadth = pd.DataFrame({"cik": range(100, 111), "start_date": s, "breadth": [5, 50, 40, 60, 80, 90, 100, 120,
                                                                              150, 170, 200],
                            "breadth_period": pd.Timestamp("2016-03-31"), "breadth_available": pd.Timestamp("2016-07-01")})
    return s, panel, companies, fund, ins, breadth


def test_assemble_universe_cohort_and_attention():
    s, panel, companies, fund, ins, breadth = feature_inputs()
    pf = ft.price_features(panel, [s])
    t0 = pf.set_index("ticker").loc["T00"]
    assert t0["drawdown"] == pytest.approx(20 / 50 - 1) and t0["dollar_volume_20d"] == pytest.approx(20 * 1e5)
    keys = ft.ticker_ciks(pf, companies, [])
    f = ft.assemble(pf, keys, fund, ins, breadth, load_config(CONFIG).data)
    by = f.set_index("ticker")
    assert not by.loc["T11", "in_universe"]                         # SIC 6770
    assert by.loc["T00", "market_cap"] == pytest.approx(20 * 2e7) and by.loc["T00", "in_universe"]
    assert by.loc["T00", "fallen"] and bool(by.loc["T00", "h2a"]) and bool(by.loc["T00", "h2b"])
    assert not bool(by.loc["T01", "h2a"])                           # not fallen
    assert bool(by.loc["T00", "low_breadth"])                       # 5 holders: bottom tercile net of size
    assert by["low_breadth"][by["in_universe"]].sum() == pytest.approx(round(by["in_universe"].sum() / 3), abs=1)
    assert by.loc["T05", "insider_any"] == False                    # noqa: E712  no Form 4 on file = no purchase
    ft.check_point_in_time(f)
    rep = ft.coverage_report(f)
    assert rep["universe"].iloc[0] == 11 and rep["h2a_cohort"].iloc[0] == 1 and rep["breadth_known"].iloc[0] == 1.0
    choice, share = ft.choose_attention(f, "2013-07-01")
    assert choice == "breadth" and share == 1.0
    st = starts_from_features(f)
    assert len(st) == 11 and "T11" not in set(st["ticker"])


def test_look_ahead_guard_trips():
    s, panel, companies, fund, ins, breadth = feature_inputs()
    fund["fund_available"] = s                                      # filed on the start date itself
    pf = ft.price_features(panel, [s])
    f = ft.assemble(pf, ft.ticker_ciks(pf, companies, []), fund, ins, breadth, load_config(CONFIG).data)
    with pytest.raises(LookAheadError):
        ft.check_point_in_time(f)


def test_bridges_and_universe_holes():
    s, panel, companies, fund, ins, breadth = feature_inputs()
    subs = pd.DataFrame({"adsh": ["a", "b"], "cik": [500, 501], "filed": [pd.Timestamp("2016-05-01")] * 2,
                         "instance": ["ghost-20160331.xml", "0001-16-000001.xml"]})
    ib = ft.instance_bridge(subs)
    assert ib["symbol"].tolist() == ["GHOST"]
    pf = ft.price_features(panel, [s])
    keys = ft.ticker_ciks(pf, None, [ib, pd.DataFrame({"symbol": ["T03"], "issuer_cik": [103],
                                                       "first_seen": [pd.Timestamp("2015-01-01")],
                                                       "last_seen": [pd.Timestamp("2016-06-01")], "filings": [3]})])
    assert keys.set_index("ticker").loc["T03", "cik"] == 103 and keys["cik"].notna().sum() == 1
    f = ft.assemble(pf, ft.ticker_ciks(pf, companies, []), fund, ins, breadth, load_config(CONFIG).data)
    ghost = pd.DataFrame({"cik": [500], "start_date": s, "public_float": 5e8})
    holes = ft.universe_holes(pd.concat([fund, ghost], ignore_index=True), f,
                              [ib, pd.DataFrame({"symbol": ["X"], "issuer_cik": [100], "first_seen": [s],
                                                 "last_seen": [s], "filings": [1]})])
    row = holes.iloc[0]
    assert row["in_range_filers"] == 2 and row["unpriced"] == 1     # GHOST has a symbol but no price


def test_unadjusted_price_undoes_later_splits():
    """A later 1:10 reverse split makes the vendor's history ten times the actual price; a later
    2:1 split halves it. The 13F implied quarter-end price restores the actual level."""
    s = pd.Timestamp("2016-07-01")
    q = pd.Timestamp("2016-03-31")
    parts = []
    for t, adj in (("RS", 20.0), ("FS", 5.0)):
        b = bseries(t, "2013-07-01", [adj] * 820)
        b["close_raw"], b["volume"], b["source"] = adj, 1e6, "stooq"
        parts.append(b)
    panel = pd.concat(parts, ignore_index=True)
    pf = ft.price_features(panel, [s])
    assert pf["price_q"].notna().all() and pf["close_raw_q"].notna().all()
    keys = ft.ticker_ciks(pf, pd.DataFrame({"ticker": ["RS", "FS"], "cik": [1, 2]}), [])
    fund = pd.DataFrame({"cik": [1, 2], "start_date": s, "fund_available": s - pd.Timedelta(days=20),
                         "shares_dei": [3e7, 1.5e7], "shares_dei_filed": s - pd.Timedelta(days=20), "sic": 3571,
                         "fcf_pos": True, "net_cash_pos": True, "s1_proxy": True, "low_dilution": True,
                         "net_cash": 1.0, "public_float": 1e8})
    p13 = pd.DataFrame({"cik": [1, 2], "start_date": s, "price_13f": [2.0, 10.0], "price_13f_period": q})
    cfg = load_config(CONFIG).data
    naive = ft.assemble(pf, keys, fund, None, None, cfg).set_index("ticker")
    fixed = ft.assemble(pf, keys, fund, None, None, cfg, prices13=p13).set_index("ticker")
    # without the fix: RS looks like US$600m and FS like US$75m – both wrong
    assert naive.loc["RS", "in_universe"] and not naive.loc["FS", "in_universe"]
    assert fixed.loc["RS", "entry_price_unadjusted"] == pytest.approx(2.0)
    assert fixed.loc["RS", "market_cap"] == pytest.approx(6e7) and not fixed.loc["RS", "in_universe"]
    assert fixed.loc["FS", "market_cap"] == pytest.approx(1.5e8) and fixed.loc["FS", "in_universe"]
    assert set(fixed["price_basis"]) == {"13F"}
    chk = ft.price_basis_check(fixed.reset_index())
    assert chk.loc[0, "share_off_10pct"] == 1.0          # this vendor's closes are split-adjusted


def test_insider_sale_placebo():
    s, panel, companies, fund, ins, breadth = feature_inputs()
    ins = ins[ins["issuer_cik"] != 100]                           # T00: fallen quality, but no purchase
    sales = pd.DataFrame({"issuer_cik": [100, 101], "start_date": s, "insider_any": True})
    pf = ft.price_features(panel, [s])
    f = ft.assemble(pf, ft.ticker_ciks(pf, companies, []), fund, ins, breadth, load_config(CONFIG).data,
                    sales=sales).set_index("ticker")
    assert bool(f.loc["T00", "h2_placebo"]) and not bool(f.loc["T00", "h2a"])
    assert not bool(f.loc["T01", "h2_placebo"])                   # sold, but not fallen
    assert ft.coverage_report(f.reset_index())["placebo_cohort"].iloc[0] == 1


def test_exchange_comes_from_the_exchange_row_of_the_same_accession_in_another_folder():
    from aem_research import delistings as dl
    seg = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC~eod", "DDD"],
                        "first_date": pd.to_datetime(["2010-01-04"] * 4),
                        "last_date": pd.to_datetime(["2016-05-02", "2018-01-10", "2019-03-01", "2020-06-01"]),
                        "last_price": [0.5, 0.4, 0.3, 0.2], "price_1y_before": [3.0, 2.0, 2.0, 2.0],
                        "cik": pd.array([1001, 2002, 3003, 4004], dtype="Int64")})
    f25 = pd.DataFrame({
        "cik": [1001, 1120193, 2002, 876661],
        "company": ["AAA CORP", "NASDAQ STOCK MARKET LLC", "BBB INC", "New York Stock Exchange LLC"],
        "form": ["25-NSE"] * 4,
        "date_filed": pd.to_datetime(["2016-05-12", "2016-05-12", "2014-02-03", "2014-02-03"]),
        # the real index lists each CIK's row under its own folder: same accession, different path
        "filename": ["edgar/data/1001/0001120193-16-000123.txt", "edgar/data/1120193/0001120193-16-000123.txt",
                     "edgar/data/2002/0000876661-14-000045.txt", "edgar/data/876661/0000876661-14-000045.txt"]})
    ex = dl.form25_with_exchange(f25)
    assert ex.set_index("cik").loc[1001, "exchange"] == "nasdaq" and ex.set_index("cik").loc[2002, "exchange"] == "nyse_amex"
    emap = pd.DataFrame({"ticker": ["CCC"], "exchange": ["nasdaq"]})
    c = dl.classify(seg, f25, None, None, exchange_map=emap).set_index("ticker")["category"]
    assert c["AAA"] == "performance_nasdaq"                     # Form 25-NSE filed by Nasdaq near the last day
    assert c["BBB"] == "performance_nyse_amex"                  # the issuer's earlier NYSE Form 25
    assert c["CCC~eod"] == "performance_nasdaq"                 # from the symbol list (EODHD)
    assert c["DDD"] == "performance"                            # nothing known: the exchange stays open


def test_breadth_liquidity_and_listing_for_the_sec_panel():
    s, panel, companies, fund, ins, breadth = feature_inputs()
    pf = ft.price_features(panel, [s])
    keys = ft.ticker_ciks(pf, companies, [])
    cfg = load_config(CONFIG).data
    listed = pd.DataFrame({"cik": range(100, 112), "start_date": s, "listed": [True] * 11 + [False]})
    listed.loc[3, "listed"] = False                                    # T03: OTC at the start date
    f = ft.assemble(pf, keys, fund, ins, breadth, cfg, liquidity="breadth", min_breadth=40, listed=listed)
    by = f.set_index("ticker")
    assert not by.loc["T00", "in_universe"] and not by.loc["T00", "liquid"]    # 5 holders < 40
    assert by.loc["T02", "in_universe"] and by.loc["T02", "liquid"]           # 40 holders
    assert not by.loc["T03", "in_universe"] and not by.loc["T03", "listed"]
    assert by.loc["T04", "in_universe"]
    vol = ft.assemble(pf, keys, fund, ins, breadth, cfg)                     # the vendor default is unchanged
    assert vol.set_index("ticker").loc["T00", "in_universe"] and vol["listed"].all()


def test_breadth_threshold_calibrates_on_the_volume_screen():
    cfg = load_config(CONFIG).data
    n = 400
    rng = np.random.default_rng(3)
    b = rng.integers(1, 60, n)
    f = pd.DataFrame({"fund_available": pd.Timestamp("2016-01-01"), "sic": 3571, "entry_price_unadjusted": 10.0,
                      "market_cap": 5e8, "breadth": b, "dollar_volume_20d": np.where(b >= 12, 5e5, 5e4)})
    out = ft.calibrate_breadth(f, cfg)
    assert out["k"] == 12 and out["agreement"] == 1.0 and out["youden"] == 1.0
    f2 = f.assign(dollar_volume_20d=5e5)                                     # nothing fails the volume screen
    assert ft.calibrate_breadth(f2, cfg)["k"] == 3 and "floor" in ft.calibrate_breadth(f2, cfg)["basis"]
    assert ft.calibrate_breadth(f.iloc[:10], cfg)["k"] == 3                 # too few windows


def test_hole_keys_match_the_hole_counts():
    s, panel, companies, fund, ins, breadth = feature_inputs()
    pf = ft.price_features(panel, [s])
    f = ft.assemble(pf, ft.ticker_ciks(pf, companies, []), fund, ins, breadth, load_config(CONFIG).data)
    ghost = pd.DataFrame({"cik": [500, 501], "start_date": s, "public_float": [5e8, 5e9]})
    sym = [pd.DataFrame({"symbol": ["G", "H"], "issuer_cik": [500, 501], "first_seen": [s] * 2, "last_seen": [s] * 2,
                         "filings": [1, 1]})]
    allf = pd.concat([fund, ghost], ignore_index=True)
    keys = ft.universe_hole_keys(allf, f, sym)
    assert keys["cik"].tolist() == [500]                                    # 501 is above the size range


def test_formation_dates_from_a_pair_or_a_range():
    from aem_research import features as ft
    pair = ft.formation_dates([2013, 2023], [1, 4, 7, 10])
    assert pair[0] == pd.Timestamp("2013-01-01") and pair[-1] == pd.Timestamp("2023-10-01") and len(pair) == 44
    rng = ft.formation_dates(range(2009, 2013), [1, 4, 7, 10])        # once read as 2009-2010 only
    assert rng[-1] == pd.Timestamp("2012-10-01") and len(rng) == 16
    assert ft.formation_dates((2009, 2012), [1]) == [pd.Timestamp(f"{y}-01-01") for y in range(2009, 2013)]
