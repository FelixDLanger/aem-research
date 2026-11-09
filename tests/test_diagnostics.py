import numpy as np
import pandas as pd

from aem_research import diagnostics as dg

CFG = {"universe": {"min_price_usd": 1.0, "market_cap_usd": {"min": 1e8, "max": 2e9},
                    "min_median_dollar_volume_20d_usd": 1e5}}
D0, D1 = pd.Timestamp("2015-01-01"), pd.Timestamp("2015-04-01")


def _a():
    """Six SEC-panel rows: two pass, one has no fresh cover-page shares, one too few 13F
    holders, one unlisted, one a SPAC shell; breadth is a nullable integer with a gap."""
    a = pd.DataFrame({
        "ticker": [f"CIK{i}" for i in (1, 2, 3, 4, 5, 6)], "cik": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
        "start_date": [D0, D0, D0, D0, D0, D1],
        "fund_available": [D0 - pd.Timedelta(days=40)] * 6,
        "sic": [3571, 3571, 3571, 3571, 3571, 6770],
        "entry_price_unadjusted": [10.0, 20.0, 5.0, 8.0, 12.0, 10.0],
        "shares_dei": [5e7, 2e7, 4e7, 5e7, 5e7, 3e7],
        "shares_dei_filed": [D0 - pd.Timedelta(days=50), D0 - pd.Timedelta(days=50), D0 - pd.Timedelta(days=900),
                             D0 - pd.Timedelta(days=50), D0 - pd.Timedelta(days=50), D1 - pd.Timedelta(days=50)],
        "wad_now": [5e7, 2e7, 4.1e7, 5e7, 5e7, 3e7],
        "breadth": pd.array([80, 60, 70, pd.NA, 90, 50], dtype="Int64"),
        "listed": [True, True, True, True, False, True],
    })
    fresh = dg.dei_fresh(a)
    a["shares"] = a["shares_dei"].where(fresh)
    a["market_cap"] = a["entry_price_unadjusted"] * a["shares"]
    a["liquid"] = (a["breadth"] >= 40).fillna(False).astype(bool)
    a["in_universe"] = [True, True, False, False, False, False]
    return a


def test_funnel_counts_each_filter_alone_so_far_and_without_it():
    fun = dg.universe_funnel(_a(), CFG)
    row = fun.set_index("filter")
    assert row.loc["filer", "alone"] == 6 and row.loc["not a SPAC shell", "so_far"] == 5
    assert row.loc["shares known", "so_far"] == 4 and np.isnan(row.loc["shares known", "all_but_this"])
    assert row.loc["listed", "so_far"] == 2 and row.loc["universe flag", "so_far"] == 2
    assert row.loc["liquid", "all_but_this"] == 3 and row.loc["listed", "all_but_this"] == 3
    assert row.loc["market cap in range", "all_but_this"] == 3          # leaving out size drops the share need too
    s = dg.funnel_summary(fun)
    assert s["rows"] == 6 and s["flag"] == 2 and s["without_mcap"] == 3 and s["binding"].endswith("(+1 without it)")


def test_detail_reads_nullable_columns_and_fresh_shares():
    d = dg.detail(_a(), CFG, min_breadth=40)
    assert d["base_rows"] == 5 and d["dei_known_base"] == 1.0 and d["dei_fresh_base"] == 0.8
    assert d["breadth_known_sized"] == 0.75 and d["breadth_ge_40_sized"] == 0.75 and d["listed_sized"] == 0.75


def test_compare_finds_vendor_windows_in_a_and_names_the_first_failure():
    a = _a()
    v = pd.DataFrame({"ticker": ["AA", "BB", "CC", "DD", "EE"], "cik": pd.array([1, 2, 3, 4, 99], dtype="Int64"),
                      "start_date": [D0] * 5, "in_universe": True,
                      "entry_price_unadjusted": [10.0, 21.0, 5.0, 8.0, 3.0], "shares": [5e7, 2e7, 4e7, 5e7, 1e8],
                      "shares_source": ["filing", "filing", "vendor", "filing", "vendor"],
                      "market_cap": [5e8, 4.4e8, 2e8, 4e8, 3e8], "breadth": pd.array([80, 60, 70, pd.NA, 5], dtype="Int64"),
                      "dollar_volume_20d": 1e6})
    c = dg.compare_universes(a, v, CFG, min_breadth=40)
    assert c["vendor_windows"] == 5 and c["found_in_a"] == 0.8 and c["cik_anywhere_in_a"] == 0.8
    assert c["a_in_universe"] == 0.5 and c["a_pass_shares"] == 0.75 and c["a_pass_liquid"] == 0.75
    assert c["a_first_fail"].startswith("none 50.0%") and "shares 25.0%" in c["a_first_fail"]
    assert c["price_a_over_v"].startswith("1 (within 10% 100%") and c["breadth_equal"] == 1.0
    assert c["v_vendor_shares_where_a_has_none"] == 1.0 and c["a_dei_fresh"] == 0.75
    ex = dg.examples(a, v, CFG, n=5)
    assert set(ex["a_fails"]) == {"shares", "liquid"} and len(ex) == 2
    assert c["a_universe_windows"] == 2 and c["a_not_priced_by_vendor"] == 0.0
    c2 = dg.compare_universes(a, v[v["cik"] != 2], CFG)          # the vendor never priced company 2
    assert c2["a_not_priced_by_vendor"] == 0.5 and c2["a_not_priced_first_date"] == 0.5


def test_what_if_counts_other_choices():
    w = dg.what_if(_a(), CFG, ks=(3, 75))
    assert w["as_run"] == 2 and w["recomputed"] == 2
    assert w["shares_or_wad"] == 3                    # the stale cover page falls back to weighted-average shares
    assert w["holders_3"] == 2 and w["holders_75"] == 1 and w["no_listing"] == 3 and w["no_liquidity"] == 3
