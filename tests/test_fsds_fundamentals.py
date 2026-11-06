import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from aem_research import fundamentals as fu
from aem_research.data import fsds
from sec_fixtures import FakeClient, fsds_zip, num_row, sub_row

K11, Q311, Q312, KA = "0001-12-000001", "0001-11-000002", "0001-12-000003", "0001-13-000004"


def acme_rows():
    subs = [sub_row(K11, 1001, "10-K", "20111231", "FY", "20120301"),
            sub_row(Q311, 1001, "10-Q", "20110930", "Q3", "20111104"),
            sub_row(Q312, 1001, "10-Q", "20120930", "Q3", "20121105"),
            # an amendment filed after the 2013-01-02 formation date must not be used then
            sub_row(KA, 1001, "10-K/A", "20111231", "FY", "20130115"),
            # a non-periodic form is ignored
            sub_row("0001-12-000009", 1001, "8-K", "20121001", None, "20121002")]
    n = []
    # FY2011 10-K with FY2010 comparatives
    for tag, fy11, fy10 in [("NetCashProvidedByUsedInOperatingActivities", 100, 80),
                            ("PaymentsToAcquirePropertyPlantAndEquipment", 20, 10),
                            ("Revenues", 1000, 900)]:
        n += [num_row(K11, tag, "20111231", 4, fy11), num_row(K11, tag, "20101231", 4, fy10)]
    n += [num_row(K11, "WeightedAverageNumberOfDilutedSharesOutstanding", "20111231", 4, 10.0e6, "shares"),
          num_row(K11, "CashAndCashEquivalentsAtCarryingValue", "20111231", 0, 300),
          num_row(K11, "LongTermDebt", "20111231", 0, 50),
          num_row(K11, "EntityCommonStockSharesOutstanding", "20120229", 0, 10.1e6, "shares", version="dei/2011"),
          num_row(K11, "EntityPublicFloat", "20110630", 0, 4.0e8, version="dei/2011")]
    # Q3 2011 10-Q: 9M 2011 and 9M 2010 (the comparative needed one year later)
    for tag, a, b in [("NetCashProvidedByUsedInOperatingActivities", 70, 55),
                      ("PaymentsToAcquirePropertyPlantAndEquipment", 12, 9),
                      ("Revenues", 700, 650)]:
        n += [num_row(Q311, tag, "20110930", 3, a), num_row(Q311, tag, "20100930", 3, b)]
    n += [num_row(Q311, "WeightedAverageNumberOfDilutedSharesOutstanding", "20110930", 1, 10.0e6, "shares")]
    # Q3 2012 10-Q: 9M 2012 and 9M 2011
    for tag, a, b in [("NetCashProvidedByUsedInOperatingActivities", 110, 70),
                      ("PaymentsToAcquirePropertyPlantAndEquipment", 15, 12),
                      ("Revenues", 800, 700)]:
        n += [num_row(Q312, tag, "20120930", 3, a), num_row(Q312, tag, "20110930", 3, b)]
    n += [num_row(Q312, "Revenues", "20120930", 1, 280),        # three-month value, not used for TTM
          num_row(Q312, "WeightedAverageNumberOfDilutedSharesOutstanding", "20120930", 1, 10.1e6, "shares"),
          num_row(Q312, "WeightedAverageNumberOfDilutedSharesOutstanding", "20110930", 1, 10.0e6, "shares"),
          num_row(Q312, "CashAndCashEquivalentsAtCarryingValue", "20120930", 0, 320),
          num_row(Q312, "LongTermDebt", "20120930", 0, 40),
          num_row(Q312, "EntityCommonStockSharesOutstanding", "20121031", 0, 10.3e6, "shares", version="dei/2012"),
          # dimensional and co-registrant rows are dropped
          num_row(Q312, "Revenues", "20120930", 3, 99999, segments="ProductOrServiceAxis=Widgets"),
          num_row(Q312, "Revenues", "20120930", 3, 88888, coreg="SubsidiaryCo")]
    n += [num_row(KA, "NetCashProvidedByUsedInOperatingActivities", "20111231", 4, 5)]   # restated later
    return subs, n


def two_class_rows():
    subs = [sub_row("0002-12-000001", 1002, "10-Q", "20120930", "Q3", "20121101", name="TWO CLASS INC")]
    n = [num_row("0002-12-000001", "EntityCommonStockSharesOutstanding", "20121031", 0, 6e6, "shares",
                 segments="ClassOfStockAxis=CommonClassA", version="dei/2012"),
         num_row("0002-12-000001", "EntityCommonStockSharesOutstanding", "20121031", 0, 4e6, "shares",
                 segments="ClassOfStockAxis=CommonClassB", version="dei/2012")]
    return subs, n


@pytest.fixture
def extracted():
    s1, n1 = acme_rows()
    s2, n2 = two_class_rows()
    raw = fsds_zip(s1 + s2, n1 + n2)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return fsds.extract_quarter(zf)


def test_extract_keeps_periodic_forms_and_wanted_facts(extracted):
    subs, facts = extracted["subs"], extracted["facts"]
    assert set(subs["form"]) == {"10-K", "10-Q", "10-K/A"}
    assert subs["filed"].dtype == "datetime64[ns]" and subs["cik"].dtype == "Int64"
    assert 88888 not in facts["value"].tolist() and 99999 not in facts["value"].tolist()
    dei_dims = facts[(facts["cik"] == 1002) & facts["dim"]]
    assert sorted(dei_dims["value"]) == [4e6, 6e6]
    assert facts["ym"].dtype == "int32" and facts["qtrs"].dtype == "int16"


def test_ttm_from_10q_with_comparatives(extracted):
    snap = fu.snapshots(extracted["facts"], extracted["subs"], ["2013-01-02"])
    a = snap[snap["cik"] == 1001].iloc[0]
    assert a["form"] == "10-Q" and a["period_end"] == pd.Timestamp("2012-09-30")
    assert a["ocf"] == pytest.approx(100 + 110 - 70)          # FY2011 + 9M2012 - 9M2011
    assert a["capex"] == pytest.approx(20 + 15 - 12)
    assert a["fcf"] == pytest.approx(140 - 23)
    assert a["ocf_prev"] == pytest.approx(80 + 70 - 55)       # FY2010 + 9M2011 - 9M2010
    assert a["fcf_prev"] == pytest.approx(95 - (10 + 12 - 9))
    assert a["revenue"] == pytest.approx(1100) and a["revenue_prev"] == pytest.approx(950)
    assert a["net_cash"] == pytest.approx(320 - 40) and bool(a["net_cash_pos"])
    assert a["shares_dei"] == pytest.approx(10.3e6)
    assert a["share_growth"] == pytest.approx(0.01) and bool(a["low_dilution"])
    g_fcf = (117 / 10.1e6) / (82 / 10.0e6) - 1
    assert g_fcf > 1100 / 950 - 1 and bool(a["s1_proxy"])
    assert a["fund_available"] == pd.Timestamp("2012-11-05") and a["fund_available"] < a["start_date"]
    assert a["public_float"] == pytest.approx(4.0e8)
    assert not bool(a["capex_missing"]) and not bool(a["debt_missing"])


def test_ttm_from_10k_and_point_in_time(extracted):
    snap = fu.snapshots(extracted["facts"], extracted["subs"], ["2012-04-02", "2013-02-01"])
    early = snap[(snap["cik"] == 1001) & (snap["start_date"] == pd.Timestamp("2012-04-02"))].iloc[0]
    assert early["form"] == "10-K" and early["ocf"] == pytest.approx(100) and early["ocf_prev"] == pytest.approx(80)
    assert early["fcf"] == pytest.approx(80) and early["shares_dei"] == pytest.approx(10.1e6)
    # after the 10-K/A is public, its restated FY2011 value replaces the original
    late = snap[(snap["cik"] == 1001) & (snap["start_date"] == pd.Timestamp("2013-02-01"))].iloc[0]
    assert late["ocf"] == pytest.approx(5 + 110 - 70)


def test_class_shares_are_summed_and_missing_flows_stay_unknown(extracted):
    snap = fu.snapshots(extracted["facts"], extracted["subs"], ["2013-01-02"])
    b = snap[snap["cik"] == 1002].iloc[0]
    assert b["shares_dei"] == pytest.approx(10e6)
    assert np.isnan(b["ocf"]) and pd.isna(b["fcf_pos"]) and pd.isna(b["s1_proxy"])
    assert bool(b["capex_missing"]) and bool(b["debt_missing"])


def test_stale_filers_drop_out():
    s, n = acme_rows()
    raw = fsds_zip(s, n)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        ex = fsds.extract_quarter(zf)
    snap = fu.snapshots(ex["facts"], ex["subs"], ["2015-01-02"])
    assert snap.empty or 1001 not in set(snap["cik"])


def test_load_caches_per_quarter(tmp_path):
    s, n = acme_rows()
    url = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2012q4.zip"
    client = FakeClient({url: fsds_zip(s, n)})
    from datetime import date
    subs, facts, stats = fsds.load(tmp_path, 2012, 2012, client=client, today=date(2013, 1, 10), log=None)
    assert stats["downloaded"] == 1 and stats["missing"] == 3 and len(facts) > 0
    assert sorted(stats["missing_labels"]) == ["2012q1", "2012q2", "2012q3"]
    again = FakeClient({})
    subs2, facts2, stats2 = fsds.load(tmp_path, 2012, 2012, client=again, today=date(2013, 1, 10), log=None)
    assert stats2["cached"] == 1 and len(facts2) == len(facts)
    assert url not in again.requests                     # served from the Drive cache


def test_paper_a_tags_are_extracted_and_a_schema_change_rebuilds_the_cache(tmp_path):
    s, n = acme_rows()
    k = s[0]                                                          # the FY2011 10-K
    n = n + [num_row(k["adsh"], "OperatingIncomeLoss", k["period"], 4, 7.0e6),
             num_row(k["adsh"], "DepreciationDepletionAndAmortization", k["period"], 4, 2.0e6),
             num_row(k["adsh"], "InterestExpense", k["period"], 4, 1.0e6),
             num_row(k["adsh"], "NetIncomeLoss", k["period"], 4, 3.0e6)]          # not wanted: dropped
    url = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2012q4.zip"
    from datetime import date
    subs, facts, stats = fsds.load(tmp_path, 2012, 2012, client=FakeClient({url: fsds_zip(s, n)}),
                                   today=date(2013, 1, 10), log=None)
    tags = set(facts["tag"].astype(str))
    assert {"OperatingIncomeLoss", "DepreciationDepletionAndAmortization", "InterestExpense"} <= tags
    assert "NetIncomeLoss" not in tags
    snap = fu.snapshots(facts, subs, ["2013-01-02"])
    assert "operating_income" not in snap.columns                    # the AEM features are unchanged
    mark = tmp_path / "fsds" / "fsds_2012q4.done.json"
    meta = json.loads(mark.read_text())
    assert meta["schema"] == fsds.SCHEMA
    meta["schema"] = "older"                                          # a cache built with another tag list
    mark.write_text(json.dumps(meta))
    client = FakeClient({url: fsds_zip(s, n)})
    _s, _f, st = fsds.load(tmp_path, 2012, 2012, client=client, today=date(2013, 1, 10), log=None)
    assert st["rebuilt"] == 1 and st["downloaded"] == 1 and url in client.requests
