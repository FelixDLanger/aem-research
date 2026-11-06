import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from aem_research.data import sec_datasets as sd
from aem_research.data import thirteenf as tf
from sec_fixtures import FakeClient, f13_zip, sec_date, tsv, zip_bytes


def test_f13_file_names_follow_the_sec_listing():
    items = tf.f13_items(date(2026, 11, 6))
    labels = [label for label, _ in items]
    assert labels[0] == "2013q2" and "2023q4" in labels and "2013q1" not in labels
    assert labels[labels.index("2023q4") + 1:] == [
        "01jan2024-29feb2024", "01mar2024-31may2024", "01jun2024-31aug2024", "01sep2024-30nov2024",
        "01dec2024-28feb2025", "01mar2025-31may2025", "01jun2025-31aug2025", "01sep2025-30nov2025",
        "01dec2025-28feb2026", "01mar2026-31may2026", "01jun2026-31aug2026"]
    urls = dict(items)["01jun2026-31aug2026"]
    assert urls[0].endswith("/structureddata/data/form-13f-data-sets/01jun2026-31aug2026_form13f.zip")
    assert "/datastandardsinnovation/" in urls[1]


def test_parse_dates_handles_every_sec_layout():
    s = sd.parse_dates(["20130515", "15-MAY-2013", "2013-05-15", "", None, "garbage"])
    assert s.dtype == "datetime64[ns]"
    assert s.iloc[:3].tolist() == [pd.Timestamp("2013-05-15")] * 3 and s.iloc[3:].isna().all()


def test_read_member_tolerates_layouts():
    raw = zip_bytes({"data/SUBMISSION.tsv": tsv(["ACCESSION_NUMBER", "CIK", "REMARKS"],
                                                [{"ACCESSION_NUMBER": "0001234567-13-000001", "CIK": "NA",
                                                  "REMARKS": 'says "hi"'}])})
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        assert sd.member(zf, "submission") == "data/SUBMISSION.tsv"
        df = sd.read_member(zf, "SUBMISSION", ["ACCESSION_NUMBER", "CIK", "MISSING_COL"])
        assert list(df.columns) == ["accession_number", "cik", "missing_col"]
        assert df["cik"].iloc[0] == "NA" and df["missing_col"].isna().all()   # "NA" stays a string
        assert sd.read_member(zf, "NOPE", ["A"]).empty


def holdings():
    subs = [
        {"ACCESSION_NUMBER": "0000900001-16-000001", "FILING_DATE": sec_date("2016-05-10"), "SUBMISSIONTYPE": "13F-HR",
         "CIK": 900001, "PERIODOFREPORT": sec_date("2016-03-31")},
        {"ACCESSION_NUMBER": "0000900002-16-000001", "FILING_DATE": sec_date("2016-05-16"), "SUBMISSIONTYPE": "13F-HR",
         "CIK": 900002, "PERIODOFREPORT": sec_date("2016-03-31")},
        # filed after the Jul 1 formation that uses this quarter -> not counted
        {"ACCESSION_NUMBER": "0000900003-16-000001", "FILING_DATE": sec_date("2016-07-15"), "SUBMISSIONTYPE": "13F-HR",
         "CIK": 900003, "PERIODOFREPORT": sec_date("2016-03-31")},
        # an amendment never counts (the filer was counted on its original)
        {"ACCESSION_NUMBER": "0000900001-16-000002", "FILING_DATE": sec_date("2016-06-01"),
         "SUBMISSIONTYPE": "13F-HR/A", "CIK": 900001, "PERIODOFREPORT": sec_date("2016-03-31")},
    ]
    info = []
    for acc in [s["ACCESSION_NUMBER"] for s in subs]:
        info += [{"ACCESSION_NUMBER": acc, "NAMEOFISSUER": "ACME CORP", "TITLEOFCLASS": "COM", "CUSIP": "000123105",
                  "SSHPRNAMT": 100, "SSHPRNAMTTYPE": "SH"},
                 {"ACCESSION_NUMBER": acc, "NAMEOFISSUER": "ACME CORP", "TITLEOFCLASS": "CALL", "CUSIP": "000123105",
                  "SSHPRNAMT": 10, "SSHPRNAMTTYPE": "SH", "PUTCALL": "Call"},
                 {"ACCESSION_NUMBER": acc, "NAMEOFISSUER": "ACME CORP", "TITLEOFCLASS": "NOTE", "CUSIP": "000123AB1",
                  "SSHPRNAMT": 1000, "SSHPRNAMTTYPE": "PRN"}]
    # the second filer holds ACME twice (two managers) and Beta once; CUSIP reported without leading zeros
    info += [{"ACCESSION_NUMBER": "0000900002-16-000001", "NAMEOFISSUER": "ACME CORP", "CUSIP": "123105",
              "SSHPRNAMT": 50, "SSHPRNAMTTYPE": "SH"},
             {"ACCESSION_NUMBER": "0000900002-16-000001", "NAMEOFISSUER": "BETA INTL HLDGS INC", "CUSIP": "08765X109",
              "SSHPRNAMT": 5, "SSHPRNAMTTYPE": "SH"}]
    return subs, info


def test_breadth_counts_filers_once_and_point_in_time(tmp_path):
    raw = f13_zip(*holdings())
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        ex = tf.extract_file(zf)
    b = ex["breadth"].set_index(["cusip", "period"])["n_filers"]
    assert b[("000123105", pd.Timestamp("2016-03-31"))] == 2       # filers 1 and 2; 3 filed too late
    assert b[("08765X109", pd.Timestamp("2016-03-31"))] == 1
    assert "000123AB1" not in {c for c, _ in b.index}               # principal amounts are not shares
    url = tf.F13_PREFIXES[0] + "2016q2_form13f.zip"
    breadth, names, stats = tf.load(tmp_path, client=FakeClient({url: raw}), today=date(2016, 7, 2), log=None)
    assert stats["downloaded"] == 1 and set(names["cusip"]) == {"000123105", "08765X109"}
    companies = pd.DataFrame({"cik": [1001, 2002], "isin": ["US0001231051", None]})
    filer_names = pd.DataFrame({"cik": [2002], "name": ["Beta International Holdings, Inc."]})
    bridge = tf.cusip_bridge(companies, names, filer_names)
    m = dict(zip(bridge["cusip"], zip(bridge["cik"], bridge["method"])))
    assert m["000123105"] == (1001, "isin")
    assert m["08765X109"] == (2002, "name")                        # INTL HLDGS ~ International Holdings
    at = tf.breadth_at(breadth, bridge, ["2016-07-05", "2016-04-01"])
    row = at[(at["cik"] == 1001)].iloc[0]
    assert row["breadth"] == 2 and row["breadth_period"] == pd.Timestamp("2016-03-31")
    assert row["breadth_available"] == pd.Timestamp("2016-07-01") <= row["start_date"]
    assert len(at[at["start_date"] == pd.Timestamp("2016-04-01")]) == 0   # Mar quarter not usable yet


@pytest.mark.parametrize("d,q", [("2016-07-01", "2016-03-31"), ("2016-07-05", "2016-03-31"),
                                 ("2016-06-30", "2015-12-31"), ("2016-01-04", "2015-09-30")])
def test_quarter_for_formation(d, q):
    assert tf.quarter_for_formation(d) == pd.Timestamp(q)


def test_process_uses_hand_downloaded_zips(tmp_path):
    raw = f13_zip(*holdings())
    local = tmp_path / "manual"
    local.mkdir()
    (local / "2016q2_form13f.zip").write_bytes(raw)
    client = FakeClient({})
    b, n, stats = tf.load(tmp_path / "drive", client=client, today=date(2016, 7, 2), log=None, local_dir=local)
    assert stats["local"] == 1 and len(b) == 2
    assert not any("2016q2" in u for u in client.requests)


def test_implied_unadjusted_prices_and_value_units():
    subs = [{"ACCESSION_NUMBER": f"0000{900000 + k:06d}-16-000001", "FILING_DATE": sec_date("2016-05-10"),
             "SUBMISSIONTYPE": "13F-HR", "CIK": 900000 + k, "PERIODOFREPORT": sec_date("2016-03-31")} for k in range(3)]
    subs.append({"ACCESSION_NUMBER": "0000900009-23-000001", "FILING_DATE": sec_date("2023-02-10"),
                 "SUBMISSIONTYPE": "13F-HR", "CIK": 900009, "PERIODOFREPORT": sec_date("2022-12-31")})
    info = [{"ACCESSION_NUMBER": s["ACCESSION_NUMBER"], "NAMEOFISSUER": "ACME CORP", "CUSIP": "000123105",
             "SSHPRNAMTTYPE": "SH", "SSHPRNAMT": 200, "VALUE": 2} for s in subs[:3]]           # thousands: $10
    info[2]["VALUE"] = 999                                                                 # one bad filer
    info.append({"ACCESSION_NUMBER": subs[3]["ACCESSION_NUMBER"], "NAMEOFISSUER": "ACME CORP",
                 "CUSIP": "000123105", "SSHPRNAMTTYPE": "SH", "SSHPRNAMT": 100, "VALUE": 1200})  # dollars: $12
    with zipfile.ZipFile(io.BytesIO(f13_zip(subs, info))) as zf:
        p = tf.extract_file(zf)["prices"].set_index("period")
    assert p.loc[pd.Timestamp("2016-03-31"), "price_13f"] == pytest.approx(10.0)      # median ignores the outlier
    assert p.loc[pd.Timestamp("2016-03-31"), "n_price_filers"] == 3
    assert p.loc[pd.Timestamp("2022-12-31"), "price_13f"] == pytest.approx(12.0)
    breadth = pd.DataFrame({"cusip": ["000123105"] * 2, "period": [pd.Timestamp("2016-03-31"), pd.Timestamp("2022-12-31")],
                            "n_filers": [3, 1], "price_13f": [10.0, 12.0], "n_price_filers": [3, 1]})
    bridge = pd.DataFrame({"cusip": ["000123105"], "cik": [1001], "method": ["isin"]})
    at = tf.prices_at(breadth, bridge, ["2016-07-01", "2023-04-03"])
    assert at["price_13f"].tolist() == [10.0]                # one filer is not enough for 2022Q4
