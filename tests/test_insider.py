import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from aem_research.data import insider
from sec_fixtures import FakeClient, insider_zip, sec_date

_n = 0


def filing(issuer, symbol, owner, rel, trans, code, filed, *, doc="4", plan=None, shares=1000, price=10.0, acq=None):
    """One Form 4 with one transaction; returns (sub, owner, trans) rows."""
    global _n
    _n += 1
    acc = f"0000{issuer:06d}-{int(filed[2:4]):02d}-{_n:06d}"
    sub = {"ACCESSION_NUMBER": acc, "FILING_DATE": sec_date(filed), "PERIOD_OF_REPORT": sec_date(trans),
           "DOCUMENT_TYPE": doc, "ISSUERCIK": issuer, "ISSUERNAME": symbol + " INC", "ISSUERTRADINGSYMBOL": symbol,
           "AFF10B5ONE": plan}
    own = {"ACCESSION_NUMBER": acc, "RPTOWNERCIK": owner, "RPTOWNERNAME": f"OWNER {owner}",
           "RPTOWNER_RELATIONSHIP": rel, "RPTOWNER_CITY": "X", "RPTOWNER_STATE": "NY", "RPTOWNER_STREET1": "1",
           "RPTOWNER_ZIPCODE": "10001"}
    tr = {"ACCESSION_NUMBER": acc, "NONDERIV_TRANS_SK": _n, "SECURITY_TITLE": "Common Stock",
          "TRANS_DATE": sec_date(trans), "TRANS_CODE": code, "TRANS_SHARES": shares, "TRANS_PRICEPERSHARE": price,
          "TRANS_ACQUIRED_DISP_CD": acq or ("A" if code == "P" else "D"), "DIRECT_INDIRECT_OWNERSHIP": "D"}
    return sub, own, tr


def build():
    rows = [
        # O1: director, trades every March 2010-2012 -> routine in 2013
        filing(1001, "ACME", 5001, "Director", "2010-03-10", "S", "2010-03-12"),
        filing(1001, "ACME", 5001, "Director", "2011-03-09", "S", "2011-03-11"),
        filing(1001, "ACME", 5001, "Director", "2012-03-08", "S", "2012-03-12"),
        filing(1001, "ACME", 5001, "Director", "2012-11-08", "P", "2012-11-09"),
        # O2: officer, trades in 2010, 2011, 2012 in different months -> opportunistic
        filing(1001, "ACME", 5002, "Officer", "2010-02-10", "P", "2010-02-11"),
        filing(1001, "ACME", 5002, "Officer", "2011-06-10", "S", "2011-06-13"),
        filing(1001, "ACME", 5002, "Officer", "2012-09-10", "P", "2012-09-11"),
        filing(1001, "ACME", 5002, "Officer", "2012-12-10", "P", "2012-12-11"),
        filing(1001, "ACME", 5002, "Officer", "2012-12-10", "P", "2012-12-20", doc="4/A"),   # repeats the original
        # O3: officer with a short record -> unclassified
        filing(1001, "ACME", 5003, "Officer", "2012-11-10", "P", "2012-11-12"),
        # O4: ten-percent owner only -> not an officer or director
        filing(1001, "ACME", 5004, "TenPercentOwner", "2012-12-03", "P", "2012-12-04"),
        # O5: director buying under a Rule 10b5-1 plan -> excluded
        filing(1001, "ACME", 5005, "Director", "2012-12-05", "P", "2012-12-06", plan="1"),
        # filed on the formation date itself -> not yet known
        filing(1001, "ACME", 5003, "Officer", "2012-12-28", "P", "2013-01-02"),
        # an award (code A) and an option exercise (code M) never count
        filing(1001, "ACME", 5003, "Officer", "2012-12-01", "A", "2012-12-03"),
        filing(1001, "ACME", 5003, "Officer", "2012-12-01", "M", "2012-12-03"),
        # BETA: one routine buyer (May 2010, 2011, 2012), buying again in November
        filing(2002, "BETA-B", 6001, "Director,Officer", "2010-05-03", "P", "2010-05-04"),
        filing(2002, "BETA-B", 6001, "Director,Officer", "2011-05-03", "P", "2011-05-04"),
        filing(2002, "BETA-B", 6001, "Director,Officer", "2012-05-03", "P", "2012-05-04"),
        filing(2002, "BETA-B", 6001, "Director,Officer", "2012-11-20", "P", "2012-11-21"),
    ]
    subs, owners, trans = (list(x) for x in zip(*rows))
    # a malformed line (remark text spilling onto its own line) must be dropped
    subs.append({"ACCESSION_NUMBER": "remark continued here", "FILING_DATE": "x"})
    return subs, owners, trans


@pytest.fixture
def tx():
    raw = insider_zip(*build())
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        out = insider.extract_quarter(zf)
    return out


def test_extract_keeps_open_market_trades_with_roles(tx):
    t = tx["tx"]
    assert set(t["code"]) == {"P", "S"}
    assert t["filing_date"].dtype == "datetime64[ns]" and t["trans_date"].min() == pd.Timestamp("2010-02-10")
    beta = t[t["issuer_cik"] == 2002]
    assert beta["is_director"].all() and beta["is_officer"].all() and set(beta["symbol"]) == {"BETA.B"}
    assert t.loc[t["owner_cik"] == 5005, "plan_10b5_1"].iloc[0] == True  # noqa: E712
    bridge = tx["bridge"]
    assert set(bridge["symbol"]) == {"ACME", "BETA.B"}
    assert bridge.loc[bridge["symbol"] == "ACME", "first_seen"].iloc[0] == pd.Timestamp("2010-02-11")


def test_amendments_are_deduplicated(tx):
    t = insider.dedupe_amendments(tx["tx"])
    o2_dec = t[(t["owner_cik"] == 5002) & (t["trans_date"] == pd.Timestamp("2012-12-10"))]
    assert len(o2_dec) == 1 and o2_dec["doc_type"].iloc[0] == "4"


def test_cohen_malloy_pomorski_classes(tx):
    c = insider.classify(insider.dedupe_amendments(tx["tx"]), [2013])
    cls = {(int(r.owner_cik), int(r.issuer_cik)): r.insider_class for r in c.itertuples()}
    assert cls[(5001, 1001)] == "routine"
    assert cls[(5002, 1001)] == "opportunistic"
    assert cls[(6001, 2002)] == "routine"
    assert (5003, 1001) not in cls                      # short record -> unclassified
    # on purchases only, O1 (a seller) has no record and O2 lacks a 2011 purchase
    p = insider.classify(insider.dedupe_amendments(tx["tx"]), [2013], purchases_only=True)
    pcls = {(int(r.owner_cik), int(r.issuer_cik)): r.insider_class for r in p.itertuples()}
    assert (5001, 1001) not in pcls and (5002, 1001) not in pcls and pcls[(6001, 2002)] == "routine"
    # the 2012 classification cannot use 2012 trades
    c12 = insider.classify(insider.dedupe_amendments(tx["tx"]), [2012])
    assert c12.empty


def test_signals_window_roles_and_classes(tx):
    t = insider.dedupe_amendments(tx["tx"])
    buys = insider.purchases(t)
    assert 5004 not in set(buys["owner_cik"]) and 5005 not in set(buys["owner_cik"])
    classes = insider.classify(t, [2013])
    reports = pd.DataFrame({"cik": [1001], "filed": [pd.Timestamp("2012-11-05")]})
    s = insider.signals(buys, classes, ["2013-01-02"], report_dates=reports)
    acme = s[s["issuer_cik"] == 1001].iloc[0]
    assert acme["insider_n"] == 3                      # O1 routine, O2 opportunistic, O3 unclassified
    assert acme["insider_any"] and acme["insider_cluster"] and acme["insider_opportunistic"]
    assert not acme["insider_routine_only"] and not acme["insider_unclassified_only"]
    assert acme["insider_post_results_cluster"]       # O1 (Nov 8) and O3 (Nov 10) after the Nov 5 10-Q
    assert acme["insider_available"] == pd.Timestamp("2012-12-11") < pd.Timestamp("2013-01-02")
    beta = s[s["issuer_cik"] == 2002].iloc[0]
    assert beta["insider_routine_only"] and not beta["insider_opportunistic"] and not beta["insider_cluster"]
    # a formation date after the window has closed sees nothing
    assert insider.signals(buys, classes, ["2014-01-02"]).empty


def test_load_reads_both_url_prefixes(tmp_path):
    raw = insider_zip(*build())
    old = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2012q4_form345.zip"
    new = "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/2013q1_form345.zip"
    client = FakeClient({old: raw, new: raw})
    tx, bridge, stats = insider.load(tmp_path, 2012, 2013, client=client, today=date(2013, 4, 2), log=None)
    assert stats["downloaded"] == 2 and stats["missing"] == 3
    assert len(tx) == len(insider.dedupe_amendments(insider.dedupe_amendments(tx)))   # idempotent
    assert bridge.loc[bridge["symbol"] == "ACME", "filings"].iloc[0] == 2 * 15
    # the same file served twice is deduplicated down to one copy of each transaction
    assert tx.duplicated(["issuer_cik", "owner_cik", "trans_date", "code", "shares", "price"]).sum() == 0
