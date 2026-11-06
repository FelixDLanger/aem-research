import gzip
from datetime import date

import pandas as pd
import pytest

from aem_research.data import sec
from aem_research.io_utils import save_table

MASTER = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 2015
Comments:              webmaster@sec.gov

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
1000045|NICHOLAS FINANCIAL INC|10-Q|2015-02-13|edgar/data/1000045/0001193125-15-047903.txt
1000180|SANDISK CORP|25-NSE|2015-03-02|edgar/data/1000180/0001354457-15-000001.txt
1001|PIPE | NAME CO|25|2015-01-20|edgar/data/1001/x.txt
1067983|BERKSHIRE HATHAWAY INC|4|2015-01-05|edgar/data/1067983/y.txt
"""


def test_parse_master_index_plain_and_gzip():
    for raw in (MASTER.encode("latin-1"), gzip.compress(MASTER.encode("latin-1"))):
        df = sec.parse_master_index(raw)
        assert len(df) == 4
        row = df[df["form"] == "25"].iloc[0]
        assert row["company"] == "PIPE | NAME CO" and row["cik"] == 1001
        assert str(df["date_filed"].dtype) == "datetime64[ns]"


def test_quarters_are_complete_only():
    q = sec.quarters(2025, 2027, today=date(2026, 11, 5))
    assert q[0] == (2025, 1) and q[-1] == (2026, 3) and (2026, 4) not in q


class FakeClient:
    def __init__(self, missing=()):
        self.calls, self.missing = [], set(missing)

    def get(self, url):
        self.calls.append(url)
        if any(m in url for m in self.missing):
            return None
        return gzip.compress(MASTER.encode("latin-1"))


def test_load_form25_index_downloads_then_uses_cache(tmp_path):
    client = FakeClient(missing=("2015/QTR2",))
    table, stats = sec.load_form25_index(tmp_path, 2015, 2015, client=client,
                                         today=date(2016, 1, 10), log=None)
    assert stats == {"cached": 0, "downloaded": 3, "missing": 1}
    assert set(table["form"]) == {"25", "25-NSE"} and len(table) == 6
    again, stats2 = sec.load_form25_index(tmp_path, 2015, 2015, offline=True,
                                          today=date(2016, 1, 10), log=None)
    assert stats2 == {"cached": 3, "downloaded": 0, "missing": 1} and len(again) == 6


def test_offline_without_cache_returns_empty(tmp_path):
    table, stats = sec.load_form25_index(tmp_path, 2015, 2015, offline=True, today=date(2016, 1, 10))
    assert table.empty and stats["missing"] == 4


class Resp:
    def __init__(self, code, content=b"ok"):
        self.status_code, self.content = code, content

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, codes):
        self.codes, self.headers_seen = list(codes), []

    def get(self, url, headers=None, timeout=None):
        self.headers_seen.append(headers)
        return Resp(self.codes.pop(0))


def test_client_status_handling(monkeypatch):
    monkeypatch.setattr(sec.time, "sleep", lambda s: None)
    c = sec.SecClient("Test Agent test@example.com", session=FakeSession([200]))
    assert c.get("u") == b"ok"
    assert c.session.headers_seen[0]["User-Agent"] == "Test Agent test@example.com"
    assert sec.SecClient("x", session=FakeSession([404])).get("u") is None
    assert sec.SecClient("x", session=FakeSession([503, 200])).get("u") == b"ok"
    with pytest.raises(sec.SecAccessError, match="SEC_USER_AGENT"):
        sec.SecClient("x", session=FakeSession([403])).get("u")


def test_user_agent_resolution(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert sec.resolve_user_agent() == (sec.GENERIC_USER_AGENT, "generic")
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    assert sec.resolve_user_agent() == ("Jane Doe jane@example.com", "environment")


def test_name_filters():
    assert sec.is_spac_name("Gores Holdings Acquisition Corp")
    assert sec.is_spac_name("Some Blank Check Co")
    assert not sec.is_spac_name("Apple Inc")
    assert sec.is_exchange_name("The NASDAQ Stock Market LLC")
    assert sec.is_exchange_name("NEW YORK STOCK EXCHANGE LLC")
    assert not sec.is_exchange_name("Exchange Income Corp")


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        sec.load_xbrl_fundamentals()
    with pytest.raises(NotImplementedError):
        sec.load_insider_purchases()
