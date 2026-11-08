import pandas as pd
import pytest

from aem_research.data import eodhd

KEY = "secret-key-123"


class _Resp:
    def __init__(self, code, data=None):
        self.status_code, self._data = code, data

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class FakeEodhd:
    """Serves the two EODHD endpoints from dictionaries; records every call without the key."""

    def __init__(self, listed=(), delisted=(), eod=None, script=None):
        self.listed, self.delisted, self.eod = list(listed), list(delisted), eod or {}
        self.script, self.calls = list(script or []), []

    def get(self, url, params=None, timeout=None):
        assert params["api_token"] == KEY and params["fmt"] == "json"
        self.calls.append((url.replace("https://eodhd.com/api/", ""), {k: v for k, v in params.items() if k != "api_token"}))
        if self.script:
            return self.script.pop(0)
        if url.endswith("exchange-symbol-list/US"):
            return _Resp(200, self.delisted if params.get("delisted") == 1 else self.listed)
        code = url.rsplit("/", 1)[-1].replace(".US", "")
        return _Resp(200, self.eod[code]) if code in self.eod else _Resp(404)


def sym(code, exchange="NASDAQ", typ="Common Stock", name=None):
    return {"Code": code, "Name": name or f"{code} Inc", "Country": "USA", "Exchange": exchange, "Currency": "USD",
            "Type": typ, "Isin": None}


def bars(n, start="2014-01-02", px=10.0):
    d = pd.bdate_range(start, periods=n)
    return [{"date": str(x.date()), "open": px, "high": px, "low": px, "close": px * 2, "adjusted_close": px,
             "volume": 1000} for x in d]


def test_client_retries_rate_limits_and_never_leaks_the_key(monkeypatch):
    monkeypatch.setattr(eodhd.time, "sleep", lambda s: None)
    sess = FakeEodhd(script=[_Resp(429), _Resp(200, [{"Code": "A"}])])
    c = eodhd.Client(KEY, session=sess, min_interval=0)
    assert c.get_json("exchange-symbol-list/US") == [{"Code": "A"}] and c.calls == 2
    assert eodhd.Client(KEY, session=FakeEodhd(script=[_Resp(404)]), min_interval=0).get_json("eod/X.US") is None
    with pytest.raises(eodhd.EodhdAccessError) as e:
        eodhd.Client(KEY, session=FakeEodhd(script=[_Resp(401)]), min_interval=0).get_json("eod/X.US")
    assert KEY not in str(e.value) and "EODHD_API_KEY" in str(e.value)


def test_key_comes_from_the_environment_outside_colab(monkeypatch):
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    assert eodhd.resolve_key() is None
    monkeypatch.setenv("EODHD_API_KEY", f" {KEY} ")
    assert eodhd.resolve_key() == KEY


def test_symbol_lists_are_cached_and_the_exchange_map_prefers_delisted_entries(tmp_path):
    sess = FakeEodhd(listed=[sym("AAA"), sym("BBB", "NYSE"), sym("ETF1", typ="ETF")],
                     delisted=[sym("BBB", "NASDAQ", name="Old BBB Corp"), sym("CCC", "NYSE MKT"), sym("DDD", None)])
    c = eodhd.Client(KEY, session=sess, min_interval=0)
    s, st = eodhd.symbol_lists(c, tmp_path, today=pd.Timestamp("2026-11-08"), log=None)
    assert st["source"] == "downloaded" and len(s) == 6 and s["delisted"].sum() == 3 and len(sess.calls) == 2
    assert sess.calls[1][1].get("delisted") == 1
    again = FakeEodhd()
    s2, st2 = eodhd.symbol_lists(eodhd.Client(KEY, session=again, min_interval=0), tmp_path,
                                 today=pd.Timestamp("2026-11-20"), log=None)
    assert st2["source"].startswith("cached") and not again.calls and len(s2) == 6
    em = eodhd.exchange_map(s).set_index("ticker")["exchange"]
    assert em["AAA"] == "nasdaq" and em["BBB"] == "nasdaq" and em["CCC"] == "nyse_amex"   # delisted BBB wins
    assert "ETF1" not in em.index and "DDD" not in em.index


def test_candidates_tie_symbols_to_xbrl_filers_that_simfin_lacks():
    s = pd.DataFrame({"code": ["AAA", "BBB", "CCC", "ETF1", "OTCX", "NULL"],
                      "name": "", "exchange": ["NASDAQ", "NYSE", "NASDAQ", "NYSE ARCA", "OTC", None],
                      "type": ["Common Stock"] * 3 + ["ETF", "Common Stock", "Common Stock"], "isin": None,
                      "delisted": [True, False, True, False, True, True]})
    bridge = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "OTCX", "NULL", "ETF1"],
                           "issuer_cik": [1, 2, 3, 4, 5, 6], "first_seen": pd.Timestamp("2012-01-01"),
                           "last_seen": pd.Timestamp("2016-01-01"), "filings": 5})
    c = eodhd.candidates(s, [bridge], xbrl_ciks=[1, 2, 4, 5, 6], simfin_ciks=[2])
    assert list(c["code"]) == ["AAA", "NULL"]          # BBB in SimFin, CCC not an XBRL filer, OTCX and ETF1 out
    assert c.set_index("code").loc["AAA", "ciks"] == [1]


def test_fetch_saves_chunks_resumes_and_returns_panel_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(eodhd, "CHUNK", 2)
    sess = FakeEodhd(eod={"AAA": bars(5), "NULL": bars(3, "2015-06-01", 4.0), "EMPTY": []})
    c = eodhd.Client(KEY, session=sess, min_interval=0)
    rows, st = eodhd.fetch_eod(["AAA", "NULL", "EMPTY", "GONE"], c, tmp_path, log=None)
    assert st == {"wanted": 4, "cached": 0, "fetched": 2, "empty": 2, "missing": 0, "tickers": 2, "rows": 8}
    assert set(rows["ticker"]) == {"AAA~eod", "NULL~eod"} and (rows["source"] == "eodhd").all()
    a = rows[rows["ticker"] == "AAA~eod"].iloc[0]
    assert a["price"] == 10.0 and a["close_raw"] == 20.0 and rows["date"].dtype == "datetime64[ns]"
    assert sess.calls[0] == ("eod/AAA.US", {"fmt": "json", "from": "2008-01-01"})
    later = FakeEodhd()
    rows2, st2 = eodhd.fetch_eod(["AAA", "NULL", "EMPTY", "GONE"], eodhd.Client(KEY, session=later, min_interval=0),
                                 tmp_path, log=None)
    assert not later.calls and st2["cached"] == 4 and len(rows2) == 8              # empty answers are remembered
    off, st3 = eodhd.fetch_eod(["AAA", "NEW"], None, tmp_path, offline=True, log=None)
    assert st3["missing"] == 1 and set(off["ticker"]) == {"AAA~eod"} and eodhd.cached(tmp_path)


def test_dedupe_keeps_the_other_source_when_both_price_the_company():
    d1, d2 = pd.Timestamp("2015-01-02"), pd.Timestamp("2016-01-04")
    pf = pd.DataFrame({"ticker": ["XYZ", "XYZ~eod", "XYZ~eod", "OLD~eod"], "start_date": [d1, d1, d2, d1]})
    keys = pd.DataFrame({"ticker": pf["ticker"], "start_date": pf["start_date"],
                         "cik": pd.array([7, 7, 7, 9], dtype="Int64"), "cik_source": ["simfin", "bridge", "bridge", "bridge"]})
    pf2, k2 = eodhd.dedupe(pf, keys)
    assert list(zip(pf2["ticker"], pf2["start_date"])) == [("XYZ", d1), ("XYZ~eod", d2), ("OLD~eod", d1)]
    assert len(k2) == 3 and "is_eod" not in k2.columns
