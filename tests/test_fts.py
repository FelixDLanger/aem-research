import json
from datetime import date
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import pytest

from aem_research.data import fts


def _hit(adsh, fname, ciks, form, file_date, file_type=None, names=None):
    return {"_id": f"{adsh}:{fname}",
            "_source": {"ciks": [f"{c:010d}" for c in ciks], "display_names": names or [f"ACME CORP  (CIK {ciks[0]:010d})"],
                        "form": form, "root_forms": [form], "file_type": file_type or form,
                        "file_date": file_date, "adsh": adsh, "sics": ["3571"]}}


class FakeFTS:
    """Serves search pages like efts.sec.gov from a list of hits; 100 hits per page."""

    def __init__(self, hits, page=100, total_override=None):
        self.hits, self.page, self.total_override, self.urls = hits, page, total_override, []

    def get(self, url):
        self.urls.append(url)
        qs = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        a, b = pd.Timestamp(qs.get("startdt", "2001-01-01")), pd.Timestamp(qs.get("enddt", "2030-12-31"))
        sel = [h for h in self.hits if a <= pd.Timestamp(h["_source"]["file_date"]) <= b]
        if "forms" in qs:
            sel = [h for h in sel if h["_source"]["form"] in qs["forms"].split(",")]
        total = len(sel) if self.total_override is None else self.total_override(a, b, len(sel))
        off = int(qs.get("from", 0))
        body = {"took": 3, "hits": {"total": {"value": total, "relation": "eq" if total < 10_000 else "gte"},
                                    "hits": sel[off:off + self.page]}}
        return json.dumps(body).encode()


def test_url_encodes_phrases_and_keeps_form_lists_readable():
    url = fts.build_url('"equity commitment letter" OR "equity commitment letters"', ["DEFM14A", "SC 14D9"],
                        "2015-01-01", "2015-12-31", offset=200, ciks=1234)
    assert url.startswith("https://efts.sec.gov/LATEST/search-index?q=%22equity%20commitment%20letter%22%20OR%20")
    qs = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    assert qs["forms"] == "DEFM14A,SC 14D9" and "forms=DEFM14A,SC%2014D9" in url
    assert qs["startdt"] == "2015-01-01" and qs["enddt"] == "2015-12-31" and qs["from"] == "200"
    assert qs["ciks"] == "0000001234" and unquote(qs["q"]).count('"') == 4


def test_parse_page_reads_both_total_styles_and_rejects_html():
    rows, total, rel = fts.parse_page(json.dumps({"hits": {"total": {"value": 2, "relation": "eq"}, "hits": [
        _hit("0001193125-15-000001", "d1defm14a.htm", [1001, 2002], "DEFM14A", "2015-03-02"),
        {"_id": "0001193125-15-000002:ex99.htm", "_source": {"ciks": ["0000001001"], "file_date": "2015-04-01",
                                                             "file_type": "EX-99.(D)(1)"}}]}}))
    assert total == 2 and rel == "eq" and rows[0]["ciks"] == "1001;2002" and rows[0]["file_name"] == "d1defm14a.htm"
    assert rows[1]["adsh"] == "0001193125-15-000002" and rows[1]["form"] is None and rows[1]["file_type"] == "EX-99.(D)(1)"
    _r, total2, _ = fts.parse_page(b'{"hits": {"total": 7, "hits": []}}')
    assert total2 == 7
    with pytest.raises(ValueError, match="did not return JSON"):
        fts.parse_page(b"<html>Your request has been blocked</html>")
    with pytest.raises(ValueError, match="no hits object"):
        fts.parse_page(b'{"error": "rate limited"}')


def test_search_pages_by_year_caches_final_windows_and_refetches_recent_ones(tmp_path):
    hits = [_hit(f"0000000000-1{y % 10}-{i:06d}", f"d{i}.htm", [1000 + i], "DEFM14A", f"{y}-0{1 + i % 9}-15")
            for y in (2015, 2016) for i in range(150)]
    hits += [_hit("0000000000-26-000001", "late.htm", [5], "DEFM14A", "2026-10-20")]
    client = FakeFTS(hits)
    out, st = fts.search('"equity commitment letter"', ["DEFM14A"], "2015-01-01", "2026-11-06", client=client,
                         cache_dir=tmp_path, today=date(2026, 11, 6), log=None)
    assert len(out) == 301 and st["windows"] == 12 and st["fetched"] == 12 and st["pages"] >= 14
    assert out["file_date"].dtype == "datetime64[ns]" and set(out["query"]) == {'"equity commitment letter"'}
    again = FakeFTS(hits)
    out2, st2 = fts.search('"equity commitment letter"', ["DEFM14A"], "2015-01-01", "2026-11-06", client=again,
                           cache_dir=tmp_path, today=date(2026, 11, 6), log=None)
    assert len(out2) == 301 and st2["cached"] == 11 and st2["fetched"] == 1           # only 2026 is not final
    assert all("startdt=2026-01-01" in u for u in again.urls)
    off, st3 = fts.search('"equity commitment letter"', ["DEFM14A"], "2015-01-01", "2026-11-06", offline=True,
                          cache_dir=tmp_path, today=date(2026, 11, 6), log=None)
    assert len(off) == 301 and st3["cached"] == 12                                   # offline uses any cache


def test_a_window_at_the_cap_is_split():
    hits = [_hit(f"0000000000-15-{i:06d}", f"d{i}.htm", [i + 1], "DEFM14A", f"2015-{1 + i % 12:02d}-10") for i in range(40)]

    def capped(a, b, n):                    # the full year reports 10,000+, halves report their true size
        return 10_000 if (b - a).days > 300 else n

    out, st = fts.search('"x"', ["DEFM14A"], "2015-01-01", "2015-12-31", client=FakeFTS(hits, total_override=capped),
                         today=date(2026, 1, 1), log=None)
    assert st["split"] == 1 and len(out) == 40


def test_search_many_caches_each_query_including_empty_answers(tmp_path):
    hits = [_hit("0000000000-19-000001", "d8k.htm", [777], "8-K", "2019-05-01", names=["BUYER INC (CIK 0000000777)"])]
    qs = [{"key": "deal1", "q": '"Acme Widgets" "completion of acquisition"', "forms": ["8-K"],
           "start": "2018-01-01", "end": "2026-11-06"},
          {"key": "deal2", "q": '"Nobody Corp" "completion of acquisition"', "forms": ["8-K"],
           "start": "2030-01-01", "end": "2030-12-31"}]
    client = FakeFTS(hits)
    out, st = fts.search_many(qs, client=client, cache_path=tmp_path / "exits", log=None)
    assert st["fetched"] == 2 and list(out["key"]) == ["deal1"] and out["ciks"].iloc[0] == "777"
    again = FakeFTS(hits)
    out2, st2 = fts.search_many(qs, client=again, cache_path=tmp_path / "exits", log=None)
    assert st2["cached"] == 2 and not again.urls and len(out2) == 1
    _o, st3 = fts.search_many(qs + [{"key": "deal3", "q": '"New"', "forms": ["8-K"]}], offline=True,
                              cache_path=tmp_path / "exits", log=None)
    assert st3["missing"] == 1 and st3["cached"] == 2


def test_a_404_says_the_endpoint_may_have_moved():
    class Gone:
        def get(self, url):
            return None
    with pytest.raises(ValueError, match="404"):
        fts.search('"x"', [], "2015-01-01", "2015-12-31", client=Gone(), today=date(2026, 1, 1), log=None)


def test_hits_outside_the_window_are_dropped_even_if_the_endpoint_ignores_dates():
    hits = [_hit(f"0000000000-1{y % 10}-000001", f"d{y}.htm", [y], "DEFM14A", f"{y}-06-01") for y in (2014, 2015, 2016)]

    class IgnoresDates(FakeFTS):
        def get(self, url):
            self.urls.append(url)
            return json.dumps({"hits": {"total": {"value": len(self.hits), "relation": "eq"}, "hits": self.hits}}).encode()

    client = IgnoresDates(hits)
    out, st = fts.search('"x"', ["DEFM14A"], "2014-01-01", "2016-12-31", client=client, today=date(2026, 1, 1), log=None)
    assert sorted(out["file_name"]) == ["d2014.htm", "d2015.htm", "d2016.htm"] and st["windows"] == 3
    assert all("dateRange=custom" in u for u in client.urls)
