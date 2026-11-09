import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from aem_research.data import ftd
from sec_fixtures import FakeClient

TEXT = """SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
20130102|000360206|AAON|2017|AAON INC|20.11
20130103|000360206|AAON|500|AAON INC|20.50
20130104|000360206|AAON|500|AAON INC|.
20130108|000360206|AAON|500|AAON INC|19.00
20130103|123456789|XYZ|100|XYZ | HOLDINGS INC|5.25
20130107|123456789|XYZ|100|XYZ | HOLDINGS INC|6.00
garbage line
Trailer record count 6
"""


def _zip(text: str, name: str = "cnsfails201301a.txt") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_parse_skips_header_trailer_and_rejoins_names_with_a_pipe():
    df = ftd.parse_text(TEXT)
    assert len(df) == 6 and df["description"].iloc[4] == "XYZ | HOLDINGS INC"
    assert df["price"].isna().sum() == 1 and df["settle_date"].dtype == "datetime64[ns]"


def test_extract_dates_the_price_the_business_day_before_settlement():
    with zipfile.ZipFile(io.BytesIO(_zip(TEXT))) as zf:
        out = ftd.extract_file(zf)
    px = out["px"].set_index("cusip")
    a = px.loc["000360206"]
    assert a["first_date"] == pd.Timestamp("2013-01-01") and a["first_price"] == 20.11
    assert a["last_date"] == pd.Timestamp("2013-01-07") and a["last_price"] == 19.0
    assert a["max_price"] == 20.5 and a["min_price"] == 19.0 and a["n_obs"] == 3
    n = out["names"].set_index("cusip")
    assert n.loc["123456789", "symbol"] == "XYZ" and n.loc["000360206", "n"] == 4


def test_file_list_from_the_sec_page_and_from_the_naming_pattern():
    html = ('<a href="/files/data/fails-deliver-data/cnsfails202607b.zip">Jul b</a>'
            '<a href="https://www.sec.gov/files/data/other/cnsfails201910a_0.zip">Oct a</a>'
            '<a href="/files/node/add/data_distribution/cnsfails200906b.zip">old</a>')
    idx = ftd.parse_index(html)
    assert idx["201910a"].endswith("cnsfails201910a_0.zip") and idx["202607b"].startswith("https://www.sec.gov/")
    its = dict(ftd.items(idx))
    assert "200906b" not in its and its["202607b"][0] == idx["202607b"]          # before the start: left out
    labels = ftd.half_month_labels("202601a", today=date(2026, 11, 9))
    assert labels[0] == "202601a" and labels[-1] == "202609a"                      # posting lag of 45 days


def test_load_caches_names_and_reads_prices_for_chosen_cusips(tmp_path):
    url = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails201301a.zip"
    page = '<a href="/files/data/fails-deliver-data/cnsfails201301a.zip">x</a>'.encode()
    client = FakeClient({ftd.INDEX_URL: page, url: _zip(TEXT)})
    names, st = ftd.load(tmp_path, client=client, log=None, start="201301a")
    assert st["file_list"] == "SEC page" and st["downloaded"] == 1 and len(names) == 2
    px = ftd.load_px(tmp_path, cusips=["123456789"])
    assert list(px["cusip"]) == ["123456789"] and px["last_price"].iloc[0] == 6.0
    again = FakeClient({ftd.INDEX_URL: page})
    names2, st2 = ftd.load(tmp_path, client=again, log=None, start="201301a")
    assert st2["cached"] == 1 and url not in again.requests and len(names2) == 2


def test_quarter_end_price_is_the_nearest_within_tolerance():
    px = pd.DataFrame({"cusip": ["A", "A", "B"],
                       "first_date": pd.to_datetime(["2012-12-20", "2013-01-03", "2012-11-01"]),
                       "first_price": [10.0, 11.0, 4.0],
                       "last_date": pd.to_datetime(["2012-12-27", "2013-01-14", "2012-11-02"]),
                       "last_price": [10.5, 12.0, 4.5],
                       "max_date": pd.NaT, "max_price": float("nan"), "min_date": pd.NaT, "min_price": float("nan"),
                       "n_obs": 2})
    q = ftd.quarter_end_prices(px, pd.to_datetime(["2012-12-31"]), tolerance_days=10)
    assert len(q) == 1 and q.iloc[0]["cusip"] == "A" and q.iloc[0]["price"] == 11.0   # Jan 3 is nearer than Dec 27
    last = ftd.last_prices(px).set_index("cusip")["price"]
    assert last["A"] == 12.0 and last["B"] == 4.5


@pytest.mark.parametrize("label,end", [("202602a", "2026-02-15"), ("202602b", "2026-02-28")])
def test_half_month_end(label, end):
    assert ftd._half_end(label) == pd.Timestamp(end)


def test_an_unreadable_page_falls_back_to_the_naming_pattern(tmp_path):
    class Refuses(FakeClient):
        def get(self, url):
            self.requests.append(url)
            if url == ftd.INDEX_URL:
                raise RuntimeError("blocked")
            return self.files.get(url)

    url = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails201301a.zip"
    client = Refuses({url: _zip(TEXT)})
    names, st = ftd.load(tmp_path, client=client, log=None, start="201301a", today=date(2013, 3, 1))
    assert st["file_list"].startswith("naming pattern (SEC page unreadable") and st["downloaded"] == 1
    assert len(names) == 2
