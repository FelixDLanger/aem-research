import os
import time
import zipfile

import pandas as pd
import pytest

from aem_research.data import stooq

HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


def member_text(ticker, rows):
    lines = [f"{ticker},D,{d},000000,{c},{c * 1.02:.4f},{c * 0.98:.4f},{c},1000,0" for d, c in rows]
    return HEADER + "\n".join(lines) + "\n"


def make_zip(path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def standard_members():
    return {
        "data/daily/us/nasdaq stocks/1/aapl.us.txt": member_text("AAPL.US", [("20040105", 1.0), ("20150102", 27.3), ("20150105", 26.5)]),
        "data/daily/us/nyse stocks/2/brk-b.us.txt": member_text("BRK-B.US", [("20150102", 150.0), ("20150105", 149.0)]),
        "data/daily/us/nasdaq stocks/2/gone.us.txt": member_text("GONE.US", [("20120103", 5.0), ("20130628", 0.5)]),
        "data/daily/us/nasdaq etfs/qqq.us.txt": member_text("QQQ.US", [("20150102", 100.0)]),
        "data/daily/us/nyse stocks/1/empty.us.txt": HEADER,
    }


def test_group_of():
    assert stooq.group_of("data/daily/us/nasdaq stocks/1/aapl.us.txt") == "nasdaq stocks"
    assert stooq.group_of("data/daily/us/nasdaq etfs/qqq.us.txt") == "nasdaq etfs"
    assert stooq.group_of("readme.txt") == "unknown"


def test_parse_member_filters_and_types():
    raw = member_text("AAPL.US", [("20040105", 1.0), ("20150102", 27.3)]).encode()
    df = stooq.parse_member(raw, min_date="2005-01-01")
    assert len(df) == 1 and df["date"].iloc[0] == pd.Timestamp("2015-01-02")
    assert str(df["date"].dtype) == "datetime64[ns]" and str(df["close"].dtype) == "float32"
    assert stooq.parse_member(HEADER.encode()) is None
    assert stooq.parse_member(b"") is None
    weekly = raw.replace(b",D,", b",W,")
    assert stooq.parse_member(weekly) is None


def test_parse_bulk_zip_keeps_stock_folders_only(tmp_path):
    zp = make_zip(tmp_path / "d_us_txt.zip", standard_members())
    prices, groups, stats = stooq.parse_bulk_zip(zp, min_date="2005-01-01", log=None)
    assert set(prices["ticker"].astype(str)) == {"AAPL", "BRK.B", "GONE"}
    assert stats == {"members": 5, "parsed": 3, "empty": 1, "skipped_group": 1}
    assert dict(zip(groups["ticker"], groups["group"]))["BRK.B"] == "nyse stocks"
    assert prices["date"].min() >= pd.Timestamp("2005-01-01")


def test_find_bulk_zip_prefers_newest(tmp_path):
    a = make_zip(tmp_path / "d_us_txt.zip", standard_members())
    b = make_zip(tmp_path / "d_us_txt (1).zip", standard_members())
    old = time.time() - 3600
    os.utime(a, (old, old))
    assert stooq.find_bulk_zip(tmp_path) == b
    assert stooq.find_bulk_zip(tmp_path / "nope") is None


def test_load_or_build_caches_and_rebuilds_on_change(tmp_path):
    sdir, cdir = tmp_path / "stooq", tmp_path / "cache"
    sdir.mkdir()
    assert stooq.load_or_build(sdir, cdir, log=None) is None
    make_zip(sdir / "d_us_txt.zip", standard_members())
    first = stooq.load_or_build(sdir, cdir, log=None)
    assert not first.from_cache and len(first.prices) == 6
    second = stooq.load_or_build(sdir, cdir, log=None)
    assert second.from_cache and len(second.prices) == len(first.prices)
    members = standard_members()
    members["data/daily/us/nasdaq stocks/3/new.us.txt"] = member_text("NEW.US", [("20200102", 3.0)])
    make_zip(sdir / "d_us_txt.zip", members)
    third = stooq.load_or_build(sdir, cdir, log=None)
    assert not third.from_cache and "NEW" in set(third.prices["ticker"].astype(str))


def test_per_symbol_api_is_not_implemented():
    with pytest.raises(NotImplementedError):
        stooq.fetch_symbol_api("aapl.us")
