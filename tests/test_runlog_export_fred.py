import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aem_research import features as ft
from aem_research.data import fred
from aem_research.data import sec_datasets as sd
from aem_research.export import write_workbook
from aem_research.report import paste_block
from aem_research.runlog import RunLog, folder_size_gb, versions


# ---------------------------------------------------------------- run log and paste block

def test_runlog_records_failures_and_carries_on(capsys):
    log = RunLog()
    with log.phase("A ok"):
        x = 1 + 1
    with log.phase("B broken"):
        raise ValueError("bad column 'filed'")
    assert x == 2
    assert [p["status"] for p in log.phases] == ["ok", "FAILED"]
    assert log.errors[0]["phase"] == "B broken" and "ValueError: bad column" in log.errors[0]["error"]
    assert any("test_runlog_export_fred.py" in w for w in log.errors[0]["where"])
    assert "Phase 'B broken' failed" in capsys.readouterr().out
    with pytest.raises(RuntimeError):
        with log.must_pass("0 tests"):
            raise RuntimeError("tests failed")
    assert log.phases[-1]["status"] == "STOPPED"
    log.skipped("E outcomes", "not locked")
    d = log.as_dict()
    assert d["phases"][-1]["status"] == "skipped (not locked)" and d["total_minutes"] >= 0
    assert versions()["pandas"] == pd.__version__


def test_paste_block_has_results_and_errors(tmp_path):
    log = RunLog()
    with log.phase("F regime"):
        raise OSError("FRED unreachable")
    report = {"generated_utc": "2026-11-07 09:00 UTC", "repo_commit": "abc1234", "config_sha256": "f" * 64,
              "lock_status": "unlocked", "versions": {"pandas": "2.2.2"},
              "simfin": {"tickers": 5123, "stopped_trading": 1234}, "stooq": {"tickers": 9876},
              "verdict": {"verdict": "optional", "missing_share": 0.061, "coverage": 0.72,
                          "optimistic": {"coverage": 0.76, "verdict": "optional"}},
              "features": {"formation_dates": 44, "as_of": "2026-10-30", "universe_windows": 81400,
                           "tickers_in_universe": 3100, "price_from_13f": 0.88, "attention_signal": "breadth"},
              "prelock": {"fcf_known": "97.0% ok"}, "power": [{"cohort": "h2a", "firms": 400, "min_lr_at_10pct": 1.45}],
              "run": log.as_dict(), "drive_gb": 1.4}
    block = paste_block(report)
    assert block.splitlines()[1].startswith("=====") and block.splitlines()[-1].startswith("=====")
    assert "EODHD OPTIONAL – unpriced 6.1%" in block and "81,400 windows" in block
    assert "POWER     cohort h2a" in block and "ERROR     [F regime] OSError: FRED unreachable" in block
    assert "Drive use 1.4 GB" in block
    (tmp_path / "x.bin").write_bytes(b"0" * 1_000_000)
    assert folder_size_gb(tmp_path) == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------- Excel export

def test_workbook_handles_nullable_types_and_limits(tmp_path):
    df = pd.DataFrame({"ticker": pd.Categorical(["A", "B", None]),
                       "flag": pd.array([True, None, False], dtype="boolean"),
                       "n": pd.array([1, None, 3], dtype="Int64"),
                       "when": pd.to_datetime(["2016-07-01", None, "2017-01-03"]),
                       "x": [1.5, np.nan, 2.0], "obj": [pd.NA, "text", {"k": 1}]})
    path = tmp_path / "out.xlsx"
    written = write_workbook(path, {"Features": df, "Summary": {"verdict": "optional", "nested": {"a": 1}},
                                    "Rows": [{"a": 1}, {"a": 2}], "Skipped": None, "Bad/Name:*": df.head(1)})
    assert written == {"Features": 3, "Summary": 2, "Rows": 2, "BadName": 1}
    back = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    f = back["Features"]
    assert f["ticker"].tolist()[:2] == ["A", "B"] and pd.isna(f["ticker"].iloc[2])
    assert f["flag"].tolist()[0] == True and pd.isna(f["flag"].iloc[1])          # noqa: E712
    assert f["n"].tolist()[0] == 1 and pd.isna(f["n"].iloc[1])
    assert back["Summary"].set_index("item").loc["verdict", "value"] == "optional"


# ---------------------------------------------------------------- FRED

def test_parse_both_fred_layouts_and_missing_values():
    old = "DATE,DFII10\n2013-01-02,-0.62\n2013-01-03,.\n2013-01-04,-0.58\n"
    new = "observation_date,DFII10\n2013-01-02,-0.62\n2013-01-03,\n2013-01-04,-0.58\n"
    for raw in (old, new.encode()):
        s = fred.parse_csv(raw)
        assert s.tolist() == [-0.62, -0.58] and s.index.dtype == "datetime64[ns]" and s.name == "DFII10"
    with pytest.raises(ValueError):
        fred.parse_csv("not a csv at all")


def test_regime_uses_only_values_published_before_the_start_date():
    days = pd.bdate_range("2014-06-02", "2016-07-15")
    s10 = pd.Series(np.arange(len(days), dtype=float), index=days)
    cpi_m = pd.date_range("2014-01-01", "2016-06-01", freq="MS")
    cpi = pd.Series(100 * 1.01 ** np.arange(len(cpi_m)), index=cpi_m)
    series = {"real_yield_10y": s10, "yield_2y": s10 / 100, "credit_spread_baa": s10 * 0 + 2.5, "cpi_nsa": cpi}
    r = fred.regime_at(series, ["2016-07-01"]).iloc[0]
    assert r["real_yield_10y"] == s10[s10.index < "2016-07-01"].iloc[-1]          # Jun 30, not Jul 1
    expected = s10[s10.index < "2016-07-01"].iloc[-1] / 100 - s10[s10.index < "2015-07-01"].iloc[-1] / 100
    assert r["tightening_2y_12m"] == pytest.approx(expected)
    assert r["inflation_cpi_yoy"] == pytest.approx(cpi["2016-05-01"] / cpi["2015-05-01"] - 1)   # May CPI, out mid-June
    assert r["credit_spread_baa"] == 2.5


class _Resp:
    def __init__(self, code, content=b""):
        self.status_code, self.content = code, content


class _Session:
    def __init__(self, ok: dict):
        self.ok, self.calls = ok, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        sid = url.rsplit("=", 1)[-1]
        return _Resp(200, self.ok[sid]) if sid in self.ok else _Resp(404)


def test_fred_load_downloads_caches_and_falls_back(tmp_path):
    csv = b"observation_date,DFII10\n2016-06-30,0.3\n"
    sess = _Session({"DFII10": csv})
    out, stats = fred.load(tmp_path / "cache", {"real_yield_10y": "DFII10", "yield_2y": "DGS2"}, session=sess, log=None)
    assert out["real_yield_10y"].iloc[-1] == 0.3 and stats["DGS2"] == "missing" and stats["DGS2_http"] == 404
    again, st2 = fred.load(tmp_path / "cache", {"real_yield_10y": "DFII10"}, session=_Session({}), log=None)
    assert st2["DFII10"].startswith("cached")                        # fresh cache: no request needed
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "DGS2.csv").write_text("DATE,DGS2\n2016-06-30,0.58\n")
    off, st3 = fred.load(tmp_path / "other", {"yield_2y": "DGS2"}, offline=True, manual_dir=manual, log=None)
    assert off["yield_2y"].iloc[0] == 0.58 and st3["DGS2"].startswith("manual")


# ---------------------------------------------------------------- power, placebo, market regime

def test_power_table_and_market_regime():
    s = pd.Timestamp("2016-07-01")
    f = pd.DataFrame({"in_universe": True, "start_date": s, "cik": np.arange(400), "drawdown": -0.2,
                      "fcf": 1e6, "market_cap": 2e8,
                      "h2a": [True] * 400, "h2b": [True] * 40 + [False] * 360, "h2_placebo": False})
    p = ft.power_table(f).set_index("cohort")
    assert p.loc["h2a", "firms"] == 400 and p.loc["h2b", "firms"] == 40 and np.isnan(p.loc["h2_placebo", "min_lr_at_10pct"])
    # 400 firms at a 10% base rate: p1 = 0.10 + 2.486 * sqrt(0.09 / 400) = 0.1373 -> odds ratio 1.43
    assert p.loc["h2a", "min_lr_at_10pct"] == pytest.approx(1.43, abs=0.01)
    assert p.loc["h2b", "min_lr_at_10pct"] > 2.0                     # small cohorts need large effects
    m = ft.market_regime(f).iloc[0]
    assert m["market_stress"] == -0.2 and m["valuation_fcf_yield"] == pytest.approx(0.005)


def test_bad_zip_is_skipped_and_reported(tmp_path):
    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("t.txt", "a\tb\n1\t2\n")
    items = [("2016q1", ["https://x/2016q1.zip"]), ("2016q2", ["https://x/2016q2.zip"])]

    class C:
        def get(self, url):
            return good.getvalue() if "q1" in url else b"PK\x03\x04truncated"

    tables, stats = sd.process(items, tmp_path, "t", lambda z: {"t": sd.read_member(z, "t", ["a", "b"])},
                               client=C(), log=None)
    assert stats["downloaded"] == 1 and stats["bad"] == 1 and stats["bad_labels"] == ["2016q2"]
    assert len(tables["t"]) == 1
