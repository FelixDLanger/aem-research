from datetime import date

import numpy as np
import pandas as pd
import pytest

from aem_research.data import wsib

# Three rows exactly as the Sep 30, 2025 report prints them, plus a subtotal over those three
PAGE = """Washington State Investment Board
Performance Summary By Strategy (Base)
Investment Name Initial Investment Date Capital Committed Paid-In Capital (A) Unfunded Commitment Current Market Value (B)
Corporate Finance/Buyout - Large
Advent International GPE VII-B, L.P. 8/27/2012 400,000,000 384,020,122 16,000,020 32,117,988 663,657,497 695,775,485 1.8x 311,755,363 13.14%
Affinity Asia Pacific Fund V, L.P. 12/11/2018 350,000,000 292,518,652 99,957,702 257,675,950 101,813,408 359,489,358 1.2x 66,970,707 7.51%
Apax Europe V-A, L.P. 4/19/2001 111,912,000 144,302,068 - - 303,563,931 303,563,931 2.1x 159,261,863 36.19%
Subtotal Corporate Finance/Buyout - Large 861,912,000 820,840,842 115,957,722 289,793,938 1,069,034,836 1,358,828,774 1.7x 537,987,933 14.02%
Venture Capital
Some Very Long Venture Partnership Name That
II, L.P. 3/1/2021 $ 25,000,000 $ 20,000,000 5,000,000 13,626,472 - 13,626,472 0.7x (6,373,528) (0.60%)
Young Fund 2024, L.P. 6/30/2024 10,000,000 1,000,000 9,000,000 1,000,000 - 1,000,000 1.0x - N/M
Subtotal Venture Capital 35,000,000 21,000,000 14,000,000 14,626,472 - 14,626,472 0.7x (6,373,528) (0.55%)
Late Row After Subtotals, L.P. 1/2/2015 5,000,000 5,000,000 - 1,000,000 6,000,000 7,000,000 1.4x 2,000,000 6.00%
Total Private Equity $ 901,912,000 $ 846,840,842 129,957,722 305,420,410 1,075,034,836 1,380,455,246 1.6x 533,614,404 13.80%
"""


def test_rows_values_and_strategies_parse_exactly():
    df, st = wsib.parse_lines(PAGE.splitlines(), "2025-09-30")
    assert st["rows"] == 6 and st["subtotals"] == 2 and st["subtotals_matched"] == 2 and st["unparsed_dated_lines"] == 0
    a = df.iloc[0]
    assert a["name"] == "Advent International GPE VII-B, L.P." and a["strategy"] == "Corporate Finance/Buyout - Large"
    assert a["initial_investment_date"] == pd.Timestamp("2012-08-27") and a["vintage_year"] == 2012
    assert (a["committed"], a["paid_in"], a["unfunded"], a["market_value"]) == (400e6, 384020122, 16000020, 32117988)
    assert (a["distributed"], a["total_value"], a["gain"]) == (663657497, 695775485, 311755363)
    assert a["multiple_printed"] == 1.8 and a["net_irr"] == pytest.approx(0.1314)
    assert a["tvpi"] == pytest.approx(695775485 / 384020122)
    apax = df[df["name"].str.startswith("Apax")].iloc[0]
    assert apax["unfunded"] == 0 and apax["market_value"] == 0 and apax["initial_investment_date"].year == 2001
    vc = df[df["strategy"] == "Venture Capital"]
    assert list(vc["name"]) == ["Some Very Long Venture Partnership Name That II, L.P.", "Young Fund 2024, L.P."]
    assert vc.iloc[0]["gain"] == -6373528 and vc.iloc[0]["net_irr"] == pytest.approx(-0.006)
    assert np.isnan(vc.iloc[1]["net_irr"]) and vc.iloc[1]["gain"] == 0          # "N/M" and "-"
    late = df[df["name"].str.startswith("Late Row")].iloc[0]
    assert pd.isna(late["strategy"]) and st["unassigned_rows"] == 1
    assert df["report_date"].unique()[0] == pd.Timestamp("2025-09-30")


def test_a_bad_row_is_counted_not_guessed():
    df, st = wsib.parse_lines(["Odd Fund, L.P. 5/5/2015 100 200 1.0x",              # too few columns
                               "Subtotal Odd 100 200 300 400 500 600 1.0x 700 8.00%"], "2025-09-30")
    assert len(df) == 0 and st["unparsed_dated_lines"] == 1


class _Resp:
    def __init__(self, code, content=b""):
        self.status_code, self.content = code, content


class _Session:
    def __init__(self, served):
        self.served, self.calls = served, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        assert "@" not in (headers or {}).get("User-Agent", "")            # no personal e-mail in requests
        return _Resp(200, self.served[url]) if url in self.served else _Resp(404)


def test_load_downloads_keeps_raw_pdfs_and_reparses_after_a_parser_change(tmp_path, monkeypatch):
    url = "https://sib.wa.gov/docs/reports/quarterly/ir093025.pdf"           # www host 404s, bare host serves
    sess = _Session({url: b"%PDF-1.4 fake"})
    lines = lambda raw: PAGE.splitlines()                                     # noqa: E731
    funds, st = wsib.load(tmp_path / "wsib", start_year=2025, today=date(2026, 1, 15), session=sess,
                          to_lines=lines, log=None)
    assert st["reports"] == 3 and st["downloaded"] == 1 and st["missing"] == 2 and st["rows"] == 6
    assert (tmp_path / "wsib" / "raw" / "ir093025.pdf").exists()
    assert sorted(st["missing_reports"]) == ["20250331", "20250630"]
    again = _Session({})
    _f, st2 = wsib.load(tmp_path / "wsib", start_year=2025, today=date(2026, 1, 15), session=again,
                        to_lines=lines, log=None)
    assert st2["cached"] == 1 and not any("093025" in u for u in again.calls)
    monkeypatch.setattr(wsib, "PARSER_VERSION", wsib.PARSER_VERSION + 1)
    _f, st3 = wsib.load(tmp_path / "wsib", start_year=2025, today=date(2026, 1, 15), session=_Session({}),
                        to_lines=lines, log=None)
    assert st3["parsed"] == 1 and st3["raw"] == 1                              # re-read from Drive, no download
    c = wsib.counts(funds)
    assert set(c.columns) == {"strategy", "vintage_year", "funds", "committed"} and c["funds"].sum() == 6
    assert "net_irr" not in c.columns


def test_unreadable_layout_saves_a_sample_for_diagnosis(tmp_path):
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "ir093025.pdf").write_bytes(b"%PDF-1.4 fake")
    _f, st = wsib.load(tmp_path / "wsib", start_year=2025, today=date(2026, 1, 15), offline=True,
                       manual_dir=manual, to_lines=lambda raw: ["A different layout", "with no fund rows"], log=None)
    assert st["failed"] == 1 and "no fund rows" in st["failed_reports"][0]
    assert (tmp_path / "wsib" / "wsib_20250930_sample.txt").read_text().startswith("A different layout")


def test_real_pdf_text_round_trip():
    pytest.importorskip("pdfplumber")
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.lib.pagesizes import landscape, letter
    import io
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    y = 560
    for line in PAGE.splitlines():
        c.setFont("Helvetica", 6)
        c.drawString(20, y, line)
        y -= 12
    c.save()
    df, st = wsib.parse_lines(wsib.pdf_lines(buf.getvalue()), "2025-09-30")
    assert st["rows"] == 6 and st["subtotals_matched"] == 2 and df.iloc[0]["committed"] == 400e6
