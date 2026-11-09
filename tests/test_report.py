from aem_research.report import render_markdown


def test_render_markdown_includes_verdict_and_sections():
    report = {"generated_utc": "2026-11-05 16:00 UTC", "config_sha256": "ab" * 32, "lock_status": "unlocked",
              "verdict": {"verdict": "optional", "reason": "About 8% unpriced.", "basis": "Form 25",
                          "optimistic": {"coverage": 0.9, "verdict": "not needed"}},
              "simfin": {"tickers": 5000, "files": {"x": 1}}, "stooq": {}}
    md = render_markdown(report)
    assert "**OPTIONAL** – About 8% unpriced." in md
    assert "coverage 90%, verdict NOT NEEDED" in md
    assert "## SimFin cache" in md and "- tickers: 5000" in md and "files" not in md
    assert "## Stooq bulk" not in md          # empty sections are skipped


def test_paste_block_shows_the_sec_panel_lines():
    from aem_research.report import paste_block
    report = {"generated_utc": "2026-11-09 01:00 UTC", "config_sha256": "ab" * 32, "lock_status": "unlocked",
              "features": {"formation_dates": 42, "as_of": "2026-07-01", "universe_windows": 50000,
                           "firms_in_universe": 3500, "price_from_13f": 0.99, "attention_signal": "breadth",
                           "price_basis": "SEC panel (A)"},
              "sec_prices": {"companies": 9000, "as_of": "2026-07-01"}, "splits": {"from_holders": 812},
              "sec_check": {"returns_agree": 0.97, "split_recall": 0.93}, "universe_b": {"sec_fill_windows": 9000},
              "calibration": {"3y_3x": 0.78}, "base_rates_b": [{"horizon_years": 3, "touch_3x": 0.12}]}
    block = paste_block(report)
    assert "UNIVERSE  42 start dates to 2026-07-01 · 50,000 windows · 3,500 firms" in block
    assert "basis SEC panel (A)" in block and "SEC PANEL companies 9000" in block
    assert "SPLITS    from_holders 812" in block and "SEC CHECK returns_agree 97.0%" in block
    assert "ROBUST B  sec_fill_windows 9000" in block and "CALIB     3y_3x 78.0%" in block
    assert "RESULT B  horizon_years 3 · touch_3x 0.12" in block
