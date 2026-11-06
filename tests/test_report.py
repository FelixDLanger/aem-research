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
