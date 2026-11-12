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


def test_results_block_and_failed_gates():
    from aem_research.report import failed_gates, results_block
    assert failed_gates({"prelock": {"fcf_known": "96.7% ok", "split_recall": "75.3% BELOW 90%",
                                     "delisting_unknown": "27.0% ABOVE 25%", "universe_windows": 58652}}) == \
        ["split_recall 75.3% BELOW 90%", "delisting_unknown 27.0% ABOVE 25%"]
    assert failed_gates({}) == ["the run has no pre-lock checks"]
    touch = [{"scope": "pooled", "group": "all", "horizon_years": 3, "windows": 50000, "firms": 3500,
              "touch_3x": 0.081, "touch_10x": 0.009},
             {"scope": "regime", "group": "before 2022", "horizon_years": 3, "windows": 40000, "firms": 3000,
              "touch_3x": 0.09, "touch_10x": 0.01},
             {"scope": "start year", "group": "2014", "horizon_years": 3, "windows": 5000, "firms": 1300,
              "touch_3x": 0.07, "touch_10x": 0.0}]
    buckets = [{"scope": "pooled", "group": "all", "windows": 30000, "firms": 2800, "share_0.1x": 0.2,
                "share_0.4x": 0.25, "share_1x": 0.3, "share_2.5x": 0.15, "share_5x": 0.07, "share_10x": 0.03,
                "mean": 1.37, "mean_lower": 1.21, "mean_upper": 1.42}]
    r = {"release": 8, "generated_utc": "2026-11-09 16:00 UTC", "repo_commit": "abc1234", "notebook_build": "x" * 12,
         "config_sha256": "f" * 64, "lock_status": "locked",
         "lock": {"locked_at_utc": "2026-11-09T15:55:00Z",
                  "data_run": {"release": 7, "repo_commit": "93fe068", "generated_utc": "2026-11-09 15:20 UTC"}},
         "problems": [], "outcomes": {"start_windows": 58652, "complete": 120000},
         "h1": {"touch": touch, "buckets": buckets}, "run": {}}
    lines = results_block(r).splitlines()
    assert lines[0] == "AEM RESULTS v8 – paste everything between the rulers into the chat"
    assert "data run v7 93fe068 2026-11-09 15:20 UTC" in lines[2] and "(locked 2026-11-09T15:55:00Z)" in lines[2]
    assert "locked, locked" not in lines[2]
    assert "(mismatch, locked 2026-11-09T15:55:00Z)" in results_block({**r, "lock_status": "mismatch"}).splitlines()[2]
    assert "(unlocked)" in results_block({**r, "lock_status": "unlocked", "lock": {}}).splitlines()[2]
    assert "CHECKS    lock ok · data run ok · gates all passed in the data run" in lines
    assert "OUTCOMES  start_windows 58,652 · complete 120,000" in lines
    assert "H1 TOUCH  pooled · 3y: 3x 8.1% 10x 0.9% (50,000 windows, 3,500 firms)" in lines
    assert "REGIME    before 2022 · 3y: 3x 9.0% 10x 1.0% (40,000 windows, 3,000 firms)" in lines
    assert ("H1 5Y     pooled · 0.1x 20.0% · 0.4x 25.0% · 1x 30.0% · 2.5x 15.0% · 5x 7.0% · 10x 3.0% · mean 1.37 "
            "(bounds 1.21 to 1.42) · 30,000 windows") in lines
    assert "BY YEAR   3y touch 3x: 2014 7.0%" in lines
    stopped = results_block({**r, "problems": ["the lock names no data run"], "h1": {}, "outcomes": {}}).splitlines()
    assert "CHECKS    STOPPED – the lock names no data run" in stopped and not any(x.startswith("H1") for x in stopped)


def test_explore_block_prints_every_row_with_shares_as_percent_and_stops_on_problems():
    from aem_research.report import explore_block
    r = {"release": 13, "generated_utc": "2026-11-11 12:00 UTC", "repo_commit": "abc1234", "notebook_build": "0" * 12,
         "lock_status_a": "locked", "lock_status_b": "locked", "inputs": {"deals": 685, "core": 471},
         "tables": {"2 margins": [{"group": "core", "n": 120, "target_margin": 0.118, "median_ratio": 0.61, "p": 0.0004}],
                    "7 exit slowdown": [], "_own_history_deals": [{"deal_id": "x"}],
                    "4 own history": [{"group": "all", "n": 300, "median_premium_to_last": 0.31, "p25": -0.02, "p75": float("nan")}]},
         "run": {}}
    lines = explore_block(r).splitlines()
    assert lines[0].startswith("AEM EXPLORE v13") and "Paper A locked · Paper B locked" in lines[2]
    assert "CHECKS    locks ok · data runs ok" in lines and "INPUTS    deals 685 · core 471" in lines
    assert "E2 margins · group core · n 120 · target margin 11.8% · median ratio 0.61 · p <0.001" in lines
    assert "E4 own history · group all · n 300 · median premium to last +31.0% · p25 -2.0% · p75 n/a" in lines
    assert "E7 exit slowdown · no rows" in lines and not any("deal id" in x for x in lines)
    stopped = explore_block({**r, "problems": ["Paper A: the pre-registration is not locked (status unlocked)"], "tables": None})
    assert "CHECKS    STOPPED – Paper A: the pre-registration is not locked" in stopped and "E2 margins" not in stopped


def test_estimates_block_prints_verdicts_and_keeps_years_as_years():
    from aem_research.report import estimates_block
    lr = {"with_signal": 1917, "windows": 52000, "firms": 3800, "hit_rate_with_signal": 0.131, "base_rate": 0.088, "lr": 1.56,
          "ci_low": 1.21, "ci_high": 1.95, "p_one_sided": 0.0005}
    r = {"release": 14, "lock_status": "locked", "lock": {"locked_at_utc": "2026-11-10T13:40:44Z"},
         "h2": {"windows": 52000, "firms": 3800, "base_rate": 0.088, "main": {"H2a": {**lr, "holm_p": 0.001, "verdict": "supported"},
                "H2b": {**lr, "with_signal": 394, "holm_p": 0.2, "verdict": "not supported"}}, "placebo": lr, "h2c": [],
                "h2c_test": {}, "ablation": [{"cohort": "without the fall", **lr}]},
         "touch_years": [{"event": "first 3x touch", "year": 2021, "touches": 1500, "share": 0.31}], "not_computed": ["no minimum price"],
         "run": {}}
    b = estimates_block(r)
    assert "H2a       1,917 windows of 52,000 (3,800 firms) · hit rate 13.1% against 8.8% · LR 1.56 (95% 1.21 to 1.95) · p <0.001 · Holm 0.001 · supported" in b
    assert "ABLATION  without the fall · " in b and "TOUCHED   event first 3x touch · year 2021 · touches 1,500 · share 31.0%" in b
    assert "PENDING   robustness versions that need a rebuilt panel, not computed here: no minimum price" in b
    assert "CHECKS    STOPPED – x" in estimates_block({"release": 14, "problems": ["x"], "run": {}})


def test_the_drive_inventory_describes_folders_and_tables_and_survives_a_broken_file(tmp_path):
    import pandas as pd
    from types import SimpleNamespace
    from aem_research import followup
    from aem_research.io_utils import save_table
    from aem_research.report import followup_block
    (tmp_path / "cache" / "paper_a" / "docs").mkdir(parents=True)
    (tmp_path / "cache" / "paper_a" / "docs" / "a.txt.gz").write_bytes(b"x" * 2000)
    save_table(pd.DataFrame({"cik": [1, 2], "start_date": pd.to_datetime(["2013-07-01", "2023-10-01"]), "kind": ["a", "b"]}),
               tmp_path / "cache" / "features")
    (tmp_path / "cache" / "broken.pkl").write_bytes(b"not a table")
    lines = followup.inventory(SimpleNamespace(work=tmp_path), [tmp_path / "cache" / "features", tmp_path / "cache" / "broken",
                                                              tmp_path / "cache" / "nothing"])
    text = "\\n".join(lines)
    assert "FOLDERS" in lines and "TABLES" in lines and "cache/ · " in text and "docs/ · 1 files" in text
    assert "features: 2 rows × 3 · start_date 2013-07-01..2023-10-01 (2 values) · cik, start_date, kind" in text
    assert "broken: unreadable" in text and "nothing: not on Drive" in text
    assert followup.value_counts_line(pd.DataFrame({"kind": ["a", "a", None]}), "kind") == "kind: a 2 · (missing) 1"
    assert followup.value_counts_line(None, "kind") == "kind: not available" and followup.describe_dir(tmp_path / "x") == ["x: missing"]
    block = followup_block({"release": 15, "lock_status": "locked", "confirmation_status": "locked (C1, C2, C3)", "inventory": lines,
                            "two_way": [{"test": "H2a", "lr": 1.07, "two_way_ci_low": 0.7, "two_way_p": 0.3}],
                            "pending": ["basis B"], "run": {}})
    assert "confirmation plan locked (C1, C2, C3)" in block and "YEARS     test H2a · lr 1.07" in block
    assert "OPEN      still to compute: basis B" in block and "DRIVE     TABLES" in block
    block = followup_block({"release": 16, "inventory": [], "run": {}, "rebuild_files": {"fsds": "63 files, 63 cached, 0 missing, 0 bad"},
                            "reproduction": {"dates_compared": 42, "universe_saved": 58652, "universe_agreement": 0.99873,
                                             "same_fcf_pos": 1.0, "weakest_agreement": 0.99873, "passed": True, "as_of_matches": False},
                            "confirm_problems": ["the 13F data on Drive now end 2026-10-01"],
                            "confirmation": {"C2": {"windows": 5200, "lr": 1.31, "p_one_sided": 0.004, "holm_p": 0.012, "verdict": "confirmed"}},
                            "confirmation_dates": [{"start_date": "2024-07-01", "windows": 1400}]})
    assert "CACHES    fsds: 63 files, 63 cached, 0 missing, 0 bad" in block
    assert ("REBUILD   dates compared 42 · universe saved 58,652 · universe agreement 99.87% · same fcf pos 100.00% · "
            "weakest agreement 99.87% · passed yes · as of matches NO") in block
    assert "CONFIRM   STOPPED – the 13F data on Drive now end 2026-10-01" in block
    assert "C2        windows 5,200 · lr 1.31 · " in block and "verdict confirmed" in block
    assert "C DATES   start date 2024-07-01 · windows 1,400" in block
    assert "DRIVE     FOLDERS" in followup_block({"release": 15, "problems": ["x"], "inventory": lines, "run": {}})     # also when stopped
