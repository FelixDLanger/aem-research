import shutil
from pathlib import Path

import pytest

from aem_research import takeprivate as tp
from aem_research.config import PreregistrationError, load_config, lock_config

ROOT = Path(__file__).resolve().parents[1]
CFG, LOCK = ROOT / "config" / "paper_a.yaml", ROOT / "config" / "paper_a_LOCK.json"


def test_paper_a_config_is_final_with_its_own_lock_file_and_a_plan_for_every_test():
    cfg = load_config(CFG, LOCK)
    assert cfg.lock_path.name == "paper_a_LOCK.json"                       # never Paper B's LOCK.json
    assert cfg.get("study.short_name") == "Paper A" and cfg.get("study.status") == "final" and not cfg.open_items
    assert [d["id"] for d in cfg.get("decisions")] == ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    assert cfg.get("sample.search.query") == tp.QUERY
    assert tuple(cfg.get("sample.search.forms")) == tp.TARGET_FORMS
    assert set(cfg.get("wsib.outcome_columns_hidden_until_lock")) >= {"net_irr", "tvpi", "gain"}
    tiers = {h["id"]: h["tier"] for h in cfg.hypotheses}
    assert tiers == {"A-H1": "confirmatory", "A-H2": "confirmatory", "A-H3": "confirmatory",
                     "A-X1": "exploratory", "A-X2": "exploratory", "A-X3": "exploratory"}
    est = cfg.get("estimation")
    assert est["alpha"] == 0.05 and est["h3"]["horizon_years"] == 5 and est["regime_split"] == "2022-01-01"
    assert all(est[k] for k in ("h1", "h2", "premium", "robustness", "exploratory"))
    acc = cfg.get("validation.term_gates.measured_accuracy")              # the terms that missed 90% carry their accuracy
    assert set(acc) == {"sponsor_name", "equity_commitment", "debt_commitment"} and all(0.7 < v < 0.9 for v in acc.values())


def test_paper_a_is_locked_on_its_data_run():
    cfg = load_config(CFG, LOCK)
    assert cfg.locked and cfg.lock_record["hypotheses"] == ["A-H1", "A-H2", "A-H3", "A-X1", "A-X2", "A-X3"]
    assert set(cfg.lock_record["data_run"]) == {"repo_commit", "release", "notebook_build", "generated_utc"}


def test_a_changed_paper_a_config_is_a_mismatch_not_a_lock(tmp_path):
    cfg_copy, lock_copy = tmp_path / "paper_a.yaml", tmp_path / "paper_a_LOCK.json"
    shutil.copy(CFG, cfg_copy)
    shutil.copy(LOCK, lock_copy)
    cfg_copy.write_text(cfg_copy.read_text(encoding="utf-8").replace("alpha: 0.05", "alpha: 0.10"), encoding="utf-8")
    assert load_config(cfg_copy, lock_copy).status == "mismatch"
    with pytest.raises(PreregistrationError):
        lock_config(cfg_copy, lock_copy)
