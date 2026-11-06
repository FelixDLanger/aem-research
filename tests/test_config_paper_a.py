import shutil
from pathlib import Path

import pytest

from aem_research import takeprivate as tp
from aem_research.config import PreregistrationError, load_config, lock_config

ROOT = Path(__file__).resolve().parents[1]
CFG, LOCK = ROOT / "config" / "paper_a.yaml", ROOT / "config" / "paper_a_LOCK.json"


def test_paper_a_config_is_a_draft_with_its_own_lock_file():
    cfg = load_config(CFG, LOCK)
    assert cfg.lock_path.name == "paper_a_LOCK.json"                       # never Paper B's LOCK.json
    assert cfg.get("study.short_name") == "Paper A" and len(cfg.open_items) >= 1
    assert [d["id"] for d in cfg.get("decisions")][:3] == ["A1", "A2", "A3"]
    assert cfg.get("sample.search.query") == tp.QUERY
    assert tuple(cfg.get("sample.search.forms")) == tp.TARGET_FORMS
    assert set(cfg.get("wsib.outcome_columns_hidden_until_lock")) >= {"net_irr", "tvpi", "gain"}
    if cfg.get("study.status") == "draft":
        assert all(h.get("tier") == "draft" for h in cfg.hypotheses)


def test_paper_a_cannot_lock_while_draft_or_open(tmp_path):
    cfg_copy = tmp_path / "paper_a.yaml"
    shutil.copy(CFG, cfg_copy)
    with pytest.raises(PreregistrationError):
        lock_config(cfg_copy, tmp_path / "paper_a_LOCK.json")
    assert not (tmp_path / "paper_a_LOCK.json").exists()
