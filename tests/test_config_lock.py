import json
import re
from datetime import datetime, timezone

import pytest
import yaml

from aem_research.config import (PreregistrationError, config_hash, load_config, lock_config,
                                 require_locked)
from conftest import CONFIG

FIXED = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)


def test_shipped_config_loads():
    # Holds before and after the lock, so finalizing the config never breaks the suite.
    cfg = load_config(CONFIG)
    assert len(cfg.sha256) == 64
    assert cfg.get("study.status") in ("draft", "final")
    ids = [h["id"] for h in cfg.hypotheses]
    assert ids[:4] == ["H1", "H2", "H3", "H4"] and len(ids) == len(set(ids))
    assert {h["tier"] for h in cfg.hypotheses} <= {"primary", "secondary", "exploratory"}
    assert [h["id"] for h in cfg.hypotheses if h["tier"] == "primary"] == ["H1", "H2"]
    assert cfg.get("outcomes.horizons_years") == [3, 5, 10]
    assert cfg.get("missing.key", "x") == "x"
    assert all(isinstance(i, str) and i for i in cfg.open_items)
    if cfg.locked:
        assert cfg.open_items == []
    models = cfg.get("estimation.models")
    assert models and all({"id", "outcome", "model"} <= set(m) for m in models)
    # every pre-registered test belongs to a hypothesis (H2c and H4_benchmark map to H2 and H4)
    assert {re.match(r"H\d+", k).group(0) for k in cfg.get("estimation.tests")} <= set(ids)
    dec = [d["id"] for d in cfg.get("decisions")]
    assert dec[:6] == ["D1", "D2", "D3", "D4", "D5", "D6"] and len(dec) == len(set(dec))
    assert set(cfg.get("regime.moderators")) >= {"real_yield_10y", "credit_spread_baa", "inflation_cpi_yoy"}
    # D8: the regime difference is confirmed once, on start dates added after the lock
    assert "D8" in dec and {"adds", "confirmation", "terminal"} <= set(cfg.get("regime.update"))
    assert cfg.get("estimation.descriptives") and "D8" in cfg.get("estimation.outputs_to_terminal")


def test_open_review_items_block_the_lock(final_config):
    data = yaml.safe_load(final_config.read_text(encoding="utf-8"))
    data["review_before_lock"] = ["Check the delisting returns.", "Confirm the start year."]
    final_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    cfg = load_config(final_config)
    assert cfg.open_items == ["Check the delisting returns.", "Confirm the start year."]
    assert "Open review items: 2" in cfg.summary() and "2. Confirm the start year." in cfg.summary()
    with pytest.raises(PreregistrationError, match="2 review_before_lock item"):
        lock_config(final_config, now=FIXED)
    assert not (final_config.parent / "LOCK.json").exists()
    data["review_before_lock"] = []
    final_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert lock_config(final_config, now=FIXED)["locked_at_utc"] == "2026-11-05T15:00:00Z"
    assert "Open review items" not in load_config(final_config).summary()


def test_open_review_items_also_block_an_amendment(final_config):
    lock_config(final_config, now=FIXED)
    data = yaml.safe_load(final_config.read_text(encoding="utf-8"))
    data["review_before_lock"] = ["New question raised after the lock."]
    final_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(PreregistrationError, match="review_before_lock"):
        lock_config(final_config, force=True, note="amend", now=FIXED)


def test_hash_ignores_comments_and_key_order(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("# comment\nx: 1\ny: [1, 2]\n", encoding="utf-8")
    b.write_text("y: [1, 2]   # trailing comment\nx: 1\n", encoding="utf-8")
    assert load_config(a).sha256 == load_config(b).sha256


def test_hash_changes_with_any_value():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    h0 = config_hash(data)
    data["outcomes"]["touch_multiples"][1] = 3.01
    assert config_hash(data) != h0


def test_draft_cannot_be_locked(tmp_path):
    # Built from the shipped config with status forced to draft, so the test still holds
    # after the real config is finalized and locked.
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["study"]["status"] = "draft"
    data["review_before_lock"] = []
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(PreregistrationError, match="final"):
        lock_config(path)


def test_lock_flow_and_idempotence(final_config):
    assert load_config(final_config).status == "unlocked"
    with pytest.raises(PreregistrationError, match="not locked"):
        require_locked(load_config(final_config))
    rec = lock_config(final_config, note="initial", now=FIXED)
    cfg = load_config(final_config)
    assert cfg.status == "locked" and cfg.locked
    assert rec["locked_at_utc"] == "2026-11-05T15:00:00Z"
    assert rec["hypotheses"] == [h["id"] for h in cfg.hypotheses]
    require_locked(cfg)
    assert lock_config(final_config) == rec  # idempotent


def test_amendment_needs_force_and_keeps_history(final_config):
    first = lock_config(final_config, now=FIXED)
    data = yaml.safe_load(final_config.read_text(encoding="utf-8"))
    data["starts"]["entry_tolerance_days"] = 5
    final_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    cfg = load_config(final_config)
    assert cfg.status == "mismatch"
    with pytest.raises(PreregistrationError, match="no longer matches"):
        require_locked(cfg)
    with pytest.raises(PreregistrationError, match="force=True"):
        lock_config(final_config)
    rec = lock_config(final_config, force=True, note="tolerance 5 days", now=FIXED)
    assert rec["amends"] == first["config_sha256"]
    payload = json.loads((final_config.parent / "LOCK.json").read_text())
    assert payload["history"][0]["config_sha256"] == first["config_sha256"]
    assert load_config(final_config).status == "locked"
