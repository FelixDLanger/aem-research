"""Pre-registration config and lock.

The config (config/preregistration.yaml) is hashed as canonical JSON, so comments
and key order do not change the hash while any value does. A lock file
(config/LOCK.json) records the hash and a UTC timestamp. Outcome computation calls
`require_locked`, so outcomes cannot be computed from a draft or from a config that
changed after it was locked.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOCK_SCHEMA = "aem-prereg-lock/1"


class PreregistrationError(RuntimeError):
    """Raised when a locked step runs without a valid pre-registration lock."""


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_hash(data: dict) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StudyConfig:
    data: dict
    sha256: str
    path: Path
    lock_path: Path
    status: str  # "locked", "unlocked" or "mismatch"
    lock_record: dict | None

    @property
    def locked(self) -> bool:
        return self.status == "locked"

    @property
    def hypotheses(self) -> list[dict]:
        return list(self.data.get("hypotheses") or [])

    @property
    def open_items(self) -> list[str]:
        """Decisions still open before the lock (`review_before_lock` in the config)."""
        items = self.data.get("review_before_lock") or []
        return [str(i) for i in items] if isinstance(items, list) else [str(items)]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def summary(self) -> str:
        lines = [
            f"Config: {self.path.name}  sha256 {self.sha256[:16]}…",
            f"Study status (in file): {self.get('study.status', 'unknown')}",
            f"Lock status: {self.status.upper()}",
        ]
        if self.lock_record:
            lines.append(f"  locked at {self.lock_record.get('locked_at_utc')}  "
                         f"hash {str(self.lock_record.get('config_sha256'))[:16]}…")
        for h in self.hypotheses:
            lines.append(f"  {h.get('id')} [{h.get('tier')}] {str(h.get('statement', ''))[:90]}…")
        if self.open_items:
            lines.append(f"Open review items: {len(self.open_items)} – the lock refuses until they are resolved:")
            lines += [f"  {n}. {item}" for n, item in enumerate(self.open_items, 1)]
        return "\n".join(lines)


def _read_lock(lock_path: Path) -> dict | None:
    if not lock_path.exists():
        return None
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "current" not in payload:
        raise PreregistrationError(f"{lock_path} is not a valid lock file")
    return payload


def load_config(path: str | Path, lock_path: str | Path | None = None) -> StudyConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    sha = config_hash(data)
    lock_path = Path(lock_path) if lock_path else path.with_name("LOCK.json")
    payload = _read_lock(lock_path)
    if payload is None:
        return StudyConfig(data, sha, path, lock_path, "unlocked", None)
    current = payload.get("current") or {}
    status = "locked" if current.get("config_sha256") == sha else "mismatch"
    return StudyConfig(data, sha, path, lock_path, status, current)


def lock_config(path: str | Path, lock_path: str | Path | None = None, *, note: str = "",
                force: bool = False, now: datetime | None = None) -> dict:
    """Write or amend the lock. Returns the current lock record.

    Idempotent when the config is already locked. A config that changed after it was
    locked is only re-locked with force=True; the previous record is kept in history.
    Every lock, amendments included, needs study.status final and an empty
    review_before_lock list.
    """
    cfg = load_config(path, lock_path)
    if cfg.status == "locked":
        return dict(cfg.lock_record or {})
    if cfg.status == "mismatch" and not force:
        raise PreregistrationError(
            "The config changed after it was locked. Amending a pre-registration must be "
            "deliberate: call lock_config(..., force=True, note='why') – the old lock stays "
            "in the history.")
    if str(cfg.get("study.status", "")).lower() != "final":
        raise PreregistrationError(
            "study.status is not 'final'. Set it to final once the hypotheses are reviewed, "
            "then lock.")
    if cfg.open_items:
        raise PreregistrationError(
            f"{len(cfg.open_items)} review_before_lock item(s) still open. Resolve each one in the "
            "config, delete it from the list, then lock:\n"
            + "\n".join(f"  - {item}" for item in cfg.open_items))
    payload = _read_lock(cfg.lock_path) or {"schema": LOCK_SCHEMA, "history": []}
    history = list(payload.get("history") or [])
    previous = payload.get("current")
    if previous:
        history.append(previous)
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    record = {
        "config_sha256": cfg.sha256,
        "config_file": cfg.path.name,
        "locked_at_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypotheses": [h.get("id") for h in cfg.hypotheses],
        "note": note,
    }
    if previous:
        record["amends"] = previous.get("config_sha256")
    out = {"schema": LOCK_SCHEMA, "current": record, "history": history}
    cfg.lock_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def require_locked(cfg: StudyConfig) -> None:
    if cfg.status == "locked":
        return
    if cfg.status == "mismatch":
        raise PreregistrationError(
            f"The config ({cfg.sha256[:12]}…) no longer matches its lock "
            f"({str((cfg.lock_record or {}).get('config_sha256'))[:12]}…). Revert the change, "
            "or re-lock deliberately with a note.")
    raise PreregistrationError(
        "The pre-registration is not locked. Review config/preregistration.yaml, set "
        "study.status to final, and lock it before computing outcomes.")
