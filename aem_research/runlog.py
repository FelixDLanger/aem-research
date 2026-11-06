"""Run log: every phase's status, duration and error, so one failure never hides the rest.

    LOG = RunLog()
    with LOG.phase("A SimFin cache"):
        ...                      # an exception here is recorded and printed, the run goes on

`LOG.must_pass` is for gates that must stop the run (the test suite, the look-ahead
guard): the error is recorded and re-raised.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager


class RunLog:
    def __init__(self):
        self.t0 = time.time()
        self.phases: list[dict] = []
        self.errors: list[dict] = []

    def _record(self, name, t, exc=None, stopped=False):
        entry = {"phase": name, "status": "ok" if exc is None else ("STOPPED" if stopped else "FAILED"),
                 "seconds": round(time.time() - t, 1)}
        self.phases.append(entry)
        if exc is not None:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            lines = "".join(tb).strip().splitlines()
            self.errors.append({"phase": name, "error": f"{type(exc).__name__}: {exc}"[:600],
                                "where": [ln.strip() for ln in lines if ln.strip().startswith("File ")][-4:]})

    @contextmanager
    def phase(self, name: str):
        t = time.time()
        try:
            yield
        except Exception as exc:          # recorded, printed, and the notebook carries on
            self._record(name, t, exc)
            print(f"\n*** Phase '{name}' failed: {type(exc).__name__}: {exc}\n"
                  "    The run continues. Paste the block printed at the end into the chat.")
        else:
            self._record(name, t)

    @contextmanager
    def must_pass(self, name: str):
        t = time.time()
        try:
            yield
        except Exception as exc:
            self._record(name, t, exc, stopped=True)
            raise
        else:
            self._record(name, t)

    def skipped(self, name: str, why: str):
        self.phases.append({"phase": name, "status": f"skipped ({why})", "seconds": 0.0})

    def as_dict(self) -> dict:
        return {"phases": self.phases, "errors": self.errors, "total_minutes": round((time.time() - self.t0) / 60, 1)}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("pandas", "numpy", "pyarrow", "openpyxl", "requests", "yaml"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "missing"
    return out


def folder_size_gb(path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 1e9, 2)


def in_notebook() -> bool:
    return "ipykernel" in sys.modules
