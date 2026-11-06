"""Run log: every phase's status, duration, memory and error, so one failure never hides the rest.

    LOG = RunLog()
    with LOG.phase("A SimFin cache"):
        ...                      # an exception here is recorded and printed, the run goes on

`LOG.must_pass` is for gates that must stop the run (the test suite): the error is
recorded, `on_stop` prints the paste-ready block, and the error is re-raised.
`LOG.warn` records a gap that does not stop anything (a FRED series that could not be
read) so it still reaches the paste block.

Once the Drive folders exist, `LOG.attach_progress(path)` appends each phase's start and
end to a small text file. If Colab crashes (out of memory) or disconnects before the last
cell, that file still shows how far the run got.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path


def peak_ram_gb() -> float | None:
    """Peak memory of this process so far, in GB (it only ever rises)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9, 1)  # bytes on macOS, KiB on Linux
    except Exception:
        return None


def _short(path: str) -> str:
    """Package files relative to the package, other files by their last two parts."""
    path = path.replace("\\", "/")
    if "aem_research/" in path:
        return path[path.rindex("aem_research/"):]
    return "/".join(path.split("/")[-2:])


def _where(exc: BaseException) -> list[str]:
    """The last frames of the error itself (not of a chained cause), without the log's own frame."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if not f.filename.endswith("runlog.py")]
    return [f"{_short(f.filename)}, line {f.lineno}, in {f.name}" for f in frames][-4:]


def total_ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


class RunLog:
    def __init__(self):
        self.t0 = time.time()
        self.phases: list[dict] = []
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.progress_path: Path | None = None

    # ---------------------------------------------------------------- progress file on Drive
    def attach_progress(self, path, header: str = "") -> None:
        """Append phase starts and ends to `path` from now on (earlier phases are written first)."""
        self.progress_path = Path(path)
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.t0))
        self._note(f"=== run started {started} UTC {header}".rstrip())
        for p in self.phases:
            self._note(f"end   {p['phase']}: {p['status']} after {p['seconds']}s")

    def _note(self, text: str) -> None:
        if self.progress_path is None:
            return
        try:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_path, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}\n")
        except OSError:
            pass                                  # the log must never be the thing that fails

    # ---------------------------------------------------------------- recording
    def _record(self, name, t, exc=None, stopped=False):
        entry = {"phase": name, "status": "ok" if exc is None else ("STOPPED" if stopped else "FAILED"),
                 "seconds": round(time.time() - t, 1), "ram_gb": peak_ram_gb()}
        self.phases.append(entry)
        note = f"end   {name}: {entry['status']} after {entry['seconds']}s · peak RAM {entry['ram_gb']} GB"
        if exc is not None:
            err = {"phase": name, "error": f"{type(exc).__name__}: {exc}"[:600], "where": _where(exc)}
            self.errors.append(err)
            note += f" · {err['error'][:300]}"
        self._note(note)

    @contextmanager
    def phase(self, name: str):
        t = time.time()
        self._note(f"start {name} · peak RAM {peak_ram_gb()} GB")
        try:
            yield
        except Exception as exc:          # recorded, printed, and the notebook carries on
            self._record(name, t, exc)
            print(f"\n*** Phase '{name}' failed: {type(exc).__name__}: {exc}\n"
                  "    The run continues. Paste the block printed at the end into the chat.")
        else:
            self._record(name, t)

    @contextmanager
    def must_pass(self, name: str, on_stop=None):
        t = time.time()
        self._note(f"start {name}")
        try:
            yield
        except Exception as exc:
            self._record(name, t, exc, stopped=True)
            if on_stop is not None:
                try:
                    on_stop()
                except Exception as cb_exc:   # never mask the real error
                    print(f"(could not print the summary block: {type(cb_exc).__name__}: {cb_exc})")
            raise
        else:
            self._record(name, t)

    def skipped(self, name: str, why: str):
        self.phases.append({"phase": name, "status": f"skipped ({why})", "seconds": 0.0})
        self._note(f"skip  {name} ({why})")

    def warn(self, phase: str, message: str):
        """A gap that does not stop the run but must reach the paste block."""
        self.warnings.append({"phase": phase, "message": message[:400]})
        self._note(f"warn  {phase}: {message[:300]}")
        print(f"Warning ({phase}): {message}")

    def as_dict(self) -> dict:
        return {"phases": self.phases, "errors": self.errors, "warnings": self.warnings,
                "total_minutes": round((time.time() - self.t0) / 60, 1),
                "peak_ram_gb": peak_ram_gb(), "total_ram_gb": total_ram_gb()}

    def finish(self) -> None:
        self._note(f"=== run finished after {round((time.time() - self.t0) / 60, 1)} min · "
                   f"{len(self.errors)} errors, {len(self.warnings)} warnings · peak RAM {peak_ram_gb()} GB")


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
