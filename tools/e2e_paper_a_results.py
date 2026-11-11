"""End-to-end check for notebook 05 (Paper A's results from the locked run). No network.

    python tools/e2e_paper_a_results.py <scratch_dir> <repo_dir>

Runs notebook 02 on the synthetic Drive (tools/e2e_paper_a.py, full), then executes every
code cell of notebook 05 against that Drive, each time on a copy of the repo:
  ok        the copy's Paper A lock names the synthetic run: sample, tests, tables, block
  mismatch  the lock names another run: stops before any result
  unlocked  no lock: stops before any result
Exits 0 when every case behaves.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path(sys.argv[1])
REPO = Path(sys.argv[2]).resolve()
t0 = time.time()

print("Running notebook 02 on its synthetic Drive …")
r = subprocess.run([sys.executable, "-I", str(REPO / "tools" / "e2e_paper_a.py"), str(SCRATCH / "nb02"), str(REPO), "full"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:], r.stderr[-3000:])
    raise SystemExit("notebook 02's end-to-end run failed")
DRIVE = next(p for p in (SCRATCH / "nb02").rglob("drive") if (p / "aem-research").is_dir())
OUTPUT = DRIVE / "aem-research" / "output"
sys.path.insert(0, str(REPO))
from aem_research.paper_a_results import run_identity  # noqa: E402

IDENTITY = run_identity((OUTPUT / "paper_a_summary.txt").read_text(encoding="utf-8"))
print("data run:", IDENTITY)
assert IDENTITY and IDENTITY["release"] >= 12


def repo_copy(name: str, data_run: dict | None) -> Path:
    copy = SCRATCH / f"repo_{name}"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    (copy / "config" / "paper_a_LOCK.json").unlink(missing_ok=True)
    if data_run is not None:
        script = ("import json, sys; from aem_research.config import lock_config; "
                  "print(lock_config('config/paper_a.yaml', 'config/paper_a_LOCK.json', note='e2e', "
                  "data_run=json.loads(sys.argv[1])))")
        subprocess.run([sys.executable, "-c", script, json.dumps(data_run)], cwd=copy, check=True)
    return copy


def run_notebook(repo: Path) -> str:
    for f in ("paper_a_results.txt", "paper_a_results.xlsx"):
        (OUTPUT / f).unlink(missing_ok=True)
    for f in OUTPUT.glob("paper_a_analysis_sample.*"):
        f.unlink()
    os.environ.update(AEM_DRIVE_ROOT=str(DRIVE), AEM_REPO_DIR=str(repo))
    nb = json.loads((repo / "notebooks" / "05_paper_a_results.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
    return (OUTPUT / "paper_a_results.txt").read_text(encoding="utf-8")


def written() -> list:
    return sorted(p.name for p in OUTPUT.iterdir() if p.name.startswith(("paper_a_results.xlsx", "paper_a_analysis_sample")))


block = run_notebook(repo_copy("ok", IDENTITY))
print(block)
assert block.startswith("PAPER A RESULTS v") and "CHECKS    lock ok · data run ok" in block
assert "SAMPLE    deals 8 · completed 8" in block and "documents scanned 7" in block and "benchmark dates 4" in block
for label in ("A-H1      core · n ", "A-H2      core · before 2022 ", "A-H3      core · visible exit by 5y", "ROBUST    EV/revenue, core",
              "ROBUST    exits, all buyers", "EXPLORE   everything below is exploratory", "ERRORS    none"):
    assert label in block, label
assert "not testable" in block                      # eight synthetic deals, all entered before 2022, carry no regime test
assert "paper_a_results.xlsx" in written() and any(f.startswith("paper_a_analysis_sample") for f in written())
import pandas as pd  # noqa: E402
wb = pd.read_excel(OUTPUT / "paper_a_results.xlsx", sheet_name=None, engine="openpyxl")
assert {"Verdicts", "Confirmatory", "Funnel", "Sample", "X premium by group", "X WSIB by vintage"} <= set(wb), set(wb)
assert len(wb["Verdicts"]) == 3 and len(wb["Sample"]) == 8

block = run_notebook(repo_copy("mismatch", {**IDENTITY, "generated_utc": "2099-01-01 00:00 UTC"}))
assert "CHECKS    STOPPED – generated_utc: the run on Drive has" in block and "A-H1" not in block, block
assert written() == [], "results written although the run is not the data run"

block = run_notebook(repo_copy("unlocked", None))
assert "CHECKS    STOPPED – the pre-registration is not locked" in block and written() == [], block

print(f"\nNOTEBOOK 05 OK in all three cases ({time.time() - t0:.0f}s)")
