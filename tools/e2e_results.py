"""End-to-end check for notebooks 04 and 07 (H1, then H2 to H6, from the locked run). No network.

    python tools/e2e_results.py <scratch_dir> <repo_dir>

Runs notebook 01 on the synthetic Drive (tools/e2e_synthetic.py, full), then executes every
code cell of notebook 04 against that Drive in four cases, each on a copy of the repo:
  ok        the copy is locked with the synthetic run as its data run: outcomes and H1
  mismatch  the lock names another run (a later start time): stops before any outcome
  unlocked  the copy is not locked: stops before any outcome
  gatefail  locked on the run, but the run's report shows a failed pre-lock gate: stops
After the ok case notebook 07 runs on the outcomes notebook 04 just wrote (every hypothesis
printed), and once more on an unlocked copy (stops). Exits 0 when every case behaves.
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

print("Building notebook 01's synthetic Drive and running notebook 01 on it …")
r = subprocess.run([sys.executable, "-I", str(REPO / "tools" / "e2e_synthetic.py"), str(SCRATCH / "nb01"),
                    str(REPO), "full"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:], r.stderr[-3000:])
    raise SystemExit("notebook 01's end-to-end run failed")
DRIVE = SCRATCH / "nb01" / "drive"
OUTPUT = DRIVE / "aem-research" / "output"
RUN = json.loads((OUTPUT / "coverage_report.json").read_text(encoding="utf-8"))
IDENTITY = {k: RUN[k] for k in ("repo_commit", "release", "notebook_build", "generated_utc")}
print("data run:", IDENTITY)


def repo_copy(name: str, data_run: dict | None) -> Path:
    """A copy of the repository, its pre-registration finalized and locked on `data_run`
    (unlocked when None)."""
    copy = SCRATCH / f"repo_{name}"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    (copy / "config" / "LOCK.json").unlink(missing_ok=True)
    if data_run is not None:
        script = ("import json, sys, yaml, pathlib; from aem_research.config import lock_config; "
                  "p = pathlib.Path('config/preregistration.yaml'); d = yaml.safe_load(p.read_text()); "
                  "d['study']['status'] = 'final'; d['review_before_lock'] = []; "
                  "p.write_text(yaml.safe_dump(d, sort_keys=False)); "
                  "print(lock_config(p, note='e2e', data_run=json.loads(sys.argv[1])))")
        subprocess.run([sys.executable, "-c", script, json.dumps(data_run)], cwd=copy, check=True)
    return copy


def run_notebook(repo: Path, drive: Path) -> str:
    """Execute notebook 04's code cells in order; return the block it saved."""
    for f in ("results_summary.txt", "aem_results_h1.xlsx"):
        (drive / "aem-research" / "output" / f).unlink(missing_ok=True)
    shutil.rmtree(drive / "aem-research" / "output" / "outcomes", ignore_errors=True)
    for f in (drive / "aem-research" / "output").glob("outcomes.*"):
        f.unlink()
    os.environ.update(AEM_DRIVE_ROOT=str(drive), AEM_REPO_DIR=str(repo))
    nb = json.loads((repo / "notebooks" / "04_results.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
    return (drive / "aem-research" / "output" / "results_summary.txt").read_text(encoding="utf-8")


def outcome_files(drive: Path) -> list:
    """The files notebook 04 writes. In a locked repository notebook 01's own Phase E leaves
    its files too (aem_outcomes.xlsx, the B table outcomes_b); those are not notebook 04's."""
    out = drive / "aem-research" / "output"
    return sorted(p.name for p in out.iterdir()
                  if p.name.split(".")[0] == "outcomes" or p.name == "aem_results_h1.xlsx")


# ok: the lock names this run
block = run_notebook(repo_copy("ok", IDENTITY), DRIVE)
print(block)
assert block.startswith("AEM RESULTS v") and "CHECKS    lock ok · data run ok · gates all passed in the data run" in block
assert "H1 TOUCH  pooled · 3y: 3x " in block and "5y: 3x " in block and "H1 5Y     pooled · 0.1x " in block
assert "REGIME    before 2022 · 3y: " in block and "BY YEAR   3y touch 3x: " in block
assert "OUTCOMES  start_windows " in block and "ERRORS    none" in block
assert "aem_results_h1.xlsx" in outcome_files(DRIVE) and any(f.startswith("outcomes") for f in outcome_files(DRIVE))



def run_estimates(repo: Path, drive: Path) -> str:
    for f in ("results_h2_h6.txt", "aem_results_h2_h6.xlsx"):
        (drive / "aem-research" / "output" / f).unlink(missing_ok=True)
    os.environ.update(AEM_DRIVE_ROOT=str(drive), AEM_REPO_DIR=str(repo))
    nb = json.loads((repo / "notebooks" / "07_estimates.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
    return (drive / "aem-research" / "output" / "results_h2_h6.txt").read_text(encoding="utf-8")


# notebook 07 on the outcomes just computed
est_block = run_estimates(SCRATCH / "repo_ok", DRIVE)
print(est_block)
assert est_block.startswith("AEM RESULTS H2–H6 v") and "CHECKS    lock ok · data run ok · gates all passed" in est_block
for label in ("WINDOWS   universe windows ", "H2        3x within 3 years", "H2a       ", "H2b       ", "PLACEBO   ", "ABLATION  full cohort (H2a)",
              "H3        ", "H4        10x within 5 years", "H4        S1 (FCF per share outgrew revenue)", "M2        10x logit",
              "M1        ordered logit", "TOUCHED   ", "ROBUST    version primary", "PENDING   robustness versions", "ERRORS    none"):
    assert label in est_block, label
import pandas as pd  # noqa: E402
wb7 = pd.read_excel(OUTPUT / "aem_results_h2_h6.xlsx", sheet_name=None, engine="openpyxl")
assert {"H2", "H2 ablation", "H3", "H4", "Robustness"} <= set(wb7), set(wb7)
assert len(wb7["H4"]) == 8 and len(wb7["H2"]) == 3
stopped = run_estimates(repo_copy("est_unlocked", None), DRIVE)
assert "CHECKS    STOPPED – the pre-registration is not locked" in stopped and "H2a" not in stopped
assert not (OUTPUT / "aem_results_h2_h6.xlsx").exists()



def run_followup(repo: Path, drive: Path) -> str:
    for f in ("followup.txt", "aem_followup.xlsx"):
        (drive / "aem-research" / "output" / f).unlink(missing_ok=True)
    os.environ.update(AEM_DRIVE_ROOT=str(drive), AEM_REPO_DIR=str(repo))
    nb = json.loads((repo / "notebooks" / "08_followup.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
    return (drive / "aem-research" / "output" / "followup.txt").read_text(encoding="utf-8")


# notebook 08 on the same Drive: the robustness versions and the inventory
fu_block = run_followup(SCRATCH / "repo_ok", DRIVE)
print(fu_block)
assert fu_block.startswith("AEM FOLLOW-UP v") and "confirmation plan locked (C1, C2, C3)" in fu_block and "ERRORS    none" in fu_block
for label in ("YEARS     test H2a", "UNIVERSE  version primary", "UNIVERSE  version market cap from US$50m", "UNIVERSE  version no minimum price",
              "BASE RATE version primary", "NASDAQ-30 version primary (Nasdaq -55%)", "NASDAQ-30 version Nasdaq -30%",
              "SPLITS    version without windows crossing a weak-evidence split", "ROUND 2   still to compute",
              "DRIVE     FOLDERS", "DRIVE     TABLES", "DRIVE     features: ", "DRIVE     sec_panel: ", "DRIVE     sec_splits evidence: "):
    assert label in fu_block, label
wb8 = pd.read_excel(OUTPUT / "aem_followup.xlsx", sheet_name=None, engine="openpyxl")
assert {"Start years", "Universes", "Base rates", "Nasdaq -30%", "Weak splits"} <= set(wb8), set(wb8)
stopped = run_followup(repo_copy("fu_unlocked", None), DRIVE)          # stopped, but the inventory is still there
assert "CHECKS    STOPPED" in stopped and "DRIVE     TABLES" in stopped and "UNIVERSE  " not in stopped

# mismatch: another run of the same version
block = run_notebook(repo_copy("mismatch", {**IDENTITY, "generated_utc": "2099-01-01 00:00 UTC"}), DRIVE)
assert "CHECKS    STOPPED – generated_utc: the run on Drive has" in block and "H1 TOUCH" not in block, block
assert outcome_files(DRIVE) == [], "outcomes written although the run is not the data run"

# unlocked
block = run_notebook(repo_copy("unlocked", None), DRIVE)
assert "CHECKS    STOPPED – the pre-registration is not locked" in block and outcome_files(DRIVE) == [], block

# gatefail: the data run itself failed a gate
drive_g = SCRATCH / "drive_gatefail"
shutil.rmtree(drive_g, ignore_errors=True)
shutil.copytree(DRIVE, drive_g)
rep = json.loads((drive_g / "aem-research" / "output" / "coverage_report.json").read_text(encoding="utf-8"))
rep["prelock"]["split_recall"] = "75.3% BELOW 90%"
(drive_g / "aem-research" / "output" / "coverage_report.json").write_text(json.dumps(rep), encoding="utf-8")
block = run_notebook(repo_copy("gatefail", IDENTITY), drive_g)
assert "CHECKS    STOPPED – gate failed in the data run: split_recall 75.3% BELOW 90%" in block, block
assert outcome_files(drive_g) == []

print(f"\nNOTEBOOKS 04, 07 AND 08 OK in all cases ({time.time() - t0:.0f}s)")
