"""End-to-end check for notebooks 05, 06 and 09 (Paper A's results, the exploration, the exit values). No network.

    python tools/e2e_paper_a_results.py <scratch_dir> <repo_dir>

Runs notebook 02 on the synthetic Drive (tools/e2e_paper_a.py, full), then executes every
code cell of notebook 05 against that Drive, each time on a copy of the repo:
  ok        the copy's Paper A lock names the synthetic run: sample, tests, tables, block
  mismatch  the lock names another run: stops before any result
  unlocked  no lock: stops before any result
Then notebook 06 on the same Drive: with both locks naming the synthetic runs (every table), and
without Paper A's lock (stops). Exits 0 when every case behaves.
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


def repo_copy(name: str, data_run: dict | None, data_run_b: dict | None = None) -> Path:
    copy = SCRATCH / f"repo_{name}"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    (copy / "config" / "paper_a_LOCK.json").unlink(missing_ok=True)
    if data_run_b is not None:                              # Paper B's lock re-pointed at the synthetic notebook-01 run
        (copy / "config" / "LOCK.json").unlink()
        script = ("import json, sys; from aem_research.config import lock_config; "
                  "lock_config('config/preregistration.yaml', note='e2e', data_run=json.loads(sys.argv[1]))")
        subprocess.run([sys.executable, "-c", script, json.dumps(data_run_b)], cwd=copy, check=True)
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



def run_explore(repo: Path) -> str:
    for f in ("explore.txt", "explore.xlsx"):
        (OUTPUT / f).unlink(missing_ok=True)
    os.environ.update(AEM_DRIVE_ROOT=str(DRIVE), AEM_REPO_DIR=str(repo))
    nb = json.loads((repo / "notebooks" / "06_explore.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
    return (OUTPUT / "explore.txt").read_text(encoding="utf-8")


RUN_B = json.loads((OUTPUT / "coverage_report.json").read_text(encoding="utf-8"))
IDENTITY_B = {k: RUN_B[k] for k in ("repo_commit", "release", "notebook_build", "generated_utc")}
block = run_explore(repo_copy("explore", IDENTITY, IDENTITY_B))
print(block)
assert block.startswith("AEM EXPLORE v") and "CHECKS    locks ok · data runs ok" in block and "ERRORS    none" in block
assert "INPUTS    deals 8 · core 7 · price panel rows " in block and "outcome rows 0" not in block
for label in ("E1 premium by year", "E1 benchmark by year · year 20", "E2 margins · group core", "E3 wider coverage · enterprise value",
              "E4 own history", "E5 target signals", "E6 public holder · windows ended in a take-private", "E7 exit slowdown",
              "E8 sponsor styles", "E9 public vs private · year 20"):
    assert label in block, label
wb = pd.read_excel(OUTPUT / "explore.xlsx", sheet_name=None, engine="openpyxl")
assert {"1 premium by year", "2 margins", "4 own history", "own_history_deals", "9 public vs private"} <= set(wb), set(wb)

block = run_explore(repo_copy("explore_unlocked", None, IDENTITY_B))
assert "CHECKS    STOPPED – Paper A: the pre-registration is not locked" in block and not (OUTPUT / "explore.xlsx").exists(), block
block = run_explore(repo_copy("explore_other_b", IDENTITY))        # Paper B's lock names the real run, not this Drive's
assert "CHECKS    STOPPED – Paper B: " in block and not (OUTPUT / "explore.xlsx").exists(), block



# ---- notebook 09: the exit values, read offline from texts placed on the synthetic Drive
import gzip  # noqa: E402
from aem_research import takeprivate as tp  # noqa: E402
from aem_research.io_utils import load_table  # noqa: E402

A_DIR = DRIVE / "aem-research" / "cache" / "paper_a"
D9 = load_table(A_DIR / "deals")
sale = D9[D9["exit_route"].eq("sold to a listed buyer")].iloc[0]
core = tp.exit_query_name(sale["target_name"])
entry_ev = float(sale["enterprise_value"])
for name, text in ((f"{sale['sale_adsh']}-index.htm.gz",
                    '<table><tr><td>1</td><td>8-K</td><td><a href="/Archives/x/d8k.htm">d8k.htm</a></td><td>8-K</td><td>1</td></tr>'
                    '<tr><td>2</td><td>Press release</td><td><a href="/Archives/x/ex99.htm">ex99.htm</a></td><td>EX-99.1</td><td>1</td></tr></table>'),
                   (f"{sale['sale_adsh']}_d8k.htm.txt.gz", f"Item 2.01 Completion of Acquisition or Disposition of Assets. {core}."),
                   (f"{sale['sale_adsh']}_ex99.htm.txt.gz",
                    f"The buyer today announced the completion of its acquisition of {core} in a transaction valued at approximately "
                    f"${entry_ev * 1.8 / 1e6:,.0f} million.")):
    (A_DIR / "docs" / name).write_bytes(gzip.compress(text.encode("utf-8")))
before = {p_.name: p_.stat().st_mtime_ns for p_ in A_DIR.iterdir() if p_.is_file()}


def run_exits(repo: Path) -> str:
    for f in ("paper_a_exits.txt", "paper_a_exits.xlsx"):
        (OUTPUT / f).unlink(missing_ok=True)
    os.environ.update(AEM_DRIVE_ROOT=str(DRIVE), AEM_REPO_DIR=str(repo), AEM_SEC_OFFLINE="1")
    nb = json.loads((repo / "notebooks" / "09_exits.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
    return (OUTPUT / "paper_a_exits.txt").read_text(encoding="utf-8")


block = run_exits(SCRATCH / "repo_explore")
print(block)
assert block.startswith("PAPER A EXITS v") and "CHECKS    locks ok · data runs ok" in block and "ERRORS    none" in block
assert "INPUTS    deals 8 · sold to a listed buyer 1 · re ipo 1" in block
assert "READ      user agent offline · sales 1 · cached 0 · read 1 · with value 1" in block
assert "VALUES    sales with value 1 · usable 1" in block
assert "X VALUE   by all · group all usable sales · n 1 · median value multiple 1.80" in block
assert "X TRACK   value growth against the benchmark's multiple · n " in block
assert f"READ 01 " in block and "(transaction)" in block and "ex99.htm" in block
assert "WSIB      reports asked " in block and "WSIB      missing: " in block and "PACE      report date 20" in block
assert "B LAST    exploratory, from the saved tables · locked windows " in block and "new windows not on Drive" in block
for label in ("B SIDES   split locked dates, five years · group with fallen", "B SIDES   split locked dates, five years · group without fcf_pos",
              "B WITHIN  sample locked dates, five years · signal fallen · hit 3x within three years",
              "B SUBSET  sample locked dates, five years · subset without pharma and biotech"):
    assert label in block, label
wb = pd.read_excel(OUTPUT / "paper_a_exits.xlsx", sheet_name=None, engine="openpyxl")
assert {"Exit values", "Exit groups", "Exit reading", "WSIB pace", "B both sides", "B fall within dates", "B subsets"} <= set(wb), set(wb)
after = {p_.name: p_.stat().st_mtime_ns for p_ in A_DIR.iterdir() if p_.is_file()}
assert all(after[k] == v for k, v in before.items()), "a table of the locked run was rewritten"
assert {n.split(".")[0] for n in set(after) - set(before)} == {"exits", "wsib_funds_full"}, set(after) - set(before)
assert "READ      user agent offline · sales 1 · cached 1 · read 0" in run_exits(SCRATCH / "repo_explore")      # a rerun reads nothing again
block = run_exits(repo_copy("exits_unlocked", None, IDENTITY_B))
assert "CHECKS    STOPPED – Paper A: the pre-registration is not locked" in block and "X VALUE" not in block and "WSIB" not in block

print(f"\nNOTEBOOKS 05, 06 AND 09 OK in all cases ({time.time() - t0:.0f}s)")
