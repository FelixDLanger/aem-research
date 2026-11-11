"""Generate notebooks/06_explore.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_explore.py notebooks/06_explore.ipynb

It joins Paper A's deals with Paper B's tables on Drive and prints exploratory tables that
answer what the confirmatory results raised. Both pre-registrations must be locked and the
runs on Drive must be the ones the locks name. Nothing is downloaded.
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""
# AEM – 06 exploration across both papers · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**What this does.** Joins Paper A's take-private deals with Paper B's price panel, features and outcomes, and prints the exploratory tables the confirmatory results asked for: (1) the premium split into the deals' multiple and the market's, year by year; (2) target margins against the benchmark's; (3) the premium with announced deal values where the filing tags no debt; (4) the offer against each target's own price history; (5) the AEM signals of targets before the deal; (6) what public holders ended with when a firm was taken private; (7) visible exits on windows with complete follow-up, around the 2022 regime change; (8) sponsor styles against deals of the same sector; (9) Paper B's universe next to WSIB's buyout funds, year by year.

**Everything here is exploratory** (decision A3 and A5): reported in full, not a finding until confirmed on other data.

**When it runs.** After both locks, on the runs they name. Nothing is downloaded; about a minute.

**Run:** Runtime → Run all, then paste the block (also saved as `explore.txt` in `My Drive/aem-research/output/`, with `explore.xlsx`).
""")

code("""
# ── Setup: mount Drive, fetch the repo, import the package ──
import os, sys, shutil, subprocess, json
from datetime import datetime, timezone

REPO_URL = "https://github.com/FelixDLanger/aem-research.git"
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive"
    REPO_DIR = "/content/aem-research"
    r = _git("-C", REPO_DIR, "pull", "--ff-only", "-q") if os.path.isdir(f"{REPO_DIR}/.git") else None
    if r is None or r.returncode != 0:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        r = _git("clone", "-q", "--depth", "1", REPO_URL, REPO_DIR)
        if r.returncode != 0:
            raise RuntimeError(f"Could not fetch {REPO_URL}: {r.stderr.strip()}")
else:  # local run: AEM_DRIVE_ROOT points at a folder laid out like My Drive
    DRIVE_ROOT = os.environ["AEM_DRIVE_ROOT"]
    REPO_DIR = os.environ.get("AEM_REPO_DIR", os.path.abspath(".."))

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
for _m in [m for m in list(sys.modules) if m == "aem_research" or m.startswith("aem_research.")]:
    del sys.modules[_m]

import numpy as np
import pandas as pd
import aem_research
from aem_research import paper_a_explore as px
from aem_research import paper_a_results as par
from aem_research.config import data_run_problems, load_config
from aem_research.env import resolve_paths
from aem_research.export import write_workbook
from aem_research.io_utils import load_table, save_table, show
from aem_research.report import explore_block
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "06_explore.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/06_explore.ipynb")
print(f"Notebook 06 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
      f"pandas {pd.__version__} · Colab: {IN_COLAB}")
""")

code("""
# ── Checks: both locks and the runs they name ──
LOG = RunLog()
cfg_a = load_config(os.path.join(REPO_DIR, "config", "paper_a.yaml"), os.path.join(REPO_DIR, "config", "paper_a_LOCK.json"))
cfg_b = load_config(os.path.join(REPO_DIR, "config", "preregistration.yaml"))
paths = resolve_paths(DRIVE_ROOT, cfg_b.get("data_sources.simfin_cache_dirs"))
A_DIR = paths.cache / "paper_a"
REPORT = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "repo_commit": COMMIT,
          "notebook_build": NOTEBOOK_BUILD, "release": NOTEBOOK_RELEASE, "lock_status_a": cfg_a.status,
          "lock_status_b": cfg_b.status, "versions": versions()}
summary_path = paths.output / "paper_a_summary.txt"
RUN_A = par.run_identity(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
run_b_path = paths.output / "coverage_report.json"
RUN_B = json.loads(run_b_path.read_text(encoding="utf-8")) if run_b_path.exists() else None
problems = ([f"Paper A: {p_.replace('notebook 01', 'notebook 02')}" for p_ in data_run_problems(cfg_a, RUN_A)]
            + [f"Paper B: {p_}" for p_ in data_run_problems(cfg_b, RUN_B)])
REPORT["problems"] = problems
for p_ in problems:
    print("  -", p_)
if not problems:
    print("Both pre-registrations are locked and the runs on Drive are the ones they name.")
""")

code("""
# ── Tables ──
TABLES = None
if not problems:
    with LOG.phase("inputs"):
        as_of = pd.Timestamp(RUN_A["generated_utc"][:10])
        book = paths.output / "paper_a_sample.xlsx"
        BENCH = pd.read_excel(book, sheet_name="Benchmark", engine="openpyxl") if book.exists() else None
        SAMPLE = load_table(paths.output / "paper_a_analysis_sample")          # notebook 05's sample, if it ran
        if SAMPLE is None:
            DEALS = load_table(A_DIR / "deals")
            if DEALS is None:
                raise FileNotFoundError("notebook 02's deals table is not on Drive")
            DEALS["stock_mentions"] = par.stock_mentions(DEALS, A_DIR / "docs")
            SAMPLE, _ = par.analysis_sample(DEALS, BENCH, cfg_a.data, as_of=as_of)
        PANEL = load_table(paths.cache / "sec_panel")
        FEATURES = load_table(paths.cache / "features")
        OUTCOMES = load_table(paths.output / "outcomes")
        WSIB = load_table(A_DIR / "wsib_funds")
        VINTAGES = par.wsib_vintages(WSIB, tuple(cfg_a.get("wsib.strategies_of_interest")))
        REPORT["inputs"] = {"deals": int(len(SAMPLE)), "core": int(SAMPLE["group"].eq("core").sum()),
                            "price_panel_rows": 0 if PANEL is None else int(len(PANEL)),
                            "feature_rows": 0 if FEATURES is None else int(len(FEATURES)),
                            "outcome_rows": 0 if OUTCOMES is None else int(len(OUTCOMES)),
                            "wsib_vintages": int(len(VINTAGES)), "as_of": str(as_of.date())}
        print(REPORT["inputs"])
    with LOG.phase("tables"):
        TABLES = px.run_all(SAMPLE, cfg_a.data, bench=BENCH, panel=PANEL, features=FEATURES, outcomes=OUTCOMES,
                            vintages=VINTAGES, as_of=as_of)
        REPORT["tables"] = {k: json.loads(v.to_json(orient="records", date_format="iso")) for k, v in TABLES.items()
                            if not k.startswith("_")}
        for name, t in TABLES.items():
            if not name.startswith("_"):
                print(name)
                show(t.head(20))
    if TABLES is not None:
        with LOG.phase("save"):
            write_workbook(paths.output / "explore.xlsx", {k.lstrip("_")[:31]: v for k, v in TABLES.items()})
else:
    LOG.skipped("tables", "checks failed")
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = explore_block(REPORT)
try:
    (paths.output / "explore.txt").write_text(block + "\\n", encoding="utf-8")
except Exception as exc:
    print(f"(could not save to Drive: {exc})")
LOG.finish()
print(block)
""")


def to_source(text):
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


# the build stamp: a hash of every cell with the placeholder in place, so it is reproducible
import hashlib  # noqa: E402
import re  # noqa: E402
# the release number (aem_research.RELEASE) goes into the cells first, so the stamp changes with it
_REL = re.search(r"^RELEASE = (\d+)", (Path(__file__).resolve().parents[1] / "aem_research" / "__init__.py")
                 .read_text(encoding="utf-8"), re.M).group(1)
CELLS = [(kind, text.replace("__RELEASE__", _REL)) for kind, text in CELLS]
_BUILD = hashlib.sha256("\x1e".join(text for _, text in CELLS).encode("utf-8")).hexdigest()[:12]
CELLS = [(kind, text.replace("__BUILD__", _BUILD)) for kind, text in CELLS]

cells = []
for i, (kind, text) in enumerate(CELLS):
    cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": to_source(text)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT} with {len(cells)} cells")
