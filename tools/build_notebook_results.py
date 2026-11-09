"""Generate notebooks/04_results.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_results.py notebooks/04_results.ipynb

It computes the study's outcomes from the tables notebook 01 saved on Drive, once the
pre-registration is locked: no download and no rebuild, a few minutes. It refuses unless
the lock names exactly that notebook-01 run as its data run and every pre-lock gate passed
in it, so the results can only come from the run that was checked before the lock.
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
# AEM – 04 results from the locked run · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**What this does.** Computes the study's outcomes – every universe window's touches and end multiple – from the tables notebook 01 saved on Drive, and prints H1: the share of windows that touched 3x and 10x within 3 and 5 years, and the five-year end multiples on the six-bucket grid with their mean, pooled, by regime and by start year. Nothing is downloaded or rebuilt; it takes a few minutes.

**When it runs.** Only after the pre-registration is locked, and only on the notebook-01 run the lock names as its data run (commit, version, notebook build and start time), in which every pre-lock gate passed. Otherwise it stops before computing anything and says why.

**Run:** Runtime → Run all, then paste the block printed at the end into the chat (it is also saved as `results_summary.txt` in `My Drive/aem-research/output/`, with `aem_results_h1.xlsx` and the outcome table).
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
from aem_research import outcomes
from aem_research.config import data_run_problems, load_config
from aem_research.env import resolve_paths
from aem_research.export import write_workbook
from aem_research.io_utils import load_table, save_table, show
from aem_research.report import failed_gates, results_block
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "04_results.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/04_results.ipynb")
print(f"Notebook 04 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
      f"pandas {pd.__version__} · Colab: {IN_COLAB}")
""")

code("""
# ── Checks: the lock, the data run it names, and that run's pre-lock gates ──
LOG = RunLog()
cfg = load_config(os.path.join(REPO_DIR, "config", "preregistration.yaml"))
paths = resolve_paths(DRIVE_ROOT, cfg.get("data_sources.simfin_cache_dirs"))
REPORT = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "repo_commit": COMMIT,
          "notebook_build": NOTEBOOK_BUILD, "release": NOTEBOOK_RELEASE, "config_sha256": cfg.sha256,
          "lock_status": cfg.status, "lock": dict(cfg.lock_record or {}), "versions": versions()}
run_path = paths.output / "coverage_report.json"
RUN = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
problems = data_run_problems(cfg, RUN)
if RUN is not None and cfg.locked:
    problems += [f"gate failed in the data run: {g}" for g in failed_gates(RUN)]
REPORT["problems"] = problems
if problems:
    print("Stopped before computing any outcome:")
    for p_ in problems:
        print("  -", p_)
else:
    dr = cfg.lock_record["data_run"]
    print(f"Locked {cfg.lock_record['locked_at_utc']} · data run v{dr.get('release')} {dr.get('repo_commit')} "
          f"(notebook {dr.get('notebook_build')}, {dr.get('generated_utc')}) is the run on Drive · "
          f"all {len(RUN['prelock'])} pre-lock entries passed")
""")

code("""
# ── Outcomes and H1 (primary price basis A: SEC quarter-end prices) ──
RESULT, H1 = None, None
if not problems:
    with LOG.phase("outcomes"):
        FEATURES = load_table(paths.cache / "features")
        DELIST = load_table(paths.cache / "delistings")
        SEC_PANEL = load_table(paths.cache / "sec_panel")
        missing = [n for n, t in (("features", FEATURES), ("delistings", DELIST), ("sec_panel", SEC_PANEL)) if t is None]
        if missing:
            raise FileNotFoundError(f"notebook 01's tables are missing on Drive: {', '.join(missing)}")
        as_of = pd.Timestamp(RUN["sec_prices"]["as_of"])
        starts = outcomes.starts_from_features(FEATURES)
        RESULT = outcomes.compute_outcomes(cfg, SEC_PANEL, starts, DELIST, as_of=as_of,
                                           overrides={"max_gap_days": cfg.get("outcomes.sec_max_gap_days", 200)})
        H1 = outcomes.h1_tables(RESULT, outcomes.OutcomeSpec.from_config(cfg.data))
        st = RESULT["status"].value_counts()
        REPORT["outcomes"] = {"start_windows": int(len(starts)), "rows": int(len(RESULT)),
                              **{k: int(st.get(k, 0)) for k in (outcomes.COMPLETE, outcomes.DELISTED, outcomes.ENDED,
                                                                outcomes.CENSORED, outcomes.NO_ENTRY)},
                              "as_of": str(as_of.date())}
        REPORT["h1"] = {k: v.to_dict("records") for k, v in H1.items()}
        print(f"{len(starts):,} start windows × {cfg.get('outcomes.horizons_years')} years · as of {as_of.date()}")
        show(H1["touch"][H1["touch"]["scope"] != "start year"])
        show(H1["buckets"])
        save_table(RESULT, paths.output / "outcomes")
        keep = [c for c in ("ticker", "start_date", "horizon_years", "status", "entry_date", "entry_price",
                            "end_multiple", "end_multiple_lower", "end_multiple_upper", "max_multiple", "bucket",
                            "touch_3x", "touch_10x", "delist_category") if c in RESULT.columns]
        write_workbook(paths.output / "aem_results_h1.xlsx",
                       {"H1 touch": H1["touch"], "H1 buckets": H1["buckets"], "Outcomes A": RESULT[keep]})
else:
    LOG.skipped("outcomes", "checks failed")
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = results_block(REPORT)
try:
    (paths.output / "results_summary.txt").write_text(block + "\\n", encoding="utf-8")
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
