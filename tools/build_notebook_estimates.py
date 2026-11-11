"""Generate notebooks/07_estimates.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_estimates.py notebooks/07_estimates.ipynb

It runs Paper B's estimators for H2 to H6 on the locked data run: the features notebook 01
saved and the outcomes notebook 04 computed from them. No download, a minute or two. It
refuses unless the lock names exactly that notebook-01 run and every gate passed in it.
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
# AEM – 07 the signals against the base rates (H2 to H6) · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**What this does.** Runs the pre-registered estimators on the locked data run. **H2** – the fallen-quality cohort with an insider purchase touches 3x within three years more often than the base rate (H2a any purchase, H2b opportunistic insiders; Holm across the two), with the insider-sale placebo, the comparison inside the cohort (H2c) and the ablation of each condition. **H3** – whether 3x in three years is as frequent as 6.24x in five, on the same windows, and the five-year multiple that is. **H4** – each AEM signal's likelihood ratio for 10x within five years, fitted on start dates to 2016 and tested on 2017 to 2021 (Holm across the eight); the fitted 10x model against the unit-weighted checklist and the base rate by Brier score. **M1** – the ordered logit on the six buckets. Then the exploratory H5 and H6, the calendar years of the touches, delisting classes, and the robustness versions the saved tables allow.

**Intervals** come from a firm-cluster bootstrap (2,000 resamples, the seed in the pre-registration): all of a firm's windows are resampled together.

**When it runs.** After Paper B's lock, on the notebook-01 run it names, after notebook 04 has computed the outcomes. Nothing is downloaded.

**Run:** Runtime → Run all, then paste the block (also saved as `results_h2_h6.txt` in `My Drive/aem-research/output/`, with `aem_results_h2_h6.xlsx`).
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
from aem_research import estimators as est
from aem_research.config import data_run_problems, load_config
from aem_research.env import resolve_paths
from aem_research.export import write_workbook
from aem_research.io_utils import load_table, save_table, show
from aem_research.report import estimates_block, failed_gates
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "07_estimates.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/07_estimates.ipynb")
print(f"Notebook 07 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
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
# ── Estimates ──
EST = None
if not problems:
    with LOG.phase("inputs"):
        FEATURES = load_table(paths.cache / "features")
        OUTCOMES = load_table(paths.output / "outcomes")
        missing = [n for n, t in (("the features (notebook 01)", FEATURES), ("the outcomes (notebook 04)", OUTCOMES)) if t is None]
        if missing:
            raise FileNotFoundError(f"not on Drive: {', '.join(missing)}")
        boot = cfg.get("estimation.bootstrap")
    with LOG.phase("estimates"):
        EST = est.run_all(FEATURES, OUTCOMES, resamples=int(boot["resamples"]), seed=int(boot["seed"]))
        W = EST["windows"]
        REPORT["inputs"] = {"universe_windows": int(len(W)), "firms": int(W["firm"].nunique()),
                            "resolved_3y": int(W["hit3_3y"].notna().sum()), "resolved_5y": int(W["hit10_5y"].notna().sum()),
                            "start_dates": int(W["start_date"].nunique()), "bootstrap": f"{boot['resamples']} resamples, seed {boot['seed']}"}
        rec = lambda t: json.loads(t.to_json(orient="records")) if t is not None and len(t) else []  # noqa: E731
        h2_ = EST["h2"]
        REPORT["h2"] = {**{k: h2_[k] for k in ("main", "placebo", "h2c_test", "windows", "firms", "base_rate")},
                        "h2c": rec(h2_["h2c"]), "ablation": rec(h2_["ablation"])}
        REPORT["h3"] = EST["h3"]
        REPORT["h4"] = {**{k: v for k, v in EST["h4"].items() if k != "table"}, "table": rec(EST["h4"]["table"])}
        REPORT["m2"] = {**{k: v for k, v in EST["m2"].items() if not isinstance(v, pd.DataFrame)},
                        "coefficients": rec(EST["m2"]["coefficients"]), "brier": rec(EST["m2"]["brier"])}
        REPORT["m1"] = {**{k: v for k, v in EST["m1"].items() if not isinstance(v, pd.DataFrame)},
                        "coefficients": rec(EST["m1"]["coefficients"])}
        for key in ("h5", "h6", "touch_years", "delisting_classes", "robustness"):
            REPORT[key] = rec(EST[key])
        REPORT["not_computed"] = list(est.NOT_COMPUTED)
        print({k: v["verdict"] for k, v in h2_["main"].items()}, "· H3", EST["h3"].get("verdict"))
        show(EST["h4"]["table"])
    with LOG.phase("basis B"):
        FEATURES_B, OUTCOMES_B = load_table(paths.cache / "features_b"), load_table(paths.output / "outcomes_b")
        if FEATURES_B is not None and OUTCOMES_B is not None and len(OUTCOMES_B):
            wb_ = est.windows(FEATURES_B, OUTCOMES_B)
            rb = est.robustness(wb_, resamples=min(int(boot["resamples"]), 1000), seed=int(boot["seed"]))
            REPORT["robustness_b"] = rec(rb[rb["version"].eq("primary")].assign(version="B: daily vendor closes with the SEC fill"))
        else:
            print("Basis B's outcomes are not on Drive (a locked notebook-01 run writes them): B is not shown.")
    with LOG.phase("save"):
        sheets = {"H2": pd.DataFrame([{"test": k, **v} for k, v in h2_["main"].items()] + [{"test": "placebo", **h2_["placebo"]}]),
                  "H2 ablation": h2_["ablation"], "H2c": h2_["h2c"], "H3": pd.DataFrame([EST["h3"]]), "H4": EST["h4"]["table"],
                  "M2": EST["m2"]["coefficients"], "Brier": EST["m2"]["brier"], "M1": EST["m1"]["coefficients"], "H5": EST["h5"],
                  "H6": EST["h6"], "Touch years": EST["touch_years"], "Delisting classes": EST["delisting_classes"],
                  "Robustness": EST["robustness"]}
        write_workbook(paths.output / "aem_results_h2_h6.xlsx", {k: v for k, v in sheets.items() if v is not None and len(v)})
else:
    LOG.skipped("estimates", "checks failed")
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = estimates_block(REPORT)
try:
    (paths.output / "results_h2_h6.txt").write_text(block + "\\n", encoding="utf-8")
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
