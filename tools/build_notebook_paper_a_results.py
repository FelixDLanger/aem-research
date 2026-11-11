"""Generate notebooks/05_paper_a_results.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_paper_a_results.py notebooks/05_paper_a_results.ipynb

It computes Paper A's results from the tables notebook 02 saved on Drive, once Paper A's
pre-registration is locked: the three confirmatory tests and every exploratory table. No
download, a few minutes. It refuses unless the lock names exactly that notebook-02 run.
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
# AEM – 05 Paper A results from the locked run · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**What this does.** Computes Paper A's results from the tables notebook 02 saved on Drive. **Confirmatory** (pre-registered in `config/paper_a.yaml`, Holm-adjusted): A-H1 – sponsors pay more than the public small-cap benchmark multiple; A-H2 – that premium is smaller for entries in 2022 or later; A-H3 – a visible exit within five years became rarer for deals completed 2017–2021. **Exploratory** (every table reported, none a finding until confirmed on other data): the premium by buyer group, year, size, sector and regime; exit probabilities; years held by exit year; leverage and sponsors from the terms that missed the 90% gate, with their measured accuracy; stated against computed deal values; WSIB's buyout funds by vintage and the pace of their distributions.

**When it runs.** Only after Paper A's lock, and only on the notebook-02 run the lock names. Otherwise it stops before computing anything and says why. Nothing is downloaded; it reads the cached documents once to drop stock mergers.

**Run:** Runtime → Run all, then paste the block printed at the end into the chat (also saved as `paper_a_results.txt` in `My Drive/aem-research/output/`, with `paper_a_results.xlsx`).
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
from aem_research import paper_a_results as par
from aem_research.config import data_run_problems, load_config
from aem_research.env import resolve_paths
from aem_research.export import write_workbook
from aem_research.io_utils import load_table, save_table, show
from aem_research.report import paper_a_results_block
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "05_paper_a_results.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/05_paper_a_results.ipynb")
print(f"Notebook 05 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
      f"pandas {pd.__version__} · Colab: {IN_COLAB}")
""")

code("""
# ── Checks: Paper A's lock and the notebook-02 run it names ──
LOG = RunLog()
cfg_a = load_config(os.path.join(REPO_DIR, "config", "paper_a.yaml"), os.path.join(REPO_DIR, "config", "paper_a_LOCK.json"))
cfg_b = load_config(os.path.join(REPO_DIR, "config", "preregistration.yaml"))
paths = resolve_paths(DRIVE_ROOT, cfg_b.get("data_sources.simfin_cache_dirs"))
A_DIR = paths.cache / "paper_a"
REPORT = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "repo_commit": COMMIT,
          "notebook_build": NOTEBOOK_BUILD, "release": NOTEBOOK_RELEASE, "config_sha256": cfg_a.sha256,
          "lock_status": cfg_a.status, "lock": dict(cfg_a.lock_record or {}), "versions": versions()}
summary_path = paths.output / "paper_a_summary.txt"
RUN = par.run_identity(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
problems = [p_.replace("notebook 01", "notebook 02") for p_ in data_run_problems(cfg_a, RUN)]
REPORT["problems"] = problems
if problems:
    print("Stopped before computing any result:")
    for p_ in problems:
        print("  -", p_)
else:
    dr = cfg_a.lock_record["data_run"]
    print(f"Locked {cfg_a.lock_record['locked_at_utc']} · data run v{dr.get('release')} {dr.get('repo_commit')} "
          f"(notebook {dr.get('notebook_build')}, {dr.get('generated_utc')}) is the run on Drive")
""")

code("""
# ── Sample, confirmatory tests, exploratory tables ──
SAMPLE, CONF, EXPL = None, None, None
if not problems:
    with LOG.phase("sample"):
        DEALS = load_table(A_DIR / "deals")
        if DEALS is None:
            raise FileNotFoundError("notebook 02's deals table is not on Drive")
        book = paths.output / "paper_a_sample.xlsx"
        BENCH = pd.read_excel(book, sheet_name="Benchmark", engine="openpyxl") if book.exists() else None
        WSIB = load_table(A_DIR / "wsib_funds")
        DEALS["stock_mentions"] = par.stock_mentions(DEALS, A_DIR / "docs")
        as_of = pd.Timestamp(RUN["generated_utc"][:10])
        SAMPLE, funnel = par.analysis_sample(DEALS, BENCH, cfg_a.data, as_of=as_of)
        funnel["documents_scanned"] = int(DEALS["stock_mentions"].notna().sum())
        funnel["benchmark_dates"] = 0 if BENCH is None else int(len(BENCH))
        funnel["as_of"] = str(as_of.date())
        REPORT["funnel"] = funnel
        print(funnel)
    with LOG.phase("confirmatory"):
        CONF = par.confirmatory(SAMPLE, cfg_a.data)
        REPORT["confirmatory"] = CONF
        print(CONF["verdict"])
    with LOG.phase("exploratory"):
        EXPL = par.exploratory(SAMPLE, WSIB, cfg_a.data)
        REPORT["exploratory"] = {k: v.to_dict("records") for k, v in EXPL.items()}
        for name in ("premium by group", "exit probability", "WSIB by vintage"):
            print(name)
            show(EXPL[name].head(12))
    with LOG.phase("save"):
        save_table(SAMPLE, paths.output / "paper_a_analysis_sample")
        conf_rows = [{"test": "A-H1 / A-H2, core, EV/EBITDA", **CONF["premium"]}, {"test": "A-H3, core", **CONF["exits"]},
                     *[{"test": f"robustness: {k}", **v} for k, v in CONF["robustness"].items()]]
        verdicts = pd.DataFrame([{"hypothesis": k, "holm_p": CONF["holm"][k], "verdict": v} for k, v in CONF["verdict"].items()])
        keep = [c for c in ("deal_id", "target_name", "group", "buyer_type", "entry_date", "post_2022", "ev_ebitda", "bench_date",
                            "bench_ev_ebitda", "log_premium", "ev_revenue", "bench_ev_revenue", "log_premium_revenue",
                            "completion_date", "cohort", "exit_event", "exit_kind", "exit_date", "exit_years", "stock_mentions",
                            "check_price", "check_value") if c in SAMPLE.columns]
        write_workbook(paths.output / "paper_a_results.xlsx",
                       {"Verdicts": verdicts, "Confirmatory": pd.DataFrame(conf_rows), "Funnel": REPORT["funnel"],
                        **{f"X {k}"[:31]: v for k, v in EXPL.items()}, "Sample": SAMPLE[keep]})
else:
    LOG.skipped("results", "checks failed")
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = paper_a_results_block(REPORT)
try:
    (paths.output / "paper_a_results.txt").write_text(block + "\\n", encoding="utf-8")
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
