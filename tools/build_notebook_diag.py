"""Generate notebooks/03_universe_a_diagnostic.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_diag.py notebooks/03_universe_a_diagnostic.ipynb

It reads what notebook 01 saved on Drive (the feature tables of A and B, the run report)
and prints one block: why universe A is the size it is. No downloads; counts and ratios
only, so it is safe before the lock.
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
# AEM – 03 universe A diagnostic (counts only)

**Why this exists.** The Nov 9, 2026 run of notebook 01 left universe A (the SEC price panel) with 147 windows, where the vendor universe B has about 40,000 on the same dates. This notebook finds which filter removes the companies, from the tables notebook 01 already saved on Drive. It downloads nothing and takes about two minutes.

**What it reads:** `cache/features` (A) and `cache/features_vendor` (B) and `output/coverage_report.json`, all in `My Drive/aem-research/`.

**What it shows:** the universe filters one at a time (how many windows pass each, and how many would pass without it), what lies under them (share counts, size, 13F holders), and, for every window of the vendor universe, what the same company's row in A says. Counts and ratios only – no returns – so the pre-registration stays clean.

**Run:** Runtime → Run all, then paste the block printed at the end into the chat (it is also saved as `diag_universe_a.txt` in `My Drive/aem-research/output/`).
""")

code("""
# ── Setup: mount Drive, fetch the repo, import the package ──
import os, sys, shutil, subprocess, json

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
from aem_research import diagnostics as dg
from aem_research.config import load_config
from aem_research.env import resolve_paths
from aem_research.io_utils import load_table, show

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "03_universe_a_diagnostic.ipynb", NOTEBOOK_BUILD,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/03_universe_a_diagnostic.ipynb")
cfg = load_config(os.path.join(REPO_DIR, "config", "preregistration.yaml"))
paths = resolve_paths(DRIVE_ROOT, cfg.get("data_sources.simfin_cache_dirs"))
print(f"aem_research {aem_research.__version__} @ {COMMIT} · pandas {pd.__version__} · numpy {np.__version__}")
""")

code("""
# ── Inputs: notebook 01's feature tables and run report ──
A = load_table(paths.cache / "features")
V = load_table(paths.cache / "features_vendor")
if A is None or V is None:
    raise FileNotFoundError(f"Notebook 01's feature tables are not in {paths.cache} – run notebook 01 first.")
REP = {}
rep_path = paths.output / "coverage_report.json"
if rep_path.exists():
    REP = json.loads(rep_path.read_text())
K = (REP.get("sec_check") or {}).get("liquidity_k")
if K is None and "liquid" in A.columns and A["liquid"].astype(bool).any():
    K = int(pd.to_numeric(A.loc[A["liquid"].astype(bool), "breadth"], errors="coerce").min())
FIRST_A = pd.Timestamp(cfg.get("price_basis.first_formation_date"))
print(f"A: {len(A):,} rows, {A['start_date'].nunique()} start dates · B (vendor): {len(V):,} rows · "
      f"13F-holder minimum used: {K} · A starts {FIRST_A.date()} · run of {REP.get('generated_utc', 'n/a')}")
""")

code("""
# ── Diagnostics (counts and ratios only) ──
FUN_A = dg.universe_funnel(A, cfg.data, start=FIRST_A)
FUN_V = dg.universe_funnel(V, cfg.data, start=FIRST_A)
print("Universe A (SEC panel) – windows passing each filter alone, so far, and without it:")
show(FUN_A)
print("Vendor universe on the same dates (dollar-volume screen in place of 13F holders):")
show(FUN_V)
DET_A = dg.detail(A, cfg.data, start=FIRST_A, min_breadth=K)
DET_V = dg.detail(V, cfg.data, start=FIRST_A, min_breadth=K)
show(pd.DataFrame({"A": pd.Series(DET_A, dtype=object), "vendor": pd.Series(DET_V, dtype=object)}))
CMP = dg.compare_universes(A, V, cfg.data, start=FIRST_A, min_breadth=K)
show(pd.Series(CMP, name="vendor universe windows, looked up in A"))
WHAT = dg.what_if(A, cfg.data, start=FIRST_A, ks=sorted({3, 10, 20, 30, int(K or 40), 60}))
show(pd.Series(WHAT, name="universe A under other choices (counts only)"))
EX = dg.examples(A, V, cfg.data, start=FIRST_A, n=10)
print("Ten vendor-universe windows whose A row fails (A values from SEC data; ratios A over vendor):")
show(EX)
PER_DATE = (A[A["start_date"] >= FIRST_A].groupby("start_date")
            .agg(rows=("ticker", "size"), universe=("in_universe", "sum")))
VPER = V[(V["start_date"] >= FIRST_A) & V["in_universe"].astype(bool)].groupby("start_date").size().rename("vendor_universe")
PER_DATE = PER_DATE.join(VPER, how="outer").fillna(0).astype(int)
show(PER_DATE)
""")

code("""
# ── The block to paste ──
def _fmt(d):
    return " · ".join(f"{k} {f'{v:.1%}' if isinstance(v, float) and abs(v) <= 1 else v}" for k, v in d.items())


lines = ["AEM DIAGNOSTIC 03 – paste everything between the rulers into the chat", "=" * 72,
         f"repo {COMMIT} · notebook {NOTEBOOK_BUILD} · pandas {pd.__version__} · numpy {np.__version__} · "
         f"notebook 01 run {REP.get('generated_utc', 'n/a')} "
         f"(repo {REP.get('repo_commit', 'n/a')}) · holders minimum {K}",
         "FUNNEL A  " + _fmt(dg.funnel_summary(FUN_A)),
         "FUNNEL V  " + _fmt(dg.funnel_summary(FUN_V)),
         "DETAIL A  " + _fmt(DET_A),
         "DETAIL V  " + _fmt(DET_V),
         "A VS V    " + _fmt(CMP),
         "WHAT IF   " + _fmt(WHAT),
         "PER DATE  " + " · ".join(f"{d:%Y-%m} {r.rows}/{r.universe}/{r.vendor_universe}"
                                   for d, r in PER_DATE.iloc[::4].iterrows()) + "  (A rows / A universe / vendor universe)",
         "DTYPES A  " + _fmt(dg.dtypes_line(A)),
         "DTYPES V  " + _fmt(dg.dtypes_line(V))]
for r in EX.itertuples(index=False):
    lines.append("EXAMPLE   " + " · ".join(f"{k} {v}" for k, v in r._asdict().items()))
lines.append("=" * 72)
BLOCK = "\\n".join(lines)
print(BLOCK)
try:
    (paths.output / "diag_universe_a.txt").write_text(BLOCK + "\\n", encoding="utf-8")
except Exception as exc:
    print(f"(could not save to Drive: {exc})")
""")


def to_source(text):
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


# the build stamp: a hash of every cell with the placeholder in place, so it is reproducible
import hashlib  # noqa: E402
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
