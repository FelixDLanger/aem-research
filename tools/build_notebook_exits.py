"""Generate notebooks/09_exits.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_exits.py notebooks/09_exits.ipynb

Paper A's last exploratory step (A-X1, A-X2): the price each visible sale to a listed buyer
was made at, read from the buyer's 8-K and press release, against the entry value and the
public benchmark's multiple; and WSIB's quarterly reports before March 2023.
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
# AEM – 09 Paper A: exit values and WSIB's history · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**What this does.** The last exploratory step of Paper A (A-X1, A-X2 – exploratory by the locked plan, never a confirmatory test).

1. **Exit values.** For every completed deal that was later sold to a listed buyer, the buyer's 8-K and its press release are fetched from EDGAR (about 135 filings, a few minutes; texts are kept on Drive) and the stated price is read from the sentences around the target's name. The block prints the tables **and a seeded sample of 30 filings with the sentence each value came from**, which I read to measure the extraction's accuracy.
2. **Value between entry and exit.** Exit value over entry enterprise value, years held, the annual rate – by entry regime, exit regime and buyer type – and whether value growth follows the public benchmark's multiple over the same span.
3. **WSIB before March 2023.** The quarterly reports from 2016 on are requested again; the block names each one that is missing or could not be read, and extends the pace of distributions as far back as the reports go.

Needs the `SEC_USER_AGENT` secret (as notebook 02). Nothing the locked runs saved is overwritten. Each step runs on its own.

**Run:** Runtime → Run all, then paste the block (also saved as `paper_a_exits.txt` in `My Drive/aem-research/output/`, with `paper_a_exits.xlsx`).
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
from aem_research import paper_a_exits as pxe
from aem_research import paper_a_results as par
from aem_research.config import data_run_problems, load_config
from aem_research.env import resolve_paths
from aem_research.export import write_workbook
from aem_research.io_utils import load_table, save_table, show
from aem_research.data import sec, wsib
from aem_research.report import exits_block
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "09_exits.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/09_exits.ipynb")
print(f"Notebook 09 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
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
# ── Exit values: read the buyers' filings, then the tables ──
BOOK, EXITS, X = {}, None, None
OFFLINE = os.environ.get("AEM_SEC_OFFLINE") == "1"
rec = lambda t: json.loads(t.to_json(orient="records", date_format="iso")) if t is not None and len(t) else []  # noqa: E731
if not problems:
    DEALS, BENCH = None, None
    with LOG.phase("inputs"):
        DEALS = load_table(A_DIR / "deals")
        if DEALS is None:
            raise FileNotFoundError("notebook 02's deals table is not on Drive")
        BENCH = load_table(A_DIR / "bench_through_2025")          # notebook 08's benchmark, through 2025
        bench_source = "notebook 08 (through 2025)"
        if BENCH is None:
            book = paths.output / "paper_a_sample.xlsx"
            BENCH = pd.read_excel(book, sheet_name="Benchmark", engine="openpyxl") if book.exists() else None
            bench_source = "notebook 02 (through 2023)" if BENCH is not None else "none"
        cand = pxe.exit_candidates(DEALS)
        REPORT["inputs"] = {"deals": int(len(DEALS)), "sold_to_a_listed_buyer": int(len(cand)),
                            "re_ipo": int(DEALS["exit_route"].eq("re-IPO").sum()) if "exit_route" in DEALS.columns else 0,
                            "benchmark": bench_source,
                            "benchmark_last_date": str(pd.to_datetime(BENCH["start_date"]).max().date()) if BENCH is not None and len(BENCH) else "n/a"}
        print(REPORT["inputs"])
    if DEALS is not None:
        with LOG.phase("exit filings"):
            ua, ua_origin = sec.resolve_user_agent()
            client = None if OFFLINE else sec.SecClient(ua, min_interval=cfg_b.get("data_sources.sec_min_interval_seconds", 0.2))
            EXITS, st = pxe.read_exits(DEALS, client=client, cache_dir=A_DIR, offline=OFFLINE)
            REPORT["exit_reading"] = {"user_agent": "offline" if OFFLINE else ua_origin, **st}
            print(REPORT["exit_reading"])
    if EXITS is not None:
        with LOG.phase("exit values"):
            X = pxe.exit_table(DEALS, EXITS, BENCH, regime_split=cfg_a.get("estimation.regime_split", "2022-01-01"))
            S = pxe.summaries(X, draws=min(int(cfg_a.get("estimation.bootstrap_draws", 5000)), 5000),
                              seed=int(cfg_a.get("estimation.seed", 20261111)))
            REPORT["exit_counts"], REPORT["exit_groups"], REPORT["exit_tracking"] = S["counts"], rec(S["groups"]), S["tracking"]
            REPORT["exit_sample"] = pxe.reading_sample(X, EXITS, DEALS, n=30, seed=20261112)
            BOOK["Exit values"] = X.drop(columns=["exit_snippet"], errors="ignore")
            BOOK["Exit groups"] = S["groups"]
            BOOK["Exit reading"] = EXITS
            show(S["groups"])
            print(S["tracking"])
else:
    LOG.skipped("exit values", "checks failed")
""")

code("""
# ── WSIB: the quarterly reports from 2016 on, requested again ──
if not problems:
    with LOG.phase("WSIB history"):
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"], check=True)
        W_DIR = A_DIR / "wsib"
        WSIB, w = wsib.load(W_DIR, start_year=cfg_a.get("wsib.first_year", 2016), offline=OFFLINE,
                            manual_dir=paths.work / "data" / "wsib")
        dates_ = pd.to_datetime(WSIB["report_date"]) if len(WSIB) else pd.Series(dtype="datetime64[ns]")
        REPORT["wsib"] = {"reports_asked": w["reports"], "cached": w["cached"], "parsed_now": w["parsed"], "missing": w["missing"],
                          "failed": w["failed"], "fund_rows": w["rows"], "reports_read": int(dates_.nunique()),
                          "first_report": str(dates_.min().date()) if len(dates_) else "n/a",
                          "last_report": str(dates_.max().date()) if len(dates_) else "n/a"}
        REPORT["wsib_missing"] = ", ".join(w["missing_reports"]) or "none"
        REPORT["wsib_failed"] = "; ".join(w["failed_reports"]) or "none"
        # a report that downloaded but did not parse: its first lines, so the parser can be fitted to the older layout
        lines_ = []
        for f_ in sorted(W_DIR.glob("wsib_*_sample.txt"))[:2]:
            lines_ += [f"{f_.name}: {x_[:200]}" for x_ in f_.read_text(encoding="utf-8").splitlines()[:25]]
        REPORT["wsib_unparsed_lines"] = lines_
        print(REPORT["wsib"], "| missing:", REPORT["wsib_missing"], "| failed:", REPORT["wsib_failed"])
        if len(WSIB):
            save_table(WSIB, A_DIR / "wsib_funds_full")
            strategies = tuple(cfg_a.get("wsib.strategies_of_interest"))
            BOOK["WSIB pace"] = par.wsib_distribution_pace(WSIB, strategies)
            BOOK["WSIB vintages"] = par.wsib_vintages(WSIB, strategies)
            REPORT["wsib_pace"], REPORT["wsib_vintages"] = rec(BOOK["WSIB pace"]), rec(BOOK["WSIB vintages"])
            show(BOOK["WSIB pace"])
    if BOOK:
        with LOG.phase("save"):
            write_workbook(paths.output / "paper_a_exits.xlsx", {k: v for k, v in BOOK.items() if len(v)})
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = exits_block(REPORT)
try:
    (paths.output / "paper_a_exits.txt").write_text(block + "\\n", encoding="utf-8")
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
