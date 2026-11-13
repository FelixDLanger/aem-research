"""Generate notebooks/11_accounting.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_accounting.py notebooks/11_accounting.ipynb

Paper B: a third batch of candidates that needs more of the financial statements than the
data run's extract kept (accruals, gross profitability, book-to-market, asset growth, R&D,
the full Piotroski score). Paper A: A-H3 on the exits read and accepted by hand.
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
# AEM – 11 accounting signals and the hand-read exits · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**Why a download.** The data run kept only the statement lines its features needed. Net income, total assets, equity, cost of revenue, R&D, liabilities and working capital are not on Drive, so the SEC's quarterly Financial Statement Data Sets (63 files, about 4 GB) are fetched once more and only those lines are kept, in their own folder (`data/sec/fsds_extra`). The data run's files are not touched. **Expect 30 to 60 minutes on the first run**; a rerun reads from Drive in a few minutes, and a file that fails is fetched again on the next run.

1. **Paper B, third batch (exploratory, fixed in advance):** accruals, gross profitability, book-to-market, asset growth, R&D to market value, return on assets, Piotroski's nine-part F-score – each on start dates to 2017, corrected for the number of candidates, counted only if it holds from 2018, with the new dates as a third look. Targets: 3x in three years, 10x in five, ended five years below half.
2. **Fourteen combinations on end multiples**, among them Piotroski's own test (cheap on book value with a high against a low score), the fallen by score, R&D-heavy companies, profitable value.
3. **The flexible model again**, now on the measures of all three batches.
4. **Paper A:** A-H3 on the exits read and accepted by hand after notebook 10 (7 sales, 5 to 6 re-IPOs), strict and wide.

Needs the `SEC_USER_AGENT` secret.

**Run:** Runtime → Run all, then paste the block (also saved as `accounting.txt` in `My Drive/aem-research/output/`, with `aem_accounting.xlsx`).
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
from aem_research.report import accounting_block
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "11_accounting.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/11_accounting.ipynb")
print(f"Notebook 11 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
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
# ── The statement lines the data run did not keep (a download on the first run) ──
BOOK, FACTS_X, SUBS_X = {}, None, None
OFFLINE = os.environ.get("AEM_SEC_OFFLINE") == "1"
rec = lambda t: json.loads(t.to_json(orient="records", date_format="iso")) if t is not None and len(t) else []  # noqa: E731
if not problems:
    with LOG.phase("statement lines"):
        from datetime import datetime, timezone
        from aem_research import accounting
        ua, ua_origin = sec.resolve_user_agent()
        client = None if OFFLINE else sec.SecClient(ua, min_interval=cfg_b.get("data_sources.sec_min_interval_seconds", 0.2))
        y0 = int(cfg_b.get("starts.years")[0])
        SUBS_X, FACTS_X, st = accounting.load_extra(paths.sec_dir, y0 - 2, datetime.now(timezone.utc).year, client=client,
                                                    offline=OFFLINE, local_dir=paths.sec_dir / "manual")
        groups = FACTS_X["tag"].astype(str).map(lambda t: accounting.EXTRA_TAGS.get(t, ("other",))[0]).value_counts().to_dict() if len(FACTS_X) else {}
        REPORT["statements"] = {"user_agent": "offline" if OFFLINE else ua_origin, "files": st["files"], "cached": st["cached"],
                                "downloaded": st["downloaded"] + st["local"], "missing": st["missing"], "bad": st["bad"],
                                "filings": int(len(SUBS_X)), "values": int(len(FACTS_X))}
        REPORT["statement_groups"] = groups
        REPORT["statement_gaps"] = ", ".join(st["missing_labels"] + st["bad_labels"]) or "none"
        print(REPORT["statements"], groups)
else:
    LOG.skipped("statement lines", "checks failed")
""")

code("""
# ── Paper B · the accounting measures, point in time, and the third batch (exploratory) ──
if FACTS_X is not None and len(FACTS_X):
    WA = WNA = None
    with LOG.phase("measures"):
        from aem_research import confirm, estimators as est, scan
        boot = cfg_b.get("estimation.bootstrap")
        F_, O_ = load_table(paths.cache / "features"), load_table(paths.output / "outcomes")
        FC_, OC_ = load_table(paths.cache / "features_confirm"), load_table(paths.output / "outcomes_confirm")
        PANEL = load_table(paths.cache / "sec_panel")
        w_locked = est.windows(F_, O_)
        w_new = confirm.windows(FC_, OC_) if FC_ is not None and OC_ is not None else None
        dates = set(pd.to_datetime(w_locked["start_date"])) | (set(pd.to_datetime(w_new["start_date"])) if w_new is not None else set())
        SNAPS = accounting.snapshots(FACTS_X, SUBS_X, dates, current_months=cfg_b.get("universe.current_filer_months", 15))
        save_table(SNAPS, paths.cache / "accounting_snapshots")
        WA = accounting.measures(w_locked, SNAPS)
        WNA = accounting.measures(w_new, SNAPS) if w_new is not None else None
        REPORT["coverage"] = accounting.coverage(WA)
        REPORT["coverage_new"] = accounting.coverage(WNA) if WNA is not None else {}
        REPORT["f_score"] = {str(int(k)): int(v) for k, v in WA["f_score"].value_counts().sort_index().items()}
        print(REPORT["coverage"])
        del FACTS_X
    if WA is not None:
        with LOG.phase("third batch"):
            ACC = scan.scan(WA, WNA, panel=PANEL, resamples=500, seed=int(boot["seed"]), continuous=accounting.ACC_CONTINUOUS,
                            flags=accounting.ACC_FLAGS)
            BOOK["B accounting scan"] = ACC
            REPORT["scan"] = rec(ACC)
            REPORT["scan_counts"] = {"tested": int(ACC["discovery_p"].notna().sum()) if len(ACC) else 0,
                                     "discovery_significant": int((ACC["verdict"] != "").sum()) if len(ACC) else 0,
                                     "held": int(ACC["verdict"].str.startswith("held").sum()) if len(ACC) else 0}
        with LOG.phase("combinations"):
            ASYM = scan.asymmetry(WA, WNA, panel=PANEL, resamples=500, seed=int(boot["seed"]), combos=accounting.ACC_COMBOS,
                                  build=accounting.with_combos)
            BOOK["B accounting combinations"] = ASYM
            REPORT["combinations"] = rec(ASYM)
        with LOG.phase("outcome model"):
            try:
                import sklearn  # noqa: F401
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=True)
            BOOK["B outcome model, all batches"] = scan.outcome_model(WA, WNA, panel=PANEL, seed=int(boot["seed"]),
                                                                      extra_continuous=accounting.ACC_CONTINUOUS,
                                                                      extra_flags=accounting.ACC_FLAGS, build=accounting.with_combos)
            REPORT["outcome_model"] = rec(BOOK["B outcome model, all batches"])
            show(BOOK["B outcome model, all batches"])
elif not problems:
    LOG.skipped("third batch", "no statement lines were read")
""")

code("""
# ── Paper A · A-H3 on the exits read and accepted by hand ──
if not problems:
    with LOG.phase("A-H3 on hand-read exits"):
        SAMPLE, VALID = load_table(paths.output / "paper_a_analysis_sample"), load_table(A_DIR / "exits_validated")
        if SAMPLE is None or VALID is None:
            raise FileNotFoundError("notebook 05's analysis sample or notebook 10's checked exits are not on Drive")
        CHOSEN = pxe.chosen_exits(VALID)
        rows, counts = [], []
        for level in ("wide", "strict"):
            s2, c = pxe.with_verified_exits(SAMPLE, CHOSEN, level=level)
            counts.append(c)
            rows.append({"measure": f"hand-read exits, {level} · core sample", **par.exit_tests(s2, cfg_a.data)})
            rows.append({"measure": f"hand-read exits, {level} · all deals", **par.exit_tests(s2, cfg_a.data, group=None)})
            if level == "wide":
                ev = s2[s2["exit_event"]].assign(years=lambda d: d["exit_years"].round(2))
                REPORT["verified_exits"] = [f"{r.target_name} · completed {pd.Timestamp(r.completion_date).date()} · {r.exit_kind} "
                                            f"after {r.years} years · group {r.group} · cohort {r.cohort or 'none'}" for r in ev.itertuples()]
        BOOK["A-H3 hand-read"] = pd.DataFrame(rows)
        REPORT["h3"], REPORT["h3_counts"] = rec(BOOK["A-H3 hand-read"]), counts
        show(BOOK["A-H3 hand-read"])
if BOOK:
    with LOG.phase("save"):
        write_workbook(paths.output / "aem_accounting.xlsx", {k: v for k, v in BOOK.items() if len(v)})
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = accounting_block(REPORT)
try:
    (paths.output / "accounting.txt").write_text(block + "\\n", encoding="utf-8")
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
