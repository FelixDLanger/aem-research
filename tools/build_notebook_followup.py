"""Generate notebooks/08_followup.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_followup.py notebooks/08_followup.ipynb

Round one of the follow-up: the pre-registered robustness versions that the saved tables
allow (start years resampled with firms, a US$50m floor, no minimum price, Nasdaq delistings
at -30%, without windows crossing weak-evidence splits), and an inventory of the Drive that
the next round (new start dates for the confirmation, basis B, the calibration, Paper A's
exit values) is built against. No download.
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
# AEM – 08 follow-up, round one · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**What this does.** Two things, from the tables already on Drive (nothing is downloaded, about five minutes).

1. **Pre-registered robustness that needs no rebuilt panel:** the likelihood-ratio intervals with start years resampled as well as firms; the universe from US$50m; the universe without the minimum price; Nasdaq performance delistings at -30%; the results without windows that cross a weak-evidence split.
2. **An inventory of the Drive** – every folder, every saved table with its columns and date ranges – so that round two is written against what is actually there: the new start dates for the confirmation (C1 to C3, locked in `config/confirmation.yaml`), basis B, the quarter-end against daily calibration, multiples in real terms, the purchases-only insider classification, and Paper A's exit values.

Each step runs on its own: one that fails is reported in the block and the others still finish.

**Run:** Runtime → Run all, then paste the block (also saved as `followup.txt` in `My Drive/aem-research/output/`, with `aem_followup.xlsx`).
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
from aem_research import followup, outcomes
from aem_research.config import data_run_problems, load_config
from aem_research.env import resolve_paths
from aem_research.export import write_workbook
from aem_research.io_utils import load_table, save_table, show
from aem_research.report import failed_gates, followup_block
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "08_followup.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/08_followup.ipynb")
print(f"Notebook 08 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
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
# ── Inventory of the Drive (always, also when a check failed: round two is built from it) ──
cfg_c = load_config(os.path.join(REPO_DIR, "config", "confirmation.yaml"), os.path.join(REPO_DIR, "config", "confirmation_LOCK.json"))
REPORT["confirmation_status"] = f"{cfg_c.status} ({', '.join(h['id'] for h in cfg_c.hypotheses)})"
with LOG.phase("inventory"):
    names_cache = ("features", "features_vendor", "features_b", "delistings", "delistings_vendor", "delistings_b", "sec_panel",
                   "sec_splits", "sec_restated_splits", "regime")
    stems = [paths.cache / n for n in names_cache] + [paths.output / n for n in ("outcomes", "outcomes_b", "paper_a_analysis_sample")]
    stems += [paths.cache / "paper_a" / n for n in ("deals", "wsib_funds", "fts", "fts_exits", "fts_second", "terms", "second")]
    inv = followup.inventory(paths, stems)
    for extra in (paths.sec_dir, paths.stooq_dir, *paths.simfin_dirs):
        if extra.exists() and paths.work not in extra.parents and extra != paths.work:
            inv += followup.describe_dir(extra, depth=1)
    splits_ = load_table(paths.cache / "sec_splits")
    inv.append("sec_splits " + followup.value_counts_line(splits_, "evidence"))
    inv.append("sec_splits " + followup.value_counts_line(splits_, "kind"))
    inv.append("delistings " + followup.value_counts_line(load_table(paths.cache / "delistings"), "category"))
    exits_ = load_table(paths.cache / "paper_a" / "deals")
    inv.append("deals " + followup.value_counts_line(exits_, "exit_route"))
    REPORT["inventory"] = inv
    print("\\n".join(inv))
""")

code("""
# ── Robustness from the saved tables ──
BOOK = {}
if not problems:
    boot = cfg.get("estimation.bootstrap")
    kw = {"resamples": min(int(boot["resamples"]), 1000), "seed": int(boot["seed"])}
    rec = lambda t: json.loads(t.to_json(orient="records")) if t is not None and len(t) else []  # noqa: E731
    FEATURES, OUTCOMES, W = None, None, None
    with LOG.phase("inputs"):
        FEATURES = load_table(paths.cache / "features")
        OUTCOMES = load_table(paths.output / "outcomes")
        SEC_PANEL = load_table(paths.cache / "sec_panel")
        DELIST = load_table(paths.cache / "delistings")
        missing = [n for n, t in (("features", FEATURES), ("outcomes (notebook 04)", OUTCOMES), ("sec_panel", SEC_PANEL),
                                  ("delistings", DELIST)) if t is None]
        if missing:
            raise FileNotFoundError(f"not on Drive: {', '.join(missing)}")
        W = est.windows(FEATURES, OUTCOMES)
        as_of = pd.Timestamp(RUN["sec_prices"]["as_of"])
        gap = {"max_gap_days": cfg.get("outcomes.sec_max_gap_days", 200)}
        primary = est.headline(W, "primary", **kw)
    if W is not None:
        with LOG.phase("start years"):
            BOOK["Start years"] = est.two_way_table(W, resamples=int(boot["resamples"]), seed=int(boot["seed"]))
            REPORT["two_way"] = rec(BOOK["Start years"])
            show(BOOK["Start years"])
        with LOG.phase("universes"):
            rows, rates = [primary], []
            done = set(zip(OUTCOMES["ticker"], pd.to_datetime(OUTCOMES["start_date"])))
            for name, kwargs in (("market cap from US$50m", {"cap_min": 5e7}), ("no minimum price", {"min_price": 0.0})):
                fv = est.universe_variant(FEATURES, cfg.data, **kwargs)
                new = fv[fv["in_universe"] & ~pd.Series(list(zip(fv["ticker"], pd.to_datetime(fv["start_date"]))), index=fv.index).isin(done)]
                extra = outcomes.compute_outcomes(cfg, SEC_PANEL, outcomes.starts_from_features(new), DELIST, as_of=as_of,
                                                  overrides=gap) if len(new) else OUTCOMES.iloc[0:0]
                wv = est.windows(fv, pd.concat([OUTCOMES, extra], ignore_index=True))
                rows.append({**est.headline(wv, name, **kw), "windows_added": int(len(new))})
                rates.append({"version": name, "windows": int(len(wv)), "firms": int(wv["firm"].nunique()),
                              "touch_3x_3y": float(wv["hit3_3y"].mean()), "touch_10x_5y": float(wv["hit10_5y"].mean()),
                              "resolved_3y": int(wv["hit3_3y"].notna().sum()), "resolved_5y": int(wv["hit10_5y"].notna().sum())})
                print(name, "·", len(new), "windows added")
            rates.insert(0, {"version": "primary", "windows": int(len(W)), "firms": int(W["firm"].nunique()),
                             "touch_3x_3y": float(W["hit3_3y"].mean()), "touch_10x_5y": float(W["hit10_5y"].mean()),
                             "resolved_3y": int(W["hit3_3y"].notna().sum()), "resolved_5y": int(W["hit10_5y"].notna().sum())})
            BOOK["Universes"], BOOK["Base rates"] = pd.DataFrame(rows), pd.DataFrame(rates)
            REPORT["universes"], REPORT["base_rates"] = rec(BOOK["Universes"]), rec(BOOK["Base rates"])
        with LOG.phase("Nasdaq -30%"):
            spec = outcomes.OutcomeSpec.from_config(cfg.data)
            returns = tuple((k, -0.30 if k == "performance_nasdaq" else v) for k, v in spec.delist_returns)
            o30 = outcomes.compute_outcomes(cfg, SEC_PANEL, outcomes.starts_from_features(FEATURES), DELIST, as_of=as_of,
                                            overrides={**gap, "delist_returns": returns})
            w30 = est.windows(FEATURES, o30)
            rows = []
            for name, ww in (("primary (Nasdaq -55%)", W), ("Nasdaq -30%", w30)):
                m = est.m1(ww)["coefficients"].set_index("regressor")
                b = ww["bucket5"].dropna()
                rows.append({"version": name, "five_year_windows": int(len(b)), "mean_end_multiple": float(ww["end5"].mean()),
                             "share_lowest_bucket": float((b == b.min()).mean()), "touch_3x_3y": float(ww["hit3_3y"].mean()),
                             "m1_fcf_pos_coef": float(m.loc["fcf_pos", "coef"]), "m1_fcf_pos_p": float(m.loc["fcf_pos", "p"]),
                             "m1_low_dilution_coef": float(m.loc["low_dilution", "coef"]),
                             "m1_low_dilution_p": float(m.loc["low_dilution", "p"])})
            BOOK["Nasdaq -30%"] = pd.DataFrame(rows)
            REPORT["nasdaq"] = rec(BOOK["Nasdaq -30%"])
        with LOG.phase("weak splits"):
            splits_ = load_table(paths.cache / "sec_splits")
            crossed = est.weak_split_windows(W, splits_, 5)
            row = est.headline(W[~crossed], "without windows crossing a weak-evidence split", **kw)
            BOOK["Weak splits"] = pd.DataFrame([primary, {**row, "windows_excluded": int(crossed.sum())}])
            REPORT["weak_splits"] = rec(BOOK["Weak splits"])
    REPORT["pending"] = ["new start dates Oct 2023 to Jul 2025 with the confirmation tests C1 to C3", "basis B (daily vendor closes)",
                         "quarter-end against daily calibration", "multiples in real terms", "insider classification on purchases only",
                         "Paper A: the benchmark for 2024-2025, exit values, WSIB reports before March 2023"]
    if BOOK:
        with LOG.phase("save"):
            write_workbook(paths.output / "aem_followup.xlsx", {k: v for k, v in BOOK.items() if len(v)})
else:
    LOG.skipped("robustness", "checks failed")
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = followup_block(REPORT)
try:
    (paths.output / "followup.txt").write_text(block + "\\n", encoding="utf-8")
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
