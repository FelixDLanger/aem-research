"""Generate notebooks/10_final.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_final.py notebooks/10_final.ipynb

Paper A: the exit measure rebuilt (every hit read, kept only where a sentence says the
target was acquired), A-H3 on it as a disclosed deviation. Paper B: whether any combination
shifts the whole outcome (exploratory).
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
# AEM – 10 final: Paper A's exits rebuilt, Paper B's net question · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**Why.** Notebook 09 read the exit filings for the first time and found that most of notebook 02's "sold to a listed buyer" matches were not sales of the target. So the exit measure is rebuilt here, and the data's last open question is answered.

1. **Exits, read one by one.** Every 8-K hit of the exit search (up to ten filings a deal) is fetched from EDGAR and kept only if a sentence says the target itself was acquired; tier A also names the deal's sponsor, tier B does not. **Every kept sale is printed with its sentence** – I read all of them – plus a handful of rejected documents and the twelve re-IPO matches.
2. **A-H3 on the checked exits.** The pre-registered test (visible exit within five years, deals completed 2012–2016 against 2017–2021), on the checked sales. This is a correction made after the lock and after the first result was seen; the paper reports it as a deviation next to the original.
3. **Exit values** where the filing states one, against the entry value.
4. **Paper B, the net question (exploratory):** eighteen combinations fixed in advance – free cash flow, the fall, momentum, cash, a floor of net cash (also net of five years of burn), buybacks, the AEM cohort, the 10x corner, an avoid list – each on five-year end multiples before and after 2018 and on the new dates: fewer wipe-outs, more multiples, both, or neither. And a flexible model that predicts the end multiple from all measures, scored out of time.

Needs the `SEC_USER_AGENT` secret. About 15 to 25 minutes (over a thousand filings are fetched once, then kept on Drive). Nothing the locked runs saved is overwritten.

**Run:** Runtime → Run all, then paste the block (also saved as `final.txt` in `My Drive/aem-research/output/`, with `aem_final.xlsx`).
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
from aem_research.report import final_block
from aem_research.runlog import RunLog, versions

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
NOTEBOOK_RELEASE = __RELEASE__
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "10_final.ipynb", NOTEBOOK_BUILD, release=NOTEBOOK_RELEASE,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/10_final.ipynb")
print(f"Notebook 10 · v{NOTEBOOK_RELEASE} · aem_research {aem_research.__version__} @ {COMMIT} · "
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
# ── Paper A · the exit measure rebuilt: read every hit, keep what says the target was acquired ──
BOOK, VALID, CHOSEN = {}, None, None
OFFLINE = os.environ.get("AEM_SEC_OFFLINE") == "1"
rec = lambda t: json.loads(t.to_json(orient="records", date_format="iso")) if t is not None and len(t) else []  # noqa: E731
if not problems:
    DEALS = SAMPLE = None
    with LOG.phase("inputs"):
        DEALS, HITS = load_table(A_DIR / "deals"), load_table(A_DIR / "fts_exits")
        SAMPLE = load_table(paths.output / "paper_a_analysis_sample")
        BENCH = load_table(A_DIR / "bench_through_2025")
        if DEALS is None or HITS is None or SAMPLE is None:
            raise FileNotFoundError("notebook 02's deals, its exit search or notebook 05's analysis sample is not on Drive")
        REPORT["inputs"] = {"deals": int(len(DEALS)), "exit_search_hits": int(len(HITS)),
                            "sold_to_a_listed_buyer_before": int(DEALS["exit_route"].eq("sold to a listed buyer").sum()),
                            "re_ipo": int(DEALS["exit_route"].eq("re-IPO").sum()), "analysis_sample": int(len(SAMPLE))}
        print(REPORT["inputs"])
    if DEALS is not None and SAMPLE is not None:
        with LOG.phase("exit documents"):
            ua, ua_origin = sec.resolve_user_agent()
            client = None if OFFLINE else sec.SecClient(ua, min_interval=cfg_b.get("data_sources.sec_min_interval_seconds", 0.2))
            VALID, st = pxe.validate_exits(DEALS, HITS, client=client, cache_dir=A_DIR, offline=OFFLINE)
            CHOSEN = pxe.chosen_exits(VALID)
            before = set(DEALS.loc[DEALS["exit_route"].eq("sold to a listed buyer"), "deal_id"])
            REPORT["exit_check"] = {"user_agent": "offline" if OFFLINE else ua_origin, **st,
                                    "deals_with_a_checked_sale": int(len(CHOSEN)), "tier_a": int(CHOSEN["tier"].eq("A").sum()),
                                    "tier_b": int(CHOSEN["tier"].eq("B").sum()),
                                    "of_the_earlier_sales_kept": int(CHOSEN["deal_id"].isin(before).sum()),
                                    "with_a_stated_value": int(CHOSEN["exit_value_usd"].notna().sum())}
            REPORT["exit_reasons"] = VALID.loc[~VALID["tier"].isin(["A", "B"]), "reason"].fillna("").str.slice(0, 60).value_counts().head(6).to_dict()
            REPORT["exit_lines"] = pxe.exit_lines(DEALS, VALID, CHOSEN)
            BOOK["Checked exits"] = CHOSEN
            BOOK["All exit documents"] = VALID.drop(columns=["exit_snippet"], errors="ignore")
            print(REPORT["exit_check"])
    if CHOSEN is not None:
        with LOG.phase("A-H3 on checked exits"):
            rows = [{"measure": "as locked (notebook 02's matches, mostly not sales)", **par.exit_tests(SAMPLE, cfg_a.data)}]
            for label, tiers, reipo in (("checked sales, tiers A and B, and re-IPOs", ("A", "B"), True),
                                        ("checked sales, tier A only, and re-IPOs", ("A",), True),
                                        ("checked sales only, tiers A and B", ("A", "B"), False)):
                s2 = pxe.with_checked_exits(SAMPLE, CHOSEN, tiers=tiers, use_reipo=reipo)
                rows.append({"measure": label, **par.exit_tests(s2, cfg_a.data)})
                if label.startswith("checked sales, tiers A and B, and"):
                    rows.append({"measure": label + " · all deals, not only the core sample", **par.exit_tests(s2, cfg_a.data, group=None)})
                    S2 = s2
            BOOK["A-H3"] = pd.DataFrame(rows)
            REPORT["h3"] = rec(BOOK["A-H3"])
            REPORT["as_of"] = str(pd.Timestamp(S2.attrs["as_of"]).date())
            ev = S2[S2["exit_event"]]
            by_year = (ev.assign(year=pd.to_datetime(ev["completion_date"]).dt.year).groupby("year")
                       .agg(exits=("deal_id", "size"), median_years_held=("exit_years", "median")).reset_index())
            done = S2.assign(year=pd.to_datetime(S2["completion_date"]).dt.year).groupby("year").size().rename("completed").reset_index()
            BOOK["Exits by completion year"] = done.merge(by_year, on="year", how="left").fillna({"exits": 0})
            REPORT["exits_by_year"] = rec(BOOK["Exits by completion year"])
            from aem_research import paper_a_explore as pax
            BOOK["Exit slowdown"] = pax.exit_slowdown(S2, as_of=S2.attrs["as_of"])
            REPORT["slowdown"] = rec(BOOK["Exit slowdown"])
            show(BOOK["A-H3"])
        with LOG.phase("exit values"):
            d2 = DEALS.drop(columns=["sale_date", "buyer_name"], errors="ignore").merge(
                CHOSEN.rename(columns={"sale_date_checked": "sale_date", "buyer_checked": "buyer_name"})[["deal_id", "sale_date", "buyer_name"]],
                on="deal_id", how="left")
            X = pxe.exit_table(d2, CHOSEN.rename(columns={"url": "exit_url"}), BENCH, regime_split=cfg_a.get("estimation.regime_split", "2022-01-01"))
            SV = pxe.summaries(X, draws=5000, seed=int(cfg_a.get("estimation.seed", 20261111)))
            REPORT["exit_counts"], REPORT["exit_groups"], REPORT["exit_tracking"] = SV["counts"], rec(SV["groups"]), SV["tracking"]
            BOOK["Exit values"] = X.drop(columns=["exit_snippet"], errors="ignore")
else:
    LOG.skipped("exits", "checks failed")
""")

code("""
# ── Paper B · the net question: does any combination shift the whole outcome? (exploratory) ──
if not problems:
    with LOG.phase("Paper B inputs"):
        from aem_research import confirm, estimators as est, scan
        boot = cfg_b.get("estimation.bootstrap")
        F_, O_ = load_table(paths.cache / "features"), load_table(paths.output / "outcomes")
        FC_, OC_ = load_table(paths.cache / "features_confirm"), load_table(paths.output / "outcomes_confirm")
        PANEL = load_table(paths.cache / "sec_panel")
        w_locked = est.windows(F_, O_)
        w_new = confirm.windows(FC_, OC_) if FC_ is not None and OC_ is not None else None
        REPORT["b_inputs"] = {"locked_windows": int(len(w_locked)), "new_windows": "not on Drive" if w_new is None else int(len(w_new))}
    with LOG.phase("combinations"):
        ASYM = scan.asymmetry(w_locked, w_new, panel=PANEL, resamples=500, seed=int(boot["seed"]))
        BOOK["B combinations"] = ASYM
        REPORT["combinations"] = rec(ASYM)
        show(ASYM[["combination", "sample", "windows", "median_end", "median_rest", "below_half", "below_half_rest", "at_least_2x",
                   "at_least_2x_rest", "shape", "verdict"]])
    with LOG.phase("outcome model"):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=True)
        BOOK["B outcome model"] = scan.outcome_model(w_locked, w_new, panel=PANEL, seed=int(boot["seed"]))
        REPORT["outcome_model"] = rec(BOOK["B outcome model"])
        show(BOOK["B outcome model"])
if BOOK:
    with LOG.phase("save"):
        write_workbook(paths.output / "aem_final.xlsx", {k: v for k, v in BOOK.items() if len(v)})
""")

code("""
# ── The block to paste ──
REPORT["run"] = LOG.as_dict()
block = final_block(REPORT)
try:
    (paths.output / "final.txt").write_text(block + "\\n", encoding="utf-8")
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
