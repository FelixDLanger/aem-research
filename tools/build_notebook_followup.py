"""Generate notebooks/08_followup.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_followup.py notebooks/08_followup.ipynb

Round one: the pre-registered robustness versions that the saved tables allow (start years
resampled with firms, a US$50m floor, no minimum price, Nasdaq delistings at -30%, without
windows crossing weak-evidence splits) and an inventory of the Drive. Round two, built
against that inventory: the data run's features rebuilt from the caches for the locked and
the new start dates (the locked dates must reproduce the saved table), the confirmation
tests C1 to C3 on the new dates, the purchases-only insider classification, multiples in
real terms, basis B with the quarter-end against daily calibration, and Paper A's benchmark
through 2025. No download.
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
# AEM – 08 follow-up, rounds one and two · v__RELEASE__

**Version v__RELEASE__.** The paste block's first line repeats it; an older open copy stops at setup and names the current version.

**What this does.** Everything comes from the files already on Drive – nothing is downloaded, and nothing the data run saved is overwritten (new tables get their own names). About 20 to 40 minutes; the first cells are quick.

1. **Round one (as in v15):** the robustness versions the saved tables allow, and the inventory of the Drive.
2. **Rebuild and check:** the data run's features are rebuilt from the caches for the locked start dates and seven new ones (January 2024 to July 2025). The rebuilt rows of the locked dates must match the saved table (99.5% on the universe and on every signal); if they do not, the block says where and no new outcome is computed.
3. **Confirmation (C1 to C3, locked in `config/confirmation.yaml`):** one- and two-year outcomes of the new start dates, then the three tests with Holm, and the plan's robustness versions.
4. **Leftovers:** the insider classification on purchases only; five-year multiples in real terms; basis B (daily vendor closes) with the quarter-end against daily calibration; Paper A's public benchmark through 2025.

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
from aem_research import confirm, followup, outcomes
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
BOOK, W = {}, None
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
else:
    LOG.skipped("robustness", "checks failed")
""")

code("""
# ── Round two · rebuild the features from the caches and check them against the data run ──
import gc
INP, SH, FEATS_ALL, FEATS_NEW, REPRO = None, None, None, None, None
confirm_problems = []
for _n in ("o30", "w30", "wv", "fv", "new", "extra"):              # round one's working tables
    globals().pop(_n, None)
gc.collect()
if not problems and W is not None:
    if not cfg_c.locked:
        confirm_problems.append(f"the confirmation plan is not locked (status {cfg_c.status})")
    K_RUN, ATT = (RUN.get("sec_check") or {}).get("liquidity_k"), (RUN.get("features") or {}).get("attention_signal")
    if K_RUN is None or ATT is None:
        confirm_problems.append("the data run's report lacks the liquidity threshold or the attention signal")
    with LOG.phase("load caches"):
        INP = confirm.load_inputs(paths, cfg, log=lambda *a, **k: None)
        REPORT["rebuild_files"] = INP["files"]
        print(INP["files"])
    if INP is not None and not confirm_problems:
        with LOG.phase("rebuild features"):
            DATES = confirm.all_dates(cfg, cfg_c)
            FEATS_ALL, SH = confirm.build_features(INP, cfg, SEC_PANEL, DATES, k=K_RUN, attention=ATT)
            NEW_DATES = [pd.Timestamp(d) for d in cfg_c.get("dates.one_year")]
            print(f"{len(FEATS_ALL):,} rows on {FEATS_ALL['start_date'].nunique()} start dates · SEC panel as of {SH['sec_as_of'].date()} · "
                  f"vendor panel as of {SH['as_of'].date()}")
            REPORT["rebuild"] = {"start_dates": int(FEATS_ALL["start_date"].nunique()), "sec_as_of": str(SH["sec_as_of"].date()),
                                 "vendor_as_of": str(SH["as_of"].date()), "vendor_dates": len(SH["form_v"])}
            INP["simfin"] = INP["stq"] = INP["eod_prices"] = None      # the vendor panel is built; free the source tables
            gc.collect()
        if FEATS_ALL is not None:
            with LOG.phase("reproduction"):
                REPRO = confirm.reproduction(FEATS_ALL, FEATURES)
                REPRO["as_of_matches"] = bool(SH["sec_as_of"] == as_of)
                REPRO["k"] = K_RUN
                REPORT["reproduction"] = REPRO
                show(pd.Series(REPRO, name="rebuilt against saved"))
                if not REPRO["passed"]:
                    confirm_problems.append(f"the rebuilt features do not reproduce the data run (universe {REPRO['universe_agreement']:.2%}, "
                                            f"weakest signal {REPRO['weakest_agreement']:.2%}; needed {confirm.MIN_AGREEMENT:.1%})")
                if not REPRO["as_of_matches"]:
                    confirm_problems.append(f"the 13F data on Drive now end {SH['sec_as_of'].date()}, the data run's panel {as_of.date()}")
                FEATS_NEW = FEATS_ALL[FEATS_ALL["start_date"].isin(NEW_DATES)]
                missing_dates = sorted(set(NEW_DATES) - set(FEATS_NEW.loc[FEATS_NEW["in_universe"].fillna(False).astype(bool), "start_date"]))
                if missing_dates:
                    confirm_problems.append("no universe window on " + ", ".join(str(d.date()) for d in missing_dates))
    REPORT["confirm_problems"] = confirm_problems
    for p_ in confirm_problems:
        print("Round two stops before the new outcomes:", p_)
else:
    LOG.skipped("round two", "checks failed or round one's inputs are missing")
""")

code("""
# ── Round two · Paper A's benchmark through 2025 and the purchases-only classification (no new outcome) ──
if FEATS_ALL is not None and REPRO is not None and REPRO["passed"]:
    with LOG.phase("Paper A benchmark"):
        from aem_research import paper_a_explore as pax, takeprivate as tp
        cfg_a = load_config(os.path.join(REPO_DIR, "config", "paper_a.yaml"), os.path.join(REPO_DIR, "config", "paper_a_LOCK.json"))
        bench = tp.benchmark(FEATS_ALL, INP["facts"], INP["subs"], current_months=cfg_a.get("sample.current_filer_months", 15))
        save_table(bench, paths.cache / "paper_a" / "bench_through_2025")
        BOOK["Benchmark by year"] = pax.benchmark_path(bench)
        REPORT["benchmark"] = rec(BOOK["Benchmark by year"])
        show(BOOK["Benchmark by year"])
    with LOG.phase("purchases only"):
        feats_p, _ = confirm.build_features(INP, cfg, SEC_PANEL, DATES, k=K_RUN, attention=ATT, purchases_only=True, shared=SH)
        wp = est.windows(feats_p[feats_p["start_date"].isin(set(pd.to_datetime(FEATURES["start_date"])))], OUTCOMES)
        rows = []
        for name, ww in (("primary (purchases and sales classify an insider)", W), ("purchases only", wp)):
            r2 = est.h2(ww, **kw)
            opp = est._flag(ww["insider_opportunistic"]).eq(1)
            rows.append({**est.headline(ww, name, **kw), "h2b_windows": r2["main"]["H2b"]["with_signal"],
                         "h2b_ci_low": r2["main"]["H2b"]["ci_low"], "h2b_ci_high": r2["main"]["H2b"]["ci_high"],
                         "opportunistic_windows": int(opp.sum())})
        BOOK["Purchases only"] = pd.DataFrame(rows)
        REPORT["purchases_only"] = rec(BOOK["Purchases only"])
        del feats_p
    INP["facts"] = INP["ins_tx"] = INP["f13_breadth"] = None   # the largest tables, not needed again
    for _k in ("fund", "pf_sec", "breadth"):
        SH.pop(_k, None)
    gc.collect()
""")

code("""
# ── Round two · the confirmation: outcomes of the new start dates, then C1 to C3 ──
if FEATS_NEW is not None and not confirm_problems:
    WC = None
    with LOG.phase("confirmation outcomes"):
        save_table(FEATS_NEW, paths.cache / "features_confirm")
        OUT_C = confirm.new_outcomes(cfg, SEC_PANEL, FEATS_NEW, DELIST, SH["sec_as_of"], sec_gap=gap["max_gap_days"])
        save_table(OUT_C, paths.output / "outcomes_confirm")
        WC = confirm.windows(FEATS_NEW, OUT_C)
        print(OUT_C.groupby(["horizon_years", "status"]).size().to_string())
    if WC is not None:
        with LOG.phase("C1 to C3"):
            RES_C = confirm.tests(WC, cfg_c, resamples=int(boot["resamples"]), seed=int(boot["seed"]))
            REPORT["confirmation"] = {k_: RES_C[k_] for k_ in ("C1", "C2", "C3")}
            REPORT["confirmation_robustness"] = RES_C["robustness"]
            REPORT["confirmation_dates"] = rec(RES_C["by_date"])
            BOOK["Confirmation"] = pd.DataFrame([{"hypothesis": k_, **RES_C[k_]} for k_ in ("C1", "C2", "C3")])
            BOOK["Confirmation robustness"] = pd.DataFrame(RES_C["robustness"])
            BOOK["Confirmation dates"] = RES_C["by_date"]
            show(BOOK["Confirmation"])
elif not problems:
    LOG.skipped("confirmation", "round two's checks failed" if confirm_problems else "no rebuilt features")
""")

code("""
# ── Round two · multiples in real terms; basis B and the quarter-end against daily calibration ──
if not problems and W is not None:
    with LOG.phase("real terms"):
        from aem_research.data import fred
        series, _ = fred.load(paths.cache / "fred", offline=True, manual_dir=paths.work / "data" / "fred", log=lambda *a, **k: None)
        if series.get("cpi_nsa") is None or not len(series["cpi_nsa"]):
            raise FileNotFoundError("the CPI series (CPIAUCNS) is not in the FRED cache on Drive")
        BOOK["Real terms"] = confirm.real_terms(W, series["cpi_nsa"])
        REPORT["real_terms"] = rec(BOOK["Real terms"])
        show(BOOK["Real terms"])
    if SH is not None:
        with LOG.phase("basis B"):
            FB, DV, DB = (load_table(paths.cache / n) for n in ("features_b", "delistings_vendor", "delistings_b"))
            EVENTS = load_table(paths.cache / "sec_splits")
            if FB is None or DV is None or DB is None or not len(FB):
                raise FileNotFoundError("features_b or its delistings are not on Drive")
            OUT_B, CALIB = confirm.basis_b(cfg, SH["panel"], SH["as_of"], FB, DV, DB, SEC_PANEL, EVENTS, paths.sec_dir, SH["bridge"],
                                           sec_gap=gap["max_gap_days"])
            save_table(OUT_B, paths.output / "outcomes_b_followup")
            wb_ = est.windows(FB.assign(in_universe=True), OUT_B)
            b0 = pd.to_datetime(FB["start_date"]).min()
            wa_ = W[W["start_date"] >= b0]
            BOOK["Basis B"] = pd.DataFrame([est.headline(wa_, f"A from {b0.date()}", **kw), est.headline(wb_, "B (daily vendor closes)", **kw)])
            BOOK["Basis B base rates"] = pd.DataFrame([confirm.base_rate_row(f"A from {b0.date()}", wa_),
                                                       confirm.base_rate_row("B (daily vendor closes)", wb_)])
            BOOK["Calibration"] = CALIB
            REPORT["basis_b"], REPORT["basis_b_rates"], REPORT["calibration"] = (rec(BOOK["Basis B"]), rec(BOOK["Basis B base rates"]),
                                                                                 rec(CALIB))
            show(BOOK["Basis B base rates"])
            show(CALIB)
    REPORT["pending"] = ["Paper A: exit values (a document extraction that needs its own reading check)",
                         "Paper A: WSIB reports before March 2023 (a download)"]
    if BOOK:
        with LOG.phase("save"):
            write_workbook(paths.output / "aem_followup.xlsx", {k: v for k, v in BOOK.items() if len(v)})
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
