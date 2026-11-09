"""Generate notebooks/01_aem_base_rate_skeleton.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook.py notebooks/01_aem_base_rate_skeleton.ipynb

Every phase after setup runs inside `with LOG.phase(...)`: an error is recorded and
printed, the run goes on, and the block printed at the end carries the details. Only
the test suite and the look-ahead guard stop or skip work on purpose.
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


def phase(name, body, before="", after=""):
    """A code cell: `before` (defaults that must exist even if the phase fails), then the
    body under LOG.phase(name), then `after`."""
    ind = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.strip("\n").split("\n"))
    parts = [before.strip("\n")] if before.strip() else []
    parts.append(f'with LOG.phase("{name}"):\n{ind}')
    if after.strip():
        parts.append(after.strip("\n"))
    code("\n".join(parts))


md("""
# AEM base-rate study – 01 coverage audit and point-in-time features

**What this does.** Builds the study's primary price basis from SEC filings alone (D10: 13F quarter-end prices, the SEC fails-to-deliver files, splits found from the holders' share counts), checks the SimFin cache and the Stooq bulk file as the robustness panel (B), builds the point-in-time features from the SEC's free data sets and the macro regime variables from FRED, and saves everything to Drive, including an Excel workbook. The outcome engine is built and tested, but stays locked until the pre-registration is locked.

**Before you run** (Runtime → Run all):
1. Colab secret `SEC_USER_AGENT` = `Your Name your@email` (key icon in the left bar, notebook access on). The SEC asks automated clients to identify themselves.
2. The Stooq bulk file `d_us_txt.zip`, still zipped, in `My Drive/aem-research/data/stooq/` – see Phase B.

**Run time.** The downloads run on Google's servers, not over your own connection: about 9 GB from the SEC on the first run, one to two hours in all. Keep the tab open and the device awake. Every file is cached on Drive as it finishes, so after a disconnect, Run all resumes where it stopped. The first run after Nov 9, 2026 reads the 13F files again (they now keep each security's top holders) and downloads the fails-to-deliver files once – about as long as the first run. Later runs take a few minutes.

**When it ends,** the last cell prints one block – results, timings and any errors. Paste it into the chat (it is also saved as `run_summary.txt` in `My Drive/aem-research/output/`). A phase that fails is recorded in that block and the run carries on. If the test suite fails, the run stops at once and prints the same block. If Colab crashes or disconnects before the end, `run_progress.txt` in the same folder shows how far it got – paste its last lines; Run all then resumes from the cached files.

| Phase | Needs the lock? | What it does |
|---|---|---|
| 0 Setup | no | Mount Drive, fetch the repo, show the lock status, run the offline test suite |
| A SimFin cache | no | Coverage of your cached prices, including names that stopped trading |
| B Stooq bulk | no | Parse the zip, coverage, overlap with SimFin, adjustment basis |
| C SEC index | no | Form 25 delistings and merger filings, cached per quarter |
| D Verdict | no | Delisting coverage → unpriced share of the vendor panel |
| F Features | no | XBRL financials, Form 4 insiders, 13F holdings, SEC fails-to-deliver files, FRED regime → SEC price panel, universe, cohorts, signals, power – counts only |
| F Calibration | no | Quarter-end vs daily touch rates on 2009–2012 start dates, before the sample – ratios only |
| E Outcomes | **yes** | Touch and end multiples – only after the lock |

Nothing in phases A–F looks at returns, so running them before the lock keeps the pre-registration clean. Raw vendor data stays on Drive; never commit it to the repository.
""")

code("""
# ── Phase 0 · Setup: mount Drive, fetch the repo, import the package ──
import os, sys, shutil, subprocess
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
    del sys.modules[_m]  # pick up a fresh pull without restarting the runtime

import numpy as np
import pandas as pd
import aem_research
from aem_research.io_utils import save_table, show, write_json

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
print(f"aem_research {aem_research.__version__} @ {COMMIT} · pandas {pd.__version__} · Colab: {IN_COLAB}")
""")

code("""
# ── Phase 0 · Pre-registration: config hash and lock status ──
from aem_research.config import load_config
from aem_research.runlog import RunLog, folder_size_gb, versions

LOG = RunLog()
cfg = load_config(os.path.join(REPO_DIR, "config", "preregistration.yaml"))
print(cfg.summary())
REPORT = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
          "repo_commit": COMMIT, "config_sha256": cfg.sha256, "lock_status": cfg.status, "versions": versions()}
""")

code("""
# ── Phase 0 · Offline test suite: must pass before anything touches data ──
from aem_research.report import paste_block


def _stop_block():
    REPORT["run"] = LOG.as_dict()
    print()
    print(paste_block(REPORT))


with LOG.must_pass("0 tests", on_stop=_stop_block):
    try:
        import pytest  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                         cwd=REPO_DIR, capture_output=True, text=True, env={**os.environ, "COLUMNS": "240"})
    print((res.stdout or res.stderr)[-1500:])
    if res.returncode != 0:
        failed = [ln for ln in (res.stdout or "").splitlines() if ln.startswith(("FAILED", "ERROR"))]
        raise RuntimeError("Test suite failed – " + ("; ".join(failed[:6]) or (res.stdout or res.stderr)[-300:])
                           + ". The run stops here: paste the block below into the chat.")
""")

code("""
# ── Phase 0 · Folders on Drive ──
from aem_research.env import resolve_paths

paths = resolve_paths(DRIVE_ROOT, cfg.get("data_sources.simfin_cache_dirs")).ensure()
print(paths.describe())
LOG.attach_progress(paths.output / "run_progress.txt", header=f"· repo {COMMIT} · config {cfg.sha256[:12]}")
print("Progress log (shows how far a run got if Colab crashes or disconnects):", paths.output / "run_progress.txt")
YEARS = cfg.get("coverage.benchmark_years")
SURV_DATES = [f"{y}-01-03" for y in (2010, 2012, 2014, 2016, 2018, 2020)]
""")

md("""
## Phase A – SimFin cache

Reads `cvm-research/data/raw/us_prices.parquet` from the CVM pipeline, or SimFin's own CSV cache. The survivorship table answers the key question directly: of the names trading on each date, how many later stopped trading? A survivor-only source shows almost none.
""")

phase("A SimFin", before="""
# ── Phase A · SimFin cache ──
from aem_research import coverage as cv
from aem_research.data import simfin_cache
from aem_research.guards import survivorship_report

simfin = simfin_cache.SimFinCache(None, None, {}, ["not loaded"])
sim_sp, sim_summary = cv.spans(None), {"source": "SimFin cache", "tickers": 0}
""", body="""
simfin = simfin_cache.load(paths.simfin_dirs)
print("Files found:", simfin.files)
for note in simfin.notes:
    print("note:", note)
if simfin.found:
    sim_sp = cv.spans(simfin.prices)
    sim_summary = cv.source_summary("SimFin cache", sim_sp)
    show(pd.Series(sim_summary, name="SimFin"))
    print("Price columns:", [c for c in simfin.prices.columns if c not in ("ticker", "date")])
    if simfin.companies is not None:
        n_cik = int(simfin.companies["cik"].notna().sum()) if "cik" in simfin.companies.columns else 0
        print(f"Companies table: {len(simfin.companies):,} rows, {n_cik:,} with a CIK")
    show(survivorship_report(sim_sp, SURV_DATES, sim_sp["last_date"].max()))
else:
    print("No SimFin price cache found – check the paths above.")
""", after="""
REPORT["simfin"] = {**sim_summary, **{f"file_{k}": v for k, v in simfin.files.items()}}
""")

md("""
## Phase B – Stooq bulk file

Stooq gates its bulk files behind a CAPTCHA, so the download is one manual step; parsing and caching are automatic. If the zip isn't on Drive yet, the cell prints the steps and the audit continues without Stooq. The overlap rows show whether Stooq adds delisted names SimFin lacks, and whether its closes are raw or adjusted.
""")

phase("B Stooq", before="""
# ── Phase B · Stooq bulk file ──
from aem_research.data import stooq

stq, stq_sp = None, cv.spans(None)
stq_summary = {"source": "Stooq bulk", "tickers": 0, "status": "not loaded"}
""", body="""
stq = stooq.load_or_build(paths.stooq_dir, paths.cache)
if stq is None:
    print(stooq.INSTRUCTIONS)
    stq_summary["status"] = "zip not on Drive yet"
else:
    print(f"{os.path.basename(stq.zip_path)} – {'cached parse' if stq.from_cache else 'parsed now'} – {stq.stats}")
    stq_sp = cv.spans(stq.prices)
    stq_summary = cv.source_summary("Stooq bulk", stq_sp)
    show(pd.Series(stq_summary, name="Stooq"))
    show(stq.groups["group"].value_counts().rename("tickers per folder"))
    show(survivorship_report(stq_sp, SURV_DATES, stq_sp["last_date"].max()))
if simfin.found and stq is not None and len(stq_sp):
    as_of_both = max(sim_sp["last_date"].max(), stq_sp["last_date"].max())
    REPORT["overlap"] = {**cv.stopped_overlap(sim_sp, stq_sp, as_of_both),
                         **cv.live_overlap(sim_sp, stq_sp, as_of_both - pd.Timedelta(days=365))}
    REPORT["adjustment"] = cv.adjustment_check(simfin.prices, stq.prices)
    show(pd.Series({**REPORT["overlap"], **REPORT["adjustment"]}, name="SimFin vs Stooq"))
""", after="""
REPORT["stooq"] = stq_summary
""")

md("""
## Phase C – SEC index: delistings (Form 25) and merger filings

Every exchange delisting is filed as Form 25 or 25-NSE; merger filings (DEFM14A, SC 14D9, SC TO-T, 425 and others) separate takeovers from failures in the delisting classification. Each quarterly EDGAR index is downloaded once, both subsets are kept and cached on Drive. Without the `SEC_USER_AGENT` secret a generic client name is tried; if the SEC refuses it, the cell says so and the verdict stays undetermined.
""")

phase("C SEC index", before="""
# ── Phase C · SEC index ──
from aem_research.data import sec

OFFLINE = os.environ.get("AEM_SEC_OFFLINE") == "1"
ua, ua_origin = sec.resolve_user_agent()
print(f"SEC user agent: {ua_origin}")
client = None if OFFLINE else sec.SecClient(ua, min_interval=cfg.get("data_sources.sec_min_interval_seconds", 0.2))
sec_note, sec_stats = "", {}
IDX_YEARS = (YEARS[0] - 1, YEARS[1])   # one extra year: merger filings precede the delisting
form25 = merger = pd.DataFrame(columns=["cik", "company", "form", "date_filed", "filename"])
""", body="""
try:
    idx_tables, sec_stats = sec.load_index_subsets(paths.sec_dir, *IDX_YEARS, client=client, offline=OFFLINE)
except sec.SecAccessError as exc:
    sec_note = str(exc)
    print(sec_note)
    idx_tables, sec_stats = sec.load_index_subsets(paths.sec_dir, *IDX_YEARS, offline=True)
form25, merger = idx_tables["form25"], idx_tables["merger"]
print(f"Form 25 filings on file: {len(form25):,} · merger filings: {len(merger):,} – {sec_stats}")
""", after="""
REPORT["sec"] = {"user_agent": ua_origin, **sec_stats, **({"error": sec_note} if sec_note else {})}
""")

md("""
## Phase D – Coverage and the EODHD verdict

Each Form 25 event is matched to a ticker through SimFin's CIK, then checked against both price sources (series ending within ±45 days of the filing). Events whose prices continue are exchange moves or non-common securities and are excluded; unmapped events count as uncovered, so coverage is conservative. Coverage and attrition give the unpriced share of the start universe – which is also the relative width of the base-rate bounds – and the pre-registered thresholds turn it into a verdict.
""")

phase("D verdict", before="""
# ── Phase D · Coverage and the EODHD verdict ──
verdict = {"verdict": "undetermined", "reason": "Phase D did not finish – see the errors at the end.", "basis": ""}
coverage, coverage_hi, att = None, None, None
""", body="""
spans_present = [sp for sp in (sim_sp, stq_sp) if len(sp)]
as_of_all = max(sp["last_date"].max() for sp in spans_present) if spans_present else None
union = cv.union_spans(sim_sp, stq_sp)
events = cv.form25_events(form25, cfg.get("coverage.benchmark_forms"), YEARS)
detail, table = cv.benchmark_against_form25(events, simfin.companies, sim_sp, stq_sp,
                                            cfg.get("coverage.match_window_days", 45))
basis = "no SEC benchmark"
if table is not None and len(table) and table["events"].sum() > 0:
    show(table)
    q_only = cv.stopped_by_year(cv.stooq_only(sim_sp, stq_sp), as_of_all, YEARS)
    coverage, coverage_hi = cv.coverage_range(table, q_only)
    tot = table[["events", "simfin", "stooq_adds", "unmapped"]].sum()
    basis = "Form 25 events matched via SimFin CIKs; unmapped events counted as uncovered"
    print(f"Delisting coverage: {coverage:.1%} conservative (SimFin {tot['simfin'] / tot['events']:.1%}, "
          f"Stooq adds {tot['stooq_adds'] / tot['events']:.1%}, unmapped {tot['unmapped'] / tot['events']:.1%}); "
          f"up to {coverage_hi:.1%} if Stooq's extra delisted names fill the unmapped events")
    gaps = detail[detail["status"].isin(["missing", "unmapped"])]
    if len(gaps):
        print("Sample of uncovered events – are these real delistings?")
        show(gaps.sample(min(15, len(gaps)), random_state=0)[["date_filed", "company", "cik", "ticker", "status"]])
elif len(events) and as_of_all is not None:
    ratio_table = cv.count_ratio(events, cv.stopped_by_year(union, as_of_all, YEARS))
    show(ratio_table)
    if ratio_table["form25_issuers"].sum():
        coverage = float(min(ratio_table["stopped_in_data"].sum() / ratio_table["form25_issuers"].sum(), 1.0))
        basis = "count ratio without a CIK bridge (rough; understates coverage)"

if coverage is not None and as_of_all is not None:
    att = cv.attrition(union, cfg.get("coverage.attrition_start_years"),
                       cfg.get("coverage.attrition_horizon_years"), as_of_all)
    show(att)


def unpriced_share(cov):
    if cov is None or att is None or not len(att):
        return None
    shares = [cv.missing_share(r.live, r.stopped_within_h, cov)[1] for r in att.itertuples() if r.live]
    shares = [s for s in shares if np.isfinite(s)]
    return float(np.median(shares)) if shares else None


THRESHOLDS = cfg.get("coverage.eodhd_thresholds")
verdict = cv.eodhd_verdict(unpriced_share(coverage), coverage, THRESHOLDS, basis)
if coverage_hi is not None and coverage is not None and coverage_hi > coverage:
    alt = cv.eodhd_verdict(unpriced_share(coverage_hi), coverage_hi, THRESHOLDS, basis)
    verdict["optimistic"] = {"coverage": round(coverage_hi, 4), "verdict": alt["verdict"],
                             "missing_share": alt.get("missing_share")}
    if alt["verdict"] != verdict["verdict"]:
        verdict["reason"] += (f" If Stooq's extra delisted names fill the unmapped events (coverage "
                              f"{coverage_hi:.0%}), the verdict would be '{alt['verdict']}' – check the "
                              "sample of uncovered events above.")
""", after="""
REPORT["verdict"] = verdict
print(f"EODHD: {verdict['verdict'].upper()} – {verdict.get('reason', '')}")
""")

code("""
# ── Save the audit to Drive now, and again after each later phase ──
from aem_research.report import paste_block, render_markdown


def save_report():
    try:
        json_path = write_json(REPORT, paths.output / "coverage_report.json")
        md_path = paths.output / "coverage_report.md"
        md_path.write_text(render_markdown(REPORT), encoding="utf-8")
        return json_path, md_path
    except Exception as exc:
        print(f"Could not save the report: {type(exc).__name__}: {exc}")
        return ()


print("Saved", ", ".join(p.name for p in save_report()), "to", paths.output)
""")

md("""
## Phase F – Point-in-time features (counts only)

Three free SEC data sets, each downloaded once per file and cached on Drive as a small extract (the raw zips are never stored):
- **Financial Statement Data Sets** (XBRL from 10-K and 10-Q): trailing free cash flow, net cash, revenue, shares, dilution, the S1 proxy.
- **Insider Transactions Data Sets** (Form 4): officer and director open-market purchases (and sales, for the placebo), split into routine and opportunistic insiders (Cohen, Malloy and Pomorski 2012), plus a symbol-to-CIK bridge.
- **Form 13F Data Sets**: institutional breadth (the attention proxy), implied quarter-end prices and each security's ten largest holders – the backbone of the SEC price panel (D10).
- **Fails-to-deliver files** (SEC, twice a month): CUSIP, ticker and the previous day's close for every security with failed deliveries – the bridge to companies SimFin never covered, prices before 2013 and the last price before a company disappears.

The next cells build the universe at every formation date (market cap US$100m–2bn, price ≥ US$1, liquidity, SPAC shells out), the fallen-quality cohort, the placebo and the H4 signals, run the pre-lock and power checks, and add the FRED regime variables. Everything is counts – no returns – and a look-ahead guard rejects the features if any of them uses information published after its start date. If the SEC blocks a download, zips saved by hand into `My Drive/aem-research/data/sec/manual/` (same file names as on sec.gov) are used instead.
""")

code("""
# ── Phase F · SEC data sets: XBRL financials, Form 4 insiders, Form 13F holdings ──
from aem_research.data import fsds, ftd, insider, thirteenf

SEC_MANUAL = paths.sec_dir / "manual"
Y0 = int(cfg.get("starts.years")[0])
THIS_YEAR = datetime.now(timezone.utc).year
kw = dict(client=client, offline=OFFLINE, local_dir=SEC_MANUAL)
subs = facts = ins_tx = ins_bridge = f13_breadth = f13_names = FTD_NAMES = None
sec_set_stats, SEC_SET_LINES = {}, {}


def keep_stats(name, st):
    sec_set_stats[name] = {k: v for k, v in st.items() if not k.endswith("_labels")}
    for k in ("missing_labels", "bad_labels"):
        if st.get(k):
            sec_set_stats[name][k.replace("_labels", "_files")] = ", ".join(st[k][:12])
    line = (f"{st['files']} files: {st['cached']} cached, {st['downloaded']} downloaded, {st['local']} manual, "
            f"{st['missing']} missing, {st['bad']} bad")
    if st.get("missing_labels"):
        line += f" (missing {', '.join(st['missing_labels'][:6])})"
    if st.get("bad_labels"):
        line += f" (bad {', '.join(st['bad_labels'][:6])})"
    SEC_SET_LINES[name] = line


with LOG.phase("F XBRL financials"):
    print("Financial Statement Data Sets – the longest step on the first run")
    subs, facts, st = fsds.load(paths.sec_dir, Y0 - 2, THIS_YEAR, **kw)
    keep_stats("fsds", st)
with LOG.phase("F Form 4 insiders"):
    print("Insider Transactions Data Sets")
    ins_tx, ins_bridge, st = insider.load(paths.sec_dir, cfg.get("data_sources.insider_history_start_year", Y0 - 4),
                                          THIS_YEAR, **kw)
    keep_stats("insider", st)
with LOG.phase("F 13F holdings"):
    print("Form 13F Data Sets (re-read once after Nov 8, 2026: the top holders find splits)")
    f13_breadth, f13_names, st = thirteenf.load(paths.sec_dir, **kw)
    keep_stats("form13f", st)
with LOG.phase("F fails-to-deliver"):
    print("SEC fails-to-deliver files (half-month, from 2009): the CUSIP-ticker bridge and daily closes")
    FTD_NAMES, st = ftd.load(paths.sec_dir, start=cfg.get("data_sources.sec_fails_start", ftd.START), **kw)
    keep_stats("fails", st)
if sec_set_stats:
    show(pd.DataFrame(sec_set_stats).T)
REPORT["sec_datasets"] = SEC_SET_LINES
""")

md("""
## Phase F – EODHD gap-fill (optional)

Not needed any more and off by default: D10 replaced it with the SEC price panel below, and EODHD's personal plans do not cover publishing results. With the Colab secret `EODHD_API_KEY` it still fills the vendor panel (B) with delisted names for a private cross-check; without the secret it is skipped.
""")

phase("F EODHD gap-fill", before="""
# ── Phase F · EODHD gap-fill (optional) ──
from aem_research import features as ft
from aem_research.data import eodhd

EOD_PRICES, EOD_EXCHANGE = None, None
EOD_KEY = eodhd.resolve_key()
EOD_DIR = paths.cache / "eodhd"
""", body="""
if EOD_KEY is None and not eodhd.cached(EOD_DIR):
    print("No EODHD_API_KEY secret and no EODHD data on Drive – skipped; the survivorship gap shows as HOLES.")
else:
    eod_offline = OFFLINE or EOD_KEY is None
    eod_client = None if eod_offline else eodhd.Client(EOD_KEY)
    syms, sym_st = eodhd.symbol_lists(eod_client, EOD_DIR, offline=eod_offline)
    EOD_EXCHANGE = eodhd.exchange_map(syms)
    bridges0 = [b for b in (ins_bridge, ft.instance_bridge(subs) if subs is not None and len(subs) else None)
                if b is not None and len(b)]
    sim_ciks = (simfin.companies["cik"].dropna() if simfin.found and simfin.companies is not None
                and "cik" in simfin.companies.columns else [])
    want = eodhd.candidates(syms, bridges0, subs["cik"].unique() if subs is not None else [], sim_ciks)
    print(f"EODHD candidates: {len(want):,} tickers ({int(want['delisted'].sum()):,} delisted) tie to XBRL filers SimFin lacks")
    EOD_PRICES, eod_st = eodhd.fetch_eod(want["code"], eod_client, EOD_DIR, offline=eod_offline)
    REPORT["eodhd"] = {"key": "colab secret" if EOD_KEY else "none (Drive cache only)",
                       "symbols": f"{sym_st['symbols']:,} ({sym_st['source']})", "candidates": f"{len(want):,}",
                       "fetched": eod_st["fetched"], "cached": eod_st["cached"], "empty": eod_st["empty"],
                       "missing": eod_st["missing"], "tickers_with_prices": eod_st.get("tickers", 0),
                       "exchange_map": f"{len(EOD_EXCHANGE):,}", "api_calls": eod_client.calls if eod_client else 0}
    print(REPORT["eodhd"])
""")

md("""
## Phase F – SEC price panel (the primary price basis, D10)

Every price the primary analysis uses comes from SEC filings, so the companies that later delisted are priced exactly like the survivors and the published numbers rest on public data. Each company gets one quarter-end price per quarter: the median of market value over shares held across the 13F filers of its main CUSIP (at least three), else the fails-to-deliver close nearest the quarter-end. Splits are found from the holders' share counts (a split multiplies every holder's shares by the same ratio) and from CUSIP changes, and prices are restated in the units after the last split. Counts only – no returns.
""")

phase("F SEC prices", before="""
# ── Phase F · SEC price panel: 13F quarter-end prices, fails-to-deliver files, splits ──
from aem_research import fundamentals as fu, secprices
from aem_research.data import ftd

SEC_PANEL, SEC_EVENTS, SEC_BRIDGE, SEC_AS_OF, FTD_PX, SEC_INFO = None, None, None, None, None, {}
SEC_FILINGS = None
""", body="""
if f13_breadth is None or not len(f13_breadth) or "price_13f" not in f13_breadth.columns:
    print("No 13F prices on Drive – the SEC panel (option A) is skipped; the vendor panel stands in.")
else:
    names_sec = subs[["cik", "name"]] if subs is not None and len(subs) else None
    comps = simfin.companies if simfin.found else None
    if names_sec is not None and comps is not None and "company_name" in comps.columns:
        names_sec = pd.concat([names_sec, comps[["cik", "company_name"]].rename(columns={"company_name": "name"})])
    sym_bridges = [b for b in (ins_bridge, ft.instance_bridge(subs) if subs is not None and len(subs) else None)
                   if b is not None and len(b)]
    SEC_BRIDGE = secprices.bridge(f13_names, FTD_NAMES, comps, names_sec, sym_bridges)
    print(f"CUSIP bridge: {len(SEC_BRIDGE):,} CUSIPs mapped ({SEC_BRIDGE['method'].value_counts().to_dict()})")
    as_of_q = secprices.complete_through(f13_breadth)
    mains = secprices.main_cusips(f13_breadth, SEC_BRIDGE) if as_of_q is not None else pd.DataFrame()
if SEC_BRIDGE is not None and (as_of_q is None or mains.empty):
    print("No company has a 13F price with three filers – the SEC panel (option A) is skipped.")
elif SEC_BRIDGE is not None:
    holders = thirteenf.load_holders(paths.sec_dir, cusips=mains["cusip"].unique())
    holders = holders.merge(mains[["cusip", "period"]].drop_duplicates(), on=["cusip", "period"], how="inner")
    FTD_PX = ftd.load_px(paths.sec_dir, cusips=SEC_BRIDGE.loc[SEC_BRIDGE["cik"].isin(set(mains["cik"])), "cusip"].unique())
    # each filing's share count (cover page, else weighted average): split evidence where holders
    # are too few or a CUSIP changed
    SEC_FILINGS = fu.filing_share_history(facts) if facts is not None and len(facts) else None
    SEC_PANEL, SEC_EVENTS, SEC_INFO = secprices.build_panel(mains, holders, FTD_PX, SEC_BRIDGE, as_of_period=as_of_q,
                                                            ftd_names=FTD_NAMES,
                                                            max_gap_days=cfg.get("outcomes.sec_max_gap_days", 200),
                                                            dei=SEC_FILINGS)
    del holders, mains
    SEC_AS_OF = secprices.as_of_date(as_of_q)
    save_table(SEC_PANEL, paths.cache / "sec_panel")
    save_table(SEC_EVENTS, paths.cache / "sec_splits")
    print(f"SEC panel: {SEC_INFO['companies']:,} companies, {SEC_INFO['series']:,} series, as of {SEC_AS_OF.date()} · "
          f"quarters from 13F {SEC_INFO['quarters_13f']:,}, from fails files {SEC_INFO['quarters_fails']:,}, "
          f"final prices {SEC_INFO['final_prices']:,}")
    print(f"Splits: {SEC_INFO['splits_holders']:,} from holders, {SEC_INFO['splits_cusip']:,} at CUSIP changes, "
          f"{SEC_INFO['splits_filings']:,} from filings, {SEC_INFO['splits_few_holders']:,} from one or two holders · "
          f"new series {SEC_INFO['new_series']:,} · one-quarter price errors {SEC_INFO['spikes_replaced']:,} replaced, "
          f"{SEC_INFO['spikes_dropped']:,} left out · large moves without evidence {SEC_INFO['unexplained_jumps']:,}")
    REPORT["sec_prices"] = {"companies": SEC_INFO["companies"], "series": SEC_INFO["series"],
                            "as_of": str(SEC_AS_OF.date()), "quarters_13f": SEC_INFO["quarters_13f"],
                            "quarters_fails": SEC_INFO["quarters_fails"], "final_prices": SEC_INFO["final_prices"],
                            "bridge": SEC_BRIDGE["method"].value_counts().to_dict()}
    REPORT["splits"] = {"from_holders": SEC_INFO["splits_holders"], "at_cusip_changes": SEC_INFO["splits_cusip"],
                        "from_filings": SEC_INFO["splits_filings"], "from_few_holders": SEC_INFO["splits_few_holders"],
                        "holder_splits_rejected": SEC_INFO["holder_splits_rejected"],
                        "new_series": SEC_INFO["new_series"], "spikes_replaced": SEC_INFO["spikes_replaced"],
                        "spikes_dropped": SEC_INFO["spikes_dropped"], "unexplained_jumps": SEC_INFO["unexplained_jumps"]}
""", after="""
save_report()
""")

phase("F features", before="""
# ── Phase F · Features, universe, pre-lock and power checks (counts only – no returns) ──
from aem_research import delistings as dl, diagnostics as dg, features as ft, fundamentals as fu, secprices
from aem_research.outcomes import build_price_panel, panel_as_of

FEATURES, DELIST, panel, PRELOCK, POWER, COVREP, holes, basis = None, None, None, None, None, None, None, None
FEATURES_V, DELIST_V, FEATURES_B, DELIST_B, LIQ, SEC_CHECK, pf_sec, keys_sec = (None,) * 8
SPLIT_CASES, B_START = None, None
as_of = None
ST = cfg.get("starts")
FORMATION = ft.formation_dates(ST["years"], ST.get("months", [1]))
FORM_ALL = list(FORMATION)
""", body="""
panel = build_price_panel(simfin.prices if simfin.found else None, stq.prices if stq is not None else None,
                          gap_fill=EOD_PRICES)
if panel.empty or facts is None or not len(facts):
    print("Features need prices and the XBRL data sets – skipped until both are on Drive.")
else:
    as_of = panel_as_of(panel)
    FORMATION = [d for d in FORMATION if d <= as_of]
    print(f"Price panel: {panel['ticker'].nunique():,} tickers, as of {as_of.date()}; {len(FORMATION)} formation dates")
    pf = ft.price_features(panel, FORMATION, tolerance_days=ST.get("entry_tolerance_days", 7),
                           max_gap_days=cfg.get("outcomes.max_gap_days", 120))
    bridges = [b for b in (ins_bridge, ft.instance_bridge(subs)) if b is not None and len(b)]
    keys = ft.ticker_ciks(pf, simfin.companies, bridges)
    if EOD_PRICES is not None and len(EOD_PRICES):
        n0 = len(pf)
        pf, keys = eodhd.dedupe(pf, keys)
        print(f"EODHD: {n0 - len(pf):,} ticker-dates dropped where SimFin or Stooq already price the company")
    print(f"Ticker → CIK: {keys['cik'].notna().mean():.1%} of ticker-dates mapped ({keys['cik_source'].value_counts().to_dict()})")
    fund = fu.snapshots(facts, subs, FORMATION, current_months=cfg.get("universe.current_filer_months", 15))
    ip = cfg.get("turnaround_cohort.insider_purchase")
    roles = tuple(ip.get("roles", ["officer", "director"]))
    signals, sale_signals = None, None
    if ins_tx is not None and len(ins_tx):
        classes = insider.classify(ins_tx, sorted({d.year for d in FORMATION}))
        reports = subs[subs["form"].isin(["10-Q", "10-K"])][["cik", "filed"]]
        sig_kw = dict(lookback_months=ip.get("lookback_months", 6), cluster_min=ip.get("cluster_variant_min_insiders", 2))
        signals = insider.signals(insider.purchases(ins_tx, roles, exclude_plan=ip.get("exclude_rule_10b5_1", True)),
                                  classes, FORMATION, report_dates=reports,
                                  post_results_days=cfg.get("features.post_results_window_days", 14), **sig_kw)
        sale_signals = insider.signals(insider.sales(ins_tx, roles), classes, FORMATION, **sig_kw)
    breadth, prices13 = None, None
    if f13_breadth is not None and len(f13_breadth):
        names = subs[["cik", "name"]]
        if simfin.companies is not None and "company_name" in simfin.companies.columns:
            names = pd.concat([names, simfin.companies[["cik", "company_name"]].rename(columns={"company_name": "name"})])
        bridge13 = thirteenf.cusip_bridge(simfin.companies, f13_names, names)
        print(f"13F CUSIP bridge: {len(bridge13):,} CUSIPs mapped {bridge13['method'].value_counts().to_dict()}")
        breadth = thirteenf.breadth_at(f13_breadth, bridge13, FORMATION)
        prices13 = thirteenf.prices_at(f13_breadth, bridge13, FORMATION)
    feats = ft.assemble(pf, keys, fund, signals, breadth, cfg.data, prices13=prices13, sales=sale_signals)
    ft.check_point_in_time(feats)        # raises if any feature postdates its start date: nothing is kept
    print("Look-ahead guard: passed")
    first = cfg.get("features.first_formation_date")
    att_v, _ = ft.choose_attention(feats, first["H4"])
    feats["low_attention"] = feats["low_breadth"] if att_v == "breadth" else feats["low_turnover"]
    FEATURES_V = feats
    basis = ft.price_basis_check(FEATURES_V)
    if len(basis):
        print("Vendor close vs the 13F implied price at quarter-end (ratio near 1 = unadjusted close; "
              "names off by >10% had later splits, which the size filter now undoes):")
        show(basis)
        REPORT["price_basis"] = {f"{r.source}_off_10pct": r.share_off_10pct for r in basis.itertuples()}
    cap = cfg.get("universe.market_cap_usd")
    holes = ft.universe_holes(fund, FEATURES_V, bridges, cap_min=cap["min"], cap_max=cap["max"])
    hole_keys = ft.universe_hole_keys(fund, FEATURES_V, bridges, cap_min=cap["min"], cap_max=cap["max"])
    if len(holes):
        print("Listed filers in the size range that never appear in the vendor price data:")
        show(holes)
        REPORT["holes"] = {"median_unpriced_share": float(holes["unpriced_share"].median()),
                           "first_date": float(holes["unpriced_share"].iloc[0]),
                           "last_date": float(holes["unpriced_share"].iloc[-1])}
    seg = dl.map_ciks(dl.ended_segments(panel, as_of, stale_gap_days=cfg.get("outcomes.stale_gap_days", 30),
                                        max_gap_days=cfg.get("outcomes.max_gap_days", 120)),
                      simfin.companies, pd.concat(bridges, ignore_index=True) if bridges else None)
    DELIST_V = dl.classify(seg, form25, merger, stq.groups if stq is not None else None, exchange_map=EOD_EXCHANGE)
    first = cfg.get("features.first_formation_date")

    if SEC_PANEL is None or SEC_PANEL.empty:
        LOG.warn("F features", "no SEC price panel – the vendor panel stands in for option A until the 13F data are on Drive")
        FEATURES, DELIST, first_a = FEATURES_V, DELIST_V, pd.Timestamp(first["H2"])
        att, att_share = ft.choose_attention(FEATURES, first["H4"])
    else:
        # ---- A (primary, D10): every price from the SEC panel ----
        first_a = pd.Timestamp(cfg.get("price_basis.first_formation_date"))
        FORM_A = [d for d in FORM_ALL if first_a <= d <= SEC_AS_OF]
        sec_gap = cfg.get("outcomes.sec_max_gap_days", 200)
        LIQ = ft.calibrate_breadth(FEATURES_V, cfg.data)
        print(f"Liquidity for A: at least {LIQ['k']} 13F holders ({LIQ['basis']}; "
              f"agreement with the dollar-volume screen {LIQ.get('agreement', 'n/a')})")
        pf_sec = ft.price_features(SEC_PANEL, FORM_A, tolerance_days=ST.get("entry_tolerance_days", 7),
                                   max_gap_days=sec_gap)
        keys_sec = secprices.keys_for(pf_sec, SEC_PANEL)
        breadth_a = thirteenf.breadth_at(f13_breadth, SEC_BRIDGE, FORM_A)
        f25x = dl.form25_with_exchange(form25) if form25 is not None and len(form25) else None
        sym_tab = secprices.symbol_table(FTD_NAMES, SEC_BRIDGE, bridges)
        sym_ex = secprices.symbol_exchange(sym_tab, stq.groups if stq is not None else None, EOD_EXCHANGE)
        listed = keys_sec[["cik", "start_date"]].assign(
            listed=secprices.listed_flags(keys_sec, keys, f25x, sym_ex).to_numpy())
        feats_a = ft.assemble(pf_sec, keys_sec, fund, signals, breadth_a, cfg.data, prices13=None,
                              sales=sale_signals, liquidity="breadth", min_breadth=LIQ["k"], listed=listed)
        ft.check_point_in_time(feats_a)
        feats_a["price_basis"] = secprices.entry_source(feats_a, SEC_PANEL)
        att, att_share = ft.choose_attention(feats_a, first["H4"])
        feats_a["low_attention"] = feats_a["low_breadth"] if att == "breadth" else feats_a["low_turnover"]
        FEATURES = feats_a
        print("Look-ahead guard (A): passed")
        seg_a = secprices.ended_segments(SEC_PANEL, SEC_AS_OF, stale_gap_days=cfg.get("outcomes.stale_gap_days", 30),
                                         max_gap_days=sec_gap)
        DELIST = dl.classify(seg_a, form25, merger, None,
                             exchange_map=secprices.exchange_map_for(SEC_PANEL, f25x, sym_ex))
        # ---- B (robustness): the vendor universe plus the companies the vendors never price, from the
        # first start date on which the vendor panel prices most of universe A (counts only) ----
        vend_ciks = set(keys["cik"].dropna().astype("int64"))
        cover = dg.vendor_coverage(FEATURES, FEATURES_V)
        b_min = float(cfg.get("price_basis.robustness_b_min_coverage", 0.8))
        ok_dates = cover.index[cover >= b_min]
        B_START = ok_dates[0] if len(ok_dates) else None
        if B_START is None:
            LOG.warn("F features", f"the vendor panel never prices {b_min:.0%} of universe A on a start date – B is empty")
            FEATURES_B = FEATURES_V.iloc[0:0].assign(panel_b="vendor")
            fill = FEATURES.iloc[0:0]
        else:
            fill = FEATURES[FEATURES["in_universe"] & (FEATURES["start_date"] >= B_START)
                            & ~FEATURES["cik"].astype("Int64").isin(vend_ciks).fillna(False)]
            FEATURES_B = pd.concat([FEATURES_V[FEATURES_V["in_universe"] & (FEATURES_V["start_date"] >= B_START)]
                                    .assign(panel_b="vendor"), fill.assign(panel_b="SEC fill")], ignore_index=True)
        DELIST_B = pd.concat([DELIST_V.assign(panel_b="vendor"),
                              DELIST[DELIST["ticker"].isin(set(fill["ticker"]))].assign(panel_b="SEC fill")],
                             ignore_index=True)
        REPORT["universe_b"] = {"start": str(B_START.date()) if B_START is not None else "none",
                                "vendor_coverage_first_date": round(float(cover.iloc[0]), 4) if len(cover) else None,
                                "windows": int(len(FEATURES_B)),
                                "vendor_windows": int((FEATURES_B["panel_b"] == "vendor").sum()),
                                "sec_fill_windows": int(len(fill)), "sec_fill_firms": int(fill["cik"].nunique())}
        print(f"B starts {REPORT['universe_b']['start']}: the first start date on which the vendor panel prices at least "
              f"{b_min:.0%} of universe A's windows (it prices {REPORT['universe_b']['vendor_coverage_first_date']:.0%} "
              f"at A's first date)")
        # ---- checks on the SEC panel, before the lock and without outcomes ----
        SEC_CHECK, SPLIT_CASES = secprices.validate(SEC_PANEL, SEC_EVENTS, panel, keys, filings=SEC_FILINGS,
                                                    return_cases=True)
        REPORT["split_cases"] = secprices.case_lines(SPLIT_CASES, per_category=4)
        if len(SPLIT_CASES):
            print("Split check – cases to look at (counts and prices only):")
            show(SPLIT_CASES["category"].value_counts().rename("cases"))
        hk = hole_keys[hole_keys["start_date"] >= first_a] if len(hole_keys) else hole_keys
        if len(hk):
            hk = hk.assign(listed=secprices.listed_flags(hk, None, f25x, sym_ex).to_numpy())
            SEC_CHECK["holes_listed_share"] = round(float(hk["listed"].mean()), 4)
            SEC_CHECK["holes_priced"] = secprices.holes_priced(hk[hk["listed"]], pf_sec, keys_sec)
        SEC_CHECK["liquidity_k"] = LIQ["k"]
        SEC_CHECK["liquidity_agreement"] = LIQ.get("agreement")
        SEC_CHECK["listed_share"] = round(float(listed["listed"].mean()), 4) if len(listed) else None
        REPORT["sec_check"] = SEC_CHECK
        REPORT["liquidity"] = LIQ
        show(pd.Series(SEC_CHECK, name="SEC panel checks"))
        # why A is the size it is: each filter alone, so far and without it; B's windows looked up in A
        FUNNEL_A = dg.universe_funnel(FEATURES, cfg.data, start=first_a)
        show(FUNNEL_A)
        REPORT["funnel_a"] = dg.funnel_summary(FUNNEL_A)
        REPORT["a_vs_v"] = dg.compare_universes(FEATURES, FEATURES_V, cfg.data, start=first_a, min_breadth=LIQ["k"])
        show(pd.Series(REPORT["a_vs_v"], name="vendor universe windows, looked up in A"))
    print(f"Attention signal for H4: {att} (breadth known for {att_share:.0%} of H4 windows; rule: 80%)")
    COVREP = ft.coverage_report(FEATURES)
    show(COVREP)
    REPORT["delistings"] = dl.summary(DELIST)
    show(pd.Series(REPORT["delistings"], name="ended series"))
    POWER = ft.power_table(FEATURES)
    print("Smallest likelihood ratio each cohort could detect (80% power, one-sided 5%, distinct firms; no outcomes used):")
    show(POWER)
    REPORT["power"] = POWER.to_dict("records")
    save_table(FEATURES, paths.cache / "features")
    save_table(DELIST, paths.cache / "delistings")
    save_table(FEATURES_V, paths.cache / "features_vendor")
    save_table(DELIST_V, paths.cache / "delistings_vendor")
    if FEATURES_B is not None:
        save_table(FEATURES_B, paths.cache / "features_b")
        save_table(DELIST_B, paths.cache / "delistings_b")

    # Pre-lock checks (config: review_before_lock) – counts only, on the primary universe
    uni = FEATURES[FEATURES["in_universe"]]

    def known(col, start):
        w = uni[uni["start_date"] >= max(pd.Timestamp(start), first_a)]
        return float(w[col].notna().mean()) if len(w) else 0.0

    checks = {"fcf_known": known("fcf_pos", first["H2"]), "net_cash_known": known("net_cash_pos", first["H2"]),
              "s1_known": known("s1_proxy", first["H4"]), "dilution_known": known("low_dilution", first["H4"]),
              "attention_known": known("low_attention", first["H4"])}
    PRELOCK = {k: f"{v:.1%} {'ok' if v >= 0.8 else 'BELOW 80%'}" for k, v in checks.items()}
    unk = REPORT["delistings"].get("share_unknown", 0.0)
    PRELOCK["delisting_unknown"] = f"{unk:.1%} {'ok' if unk < 0.25 else 'ABOVE 25%'}"
    if SEC_CHECK:
        for name, key, bar in (("sec_returns_agree", "returns_agree", 0.95), ("split_precision", "split_precision", 0.9),
                               ("split_recall", "split_recall", 0.9), ("holes_priced", "holes_priced", 0.8)):
            v = SEC_CHECK.get(key)
            PRELOCK[name] = "n/a" if v is None else f"{v:.1%} {'ok' if v >= bar else f'BELOW {bar:.0%}'}"
    PRELOCK["universe_windows"] = int(len(uni))
    PRELOCK["h2a_cohort_windows"] = int(uni["h2a"].fillna(False).sum())
    PRELOCK["h2b_cohort_windows"] = int(uni["h2b"].fillna(False).sum())
    PRELOCK["placebo_cohort_windows"] = int(uni["h2_placebo"].fillna(False).sum())
    REPORT["prelock"] = PRELOCK
    REPORT["s1_gaps"] = ft.s1_gaps(FEATURES, max(pd.Timestamp(first["H4"]), first_a))
    REPORT["coverage_summary"] = {"universe_per_date_median": int(COVREP["universe"].median()),
                                  "universe_first_date": int(COVREP["universe"].iloc[0]),
                                  "universe_last_date": int(COVREP["universe"].iloc[-1])}
    REPORT["features"] = {"formation_dates": int(FEATURES["start_date"].nunique()),
                          "price_basis": "SEC panel (A)" if FEATURES is not FEATURES_V else "vendor panel (A unavailable)",
                          "as_of": str((SEC_AS_OF if FEATURES is not FEATURES_V else as_of).date()),
                          "vendor_as_of": str(as_of.date()), "attention_signal": att,
                          "universe_windows": int(len(uni)), "firms_in_universe": int(uni["cik"].nunique()),
                          "price_from_13f": round(float((uni["price_basis"] == "13F").mean()), 4)}
    show(pd.Series(PRELOCK, name="pre-lock checks"))
""", after="""
save_report()
""")

md("""
## Phase F – Quarter-end vs daily touch rates (after the lock)

The primary analysis counts a touch only at a quarter-end close (D10). How much that lowers touch rates is measured after the lock, on B's vendor windows: quarter-end against daily closes of the same series, so only the sampling differs. The pre-sample design (2009–2012 start dates) was dropped: the vendor panel prices almost none of those companies (the Nov 9 run found 51 windows).
""")

phase("F calibration", before="""
# ── Phase F · Quarter-end over daily touch rates: measured after the lock ──
from aem_research.outcomes import OutcomeSpec, touch_sampling_ratios

CALIB = None
""", body="""
REPORT["calibration"] = {"basis": "after the lock – B's vendor windows, quarter-end against daily closes of the same series"}
print(REPORT["calibration"]["basis"])
""", after="""
save_report()
""")

md("""
## Phase F – Macro regime (FRED)

The 2022 split is the primary regime definition; these moderators test the mechanisms behind it, each measured on the start date with only what was published by then: the 10-year real yield (DFII10), the 12-month change in the 2-year yield (DGS2), the Baa credit spread (BAA10Y), CPI inflation (CPIAUCNS, the release available on the start date), the universe's median drawdown and its median free-cash-flow yield. FRED needs no key. Macro variables vary only across start dates, so their inference resamples calendar years, not firms. The same table feeds Paper A.
""")

phase("F regime", before="""
# ── Phase F · Macro regime moderators ──
from aem_research.data import fred

REGIME = None
""", body="""
series, fred_stats = fred.load(paths.cache / "fred", offline=OFFLINE, manual_dir=paths.work / "data" / "fred")
REPORT["fred"] = fred_stats
for gap in fred.gaps(series, fred_stats):          # one missing series must not cost the other moderators
    LOG.warn("F regime", gap)
REGIME = fred.regime_at(series, FORMATION)
if FEATURES is not None:
    REGIME = REGIME.merge(ft.market_regime(FEATURES), on="start_date", how="left")
show(REGIME.round(4))
save_table(REGIME, paths.cache / "regime")
REPORT["regime_summary"] = {c: f"{REGIME[c].min():.3g} to {REGIME[c].max():.3g}"
                            for c in REGIME.columns if c != "start_date" and REGIME[c].notna().any()}
""")

phase("Excel export", before="""
# ── Save every table to Drive as one Excel workbook (raw data stays in parquet) ──
from aem_research.export import write_workbook

FEATURE_COLS = ["ticker", "cik", "start_date", "entry_date", "entry_price_unadjusted", "price_basis", "market_cap",
                "shares_source", "shares_kind", "dollar_volume_20d", "drawdown", "fallen", "fcf_pos", "net_cash_pos", "s1_proxy",
                "low_dilution", "insider_any", "insider_cluster", "insider_opportunistic", "insider_routine_only",
                "insider_unclassified_only", "insider_post_results_cluster", "sale_any", "breadth", "low_breadth",
                "low_turnover", "low_attention", "h2a", "h2b", "h2_placebo", "sic", "source", "liquid", "listed"]
""", body="""
summary = {"run (UTC)": REPORT["generated_utc"], "repo commit": COMMIT, "config sha256": cfg.sha256,
           "lock status": cfg.status, "EODHD verdict": verdict.get("verdict"), "verdict reason": verdict.get("reason"),
           **{f"features: {k}": v for k, v in (REPORT.get("features") or {}).items()},
           **{f"pre-lock: {k}": v for k, v in (PRELOCK or {}).items()},
           **{f"delistings: {k}": v for k, v in (REPORT.get("delistings") or {}).items()}}
uni_rows = None
if FEATURES is not None:
    uni_rows = FEATURES.loc[FEATURES["in_universe"], [c for c in FEATURE_COLS if c in FEATURES.columns]]
prelock_sheet = {**(PRELOCK or {}), **{f"S1 gap: {k}": v for k, v in (REPORT.get("s1_gaps") or {}).items()},
                 **{f"SEC check: {k}": v for k, v in (REPORT.get("sec_check") or {}).items()},
                 **{f"EODHD: {k}": v for k, v in (REPORT.get("eodhd") or {}).items()}} or None
uni_b = None
if FEATURES_B is not None:
    uni_b = FEATURES_B[[c for c in FEATURE_COLS + ["panel_b"] if c in FEATURES_B.columns]]
sec_sheet = {**{f"panel: {k}": v for k, v in (REPORT.get("sec_prices") or {}).items()},
             **{f"splits: {k}": v for k, v in (REPORT.get("splits") or {}).items()},
             **{f"universe B: {k}": v for k, v in (REPORT.get("universe_b") or {}).items()}} or None
sheets = {"Summary": summary, "Coverage by date": COVREP, "Pre-lock checks": prelock_sheet, "Power": POWER,
          "Regime": REGIME, "SEC price panel": sec_sheet, "Splits": SEC_EVENTS, "Split check": SPLIT_CASES,
          "Calibration": CALIB,
          "Universe holes": holes, "Price basis": basis, "Delistings": DELIST,
          "SEC data sets": pd.DataFrame(sec_set_stats).T.reset_index() if sec_set_stats else None,
          "Universe windows": uni_rows, "Universe B windows": uni_b}
written = write_workbook(paths.output / "aem_results.xlsx", sheets)
print(f"Saved aem_results.xlsx to {paths.output}: {written}")
""", after="""
save_report()
""")

md("""
## Phase E – Outcome engine (locked)

The engine computes touch and end-of-window multiples with delisting returns and bounds, on the universe Phase F built and with the delisting classes Phase F assigned. It runs on real data only once the pre-registration is locked; until then this cell says what is pending, and the engine itself is covered by the test suite.
""")

code("""
# ── Phase E · Outcomes (locked) ──
from aem_research import outcomes, universe

result = None
if not cfg.locked:
    print(f"Locked phase – pre-registration status: {cfg.status}. Once the pre-lock checks pass, "
          "I lock it and commit LOCK.json.")
    LOG.skipped("E outcomes", "not locked")
elif not universe.READY or FEATURES is None:
    print("Pre-registration locked. Waiting for the Phase F features before computing outcomes.")
    LOG.skipped("E outcomes", "no features")
else:
    with LOG.phase("E outcomes"):
        sec_kw = dict(overrides={"max_gap_days": cfg.get("outcomes.sec_max_gap_days", 200)})
        starts = outcomes.starts_from_features(FEATURES)
        if SEC_PANEL is not None and len(SEC_PANEL) and FEATURES is not FEATURES_V:
            result = outcomes.compute_outcomes(cfg, SEC_PANEL, starts, DELIST, as_of=SEC_AS_OF, **sec_kw)
        else:
            result = outcomes.compute_outcomes(cfg, panel, starts, DELIST)
        br = outcomes.base_rates(result, multiples=(3.0, 10.0))
        print("A – SEC quarter-end prices (primary):")
        show(br)
        REPORT["base_rates"] = br.round(4).to_dict("records")
        print(f"{len(starts):,} start windows · saved {save_table(result, paths.output / 'outcomes').name}")
        keep = [c for c in ("ticker", "start_date", "horizon_years", "status", "entry_price", "end_multiple",
                            "end_multiple_lower", "end_multiple_upper", "max_multiple", "bucket", "touch_3x",
                            "touch_10x", "delist_category") if c in result.columns]
        sheets_e = {"Base rates A": br, "Outcomes A": result[keep]}
        if FEATURES_B is not None and len(FEATURES_B):
            sb = outcomes.starts_from_features(FEATURES_B.assign(in_universe=True))
            fill_t = set(FEATURES_B.loc[FEATURES_B["panel_b"] == "SEC fill", "ticker"])
            res_v = outcomes.compute_outcomes(cfg, panel, sb[~sb["ticker"].isin(fill_t)], DELIST_V)
            dense = secprices.dense_panel(SEC_PANEL, SEC_EVENTS, FTD_PX, fill_t)
            res_f = outcomes.compute_outcomes(cfg, dense, sb[sb["ticker"].isin(fill_t)], DELIST_B, as_of=as_of, **sec_kw)
            result_b = pd.concat([res_v, res_f], ignore_index=True)
            br_b = outcomes.base_rates(result_b, multiples=(3.0, 10.0))
            print("B – daily vendor closes with the SEC fill (robustness):")
            show(br_b)
            REPORT["base_rates_b"] = br_b.round(4).to_dict("records")
            save_table(result_b, paths.output / "outcomes_b")
            sheets_e.update({"Base rates B": br_b, "Outcomes B": result_b[[c for c in keep if c in result_b.columns]]})
            # quarter-end against daily touch rates on B's vendor windows – only the sampling differs
            CALIB = touch_sampling_ratios(OutcomeSpec.from_config(cfg.data), panel,
                                          sb.loc[~sb["ticker"].isin(fill_t), ["ticker", "start_date"]],
                                          sample_start=pd.Timestamp.max, horizons=(3, 5),
                                          quarterly_max_gap_days=cfg.get("outcomes.sec_max_gap_days", 200))
            show(CALIB)
            c3 = CALIB[CALIB["horizon_years"] == 3] if len(CALIB) else CALIB
            REPORT["calibration"] = {"basis": "B's vendor windows (after the lock)",
                                     "windows_3y": int(c3["windows"].max()) if len(c3) else 0,
                                     **{f"3y_{m:g}x": v for m, v in zip(c3["multiple"], c3["quarter_end_over_daily"])}}
            sheets_e["Calibration"] = CALIB
        write_workbook(paths.output / "aem_outcomes.xlsx", sheets_e)
""")

code("""
# ── Run summary: paste everything between the rulers into the chat ──
REPORT["run"] = LOG.as_dict()
REPORT["drive_gb"] = folder_size_gb(paths.work)
save_report()
block = paste_block(REPORT)
(paths.output / "run_summary.txt").write_text(block, encoding="utf-8")
LOG.finish()
print(block)
print(f"Saved to {paths.output}: run_summary.txt, run_progress.txt, coverage_report.md, coverage_report.json, aem_results.xlsx")
""")

md("""
## Next steps

1. **Paste the run summary** (the block above) into the chat.
2. **SEC price panel (D10):** quarterly returns agree with the vendors' split-adjusted prices for at least 95% of company-quarters, split detection reaches 90% precision and recall, and the panel prices at least 80% of the listed holes. The vendor panel's EODHD verdict is kept for reference only.
3. **Pre-lock checks:** every feature known for at least 80% of universe windows, fewer than 25% "unknown" delistings, and a look at the power table. If one fails, the definition is fixed in the config before the lock.
4. **Lock:** I set `study.status` to final, lock, and commit `LOCK.json` – the public timestamp. D7–D10 (regime, its confirmation, unplanned findings, the SEC price basis) are already settled.
5. **Next build:** the estimators – likelihood ratios with firm-cluster bootstrap and Holm correction, the ordered logit and the 10x logit, the checklist benchmark, the placebo.
""")


def to_source(text):
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


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
