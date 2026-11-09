"""Generate notebooks/02_paper_a_take_private_sample.ipynb (nbformat 4.5) from plain cell sources.

The notebook is generated – edit the cells here, then run:
    python tools/build_notebook_paper_a.py notebooks/02_paper_a_take_private_sample.ipynb

It runs after notebook 01 and reuses its Drive caches. Every phase after setup runs
inside `with LOG.phase(...)`: an error is recorded, the run goes on, and the block
printed at the end carries the details. Only a failing test suite stops it.
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
    ind = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.strip("\n").split("\n"))
    parts = [before.strip("\n")] if before.strip() else []
    parts.append(f'with LOG.phase("{name}"):\n{ind}')
    if after.strip():
        parts.append(after.strip("\n"))
    code("\n".join(parts))


md("""
# Paper A – 02 take-private sample from SEC filings (counts only)

**What this does.** Builds the sample for Paper A – regime-conditional value creation in US take-private buyouts – from free, public SEC filings. It finds sponsor-backed take-privates through EDGAR full-text search, reads each deal's merger document for the offer price, the equity and debt commitments and the sponsor, attaches the target's last XBRL financials (EBITDA, net debt) and the public small-cap benchmark multiples, detects which exits became public again, and reads WSIB's quarterly private equity fund table. Everything is saved to Drive, with an Excel workbook.

**Run notebook 01 first.** This notebook reuses its caches on Drive – the SEC financial data sets, the EDGAR index, the universe features, the delistings and the FRED series – and downloads only what is new: the search results, about one document per deal, the Form 15 and IPO filings from the EDGAR index, and WSIB's quarterly PDFs.

**Before you run** (Runtime → Run all): the same Colab secret `SEC_USER_AGENT` as for notebook 01. Nothing else.

**Run time.** Roughly 30–60 minutes the first time, mostly one merger document per deal at the SEC's pace; later runs take minutes, because every search, document and report is cached on Drive. After a disconnect, Run all resumes.

**Blinding (decision A2).** Until `config/paper_a.yaml` is locked, this notebook shows counts and entry-side data only: no exit values, no exit dates (they would give holding periods), no value-creation components and no fund returns. Those inputs are cached for after the lock.

**When it ends,** the last cell prints one block – counts, timings, warnings and errors. Paste it into the chat (it is also saved as `paper_a_summary.txt` in `My Drive/aem-research/output/`). If Colab crashes or disconnects first, `paper_a_progress.txt` in the same folder shows how far it got.

| Phase | What it does |
|---|---|
| 0 Setup | Mount Drive, fetch the repo, the two configs, the offline test suite |
| A1 Inputs | Notebook 01's caches: XBRL financials, EDGAR index (plus Form 15 and IPO filings), features, delistings, FRED |
| A2 Search | EDGAR full-text search for "equity commitment letter" in target-side merger filings |
| A3 Deals | Hits grouped into deals by target; completion from Form 25, Form 15 or the delisting |
| A4 Terms | Offer price, equity and debt commitments, sponsor – from each deal's definitive document |
| A5 Entry | XBRL financials before the first filing → EBITDA, net debt, enterprise value, multiples; the regime at entry |
| A6 Exits | Still files with the SEC, re-IPO, sold to a listed buyer – which route, not when or at what value |
| A7 Benchmark | Median EV/EBITDA and EV/revenue of the public small-cap universe at each start date |
| A8 WSIB | The quarterly fund table: funds by strategy and vintage (returns hidden until the lock) |
| A9 Save | Excel workbook, cached tables, the paste block |
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
from aem_research.io_utils import load_table, save_table, show

COMMIT = _git("-C", REPO_DIR, "rev-parse", "--short", "HEAD").stdout.strip() or "n/a"
# a hash of this notebook's cells, set by its builder: an older copy (Colab keeps an opened
# notebook as it was) would run the newest code with old steps, so it stops here instead
NOTEBOOK_BUILD = "__BUILD__"
from aem_research.env import check_notebook_fresh
check_notebook_fresh(REPO_DIR, "02_paper_a_take_private_sample.ipynb", NOTEBOOK_BUILD,
                     colab_url="https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/02_paper_a_take_private_sample.ipynb")
print(f"aem_research {aem_research.__version__} @ {COMMIT} · pandas {pd.__version__} · Colab: {IN_COLAB}")
""")

code("""
# ── Phase 0 · Configs: Paper A (draft, its own lock) and Paper B (shared paths and regime) ──
from aem_research.config import load_config
from aem_research.runlog import RunLog, folder_size_gb, versions

LOG = RunLog()
cfg_a = load_config(os.path.join(REPO_DIR, "config", "paper_a.yaml"),
                    os.path.join(REPO_DIR, "config", "paper_a_LOCK.json"))
cfg_b = load_config(os.path.join(REPO_DIR, "config", "preregistration.yaml"))
print(cfg_a.summary())
REPORT = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "repo_commit": COMMIT,
          "notebook_build": NOTEBOOK_BUILD, "config_sha256": cfg_a.sha256, "lock_status": cfg_a.status,
          "versions": versions()}
""")

code("""
# ── Phase 0 · Offline test suite: must pass before anything touches data ──
from aem_research.report import paper_a_block


def _stop_block():
    REPORT["run"] = LOG.as_dict()
    print()
    print(paper_a_block(REPORT))


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

paths = resolve_paths(DRIVE_ROOT, cfg_b.get("data_sources.simfin_cache_dirs")).ensure()
A_DIR = paths.cache / "paper_a"
A_DIR.mkdir(parents=True, exist_ok=True)
print(paths.describe())
print("Paper A cache:", A_DIR)
LOG.attach_progress(paths.output / "paper_a_progress.txt", header=f"· repo {COMMIT} · config A {cfg_a.sha256[:12]}")
SAMPLE = cfg_a.get("sample")
Y0, Y1 = SAMPLE["announce_years"]
TODAY = pd.Timestamp(datetime.now(timezone.utc).date())
THIS_YEAR = TODAY.year
OFFLINE = os.environ.get("AEM_SEC_OFFLINE") == "1"
""")

md("""
## A1 – Inputs from notebook 01

The XBRL financial data sets and the EDGAR index come from notebook 01's cache; quarters published since are downloaded. The index adds two subsets for Paper A: Form 15 deregistrations (completion) and S-1 or 424B prospectuses (re-IPOs) – each quarter's index is read once more for them.
""")

phase("A1 inputs", before="""
# ── A1 · Inputs ──
from aem_research import takeprivate as tp
from aem_research.data import fred, fsds, fts, sec

ua, ua_origin = sec.resolve_user_agent()
client = None if OFFLINE else sec.SecClient(ua, min_interval=cfg_b.get("data_sources.sec_min_interval_seconds", 0.2))
subs, facts, series = None, None, {}
form25 = merger = dereg = ipo = None
FEATURES = load_table(paths.cache / "features")
DELIST = load_table(paths.cache / "delistings")
""", body="""
print(f"SEC user agent: {ua_origin}")
if FEATURES is None:
    print("Notebook 01's features are not on Drive yet – run notebook 01 first. The benchmark is skipped.")
fs0 = cfg_b.get("coverage.benchmark_years")[0] - 2
print("XBRL financial data sets (notebook 01's cache; new quarters are downloaded)")
subs, facts, fs_st = fsds.load(paths.sec_dir, fs0, THIS_YEAR, client=client, offline=OFFLINE,
                               local_dir=paths.sec_dir / "manual")
print("EDGAR index: Form 25, merger filings, Form 15 deregistrations, IPO prospectuses")
idx, idx_st = sec.load_index_subsets(paths.sec_dir, Y0, THIS_YEAR, {**sec.INDEX_SUBSETS, **tp.INDEX_SUBSETS},
                                     client=client, offline=OFFLINE)
form25, merger, dereg, ipo = idx["form25"], idx["merger"], idx["dereg"], idx["ipo"]
series, fred_stats = fred.load(paths.cache / "fred", offline=OFFLINE, manual_dir=paths.work / "data" / "fred")
for gap in fred.gaps(series, fred_stats):
    LOG.warn("A1 inputs", gap)
REPORT["inputs"] = {
    "xbrl": f"{len(subs):,} filings, {len(facts):,} facts ({fs_st['cached']} quarters cached, "
            f"{fs_st['downloaded'] + fs_st['local']} new, {fs_st['missing']} missing)",
    "universe_windows": f"{int(FEATURES['in_universe'].sum()):,}" if FEATURES is not None else "none – run notebook 01",
    "delistings": f"{len(DELIST):,}" if DELIST is not None else "none – run notebook 01",
    "fred": ", ".join(f"{k} {v.split(',')[0]}" for k, v in fred_stats.items() if "_" not in k)}
REPORT["index"] = {"quarters": f"{idx_st['cached']} cached, {idx_st['downloaded']} downloaded, {idx_st['missing']} missing",
                   "form25": f"{len(form25):,}", "merger_filings": f"{len(merger):,}", "form15": f"{len(dereg):,}",
                   "ipo_prospectuses": f"{len(ipo):,}"}
print(REPORT["inputs"], REPORT["index"], sep="\\n")
""")

md("""
## A2 – Full-text search: sponsor-backed take-privates

EDGAR full-text search covers every electronic filing since 2001. The query is the sponsor's "equity commitment letter" in target-side merger filings (definitive and preliminary proxies, information statements, Schedule 14D-9 and 13E-3); strategic buyers rarely need an equity commitment. One cached window per calendar year; past years are never searched again.
""")

phase("A2 search", before="""
# ── A2 · Search ──
hits = None
""", body="""
s = SAMPLE["search"]
hits, search_st = fts.search(s["query"], s["forms"], f"{Y0}-01-01", TODAY, client=client,
                             cache_dir=A_DIR / "fts", offline=OFFLINE)
REPORT["search"] = {"query": "equity commitment letter", "documents": f"{search_st['hits']:,}",
                    "filings": f"{search_st['filings']:,}",
                    "windows": f"{search_st['windows']} ({search_st['cached']} cached, {search_st['fetched']} fetched, "
                               f"{search_st['missing']} missing)", "windows_split": search_st["split"]}
print(REPORT["search"])
""")

md("""
## A3 – Deals and completion

Hits are grouped by target – the CIK among the filers that files 10-Ks and 10-Qs – and a new deal starts after an 18-month gap. A deal is completed when the target's Form 25 or Form 15, or a merger delisting in the price data, falls within 18 months of its first filing.
""")

phase("A3 deals", before="""
# ── A3 · Deals ──
DEALS = None
""", body="""
DEALS = tp.group_deals(hits, subs, gap_months=SAMPLE["deal_gap_months"], current_months=SAMPLE["current_filer_months"])
DEALS = DEALS[DEALS["announce_year"].between(Y0, Y1)].reset_index(drop=True)
DEALS = tp.completion(DEALS, form25, dereg, DELIST, window_months=SAMPLE["completion_window_months"])
s = tp.summary(DEALS)
REPORT["deals"] = {k: s[k] for k in ("targets", "deals", "announced", "completed", "not_completed_or_unseen",
                                     "financial_firms") if k in s}
REPORT["deals"]["xbrl_filer_share"] = s.get("fsds_filer")
REPORT["by_year"] = {str(k): int(v) for k, v in DEALS.groupby("announce_year").size().items()}
print(REPORT["deals"])
print("Deals by announcement year:", REPORT["by_year"])
""")

md("""
## A4 – Terms from each deal's definitive document

The definitive proxy (or 14D-9, 13E-3, information statement) is read once and kept on Drive as compressed text. Offer price, equity commitment, the largest debt amount committed and the sponsor are heuristics, so each keeps the sentence it came from: the Review sheet is for hand-checking.
""")

phase("A4 terms", body="""
terms, doc_st = tp.read_documents(DEALS, client=client, cache_dir=A_DIR, offline=OFFLINE)
DEALS = DEALS.merge(terms, on=["deal_id", "doc_adsh"], how="left")
REPORT["documents"] = doc_st
print(doc_st)
""")

md("""
## A5 – Entry: financials, enterprise value, multiples, regime

The target's latest 10-K or 10-Q filed before the first merger filing gives trailing revenue, EBITDA (operating income plus depreciation and amortization), cash, debt and the share count (the cover page where the SEC data sets carry it, else the weighted-average shares); the offer price turns them into enterprise value and entry multiples. The macro moderators of Paper B's D7 are measured on the same date.
""")

phase("A5 entry", body="""
fin = tp.entry_financials(DEALS, facts, subs, current_months=SAMPLE["current_filer_months"])
DEALS = tp.entry_values(DEALS.merge(fin, on="deal_id", how="left"))
reg = fred.regime_at(series, DEALS["first_filed"]).drop(columns="start_date").add_prefix("entry_")
DEALS = pd.concat([DEALS.reset_index(drop=True), reg.reset_index(drop=True)], axis=1)
s = tp.summary(DEALS)
REPORT["entry"] = {k: s[k] for k in ("document_read", "main_document", "offer_price_known", "ebitda_known",
                                     "ev_ebitda_known", "equity_commitment_known", "debt_commitment_known",
                                     "sponsor_named") if k in s}
print(REPORT["entry"])
""")

md("""
## A6 – Exits: which became public again (detection only)

Three routes leave a public trace: the company still files 10-Ks (registered debt – post-buyout financials), it re-lists (an S-1 or 424B prospectus under a matching name), or a listed buyer acquires it (an 8-K "completion of acquisition" naming it; one full-text search per deal). Only the route is shown; exit dates and values wait for Paper A's lock.
""")

phase("A6 exits", body="""
queries = tp.exit_queries(DEALS, days_after=cfg_a.get("exits.exit_days", 180), today=TODAY)
sale_hits, exit_st = fts.search_many(queries, client=client, cache_path=A_DIR / "fts_exits", offline=OFFLINE)
DEALS = tp.exit_routes(DEALS, subs, ipo, sale_hits, still_days=cfg_a.get("exits.still_files_days", 90),
                       exit_days=cfg_a.get("exits.exit_days", 180))
s = tp.summary(DEALS)
REPORT["exits"] = {k.replace("exit_", ""): v for k, v in s.items() if k.startswith("exit_")}
REPORT["exits"]["post_buyout_financials"] = s.get("post_buyout_financials", 0)
REPORT["exits"]["searches"] = f"{exit_st['queries']} ({exit_st['cached']} cached, {exit_st['fetched']} fetched, {exit_st['missing']} missing)"
print(REPORT["exits"])
""")

md("""
## A7 – Public benchmark multiples

The median EV/EBITDA and EV/revenue of Paper B's universe (US$100m–2bn) at each quarterly start date – the public yardstick that splits a deal's multiple change into the market's part and the deal's own.
""")

phase("A7 benchmark", before="""
# ── A7 · Benchmark ──
BENCH = None
""", body="""
if FEATURES is None:
    print("Skipped: notebook 01's features are not on Drive yet.")
else:
    BENCH = tp.benchmark(FEATURES, facts, subs, current_months=SAMPLE["current_filer_months"])
    if len(BENCH):
        e = BENCH["median_ev_ebitda"].dropna()
        REPORT["bench"] = {"start_dates": int(len(BENCH)),
                           "median_ev_ebitda": f"{e.min():.1f} to {e.max():.1f}" if len(e) else "n/a",
                           "firms_per_date": f"{int(BENCH['firms'].median()):,}"}
        show(BENCH.tail(8))
""")

md("""
## A8 – WSIB quarterly fund table

Washington State Investment Board publishes every private equity partnership it holds each quarter: commitment, paid-in capital, distributions, value, net IRR and multiple, with the first investment date as the vintage. Raw PDFs are kept on Drive. Before Paper A's lock only the counts are shown – funds by strategy and vintage.
""")

phase("A8 WSIB", before="""
# ── A8 · WSIB ──
from aem_research.data import wsib

WSIB, WSIB_COUNTS = None, None
""", body="""
try:
    import pdfplumber  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"], check=True)
WSIB, w = wsib.load(A_DIR / "wsib", start_year=cfg_a.get("wsib.first_year", 2016), offline=OFFLINE,
                    manual_dir=paths.work / "data" / "wsib")
WSIB_COUNTS = wsib.counts(WSIB)
latest = WSIB["report_date"].max() if len(WSIB) else None
buyout = WSIB_COUNTS[WSIB_COUNTS["strategy"].astype(str).str.contains("Buyout", na=False)]["funds"].sum() if len(WSIB_COUNTS) else 0
REPORT["wsib"] = {"reports": f"{w['parsed'] + w['cached']} of {w['reports']} read ({w['missing']} not published or missing, "
                             f"{w['failed']} failed)",
                  "fund_rows": f"{w['rows']:,}", "latest": str(latest.date()) if latest is not None else "none",
                  "buyout_funds_latest": int(buyout),
                  "subtotal_checks": f"{w['subtotals_matched']} of {w['subtotals']} match"}
if w["failed_reports"]:
    LOG.warn("A8 WSIB", "reports not read: " + ", ".join(w["failed_reports"][:6]))
print(REPORT["wsib"])
""")

md("""
## A9 – Save: Excel workbook, cached tables, the paste block
""")

phase("A9 save", before="""
# ── A9 · Save ──
from aem_research.export import write_workbook

DEAL_COLS = ["deal_id", "target_cik", "target_name", "sic", "announce_year", "first_filed", "last_filed", "forms",
             "doc_form", "doc_url", "completed", "completion_date", "completion_evidence", "offer_price", "shares_filing",
             "shares_filing_kind", "shares_dei", "equity_value", "cash_total", "debt", "net_cash", "enterprise_value", "revenue", "operating_income", "dna",
             "ebitda", "ebitda_source", "ev_ebitda", "ev_revenue", "equity_commitment_usd", "debt_commitment_usd",
             "leverage_x", "equity_share", "sponsor", "fin_filed", "period_end", "entry_real_yield_10y",
             "entry_tightening_2y_12m", "entry_credit_spread_baa", "entry_inflation_cpi_yoy", "exit_route",
             "post_buyout_financials"]
REVIEW_COLS = ["deal_id", "target_name", "doc_url", "offer_price", "price_snippet", "equity_commitment_usd",
               "equity_snippet", "debt_commitment_usd", "debt_snippet", "sponsor", "sponsor_mentions", "doc_error"]
EXIT_COLS = ["deal_id", "target_name", "completed", "completion_evidence", "exit_route", "post_buyout_financials",
             "reipo_company", "reipo_form", "buyer_name", "sale_adsh"]
""", body="""
pick = lambda df, cols: None if df is None else df[[c for c in cols if c in df.columns]]  # noqa: E731
if DEALS is not None:
    save_table(DEALS, A_DIR / "deals")                       # everything, exit dates included, for after the lock
if WSIB is not None and len(WSIB):
    save_table(WSIB, A_DIR / "wsib_funds")
summary = {"run (UTC)": REPORT["generated_utc"], "repo commit": COMMIT, "Paper A config sha256": cfg_a.sha256,
           "lock status": cfg_a.status, "blinding": "counts and entry-side data only until Paper A's lock (A2)",
           **{f"{sec_}: {k}": v for sec_ in ("inputs", "search", "deals", "entry", "exits", "bench", "wsib")
              for k, v in (REPORT.get(sec_) or {}).items()}}
wsib_cols = [c for c in (WSIB.columns if WSIB is not None else []) if c not in wsib.OUTCOME_COLUMNS]
sheets = {"Summary": summary, "Deals": pick(DEALS, DEAL_COLS), "Review": pick(DEALS, REVIEW_COLS),
          "Exit routes": pick(DEALS, EXIT_COLS), "Benchmark": BENCH,
          "Search hits": pick(hits, ["adsh", "form", "file_type", "file_date", "names", "ciks", "file_name"]),
          "WSIB funds": pick(WSIB, wsib_cols), "WSIB counts": WSIB_COUNTS}
written = write_workbook(paths.output / "paper_a_sample.xlsx", sheets)
print(f"Saved paper_a_sample.xlsx to {paths.output}: {written}")
""")

code("""
# ── Run summary: paste the block below into the chat ──
REPORT["run"] = LOG.as_dict()
REPORT["drive_gb"] = folder_size_gb(paths.work)
block = paper_a_block(REPORT)
(paths.output / "paper_a_summary.txt").write_text(block, encoding="utf-8")
LOG.finish()
print(block)
print(f"Saved to {paths.output}: paper_a_summary.txt, paper_a_progress.txt, paper_a_sample.xlsx")
""")

md("""
## Next steps

1. **Paste the block** into the chat.
2. **Hand-check:** I draw 30 deals at random from the Review sheet and check the offer price, the commitments and the sponsor against the filings it links to (target: 90% correct).
3. **Sample rules and hypotheses:** we fix the rules in `config/paper_a.yaml`, agree the draft hypotheses, and lock Paper A – the public timestamp.
4. **After the lock:** exit values (re-IPO valuations, sale prices, post-buyout financials), the value bridge by entry regime, WSIB net returns by vintage; the PCRI proposal runs in parallel.
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
