# aem-research

Base rates for extreme multiples in US small caps, and whether the AEM screening signals carry information. Empirical code for the AEM paper (Paper B) and, from the same data layer, the take-private sample for Paper A on regime-conditional value creation in buyouts.

[Open notebook 01 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/01_aem_base_rate_skeleton.ipynb) – the AEM base-rate study (run first)
[Open notebook 02 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/02_paper_a_take_private_sample.ipynb) – Paper A's take-private sample (runs on notebook 01's caches)
[Open notebook 03 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/03_universe_a_diagnostic.ipynb) – why universe A is the size it is (reads notebook 01's caches; counts only, about two minutes)
[Open notebook 04 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/04_results.ipynb) – the results, after the lock: outcomes and H1 from the notebook-01 run the lock names (a few minutes)

[Open notebook 05 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/05_paper_a_results.ipynb) – Paper A's results, after its lock: the three confirmatory tests and every exploratory table, from the notebook-02 run the lock names (a few minutes)

[Open notebook 06 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/06_explore.ipynb) – exploration across both papers, after both locks: the premium split by year, margins, own price history, AEM signals of targets, public holders' outcomes in buyouts, the exit slowdown, sponsor styles, public against private (about a minute; everything exploratory)

[Open notebook 07 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/07_estimates.ipynb) – Paper B's H2 to H6, after the lock and notebook 04: the fallen-quality cohort, the horizon conversion, each signal's likelihood ratio out of sample, the 10x model against the checklist, with firm-cluster intervals (a minute or two)

[Open notebook 08 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/08_followup.ipynb) – follow-up, rounds one and two, from the files on Drive only: the robustness versions, the data run's features rebuilt for the locked and the new start dates (the locked dates must reproduce the saved table), the confirmation of C1 to C3 on start dates from October 2023 to July 2025, the purchases-only insider classification, multiples in real terms, basis B with the calibration, Paper A's benchmark through 2025; 20 to 40 minutes

[Open notebook 09 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/09_exits.ipynb) – Paper A, exploratory (A-X1, A-X2): the price each visible sale to a listed buyer was made at, read from the buyer's 8-K and press release, against the entry value and the public benchmark's multiple, with a seeded sample of 30 filings to read; WSIB's quarterly reports before March 2023; Paper B's last checks (the loss side of each signal, the fall within start dates, without pharma and biotech, by size) and a signal scan over every other measure in the feature table with a discovery and a holdout period; needs the `SEC_USER_AGENT` secret; about ten minutes

[Open notebook 10 · v21 in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/10_final.ipynb) – Paper A: the exit measure rebuilt (every hit of the exit search read, kept only where a sentence says the target was acquired; every kept sale printed with its sentence), A-H3 on the checked exits as a disclosed deviation, exit values; Paper B: eighteen combinations fixed in advance and a flexible model, on end multiples before and after 2018 and on the new dates (exploratory); needs the `SEC_USER_AGENT` secret; 15 to 25 minutes

The version (v21) is in each notebook's title and in the first line of its paste block. When a message asks for a new run, it names the version; a copy that is still open from an earlier version stops at setup and says which one to open.

## Design at a glance (pre-registration v0.3, draft)

- **Prices (D10):** every price the primary analysis uses comes from SEC filings – quarter-end prices from Form 13F (market value over shares held, median across at least three filers), the SEC fails-to-deliver files where 13F has none, and splits found from the filings' restated history (later 10-Qs and 10-Ks restate past share counts by exactly the split ratio), the holders' share counts and CUSIP changes. A reverse split whose ratio no evidence gives is neutralized, so it can never count as a gain. Companies that later delisted are priced exactly like the survivors, and the published numbers rest on public data only. Robustness B keeps daily SimFin and Stooq closes and fills only the companies they never priced.
- **Universe:** US exchange-listed common stocks with market cap US$100m–2bn (AEM-2 v2.6), price at least US$1, liquid (a minimum number of 13F holders, calibrated on the US$100k median dollar-volume screen), filing 10-K and 10-Q reports with XBRL financials, no SPAC shells.
- **Formation dates:** the first trading day of January, April, July and October, 2013–2023 (quarterly confirmed Nov 6, 2026; January alone is the AEM-2 v2.6 robustness check). Smaller companies tagged XBRL financials from mid-2011, so trailing free cash flow is known from filings for every company from January 2013; the SEC price panel starts in July 2013, the first start date with usable 13F holdings.
- **Outcomes:** touching 2x to 100x within 3, 5 and 10 years – at a quarter-end close in the primary analysis, at a daily close in robustness B; the ratio between the two is measured on 2009–2012 start dates, outside the sample – and the five-year end multiple on the six-bucket grid. A series that stops takes a delisting return – Shumway (1997): -30% on NYSE and NYSE American, Shumway and Warther (1999): -55% on Nasdaq, 0% for mergers – always with bounds.
- **Hypotheses:** H1 base rates; H2 the fallen-quality cohort, tested with any insider purchase (H2a) and with opportunistic insiders only (H2b, Cohen, Malloy and Pomorski 2012); H3 the horizon conversion 3x/3y = 6.24x/5y; H4 likelihood ratios for the AEM-2 v2.6 signals, plus a checklist benchmark; H5 regime; H6 post-results insider clusters.
- **Regime:** the 2022 split is primary; six exploratory moderators from FRED and the universe itself (real yield, tightening, credit spread, CPI inflation, market stress, starting valuation), measured on each start date and shared with Paper A. A regime difference found in the locked sample is tested once, in late 2028, on start dates added after the lock (D8).
- **Falsification and power:** an insider-sale placebo for H2, and the smallest detectable likelihood ratio per cohort before the lock.
- **Discovery:** the lock fixes what is tested, not what may be learned. Unplanned findings are welcome, reported as exploratory with every analysis tried, and earn weight in the screening terminal once they hold on data they were not found in (D9).
- **Point in time:** every feature is used only after its filing date; a guard rejects the features otherwise. Size and price filters undo later split adjustments with 13F implied prices.

## What is here

| Path | Contents |
|---|---|
| `config/preregistration.yaml` | Decisions D1–D10, open review items, hypotheses H1–H6, every parameter. Hashed; `config/LOCK.json` locks it. |
| `aem_research/config.py` | Canonical hash, lock, amendment history; open review items block the lock |
| `aem_research/data/simfin_cache.py`, `stooq.py` | Price sources: your SimFin cache, the Stooq bulk file |
| `aem_research/data/sec.py` | EDGAR quarterly index: Form 25 delistings and merger filings; the SEC client |
| `aem_research/data/sec_datasets.py` | Download, reduce and cache the SEC data sets per file – resumable |
| `aem_research/data/fsds.py`, `aem_research/fundamentals.py` | XBRL facts → trailing free cash flow, net cash, revenue, dilution, the S1 proxy |
| `aem_research/data/insider.py` | Form 4 open-market purchases; routine, opportunistic and unclassified insiders; symbol-to-CIK bridge |
| `aem_research/data/thirteenf.py` | Form 13F: institutional breadth (attention), implied unadjusted prices and each security's top holders |
| `aem_research/data/ftd.py` | SEC fails-to-deliver files: the CUSIP-ticker bridge and daily closes, cached per half-month file |
| `aem_research/restatements.py` | Splits from the filings' restated history: the same period's share count as first reported and as later 10-Qs and 10-Ks report it again differs by exactly the splits in between |
| `aem_research/splits.py`, `secprices.py` | Splits from holders and CUSIP changes; the SEC price panel (D10), its listing, liquidity and validation checks |
| `aem_research/features.py` | Universe, fallen-quality cohort, H4 signals, coverage report, universe holes |
| `aem_research/delistings.py` | Ended series classified as merger, performance (by exchange) or unknown |
| `aem_research/outcomes.py` | Outcome engine: touch and end multiples, delisting returns with bounds |
| `aem_research/data/fred.py` | FRED macro series (no key) and the regime moderators on each start date |
| `aem_research/coverage.py`, `report.py` | Coverage audit, the EODHD verdict, and the paste-ready run summary |
| `aem_research/export.py`, `runlog.py` | Excel workbook of every table; per-phase status, timings and errors |
| `aem_research/guards.py` | Look-ahead and survivorship guards that fail loudly |
| `notebooks/01_aem_base_rate_skeleton.ipynb` | Colab notebook: setup, tests, coverage audit, verdict, features, regime, Excel export, locked outcome phase, run summary |
| `config/paper_a.yaml` | Paper A's draft pre-registration (decisions A1–A3, sample, entry and exit rules, draft hypotheses); `config/paper_a_LOCK.json` locks it |
| `aem_research/data/fts.py` | EDGAR full-text search: phrase search across every filing since 2001, cached per year on Drive |
| `aem_research/takeprivate.py` | Paper A: take-private deals, completion, terms from the merger documents, entry EBITDA and enterprise value, exit routes, public benchmark multiples |
| `aem_research/data/wsib.py` | Washington State Investment Board's quarterly private equity table, fund by fund |
| `notebooks/02_paper_a_take_private_sample.ipynb` | Colab notebook for Paper A's sample – counts and entry-side data only until Paper A's lock |
| `aem_research/diagnostics.py`, `notebooks/03_universe_a_diagnostic.ipynb` | Universe funnel: windows passing each filter alone, so far and without it; the vendor universe's windows looked up in A; counts and ratios only |
| `tests/` | Offline test suite on synthetic data in the SEC's published layouts – runs in the notebook before any data is touched |
| `tools/` | `build_notebook.py`, `build_notebook_paper_a.py` and `build_notebook_diag.py` generate the notebooks (edit cells there, not in the .ipynb); `e2e_synthetic.py` runs every cell of notebook 01 against a synthetic Drive in nine scenarios, including truncated downloads, a corrupt FRED file, a planted failing test, planted splits (restated in the synthetic filings, as filers do), companies SimFin never had (priced from SEC data alone), filings without cover pages and an EODHD gap-fill; `e2e_paper_a.py` runs notebook 02 on top of it, and on an empty Drive |

## Run it

1. Add a Colab secret `SEC_USER_AGENT` with `Your Name your@email` – the SEC asks automated clients to identify themselves.
2. Download `d_us_txt.zip` from [stooq.com/db/h](https://stooq.com/db/h/) (U.S., Daily, ASCII – a CAPTCHA blocks automation) and save it, still zipped, to `My Drive/aem-research/data/stooq/`.
3. Runtime → Run all. The first run downloads about 9 GB of SEC data sets and takes one to two hours. Each file is cached on Drive as it finishes, so after a disconnect, Run all resumes where it stopped. Later runs take minutes – except the first run after Nov 9, 2026, which reads the 13F files again and downloads the fails-to-deliver files once.

EODHD is no longer part of the plan (D10): its personal plans do not cover publishing results. The notebook's EODHD step stays off unless you add a Colab secret `EODHD_API_KEY` for a private cross-check.

If the SEC refuses downloads from Colab, save the zips on a computer under their original names into `My Drive/aem-research/data/sec/manual/`; the notebook uses them before the network.

The downloads run on Google's servers, so your own connection only carries the notebook page; keep the tab open and the device awake. The raw SEC zips pass through memory and are never stored – Drive keeps reduced extracts, roughly 1–2 GB.

When the run ends, the last cell prints one block with every result, timing, memory peak, warning and error: paste it into the chat. A phase that fails is recorded there and the run carries on. Only a failing test suite stops the run on purpose, and it prints the same block first; the look-ahead guard discards the features rather than keep any that use the future. If Colab crashes or disconnects before the end, `run_progress.txt` shows how far the run got – paste its last lines, then Run all resumes from the cached files.

The notebook reads the SimFin data the CVM pipeline cached in `My Drive/cvm-research/data/raw/`. Outputs in `My Drive/aem-research/output/`:
- `aem_results.xlsx` – summary, coverage by start date, pre-lock checks, power, regime, the SEC price panel and its splits, the quarter-end vs daily calibration, universe holes, price basis, delisting classes, SEC data set status, and every universe window (A and B) with its features;
- `run_summary.txt` – the paste-ready block; `run_progress.txt` – each phase's start and end, kept across reruns; `coverage_report.md` and `.json` – the audit;
- after the lock: `aem_outcomes.xlsx`, `outcomes.parquet` (A) and `outcomes_b.parquet` (robustness B).

Features, delisting classes and the regime table are also cached as parquet in `My Drive/aem-research/cache/`; raw inputs (tens of millions of rows) stay as parquet there too, beyond Excel's 1,048,576-row limit.

## Paper A – take-private sample (notebook 02)

Run it after notebook 01: it reuses that run's Drive caches and downloads only what is new. Sponsor-backed take-privates are found with EDGAR full-text search ("equity commitment letter" in target-side merger filings, 2012 on); each deal's definitive document gives the offer price, the equity and debt commitments and the sponsor, each with the sentence it came from; the terms are checked against a second filing for every deal (the target's 8-K around the announcement, else another merger filing) and on 30 deals drawn by a fixed seed, shown with their passages; the target's last XBRL financials give entry EBITDA, net debt and enterprise value; the exit routes that leave a public trace (still files with the SEC, re-IPO, sale to a listed buyer) are detected; the public small-cap benchmark multiples and WSIB's quarterly fund table complete the inputs. The method follows Guo, Hotchkiss and Song (2011, Journal of Finance 66(2), 479–517).

Until `config/paper_a.yaml` is locked (decision A2), notebook 02 shows counts and entry-side data only – no exit values, exit dates, value-creation components or fund returns; those are cached for after the lock. Outputs: `paper_a_sample.xlsx` (deals, review snippets, exit routes, benchmark, WSIB counts), `paper_a_summary.txt` (the paste block) and `paper_a_progress.txt` in `My Drive/aem-research/output/`.

## Pre-registration

Phases A–F look only at whether data exist and at counts, never at returns, so they run before the lock. The outcome engine refuses to run until `config/LOCK.json` matches the config hash. Changing the config afterwards is an amendment: it needs a deliberate re-lock with a note, and the earlier lock stays in the history. A public commit of `LOCK.json` is the timestamp that makes the pre-registration verifiable.

The lock also refuses while `review_before_lock` lists open items. One remains: running Phase F once to check that every feature is known for at least 80% of universe windows, that fewer than 25% of ended series are "unknown" delistings, how large an effect each cohort can detect, and that the SEC price panel agrees with the vendors' split-adjusted prices (returns within 10% for 95% of company-quarters, splits found with 90% precision and recall). The `estimation` section fixes the tests in advance: likelihood ratios with a firm-cluster bootstrap, Holm corrections, an ordered logit on the six buckets, a 10x logit, start-date fixed effects and the out-of-sample split.

## Outcome rules (summary)

- Entry: first price on or after the start date, within 7 days. On the SEC panel the quarter-end price is dated the day after the quarter-end, the start date it serves.
- Windows that end after the data's as-of date are censored, never partially counted. The vendor panel's as-of date is the earliest of the sources' last dates; the SEC panel's is the last quarter whose 13F filings are complete.
- Touch m: a close at least m times the entry price inside the window – quarter-end closes plus a series' last price before it stops (A), daily closes (B).
- A series that stops inside a window takes the delisting return of its class (merger filing → 0%; price collapse or exchange-filed Form 25-NSE → -30% NYSE/AMEX, -55% Nasdaq; otherwise -30%), always with bounds (total loss, no haircut).
- A gap longer than 120 days ends a vendor series (200 days, two missing quarters, on the SEC panel); SimFin tickers used by two companies are split by SimFinId; on the SEC panel a change of CUSIP that is neither a split nor the same security starts a new series. None can fabricate a jump.

## Data and licences

SimFin data is your own cached download. Stooq bulk files are for personal use. The primary analysis therefore rests on SEC data alone (D10); SimFin and Stooq feed robustness B and the checks. SEC EDGAR data, the SEC's structured data sets and the fails-to-deliver files are public, subject to the SEC's fair-access policy. FRED needs no key; BAA10Y is Moody's copyrighted data, so the series and the regime table stay on Drive and the papers report estimates, not the series. EDGAR full-text search and the filings it finds are public; WSIB's quarterly reports are public documents, read with a generic client name. PCRI's deal-level database is for academic researchers through its own application and enclave; nothing from it enters this repository. EODHD's personal plans cover private research, not publication, so EODHD is not used for published results. Raw vendor data stays on Drive and is never committed here – `.gitignore` blocks data files, and a test fails if any appear in the repository.

The code is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).

## Status

Built and tested: config lock with review gate; the SEC price panel with split detection and its validation (D10); SimFin and Stooq readers; SEC index (Form 25, merger filings); the three SEC data sets and the fails-to-deliver files; point-in-time fundamentals, insider signals and attention; universe, cohort, placebo and signal flags; delisting classification; FRED regime moderators; power table; coverage audit with the EODHD verdict; outcome engine; guards; Excel export; run log with a paste-ready summary.
Paper A (notebook 02): take-private sample, terms, entry EBITDA and enterprise value, exit-route detection, public benchmark, WSIB fund table – counts only until Paper A's lock.
Next: the estimators – likelihood ratios with firm-cluster bootstrap and Holm correction, the ordered logit and the 10x logit, the checklist benchmark; for Paper A, reading the 30 seeded deals and the cross-check's disagreements, its hypotheses and lock, then exit values and the value bridge.
