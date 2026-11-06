# aem-research

Base rates for extreme multiples in US small caps, and whether the AEM screening signals carry information. Empirical code for the AEM paper.

[Open the notebook in Colab](https://colab.research.google.com/github/FelixDLanger/aem-research/blob/main/notebooks/01_aem_base_rate_skeleton.ipynb)

## Design at a glance (pre-registration v0.2, draft)

- **Universe:** US common stocks with market cap US$100m–2bn (AEM-2 v2.6), price at least US$1, median 20-day dollar volume at least US$100k, filing 10-K and 10-Q reports with XBRL financials, no SPAC shells.
- **Formation dates:** the first trading day of January, April, July and October, 2013–2023. Smaller companies tagged XBRL financials from mid-2011, so trailing free cash flow is known from filings for every company from January 2013; earlier years are skipped.
- **Outcomes:** touching 2x to 100x within 3, 5 and 10 years, and the five-year end multiple on the six-bucket grid. A series that stops takes a delisting return – Shumway (1997): -30% on NYSE and NYSE American, Shumway and Warther (1999): -55% on Nasdaq, 0% for mergers – always with bounds.
- **Hypotheses:** H1 base rates; H2 the fallen-quality cohort, tested with any insider purchase (H2a) and with opportunistic insiders only (H2b, Cohen, Malloy and Pomorski 2012); H3 the horizon conversion 3x/3y = 6.24x/5y; H4 likelihood ratios for the AEM-2 v2.6 signals, plus a checklist benchmark; H5 regime; H6 post-results insider clusters.
- **Point in time:** every feature is used only after its filing date; a guard fails the run otherwise. Size and price filters undo later split adjustments with 13F implied prices.

## What is here

| Path | Contents |
|---|---|
| `config/preregistration.yaml` | Decisions D1–D6, open review items, hypotheses H1–H6, every parameter. Hashed; `config/LOCK.json` locks it. |
| `aem_research/config.py` | Canonical hash, lock, amendment history; open review items block the lock |
| `aem_research/data/simfin_cache.py`, `stooq.py` | Price sources: your SimFin cache, the Stooq bulk file |
| `aem_research/data/sec.py` | EDGAR quarterly index: Form 25 delistings and merger filings; the SEC client |
| `aem_research/data/sec_datasets.py` | Download, reduce and cache the SEC data sets per file – resumable |
| `aem_research/data/fsds.py`, `aem_research/fundamentals.py` | XBRL facts → trailing free cash flow, net cash, revenue, dilution, the S1 proxy |
| `aem_research/data/insider.py` | Form 4 open-market purchases; routine, opportunistic and unclassified insiders; symbol-to-CIK bridge |
| `aem_research/data/thirteenf.py` | Form 13F: institutional breadth (attention) and implied unadjusted prices |
| `aem_research/features.py` | Universe, fallen-quality cohort, H4 signals, coverage report, universe holes |
| `aem_research/delistings.py` | Ended series classified as merger, performance (by exchange) or unknown |
| `aem_research/outcomes.py` | Outcome engine: touch and end multiples, delisting returns with bounds |
| `aem_research/coverage.py`, `report.py` | Coverage audit and the EODHD verdict |
| `aem_research/guards.py` | Look-ahead and survivorship guards that fail loudly |
| `notebooks/01_aem_base_rate_skeleton.ipynb` | Colab notebook: setup, tests, coverage audit, verdict, features, locked outcome phase |
| `tests/` | Offline test suite on synthetic data in the SEC's published layouts – runs in the notebook before any data is touched |
| `tools/` | `build_notebook.py` generates the notebook (edit cells there, not in the .ipynb); `e2e_synthetic.py` runs every notebook cell against a synthetic Drive in four scenarios |

## Run it

1. Add a Colab secret `SEC_USER_AGENT` with `Your Name your@email` – the SEC asks automated clients to identify themselves.
2. Download `d_us_txt.zip` from [stooq.com/db/h](https://stooq.com/db/h/) (U.S., Daily, ASCII – a CAPTCHA blocks automation) and save it, still zipped, to `My Drive/aem-research/data/stooq/`.
3. Runtime → Run all. The first run downloads about 9 GB of SEC data sets and takes one to two hours. Each file is cached on Drive as it finishes, so after a disconnect, Run all resumes where it stopped. Later runs take minutes.

If the SEC refuses downloads from Colab, save the zips on a computer under their original names into `My Drive/aem-research/data/sec/manual/`; the notebook uses them before the network.

The notebook reads the SimFin data the CVM pipeline cached in `My Drive/cvm-research/data/raw/`. Outputs go to `My Drive/aem-research/output/` (`coverage_report.md`, `coverage_report.json`); features and delisting classes are cached in `My Drive/aem-research/cache/`.

## Pre-registration

Phases A–F look only at whether data exist and at counts, never at returns, so they run before the lock. The outcome engine refuses to run until `config/LOCK.json` matches the config hash. Changing the config afterwards is an amendment: it needs a deliberate re-lock with a note, and the earlier lock stays in the history. A public commit of `LOCK.json` is the timestamp that makes the pre-registration verifiable.

The lock also refuses while `review_before_lock` lists open items. Two remain: confirming quarterly formation dates (D3), and running Phase F once to check that every feature is known for at least 80% of universe windows and that fewer than 25% of ended series are "unknown" delistings. The `estimation` section fixes the tests in advance: likelihood ratios with a firm-cluster bootstrap, Holm corrections, an ordered logit on the six buckets, a 10x logit, start-date fixed effects and the out-of-sample split.

## Outcome rules (summary)

- Entry: first price on or after the start date, within 7 days.
- Windows that end after the data's as-of date are censored, never partially counted. The as-of date is the earliest of the sources' last dates, so a source that ends sooner never looks like a wave of delistings.
- Touch m: a daily close at least m times the entry price inside the window.
- A series that stops inside a window takes the delisting return of its class (merger filing → 0%; price collapse or exchange-filed Form 25-NSE → -30% NYSE/AMEX, -55% Nasdaq; otherwise -30%), always with bounds (total loss, no haircut).
- A gap longer than 120 days ends a series; SimFin tickers used by two companies are split by SimFinId. Neither can fabricate a jump.

## Data and licences

SimFin data is your own cached download. Stooq bulk files are for personal use. SEC EDGAR data and the SEC's structured data sets are public, subject to the SEC's fair-access policy. Raw vendor data stays on Drive and is never committed here – `.gitignore` blocks data files, and a test fails if any appear in the repository.

The code is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).

## Status

Built and tested: config lock with review gate; SimFin and Stooq readers; SEC index (Form 25, merger filings); the three SEC data sets; point-in-time fundamentals, insider signals and attention; universe, cohort and signal flags; delisting classification; coverage audit with the EODHD verdict; outcome engine; guards.
Next: the estimators – likelihood ratios with firm-cluster bootstrap and Holm correction, the ordered logit and the 10x logit, the checklist benchmark.
