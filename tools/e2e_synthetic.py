"""End-to-end check: build a synthetic Drive, then execute every notebook code cell in order.

Usage (outside Colab, no network):
    python tools/e2e_synthetic.py <scratch_dir> <repo_dir> [full|nocik_nostooq|empty|locked|broken|testsfail]
Scenarios:
  full           SimFin with CIKs and ISINs, a Stooq zip, SEC index caches, and the SEC data
                 sets (XBRL, Form 4, 13F) as hand-downloaded zips in data/sec/manual
  nocik_nostooq  SimFin without CIKs, no Stooq, no SEC data sets
  empty          no data at all
  locked         as full, on a copy of the repo whose pre-registration is finalized and
                 locked, so Phase E computes outcomes
  broken         as full, plus a truncated Stooq zip, a truncated 13F zip and a corrupt FRED
                 file: the run must finish, report Phase B as failed with what to do, skip the
                 bad 13F zip, keep the other FRED series, and list all three in the paste block
  testsfail      as full, on a copy of the repo with one planted failing test: the run must
                 stop at the test cell and print the paste block with the failing test
Every scenario must finish with exit code 0.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(sys.argv[1])
REPO = Path(sys.argv[2])
SCENARIO = sys.argv[3] if len(sys.argv) > 3 else "full"
FULL = SCENARIO in ("full", "locked", "broken", "testsfail")
DRIVE = SCRATCH / "drive"
shutil.rmtree(DRIVE, ignore_errors=True)
rng = np.random.default_rng(42)
sys.path.insert(0, str(REPO / "tests"))
from sec_fixtures import (f13_zip, fsds_zip, insider_zip, num_row, sec_date, sub_row)  # noqa: E402

# ---------- synthetic SimFin cache (CVM layout: data/raw/us_prices.pkl with (Ticker, Date) index)
dates = pd.bdate_range("2008-01-02", "2025-12-31")
n = 300
tickers = [f"S{i:03d}" for i in range(n)]
ciks = {t: 100000 + i for i, t in enumerate(tickers)}
cusips = {t: f"{i + 1:06d}105" for i, t in enumerate(tickers)}
# 30% delist at random dates 2010-2024; the rest survive
delist = {t: pd.Timestamp(rng.choice(pd.bdate_range("2010-01-01", "2024-06-30"))) for t in tickers[:90]}
rows = []
for i, t in enumerate(tickers):
    end = delist.get(t, dates[-1])
    d = dates[dates <= end]
    p = 10 * np.exp(np.cumsum(rng.normal(0.0002, 0.025, len(d))))
    rows.append(pd.DataFrame({"Ticker": t, "Date": d, "SimFinId": 9000 + i, "Open": p, "High": p * 1.01,
                              "Low": p * 0.99, "Close": p, "Adj. Close": p * 0.97, "Volume": 1e5,
                              "Dividend": np.nan, "Shares Outstanding": 1e7}))
prices = pd.concat(rows).set_index(["Ticker", "Date"])
raw = DRIVE / "cvm-research" / "data" / "raw"
raw.mkdir(parents=True)
if SCENARIO != "empty":
    prices.to_pickle(raw / "us_prices.pkl")
comp = pd.DataFrame({"Ticker": tickers, "SimFinId": [9000 + i for i in range(n)],
                     "Company Name": [f"{t} CORP" for t in tickers],
                     "ISIN": [f"US{cusips[t]}0" for t in tickers],
                     "CIK": [float(ciks[t]) if i % 50 else np.nan for i, t in enumerate(tickers)]}).set_index("Ticker")
if SCENARIO == "nocik_nostooq":
    comp = comp.drop(columns=["CIK", "ISIN"])
if SCENARIO != "empty":
    comp.to_pickle(raw / "us_companies.pkl")

# ---------- synthetic Stooq zip: survivors overlap SimFin; 20 delisted names SimFin lacks; ETFs
stq_dir = DRIVE / "aem-research" / "data" / "stooq"
stq_dir.mkdir(parents=True)
HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


def member(tk, d, p):
    body = "\n".join(f"{tk}.US,D,{x:%Y%m%d},000000,{v:.4f},{v:.4f},{v:.4f},{v:.4f},1000,0" for x, v in zip(d, p))
    return HEADER + body + "\n"


stq_extra = {f"Q{i:02d}": pd.Timestamp(rng.choice(pd.bdate_range("2011-01-01", "2023-12-31"))) for i in range(20)}
with zipfile.ZipFile(stq_dir / ("d_us_txt.zip" if FULL else "ignore.bin"), "w", zipfile.ZIP_DEFLATED) as zf:
    for i, t in enumerate(tickers[90:]):                       # survivors, Stooq close = SimFin adj close
        sub = prices.loc[t]
        zf.writestr(f"data/daily/us/nasdaq stocks/{i % 3 + 1}/{t.lower()}.us.txt",
                    member(t, sub.index, sub["Adj. Close"].to_numpy()))
    for t, end in stq_extra.items():
        d = dates[dates <= end]
        zf.writestr(f"data/daily/us/nyse stocks/1/{t.lower()}.us.txt",
                    member(t, d, 5 * np.exp(np.cumsum(rng.normal(0, 0.02, len(d))))))
    zf.writestr("data/daily/us/nasdaq etfs/qqq.us.txt", member("QQQ", dates[:10], np.ones(10)))
    zf.writestr("data/daily/us/nyse stocks/2/empty.us.txt", HEADER)

# ---------- synthetic SEC index caches: Form 25 (with the exchange's row) and merger filings
sec_dir = DRIVE / "aem-research" / "data" / "sec"
sec_dir.mkdir(parents=True)
COLS = ["cik", "company", "form", "date_filed", "filename"]
ev, mg = [], []
for k, (t, end) in enumerate(delist.items()):                   # real delistings of SimFin names
    when = end + pd.Timedelta(days=int(rng.integers(-10, 11)))
    fn = f"edgar/data/{ciks[t]}/f25-{k}.txt"
    ev.append((ciks[t], f"{t} CORP", "25-NSE", when, fn))
    ev.append((1120193, "THE NASDAQ STOCK MARKET LLC", "25-NSE", when, fn))
    if k % 3 == 0:                                              # a third were takeovers
        mg.append((ciks[t], f"{t} CORP", "DEFM14A", end - pd.Timedelta(days=90), f"edgar/data/{ciks[t]}/m-{k}.txt"))
for i, t in enumerate(tickers[90:110]):                        # exchange moves: prices continue
    ev.append((ciks[t], f"{t} CORP", "25", pd.Timestamp("2016-03-01") + pd.Timedelta(days=i), f"edgar/data/x{i}.txt"))
for i in range(25):                                            # issuers SimFin never covered
    ev.append((800000 + i, f"UNKNOWN {i} INC", "25-NSE", pd.Timestamp("2014-05-01") + pd.Timedelta(days=9 * i),
               f"edgar/data/u{i}.txt"))
for i in range(5):
    ev.append((900000 + i, f"ALPHA {i} ACQUISITION CORP", "25-NSE", pd.Timestamp("2022-02-01"), f"edgar/data/a{i}.txt"))
idx = {"form25": pd.DataFrame(ev, columns=COLS), "merger": pd.DataFrame(mg, columns=COLS)}
for name, tab in idx.items():
    tab["date_filed"] = tab["date_filed"].astype("datetime64[ns]")
    if SCENARIO == "empty":
        tab = tab.iloc[0:0]
    for y in range(2012, 2027):
        for q in range(1, 5):
            if (y, q) >= (2026, 4):
                continue
            g = tab[(tab["date_filed"].dt.year == y) & (tab["date_filed"].dt.quarter == q)]
            if SCENARIO != "empty" or g.empty:
                g[COLS].reset_index(drop=True).to_pickle(sec_dir / f"{name}_{y}Q{q}.pkl")

# ---------- synthetic SEC data sets, as zips saved by hand (data/sec/manual)
manual = sec_dir / "manual"
manual.mkdir(parents=True)


def q_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}q{(ts.month - 1) // 3 + 1}"


def f13_label(ts: pd.Timestamp) -> str:
    if ts < pd.Timestamp("2024-01-01"):
        return q_label(ts)
    if ts < pd.Timestamp("2024-03-01"):
        return "01jan2024-29feb2024"
    m0 = ((ts.month - 3) // 3) * 3 + 3 if ts.month >= 3 else 12
    y0 = ts.year if ts.month >= 3 else ts.year - 1
    a = pd.Timestamp(year=y0, month=m0, day=1)
    b = (a + pd.DateOffset(months=2)) + pd.offsets.MonthEnd(0)
    return f"{a:%d%b%Y}-{b:%d%b%Y}".lower()


if FULL:
    fs_files: dict = {}
    for i, t in enumerate(tickers):
        if not i % 50:
            continue                                            # SimFin has no CIK for these
        cik, last = ciks[t], delist.get(t, pd.Timestamp("2025-12-31"))
        qe = pd.date_range("2010-03-31", "2025-09-30", freq="QE")
        flows = {"ocf": rng.normal(4e6, 3e6, len(qe)), "capex": np.abs(rng.normal(1e6, 3e5, len(qe))),
                 "rev": np.abs(rng.normal(30e6, 5e6, len(qe)))}
        flows = {k: pd.Series(v, index=qe) for k, v in flows.items()}
        wad = pd.Series(5e7 * (1 + 0.01 * np.arange(len(qe)) / 4), index=qe)
        for k_, P in enumerate(qe):
            if P < pd.Timestamp("2011-06-30") or P > last:
                continue
            k = (P.month - 1) // 3 + 1
            filed = P + pd.Timedelta(days=60 if k == 4 else 40)
            if filed > last + pd.Timedelta(days=30):
                continue
            adsh = f"{cik:010d}-{filed:%y}-{k_:06d}"
            form, fp = ("10-K", "FY") if k == 4 else ("10-Q", f"Q{k}")
            subs_ = [sub_row(adsh, cik, form, f"{P:%Y%m%d}", fp, f"{filed:%Y%m%d}", name=f"{t} CORP",
                             sic=6770 if i % 37 == 0 else 3571)]
            subs_[0]["instance"] = f"{t.lower()}-{P:%Y%m%d}.xml"
            nums = []
            for P_ in (P, P - pd.DateOffset(months=12) + pd.offsets.MonthEnd(0)):
                fy_start = pd.Timestamp(year=P_.year, month=1, day=1)
                ytd = slice(fy_start, P_)
                q = 4 if k == 4 else k
                for tag, key in (("NetCashProvidedByUsedInOperatingActivities", "ocf"),
                                 ("PaymentsToAcquirePropertyPlantAndEquipment", "capex"), ("Revenues", "rev")):
                    nums.append(num_row(adsh, tag, f"{P_:%Y%m%d}", q, round(float(flows[key][ytd].sum()), 0)))
                nums.append(num_row(adsh, "WeightedAverageNumberOfDilutedSharesOutstanding", f"{P_:%Y%m%d}",
                                    1 if k < 4 else 4, round(float(wad[P_]), 0), "shares"))
            nums += [num_row(adsh, "CashAndCashEquivalentsAtCarryingValue", f"{P:%Y%m%d}", 0,
                             round(float(rng.uniform(5e6, 6e7)), 0)),
                     num_row(adsh, "LongTermDebt", f"{P:%Y%m%d}", 0, round(float(rng.uniform(0, 5e7)), 0)),
                     num_row(adsh, "EntityCommonStockSharesOutstanding", f"{filed + pd.offsets.MonthEnd(0):%Y%m%d}",
                             0, round(float(wad[P]), 0), "shares", version="dei/2014")]
            if k == 4:
                nums.append(num_row(adsh, "EntityPublicFloat", f"{P - pd.DateOffset(months=6) + pd.offsets.MonthEnd(0):%Y%m%d}",
                                    0, round(float(wad[P] * 8), 0), version="dei/2014"))
            lab = q_label(filed)
            fs_files.setdefault(lab, ([], []))
            fs_files[lab][0].extend(subs_)
            fs_files[lab][1].extend(nums)
    for lab, (s_, n_) in fs_files.items():
        (manual / f"{lab}.zip").write_bytes(fsds_zip(s_, n_))

    ins_files: dict = {}
    acc = 0

    def add_trade(issuer, symbol, owner, rel, trans, code):
        global acc
        acc += 1
        filed = trans + pd.Timedelta(days=2)
        a = f"{issuer:010d}-{filed:%y}-{acc:06d}"
        lab = q_label(filed)
        ins_files.setdefault(lab, ([], [], []))
        ins_files[lab][0].append({"ACCESSION_NUMBER": a, "FILING_DATE": sec_date(f"{filed:%Y-%m-%d}"),
                                  "PERIOD_OF_REPORT": sec_date(f"{trans:%Y-%m-%d}"), "DOCUMENT_TYPE": "4",
                                  "ISSUERCIK": issuer, "ISSUERNAME": f"{symbol} CORP", "ISSUERTRADINGSYMBOL": symbol})
        ins_files[lab][1].append({"ACCESSION_NUMBER": a, "RPTOWNERCIK": owner, "RPTOWNERNAME": f"O{owner}",
                                  "RPTOWNER_RELATIONSHIP": rel})
        ins_files[lab][2].append({"ACCESSION_NUMBER": a, "NONDERIV_TRANS_SK": acc, "SECURITY_TITLE": "Common",
                                  "TRANS_DATE": sec_date(f"{trans:%Y-%m-%d}"), "TRANS_CODE": code,
                                  "TRANS_SHARES": 1000, "TRANS_PRICEPERSHARE": 10,
                                  "TRANS_ACQUIRED_DISP_CD": "A" if code == "P" else "D",
                                  "DIRECT_INDIRECT_OWNERSHIP": "D"})

    for i, t in enumerate(tickers):
        cik, last = ciks[t], delist.get(t, pd.Timestamp("2025-12-31"))
        for y in range(2009, 2026):
            mar = pd.Timestamp(year=y, month=3, day=10)
            if mar < last:
                add_trade(cik, t, 5_000_000 + 2 * i, "Director", mar, "S")        # routine seller
            if rng.random() < 0.35:
                d = pd.Timestamp(year=y, month=int(rng.integers(1, 13)), day=15)
                if d < last:
                    add_trade(cik, t, 5_000_001 + 2 * i, "Officer", d, "P")      # opportunistic buyer
    for i, (t, end) in enumerate(stq_extra.items()):                            # Stooq-only names get a CIK
        for y in range(2009, end.year + 1):
            add_trade(700000 + i, t, 6_000_000 + i, "Director", pd.Timestamp(year=y, month=6, day=1), "S")
    for lab, (s_, o_, t_) in ins_files.items():
        if lab >= "2009q1":
            (manual / f"{lab}_form345.zip").write_bytes(insider_zip(s_, o_, t_))

    f13_files: dict = {}
    periods = pd.date_range("2013-03-31", "2025-09-30", freq="QE")
    closes = prices["Close"]

    def close_at(t, when):
        s = closes.loc[t]
        return s[s.index <= when].iloc[-1]

    for j, P in enumerate(periods):
        filed = P + pd.Timedelta(days=40)
        lab = f13_label(filed)
        f13_files.setdefault(lab, ([], []))
        for f in range(25):
            a = f"{900001 + f:010d}-{filed:%y}-{j:06d}"
            f13_files[lab][0].append({"ACCESSION_NUMBER": a, "FILING_DATE": sec_date(f"{filed:%Y-%m-%d}"),
                                      "SUBMISSIONTYPE": "13F-HR", "CIK": 900001 + f,
                                      "PERIODOFREPORT": sec_date(f"{P:%Y-%m-%d}")})
            for i, t in enumerate(tickers):
                if P > delist.get(t, pd.Timestamp("2030-01-01")):
                    continue
                if rng.random() < 0.2 + 0.6 * (i % 10) / 10:
                    close = float(close_at(t, P))                  # the unadjusted quarter-end close
                    value = close * 100_000 / (1000.0 if filed < pd.Timestamp("2023-01-03") else 1.0)
                    f13_files[lab][1].append({"ACCESSION_NUMBER": a, "NAMEOFISSUER": f"{t} CORP", "CUSIP": cusips[t],
                                              "SSHPRNAMT": 100_000, "VALUE": int(round(value)),
                                              "SSHPRNAMTTYPE": "SH", "TITLEOFCLASS": "COM"})
    for lab, (s_, i_) in f13_files.items():
        (manual / f"{lab}_form13f.zip").write_bytes(f13_zip(s_, i_))
    print(f"SEC data sets: {len(fs_files)} XBRL, {len(ins_files)} insider, {len(f13_files)} 13F zips")

    # FRED series as saved by hand (fred.stlouisfed.org download button: <SERIES>.csv)
    fred_dir = DRIVE / "aem-research" / "data" / "fred"
    fred_dir.mkdir(parents=True)
    days = pd.bdate_range("2011-01-03", "2025-12-31")
    months = pd.date_range("2010-01-01", "2025-11-01", freq="MS")
    walk = lambda n, s: np.round(np.cumsum(rng.normal(0, s, n)), 3)
    for sid, idx, base, step, hdr in (("DFII10", days, 0.5, 0.02, "observation_date"),
                                      ("DGS2", days, 1.0, 0.02, "observation_date"),
                                      ("BAA10Y", days, 2.5, 0.01, "DATE"),
                                      ("CPIAUCNS", months, 220.0, 0.0, "observation_date")):
        vals = (base + walk(len(idx), step)) if step else base * np.cumprod(np.full(len(idx), 1.002))
        body = "\n".join(f"{d:%Y-%m-%d},{'.' if k % 97 == 5 else round(float(v), 3)}" for k, (d, v) in enumerate(zip(idx, vals)))
        (fred_dir / f"{sid}.csv").write_text(f"{hdr},{sid}\n{body}\n")
    if SCENARIO == "broken":
        bad = sorted(manual.glob("*_form13f.zip"))[3]
        bad.write_bytes(bad.read_bytes()[:500])                  # a truncated download
        (fred_dir / "BAA10Y.csv").write_text("not a csv at all")  # a corrupt hand-saved file
        stz = stq_dir / "d_us_txt.zip"
        stz.write_bytes(stz.read_bytes()[:2000])                 # an interrupted browser download

# ---------- locked scenario: a copy of the repo with a finalized, locked pre-registration
if SCENARIO == "locked":
    copy = SCRATCH / "repo_locked"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    script = ("import yaml, pathlib; from aem_research.config import lock_config; "
              "p = pathlib.Path('config/preregistration.yaml'); d = yaml.safe_load(p.read_text()); "
              "d['study']['status'] = 'final'; d['review_before_lock'] = []; "
              "p.write_text(yaml.safe_dump(d, sort_keys=False)); print(lock_config(p, note='e2e'))")
    subprocess.run([sys.executable, "-c", script], cwd=copy, check=True)
    REPO = copy

# ---------- testsfail scenario: a copy of the repo with one planted failing test
if SCENARIO == "testsfail":
    copy = SCRATCH / "repo_testsfail"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    (copy / "tests" / "test_zz_planted_failure.py").write_text(
        "def test_planted_failure():\n    assert 1 == 2, 'planted by e2e testsfail'\n")
    REPO = copy

# ---------- execute the notebook's code cells in order
os.environ.update(AEM_DRIVE_ROOT=str(DRIVE), AEM_REPO_DIR=str(REPO), AEM_SEC_OFFLINE="1")
nb = json.loads((REPO / "notebooks" / "01_aem_base_rate_skeleton.ipynb").read_text())
ns = {"__name__": "__main__"}
t0 = time.time()
for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
    print(f"\n======== code cell {i} ========")
    if SCENARIO != "testsfail":
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
        continue
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
    except RuntimeError as exc:                       # Colab's Run all stops at the first exception
        printed = buf.getvalue()
        print(printed)
        assert "Test suite failed – FAILED tests/test_zz_planted_failure.py::test_planted_failure" in str(exc), exc
        assert "AEM RUN SUMMARY" in printed and "0 tests STOPPED" in printed, "no paste block at the stop"
        assert "ERROR     [0 tests] RuntimeError: Test suite failed – FAILED tests/test_zz_planted_failure.py" in printed
        assert "paths" not in ns, "cells after the failed tests must not run"
        print(f"\nSTOPPED AT THE TEST CELL AS INTENDED ({type(exc).__name__}) in {time.time() - t0:.1f}s")
        sys.exit(0)
    print(buf.getvalue())
if SCENARIO == "testsfail":
    raise SystemExit("the planted failing test did not stop the run")
print(f"\nALL CELLS OK in {time.time() - t0:.1f}s")
report = json.loads((DRIVE / "aem-research" / "output" / "coverage_report.json").read_text())
print(json.dumps(report["verdict"], indent=1))
if FULL:
    assert report.get("features"), "Phase F did not build features"
    print("prelock:", report["prelock"])
summary_text = (DRIVE / "aem-research" / "output" / "run_summary.txt").read_text()
print(summary_text)
progress = (DRIVE / "aem-research" / "output" / "run_progress.txt").read_text()
assert "=== run started" in progress and "=== run finished" in progress and "start F features" in progress
if SCENARIO == "broken":
    assert "ERROR     [B Stooq] ValueError: d_us_txt.zip is not a complete zip" in summary_text, "Stooq failure missing"
    assert "download it again" in summary_text
    assert "WARNING   [F regime] FRED BAA10Y missing, so credit_spread_baa is not computed" in summary_text
    assert "BAA10Y missing (manual BAA10Y.csv unreadable" in summary_text
    assert "real_yield_10y" in summary_text and "credit_spread_baa" not in summary_text.split("REGIME")[1].split("\n")[0]
    assert "1 bad" in summary_text and "(bad " in summary_text, "the truncated 13F zip was not reported"
    assert report.get("features"), "a bad Stooq or 13F zip must not stop the features"
    assert "end   B Stooq: FAILED" in progress
elif FULL:
    assert "ERRORS    none" in summary_text and "WARNING" not in summary_text, "unexpected problems in a clean run"
if SCENARIO == "locked":
    out = ns["result"]
    assert len(out) and set(out["status"]) <= {"complete", "delisted", "series_ended", "censored", "no_entry"}
    print("outcome statuses:", out["status"].value_counts().to_dict())
