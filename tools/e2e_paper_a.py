"""End-to-end check for notebook 02, on top of notebook 01's synthetic Drive. No network.

    python tools/e2e_paper_a.py <scratch_dir> <repo_dir> [full|fresh]

  full   runs tools/e2e_synthetic.py (full) first, then adds Paper A's inputs the way the SEC
         and WSIB would serve them – full-text search results, deal documents (one reachable
         only through its filing index, one missing), Form 15 and IPO index subsets, an 8-K
         exit, a post-buyout 10-K, two WSIB reports (one corrupt), second documents for the
         check of the terms (an agreeing press release, a disagreeing 8-K, a second proxy) –
         and executes every code
         cell of notebook 02 offline. Checks counts, the blinding, the workbook and the block.
  fresh  notebook 02 on an empty Drive (notebook 01 never ran): it must finish and say so.
Every scenario must finish with exit code 0.
"""
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRATCH = Path(sys.argv[1])
REPO = Path(sys.argv[2]).resolve()
SCENARIO = sys.argv[3] if len(sys.argv) > 3 else "full"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from aem_research import takeprivate as tp  # noqa: E402
from aem_research.data import fsds, fts, sec  # noqa: E402
from aem_research.io_utils import load_table, save_table  # noqa: E402
from test_fts import FakeFTS, _hit  # noqa: E402
from test_wsib import PAGE  # noqa: E402

TODAY = pd.Timestamp(datetime.now(timezone.utc).date())
t0 = time.time()

if SCENARIO == "fresh":
    DRIVE = SCRATCH / "drive"
    shutil.rmtree(DRIVE, ignore_errors=True)
    DRIVE.mkdir(parents=True)
else:
    print("Building notebook 01's synthetic Drive and running notebook 01 on it …")
    r = subprocess.run([sys.executable, "-I", str(REPO / "tools" / "e2e_synthetic.py"), str(SCRATCH / "nb01"),
                        str(REPO), "full"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("notebook 01's end-to-end run failed")
    DRIVE = SCRATCH / "nb01" / "drive"
    sec_dir = DRIVE / "aem-research" / "data" / "sec"
    A_DIR = DRIVE / "aem-research" / "cache" / "paper_a"

    # ---- targets: merged names that file XBRL, announced 2012-2024
    merger = pd.concat([pd.read_pickle(p) for p in sorted(sec_dir.glob("merger_*Q*.pkl"))], ignore_index=True)
    subs, _facts, _st = fsds.load(sec_dir, 2011, TODAY.year, offline=True, log=None)
    filers = set(subs["cik"].astype(int))
    m = merger[merger["cik"].isin(filers) & merger["date_filed"].dt.year.between(2012, 2023)]
    m = m.sort_values("date_filed").drop_duplicates("cik").head(8).reset_index(drop=True)
    assert len(m) >= 6, f"too few synthetic merger targets: {len(m)}"
    names = ["Acme Widgets", "Brightline Software", "Cobalt Logistics", "Delta Dental Labs", "Evergreen Foods",
             "Falcon Aerostructures", "Granite Specialty Chemicals", "Harbor Medical Devices"]
    sponsors = ["Thoma Bravo", "Vista Equity", "KKR", "Apollo", "Blackstone", "Silver Lake", "Bain Capital", "TPG"]
    m["name"] = [f"{n.upper()} INC" for n in names[:len(m)]]

    # ---- realistic target names in the cached XBRL submissions (the synthetic "S012 CORP" is too generic)
    rename = dict(zip(m["cik"].astype(int), m["name"]))
    for f in sorted((sec_dir / "fsds").glob("fsds_*_subs.*")):
        base = f.with_suffix("")
        s = load_table(base)
        hit = s["cik"].astype(int).isin(rename)
        if hit.any():
            s.loc[hit, "name"] = s.loc[hit, "cik"].astype(int).map(rename)
            save_table(s, base)
    # a post-buyout 10-K for target 0 (registered notes), filed 400 days after its merger proxy
    c0, d0 = int(m.loc[0, "cik"]), m.loc[0, "date_filed"]
    filed = d0 + pd.Timedelta(days=400)
    lab = f"{filed.year}q{(filed.month - 1) // 3 + 1}"
    base = sec_dir / "fsds" / f"fsds_{lab}_subs"
    s = load_table(base)
    row = s.iloc[[0]].copy()
    row[["adsh", "cik", "name", "form", "fp", "filed", "period"]] = [f"{c0:010d}-99-000001", c0, rename[c0], "10-K", "FY",
                                                                     filed, filed - pd.Timedelta(days=60)]
    save_table(pd.concat([s, row], ignore_index=True), base)

    # ---- Form 15 and IPO index subsets for every quarter notebook 02 reads
    COLS = ["cik", "company", "form", "date_filed", "filename"]
    dereg = pd.DataFrame([(int(m.loc[1, "cik"]), m.loc[1, "name"], "15-12B", m.loc[1, "date_filed"] + pd.Timedelta(days=100),
                           "edgar/data/x/15.txt")], columns=COLS)
    ipo = pd.DataFrame([(777001, f"{names[2].upper()} HOLDINGS, INC.", "S-1", m.loc[2, "date_filed"] + pd.Timedelta(days=1100),
                         "edgar/data/777001/s1.txt"),
                        (777002, "UNRELATED CO INC", "S-1", pd.Timestamp("2019-01-15"), "edgar/data/777002/s1.txt")],
                       columns=COLS)
    for name, tab in (("dereg", dereg), ("ipo", ipo)):
        for y in range(2012, TODAY.year + 1):
            for q in (1, 2, 3, 4):
                if (y, q) >= (TODAY.year, (TODAY.month - 1) // 3 + 1):
                    continue
                g = tab[(tab["date_filed"].dt.year == y) & (tab["date_filed"].dt.quarter == q)]
                g[COLS].astype({"date_filed": "datetime64[ns]"}).reset_index(drop=True).to_pickle(sec_dir / f"{name}_{y}Q{q}.pkl")
    for y in range(2012, TODAY.year + 1):                  # notebook 01 cached 2012 on; make sure 2012 is complete
        for q in (1, 2, 3, 4):
            for name in ("form25", "merger"):
                p = sec_dir / f"{name}_{y}Q{q}.pkl"
                if not p.exists() and (y, q) < (TODAY.year, (TODAY.month - 1) // 3 + 1):
                    pd.DataFrame(columns=COLS).astype({"date_filed": "datetime64[ns]"}).to_pickle(p)

    # ---- full-text search results, cached exactly as the notebook's search would cache them
    hits, docs, info = [], {}, {}
    for i, r_ in m.iterrows():
        cik, when = int(r_["cik"]), r_["date_filed"].strftime("%Y-%m-%d")
        adsh = f"0001193125-{r_['date_filed']:%y}-{900000 + i:06d}"
        if i == 2:                                          # only the equity commitment letter (an exhibit) matched
            hits.append(_hit(adsh, "dex99d1.htm", [5555, cik], "SC 13E3", when, file_type="EX-99.(D)(1)",
                             names=[f"{r_['name']}  (CIK {cik:010d})"]))
            main = "d13e3.htm"
            index_html = (f'<table><tr><td>1</td><td>SCHEDULE 13E-3</td><td><a href="/Archives/edgar/data/{cik}/'
                          f'{adsh.replace("-", "")}/{main}">{main}</a></td><td>SC 13E3</td><td>1</td></tr></table>')
            docs[A_DIR / "docs" / f"{adsh}-index.htm.gz"] = index_html
        else:
            main = f"d{cik}defm14a.htm"
            hits.append(_hit(adsh, main, [cik], "DEFM14A", when, names=[f"{r_['name']}  (CIK {cik:010d})"]))
        if i != 3:                                          # deal 3's document cannot be read
            price, eq, debt, sp = 10 + i, 100 + 10 * i, 200 + 10 * i, sponsors[i]
            html = (f"<p>Each share will be converted into the right to receive ${price}.50 in cash.</p>"
                    f"<p>Parent is an affiliate of {sp}. {sp} delivered an equity commitment letter for equity financing "
                    f"of up to ${eq}.0 million and a limited guarantee of the termination fee of $9 million.</p>"
                    f"<p>The debt commitment letter provides debt financing of ${debt} million. {sp} will own the company. "
                    f"Stockholders receive ${price}.50 per share in cash. {sp} {sp}</p>")
            docs[A_DIR / "docs" / f"{adsh}_{tp._safe(main)}.txt.gz"] = tp.html_to_text(html)
            info[i] = {"cik": cik, "date": r_["date_filed"], "price": price, "sponsor": sp, "html": html, "name": r_["name"]}
    # the check's second documents: deal 4 has a second proxy (filed five days later, same terms) …
    x = info[4]
    adsh_p = "0001193125-99-444444"
    hits.append(_hit(adsh_p, "dprem14a.htm", [x["cik"]], "PREM14A", (x["date"] + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                     names=[f"{x['name']}  (CIK {x['cik']:010d})"]))
    docs[A_DIR / "docs" / f"{adsh_p}_dprem14a.htm.txt.gz"] = tp.html_to_text(x["html"])
    # … deal 0 an agreeing press release, deal 1 an 8-K that states another price
    second_8k = {}
    for i, (price, fname, ftype) in {0: (f"{info[0]['price']}.50", "dex991.htm", "EX-99.1"),
                                     1: ("99.00", "d8k.htm", "8-K")}.items():
        x = info[i]
        adsh8 = f"0001193125-99-55555{i}"
        second_8k[x["cik"]] = _hit(adsh8, fname, [x["cik"]], "8-K", (x["date"] - pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
                                   file_type=ftype)
        docs[A_DIR / "docs" / f"{adsh8}_{fname}.txt.gz"] = tp.html_to_text(
            f"<p>{x['name']} today announced that it has entered into a definitive merger agreement to be acquired by "
            f"{x['sponsor']} in an all-cash transaction valued at approximately $1.2 billion. Stockholders will "
            f"receive ${price} per share in cash.</p>")
    hits.append(_hit("0001193125-16-999999", "x8k.htm", [int(m.loc[0, "cik"])], "8-K", "2016-05-01"))   # not merger-side
    fts.search(tp.QUERY, tp.TARGET_FORMS, "2012-01-01", TODAY, client=FakeFTS(hits), cache_dir=A_DIR / "fts",
               today=TODAY.date(), log=None)
    for path, text in docs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(text.encode()))

    # ---- one listed buyer's 8-K, cached for exactly the queries the notebook will make
    cached = fts.search(tp.QUERY, tp.TARGET_FORMS, "2012-01-01", TODAY, cache_dir=A_DIR / "fts", offline=True,
                        today=TODAY.date(), log=None)[0]
    idx, _ = sec.load_index_subsets(sec_dir, 2012, TODAY.year, {**sec.INDEX_SUBSETS, **tp.INDEX_SUBSETS}, offline=True,
                                    log=None)
    subs2, _f2, _s2 = fsds.load(sec_dir, 2011, TODAY.year, offline=True, log=None)
    delist = load_table(DRIVE / "aem-research" / "cache" / "delistings")
    deals = tp.completion(tp.group_deals(cached, subs2), idx["form25"], idx["dereg"], delist)
    queries = tp.exit_queries(deals, today=TODAY)
    buyer_target = names[4]

    class Exits:
        def get(self, url):
            hit = [_hit("0000999999-25-000001", "d8k.htm", [999999], "8-K", "2025-06-01",
                        names=["BIGCO INDUSTRIES INC  (BIG)  (CIK 0000999999)"])] if buyer_target.replace(" ", "%20") in url else []
            return json.dumps({"hits": {"total": {"value": len(hit), "relation": "eq"}, "hits": hit}}).encode()

    fts.search_many(queries, client=Exits(), cache_path=A_DIR / "fts_exits", log=None)

    class Second:                                           # the target's own 8-Ks, asked for by CIK
        def get(self, url):
            hit = [h for c, h in second_8k.items() if f"ciks={c:010d}" in url]
            return json.dumps({"hits": {"total": {"value": len(hit), "relation": "eq"}, "hits": hit}}).encode()

    second_q = tp.second_queries(deals)
    assert all("ciks" in q for q in second_q) and len(second_q) == len(deals)
    fts.search_many(second_q, client=Second(), cache_path=A_DIR / "fts_second", log=None)
    print(f"Paper A inputs: {len(m)} targets, {len(queries)} exit searches, {len(docs)} cached documents")

    # ---- WSIB: one real-layout report and one corrupt download, saved by hand
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
    wdir = DRIVE / "aem-research" / "data" / "wsib"
    wdir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    y = 560
    for line in PAGE.splitlines():
        c.setFont("Helvetica", 6)
        c.drawString(20, y, line)
        y -= 12
    c.save()
    (wdir / "ir093025.pdf").write_bytes(buf.getvalue())
    (wdir / "ir063025.pdf").write_bytes(b"%PDF-1.4 truncated")
    EXPECT = {"deals": len(m), "docs_missing": 1}

# ---------- execute notebook 02's code cells in order
os.environ.update(AEM_DRIVE_ROOT=str(DRIVE), AEM_REPO_DIR=str(REPO), AEM_SEC_OFFLINE="1")
nb = json.loads((REPO / "notebooks" / "02_paper_a_take_private_sample.ipynb").read_text(encoding="utf-8"))
ns = {"__name__": "__main__"}
for i, cell in enumerate(cc for cc in nb["cells"] if cc["cell_type"] == "code"):
    print(f"\n======== code cell {i} ========")
    exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
print(f"\nALL CELLS OK in {time.time() - t0:.1f}s")

out = DRIVE / "aem-research" / "output"
block = (out / "paper_a_summary.txt").read_text()
print(block)
assert "PAPER A RUN SUMMARY" in block and "ERRORS    none" in block, "unexpected errors"
progress = (out / "paper_a_progress.txt").read_text()
assert "=== run started" in progress and "=== run finished" in progress

if SCENARIO == "fresh":
    assert "universe windows none – run notebook 01" in block
    assert "deals 0" in block or "DEALS" in block
    print("\nFRESH DRIVE: notebook 02 finished and pointed to notebook 01")
    raise SystemExit(0)

D = ns["DEALS"].set_index("deal_id")
assert len(D) == EXPECT["deals"], (len(D), EXPECT)
assert D["completed"].sum() >= EXPECT["deals"] - 1
assert D["offer_price"].notna().sum() == EXPECT["deals"] - EXPECT["docs_missing"]
assert (D["doc_error"].astype(str) == "not available").sum() == EXPECT["docs_missing"]
assert D["sponsor"].fillna("").ne("").sum() == EXPECT["deals"] - EXPECT["docs_missing"]
assert D["ebitda"].notna().all() and (D["ev_ebitda"].notna()).sum() >= EXPECT["deals"] - 2
assert (D["exit_route"] == "re-IPO").sum() == 1 and (D["exit_route"] == "sold to a listed buyer").sum() == 1
assert D["post_buyout_financials"].sum() == 1
assert "main_file" in D.columns and (D["main_file"] == "d13e3.htm").sum() == 1          # found via the filing index
for k in ("reIPO 1", "sold to a listed buyer 1", "post buyout financials 1"):
    assert k in block, k
assert "WARNING   [A8 WSIB] reports not read: 20250630" in block
assert "1 of 1 match" in block or "subtotal checks 2 of 2 match" in block

wb = pd.read_excel(out / "paper_a_sample.xlsx", sheet_name=None, engine="openpyxl")
assert {"Summary", "Deals", "Review", "Exit routes", "Benchmark", "Search hits", "WSIB funds", "WSIB counts",
        "Cross-check", "Check 30"} <= set(wb)

# ---- the check of the terms: second documents for every deal, the seeded reading sample
kinds = D["second_kind"].fillna("").value_counts().to_dict()
assert kinds.get("8-K announcement") == 2 and kinds.get("other merger filing") == 1, kinds
by_cik = D.reset_index().set_index("target_cik")
d0, d1, d4 = (by_cik.loc[info[i]["cik"]] for i in (0, 1, 4))
assert d0["check_price"] == "agree" and d0["check_sponsor"] == "agree" and d0["second_deal_value_usd"] == 1.2e9
assert d0["equity_commitment_usd"] == 100e6 and d0["debt_commitment_usd"] == 200e6    # each its own amount
assert d0["second_file"] == "dex991.htm" and d0["check_value"] in ("agree", "disagree")
assert d1["check_price"] == "disagree" and d1["check_flag"] == "disagree" and d1["second_offer_price"] == 99.0
assert d4["check_price"] == "agree" and d4["check_equity"] == "agree" and d4["check_debt"] == "agree"
assert d4["second_form"] == "PREM14A" and d4["doc_form"] == "DEFM14A"
assert (D["check_flag"] == "unchecked").sum() == EXPECT["deals"] - 3
assert "CHECK     deals 8" in block and "second 8k 2 · second other filing 1" in block and "price 2 of 3 agree" in block
ids = ns["CHECK_IDS"]
assert ids == tp.check_sample(ns["DEALS"], 30, 20261110) and 1 <= len(ids) <= 30
assert set(ids) <= set(D.index[D["completed"] & (D["text_chars"].fillna(0) > 0)])
assert sum(line.startswith("CHECK 0") for line in block.splitlines()) == len(ids)
assert "offer price: 10.5 | «" in block and "|| second: 10.5 | «" in block and "[agree]" in block and "[disagree]" in block
assert len(wb["Check 30"]) == 5 * len(ids) and wb["Cross-check"]["check_flag"].iloc[0] == "disagree"
assert "SEC ID    offline" in block
hidden = set(tp.HIDDEN_UNTIL_LOCK)
assert not hidden & set(wb["Deals"].columns) and not hidden & set(wb["Exit routes"].columns)
assert "net_irr" not in wb["WSIB funds"].columns and "tvpi" not in wb["WSIB funds"].columns
assert len(wb["Benchmark"]) > 10 and wb["Benchmark"]["median_ev_ebitda"].notna().any()
cached_deals = load_table(DRIVE / "aem-research" / "cache" / "paper_a" / "deals")
assert hidden <= set(cached_deals.columns)                                             # kept for after the lock
for word in ("IRR", "net_irr", "tvpi", "holding"):
    assert word not in block, f"{word} must not appear before Paper A's lock"
print(f"\nPAPER A FULL RUN OK in {time.time() - t0:.1f}s")
