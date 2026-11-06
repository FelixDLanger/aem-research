import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from aem_research import takeprivate as tp
from aem_research.data import fsds
from sec_fixtures import fsds_zip, num_row, sub_row

PROXY = """<html><head><style>p {color: red}</style><script>var x = "$999.00 per share in cash";</script></head>
<body><p>SUMMARY TERM SHEET</p>
<p>At the effective time, each share of common stock will be converted into the right to receive $27.50 in cash,
without interest (the &ldquo;Merger Consideration&rdquo;).</p>
<p>Parent is controlled by Thoma Bravo Fund XIV, L.P., an affiliate of Thoma Bravo.</p>
<p>Financing of the Merger. Parent has delivered an equity commitment letter from Thoma Bravo Fund XIV, L.P.,
pursuant to which it has committed to provide equity financing of up to $1,250.0 million. Thoma Bravo has also
delivered a limited guarantee in favor of the Company in respect of the Parent termination fee of $95 million.</p>
<p>Parent has obtained a debt commitment letter pursuant to which the lenders have committed to provide debt
financing consisting of a $900 million senior secured term loan facility and a $150 million revolving credit
facility.</p>
<p>The Merger Consideration of $27.50 represents a premium of 35%. Thoma Bravo&#8217;s funds and Thoma Bravo
partners will hold the company. Stockholders will receive $27.50 per share in cash.</p>
<p>An earlier unsolicited proposal of $24.00 per share in cash was rejected.</p></body></html>"""


def test_html_to_text_drops_scripts_and_decodes_entities():
    text = tp.html_to_text(PROXY.encode())
    assert "$999.00" not in text and "color: red" not in text
    assert "“Merger Consideration”" in text and "Thoma Bravo's funds" in text


def test_terms_pick_the_offer_price_commitments_and_sponsor_with_their_sentences():
    t = tp.extract_terms(tp.html_to_text(PROXY))
    assert t["offer_price"] == 27.50 and t["price_mentions"] >= 4
    assert "right to receive $27.50 in cash" in t["price_snippet"]
    assert t["equity_commitment_usd"] == 1_250_000_000                 # not the $95 million termination fee
    assert "equity commitment letter" in t["equity_snippet"]
    assert t["debt_commitment_usd"] == 900_000_000 and "term loan" in t["debt_snippet"]
    assert t["sponsor"] == "Thoma Bravo" and t["sponsor_mentions"] >= 4
    empty = tp.extract_terms("A strategic merger paid in stock.")
    assert np.isnan(empty["offer_price"]) and empty["sponsor"] == "" and np.isnan(empty["equity_commitment_usd"])


INDEX_PAGE = """<table class="tableFile" summary="Document Format Files">
<tr><th scope="col">Seq</th><th scope="col">Description</th><th scope="col">Document</th><th scope="col">Type</th><th scope="col">Size</th></tr>
<tr><td scope="row">1</td><td scope="row">DEFINITIVE PROXY</td><td scope="row"><a href="/Archives/edgar/data/1001/000119312515000001/d1defm14a.htm">d1defm14a.htm</a></td><td scope="row">DEFM14A</td><td scope="row">2345678</td></tr>
<tr><td scope="row">2</td><td scope="row">EQUITY COMMITMENT LETTER</td><td scope="row"><a href="/Archives/edgar/data/1001/000119312515000001/dex99d1.htm">dex99d1.htm</a></td><td scope="row">EX-99.(D)(1)</td><td scope="row">45678</td></tr>
</table>"""


def test_filing_index_lists_documents_and_urls_are_built_from_accession():
    docs = tp.filing_documents(INDEX_PAGE)
    assert list(docs["type"]) == ["DEFM14A", "EX-99.(D)(1)"] and docs["file_name"].iloc[0] == "d1defm14a.htm"
    assert tp.document_url(1001, "0001193125-15-000001", "d1defm14a.htm") == \
        "https://www.sec.gov/Archives/edgar/data/1001/000119312515000001/d1defm14a.htm"
    assert tp.index_url(1001, "0001193125-15-000001").endswith("/000119312515000001/0001193125-15-000001-index.htm")


def _fsds(rows_subs, rows_nums):
    with zipfile.ZipFile(io.BytesIO(fsds_zip(rows_subs, rows_nums))) as zf:
        ex = fsds.extract_quarter(zf)
    return ex["subs"], ex["facts"]


def acme_filings():
    """ACME (CIK 1001): 10-K for FY2014 filed Mar 2015; post-buyout 10-K filed in 2017 (registered notes)."""
    k14, k16 = "0001001-15-000001", "0001001-17-000001"
    subs = [sub_row(k14, 1001, "10-K", "20141231", "FY", "20150302", name="ACME WIDGETS INC", sic=3571),
            sub_row(k16, 1001, "10-K", "20161231", "FY", "20170310", name="ACME WIDGETS INC", sic=3571),
            sub_row("0002002-15-000001", 2002, "10-K", "20141231", "FY", "20150301", name="BETA BANCORP", sic=6022)]
    nums = []
    for tag, v in (("Revenues", 500e6), ("OperatingIncomeLoss", 60e6), ("DepreciationDepletionAndAmortization", 40e6),
                   ("NetCashProvidedByUsedInOperatingActivities", 80e6),
                   ("PaymentsToAcquirePropertyPlantAndEquipment", 20e6)):
        nums.append(num_row(k14, tag, "20141231", 4, v))
        nums.append(num_row(k14, tag, "20131231", 4, v * 0.9))
    nums += [num_row(k14, "CashAndCashEquivalentsAtCarryingValue", "20141231", 0, 50e6),
             num_row(k14, "LongTermDebt", "20141231", 0, 150e6),
             num_row(k14, "EntityCommonStockSharesOutstanding", "20150228", 0, 40e6, "shares", version="dei/2014"),
             num_row(k16, "Revenues", "20161231", 4, 600e6)]
    return _fsds(subs, nums)


def _hits():
    rows = [
        # ACME: preliminary and definitive proxy, plus a 13E-3 co-filed by the sponsor's shell (CIK 5555)
        ("0001193125-15-000010", "d1prem14a.htm", "1001", "PREM14A", "PREM14A", "2015-05-01"),
        ("0001193125-15-000020", "d1defm14a.htm", "1001", "DEFM14A", "DEFM14A", "2015-06-01"),
        ("0001193125-15-000021", "dex99d1.htm", "5555;1001", "SC 13E3", "EX-99.(D)(1)", "2015-06-02"),
        # BETA: a non-XBRL target in our data (no FSDS filings before the hit)
        ("0001193125-16-000030", "beta14d9.htm", "3003", "SC 14D9", "SC 14D9", "2016-02-01"),
        # an 8-K is not a target-side merger filing
        ("0001193125-16-000040", "d8k.htm", "1001", "8-K", "8-K", "2016-03-01"),
    ]
    df = pd.DataFrame(rows, columns=["adsh", "file_name", "ciks", "form", "file_type", "file_date"])
    df["root_form"], df["names"], df["sics"], df["query"] = df["form"], "ACME WIDGETS INC  (ACME)  (CIK 0000001001)", "", tp.QUERY
    df.loc[3, "names"] = "BETA SOFTWARE CORP  (BETA)  (CIK 0000003003)"
    df["file_date"] = pd.to_datetime(df["file_date"]).astype("datetime64[ns]")
    return df


def test_deals_group_by_target_and_pick_the_definitive_main_document():
    subs, _facts = acme_filings()
    deals = tp.group_deals(_hits(), subs)
    assert len(deals) == 2
    acme = deals.set_index("target_cik").loc[1001]
    assert acme["first_filed"] == pd.Timestamp("2015-05-01") and acme["n_filings"] == 3
    assert acme["doc_form"] == "DEFM14A" and acme["doc_file"] == "d1defm14a.htm" and bool(acme["doc_is_main"])
    assert acme["target_name"] == "ACME WIDGETS INC" and acme["fsds_filer"] and acme["announce_year"] == 2015
    beta = deals.set_index("target_cik").loc[3003]
    assert not beta["fsds_filer"] and beta["target_name"] == "BETA SOFTWARE CORP"
    later = _hits().iloc[[1]].assign(file_date=pd.Timestamp("2019-01-10"), adsh="0001193125-19-000001")
    two = tp.group_deals(pd.concat([_hits(), later], ignore_index=True), subs)
    assert (two["target_cik"] == 1001).sum() == 2                          # a second deal after an 18-month gap


def test_completion_from_form25_form15_or_a_merger_delisting():
    subs, _f = acme_filings()
    deals = tp.group_deals(_hits(), subs)
    f25 = pd.DataFrame({"cik": [1001, 1001], "company": "ACME", "form": "25-NSE",
                        "date_filed": pd.to_datetime(["2015-07-20", "2012-01-01"]), "filename": ""})
    dereg = pd.DataFrame({"cik": [3003], "company": "BETA", "form": "15-12B", "date_filed": pd.to_datetime(["2016-04-10"]),
                          "filename": ""})
    delist = pd.DataFrame({"ticker": ["ACME"], "delist_date": pd.to_datetime(["2015-07-30"]),
                           "category": ["merger"], "cik": [1001]})
    c = tp.completion(deals, f25, dereg, delist).set_index("target_cik")
    assert c.loc[1001, "completion_date"] == pd.Timestamp("2015-07-20")
    assert c.loc[1001, "completion_evidence"] == "form25, delisting"
    assert c.loc[3003, "completion_date"] == pd.Timestamp("2016-04-10") and bool(c.loc[3003, "completed"])
    none = tp.completion(deals)
    assert not none["completed"].any()


def test_entry_financials_ebitda_and_enterprise_value_are_point_in_time():
    subs, facts = acme_filings()
    deals = tp.group_deals(_hits(), subs)
    fin = tp.entry_financials(deals, facts, subs)
    a = fin.set_index("deal_id").loc["1001-20150501"]
    assert a["fin_filed"] == pd.Timestamp("2015-03-02") and a["revenue"] == 500e6       # not the 2017 10-K
    assert a["ebitda"] == 100e6 and a["ebitda_source"] == "operating income + D&A" and a["shares_dei"] == 40e6
    assert a["net_cash"] == -100e6
    d = deals.merge(fin, on="deal_id", how="left").assign(offer_price=27.50, equity_commitment_usd=1.25e9,
                                                         debt_commitment_usd=0.9e9)
    v = tp.entry_values(d).set_index("deal_id").loc["1001-20150501"]
    assert v["equity_value"] == 1.1e9 and v["enterprise_value"] == 1.2e9
    assert v["ev_ebitda"] == pytest.approx(12.0) and v["ev_revenue"] == pytest.approx(2.4)
    assert v["leverage_x"] == pytest.approx(9.0) and v["equity_share"] == pytest.approx(1.25 / 1.2)


def test_exits_still_files_reipo_and_sale_are_detected_without_values():
    subs, _f = acme_filings()
    deals = tp.completion(tp.group_deals(_hits(), subs),
                          pd.DataFrame({"cik": [1001, 3003], "company": "", "form": "25-NSE",
                                        "date_filed": pd.to_datetime(["2015-07-20", "2016-04-01"]), "filename": ""}))
    ipo = pd.DataFrame({"cik": [7777, 8888], "company": ["ACME WIDGETS HOLDINGS, INC.", "ACME"],
                        "form": ["S-1", "S-1"], "date_filed": pd.to_datetime(["2019-05-01", "2019-06-01"]),
                        "filename": ""})
    q = tp.exit_queries(deals, today=pd.Timestamp("2026-11-06"))
    assert [x["q"] for x in q] == ['"Acme Widgets" "completion of acquisition"',
                                   '"Beta Software" "completion of acquisition"']
    assert q[0]["start"] == "2016-01-16" and q[0]["forms"] == ["8-K"]
    hits = pd.DataFrame({"key": ["3003-20160201", "3003-20160201"], "ciks": ["9999", "3003"],
                         "names": ["BIGCO INC  (BIG)  (CIK 0000009999)", "BETA"], "adsh": ["a", "b"],
                         "file_date": pd.to_datetime(["2021-03-01", "2020-01-01"])})
    r = tp.exit_routes(deals, subs, ipo, hits).set_index("target_cik")
    assert r.loc[1001, "exit_route"] == "re-IPO" and r.loc[1001, "reipo_cik"] == 7777      # "ACME" alone is too generic
    assert r.loc[1001, "still_files_from"] == pd.Timestamp("2017-03-10") and bool(r.loc[1001, "post_buyout_financials"])
    assert r.loc[3003, "exit_route"] == "sold to a listed buyer" and r.loc[3003, "buyer_name"] == "BIGCO INC"
    assert r.loc[3003, "sale_date"] == pd.Timestamp("2021-03-01")                           # the target's own 8-K ignored
    s = tp.summary(r.reset_index())
    assert s["deals"] == 2 and s["completed"] == 2 and s["exit_reIPO"] == 1 and s["exit_sold_to_a_listed_buyer"] == 1
    assert s["post_buyout_financials"] == 1
    assert not any("value" in k or "return" in k for k in s)                                # counts only


def test_generic_names_are_not_searched():
    assert tp.exit_query_name("ACME INC") is None and tp.exit_query_name("Acme Widgets Inc") == "Acme Widgets"
    assert tp.exit_query_name("Specialty Chemicals Holdings LLC") == "Specialty Chemicals"


def test_public_benchmark_multiples_by_start_date():
    subs, facts = acme_filings()
    feats = pd.DataFrame({"cik": [1001], "start_date": [pd.Timestamp("2015-04-01")], "market_cap": [1.0e9],
                          "in_universe": [True]})
    b = tp.benchmark(feats, facts, subs)
    row = b.iloc[0]
    assert row["firms"] == 1 and row["median_ev_ebitda"] == pytest.approx(11.0)            # (1.0bn + 0.1bn) / 0.1bn
    assert row["median_ev_revenue"] == pytest.approx(2.2) and row["ebitda_positive"] == 1.0


def test_documents_are_read_once_kept_on_drive_and_re_read_after_an_extractor_change(tmp_path, monkeypatch):
    from sec_fixtures import FakeClient
    deals = pd.DataFrame({"deal_id": ["1001-20150501", "3003-20160201", "4004-20170101"],
                          "target_cik": [1001, 3003, 4004],
                          "doc_adsh": ["0001193125-15-000020", "0001193125-16-000030", "0001193125-17-000040"],
                          "doc_file": ["d1defm14a.htm", "dex99.htm", "gone.htm"], "doc_form": ["DEFM14A", "SC 14D9", "DEFM14A"],
                          "doc_is_main": [True, False, True]})
    beta_index = INDEX_PAGE.replace("1001", "3003").replace("DEFM14A", "SC 14D9").replace("d1defm14a", "beta14d9")
    served = {tp.document_url(1001, "0001193125-15-000020", "d1defm14a.htm"): PROXY.encode(),
              tp.index_url(3003, "0001193125-16-000030"): beta_index.encode(),
              tp.document_url(3003, "0001193125-16-000030", "beta14d9.htm"): b"<p>right to receive $12.00 in cash</p>"}
    client = FakeClient(served)
    terms, st = tp.read_documents(deals, client=client, cache_dir=tmp_path, log=None)
    t = terms.set_index("deal_id")
    assert st == {"deals": 3, "cached": 0, "read": 2, "missing": 1, "failed": 0, "index_pages": 1}
    assert t.loc["1001-20150501", "offer_price"] == 27.5 and t.loc["3003-20160201", "offer_price"] == 12.0
    assert t.loc["3003-20160201", "main_file"] == "beta14d9.htm"            # found through the filing index
    assert t.loc["4004-20170101", "doc_error"] == "not available"
    again = FakeClient({})
    _t2, st2 = tp.read_documents(deals, client=again, cache_dir=tmp_path, log=None)
    assert st2["cached"] == 2 and st2["missing"] == 1
    assert all("gone.htm" in u for u in again.requests)                    # only the missing one is retried
    monkeypatch.setattr(tp, "TERMS_VERSION", tp.TERMS_VERSION + 1)
    none = FakeClient({})
    t3, st3 = tp.read_documents(deals, client=none, cache_dir=tmp_path, log=None)
    assert st3["read"] == 2 and not any("d1defm14a" in u for u in none.requests)   # re-read from Drive
    assert t3.set_index("deal_id").loc["1001-20150501", "sponsor"] == "Thoma Bravo"
