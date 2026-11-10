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


# ---- the check of the terms: a second filing for every deal, a seeded sample to read

PRESS = ("<p>Acme Widgets today announced that it has entered into a definitive merger agreement to be acquired by "
         "Thoma Bravo in an all-cash transaction valued at approximately $2.1 billion. Under the terms of the "
         "agreement, stockholders will receive $27.50 per share in cash.</p>")


def _check_deals():
    return pd.DataFrame({
        "deal_id": ["100-20200301", "200-20210601", "300-20220115"], "target_cik": [100, 200, 300],
        "target_name": ["ACME WIDGETS INC", "BRIGHTLINE INC", "COBALT INC"],
        "doc_adsh": ["A-1", "B-1", "C-1"], "doc_form": ["DEFM14A", "DEFM14A", "SC 14D9"],
        "doc_url": ["u1", "u2", "u3"], "first_filed": pd.to_datetime(["2020-03-01", "2021-06-01", "2022-01-15"]),
        "last_filed": pd.to_datetime(["2020-04-01", "2021-06-20", "2022-01-15"]), "completed": [True, True, False],
        "offer_price": [27.5, 12.0, 8.0], "price_snippet": ["right to receive $27.50 in cash", "$12.00 per share", "x"],
        "equity_commitment_usd": [1.25e9, 3e8, np.nan], "equity_snippet": ["e1", "e2", ""],
        "debt_commitment_usd": [9e8, 2e8, np.nan], "debt_snippet": ["d1", "d2", ""],
        "sponsor": ["Thoma Bravo", "KKR", ""], "sponsor_snippet": ["an affiliate of Thoma Bravo", "KKR", ""],
        "text_chars": [5000, 4000, 0], "equity_value": [2.0e9, 5e8, np.nan], "enterprise_value": [2.6e9, 6e8, np.nan]})


def test_deal_value_and_every_sponsor_are_read_from_an_announcement():
    text = tp.html_to_text(PRESS)
    v = tp.extract_deal_value(text)
    assert v["deal_value_usd"] == 2.1e9 and v["deal_value_kind"] == "transaction" and "valued at" in v["value_snippet"]
    assert tp.extract_deal_value("an enterprise value of approximately $850 million")["deal_value_kind"] == "enterprise"
    assert np.isnan(tp.extract_deal_value("a termination fee of $20 million")["deal_value_usd"])
    assert tp.sponsors_named("KKR and Thoma Bravo; Thoma Bravo leads") == ["Thoma Bravo", "KKR"]
    t = tp.extract_terms(text)
    assert t["offer_price"] == 27.5 and t["sponsor"] == "Thoma Bravo" and "Thoma Bravo" in t["sponsor_snippet"]
    assert tp.extract_terms("holders receive $31.25 in cash, without interest, per share")["offer_price"] == 31.25
    # a larger debt amount right after the equity sentence stays the debt's
    both = tp.extract_terms("Sponsor delivered an equity commitment letter for equity financing of up to $170.0 million. "
                            "The debt commitment letter provides debt financing of $270 million.")
    assert both["equity_commitment_usd"] == 170e6 and both["debt_commitment_usd"] == 270e6


def test_second_queries_ask_for_the_targets_own_8k_around_the_announcement():
    q = tp.second_queries(_check_deals())
    assert len(q) == 3 and q[0] == {"key": "100-20200301", "q": tp.SECOND_QUERY, "forms": ["8-K"], "ciks": 100,
                                    "start": "2019-10-03", "end": "2020-03-31"}


def test_the_second_document_is_an_8k_first_then_another_merger_filing_never_the_same_filing():
    deals = _check_deals()
    h8 = pd.DataFrame({"key": ["100-20200301"] * 3 + ["200-20210601"], "adsh": ["K-2", "K-1", "K-1", "B-1"],
                       "file_name": ["late.htm", "d8k.htm", "ex991.htm", "same.htm"],
                       "form": ["8-K"] * 4, "root_form": ["8-K"] * 4, "file_type": ["8-K", "8-K", "EX-99.1", "8-K"],
                       "file_date": pd.to_datetime(["2020-02-20", "2020-01-10", "2020-01-10", "2021-05-01"])})
    hits = pd.DataFrame({"adsh": ["A-1", "A-2", "B-2", "B-3", "B-4"], "ciks": ["100", "100", "200;999", "200", "200"],
                         "file_name": ["defm.htm", "prem.htm", "ex_d1.htm", "defm_a.htm", "sc13e3.htm"],
                         "form": ["DEFM14A", "PREM14A", "SC 13E3", "DEFM14A/A", "SC 13E3"],
                         "root_form": ["DEFM14A", "PREM14A", "SC 13E3", "DEFM14A", "SC 13E3"],
                         "file_type": ["DEFM14A", "PREM14A", "EX-99.(D)(1)", "DEFM14A/A", "SC 13E3"],
                         "file_date": pd.to_datetime(["2020-03-01", "2020-03-01", "2021-06-01", "2021-06-10", "2021-06-12"])})
    p = tp.pick_second(deals, h8, hits).set_index("deal_id")
    assert p.loc["100-20200301", ["second_adsh", "second_file", "second_kind"]].tolist() == \
        ["K-1", "ex991.htm", "8-K announcement"]                       # earliest 8-K, its press release first
    # deal B: its only 8-K hit is the filing the terms came from, so another form's main document is taken
    assert p.loc["200-20210601", ["second_adsh", "second_form", "second_kind"]].tolist() == \
        ["B-4", "SC 13E3", "other merger filing"]
    assert "300-20220115" not in p.index and len(tp.pick_second(deals, None, None)) == 0


def test_second_documents_are_read_cached_and_compared_term_by_term(tmp_path):
    deals = _check_deals()
    picks = pd.DataFrame({"deal_id": ["100-20200301", "200-20210601"], "second_adsh": ["K-1", "B-4"],
                          "second_file": ["ex991.htm", "sc13e3.htm"], "second_form": ["8-K", "SC 13E3"],
                          "second_kind": ["8-K announcement", "other merger filing"],
                          "second_filed": pd.to_datetime(["2020-01-10", "2021-06-12"])})
    other = ("<p>Holders receive $12.60 per share in cash. Apollo delivered an equity commitment letter for equity "
             "financing of $300 million. The debt commitment letter provides debt financing of $250 million.</p>")

    class Client:
        calls = 0

        def get(self, url):
            Client.calls += 1
            return (PRESS if "ex991" in url else other).encode()

    second, st = tp.read_second(deals, picks, client=Client(), cache_dir=tmp_path)
    assert st["read"] == 2 and Client.calls == 2 and second["second_url"].str.contains("/Archives/edgar/data/").all()
    again, st2 = tp.read_second(deals, picks, client=None, cache_dir=tmp_path, offline=True)
    assert st2["cached"] == 2 and again["second_offer_price"].tolist() == [27.5, 12.6]
    d = tp.cross_check(deals.merge(second, on="deal_id", how="left")).set_index("deal_id")
    a, b, c = d.loc["100-20200301"], d.loc["200-20210601"], d.loc["300-20220115"]
    # the press release: price and sponsor agree, $2.1bn is within 15% of the $2.0bn equity value
    assert (a["check_price"], a["check_sponsor"], a["check_value"], a["check_flag"]) == ("agree",) * 4
    assert a["check_equity"] == "one source"                 # an announcement is not asked for commitments
    # the other filing: 12.60 against 12.00 is outside 0.5%, another sponsor, equity agrees, debt does not
    assert (b["check_price"], b["check_sponsor"], b["check_equity"], b["check_debt"]) == \
        ("disagree", "disagree", "agree", "disagree") and b["check_flag"] == "disagree"
    assert c["check_flag"] == "unchecked" and c["check_price"] == "one source" and c["check_sponsor"] == "none"
    s = tp.check_summary(d.reset_index(), ["100-20200301"], 7)
    assert s["second_8k"] == 1 and s["second_other_filing"] == 1 and s["second_none"] == 1
    assert s["price"] == "1 of 2 agree" and s["deals_flagged"] == 1 and s["reading_sample"] == "1 deals, seed 7"
    # without a second document nothing breaks and nothing is confirmed
    empty = tp.cross_check(deals.iloc[0:0])                  # no deals at all (notebook 01 never ran)
    assert len(empty) == 0 and set(tp.CHECK_COLUMNS) <= set(empty.columns) and tp.check_summary(empty, [], 1) == {}
    assert len(tp.read_second(deals.iloc[0:0], tp.pick_second(deals.iloc[0:0], None, None), cache_dir=tmp_path / "e")[0]) == 0
    bare = tp.cross_check(deals)
    assert bare["check_flag"].eq("unchecked").all() and tp.check_summary(bare, [], 1)["second_none"] == 3


def test_the_reading_sample_is_seeded_stable_and_shown_with_its_passages(tmp_path):
    deals = tp.cross_check(_check_deals())
    ids = tp.check_sample(deals, n=30, seed=20261110)
    assert sorted(ids) == ["100-20200301", "200-20210601"]           # completed deals with a read document
    assert ids == tp.check_sample(deals.iloc[::-1], n=30, seed=20261110)      # the order of the table does not matter
    assert tp.check_sample(deals, n=1, seed=20261110) == ids[:1]     # a larger sample keeps the smaller one
    many = pd.concat([deals.assign(deal_id=[f"{k}-{i}" for i in range(3)]) for k in range(40)], ignore_index=True)
    assert tp.check_sample(many, 30, 1) != tp.check_sample(many, 30, 2) and len(tp.check_sample(many, 30, 1)) == 30
    assert tp.check_sample(deals.drop(columns="text_chars"), 30, 1) == [] and tp.check_sample(None, 30, 1) == []
    sheet = tp.check_sheet(deals, ids)
    assert len(sheet) == 10 and set(sheet["term"]) == {"offer price", "equity commitment", "debt commitment",
                                                       "sponsor", "deal value"}
    row = sheet[(sheet["deal_id"] == "100-20200301") & (sheet["term"] == "offer price")].iloc[0]
    assert row["extracted"] == "27.5" and row["passage"] == "right to receive $27.50 in cash" and row["source_url"] == "u1"
    lines = tp.check_lines(deals, ids)
    assert sum(x.startswith("CHECK 0") for x in lines) == 2 and any("second source: none found" in x for x in lines)
    assert any(x.startswith("  equity commitment: $1.25bn | «e1» [one source]") for x in lines)
    assert any("deal value: equity $2.00bn, enterprise $2.60bn" in x for x in lines)
    assert tp.check_lines(deals, []) == [] and len(tp.check_sheet(deals, [])) == 0
