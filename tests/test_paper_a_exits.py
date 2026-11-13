import gzip

import numpy as np
import pandas as pd
import pytest

from aem_research import paper_a_exits as pxe
from aem_research import takeprivate as tp
from aem_research.io_utils import load_table


def test_the_value_is_read_only_from_sentences_around_the_targets_name():
    ev = pxe.extract_exit_value
    r = ev("Buyer Inc. today announced that it has completed its acquisition of Acme Widgets, a maker of widgets, in a "
           "transaction valued at approximately $1.25 billion, including debt.", "Acme Widgets")
    assert r["exit_value_usd"] == pytest.approx(1.25e9) and r["exit_value_kind"] == "transaction" and "Acme Widgets" in r["exit_snippet"]
    r = ev("On May 1, the Company completed the acquisition of Acme Widgets for approximately $450 million in cash.", "Acme Widgets")
    assert r["exit_value_usd"] == pytest.approx(4.5e8) and r["exit_value_kind"] == "purchase price"
    r = ev("The aggregate purchase price was $310.5 million, subject to adjustments. Acme Widgets is now a subsidiary.", "ACME WIDGETS")
    assert r["exit_value_usd"] == pytest.approx(3.105e8) and r["exit_name_mentions"] == 1
    # the strongest statement wins over a nearer, weaker one
    r = ev("Buyer acquired Acme Widgets for $900 million in cash, implying an enterprise value of approximately $1.1 billion.", "Acme Widgets")
    assert r["exit_value_usd"] == pytest.approx(1.1e9) and r["exit_value_kind"] == "enterprise"
    # another deal's price far from the name, a document that never names the target, and financing or sales amounts
    far = "Buyer acquired Other Co for $2.0 billion. " + "Filler sentence about operations. " * 40 + "Separately, Acme Widgets was integrated."
    assert np.isnan(ev(far, "Acme Widgets")["exit_value_usd"]) and ev(far, "Acme Widgets")["exit_name_mentions"] == 1
    assert np.isnan(ev("Buyer completed an acquisition valued at $2.0 billion.", "Acme Widgets")["exit_value_usd"])
    assert np.isnan(ev("Buyer acquired Acme Widgets, which had revenue of $300 million, and financed the purchase "
                       "with a term loan of $500 million.", "Acme Widgets")["exit_value_usd"])
    assert np.isnan(ev("Buyer acquired Acme Widgets for $21.50 per share.", "Acme Widgets")["exit_value_usd"])
    assert np.isnan(ev("", "Acme Widgets")["exit_value_usd"]) and np.isnan(ev("text", None)["exit_value_usd"])


def _deals():
    return pd.DataFrame({
        "deal_id": ["d1", "d2", "d3", "d4"], "target_name": ["ACME WIDGETS INC", "BETA SYSTEMS CORP", "GAMMA FOODS INC", "DELTA LABS INC"],
        "target_cik": [11, 12, 13, 14], "first_filed": pd.to_datetime(["2015-03-01", "2018-06-01", "2022-05-01", "2016-01-01"]),
        "completion_date": pd.to_datetime(["2015-07-01", "2018-10-01", "2022-09-01", "2016-04-01"]),
        "exit_route": ["sold to a listed buyer", "sold to a listed buyer", "sold to a listed buyer", "not visible"],
        "sale_adsh": ["0000900001-19-000001", "0000900002-23-000002", "0000900003-25-000003", None],
        "sale_date": pd.to_datetime(["2019-07-01", "2023-10-01", "2025-03-01", None]), "buyer_cik": [900001, 900002, 900003, None],
        "buyer_name": ["Buyer One", "Buyer Two", "Buyer Three", None], "enterprise_value": [5e8, 1e9, 2e9, 3e8],
        "core_sample": [True, True, False, True], "buyer_type": ["sponsor", "sponsor", "strategic or unclear", "sponsor"],
        "post_buyout_financials": [True, False, False, False]})


def _put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode("utf-8")))


def test_exit_filings_are_read_from_the_press_release_first_and_cached(tmp_path):
    d = _deals()
    docs = tmp_path / "docs"
    index = ('<table><tr><td>1</td><td>8-K</td><td><a href="/Archives/x/d8k.htm">d8k.htm</a></td><td>8-K</td><td>1</td></tr>'
             '<tr><td>2</td><td>Press release</td><td><a href="/Archives/x/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>1</td></tr></table>')
    _put(docs / "0000900001-19-000001-index.htm.gz", index)
    _put(docs / "0000900001-19-000001_ex991.htm.txt.gz", "Buyer One completed the acquisition of Acme Widgets for approximately $1.0 billion.")
    _put(docs / "0000900001-19-000001_d8k.htm.txt.gz", "Item 2.01 Completion of Acquisition. Acme Widgets. See Exhibit 99.1.")
    _put(docs / "0000900002-23-000002-index.htm.gz", index)
    _put(docs / "0000900002-23-000002_d8k.htm.txt.gz", "Item 2.01. Buyer Two completed its acquisition of Beta Systems. The total consideration "
                                                     "of approximately $1.5 billion was paid in cash.")
    out, st = pxe.read_exits(d, client=None, cache_dir=tmp_path, offline=True, log=None)
    assert st == {"sales": 3, "cached": 0, "read": 2, "with_value": 2, "no_name_in_text": 1, "missing": 1, "failed": 0}
    o = out.set_index("deal_id")
    assert o.loc["d1", "exit_value_usd"] == pytest.approx(1e9) and o.loc["d1", "exit_file"] == "ex991.htm"
    assert o.loc["d2", "exit_value_usd"] == pytest.approx(1.5e9) and o.loc["d2", "exit_file"] == "d8k.htm"
    assert o.loc["d3", "exit_error"] == "filing index not available" and "d4" not in o.index
    assert o.loc["d1", "exit_url"].endswith("/900001/000090000119000001/ex991.htm")
    again, st2 = pxe.read_exits(d, client=None, cache_dir=tmp_path, offline=True, log=None)
    assert st2["cached"] == 2 and st2["read"] == 0 and len(load_table(tmp_path / "exits")) == 2

    bench = pd.DataFrame({"start_date": pd.to_datetime(["2015-01-01", "2018-04-01", "2019-07-01", "2023-10-01"]),
                          "median_ev_ebitda": [10.0, 12.0, 11.0, 9.0]})
    x = pxe.exit_table(d, out, bench).set_index("deal_id")
    assert list(x.index) == ["d1", "d2"]
    assert x.loc["d1", "value_multiple"] == pytest.approx(2.0) and x.loc["d1", "years_held"] == pytest.approx(4.0, abs=0.01)
    assert x.loc["d1", "annual_rate"] == pytest.approx(2 ** 0.25 - 1, abs=1e-3)
    assert x.loc["d1", "bench_entry"] == 10.0 and x.loc["d1", "bench_exit"] == 11.0 and x.loc["d2", "bench_change"] == pytest.approx(0.75)
    assert x.loc["d1", "exit_regime"] == "sold before 2022" and x.loc["d2", "exit_regime"] == "sold 2022 or later"
    s = pxe.summaries(x.reset_index(), draws=200)
    g = s["groups"].set_index(["by", "group"])
    assert g.loc[("all", "all usable sales"), "n"] == 2 and g.loc[("all", "all usable sales"), "median_value_multiple"] == pytest.approx(1.75)
    assert s["counts"] == {"sales_with_value": 2, "usable": 2, "left_out_implausible_or_short": 0, "entry_value_missing": 0,
                           "with_post_buyout_financials": 1}
    assert s["tracking"] == {"n": 2}                                     # too few for a slope: nothing is printed as an estimate
    lines = pxe.reading_sample(x.reset_index(), out, d, n=5)
    assert len(lines) == 2 and all("→ Buyer" in line and "names " in line and "https://www.sec.gov/Archives" in line for line in lines)
    assert pxe.reading_sample(x.reset_index(), out, d, n=5) == lines     # seeded: the same sample every run


def test_implausible_multiples_are_kept_out_of_the_statistics_and_tracking_needs_ten_sales():
    rng = np.random.default_rng(1)
    n = 40
    bc = np.exp(rng.normal(0, 0.2, n))
    x = pd.DataFrame({"deal_id": [f"d{i}" for i in range(n)], "value_multiple": np.exp(0.3 + 1.2 * np.log(bc) + rng.normal(0, 0.1, n)),
                      "years_held": rng.uniform(1, 6, n), "bench_change": bc, "usable": True, "exit_value_kind": "transaction",
                      "entry_regime": ["entered before 2022"] * 30 + ["entered 2022 or later"] * 10,
                      "exit_regime": "sold 2022 or later", "buyer_type": "sponsor", "core_sample": True, "post_buyout_financials": False})
    x["annual_rate"] = x["value_multiple"] ** (1 / x["years_held"]) - 1
    x.loc[0, ["value_multiple", "usable"]] = [40.0, False]
    s = pxe.summaries(x, draws=300)
    assert s["counts"]["usable"] == 39 and s["counts"]["left_out_implausible_or_short"] == 1
    t = s["tracking"]
    assert t["n"] == 39 and 0.9 < t["slope"] < 1.5 and t["slope_p"] < 0.001 and t["spearman"] > 0.7
    g = s["groups"]
    assert set(g["by"]) == {"all", "entry regime", "exit regime", "exit value kind", "buyer type", "sample"}
    assert g.loc[g["group"].eq("entered 2022 or later"), "n"].iloc[0] == 10 and (g["ci_low"] <= g["median_value_multiple"]).all()
    assert not len(pxe.exit_candidates(pd.DataFrame({"deal_id": ["a"]})))


def test_an_exit_counts_only_where_a_sentence_says_the_target_was_acquired():
    v = pxe.validate_exit
    a = v("Intercontinental Exchange, Inc. today announced that it has completed its acquisition of Ellie Mae from private equity firm "
          "Thoma Bravo. Other text follows.", "Ellie Mae", "Thoma Bravo")
    assert a["tier"] == "A" and a["sponsor_named"] and "acquisition of Ellie Mae" in a["sentence"]
    assert v("The Company completed the acquisition of Acme Widgets.", "Acme Widgets", "KKR")["tier"] == "B"          # sponsor not named
    assert v("Acme Widgets was acquired by Buyer Corp. for $2.0 billion.", "Acme Widgets", None)["tier"] == "B"
    assert v("Following the merger, Acme Widgets became an indirect wholly owned subsidiary of Buyer.", "Acme Widgets", None)["tier"] == "B"
    # what notebook 02 let through
    assert v("CommerceHub, Inc. today announced it has completed its acquisition of ChannelAdvisor Corporation. More.", "Commercehub",
             "GTCR")["reason"] == "the target is the acquirer"
    assert v("The Company completed the acquisition of a portfolio of 11 retail properties anchored by PetSmart and other tenants.",
             "Petsmart", "BC Partners")["tier"] == ""
    assert v("Coherus completed the divestiture of its ophthalmology franchise group of products.", "Franchise Group",
             None)["reason"] == "the name does not appear as a name"
    assert v("Nordstrom is a customer of the Company. The Company completed the acquisition of Maddox Industries.", "Nordstrom", None)["tier"] == ""
    assert v("RingCentral acquired intellectual property from Mitel for $650 million.", "Mitel", None)["tier"] == ""
    assert v("", "Acme", None)["tier"] == "" and v("text", None, None)["tier"] == ""


def test_every_hit_is_read_the_best_one_chosen_and_the_sample_recomputed(tmp_path):
    d = _deals().assign(sponsor=["Thoma Bravo", "KKR", None, "Apollo"], reipo_date=pd.NaT)
    hits = pd.DataFrame({
        "key": ["d1", "d1", "d2", "d3", "d1"],
        "adsh": ["0000900009-16-000001", "0000900001-19-000001", "0000900002-23-000002", "0000900003-25-000003", "0000000011-17-000001"],
        "file_name": ["early.htm", "ex991.htm", "d8k.htm", "d8k.htm", "own.htm"],
        "ciks": ["900009", "900001;11", "900002", "900003", "11"], "names": ["Landlord REIT  (LR)", "Buyer One  (B1)", "Buyer Two", "Buyer Three", "Acme Widgets Inc"],
        "form": "8-K", "file_type": ["8-K", "EX-99.1", "8-K", "8-K", "8-K"],
        "file_date": pd.to_datetime(["2016-03-01", "2019-07-01", "2023-10-01", "2025-03-01", "2017-01-01"])})
    docs = tmp_path / "docs"
    _put(docs / "0000900009-16-000001_early.htm.txt.gz", "The Company completed the acquisition of a property leased to Acme Widgets.")
    _put(docs / "0000900001-19-000001_ex991.htm.txt.gz", "Buyer One completed its acquisition of Acme Widgets from Thoma Bravo in a "
                                                         "transaction valued at approximately $1.0 billion.")
    _put(docs / "0000900002-23-000002_d8k.htm.txt.gz", "Buyer Two completed the acquisition of Beta Systems.")
    h = pxe.exit_hits(d, hits)
    assert len(h) == 4 and "0000000011-17-000001" not in set(h["adsh"])                 # the target's own filing is not a buyer's
    out, st = pxe.validate_exits(d, hits, client=None, cache_dir=tmp_path, offline=True, log=None)
    assert st == {"deals_with_hits": 3, "documents": 4, "cached": 0, "read": 3, "missing": 1, "failed": 0}
    c = pxe.chosen_exits(out).set_index("deal_id")
    assert list(c.index) == ["d1", "d2"] and c.loc["d1", "tier"] == "A" and c.loc["d2", "tier"] == "B"
    assert c.loc["d1", "sale_date_checked"] == pd.Timestamp("2019-07-01")               # not the landlord's filing of 2016
    assert c.loc["d1", "exit_value_usd"] == pytest.approx(1e9)
    _, st2 = pxe.validate_exits(d, hits, client=None, cache_dir=tmp_path, offline=True, log=None)
    assert st2["cached"] == 3 and st2["read"] == 0
    sample = d.assign(group="core", exit_event=[True, True, True, False], exit_years=[0.7, 5.0, 2.5, 10.0], cohort="")
    s2 = pxe.with_checked_exits(sample, c.reset_index())
    assert s2.attrs["as_of"] == pd.Timestamp("2016-04-01") + pd.Timedelta(days=round(10.0 * 365.25))   # read back from the censored row
    assert s2["exit_event"].tolist() == [True, True, False, False] and s2.loc[0, "exit_years"] == pytest.approx(4.0, abs=0.01)
    assert s2.loc[2, "exit_years"] == pytest.approx((s2.attrs["as_of"] - pd.Timestamp("2022-09-01")).days / 365.25)
    only_a = pxe.with_checked_exits(sample, c.reset_index(), tiers=("A",))
    assert only_a["exit_event"].tolist() == [True, False, False, False]
    lines = pxe.exit_lines(d, out, c.reset_index())
    assert lines[0].startswith("SALE 01 tier A · ACME WIDGETS INC (Thoma Bravo) → Buyer One") and "acquisition of Acme Widgets" in lines[0]
    assert lines[1].startswith("SALE 02 tier B") and any(x.startswith("REJECTED 01 GAMMA FOODS INC") for x in lines)


def test_only_exits_accepted_by_hand_count_and_strict_is_a_subset_of_wide():
    sample = pd.DataFrame({
        "deal_id": ["a", "b", "c", "d", "e"],
        "target_name": ["ELLIE MAE INC", "PLY GEM HOLDINGS INC", "NORDSTROM INC", "GARDNER DENVER INC", "DOLE FOOD CO INC"],
        "completion_date": pd.to_datetime(["2019-04-17", "2018-04-12", "2019-01-01", "2013-07-25", "2013-10-31"]),
        "reipo_date": pd.to_datetime([None, None, None, "2017-02-28", "2017-04-24"]), "group": "core", "cohort": "",
        "exit_event": [True, True, True, True, True], "exit_years": [1.4, 0.6, 2.0, 3.6, 3.5]})
    sample.loc[5] = ["f", "QUIET CO", pd.Timestamp("2016-01-01"), pd.NaT, "core", "", False, 10.0]      # censored: gives the as-of date
    chosen = pd.DataFrame({"deal_id": ["a", "b", "c"], "sale_date_checked": pd.to_datetime(["2020-09-04", "2018-11-20", "2021-01-01"]),
                           "buyer_checked": ["ICE", "NCI", "X"], "tier": ["B", "A", "B"]})
    wide, cw = pxe.with_verified_exits(sample, chosen, level="wide")
    assert wide["exit_event"].tolist() == [True, True, False, True, False, False]       # Nordstrom's match and Dole's filing are not exits
    assert cw["sales_in_sample"] == 2 and cw["re_ipos_in_sample"] == 1 and cw["exits_counted"] == 3
    strict, cs = pxe.with_verified_exits(sample, chosen, level="strict")
    assert strict["exit_event"].tolist() == [True, False, False, True, False, False] and cs["exits_counted"] == 2
    assert set(k for k, v in pxe.VERIFIED_SALES.items() if v[0] == "strict") < set(pxe.VERIFIED_SALES)
    assert not set(pxe.REJECTED_REIPOS) & set(pxe.VERIFIED_REIPOS)
