import numpy as np
import pandas as pd
import pytest

from aem_research import fundamentals as fu
from aem_research import restatements as rs
from aem_research import secprices as sp
from sec_fixtures import share_facts

QE = pd.date_range("2014-03-31", "2016-12-31", freq="QE")


def _ym(ts) -> int:
    ts = pd.Timestamp(ts)
    return ts.year * 12 + ts.month - 1


def filings(cik, splits=(), issue=None, rounding=None, bs=True):
    return share_facts(cik, QE, splits=splits, issue=issue, rounding=rounding, bs=bs)


def test_restated_history_gives_the_exact_ratio_where_issuance_blurs_the_counts():
    # 1-for-10 reverse split on Aug 20, 2015; the company then doubles its share count by October
    split = pd.Timestamp("2015-08-20")
    issue = lambda d: 1e7 * (2.0 if pd.Timestamp(d) >= pd.Timestamp("2015-09-15") else 1.0)  # noqa: E731
    f = filings(1, splits=[(split, 0.1)], issue=issue)
    ev, spans, info = rs.evidence(f)
    assert len(ev) == 1 and ev.iloc[0]["ratio"] == pytest.approx(0.1)
    e = ev.iloc[0]
    assert e["lo"] < split <= e["hi"] and (e["hi"] - e["lo"]).days <= 125 and e["n_periods"] >= 2
    # the counts read at two dates say 1-for-5: the issuance shrinks the reverse split
    hist = fu.filing_share_history(f)
    idx = sp._dei_index(hist, {1})
    assert sp._dei_ratio(idx, 1, "2015-06-30", "2015-09-30") == pytest.approx(0.2)
    # lookups: the quarter that holds the split has an event; quarters before it show no split
    x = rs.index(ev, spans)
    assert [round(r[2], 3) for r in rs.events_between(x, 1, "2015-06-30", "2015-09-30")] == [0.1]
    assert rs.no_split(x, 1, "2014-12-31", "2015-03-31") and not rs.no_split(x, 1, "2015-06-30", "2015-09-30")
    assert info["events"] == 1 and info["obs_split"] >= 3


def test_no_split_means_no_events_and_spans_that_cover_the_quarters():
    ev, spans, info = rs.evidence(filings(2))
    assert ev.empty and info["obs_split"] == 0 and info["obs_same"] > 10
    x = rs.index(ev, spans)
    assert rs.no_split(x, 2, "2015-03-31", "2015-06-30") and rs.no_split(x, 2, "2015-12-31", "2016-03-31")
    assert not rs.no_split(x, 3, "2015-03-31", "2015-06-30")                  # unknown company: no evidence


def test_stock_dividends_and_rounded_values_still_show_their_ratio():
    # values in thousands: a 1-for-1000 split leaves 123,400 shares, reported as 123,000 (0.3% off)
    big = lambda d: 1.234e8 * (1.0 + 0.01 * (pd.Timestamp(d).year - 2013))       # noqa: E731
    f = pd.concat([filings(4, splits=[(pd.Timestamp("2015-05-05"), 1.25)], rounding=1000),
                   filings(5, splits=[(pd.Timestamp("2015-11-10"), 0.001)], issue=big, rounding=1000)],
                  ignore_index=True)
    ev, _spans, _info = rs.evidence(f)
    got = dict(zip(ev["cik"], ev["ratio"]))
    assert got[4] == pytest.approx(1.25) and got[5] == pytest.approx(0.001) and len(ev) == 2
    # a value rounded harder than ROUND_CAP allows (10,400 shares shown as 10,000) is no evidence
    tiny = filings(8, splits=[(pd.Timestamp("2015-11-10"), 0.001)], issue=lambda d: 1.04e7, rounding=1000)
    assert rs.evidence(tiny)[0].empty


def test_corrections_and_amendments_say_nothing():
    f = filings(6)
    # a later filing corrects one period's count by 7%: neither a split nor no split
    later = f["filed"] > pd.Timestamp("2015-06-01")
    hit = later & (f["ym"] == _ym("2014-06-30")) & (f["qtrs"] == 1)
    f.loc[hit, "value"] *= 1.07
    o = rs.observations(f)
    assert set(o.loc[o["kind"] != "same", "kind"]) == {"other"} and rs.evidence(f)[0].empty
    # an amendment re-filing old (pre-split) numbers after a split is left out when the forms are known
    g = filings(7, splits=[(pd.Timestamp("2015-08-20"), 0.5)])
    amend = g[g["filed"] == g["filed"].min()].assign(filed=pd.Timestamp("2016-02-01"), adsh="0000000007-16-999999")
    g2 = pd.concat([g, amend], ignore_index=True)
    subs = pd.DataFrame({"adsh": g2["adsh"].unique()}).assign(
        form=lambda d: np.where(d["adsh"].eq("0000000007-16-999999"), "10-Q/A", "10-Q"))
    clean, _s, _i = rs.evidence(g2, subs)
    assert len(clean) == 1 and clean.iloc[0]["ratio"] == pytest.approx(0.5)
    split = pd.Timestamp("2015-08-20")

    def across(o):                                  # 'no split' observations that span the split date
        return int(((o["kind"] == "same") & (o["lo"] < split) & (o["hi"] > split)).sum())
    assert across(rs.observations(g2, subs)) == 0
    assert across(rs.observations(g2)) > 0          # the amendment's old numbers would deny the split
