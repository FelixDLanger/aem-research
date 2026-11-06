import pandas as pd
import pytest

from aem_research.guards import (LookAheadError, SurvivorshipError, assert_point_in_time,
                                 assert_survivorship_free, survivorship_report)


def test_point_in_time_passes_and_fails():
    ok = pd.DataFrame({"start_date": pd.to_datetime(["2015-01-01"] * 2),
                       "available_date": pd.to_datetime(["2014-11-10", "2015-01-01"])})
    assert_point_in_time(ok)
    bad = ok.assign(available_date=pd.to_datetime(["2014-11-10", "2015-02-15"]))
    with pytest.raises(LookAheadError, match="1 feature rows"):
        assert_point_in_time(bad)
    with pytest.raises(LookAheadError, match="lacks"):
        assert_point_in_time(ok.drop(columns="available_date"))
    with pytest.raises(LookAheadError, match="no availability date"):
        assert_point_in_time(ok.assign(available_date=[pd.NaT, pd.Timestamp("2014-01-01")]))


def _spans(n_live, n_stopped):
    first = pd.Timestamp("2005-01-03")
    last = [pd.Timestamp("2026-10-30")] * (n_live - n_stopped) + [pd.Timestamp("2014-06-30")] * n_stopped
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n_live)], "first_date": first, "last_date": last})


def test_survivor_only_source_is_rejected():
    with pytest.raises(SurvivorshipError, match="survivor-only"):
        assert_survivorship_free(_spans(100, 0), ["2012-01-03"], "2026-10-30")


def test_source_with_delisted_names_passes():
    rep = assert_survivorship_free(_spans(100, 25), ["2012-01-03", "2024-01-02"], "2026-10-30")
    assert rep.loc[0, "share_stopped"] == pytest.approx(0.25)
    assert not rep.loc[1, "checked"]          # too recent to judge


def test_report_counts_live_names_only():
    sp = _spans(10, 2)
    sp.loc[0, "first_date"] = pd.Timestamp("2016-01-04")   # not yet listed in 2012
    rep = survivorship_report(sp, ["2012-01-03"], "2026-10-30")
    assert rep.loc[0, "live"] == 9
