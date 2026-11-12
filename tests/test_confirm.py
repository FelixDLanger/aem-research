import numpy as np
import pandas as pd
import pytest

from aem_research import confirm
from aem_research.config import load_config
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg_c():
    return load_config(ROOT / "config" / "confirmation.yaml", ROOT / "config" / "confirmation_LOCK.json")


def _features(n=900, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(["2023-10-01", "2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01", "2025-07-01"])
    rows = []
    for d in dates:
        for i in range(n // len(dates) * 4):
            rows.append({"ticker": f"T{i}", "cik": 1000 + i, "start_date": d})
    f = pd.DataFrame(rows)
    m = len(f)
    f["in_universe"] = True
    for c, p in (("fallen", 0.4), ("fcf_pos", 0.6), ("net_cash_pos", 0.5), ("low_dilution", 0.5), ("s1_proxy", 0.3), ("low_attention", 0.33),
                 ("insider_any", 0.2), ("insider_cluster", 0.08), ("insider_opportunistic", 0.05), ("insider_post_results_cluster", 0.06),
                 ("h2a", 0.05), ("h2b", 0.02)):
        f[c] = rng.random(m) < p
    f["market_cap"], f["net_cash"] = rng.uniform(1e8, 2e9, m), rng.normal(0, 1e8, m)
    return f, rng


def _outcomes(f, rng):
    rows = []
    for r in f.itertuples():
        lift = 0.25 * r.fcf_pos + 0.2 * r.low_dilution + 0.5 * r.fallen * rng.random()
        for h in (1, 2):
            end = float(np.exp(rng.normal(0.05 * h + lift * 0.4, 0.5)))
            top = end * float(np.exp(abs(rng.normal(0, 0.3 + 0.3 * r.fallen))))
            done = r.start_date + pd.DateOffset(years=h) <= pd.Timestamp("2026-07-01")
            rows.append({"ticker": r.ticker, "start_date": r.start_date, "horizon_years": h, "status": "complete" if done else "censored",
                         "end_multiple": end if done else np.nan, "bucket": [0.1, 0.4, 1.0, 2.5, 5.0, 10.0][int(np.searchsorted(
                             [0.2, 0.632, 1.58, 3.54, 7.07], end))] if done else np.nan,
                         "touch_1.5x": float(top >= 1.5) if done else np.nan, "touch_2x": float(top >= 2) if done else np.nan,
                         "touch_3x": float(top >= 3) if done else np.nan})
    return pd.DataFrame(rows)


def test_windows_keep_unresolved_horizons_missing_and_the_tests_follow_the_locked_plan(cfg_c):
    f, rng = _features()
    w = confirm.windows(f, _outcomes(f, rng))
    assert len(w) == len(f) and w.loc[w["start_date"] > "2024-07-01", "hit2_2y"].isna().all()
    assert w.loc[w["start_date"] <= "2024-07-01", "hit2_2y"].notna().all() and w["hit15_1y"].notna().all()
    r = confirm.tests(w, cfg_c, resamples=300)
    two = int((w["start_date"] <= "2024-07-01").sum())
    assert r["C1"]["windows"] == two and r["C2"]["windows"] == two and r["C3"]["windows"] == two
    assert r["C1"]["fcf_pos_coef"] > 0 and r["C1"]["low_dilution_coef"] > 0 and r["C1"]["verdict"] == "confirmed"
    # C1's p is the weaker coefficient after Holm across the two, and Holm runs across the three hypotheses
    assert r["C1"]["p"] >= max(r["C1"]["fcf_pos_p_one_sided"], r["C1"]["low_dilution_p_one_sided"]) - 1e-12
    assert r["C2"]["lr"] > 1 and r["C2"]["verdict"] == "confirmed" and r["C2"]["holm_p"] >= r["C2"]["p_one_sided"]
    assert r["C3"]["verdict"] in ("confirmed", "not confirmed") and r["C3"]["with_signal"] > 0
    assert [x["version"] for x in r["robustness"]][:3] == ["one year · C1 (one-year buckets)", "one year · C2 (touch 1.5x)",
                                                         "one year · C3 (touch 1.5x)"]
    assert r["robustness"][1]["windows"] == len(w) and len(r["by_date"]) == 8
    assert r["by_date"]["resolved_2y"].tolist()[4:] == [0, 0, 0, 0]
    # a signal that never fires is not estimable, and nothing is called confirmed without an estimate
    r0 = confirm.tests(w.assign(insider_post_results_cluster=False), cfg_c, resamples=200)
    assert r0["C3"]["verdict"] == "not estimable" and r0["C2"]["verdict"] == "confirmed"
    # a lift in the wrong direction is never confirmed
    flipped = confirm.tests(w.assign(fallen=~w["fallen"]), cfg_c, resamples=200)
    assert flipped["C2"]["lr"] < 1 and flipped["C2"]["verdict"] == "not confirmed"


def test_reproduction_passes_on_the_same_table_and_fails_on_a_changed_universe_or_signal():
    f, _ = _features(400)
    old = f[f["start_date"] <= "2023-10-01"]
    ok = confirm.reproduction(f, old)
    assert ok["passed"] and ok["universe_agreement"] == 1.0 and ok["weakest_agreement"] == 1.0 and ok["dates_compared"] == 1
    assert ok["rows_rebuilt"] == len(old)                                    # only the dates both hold are compared
    moved = f.copy()
    moved.loc[moved.index[:40], "in_universe"] = False
    bad = confirm.reproduction(moved, old)
    assert not bad["passed"] and bad["universe_agreement"] < confirm.MIN_AGREEMENT
    sig = f.copy()
    sig.loc[sig.index[:30], "low_dilution"] = ~sig.loc[sig.index[:30], "low_dilution"]
    bad = confirm.reproduction(sig, old)
    assert not bad["passed"] and bad["same_low_dilution"] < confirm.MIN_AGREEMENT and bad["same_fcf_pos"] == 1.0
    assert not confirm.reproduction(f, old.assign(in_universe=False))["passed"]     # nothing to compare is not a pass


def test_real_terms_divide_by_the_cpi_of_the_window(cfg_c):
    cpi = pd.Series([100.0, 100.0, 110.0, 121.0], index=pd.to_datetime(["2012-12-01", "2013-12-01", "2018-12-01", "2019-12-01"]))
    w = pd.DataFrame({"start_date": pd.to_datetime(["2014-01-01", "2015-01-01", "2022-01-01"]), "end5": [1.1, 2.42, np.nan],
                      "regime": ["before 2022", "before 2022", "2022 and later"]})
    t = confirm.real_terms(w, cpi).set_index("group")
    assert t.loc["all", "windows"] == 2 and t.loc["all", "mean_real"] == pytest.approx((1.0 + 2.0) / 2)
    assert t.loc["all", "share_real_below_1"] == 0.0 and t.loc["all", "median_cpi_factor"] == pytest.approx(1.155)
    assert list(t.index) == ["all", "before 2022"] and not len(confirm.real_terms(w, pd.Series(dtype=float)))


def test_the_calendar_continues_the_locked_one_to_the_last_confirmation_date(cfg_c):
    class Cfg:
        def get(self, key):
            return {"years": [2013, 2023], "months": [1, 4, 7, 10]}
    d = confirm.all_dates(Cfg(), cfg_c)
    assert d[0] == pd.Timestamp("2013-01-01") and d[-1] == pd.Timestamp("2025-07-01") and len(d) == 51
    assert set(pd.to_datetime(cfg_c.get("dates.one_year"))) <= set(d)
