"""Coverage audit: how much of the delisted universe each source prices, and whether a
paid source (EODHD) would add value. Uses price availability only – no returns, no
outcomes – so it is allowed before the pre-registration is locked."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.sec import is_exchange_name, is_spac_name


def base_symbol(ticker: str) -> str:
    """'XYZ~12345' (a SimFin ticker split by SimFinId) -> 'XYZ' for cross-source matching."""
    return str(ticker).split("~", 1)[0]


def spans(prices: pd.DataFrame | None) -> pd.DataFrame:
    """One row per ticker: first_date, last_date, n_obs."""
    cols = ["ticker", "first_date", "last_date", "n_obs"]
    if prices is None or len(prices) == 0:
        return pd.DataFrame({"ticker": pd.Series(dtype=object),
                             "first_date": pd.Series(dtype="datetime64[ns]"),
                             "last_date": pd.Series(dtype="datetime64[ns]"),
                             "n_obs": pd.Series(dtype="int64")})
    g = prices.groupby("ticker", observed=True)["date"]
    out = pd.DataFrame({"first_date": g.min(), "last_date": g.max(), "n_obs": g.size()}).reset_index()
    out["ticker"] = out["ticker"].astype(str)
    return out[cols]


def source_summary(name: str, sp: pd.DataFrame, as_of=None, gap_days: int = 30) -> dict:
    if len(sp) == 0:
        return {"source": name, "tickers": 0}
    as_of = pd.Timestamp(as_of) if as_of is not None else sp["last_date"].max()
    stopped = sp["last_date"] < as_of - pd.Timedelta(days=gap_days)
    return {
        "source": name,
        "tickers": int(len(sp)),
        "first_date": str(sp["first_date"].min().date()),
        "as_of": str(as_of.date()),
        "stopped_trading": int(stopped.sum()),
        "share_stopped": round(float(stopped.mean()), 4),
        "median_history_days": int(sp["n_obs"].median()),
    }


def stopped_by_year(sp: pd.DataFrame, as_of, years, gap_days: int = 30) -> pd.Series:
    as_of = pd.Timestamp(as_of)
    st = sp[sp["last_date"] < as_of - pd.Timedelta(days=gap_days)]
    counts = st["last_date"].dt.year.value_counts()
    return pd.Series({y: int(counts.get(y, 0)) for y in range(int(years[0]), int(years[1]) + 1)})


def _simfin_symbols(simfin_sp: pd.DataFrame) -> set:
    return {base_symbol(t) for t in simfin_sp["ticker"]} if len(simfin_sp) else set()


def stooq_only(simfin_sp: pd.DataFrame, stooq_sp: pd.DataFrame) -> pd.DataFrame:
    """Stooq spans for symbols SimFin does not have at all."""
    if not len(stooq_sp):
        return stooq_sp
    return stooq_sp[~stooq_sp["ticker"].isin(_simfin_symbols(simfin_sp))]


def union_spans(simfin_sp: pd.DataFrame, stooq_sp: pd.DataFrame) -> pd.DataFrame:
    """SimFin spans, plus Stooq spans for symbols SimFin lacks (same rule as the price panel)."""
    parts = [simfin_sp.assign(source="simfin")]
    if len(stooq_sp):
        parts.append(stooq_only(simfin_sp, stooq_sp).assign(source="stooq"))
    return pd.concat(parts, ignore_index=True) if len(simfin_sp) or len(stooq_sp) else simfin_sp


def stopped_overlap(simfin_sp: pd.DataFrame, stooq_sp: pd.DataFrame, as_of, gap_days: int = 30,
                    tol_days: int = 10) -> dict:
    """Stopped-trading names in each source, matched on ticker and last date (±tol_days)."""
    as_of = pd.Timestamp(as_of)
    cut = as_of - pd.Timedelta(days=gap_days)
    a = simfin_sp[simfin_sp["last_date"] < cut][["ticker", "last_date"]]
    a = a.assign(ticker=a["ticker"].map(base_symbol))
    b = stooq_sp[stooq_sp["last_date"] < cut][["ticker", "last_date"]]
    m = a.merge(b, on="ticker", suffixes=("_simfin", "_stooq"))
    both = int(((m["last_date_simfin"] - m["last_date_stooq"]).abs() <= pd.Timedelta(days=tol_days)).sum())
    return {"simfin_stopped": int(len(a)), "stooq_stopped": int(len(b)), "matched_both": both,
            "stooq_only": int(len(b)) - both, "simfin_only": int(len(a)) - both}


def live_overlap(simfin_sp: pd.DataFrame, stooq_sp: pd.DataFrame, ref_date) -> dict:
    """Share of names Stooq lists on ref_date that SimFin also prices (universe coverage)."""
    ref = pd.Timestamp(ref_date)

    def live(sp):
        return {base_symbol(t) for t in sp[(sp["first_date"] <= ref) & (sp["last_date"] >= ref)]["ticker"]}

    q, s = live(stooq_sp), live(simfin_sp)
    return {"ref_date": str(ref.date()), "stooq_live": len(q), "simfin_live": len(s),
            "stooq_live_in_simfin": len(q & s),
            "share_of_stooq_in_simfin": round(len(q & s) / len(q), 4) if q else None}


def adjustment_check(simfin_prices: pd.DataFrame | None, stooq_prices: pd.DataFrame | None,
                     n_tickers: int = 300, seed: int = 0) -> dict:
    """Is Stooq's close raw or adjusted? Compare it with SimFin's close and adjusted close on
    shared (ticker, date) pairs for a random sample of tickers."""
    if simfin_prices is None or stooq_prices is None or not len(simfin_prices) or not len(stooq_prices):
        return {}
    common = sorted(set(map(str, simfin_prices["ticker"].unique()))
                    & set(map(str, stooq_prices["ticker"].unique())))
    if not common:
        return {"pairs": 0}
    rng = np.random.default_rng(seed)
    sample = [str(x) for x in rng.choice(common, size=min(n_tickers, len(common)), replace=False)]
    cols = ["ticker", "date"] + [c for c in ("close", "adj_close") if c in simfin_prices.columns]
    a = simfin_prices.loc[simfin_prices["ticker"].isin(sample), cols].copy()
    b = stooq_prices.loc[stooq_prices["ticker"].isin(sample), ["ticker", "date", "close"]].copy()
    a["ticker"], b["ticker"] = a["ticker"].astype(str), b["ticker"].astype(str)
    m = a.merge(b.rename(columns={"close": "stooq_close"}), on=["ticker", "date"])
    out = {"tickers_sampled": len(sample), "pairs": int(len(m))}
    for col in ("close", "adj_close"):
        if col in m.columns and len(m):
            r = np.abs(np.log(m["stooq_close"].astype(float) / m[col].astype(float)))
            r = r[np.isfinite(r)]
            out[f"within_1pct_of_simfin_{col}"] = round(float((r < 0.01).mean()), 4) if len(r) else None
    return out


def form25_events(form25: pd.DataFrame, forms, years) -> pd.DataFrame:
    """One delisting event per issuer CIK and year; exchanges and SPAC shells flagged."""
    cols = ["cik", "company", "date_filed", "year", "spac"]
    if form25 is None or len(form25) == 0:
        return pd.DataFrame(columns=cols)
    f = form25[form25["form"].isin(list(forms))].copy()
    f = f[~f["company"].map(is_exchange_name)]
    f["year"] = f["date_filed"].dt.year
    f = f[(f["year"] >= int(years[0])) & (f["year"] <= int(years[1]))]
    f = f.sort_values("date_filed").drop_duplicates(["cik", "year"], keep="first")
    f["spac"] = f["company"].map(is_spac_name)
    return f[cols].reset_index(drop=True)


def benchmark_against_form25(events: pd.DataFrame, companies: pd.DataFrame | None,
                             simfin_sp: pd.DataFrame, stooq_sp: pd.DataFrame,
                             window_days: int = 45) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Classify each non-SPAC Form 25 event via the SimFin CIK -> ticker bridge.

    Status per event:
      continued     SimFin prices continue past the filing (exchange move or a non-common
                    security) -> not a terminal delisting, excluded from the denominator
      simfin        SimFin series ends within ±window of the filing
      stooq         SimFin lacks the end, Stooq's series ends within ±window
      missing       mapped to a ticker, but no source shows the series ending
      unmapped      CIK not in SimFin's companies table -> counted as uncovered
    Returns (per-event detail, per-year table) or (None, None) without a CIK bridge.
    """
    if companies is None or "cik" not in companies.columns or "ticker" not in companies.columns:
        return None, None
    if events is None or len(events) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=["year", "events", "simfin", "stooq_adds", "unmapped"])
    ev = events[~events["spac"].astype(bool)].copy()
    bridge = companies.dropna(subset=["cik"]).drop_duplicates("cik")[["cik", "ticker"]]
    ev["cik"] = pd.to_numeric(ev["cik"], errors="coerce").astype("Int64")
    bridge = bridge.assign(cik=bridge["cik"].astype("Int64"))
    ev = ev.merge(bridge, on="cik", how="left")
    sim = simfin_sp.set_index("ticker")["last_date"]
    stq = stooq_sp.set_index("ticker")["last_date"] if len(stooq_sp) else pd.Series(dtype="datetime64[ns]")
    win = pd.Timedelta(days=window_days)

    def classify(row) -> str:
        t = row["ticker"]
        if pd.isna(t):
            return "unmapped"
        filed = row["date_filed"]
        s_last = sim.get(t)
        if s_last is not None and pd.notna(s_last):
            if s_last > filed + win:
                return "continued"
            if s_last >= filed - win:
                return "simfin"
        q_last = stq.get(base_symbol(t)) if len(stq) else None
        if q_last is not None and pd.notna(q_last) and filed - win <= q_last <= filed + win:
            return "stooq"
        return "missing"

    ev["status"] = ev.apply(classify, axis=1) if len(ev) else pd.Series(dtype=str)
    rows = []
    for y, g in ev.groupby("year"):
        terminal = g[g["status"] != "continued"]
        n = len(terminal)
        n_s = int((terminal["status"] == "simfin").sum())
        n_q = int((terminal["status"] == "stooq").sum())
        n_u = int((terminal["status"] == "unmapped").sum())
        rows.append({"year": int(y), "events": n, "simfin": n_s, "stooq_adds": n_q,
                     "unmapped": n_u, "missing": int((terminal["status"] == "missing").sum()),
                     "continued_excluded": int((g["status"] == "continued").sum()),
                     "coverage_union": round((n_s + n_q) / n, 4) if n else None,
                     "coverage_simfin": round(n_s / n, 4) if n else None,
                     "coverage_if_mapped": round((n_s + n_q) / (n - n_u), 4) if n - n_u else None})
    return ev, pd.DataFrame(rows)


def coverage_range(table: pd.DataFrame, stooq_only_stopped: pd.Series) -> tuple[float, float]:
    """(conservative, optimistic) delisting coverage.

    Conservative counts unmapped events as uncovered. Optimistic credits, per year, up to
    as many unmapped events as Stooq has stopped-trading symbols that SimFin lacks – the
    most Stooq could add for issuers SimFin never covered."""
    events = int(table["events"].sum())
    if events == 0:
        return float("nan"), float("nan")
    covered = int((table["simfin"] + table["stooq_adds"]).sum())
    credit = sum(min(int(r.unmapped), int(stooq_only_stopped.get(r.year, 0))) for r in table.itertuples())
    return covered / events, (covered + credit) / events


def count_ratio(events: pd.DataFrame, stopped_union: pd.Series) -> pd.DataFrame:
    """Rough fallback without a CIK bridge: stopped names per year vs non-SPAC Form 25 issuers.
    The Form 25 count includes non-common securities, so this understates coverage."""
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["year", "form25_issuers", "stopped_in_data", "ratio"])
    ev = events[~events["spac"].astype(bool)].groupby("year").size()
    rows = []
    for y, n in ev.items():
        st = int(stopped_union.get(y, 0))
        rows.append({"year": int(y), "form25_issuers": int(n), "stopped_in_data": st,
                     "ratio": round(min(st / n, 1.0), 4) if n else None})
    return pd.DataFrame(rows)


def attrition(sp: pd.DataFrame, start_years, horizon_years: int, as_of, gap_days: int = 30) -> pd.DataFrame:
    """Share of names live at each start date whose series stops within the horizon."""
    as_of = pd.Timestamp(as_of)
    cut = as_of - pd.Timedelta(days=gap_days)
    rows = []
    for y in start_years:
        s = pd.Timestamp(year=int(y), month=1, day=1)
        end = s + pd.DateOffset(years=int(horizon_years))
        if end > as_of:
            continue
        live = sp[(sp["first_date"] <= s + pd.Timedelta(days=7)) & (sp["last_date"] >= s)]
        stopped = int(((live["last_date"] < end) & (live["last_date"] < cut)).sum())
        rows.append({"start_year": int(y), "live": int(len(live)), "stopped_within_h": stopped,
                     "attrition_observed": round(stopped / len(live), 4) if len(live) else None})
    return pd.DataFrame(rows)


def missing_share(n_live: int, n_stopped: int, coverage: float) -> tuple[float, float]:
    """(true attrition, share of the start universe unpriced after delisting).

    Survivors are fully priced; only a share `coverage` of names that later stop are
    captured. True stops D = captured / coverage. The unpriced share of the universe
    is D(1 - coverage) / (survivors + D), which is also the relative width of the
    base-rate bounds."""
    if not 0 < coverage <= 1 or n_live <= 0:
        return float("nan"), float("nan")
    survivors = n_live - n_stopped
    d_true = n_stopped / coverage
    total = survivors + d_true
    return d_true / total, d_true * (1 - coverage) / total


def eodhd_verdict(missing: float | None, coverage: float | None, thresholds: dict,
                  basis: str, example_rate: float = 0.03) -> dict:
    if coverage is None:
        return {"verdict": "undetermined", "basis": basis,
                "reason": "No delisting benchmark yet – set the SEC_USER_AGENT secret and rerun the SEC step."}
    if coverage <= 0.01:
        return {"verdict": "recommended", "coverage": round(float(coverage), 4), "basis": basis,
                "reason": "The free sources price almost none of the names that were delisted – "
                          "base rates would be survivor-only without a source that keeps delisted tickers."}
    if missing is None or not np.isfinite(missing):
        return {"verdict": "undetermined", "coverage": round(float(coverage), 4), "basis": basis,
                "reason": "Coverage is known but attrition could not be measured (no price spans)."}
    lo, hi = thresholds["not_needed_max_missing_share"], thresholds["optional_max_missing_share"]
    verdict = "not needed" if missing <= lo else ("optional" if missing <= hi else "recommended")
    low = example_rate * (1 - missing)
    reason = (f"About {missing:.1%} of the start universe would be unpriced after delisting "
              f"(delisting coverage {coverage:.0%}). A {example_rate:.1%} base rate would carry "
              f"bounds of {low:.2%}–{example_rate:.2%}. Thresholds: not needed ≤ {lo:.0%}, "
              f"optional ≤ {hi:.0%}, recommended above.")
    return {"verdict": verdict, "missing_share": round(float(missing), 4),
            "coverage": round(float(coverage), 4), "basis": basis, "reason": reason}
