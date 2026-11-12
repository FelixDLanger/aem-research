"""Notebook 08, round two: the confirmation of C1 to C3 on new start dates and the leftovers.

The data run's universe-A features are rebuilt from the caches on Drive – nothing is
downloaded, nothing the data run saved is overwritten – for the locked start dates and the
confirmation's new ones in one pass. The rebuilt rows of the locked dates must match the
saved table (`reproduction`): only then is the code that builds the new dates the code that
built the study, and only then are the new dates' outcomes computed.

The steps mirror notebook 01 (Phases A, B, C, F: the vendor panel for the listing evidence,
the SEC data sets, the features of universe A); the SEC price panel and the delistings are
read from the data run's saved tables.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import delistings as dl
from . import estimators as est
from . import features as ft
from . import fundamentals as fu
from . import outcomes as oc
from . import secprices
from .data import eodhd, fsds, ftd, insider, sec, simfin_cache, stooq, thirteenf

HORIZONS = (1, 2)
MULTIPLES = (1.5, 2.0, 3.0)
CHECK_FLAGS = ("fallen", "fcf_pos", "net_cash_pos", "low_dilution", "s1_proxy", "low_attention", "insider_any",
               "insider_cluster", "insider_opportunistic", "insider_post_results_cluster", "h2a", "h2b")
MIN_AGREEMENT = 0.995


def all_dates(cfg, cfg_c) -> list:
    """The locked quarterly calendar, continued to the confirmation's last start date."""
    st = cfg.get("starts")
    last = max(pd.Timestamp(d) for d in cfg_c.get("dates.one_year"))
    return [d for d in ft.formation_dates([int(st["years"][0]), last.year], st.get("months", [1])) if d <= last]


def load_inputs(paths, cfg, log=print) -> dict:
    """Everything notebook 01 loads before its features, from the caches only."""
    y0, this_year = int(cfg.get("starts.years")[0]), datetime.now(timezone.utc).year
    kw = dict(client=None, offline=True, local_dir=paths.sec_dir / "manual", log=log)
    simfin = simfin_cache.load(paths.simfin_dirs)
    stq = stooq.load_or_build(paths.stooq_dir, paths.cache, log=log)
    years = cfg.get("coverage.benchmark_years")
    idx, _ = sec.load_index_subsets(paths.sec_dir, years[0] - 1, years[1], offline=True, log=log)
    subs, facts, st_f = fsds.load(paths.sec_dir, y0 - 2, this_year, **kw)
    ins_tx, ins_bridge, st_i = insider.load(paths.sec_dir, cfg.get("data_sources.insider_history_start_year", y0 - 4),
                                            this_year, **kw)
    f13_breadth, f13_names, st_13 = thirteenf.load(paths.sec_dir, **kw)
    ftd_names, st_ftd = ftd.load(paths.sec_dir, start=cfg.get("data_sources.sec_fails_start", ftd.START), **kw)
    bridges = [b for b in (ins_bridge, ft.instance_bridge(subs) if subs is not None and len(subs) else None)
               if b is not None and len(b)]
    eod_prices = eod_exchange = None
    eod_dir = paths.cache / "eodhd"
    if eodhd.cached(eod_dir):                             # only when the data run had it on Drive
        syms, _ = eodhd.symbol_lists(None, eod_dir, offline=True)
        eod_exchange = eodhd.exchange_map(syms)
        sim_ciks = (simfin.companies["cik"].dropna() if simfin.found and simfin.companies is not None
                    and "cik" in simfin.companies.columns else [])
        want = eodhd.candidates(syms, bridges, subs["cik"].unique(), sim_ciks)
        eod_prices, _ = eodhd.fetch_eod(want["code"], None, eod_dir, offline=True)
    files = {name: f"{s['files']} files, {s['cached']} cached, {s['missing']} missing, {s['bad']} bad"
             for name, s in (("fsds", st_f), ("insider", st_i), ("form13f", st_13), ("fails", st_ftd))}
    return {"simfin": simfin, "stq": stq, "form25": idx["form25"], "merger": idx["merger"], "subs": subs, "facts": facts,
            "ins_tx": ins_tx, "bridges": bridges, "f13_breadth": f13_breadth, "f13_names": f13_names,
            "ftd_names": ftd_names, "eod_prices": eod_prices, "eod_exchange": eod_exchange, "files": files}


def build_features(inp: dict, cfg, sec_panel: pd.DataFrame, dates, *, k: int, attention: str,
                   purchases_only: bool = False, shared: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Universe A's feature table for `dates`, as notebook 01's Phase F builds it, with the
    liquidity threshold `k` and the attention signal of the data run. `shared` carries the
    parts that do not depend on the insider classification from one call to the next."""
    subs = inp["subs"]
    st = cfg.get("starts")
    tol = st.get("entry_tolerance_days", 7)
    sh = shared if shared is not None else {}
    if "panel" not in sh:
        simfin, stq, facts = inp["simfin"], inp["stq"], inp["facts"]
        comps = simfin.companies if simfin.found else None
        sh["panel"] = oc.build_price_panel(simfin.prices if simfin.found else None,
                                           stq.prices if stq is not None else None, gap_fill=inp["eod_prices"])
        sh["as_of"] = oc.panel_as_of(sh["panel"])
        form_v = [d for d in dates if d <= sh["as_of"]]
        pf = ft.price_features(sh["panel"], form_v, tolerance_days=tol, max_gap_days=cfg.get("outcomes.max_gap_days", 120))
        keys = ft.ticker_ciks(pf, comps, inp["bridges"])
        if inp["eod_prices"] is not None and len(inp["eod_prices"]):
            pf, keys = eodhd.dedupe(pf, keys)
        # the vendor panel only supplies listing evidence here; financials and insider signals are
        # taken on every date (the locked dates all lie before the vendor panel's as-of date)
        sh["form_v"], sh["dates"], sh["keys"] = form_v, list(dates), keys
        sh["fund"] = fu.snapshots(facts, subs, list(dates), current_months=cfg.get("universe.current_filer_months", 15))
        names_sec = subs[["cik", "name"]]
        if comps is not None and "company_name" in comps.columns:
            names_sec = pd.concat([names_sec, comps[["cik", "company_name"]].rename(columns={"company_name": "name"})])
        sh["bridge"] = secprices.bridge(inp["f13_names"], inp["ftd_names"], comps, names_sec, inp["bridges"])
        sec_as_of = secprices.as_of_date(secprices.complete_through(inp["f13_breadth"]))
        first_a = pd.Timestamp(cfg.get("price_basis.first_formation_date"))
        form_a = [d for d in dates if first_a <= d <= sec_as_of]
        sh["sec_as_of"], sh["form_a"] = sec_as_of, form_a
        sh["pf_sec"] = ft.price_features(sec_panel, form_a, tolerance_days=tol,
                                         max_gap_days=cfg.get("outcomes.sec_max_gap_days", 200))
        sh["keys_sec"] = secprices.keys_for(sh["pf_sec"], sec_panel)
        sh["breadth"] = thirteenf.breadth_at(inp["f13_breadth"], sh["bridge"], form_a)
        f25 = inp["form25"]
        f25x = dl.form25_with_exchange(f25) if f25 is not None and len(f25) else None
        sym_tab = secprices.symbol_table(inp["ftd_names"], sh["bridge"], inp["bridges"])
        sym_ex = secprices.symbol_exchange(sym_tab, stq.groups if stq is not None else None, inp["eod_exchange"])
        sh["listed"] = sh["keys_sec"][["cik", "start_date"]].assign(
            listed=secprices.listed_flags(sh["keys_sec"], keys, f25x, sym_ex).to_numpy())
    ip = cfg.get("turnaround_cohort.insider_purchase")
    roles = tuple(ip.get("roles", ["officer", "director"]))
    signals = sale_signals = None
    tx = inp["ins_tx"]
    if tx is not None and len(tx):
        classes = insider.classify(tx, sorted({d.year for d in sh["dates"]}), purchases_only=purchases_only)
        reports = subs[subs["form"].isin(["10-Q", "10-K"])][["cik", "filed"]]
        sig_kw = dict(lookback_months=ip.get("lookback_months", 6), cluster_min=ip.get("cluster_variant_min_insiders", 2))
        signals = insider.signals(insider.purchases(tx, roles, exclude_plan=ip.get("exclude_rule_10b5_1", True)), classes,
                                  sh["dates"], report_dates=reports,
                                  post_results_days=cfg.get("features.post_results_window_days", 14), **sig_kw)
        sale_signals = insider.signals(insider.sales(tx, roles), classes, sh["dates"], **sig_kw)
    feats = ft.assemble(sh["pf_sec"], sh["keys_sec"], sh["fund"], signals, sh["breadth"], cfg.data, prices13=None,
                        sales=sale_signals, liquidity="breadth", min_breadth=int(k), listed=sh["listed"])
    ft.check_point_in_time(feats)                         # raises if any feature postdates its start date
    feats["price_basis"] = secprices.entry_source(feats, sec_panel)
    feats["low_attention"] = feats["low_breadth"] if attention == "breadth" else feats["low_turnover"]
    return feats, sh


def reproduction(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """The rebuilt table against the data run's saved one, on the start dates both hold: the
    universe's windows and each signal on them."""
    key = ["ticker", "start_date"]
    o = old.assign(start_date=pd.to_datetime(old["start_date"]).astype("datetime64[ns]"), ticker=old["ticker"].astype(str))
    n = new.assign(start_date=pd.to_datetime(new["start_date"]).astype("datetime64[ns]"), ticker=new["ticker"].astype(str))
    n = n[n["start_date"].isin(set(o["start_date"]))]
    uo = o[o["in_universe"].fillna(False).astype(bool)]
    un = n[n["in_universe"].fillna(False).astype(bool)]
    so, sn = set(map(tuple, uo[key].to_numpy())), set(map(tuple, un[key].to_numpy()))
    both = uo.merge(un, on=key, suffixes=("_old", "_new"))
    out = {"dates_compared": int(o["start_date"].nunique()), "rows_saved": int(len(o)), "rows_rebuilt": int(len(n)),
           "universe_saved": len(so), "universe_rebuilt": len(sn), "universe_in_both": len(so & sn),
           "universe_agreement": round(len(so & sn) / len(so | sn), 5) if so | sn else float("nan")}
    worst = 1.0
    for c in CHECK_FLAGS:
        if f"{c}_old" not in both.columns or not len(both):
            continue
        a, b = est._flag(both[f"{c}_old"]), est._flag(both[f"{c}_new"])
        same = float(((a == b) | (a.isna() & b.isna())).mean())
        out[f"same_{c}"] = round(same, 5)
        worst = min(worst, same)
    if len(both) and "market_cap_old" in both.columns:
        a, b = pd.to_numeric(both["market_cap_old"], errors="coerce"), pd.to_numeric(both["market_cap_new"], errors="coerce")
        out["same_market_cap"] = round(float((np.isclose(a, b, rtol=1e-6) | (a.isna() & b.isna())).mean()), 5)
        worst = min(worst, out["same_market_cap"])
    out["weakest_agreement"] = round(worst, 5)
    out["passed"] = bool(len(so) > 0 and out["universe_agreement"] >= MIN_AGREEMENT and worst >= MIN_AGREEMENT)
    return out


def new_outcomes(cfg, sec_panel, feats_new, delist, as_of, *, sec_gap: int = 200) -> pd.DataFrame:
    """One- and two-year outcomes of the confirmation's windows on the SEC panel."""
    return oc.compute_outcomes(cfg, sec_panel, oc.starts_from_features(feats_new), delist, as_of=as_of,
                               overrides={"max_gap_days": int(sec_gap), "horizons_years": HORIZONS,
                                          "touch_multiples": MULTIPLES})


def windows(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per universe window of the new dates: hit2_2y, bucket2, end2 (two years) and
    hit15_1y, bucket1, end1 (one year); a horizon that is not resolved stays missing."""
    f = features[features["in_universe"].fillna(False).astype(bool)].copy()
    f["start_date"] = pd.to_datetime(f["start_date"]).astype("datetime64[ns]")
    f["ticker"] = f["ticker"].astype(str)
    f["firm"] = pd.to_numeric(f["cik"], errors="coerce")
    o = outcomes[outcomes["status"].isin(est.RESOLVED)].copy()
    o["start_date"] = pd.to_datetime(o["start_date"]).astype("datetime64[ns]")
    o["ticker"] = o["ticker"].astype(str)
    num = lambda c: pd.to_numeric(o[c], errors="coerce")  # noqa: E731
    o2 = o[o["horizon_years"] == 2].assign(hit2_2y=num("touch_2x"), bucket2=num("bucket"), end2=num("end_multiple"))[
        ["ticker", "start_date", "hit2_2y", "bucket2", "end2"]]
    o1 = o[o["horizon_years"] == 1].assign(hit15_1y=num("touch_1.5x"), bucket1=num("bucket"), end1=num("end_multiple"))[
        ["ticker", "start_date", "hit15_1y", "bucket1", "end1"]]
    w = f.merge(o2, on=["ticker", "start_date"], how="left").merge(o1, on=["ticker", "start_date"], how="left")
    w["start_year"] = w["start_date"].dt.year
    return w


def _one_sided(z: float) -> float:
    return float(0.5 * math.erfc(z / math.sqrt(2))) if z is not None and np.isfinite(z) else float("nan")


def _c1(w: pd.DataFrame, column: str) -> dict:
    """Both coefficients above zero: one-sided z-tests, Holm across the two; C1's p is the weaker one."""
    m = est.m1(w, column=column)
    out = {"windows": m["windows"], "converged": m.get("converged"), "p": float("nan")}
    t = m["coefficients"]
    if not len(t):
        out["note"] = "fewer than 300 complete windows"
        return out
    t = t.set_index("regressor")
    ps = {}
    for c in ("fcf_pos", "low_dilution"):
        z = float(t.loc[c, "z"]) if c in t.index and pd.notna(t.loc[c, "z"]) else float("nan")
        ps[c] = _one_sided(z)
        out[f"{c}_coef"] = float(t.loc[c, "coef"]) if c in t.index and pd.notna(t.loc[c, "coef"]) else float("nan")
        out[f"{c}_odds_ratio"] = float(math.exp(out[f"{c}_coef"])) if np.isfinite(out[f"{c}_coef"]) else float("nan")
        out[f"{c}_p_one_sided"] = ps[c]
        if c in t.index and isinstance(t.loc[c, "note"], str) and t.loc[c, "note"]:
            out[f"{c}_note"] = t.loc[c, "note"]
    adj = est.holm(ps)
    out["p"] = float(max(adj.values())) if all(np.isfinite(v) for v in adj.values()) else float("nan")
    return out


def _lr(w: pd.DataFrame, signal: str, hit: str, kw: dict) -> dict:
    x = w[w[hit].notna()]
    if signal not in x.columns or not len(x):
        return {"windows": int(len(x)), "lr": float("nan"), "p_one_sided": float("nan")}
    r = est.likelihood_ratio(x, signal, hit, **kw)
    return {k: r.get(k) for k in ("windows", "with_signal", "firms", "base_rate", "hit_rate_with_signal", "lr", "ci_low",
                                  "ci_high", "p_one_sided")}


def tests(w: pd.DataFrame, cfg_c, *, resamples: int = 2000, seed: int = 20261106) -> dict:
    """C1 to C3 with Holm across the three, and the robustness versions of the locked plan."""
    kw = {"resamples": int(resamples), "seed": int(seed)}
    alpha = float(cfg_c.get("estimation.alpha", 0.05))
    two = w[w["start_date"].isin([pd.Timestamp(d) for d in cfg_c.get("dates.two_year")])]
    one = w[w["start_date"].isin([pd.Timestamp(d) for d in cfg_c.get("dates.one_year")])]
    c1 = _c1(two, "bucket2")
    c2 = _lr(two, "fallen", "hit2_2y", kw)
    c3 = _lr(two, "insider_post_results_cluster", "hit2_2y", kw)
    adj = est.holm({"C1": c1["p"], "C2": c2["p_one_sided"], "C3": c3["p_one_sided"]})

    def verdict(name, direction_ok):
        p = adj[name]
        if p is None or not np.isfinite(p):
            return "not estimable"
        return "confirmed" if (p <= alpha and direction_ok) else "not confirmed"
    c1.update(holm_p=adj["C1"], verdict=verdict("C1", (c1.get("fcf_pos_coef", np.nan) > 0) and (c1.get("low_dilution_coef", np.nan) > 0)))
    c2.update(holm_p=adj["C2"], verdict=verdict("C2", (c2.get("lr") or 0) > 1))
    c3.update(holm_p=adj["C3"], verdict=verdict("C3", (c3.get("lr") or 0) > 1))
    fcf2 = two[est._flag(two["fcf_pos"]).eq(1)]
    bought = two[est._flag(two["insider_any"]).eq(1)]
    robust = [{"version": "one year · C1 (one-year buckets)", **_c1(one, "bucket1")},
              {"version": "one year · C2 (touch 1.5x)", **_lr(one, "fallen", "hit15_1y", kw)},
              {"version": "one year · C3 (touch 1.5x)", **_lr(one, "insider_post_results_cluster", "hit15_1y", kw)},
              {"version": "positive free cash flow only · C2", **_lr(fcf2, "fallen", "hit2_2y", kw)},
              {"version": "positive free cash flow only · C3", **_lr(fcf2, "insider_post_results_cluster", "hit2_2y", kw)},
              {"version": "windows with a purchase only · C3", **_lr(bought, "insider_post_results_cluster", "hit2_2y", kw)}]
    by_date = (w.groupby("start_date").agg(windows=("ticker", "size"), firms=("firm", "nunique"), resolved_2y=("hit2_2y", "count"),
                                           touch_2x_2y=("hit2_2y", "mean"), resolved_1y=("hit15_1y", "count"),
                                           touch_1_5x_1y=("hit15_1y", "mean"), mean_end_1y=("end1", "mean"),
                                           mean_end_2y=("end2", "mean")).reset_index())
    by_date["start_date"] = by_date["start_date"].dt.strftime("%Y-%m-%d")
    return {"C1": c1, "C2": c2, "C3": c3, "robustness": robust, "by_date": by_date, "alpha": alpha}


# ---------------------------------------------------------------------------
# Leftovers of the locked study
# ---------------------------------------------------------------------------

def real_terms(w: pd.DataFrame, cpi: pd.Series, years: int = 5, column: str = "end5") -> pd.DataFrame:
    """End multiples in nominal and in real terms (the CPI of the month before the window's
    end over the month before its start), pooled and by regime."""
    x = w[w[column].notna()].copy()
    cols = ["group", "windows", "mean_nominal", "mean_real", "median_nominal", "median_real", "share_real_below_1",
            "share_nominal_below_1", "median_cpi_factor"]
    c = pd.Series(cpi).dropna().sort_index()
    if not len(x) or not len(c):
        return pd.DataFrame(columns=cols)
    c.index = pd.to_datetime(c.index)

    def at(dates):
        pos = np.searchsorted(c.index.to_numpy(), pd.to_datetime(dates).to_numpy(), side="left") - 1
        return np.where(pos >= 0, c.to_numpy()[np.clip(pos, 0, len(c) - 1)], np.nan)
    start = pd.to_datetime(x["start_date"])
    factor = at(start + pd.DateOffset(years=int(years))) / at(start)
    x["_real"], x["_factor"] = x[column].to_numpy(dtype=float) / factor, factor
    x = x[np.isfinite(x["_real"])]
    rows = []
    for name, g in [("all", x)] + ([(k, g) for k, g in x.groupby("regime")] if "regime" in x.columns else []):
        rows.append({"group": name, "windows": int(len(g)), "mean_nominal": float(g[column].mean()),
                     "mean_real": float(g["_real"].mean()), "median_nominal": float(g[column].median()),
                     "median_real": float(g["_real"].median()), "share_real_below_1": float((g["_real"] < 1).mean()),
                     "share_nominal_below_1": float((g[column] < 1).mean()), "median_cpi_factor": float(g["_factor"].median())})
    return pd.DataFrame(rows, columns=cols)


def basis_b(cfg, vendor_panel, vendor_as_of, features_b, delist_v, delist_b, sec_panel, sec_events, sec_dir, bridge,
            *, sec_gap: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustness B as notebook 01's locked Phase E computes it: daily vendor closes, and for
    the companies the vendors never price the SEC series with the fails-to-deliver prices
    in between. Returns (outcomes, the quarter-end over daily calibration)."""
    sb = oc.starts_from_features(features_b.assign(in_universe=True))
    fill_t = set(features_b.loc[features_b["panel_b"] == "SEC fill", "ticker"])
    is_fill = sb["ticker"].isin(fill_t)
    res_v = oc.compute_outcomes(cfg, vendor_panel, sb[~is_fill], delist_v)
    parts = [res_v]
    if is_fill.any():
        ciks = set(pd.to_numeric(sec_panel.loc[sec_panel["ticker"].isin(fill_t), "cik"], errors="coerce").dropna().astype("int64"))
        cusips = bridge.loc[pd.to_numeric(bridge["cik"], errors="coerce").isin(ciks), "cusip"].unique()
        px = ftd.load_px(sec_dir, cusips=cusips)
        dense = secprices.dense_panel(sec_panel, sec_events, px, fill_t)
        parts.append(oc.compute_outcomes(cfg, dense, sb[is_fill], delist_b, as_of=vendor_as_of,
                                         overrides={"max_gap_days": int(sec_gap)}))
    result = pd.concat(parts, ignore_index=True)
    calib = oc.touch_sampling_ratios(oc.OutcomeSpec.from_config(cfg.data), vendor_panel,
                                     sb.loc[~is_fill, ["ticker", "start_date"]], sample_start=pd.Timestamp.max,
                                     horizons=(3, 5), quarterly_max_gap_days=int(sec_gap))
    return result, calib


def base_rate_row(name: str, w: pd.DataFrame) -> dict:
    return {"version": name, "windows": int(len(w)), "firms": int(w["firm"].nunique()),
            "touch_3x_3y": float(w["hit3_3y"].mean()), "touch_10x_5y": float(w["hit10_5y"].mean()),
            "resolved_3y": int(w["hit3_3y"].notna().sum()), "resolved_5y": int(w["hit10_5y"].notna().sum()),
            "mean_end_5y": float(w["end5"].mean())}


# ---------------------------------------------------------------------------
# Last checks before the papers (exploratory; notebook 09): what a reviewer asks of C2 and of
# the free-cash-flow screen once the headline tests are in
# ---------------------------------------------------------------------------

PHARMA_SIC = (2833, 2834, 2835, 2836, 8731)


def both_sides(w: pd.DataFrame, flag: str, end: str, label: str) -> list[dict]:
    """The whole distribution of the end multiple with and without a flag: a signal that only
    widens the distribution shows a fatter loss side next to its fatter gain side."""
    x = w[w[end].notna() & est._flag(w[flag]).notna()] if flag in w.columns and end in w.columns else w.iloc[0:0]
    f = est._flag(x[flag]) if len(x) else pd.Series(dtype=float)
    rows = []
    for name, v in (("with", 1.0), ("without", 0.0)):
        g = x[f.eq(v)]
        e = g[end].to_numpy(dtype=float)
        if not len(e):
            continue
        rows.append({"split": label, "group": f"{name} {flag}", "windows": int(len(g)), "firms": int(g["firm"].nunique()),
                     "mean_end": float(e.mean()), "median_end": float(np.median(e)), "p10_end": float(np.quantile(e, 0.1)),
                     "p90_end": float(np.quantile(e, 0.9)), "share_end_below_half": float((e < 0.5).mean()),
                     "share_end_below_1": float((e < 1).mean()), "share_end_at_least_2x": float((e >= 2).mean())})
    return rows


def within_dates(w: pd.DataFrame, signal: str, hit: str, *, resamples: int = 1000, seed: int = 20261106) -> tuple[pd.DataFrame, dict]:
    """The hit rate with the signal over the hit rate without it, start date by start date and
    combined within start dates (Mantel-Haenszel weights): a signal that is merely more common
    on dates before a market-wide rebound loses its ratio here. Firm-cluster bootstrap."""
    x = w[w[hit].notna() & est._flag(w[signal]).notna()] if signal in w.columns and hit in w.columns else w.iloc[0:0]
    if not len(x):
        return pd.DataFrame(), {"windows": 0}
    s, h = est._flag(x[signal]).to_numpy(dtype=float), x[hit].to_numpy(dtype=float)
    fc, firms = pd.factorize(x["firm"].to_numpy())
    dc, dates = pd.factorize(pd.to_datetime(x["start_date"]).to_numpy(), sort=True)
    cols = np.column_stack([s * h, s, (1 - s) * h, 1 - s])                # a, n1, c, n0
    cell = np.zeros((len(firms), len(dates), 4))
    np.add.at(cell, (fc, dc), cols)
    tot = cell.sum(axis=0)
    by_date = pd.DataFrame({"start_date": pd.to_datetime(dates).strftime("%Y-%m-%d"), "windows": (tot[:, 1] + tot[:, 3]).astype(int),
                            "with_signal": tot[:, 1].astype(int),
                            "hit_rate_with": np.divide(tot[:, 0], tot[:, 1], out=np.full(len(dates), np.nan), where=tot[:, 1] > 0),
                            "hit_rate_without": np.divide(tot[:, 2], tot[:, 3], out=np.full(len(dates), np.nan), where=tot[:, 3] > 0)})
    by_date["ratio"] = by_date["hit_rate_with"] / by_date["hit_rate_without"].where(by_date["hit_rate_without"] > 0)

    def mh(t):                                             # t: (..., dates, 4)
        n = t[..., 1] + t[..., 3]
        num = np.where(n > 0, t[..., 0] * t[..., 3] / np.where(n > 0, n, 1), 0).sum(axis=-1)
        den = np.where(n > 0, t[..., 2] * t[..., 1] / np.where(n > 0, n, 1), 0).sum(axis=-1)
        return np.divide(num, den, out=np.full(np.shape(num), np.nan), where=den > 0)

    def pooled(t):
        a, n1, c, n0 = (t[..., k].sum(axis=-1) for k in range(4))
        return np.divide(a * n0, c * n1, out=np.full(np.shape(a), np.nan), where=(c * n1) > 0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(firms), size=(int(resamples), len(firms)))
    wts = np.zeros((int(resamples), len(firms)))
    np.add.at(wts, (np.arange(int(resamples))[:, None], draws), 1.0)
    boot = np.einsum("bf,fdk->bdk", wts, cell, optimize=True)
    b = mh(boot)
    lo, hi = est._ci(b)
    out = {"windows": int(len(x)), "dates": int(len(dates)), "ratio_pooled": float(pooled(tot)), "ratio_within_dates": float(mh(tot)),
           "ci_low": lo, "ci_high": hi, "p_one_sided": est._p_above(b),
           "dates_with_ratio_above_1": int((by_date["ratio"] > 1).sum())}
    return by_date, out


def subsets(w: pd.DataFrame, hit: str, bucket: str, *, resamples: int = 1000, seed: int = 20261106) -> pd.DataFrame:
    """The fall's ratio and the free-cash-flow coefficient without pharma and biotech, and in
    the smaller and the larger half of each start date's universe by market cap."""
    sic = pd.to_numeric(w.get("sic"), errors="coerce") if "sic" in w.columns else pd.Series(np.nan, index=w.index)
    cap = pd.to_numeric(w["market_cap"], errors="coerce")
    big = cap >= cap.groupby(w["start_date"]).transform("median")
    rows = []
    for name, g in (("all", w), ("without pharma and biotech", w[~sic.isin(PHARMA_SIC)]), ("pharma and biotech only", w[sic.isin(PHARMA_SIC)]),
                    ("smaller half by market cap", w[~big]), ("larger half by market cap", w[big])):
        r = _lr(g, "fallen", hit, {"resamples": int(resamples), "seed": int(seed)})
        row = {"subset": name, "windows": int(g[hit].notna().sum()), "fall_lr": r.get("lr"), "fall_ci_low": r.get("ci_low"),
               "fall_ci_high": r.get("ci_high")}
        try:
            c = _c1(g, bucket)
        except Exception as exc:                           # a subset the model cannot be fitted on is reported, not fatal
            c = {"note": f"{type(exc).__name__}"}
        row.update({"fcf_odds_ratio": c.get("fcf_pos_odds_ratio", np.nan), "fcf_p_one_sided": c.get("fcf_pos_p_one_sided", np.nan),
                    "low_dilution_odds_ratio": c.get("low_dilution_odds_ratio", np.nan), "model_windows": c.get("windows", 0)})
        rows.append(row)
    return pd.DataFrame(rows)


def last_checks(w_locked: pd.DataFrame | None, w_new: pd.DataFrame | None, *, resamples: int = 1000, seed: int = 20261106) -> dict:
    """Every last check on the windows that are on Drive; a missing table is skipped."""
    sides, dates, strat, subs = [], [], [], []
    kw = {"resamples": resamples, "seed": seed}
    for label, w, end, hit, bucket in (("new dates, two years", w_new, "end2", "hit2_2y", "bucket2"),
                                       ("locked dates, five years", w_locked, "end5", "hit3_3y", "bucket5")):
        if w is None or not len(w):
            continue
        for flag in ("fallen", "fcf_pos", "low_dilution"):
            sides += both_sides(w, flag, end, label)
        horizon = "2x within two years" if hit == "hit2_2y" else "3x within three years"
        bd, st = within_dates(w, "fallen", hit, **kw)
        strat.append({"sample": label, "signal": "fallen", "hit": horizon, **st})
        if hit == "hit2_2y" and len(bd):
            dates += [{"sample": label, **r} for r in bd.dropna(subset=["ratio"]).to_dict("records")]
        s = subsets(w, hit, bucket, **kw)
        subs += [{"sample": label, **r} for r in s.to_dict("records")]
    return {"sides": sides, "dates": dates, "stratified": strat, "subsets": subs}
