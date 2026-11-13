"""A third batch of candidates: the accounting patterns the feature table could not measure
(exploratory; notebook 11).

The data run's extract of the SEC Financial Statement Data Sets kept only the tags its
features needed. Accruals, gross profitability, book-to-market, asset growth, R&D intensity
and the full Piotroski score need more of the statements – net income, total assets,
equity, cost of revenue, R&D expense, liabilities, current assets and liabilities – so the
quarterly files are read once more for those tags into their own cache (`fsds_extra`); the
data run's cache is not touched. Values are taken point in time with the data run's own
rules (the latest 10-K or 10-Q filed before the start date; restatements count only once
filed).

Candidates and combinations were fixed on Nov 13, 2026, before any was set against an outcome:
accruals (Sloan), gross profitability (Novy-Marx), book-to-market (Fama and French), asset
growth (Cooper, Gulen and Schill), R&D to market value (Chan, Lakonishok and Sougiannis),
return on assets, and Piotroski's nine-part F-score (leverage is measured on total
liabilities, not long-term debt alone).
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import estimators as est
from . import fundamentals as fu
from . import scan
from .data.fsds import PERIODIC_FORMS, UNITS, _ym
from .data.sec_datasets import FSDS_URLS, parse_dates, process, quarter_items, read_member, to_int

EXTRA_TAG_GROUPS: dict[str, tuple[str, ...]] = {
    "net_income": ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    "assets": ("Assets",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "liabilities": ("Liabilities",),
    "assets_current": ("AssetsCurrent",),
    "liabilities_current": ("LiabilitiesCurrent",),
    "cogs": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"),
    "gross_profit": ("GrossProfit",),
    "rnd": ("ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
}
EXTRA_TAGS = {t: (g, i) for g, tags in EXTRA_TAG_GROUPS.items() for i, t in enumerate(tags)}
EXTRA_SCHEMA = hashlib.sha1(json.dumps({"tags": EXTRA_TAG_GROUPS, "forms": PERIODIC_FORMS}, sort_keys=True).encode()).hexdigest()[:12]
FLOWS = ("net_income", "cogs", "gross_profit", "rnd")
BALANCES = ("assets", "equity", "liabilities", "assets_current", "liabilities_current")

ACC_CONTINUOUS = ("accruals", "gross_profitability", "book_to_market", "asset_growth", "rnd_to_cap", "return_on_assets", "f_score")
ACC_FLAGS = ("f_score_high", "f_score_low", "negative_equity", "reports_rnd")
ACC_COMBOS = (
    ("fallen with an F-score of 7 to 9", "a_fallen_f_high"),
    ("fallen with an F-score of 0 to 3", "a_fallen_f_low"),
    ("Piotroski's own test: book-to-market in the top fifth, F-score 7 to 9", "a_value_f_high"),
    ("book-to-market in the top fifth, F-score 0 to 3", "a_value_f_low"),
    ("book-to-market in the top fifth", "a_value"),
    ("profitable value: gross profitability and book-to-market both in the top two fifths", "a_profitable_value"),
    ("gross profitability in the top fifth", "a_gross"),
    ("R&D to market value in the top fifth", "a_rnd"),
    ("R&D in the top fifth and fallen", "a_rnd_fallen"),
    ("R&D in the top fifth and cash-positive", "a_rnd_fcf"),
    ("accruals in the lowest fifth and cash-positive", "a_clean_fcf"),
    ("accruals in the top fifth", "a_accruals_high"),
    ("asset growth in the top fifth", "a_growth_high"),
    ("fallen, F-score 7 to 9, floor net of burn", "a_fallen_f_floor"),
)


def extract_extra(zf: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    """One quarterly file: the periodic filings and the extra tags, non-dimensional, in USD."""
    sub = read_member(zf, "sub", ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "afs", "fye", "instance",
                                  "prevrpt", "countryba"])
    sub = sub[sub["form"].isin(PERIODIC_FORMS)].copy()
    keep = set(sub["adsh"])

    def only_wanted(chunk: pd.DataFrame) -> pd.DataFrame:
        ok = chunk["tag"].isin(EXTRA_TAGS.keys()) & chunk["uom"].isin(UNITS) & chunk["adsh"].isin(keep) & chunk["segments"].isna()
        if "coreg" in chunk.columns:
            ok &= chunk["coreg"].isna()
        return chunk[ok]

    num = read_member(zf, "num", ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg", "value"],
                      chunk_filter=only_wanted)
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    num = num.dropna(subset=["value"])
    sub["cik"] = to_int(sub["cik"])
    sub["filed"] = parse_dates(sub["filed"])
    sub["period"] = parse_dates(sub["period"])
    subs = sub[["adsh", "cik", "form", "period", "fp", "filed"]].reset_index(drop=True)
    f = num.merge(subs[["adsh", "cik", "filed"]], on="adsh", how="inner")
    f["ym"] = _ym(parse_dates(f["ddate"]))
    f = f.dropna(subset=["cik", "ym", "filed"]).reset_index(drop=True)
    facts = pd.DataFrame({"cik": f["cik"].astype("int64"), "tag": f["tag"].astype("category"), "ym": f["ym"].astype("int32"),
                          "qtrs": pd.to_numeric(f["qtrs"], errors="coerce").fillna(-1).astype("int16"),
                          "value": f["value"].astype("float64"), "filed": f["filed"], "adsh": f["adsh"].astype(str)})
    return {"subs": subs, "facts": facts}


def load_extra(sec_dir, start_year: int, end_year: int, *, client=None, offline: bool = False, today=None, log=print,
               local_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(subs, facts, stats) of the extra tags for every quarter, cached per quarter in `fsds_extra`."""
    items = quarter_items(FSDS_URLS, start_year, end_year, today)
    tables, stats = process(items, Path(sec_dir) / "fsds_extra", "fsdsx", extract_extra, client=client, offline=offline, log=log,
                            local_dir=local_dir, schema=EXTRA_SCHEMA)
    subs = tables.get("subs", pd.DataFrame(columns=["adsh", "cik", "form", "period", "fp", "filed"]))
    facts = tables.get("facts", pd.DataFrame(columns=["cik", "tag", "ym", "qtrs", "value", "filed", "adsh"]))
    if len(facts):
        facts = facts.drop_duplicates(["adsh", "tag", "ym", "qtrs", "value"]).reset_index(drop=True)
    if len(subs):
        subs = subs.drop_duplicates("adsh", keep="last").reset_index(drop=True)
    return subs, facts, stats


def prepare(facts: pd.DataFrame) -> dict:
    """Sorted by filing date with each tag's group and priority – the shape `fundamentals._latest` reads."""
    f = facts.copy()
    f["filed"] = pd.to_datetime(f["filed"]).astype("datetime64[ns]")
    tag = f["tag"].astype(str)
    f["tag"] = tag.astype("category")
    f["group"] = tag.map(lambda t: EXTRA_TAGS.get(t, ("other", 99))[0]).astype("category")
    f["prio"] = tag.map(lambda t: EXTRA_TAGS.get(t, ("other", 99))[1]).astype("int16")
    f = f.sort_values("filed", kind="mergesort").reset_index(drop=True)
    return {"plain": f, "plain_filed": f["filed"].to_numpy(dtype="datetime64[ns]")}


def snapshot(prep: dict, subs: pd.DataFrame, cutoff, *, current_months: int = 15) -> pd.DataFrame:
    """The extra statement values known before `cutoff`: trailing twelve months for the
    flows (and the twelve months before), the balance sheet at the anchor period and a year earlier."""
    cutoff = pd.Timestamp(cutoff)
    a = fu._anchors(subs, cutoff, current_months)
    if a.empty:
        return pd.DataFrame()
    look = fu._Lookup(fu._latest(prep, cutoff, fu.month_index(cutoff) - 12 * 4))
    cik, P, k = a["cik"].to_numpy(), a["P"].to_numpy(dtype="float64"), a["k"].to_numpy(dtype="float64")
    out = pd.DataFrame({"firm": cik.astype("int64"), "start_date": cutoff})
    for g in FLOWS:
        out[g] = fu._ttm(look, g, cik, P, k)
        out[f"{g}_prev"] = fu._ttm(look, g, cik, P, k, back=1)
    for g in BALANCES:
        out[g] = look.get(g, cik, P, 0)
        out[f"{g}_prev"] = look.get(g, cik, P - 12, 0)
    return out


def snapshots(facts: pd.DataFrame, subs: pd.DataFrame, cutoffs, *, current_months: int = 15) -> pd.DataFrame:
    prep = prepare(facts)
    s = subs.copy()
    s["filed"] = pd.to_datetime(s["filed"]).astype("datetime64[ns]")
    s["period"] = pd.to_datetime(s["period"]).astype("datetime64[ns]")
    parts = [snapshot(prep, s, c, current_months=current_months) for c in sorted(pd.to_datetime(list(cutoffs)))]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["firm", "start_date"])


def measures(w: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """The windows with the accounting measures; the feature table supplies operating cash
    flow, revenue (both trailing twelve months, and the year before) and market cap."""
    x = w.copy()
    x["start_date"] = pd.to_datetime(x["start_date"]).astype("datetime64[ns]")
    s = snaps.assign(start_date=pd.to_datetime(snaps["start_date"]).astype("datetime64[ns]")).drop_duplicates(["firm", "start_date"])
    x = x.merge(s, on=["firm", "start_date"], how="left")
    n = lambda c: pd.to_numeric(x[c], errors="coerce") if c in x.columns else pd.Series(np.nan, index=x.index)  # noqa: E731
    pos = lambda v: v.where(v > 0)  # noqa: E731
    assets, assets_prev, cap = pos(n("assets")), pos(n("assets_prev")), pos(n("market_cap"))
    ni, ni_prev, ocf = n("net_income"), n("net_income_prev"), n("ocf")
    rev, rev_prev = pos(n("revenue")), pos(n("revenue_prev"))
    gross = n("gross_profit").where(n("gross_profit").notna(), rev - n("cogs"))
    gross_prev = n("gross_profit_prev").where(n("gross_profit_prev").notna(), rev_prev - n("cogs_prev"))
    x["accruals"] = (ni - ocf) / assets
    x["gross_profitability"] = gross / assets
    x["book_to_market"] = pos(n("equity")) / cap
    x["negative_equity"] = (n("equity") <= 0).astype(float).where(n("equity").notna())
    x["asset_growth"] = assets / assets_prev - 1
    x["rnd_to_cap"] = pos(n("rnd")) / cap
    x["reports_rnd"] = (n("rnd") > 0).astype(float).where(assets.notna())       # no R&D line in a filing that has a balance sheet: none
    x["return_on_assets"] = ni / assets
    roa, roa_prev = ni / assets, ni_prev / assets_prev
    parts = [(roa > 0, roa.notna()), (ocf > 0, ocf.notna()), (roa > roa_prev, roa.notna() & roa_prev.notna()),
             (ocf > ni, ocf.notna() & ni.notna()),
             (n("liabilities") / assets < n("liabilities_prev") / assets_prev, (n("liabilities") / assets).notna() & (n("liabilities_prev") / assets_prev).notna()),
             (n("assets_current") / pos(n("liabilities_current")) > n("assets_current_prev") / pos(n("liabilities_current_prev")),
              (n("assets_current") / pos(n("liabilities_current"))).notna() & (n("assets_current_prev") / pos(n("liabilities_current_prev"))).notna()),
             (n("share_growth") <= 0.01, n("share_growth").notna()),
             (gross / rev > gross_prev / rev_prev, (gross / rev).notna() & (gross_prev / rev_prev).notna()),
             (rev / assets > rev_prev / assets_prev, (rev / assets).notna() & (rev_prev / assets_prev).notna())]
    known = np.sum([k.to_numpy() for _, k in parts], axis=0)
    score = np.sum([(c & k).to_numpy().astype(float) for c, k in parts], axis=0)
    # companies without a classified balance sheet or a cost-of-revenue line lack one or two parts: the
    # score is scaled to nine where at least seven parts are known
    x["f_parts_known"] = known
    x["f_score"] = np.where(known >= 7, np.round(score * 9 / np.maximum(known, 1)), np.nan)
    x["f_score_high"] = np.where(known >= 7, (x["f_score"] >= 7).astype(float), np.nan)
    x["f_score_low"] = np.where(known >= 7, (x["f_score"] <= 3).astype(float), np.nan)
    return x


def coverage(x: pd.DataFrame) -> dict:
    """Share of windows with each measure (a measure most companies lack is no screen)."""
    return {c: round(float(x[c].notna().mean()), 3) for c in (*ACC_CONTINUOUS, *ACC_FLAGS) if c in x.columns} if len(x) else {}


def with_combos(x: pd.DataFrame, panel=None) -> pd.DataFrame:
    """The combinations of the first two batches (scan.with_combos) plus this batch's."""
    y = scan.with_combos(x, panel)
    for c in (*ACC_CONTINUOUS, *ACC_FLAGS):
        if c in x.columns:
            y[c] = x[c].to_numpy()
    fallen = est._flag(y["fallen"]).eq(1) if "fallen" in y.columns else pd.Series(False, index=y.index)
    pos = pd.to_numeric(y["fcf"], errors="coerce") > 0 if "fcf" in y.columns else pd.Series(False, index=y.index)
    known_fcf = pd.to_numeric(y["fcf"], errors="coerce").notna() if "fcf" in y.columns else pd.Series(False, index=y.index)
    top = lambda c: scan._quintile_flags(y, c)[1]  # noqa: E731
    low_acc, high_acc = scan._quintile_flags(y, "accruals")
    value, gross, rnd, growth = top("book_to_market"), top("gross_profitability"), top("rnd_to_cap"), top("asset_growth")
    rank = lambda c: pd.to_numeric(y[c], errors="coerce").groupby(y["start_date"]).rank(pct=True)  # noqa: E731
    fh, fl = y["f_score_high"], y["f_score_low"]

    def put(name, cond, need):
        y[name] = cond.astype(float).where(need)
    put("a_fallen_f_high", fallen & fh.eq(1), fh.notna())
    put("a_fallen_f_low", fallen & fl.eq(1), fl.notna())
    put("a_value_f_high", value.eq(1) & fh.eq(1), value.notna() & fh.notna())
    put("a_value_f_low", value.eq(1) & fl.eq(1), value.notna() & fl.notna())
    put("a_value", value.eq(1), value.notna())
    rg, rb = rank("gross_profitability"), rank("book_to_market")
    put("a_profitable_value", (rg > 0.6) & (rb > 0.6), rg.notna() & rb.notna())
    put("a_gross", gross.eq(1), gross.notna())
    put("a_rnd", rnd.eq(1), rnd.notna())
    put("a_rnd_fallen", rnd.eq(1) & fallen, rnd.notna())
    put("a_rnd_fcf", rnd.eq(1) & pos, rnd.notna() & known_fcf)
    put("a_clean_fcf", low_acc.eq(1) & pos, low_acc.notna() & known_fcf)
    put("a_accruals_high", high_acc.eq(1), high_acc.notna())
    put("a_growth_high", growth.eq(1), growth.notna())
    floor_net = y["c_floor_net"] if "c_floor_net" in y.columns else pd.Series(np.nan, index=y.index)
    put("a_fallen_f_floor", fallen & fh.eq(1) & floor_net.eq(1), fh.notna() & floor_net.notna())
    return y
