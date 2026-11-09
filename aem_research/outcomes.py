"""Outcome engine: touch and end-of-window multiples per (ticker, start date, horizon).

Rules (pre-registered in config/preregistration.yaml):
  * Entry: first price on or after the start date, within entry_tolerance_days.
  * Window: entry date + horizon years. Windows ending after the data's as-of date
    are censored (excluded), never partially counted.
  * Touch m: some daily price in (entry, window end] is at least m x entry price.
  * End multiple: last price on or before the window end over the entry price. If the
    series stops more than stale_gap_days before the window end, the delisting return
    is applied to the last price – observed if supplied, else imputed by category,
    with bounds: lower = total loss, upper = no haircut.
  * A gap longer than max_gap_days ends a series: prices after it under the same
    symbol are treated as a different security, never as a jump in the old one.
  * Bucket: six-bucket grid; each bucket includes its lower edge.

`compute_outcomes` refuses to run without a valid pre-registration lock.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import StudyConfig, require_locked

COMPLETE, DELISTED, ENDED, CENSORED, NO_ENTRY = "complete", "delisted", "series_ended", "censored", "no_entry"


def touch_col(m: float) -> str:
    return f"touch_{m:g}x"


def days_col(m: float) -> str:
    return f"days_to_{m:g}x"


@dataclass(frozen=True)
class OutcomeSpec:
    horizons_years: tuple = (3, 5, 10)
    touch_multiples: tuple = (2.0, 3.0, 4.0, 6.24, 10.0, 100.0)
    bucket_edges: tuple = (0.2, 0.632, 1.58, 3.54, 7.07)
    bucket_labels: tuple = (0.1, 0.4, 1.0, 2.5, 5.0, 10.0)
    entry_tolerance_days: int = 7
    stale_gap_days: int = 30
    max_gap_days: int = 120
    delist_returns: tuple = (("merger", 0.0), ("performance", -0.30), ("unknown", -0.30))

    def __post_init__(self):
        if len(self.bucket_labels) != len(self.bucket_edges) + 1:
            raise ValueError("bucket_labels must have one more entry than bucket_edges")
        if list(self.bucket_edges) != sorted(self.bucket_edges):
            raise ValueError("bucket_edges must be increasing")
        if any(int(h) != h or h <= 0 for h in self.horizons_years):
            raise ValueError("horizons_years must be positive whole years")
        if any(m <= 1 for m in self.touch_multiples):
            raise ValueError("touch multiples must exceed 1")

    @classmethod
    def from_config(cls, data: dict) -> "OutcomeSpec":
        o = data.get("outcomes", {})
        s = data.get("starts", {})
        returns = (data.get("delisting", {}) or {}).get("returns", {}) or {}
        return cls(
            horizons_years=tuple(int(h) for h in o.get("horizons_years", cls.horizons_years)),
            touch_multiples=tuple(float(m) for m in o.get("touch_multiples", cls.touch_multiples)),
            bucket_edges=tuple(float(e) for e in o.get("bucket_edges", cls.bucket_edges)),
            bucket_labels=tuple(float(b) for b in o.get("bucket_labels", cls.bucket_labels)),
            entry_tolerance_days=int(s.get("entry_tolerance_days", cls.entry_tolerance_days)),
            stale_gap_days=int(o.get("stale_gap_days", cls.stale_gap_days)),
            max_gap_days=int(o.get("max_gap_days", cls.max_gap_days)),
            delist_returns=tuple((str(k), float(v)) for k, v in returns.items()) or cls.delist_returns,
        )

    def delist_return(self, category: str) -> float:
        table = dict(self.delist_returns)
        return float(table.get(category, table.get("unknown", -0.30)))

    def bucket(self, multiple: float) -> float:
        if multiple is None or not np.isfinite(multiple):
            return np.nan
        return float(self.bucket_labels[int(np.searchsorted(self.bucket_edges, multiple, side="right"))])


def _ns(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")


def _delisting_lookup(delistings: pd.DataFrame | None) -> dict:
    if delistings is None or len(delistings) == 0:
        return {}
    d = delistings.copy()
    d["ticker"] = d["ticker"].astype(str)
    d["delist_date"] = _ns(d["delist_date"])
    if "category" not in d.columns:
        d["category"] = "unknown"
    if "delist_return" not in d.columns:
        d["delist_return"] = np.nan
    out: dict = {}
    for row in d.dropna(subset=["delist_date"]).itertuples(index=False):
        out.setdefault(row.ticker, []).append(
            (np.datetime64(row.delist_date, "ns"), str(row.category), float(row.delist_return)))
    return out


def _match_delisting(records, last_date, window_end, tolerance) -> tuple | None:
    if not records:
        return None
    best, best_gap = None, None
    for when, cat, ret in records:
        if last_date - tolerance <= when <= window_end:
            gap = abs(when - last_date)
            if best_gap is None or gap < best_gap:
                best, best_gap = (cat, ret), gap
    return best


def _compute_outcomes(spec: OutcomeSpec, prices: pd.DataFrame, starts: pd.DataFrame,
                      delistings: pd.DataFrame | None = None, *, price_col: str = "price",
                      as_of=None) -> pd.DataFrame:
    """Engine without the lock check – for tests and validated callers only."""
    need = {"ticker", "date", price_col}
    if need - set(prices.columns):
        raise ValueError(f"prices needs columns {sorted(need)}")
    if {"ticker", "start_date"} - set(starts.columns):
        raise ValueError("starts needs columns ['start_date', 'ticker']")

    # Work on positions per ticker; no global string conversion or sort of a 50M-row table.
    dates_all = _ns(prices["date"]).to_numpy(dtype="datetime64[ns]")
    price_all = pd.to_numeric(prices[price_col], errors="coerce").to_numpy(dtype="float64")
    positions = {str(k): v for k, v in
                 prices.groupby("ticker", observed=True, sort=False).indices.items()}
    if as_of is None and "source" in prices.columns:
        as_of = panel_as_of(prices)       # a source that ends sooner must not look like delistings
    if as_of is None:
        valid = dates_all[~np.isnat(dates_all)]
        as_of = valid.max() if valid.size else np.datetime64("NaT", "ns")
    as_of = np.datetime64(pd.Timestamp(as_of), "ns")

    st = starts[["ticker", "start_date"]].copy()
    st["ticker"] = st["ticker"].astype(str)
    st["start_date"] = _ns(st["start_date"])
    st = st.dropna().drop_duplicates()

    def series_for(t):
        pos = positions.get(t)
        if pos is None:
            return None
        d, p = dates_all[pos], price_all[pos]
        ok = ~np.isnat(d) & np.isfinite(p) & (p > 0)
        d, p = d[ok], p[ok]
        if d.size == 0:
            return None
        order = np.argsort(d, kind="mergesort")
        d, p = d[order], p[order]
        keep = np.ones(d.size, dtype=bool)
        keep[:-1] = d[1:] != d[:-1]          # duplicate dates: keep the last
        return d[keep], p[keep]

    dl = _delisting_lookup(delistings)
    tol = np.timedelta64(spec.entry_tolerance_days, "D")
    gap = np.timedelta64(spec.stale_gap_days, "D")
    max_gap = np.timedelta64(spec.max_gap_days, "D")
    match_tol = np.timedelta64(max(spec.stale_gap_days, 30), "D")
    one_day = np.timedelta64(1, "D")
    end_cache: dict = {}
    mults = spec.touch_multiples
    records: list[dict] = []

    def blank(t, s, h, status, entry=None, p0=np.nan, we=None):
        row = {"ticker": t, "start_date": s, "horizon_years": h, "status": status,
               "entry_date": entry, "entry_price": p0, "window_end": we,
               "end_multiple": np.nan, "end_multiple_lower": np.nan, "end_multiple_upper": np.nan,
               "max_multiple": np.nan, "bucket": np.nan, "delist_category": None,
               "delist_return_used": np.nan, "delist_return_observed": None}
        for m in mults:
            row[touch_col(m)] = np.nan
            row[days_col(m)] = np.nan
        return row

    for t, grp in st.groupby("ticker", sort=False):
        series = series_for(t)
        sdates = grp["start_date"].to_numpy(dtype="datetime64[ns]")
        if series is None:
            for s in sdates:
                records.extend(blank(t, s, h, NO_ENTRY) for h in spec.horizons_years)
            continue
        dd, pp = series
        # A gap longer than max_gap_days ends a series: whatever trades after it under the
        # same symbol is treated as a different security (symbol reuse, relisting).
        breaks = np.flatnonzero(np.diff(dd) > max_gap) + 1
        for s in sdates:
            j0 = int(np.searchsorted(dd, s, side="left"))
            if j0 >= len(dd) or dd[j0] - s > tol:
                records.extend(blank(t, s, h, NO_ENTRY) for h in spec.horizons_years)
                continue
            k = int(np.searchsorted(breaks, j0, side="right"))
            lo_i = int(breaks[k - 1]) if k > 0 else 0
            hi_i = int(breaks[k]) if k < len(breaks) else len(dd)
            d, p, i0 = dd[lo_i:hi_i], pp[lo_i:hi_i], j0 - lo_i
            series_end = d[-1]
            e, p0 = d[i0], p[i0]
            for h in spec.horizons_years:
                key = (e, h)
                we = end_cache.get(key)
                if we is None:
                    we = np.datetime64((pd.Timestamp(e) + pd.DateOffset(years=int(h))).to_datetime64(), "ns")
                    end_cache[key] = we
                if we > as_of:
                    records.append(blank(t, s, h, CENSORED, e, p0, we))
                    continue
                row = blank(t, s, h, COMPLETE, e, p0, we)
                i1 = int(np.searchsorted(d, we, side="right"))
                seg_d, rel = d[i0 + 1:i1], p[i0 + 1:i1] / p0
                row["max_multiple"] = max(float(rel.max()) if rel.size else 1.0, 1.0)
                if series_end >= we - gap:
                    base = lower = upper = p[i1 - 1] / p0
                else:
                    last_p = p[-1]
                    hit = _match_delisting(dl.get(t), series_end, we, match_tol)
                    if hit is not None:
                        cat, observed = hit
                        known = bool(np.isfinite(observed))
                        r = observed if known else spec.delist_return(cat)
                        row["status"] = DELISTED
                    else:
                        cat, known, r = "unknown", False, spec.delist_return("unknown")
                        row["status"] = ENDED
                    base = last_p * (1.0 + r) / p0
                    lower, upper = (base, base) if known else (0.0, last_p / p0)
                    row.update(delist_category=cat, delist_return_used=r, delist_return_observed=known)
                row.update(end_multiple=base, end_multiple_lower=lower, end_multiple_upper=upper,
                           bucket=spec.bucket(base))
                for m in mults:
                    hit_mask = rel >= m
                    if hit_mask.any():
                        row[touch_col(m)] = 1.0
                        row[days_col(m)] = float((seg_d[int(np.argmax(hit_mask))] - e) / one_day)
                    else:
                        row[touch_col(m)] = 0.0
                records.append(row)

    out = pd.DataFrame.from_records(records)
    if len(out):
        for col in ("start_date", "entry_date", "window_end"):
            out[col] = _ns(out[col])
    return out


def compute_outcomes(cfg: StudyConfig, prices: pd.DataFrame, starts: pd.DataFrame,
                     delistings: pd.DataFrame | None = None, *, price_col: str = "price",
                     as_of=None, overrides: dict | None = None) -> pd.DataFrame:
    """Outcomes for the study. Requires a locked pre-registration. `overrides` replaces spec
    fields for a panel with other spacing (the SEC quarter-end panel: max_gap_days)."""
    require_locked(cfg)
    spec = OutcomeSpec.from_config(cfg.data)
    if overrides:
        spec = replace(spec, **overrides)
    return _compute_outcomes(spec, prices, starts, delistings, price_col=price_col, as_of=as_of)


def quarter_end_panel(panel: pd.DataFrame, *, price_col: str = "price", tolerance_days: int = 10) -> pd.DataFrame:
    """A daily panel sampled like the SEC panel: per ticker, the last price on or before each
    quarter-end (within tolerance_days), dated the day after the quarter-end."""
    cols = ["ticker", "date", price_col, "source"]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=cols)
    p = panel[["ticker", "date", price_col] + (["source"] if "source" in panel.columns else [])].copy()
    p["date"] = _ns(p["date"])
    p = p.dropna(subset=["date"])
    p["qe"] = p["date"].dt.to_period("Q").dt.end_time.dt.normalize().astype("datetime64[ns]")
    p = p.sort_values(["ticker", "date"], kind="mergesort")
    last = p.groupby(["ticker", "qe"], sort=False, observed=True).tail(1)
    last = last[(last["qe"] - last["date"]) <= pd.Timedelta(days=tolerance_days)]
    out = last.assign(date=(last["qe"] + pd.Timedelta(days=1)).astype("datetime64[ns]")).drop(columns="qe")
    if "source" not in out.columns:
        out["source"] = "vendor"
    return out[cols].reset_index(drop=True)


def touch_sampling_ratios(spec: OutcomeSpec, panel: pd.DataFrame, starts: pd.DataFrame, *, sample_start,
                          horizons=(3,), multiples=None, quarterly_max_gap_days: int = 200) -> pd.DataFrame:
    """How much quarter-end sampling lowers touch rates (D10): the same series sampled at
    quarter-ends and daily, on the same windows.

    Every start date must precede `sample_start` – before the lock, the sample's first
    start date, so no window of the locked sample is computed; after the lock (where the
    notebook measures it, on B's vendor windows), any later date. Returns per horizon and
    multiple only the ratio of the quarter-end touch rate to the daily touch rate, and the
    window count."""
    st = starts.copy()
    st["start_date"] = _ns(st["start_date"])
    if len(st) and st["start_date"].max() >= pd.Timestamp(sample_start):
        raise ValueError("touch_sampling_ratios is for start dates before the sample only")
    cols = ["horizon_years", "multiple", "windows", "quarter_end_over_daily"]
    if panel is None or panel.empty or st.empty:
        return pd.DataFrame(columns=cols)
    multiples = tuple(multiples or spec.touch_multiples)
    as_of = panel_as_of(panel)
    # only the tickers and years the windows need: the full vendor panel is never copied
    last_day = st["start_date"].max() + pd.DateOffset(years=int(max(horizons))) + pd.Timedelta(days=31)
    keep = panel["ticker"].isin(set(st["ticker"])).to_numpy() & (_ns(panel["date"]) <= last_day).to_numpy()
    panel = panel[keep]
    daily = _compute_outcomes(spec, panel, st, as_of=as_of)
    qspec = replace(spec, max_gap_days=int(quarterly_max_gap_days))
    qtr = _compute_outcomes(qspec, quarter_end_panel(panel), st, as_of=as_of)
    ok = [COMPLETE, DELISTED, ENDED]
    keys = ["ticker", "start_date", "horizon_years"]
    tcols = [touch_col(m) for m in multiples]
    both = (daily[daily["status"].isin(ok)][keys + tcols]
            .merge(qtr[qtr["status"].isin(ok)][keys + tcols], on=keys, suffixes=("_d", "_q")))
    rows = []
    for h in horizons:
        b = both[both["horizon_years"] == h]
        for m in multiples:
            d, q = b[f"{touch_col(m)}_d"].mean(), b[f"{touch_col(m)}_q"].mean()
            rows.append({"horizon_years": int(h), "multiple": float(m), "windows": int(len(b)),
                         "quarter_end_over_daily": round(float(q / d), 3) if d and d > 0 else np.nan})
    return pd.DataFrame(rows, columns=cols)


def base_rates(outcomes: pd.DataFrame, multiples=(3.0, 10.0), group_cols=("horizon_years",)) -> pd.DataFrame:
    """Share of resolved windows that touched each multiple (censored/no-entry excluded)."""
    resolved = outcomes[outcomes["status"].isin([COMPLETE, DELISTED, ENDED])]
    cols = {touch_col(m): "mean" for m in multiples}
    cols["ticker"] = "count"
    out = resolved.groupby(list(group_cols)).agg(cols).rename(columns={"ticker": "n_windows"})
    return out.reset_index()


H1_MULTIPLES = (3.0, 10.0)
H1_HORIZONS = (3, 5)
H1_BUCKET_HORIZON = 5
REGIME_SPLIT = "2022-01-01"


def h1_tables(outcomes: pd.DataFrame, spec: OutcomeSpec | None = None, *, multiples=H1_MULTIPLES,
              horizons=H1_HORIZONS, bucket_horizon: int = H1_BUCKET_HORIZON,
              regime_split=REGIME_SPLIT) -> dict:
    """H1 as pre-registered: the share of resolved windows (complete, delisted, series ended;
    censored and no-entry excluded) that touched 3x and 10x within 3 and 5 years, and the
    distribution of five-year end multiples on the six-bucket grid with its mean (and the
    means of its delisting bounds) – pooled, by start year, and by regime (start dates
    before 2022 versus 2022 and later). A firm is a CIK (a series id without its ~segment).

    Returns {"touch": scope, group, horizon_years, windows, firms, touch_<m>x ...;
             "buckets": scope, group, windows, firms, share_<label>x ..., mean, mean_lower, mean_upper}."""
    spec = spec or OutcomeSpec()
    o = outcomes[outcomes["status"].isin([COMPLETE, DELISTED, ENDED])].copy()
    o["firm"] = o["ticker"].astype(str).str.split("~").str[0]
    start = _ns(o["start_date"])
    o["start_year"] = start.dt.year.astype("Int64").astype(str)
    o["regime"] = np.where(start < pd.Timestamp(regime_split), "before 2022", "2022 and later")
    scopes = (("pooled", None), ("regime", "regime"), ("start year", "start_year"))
    touch, buckets = [], []
    for scope, key in scopes:
        groups = [("all", o)] if key is None else sorted(o.groupby(key), key=lambda kv: str(kv[0]))
        for g, d in groups:
            for h in horizons:
                x = d[d["horizon_years"] == h]
                if not len(x):
                    continue
                row = {"scope": scope, "group": str(g), "horizon_years": int(h), "windows": int(len(x)),
                       "firms": int(x["firm"].nunique())}
                for m in multiples:
                    row[touch_col(m)] = round(float(pd.to_numeric(x[touch_col(m)], errors="coerce").mean()), 4)
                touch.append(row)
            x = d[d["horizon_years"] == bucket_horizon]
            if len(x):
                row = {"scope": scope, "group": str(g), "windows": int(len(x)), "firms": int(x["firm"].nunique())}
                b = pd.to_numeric(x["bucket"], errors="coerce")
                for lab in spec.bucket_labels:
                    row[f"share_{lab:g}x"] = round(float((b == lab).mean()), 4)
                for col, name in (("end_multiple", "mean"), ("end_multiple_lower", "mean_lower"),
                                  ("end_multiple_upper", "mean_upper")):
                    row[name] = round(float(pd.to_numeric(x[col], errors="coerce").mean()), 4)
                buckets.append(row)
    return {"touch": pd.DataFrame(touch), "buckets": pd.DataFrame(buckets)}


def panel_as_of(panel: pd.DataFrame) -> pd.Timestamp | None:
    """The earliest of the sources' last dates (pre-registered price_panel.as_of).

    Only sources that run to within a year of the latest date count, so a gap-fill source
    holding nothing but delisted names cannot drag the as-of date back."""
    if panel is None or panel.empty:
        return None
    if "source" not in panel.columns:
        return pd.Timestamp(_ns(panel["date"]).max())
    last = pd.DataFrame({"source": panel["source"].astype(str), "date": _ns(panel["date"])}).groupby("source")["date"].max()
    if not len(last):
        return None
    live = last[last >= last.max() - pd.Timedelta(days=365)]
    return pd.Timestamp(live.min())


def build_price_panel(simfin_prices: pd.DataFrame | None, stooq_prices: pd.DataFrame | None,
                      simfin_col: str = "adj_close", *, keep_volume: bool = True,
                      gap_fill: pd.DataFrame | None = None) -> pd.DataFrame:
    """One price series per ticker: SimFin where it has the ticker, Stooq for the rest.

    No mixing inside a ticker. Known limitation: a symbol reused by a different company
    on Stooq is only used when SimFin lacks that symbol entirely. Volume and the vendor
    share count ride along when the source has them (dollar volume and turnover).
    """
    parts = []
    taken: set = set()
    if simfin_prices is not None and len(simfin_prices):
        col = simfin_col if simfin_col in simfin_prices.columns else "close"
        extra = [c for c in ("volume", "shares_out") if keep_volume and c in simfin_prices.columns]
        sp = simfin_prices[["ticker", "date", col] + extra].rename(columns={col: "price"}).copy()
        if col != "close" and "close" in simfin_prices.columns:
            sp["price"] = sp["price"].fillna(simfin_prices["close"])
        if keep_volume and "close" in simfin_prices.columns:
            sp["close_raw"] = simfin_prices["close"]      # the vendor's close, for size and price filters
        sp["ticker"] = sp["ticker"].astype(str)
        sp["source"] = "simfin"
        parts.append(sp)
        taken = {str(t).split("~", 1)[0] for t in sp["ticker"].unique()}   # base symbols
    if stooq_prices is not None and len(stooq_prices):
        extra = [c for c in ("volume",) if keep_volume and c in stooq_prices.columns]
        qp = stooq_prices[["ticker", "date", "close"] + extra].rename(columns={"close": "price"}).copy()
        if keep_volume:
            qp["close_raw"] = qp["price"]
        qp["ticker"] = qp["ticker"].astype(str)
        qp = qp[~qp["ticker"].isin(taken)]
        qp["source"] = "stooq"
        parts.append(qp)
    if gap_fill is not None and len(gap_fill):
        # series already in panel form, under their own ticker names (EODHD: "<CODE>~eod")
        cols = ["ticker", "date", "price", "source"] + [c for c in ("close_raw", "volume")
                                                        if keep_volume and c in gap_fill.columns]
        ex = gap_fill[cols].copy()
        ex["ticker"] = ex["ticker"].astype(str)
        parts.append(ex)
    if not parts:
        return pd.DataFrame(columns=["ticker", "date", "price", "source"])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = _ns(out["date"])
    out["price"] = out["price"].astype("float32")
    for c in ("volume", "shares_out", "close_raw"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def build_starts(panel: pd.DataFrame, start_years, *, min_price: float = 1.0,
                 tolerance_days: int = 7, months=(1,)) -> pd.DataFrame:
    """Minimal start universe: names with a price >= min_price in the first trading days of
    each formation month (January by default) of each start year. The full universe
    filters are in features.assemble."""
    rows = []
    for y in range(int(start_years[0]), int(start_years[1]) + 1):
        for mo in months:
            s = pd.Timestamp(year=y, month=int(mo), day=1)
            win = panel[(panel["date"] >= s) & (panel["date"] <= s + pd.Timedelta(days=tolerance_days))]
            first = win.sort_values("date").groupby("ticker", sort=False, observed=True).head(1)
            first = first[first["price"] >= min_price]
            rows.append(pd.DataFrame({"ticker": first["ticker"].astype(str).to_numpy(), "start_date": s}))
    if not rows:
        return pd.DataFrame(columns=["ticker", "start_date"])
    out = pd.concat(rows, ignore_index=True)
    out["start_date"] = _ns(out["start_date"])
    return out


def starts_from_features(features: pd.DataFrame) -> pd.DataFrame:
    """The study's start universe: rows that pass the pre-registered universe filters."""
    u = features[features["in_universe"]]
    return u[["ticker", "start_date"]].drop_duplicates().reset_index(drop=True)
