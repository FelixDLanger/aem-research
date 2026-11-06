"""FRED macro series for the regime moderators (free, no key).

Series (config: regime.moderators): DFII10 10-year real yield (daily), DGS2 2-year
Treasury yield (daily), BAA10Y Baa corporate spread over the 10-year Treasury (daily),
CPIAUCNS CPI for all urban consumers, not seasonally adjusted (monthly; the
unadjusted index is not revised, so today's file shows what was published then).

Each series is cached on Drive and refreshed when older than `max_age_days`. CSV files
saved by hand into the manual folder (named <SERIES>.csv, as FRED's download button
names them) are used when the network is not available.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import load_table, save_table

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SERIES = {"real_yield_10y": "DFII10", "yield_2y": "DGS2", "credit_spread_baa": "BAA10Y", "cpi_nsa": "CPIAUCNS"}
USER_AGENT = "aem-research/0.2 (academic research; github.com/FelixDLanger/aem-research)"


def parse_csv(raw: bytes | str) -> pd.Series:
    """FRED CSV (header DATE or observation_date, then the series id) -> float Series by date.
    Missing values ('.' or empty) are dropped."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError("not a FRED CSV: fewer than two columns")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    vals = pd.to_numeric(df.iloc[:, 1].str.strip().replace({".": ""}), errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(dates).astype("datetime64[ns]"), name=str(df.columns[1]))
    return s[s.index.notna() & s.notna()].sort_index()


def load(cache_dir: str | Path, series: dict | None = None, *, session=None, offline: bool = False,
         max_age_days: int = 7, manual_dir: str | Path | None = None, log=print) -> tuple[dict, dict]:
    """({name: Series}, stats). Network errors fall back to the cache, then to manual CSVs."""
    series = series or SERIES
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, stats = {}, {}
    for name, sid in series.items():
        base = cache_dir / f"fred_{sid}"
        cached = load_table(base)
        fresh = False
        if cached is not None:
            stamp = next((p for p in (base.with_suffix(".parquet"), base.with_suffix(".pkl")) if p.exists()), None)
            fresh = stamp is not None and (time.time() - stamp.stat().st_mtime) < max_age_days * 86400
        s, how = None, None
        if not offline and not fresh:
            try:
                import requests
                sess = session or requests.Session()
                r = sess.get(CSV_URL.format(series=sid), headers={"User-Agent": USER_AGENT}, timeout=60)
                if r.status_code == 200:
                    s, how = parse_csv(r.content), "downloaded"
                    save_table(pd.DataFrame({"date": s.index, "value": s.to_numpy()}), base)
                else:
                    stats[f"{sid}_http"] = r.status_code
            except Exception as exc:          # offline, DNS, timeout: use what is on Drive
                stats[f"{sid}_error"] = f"{type(exc).__name__}"
        if s is None and cached is not None:
            s, how = pd.Series(cached["value"].to_numpy(), index=pd.DatetimeIndex(cached["date"]).astype("datetime64[ns]"),
                               name=sid), "cached"
        if s is None and manual_dir is not None and (Path(manual_dir) / f"{sid}.csv").exists():
            s, how = parse_csv((Path(manual_dir) / f"{sid}.csv").read_bytes()), "manual"
        if s is not None and len(s):
            out[name] = s
            stats[sid] = f"{how}, {s.index.min().date()} to {s.index.max().date()}"
        else:
            stats[sid] = "missing"
        if log:
            log(f"  FRED {sid}: {stats[sid]}")
    return out, stats


def _last_before(s: pd.Series, when: pd.Timestamp) -> float:
    i = int(np.searchsorted(s.index.to_numpy(), np.datetime64(when, "ns"), side="left")) - 1
    return float(s.iloc[i]) if i >= 0 else np.nan


def regime_at(series: dict, formation_dates) -> pd.DataFrame:
    """Macro moderators known on each start date (pre-registered rules in config: regime)."""
    rows = []
    for d in pd.to_datetime(list(formation_dates)):
        row = {"start_date": d}
        if "real_yield_10y" in series:
            row["real_yield_10y"] = _last_before(series["real_yield_10y"], d)
        if "yield_2y" in series:
            now = _last_before(series["yield_2y"], d)
            then = _last_before(series["yield_2y"], d - pd.DateOffset(years=1))
            row["tightening_2y_12m"] = now - then
        if "credit_spread_baa" in series:
            row["credit_spread_baa"] = _last_before(series["credit_spread_baa"], d)
        if "cpi_nsa" in series:
            cpi = series["cpi_nsa"]
            m = (d - pd.DateOffset(months=2)).to_period("M").to_timestamp()   # latest release on the start date
            a, b = cpi.get(m, np.nan), cpi.get(m - pd.DateOffset(years=1), np.nan)
            row["inflation_cpi_yoy"] = float(a / b - 1) if np.isfinite(a) and np.isfinite(b) and b else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["start_date"] = pd.to_datetime(out["start_date"]).astype("datetime64[ns]")
    return out
