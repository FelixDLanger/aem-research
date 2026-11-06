"""Read the SimFin data already cached on Drive by the CVM pipeline.

The CVM notebook saved raw SimFin frames as `cvm-research/data/raw/us_<kind>.parquet`
(prices keep SimFin's (Ticker, Date) index). The simfin library's own cache in
`cvm-research/data/` holds semicolon-separated CSVs such as `us-shareprices-daily.csv`.
Both layouts are accepted. Nothing here downloads data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..io_utils import read_any
from ..tickers import canonical_series

PRICE_FILES = (
    "us_prices.parquet", "us_prices.pkl",
    "us-shareprices-daily.csv", "us-shareprices-daily.zip", "us-shareprices-daily.csv.gz",
)
COMPANY_FILES = (
    "us_companies.parquet", "us_companies.pkl",
    "us-companies.csv", "us-companies.zip", "us-companies.csv.gz",
)

# Case-insensitive SimFin label -> snake_case (compatible with cvm_research.pit_data).
CANONICAL = {
    "ticker": "ticker", "simfinid": "simfin_id", "simfin id": "simfin_id",
    "company name": "company_name", "cik": "cik", "isin": "isin", "market": "market",
    "industryid": "industry_id", "date": "date", "open": "open", "high": "high",
    "low": "low", "close": "close", "adj. close": "adj_close", "adj close": "adj_close",
    "volume": "volume", "dividend": "dividend",
    "shares outstanding": "shares_out", "common shares outstanding": "shares_out",
    "publish date": "publish_date", "report date": "report_date",
}
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume", "shares_out")


@dataclass
class SimFinCache:
    prices: pd.DataFrame | None
    companies: pd.DataFrame | None
    files: dict
    notes: list = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.prices is not None and len(self.prices) > 0


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Promote named index levels to columns and rename SimFin labels to snake_case."""
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        target = CANONICAL.get(key)
        if target and (target not in df.columns or target == col):
            rename[col] = target
    return df.rename(columns=rename)


def discover(roots: Iterable[str | Path]) -> dict:
    roots = [Path(r) for r in roots]
    found = {"prices": None, "companies": None}
    for kind, names in (("prices", PRICE_FILES), ("companies", COMPANY_FILES)):
        for root in roots:
            hit = next((root / n for n in names if (root / n).exists()), None)
            if hit is not None:
                found[kind] = hit
                break
    return found


def _cik_to_int(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")


def tidy_prices(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["ticker", "date"] + [c for c in PRICE_COLUMNS if c in df.columns]
    if "simfin_id" in df.columns:
        keep.append("simfin_id")
    out = df[keep].copy()
    out = out.dropna(subset=["ticker", "date"])
    # Canonicalise each distinct ticker once (the price table has ~20M rows).
    codes, uniques = pd.factorize(out["ticker"].astype(str), sort=False)
    canon = canonical_series(pd.Series(uniques)).to_numpy()
    cats, inverse = np.unique(canon, return_inverse=True)
    out["ticker"] = pd.Categorical.from_codes(inverse[codes], categories=cats)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").astype("datetime64[ns]")
    out = out.dropna(subset=["date"])
    for col in PRICE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    key = ["ticker", "simfin_id", "date"] if "simfin_id" in out.columns else ["ticker", "date"]
    out = out.sort_values(key, kind="mergesort").drop_duplicates(key, keep="last")
    return out.reset_index(drop=True)


def disambiguate(prices: pd.DataFrame, companies: pd.DataFrame | None) -> tuple:
    """Split tickers that SimFin uses for more than one company (symbol reuse).

    Rows of a reused ticker are relabelled 'TICKER~SimFinId' in prices and companies, so
    two companies are never merged into one series – a merge would fabricate jumps
    (e.g. an old company's last $0.50 followed by a new company's $50)."""
    if prices is None or "simfin_id" not in prices.columns:
        return prices, companies, 0
    ids = prices.groupby("ticker", observed=True)["simfin_id"].nunique()
    dup = set(ids[ids > 1].index.astype(str))
    if not dup:
        return prices, companies, 0
    mask = prices["ticker"].astype(str).isin(dup).to_numpy()
    sub = prices.loc[mask, ["ticker", "simfin_id"]]
    labels = sub["ticker"].astype(str) + "~" + sub["simfin_id"].astype("Int64").astype(str)
    base = [str(c) for c in prices["ticker"].cat.categories]
    tick = prices["ticker"].cat.set_categories(base + sorted(set(labels) - set(base)))
    tick = tick.copy()
    tick[mask] = labels.to_numpy()
    prices = prices.assign(ticker=tick.cat.remove_unused_categories())
    if companies is not None and {"ticker", "simfin_id"} <= set(companies.columns):
        ct = companies["ticker"].astype(str)
        cm = ct.isin(dup)
        relabel = ct + "~" + companies["simfin_id"].astype("Int64").astype(str)
        companies = companies.assign(ticker=ct.where(~cm, relabel))
    return prices, companies, len(dup)


def tidy_companies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ticker" in out.columns:
        out = out.dropna(subset=["ticker"])
        out["ticker"] = canonical_series(out["ticker"])
    if "cik" in out.columns:
        out["cik"] = _cik_to_int(out["cik"])
    return out.reset_index(drop=True)


def load(roots: Iterable[str | Path]) -> SimFinCache:
    files = discover(roots)
    notes: list[str] = []
    companies = None
    if files["companies"] is not None:
        companies = tidy_companies(normalise(read_any(files["companies"])))
        if "cik" not in companies.columns:
            notes.append("companies table has no CIK column – the SEC cross-check falls back to counts")
    else:
        notes.append("no US companies file found in the SimFin cache")
    prices = None
    if files["prices"] is not None:
        raw = normalise(read_any(files["prices"]))
        if "ticker" not in raw.columns and "simfin_id" in raw.columns and companies is not None \
                and {"simfin_id", "ticker"} <= set(companies.columns):
            raw = raw.merge(companies[["simfin_id", "ticker"]].drop_duplicates("simfin_id"),
                            on="simfin_id", how="left")
            notes.append("prices had no ticker column – joined via SimFinId")
        if {"ticker", "date"} <= set(raw.columns):
            prices = tidy_prices(raw)
            prices, companies, n_dup = disambiguate(prices, companies)
            if n_dup:
                notes.append(f"{n_dup} tickers belong to more than one SimFin company – split as TICKER~SimFinId")
        else:
            notes.append(f"prices file lacks ticker/date columns: {list(raw.columns)[:12]}")
    else:
        notes.append("no US share-price file found in the SimFin cache")
    return SimFinCache(prices=prices, companies=companies,
                       files={k: (str(v) if v else None) for k, v in files.items()}, notes=notes)


def best_price_column(prices: pd.DataFrame) -> str:
    if "adj_close" in prices.columns and prices["adj_close"].notna().mean() > 0.5:
        return "adj_close"
    return "close"
