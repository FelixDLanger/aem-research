"""EODHD gap-fill: end-of-day prices for the US companies SimFin and Stooq miss, and the
listing exchange of US tickers (for D1's exchange-specific delisting returns).

Why: the first real run (Nov 8, 2026) found that 31.8% of the filers that look like
universe members at the first start date never appear in SimFin or Stooq – companies
that later delisted. Base rates measured on survivors alone come out too high.

Licence (eodhd.com/help/storage and /help/personal-use, read Nov 8, 2026): personal plans
cover private research, and data downloaded while subscribed may be kept for private,
non-commercial use after cancelling. Publication is "a different use case", with an
academic licence route – confirm with EODHD before Paper B is published. Raw prices
stay on Drive; nothing from EODHD enters the repository.

Endpoints (the API key comes from the Colab secret EODHD_API_KEY and never appears in
code, logs or errors):
  GET https://eodhd.com/api/exchange-symbol-list/US?api_token=…&fmt=json[&delisted=1]
      → [{Code, Name, Country, Exchange, Currency, Type, Isin}]
  GET https://eodhd.com/api/eod/{CODE}.US?api_token=…&from=YYYY-MM-DD&fmt=json
      → [{date, open, high, low, close, adjusted_close, volume}]
      close is as traded (not adjusted); adjusted_close adjusts for splits and dividends;
      a delisted ticker keeps its history (EODHD's example: TWTR to October 2022).

What is fetched: every common stock on a US exchange (listed or delisted) whose symbol
the SEC symbol bridges tie to an XBRL filer that SimFin does not price. Series enter the
panel as "<CODE>~eod"; where the same company is also priced by SimFin or Stooq on a
start date, the other source wins (dedupe). Downloads are saved in chunks of 250
tickers with an index, so a disconnect loses little and later runs read Drive only.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd

from ..env import colab_secret
from ..io_utils import load_table, save_table
from ..tickers import canonical_series

BASE = "https://eodhd.com/api"
SUFFIX = "~eod"
US_EXCHANGES = {"NASDAQ", "NYSE", "NYSE MKT", "NYSE AMERICAN", "AMEX", "NYSE ARCA", "BATS"}
CHUNK = 250
SYMBOL_COLUMNS = ["code", "name", "exchange", "type", "isin", "delisted"]


class EodhdAccessError(RuntimeError):
    """The key was refused (401/403): wrong key, or the plan lacks US end-of-day data."""


def resolve_key() -> str | None:
    value = colab_secret("EODHD_API_KEY") or os.environ.get("EODHD_API_KEY")
    return value.strip() if value else None


class Client:
    def __init__(self, key: str, session=None, min_interval: float = 0.05, timeout: int = 60):
        import requests
        self._key = key
        self.session = session or requests.Session()
        self.min_interval, self.timeout, self._last = float(min_interval), timeout, 0.0
        self.calls = 0

    def get_json(self, path: str, **params):
        """Parsed JSON, or None on 404. The key never appears in an error message."""
        q = {"api_token": self._key, "fmt": "json", **params}
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            r = self.session.get(f"{BASE}/{path}", params=q, timeout=self.timeout)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as exc:
                    raise ValueError(f"EODHD {path}: not JSON") from exc
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise EodhdAccessError(f"EODHD refused the key ({r.status_code}) for {path} – check the EODHD_API_KEY "
                                       "secret and that the plan includes US end-of-day data")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 5 * (attempt + 1)))
                continue
            raise RuntimeError(f"EODHD {path}: HTTP {r.status_code}")
        raise RuntimeError(f"EODHD {path}: gave up after repeated rate limits or server errors")


def _symbols_frame(rows, delisted: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    out = pd.DataFrame({"code": df.get("Code", pd.Series(dtype=str)).astype(str),
                        "name": df.get("Name", pd.Series(dtype=str)),
                        "exchange": df.get("Exchange", pd.Series(dtype=object)),
                        "type": df.get("Type", pd.Series(dtype=object)),
                        "isin": df.get("Isin", pd.Series(dtype=object))})
    out["delisted"] = bool(delisted)
    return out[SYMBOL_COLUMNS]


def symbol_lists(client=None, cache_dir=None, *, offline: bool = False, max_age_days: int = 30,
                 today: date | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    """US tickers, listed and delisted (two calls), cached on Drive for `max_age_days`."""
    cache = Path(cache_dir) / "eodhd_symbols" if cache_dir is not None else None
    mark = Path(cache_dir) / "eodhd_symbols.json" if cache_dir is not None else None
    today = pd.Timestamp(today or date.today())
    if cache is not None and mark.exists():
        meta = json.loads(mark.read_text(encoding="utf-8"))
        fresh = today - pd.Timestamp(meta["fetched"]) <= pd.Timedelta(days=max_age_days)
        t = load_table(cache)
        if t is not None and (fresh or offline or client is None):
            return t, {"symbols": int(len(t)), "source": f"cached {meta['fetched']}"}
    if offline or client is None:
        return pd.DataFrame(columns=SYMBOL_COLUMNS), {"symbols": 0, "source": "missing"}
    listed = _symbols_frame(client.get_json("exchange-symbol-list/US"), False)
    gone = _symbols_frame(client.get_json("exchange-symbol-list/US", delisted=1), True)
    t = pd.concat([listed, gone], ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_table(t, cache)
        mark.write_text(json.dumps({"fetched": str(today.date()), "rows": int(len(t))}), encoding="utf-8")
    if log:
        log(f"  EODHD symbol lists: {len(listed):,} listed, {len(gone):,} delisted")
    return t, {"symbols": int(len(t)), "source": "downloaded"}


def exchange_label(name) -> str:
    n = str(name or "").upper()
    if "NASDAQ" in n:
        return "nasdaq"
    if n.startswith("NYSE") or n in ("AMEX", "NYSE MKT", "NYSE AMERICAN"):
        return "nyse_amex"
    return "other"


def exchange_map(symbols: pd.DataFrame) -> pd.DataFrame:
    """ticker → nasdaq / nyse_amex for the delisting returns; a delisted entry wins over a
    listed one with the same symbol, since only ended series are classified."""
    if symbols is None or not len(symbols):
        return pd.DataFrame(columns=["ticker", "exchange"])
    s = symbols[symbols["type"].astype(str).eq("Common Stock")].copy()
    s["label"] = s["exchange"].map(exchange_label)
    s = s[s["label"].isin(["nasdaq", "nyse_amex"])].sort_values("delisted", ascending=False)
    s = s.drop_duplicates("code")
    return pd.DataFrame({"ticker": s["code"].astype(str), "exchange": s["label"]}).reset_index(drop=True)


def candidates(symbols: pd.DataFrame, bridges: list, xbrl_ciks, simfin_ciks=()) -> pd.DataFrame:
    """Common stocks on US exchanges whose symbol the SEC bridges tie to an XBRL filer that
    SimFin does not price. Columns: code, delisted, ciks (the bridge's candidates)."""
    cols = ["code", "delisted", "ciks"]
    usable = [b for b in bridges if b is not None and len(b)]
    if symbols is None or not len(symbols) or not usable:
        return pd.DataFrame(columns=cols)
    s = symbols[symbols["type"].astype(str).eq("Common Stock")].copy()
    ex = s["exchange"].astype(str).str.upper()
    s = s[ex.isin(US_EXCHANGES) | (s["delisted"].astype(bool) & s["exchange"].isna())]
    s["symbol"] = canonical_series(s["code"].astype(str))
    b = pd.concat(usable, ignore_index=True)[["symbol", "issuer_cik"]].dropna()
    b["symbol"] = canonical_series(b["symbol"])
    b["issuer_cik"] = pd.to_numeric(b["issuer_cik"], errors="coerce").astype("Int64")
    m = s.merge(b.drop_duplicates(), on="symbol")
    xbrl = set(pd.to_numeric(pd.Series(list(xbrl_ciks)), errors="coerce").dropna().astype("int64"))
    sim = set(pd.to_numeric(pd.Series(list(simfin_ciks)), errors="coerce").dropna().astype("int64"))
    m = m[m["issuer_cik"].astype("int64").isin(xbrl)]
    g = m.groupby(["code", "delisted"])["issuer_cik"].agg(lambda x: sorted(set(int(v) for v in x))).reset_index()
    g = g[g["issuer_cik"].map(lambda c: not set(c) <= sim)]          # every candidate CIK already in SimFin: skip
    g = g.rename(columns={"issuer_cik": "ciks"}).sort_values(["delisted", "code"], ascending=[False, True])
    return g.drop_duplicates("code")[cols].reset_index(drop=True)


def cached(cache_dir) -> bool:
    """Whether EODHD data from an earlier run are on Drive (usable without a key)."""
    d = Path(cache_dir)
    return any((d / f"eodhd_index{ext}").exists() for ext in (".parquet", ".pkl"))


def _index(cache_dir: Path) -> pd.DataFrame:
    t = load_table(cache_dir / "eodhd_index")
    return t if t is not None else pd.DataFrame(columns=["code", "chunk", "rows", "first_date", "last_date"])


def fetch_eod(codes, client=None, cache_dir=None, *, offline: bool = False, start: str = "2008-01-01",
              log=print) -> tuple[pd.DataFrame, dict]:
    """End-of-day history per code, cached on Drive in chunks. Returns panel rows:
    ticker (<CODE>~eod), date, price (adjusted close), close_raw (as traded), volume, source."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx = _index(cache_dir)
    done = set(idx["code"].astype(str))
    todo = [c for c in dict.fromkeys(str(c) for c in codes) if c not in done]
    stats = {"wanted": len(set(map(str, codes))), "cached": len(set(map(str, codes)) & done), "fetched": 0,
             "empty": 0, "missing": 0}
    if todo and (offline or client is None):
        stats["missing"] = len(todo)
        todo = []
    chunk_no = int(idx["chunk"].max()) + 1 if len(idx) else 0
    buf, buf_idx = [], []

    def flush():
        nonlocal buf, buf_idx, chunk_no, idx
        if buf_idx:
            frame = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame(columns=["code", "date", "close",
                                                                                         "adjusted_close", "volume"])
            save_table(frame, cache_dir / f"eodhd_prices_{chunk_no:04d}")
            idx = pd.concat([idx, pd.DataFrame(buf_idx)], ignore_index=True)
            save_table(idx, cache_dir / "eodhd_index")       # the index last: a chunk is only listed once saved
            chunk_no += 1
            buf, buf_idx = [], []

    for i, code in enumerate(todo, 1):
        rows = client.get_json(f"eod/{code}.US", **{"from": start}) or []
        df = pd.DataFrame(rows)
        if len(df) and {"date", "close", "adjusted_close"} <= set(df.columns):
            df = df[["date", "close", "adjusted_close"] + (["volume"] if "volume" in df.columns else [])].copy()
            df.insert(0, "code", code)
            buf.append(df)
            buf_idx.append({"code": code, "chunk": chunk_no, "rows": int(len(df)),
                            "first_date": str(df["date"].min()), "last_date": str(df["date"].max())})
            stats["fetched"] += 1
        else:
            buf_idx.append({"code": code, "chunk": -1, "rows": 0, "first_date": None, "last_date": None})
            stats["empty"] += 1
        if len(buf_idx) >= CHUNK:
            flush()
            if log:
                log(f"  EODHD: {i:,} of {len(todo):,} tickers downloaded")
    flush()
    want = set(map(str, codes))
    use = idx[idx["code"].astype(str).isin(want) & (idx["chunk"] >= 0)]
    parts = []
    for ch in sorted(use["chunk"].astype(int).unique()):
        t = load_table(cache_dir / f"eodhd_prices_{ch:04d}")
        if t is not None and len(t):
            parts.append(t[t["code"].astype(str).isin(want)])
    if not parts:
        return pd.DataFrame(columns=["ticker", "date", "price", "close_raw", "volume", "source"]), stats
    p = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({"ticker": p["code"].astype(str) + SUFFIX,
                        "date": pd.to_datetime(p["date"], errors="coerce").astype("datetime64[ns]"),
                        "price": pd.to_numeric(p["adjusted_close"], errors="coerce"),
                        "close_raw": pd.to_numeric(p["close"], errors="coerce"),
                        "volume": pd.to_numeric(p.get("volume"), errors="coerce") if "volume" in p.columns else float("nan"),
                        "source": "eodhd"})
    out = out.dropna(subset=["date", "price"])
    out = out[out["price"] > 0]
    stats["tickers"] = int(out["ticker"].nunique())
    stats["rows"] = int(len(out))
    return out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True), stats


def dedupe(pf: pd.DataFrame, keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop EODHD ticker-dates whose company is already priced by SimFin or Stooq that day."""
    if pf is None or not len(pf) or keys is None or not len(keys):
        return pf, keys
    k = keys.copy()
    k["is_eod"] = k["ticker"].astype(str).str.endswith(SUFFIX)
    other = set(map(tuple, k.loc[~k["is_eod"] & k["cik"].notna(), ["cik", "start_date"]].astype(object).to_numpy()))
    dup = k["is_eod"] & k["cik"].notna() & pd.Series([(c, d) in other for c, d in zip(k["cik"].astype(object),
                                                                                     k["start_date"])], index=k.index)
    drop = set(map(tuple, k.loc[dup, ["ticker", "start_date"]].to_numpy()))
    if not drop:
        return pf, keys
    keep_pf = [(t, d) not in drop for t, d in zip(pf["ticker"], pf["start_date"])]
    return pf[keep_pf].reset_index(drop=True), k.loc[~dup].drop(columns="is_eod").reset_index(drop=True)
