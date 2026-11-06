"""Stooq bulk daily data (d_us_txt.zip).

Stooq puts its bulk files behind a CAPTCHA, so the download itself cannot be
automated: download d_us_txt.zip once from https://stooq.com/db/h/ (Daily, ASCII,
U.S.) and save it to Drive under aem-research/data/stooq/. Everything after that
is automated: this module finds the newest zip, parses the stock folders, caches a
consolidated table, and re-parses only when the zip changes.

Member layout inside the zip: data/daily/us/<group>/<sub>/<symbol>.us.txt, with the
header <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>.
Stooq's bulk files are for personal use; never commit them or derived raw prices to
a public repository.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from ..io_utils import load_table, save_table
from ..tickers import canonical_ticker

DOWNLOAD_PAGE = "https://stooq.com/db/h/"
INSTRUCTIONS = (
    "Stooq bulk zip not found. One-time manual step (Stooq requires a CAPTCHA):\n"
    f"  1. Open {DOWNLOAD_PAGE} – row 'U.S.', column 'Daily', link 'ASCII' (file d_us_txt.zip).\n"
    "  2. Solve the CAPTCHA and download.\n"
    "  3. Save the zip to Google Drive: My Drive/aem-research/data/stooq/\n"
    "     (iPhone: Safari download -> Share -> Save to Files -> Google Drive).\n"
    "  4. Re-run this cell. Parsing and caching are automatic from here."
)
_COLS = {
    "<TICKER>": "ticker", "<PER>": "per", "<DATE>": "date", "<TIME>": "time",
    "<OPEN>": "open", "<HIGH>": "high", "<LOW>": "low", "<CLOSE>": "close",
    "<VOL>": "volume", "<OPENINT>": "openint",
}
EMPTY = pd.DataFrame({"ticker": pd.Series(dtype="category"),
                      "date": pd.Series(dtype="datetime64[ns]"),
                      "high": pd.Series(dtype="float32"),
                      "close": pd.Series(dtype="float32"),
                      "volume": pd.Series(dtype="float32")})


@dataclass
class StooqData:
    prices: pd.DataFrame
    groups: pd.DataFrame          # ticker -> folder group (e.g. 'nasdaq stocks')
    zip_path: str
    stats: dict = field(default_factory=dict)
    from_cache: bool = False


def find_bulk_zip(stooq_dir: str | Path) -> Path | None:
    stooq_dir = Path(stooq_dir)
    if not stooq_dir.exists():
        return None
    zips = [p for p in stooq_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".zip" and p.name.lower().startswith("d_us_txt")]
    return max(zips, key=lambda p: p.stat().st_mtime) if zips else None


def group_of(member: str) -> str:
    parts = PurePosixPath(member).parts
    try:
        i = parts.index("us")
    except ValueError:
        return "unknown"
    return parts[i + 1].lower() if len(parts) > i + 2 else "unknown"


def parse_member(raw: bytes, min_date=None) -> pd.DataFrame | None:
    """Parse one <symbol>.us.txt member; None when it holds no usable daily rows."""
    if not raw or len(raw) < 20:
        return None
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, on_bad_lines="skip")
    except Exception:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    if "<DATE>" not in df.columns or "<CLOSE>" not in df.columns:
        return None
    df = df.rename(columns=_COLS)
    if "per" in df.columns:
        df = df[df["per"].astype(str).str.strip().str.upper().eq("D")]
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if df.empty:
        return None
    high = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else np.nan
    vol = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else np.nan
    return pd.DataFrame({
        "ticker_raw": df["ticker"].astype(str).str.strip() if "ticker" in df.columns else "",
        "date": df["date"].astype("datetime64[ns]"),
        "high": pd.Series(high, index=df.index).astype("float32"),
        "close": df["close"].astype("float32"),
        "volume": pd.Series(vol, index=df.index).astype("float32"),
    })


def parse_bulk_zip(zip_path: str | Path, include=("stocks",), min_date="2005-01-01",
                   log_every: int = 2000, log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse the stock folders of a Stooq bulk zip into one long table."""
    frames, codes, tickers, groups = [], {}, [], {}
    stats = {"members": 0, "parsed": 0, "empty": 0, "skipped_group": 0}
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith(".txt")]
        stats["members"] = len(members)
        for n, m in enumerate(members, 1):
            grp = group_of(m.filename)
            if not any(k in grp for k in include):
                stats["skipped_group"] += 1
                continue
            df = parse_member(zf.read(m), min_date)
            if df is None:
                stats["empty"] += 1
                continue
            raw_t = str(df["ticker_raw"].iloc[0]).strip()
            if raw_t.lower() in ("", "nan", "none"):
                raw_t = PurePosixPath(m.filename).name[:-4]
            t = canonical_ticker(raw_t)
            if t not in codes:
                codes[t] = len(tickers)
                tickers.append(t)
                groups[t] = grp
            frames.append(pd.DataFrame({
                "code": np.full(len(df), codes[t], dtype=np.int32),
                "date": df["date"].to_numpy(), "high": df["high"].to_numpy(),
                "close": df["close"].to_numpy(), "volume": df["volume"].to_numpy()}))
            stats["parsed"] += 1
            if log and log_every and n % log_every == 0:
                log(f"  … {n:,}/{len(members):,} members read")
    if not frames:
        return EMPTY.copy(), pd.DataFrame(columns=["ticker", "group"]), stats
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "ticker", pd.Categorical.from_codes(out.pop("code"), categories=tickers))
    out["date"] = out["date"].astype("datetime64[ns]")
    out = out.sort_values(["ticker", "date"], kind="mergesort").drop_duplicates(["ticker", "date"], keep="last")
    grp = pd.DataFrame({"ticker": list(groups), "group": list(groups.values())})
    return out.reset_index(drop=True), grp, stats


def load_or_build(stooq_dir: str | Path, cache_dir: str | Path, *, include=("stocks",),
                  min_date="2005-01-01", log=print) -> StooqData | None:
    """Parse the newest bulk zip, or load the cached parse if the zip is unchanged."""
    zp = find_bulk_zip(stooq_dir)
    if zp is None:
        return None
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "stooq_us_daily.meta.json"
    fingerprint = {"zip": zp.name, "size": zp.stat().st_size, "include": list(include),
                   "min_date": str(min_date)}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            prices = load_table(cache_dir / "stooq_us_daily")
            groups = load_table(cache_dir / "stooq_us_groups")
            if prices is not None and groups is not None:
                if "ticker" in prices.columns:
                    prices["ticker"] = prices["ticker"].astype("category")
                return StooqData(prices, groups, str(zp), meta.get("stats", {}), from_cache=True)
    if log:
        log(f"Parsing {zp.name} ({zp.stat().st_size / 1e6:,.0f} MB) – first run takes a few minutes …")
    prices, groups, stats = parse_bulk_zip(zp, include=include, min_date=min_date, log=log)
    save_table(prices, cache_dir / "stooq_us_daily")
    save_table(groups, cache_dir / "stooq_us_groups")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"fingerprint": fingerprint, "stats": stats}, indent=2))
    return StooqData(prices, groups, str(zp), stats, from_cache=False)


def fetch_symbol_api(*_args, **_kwargs):
    """Per-symbol downloads are deliberately not implemented.

    Stooq's per-symbol CSV endpoint has required an API key since 2026 and its exact
    parameters are unverified. Use the bulk zip; for gaps, see data/eodhd.py.
    """
    raise NotImplementedError(fetch_symbol_api.__doc__)
