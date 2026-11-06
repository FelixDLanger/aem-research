"""SEC structured data sets: download, reduce, cache.

Three of the SEC's free bulk data sets feed the point-in-time features:
  * Financial Statement Data Sets (XBRL facts from 10-K and 10-Q filings, from 2009Q2)
  * Insider Transactions Data Sets (Forms 3, 4 and 5, from 2006Q1)
  * Form 13F Data Sets (institutional holdings, XML filings from May 2013)

Each zip is fetched into memory, reduced by an extract function to the few columns
the study needs, and saved to Drive as small tables plus a marker file. A rerun reads
the extracts and downloads only what is missing, so an interrupted first run resumes
where it stopped. Raw zips are never written to Drive.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..io_utils import load_table, save_table
from .sec import SecAccessError, quarters

FSDS_URLS = ("https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{q}.zip",)
INSIDER_URLS = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{q}_form345.zip",
    "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{year}q{q}_form345.zip",
)
F13_PREFIXES = (
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/",
    "https://www.sec.gov/files/datastandardsinnovation/data/form-13f-data-sets/",
)
ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")


# ---------------------------------------------------------------------------
# Which files exist
# ---------------------------------------------------------------------------

def quarter_items(templates: Iterable[str], start_year: int, end_year: int,
                  today: date | None = None) -> list[tuple[str, list[str]]]:
    """(label, candidate URLs) for every complete quarter, e.g. ('2013q1', [...])."""
    return [(f"{y}q{q}", [t.format(year=y, q=q) for t in templates])
            for y, q in quarters(start_year, end_year, today)]


def _month_end(year: int, month: int) -> date:
    return (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()


def f13_items(today: date | None = None, start: str = "2013q2") -> list[tuple[str, list[str]]]:
    """Form 13F data set files: calendar quarters 2013Q2-2023Q4, then (from 2024) files
    named by filing-date window: Jan-Feb 2024, then Mar-May, Jun-Aug, Sep-Nov, Dec-Feb."""
    today = today or date.today()
    names = [label for label, _ in quarter_items(["{year}q{q}"], 2013, 2023, today)
             if label >= start]
    names = [f"{n}_form13f.zip" for n in names]
    windows = [(date(2024, 1, 1), date(2024, 2, 29))]
    y, m = 2024, 3
    while True:
        end_y, end_m = (y, m + 2) if m + 2 <= 12 else (y + 1, m + 2 - 12)
        windows.append((date(y, m, 1), _month_end(end_y, end_m)))
        if windows[-1][1] >= today:
            break
        y, m = (y, m + 3) if m + 3 <= 12 else (y + 1, m + 3 - 12)
    for a, b in windows:
        if b < today:
            names.append(f"{a:%d%b%Y}-{b:%d%b%Y}".lower() + "_form13f.zip")
    return [(n.replace("_form13f.zip", ""), [p + n for p in F13_PREFIXES]) for n in names]


# ---------------------------------------------------------------------------
# Reading members of a zip
# ---------------------------------------------------------------------------

def member(zf: zipfile.ZipFile, stem: str) -> str | None:
    """Zip member whose base name (without extension) is `stem`, case-insensitive."""
    stem = stem.lower()
    names = [n for n in zf.namelist() if not n.endswith("/")]
    for n in names:
        base = n.rsplit("/", 1)[-1].lower()
        if base.rsplit(".", 1)[0] == stem:
            return n
    for n in names:
        if n.rsplit("/", 1)[-1].lower().startswith(stem):
            return n
    return None


def read_member(zf: zipfile.ZipFile, stem: str, columns: Iterable[str], *,
                chunk_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read the named columns of a tab-delimited member as strings, lower-case names.

    Missing columns come back empty, so older and newer layouts both work. Quotes are
    taken literally (the SEC files are not quoted), only empty fields count as missing,
    and malformed lines are skipped. `chunk_filter` reduces each chunk as it is read.
    """
    wanted = [c.lower() for c in columns]
    name = member(zf, stem)
    if name is None:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in wanted})
    with zf.open(name) as fh:
        reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline=""),
                             sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                             na_values=[""], on_bad_lines="skip", chunksize=chunksize,
                             usecols=lambda c: str(c).strip().lower() in wanted)
        parts = []
        for chunk in reader:
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            if chunk_filter is not None:
                chunk = chunk_filter(chunk)
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=wanted)
    for c in wanted:
        if c not in df.columns:
            df[c] = pd.Series(np.nan, index=df.index, dtype="object")
    return df[wanted]


def parse_dates(values) -> pd.Series:
    """SEC dates in any of their layouts: yyyymmdd, DD-MON-YYYY, ISO."""
    s = pd.Series(values, dtype="object").astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    for fmt in ("%d-%b-%Y", "ISO8601"):
        rest = out.isna() & s.ne("") & s.ne("nan")
        if not rest.any():
            break
        out.loc[rest] = pd.to_datetime(s[rest], format=fmt, errors="coerce")
    return out.astype("datetime64[ns]")


def to_int(values) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("Int64")


def valid_accession(values) -> pd.Series:
    return pd.Series(values, dtype="object").astype(str).str.strip().str.match(ACCESSION)


# ---------------------------------------------------------------------------
# Download, extract, cache
# ---------------------------------------------------------------------------

def _marker(cache_dir: Path, prefix: str, label: str) -> Path:
    return cache_dir / f"{prefix}_{label}.done.json"


def process(items: list[tuple[str, list[str]]], cache_dir: str | Path, prefix: str,
            extract: Callable[[zipfile.ZipFile], dict[str, pd.DataFrame]], *,
            client=None, offline: bool = False, log=print,
            local_dir: str | Path | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Run `extract` over every file, caching each file's tables on Drive.

    `local_dir` can hold zips downloaded by hand (same file names as on the SEC site);
    they are used before the network. Returns ({table name: all files concatenated},
    stats). A file the SEC does not have (404) is counted as missing and retried on
    the next run.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[pd.DataFrame]] = {}
    stats = {"files": len(items), "cached": 0, "downloaded": 0, "local": 0, "missing": 0, "bad": 0,
             "missing_labels": [], "bad_labels": []}
    for label, urls in items:
        mark = _marker(cache_dir, prefix, label)
        if mark.exists():
            names = json.loads(mark.read_text(encoding="utf-8")).get("tables", [])
            loaded = {n: load_table(cache_dir / f"{prefix}_{label}_{n}") for n in names}
            if all(v is not None for v in loaded.values()):
                for n, df in loaded.items():
                    tables.setdefault(n, []).append(df)
                stats["cached"] += 1
                continue
        raw, source = None, None
        if local_dir is not None:
            for u in urls:
                p = Path(local_dir) / u.rsplit("/", 1)[-1]
                if p.exists():
                    raw, source = p.read_bytes(), "local"
                    break
        if raw is None and not offline and client is not None:
            for u in urls:
                raw = client.get(u)          # None on 404; SecAccessError on 403
                if raw is not None:
                    source = "downloaded"
                    break
        if raw is None:
            stats["missing"] += 1
            stats["missing_labels"].append(label)
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                out = extract(zf)
        except zipfile.BadZipFile as exc:      # a truncated download: skip it, report it, retry next run
            stats["bad"] += 1
            stats["bad_labels"].append(label)
            if log:
                log(f"  {prefix} {label}: not a valid zip ({exc}) – skipped; it is retried on the next run")
            continue
        for n, df in out.items():
            save_table(df, cache_dir / f"{prefix}_{label}_{n}")
            tables.setdefault(n, []).append(df)
        mark.write_text(json.dumps({"tables": sorted(out), "rows": {n: int(len(d)) for n, d in out.items()},
                                    "source": source}), encoding="utf-8")
        stats[source] += 1
        if log:
            log(f"  {prefix} {label}: " + ", ".join(f"{n} {len(d):,}" for n, d in out.items()))
    merged = {n: pd.concat(parts, ignore_index=True) for n, parts in tables.items()}
    return merged, stats


__all__ = ["FSDS_URLS", "INSIDER_URLS", "F13_PREFIXES", "SecAccessError", "quarter_items",
           "f13_items", "member", "read_member", "parse_dates", "to_int", "valid_accession", "process"]
