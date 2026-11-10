"""EDGAR full-text search: phrase search across every electronic filing since 2001.

    GET https://efts.sec.gov/LATEST/search-index?q=...&forms=...&startdt=YYYY-MM-DD&enddt=YYYY-MM-DD&from=N

The response is Elasticsearch-style JSON: `hits.total.value` (with `relation` "eq" or
"gte"), and `hits.hits[]` with `_id` = "<accession>:<file name>" and `_source` holding
`ciks`, `display_names`, `form`, `root_forms`, `file_type`, `file_date`, `sics`. A page
holds at most 100 hits and a query at most 10,000, so long ranges are searched one
calendar year at a time and a window that reaches the cap is split in half. The same
SEC fair-access rules apply as everywhere else: a User-Agent and at most 10 requests a
second (the SecClient handles both).

Two caches on Drive:
  * `search` – one table per (query, forms, window); windows that ended more than 30 days
    ago are final and never fetched again;
  * `search_many` – many small queries (one per deal) in one table, saved every 100
    queries, so a disconnect loses little.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from ..io_utils import load_table, save_table

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
MAX_HITS = 10_000
FINAL_AFTER_DAYS = 30
HIT_COLUMNS = ["query", "adsh", "file_name", "ciks", "names", "form", "root_form", "file_type", "file_date", "sics"]


def build_url(q: str, forms=(), start=None, end=None, offset: int = 0, ciks=None) -> str:
    params = [("q", q)]
    if forms:
        params.append(("forms", ",".join(forms)))
    if start is not None or end is not None:
        params.append(("dateRange", "custom"))         # the search page sends it with explicit dates
    if start is not None:
        params.append(("startdt", str(pd.Timestamp(start).date())))
    if end is not None:
        params.append(("enddt", str(pd.Timestamp(end).date())))
    if offset:
        params.append(("from", str(int(offset))))
    if ciks is not None:
        params.append(("ciks", f"{int(ciks):010d}"))
    return FTS_URL + "?" + "&".join(f"{k}={quote(str(v), safe=',')}" for k, v in params)


def parse_page(raw) -> tuple[list[dict], int, str]:
    """(hits, total, relation) from one response page; ValueError if it is not a search result."""
    try:
        data = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    except (ValueError, UnicodeDecodeError) as exc:
        head = raw[:160] if isinstance(raw, (bytes, bytearray, str)) else raw
        raise ValueError(f"full-text search did not return JSON ({exc}): {head!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("hits"), dict):
        raise ValueError(f"full-text search returned no hits object: {str(data)[:200]}")
    hits = data["hits"]
    total = hits.get("total", 0)
    if isinstance(total, dict):
        value, relation = int(total.get("value") or 0), str(total.get("relation") or "eq")
    else:
        value, relation = int(total or 0), "eq"
    rows = []
    for h in hits.get("hits") or []:
        src = h.get("_source") or {}
        adsh_id, _, fname = str(h.get("_id") or "").partition(":")
        roots = src.get("root_forms") or []
        ciks = [int(c) for c in (src.get("ciks") or []) if str(c).strip().isdigit()]
        rows.append({"adsh": str(src.get("adsh") or adsh_id), "file_name": fname,
                     "ciks": ";".join(str(c) for c in ciks),
                     "names": " | ".join(str(n) for n in (src.get("display_names") or [])),
                     "form": src.get("form") or (roots[0] if roots else None),
                     "root_form": roots[0] if roots else src.get("form"),
                     "file_type": src.get("file_type"), "file_date": src.get("file_date"),
                     "sics": ";".join(str(s) for s in (src.get("sics") or []))})
    return rows, value, relation


def _frame(rows: list[dict], query: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[c for c in HIT_COLUMNS if c != "query"])
    df.insert(0, "query", query)
    df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce").astype("datetime64[ns]")
    return df


def fetch(client, q: str, forms=(), start=None, end=None, *, max_hits: int = MAX_HITS, ciks=None):
    """All pages of one query: (hits frame, total, relation, pages). Hits outside [start, end]
    are dropped, so the result is right even if the endpoint ignored the dates."""
    rows, offset, pages, total, relation = [], 0, 0, 0, "eq"
    while True:
        raw = client.get(build_url(q, forms, start, end, offset, ciks))
        if raw is None:
            raise ValueError("full-text search answered 404 – the endpoint may have moved")
        page, total, relation = parse_page(raw)
        pages += 1
        rows.extend(page)
        offset += len(page)
        if not page or offset >= min(total, max_hits):
            break
    out = _frame(rows, q)
    if start is not None:
        out = out[out["file_date"].isna() | (out["file_date"] >= pd.Timestamp(start).normalize())]
    if end is not None:
        out = out[out["file_date"].isna() | (out["file_date"] <= pd.Timestamp(end).normalize() + pd.Timedelta(days=1))]
    return out.reset_index(drop=True), total, relation, pages


def _key(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _windows(start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out = []
    for y in range(start.year, end.year + 1):
        a, b = max(start, pd.Timestamp(year=y, month=1, day=1)), min(end, pd.Timestamp(year=y, month=12, day=31))
        if a <= b:
            out.append((a, b))
    return out


def search(q: str, forms=(), start="2001-01-01", end=None, *, client=None, cache_dir=None,
           offline: bool = False, today: date | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    """Every hit of a query between `start` and `end`, one cached window per calendar year."""
    today = pd.Timestamp(today or date.today())
    end = pd.Timestamp(end) if end is not None else today
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    stats = {"windows": 0, "cached": 0, "fetched": 0, "pages": 0, "missing": 0, "split": 0, "capped": 0}
    parts = []

    def run(a, b, depth=0):
        stats["windows"] += 1
        key = _key(q, ",".join(forms), a.date(), b.date())
        mark = cache_dir / f"fts_{key}.done.json" if cache_dir is not None else None
        final = b < today - pd.Timedelta(days=FINAL_AFTER_DAYS)
        if mark is not None and mark.exists():
            meta = json.loads(mark.read_text(encoding="utf-8"))
            if meta.get("final") or offline or client is None:
                t = load_table(cache_dir / f"fts_{key}")
                if t is not None:
                    parts.append(t)
                    stats["cached"] += 1
                    return
        if offline or client is None:
            stats["missing"] += 1
            return
        hits, total, relation, pages = fetch(client, q, forms, a, b)
        stats["pages"] += pages
        if total >= MAX_HITS and (b - a).days >= 1 and depth < 12:
            stats["split"] += 1                   # past the cap: halve the window and search both halves
            mid = a + (b - a) / 2
            run(a, mid.normalize(), depth + 1)
            run(mid.normalize() + pd.Timedelta(days=1), b, depth + 1)
            return
        if total >= MAX_HITS:
            stats["capped"] += 1
        stats["fetched"] += 1
        parts.append(hits)
        if mark is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            save_table(hits, cache_dir / f"fts_{key}")
            mark.write_text(json.dumps({"q": q, "forms": list(forms), "start": str(a.date()), "end": str(b.date()),
                                        "total": total, "relation": relation, "final": bool(final)}), encoding="utf-8")
        if log:
            log(f"  search {a.date()} to {b.date()}: {len(hits):,} hits")

    for a, b in _windows(start, end):
        run(a, b)
    out = pd.concat(parts, ignore_index=True) if parts else _frame([], q)
    out = out.drop_duplicates(["adsh", "file_name"]).reset_index(drop=True)
    stats["hits"] = int(len(out))
    stats["filings"] = int(out["adsh"].nunique()) if len(out) else 0
    return out, stats


def search_many(queries: list[dict], *, client=None, cache_path=None, offline: bool = False,
                max_hits: int = 100, save_every: int = 100, log=print) -> tuple[pd.DataFrame, dict]:
    """Run many small queries ({"key", "q", "forms", "start", "end", optional "ciks"}), first page only by
    default; results are cached in one table keyed by `key` plus the query text."""
    cache_path = Path(cache_path) if cache_path is not None else None
    cached = load_table(cache_path) if cache_path is not None else None
    done = cached if cached is not None else pd.DataFrame(columns=["key", "qkey", *HIT_COLUMNS])
    seen = set(done["qkey"].astype(str)) if len(done) else set()
    new, stats = [], {"queries": len(queries), "cached": 0, "fetched": 0, "missing": 0}

    def flush():
        nonlocal done, new
        if new:
            done = pd.concat([done, *new], ignore_index=True)
            new = []
        if cache_path is not None:
            save_table(done, cache_path)

    def qkey_of(qd):                               # the filer filter joins the key only where it is used
        extra = [f"cik{int(qd['ciks'])}"] if qd.get("ciks") is not None else []
        return _key(qd["key"], qd["q"], ",".join(qd.get("forms", ())), qd.get("start"), qd.get("end"), *extra)

    for i, qd in enumerate(queries):
        qkey = qkey_of(qd)
        if qkey in seen:
            stats["cached"] += 1
            continue
        if offline or client is None:
            stats["missing"] += 1
            continue
        hits, total, _rel, _pages = fetch(client, qd["q"], qd.get("forms", ()), qd.get("start"), qd.get("end"),
                                         max_hits=max_hits, ciks=qd.get("ciks"))
        if hits.empty:                             # remember empty answers too
            hits = _frame([{"adsh": "", "file_name": "", "ciks": "", "names": "", "form": None,
                            "root_form": None, "file_type": None, "file_date": None, "sics": ""}], qd["q"])
        hits.insert(0, "qkey", qkey)
        hits.insert(0, "key", str(qd["key"]))
        new.append(hits)
        seen.add(qkey)
        stats["fetched"] += 1
        if stats["fetched"] % save_every == 0:
            flush()
            if log:
                log(f"  {stats['fetched']:,} of {len(queries):,} searches done")
    flush()
    keys = {qkey_of(qd) for qd in queries}
    out = done[done["qkey"].astype(str).isin(keys) & done["adsh"].astype(str).ne("")].reset_index(drop=True)
    if len(out):
        out["file_date"] = pd.to_datetime(out["file_date"], errors="coerce").astype("datetime64[ns]")
    return out, stats
