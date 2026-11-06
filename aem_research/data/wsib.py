"""Washington State Investment Board: the quarterly private equity table, fund by fund.

Each quarter WSIB publishes `ir{mmddyy}.pdf` ("Performance Summary By Strategy"), one row
per partnership, in whole dollars:

    Investment Name | Initial Investment Date | Capital Committed | Paid-In Capital (A) |
    Unfunded Commitment | Current Market Value (B) | Capital Distributed (1) (C) |
    Total Value (B+C) | Total Value Multiple (B+C)/A | Gain Since Inception (B+C)-A | Net IRR (2)

    Advent International GPE VII-B, L.P. 8/27/2012 400,000,000 384,020,122 16,000,020
        32,117,988 663,657,497 695,775,485 1.8x 311,755,363 13.14%

A lone "-" is zero, negatives are in parentheses. Each strategy ends with a line
"Subtotal <strategy> ..." – rows are assigned the strategy of the subtotal that follows
them, and the subtotal's capital committed is checked against the sum of its rows.
The initial investment date stands in for the vintage year.

Raw PDFs are kept on Drive, so a better parser re-reads them without downloading again
(PARSER_VERSION). Net IRR, market value, distributions and gains are outcomes for Paper A:
they are cached, never summarized before Paper A's lock.
"""
from __future__ import annotations

import io
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import load_table, save_table

URLS = ("https://www.sib.wa.gov/docs/reports/quarterly/ir{tag}.pdf",
        "https://sib.wa.gov/docs/reports/quarterly/ir{tag}.pdf")
USER_AGENT = "Mozilla/5.0 (compatible; aem-research/0.3; +https://github.com/FelixDLanger/aem-research)"
PARSER_VERSION = 1
VALUE_COLUMNS = ["committed", "paid_in", "unfunded", "market_value", "distributed", "total_value",
                 "multiple_printed", "gain", "net_irr"]
OUTCOME_COLUMNS = ["market_value", "distributed", "total_value", "multiple_printed", "gain", "net_irr", "tvpi"]
_DATE = re.compile(r"\s(\d{1,2}/\d{1,2}/(?:19|20)\d{2})\s")
_HEADER_WORDS = ("investment name", "capital committed", "performance summary", "washington state", "paid-in",
                 "market value", "total value", "gain since", "net irr", "unfunded", "contributions",
                 "distributions", "page ", "initial investment")
_CONTINUATION = re.compile(r"^(?:[a-z0-9(&]|L\.?P\b|LP\b|Fund\b|(?:[IVX]{1,5})\b)")


def report_dates(start_year: int, today: date | None = None) -> list[pd.Timestamp]:
    today = pd.Timestamp(today or date.today())
    out = []
    for y in range(int(start_year), today.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            ts = pd.Timestamp(year=y, month=m, day=d)
            if ts < today - pd.Timedelta(days=45):          # published about two months after quarter-end
                out.append(ts)
    return out


def file_name(report_date) -> str:
    return f"ir{pd.Timestamp(report_date):%m%d%y}.pdf"


def _amount(tok: str) -> float:
    t = tok.replace("$", "").replace(",", "").strip()
    if t in ("-", "–", ""):
        return 0.0
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
    except ValueError:
        return np.nan
    return -v if neg else v


def _irr(tok: str) -> float:
    t = tok.replace(",", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").rstrip("%")
    try:
        v = float(t) / 100.0
    except ValueError:
        return np.nan                                        # "N/M", "NM", "N/A"
    return -v if neg else v


def _multiple(tok: str) -> float:
    t = tok.strip().strip("()").rstrip("xX")
    try:
        return float(t)
    except ValueError:
        return np.nan


def _values(tokens: list[str]) -> dict | None:
    toks = [t for t in tokens if t != "$"]
    if len(toks) != 9 or not toks[6].rstrip(")").lower().endswith("x"):
        return None
    out = {c: _amount(t) for c, t in zip(VALUE_COLUMNS[:6], toks[:6])}
    out["multiple_printed"] = _multiple(toks[6])
    out["gain"] = _amount(toks[7])
    out["net_irr"] = _irr(toks[8])
    return out if np.isfinite(out["committed"]) and np.isfinite(out["paid_in"]) else None


def _is_header(line: str) -> bool:
    low = line.lower()
    return any(w in low for w in _HEADER_WORDS)


def parse_lines(lines: list[str], report_date=None) -> tuple[pd.DataFrame, dict]:
    """Fund rows from the report's text lines, with the subtotal checks."""
    rows, pending, fragment = [], [], None
    stats = {"lines": len(lines), "rows": 0, "subtotals": 0, "subtotals_matched": 0, "unparsed_dated_lines": 0,
             "unassigned_rows": 0}
    for i, raw in enumerate(lines):
        line = " ".join(str(raw).split())
        if not line:
            continue
        if line.lower().startswith("subtotal"):
            m = re.match(r"(?i)subtotal\s+(.*?)\s+((?:\$\s?)?[\d(].*)$", line)
            if m:
                strategy = m.group(1).strip(" -")
                vals = _values(m.group(2).split())
                stats["subtotals"] += 1
                for r in pending:
                    r["strategy"] = strategy
                if vals is not None and pending:
                    s = sum(r["committed"] for r in pending)
                    if abs(s - vals["committed"]) <= max(1000.0, 0.001 * abs(vals["committed"])):
                        stats["subtotals_matched"] += 1
                rows.extend(pending)
                pending = []
            fragment = None
            continue
        m = _DATE.search(f" {line} ")
        if m is None:
            low = line.lower()
            if low.startswith(("total", "grand total")) or _is_header(line):
                fragment = None
            elif len(line) < 80 and " - " not in line and not re.search(r"(\d[\d,]{3,}\s+){2}", line):
                fragment = line                              # possibly the first half of a wrapped name
            continue
        name = f" {line} "[:m.start()].strip()
        vals = _values(f" {line} "[m.end():].split())
        if vals is None or not name:
            stats["unparsed_dated_lines"] += 1
            fragment = None
            continue
        if fragment and _CONTINUATION.match(name):
            name = f"{fragment} {name}"
        fragment = None
        pending.append({"name": name, "initial_investment_date": pd.to_datetime(m.group(1), format="%m/%d/%Y",
                                                                               errors="coerce"),
                        **vals, "line_no": i})
    for r in pending:                                        # rows after the last subtotal
        r["strategy"] = None
        stats["unassigned_rows"] += 1
    rows.extend(pending)
    df = pd.DataFrame(rows, columns=["strategy", "name", "initial_investment_date", *VALUE_COLUMNS, "line_no"])
    if len(df):
        df["initial_investment_date"] = pd.to_datetime(df["initial_investment_date"]).astype("datetime64[ns]")
        df["tvpi"] = np.where(df["paid_in"] > 0, df["total_value"] / df["paid_in"].where(df["paid_in"] > 0), np.nan)
        df["vintage_year"] = df["initial_investment_date"].dt.year.astype("Int64")
    else:
        df["tvpi"], df["vintage_year"] = pd.Series(dtype="float64"), pd.Series(dtype="Int64")
    df.insert(0, "report_date", pd.Timestamp(report_date) if report_date is not None else pd.NaT)
    df["report_date"] = pd.to_datetime(df["report_date"]).astype("datetime64[ns]")
    stats["rows"] = int(len(df))
    return df, stats


def pdf_lines(raw: bytes) -> list[str]:
    """Text lines of every page (pdfplumber keeps a table row on one line)."""
    import pdfplumber
    out = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            out.extend((page.extract_text() or "").splitlines())
    return out


def _download(session, report_date, timeout=60):
    tag = f"{pd.Timestamp(report_date):%m%d%y}"
    last = None
    for tpl in URLS:
        r = session.get(tpl.format(tag=tag), headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            return r.content, None
        last = r.status_code
    return None, last


def load(cache_dir, *, start_year: int = 2016, today: date | None = None, session=None, offline: bool = False,
         manual_dir=None, raw_dir=None, to_lines=None, log=print) -> tuple[pd.DataFrame, dict]:
    """Every published quarterly report from `start_year` on: (fund rows, stats).

    Order: parsed cache (same parser version) → raw PDF on Drive → manual folder → download.
    """
    cache_dir = Path(cache_dir)
    raw_dir = Path(raw_dir) if raw_dir is not None else cache_dir / "raw"
    to_lines = to_lines or pdf_lines
    if session is None and not offline:
        import requests
        session = requests.Session()
    parts, stats = [], {"reports": 0, "cached": 0, "parsed": 0, "downloaded": 0, "manual": 0, "raw": 0, "missing": 0,
                        "failed": 0, "rows": 0, "subtotals": 0, "subtotals_matched": 0, "unparsed_dated_lines": 0,
                        "missing_reports": [], "failed_reports": []}
    for rd in report_dates(start_year, today):
        stats["reports"] += 1
        label = f"{rd:%Y%m%d}"
        mark = cache_dir / f"wsib_{label}.done.json"
        if mark.exists():
            meta = json.loads(mark.read_text(encoding="utf-8"))
            t = load_table(cache_dir / f"wsib_{label}") if meta.get("parser") == PARSER_VERSION else None
            if t is not None:
                parts.append(t)
                stats["cached"] += 1
                for k in ("subtotals", "subtotals_matched", "unparsed_dated_lines"):
                    stats[k] += int(meta.get(k, 0))
                continue
        raw, source = None, None
        for folder, src in ((raw_dir, "raw"), (Path(manual_dir) if manual_dir else None, "manual")):
            if folder is not None and (folder / file_name(rd)).exists():
                raw, source = (folder / file_name(rd)).read_bytes(), src
                break
        if raw is None and not offline and session is not None:
            try:
                raw, code = _download(session, rd)
            except Exception as exc:                          # one bad quarter never stops the rest
                raw, code = None, f"{type(exc).__name__}"
            if raw is not None:
                source = "downloaded"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / file_name(rd)).write_bytes(raw)
            elif code not in (404, None):
                stats["failed_reports"].append(f"{label} ({code})")
        if raw is None:
            stats["missing"] += 1
            stats["missing_reports"].append(label)
            continue
        try:
            df, st = parse_lines(to_lines(raw), rd)
        except Exception as exc:
            stats["failed"] += 1
            stats["failed_reports"].append(f"{label} ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        if not len(df):
            stats["failed"] += 1
            stats["failed_reports"].append(f"{label} (no fund rows recognized)")
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"wsib_{label}_sample.txt").write_text("\n".join(to_lines(raw)[:80]), encoding="utf-8")
            continue
        save_table(df, cache_dir / f"wsib_{label}")
        mark.write_text(json.dumps({"parser": PARSER_VERSION, "source": source, **{k: v for k, v in st.items()}}),
                        encoding="utf-8")
        parts.append(df)
        stats["parsed"] += 1
        stats[source] += 1
        for k in ("subtotals", "subtotals_matched", "unparsed_dated_lines"):
            stats[k] += int(st.get(k, 0))
        if log:
            log(f"  WSIB {rd.date()}: {len(df):,} funds, {st['subtotals_matched']}/{st['subtotals']} subtotals match")
    out = pd.concat(parts, ignore_index=True) if parts else parse_lines([], None)[0]
    stats["rows"] = int(len(out))
    return out, stats


def counts(funds: pd.DataFrame) -> pd.DataFrame:
    """Funds per strategy and vintage in the latest report – no returns."""
    if funds is None or not len(funds):
        return pd.DataFrame(columns=["strategy", "vintage_year", "funds", "committed"])
    latest = funds[funds["report_date"] == funds["report_date"].max()]
    return (latest.groupby(["strategy", "vintage_year"], dropna=False)
            .agg(funds=("name", "size"), committed=("committed", "sum")).reset_index())
