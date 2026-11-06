"""SEC EDGAR access: the quarterly index (Form 25 delistings, merger filings) and the client.

The structured data sets (XBRL financials, Form 4 insiders, Form 13F holdings) are read
by data.sec_datasets and the modules built on it (data.fsds, data.insider,
data.thirteenf), through the same client.

Fair access: the SEC asks automated clients to declare a User-Agent and to stay
under 10 requests per second. Set a Colab secret SEC_USER_AGENT such as
"Your Name your@email". Without it a generic client name is tried; if the SEC refuses
it (HTTP 403), the notebook says so and the rest of the audit still runs.
"""
from __future__ import annotations

import gzip
import os
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

from ..env import colab_secret
from ..io_utils import load_table, save_table

INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.{ext}"
GENERIC_USER_AGENT = "aem-research/0.1 (academic research; github.com/FelixDLanger/aem-research)"
FORM25_TYPES = ("25", "25/A", "25-NSE", "25-NSE/A")
_SPAC = re.compile(r"\bACQUISITION\b|\bACQUISITIONS?\s+CORP|\bSPAC\b|BLANK\s+CHECK", re.I)
_EXCHANGE = re.compile(r"STOCK\s+MARKET|STOCK\s+EXCHANGE|\bEXCHANGE\s+LLC|\bNYSE\b|\bNASDAQ\b|\bCBOE\b|\bIEX\b", re.I)
_COLUMNS = ["cik", "company", "form", "date_filed", "filename"]


class SecAccessError(RuntimeError):
    pass


def resolve_user_agent() -> tuple[str, str]:
    """(user_agent, origin). Colab secret first, then environment, then generic."""
    value = colab_secret("SEC_USER_AGENT")
    if value:
        return value, "colab secret"
    value = os.environ.get("SEC_USER_AGENT")
    if value:
        return value, "environment"
    return GENERIC_USER_AGENT, "generic"


class SecClient:
    def __init__(self, user_agent: str, min_interval: float = 0.2, session=None, timeout: int = 60):
        import requests
        self.user_agent = user_agent
        self.min_interval = max(float(min_interval), 0.11)
        self.session = session or requests.Session()
        self.timeout = timeout
        self._last = 0.0

    def get(self, url: str) -> bytes | None:
        """Bytes of the resource, None on 404. Raises SecAccessError on 403."""
        for attempt in range(4):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.session.get(url, headers={"User-Agent": self.user_agent,
                                               "Accept-Encoding": "gzip, deflate"},
                                 timeout=self.timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            if r.status_code == 403:
                raise SecAccessError(
                    "SEC refused the request (HTTP 403). Add a Colab secret SEC_USER_AGENT with "
                    "'Your Name your@email' (SEC fair-access policy), enable notebook access, rerun.")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
        raise SecAccessError(f"SEC kept failing for {url}")


def parse_master_index(raw: bytes) -> pd.DataFrame:
    """Parse EDGAR master.idx (pipe-delimited) into cik/company/form/date_filed/filename."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("latin-1")
    rows, started = [], False
    for line in text.splitlines():
        if not started:
            started = line.startswith("-----")
            continue
        parts = line.split("|")
        if len(parts) < 5 or not parts[0].strip().isdigit():
            continue
        rows.append((int(parts[0]), "|".join(parts[1:-3]).strip(), parts[-3].strip(),
                     parts[-2].strip(), parts[-1].strip()))
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["date_filed"] = pd.to_datetime(df["date_filed"], errors="coerce").astype("datetime64[ns]")
    return df


def quarters(start_year: int, end_year: int, today: date | None = None) -> list[tuple[int, int]]:
    today = today or date.today()
    current = (today.year, (today.month - 1) // 3 + 1)
    out = []
    for y in range(int(start_year), int(end_year) + 1):
        for q in (1, 2, 3, 4):
            if (y, q) < current:          # complete quarters only
                out.append((y, q))
    return out


def load_form25_index(sec_dir: str | Path, start_year: int, end_year: int, *,
                      client: SecClient | None = None, offline: bool = False,
                      today: date | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    """Form 25 filings per quarter, cached on Drive. Returns (table, stats)."""
    sec_dir = Path(sec_dir)
    frames, stats = [], {"cached": 0, "downloaded": 0, "missing": 0}
    for y, q in quarters(start_year, end_year, today):
        base = sec_dir / f"form25_{y}Q{q}"
        cached = load_table(base)
        if cached is not None:
            frames.append(cached)
            stats["cached"] += 1
            continue
        if offline or client is None:
            stats["missing"] += 1
            continue
        raw = client.get(INDEX_URL.format(year=y, q=q, ext="gz"))
        if raw is None:
            raw = client.get(INDEX_URL.format(year=y, q=q, ext="idx"))
        if raw is None:
            stats["missing"] += 1
            continue
        idx = parse_master_index(raw)
        sub = idx[idx["form"].isin(FORM25_TYPES)].reset_index(drop=True)
        save_table(sub, base)
        frames.append(sub)
        stats["downloaded"] += 1
        if log:
            log(f"  {y}Q{q}: {len(sub):,} Form 25 filings")
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLUMNS)
    if len(table):
        table["date_filed"] = pd.to_datetime(table["date_filed"]).astype("datetime64[ns]")
    return table, stats


MERGER_FORMS = ("DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "SC 14D9", "SC 14D9/A", "SC TO-T", "SC TO-T/A",
                "SC 13E3", "SC 13E3/A", "425")
INDEX_SUBSETS = {"form25": FORM25_TYPES, "merger": MERGER_FORMS}


def load_index_subsets(sec_dir: str | Path, start_year: int, end_year: int, subsets: dict | None = None, *,
                       client: SecClient | None = None, offline: bool = False, today: date | None = None,
                       log=print) -> tuple[dict[str, pd.DataFrame], dict]:
    """Several form subsets of the quarterly EDGAR index, one download per quarter.

    Default subsets: Form 25 (delisting benchmark) and merger filings (DEFM14A, SC 14D9,
    SC TO-T, 425 and others; the delisting classification). Each subset is cached per
    quarter as `<name>_<year>Q<q>`, so the Form 25 cache from earlier runs is reused and
    only quarters missing a subset are downloaded again.
    """
    subsets = subsets or INDEX_SUBSETS
    sec_dir = Path(sec_dir)
    frames: dict[str, list] = {k: [] for k in subsets}
    stats = {"cached": 0, "downloaded": 0, "missing": 0}
    for y, q in quarters(start_year, end_year, today):
        cached = {k: load_table(sec_dir / f"{k}_{y}Q{q}") for k in subsets}
        if all(v is not None for v in cached.values()):
            for k, v in cached.items():
                frames[k].append(v)
            stats["cached"] += 1
            continue
        raw = None
        if not offline and client is not None:
            raw = client.get(INDEX_URL.format(year=y, q=q, ext="gz")) or client.get(INDEX_URL.format(year=y, q=q, ext="idx"))
        if raw is None:
            for k, v in cached.items():
                if v is not None:
                    frames[k].append(v)
            stats["missing"] += 1
            continue
        idx = parse_master_index(raw)
        for k, forms in subsets.items():
            sub = idx[idx["form"].isin(forms)].reset_index(drop=True)
            save_table(sub, sec_dir / f"{k}_{y}Q{q}")
            frames[k].append(sub)
        stats["downloaded"] += 1
        if log:
            log(f"  {y}Q{q}: " + ", ".join(f"{k} {len(frames[k][-1]):,}" for k in subsets))
    tables = {}
    for k, parts in frames.items():
        t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_COLUMNS)
        if len(t):
            t["date_filed"] = pd.to_datetime(t["date_filed"]).astype("datetime64[ns]")
        tables[k] = t
    return tables, stats


def is_spac_name(name: str) -> bool:
    return bool(_SPAC.search(str(name)))


def is_exchange_name(name: str) -> bool:
    return bool(_EXCHANGE.search(str(name)))


# ---------------------------------------------------------------------------
# Superseded placeholders, kept so older notebooks fail with a pointer, not a NameError
# ---------------------------------------------------------------------------

def load_xbrl_fundamentals(*_args, **_kwargs):
    """Superseded: use aem_research.data.fsds.load and aem_research.fundamentals.snapshots.

    Source: SEC Financial Statement Data Sets (quarterly; sub.txt + num.txt). Every value
    is stamped with the filing's `filed` date, never the period end, so features are
    known on the date they are used.
    """
    raise NotImplementedError(load_xbrl_fundamentals.__doc__)


def load_insider_purchases(*_args, **_kwargs):
    """Superseded: use aem_research.data.insider.load, purchases, classify and signals.

    Transaction code P only (not option exercises, awards or Rule 10b5-1 plan trades),
    officer and director roles, the Form 4 filing date as the availability date.
    """
    raise NotImplementedError(load_insider_purchases.__doc__)
