"""SEC fails-to-deliver files: a public CUSIP-ticker bridge and daily closing prices.

The SEC publishes, twice a month, every equity security with a balance of failed
deliveries on each settlement day (all securities from September 16, 2008): settlement
date, CUSIP, ticker symbol, quantity, company name and the closing price on the previous
day (https://www.sec.gov/data/foiadocsfailsdatahtm). The study uses them for three
things the Form 13F data cannot give:
  * the bridge from 13F's CUSIPs to tickers, and through the SEC symbol bridges to the
    filer (CIK), for companies SimFin never covered;
  * quarter-end prices where 13F has none (before the 13F data sets begin in 2013);
  * the last price before a company disappears, and extra prices between quarter-ends
    for the robustness panel.

Each half-month zip is reduced to two small tables and cached on Drive like the other
SEC data sets: `names` (CUSIP, symbol, company name, first and last settlement date)
and `px` (per CUSIP: the first, last, highest and lowest price in the file, with their
dates). The price is dated the business day before the settlement date. The SEC notes
it cannot guarantee the price matches other sources; prices missing or below a cent are
blank in the files.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .sec_datasets import cached_table, process
from .thirteenf import normalize_cusip

INDEX_URL = "https://www.sec.gov/data/foiadocsfailsdatahtm"
PREFIXES = (
    "https://www.sec.gov/files/data/fails-deliver-data/",
    "https://www.sec.gov/files/data/frequently-requested-foia-document-fails-deliver-data/",
    "https://www.sec.gov/files/data/other/",
    "https://www.sec.gov/files/node/add/data_distribution/",
)
START = "200907a"          # first half-month file; 3-year lookbacks for 2013 start dates need 2010 prices
SCHEMA = "ftd-v1"
POSTING_LAG_DAYS = 45      # without the index page: only half-months that ended this long ago are tried
_HREF = re.compile(r"""href\s*=\s*["']([^"']*?cnsfails(\d{6}[ab]?)(?:_\d+)?\.zip)["']""", re.I)
_PX_COLS = ["cusip", "first_date", "first_price", "last_date", "last_price", "max_date", "max_price",
            "min_date", "min_price", "n_obs"]
_NAME_COLS = ["cusip", "symbol", "description", "first_date", "last_date", "n"]

__all__ = ["INDEX_URL", "PREFIXES", "START", "SCHEMA", "parse_index", "half_month_labels", "items", "parse_text",
           "extract_file", "load", "load_px", "quarter_end_prices", "last_prices"]


def parse_index(html: str) -> dict[str, str]:
    """{label: absolute URL} for every file linked on the SEC's page: half-months ('202301a') and
    the occasional whole month ('201003')."""
    out: dict[str, str] = {}
    for href, label in _HREF.findall(html or ""):
        url = href if href.lower().startswith("http") else "https://www.sec.gov" + ("" if href.startswith("/") else "/") + href
        out.setdefault(label.lower(), url)
    return out


def _half_end(label: str) -> pd.Timestamp:
    y, m, half = int(label[:4]), int(label[4:6]), label[6]
    first = pd.Timestamp(year=y, month=m, day=1)
    return first + pd.Timedelta(days=14) if half == "a" else first + pd.offsets.MonthEnd(0)


def half_month_labels(start: str = START, today: date | None = None, lag_days: int = POSTING_LAG_DAYS) -> list[str]:
    """'YYYYMMa' and 'YYYYMMb' labels from `start` to the last half-month that ended `lag_days` ago."""
    today = pd.Timestamp(today or date.today())
    y, m = int(start[:4]), int(start[4:6])
    out = []
    while (y, m) <= (today.year, today.month):
        for half in ("a", "b"):
            lab = f"{y:04d}{m:02d}{half}"
            if lab >= start and _half_end(lab) <= today - pd.Timedelta(days=lag_days):
                out.append(lab)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _candidates(label: str) -> list[str]:
    return [f"{p}cnsfails{label}.zip" for p in PREFIXES] + [f"{PREFIXES[0]}cnsfails{label}_0.zip"]


def items(index: dict[str, str] | None, start: str = START, today: date | None = None) -> list[tuple[str, list[str]]]:
    """(label, candidate URLs) per file: the links on the SEC page when it was read, else the
    published naming pattern under each folder the SEC has used."""
    if index:
        return [(lab, [index[lab]] + [u for u in _candidates(lab) if u != index[lab]])
                for lab in sorted(index) if lab >= start]
    return [(lab, _candidates(lab)) for lab in half_month_labels(start, today)]


def parse_text(text: str) -> pd.DataFrame:
    """Rows of one file: settle_date, cusip, symbol, quantity, description, price.

    Pipe-delimited with a header; header, trailer and malformed lines are skipped. A
    company name that itself contains a pipe is rejoined; '.' (no usable price) is blank."""
    rows = []
    for line in str(text).splitlines():
        parts = line.rstrip("\r\n").split("|")
        if len(parts) < 6 or not re.fullmatch(r"\d{8}", parts[0].strip()):
            continue
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip(),
                     "|".join(parts[4:-1]).strip(), parts[-1].strip()))
    df = pd.DataFrame(rows, columns=["settle_date", "cusip", "symbol", "quantity", "description", "price"])
    df["settle_date"] = pd.to_datetime(df["settle_date"], format="%Y%m%d", errors="coerce").astype("datetime64[ns]")
    df["cusip"] = normalize_cusip(df["cusip"]).to_numpy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"].str.replace(",", "", regex=False), errors="coerce")
    df["price"] = pd.to_numeric(df["price"].str.replace(",", "", regex=False), errors="coerce")
    df.loc[~(df["price"] > 0), "price"] = np.nan
    return df[df["settle_date"].notna() & df["cusip"].ne("")].reset_index(drop=True)


def extract_file(zf) -> dict[str, pd.DataFrame]:
    """The two cached tables of one half-month file (see the module docstring)."""
    member = next((n for n in zf.namelist() if not n.endswith("/")), None)
    if member is None:
        return {"px": pd.DataFrame(columns=_PX_COLS), "names": pd.DataFrame(columns=_NAME_COLS)}
    df = parse_text(zf.read(member).decode("latin-1", errors="replace"))
    names = (df.sort_values("settle_date", kind="mergesort")
             .groupby(["cusip", "symbol"], as_index=False)
             .agg(description=("description", "last"), first_date=("settle_date", "min"),
                  last_date=("settle_date", "max"), n=("settle_date", "size")))
    p = df.dropna(subset=["price"]).copy()
    if p.empty:
        return {"px": pd.DataFrame(columns=_PX_COLS), "names": names[_NAME_COLS]}
    p["price_date"] = (p["settle_date"] - pd.offsets.BDay(1)).astype("datetime64[ns]")
    p = p.sort_values(["cusip", "price_date"], kind="mergesort").drop_duplicates(["cusip", "price_date"], keep="last")
    g = p.groupby("cusip", sort=False)
    first, last = g.head(1).set_index("cusip"), g.tail(1).set_index("cusip")
    hi = p.loc[g["price"].idxmax()].set_index("cusip")
    lo = p.loc[g["price"].idxmin()].set_index("cusip")
    px = pd.DataFrame({"first_date": first["price_date"], "first_price": first["price"],
                       "last_date": last["price_date"], "last_price": last["price"],
                       "max_date": hi["price_date"], "max_price": hi["price"],
                       "min_date": lo["price_date"], "min_price": lo["price"],
                       "n_obs": g.size()}).reset_index().rename(columns={"index": "cusip"})
    px["n_obs"] = px["n_obs"].astype("int32")
    return {"px": px[_PX_COLS], "names": names[_NAME_COLS]}


def load(sec_dir: str | Path, *, client=None, offline: bool = False, today: date | None = None, log=print,
         local_dir=None, start: str = START) -> tuple[pd.DataFrame, dict]:
    """(names: cusip, symbol, description, first_date, last_date, n – one row per CUSIP and
    symbol over all files; stats). The price summaries stay on Drive for `load_px`."""
    index, source = None, "naming pattern"
    if client is not None and not offline:
        try:
            raw = client.get(INDEX_URL)
        except Exception as exc:                 # the page only lists the files; the pattern still finds them
            raw = None
            source = f"naming pattern (SEC page unreadable: {type(exc).__name__})"
        if raw:
            index = parse_index(raw.decode("utf-8", errors="replace")) or None
            source = "SEC page" if index else source
    its = items(index, start, today)
    tables, stats = process(its, Path(sec_dir) / "ftd", "ftd", extract_file, client=client, offline=offline,
                            log=log, local_dir=local_dir, schema=SCHEMA, skip_load=("px",))
    stats["file_list"] = source
    n = tables.get("names")
    if n is None or n.empty:
        return pd.DataFrame(columns=_NAME_COLS), stats
    n = n.sort_values("last_date", kind="mergesort")
    n = (n.groupby(["cusip", "symbol"], as_index=False)
         .agg(description=("description", "last"), first_date=("first_date", "min"),
              last_date=("last_date", "max"), n=("n", "sum")))
    for c in ("first_date", "last_date"):
        n[c] = pd.to_datetime(n[c]).astype("datetime64[ns]")
    return n[_NAME_COLS], stats


def load_px(sec_dir: str | Path, cusips=None) -> pd.DataFrame:
    """Price summaries per CUSIP and file (columns of `extract_file`'s px), for the given CUSIPs."""
    want = None if cusips is None else set(pd.Series(list(cusips), dtype="object").astype(str))

    def keep(df):
        return df[df["cusip"].isin(want)] if want is not None else df

    px = cached_table(Path(sec_dir) / "ftd", "ftd", "px", keep, schema=SCHEMA)
    if px is None or px.empty:
        return pd.DataFrame(columns=_PX_COLS)
    for c in ("first_date", "last_date", "max_date", "min_date"):
        px[c] = pd.to_datetime(px[c]).astype("datetime64[ns]")
    return px.reset_index(drop=True)


def _points(px: pd.DataFrame) -> pd.DataFrame:
    """Every dated price the summaries hold (first and last of each file), one per CUSIP and day."""
    a = px[["cusip", "first_date", "first_price"]].rename(columns={"first_date": "price_date", "first_price": "price"})
    b = px[["cusip", "last_date", "last_price"]].rename(columns={"last_date": "price_date", "last_price": "price"})
    pts = pd.concat([a, b], ignore_index=True).dropna()
    pts = pts[pts["price"] > 0]
    return pts.sort_values(["cusip", "price_date"], kind="mergesort").drop_duplicates(["cusip", "price_date"])


def quarter_end_prices(px: pd.DataFrame, quarter_ends, tolerance_days: int = 10, *,
                       points: pd.DataFrame | None = None) -> pd.DataFrame:
    """(cusip, period, price, price_date): per CUSIP and quarter-end, the price dated nearest
    the quarter-end within `tolerance_days` (the earlier one on a tie). `points` reuses an
    earlier `_points(px)`."""
    cols = ["cusip", "period", "price", "price_date"]
    if points is None and (px is None or px.empty):
        return pd.DataFrame(columns=cols)
    pts = points if points is not None else _points(px)
    if pts.empty:
        return pd.DataFrame(columns=cols)
    q = pd.DatetimeIndex(pd.to_datetime(list(quarter_ends))).astype("datetime64[ns]")
    out = []
    tol = pd.Timedelta(days=tolerance_days)
    for cusip, g in pts.groupby("cusip", sort=False):
        d = g["price_date"].to_numpy(dtype="datetime64[ns]")
        p = g["price"].to_numpy(dtype="float64")
        lo = d.min() - np.timedelta64(tolerance_days, "D")
        hi = d.max() + np.timedelta64(tolerance_days, "D")
        qs = q[(q >= lo) & (q <= hi)]
        if not len(qs):
            continue
        qv = qs.to_numpy(dtype="datetime64[ns]")
        j = np.searchsorted(d, qv, side="left")
        before = np.clip(j - 1, 0, len(d) - 1)
        after = np.clip(j, 0, len(d) - 1)
        gap_b = np.abs(qv - d[before])
        gap_a = np.abs(d[after] - qv)
        pick = np.where(gap_b <= gap_a, before, after)
        gap = np.minimum(gap_b, gap_a)
        ok = gap <= tol.to_timedelta64()
        if ok.any():
            out.append(pd.DataFrame({"cusip": cusip, "period": qv[ok], "price": p[pick[ok]], "price_date": d[pick[ok]]}))
    if not out:
        return pd.DataFrame(columns=cols)
    res = pd.concat(out, ignore_index=True)
    res["period"] = res["period"].astype("datetime64[ns]")
    res["price_date"] = res["price_date"].astype("datetime64[ns]")
    return res[cols]


def last_prices(px: pd.DataFrame) -> pd.DataFrame:
    """(cusip, price_date, price): the latest price each CUSIP has in the files."""
    cols = ["cusip", "price_date", "price"]
    if px is None or px.empty:
        return pd.DataFrame(columns=cols)
    pts = _points(px)
    return pts.groupby("cusip", sort=False).tail(1).reset_index(drop=True)[cols]
