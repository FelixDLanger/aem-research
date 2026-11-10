"""Paper A: US take-private buyouts from SEC filings – sample, entry terms, visible exits.

The method follows Guo, Hotchkiss and Song (2011, JF 66(2), 479-517): public-to-private
buyouts identified in SEC filings, entry terms from the merger documents and the last
financials before the deal, exits observed where the company becomes public again.

1. Candidates – EDGAR full-text search for the sponsor's "equity commitment letter" in
   target-side merger filings (proxy, information statement, 14D-9, 13E-3). Strategic
   buyers rarely need an equity commitment, so the phrase marks sponsor-backed deals.
2. Deals – hits grouped by target (the CIK that files 10-Ks and 10-Qs); a new deal
   starts after an 18-month gap.
3. Completion – a Form 25 or Form 15 for the target, or a merger delisting in the price
   data, within 18 months of the first filing.
4. Terms – read from the definitive document: offer price per share, the equity
   commitment, the largest debt amount committed, the sponsor. These are heuristics, so
   each value keeps the sentence it came from, for hand-checking.
5. Entry – XBRL financials filed before the first filing: trailing revenue, operating
   income and D&A (EBITDA), cash, debt and cover-page shares, so enterprise value at the
   offer price.
6. Exits, detection only – the target still files 10-Ks (registered debt), a re-IPO under
   a matching name (S-1 or 424B), or a sale to a listed buyer (an 8-K "completion of
   acquisition" naming the target). No exit value is computed before Paper A's lock.
"""
from __future__ import annotations

import gzip
import html as _html
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import fundamentals as fu
from .data.sec import SecAccessError
from .data.thirteenf import normalize_name
from .io_utils import load_table, save_table

TARGET_FORMS = ("DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "SC 14D9", "SC 13E3")
DEFINITIVE_ORDER = ("DEFM14A", "SC 14D9", "SC 13E3", "DEFM14C", "PREM14A", "PREM14C")
QUERY = '"equity commitment letter" OR "equity commitment letters"'
EXIT_PHRASE = '"completion of acquisition"'
INDEX_SUBSETS = {"dereg": ("15-12B", "15-12G", "15-15D", "15-12B/A", "15-12G/A", "15-15D/A"),
                 "ipo": ("S-1", "424B4", "424B1")}
EXTRA_FLOWS = ("operating_income", "dna", "amortization", "pretax_income", "interest_expense")
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"
TERMS_VERSION = 2          # bump when extract_terms changes: cached texts are re-read, nothing is downloaded
TERM_COLUMNS = ["deal_id", "doc_adsh", "main_file", "doc_url", "offer_price", "price_snippet", "price_mentions",
                "equity_commitment_usd", "equity_snippet", "debt_commitment_usd", "debt_snippet", "sponsor",
                "sponsor_mentions", "sponsor_snippet", "text_chars", "terms_version", "doc_error"]
# Exit dates would give holding periods (an A-H3 outcome): cached, not shown before Paper A's lock.
HIDDEN_UNTIL_LOCK = ["still_files_from", "reipo_date", "sale_date"]

# Sponsors named often enough in US take-privates to be worth recognizing (public knowledge).
SPONSORS = (
    "Apollo", "Blackstone", "KKR", "Kohlberg Kravis Roberts", "Carlyle", "TPG", "Bain Capital", "Thoma Bravo",
    "Vista Equity", "Silver Lake", "Advent International", "Hellman & Friedman", "Warburg Pincus",
    "Leonard Green", "Clearlake", "Francisco Partners", "Insight Partners", "Permira", "EQT", "CVC", "Cinven",
    "Apax", "Brookfield", "Platinum Equity", "American Securities", "Searchlight", "Siris", "Vector Capital",
    "Marlin Equity", "Sycamore Partners", "Cerberus", "Stone Point", "Veritas Capital", "Golden Gate Capital",
    "Elliott", "Evergreen Coast", "Sun Capital", "Gores", "H.I.G.", "Genstar", "GTCR", "Madison Dearborn",
    "Welsh, Carson", "Thomas H. Lee", "Berkshire Partners", "Onex", "Ares", "Oaktree", "Centerbridge",
    "Court Square", "Kelso", "Littlejohn", "Lone Star", "Partners Group", "Ardian", "BC Partners",
    "Providence Equity", "Summit Partners", "TA Associates", "Nordic Capital", "PAI Partners", "Triton",
    "Sentinel Capital", "Audax", "Charlesbank", "Harvest Partners", "Kohlberg & Co", "OpenGate", "Turn/River",
    "Symphony Technology", "STG Partners", "Arlington Capital", "Crestview", "Lindsay Goldberg", "New Mountain",
    "Stonepeak", "Global Infrastructure Partners", "I Squared", "Macquarie", "Blue Owl", "KPS Capital",
    "Wind Point", "Mill Point", "Searchlight Capital", "Rhône", "Bregal", "Vestar", "Riverstone", "EnCap",
)
_SPONSOR_RE = [(s, re.compile(r"(?<![A-Za-z])" + re.escape(s) + r"(?![a-z])")) for s in SPONSORS]

_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(million|billion|thousand)?", re.I)
_PRICE_PATTERNS = [
    re.compile(r"right to receive (?:an amount (?:in cash )?equal to )?\$\s?(\d{1,4}(?:\.\d{1,4})?)"
               r"(?: per share)?,? in cash", re.I),
    re.compile(r"\$\s?(\d{1,4}(?:\.\d{1,4})?) per share,? (?:net to the (?:seller|holder) )?in cash", re.I),
    re.compile(r"(?:offer price|merger consideration|per share merger consideration|per share price)"
               r",? (?:of|equal to|is|which is) \$\s?(\d{1,4}(?:\.\d{1,4})?)", re.I),
    re.compile(r"at a (?:purchase )?price of \$\s?(\d{1,4}(?:\.\d{1,4})?) per share", re.I),
    re.compile(r"\$\s?(\d{1,4}(?:\.\d{1,4})?) in cash,? (?:without interest,? )?(?:per|for each) share", re.I),
]
# The stated size of the deal, as announcements put it ("valued at approximately $2.1 billion").
_VALUE_RE = re.compile(
    r"(?:valued at|(?:enterprise|equity|transaction|aggregate) value of|"
    r"total (?:transaction |equity |enterprise )?(?:value|consideration) of)\s+(?:approximately |about |roughly )?"
    r"(?:US)?\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(million|billion)", re.I)
# The check of the extracted terms: a second filing per deal and a seeded sample for reading.
SECOND_VERSION = 1
SECOND_QUERY = '"merger agreement" "per share"'
SECOND_COLUMNS = ["deal_id", "second_adsh", "second_file", "second_form", "second_kind", "second_filed", "second_url",
                  "second_offer_price", "second_price_snippet", "second_equity_commitment_usd",
                  "second_debt_commitment_usd", "second_sponsors", "second_deal_value_usd", "second_deal_value_kind",
                  "second_value_snippet", "second_text_chars", "second_version", "second_error"]
CHECK_COLUMNS = ["check_price", "check_sponsor", "check_value", "check_equity", "check_debt", "check_flag"]
_DEBT_ANCHOR = re.compile(r"debt commitment|debt financing|committed (?:debt|financing)|(?:term loan|credit) facilit", re.I)
_FEE_CONTEXT = re.compile(r"termination fee|fee of|fees|guarantee|expense|reimburse|cap of|escrow|deposit", re.I)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def html_to_text(raw) -> str:
    s = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</h\d>|</li>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s).replace("\xa0", " ").replace("’", "'")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def _usd(whole: str, frac: str | None, unit: str | None) -> float:
    v = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
    u = (unit or "").lower()
    return v * (1e9 if u == "billion" else 1e6 if u == "million" else 1e3 if u == "thousand" else 1.0)


def _snippet(text: str, start: int, end: int, pad: int = 160) -> str:
    return " ".join(text[max(0, start - pad):min(len(text), end + pad)].split())


def _amounts_near(text: str, anchor: re.Pattern, *, before: int = 100, after: int = 500, minimum: float = 1e6,
                  stop: re.Pattern | None = None):
    """Largest dollar amount near each anchor, skipping fees, guarantees and expense caps.
    The window ends where `stop` begins (the debt paragraph must not lend the equity its amount)."""
    out = []
    for m in anchor.finditer(text):
        lo, hi = max(0, m.start() - before), min(len(text), m.end() + after)
        nxt = stop.search(text, m.end(), hi) if stop is not None else None
        if nxt is not None:
            hi = nxt.start()
        best = None
        for a in _MONEY.finditer(text, lo, hi):
            if _FEE_CONTEXT.search(text[max(lo, a.start() - 60):a.start()]):
                continue
            v = _usd(*a.groups())
            if v >= minimum and (best is None or v > best[0]):
                best = (v, a.start(), a.end())
        if best is not None:
            out.append(best)
    return out


def _mode(values: list[tuple[float, int, int]]):
    if not values:
        return None
    counts = Counter(round(v[0], 2) for v in values)
    top = max(counts.values())
    winners = {k for k, c in counts.items() if c == top}
    return next(v for v in values if round(v[0], 2) in winners)


def extract_terms(text: str) -> dict:
    """Offer price, equity commitment, largest debt commitment and sponsor, each with its sentence."""
    t = " ".join(text.split())
    out = {"offer_price": np.nan, "price_snippet": "", "price_mentions": 0, "equity_commitment_usd": np.nan,
           "equity_snippet": "", "debt_commitment_usd": np.nan, "debt_snippet": "", "sponsor": "",
           "sponsor_mentions": 0, "sponsor_snippet": "", "text_chars": len(t)}
    prices = []
    for pat in _PRICE_PATTERNS:
        for m in pat.finditer(t):
            v = float(m.group(1))
            if 0.01 <= v <= 10_000:
                prices.append((v, m.start(), m.end()))
    best = _mode(prices)
    if best is not None:
        out["offer_price"], out["price_mentions"] = best[0], len(prices)
        out["price_snippet"] = _snippet(t, best[1], best[2])
    eq = _mode(_amounts_near(t, re.compile(r"equity commitment|equity financing|equity contribution", re.I),
                             stop=_DEBT_ANCHOR))
    if eq is not None:
        out["equity_commitment_usd"], out["equity_snippet"] = eq[0], _snippet(t, eq[1], eq[2])
    debt = _amounts_near(t, _DEBT_ANCHOR)
    if debt:
        top = max(debt, key=lambda v: v[0])
        out["debt_commitment_usd"], out["debt_snippet"] = top[0], _snippet(t, top[1], top[2])
    found = [(s, len(rx.findall(t))) for s, rx in _SPONSOR_RE]
    found = [f for f in found if f[1] > 0]
    if found:
        s, n = max(found, key=lambda f: (f[1], len(f[0])))
        out["sponsor"], out["sponsor_mentions"] = s, n
        first = dict(_SPONSOR_RE)[s].search(t)
        out["sponsor_snippet"] = _snippet(t, first.start(), first.end(), pad=120)
    return out


def sponsors_named(text: str) -> list[str]:
    """Every known sponsor the text names, most mentioned first."""
    t = " ".join(str(text).split())
    found = [(s, len(rx.findall(t))) for s, rx in _SPONSOR_RE]
    return [s for s, n in sorted(found, key=lambda f: (-f[1], -len(f[0]))) if n > 0]


def extract_deal_value(text: str) -> dict:
    """The deal's stated size (the first statement in the text), its kind and its sentence."""
    t = " ".join(str(text).split())
    m = _VALUE_RE.search(t)
    if m is None:
        return {"deal_value_usd": np.nan, "deal_value_kind": "", "value_snippet": ""}
    near = t[max(0, m.start() - 80):m.end()].lower()
    kind = "enterprise" if "enterprise" in near else "equity" if "equity" in near else "transaction"
    return {"deal_value_usd": _usd(m.group(1), m.group(2), m.group(3)), "deal_value_kind": kind,
            "value_snippet": _snippet(t, m.start(), m.end())}


def filing_documents(index_html) -> pd.DataFrame:
    """Rows of an EDGAR filing index page (<accession>-index.htm): seq, description, file, type."""
    s = index_html.decode("utf-8", errors="ignore") if isinstance(index_html, (bytes, bytearray)) else str(index_html)
    rows = []
    for tr in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", s):
        cells = [" ".join(_html.unescape(re.sub(r"<[^>]+>", " ", c)).split())
                 for c in re.findall(r"(?is)<td[^>]*>(.*?)</td>", tr)]
        href = re.search(r'(?i)href="([^"]+)"', tr)
        if len(cells) >= 4 and href:
            rows.append({"seq": cells[0], "description": cells[1], "file_name": href.group(1).rsplit("/", 1)[-1],
                         "type": cells[3]})
    return pd.DataFrame(rows, columns=["seq", "description", "file_name", "type"])


def document_url(cik: int, adsh: str, name: str) -> str:
    return ARCHIVES.format(cik=int(cik), folder=str(adsh).replace("-", ""), name=name)


def index_url(cik: int, adsh: str) -> str:
    return document_url(cik, adsh, f"{adsh}-index.htm")


def _cached_text(client, url: str, path: Path, offline: bool, transform=None) -> str | None:
    """Text of `url`, kept gzipped on Drive; None when it is not there (404) or not cached offline."""
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode("utf-8", errors="ignore")
    if offline or client is None:
        return None
    raw = client.get(url)
    if raw is None:
        return None
    text = transform(raw) if transform else raw.decode("utf-8", errors="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode("utf-8")))
    return text


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))[:80]


def read_documents(deals: pd.DataFrame, *, client=None, cache_dir, offline: bool = False, log=print,
                   save_every: int = 50) -> tuple[pd.DataFrame, dict]:
    """Terms of every deal from its definitive document. The text is kept on Drive (gzipped),
    so a better extractor (TERMS_VERSION) re-reads it without downloading again."""
    cache_dir = Path(cache_dir)
    docs = cache_dir / "docs"
    old = load_table(cache_dir / "terms")
    have = {}
    if old is not None and len(old) and "terms_version" in old.columns:
        old = old[old["terms_version"] == TERMS_VERSION]
        have = {(r["deal_id"], r["doc_adsh"]): r for r in old.to_dict("records")}
    rows, stats = [], {"deals": int(len(deals)), "cached": 0, "read": 0, "missing": 0, "failed": 0, "index_pages": 0}
    done = []

    def flush():
        if done:
            save_table(pd.DataFrame(list(have.values()) + [r for r in done if (r["deal_id"], r["doc_adsh"]) not in have]),
                       cache_dir / "terms")

    for n, d in enumerate(deals.itertuples(), 1):
        key = (d.deal_id, d.doc_adsh)
        if key in have:
            rows.append(have[key])
            stats["cached"] += 1
            continue
        cik, adsh = int(d.target_cik), str(d.doc_adsh)
        main = d.doc_file if bool(d.doc_is_main) else None
        try:
            if main is None:
                page = _cached_text(client, index_url(cik, adsh), docs / f"{adsh}-index.htm.gz", offline)
                if page is not None:
                    stats["index_pages"] += 1
                    listed = filing_documents(page)
                    base = str(d.doc_form).replace("/A", "").upper()
                    hit = listed[listed["type"].str.upper().str.replace("/A", "", regex=False) == base]
                    main = hit["file_name"].iloc[0] if len(hit) else d.doc_file
                else:
                    main = d.doc_file
            url = document_url(cik, adsh, main)
            text = _cached_text(client, url, docs / f"{adsh}_{_safe(main)}.txt.gz", offline, transform=html_to_text)
        except SecAccessError:
            raise                                  # a refused User-Agent stops the phase with its message
        except Exception as exc:                   # one bad document never stops the rest
            stats["failed"] += 1
            rows.append({"deal_id": d.deal_id, "doc_adsh": adsh, "doc_error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        if text is None:
            stats["missing"] += 1
            rows.append({"deal_id": d.deal_id, "doc_adsh": adsh, "doc_error": "not available"})
            continue
        rec = {"deal_id": d.deal_id, "doc_adsh": adsh, "main_file": main, "doc_url": url,
               **extract_terms(text), "terms_version": TERMS_VERSION}
        rows.append(rec)
        done.append(rec)
        stats["read"] += 1
        if stats["read"] % save_every == 0:
            flush()
            if log:
                log(f"  documents read: {stats['read']:,} of {len(deals):,}")
    flush()
    out = pd.DataFrame(rows).reindex(columns=TERM_COLUMNS)
    return out.astype({"deal_id": "object", "doc_adsh": "object"}), stats


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------

def _clean_display_name(name: str) -> str:
    return re.sub(r"\s*\((?:CIK [0-9]+|[A-Z0-9.\-, ]{1,20})\)", "", str(name).split(" | ")[0]).strip()


def _map_dates(keys: pd.Series, mapping: pd.Series) -> pd.Series:
    """keys.map(mapping) as dates; an empty mapping gives NaT (pandas cannot map an empty date Series)."""
    if mapping is None or not len(mapping):
        return pd.Series(pd.NaT, index=keys.index, dtype="datetime64[ns]")
    return keys.map(mapping).astype("datetime64[ns]")


def _periodic(subs: pd.DataFrame) -> pd.DataFrame:
    subs = subs.reindex(columns=sorted({*subs.columns, "cik", "filed", "name", "sic", "form"}))
    s = subs[subs["form"].astype(str).str.match(r"^10-[KQ]")][["cik", "filed", "name", "sic"]].dropna(subset=["cik", "filed"]).copy()
    s["cik"] = s["cik"].astype("int64")
    s["filed"] = pd.to_datetime(s["filed"]).astype("datetime64[ns]")
    return s


def group_deals(hits: pd.DataFrame, subs: pd.DataFrame, *, gap_months: int = 18,
                current_months: int = 15) -> pd.DataFrame:
    """One row per target and deal from the full-text search hits."""
    cols = ["deal_id", "target_cik", "target_name", "sic", "first_filed", "last_filed", "n_filings", "forms",
            "doc_adsh", "doc_file", "doc_form", "doc_is_main", "fsds_filer", "announce_year"]
    if hits is None or not len(hits):
        return pd.DataFrame(columns=cols)
    h = hits.copy().reset_index(drop=True)
    h["row"] = np.arange(len(h))
    h["form"] = h["form"].fillna(h["root_form"]).astype(str)
    h = h[h["form"].str.replace("/A", "", regex=False).isin(TARGET_FORMS)]
    ex = h[["row", "ciks", "file_date"]].assign(cik=h["ciks"].astype(str).str.split(";")).explode("cik")
    ex = ex[ex["cik"].astype(str).str.isdigit()].astype({"cik": "int64"})
    per = _periodic(subs)
    j = ex.merge(per[["cik", "filed"]], on="cik", how="left")
    in_win = (j["filed"] < j["file_date"]) & (j["filed"] >= j["file_date"] - pd.DateOffset(months=current_months))
    score = j[in_win].groupby(["row", "cik"]).size().rename("n").reset_index()
    best = score.sort_values(["row", "n"], ascending=[True, False]).drop_duplicates("row")
    first_cik = ex.drop_duplicates("row").set_index("row")["cik"]
    h = h.merge(best[["row", "cik"]], on="row", how="left")
    h["fsds_filer"] = h["cik"].notna()
    h["target_cik"] = h["cik"].fillna(h["row"].map(first_cik)).astype("Int64")
    h = h.dropna(subset=["target_cik", "file_date"]).sort_values(["target_cik", "file_date"])
    gap = h.groupby("target_cik")["file_date"].diff() > pd.Timedelta(days=int(gap_months * 30.5))
    h["deal_no"] = gap.astype(int).groupby(h["target_cik"]).cumsum()
    names = per.sort_values("filed").drop_duplicates("cik", keep="last").set_index("cik")
    rows = []
    for (cik, _n), g in h.groupby(["target_cik", "deal_no"], sort=False):
        g = g.copy()
        g["base"] = g["form"].str.replace("/A", "", regex=False)
        g["rank"] = g["base"].map({f: i for i, f in enumerate(DEFINITIVE_ORDER)}).fillna(99)
        g["is_main"] = g["file_type"].astype(str).str.upper().eq(g["form"].str.upper())
        g["is_amend"] = g["form"].str.endswith("/A")
        pick = g.sort_values(["rank", "is_amend", "is_main", "file_date"], ascending=[True, True, False, False]).iloc[0]
        first = g["file_date"].min()
        nm = names["name"].get(int(cik)) if int(cik) in names.index else _clean_display_name(pick["names"])
        rows.append({"deal_id": f"{int(cik)}-{first:%Y%m%d}", "target_cik": int(cik), "target_name": nm,
                     "sic": names["sic"].get(int(cik)) if int(cik) in names.index else np.nan,
                     "first_filed": first, "last_filed": g["file_date"].max(), "n_filings": g["adsh"].nunique(),
                     "forms": ", ".join(sorted(g["form"].unique())), "doc_adsh": pick["adsh"],
                     "doc_file": pick["file_name"], "doc_form": pick["form"], "doc_is_main": bool(pick["is_main"]),
                     "fsds_filer": bool(g["fsds_filer"].any()), "announce_year": int(first.year)})
    return pd.DataFrame(rows, columns=cols)


def completion(deals: pd.DataFrame, form25=None, dereg=None, delist=None, *, window_months: int = 18) -> pd.DataFrame:
    """Completion date and evidence: Form 25, Form 15, or a merger delisting in the price data."""
    d = deals.copy()
    d["completion_date"], d["completion_evidence"] = pd.NaT, ""
    d["completion_date"] = d["completion_date"].astype("datetime64[ns]")
    if not len(d):
        d["completed"] = pd.Series(dtype=bool)
        return d
    lo = d["first_filed"] - pd.Timedelta(days=30)
    hi = d["first_filed"] + pd.DateOffset(months=window_months)
    found = {}
    for label, tab, dcol in (("form25", form25, "date_filed"), ("form15", dereg, "date_filed"),
                             ("delisting", delist, "delist_date")):
        if tab is None or not len(tab):
            continue
        t = tab.copy()
        if label == "delisting" and "category" in t.columns:
            t = t[t["category"].astype(str).str.startswith("merger")]
        t = t.dropna(subset=["cik", dcol])
        t = t.assign(cik=t["cik"].astype("int64"), when=pd.to_datetime(t[dcol]).astype("datetime64[ns]"))
        j = d[["deal_id", "target_cik"]].assign(lo=lo, hi=hi).merge(t[["cik", "when"]], left_on="target_cik",
                                                                     right_on="cik")
        j = j[(j["when"] >= j["lo"]) & (j["when"] <= j["hi"])]
        found[label] = j.groupby("deal_id")["when"].min()
    found = {k: v for k, v in found.items() if len(v)}
    if found:
        ev = pd.DataFrame(found)
        d["completion_date"] = _map_dates(d["deal_id"], ev.min(axis=1))
        evidence = {i: ", ".join(k for k, v in r.items() if v) for i, r in ev.notna().iterrows()}
        d["completion_evidence"] = d["deal_id"].map(evidence).fillna("")
    d["completed"] = d["completion_date"].notna()
    return d


def entry_financials(deals: pd.DataFrame, facts: pd.DataFrame, subs: pd.DataFrame, *,
                     current_months: int = 15) -> pd.DataFrame:
    """XBRL financials filed before each deal's first filing (point in time)."""
    cols = ["deal_id", "fin_filed", "period_end", "fin_form", "revenue", "operating_income", "dna", "dna_tag",
            "amortization", "pretax_income", "interest_expense", "cash_total", "debt", "net_cash", "shares_dei",
            "shares_filing", "shares_filing_kind", "ebitda", "ebitda_source"]
    if not len(deals) or facts is None or not len(facts):
        return pd.DataFrame(columns=cols)
    ciks = deals["target_cik"].dropna().astype("int64").unique()
    f = facts[facts["cik"].astype("int64").isin(ciks)]
    s = subs[subs["cik"].astype("int64").isin(ciks)].copy()
    s["cik"] = s["cik"].astype("int64")
    s["filed"] = pd.to_datetime(s["filed"]).astype("datetime64[ns]")
    s["period"] = pd.to_datetime(s["period"]).astype("datetime64[ns]")
    if not len(f):
        return pd.DataFrame(columns=cols)
    prep = fu.prepare(f)
    by_cik = {c: g for c, g in s.groupby("cik")}
    rows = []
    for d in deals.itertuples():
        g = by_cik.get(int(d.target_cik)) if pd.notna(d.target_cik) else None
        if g is None:
            continue
        snap = fu.snapshot(fu.restrict(prep, [int(d.target_cik)]), g, d.first_filed, current_months=current_months,
                           extra_flows=EXTRA_FLOWS)
        if not len(snap):
            continue
        r = snap.iloc[0]
        rows.append({"deal_id": d.deal_id, "fin_filed": r["fund_available"], "period_end": r["period_end"],
                     "fin_form": r["form"], **{k: r.get(k) for k in ("revenue", "operating_income", "dna", "dna_tag",
                                                                     "amortization", "pretax_income",
                                                                     "interest_expense", "cash_total", "debt",
                                                                     "net_cash", "shares_dei", "shares_filing",
                                                                     "shares_filing_kind")}})
    out = pd.DataFrame(rows, columns=cols[:-2])
    if not len(out):
        return pd.DataFrame(columns=cols)
    dna = out["dna"] + np.where(out["dna_tag"].astype(str).eq("Depreciation"), out["amortization"].fillna(0.0), 0.0)
    via_op = out["operating_income"] + dna
    via_pretax = out["pretax_income"] + out["interest_expense"] + dna
    out["ebitda"] = via_op.where(via_op.notna(), via_pretax)
    out["ebitda_source"] = np.where(via_op.notna(), "operating income + D&A",
                                    np.where(via_pretax.notna(), "pre-tax income + interest + D&A", ""))
    return out


def entry_values(deals: pd.DataFrame) -> pd.DataFrame:
    """Enterprise value at the offer price and the entry multiples (inputs, not outcomes)."""
    d = deals.copy()
    for c in ("offer_price", "shares_dei", "shares_filing", "net_cash", "ebitda", "revenue", "debt_commitment_usd",
              "equity_commitment_usd"):
        if c not in d.columns:                     # an earlier phase failed: the values stay unknown
            d[c] = np.nan
    # the latest filing's share count (cover page, else weighted-average shares: the SEC data
    # sets rarely carry the cover page), the cover page alone where that is all there is
    shares = pd.to_numeric(d["shares_filing"], errors="coerce")
    shares = shares.where(shares > 0, pd.to_numeric(d["shares_dei"], errors="coerce"))
    d["equity_value"] = d["offer_price"] * shares
    d["enterprise_value"] = d["equity_value"] - d["net_cash"]
    d["ev_ebitda"] = (d["enterprise_value"] / d["ebitda"]).where(d["ebitda"] > 0)
    d["ev_revenue"] = (d["enterprise_value"] / d["revenue"]).where(d["revenue"] > 0)
    d["leverage_x"] = (d["debt_commitment_usd"] / d["ebitda"]).where(d["ebitda"] > 0)
    d["equity_share"] = (d["equity_commitment_usd"] / d["enterprise_value"]).where(d["enterprise_value"] > 0)
    return d


# ---------------------------------------------------------------------------
# Exits (detection only)
# ---------------------------------------------------------------------------

_EXTRA_SUFFIX = re.compile(r"\b(PARENT|TOPCO|HOLDCO|INTERMEDIATE)\b")


def match_name(values) -> pd.Series:
    return normalize_name(values).str.replace(_EXTRA_SUFFIX, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def _specific(core: str) -> bool:
    return len(core) >= 8 or len(core.split()) >= 2


def exit_query_name(name: str) -> str | None:
    """The target's name core for an exact-phrase search, or None when it is too generic."""
    core = match_name([name]).iloc[0].title()
    return core if _specific(core) and '"' not in core else None


def still_files(deals: pd.DataFrame, subs: pd.DataFrame, *, days_after: int = 90) -> pd.Series:
    """First 10-K or 10-Q the target filed more than `days_after` days after completion."""
    per = _periodic(subs)
    j = deals[["deal_id", "target_cik", "completion_date"]].dropna().merge(per[["cik", "filed"]],
                                                                         left_on="target_cik", right_on="cik")
    j = j[j["filed"] > j["completion_date"] + pd.Timedelta(days=days_after)]
    return _map_dates(deals["deal_id"], j.groupby("deal_id")["filed"].min())


def reipo_candidates(deals: pd.DataFrame, ipo: pd.DataFrame, *, days_after: int = 180) -> pd.DataFrame:
    """S-1 or 424B filings after completion under a name that matches the target's."""
    cols = ["deal_id", "reipo_date", "reipo_cik", "reipo_company", "reipo_form"]
    if ipo is None or not len(ipo) or not len(deals):
        return pd.DataFrame(columns=cols)
    d = deals[["deal_id", "target_name", "completion_date"]].dropna().copy()
    d["key"] = match_name(d["target_name"]).to_numpy()
    i = ipo.copy()
    i["key"] = match_name(i["company"]).to_numpy()
    i["date_filed"] = pd.to_datetime(i["date_filed"]).astype("datetime64[ns]")
    j = d[d["key"].map(_specific)].merge(i, on="key")
    j = j[j["date_filed"] > j["completion_date"] + pd.Timedelta(days=days_after)]
    j = j.sort_values("date_filed").drop_duplicates("deal_id")
    return j.rename(columns={"date_filed": "reipo_date", "cik": "reipo_cik", "company": "reipo_company",
                             "form": "reipo_form"})[cols].reset_index(drop=True)


def exit_queries(deals: pd.DataFrame, *, days_after: int = 180, today=None) -> list[dict]:
    """One full-text search per completed deal: an 8-K "completion of acquisition" naming it."""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    out = []
    for d in deals.itertuples():
        if pd.isna(d.completion_date):
            continue
        core = exit_query_name(d.target_name)
        start = d.completion_date + pd.Timedelta(days=days_after)
        if core and start < today:
            out.append({"key": d.deal_id, "q": f'"{core}" {EXIT_PHRASE}', "forms": ["8-K"],
                        "start": str(start.date()), "end": str(today.date())})
    return out


def sale_candidates(deals: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """The earliest 8-K hit filed by someone other than the target."""
    cols = ["deal_id", "sale_date", "buyer_cik", "buyer_name", "sale_adsh"]
    if hits is None or not len(hits):
        return pd.DataFrame(columns=cols)
    h = hits.merge(deals[["deal_id", "target_cik"]], left_on="key", right_on="deal_id")
    first_cik = h["ciks"].astype(str).str.split(";").str[0]
    h = h[first_cik.str.isdigit() & (pd.to_numeric(first_cik, errors="coerce") != h["target_cik"])]
    h = h.assign(buyer_cik=pd.to_numeric(first_cik, errors="coerce").astype("Int64"),
                 buyer_name=h["names"].astype(str).map(_clean_display_name))
    h = h.sort_values("file_date").drop_duplicates("deal_id")
    return h.rename(columns={"file_date": "sale_date", "adsh": "sale_adsh"})[cols].reset_index(drop=True)


def exit_routes(deals: pd.DataFrame, subs: pd.DataFrame, ipo: pd.DataFrame | None, sale_hits: pd.DataFrame | None,
                *, still_days: int = 90, exit_days: int = 180) -> pd.DataFrame:
    """Which exits are visible, and when – flags and dates only, no values."""
    d = deals.copy()
    d["still_files_from"] = still_files(d, subs, days_after=still_days)
    d = d.merge(reipo_candidates(d, ipo, days_after=exit_days), on="deal_id", how="left")
    d = d.merge(sale_candidates(d, sale_hits), on="deal_id", how="left")
    first = d[["reipo_date", "sale_date"]].min(axis=1)
    d["exit_route"] = np.select([d["completion_date"].isna(), first.notna() & d["reipo_date"].eq(first),
                                 first.notna()], ["not completed", "re-IPO", "sold to a listed buyer"],
                                default="not visible")
    d["post_buyout_financials"] = d["still_files_from"].notna()
    return d


# ---------------------------------------------------------------------------
# Check of the extracted terms: a second filing for every deal, a seeded sample to read
# ---------------------------------------------------------------------------

def second_queries(deals: pd.DataFrame, *, days_before: int = 150, days_after: int = 30) -> list[dict]:
    """One full-text search per deal: the target's own 8-K around the announcement that
    speaks of the merger agreement and a per-share price (the press release or the agreement)."""
    out = []
    for d in deals.itertuples():
        if pd.isna(d.target_cik) or pd.isna(d.first_filed):
            continue
        out.append({"key": d.deal_id, "q": SECOND_QUERY, "forms": ["8-K"], "ciks": int(d.target_cik),
                    "start": str((d.first_filed - pd.Timedelta(days=days_before)).date()),
                    "end": str((d.first_filed + pd.Timedelta(days=days_after)).date())})
    return out


def pick_second(deals: pd.DataFrame, hits_8k: pd.DataFrame | None, hits: pd.DataFrame | None) -> pd.DataFrame:
    """A second document for each deal, never from the filing the terms were read from: the
    target's earliest 8-K hit (press release before the 8-K itself before the agreement), else
    another target-side merger filing (another form first, main documents first)."""
    cols = ["deal_id", "second_adsh", "second_file", "second_form", "second_kind", "second_filed"]
    if deals is None or not len(deals):
        return pd.DataFrame(columns=cols)
    base = deals[["deal_id", "target_cik", "doc_adsh", "doc_form", "first_filed", "last_filed"]]
    parts = []
    if hits_8k is not None and len(hits_8k):
        h = hits_8k.merge(base, left_on="key", right_on="deal_id")
        h = h[h["adsh"].astype(str).ne(h["doc_adsh"].astype(str)) & h["file_name"].astype(str).ne("")]
        ft = h["file_type"].astype(str).str.upper()
        h = h.assign(rank=np.select([ft.str.startswith("EX-99"), ft.str.startswith("8-K"), ft.str.startswith("EX-2")],
                                    [0, 1, 2], default=3), second_kind="8-K announcement",
                     form=h["form"].fillna(h["root_form"]).fillna("8-K"))
        parts.append(h.sort_values(["deal_id", "file_date", "rank"]).drop_duplicates("deal_id"))
    if hits is not None and len(hits):
        x = hits.assign(cik=hits["ciks"].astype(str).str.split(";")).explode("cik")
        x = x[x["cik"].astype(str).str.isdigit()].astype({"cik": "int64"})
        h = x.merge(base.dropna(subset=["target_cik"]).astype({"target_cik": "int64"}), left_on="cik", right_on="target_cik")
        h = h[(h["file_date"] >= h["first_filed"] - pd.Timedelta(days=30))
              & (h["file_date"] <= h["last_filed"] + pd.Timedelta(days=1))
              & h["adsh"].astype(str).ne(h["doc_adsh"].astype(str))]
        form = h["form"].fillna(h["root_form"]).astype(str)
        same = form.str.replace("/A", "", regex=False).eq(h["doc_form"].astype(str).str.replace("/A", "", regex=False))
        exhibit = ~h["file_type"].astype(str).str.upper().eq(form.str.upper())
        h = h.assign(form=form, same=same, exhibit=exhibit, second_kind="other merger filing")
        parts.append(h.sort_values(["deal_id", "exhibit", "same", "file_date"]).drop_duplicates("deal_id"))
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True).drop_duplicates("deal_id")          # the 8-K first
    out = out.rename(columns={"adsh": "second_adsh", "file_name": "second_file", "form": "second_form",
                              "file_date": "second_filed"})
    return out[cols].reset_index(drop=True)


def read_second(deals: pd.DataFrame, picks: pd.DataFrame, *, client=None, cache_dir, offline: bool = False,
                log=print, save_every: int = 50) -> tuple[pd.DataFrame, dict]:
    """The same terms from each deal's second document, with the stated deal value."""
    cache_dir = Path(cache_dir)
    docs = cache_dir / "docs"
    stats = {"deals": int(len(deals)), "with_second": int(len(picks)), "cached": 0, "read": 0, "missing": 0, "failed": 0}
    old = load_table(cache_dir / "second")
    have = {}
    if old is not None and len(old) and "second_version" in old.columns:
        old = old[old["second_version"] == SECOND_VERSION]
        have = {(r["deal_id"], r["second_adsh"], r["second_file"]): r for r in old.to_dict("records")}
    cik = dict(zip(deals["deal_id"], deals["target_cik"])) if len(deals) else {}
    rows, done = [], []

    def flush():
        if done:
            keys = {(r["deal_id"], r["second_adsh"], r["second_file"]) for r in done}
            save_table(pd.DataFrame([r for k, r in have.items() if k not in keys] + done).reindex(columns=SECOND_COLUMNS),
                       cache_dir / "second")

    for p in picks.itertuples():
        key = (p.deal_id, p.second_adsh, p.second_file)
        if key in have:
            rows.append(have[key])
            stats["cached"] += 1
            continue
        rec = {"deal_id": p.deal_id, "second_adsh": p.second_adsh, "second_file": p.second_file,
               "second_form": p.second_form, "second_kind": p.second_kind, "second_filed": p.second_filed}
        try:
            url = document_url(int(cik[p.deal_id]), str(p.second_adsh), str(p.second_file))
            text = _cached_text(client, url, docs / f"{p.second_adsh}_{_safe(p.second_file)}.txt.gz", offline,
                                transform=html_to_text)
        except SecAccessError:
            raise
        except Exception as exc:
            stats["failed"] += 1
            rows.append({**rec, "second_error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        if text is None:
            stats["missing"] += 1
            rows.append({**rec, "second_url": url, "second_error": "not available"})
            continue
        t, v = extract_terms(text), extract_deal_value(text)
        rec.update({"second_url": url, "second_offer_price": t["offer_price"], "second_price_snippet": t["price_snippet"],
                    "second_equity_commitment_usd": t["equity_commitment_usd"],
                    "second_debt_commitment_usd": t["debt_commitment_usd"],
                    "second_sponsors": "; ".join(sponsors_named(text)), "second_deal_value_usd": v["deal_value_usd"],
                    "second_deal_value_kind": v["deal_value_kind"], "second_value_snippet": v["value_snippet"],
                    "second_text_chars": t["text_chars"], "second_version": SECOND_VERSION})
        rows.append(rec)
        done.append(rec)
        stats["read"] += 1
        if stats["read"] % save_every == 0:
            flush()
            if log:
                log(f"  second documents read: {stats['read']:,} of {len(picks):,}")
    flush()
    out = pd.DataFrame(rows).reindex(columns=SECOND_COLUMNS)
    out["second_filed"] = pd.to_datetime(out["second_filed"], errors="coerce").astype("datetime64[ns]")
    return out.astype({"deal_id": "object"}), stats


def _compare(a: pd.Series, b: pd.Series, tol: float) -> pd.Series:
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    close = (a - b).abs() <= tol * np.maximum(a.abs(), b.abs())
    return pd.Series(np.select([a.notna() & b.notna() & close, a.notna() & b.notna(), a.notna() | b.notna()],
                               ["agree", "disagree", "one source"], default="none"), index=a.index)


def cross_check(deals: pd.DataFrame, *, price_tol: float = 0.005, value_tol: float = 0.15,
                commitment_tol: float = 0.01) -> pd.DataFrame:
    """Compare each deal's terms with its second document. Per term: agree, disagree, one
    source, none. The stated deal value agrees when it is within `value_tol` of the equity
    value or the enterprise value computed at the offer price (announcements quote either).
    Commitments are compared only against another merger filing (announcements rarely give
    them). check_flag: disagree if any term disagrees, agree if at least one agrees."""
    d = deals.copy()
    for c in (*SECOND_COLUMNS[1:], "offer_price", "sponsor", "equity_value", "enterprise_value",
              "equity_commitment_usd", "debt_commitment_usd"):
        if c not in d.columns:
            d[c] = "" if c in ("sponsor", "second_sponsors", "second_kind") else np.nan
    d["check_price"] = _compare(d["offer_price"], d["second_offer_price"], price_tol)
    first = d["sponsor"].fillna("").astype(str)
    second = d["second_sponsors"].fillna("").astype(str)
    named = pd.Series([f != "" and f in s.split("; ") for f, s in zip(first, second)], index=d.index, dtype=bool)
    d["check_sponsor"] = np.select([named, first.ne("") & second.ne(""), first.ne("") | second.ne("")],
                                   ["agree", "disagree", "one source"], default="none")
    stated = pd.to_numeric(d["second_deal_value_usd"], errors="coerce")
    by_equity, by_ev = _compare(d["equity_value"], stated, value_tol), _compare(d["enterprise_value"], stated, value_tol)
    d["check_value"] = np.select([by_equity.eq("agree") | by_ev.eq("agree"),
                                  by_equity.eq("disagree") | by_ev.eq("disagree"),
                                  by_equity.eq("one source") | by_ev.eq("one source")],
                                 ["agree", "disagree", "one source"], default="none")
    other = d["second_kind"].fillna("").astype(str).eq("other merger filing")
    for term in ("equity", "debt"):
        c = _compare(d[f"{term}_commitment_usd"], d[f"second_{term}_commitment_usd"].where(other), commitment_tol)
        d[f"check_{term}"] = c
    terms = d[CHECK_COLUMNS[:-1]]
    d["check_flag"] = np.select([terms.eq("disagree").any(axis=1).to_numpy(dtype=bool),
                                 terms.eq("agree").any(axis=1).to_numpy(dtype=bool)],
                                ["disagree", "agree"], default="unchecked")
    return d


def check_sample(deals: pd.DataFrame, n: int = 30, seed: int = 0) -> list:
    """`n` deals to read, drawn by a hash of the seed and the deal id – nobody chooses them,
    and a deal keeps its place when the sample grows. Pool: completed deals whose document
    was read (every read deal when none is completed)."""
    import hashlib
    if deals is None or not len(deals) or "text_chars" not in deals.columns:
        return []
    pool = deals[pd.to_numeric(deals["text_chars"], errors="coerce").fillna(0) > 0]
    if "completed" in pool.columns and pool["completed"].fillna(False).astype(bool).any():
        pool = pool[pool["completed"].fillna(False).astype(bool)]
    ids = sorted(pool["deal_id"].astype(str), key=lambda i: hashlib.sha256(f"{int(seed)}:{i}".encode()).hexdigest())
    return ids[:int(n)]


def _money(v) -> str:
    v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.isna(v):
        return "n/a"
    return f"${v / 1e9:.2f}bn" if abs(v) >= 1e9 else f"${v / 1e6:.1f}m" if abs(v) >= 1e6 else f"${v:,.0f}"


def _num(v) -> str:
    v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(v) else f"{v:g}"


def _txt(v) -> str:
    return "" if v is None or v is pd.NaT or v is pd.NA or (isinstance(v, float) and np.isnan(v)) else str(v)


def check_sheet(deals: pd.DataFrame, ids: list) -> pd.DataFrame:
    """The reading sheet: one row per sampled deal and term – the extracted value, the passage
    it came from, the second document's value and passage, and the comparison."""
    cols = ["n", "deal_id", "target_name", "term", "extracted", "passage", "source_url", "second_kind", "second_value",
            "second_passage", "second_url", "check", "reader_verdict", "reader_note"]
    if not ids:
        return pd.DataFrame(columns=cols)
    by = deals.assign(deal_id=deals["deal_id"].astype(str)).set_index("deal_id")
    rows = []
    for n, i in enumerate(ids, 1):
        r = by.loc[i]
        g = lambda c: r.get(c) if c in by.columns else None  # noqa: E731
        common = {"n": n, "deal_id": i, "target_name": _txt(g("target_name")), "source_url": _txt(g("doc_url")),
                  "second_kind": _txt(g("second_kind")), "second_url": _txt(g("second_url")),
                  "reader_verdict": "", "reader_note": ""}
        value = (f"{_money(g('second_deal_value_usd'))} ({_txt(g('second_deal_value_kind'))})"
                 if _money(g("second_deal_value_usd")) != "n/a" else "n/a")
        for term, extracted, passage, second, second_passage, check in (
                ("offer price", _num(g("offer_price")), g("price_snippet"), _num(g("second_offer_price")),
                 g("second_price_snippet"), g("check_price")),
                ("equity commitment", _money(g("equity_commitment_usd")), g("equity_snippet"),
                 _money(g("second_equity_commitment_usd")), "", g("check_equity")),
                ("debt commitment", _money(g("debt_commitment_usd")), g("debt_snippet"),
                 _money(g("second_debt_commitment_usd")), "", g("check_debt")),
                ("sponsor", _txt(g("sponsor")) or "n/a", g("sponsor_snippet"), _txt(g("second_sponsors")) or "n/a", "",
                 g("check_sponsor")),
                ("deal value", f"equity {_money(g('equity_value'))}, enterprise {_money(g('enterprise_value'))}",
                 "offer price × shares; less net cash", value, g("second_value_snippet"), g("check_value"))):
            rows.append({**common, "term": term, "extracted": extracted, "passage": _txt(passage),
                         "second_value": second, "second_passage": _txt(second_passage), "check": _txt(check)})
    return pd.DataFrame(rows, columns=cols)


def check_lines(deals: pd.DataFrame, ids: list) -> list[str]:
    """The reading sheet as text for the paste block: per deal a header, then one line per term."""
    sheet = check_sheet(deals, ids)
    if not len(sheet):
        return []
    by = deals.assign(deal_id=deals["deal_id"].astype(str)).set_index("deal_id")
    lines = []
    for (n, i), g in sheet.groupby(["n", "deal_id"], sort=True):
        r, h = by.loc[i], g.iloc[0]
        filed = pd.to_datetime(r.get("first_filed"), errors="coerce")
        lines.append(f"CHECK {n:02d}  {i} · {h['target_name']} · {_txt(r.get('doc_form'))} · first filed "
                     f"{'n/a' if pd.isna(filed) else filed.date()} · {h['source_url'] or 'no document'}")
        second_filed = pd.to_datetime(r.get("second_filed"), errors="coerce") if "second_filed" in by.columns else pd.NaT
        lines.append(f"  second source: {h['second_kind'] or 'none found'}"
                     + (f" · {_txt(r.get('second_form'))} filed {second_filed.date()}" if pd.notna(second_filed) else "")
                     + (f" · {h['second_url']}" if h["second_url"] else ""))
        for x in g.itertuples():
            line = f"  {x.term}: {x.extracted}"
            if x.passage:
                line += f" | «{x.passage}»"
            if x.second_value != "n/a" or x.second_passage:
                line += f" || second: {x.second_value}" + (f" | «{x.second_passage}»" if x.second_passage else "")
            lines.append(line + f" [{x.check or 'none'}]")
    return lines


def check_summary(deals: pd.DataFrame, ids: list, seed: int) -> dict:
    """Counts for the paste block: second documents found and agreement by term, all deals."""
    if deals is None or not len(deals) or "check_flag" not in deals.columns:
        return {}
    kind = deals["second_kind"].fillna("").astype(str)
    read = pd.to_numeric(deals["second_text_chars"], errors="coerce").fillna(0) > 0
    out = {"deals": int(len(deals)), "second_8k": int((kind.eq("8-K announcement") & read).sum()),
           "second_other_filing": int((kind.eq("other merger filing") & read).sum()),
           "second_none": int((~read).sum())}
    for term in ("price", "sponsor", "value", "equity", "debt"):
        c = deals[f"check_{term}"].value_counts()
        both = int(c.get("agree", 0) + c.get("disagree", 0))
        out[term] = (f"{int(c.get('agree', 0))} of {both} agree" if both else "no pairs")
    out["deals_flagged"] = int(deals["check_flag"].eq("disagree").sum())
    out["deals_confirmed"] = int(deals["check_flag"].eq("agree").sum())
    out["reading_sample"] = f"{len(ids)} deals, seed {seed}"
    return out


# ---------------------------------------------------------------------------
# Public benchmark and summary
# ---------------------------------------------------------------------------

def benchmark(features: pd.DataFrame, facts: pd.DataFrame, subs: pd.DataFrame, *, current_months: int = 15) -> pd.DataFrame:
    """Median EV/EBITDA and EV/revenue of the Paper B universe at each start date."""
    cols = ["start_date", "firms", "ebitda_positive", "median_ev_ebitda", "median_ev_revenue"]
    if features is None or not len(features) or facts is None or not len(facts):
        return pd.DataFrame(columns=cols)
    uni = features[features["in_universe"].astype(bool)][["cik", "start_date", "market_cap"]].dropna()
    uni = uni.assign(cik=uni["cik"].astype("int64"), start_date=pd.to_datetime(uni["start_date"]).astype("datetime64[ns]"))
    ciks = uni["cik"].unique()
    f = facts[facts["cik"].astype("int64").isin(ciks)]
    s = subs[subs["cik"].astype("int64").isin(ciks)]
    snap = fu.snapshots(f, s, sorted(uni["start_date"].unique()), current_months=current_months,
                        extra_flows=EXTRA_FLOWS)
    if not len(snap):
        return pd.DataFrame(columns=cols)
    snap["cik"] = snap["cik"].astype("int64")
    snap["start_date"] = pd.to_datetime(snap["start_date"]).astype("datetime64[ns]")
    dna = snap["dna"] + np.where(snap["dna_tag"].astype(str).eq("Depreciation"), snap["amortization"].fillna(0.0), 0.0)
    snap["ebitda"] = snap["operating_income"] + dna
    j = uni.merge(snap[["cik", "start_date", "ebitda", "revenue", "net_cash"]], on=["cik", "start_date"])
    j["ev"] = j["market_cap"] - j["net_cash"]
    j["ev_ebitda"] = (j["ev"] / j["ebitda"]).where(j["ebitda"] > 0)
    j["ev_revenue"] = (j["ev"] / j["revenue"]).where(j["revenue"] > 0)
    return (j.groupby("start_date").agg(firms=("cik", "size"), ebitda_positive=("ebitda", lambda x: float((x > 0).mean())),
                                        median_ev_ebitda=("ev_ebitda", "median"),
                                        median_ev_revenue=("ev_revenue", "median")).reset_index())


def summary(deals: pd.DataFrame) -> dict:
    """Counts for the paste block – no exit values, no returns."""
    if deals is None or not len(deals):
        return {"deals": 0}
    d = deals
    done = d[d["completed"]] if "completed" in d.columns else d.iloc[0:0]
    out = {"targets": int(d["target_cik"].nunique()), "deals": int(len(d)),
           "announced": f"{int(d['announce_year'].min())}-{int(d['announce_year'].max())}",
           "completed": int(len(done)), "not_completed_or_unseen": int(len(d) - len(done)),
           "fsds_filer": float(d["fsds_filer"].mean()), "main_document": float(d["doc_is_main"].mean())}
    if "text_chars" in d.columns:
        out["document_read"] = float((d["text_chars"].fillna(0) > 0).mean())
    for col, key in (("offer_price", "offer_price"), ("ebitda", "ebitda"), ("ev_ebitda", "ev_ebitda"),
                     ("equity_commitment_usd", "equity_commitment"), ("debt_commitment_usd", "debt_commitment")):
        if col in d.columns:
            out[f"{key}_known"] = float(d[col].notna().mean())
    if "sponsor" in d.columns:
        out["sponsor_named"] = float(d["sponsor"].fillna("").ne("").mean())
    if "sic" in d.columns:
        sic = pd.to_numeric(d["sic"], errors="coerce")
        out["financial_firms"] = int(sic.between(6000, 6799).sum())
    if "exit_route" in d.columns and len(done):
        routes = done["exit_route"].value_counts().to_dict()
        out.update({f"exit_{k.replace(' ', '_').replace('-', '')}": int(v) for k, v in routes.items()})
        out["post_buyout_financials"] = int(done["post_buyout_financials"].sum())
    return out
