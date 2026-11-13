"""Paper A, exploratory (A-X1, A-X2): what the visible exits were sold for.

An exit to a listed buyer is visible as the buyer's 8-K "completion of acquisition" that
names the target (notebook 02 found the filings; no value was read before the lock). Here
the 8-K and its press release are read and the stated price is taken from the sentences
around the target's name. The extraction is pattern-based: every value carries its sentence,
a seeded sample is printed for reading, and the tables are exploratory.

What the public record cannot give: the target's EBITDA at exit (it no longer files), so
the change in the multiple itself is not observed. The tables compare the growth of the
enterprise value between entry and exit with the public benchmark's multiple over the
same span.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import takeprivate as tp
from .data.sec import SecAccessError
from .io_utils import load_table, save_table

EXIT_VERSION = 1
EXIT_COLUMNS = ["deal_id", "sale_adsh", "exit_file", "exit_url", "exit_value_usd", "exit_value_kind", "exit_snippet",
                "exit_name_mentions", "exit_docs_read", "exit_error", "exit_version"]
_M = r"(?:US)?\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(million|billion)"
_APPROX = r"(?:approximately |about |roughly |an aggregate of |a total of |up to )*"
# strongest statement first; each is searched only in the sentences around the target's name
_PATTERNS = [
    ("enterprise", re.compile(r"enterprise value of\s+" + _APPROX + _M, re.I)),
    ("transaction", re.compile(r"(?:valued at|(?:transaction|aggregate|equity) value of|"
                               r"total (?:transaction |equity |purchase )?(?:value|consideration|price) of)\s+" + _APPROX + _M, re.I)),
    ("purchase price", re.compile(r"(?:purchase price|(?:aggregate |cash |total |merger |base )+consideration|"
                                  r"(?:aggregate |total )?cash (?:purchase )?price)(?: for the (?:acquisition|transaction))?"
                                  r"(?:,? (?:of|was|is|were|totaled|totaling|equal to|paid (?:at closing )?(?:of|was)))\s+"
                                  + _APPROX + _M, re.I)),
    ("purchase price", re.compile(r"(?:acquire[ds]?|acquisition of|purchase[ds]?|bought)\b[^.$]{0,220}?\bfor\s+"
                                  r"(?:(?:total |aggregate |net )?(?:cash )?(?:consideration|proceeds) of |a (?:purchase )?price of )?"
                                  + _APPROX + _M, re.I)),
]
_SKIP = re.compile(r"termination fee|revenue|sales of|net income|ebitda|credit facilit|term loan|senior notes|"
                   r"notes due|borrow|financ(?:ed|ing) (?:with|through)|per share|backlog|assets under", re.I)
WINDOW = 700


def exit_candidates(deals: pd.DataFrame) -> pd.DataFrame:
    """Completed deals sold to a listed buyer, with the buyer's filing."""
    need = {"exit_route", "sale_adsh", "buyer_cik"}
    if deals is None or not len(deals) or need - set(deals.columns):
        return pd.DataFrame(columns=["deal_id", "target_name", "sale_adsh", "buyer_cik"])
    d = deals[deals["exit_route"].eq("sold to a listed buyer") & deals["sale_adsh"].notna() & deals["buyer_cik"].notna()]
    return d[["deal_id", "target_name", "sale_adsh", "buyer_cik"]].reset_index(drop=True)


def extract_exit_value(text: str, target_core: str | None) -> dict:
    """The price stated for the target: the strongest statement within `WINDOW` characters of
    a mention of its name. Without a mention nothing is taken – another deal's price is worse
    than none."""
    out = {"exit_value_usd": np.nan, "exit_value_kind": "", "exit_snippet": "", "exit_name_mentions": 0}
    t = " ".join(str(text or "").split())
    if not t or not target_core:
        return out
    spots = [m.start() for m in re.finditer(re.escape(target_core), t, flags=re.I)]
    out["exit_name_mentions"] = len(spots)
    best = None
    for s in spots:
        lo, hi = max(0, s - WINDOW), min(len(t), s + len(target_core) + WINDOW)
        chunk = t[lo:hi]
        for rank, (kind, pat) in enumerate(_PATTERNS):
            for m in pat.finditer(chunk):
                # the two weaker forms are not taken where the sentence speaks of something else
                if rank >= 2 and _SKIP.search(chunk[max(0, m.start() - 80):m.end() + 30]):
                    continue
                value = tp._usd(m.group(1), m.group(2), m.group(3))
                if not np.isfinite(value) or value < 5e6:
                    continue
                dist = abs((lo + m.start()) - s)
                cand = (rank, dist, value, kind, tp._snippet(t, lo + m.start(), lo + m.end(), pad=200))
                if best is None or cand[:2] < best[:2]:
                    best = cand
    if best is not None:
        out.update(exit_value_usd=float(best[2]), exit_value_kind=best[3], exit_snippet=best[4])
    return out


def _order(listed: pd.DataFrame) -> list[str]:
    """Press releases first (EX-99), then the 8-K itself, then other exhibits; text documents only."""
    if listed is None or not len(listed):
        return []
    f = listed[listed["file_name"].astype(str).str.lower().str.endswith((".htm", ".html", ".txt"))].copy()
    typ = f["type"].astype(str).str.upper()
    f["_rank"] = np.select([typ.str.startswith("EX-99"), typ.str.startswith("8-K"), typ.str.startswith("EX-2")], [0, 1, 3], 2)
    return list(f.sort_values("_rank", kind="mergesort")["file_name"].head(5))


def read_exits(deals: pd.DataFrame, *, client=None, cache_dir, offline: bool = False, log=print,
               save_every: int = 25) -> tuple[pd.DataFrame, dict]:
    """The stated exit value of every visible sale. Texts are kept on Drive (gzipped), so a
    better extractor (EXIT_VERSION) re-reads them without downloading again."""
    cache_dir = Path(cache_dir)
    docs = cache_dir / "docs"
    cand = exit_candidates(deals)
    stats = {"sales": int(len(cand)), "cached": 0, "read": 0, "with_value": 0, "no_name_in_text": 0, "missing": 0, "failed": 0}
    old = load_table(cache_dir / "exits")
    have = {}
    if old is not None and len(old) and "exit_version" in old.columns:
        have = {(r["deal_id"], r["sale_adsh"]): r for r in old[old["exit_version"] == EXIT_VERSION].to_dict("records")}
    rows, done = [], []

    def flush():
        if done:
            keys = {(r["deal_id"], r["sale_adsh"]) for r in done}
            save_table(pd.DataFrame([r for k, r in have.items() if k not in keys] + done).reindex(columns=EXIT_COLUMNS),
                       cache_dir / "exits")

    for c in cand.itertuples():
        key = (c.deal_id, c.sale_adsh)
        if key in have:
            rows.append(have[key])
            stats["cached"] += 1
            continue
        cik, adsh = int(c.buyer_cik), str(c.sale_adsh)
        core = tp.exit_query_name(c.target_name)
        rec = {"deal_id": c.deal_id, "sale_adsh": adsh, "exit_version": EXIT_VERSION}
        try:
            page = tp._cached_text(client, tp.index_url(cik, adsh), docs / f"{adsh}-index.htm.gz", offline)
            if page is None:
                stats["missing"] += 1
                rows.append({**rec, "exit_error": "filing index not available"})
                continue
            best, read = None, 0
            for name in _order(tp.filing_documents(page)):
                url = tp.document_url(cik, adsh, name)
                text = tp._cached_text(client, url, docs / f"{adsh}_{tp._safe(name)}.txt.gz", offline, transform=tp.html_to_text)
                if text is None:
                    continue
                read += 1
                v = extract_exit_value(text, core)
                v.update(exit_file=name, exit_url=url)
                if best is None or (np.isfinite(v["exit_value_usd"]) and not np.isfinite(best["exit_value_usd"])) or \
                        (not np.isfinite(best["exit_value_usd"]) and v["exit_name_mentions"] > best["exit_name_mentions"]):
                    best = v
                if np.isfinite(v["exit_value_usd"]) and v["exit_value_kind"] in ("enterprise", "transaction"):
                    break
        except SecAccessError:
            raise                                  # a refused User-Agent stops the phase with its message
        except Exception as exc:                   # one bad filing never stops the rest
            stats["failed"] += 1
            rows.append({**rec, "exit_error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        if best is None:
            stats["missing"] += 1
            rows.append({**rec, "exit_docs_read": read, "exit_error": "no document available"})
            continue
        rec.update(best, exit_docs_read=read)
        rows.append(rec)
        done.append(rec)
        stats["read"] += 1
        if stats["read"] % save_every == 0:
            flush()
            if log:
                log(f"  exit filings read: {stats['read']:,} of {len(cand):,}")
    flush()
    out = pd.DataFrame(rows).reindex(columns=EXIT_COLUMNS)
    val = pd.to_numeric(out["exit_value_usd"], errors="coerce")
    stats["with_value"] = int(val.notna().sum())
    stats["no_name_in_text"] = int((pd.to_numeric(out["exit_name_mentions"], errors="coerce").fillna(0) == 0).sum())
    return out.astype({"deal_id": "object"}), stats


def _bench_at(bench: pd.DataFrame | None, dates: pd.Series) -> np.ndarray:
    """The benchmark's median EV/EBITDA at the latest start date on or before each date."""
    if bench is None or not len(bench):
        return np.full(len(dates), np.nan)
    b = bench.dropna(subset=["median_ev_ebitda"]).assign(start_date=lambda x: pd.to_datetime(x["start_date"])).sort_values("start_date")
    pos = np.searchsorted(b["start_date"].to_numpy(), pd.to_datetime(dates).to_numpy(), side="right") - 1
    vals = b["median_ev_ebitda"].to_numpy(dtype=float)
    return np.where((pos >= 0) & pd.to_datetime(dates).notna().to_numpy(), vals[np.clip(pos, 0, len(vals) - 1)], np.nan)


def exit_table(deals: pd.DataFrame, exits: pd.DataFrame, bench: pd.DataFrame | None, *, regime_split="2022-01-01",
               plausible=(0.1, 15.0)) -> pd.DataFrame:
    """One row per visible sale with a stated value: entry and exit enterprise value, the
    multiple of value, years held, the annual rate, and the benchmark's multiple at both dates."""
    keep = ["deal_id", "target_name", "buyer_name", "first_filed", "completion_date", "sale_date", "enterprise_value", "ev_ebitda",
            "core_sample", "comparison_group", "buyer_type", "post_buyout_financials"]
    d = deals[[c for c in keep if c in deals.columns]].merge(
        exits[["deal_id", "exit_value_usd", "exit_value_kind", "exit_snippet", "exit_url"]], on="deal_id")
    d = d[pd.to_numeric(d["exit_value_usd"], errors="coerce").notna()].copy()
    for c in ("first_filed", "completion_date", "sale_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    ev = pd.to_numeric(d["enterprise_value"], errors="coerce")
    d["value_multiple"] = (d["exit_value_usd"] / ev.where(ev > 0)).astype(float)
    start = d["completion_date"].fillna(d["first_filed"])
    d["years_held"] = (d["sale_date"] - start).dt.days / 365.25
    ok = d["value_multiple"].between(*plausible) & (d["years_held"] > 0.25)
    d["usable"] = ok.fillna(False)
    d["annual_rate"] = np.where(d["usable"], d["value_multiple"] ** (1 / d["years_held"].where(d["years_held"] > 0)) - 1, np.nan)
    d["bench_entry"], d["bench_exit"] = _bench_at(bench, d["first_filed"]), _bench_at(bench, d["sale_date"])
    d["bench_change"] = d["bench_exit"] / d["bench_entry"]
    d["entry_regime"] = np.where(d["first_filed"] < pd.Timestamp(regime_split), "entered before 2022", "entered 2022 or later")
    d["exit_regime"] = np.where(d["sale_date"] < pd.Timestamp(regime_split), "sold before 2022", "sold 2022 or later")
    return d.reset_index(drop=True)


def _ci_median(v: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    if len(v) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), size=(draws, len(v)))], axis=1)
    return (float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975)))


def summaries(x: pd.DataFrame, *, draws: int = 5000, seed: int = 20261111) -> dict:
    """The tables: the multiple of value by group, and value growth against the benchmark's multiple."""
    cols = ["by", "group", "n", "median_value_multiple", "ci_low", "ci_high", "median_years_held", "median_annual_rate",
            "share_below_entry_value", "median_bench_change"]
    u = x[x["usable"]] if len(x) else x
    rows = []

    def add(by, name, g):
        v = g["value_multiple"].to_numpy(dtype=float)
        lo, hi = _ci_median(v, draws, seed)
        rows.append({"by": by, "group": name, "n": int(len(g)), "median_value_multiple": float(np.median(v)) if len(v) else np.nan,
                     "ci_low": lo, "ci_high": hi, "median_years_held": float(g["years_held"].median()) if len(g) else np.nan,
                     "median_annual_rate": float(g["annual_rate"].median()) if len(g) else np.nan,
                     "share_below_entry_value": float((v < 1).mean()) if len(v) else np.nan,
                     "median_bench_change": float(g["bench_change"].median()) if len(g) else np.nan})
    if len(u):
        add("all", "all usable sales", u)
        for by in ("entry_regime", "exit_regime", "exit_value_kind", "buyer_type"):
            if by in u.columns:
                for name, g in u.groupby(by):
                    add(by.replace("_", " "), str(name), g)
        if "core_sample" in u.columns:
            add("sample", "core sample", u[u["core_sample"].fillna(False).astype(bool)])
    groups = pd.DataFrame(rows, columns=cols)
    # A-X2 in the form the record allows: does value growth follow the public multiple?
    track = {"n": 0}
    t = u.dropna(subset=["bench_change"]) if len(u) else u
    t = t[t["bench_change"] > 0] if len(t) else t
    track = {"n": int(len(t))}                             # fewer than ten sales: the count only, no estimate
    if len(t) >= 10:
        from scipy import stats
        ly, lx = np.log(t["value_multiple"].to_numpy(dtype=float)), np.log(t["bench_change"].to_numpy(dtype=float))
        rho = stats.spearmanr(lx, ly)
        X = np.column_stack([np.ones(len(lx)), lx])
        beta = np.linalg.lstsq(X, ly, rcond=None)[0]
        resid = ly - X @ beta
        bread = np.linalg.inv(X.T @ X)
        cov = bread @ (X.T * resid ** 2) @ X @ bread * len(lx) / max(len(lx) - 2, 1)       # HC1
        se = float(np.sqrt(cov[1, 1]))
        track = {"n": int(len(t)), "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue), "slope": float(beta[1]),
                 "slope_se": se, "slope_p": float(2 * stats.norm.sf(abs(beta[1] / se))) if se > 0 else np.nan,
                 "share_same_direction": float((np.sign(lx) == np.sign(ly)).mean())}
    counts = {"sales_with_value": int(len(x)), "usable": int(len(u)),
              "left_out_implausible_or_short": int(len(x) - len(u)),
              "entry_value_missing": int(x["value_multiple"].isna().sum()) if len(x) else 0,
              "with_post_buyout_financials": int(u["post_buyout_financials"].fillna(False).astype(bool).sum())
              if len(u) and "post_buyout_financials" in u.columns else 0}
    return {"groups": groups, "tracking": track, "counts": counts}


def reading_sample(x: pd.DataFrame, exits: pd.DataFrame, deals: pd.DataFrame, n: int = 30, seed: int = 20261112) -> list[str]:
    """A seeded sample of the read filings (with and without a value) for reading: one line each."""
    e = exits.merge(deals[["deal_id", "target_name", "buyer_name", "sale_date"]], on="deal_id", how="left")
    e = e[e["exit_error"].isna()] if "exit_error" in e.columns else e
    if not len(e):
        return []
    order = sorted(e["deal_id"], key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())[:n]
    e = e.set_index("deal_id")
    lines = []
    for i, k in enumerate(order, 1):
        r = e.loc[k]
        v = r["exit_value_usd"]
        value = f"${v / 1e6:,.0f}m ({r['exit_value_kind']})" if pd.notna(v) else "no value"
        when = pd.Timestamp(r["sale_date"]).date() if pd.notna(r["sale_date"]) else "n/a"
        lines.append(f"{i:02d} {str(r['target_name'])[:40]} → {str(r['buyer_name'])[:34]} · {when} · {value} · names {int(r['exit_name_mentions'] or 0)} · "
                     f"“{str(r['exit_snippet'] or '')[:330]}” · {r['exit_url']}")
    return lines


# ---------------------------------------------------------------------------
# The exit measure rebuilt (notebook 10). Notebook 02 took the earliest 8-K of any listed
# company that named the target next to "completion of acquisition"; read for the first time
# in notebook 09, most of those were not sales of the target. Here every hit is read and
# kept only if a sentence says the target itself was acquired. This is a correction made
# after the lock and after the results were seen; A-H3 on it is reported as a deviation.
# ---------------------------------------------------------------------------

VALID_VERSION = 1
VALID_COLUMNS = ["deal_id", "adsh", "file_name", "buyer_cik", "buyer_name", "file_date", "tier", "reason", "sentence",
                 "sponsor_named", "exit_value_usd", "exit_value_kind", "exit_snippet", "url", "valid_version"]
_OBJECT = (r"(?:acquisition|purchase|merger|buyout)\s+of\s+(?:all\s+|100%\s+|substantially\s+all\s+)?[^.;]{{0,110}}?{core}",
           r"(?:acquired?|acquires|acquiring|purchased?|bought)\s+(?:all\s+|100%\s+)?[^.;]{{0,110}}?{core}",
           r"{core}[^.;]{{0,160}}?(?:became|become|becoming|is\s+now|will\s+become|surviv\w+\s+as)\s+an?\s+(?:indirect\s+|direct\s+)?"
           r"(?:wholly[- ]owned\s+)?subsidiary",
           r"merge[drs]?\s+(?:with\s+and\s+)?(?:into|with)\s+[^.;]{{0,60}}?{core}")
_SUBJECT = (r"{core}(?:,?\s+(?:Inc|Corp|LLC|Ltd|L\.P|Holdings|Corporation|Company|Group)\.?,?)*(?:\s+\([^)]{{0,40}}\))?\s+"
            r"(?:today\s+)?(?:has\s+|have\s+|had\s+)?(?:completed|closed|announced|acquired|agreed\s+to\s+acquire|to\s+acquire)\b"
            r"(?![^.;]{{0,40}}\bby\b)")
_PASSIVE = (r"{core}[^.;]{{0,160}}?(?:became|become|becoming|is\s+now|will\s+become|surviv\w+\s+as)\s+an?\s+(?:indirect\s+|direct\s+)?"
            r"(?:wholly[- ]owned\s+)?subsidiary",
            r"{core}[^.;]{{0,80}}?\b(?:was|were|has\s+been|had\s+been|being|to\s+be)\s+acquired\s+by")
_SENTENCE_END = re.compile(r"(?<!\bInc)(?<!\bCorp)(?<!\bLtd)(?<!\bCo)(?<!\bL\.P)(?<!\bNo)(?<!\bU\.S)(?<!\bMr)(?<!\bMs)(?<!\bSt)"
                           r"\.\s+(?=[A-Z\"“(])")
_LESSOR = re.compile(r"tenant|leased? to|anchored by|lease with|net lease|occupied by|portfolio of \d+ (?:retail |net lease )?properties", re.I)


def _sponsor_names(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parts = re.split(r"[;|/]|,\s*(?=[A-Z])", str(value))
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def validate_exit(text: str, target_core: str | None, sponsors=None) -> dict:
    """Does this document say the target was acquired? Tier A: a sentence in which the target
    is what was acquired, and the deal's sponsor is named in the document (the seller).
    Tier B: such a sentence, sponsor not named (or not known). Otherwise rejected, with the reason."""
    out = {"tier": "", "reason": "", "sentence": "", "sponsor_named": False}
    t = " ".join(str(text or "").split())
    if not t or not target_core:
        return {**out, "reason": "no text or no specific name"}
    core = re.escape(target_core).replace(r"\ ", r"\s+")
    names = [m for m in re.finditer(core, t, flags=re.I) if m.group(0)[:1].isupper()]
    if not names:
        return {**out, "reason": "the name does not appear as a name"}
    low = t.lower()
    out["sponsor_named"] = any(s.lower() in low for s in _sponsor_names(sponsors))
    found, acquirer = None, False
    ends = [e.start() + 1 for e in _SENTENCE_END.finditer(t)]
    for m in names:
        before = [e for e in ends if e <= m.start()]
        after = [e for e in ends if e >= m.end()]
        lo = max(before[-1] if before else 0, m.start() - 400)
        hi = min(after[0] if after else len(t), m.end() + 400)
        sent = t[lo:hi].strip()
        if _LESSOR.search(sent):
            continue                                        # the target is a tenant or a counterparty, not what was bought
        if any(re.search(p.format(core=core), sent, flags=re.I) for p in _PASSIVE):
            found = sent
            break
        if re.search(_SUBJECT.format(core=core), sent, flags=re.I):
            acquirer = True                                 # the target is the one acquiring
            continue
        hit = False
        for p in _OBJECT:
            mm = re.search(p.format(core=core), sent, flags=re.I)
            # "acquired patents from Target": the target is the source, not the object
            if mm is None:
                continue
            tail = re.search(core + r"$", mm.group(0), flags=re.I)
            prefix = mm.group(0)[:tail.start()] if tail else ""
            if not re.search(r"\b(?:from|by|through|via)\s+(?:the\s+)?$", prefix, flags=re.I):
                hit = True
                break
        if hit:
            found = sent
            break
    if found is None:
        return {**out, "reason": "the target is the acquirer" if acquirer else "no sentence says the target was acquired"}
    return {**out, "tier": "A" if out["sponsor_named"] else "B", "sentence": found[:420]}


def exit_hits(deals: pd.DataFrame, hits: pd.DataFrame | None, *, per_deal: int = 10) -> pd.DataFrame:
    """Every 8-K hit of notebook 02's exit search that someone other than the target filed:
    up to `per_deal` filings a deal, earliest first, the press release before the 8-K itself."""
    cols = ["deal_id", "target_name", "sponsor", "adsh", "file_name", "buyer_cik", "buyer_name", "file_date"]
    if hits is None or not len(hits) or deals is None or not len(deals):
        return pd.DataFrame(columns=cols)
    d = deals[["deal_id", "target_cik", "target_name", "completion_date"] + (["sponsor"] if "sponsor" in deals.columns else [])]
    h = hits.merge(d, left_on="key", right_on="deal_id")
    first = h["ciks"].astype(str).str.split(";").str[0]
    h = h[first.str.isdigit() & (pd.to_numeric(first, errors="coerce") != pd.to_numeric(h["target_cik"], errors="coerce"))
          & h["file_name"].astype(str).ne("") & h["file_name"].notna()].copy()
    h["buyer_cik"] = pd.to_numeric(h["ciks"].astype(str).str.split(";").str[0], errors="coerce").astype("int64")
    h["buyer_name"] = h["names"].astype(str).map(tp._clean_display_name)
    same = tp.match_name(h["buyer_name"]).to_numpy() == tp.match_name(h["target_name"]).to_numpy()
    h = h[~same]                                           # a successor that carries the target's own name is not a buyer
    h["file_date"] = pd.to_datetime(h["file_date"]).astype("datetime64[ns]")
    h = h[h["file_date"] > pd.to_datetime(h["completion_date"])]
    ft = h["file_type"].astype(str).str.upper() if "file_type" in h.columns else pd.Series("", index=h.index)
    h["_rank"] = np.select([ft.str.startswith("EX-99"), ft.str.startswith("8-K")], [0, 1], 2)
    h = h.sort_values(["deal_id", "file_date", "_rank"]).drop_duplicates(["deal_id", "adsh", "file_name"])
    filings = h.drop_duplicates(["deal_id", "adsh"]).groupby("deal_id").cumcount()
    keep = h[["deal_id", "adsh"]].drop_duplicates().assign(_n=filings.to_numpy())
    h = h.merge(keep[keep["_n"] < per_deal][["deal_id", "adsh"]], on=["deal_id", "adsh"])
    if "sponsor" not in h.columns:
        h["sponsor"] = None
    return h[cols].reset_index(drop=True)


def validate_exits(deals: pd.DataFrame, hits: pd.DataFrame | None, *, client=None, cache_dir, offline: bool = False, log=print,
                   save_every: int = 100, per_deal: int = 10) -> tuple[pd.DataFrame, dict]:
    """Read every hit and judge it; one row per hit document. Texts and verdicts are cached."""
    cache_dir = Path(cache_dir)
    docs = cache_dir / "docs"
    cand = exit_hits(deals, hits, per_deal=per_deal)
    stats = {"deals_with_hits": int(cand["deal_id"].nunique()), "documents": int(len(cand)), "cached": 0, "read": 0, "missing": 0,
             "failed": 0}
    old = load_table(cache_dir / "exits_validated")
    have = {}
    if old is not None and len(old) and "valid_version" in old.columns:
        have = {(r["deal_id"], r["adsh"], r["file_name"]): r for r in old[old["valid_version"] == VALID_VERSION].to_dict("records")}
    rows, new = [], 0

    def flush():
        save_table(pd.DataFrame(list(have.values())).reindex(columns=VALID_COLUMNS), cache_dir / "exits_validated")

    for c in cand.itertuples():
        key = (c.deal_id, c.adsh, c.file_name)
        if key in have:
            rows.append(have[key])
            stats["cached"] += 1
            continue
        url = tp.document_url(int(c.buyer_cik), str(c.adsh), str(c.file_name))
        rec = {"deal_id": c.deal_id, "adsh": c.adsh, "file_name": c.file_name, "buyer_cik": int(c.buyer_cik), "buyer_name": c.buyer_name,
               "file_date": c.file_date, "url": url, "valid_version": VALID_VERSION}
        try:
            text = tp._cached_text(client, url, docs / f"{c.adsh}_{tp._safe(str(c.file_name))}.txt.gz", offline, transform=tp.html_to_text)
        except SecAccessError:
            raise
        except Exception as exc:
            stats["failed"] += 1
            rows.append({**rec, "tier": "", "reason": f"{type(exc).__name__}: {str(exc)[:100]}"})
            continue
        if text is None:
            stats["missing"] += 1
            rows.append({**rec, "tier": "", "reason": "document not available"})
            continue
        core = tp.exit_query_name(c.target_name)
        v = validate_exit(text, core, c.sponsor)
        rec.update(v)
        if v["tier"]:
            rec.update({k: x for k, x in extract_exit_value(text, core).items() if k != "exit_name_mentions"})
        rows.append(rec)
        have[key] = rec
        stats["read"] += 1
        new += 1
        if new % save_every == 0:
            flush()
            if log:
                log(f"  exit documents read: {stats['read']:,} of {len(cand):,}")
    if new:
        flush()
    out = pd.DataFrame(rows).reindex(columns=VALID_COLUMNS)
    return out, stats


def chosen_exits(validated: pd.DataFrame) -> pd.DataFrame:
    """One sale per deal: the earliest document of the best tier (A before B)."""
    cols = ["deal_id", "sale_date_checked", "buyer_checked", "tier", "sentence", "exit_value_usd", "exit_value_kind", "exit_snippet", "url"]
    v = validated[validated["tier"].isin(["A", "B"])] if validated is not None and len(validated) else None
    if v is None or not len(v):
        return pd.DataFrame(columns=cols)
    v = v.sort_values(["deal_id", "tier", "file_date"]).drop_duplicates("deal_id")
    return v.rename(columns={"file_date": "sale_date_checked", "buyer_name": "buyer_checked"})[cols].reset_index(drop=True)


def with_checked_exits(sample: pd.DataFrame, chosen: pd.DataFrame, *, tiers=("A", "B"), use_reipo: bool = True) -> pd.DataFrame:
    """The analysis sample with its exit columns recomputed from the checked sales (and the
    re-IPOs as notebook 02 found them). The run's as-of date is read back from the saved rows."""
    d = sample.copy()
    done = pd.to_datetime(d["completion_date"]).astype("datetime64[ns]")
    censored = ~d["exit_event"].fillna(False).astype(bool)
    ends = done[censored] + pd.to_timedelta((pd.to_numeric(d.loc[censored, "exit_years"], errors="coerce") * 365.25).round(), unit="D")
    as_of = ends.dropna().max() if censored.any() and ends.notna().any() else pd.Timestamp.today().normalize()
    c = chosen[chosen["tier"].isin(tiers)] if len(chosen) else chosen
    sale = d["deal_id"].map(c.set_index("deal_id")["sale_date_checked"]) if len(c) else pd.Series(pd.NaT, index=d.index)
    sale = pd.to_datetime(sale).astype("datetime64[ns]")
    reipo = pd.to_datetime(d["reipo_date"]).astype("datetime64[ns]") if use_reipo and "reipo_date" in d.columns else pd.Series(pd.NaT, index=d.index)
    exit_date = pd.concat([reipo, sale], axis=1).min(axis=1)
    exit_date = exit_date.where(exit_date > done)
    d["sale_date"], d["exit_date"] = sale, exit_date
    d["exit_kind"] = np.where(exit_date.isna(), "", np.where(reipo.eq(exit_date), "reIPO", "sold to a listed buyer"))
    d["exit_event"] = exit_date.notna() & (exit_date <= as_of)
    end = exit_date.where(d["exit_event"], as_of)
    d["exit_years"] = ((end - done).dt.days / 365.25).clip(lower=0)
    d.attrs["as_of"] = as_of
    return d


def exit_lines(deals: pd.DataFrame, validated: pd.DataFrame, chosen: pd.DataFrame, *, rejected: int = 12, seed: int = 20261113) -> list[str]:
    """Every checked sale with its sentence (all of them are read), then a seeded handful of
    rejected documents (to see what the rule turns away), then the re-IPOs as notebook 02 matched them."""
    names = deals.set_index("deal_id")
    lines = []
    for i, r in enumerate(chosen.sort_values(["tier", "sale_date_checked"]).itertuples(), 1):
        d = names.loc[r.deal_id]
        v = f"${r.exit_value_usd / 1e6:,.0f}m ({r.exit_value_kind})" if pd.notna(r.exit_value_usd) else "no value"
        lines.append(f"SALE {i:02d} tier {r.tier} · {str(d['target_name'])[:38]} ({str(d.get('sponsor') or 'no sponsor')[:30]}) → "
                     f"{str(r.buyer_checked)[:34]} · {pd.Timestamp(r.sale_date_checked).date()} · {v} · “{str(r.sentence)[:300]}” · "
                     f"value: “{str(r.exit_snippet or '')[:220]}” · {r.url}")
    rej = validated[~validated["tier"].isin(["A", "B"]) & ~validated["deal_id"].isin(set(chosen["deal_id"]))] if len(validated) else validated
    if len(rej):
        rej = rej.drop_duplicates("deal_id")
        order = sorted(rej["deal_id"], key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())[:rejected]
        for i, r in enumerate(rej.set_index("deal_id").loc[order].reset_index().itertuples(), 1):
            lines.append(f"REJECTED {i:02d} {str(names.loc[r.deal_id, 'target_name'])[:38]} → {str(r.buyer_name)[:34]} · "
                         f"{pd.Timestamp(r.file_date).date()} · {r.reason} · {r.url}")
    if "reipo_date" in deals.columns:
        for i, r in enumerate(deals[deals["reipo_date"].notna()].sort_values("reipo_date").itertuples(), 1):
            lines.append(f"REIPO {i:02d} {str(r.target_name)[:40]} (completed {pd.Timestamp(r.completion_date).date() if pd.notna(r.completion_date) else 'n/a'}) "
                         f"→ {str(getattr(r, 'reipo_company', ''))[:40]} · {getattr(r, 'reipo_form', '')} · {pd.Timestamp(r.reipo_date).date()}")
    return lines
