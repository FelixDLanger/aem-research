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
