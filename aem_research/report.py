"""Render the coverage audit as a short Markdown report for Drive, and the paste-ready
block that ends every run."""
from __future__ import annotations


def _pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return str(v)


def paste_block(report: dict) -> str:
    """One compact text block with every result and the audit trail (no returns before the lock)."""
    r = report
    v = r.get("verdict") or {}
    f = r.get("features") or {}
    lines = [
        "AEM RUN SUMMARY – paste everything between the rulers into the chat",
        "=" * 72,
        f"run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · config {str(r.get('config_sha256', ''))[:12]} "
        f"({r.get('lock_status', '')}) · {r.get('versions', {})}",
        f"DATA      SimFin {r.get('simfin', {}).get('tickers', 0):,} tickers ({r.get('simfin', {}).get('stopped_trading', 0):,} stopped)"
        f" · Stooq {r.get('stooq', {}).get('tickers', 0):,} · SEC index {r.get('sec', {})}",
        "SEC SETS  " + (" | ".join(f"{k} {x}" for k, x in (r.get("sec_datasets") or {}).items()) or "not loaded"),
        f"VERDICT   EODHD {str(v.get('verdict', 'undetermined')).upper()} – unpriced {_pct(v.get('missing_share'))}, "
        f"coverage {_pct(v.get('coverage'))}" + (f", optimistic {_pct(v['optimistic'].get('coverage'))} "
                                                f"-> {str(v['optimistic'].get('verdict', '')).upper()}" if v.get("optimistic") else ""),
    ]
    if f:
        lines.append(f"UNIVERSE  {f.get('formation_dates')} start dates to {f.get('as_of')} · {f.get('universe_windows', 0):,} windows"
                     f" · {f.get('tickers_in_universe', 0):,} tickers · price from 13F {_pct(f.get('price_from_13f'))}"
                     f" · attention signal {f.get('attention_signal')}")
    for key, label in (("coverage_summary", "COVERAGE "), ("prelock", "PRE-LOCK  "), ("delistings", "DELISTED  "),
                       ("price_basis", "PRICES    "), ("holes", "HOLES     "), ("regime_summary", "REGIME    ")):
        if r.get(key):
            items = r[key].items() if isinstance(r[key], dict) else []
            lines.append(label + " · ".join(f"{k} {_pct(x) if isinstance(x, float) and abs(x) <= 1 else x}" for k, x in items))
    for row in r.get("power") or []:
        lines.append("POWER     " + " · ".join(f"{k} {x}" for k, x in row.items()))
    for row in r.get("base_rates") or []:
        lines.append("RESULT    " + " · ".join(f"{k} {x}" for k, x in row.items()))
    run = r.get("run") or {}
    if run.get("phases"):
        lines.append("PHASES    " + " · ".join(f"{p['phase']} {p['status']} {p['seconds']}s" for p in run["phases"]))
        lines.append(f"TIME      {run.get('total_minutes')} min · Drive use {r.get('drive_gb', '?')} GB")
    errs = run.get("errors") or []
    if errs:
        for e in errs:
            lines.append(f"ERROR     [{e['phase']}] {e['error']}")
            for w in e.get("where", []):
                lines.append(f"          {w}")
    else:
        lines.append("ERRORS    none")
    lines.append("=" * 72)
    return "\n".join(lines)


def _flat(d: dict) -> list[str]:
    return [f"- {k}: {v}" for k, v in d.items() if not isinstance(v, (dict, list))]


def render_markdown(report: dict) -> str:
    v = report.get("verdict") or {}
    lines = [
        "# AEM base-rate study – coverage audit",
        "",
        f"Generated: {report.get('generated_utc', '')}",
        f"Config: {str(report.get('config_sha256', ''))[:16]}… (lock: {report.get('lock_status', '')})",
        "",
        "## Is EODHD needed?",
        f"**{str(v.get('verdict', 'undetermined')).upper()}** – {v.get('reason', '')}",
        f"Basis: {v.get('basis', '')}",
        "",
    ]
    opt = v.get("optimistic")
    if opt:
        lines[-1:-1] = [f"Optimistic case (Stooq's extra delisted names fill the unmapped events): coverage "
                        f"{opt.get('coverage', 0):.0%}, verdict {str(opt.get('verdict', '')).upper()}"]
    for key, title in (("simfin", "SimFin cache"), ("stooq", "Stooq bulk"), ("overlap", "Overlap"),
                       ("adjustment", "Stooq adjustment basis"), ("sec", "SEC Form 25 index"),
                       ("sec_datasets", "SEC data sets"), ("features", "Features (counts only)"),
                       ("prelock", "Pre-lock checks"), ("delistings", "Delisting classes"),
                       ("price_basis", "Vendor price basis (share of names off the 13F price by >10%)"),
                       ("coverage_summary", "Feature coverage"), ("holes", "Universe holes"),
                       ("regime_summary", "Regime moderators (range across start dates)")):
        if report.get(key):
            lines += [f"## {title}", *_flat(report[key]), ""]
    return "\n".join(lines)
