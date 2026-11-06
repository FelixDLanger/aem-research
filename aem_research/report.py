"""Render the coverage audit as a short Markdown report for Drive."""
from __future__ import annotations


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
                       ("price_basis", "Vendor price basis (share of names off the 13F price by >10%)")):
        if report.get(key):
            lines += [f"## {title}", *_flat(report[key]), ""]
    return "\n".join(lines)
