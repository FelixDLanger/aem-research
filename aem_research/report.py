"""Render the coverage audit as a short Markdown report for Drive, and the paste-ready
block that ends every run."""
from __future__ import annotations

import math


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
    v_ = f"v{r['release']}" if r.get("release") is not None else "v?"
    lines = [
        f"AEM RUN SUMMARY {v_} – paste everything between the rulers into the chat",
        "=" * 72,
        f"{v_} · run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · notebook {r.get('notebook_build', 'n/a')} "
        f"· config {str(r.get('config_sha256', ''))[:12]} "
        f"({r.get('lock_status', '')}) · {r.get('versions', {})}",
    ]
    if "simfin" in r or "stooq" in r:
        lines.append(f"DATA      SimFin {r.get('simfin', {}).get('tickers', 0):,} tickers ({r.get('simfin', {}).get('stopped_trading', 0):,} stopped)"
                     f" · Stooq {r.get('stooq', {}).get('tickers', 0):,} · SEC index {r.get('sec', {})}")
    else:
        lines.append("DATA      not loaded yet")
    lines.append("SEC SETS  " + (" | ".join(f"{k} {x}" for k, x in (r.get("sec_datasets") or {}).items()) or "not loaded"))
    if v:
        lines.append(f"VERDICT   EODHD {str(v.get('verdict', 'undetermined')).upper()} – unpriced {_pct(v.get('missing_share'))}, "
                     f"coverage {_pct(v.get('coverage'))}" + (f", optimistic {_pct(v['optimistic'].get('coverage'))} "
                                                             f"-> {str(v['optimistic'].get('verdict', '')).upper()}" if v.get("optimistic") else ""))
    else:
        lines.append("VERDICT   not reached")
    if f:
        who = (f"{f['firms_in_universe']:,} firms" if "firms_in_universe" in f
               else f"{f.get('tickers_in_universe', 0):,} tickers")
        lines.append(f"UNIVERSE  {f.get('formation_dates')} start dates to {f.get('as_of')} · {f.get('universe_windows', 0):,} windows"
                     f" · {who} · price from 13F {_pct(f.get('price_from_13f'))}"
                     f" · attention signal {f.get('attention_signal')}"
                     + (f" · basis {f['price_basis']}" if f.get("price_basis") else ""))
    for key, label in (("coverage_summary", "COVERAGE "), ("prelock", "PRE-LOCK  "), ("sec_prices", "SEC PANEL "),
                       ("splits", "SPLITS    "), ("restated", "RESTATED  "), ("sec_check", "SEC CHECK "),
                       ("liquidity", "LIQUIDITY "),
                       ("funnel_a", "FUNNEL A  "), ("a_vs_v", "A VS V    "), ("universe_b", "ROBUST B  "),
                       ("calibration", "CALIB     "), ("delistings", "DELISTED  "),
                       ("price_basis", "PRICES    "), ("holes", "HOLES     "), ("regime_summary", "REGIME    "),
                       ("eodhd", "EODHD     "), ("s1_gaps", "S1 GAPS   ")):
        if r.get(key):
            items = r[key].items() if isinstance(r[key], dict) else []
            lines.append(label + " · ".join(f"{k} {_pct(x) if isinstance(x, float) and abs(x) <= 1 else x}" for k, x in items))
    fred = r.get("fred") or {}
    if fred:
        parts = []
        for sid, val in fred.items():
            if "_" in sid:
                continue
            why = fred.get(f"{sid}_error") or (f"HTTP {fred[f'{sid}_http']}" if f"{sid}_http" in fred else "")
            parts.append(f"{sid} {val}" + (f" ({why})" if why and str(val).startswith("missing") else ""))
        lines.append("FRED      " + " · ".join(parts))
    for line in r.get("split_cases") or []:
        lines.append("SPLIT CASE " + str(line))
    for row in r.get("power") or []:
        lines.append("POWER     " + " · ".join(f"{k} {x}" for k, x in row.items()))
    for row in r.get("base_rates") or []:
        lines.append("RESULT    " + " · ".join(f"{k} {x}" for k, x in row.items()))
    for row in r.get("base_rates_b") or []:
        lines.append("RESULT B  " + " · ".join(f"{k} {x}" for k, x in row.items()))
    lines += _run_lines(r)
    lines.append("=" * 72)
    return "\n".join(lines)


def failed_gates(report: dict | None) -> list[str]:
    """The pre-lock checks a notebook-01 run failed (its PRE-LOCK entries marked BELOW or
    ABOVE); a run without pre-lock checks fails as such."""
    pl = (report or {}).get("prelock") or {}
    if not pl:
        return ["the run has no pre-lock checks"]
    return [f"{k} {v}" for k, v in pl.items() if isinstance(v, str) and ("BELOW" in v or "ABOVE" in v)]


def _pc(v) -> str:
    try:
        return f"{100 * float(v):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def results_block(report: dict) -> str:
    """Notebook 04's paste block: the checks, the outcome statuses and H1 (counts and rates)."""
    r = report
    v_ = f"v{r['release']}" if r.get("release") is not None else "v?"
    lock = r.get("lock") or {}
    dr = lock.get("data_run") or {}
    status, when = r.get("lock_status", ""), lock.get("locked_at_utc")
    state = (f"locked {when}" if status == "locked" and when
             else f"{status}, locked {when}" if when else status)
    lines = [f"AEM RESULTS {v_} – paste everything between the rulers into the chat", "=" * 72,
             f"{v_} · run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · notebook "
             f"{r.get('notebook_build', 'n/a')} · config {str(r.get('config_sha256', ''))[:12]} ({state})"
             + (f" · data run v{dr.get('release')} {dr.get('repo_commit')} {dr.get('generated_utc')}" if dr else "")]
    problems = r.get("problems") or []
    lines.append("CHECKS    " + ("lock ok · data run ok · gates all passed in the data run" if not problems
                                 else "STOPPED – " + " · ".join(problems)))
    o = r.get("outcomes") or {}
    if o:
        lines.append("OUTCOMES  " + " · ".join(f"{k} {x:,}" if isinstance(x, int) else f"{k} {x}" for k, x in o.items()))
    h1 = r.get("h1") or {}
    touch = h1.get("touch") or []
    buckets = h1.get("buckets") or []

    def tline(rows):
        return " · ".join(f"{x['horizon_years']}y: 3x {_pc(x.get('touch_3x'))} 10x {_pc(x.get('touch_10x'))} "
                          f"({x['windows']:,} windows, {x['firms']:,} firms)" for x in rows)

    def bline(x):
        shares = " · ".join(f"{k.split('_')[1]} {_pc(val)}" for k, val in x.items() if k.startswith("share_"))
        return (f"{shares} · mean {x['mean']:.2f} (bounds {x['mean_lower']:.2f} to {x['mean_upper']:.2f}) · "
                f"{x['windows']:,} windows")

    pooled = [x for x in touch if x["scope"] == "pooled"]
    if pooled:
        lines.append("H1 TOUCH  pooled · " + tline(pooled))
    for g in ("before 2022", "2022 and later"):
        rows = [x for x in touch if x["scope"] == "regime" and x["group"] == g]
        if rows:
            lines.append(f"REGIME    {g} · " + tline(rows))
    for x in buckets:
        if x["scope"] in ("pooled", "regime"):
            lines.append(f"H1 5Y     {x['group'] if x['scope'] == 'regime' else 'pooled'} · " + bline(x))
    for h, m in ((3, "3x"), (5, "10x")):
        rows = [x for x in touch if x["scope"] == "start year" and x["horizon_years"] == h]
        if rows:
            lines.append(f"BY YEAR   {h}y touch {m}: " + " · ".join(f"{x['group']} {_pc(x.get('touch_' + m))}" for x in rows))
    lines += _run_lines(r)
    lines.append("=" * 72)
    return "\n".join(lines)


def _run_lines(r: dict) -> list[str]:
    """Phases, time, memory, warnings and errors – the audit trail shared by both papers."""
    lines = []
    run = r.get("run") or {}
    if run.get("phases"):
        lines.append("PHASES    " + " · ".join(f"{p['phase']} {p['status']} {p['seconds']}s" for p in run["phases"]))
        drive = f" · Drive use {r['drive_gb']} GB" if r.get("drive_gb") is not None else ""
        lines.append(f"TIME      {run.get('total_minutes')} min{drive}")
        rose, top = [], 0.0                        # peak memory only rises: show where it rose
        for p in run["phases"]:
            if p.get("ram_gb") is not None and p["ram_gb"] > top:
                rose.append(f"{p['phase']} {p['ram_gb']}")
                top = p["ram_gb"]
        if run.get("peak_ram_gb") is not None:
            lines.append(f"RAM       peak {run['peak_ram_gb']} of {run.get('total_ram_gb', '?')} GB"
                         + (" · rose in: " + " · ".join(rose) if rose else ""))
    for w in run.get("warnings") or []:
        lines.append(f"WARNING   [{w['phase']}] {w['message']}")
    errs = run.get("errors") or []
    if errs:
        for e in errs:
            lines.append(f"ERROR     [{e['phase']}] {e['error']}")
            for w in e.get("where", []):
                lines.append(f"          {w}")
    else:
        lines.append("ERRORS    none")
    return lines


def _items(d: dict) -> str:
    return " · ".join(f"{k.replace('_', ' ')} {_pct(x) if isinstance(x, float) and abs(x) <= 1 else x}"
                      for k, x in d.items())


def paper_a_block(report: dict) -> str:
    """Paper A's paste block: inputs, search, deals, entry coverage, exit routes, WSIB – counts only."""
    r = report
    v_ = f"v{r['release']}" if r.get("release") is not None else "v?"
    lines = [
        f"PAPER A RUN SUMMARY {v_} – paste everything between the rulers into the chat",
        "=" * 72,
        f"{v_} · run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · notebook {r.get('notebook_build', 'n/a')} "
        f"· config A {str(r.get('config_sha256', ''))[:12]} "
        f"({r.get('lock_status', '')}) · {r.get('versions', {})}",
    ]
    for key, label in (("inputs", "INPUTS    "), ("index", "INDEX     "), ("search", "SEARCH    "),
                       ("deals", "DEALS     "), ("buyers", "BUYERS    "), ("entry", "ENTRY     "), ("exits", "EXITS     "),
                       ("by_year", "BY YEAR   "), ("bench", "BENCH     "), ("regime", "REGIME    "),
                       ("wsib", "WSIB      "), ("check", "CHECK     ")):
        if r.get(key):
            lines.append(label + (_items(r[key]) if isinstance(r[key], dict) else str(r[key])))
        elif key in ("inputs", "search", "deals"):
            lines.append(label + "not reached")
    if r.get("sec_user_agent"):                    # a refused request is usually a secret the notebook cannot read
        lines.insert(3, f"SEC ID    {r['sec_user_agent']}")
    lines += list(r.get("check_lines") or [])      # the seeded reading sample: values with their passages
    lines += _run_lines(r)
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
                       ("prelock", "Pre-lock checks"), ("sec_prices", "SEC price panel (D10)"),
                       ("splits", "Splits found"), ("sec_check", "SEC panel checks (no outcomes)"),
                       ("universe_b", "Robustness B universe"),
                       ("calibration", "Quarter-end over daily touch rates, 2009-2012 (ratios only)"),
                       ("delistings", "Delisting classes"),
                       ("price_basis", "Vendor price basis (share of names off the 13F price by >10%)"),
                       ("coverage_summary", "Feature coverage"), ("holes", "Universe holes"),
                       ("regime_summary", "Regime moderators (range across start dates)")):
        if report.get(key):
            lines += [f"## {title}", *_flat(report[key]), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper A results (notebook 05)
# ---------------------------------------------------------------------------

def _fin(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _share(v, signed: bool = False) -> str:
    if not _fin(v):
        return "n/a"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _p(v) -> str:
    if not _fin(v):
        return "n/a"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _f(v, digits: int = 2) -> str:
    return "n/a" if not _fin(v) else f"{v:.{digits}f}"


def _premium_line(t: dict) -> str:
    return (f"n {t['n']} · median premium {_share(t['median_premium_pct'], True)} (95% {_share(t['ci_low_pct'], True)} to "
            f"{_share(t['ci_high_pct'], True)}) · above benchmark {_share(t['share_above_benchmark'])} · p {_p(t['h1_p'])}")


def _regime_line(t: dict) -> str:
    return (f"before 2022 {_share(t['median_before_pct'], True)} (n {t['n_before_2022']}) · 2022 on "
            f"{_share(t['median_2022_on_pct'], True)} (n {t['n_2022_on']}) · difference in logs {_f(t['diff_log'])} "
            f"(95% {_f(t['diff_ci_low'])} to {_f(t['diff_ci_high'])}) · p {_p(t['h2_p'])}")


def _exit_line(t: dict) -> str:
    return (f"visible exit by {t['horizon_years']:g}y · completed {t['early_cohort']} {_share(t['early_rate'])} "
            f"({t['early_exits']} of {t['early_n']}) · completed {t['late_cohort']} {_share(t['late_rate'])} "
            f"({t['late_exits']} of {t['late_n']}) · p {_p(t['h3_p'])}")


def _rows(records, fmt, per_line: int = 6) -> list[str]:
    items = [fmt(r) for r in records]
    return [" · ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


def paper_a_results_block(report: dict) -> str:
    """Notebook 05's paste block: the checks, the sample, the three confirmatory tests with
    their robustness versions, then every exploratory table (prefixed X)."""
    r = report
    v_ = f"v{r['release']}" if r.get("release") is not None else "v?"
    lock, dr = r.get("lock") or {}, (r.get("lock") or {}).get("data_run") or {}
    status, when = r.get("lock_status", ""), lock.get("locked_at_utc")
    state = f"locked {when}" if status == "locked" and when else f"{status}, locked {when}" if when else status
    lines = [f"PAPER A RESULTS {v_} – paste everything between the rulers into the chat", "=" * 72,
             f"{v_} · run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · notebook "
             f"{r.get('notebook_build', 'n/a')} · config A {str(r.get('config_sha256', ''))[:12]} ({state})"
             + (f" · data run v{dr.get('release')} {dr.get('repo_commit')} {dr.get('generated_utc')}" if dr else "")]
    problems = r.get("problems") or []
    lines.append("CHECKS    " + ("lock ok · data run ok" if not problems else "STOPPED – " + " | ".join(problems)))
    if r.get("funnel"):
        lines.append("SAMPLE    " + _items(r["funnel"]))
    c = r.get("confirmatory")
    if c:
        holm, verdict = c["holm"], c["verdict"]
        lines.append(f"A-H1      core · {_premium_line(c['premium'])} · Holm {_p(holm['A-H1'])} · {verdict['A-H1']}")
        lines.append(f"A-H2      core · {_regime_line(c['premium'])} · Holm {_p(holm['A-H2'])} · {verdict['A-H2']}")
        lines.append(f"A-H3      core · {_exit_line(c['exits'])} · Holm {_p(holm['A-H3'])} · {verdict['A-H3']}")
        for name, t in c["robustness"].items():
            if "h1_p" in t:
                lines.append(f"ROBUST    {name} · {_premium_line(t)}")
                lines.append(f"ROBUST    {name} · {_regime_line(t)}")
            else:
                lines.append(f"ROBUST    {name} · {_exit_line(t)}")
    x = r.get("exploratory") or {}
    if x:
        lines.append("EXPLORE   everything below is exploratory: reported in full, not a finding until confirmed on other data")
    by: dict = {}
    for row in x.get("premium by group", []):
        by.setdefault(row["by"], []).append(row)
    for key, rows in by.items():
        for line in _rows(rows, lambda q: f"{q['group']} {_share(q['median_premium_pct'], True)} "
                                         f"({_share(q['share_above_benchmark'])} above, n {q['n']})", 5):
            lines.append(f"X PREMIUM by {key} · {line}")
    if x.get("premium regression"):
        lines.append("X REGRESS log premium on · " + " · ".join(
            f"{q['term']} {q['coef']:+.3f} (se {_f(q['se'], 3)}, p {_p(q['p'])})" for q in x["premium regression"])
            + f" · n {int(x['premium regression'][0]['n'])}")
    by = {}
    for row in x.get("exit probability", []):
        by.setdefault(row["by"], []).append(row)
    for key, rows in by.items():
        for line in _rows(rows, lambda q: f"{q['group']} 3y {_share(q.get('exit_by_3y'))} 5y {_share(q.get('exit_by_5y'))} "
                                         f"7y {_share(q.get('exit_by_7y'))} (n {q['n']}, exits {q['exits_seen']}, "
                                         f"reIPO share {_share(q['reipo_share_of_exits'])})", 3):
            lines.append(f"X EXITS   by {key} · {line}")
    for line in _rows(x.get("holding by exit year", []),
                      lambda q: f"{int(q['exit_year'])}: {int(q['exits'])} exits, {_f(q['median_years_held'], 1)}y held, "
                                f"{int(q['reipo'])} reIPO", 5):
        lines.append(f"X HOLDING exits seen by exit year · {line}")
    for line in _rows(x.get("leverage (unvalidated)", []),
                      lambda q: f"{q['group']} {q['measure']} {_f(q['median'])} (IQR {_f(q['p25'])} to {_f(q['p75'])}, "
                                f"n {q['n']}, term accuracy {_share(q['term_accuracy_in_reading'])})", 3):
        lines.append(f"X LEVERAGE unvalidated terms · {line}")
    for line in _rows((x.get("sponsors (unvalidated)") or [])[:15],
                      lambda q: f"{q['sponsor']} {int(q['deals'])} deals ({_share(q['median_premium_pct'], True)}, "
                                f"{int(q['visible_exits'])} exits)", 5):
        lines.append(f"X SPONSORS name accuracy {_share((x['sponsors (unvalidated)'][0]).get('sponsor_name_accuracy_in_reading'))} · {line}")
    for q in x.get("measurement", []):
        lines.append(f"X MEASURE {q['measure']} · n {q['n']} · median {_f(q['median'])} · p10 {_f(q['p10'])} · p90 "
                     f"{_f(q['p90'])} · within 15% {_share(q['within_15pct'])}")
    for line in _rows(x.get("WSIB by vintage", []),
                      lambda q: f"{int(q['vintage_year'])}: {int(q['funds'])} funds TVPI {_f(q['pooled_tvpi'])} DPI "
                                f"{_f(q['pooled_dpi'])} IRR {_share(q['median_net_irr'])}", 5):
        lines.append(f"X WSIB    buyout funds by vintage · {line}")
    for line in _rows(x.get("WSIB distribution pace", []),
                      lambda q: f"{str(q['report_date'])[:10]}: DPI {_f(q['dpi'])}"
                                + (f", distributed since last {q['distributed_since_last_report'] / 1e9:.2f}bn"
                                   if _fin(q.get("distributed_since_last_report")) else ""), 5):
        lines.append(f"X WSIB    distribution pace · {line}")
    for q in x.get("premium vs fund returns", []):
        lines.append(f"X WSIB    entry-year premium against the vintage's fund multiple · years {q['years']} · Spearman "
                     f"{_f(q['spearman'])}")
    lines += _run_lines(r)
    lines.append("=" * 72)
    return "\n".join(lines)
