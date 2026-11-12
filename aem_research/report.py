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


# ---------------------------------------------------------------------------
# Exploration across both papers (notebook 06)
# ---------------------------------------------------------------------------

_SHARE_COLUMNS = {"median_premium_pct", "ci_low_pct", "ci_high_pct", "before_2022_pct", "from_2022_pct", "target_margin",
                  "benchmark_implied_margin", "share_below_benchmark", "targets_share", "universe_share", "difference",
                  "exit_rate", "median_premium_to_last", "p25", "p75", "share_offer_below_own_high", "public_mean_annual",
                  "fund_median_net_irr", "vs_sector_deals_pct", "touch_3x", "share_below_1x", "share_2x_or_more",
                  "sponsor_name_accuracy_in_reading"}
_SIGNED = {"median_premium_pct", "ci_low_pct", "ci_high_pct", "before_2022_pct", "from_2022_pct", "difference",
           "median_premium_to_last", "p25", "p75", "vs_sector_deals_pct", "public_mean_annual"}


def _cell(col: str, v) -> str:
    if col in ("p", "p_regime"):
        return _p(v)
    if col in _SHARE_COLUMNS:
        return _share(v, col in _SIGNED)
    if isinstance(v, str):
        return v
    if not _fin(v):
        return "n/a"
    f = float(v)
    return f"{int(f)}" if f == int(f) and abs(f) < 1e7 else f"{f:.2f}"


def explore_block(report: dict) -> str:
    """Notebook 06's paste block: one line per row of every exploratory table."""
    r = report
    v_ = f"v{r['release']}" if r.get("release") is not None else "v?"
    lines = [f"AEM EXPLORE {v_} – paste everything between the rulers into the chat", "=" * 72,
             f"{v_} · run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · notebook "
             f"{r.get('notebook_build', 'n/a')} · Paper A {r.get('lock_status_a', '')} · Paper B {r.get('lock_status_b', '')}"]
    problems = r.get("problems") or []
    lines.append("CHECKS    " + ("locks ok · data runs ok" if not problems else "STOPPED – " + " | ".join(problems)))
    if r.get("inputs"):
        lines.append("INPUTS    " + _items(r["inputs"]))
    if r.get("tables"):
        lines.append("EXPLORE   everything below is exploratory: reported in full, not a finding until confirmed on other data")
    for name, rows in (r.get("tables") or {}).items():
        if name.startswith("_"):
            continue
        if not rows:
            lines.append(f"E{name} · no rows")
        for row in rows:
            lines.append(f"E{name} · " + " · ".join(f"{k.replace('_', ' ')} {_cell(k, v)}" for k, v in row.items()))
    lines += _run_lines(r)
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper B's H2 to H6 (notebook 07)
# ---------------------------------------------------------------------------

_RATE_KEYS = {"base_rate", "hit_rate_with_signal", "hit_rate", "left_out_share", "test_base_rate", "test_hit_rate_with_signal",
              "p_3x_3y", "p_624x_5y", "share", "difference"}
_P_KEYS = {"p", "p_one_sided", "p_two_sided", "holm_p"}


def _kv(row: dict, skip=()) -> str:
    out = []
    for k, v in row.items():
        if k in skip or (isinstance(v, float) and not _fin(v) and k in ("ci_low", "ci_high", "p_two_sided")):
            continue
        if k in _P_KEYS:
            cell = _p(v)
        elif k in _RATE_KEYS:
            cell = _share(v, k == "difference")
        elif isinstance(v, str):
            cell = v
        elif not _fin(v):
            cell = "n/a"
        elif k in ("year", "start_year"):
            cell = f"{int(float(v))}"
        else:
            f = float(v)
            cell = f"{int(f):,}" if f == int(f) and abs(f) >= 100 else f"{int(f)}" if f == int(f) else f"{f:.4f}" if abs(f) < 0.1 else f"{f:.2f}"
        out.append(f"{k.replace('_', ' ')} {cell}")
    return " · ".join(out)


def _lr_line(r: dict) -> str:
    return (f"{r.get('with_signal', 0):,} windows of {r.get('windows', 0):,} ({r.get('firms', 0):,} firms) · hit rate "
            f"{_share(r.get('hit_rate_with_signal'))} against {_share(r.get('base_rate'))} · LR {_f(r.get('lr'))} "
            f"(95% {_f(r.get('ci_low'))} to {_f(r.get('ci_high'))}) · p {_p(r.get('p_one_sided'))}")


def _coef_line(label: str, row: dict) -> str:
    if row.get("note") or not _fin(row.get("coef")):
        return f"{label}{row['regressor']} · not estimated: {row.get('note') or 'unknown'}"
    return (f"{label}{row['regressor']} · odds ratio {_f(row['odds_ratio'])} · coef {row['coef']:+.3f} "
            f"(se {_f(row['se'], 3)}) · p {_p(row['p'])}")


def estimates_block(report: dict) -> str:
    """Notebook 07's paste block: H2 to H4 with their verdicts, the models, then the
    exploratory hypotheses, descriptives and robustness versions."""
    r = report
    v_ = f"v{r['release']}" if r.get("release") is not None else "v?"
    lock, dr = r.get("lock") or {}, (r.get("lock") or {}).get("data_run") or {}
    lines = [f"AEM RESULTS H2–H6 {v_} – paste everything between the rulers into the chat", "=" * 72,
             f"{v_} · run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · notebook "
             f"{r.get('notebook_build', 'n/a')} · config {str(r.get('config_sha256', ''))[:12]} ({r.get('lock_status', '')}"
             + (f" {lock.get('locked_at_utc')}" if lock.get("locked_at_utc") else "") + ")"
             + (f" · data run v{dr.get('release')} {dr.get('repo_commit')} {dr.get('generated_utc')}" if dr else "")]
    problems = r.get("problems") or []
    lines.append("CHECKS    " + ("lock ok · data run ok · gates all passed in the data run" if not problems
                                 else "STOPPED – " + " | ".join(problems)))
    if r.get("inputs"):
        lines.append("WINDOWS   " + _items(r["inputs"]))
    h2 = r.get("h2")
    if h2:
        lines.append(f"H2        3x within 3 years · {h2['windows']:,} windows with the cohort's inputs known ({h2['firms']:,} firms) · "
                     f"base rate {_share(h2['base_rate'])}")
        for k in ("H2a", "H2b"):
            m = h2["main"][k]
            lines.append(f"{k}       {_lr_line(m)} · Holm {_p(m.get('holm_p'))} · {m.get('verdict')}")
        lines.append(f"PLACEBO   insider sale, no purchase · {_lr_line(h2['placebo'])}")
        for row in h2.get("h2c") or []:
            lines.append("H2c       " + _kv(row))
        t = h2.get("h2c_test") or {}
        lines.append(f"H2c       opportunistic minus routine only · difference {_share(t.get('difference'), True)} "
                     f"(95% {_share(t.get('ci_low'), True)} to {_share(t.get('ci_high'), True)}) · p {_p(t.get('p_two_sided'))}")
        for row in h2.get("ablation") or []:
            lines.append(f"ABLATION  {row['cohort']} · {_lr_line(row)}")
    h3 = r.get("h3")
    if h3 and h3.get("windows"):
        lines.append(f"H3        {h3['windows']:,} paired windows ({h3['firms']:,} firms) · P(3x in 3y) {_share(h3['p_3x_3y'])} · "
                     f"P(6.24x in 5y) {_share(h3['p_624x_5y'])} · ratio {_f(h3['ratio'])} (95% {_f(h3['ci_low'])} to "
                     f"{_f(h3['ci_high'])}) · p {_p(h3['p_two_sided'])} · {h3['verdict']} · five-year multiple reached as often "
                     f"as 3x in three years: {_f(h3['equivalent_multiple_5y'])}x")
    h4 = r.get("h4")
    if h4:
        lines.append(f"H4        10x within 5 years · fitted on start dates to 2016 ({h4['fit_windows']:,} windows, base rate "
                     f"{_share(h4['fit_base_rate'])}) · tested on 2017–2021 ({h4['test_windows']:,} windows, base rate "
                     f"{_share(h4['test_base_rate'])})")
        for s in h4.get("table") or []:
            lines.append(f"H4        {s['name']} · fit LR {_f(s['fit_lr'])} · test LR {_f(s['test_lr'])} (95% {_f(s['test_ci_low'])} to "
                         f"{_f(s['test_ci_high'])}) · hit rate {_share(s['test_hit_rate_with_signal'])} with the signal "
                         f"({s['test_with_signal']:,} windows) · p {_p(s['p_one_sided'])} · Holm {_p(s['holm_p'])} · {s['verdict']}"
                         f" · unknown {_share(s['left_out_share'])}")
    m2 = r.get("m2")
    if m2:
        lines.append(f"M2        10x logit, start-date fixed effects, firm clusters · {m2.get('windows', 0):,} complete windows "
                     f"(left out {_share(m2.get('left_out_share'))})")
        for row in m2.get("coefficients") or []:
            lines.append(_coef_line("M2        ", row))
        for row in m2.get("brier") or []:
            lines.append("BRIER     " + _kv({k: (f"{v:.5f}" if k in ("brier", "ci_low", "ci_high") and _fin(v) else v) for k, v in row.items()}))
        if m2.get("terminal_uses"):
            lines.append(f"BRIER     the terminal uses {m2['terminal_uses']} (H4 benchmark rule)")
    m1 = r.get("m1")
    if m1:
        lines.append(f"M1        ordered logit on the five-year buckets · {m1.get('windows', 0):,} complete windows · converged "
                     f"{m1.get('converged')}")
        for row in m1.get("coefficients") or []:
            lines.append(_coef_line("M1        ", row))
    for label, key in (("H5 expl  ", "h5"), ("H6 expl  ", "h6"), ("TOUCHED   ", "touch_years"), ("DELISTED  ", "delisting_classes"),
                       ("ROBUST    ", "robustness"), ("ROBUST B  ", "robustness_b")):
        for row in r.get(key) or []:
            lines.append(label + _kv(row))
    if r.get("not_computed"):
        lines.append("PENDING   robustness versions that need a rebuilt panel, not computed here: " + "; ".join(r["not_computed"]))
    lines += _run_lines(r)
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Follow-up, round one (notebook 08)
# ---------------------------------------------------------------------------

def followup_block(report: dict) -> str:
    """Notebook 08's paste block: the robustness versions computed from the saved tables, the
    state of the confirmation plan, and the inventory of the Drive for the next round."""
    r = report
    v_ = f"v{r['release']}" if r.get("release") is not None else "v?"
    lines = [f"AEM FOLLOW-UP {v_} – paste everything between the rulers into the chat", "=" * 72,
             f"{v_} · run {r.get('generated_utc', '')} · repo {r.get('repo_commit', '')} · notebook "
             f"{r.get('notebook_build', 'n/a')} · Paper B {r.get('lock_status', '')} · confirmation plan "
             f"{r.get('confirmation_status', '')}"]
    problems = r.get("problems") or []
    lines.append("CHECKS    " + ("lock ok · data run ok · gates all passed in the data run" if not problems
                                 else "STOPPED – " + " | ".join(problems)))
    for label, key in (("YEARS     ", "two_way"), ("UNIVERSE  ", "universes"), ("BASE RATE ", "base_rates"),
                       ("NASDAQ-30 ", "nasdaq"), ("SPLITS    ", "weak_splits")):
        for row in r.get(key) or []:
            lines.append(label + _kv(row))
    if r.get("pending"):
        lines.append("ROUND 2   still to compute: " + "; ".join(r["pending"]))
    for line in r.get("inventory") or []:
        lines.append("DRIVE     " + line)
    lines += _run_lines(r)
    lines.append("=" * 72)
    return "\n".join(lines)
