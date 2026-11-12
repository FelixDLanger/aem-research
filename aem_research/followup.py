"""Notebook 08, round one: what is on Drive, described precisely enough to build the next
round against it without guessing. Counts, names, columns and date ranges only."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .io_utils import load_table

TABLE_SUFFIXES = (".pkl", ".parquet")


def describe_dir(path: Path, depth: int = 2, max_names: int = 8) -> list[str]:
    """Directories under `path` down to `depth`: files, megabytes, newest change, a few names."""
    lines = []
    path = Path(path)
    if not path.exists():
        return [f"{path.name}: missing"]

    def walk(p: Path, level: int):
        try:
            entries = sorted(p.iterdir(), key=lambda e: e.name)
        except OSError as exc:
            lines.append(f"{'  ' * level}{p.name}/: unreadable ({type(exc).__name__})")
            return
        files = [e for e in entries if e.is_file()]
        size = sum(e.stat().st_size for e in files) / 1e6
        newest = max((e.stat().st_mtime for e in files), default=None)
        kinds = pd.Series([e.suffix or "(none)" for e in files]).value_counts().head(4).to_dict() if files else {}
        names = ", ".join(e.name for e in files[:max_names]) + (" …" if len(files) > max_names else "")
        lines.append(f"{'  ' * level}{p.name}/ · {len(files)} files · {size:,.1f} MB"
                     + (f" · newest {datetime.fromtimestamp(newest, timezone.utc):%Y-%m-%d}" if newest else "")
                     + (f" · {kinds}" if kinds else "") + (f" · {names}" if names else ""))
        if level < depth:
            for e in entries:
                if e.is_dir():
                    walk(e, level + 1)
    walk(path, 0)
    return lines


def describe_table(path_stem: Path, max_cols: int = 60) -> str:
    """One line for a saved table: rows, columns, the range of each date column, the columns."""
    try:
        t = load_table(path_stem)
    except Exception as exc:                                # a broken file is a finding, not a crash
        return f"{Path(path_stem).name}: unreadable ({type(exc).__name__}: {str(exc)[:80]})"
    if t is None:
        return f"{Path(path_stem).name}: not on Drive"
    dates = []
    for c in t.columns:
        if pd.api.types.is_datetime64_any_dtype(t[c]) and t[c].notna().any():
            dates.append(f"{c} {t[c].min():%Y-%m-%d}..{t[c].max():%Y-%m-%d} ({t[c].nunique()} values)")
    cols = ", ".join(map(str, t.columns[:max_cols])) + (" …" if len(t.columns) > max_cols else "")
    return f"{Path(path_stem).name}: {len(t):,} rows × {len(t.columns)} · " + (" · ".join(dates[:4]) + " · " if dates else "") + cols


def value_counts_line(table: pd.DataFrame | None, column: str, top: int = 12) -> str:
    if table is None or column not in getattr(table, "columns", ()):
        return f"{column}: not available"
    vc = table[column].fillna("(missing)").astype(str).value_counts().head(top)
    return f"{column}: " + " · ".join(f"{k} {v:,}" for k, v in vc.items())


def inventory(paths, tables=()) -> list[str]:
    """Everything round two needs to know about the Drive: the folder tree and each named table."""
    lines = ["FOLDERS"]
    lines += describe_dir(paths.work, depth=3)
    lines.append("TABLES")
    for stem in tables:
        lines.append(describe_table(Path(stem)))
    return lines
