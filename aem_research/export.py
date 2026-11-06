"""Excel export of every analysis table (the raw data stays in parquet: tens of millions of
rows, beyond Excel's 1,048,576 rows per sheet).

Values are made Excel-safe: nullable booleans and integers become plain values or
blanks, categories become text, time zones are dropped, and a sheet longer than Excel's
limit is cut with a note in the Summary sheet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXCEL_MAX_ROWS = 1_048_575          # one header row


def _safe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(str).where(s.notna(), None)
        elif pd.api.types.is_datetime64_any_dtype(s):
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_localize(None)
            s = s.astype("datetime64[ns]")
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s) or str(s.dtype) in ("boolean", "Int64"):
            s = s.astype(object).where(s.notna(), None)
        elif s.dtype == object:
            s = s.map(lambda v: None if v is pd.NA or (isinstance(v, float) and np.isnan(v)) else
                      (v if isinstance(v, (str, int, float, bool, pd.Timestamp)) or v is None else str(v)))
        out[str(c)[:250]] = s
    return out


def write_workbook(path: str | Path, sheets: dict[str, pd.DataFrame | dict | list], *, engine: str | None = None) -> dict:
    """Write {sheet name: table, dict or list of dicts} to one .xlsx. Returns {sheet: rows written}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if engine is None:
        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except Exception:
            engine = "openpyxl"
    written, cut = {}, []
    used = set()
    with pd.ExcelWriter(path, engine=engine) as xw:
        for name, obj in sheets.items():
            if obj is None:
                continue
            if isinstance(obj, dict):
                df = pd.DataFrame({"item": list(obj), "value": [str(v) if isinstance(v, (dict, list)) else v
                                                                for v in obj.values()]})
            elif isinstance(obj, list):
                df = pd.DataFrame(obj)
            else:
                df = obj
            if df is None or not isinstance(df, pd.DataFrame):
                continue
            sheet = "".join(ch for ch in str(name) if ch not in "[]:*?/\\")[:31] or "Sheet"
            while sheet in used:
                sheet = sheet[:28] + f"_{len(used)}"
            used.add(sheet)
            if len(df) > EXCEL_MAX_ROWS:
                cut.append(f"{sheet}: {len(df):,} rows cut to {EXCEL_MAX_ROWS:,}")
                df = df.iloc[:EXCEL_MAX_ROWS]
            _safe(df.reset_index(drop=True)).to_excel(xw, sheet_name=sheet, index=False)
            written[sheet] = int(len(df))
        if cut:
            pd.DataFrame({"note": cut}).to_excel(xw, sheet_name="Notes", index=False)
    return written
