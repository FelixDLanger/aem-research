"""Ticker normalisation so SimFin, Stooq and other sources can be matched."""
from __future__ import annotations

import pandas as pd


def canonical_ticker(value) -> str:
    """'aapl.us' -> 'AAPL'; 'BRK-B' / 'BRK/B' / 'BRK_B' -> 'BRK.B'."""
    s = str(value).strip().upper()
    if s.endswith(".US"):
        s = s[:-3]
    for ch in ("-", "/", "_"):
        s = s.replace(ch, ".")
    return s


def canonical_series(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\.US$", "", regex=True)
    return s.str.replace(r"[-/_]", ".", regex=True)
