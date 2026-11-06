"""Small I/O helpers: format-tolerant reads, cache writes, display."""
from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd


def parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


def save_table(df: pd.DataFrame, base: str | Path) -> Path:
    """Write parquet when an engine is installed (Colab has pyarrow), else pickle."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    if parquet_available():
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    else:
        path = base.with_suffix(".pkl")
        df.to_pickle(path)
    return path


def load_table(base: str | Path) -> pd.DataFrame | None:
    base = Path(base)
    pq, pk = base.with_suffix(".parquet"), base.with_suffix(".pkl")
    if pq.exists() and parquet_available():
        return pd.read_parquet(pq)
    if pk.exists():
        return pd.read_pickle(pk)
    return None


def table_exists(base: str | Path) -> bool:
    base = Path(base)
    return base.with_suffix(".parquet").exists() or base.with_suffix(".pkl").exists()


def _first_line(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            member = next(m for m in zf.namelist() if not m.endswith("/"))
            with zf.open(member) as fh:
                return fh.readline().decode("utf-8", "replace")
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.readline()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.readline()


def read_any(path: str | Path) -> pd.DataFrame:
    """Read parquet, pickle, or a CSV (plain, .zip or .gz) with ';' or ',' separators."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    if name.endswith((".csv", ".txt", ".zip", ".gz", ".csv.gz")):
        header = _first_line(path)
        sep = ";" if header.count(";") > header.count(",") else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def write_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def show(obj, max_rows: int = 60) -> None:
    """Rich display in Jupyter/Colab, plain text elsewhere."""
    try:
        from IPython import get_ipython  # type: ignore
        from IPython.display import display  # type: ignore
        if get_ipython() is not None:
            display(obj)
            return
    except Exception:
        pass
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj.to_string(max_rows=max_rows))
    else:
        print(obj)
