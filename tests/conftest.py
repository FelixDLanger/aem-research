import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config" / "preregistration.yaml"


@pytest.fixture
def final_config(tmp_path):
    """Copy of the shipped config made ready to lock (study.status final, no open
    review items), in a temp folder."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["study"]["status"] = "final"
    data["review_before_lock"] = []
    path = tmp_path / "preregistration.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def bseries(ticker, start, values):
    """Business-day price series for one ticker."""
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"ticker": ticker, "date": dates, "price": np.asarray(values, dtype=float)})
