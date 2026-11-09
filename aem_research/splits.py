"""Stock splits from public filings, for the SEC price panel (D10).

13F and fails-to-deliver prices are the prices of the day, not adjusted for later
splits, so a 1-for-10 reverse split would look like a tenfold gain. Two pieces of
public evidence find splits without a vendor:

  * holders – a split multiplies every holder's share count by the same ratio. Between
    two consecutive quarter-ends, the largest 13F holders present in both are matched;
    when their share counts changed by a common split ratio (the median within 3% of a
    standard ratio, at least half of them within 5% of it, at least three of them, and
    a ratio of at least 1.2 or at most 1/1.2), the quarter holds a split. The standard
    ratios are those companies use (6:5 to 100:1 forward, 1:2 to 1:1000 reverse).
    Ordinary trading moves holders by different amounts, so it cannot pass this test.
  * a change of CUSIP – reverse splits usually bring a new CUSIP. Where holders cannot
    be matched (before 13F data begin in 2013), the last daily price under the old
    CUSIP and the first under the new one, at most 30 days apart, give the ratio
    (within 6% of a standard ratio).
  * the filings – the cover-page share count (dei:EntityCommonStockSharesOutstanding)
    changes by the split ratio. aem_research.secprices uses it to confirm or refute a
    ratio read from prices at a CUSIP change, and to find splits where holders are too
    few, when the price moved the other way by about as much.

A ratio r means one old share became r new shares (2 for a 2-for-1 split, 0.1 for a
1-for-10 reverse split): prices before the split are divided by r to match later ones.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

MEDIAN_TOL = 0.03          # holders' median ratio within 3% of a standard ratio
AGREE_TOL = 0.05           # a holder agrees when its own ratio is within 5% of it
MIN_MATCHED = 3            # holders present in both quarters
MIN_AGREE = 0.5            # share of matched holders that agree
MIN_MOVE = 1.2             # smaller ratios (stock dividends) are ignored
SAME_TOL = 0.15            # at a CUSIP change, holders' median within 15% of 1: the same security
PRICE_TOL = 0.06           # CUSIP change before 13F data: price ratio within 6% of a standard ratio
PRICE_SAME_TOL = 0.25      # ... or within 25% of 1: the same security
SWITCH_DAYS = 30


def _standard_ratios() -> np.ndarray:
    """Ratios companies actually use: forward 6:5 to 100:1, reverse 1:2 to 1:1000."""
    fwd = {6 / 5, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 2.0, 5 / 2, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0}
    rev = {1.0 / x for x in fwd} | {1.0 / k for k in range(2, 51)} | \
          {1.0 / k for k in (60, 70, 75, 80, 90, 100, 120, 125, 150, 200, 250, 300, 400, 500, 750, 1000)}
    return np.array(sorted(fwd | rev), dtype="float64")


STANDARD = _standard_ratios()
_LOG_STD = np.log(STANDARD)
# The ratios most splits use. Below 1:10 the standard set is so dense (every 1:k) that,
# with a few percent of tolerance, almost any price ratio lands near one of them; price
# evidence alone (a vendor's implied ratio, a price switch at a CUSIP change) is
# therefore read against this sparser set, where a match means something.
COMMON = np.array(sorted({5 / 4, 4 / 3, 3 / 2, 2.0, 5 / 2, 3.0, 4.0, 5.0, 10.0}
                         | {1.0 / k for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)}),
                  dtype="float64")
_LOG_COMMON = np.log(COMMON)


def _snap(x, ratios: np.ndarray, log_ratios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype="float64")
    lx = np.log(np.where(x > 0, x, np.nan))
    j = np.clip(np.searchsorted(log_ratios, lx), 1, len(ratios) - 1)
    lo, hi = log_ratios[j - 1], log_ratios[j]
    pick = np.where(np.abs(lx - lo) <= np.abs(lx - hi), j - 1, j)
    r = ratios[pick]
    dist = np.abs(lx - log_ratios[pick])
    r = np.where(np.isfinite(lx), r, np.nan)
    return r, np.where(np.isfinite(lx), dist, np.nan)


def snap(x) -> tuple[np.ndarray, np.ndarray]:
    """(nearest standard ratio, |log distance|) for each value (NaN where x <= 0)."""
    return _snap(x, STANDARD, _LOG_STD)


def snap_common(x) -> tuple[np.ndarray, np.ndarray]:
    """(nearest common ratio, |log distance|) for each value (NaN where x <= 0)."""
    return _snap(x, COMMON, _LOG_COMMON)


def holder_ratios(holders: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per consecutive pair of quarter-ends of a company: matched holders, their median share
    ratio, the standard ratio nearest to it and the share of holders that agree with it.

    holders: (cik, period, filer_cik, shares) – the holders of the company's main CUSIP
    in each quarter. pairs: (cik, period, prev_period)."""
    cols = ["cik", "period", "matched", "median_ratio", "ratio", "agree"]
    if holders is None or holders.empty or pairs is None or pairs.empty:
        return pd.DataFrame(columns=cols)
    h = holders[["cik", "period", "filer_cik", "shares"]]
    cur = h.merge(pairs[["cik", "period", "prev_period"]], on=["cik", "period"], how="inner")
    prev = h.rename(columns={"period": "prev_period", "shares": "shares_prev"})
    m = cur.merge(prev, on=["cik", "prev_period", "filer_cik"], how="inner")
    m = m[(m["shares"] > 0) & (m["shares_prev"] > 0)]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m = m.assign(r=m["shares"] / m["shares_prev"])
    g = m.groupby(["cik", "period"], as_index=False).agg(matched=("r", "size"), median_ratio=("r", "median"))
    std, _ = snap(g["median_ratio"].to_numpy())
    g["ratio"] = std
    m = m.merge(g[["cik", "period", "ratio"]], on=["cik", "period"], how="left")
    m["agrees"] = np.abs(np.log(m["r"] / m["ratio"])) <= np.log(1 + AGREE_TOL)
    agree = m.groupby(["cik", "period"], as_index=False)["agrees"].mean().rename(columns={"agrees": "agree"})
    return g.merge(agree, on=["cik", "period"], how="left")[cols]


def holder_split(row) -> float | None:
    """The split ratio a holder_ratios row shows, or None."""
    if row is None or not np.isfinite(row.get("median_ratio", np.nan)) or row.get("matched", 0) < MIN_MATCHED:
        return None
    r = float(row["ratio"])
    if abs(np.log(r)) < np.log(MIN_MOVE):
        return None
    if abs(np.log(float(row["median_ratio"]) / r)) > np.log(1 + MEDIAN_TOL) or float(row["agree"]) < MIN_AGREE:
        return None
    return r


def holders_same(row) -> bool:
    """At a CUSIP change: do the matched holders show the same security (ratio near 1)?"""
    if row is None or row.get("matched", 0) < MIN_MATCHED or not np.isfinite(row.get("median_ratio", np.nan)):
        return False
    return abs(np.log(float(row["median_ratio"]))) <= np.log(1 + SAME_TOL)


def price_switch(old_last: float, new_first: float) -> tuple[str, float | None]:
    """At a CUSIP change without holder evidence: ('split', r), ('same', None) or ('break', None),
    from the last daily price under the old CUSIP and the first under the new one."""
    if not (np.isfinite(old_last) and np.isfinite(new_first)) or old_last <= 0 or new_first <= 0:
        return "break", None
    x = old_last / new_first                      # = r when the price change is the split alone
    if abs(np.log(x)) <= np.log(1 + PRICE_SAME_TOL):
        return "same", None
    r, dist = snap([x])
    if abs(np.log(r[0])) >= np.log(MIN_MOVE) and dist[0] <= np.log(1 + PRICE_TOL):
        return "split", float(r[0])
    return "break", None


__all__ = ["STANDARD", "COMMON", "snap", "snap_common", "holder_ratios", "holder_split", "holders_same",
           "price_switch", "MIN_MATCHED"]
