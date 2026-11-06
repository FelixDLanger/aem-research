"""Hard guards that fail loudly: look-ahead and survivorship."""
from __future__ import annotations

import pandas as pd


class LookAheadError(AssertionError):
    pass


class SurvivorshipError(AssertionError):
    pass


def assert_point_in_time(features: pd.DataFrame, *, start_col: str = "start_date",
                         available_col: str = "available_date") -> None:
    """Every feature must have been public on or before the start date it is used at."""
    missing = {start_col, available_col} - set(features.columns)
    if missing:
        raise LookAheadError(f"feature table lacks {sorted(missing)} – availability cannot be checked")
    avail = pd.to_datetime(features[available_col])
    start = pd.to_datetime(features[start_col])
    if avail.isna().any():
        raise LookAheadError(f"{int(avail.isna().sum())} feature rows have no availability date")
    bad = features[avail > start]
    if len(bad):
        sample = bad.head(3).to_dict("records")
        raise LookAheadError(f"{len(bad)} feature rows use information published after the start date, e.g. {sample}")


def survivorship_report(spans: pd.DataFrame, start_dates, as_of, *, gap_days: int = 30,
                        min_years_before_as_of: int = 5) -> pd.DataFrame:
    """For each start date: names live then, and the share whose series later stopped.

    `spans` has one row per ticker with first_date and last_date. A survivorship-free
    source shows a material share of names that later stop trading (mergers,
    failures); a biased one shows almost none.
    """
    as_of = pd.Timestamp(as_of)
    cutoff = as_of - pd.Timedelta(days=gap_days)
    rows = []
    for s in pd.to_datetime(list(start_dates)):
        live = spans[(spans["first_date"] <= s) & (spans["last_date"] >= s)]
        stopped = int((live["last_date"] < cutoff).sum())
        eligible = s <= as_of - pd.DateOffset(years=min_years_before_as_of)
        rows.append({"start_date": s.date(), "live": len(live), "later_stopped": stopped,
                     "share_stopped": stopped / len(live) if len(live) else float("nan"),
                     "checked": bool(eligible)})
    return pd.DataFrame(rows)


def assert_survivorship_free(spans: pd.DataFrame, start_dates, as_of, *, min_share: float = 0.05,
                             gap_days: int = 30, min_years_before_as_of: int = 5) -> pd.DataFrame:
    rep = survivorship_report(spans, start_dates, as_of, gap_days=gap_days,
                              min_years_before_as_of=min_years_before_as_of)
    checked = rep[rep["checked"] & (rep["live"] > 0)]
    failing = checked[checked["share_stopped"] < min_share]
    if len(failing):
        raise SurvivorshipError(
            f"Only {failing['share_stopped'].min():.1%} of names live at {failing['start_date'].iloc[0]} "
            f"later stop trading (minimum {min_share:.0%}) – this price source looks survivor-only.")
    return rep
