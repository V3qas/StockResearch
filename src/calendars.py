from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import REFERENCE_CALENDAR


@lru_cache(maxsize=32)
def reference_sessions(
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_days: int = 252,
    calendar_name: str = REFERENCE_CALENDAR,
) -> pd.DatetimeIndex:
    """Independent exchange sessions, including the planned future horizon."""
    import exchange_calendars as xcals

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError("Provide a valid reference-calendar date range.")
    if forecast_days < 1:
        raise ValueError("forecast_days must be positive.")
    calendar = xcals.get_calendar(
        calendar_name,
        start=start - pd.Timedelta(days=14),
        end=end + pd.DateOffset(years=forecast_days // 200 + 2),
    )
    sessions = calendar.sessions.tz_localize(None).as_unit("ns")
    if int((sessions > end).sum()) < forecast_days:
        raise ValueError("Reference calendar does not cover the requested horizon.")
    return sessions


def benchmark_coverage(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Audit expected closes; missing sessions never disappear from the calendar."""
    expected = sessions[(sessions >= start) & (sessions <= end)]
    close = pd.to_numeric(prices.Close.reindex(expected), errors="coerce")
    audit = pd.DataFrame({"Close": close, "available": np.isfinite(close) & close.gt(0)})
    audit.index.name = "session"
    return audit
