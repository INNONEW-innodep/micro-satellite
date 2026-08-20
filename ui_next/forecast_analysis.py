"""Pure helpers for the daily water-level forecast presentation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

STANDARD_WINDOWS = (7, 14, 30)


def forecast_window_options(step_count: int) -> tuple[int, ...]:
    """Return useful D+ windows without offering unavailable horizons."""

    if step_count <= 0:
        return ()
    options = [days for days in STANDARD_WINDOWS if days <= step_count]
    if step_count not in options:
        options.append(step_count)
    return tuple(sorted(set(options)))


def horizon_row(
    rows: Sequence[Mapping[str, Any]], horizon: int
) -> Mapping[str, Any] | None:
    """Find an exact horizon row; list position is only a compatibility fallback."""

    for index, row in enumerate(rows, start=1):
        raw = row.get("frame", row.get("horizon", index))
        try:
            if int(raw) == horizon:
                return row
        except (TypeError, ValueError):
            continue
    return None


def cumulative_rmse(pairs: Sequence[Mapping[str, Any]]) -> list[float]:
    """Calculate the RMSE visible at each successive forecast horizon."""

    squared_errors: list[float] = []
    values: list[float] = []
    for pair in pairs:
        try:
            residual = float(pair["residual_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(residual):
            continue
        squared_errors.append(residual**2)
        values.append(math.sqrt(sum(squared_errors) / len(squared_errors)))
    return values


def precipitation_by_date(
    weather_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Normalize common precipitation/date aliases for forecast charts."""

    result: dict[str, float] = {}
    for row in weather_rows:
        raw_date = row.get("date") or row.get("timestamp")
        if raw_date is None:
            continue
        date_key = str(raw_date)[:10]
        raw_value = next(
            (
                row.get(key)
                for key in (
                    "precipitation_mm",
                    "rainfall_mm",
                    "total_precipitation_mm",
                )
                if row.get(key) is not None
            ),
            0.0,
        )
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[date_key] = max(0.0, value)
    return result


__all__ = [
    "STANDARD_WINDOWS",
    "cumulative_rmse",
    "forecast_window_options",
    "horizon_row",
    "precipitation_by_date",
]
