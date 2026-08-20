from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from ..schemas import ModelInfo
from .base import PredictionAdapter, PredictionContext, PredictionOutput


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool, copy=False), 1, mode="constant")
    neighbours = [
        padded[row : row + mask.shape[0], column : column + mask.shape[1]]
        for row in range(3)
        for column in range(3)
    ]
    return np.logical_or.reduce(neighbours)


def _erode_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(
        mask.astype(bool, copy=False),
        1,
        mode="constant",
        constant_values=False,
    )
    neighbours = [
        padded[row : row + mask.shape[0], column : column + mask.shape[1]]
        for row in range(3)
        for column in range(3)
    ]
    return np.logical_and.reduce(neighbours)


def _stable_subset(coordinates: np.ndarray, count: int) -> np.ndarray:
    """Choose an exact, deterministic subset without depending on file order."""

    if count >= len(coordinates):
        return coordinates
    rows = coordinates[:, 0].astype(np.uint64)
    columns = coordinates[:, 1].astype(np.uint64)
    priorities = (
        rows * np.uint64(73_856_093)
        ^ columns * np.uint64(19_349_663)
        ^ (rows + columns) * np.uint64(83_492_791)
    )
    selected = np.argpartition(priorities, count - 1)[:count]
    return coordinates[selected]


def _mask_with_area(mask: np.ndarray, target_pixels: int) -> np.ndarray:
    """Expand or contract from the latest boundary to an exact pixel count."""

    current = mask.astype(bool, copy=True)
    target = min(max(0, int(target_pixels)), current.size)
    current_pixels = int(np.count_nonzero(current))
    if target == current_pixels:
        return current

    if target > current_pixels:
        while current_pixels < target:
            ring = _dilate_once(current) & ~current
            candidates = np.argwhere(ring)
            if not len(candidates):
                # An all-dry seed has no meaningful shoreline from which to grow.
                break
            selected = _stable_subset(candidates, target - current_pixels)
            current[selected[:, 0], selected[:, 1]] = True
            current_pixels += len(selected)
        return current

    while current_pixels > target:
        boundary = current & ~_erode_once(current)
        candidates = np.argwhere(boundary)
        if not len(candidates):
            break
        selected = _stable_subset(candidates, current_pixels - target)
        current[selected[:, 0], selected[:, 1]] = False
        current_pixels -= len(selected)
    return current


def _positive_float(options: Mapping[str, Any], key: str) -> float:
    value = float(options[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"model option {key!r} must be a positive number")
    return value


def _bounded_float(
    options: Mapping[str, Any], key: str, *, minimum: float, maximum: float
) -> float:
    value = float(options[key])
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            f"model option {key!r} must be between {minimum} and {maximum}"
        )
    return value


def _precipitation(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    for key in (
        "precipitation_mm",
        "rainfall_mm",
        "daily_precipitation_mm",
        "sum_rn",
        "rn_day",
    ):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return max(0.0, value)
    return None


def _elapsed_days(values: Sequence[date], frame_count: int) -> np.ndarray:
    if len(values) == frame_count:
        first = values[0]
        return np.asarray([(value - first).days for value in values], dtype=np.float64)
    return np.arange(frame_count, dtype=np.float64)


class IrregularAreaTrendAdapter(PredictionAdapter):
    """Transparent multi-frame baseline for sparse, irregular mask sequences.

    A log-linear trend is fitted to every observed mask area against its real
    observation date.  The predicted area is then rendered from the latest
    shoreline by deterministic boundary expansion or contraction.  This is an
    integration and sanity-check baseline, not a trained hydrological model.
    """

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            id="irregular-area-trend",
            name="Irregular-date water-area trend baseline",
            version="1.0.0",
            description=(
                "Fits a log-linear water-area trend using every dated input mask, then "
                "adds a configurable rainfall-memory response and expands or contracts "
                "the latest shoreline to daily target areas."
            ),
            min_frames=2,
            supports_weather=True,
            produces_water_level=False,
            options={
                "max_daily_area_change_pct": 1.0,
                "max_total_area_change_pct": 25.0,
                "rainfall_response_pct_per_20mm": 0.2,
                "rainfall_memory_decay": 0.65,
            },
        )

    def predict(
        self,
        frames: np.ndarray,
        weather: Sequence[Mapping[str, Any]],
        request: PredictionContext,
    ) -> PredictionOutput:
        options = {**self.info.options, **dict(request.options)}
        max_daily_pct = _positive_float(options, "max_daily_area_change_pct")
        max_total_pct = _positive_float(options, "max_total_area_change_pct")
        rainfall_response_pct = _bounded_float(
            options,
            "rainfall_response_pct_per_20mm",
            minimum=0.0,
            maximum=10.0,
        )
        rainfall_decay = _bounded_float(
            options, "rainfall_memory_decay", minimum=0.0, maximum=1.0
        )

        observed = frames >= request.threshold
        areas = np.count_nonzero(observed, axis=(1, 2)).astype(np.float64)
        if np.any(areas <= 0):
            raise ValueError(
                "irregular-area-trend requires at least one water pixel in every input frame"
            )

        elapsed = _elapsed_days(request.source_dates, len(observed))
        centered = elapsed - float(np.mean(elapsed))
        denominator = float(np.dot(centered, centered))
        if denominator <= 0:
            raise ValueError("input observations must have distinct dates or frame positions")
        log_areas = np.log(areas)
        slope = float(np.dot(centered, log_areas - float(np.mean(log_areas))) / denominator)

        latest_area = float(areas[-1])
        maximum_total_delta = latest_area * max_total_pct / 100.0
        masks: list[np.ndarray] = []
        targets: list[dict[str, Any]] = []
        weather_by_date = {
            str(row.get("date")): row for row in weather if row.get("date") is not None
        }
        rainfall_memory_mm = 0.0
        matched_weather_rows = 0
        precipitation_value_count = 0
        for index in range(request.horizon_steps):
            target_date = (
                request.target_dates[index]
                if index < len(request.target_dates)
                else None
            )
            if target_date is not None and len(request.source_dates) == len(observed):
                days_ahead = max(1.0, float((target_date - request.source_dates[-1]).days))
            else:
                days_ahead = float(index + 1)

            # The slope uses every historical observation, while the forecast
            # remains continuous at the latest actually observed area.
            trend_target = float(latest_area * math.exp(slope * days_ahead))
            if target_date is not None:
                weather_row = weather_by_date.get(target_date.isoformat())
            elif index < len(weather):
                weather_row = weather[index]
            else:
                weather_row = None
            if weather_row is not None:
                matched_weather_rows += 1
            precipitation_mm = _precipitation(weather_row)
            if precipitation_mm is not None:
                precipitation_value_count += 1
                rainfall_memory_mm = (
                    rainfall_memory_mm * rainfall_decay + precipitation_mm
                )
            else:
                rainfall_memory_mm *= rainfall_decay
            weather_adjustment_pixels = (
                latest_area
                * rainfall_response_pct
                / 100.0
                * rainfall_memory_mm
                / 20.0
            )
            raw_target = trend_target + weather_adjustment_pixels
            daily_delta = latest_area * max_daily_pct * days_ahead / 100.0
            allowed_delta = min(daily_delta, maximum_total_delta)
            minimum = max(1.0, latest_area - allowed_delta)
            maximum = min(float(observed[-1].size), latest_area + allowed_delta)
            capped_target = min(max(raw_target, minimum), maximum)
            target_pixels = round(capped_target)
            masks.append(_mask_with_area(observed[-1], target_pixels).astype(np.float32))
            targets.append(
                {
                    "horizon": index + 1,
                    "target_date": target_date.isoformat() if target_date else None,
                    "days_ahead": days_ahead,
                    "trend_target_pixels": trend_target,
                    "precipitation_mm": precipitation_mm,
                    "rainfall_memory_mm": rainfall_memory_mm,
                    "weather_adjustment_pixels": weather_adjustment_pixels,
                    "raw_target_pixels": raw_target,
                    "capped_target_pixels": target_pixels,
                    "was_capped": not math.isclose(raw_target, capped_target),
                }
            )

        daily_pct = (math.exp(slope) - 1.0) * 100.0
        warnings = [
            "This is a sparse multi-frame area-trend baseline, not a trained hydrological model.",
            "Predicted shoreline geometry is a deterministic rendering of target area, not learned flow physics.",
        ]
        if len(observed) < 8:
            warnings.append(
                f"Only {len(observed)} observations were available; trend uncertainty is not estimable reliably."
            )
        if weather and matched_weather_rows == 0:
            warnings.append(
                "Weather rows were supplied but none matched the requested target dates."
            )
        elif not weather:
            warnings.append("No weather rows were supplied; rainfall adjustment is zero.")
        return PredictionOutput(
            masks=np.stack(masks, axis=0),
            metadata={
                "strategy": "irregular_log_area_trend",
                "uses_all_input_frames": True,
                "observation_count": len(observed),
                "source_dates": [value.isoformat() for value in request.source_dates],
                "elapsed_days": elapsed.tolist(),
                "observed_area_pixels": areas.astype(int).tolist(),
                "fitted_log_slope_per_day": slope,
                "fitted_area_change_pct_per_day": daily_pct,
                "max_daily_area_change_pct": max_daily_pct,
                "max_total_area_change_pct": max_total_pct,
                "rainfall_response_pct_per_20mm": rainfall_response_pct,
                "rainfall_memory_decay": rainfall_decay,
                "matched_target_weather_rows": matched_weather_rows,
                "target_precipitation_value_count": precipitation_value_count,
                "targets": targets,
            },
            warnings=warnings,
        )


__all__ = ["IrregularAreaTrendAdapter"]
