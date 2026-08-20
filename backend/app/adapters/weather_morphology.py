from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from ..schemas import ModelInfo
from .base import PredictionAdapter, PredictionContext, PredictionOutput


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbours = [
            padded[row : row + result.shape[0], col : col + result.shape[1]]
            for row in range(3)
            for col in range(3)
        ]
        result = np.logical_or.reduce(neighbours)
    return result


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbours = [
            padded[row : row + result.shape[0], col : col + result.shape[1]]
            for row in range(3)
            for col in range(3)
        ]
        result = np.logical_and.reduce(neighbours)
    return result


def _cap_changed_pixels(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_change_pct: float | None,
    seed: int,
) -> tuple[np.ndarray, bool]:
    """Limit one morphology step for coarse demo grids.

    A full one-pixel dilation can nearly double the area of thin rivers after a
    large source scene has been downsampled.  ``max_change_pct`` lets a caller
    cap that discrete jump while preserving the original uncapped behaviour
    when the option is absent.  Candidate boundary pixels are selected by a
    stable coordinate hash, so repeated requests remain reproducible.
    """

    if max_change_pct is None:
        return after, False
    before_bool = before.astype(bool, copy=False)
    after_bool = after.astype(bool, copy=False)
    changed = np.argwhere(before_bool != after_bool)
    if changed.size == 0:
        return after_bool.copy(), False
    reference_pixels = max(1, int(np.count_nonzero(before_bool)))
    maximum = max(1, math.floor(reference_pixels * max_change_pct / 100.0))
    if len(changed) <= maximum:
        return after_bool.copy(), False

    rows = changed[:, 0].astype(np.uint64)
    columns = changed[:, 1].astype(np.uint64)
    priorities = (
        rows * np.uint64(73_856_093)
        ^ columns * np.uint64(19_349_663)
        ^ np.uint64(seed * 83_492_791)
    )
    selected = changed[np.argpartition(priorities, maximum - 1)[:maximum]]
    limited = before_bool.copy()
    limited[selected[:, 0], selected[:, 1]] = after_bool[
        selected[:, 0], selected[:, 1]
    ]
    return limited, True


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _precipitation(row: Mapping[str, Any]) -> float | None:
    for key in (
        "precipitation_mm",
        "rainfall_mm",
        "daily_precipitation_mm",
        "sum_rn",
        "rn_day",
    ):
        value = row.get(key)
        if value is not None and value != "":
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                return max(0.0, parsed)
    return None


class WeatherMorphologyAdapter(PredictionAdapter):
    """Explainable demo baseline that expands/contracts a mask from rainfall.

    It is intentionally labelled as a baseline, not as a learned hydrological
    model. It remains useful for end-to-end integration before a trained adapter
    is supplied.
    """

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            id="weather-morphology",
            name="Weather morphology baseline",
            version="1.0.0",
            description=(
                "Rule-based mask dilation after rainfall and erosion after a configurable dry streak."
            ),
            min_frames=1,
            supports_weather=True,
            produces_water_level=False,
            options={
                "rain_mm_per_dilation": 20.0,
                "dry_threshold_mm": 0.1,
                "dry_days_per_erosion": 3,
                "max_iterations_per_step": 3,
            },
        )

    def predict(
        self,
        frames: np.ndarray,
        weather: Sequence[Mapping[str, Any]],
        request: PredictionContext,
    ) -> PredictionOutput:
        options = {**self.info.options, **dict(request.options)}
        rain_unit = _positive_float(options, "rain_mm_per_dilation")
        dry_threshold = _non_negative_float(options, "dry_threshold_mm")
        dry_days = _positive_int(options, "dry_days_per_erosion")
        max_iterations = _positive_int(options, "max_iterations_per_step")
        max_area_change_pct = _optional_positive_float(
            options, "max_area_change_pct_per_step"
        )

        weather_by_date = {
            parsed: row
            for row in weather
            if (parsed := _as_date(row.get("date"))) is not None
        }
        has_dated_weather = bool(weather_by_date)
        current = frames[-1] >= request.threshold
        outputs: list[np.ndarray] = []
        operations: list[dict[str, Any]] = []
        warnings: list[str] = []
        dry_streak = 0

        for index in range(request.horizon_steps):
            target = (
                request.target_dates[index]
                if index < len(request.target_dates)
                else None
            )
            if target is not None and has_dated_weather:
                row: Mapping[str, Any] | None = weather_by_date.get(target)
                if row is None:
                    warnings.append(
                        f"No weather row matches target date {target.isoformat()}; mask is unchanged."
                    )
            elif index < len(weather):
                row = weather[index]
            else:
                row = None

            precipitation = _precipitation(row) if row is not None else None
            operation = "unchanged"
            iterations = 0
            uncapped_area_pixels = int(np.count_nonzero(current))
            capped = False
            if precipitation is None:
                dry_streak = 0
            elif precipitation >= rain_unit:
                iterations = min(
                    max_iterations, max(1, int(precipitation // rain_unit))
                )
                expanded = _dilate(current, iterations)
                uncapped_area_pixels = int(np.count_nonzero(expanded))
                current, capped = _cap_changed_pixels(
                    current,
                    expanded,
                    max_change_pct=max_area_change_pct,
                    seed=index + 1,
                )
                dry_streak = 0
                operation = "dilate"
            elif precipitation <= dry_threshold:
                dry_streak += 1
                if dry_streak % dry_days == 0:
                    iterations = 1
                    contracted = _erode(current, iterations)
                    uncapped_area_pixels = int(np.count_nonzero(contracted))
                    current, capped = _cap_changed_pixels(
                        current,
                        contracted,
                        max_change_pct=max_area_change_pct,
                        seed=index + 1,
                    )
                    operation = "erode"
            else:
                dry_streak = 0

            outputs.append(current.astype(np.float32))
            operations.append(
                {
                    "horizon": index + 1,
                    "target_date": target.isoformat() if target else None,
                    "precipitation_mm": precipitation,
                    "operation": operation,
                    "iterations": iterations,
                    "max_area_change_pct_per_step": max_area_change_pct,
                    "uncapped_area_pixels": uncapped_area_pixels,
                    "area_change_capped": capped,
                }
            )

        if not weather:
            warnings.append("No weather rows were supplied; masks remain unchanged.")
        return PredictionOutput(
            masks=np.stack(outputs, axis=0),
            metadata={"strategy": "weather_morphology", "operations": operations},
            warnings=warnings,
        )


def _positive_float(options: Mapping[str, Any], key: str) -> float:
    value = float(options[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"model option {key!r} must be a positive number")
    return value


def _non_negative_float(options: Mapping[str, Any], key: str) -> float:
    value = float(options[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"model option {key!r} must be a non-negative number")
    return value


def _positive_int(options: Mapping[str, Any], key: str) -> int:
    value = int(options[key])
    if value <= 0:
        raise ValueError(f"model option {key!r} must be a positive integer")
    return value


def _optional_positive_float(
    options: Mapping[str, Any], key: str
) -> float | None:
    raw = options.get(key)
    if raw is None or raw == "":
        return None
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"model option {key!r} must be a positive number when supplied")
    return value
