from __future__ import annotations

import csv
import math
import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.base import PredictionContext
from ..adapters.registry import AdapterRegistry
from ..schemas import (
    ArtifactInfo,
    ForecastStep,
    InputSummary,
    PredictionParameters,
    PredictionResult,
    RiskLevel,
    WaterLevelConfig,
    WeatherRow,
)
from .artifacts import write_mask_artifacts
from .evaluation import evaluate_water_levels
from .inputs import LoadedFrames
from .store import FileResultStore


class PredictionService:
    def __init__(
        self, registry: AdapterRegistry, store: FileResultStore, api_prefix: str
    ) -> None:
        self.registry = registry
        self.store = store
        self.api_prefix = api_prefix.rstrip("/")

    def run(
        self,
        *,
        loaded: LoadedFrames,
        model_id: str,
        horizon_steps: int,
        threshold: float,
        source_dates: Sequence[date],
        target_dates: Sequence[date],
        input_metadata: dict[str, Any],
        weather: Sequence[WeatherRow],
        historical_water_levels_m: Sequence[float | None],
        reference_water_levels_m: Sequence[float | None],
        evaluation_kind: str | None,
        evaluation_truth_provenance: str | None,
        pixel_area_m2: float,
        water_level_config: WaterLevelConfig | None,
        caution_pct: float,
        risk_pct: float,
        model_options: dict[str, Any],
    ) -> PredictionResult:
        adapter = self.registry.get(model_id)
        self._validate_request(
            loaded=loaded,
            minimum_frames=adapter.info.min_frames,
            horizon_steps=horizon_steps,
            source_dates=source_dates,
            target_dates=target_dates,
            historical_water_levels_m=historical_water_levels_m,
            reference_water_levels_m=reference_water_levels_m,
            pixel_area_m2=pixel_area_m2,
            caution_pct=caution_pct,
            risk_pct=risk_pct,
        )
        resolved_targets = _resolve_target_dates(
            source_dates, target_dates, horizon_steps
        )
        prediction_id = uuid.uuid4().hex
        created_at = datetime.now(UTC)
        output_directory = self.store.create_directory(prediction_id)
        parameters = PredictionParameters(
            horizon_steps=horizon_steps,
            target_dates=resolved_targets,
            weather=list(weather),
            historical_water_levels_m=list(historical_water_levels_m),
            reference_water_levels_m=list(reference_water_levels_m),
            evaluation_kind=evaluation_kind,
            evaluation_truth_provenance=evaluation_truth_provenance,
            model_options=model_options,
            water_level_config=water_level_config,
            caution_pct=caution_pct,
            risk_pct=risk_pct,
        )
        input_summary = InputSummary(
            frame_count=int(loaded.frames.shape[0]),
            height=int(loaded.frames.shape[1]),
            width=int(loaded.frames.shape[2]),
            source_files=loaded.source_files,
            source_dates=list(source_dates),
            threshold=threshold,
            pixel_area_m2=pixel_area_m2,
            georeferenced=loaded.geo_reference is not None,
            source_metadata=input_metadata,
        )
        base_url = f"{self.api_prefix}/predictions/{prediction_id}"

        try:
            context = PredictionContext(
                horizon_steps=horizon_steps,
                threshold=threshold,
                source_dates=tuple(source_dates),
                target_dates=tuple(resolved_targets),
                historical_water_levels_m=tuple(historical_water_levels_m),
                pixel_area_m2=pixel_area_m2,
                options=model_options,
            )
            output = adapter.predict(
                loaded.frames.astype(np.float32, copy=False),
                [row.model_dump(mode="json", exclude_none=True) for row in weather],
                context,
            )
            masks = _validate_output_masks(
                output.masks, horizon_steps, loaded.frames.shape[1:], threshold
            )
            levels, water_level_source = _resolve_water_levels(
                output.water_levels_m,
                masks,
                loaded.frames[-1],
                pixel_area_m2,
                water_level_config,
                horizon_steps,
            )
            steps, artifacts = self._write_steps(
                output_directory=output_directory,
                base_url=base_url,
                masks=masks,
                target_dates=resolved_targets,
                input_mask=loaded.frames[-1],
                pixel_area_m2=pixel_area_m2,
                levels=levels,
                water_level_source=water_level_source,
                reference_levels=reference_water_levels_m,
                historical_levels=historical_water_levels_m,
                water_level_config=water_level_config,
                caution_pct=caution_pct,
                risk_pct=risk_pct,
                loaded=loaded,
            )
            evaluation = evaluate_water_levels(
                levels,
                reference_water_levels_m,
                resolved_targets,
                kind=evaluation_kind or "user_supplied",
                truth_provenance=(
                    evaluation_truth_provenance
                    or "Reference levels supplied with the prediction request."
                ),
            )
            warnings = [*loaded.warnings, *output.warnings]
            if int(loaded.frames[-1].sum()) == 0:
                warnings.append(
                    "The latest input mask has zero water pixels; percentage change uses 100% for growth."
                )
            result = PredictionResult(
                id=prediction_id,
                status="completed",
                model_id=model_id,
                model_version=adapter.info.version,
                created_at=created_at,
                completed_at=datetime.now(UTC),
                input=input_summary,
                parameters=parameters,
                steps=steps,
                artifacts=artifacts,
                result_url=base_url,
                artifacts_url=f"{base_url}/files",
                bundle_url=f"{base_url}/bundle",
                warnings=warnings,
                adapter_metadata=_json_safe(output.metadata),
                evaluation=evaluation,
            )
        except Exception as exc:  # noqa: BLE001 - adapter boundary is persisted as a failed result
            result = PredictionResult(
                id=prediction_id,
                status="failed",
                model_id=model_id,
                model_version=adapter.info.version,
                created_at=created_at,
                completed_at=datetime.now(UTC),
                input=input_summary,
                parameters=parameters,
                result_url=base_url,
                artifacts_url=f"{base_url}/files",
                bundle_url=f"{base_url}/bundle",
                warnings=loaded.warnings,
                error=f"{type(exc).__name__}: {exc}",
            )
        self.store.save(result)
        return result

    def _write_steps(
        self,
        *,
        output_directory: Path,
        base_url: str,
        masks: np.ndarray,
        target_dates: Sequence[date],
        input_mask: np.ndarray,
        pixel_area_m2: float,
        levels: list[float | None],
        water_level_source: str,
        reference_levels: Sequence[float | None],
        historical_levels: Sequence[float | None],
        water_level_config: WaterLevelConfig | None,
        caution_pct: float,
        risk_pct: float,
        loaded: LoadedFrames,
    ) -> tuple[list[ForecastStep], list[ArtifactInfo]]:
        steps: list[ForecastStep] = []
        artifacts: list[ArtifactInfo] = []
        baseline_pixels = int(input_mask.sum())
        baseline_level = next(
            (v for v in reversed(historical_levels) if v is not None), None
        )
        if baseline_level is None and water_level_config is not None:
            baseline_level = water_level_config.baseline_level_m

        for index, mask in enumerate(masks):
            horizon = index + 1
            npy_path, png_path, tif_path = write_mask_artifacts(
                output_directory, mask, horizon, loaded.geo_reference
            )
            urls = {
                path.suffix: f"{base_url}/files/{path.name}"
                for path in (npy_path, png_path, tif_path)
            }
            artifacts.extend(
                [
                    _artifact(
                        npy_path, "mask", "application/x-npy", urls[".npy"], horizon
                    ),
                    _artifact(png_path, "preview", "image/png", urls[".png"], horizon),
                    _artifact(tif_path, "raster", "image/tiff", urls[".tif"], horizon),
                ]
            )
            area_pixels = int(mask.sum())
            area_m2 = float(area_pixels * pixel_area_m2)
            change_pct = _percentage_change(baseline_pixels, area_pixels)
            level = levels[index]
            reference_level = (
                reference_levels[index] if index < len(reference_levels) else None
            )
            steps.append(
                ForecastStep(
                    horizon=horizon,
                    target_date=target_dates[index]
                    if index < len(target_dates)
                    else None,
                    water_area_pixels=area_pixels,
                    water_area_m2=area_m2,
                    water_area_km2=area_m2 / 1_000_000.0,
                    area_change_pct=change_pct,
                    water_level_m=level,
                    water_level_change_m=(
                        level - baseline_level
                        if level is not None and baseline_level is not None
                        else None
                    ),
                    water_level_source=water_level_source,
                    reference_water_level_m=reference_level,
                    water_level_error_m=(
                        level - reference_level
                        if level is not None and reference_level is not None
                        else None
                    ),
                    risk=_risk(change_pct, caution_pct, risk_pct),
                    mask_npy_url=urls[".npy"],
                    preview_png_url=urls[".png"],
                    mask_tif_url=urls[".tif"],
                )
            )
        csv_path = output_directory / "results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = [
                "horizon",
                "target_date",
                "water_area_pixels",
                "water_area_m2",
                "water_area_km2",
                "area_change_pct",
                "water_level_m",
                "water_level_change_m",
                "water_level_source",
                "reference_water_level_m",
                "water_level_error_m",
                "risk",
                "rule_based",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for step in steps:
                values = step.model_dump(mode="json", include=set(fieldnames))
                writer.writerow(values)
        artifacts.append(
            _artifact(
                csv_path,
                "metadata",
                "text/csv; charset=utf-8",
                f"{base_url}/files/{csv_path.name}",
                None,
            )
        )
        return steps, artifacts

    @staticmethod
    def _validate_request(
        *,
        loaded: LoadedFrames,
        minimum_frames: int,
        horizon_steps: int,
        source_dates: Sequence[date],
        target_dates: Sequence[date],
        historical_water_levels_m: Sequence[float | None],
        reference_water_levels_m: Sequence[float | None],
        pixel_area_m2: float,
        caution_pct: float,
        risk_pct: float,
    ) -> None:
        frame_count = loaded.frames.shape[0]
        if frame_count < minimum_frames:
            raise ValueError(
                f"selected model requires at least {minimum_frames} input frame(s)"
            )
        if not 1 <= horizon_steps <= 365:
            raise ValueError("horizon_steps must be between 1 and 365")
        if source_dates and len(source_dates) != frame_count:
            raise ValueError(
                "source_dates length must match the decoded input frame count"
            )
        if source_dates and not _strictly_increasing(source_dates):
            raise ValueError("source_dates must be strictly increasing in file order")
        if target_dates and len(target_dates) != horizon_steps:
            raise ValueError("target_dates length must match horizon_steps")
        if target_dates and not _strictly_increasing(target_dates):
            raise ValueError("target_dates must be strictly increasing")
        if source_dates and target_dates and target_dates[0] <= source_dates[-1]:
            raise ValueError("every target date must be after the latest source date")
        if historical_water_levels_m and len(historical_water_levels_m) != frame_count:
            raise ValueError(
                "historical_water_levels_m length must match the decoded input frame count"
            )
        for value in historical_water_levels_m:
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    "historical_water_levels_m must contain finite numbers or null"
                )
        if reference_water_levels_m and len(reference_water_levels_m) != horizon_steps:
            raise ValueError(
                "reference_water_levels_m length must match horizon_steps"
            )
        for value in reference_water_levels_m:
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    "reference_water_levels_m must contain finite numbers or null"
                )
        if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
            raise ValueError("pixel_area_m2 must be a positive finite number")
        if not math.isfinite(caution_pct) or caution_pct < 0:
            raise ValueError("caution_pct must be a non-negative finite number")
        if not math.isfinite(risk_pct) or risk_pct <= caution_pct:
            raise ValueError("risk_pct must be finite and greater than caution_pct")


def _resolve_target_dates(
    source_dates: Sequence[date], target_dates: Sequence[date], horizon_steps: int
) -> list[date]:
    if target_dates:
        return list(target_dates)
    if source_dates:
        last_date = source_dates[-1]
        return [
            last_date + timedelta(days=index) for index in range(1, horizon_steps + 1)
        ]
    return []


def _strictly_increasing(values: Sequence[date]) -> bool:
    return all(previous < current for previous, current in pairwise(values))


def _validate_output_masks(
    masks: Any, horizon_steps: int, expected_shape: tuple[int, int], threshold: float
) -> np.ndarray:
    values = np.asarray(masks)
    if values.shape != (horizon_steps, *expected_shape):
        raise ValueError(
            "adapter returned mask shape "
            f"{values.shape}, expected {(horizon_steps, *expected_shape)}"
        )
    if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
        raise ValueError("adapter masks must be numeric or boolean")
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("adapter masks contain NaN or infinite values")
    return (values >= threshold).astype(np.uint8)


def _resolve_water_levels(
    model_levels: Sequence[float | None] | None,
    masks: np.ndarray,
    input_mask: np.ndarray,
    pixel_area_m2: float,
    config: WaterLevelConfig | None,
    horizon_steps: int,
) -> tuple[list[float | None], str]:
    if model_levels is not None:
        if len(model_levels) != horizon_steps:
            raise ValueError("adapter water_levels_m length must match horizon_steps")
        levels: list[float | None] = []
        for level in model_levels:
            if level is None:
                levels.append(None)
            else:
                parsed = float(level)
                if not math.isfinite(parsed):
                    raise ValueError("adapter water levels must be finite or null")
                levels.append(parsed)
        return levels, "adapter_output"
    if config is None:
        return [None] * horizon_steps, "unavailable"

    observed_area = float(input_mask.sum() * pixel_area_m2)
    baseline_area = config.baseline_area_m2 or observed_area
    if baseline_area <= 0:
        raise ValueError("water-level calibration requires a non-zero baseline area")
    return (
        [
            config.baseline_level_m
            + config.meters_per_area_ratio
            * ((float(mask.sum()) * pixel_area_m2 / baseline_area) - 1.0)
            for mask in masks
        ],
        "area_level_calibration",
    )


def _percentage_change(baseline: int, current: int) -> float:
    if baseline == 0:
        return 0.0 if current == 0 else 100.0
    return (current - baseline) / baseline * 100.0


def _risk(change_pct: float, caution_pct: float, risk_pct: float) -> RiskLevel:
    if change_pct >= risk_pct:
        return RiskLevel.flood_risk
    if change_pct <= -risk_pct:
        return RiskLevel.drought_risk
    if abs(change_pct) >= caution_pct:
        return RiskLevel.caution
    return RiskLevel.normal


def _artifact(
    path: Path, kind: str, media_type: str, url: str, horizon: int | None
) -> ArtifactInfo:
    return ArtifactInfo(
        name=path.name,
        kind=kind,  # type: ignore[arg-type]
        media_type=media_type,
        size_bytes=path.stat().st_size,
        url=url,
        horizon=horizon,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
