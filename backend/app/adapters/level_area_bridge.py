"""수위-면적 곡선 연동 면적 예측 어댑터 — ``level-area-bridge``.

면적 시계열을 따로 모델링하지 않고 **수위 시계열 예측에 종속**시킨다(2026-10-15, 사용자 제안).

    면적(t+h) = 마지막 탐지 면적 + 기울기 × (수위 전망(t+h) − 원점 수위)

- 수위 전망은 지역 레지스트리의 학습 모델(``GaugeForecastService.forecast``)에서, 원점 수위는 그 응답의
  ``origin_level_m`` 에서 가져온다.
- 기울기(km²/m)는 ① 옵션 ``slope_km2_per_m`` → ② 지역 매니페스트 ``satellite.level_area.slope_km2_per_m``
  → ③ 지역 면적 계열(``satellite/area_series.json`` 의 공통 격자 면적 × 같은 날 수위)의 선형 적합 순으로 정한다.
  ③ 은 관측이 적어 잡음에 맞출 수 있으므로 경고를 남기고, 기울기는 0 이상으로 자른다(수위가 오르면 면적도 넓어진다).
- 마스크는 마지막 프레임의 수체 경계에서 화소를 더하거나 빼서 목표 면적에 맞춘다(``irregular-area-trend`` 와 같은 방식).
- 선택 ``anchor="mean_reversion"`` 이면 출발 면적을 직전값 대신 평균회귀(직전값 + 0.5 × (평균 − 직전값))로 둔다 —
  공식 지표의 면적 모델에 수위 항을 더한 형태.

옵션(``model_options_json``): region_id, station_id, forecaster, as_of, slope_km2_per_m, anchor(last|mean_reversion).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date as Date
from typing import Any

import numpy as np

from ..schemas import ModelInfo
from .base import PredictionAdapter, PredictionContext, PredictionOutput
from .irregular_area_trend import _mask_with_area

MEAN_REVERSION_PHI = 0.5


def fit_slope(levels: Sequence[float], areas_km2: Sequence[float]) -> tuple[float | None, float | None, int]:
    """같은 날 (수위, 면적) 쌍의 최소제곱 기울기(km²/m)와 피어슨 상관. 쌍이 3개 미만이거나 수위 변동이 없으면 None."""

    pairs = [(float(h), float(a)) for h, a in zip(levels, areas_km2) if h is not None and a is not None]
    if len(pairs) < 3:
        return None, None, len(pairs)
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    sx = float(np.sum((x - x.mean()) ** 2)); sy = float(np.sum((y - y.mean()) ** 2))
    if sx <= 1e-12 or sy <= 1e-12:
        return None, None, len(pairs)
    cov = float(np.sum((x - x.mean()) * (y - y.mean())))
    return cov / sx, cov / math.sqrt(sx * sy), len(pairs)


class LevelAreaBridgeAdapter(PredictionAdapter):
    def __init__(self, gauge_service: Any) -> None:
        self._gauge = gauge_service

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            id="level-area-bridge",
            name="Level-to-area bridge (gauge forecast × stage-area slope)",
            version="1.0.0",
            description=(
                "면적을 따로 예측하지 않고 수위 전망에 종속시킨다: 면적(t+h) = 마지막 탐지 면적 + 기울기 × (수위 전망 − 원점 수위). "
                "수위 전망은 지역 레지스트리의 학습 모델, 기울기는 옵션 → 지역 매니페스트(satellite.level_area) → 면적 계열 적합 순. "
                "옵션 region_id·station_id·forecaster·as_of·slope_km2_per_m·anchor(last|mean_reversion)."
            ),
            min_frames=1,
            supports_weather=False,
            produces_water_level=True,
            built_in=True,
            options={"region_id": None, "station_id": None, "forecaster": None, "as_of": None, "slope_km2_per_m": None, "anchor": "last"},
        )

    # ── 대상 지역·지점 (gauge_level 어댑터와 같은 규칙) ──────────────────────────
    def _pick_target(self, options: Mapping[str, Any]) -> tuple[str, str]:
        regions = self._gauge.regions.list()
        region_id = options.get("region_id")
        if not region_id:
            def trained(record) -> bool:
                try:
                    return any(s.get("model_trained") for s in self._gauge.region_summary(record)["stations"])
                except Exception:  # noqa: BLE001
                    return False

            ordered = sorted(regions, key=lambda r: (r.origin != "builtin", not r.satellite_available, r.id))
            preferred = [r for r in ordered if r.gauge_available and trained(r)]
            if not preferred:
                raise ValueError("학습된 수위계 지점이 있는 지역이 없습니다 — model_options.region_id 를 지정하세요")
            region_id = preferred[0].id
        record = self._gauge.regions.get(region_id)
        summary = self._gauge.region_summary(record)
        stations = [s for s in summary["stations"] if s.get("model_trained")]
        if not stations:
            raise ValueError(f"지역 {region_id!r} 에 학습된 지점이 없습니다")
        station_id = options.get("station_id")
        if not station_id:
            configured = (record.manifest.get("satellite") or {}).get("level_area") or {}
            station_id = configured.get("station") or next((s["id"] for s in stations if s.get("satellite_location")), stations[0]["id"])
        return str(region_id), str(station_id)

    # ── 기울기 ─────────────────────────────────────────────────────────────────
    def _slope(self, options: Mapping[str, Any], region_id: str, station_id: str, warnings: list[str]) -> tuple[float, str]:
        raw = options.get("slope_km2_per_m")
        if raw not in (None, ""):
            return max(0.0, float(raw)), "option"
        record = self._gauge.regions.get(region_id)
        configured = (record.manifest.get("satellite") or {}).get("level_area") or {}
        if configured.get("slope_km2_per_m") is None:
            # 사용자 사본(REGIONS_USER_DIR)이 내장 지역을 가리면 사본에는 설정이 없을 수 있다 — 내장 매니페스트로 폴백
            builtin = getattr(self._gauge.regions, "builtin_record", lambda _id: None)(region_id)
            if builtin is not None and builtin is not record:
                configured = (builtin.manifest.get("satellite") or {}).get("level_area") or {}
        if configured.get("slope_km2_per_m") is not None:
            return max(0.0, float(configured["slope_km2_per_m"])), f"region_manifest:{configured.get('source') or 'satellite.level_area'}"
        # 면적 계열에서 적합 — 센서별 공통 격자 면적 × 같은 날 수위(전 센서 합산)
        levels: list[float] = []
        areas: list[float] = []
        payload = record.area_series() or {}
        for block in (payload.get("sensors") or {}).values():
            for row in block.get("rows") or []:
                level = (row.get("gauge_levels") or {}).get(station_id)
                if level is not None and row.get("area_km2_common_frame") is not None:
                    levels.append(float(level)); areas.append(float(row["area_km2_common_frame"]))
        slope, corr, n = fit_slope(levels, areas)
        if slope is None:
            warnings.append(f"수위-면적 기울기를 정할 자료가 없어 0 으로 둡니다 (쌍 {n}개) — 면적은 직전값과 같습니다")
            return 0.0, "none"
        warnings.append(f"기울기를 면적 계열 {n}쌍에서 적합했습니다 (r={corr:+.2f}, {slope:+.2f} km²/m) — 관측이 적어 잡음에 맞았을 수 있습니다")
        return max(0.0, float(slope)), f"area_series_fit:n={n}"

    def predict(
        self,
        frames: np.ndarray,
        weather: Sequence[Mapping[str, Any]],
        request: PredictionContext,
    ) -> PredictionOutput:
        options = {**self.info.options, **dict(request.options or {})}
        observed = frames >= request.threshold
        areas_px = np.count_nonzero(observed, axis=(1, 2)).astype(np.float64)
        if len(areas_px) == 0:
            raise ValueError("level-area-bridge requires at least one input frame")
        last_px = float(areas_px[-1])
        anchor = str(options.get("anchor") or "last")
        if anchor == "mean_reversion":
            anchor_px = last_px + MEAN_REVERSION_PHI * (float(np.mean(areas_px)) - last_px)
        else:
            anchor = "last"; anchor_px = last_px
        warnings: list[str] = []
        region_id, station_id = self._pick_target(options)  # ValueError → 서비스가 422
        as_of_raw = options.get("as_of")
        if as_of_raw:
            as_of = Date.fromisoformat(str(as_of_raw)[:10])
        elif request.source_dates:
            as_of = request.source_dates[-1]
        else:
            as_of = None
        payload = self._gauge.forecast(
            region_id, station_id, as_of=as_of, horizon=int(request.horizon_steps), history_days=7,
            forecaster_id=options.get("forecaster") or None,
        )
        if as_of is not None and str(payload.get("as_of")) != as_of.isoformat():
            warnings.append(f"gauge series for {station_id} ends at {payload.get('as_of')}; forecast origin was clamped to that date")
        origin_level = float(payload["origin_level_m"])
        slope, slope_source = self._slope(options, region_id, station_id, warnings)
        px_per_m = slope * 1e6 / float(request.pixel_area_m2 or 1.0)  # km²/m → 화소/m
        levels: list[float | None] = [None] * int(request.horizon_steps)
        for step in payload.get("steps") or []:
            index = int(step["horizon"]) - 1
            if 0 <= index < len(levels):
                levels[index] = step.get("forecast_m")
        masks: list[np.ndarray] = []
        targets: list[dict[str, Any]] = []
        seed = observed[-1]
        for index in range(int(request.horizon_steps)):
            level = levels[index]
            delta_m = (float(level) - origin_level) if level is not None else 0.0
            target_px = max(0.0, anchor_px + px_per_m * delta_m)
            mask = _mask_with_area(seed, int(round(target_px)))
            masks.append(mask)
            targets.append({
                "horizon": index + 1, "forecast_level_m": level, "delta_level_m": round(delta_m, 4),
                "target_area_km2": round(target_px * float(request.pixel_area_m2 or 1.0) / 1e6, 4),
                "mask_area_km2": round(int(np.count_nonzero(mask)) * float(request.pixel_area_m2 or 1.0) / 1e6, 4),
            })
            if level is None:
                warnings.append(f"D+{index + 1}: 수위 전망이 없어 면적을 출발값으로 둡니다")
        metadata = {
            "model": "level-area-bridge",
            "anchor": anchor,
            "anchor_area_km2": round(anchor_px * float(request.pixel_area_m2 or 1.0) / 1e6, 4),
            "slope_km2_per_m": slope, "slope_source": slope_source,
            "origin_level_m": origin_level,
            "gauge_region_id": region_id, "gauge_station_id": station_id, "gauge_as_of": payload.get("as_of"),
            "gauge_models_by_horizon": payload.get("models_by_horizon"), "gauge_params_used": payload.get("params_used"),
            "water_level_source_detail": "gauge_forecast",
            "steps": targets,
            "note_ko": "면적 = 출발 면적 + 기울기 × (수위 전망 − 원점 수위). 면적 오차는 기울기 × 수위 전망 오차에 비례한다.",
        }
        return PredictionOutput(masks=np.stack(masks), water_levels_m=levels, metadata=metadata, warnings=warnings)


__all__ = ["LevelAreaBridgeAdapter", "fit_slope"]
