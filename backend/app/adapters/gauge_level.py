"""기존 예측 API(`POST /predictions`)에 학습된 지점 수위 예측기를 붙이는 어댑터.

외부 모듈이 `model_id=persistence` 로 수위를 가져가면 모든 날짜가 같은 값이다(직전값
유지). 이 어댑터는 같은 API 계약(마스크 시퀀스 → 미래 마스크 + 수위)을 지키면서
**수위만 지역 레지스트리의 학습 모델**(검증 연도별 폴드에서 고른 지점별 모델)에서
가져온다. 마스크는 `irregular-area-trend` 와 같은 방식으로 만든다.

옵션(`model_options_json`):
- ``region_id`` (기본: 위성·수위계가 모두 있는 첫 지역), ``station_id`` (기본: 위성 연결 지점),
- ``forecaster`` (기본: 검증에서 고른 모델), ``as_of`` (기본: 마지막 입력 프레임 날짜).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date as Date
from typing import Any

import numpy as np

from ..schemas import ModelInfo
from .base import PredictionAdapter, PredictionContext, PredictionOutput
from .irregular_area_trend import IrregularAreaTrendAdapter


class GaugeLevelForecastAdapter(PredictionAdapter):
    def __init__(self, gauge_service: Any) -> None:
        self._gauge = gauge_service
        self._area = IrregularAreaTrendAdapter()

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            id="gauge-level-forecast",
            name="Station water-level forecast (trained on 10-year gauge records)",
            version="1.0.0",
            description=(
                "수위는 지역 레지스트리의 학습 모델(지점×예측일수별 검증 선택)에서, 마스크는 irregular-area-trend 와 "
                "같은 면적 추세로 만든다. persistence 와 달리 날짜마다 다른 수위를 낸다. "
                "옵션 region_id·station_id·forecaster·as_of."
            ),
            min_frames=1,
            supports_weather=True,
            produces_water_level=True,
            built_in=True,
            options={"region_id": None, "station_id": None, "forecaster": None, "as_of": None},
        )

    def _pick_target(self, options: Mapping[str, Any]) -> tuple[str, str]:
        regions = self._gauge.regions.list()
        region_id = options.get("region_id")
        if not region_id:
            # 기본 지역: 내장(저장소) 지역을 사용자 지역보다 먼저, 그중 위성·수위계 둘 다 있고 학습된 지점이 있는 곳
            def trained(record) -> bool:
                try:
                    return any(s.get("model_trained") for s in self._gauge.region_summary(record)["stations"])
                except Exception:  # noqa: BLE001 - 요약 실패 지역은 후보에서 제외
                    return False

            ordered = sorted(regions, key=lambda r: (r.origin != "builtin", not r.satellite_available, r.id))
            preferred = [r for r in ordered if r.gauge_available and trained(r)]
            if not preferred:
                raise ValueError("학습된 수위계 지점이 있는 지역이 없습니다 — model_options.region_id 를 지정하세요")
            region_id = preferred[0].id
        summary = self._gauge.region_summary(self._gauge.regions.get(region_id))
        stations = [s for s in summary["stations"] if s.get("model_trained")]
        if not stations:
            raise ValueError(f"지역 {region_id!r} 에 학습된 지점이 없습니다")
        station_id = options.get("station_id")
        if not station_id:
            station_id = next((s["id"] for s in stations if s.get("satellite_location")), stations[0]["id"])
        return str(region_id), str(station_id)

    def predict(
        self,
        frames: np.ndarray,
        weather: Sequence[Mapping[str, Any]],
        request: PredictionContext,
    ) -> PredictionOutput:
        options = dict(request.options or {})
        base = self._area.predict(frames, weather, request)
        try:
            region_id, station_id = self._pick_target(options)
        except ValueError as exc:
            return PredictionOutput(masks=base.masks, water_levels_m=None, metadata=base.metadata, warnings=[*base.warnings, str(exc)])
        as_of_raw = options.get("as_of")
        if as_of_raw:
            as_of = Date.fromisoformat(str(as_of_raw)[:10])
        elif request.source_dates:
            as_of = request.source_dates[-1]
        else:
            as_of = None
        forecaster = options.get("forecaster") or None
        payload = self._gauge.forecast(
            region_id, station_id, as_of=as_of, horizon=int(request.horizon_steps), history_days=7, forecaster_id=forecaster,
        )
        steps = payload.get("steps") or []
        levels: list[float | None] = [None] * int(request.horizon_steps)
        for step in steps:
            index = int(step["horizon"]) - 1
            if 0 <= index < len(levels):
                levels[index] = step.get("forecast_m")
        warnings = list(base.warnings)
        if as_of is not None and str(payload.get("as_of")) != as_of.isoformat():
            warnings.append(
                f"gauge series for {station_id} ends at {payload.get('as_of')}; forecast origin was clamped to that date"
            )
        metadata = {
            **base.metadata,
            "water_level_source_detail": "gauge_forecast",
            "gauge_region_id": region_id,
            "gauge_station_id": station_id,
            "gauge_as_of": payload.get("as_of"),
            "gauge_models_by_horizon": payload.get("models_by_horizon"),
            "gauge_params_used": payload.get("params_used"),
            "gauge_sigma_m_by_horizon": {str(s["horizon"]): s.get("sigma_m") for s in steps},
            "gauge_note_ko": payload.get("forecaster_note_ko"),
        }
        return PredictionOutput(masks=base.masks, water_levels_m=levels, metadata=metadata, warnings=warnings)
