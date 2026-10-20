"""``POST /scenes/{scene_id}/predict`` — 장면 파이프라인의 predict 단계만 선택 모델·예측일수로 다시 돌린다.

서비스 화면 ④ 에서 "면적 전망" 모델(``GET /models`` 의 어댑터)과 예측일수를 바꿔 볼 때 쓴다. 등록·탐지·면적·수위는
다시 하지 않는다. 결과는 장면 status.json 의 ``outputs.prediction_id/prediction_url`` 과 ``predictions[]`` 이력에 남는다
(``ScenePipelineJobs.record_prediction``). 파이프라인 기록이 없는 기존 장면(내장 지역 자산)도 지역 장면 기록으로 합성해 받는다.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..gauge.regions import RegionNotFoundError
from ..pipeline.jobs import ScenePipelineJobs
from ..pipeline.steps import PREDICT_MODEL_ID, StepSkipped, predict_area_series

router = APIRouter(tags=["scenes"])


class ScenePredictRequest(BaseModel):
    model_id: str = Field(default=PREDICT_MODEL_ID, min_length=1, description="GET /models 의 어댑터 id (예: irregular-area-trend, persistence)")
    horizon: int = Field(default=7, ge=1, le=30, description="예측 단계 수(장면 간격 단위) 1..30")
    as_of: Date | None = Field(
        default=None,
        description="시나리오 원점 — 이 날짜(포함)까지의 영상만으로 예측하고, 그 뒤 영상의 실제 면적은 summary.later_frames 로 돌려준다. 비우면 영상 전부",
    )
    exclude_scene_ids: list[str] = Field(default_factory=list, max_length=200, description="면적 계열에서 뺄 장면 id 목록(품질이 나쁜 장면 등). 출발 장면은 빼지 않는다 — summary.excluded_scene_ids 로 확인")


def _jobs(request: Request) -> ScenePipelineJobs:
    jobs = getattr(request.app.state, "scene_pipeline", None)
    if jobs is None:
        raise HTTPException(status_code=503, detail="scene pipeline is not installed (install_pipeline(app))")
    return jobs


@router.post(
    "/scenes/{scene_id}/predict",
    responses={
        404: {"description": "모르는 장면"},
        409: {"description": "이 장면의 파이프라인이 아직 진행 중"},
        422: {"description": "모르는 모델, 또는 센서 공통 격자 프레임이 3개 미만(예측 불가), 또는 이 장면에 마스크 없음"},
    },
    summary="Re-run only the predict step of a scene with a chosen area model and horizon",
    description=(
        "`model_id` 는 `GET /models` 의 어댑터 id, `horizon` 1..30(기본 7). 장면 센서의 공통 격자 프레임 전부(최소 3개)로 "
        "면적 시계열 예측을 돌리고 장면 status.json 의 `outputs.prediction_id/prediction_url` 을 갱신하며 `predictions[]` "
        "이력에 {model_id, horizon, prediction_id, created_at(, as_of)} 을 덧붙인다. 응답 `summary.steps[]` {horizon, target_date, "
        "water_area_km2, area_change_pct, risk}. `as_of`(시나리오 원점)를 주면 그 날짜까지의 프레임만 쓰고 그 뒤 프레임의 실제 "
        "면적을 `summary.later_frames[]` {date, scene_id, water_area_km2} 로 돌려준다. 전체 결과는 `prediction_url`(GET /predictions/{id})."
    ),
)
async def scene_predict(scene_id: str, payload: ScenePredictRequest, request: Request) -> dict[str, Any]:
    jobs = _jobs(request)
    services = jobs.services
    record = jobs.get(scene_id)
    if record is None:
        record = await run_in_threadpool(jobs.synthesize, scene_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown scene {scene_id!r}")
    if jobs.active(scene_id):
        raise HTTPException(status_code=409, detail=f"장면 {scene_id} 의 파이프라인이 아직 진행 중입니다 — 끝난 뒤 다시 예측하세요")
    registry = services.prediction_service.registry
    try:
        registry.get(payload.model_id)
    except KeyError:
        known = sorted(m.id for m in registry.list())
        raise HTTPException(status_code=422, detail=f"모르는 예측 모델 {payload.model_id!r} — 가능: {known}") from None
    region_id, sensor = str(record["region_id"]), str(record["sensor"])
    try:
        region = services.regions.get(region_id)
    except RegionNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r} of scene {scene_id!r}") from None
    own = next((s for s in (region.scenes() or {}).get("scenes") or [] if str(s.get("scene_id")) == scene_id), None)
    if own is None or not (own.get("frame") or {}).get("path"):
        raise HTTPException(
            status_code=422,
            detail=f"장면 {scene_id} 에는 공통 격자 프레임(마스크)이 없어 면적 시계열에 넣을 수 없습니다 — 탐지가 끝난 장면을 고르세요",
        )
    try:
        summary = await run_in_threadpool(
            predict_area_series, services,
            region_id=region_id, sensor=sensor, scene_id=scene_id,
            model_id=payload.model_id, horizon_steps=payload.horizon,
            pipeline_id=record.get("pipeline_id"), fallback_model_id=None, as_of=payload.as_of,
            exclude_scene_ids=payload.exclude_scene_ids,
        )
    except StepSkipped as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, TypeError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=f"예측 실패: {exc}") from exc
    view = await run_in_threadpool(jobs.record_prediction, scene_id, summary)
    prefix = services.api_prefix
    return {
        "scene_id": scene_id,
        "region_id": region_id,
        "sensor": sensor,
        "model_id": summary["model_id"],
        "model_version": summary.get("model_version"),
        "horizon": int(summary["horizon_steps"]),
        "as_of": summary.get("as_of"),
        "prediction_id": summary["prediction_id"],
        "prediction_url": summary["result_url"],
        "summary": {
            "steps": summary["steps"],
            "frame_count": summary.get("frame_count"),
            "source_dates": summary.get("source_dates"),
            "scene_ids": summary.get("scene_ids"),
            "excluded_scene_ids": summary.get("excluded_scene_ids") or [],
            "pixel_area_m2": summary.get("pixel_area_m2"),
            "later_frames": summary.get("later_frames") or [],
            "adapter_metadata": summary.get("adapter_metadata") or {},
            "warnings": summary.get("warnings") or [],
        },
        "predictions": view.get("predictions") or [],
        "status_url": f"{prefix}/scenes/{scene_id}",
    }


__all__ = ["ScenePredictRequest", "router"]
