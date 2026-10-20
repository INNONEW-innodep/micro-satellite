"""통합 장면 파이프라인 — 위성 장면 하나의 전체 체인을 한 입구(``POST /scenes``)로.

기존 조각(장면 등록·WBMS 탐지 잡·공통 격자 면적 계열·③ calc_wlwa·면적 시계열 예측·연계 규약 파일/로그)을
새로 만들지 않고 **엮기만** 한다. ``install_pipeline(app)`` 이 ``app.state`` 의 기존 서비스 객체로 잡 관리자를
만들고 라우터를 ``settings.api_prefix`` 아래에 붙인다 — ``main.create_app`` 의 끝에서 한 번 호출한다.
"""

from __future__ import annotations

from typing import Any

from .jobs import SceneConflictError, ScenePipelineJobs
from .router import router
from .steps import (
    STEP_LABELS,
    STEP_ORDER,
    PipelineServices,
    ScenePipelineRunner,
    StepSkipped,
)


def install_pipeline(app: Any, **options: Any) -> ScenePipelineJobs:
    """``app.state`` 의 regions·gauge_service·segmenters·wbms_segmentation·prediction_service·store·settings 로
    잡 관리자를 만들어 ``app.state.scene_pipeline`` 에 두고 라우터를 ``/api/v1`` (settings.api_prefix) 아래에 포함한다.

    두 번 불러도 한 번만 설치한다. ``options`` 는 ``ScenePipelineJobs`` 로 넘어간다(``max_workers``, ``poll_s`` — 테스트용).
    """

    existing = getattr(app.state, "scene_pipeline", None)
    if existing is not None:
        return existing
    services = PipelineServices.from_app(app)
    jobs = ScenePipelineJobs(services, **options)
    app.state.scene_pipeline = jobs
    prefix = str(getattr(app.state.settings, "api_prefix", "/api/v1"))
    app.include_router(router, prefix=prefix)
    return jobs


__all__ = [
    "STEP_LABELS",
    "STEP_ORDER",
    "PipelineServices",
    "SceneConflictError",
    "ScenePipelineJobs",
    "ScenePipelineRunner",
    "StepSkipped",
    "install_pipeline",
    "router",
]
