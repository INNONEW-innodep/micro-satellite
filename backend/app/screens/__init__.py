"""서비스 화면(/app)·모델 평가 화면(/eval)용 백엔드 계약 — docs/ui/서비스_화면_설계.md §4.

새 알고리즘은 없다. 기존 지역 레지스트리·수위계 서비스·장면 파이프라인·기상 서비스·WAMIS 스크립트를
화면이 바로 쓸 수 있는 모양으로 **엮기만** 한다.

- ``geometry``        — 지역 지도 기하(경계·지점·AOI·장면 촬영범위)와 준비 상태(readiness)
- ``wamis``           — WAMIS 유역 관측소 목록(캐시)·일 수위 수집 잡 → 지역 시계열 병합
- ``weather_collect`` — 기상청 ASOS 일자료 수집 잡 → 지역 사본 weather/asos_daily.json + 매니페스트
- ``scene_predict``   — 장면 파이프라인 predict 단계만 선택 모델·예측일수(·원점)로 다시 실행
- ``scenarios``       — 자료가 있는 기간(수위·기상·위성)과 전망 시나리오(추천 출발일·예측 창) — 2026-10-14 추가
- ``gauge_hourly``    — WAMIS 시 수위 수집 잡·자료 현황·시 단위 전망(릿지 AR 즉석 학습) — 2026-10-20 추가
- ``exog_feeds``      — 기상청 단기예보 수집·조회, 댐 운영 일자료(WAMIS) 수집 잡·조회 — 2026-10-20 추가
- ``jobs``            — 수집 잡 공용 저장소(GaugeTrainJobs 와 같은 꼴)

설치는 ``backend/app/screens_api.py`` 의 ``install_screens_api(app)`` 한 곳에서 한다.
"""

from __future__ import annotations

from .exog_feeds import router as exog_feeds_router
from .gauge_hourly import router as gauge_hourly_router
from .geometry import router as geometry_router
from .jobs import JobConflictError, ScreenJobs
from .scenarios import router as scenarios_router
from .scene_predict import router as scene_predict_router
from .wamis import WamisStationCache
from .wamis import router as wamis_router
from .weather_collect import router as weather_collect_router

SCREEN_ROUTERS = (geometry_router, wamis_router, weather_collect_router, scene_predict_router, scenarios_router, gauge_hourly_router, exog_feeds_router)

__all__ = [
    "SCREEN_ROUTERS",
    "JobConflictError",
    "ScreenJobs",
    "WamisStationCache",
    "exog_feeds_router",
    "gauge_hourly_router",
    "geometry_router",
    "scenarios_router",
    "scene_predict_router",
    "wamis_router",
    "weather_collect_router",
]
