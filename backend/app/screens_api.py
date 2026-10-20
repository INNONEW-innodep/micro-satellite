"""서비스 화면(/app)·모델 평가 화면(/eval)용 백엔드 계약 설치 지점 — ``install_screens_api(app)``.

``main.create_app`` 의 끝(``install_pipeline(app)`` 뒤)에서 한 번 부른다. 라우터(``backend/app/screens``)를
``settings.api_prefix`` 아래에 포함하고, 수집 잡 저장소·WAMIS 관측소 캐시를 ``app.state`` 에 만든다.
두 번 불러도 한 번만 설치한다. 장면 재예측(``POST /scenes/{id}/predict``)은 ``app.state.scene_pipeline`` 을 쓰므로
파이프라인이 먼저 설치돼 있어야 한다(없으면 그 엔드포인트만 503).
"""

from __future__ import annotations

from typing import Any

from .screens import SCREEN_ROUTERS, ScreenJobs, WamisStationCache


def install_screens_api(app: Any) -> dict[str, Any]:
    """화면용 라우터를 설치하고 ``app.state`` 에 잡 관리자를 둔다. 멱등 — 이미 설치됐으면 그 상태를 돌려준다."""

    installed = getattr(app.state, "screens_api", None)
    if installed is not None:
        return installed
    if getattr(app.state, "gauge_collect_jobs", None) is None:
        app.state.gauge_collect_jobs = ScreenJobs("gauge_collect")       # WAMIS 수집(느림·순차)
    if getattr(app.state, "weather_collect_jobs", None) is None:
        app.state.weather_collect_jobs = ScreenJobs("weather_collect")   # ASOS 수집(빠름) — 수위 수집 뒤에 줄 서지 않게 따로
    if getattr(app.state, "gauge_hourly_jobs", None) is None:
        app.state.gauge_hourly_jobs = ScreenJobs("gauge_hourly_collect")  # WAMIS 시 수위 수집(월 단위 요청, 느림)
    if getattr(app.state, "dam_collect_jobs", None) is None:
        app.state.dam_collect_jobs = ScreenJobs("dam_collect")              # WAMIS 댐 운영 자료 수집(월 단위 요청)
    if getattr(app.state, "wamis_station_cache", None) is None:
        app.state.wamis_station_cache = WamisStationCache()
    prefix = str(getattr(app.state.settings, "api_prefix", "/api/v1"))
    for router in SCREEN_ROUTERS:
        app.include_router(router, prefix=prefix)
    app.state.screens_api = {
        "installed": True,
        "prefix": prefix,
        "routes": sorted({f"{prefix}{route.path}" for router in SCREEN_ROUTERS for route in router.routes}),
    }
    return app.state.screens_api


__all__ = ["install_screens_api"]
