"""서비스 화면(/app) — 자료 넣기·위치 확인 → 탐지 → 시각화 → 예측을 한 페이지에서 끝내는 운영자용 화면.

위성·소프트웨어를 모르는 운영자가 쓰는 화면이라 외부 자원(CDN·웹폰트·프레임워크) 없이
``backend/app/static/app/index.html`` 한 파일로 동작한다(지도·차트도 인라인 SVG). 통합 측은 ``create_app`` 안에서

    from .service_ui import mount_service_ui
    mount_service_ui(app)

한 줄만 호출하면 된다. 루트(``/``)는 건드리지 않는다(기존 ``/docs`` 리다이렉트 유지 — ``/`` → ``/app`` 전환은 통합 측 몫).
페이지가 호출하는 API(같은 출처, 상대 경로):

    GET  /api/v1/health
    GET  /api/v1/scenes/options
    GET  /api/v1/scenes                                   (최근 결과 · 최근 장면 고르기)
    POST /api/v1/scenes                                   (multipart)
    GET  /api/v1/scenes/{scene_id}                        (2초 간격 폴링)
    POST /api/v1/scenes/{scene_id}/predict                {model_id, horizon, as_of?}  (as_of = 면적 시나리오 원점)
    GET  /api/v1/regions/{region_id}/scenarios?as_of=     (④ 자료가 있는 기간 · 추천 출발일 · 영상 날짜 보정)
    GET  /api/v1/regions/{region_id}/geometry             (위치 지도)
    GET  /api/v1/regions/{region_id}/satellite/area-series
    GET  /api/v1/regions/{region_id}/satellite/levels
    GET  /api/v1/regions/{region_id}/gauge/stations
    GET  /api/v1/regions/{region_id}/gauge/{station}/forecast?horizon=&forecaster=&as_of=
    GET  /api/v1/gauge/forecasters
    GET  /api/v1/models

외부로 나가는 링크는 "지도에서 보기(OpenStreetMap)" 하나뿐이다.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

SERVICE_UI_DIR = Path(__file__).resolve().parent / "static" / "app"
INDEX_FILE = SERVICE_UI_DIR / "index.html"
DEFAULT_PATH = "/app"


def read_index_html() -> str:
    """index.html 본문을 매 요청마다 읽는다 — 파일 하나라 비용이 없고, 재시작 없이 수정이 반영된다."""
    return INDEX_FILE.read_text(encoding="utf-8")


def mount_service_ui(app: FastAPI, path: str = DEFAULT_PATH) -> str:
    """``GET {path}`` 와 ``GET {path}/`` 에 서비스 화면을 붙인다. 붙인 경로를 돌려준다.

    * OpenAPI 문서에는 노출하지 않는다(include_in_schema=False).
    * ``Cache-Control: no-cache`` 로 배포 직후 브라우저 캐시 때문에 옛 화면이 보이는 일을 막는다.
    * 루트(``/``)나 다른 라우트는 건드리지 않는다.
    """
    path = "/" + path.strip("/")

    async def service_index() -> HTMLResponse:
        return HTMLResponse(read_index_html(), headers={"Cache-Control": "no-cache"})

    app.add_api_route(path, service_index, methods=["GET"], include_in_schema=False, name="service_ui_index")
    app.add_api_route(
        path + "/", service_index, methods=["GET"], include_in_schema=False, name="service_ui_index_slash"
    )
    return path
