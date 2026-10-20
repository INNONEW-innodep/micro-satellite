"""쉬운 화면(/easy) — 위성 영상 한 장으로 물 영역을 확인하는 단일 페이지.

위성·소프트웨어를 모르는 사람이 쓰는 화면이라 외부 자원(CDN·웹폰트·프레임워크) 없이
``backend/app/static/easy/index.html`` 한 파일로 동작한다. 통합 측은 ``create_app`` 안에서

    from .easy_ui import mount_easy_ui
    mount_easy_ui(app)

한 줄만 호출하면 된다. 루트(``/``)는 건드리지 않는다(기존 ``/docs`` 리다이렉트 유지).
페이지가 호출하는 API(같은 출처, 상대 경로):

    GET  /api/v1/health
    GET  /api/v1/scenes/options
    POST /api/v1/scenes            (multipart)
    GET  /api/v1/scenes            (최근 결과)
    GET  /api/v1/scenes/{scene_id} (2초 간격 폴링)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

EASY_UI_DIR = Path(__file__).resolve().parent / "static" / "easy"
INDEX_FILE = EASY_UI_DIR / "index.html"
ASSETS_DIR = EASY_UI_DIR / "assets"
DEFAULT_PATH = "/easy"


def read_index_html() -> str:
    """index.html 본문을 매 요청마다 읽는다 — 파일 하나라 비용이 없고, 재시작 없이 수정이 반영된다."""
    return INDEX_FILE.read_text(encoding="utf-8")


def mount_easy_ui(app: FastAPI, path: str = DEFAULT_PATH) -> str:
    """``GET {path}`` 와 ``GET {path}/`` 에 쉬운 화면을 붙인다. 붙인 경로를 돌려준다.

    * OpenAPI 문서에는 노출하지 않는다(include_in_schema=False).
    * ``Cache-Control: no-cache`` 로 배포 직후 브라우저 캐시 때문에 옛 화면이 보이는 일을 막는다.
    * ``static/easy/assets/`` 폴더가 있으면 ``{path}/assets`` 로 정적 서빙한다(현재는 단일 파일이라 비어 있어도 된다).
    * 루트(``/``)나 다른 라우트는 건드리지 않는다.
    """
    path = "/" + path.strip("/")

    async def easy_index() -> HTMLResponse:
        return HTMLResponse(read_index_html(), headers={"Cache-Control": "no-cache"})

    app.add_api_route(path, easy_index, methods=["GET"], include_in_schema=False, name="easy_ui_index")
    app.add_api_route(
        path + "/", easy_index, methods=["GET"], include_in_schema=False, name="easy_ui_index_slash"
    )
    if ASSETS_DIR.is_dir():
        app.mount(path + "/assets", StaticFiles(directory=str(ASSETS_DIR)), name="easy_ui_assets")
    return path
