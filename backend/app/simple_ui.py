"""간단 화면(/simple) — 지역·날짜를 고르고 버튼 하나로 예측 결과(그림·수위·면적)를 보는 아이콘 위주 한 페이지.

설명 문장 없이 아이콘·큰 숫자·그래프만 보여 준다(2026-10-15, 사용자 요구 "날짜 고르고 예측, 그림이랑 아주 쉽게").
외부 자원(CDN·웹폰트·프레임워크) 없이 ``backend/app/static/simple/index.html`` 한 파일로 동작한다. 통합 측은

    from .simple_ui import mount_simple_ui
    mount_simple_ui(app)

한 줄만 호출하면 된다. 루트(``/``)·기존 화면(/app, /easy, /eval)은 건드리지 않는다. 페이지가 호출하는 API(같은 출처):

    GET  /api/v1/health
    GET  /api/v1/scenes/options                          (지역 칩)
    GET  /api/v1/scenes?region_id=                       (날짜 칩 = 그 지역의 끝난 장면)
    GET  /api/v1/regions/{id}/scenarios?as_of=&horizon=  (수위 자료 기간 표시 · 영상 날짜가 자료 밖이면 출발일 보정)
    GET  /api/v1/scenes/{scene_id}                       (물 영역 그림 · 면적)
    GET  /api/v1/regions/{id}/gauge/stations             (지점 칩)
    GET  /api/v1/regions/{id}/gauge/{station}/forecast   (수위 그래프)
    POST /api/v1/scenes/{scene_id}/predict               (면적 전망: level-area-bridge → 실패 시 irregular-area-trend)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

SIMPLE_UI_DIR = Path(__file__).resolve().parent / "static" / "simple"
INDEX_FILE = SIMPLE_UI_DIR / "index.html"
DEFAULT_PATH = "/simple"


def read_index_html() -> str:
    return INDEX_FILE.read_text(encoding="utf-8")


def mount_simple_ui(app: FastAPI, path: str = DEFAULT_PATH) -> str:
    """``GET {path}``·``{path}/``·``{path}/{alias}`` 에 간단 화면을 붙인다. OpenAPI 에는 노출하지 않고 ``Cache-Control: no-cache``."""

    path = "/" + path.strip("/")

    async def simple_index() -> HTMLResponse:
        return HTMLResponse(read_index_html(), headers={"Cache-Control": "no-cache"})

    app.add_api_route(path, simple_index, methods=["GET"], include_in_schema=False, name="simple_ui_index")
    app.add_api_route(path + "/", simple_index, methods=["GET"], include_in_schema=False, name="simple_ui_index_slash")
    # 시연용 바로 가기 — /simple/busan · /simple/daecheong (또는 지역 id). 같은 페이지를 주고 화면이 주소를 읽어 그 지역으로 바로 들어간다.
    app.add_api_route(path + "/{alias}", simple_alias, methods=["GET"], include_in_schema=False, name="simple_ui_alias")
    return path


async def simple_alias(alias: str):
    if alias == "korea.json":  # 지도 바탕(시도 경계, 단순화) — 새 지역 등록 화면이 같은 출처에서 읽는다
        return FileResponse(SIMPLE_UI_DIR / "korea.json", media_type="application/json", headers={"Cache-Control": "public, max-age=86400"})
    return HTMLResponse(read_index_html(), headers={"Cache-Control": "no-cache"})


__all__ = ["DEFAULT_PATH", "INDEX_FILE", "SIMPLE_UI_DIR", "mount_simple_ui", "read_index_html"]
