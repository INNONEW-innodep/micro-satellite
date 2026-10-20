"""모델 평가 화면(/eval) — 새 지역의 수위·기상 자료 수집 → 학습 → 성능표 → 시험까지 한 페이지.

자료를 다루는 사람이 문서 없이 지역을 올리고 모델 성능을 보는 화면이라, 외부 자원(CDN·웹폰트·프레임워크)
없이 ``backend/app/static/eval/index.html`` 한 파일로 동작한다(CSP self). 통합 측은 ``create_app`` 안에서

    from .eval_ui import mount_eval_ui
    mount_eval_ui(app)

한 줄만 호출하면 된다. 루트(``/``)는 건드리지 않는다. 페이지가 호출하는 API(같은 출처, 상대 경로):

    GET  /api/v1/health
    GET  /api/v1/regions                              POST /api/v1/regions
    GET  /api/v1/regions/{id}/readiness
    GET  /api/v1/wamis/stations?basin=&match=
    POST /api/v1/regions/{id}/gauge/collect           GET  …/gauge/collect/jobs/{job_id}   (2초 폴링)
    POST /api/v1/regions/{id}/gauge/{station}/series/file   (multipart)
    GET  /api/v1/weather/stations
    POST /api/v1/regions/{id}/weather/collect         GET  …/weather/collect/jobs/{job_id} (2초 폴링)
    GET  /api/v1/gauge/forecasters
    POST /api/v1/regions/{id}/gauge/train (async)     GET  …/gauge/train/jobs/{job_id}     (2초 폴링)
    GET  /api/v1/regions/{id}/gauge/evaluation
    GET  /api/v1/regions/{id}/gauge/{station}/forecast/compare?horizon=
    GET  /api/v1/regions/{id}/gauge/{station}/backtest?horizon=
    GET  /api/v1/regions/{id}/gauge/experiments
    GET  /api/v1/regions/{id}/gauge/{station}/forecast?as_of=&forecaster=&horizon=
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

EVAL_UI_DIR = Path(__file__).resolve().parent / "static" / "eval"
INDEX_FILE = EVAL_UI_DIR / "index.html"
ASSETS_DIR = EVAL_UI_DIR / "assets"
DEFAULT_PATH = "/eval"


def read_index_html() -> str:
    """index.html 본문을 매 요청마다 읽는다 — 파일 하나라 비용이 없고, 재시작 없이 수정이 반영된다."""
    return INDEX_FILE.read_text(encoding="utf-8")


def mount_eval_ui(app: FastAPI, path: str = DEFAULT_PATH) -> str:
    """``GET {path}`` 와 ``GET {path}/`` 에 모델 평가 화면을 붙인다. 붙인 경로를 돌려준다.

    * OpenAPI 문서에는 노출하지 않는다(include_in_schema=False).
    * ``Cache-Control: no-cache`` 로 배포 직후 브라우저 캐시 때문에 옛 화면이 보이는 일을 막는다.
    * ``static/eval/assets/`` 폴더가 있으면 ``{path}/assets`` 로 정적 서빙한다(현재는 단일 파일이라 없어도 된다).
    * 루트(``/``)나 다른 라우트는 건드리지 않는다.
    """
    path = "/" + path.strip("/")

    async def eval_index() -> HTMLResponse:
        return HTMLResponse(read_index_html(), headers={"Cache-Control": "no-cache"})

    app.add_api_route(path, eval_index, methods=["GET"], include_in_schema=False, name="eval_ui_index")
    app.add_api_route(
        path + "/", eval_index, methods=["GET"], include_in_schema=False, name="eval_ui_index_slash"
    )
    if ASSETS_DIR.is_dir():
        app.mount(path + "/assets", StaticFiles(directory=str(ASSETS_DIR)), name="eval_ui_assets")
    return path
