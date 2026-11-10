"""시험 자료 현황 화면(/data) — 예측에 실제로 쓰이는 원자료를 표·그래프로 확인하는 한 페이지 (2026-11-10).

시험원(TTA)이 "어떤 자료로 예측했는가"를 서버 응답 그대로 볼 수 있게 한다. 보여 주는 것은 두 가지 원자료와 그 발췌다.

    1. 지상 수위관측소 일 수위 (지점 선택 · 전체 기간 / 시험 연도 / 촬영일 전후 · 표 · 그래프 · CSV)
    2. 촬영일 기준 실측 수위 — 원점 · D+1 · D+7 (위성 연계 비교 지점, 1번 자료에서 날짜만 뽑은 것)
    3. 위성 장면별 수체 면적 (센서별 공통 격자 면적 · 유효 면적 · 장면 범위 면적(참고) · 정답 면적 · 탐지 결과 퀵룩)

외부 자원(CDN·웹폰트·프레임워크) 없이 ``backend/app/static/data/index.html`` 한 파일로 동작한다(CSP self).
통합 측은 ``create_app`` 안에서

    from .data_ui import mount_data_ui
    mount_data_ui(app)

한 줄만 호출하면 된다. 루트(``/``)·기존 화면(/app, /easy, /eval, /simple)은 건드리지 않는다.
페이지가 호출하는 API(같은 출처, 읽기만):

    GET /api/v1/health
    GET /api/v1/regions                                   (지역 고르기 — ?region= 로 미리 고를 수 있음, 기본 nakdong_busan)
    GET /api/v1/regions/{id}                              (지점 목록·위성 연계·자료 기간)
    GET /api/v1/regions/{id}/satellite/scenes             (촬영일 = 예측 원점, 퀵룩 URL)
    GET /api/v1/regions/{id}/satellite/area-series        (센서별 공통 격자 면적)
    GET /api/v1/regions/{id}/gauge/{station}/series       (지점 일 수위 — 시험 스크립트와 같은 호출)
    GET /api/v1/regions/{id}/satellite/files/quicklooks/… (장면 응답 image_urls 의 퀵룩 PNG — 썸네일·원본 보기)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

DATA_UI_DIR = Path(__file__).resolve().parent / "static" / "data"
INDEX_FILE = DATA_UI_DIR / "index.html"
DEFAULT_PATH = "/data"


def read_index_html() -> str:
    """index.html 본문을 매 요청마다 읽는다 — 파일 하나라 비용이 없고, 재시작 없이 수정이 반영된다."""
    return INDEX_FILE.read_text(encoding="utf-8")


def mount_data_ui(app: FastAPI, path: str = DEFAULT_PATH) -> str:
    """``GET {path}`` 와 ``GET {path}/`` 에 시험 자료 현황 화면을 붙인다. 붙인 경로를 돌려준다.

    * OpenAPI 문서에는 노출하지 않는다(include_in_schema=False).
    * ``Cache-Control: no-cache`` 로 배포 직후 브라우저 캐시 때문에 옛 화면이 보이는 일을 막는다.
    """
    path = "/" + path.strip("/")

    async def data_index() -> HTMLResponse:
        return HTMLResponse(read_index_html(), headers={"Cache-Control": "no-cache"})

    app.add_api_route(path, data_index, methods=["GET"], include_in_schema=False, name="data_ui_index")
    app.add_api_route(path + "/", data_index, methods=["GET"], include_in_schema=False, name="data_ui_index_slash")
    return path


__all__ = ["DATA_UI_DIR", "DEFAULT_PATH", "INDEX_FILE", "mount_data_ui", "read_index_html"]
