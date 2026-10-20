"""간단 화면(/simple) — 지역·날짜 고르고 버튼 하나로 그림·수위·면적을 보는 아이콘 위주 페이지 (2026-10-15).

서비스 화면 스텁(test_service_ui.build_stub_app)이 페이지가 쓰는 API 를 전부 구현하므로 그 위에 ``mount_simple_ui`` 를
붙여 본다. 페이지 자체는 설명 문장이 없어야 하므로(요구: 아이콘·그림·숫자만) 본문 글자 수 상한도 검사한다.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from backend.app.simple_ui import INDEX_FILE, mount_simple_ui
from backend.tests.test_service_ui import _check_html, build_stub_app


@pytest.fixture()
def client() -> TestClient:
    app = build_stub_app(tick=1, seed=True)
    mount_simple_ui(app)
    return TestClient(app)


def test_simple_page_is_served_with_icon_first_labels(client: TestClient) -> None:
    for path in ("/simple", "/simple/"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith("text/html")
        assert res.headers.get("cache-control") == "no-cache"
    for alias in ("/simple/busan", "/simple/daecheong", "/simple/nakdong_busan"):
        assert client.get(alias).status_code == 200, alias  # 시연용 바로 가기 — 같은 페이지, 화면이 주소를 읽어 그 지역으로
    geo = client.get("/simple/korea.json")
    assert geo.status_code == 200 and geo.json()["type"] == "FeatureCollection" and len(geo.json()["features"]) >= 10  # 지도 바탕
    html = client.get("/simple").text
    assert '<html lang="ko">' in html
    assert "지역을 고르면" in html and "homeCards" in html and "flowDiagram" in html  # 첫 화면 = 흐름도 + 지역 큰 카드
    # 단계형 흐름(2026-10-16): 홈(흐름도·지역 카드) → ② 위성 데이터 → ③ 입력 정보(세부 지점·기간) → ④ 예측 범위(지상/위성 수위) → ⑤ 모델 → ⑥ 결과
    for label in ("위성 데이터", "입력 정보", "예측 범위", "모델", "결과", "세부 지점", "지상 수위계", "위성 수위계", "물 영역", "수위 전망", "면적 전망", "일평균 수위", "공통 격자", "흔들림 폭", "자세히", "새 영상 올리기", "영상 없이"):
        assert label in html, label
    assert html.index("위성 데이터 넣기") < html.index("입력 정보 — 영상과") < html.index("예측 범위 — ") < html.index("<span class=\"n\">5</span>수위 모델") < html.index("<h2>물 영역</h2>")
    for label in ("촬영 시각", "3-class 마스크", "수체 마스크", "정답 라벨", "서버에 있는 파일", "연계 정보", "입력 점검", "접수하고 탐지 시작", "계열 포함", "빼기", "새 지역 등록", "지도에서 고르기", "후보 관측소"):
        assert label in html, label  # 접수 규격의 입력이 전부 폼에 있다
    for label in ("단위 — 하루 평균 / 시간", "unitChips", "시간 단위", "forecast-hourly", "시간 뒤까지", "시각 직접 고르기", "hourAsOf", "시 단위 릿지 자기회귀"):
        assert label in html, label  # 2026-10-20: ④ 단위(일/시간) 토글 → 시 단위 전망(지상 수위계 전용)
    body = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
    visible = re.sub(r"<[^>]+>", " ", body)
    visible = re.sub(r"\s+", " ", visible).strip()
    # 단계마다 한두 줄 안내만 정적으로 둔다(수치·설명 줄은 결과가 나올 때 자료 값으로 채운다)
    assert len(visible) < 2200, visible
    for key in ("예상 오차", "RMSE", "실제와 맞춰", "직전값 유지", "기울기"):
        assert key in html, key  # 결과 설명 줄 문구(스크립트 안)
    for jargon in ("GeoTIFF", "pipeline", "endpoint", "multipart", "API"):
        assert jargon not in visible, jargon
    assert "/app" in html  # '자세히' 링크 하나만 다른 화면으로


def test_simple_page_is_offline_and_well_formed() -> None:
    html = INDEX_FILE.read_text(encoding="utf-8")
    urls = [u for u in re.findall(r"https?://[^\s\"'<>)]+", html) if u != "http://www.w3.org/2000/svg"]
    assert urls == [], urls  # 외부 링크 없음
    assert "<script src=" not in html and 'rel="stylesheet"' not in html and "@import" not in html and "@font-face" not in html
    assert "Content-Security-Policy" in html and "default-src 'self'" in html and "img-src 'self' data:" in html
    assert INDEX_FILE.stat().st_size < 170 * 1024
    parsed = _check_html(html)
    assert parsed.tags > 60 and parsed.problems == []


def test_simple_page_contract_is_covered_by_stub(client: TestClient) -> None:
    """페이지가 호출하는 API 가 스텁에서 전부 답한다 — 지역 → 날짜 → 예측 순서 그대로."""

    regions = client.get("/api/v1/scenes/options").json()["regions"]
    region = next(r for r in regions if r["id"] == "geum_daecheong")
    scenes = [it for it in client.get("/api/v1/scenes", params={"region_id": region["id"]}).json()["items"] if it["status"] == "done"]
    assert scenes, "날짜 칩이 될 끝난 장면이 있어야 한다"
    scene = sorted(scenes, key=lambda it: it["image_date"])[-1]
    sc = client.get(f"/api/v1/regions/{region['id']}/scenarios", params={"as_of": scene["image_date"], "horizon": 7}).json()
    assert sc["gauge"]["period"] and sc["requested"]["horizon"] == 7
    detail = client.get(f"/api/v1/scenes/{scene['scene_id']}").json()
    assert detail["outputs"]["overlay_url"] and detail["outputs"]["quicklook_url"]
    assert client.get(detail["outputs"]["overlay_url"]).status_code == 200
    stations = client.get(f"/api/v1/regions/{region['id']}/gauge/stations").json()
    station = next(s for s in stations if s.get("model_trained"))
    origin = sc["requested"]["resolved_as_of"] if sc["requested"]["available"] is False else scene["image_date"]
    fc = client.get(f"/api/v1/regions/{region['id']}/gauge/{station['id']}/forecast", params={"as_of": origin, "horizon": 7, "history_days": 30}).json()
    assert len(fc["steps"]) == 7 and fc["origin_level_m"] is not None and "data_period" in fc
    pred = client.post(f"/api/v1/scenes/{scene['scene_id']}/predict", json={"model_id": "level-area-bridge", "horizon": 7, "as_of": scene["image_date"]})
    assert pred.status_code == 200 and len(pred.json()["summary"]["steps"]) == 7
    fallback = client.post(f"/api/v1/scenes/{scene['scene_id']}/predict", json={"model_id": "irregular-area-trend", "horizon": 7, "as_of": scene["image_date"]})
    assert fallback.status_code == 200
