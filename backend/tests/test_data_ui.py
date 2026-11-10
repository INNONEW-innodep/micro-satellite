"""시험 자료 현황 화면(/data) — 예측에 실제로 쓰이는 원자료(지점 일 수위·장면별 수체 면적)를 표·그래프로 보는 페이지 (2026-11-10).

페이지는 읽기 전용이고 호출하는 API 가 전부 기존 것이라, (1) 서빙·캐시 헤더·오프라인·정합성, (2) 내장 낙동강 하구 지역에서
페이지가 읽는 필드가 실제 응답에 있는지(시험 스크립트가 쓰는 원점·D+1·D+7 날짜의 실측 포함)를 본다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.data_ui import INDEX_FILE, mount_data_ui
from backend.app.main import create_app
from backend.tests.test_service_ui import _check_html, build_stub_app

REAL_REGIONS_DIR = Path(__file__).resolve().parents[2] / "regions"
HAS_BUSAN = (REAL_REGIONS_DIR / "nakdong_busan" / "region.json").is_file()


@pytest.fixture()
def client() -> TestClient:
    app = build_stub_app(tick=1, seed=True)
    mount_data_ui(app)
    return TestClient(app)


def test_data_page_is_served_with_raw_data_sections(client: TestClient) -> None:
    for path in ("/data", "/data/"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith("text/html")
        assert res.headers.get("cache-control") == "no-cache"
    html = client.get("/data").text
    assert '<html lang="ko">' in html and "시험 자료 현황" in html
    # 세 절: ① 수위 raw(지점·구간·표·그래프·CSV·원본 JSON) ② 촬영일 기준 실측(원점·D+1·D+7) ③ 수체 면적 raw(공통 격자·참고값·퀵룩)
    for label in ("수위 Raw 데이터", "지상 수위관측소 일 수위", "전체 기간", "시험 연도", "촬영일 전후", "CSV 내려받기", "원본 JSON",
                  "촬영일 기준 실측 수위", "D+1", "D+7", "위성 연계", "관측범위", "정규화 분모",
                  "수체 면적 Raw 데이터", "공통 격자 면적", "장면 자체 범위에서 센 수체 면적", "비교 금지", "정답 라벨 면적", "예측 비교에 안 씀",
                  "초소형위성 촬영일", "이번 시험에 쓰는 구간", "비교 지점 아님", "학습에만 쓰임", "촬영 기준",
                  # ③ 자료 출처·위치(어디 자료인지): 지역·창 위치(경위도 환산)·영상 파일명·수체 판정(탐지 마스크)
                  "어디", "창 위치", "경위도 환산", "원본 영상 → 탐지 마스크", "수체 판정", "탐지 모델(WBMS)"):
        assert label in html, label
    assert html.index("수위 Raw 데이터") < html.index("촬영일 기준 실측 수위") < html.index("수체 면적 Raw 데이터")
    # 페이지가 부르는 API 는 전부 기존 읽기 경로
    for call in ("'/health'", "'/regions'", "/satellite/scenes", "/satellite/area-series", "/gauge/' + enc(", "/series'"):
        assert call in html, call
    assert "POST" not in html and "fetch(url, { headers: { Accept: 'application/json' } })" in html  # 읽기만 한다
    assert 'href="/app"' in html and 'href="/eval"' in html
    assert "가공 없이" in html  # 값은 서버 응답 그대로라는 안내


def test_data_page_is_offline_and_well_formed() -> None:
    html = INDEX_FILE.read_text(encoding="utf-8")
    urls = [u for u in re.findall(r"https?://[^\s\"'<>)]+", html) if u != "http://www.w3.org/2000/svg"]
    assert urls == [], urls  # 외부 링크 없음 — 시험장 LAN 에 인터넷이 없어도 뜬다
    assert "<script src=" not in html and 'rel="stylesheet"' not in html and "@import" not in html and "@font-face" not in html
    assert "Content-Security-Policy" in html and "default-src 'self'" in html and "img-src 'self' data: blob:" in html
    assert INDEX_FILE.stat().st_size < 120 * 1024
    parsed = _check_html(html)
    assert parsed.tags > 60 and parsed.problems == []
    # 자료 값은 전부 textContent 로만 넣는다(innerHTML 없음) — 지점 이름·노트가 서버 자료라서
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html
    assert "notes_ko" not in html  # 면적 설명은 페이지의 한글 주석으로 — 개발자용 서버 노트(영문 필드명)를 그대로 찍지 않는다
    assert "seq !== state.seq" in html and "++state.seq" in html  # 지역을 바꾸는 중 늦게 온 이전 지역 응답을 버리는 토큰


def test_mount_does_not_touch_root_or_api(client: TestClient) -> None:
    assert client.get("/", follow_redirects=False).status_code == 404  # 스텁에는 루트가 없고, 마운트도 루트를 만들지 않는다
    assert "/data" not in client.get("/openapi.json").json()["paths"]
    assert client.get("/api/v1/health").status_code == 200


def test_real_app_mounts_data_page(tmp_path: Path) -> None:
    app = create_app(Settings(result_dir=tmp_path / "results"))
    real = TestClient(app)
    assert real.get("/data").status_code == 200
    assert real.get("/data/").status_code == 200
    root = real.get("/", follow_redirects=False)
    assert root.headers["location"] == "/app"  # 정문은 그대로 서비스 화면
    assert "/data" not in real.get("/openapi.json").json()["paths"]  # 화면은 API 문서에 안 보임


@pytest.mark.skipif(not HAS_BUSAN, reason="저장소 내장 지역이 없다")
def test_data_page_contract_on_builtin_busan(tmp_path: Path) -> None:
    """페이지가 읽는 필드가 내장 낙동강 하구 응답에 전부 있고, 촬영일(원점)·D+1·D+7 의 실측이 비교 지점에 존재한다."""

    settings = Settings(result_dir=tmp_path / "results", regions_dir=REAL_REGIONS_DIR, regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    rid = "nakdong_busan"

    regions = client.get("/api/v1/regions").json()["items"]
    me = next(r for r in regions if r["id"] == rid)
    assert me["gauge_available"] and me["satellite_available"] and me["name"]

    detail = client.get(f"/api/v1/regions/{rid}").json()
    stations = detail["stations"]
    assert len(stations) == 6 and all(s["id"] and s["period"] and s["source"] for s in stations)
    compared = [s for s in stations if s.get("satellite_location")]
    assert {s["id"] for s in compared} == {"구포대교", "호포대교", "정천교"}

    scenes = client.get(f"/api/v1/regions/{rid}/satellite/scenes").json()
    assert scenes["n_scenes"] == 8 and len(scenes["scenes"]) == 8
    origins = sorted(sc["date"] for sc in scenes["scenes"])
    assert {d[:4] for d in origins} == {"2020"}  # 시험 연도 = 촬영 연도 하나
    assert detail["crs"] == "EPSG:32652" and len(detail["bbox_wgs84"]) == 4 and detail["description"]  # ③ '어디' 행
    for sc in scenes["scenes"]:
        assert sc["sensor"] in ("ICEYE", "PlanetScope") and sc["scene_id"]
        # ③ 파일명 열(원본 영상 → 탐지 마스크)과 장면 촬영 범위 툴팁이 읽는 필드
        assert sc["source_pre"].endswith(".tif") and sc["source_mask"].endswith(".tif") and "/" in sc["source_pre"]
        assert len(sc["quicklook"]["footprint_lonlat"]) == 4 and len(sc["frame"]["bounds_epsg32652"]) == 4
        assert sc["image_urls"]["overlay"].startswith(f"/api/v1/regions/{rid}/satellite/files/")
    assert client.get(scenes["scenes"][0]["image_urls"]["overlay"]).status_code == 200  # 퀵룩 썸네일

    area = client.get(f"/api/v1/regions/{rid}/satellite/area-series").json()
    assert area["available"] and set(area["sensors"]) == {"ICEYE", "PlanetScope"} and area["notes_ko"]
    for sensor, block in area["sensors"].items():
        assert len(block["rows"]) == 4 and block["frame"]["coverage_km2"] > 0, sensor
        assert len(block["frame"]["bounds_epsg32652"]) == 4  # ③ 센서 블록 '창 위치' 줄(UTM → 경위도 환산)
        for row in block["rows"]:
            assert row["date"] in origins and row["scene_id"] and row["area_km2_common_frame"] > 0
            assert "area_km2_scene_footprint" in row and "valid_area_km2" in row

    from datetime import date, timedelta

    for s in compared:
        series = client.get(f"/api/v1/regions/{rid}/gauge/{s['id']}/series").json()
        by_date = {r["date"]: r["water_level_m"] for r in series["rows"]}
        assert series["station"]["id"] == s["id"] and len(by_date) > 3000
        for o in origins:
            d = date.fromisoformat(o)
            for h in (0, 1, 7):
                key = (d + timedelta(days=h)).isoformat()
                assert by_date.get(key) is not None, (s["id"], o, h)  # 원점·D+1·D+7 실측이 있다 — 2번 표가 빈칸 없이 찬다
