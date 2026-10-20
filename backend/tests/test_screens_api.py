"""서비스/모델 평가 화면용 백엔드 계약 6종(docs/ui/서비스_화면_설계.md §4) — ``install_screens_api(app)`` 로 설치.

토이 지역(test_gauge.write_region)·장면 파이프라인 픽스처(test_scene_pipeline._app)를 재사용한다. WAMIS 는 인터넷이
필요하므로 목록·수집 함수를 바꿔 끼우고, 기상은 합성 샘플 소스(source=sample)로 받는다.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.screens import wamis as wamis_module
from backend.app.screens.weather_collect import PROVIDER_SAMPLE
from backend.app.screens_api import install_screens_api
from backend.tests.test_gauge import SPLIT, _client, _geotiff_bytes, write_region
from backend.tests.test_scene_pipeline import REGION_ID, _app, _steps, _wait

REAL_REGIONS_DIR = Path(__file__).resolve().parents[2] / "regions"
CONTRACT_PATHS = {
    "/api/v1/regions/{region_id}/geometry",
    "/api/v1/regions/{region_id}/readiness",
    "/api/v1/regions/{region_id}/scenarios",  # 2026-10-14: 자료가 있는 기간 · 시나리오
    "/api/v1/wamis/stations",
    "/api/v1/regions/{region_id}/gauge/collect",
    "/api/v1/regions/{region_id}/gauge/collect/jobs",
    "/api/v1/regions/{region_id}/gauge/collect/jobs/{job_id}",
    "/api/v1/regions/{region_id}/weather/collect",
    "/api/v1/regions/{region_id}/weather/collect/jobs",
    "/api/v1/regions/{region_id}/weather/collect/jobs/{job_id}",
    "/api/v1/scenes/{scene_id}/predict",
    # 2026-10-20: 시 단위 수위 — WAMIS 시 자료 수집 잡·현황·전망
    "/api/v1/regions/{region_id}/gauge/collect-hourly",
    "/api/v1/regions/{region_id}/gauge/collect-hourly/jobs",
    "/api/v1/regions/{region_id}/gauge/collect-hourly/jobs/{job_id}",
    "/api/v1/regions/{region_id}/gauge/hourly",
    "/api/v1/regions/{region_id}/gauge/{station_id}/hourly",
    "/api/v1/regions/{region_id}/gauge/{station_id}/forecast-hourly",
    # 2026-10-20: 외생 피드 — 기상청 단기예보·댐 운영 일자료
    "/api/v1/regions/{region_id}/weather/forecast-collect",
    "/api/v1/regions/{region_id}/weather/forecast",
    "/api/v1/regions/{region_id}/dam/collect",
    "/api/v1/regions/{region_id}/dam/collect/jobs/{job_id}",
    "/api/v1/regions/{region_id}/dam",
}


def _screens_client(tmp_path: Path) -> TestClient:
    client = _client(tmp_path)
    install_screens_api(client.app)
    return client


def _poll(client: TestClient, url: str, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(url).json()
        if payload.get("status") in ("done", "failed"):
            return payload
        time.sleep(0.02)
    pytest.fail(f"job did not finish within {timeout_s}s: {payload}")


def _wamis_rows(year: int, n: int, *, base: float = 2.0) -> list[dict]:
    """합성 WAMIS 일 수위 행(date YYYYMMDD) — 계절 사인 + 작은 요동. QC 가 지우지 않는 평범한 수위."""

    import datetime as dt
    import math

    first = dt.date(year, 1, 1)
    out = []
    for i in range(n):
        day = first + dt.timedelta(days=i)
        level = base + 0.5 * math.sin(2 * math.pi * i / 365.25) + 0.03 * math.sin(i * 1.7)
        out.append({"date": day.strftime("%Y%m%d"), "water_level_m": round(level, 4), "obscd": "0000000"})
    return out


# ── install ─────────────────────────────────────────────────────────────────────


def test_install_screens_api_is_idempotent_and_registers_contract_paths(tmp_path: Path) -> None:
    client = _screens_client(tmp_path)
    app = client.app
    first = app.state.screens_api
    n_routes = len(app.routes)
    assert install_screens_api(app) is first
    assert len(app.routes) == n_routes  # 두 번 설치해도 라우터를 다시 붙이지 않는다
    assert app.state.gauge_collect_jobs is not None and app.state.weather_collect_jobs is not None
    paths = client.get("/openapi.json").json()["paths"]
    assert CONTRACT_PATHS <= set(paths)
    assert set(first["routes"]) == CONTRACT_PATHS


def test_forecasters_carry_korean_names(tmp_path: Path) -> None:
    client = _screens_client(tmp_path)
    items = client.get("/api/v1/gauge/forecasters").json()
    by_id = {i["id"]: i for i in items}
    assert all({"id", "name", "description"} <= set(i) for i in items)
    assert by_id["ridge_ar"]["name_ko"] == "릿지 자기회귀 (AR)" and by_id["persistence"]["name_ko"] == "직전값 유지 (기준선)"


# ── 1. geometry ────────────────────────────────────────────────────────────────


def test_geometry_of_toy_region_unions_stations_and_aois(tmp_path: Path) -> None:
    client = _screens_client(tmp_path)
    payload = client.get("/api/v1/regions/toy/geometry").json()
    assert payload["region_id"] == "toy" and payload["name"] == "장난감 지역"
    stations = {s["id"]: s for s in payload["stations"]}
    assert set(stations) == {"A", "B"}
    assert stations["A"]["lat"] == pytest.approx(35.2036) and stations["A"]["lon"] == pytest.approx(128.9933)
    assert stations["A"]["satellite_location"] == "gupo" and stations["A"]["terrain"] == "toy"
    assert stations["B"]["lat"] is None and stations["B"]["lon"] is None  # 좌표 없는 지점은 null — 지어내지 않는다
    assert len(payload["aois"]) == 1
    aoi = payload["aois"][0]
    assert aoi["loc_id"] == "gupo" and aoi["station_id"] == "A"
    assert len(aoi["polygon_lonlat"]) == 4 and all(len(p) == 2 for p in aoi["polygon_lonlat"])
    assert aoi["anchor_lonlat"] == pytest.approx([128.993333, 35.203611])
    assert payload["scenes"] == [] and payload["n_scenes"] == 0
    # 토이 지역에는 bbox_wgs84 가 없다 → 지점·AOI 합집합 + 여백
    minlon, minlat, maxlon, maxlat = payload["bbox_wgs84"]
    assert payload["bbox_source"] == "union"
    assert minlon < 128.9933 < maxlon and minlat < 35.2036 < maxlat
    assert all(minlon < lon < maxlon and minlat < lat < maxlat for lon, lat in aoi["polygon_lonlat"])
    assert maxlon - minlon < 0.1 and maxlat - minlat < 0.1
    assert client.get("/api/v1/regions/nope/geometry").status_code == 404


def test_geometry_projected_bbox_is_transformed_when_lonlat_missing(tmp_path: Path) -> None:
    """지역 정의에 bbox_lonlat 이 없고 EPSG:32652 상자만 있어도 경위도 다각형을 만든다(rasterio.warp)."""

    builtin = tmp_path / "builtin"
    region = write_region(builtin, region_id="proj")
    manifest_path = region / "region.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loc = manifest["satellite"]["locations"]["gupo"]
    expected = loc.pop("bbox_lonlat")
    loc.pop("anchor_lonlat")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    install_screens_api(client.app)
    aoi = client.get("/api/v1/regions/proj/geometry").json()["aois"][0]
    assert aoi["anchor_lonlat"] is None
    for got, want in zip(aoi["polygon_lonlat"], expected):
        assert got[0] == pytest.approx(want[0], abs=2e-5) and got[1] == pytest.approx(want[1], abs=2e-5)


@pytest.mark.skipif(not (REAL_REGIONS_DIR / "nakdong_busan" / "region.json").is_file(), reason="저장소 내장 지역이 없다")
def test_geometry_of_builtin_busan_region(tmp_path: Path) -> None:
    settings = Settings(result_dir=tmp_path / "results", regions_dir=REAL_REGIONS_DIR, regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    install_screens_api(client.app)
    payload = client.get("/api/v1/regions/nakdong_busan/geometry").json()
    assert payload["bbox_wgs84"] == [128.85, 35.05, 129.05, 35.35] and payload["bbox_source"] == "manifest"
    assert payload["n_stations"] == 6 and all(124 <= s["lon"] <= 132 and 33 <= s["lat"] <= 39 for s in payload["stations"])
    aois = {a["loc_id"]: a for a in payload["aois"]}
    assert set(aois) == {"gupo", "gimhae", "jeongcheon", "hupo"}
    assert aois["gupo"]["station_id"] == "구포대교" and aois["gimhae"]["station_id"] is None
    assert all(len(a["polygon_lonlat"]) == 4 for a in aois.values())
    assert payload["n_scenes"] == 8
    for scene in payload["scenes"]:
        assert len(scene["footprint_lonlat"]) == 4 and scene["detection_available"] is True
        assert scene["quicklook_url"].startswith("/api/v1/regions/nakdong_busan/satellite/files/")
        assert scene["overlay_url"].endswith("_overlay.png") and scene["water_area_km2"] > 0
        assert client.get(scene["quicklook_url"]).status_code == 200
    assert not (tmp_path / "user").exists() or not any((tmp_path / "user").iterdir())  # 읽기만 했다


# ── 2. readiness + 5. weather collect ───────────────────────────────────────────


def test_readiness_tracks_weather_collection_and_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KMA_API_KEY", raising=False)
    client = _screens_client(tmp_path)
    before = client.get("/api/v1/regions/toy/readiness").json()
    assert before["gauge"]["n_stations"] == 2 and before["gauge"]["period"] == ["2018-01-01", "2023-12-30"]  # make_series: 2190일
    assert before["gauge"]["n_days"] > 2000
    assert before["weather"] == {"available": False, "station": None, "station_name": None, "provider": None, "period": None, "n_days": 0}
    assert before["models"]["trained"] is False and before["models"]["trained_at"] is None
    assert before["satellite"]["n_scenes"] == 0 and before["ready"] is False
    assert any("기상 자료가 없습니다 — ③" in s for s in before["next_steps"])
    assert any("학습되지 않았습니다 — ④" in s for s in before["next_steps"])
    assert not any("수위 자료가 없습니다" in s for s in before["next_steps"])

    # ③ 기상 수집(합성 샘플, 동기) → 사용자 사본 weather/asos_daily.json + 매니페스트
    response = client.post(
        "/api/v1/regions/toy/weather/collect",
        json={"station_id": "133", "start": "2024-01-01", "end": "2024-02-29", "async_run": False},
    )
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "done" and job["progress"] == 1.0 and job["error"] is None
    assert job["source"] == "sample" and job["n_rows"] == 60
    assert job["years"] == [{"year": 2024, "start": "2024-01-01", "end": "2024-02-29", "n_rows": 60, "source": "sample", "status": "done", "note": None}]
    assert any("합성 샘플" in w for w in job["warnings"])
    weather_file = tmp_path / "user" / "toy" / "weather" / "asos_daily.json"
    document = json.loads(weather_file.read_text(encoding="utf-8"))
    assert document["schema"] == "watercast.weather.daily/1" and document["provider"] == PROVIDER_SAMPLE
    assert document["station_id"] == "133" and document["station_name"] == "대전" and len(document["rows"]) == 60
    assert set(document["rows"][0]) == {"date", "precipitation_mm", "avg_temperature_c", "min_temperature_c", "max_temperature_c", "humidity_pct"}
    assert document["rows"][0]["date"] == "2024-01-01" and document["missing_dates"] == []
    manifest = json.loads((tmp_path / "user" / "toy" / "region.json").read_text(encoding="utf-8"))
    assert manifest["weather"]["station_id"] == "133" and manifest["weather"]["station_name"] == "대전"
    assert manifest["weather"]["provider"] == PROVIDER_SAMPLE and manifest["weather"]["source"] == "sample"
    stored = client.get("/api/v1/regions/toy/weather").json()
    assert stored["n"] == 60 and stored["station_id"] == "133" and stored["range"] == ["2024-01-01", "2024-02-29"]
    assert client.get(f"/api/v1/regions/toy/weather/collect/jobs/{job['job_id']}").json()["status"] == "done"
    assert client.get("/api/v1/regions/toy/weather/collect/jobs").json()["total"] == 1

    # 같은 지점·같은 출처로 기간을 이어 붙이면 날짜 기준 병합
    more = client.post(
        "/api/v1/regions/toy/weather/collect",
        json={"station_id": "133", "start": "2024-02-15", "end": "2024-03-31", "async_run": False, "source": "sample"},
    ).json()
    assert more["status"] == "done" and more["n_rows_new"] == 46 and more["n_rows"] == 91
    assert more["period"] == ["2024-01-01", "2024-03-31"]

    after = client.get("/api/v1/regions/toy/readiness").json()
    assert after["weather"]["available"] is True and after["weather"]["station"] == "133"
    assert after["weather"]["period"] == ["2024-01-01", "2024-03-31"] and after["weather"]["n_days"] == 91
    assert not any("기상 자료" in s for s in after["next_steps"])

    # 학습 산출물이 생기면 trained — 지점 2개 중 1개만 산출물이 있으면 안내가 붙는다
    artifacts = tmp_path / "user" / "toy" / "models" / "gauge"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "evaluation.json").write_text(json.dumps({"generated_at": "2026-10-14T00:00:00+00:00", "split": SPLIT, "n_stations": 1, "rows": []}), encoding="utf-8")
    (artifacts / "A.json").write_text(json.dumps({"station": "A", "split": SPLIT, "horizons": {}}), encoding="utf-8")
    trained = client.get("/api/v1/regions/toy/readiness").json()
    assert trained["models"] == {
        "trained": True, "trained_at": "2026-10-14T00:00:00+00:00", "split": SPLIT, "custom_split": False,
        "artifacts_origin": "user", "n_stations_trained": 1,
    }
    assert trained["ready"] is True
    assert any("학습 산출물이 없는 지점이 1개" in s for s in trained["next_steps"])
    assert trained["next_steps"][-1].startswith("준비 완료")
    assert client.get("/api/v1/regions/nope/readiness").status_code == 404


def test_weather_collect_async_job_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KMA_API_KEY", raising=False)
    client = _screens_client(tmp_path)
    accepted = client.post(
        "/api/v1/regions/toy/weather/collect",
        json={"station_id": "159", "start": "2023-11-01", "end": "2024-01-31", "async_run": True},
    )
    assert accepted.status_code == 202, accepted.text
    body = accepted.json()
    assert body["status_url"] == f"/api/v1/regions/toy/weather/collect/jobs/{body['job_id']}" and body["source"] == "sample"
    job = _poll(client, body["status_url"])
    assert job["status"] == "done" and [y["year"] for y in job["years"]] == [2023, 2024] and job["n_rows"] == 92
    assert job["station_name"] == "부산"
    # 실측(asos)은 키가 없으면 422 — 합성으로 조용히 떨어지지 않는다
    refused = client.post("/api/v1/regions/toy/weather/collect", json={"station_id": "159", "start": "2024-01-01", "end": "2024-01-31", "source": "asos"})
    assert refused.status_code == 422 and "KMA_API_KEY" in refused.json()["detail"]
    future = client.post("/api/v1/regions/toy/weather/collect", json={"station_id": "159", "start": "2024-01-01", "end": "2999-01-31"})
    assert future.status_code == 422 and "D-1" in future.json()["detail"]
    assert client.post("/api/v1/regions/toy/weather/collect", json={"station_id": "159", "start": "2024-02-01", "end": "2024-01-31"}).status_code == 422
    assert client.post("/api/v1/regions/toy/weather/collect", json={"station_id": "없는지점", "start": "2024-01-01", "end": "2024-01-31"}).status_code == 422
    assert client.post("/api/v1/regions/nope/weather/collect", json={"station_id": "159", "start": "2024-01-01", "end": "2024-01-31"}).status_code == 404
    assert client.get("/api/v1/regions/toy/weather/collect/jobs/nope").status_code == 404


# ── 3. WAMIS stations ───────────────────────────────────────────────────────────


def test_wamis_stations_are_cached_per_basin_and_filtered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_list(basin: int) -> list[dict[str, str]]:
        calls.append(basin)
        if basin == 3:
            raise OSError("connection reset")
        return [
            {"bbsnnm": "영산강", "obscd": "5004650", "obsnm": "나주시(나주대교)", "clsyn": "운영", "obsknd": "T/M", "sbsncd": "500401", "mngorg": "환경부"},
            {"bbsnnm": "영산강", "obscd": "5001610", "obsnm": "담양군(담양댐)", "clsyn": "운영", "obsknd": "T/M", "sbsncd": "500103", "mngorg": "환경부"},
            {"bbsnnm": "영산강", "obscd": "5001000", "obsnm": "옛관측소", "clsyn": "폐쇄", "obsknd": "", "sbsncd": "", "mngorg": ""},
        ]

    monkeypatch.setattr(wamis_module, "wamis_list_basin_stations", fake_list)
    client = _screens_client(tmp_path)
    first = client.get("/api/v1/wamis/stations", params={"basin": 5}).json()
    assert first["basin"] == 5 and first["basin_name"] == "영산강" and first["n"] == 3 and first["cached"] is False
    assert first["items"][0] == {
        "code": "5004650", "name": "나주시(나주대교)", "river": "영산강", "address": None, "status": "운영", "kind": "T/M", "org": "환경부", "sub_basin": "500401",
    }
    filtered = client.get("/api/v1/wamis/stations", params={"basin": 5, "match": "나주|담양"}).json()
    assert [i["code"] for i in filtered["items"]] == ["5004650", "5001610"] and filtered["cached"] is True and filtered["n_total"] == 3
    operating = client.get("/api/v1/wamis/stations", params={"basin": 5, "operating_only": "true"}).json()
    assert [i["code"] for i in operating["items"]] == ["5004650", "5001610"]
    assert calls == [5]  # 유역 목록은 프로세스 캐시 — 한 번만 받는다
    assert client.get("/api/v1/wamis/stations", params={"basin": 5, "match": "("}).status_code == 422
    assert client.get("/api/v1/wamis/stations", params={"basin": 9}).status_code == 422
    assert client.get("/api/v1/wamis/stations").status_code == 422
    # WAMIS 가 실패하거나 빈 목록을 주면 저장소 사본(backend/app/wamis_cache)으로 받친다 — 목록 응답이 자주 잘리기 때문(2026-10-16)
    failed = client.get("/api/v1/wamis/stations", params={"basin": 3})
    assert failed.status_code == 200 and failed.json()["source"] == "cache" and failed.json()["n_total"] > 100 and "사본" in failed.json()["note_ko"]
    assert calls == [5, 3]
    monkeypatch.setattr(wamis_module, "wamis_list_basin_stations", lambda basin: [])
    empty = client.get("/api/v1/wamis/stations", params={"basin": 1})
    assert empty.status_code == 200 and empty.json()["source"] == "cache" and empty.json()["n_total"] > 100
    monkeypatch.setattr(wamis_module, "cached_basin_rows", lambda basin: [])
    none = client.get("/api/v1/wamis/stations", params={"basin": 2})
    assert none.status_code == 502 and "사본도 없" in none.json()["detail"]


# ── 4. gauge collect ────────────────────────────────────────────────────────────


def test_gauge_collect_sync_merges_rows_and_flags_low_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic = {
        "5004650": _wamis_rows(2024, 366, base=3.0),       # 전체 기간 → 커버리지 100%
        "5001610": _wamis_rows(2024, 100, base=1.0),       # 100/366 → 기준 미달
        "9999999": [],                                      # 빈 응답
        "2022680": _wamis_rows(2024, 366, base=2.0),       # 기존 지점 A(2018~2023)에 이어 붙이기
    }
    requested: list[tuple[dict, str, str]] = []

    def fake_collect(stations: dict[str, str], start: str, end: str):
        requested.append((dict(stations), start, end))
        out = {name: [dict(r, obscd=code) for r in synthetic[code]] for name, code in stations.items()}
        failures = ["옛관측소(9999999) 2024: 0행 (응답 12자)"] if "9999999" in stations.values() else []
        return out, failures

    monkeypatch.setattr(wamis_module, "wamis_collect", fake_collect)
    client = _screens_client(tmp_path)
    response = client.post(
        "/api/v1/regions/toy/gauge/collect",
        json={
            "stations": [
                {"id": "나주대교", "name": "나주대교", "code": "5004650", "terrain": "mainstem", "terrain_note": "본류 중류"},
                {"name": "담양댐", "code": "5001610", "terrain": "dam"},
                {"id": "빈지점", "code": "9999999"},
                {"id": "A", "code": "2022680"},
            ],
            "start": "2024-01-01", "end": "2024-12-31", "min_coverage": 0.85, "async_run": False,
        },
    )
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "done" and job["progress"] == 1.0 and job["error"] is None and job["current"] is None
    assert job["n_stations"] == 4 and job["n_merged"] == 3 and job["n_low_coverage"] == 1 and job["n_failed"] == 1
    assert requested == [({"나주대교": "5004650"}, "20240101", "20241231"), ({"담양댐": "5001610"}, "20240101", "20241231"),
                         ({"빈지점": "9999999"}, "20240101", "20241231"), ({"A": "2022680"}, "20240101", "20241231")]
    by_id = {s["id"]: s for s in job["stations"]}
    assert by_id["나주대교"]["status"] == "done" and by_id["나주대교"]["coverage"] == 1.0 and by_id["나주대교"]["n_days"] == 366
    assert by_id["나주대교"]["period"] == ["2024-01-01", "2024-12-31"] and by_id["나주대교"]["note"] is None
    assert by_id["담양댐"]["status"] == "low_coverage" and by_id["담양댐"]["coverage"] == pytest.approx(100 / 366, abs=1e-4)
    assert "커버리지 27.3%" in by_id["담양댐"]["note"] and "85%" in by_id["담양댐"]["note"]
    assert by_id["빈지점"]["status"] == "empty" and "수집된 행이 없습니다" in by_id["빈지점"]["note"] and by_id["빈지점"]["fetch_failures"]
    assert by_id["A"]["status"] == "done" and by_id["A"]["period_total"] == ["2018-01-01", "2024-12-31"]
    assert by_id["A"]["n_days_total"] > 366 + 2000  # 기존 2018~2023 자료 위에 이어 붙였다(replace=false)

    # 지역 시계열에는 series 업로드 엔드포인트와 같은 메타(source WAMIS·source_code·terrain)로 들어갔다
    stations = {s["id"]: s for s in client.get("/api/v1/regions/toy/gauge/stations").json()}
    assert set(stations) == {"A", "B", "나주대교", "담양댐"}  # 빈 지점은 들어가지 않는다
    assert stations["나주대교"]["source"] == "WAMIS" and stations["나주대교"]["source_code"] == "5004650"
    assert stations["나주대교"]["terrain"] == "mainstem" and stations["나주대교"]["terrain_note"] == "본류 중류"
    assert stations["담양댐"]["name"] == "담양댐" and stations["담양댐"]["terrain"] == "dam" and stations["담양댐"]["n_days"] == 100
    assert stations["A"]["terrain"] == "toy" and stations["A"]["lat"] == pytest.approx(35.2036)  # 이전 메타·좌표 유지
    series = client.get("/api/v1/regions/toy/gauge/나주대교/series").json()
    assert series["n"] == 366 and series["rows"][0] == {"date": "2024-01-01", "water_level_m": 3.0}
    saved = json.loads((tmp_path / "user" / "toy" / "gauge" / "series.json").read_text(encoding="utf-8"))
    assert next(s for s in saved["stations"] if s["id"] == "나주대교")["source"] == "WAMIS"
    listed = client.get("/api/v1/regions/toy/gauge/collect/jobs").json()
    assert listed["total"] == 1 and listed["items"][0]["job_id"] == job["job_id"]
    assert client.get(f"/api/v1/regions/toy/gauge/collect/jobs/{job['job_id']}").json()["status"] == "done"
    readiness = client.get("/api/v1/regions/toy/readiness").json()
    assert readiness["gauge"]["n_stations"] == 4 and readiness["gauge"]["period"] == ["2018-01-01", "2024-12-31"]


def test_gauge_collect_async_accepts_polls_and_refuses_concurrent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def blocking_collect(stations: dict[str, str], start: str, end: str):
        assert release.wait(10.0), "test did not release the fake WAMIS fetch"
        return {name: _wamis_rows(2024, 366) for name in stations}, []

    monkeypatch.setattr(wamis_module, "wamis_collect", blocking_collect)
    client = _screens_client(tmp_path)
    body = {"stations": [{"id": "나주대교", "code": "5004650"}, {"id": "영산포", "code": "5004660"}], "start": "2024-01-01", "end": "2024-12-31"}
    accepted = client.post("/api/v1/regions/toy/gauge/collect", json=body)
    assert accepted.status_code == 202, accepted.text
    payload = accepted.json()
    assert payload["status_url"] == f"/api/v1/regions/toy/gauge/collect/jobs/{payload['job_id']}"
    conflict = client.post("/api/v1/regions/toy/gauge/collect", json=body)
    assert conflict.status_code == 409 and "진행 중" in conflict.json()["detail"]
    running = client.get(payload["status_url"]).json()
    assert running["status"] in ("queued", "running") and running["progress"] < 1.0
    assert [s["status"] for s in running["stations"]] in (["queued", "queued"], ["running", "queued"])
    release.set()
    done = _poll(client, payload["status_url"])
    assert done["status"] == "done" and [s["status"] for s in done["stations"]] == ["done", "done"]
    assert {s["id"] for s in client.get("/api/v1/regions/toy/gauge/stations").json()} == {"A", "B", "나주대교", "영산포"}
    # 끝난 뒤에는 다시 받을 수 있다 — 이 잡도 끝까지 기다린다(테스트가 끝나 monkeypatch 가 풀린 뒤 진짜 WAMIS 를 부르지 않게)
    again = client.post("/api/v1/regions/toy/gauge/collect", json=body)
    assert again.status_code == 202
    assert _poll(client, again.json()["status_url"])["status"] == "done"
    assert client.get("/api/v1/regions/toy/gauge/collect/jobs").json()["total"] == 2


def test_gauge_collect_validation(tmp_path: Path) -> None:
    client = _screens_client(tmp_path)
    base = {"start": "2024-01-01", "end": "2024-12-31", "async_run": False}
    assert client.post("/api/v1/regions/toy/gauge/collect", json={**base, "stations": []}).status_code == 422
    assert client.post("/api/v1/regions/toy/gauge/collect", json={**base, "stations": [{"id": "x", "code": "123"}]}).status_code == 422
    assert client.post("/api/v1/regions/toy/gauge/collect", json={**base, "stations": [{"code": "1234567"}]}).status_code == 422
    assert client.post("/api/v1/regions/toy/gauge/collect", json={**base, "stations": [{"id": "x", "code": "1234567"}, {"id": "x", "code": "7654321"}]}).status_code == 422
    assert client.post("/api/v1/regions/toy/gauge/collect", json={"stations": [{"id": "x", "code": "1234567"}], "start": "2024-12-31", "end": "2024-01-01"}).status_code == 422
    assert client.post("/api/v1/regions/nope/gauge/collect", json={**base, "stations": [{"id": "x", "code": "1234567"}]}).status_code == 404
    assert client.get("/api/v1/regions/toy/gauge/collect/jobs/nope").status_code == 404


# ── 6. scene predict ────────────────────────────────────────────────────────────


def test_scene_predict_reruns_only_predict_step_with_chosen_model(tmp_path: Path) -> None:
    client, *_ = _app(tmp_path)
    install_screens_api(client.app)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    for day in ("20200301", "20200302", "20200303"):
        response = client.post(
            "/api/v1/scenes",
            data={"region_id": REGION_ID, "sensor": "planet", "image_date": day},
            files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
        )
        assert response.status_code == 202, response.text
        last = _wait(client, f"{day}_PlanetScope")
        assert last["status"] == "done", last
    scene_id = "20200303_PlanetScope"
    original = last["outputs"]["prediction_id"]
    assert original and len(last["predictions"]) == 1
    assert last["predictions"][0]["model_id"] == "irregular-area-trend" and last["predictions"][0]["horizon"] == 3  # 파이프라인 기본값 그대로

    response = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 5})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["scene_id"] == scene_id and payload["model_id"] == "persistence" and payload["horizon"] == 5
    assert payload["prediction_id"] != original and payload["prediction_url"] == f"/api/v1/predictions/{payload['prediction_id']}"
    steps = payload["summary"]["steps"]
    assert [s["horizon"] for s in steps] == [1, 2, 3, 4, 5]
    assert set(steps[0]) == {"horizon", "target_date", "water_area_km2", "area_change_pct", "risk"}
    assert steps[0]["target_date"] > "2020-03-03" and steps[0]["water_area_km2"] > 0 and steps[0]["risk"] == "normal"
    assert payload["summary"]["frame_count"] == 3 and payload["summary"]["source_dates"] == ["2020-03-01", "2020-03-02", "2020-03-03"]
    assert [p["model_id"] for p in payload["predictions"]] == ["irregular-area-trend", "persistence"]
    full = client.get(payload["prediction_url"]).json()
    assert full["status"] == "completed" and full["model_id"] == "persistence" and full["parameters"]["horizon_steps"] == 5
    assert full["input"]["source_metadata"]["scene_id"] == scene_id

    status = client.get(f"/api/v1/scenes/{scene_id}").json()
    assert status["outputs"]["prediction_id"] == payload["prediction_id"] and status["outputs"]["prediction_url"] == payload["prediction_url"]
    assert status["status"] == "done" and status["pipeline_id"] == last["pipeline_id"]
    history = status["predictions"]
    assert len(history) == 2 and history[-1] == {
        "model_id": "persistence", "horizon": 5, "prediction_id": payload["prediction_id"],
        "prediction_url": payload["prediction_url"], "created_at": history[-1]["created_at"],
    }
    predict_step = _steps(status)["predict"]
    assert predict_step["status"] == "done" and "persistence" in predict_step["detail"] and "재예측" in predict_step["detail"]
    assert _steps(status)["register"] == _steps(last)["register"]  # 다른 단계는 손대지 않는다
    folder = tmp_path / last["outputs"]["folder"]
    on_disk = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    assert on_disk["outputs"]["prediction_id"] == payload["prediction_id"] and len(on_disk["predictions"]) == 2
    assert json.loads((folder / "prediction.json").read_text(encoding="utf-8"))["prediction_id"] == payload["prediction_id"]

    # 기본값: irregular-area-trend · 7단계
    default = client.post(f"/api/v1/scenes/{scene_id}/predict", json={}).json()
    assert default["model_id"] == "irregular-area-trend" and default["horizon"] == 7 and len(default["summary"]["steps"]) == 7
    assert len(client.get(f"/api/v1/scenes/{scene_id}").json()["predictions"]) == 3

    unknown = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "no-such-model"})
    assert unknown.status_code == 422 and "모르는 예측 모델" in unknown.json()["detail"] and "persistence" in unknown.json()["detail"]
    assert client.post(f"/api/v1/scenes/{scene_id}/predict", json={"horizon": 0}).status_code == 422
    assert client.post(f"/api/v1/scenes/{scene_id}/predict", json={"horizon": 31}).status_code == 422
    assert client.post("/api/v1/scenes/nope/predict", json={}).status_code == 404

    # 센서 프레임이 3개 미만이면 422 — 파이프라인의 '최소 3개' 규칙 그대로
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": REGION_ID, "sensor": "iceye", "image_date": "20200401"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    assert _wait(client, "20200401_ICEYE")["status"] == "done"
    few = client.post("/api/v1/scenes/20200401_ICEYE/predict", json={"model_id": "persistence"})
    assert few.status_code == 422 and "최소 3개" in few.json()["detail"] and "1개" in few.json()["detail"]


def test_scene_predict_on_scene_without_pipeline_record_creates_one(tmp_path: Path) -> None:
    """파이프라인을 거치지 않은 장면(내장 자산처럼 지역 장면 기록만 있는 것)도 재예측하고 status.json 을 처음 만든다."""

    client, *_ = _app(tmp_path)
    install_screens_api(client.app)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    for day in ("2020-03-01", "2020-03-02", "2020-03-03"):
        created = client.post(
            f"/api/v1/regions/{REGION_ID}/satellite/scenes",
            data={"sensor": "PlanetScope", "sensor_type": "OPTIC", "image_date": day},
            files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
        )
        assert created.status_code == 201, created.text
    scene_id = "20200303_PlanetScope"
    synthesized = client.get(f"/api/v1/scenes/{scene_id}").json()
    assert synthesized["pipeline_id"] is None and synthesized["outputs"]["prediction_id"] is None
    assert client.get(f"/api/v1/scenes/{scene_id}/files/status.json").status_code == 404

    payload = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 2}).json()
    assert payload["summary"]["frame_count"] == 3 and [s["horizon"] for s in payload["summary"]["steps"]] == [1, 2]
    status = client.get(f"/api/v1/scenes/{scene_id}").json()
    assert status["pipeline_id"] is None and status["status"] == "done"
    assert status["outputs"]["prediction_id"] == payload["prediction_id"] and len(status["predictions"]) == 1
    assert _steps(status)["predict"]["status"] == "done" and _steps(status)["register"]["status"] == "done"
    assert any("재예측" in w for w in status["warnings"])
    status_file = tmp_path / "scenes" / REGION_ID / scene_id / "status.json"
    assert status_file.is_file() and json.loads(status_file.read_text(encoding="utf-8"))["predictions"][0]["model_id"] == "persistence"
    assert client.get(f"/api/v1/scenes/{scene_id}/files/status.json").status_code == 200
    listed = [i for i in client.get("/api/v1/scenes").json()["items"] if i["scene_id"] == scene_id]
    assert len(listed) == 1 and listed[0]["pipeline_id"] is None  # 목록에 두 번 나오지 않는다
    # 마스크 없는 장면(사진만)은 프레임이 없어 422
    photo = client.post(
        f"/api/v1/regions/{REGION_ID}/satellite/scenes",
        data={"sensor": "PlanetScope", "sensor_type": "OPTIC", "image_date": "2020-03-04"},
        files={"pre_file": ("pre.tif", pre, "image/tiff")},
    )
    assert photo.status_code == 201, photo.text
    refused = client.post("/api/v1/scenes/20200304_PlanetScope/predict", json={"model_id": "persistence"})
    assert refused.status_code == 422 and "프레임" in refused.json()["detail"]


# ── 7. scenarios(자료 기간·시나리오) · forecast 원점 보정 · predict as_of (2026-10-14) ───────


def test_scenarios_of_toy_region_lists_window_and_resolves_requested_origin(tmp_path: Path) -> None:
    client = _screens_client(tmp_path)
    sc = client.get("/api/v1/regions/toy/scenarios", params={"as_of": "2026-09-20", "horizon": 30}).json()
    assert sc["gauge"]["period"] == ["2018-01-01", "2023-12-30"] and sc["gauge"]["n_stations"] == 2
    assert all(s["model_trained"] is False for s in sc["gauge"]["stations"])
    assert sc["weather"]["available"] is False and sc["satellite"]["n_scenes"] == 0 and sc["satellite"]["available"] is False
    assert sc["models"]["trained"] is False and sc["models"]["horizon_max"] == 7 and sc["models"]["split"] == SPLIT
    assert sc["forecast_window"] == {
        "origin_min": "2018-03-02", "origin_max": "2023-12-30", "horizon_max": 7, "verifiable_until": "2023-12-30", "min_history_days": 60,
    }
    assert [s["kind"] for s in sc["scenarios"]] == ["test_start", "verify", "latest"]  # 위성 장면이 없으니 영상 날짜 항목은 없다
    by_kind = {s["kind"]: s for s in sc["scenarios"]}
    assert by_kind["test_start"]["as_of"] == "2023-01-01" and by_kind["test_start"]["in_sample"] is False and by_kind["test_start"]["observed_days_after"] == 7
    assert by_kind["verify"]["as_of"] == "2023-12-23" and by_kind["verify"]["available"] is True
    assert by_kind["latest"]["as_of"] == "2023-12-30" and by_kind["latest"]["observed_days_after"] == 0
    req = sc["requested"]
    # 자료 뒤의 날짜 → '자료 끝 H일 전'(전망 전부 비교 가능)으로 맞춘다. 화면 기본 출발일과 같은 날짜.
    assert req["clamped"] is True and req["available"] is False and req["resolved_as_of"] == "2023-12-23"
    assert req["horizon"] == 7 and req["target_date"] == "2023-12-30" and req["observed_days_after"] == 7  # 30 → 최대 예측일수 7 로 줄인다
    assert "2026-09-20 에는 수위 자료가 없어요" in req["note_ko"] and "2018-01-01 ~ 2023-12-30" in req["note_ko"]
    assert "자료 끝 7일 전인 2023-12-23" in req["note_ko"]
    assert "학습된 모델이 없어요" in sc["note_ko"]

    inside = client.get("/api/v1/regions/toy/scenarios", params={"as_of": "2023-06-01", "horizon": 5}).json()["requested"]
    assert inside == {
        "as_of": "2023-06-01", "available": True, "resolved_as_of": "2023-06-01", "clamped": False, "observed_days_after": 5,
        "in_sample": False, "horizon": 5, "target_date": "2023-06-06", "note_ko": None,
    }
    tail = client.get("/api/v1/regions/toy/scenarios", params={"as_of": "2023-12-28", "horizon": 7}).json()["requested"]
    assert tail["observed_days_after"] == 2 and tail["clamped"] is False  # 끝까지 2일만 실측
    early = client.get("/api/v1/regions/toy/scenarios", params={"as_of": "2018-01-10"}).json()["requested"]
    assert early["clamped"] is False and early["available"] is True and "전망이 비어 있을 수 있어요" in early["note_ko"]
    before = client.get("/api/v1/regions/toy/scenarios", params={"as_of": "2010-01-01"}).json()["requested"]
    assert before["clamped"] is True and before["resolved_as_of"] == "2018-03-02" and "시작되기 전" in before["note_ko"]
    plain = client.get("/api/v1/regions/toy/scenarios").json()
    assert plain["requested"] is None and plain["forecast_window"]["origin_max"] == "2023-12-30"
    assert client.get("/api/v1/regions/nope/scenarios").status_code == 404
    assert client.get("/api/v1/regions/toy/scenarios", params={"as_of": "not-a-date"}).status_code == 422
    assert client.get("/api/v1/regions/toy/scenarios", params={"horizon": 0}).status_code == 422
    assert "/api/v1/regions/{region_id}/scenarios" in client.app.state.screens_api["routes"]


@pytest.mark.skipif(not (REAL_REGIONS_DIR / "geum_daecheong" / "region.json").is_file(), reason="저장소 내장 지역이 없다")
def test_scenarios_of_builtin_daecheong_region(tmp_path: Path) -> None:
    settings = Settings(result_dir=tmp_path / "results", regions_dir=REAL_REGIONS_DIR, regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    install_screens_api(client.app)
    sc = client.get("/api/v1/regions/geum_daecheong/scenarios", params={"as_of": "2026-09-20", "horizon": 30}).json()
    assert sc["gauge"]["period"] == ["2019-01-01", "2024-12-31"] and sc["gauge"]["n_stations"] == 3
    assert all(s["model_trained"] for s in sc["gauge"]["stations"])
    assert sc["weather"]["period"] == ["2015-01-01", "2024-12-31"] and sc["weather"]["station_name"] == "대전"
    assert sc["models"]["trained"] is True and sc["models"]["horizon_max"] == 30 and sc["models"]["split"]["test_start"] == "2024-01-01"
    assert sc["satellite"]["n_scenes"] == 10 and sc["satellite"]["sensors"] == ["PlanetScope"]
    assert all(s["in_gauge_period"] and s["has_frame"] for s in sc["satellite"]["scenes"])
    assert sc["satellite"]["scenes"][0]["area_km2_common_frame"] == pytest.approx(16.337)
    assert sc["forecast_window"] == {
        "origin_min": "2019-03-02", "origin_max": "2024-12-31", "horizon_max": 30, "verifiable_until": "2024-12-31", "min_history_days": 60,
    }
    scene_items = [s for s in sc["scenarios"] if s["kind"] == "scene"]
    assert len(scene_items) == 10 and [s["as_of"] for s in scene_items][:2] == ["2019-03-23", "2019-03-29"]
    assert all(s["available"] and s["observed_days_after"] == 30 and s["in_sample"] is True for s in scene_items)
    tail = {s["kind"]: s for s in sc["scenarios"] if s["kind"] != "scene"}
    assert tail["test_start"]["as_of"] == "2024-01-01" and tail["verify"]["as_of"] == "2024-12-01" and tail["latest"]["observed_days_after"] == 0
    req = sc["requested"]
    assert req["clamped"] is True and req["resolved_as_of"] == "2024-12-01" and req["observed_days_after"] == 30
    assert "2019-01-01 ~ 2024-12-31" in req["note_ko"] and "자료 끝 30일 전인 2024-12-01" in req["note_ko"]
    assert sc["note_ko"].startswith("출발일은 2019-03-02 ~ 2024-12-31")
    ex = sc["expansion"]  # 다지역 확장 경로 판정(요청서 초안 규칙): 대청은 광학 10장·홍수기 없음·라벨 없음 → 경로 B
    assert ex["mode"] == "gauge_first" and ex["label_ko"].startswith("경로 B")
    assert ex["checks"]["scenes_per_sensor"]["value"] == {"PlanetScope": 10} and ex["checks"]["scenes_per_sensor"]["ok"] is False
    assert ex["checks"]["years"]["value"] == pytest.approx(3.17, abs=0.01) and ex["checks"]["years"]["ok"] is True
    assert ex["checks"]["flood_season"]["ok"] is False and ex["checks"]["labels_or_pairs"]["labels"] is False
    assert 0 < ex["checks"]["labels_or_pairs"]["pairs"] <= 10 and ex["checks"]["gauge"]["ok"] is True and ex["checks"]["dem"]["ok"] is True
    assert ex["reasons_ko"][0] == "센서당 장면 최대 10장 (< 36장)" and "경로 B 가동" in ex["reasons_ko"][-1]
    assert "≥ 36" in ex["rule_ko"]
    assert not (tmp_path / "user").exists() or not any((tmp_path / "user").iterdir())  # 읽기만 했다


def test_forecast_reports_requested_origin_and_rejects_origin_before_data(tmp_path: Path) -> None:
    client = _screens_client(tmp_path)
    trained = client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["A"], "forecasters": ["mean_reversion"], "horizon_max": 5})
    assert trained.status_code == 200, trained.text
    beyond = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": "2026-09-20", "horizon": 5}).json()
    assert beyond["as_of"] == "2023-12-30" and beyond["requested_as_of"] == "2026-09-20" and beyond["as_of_clamped"] is True
    assert beyond["data_period"] == ["2018-01-01", "2023-12-30"] and beyond["observed_days_after"] == 0
    assert "2026-09-20" in beyond["origin_note_ko"] and "2023-12-30" in beyond["origin_note_ko"]
    inside = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": "2023-06-01", "horizon": 5}).json()
    assert inside["as_of"] == "2023-06-01" and inside["requested_as_of"] == "2023-06-01" and inside["as_of_clamped"] is False
    assert inside["origin_note_ko"] is None and 1 <= inside["observed_days_after"] <= 5
    assert inside["observed_days_after"] == sum(1 for s in inside["steps"] if s["observed_m"] is not None)
    plain = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"horizon": 3}).json()
    assert plain["requested_as_of"] is None and plain["as_of_clamped"] is False and plain["as_of"] == "2023-12-30"
    before = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": "2010-01-01"})
    assert before.status_code == 422 and "2018-01-01 ~ 2023-12-30" in before.json()["detail"]
    # 시나리오 계약이 말한 '자료 끝 H일 전' 원점은 전망 전부가 실측과 비교된다
    sc = client.get("/api/v1/regions/toy/scenarios", params={"horizon": 5}).json()
    verify = next(s for s in sc["scenarios"] if s["kind"] == "verify")
    checked = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": verify["as_of"], "horizon": 5}).json()
    assert checked["observed_days_after"] >= 4 and checked["verification"] is not None


def test_scene_predict_with_origin_uses_frames_up_to_origin_and_reports_later_frames(tmp_path: Path) -> None:
    client, *_ = _app(tmp_path)
    install_screens_api(client.app)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    for day in ("20200301", "20200302", "20200303", "20200305"):
        response = client.post(
            "/api/v1/scenes",
            data={"region_id": REGION_ID, "sensor": "planet", "image_date": day},
            files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
        )
        assert response.status_code == 202, response.text
        assert _wait(client, f"{day}_PlanetScope")["status"] == "done"
    scene_id = "20200305_PlanetScope"
    response = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 2, "as_of": "2020-03-03"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["as_of"] == "2020-03-03" and body["summary"]["frame_count"] == 3
    assert body["summary"]["source_dates"] == ["2020-03-01", "2020-03-02", "2020-03-03"]
    later = body["summary"]["later_frames"]
    assert [f["date"] for f in later] == ["2020-03-05"] and later[0]["scene_id"] == scene_id and later[0]["water_area_km2"] is not None
    assert body["summary"]["steps"][0]["target_date"] > "2020-03-03"
    status = client.get(f"/api/v1/scenes/{scene_id}").json()
    assert status["predictions"][-1]["as_of"] == "2020-03-03" and "as_of" not in status["predictions"][0]  # 파이프라인 기본 이력 모양은 그대로
    assert "원점 2020-03-03" in _steps(status)["predict"]["detail"]
    few = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 2, "as_of": "2020-03-02"})
    assert few.status_code == 422 and "원점 2020-03-02 까지" in few.json()["detail"] and "최소 3개" in few.json()["detail"]
    whole = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 2}).json()
    assert whole["as_of"] is None and whole["summary"]["later_frames"] == [] and whole["summary"]["frame_count"] == 4
    # 계열에서 장면 빼기 — 출발 장면은 못 빼고, 모르는 id 는 무시, 너무 빼면 422
    sub = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 2, "exclude_scene_ids": ["20200302_PlanetScope", scene_id, "nope"]}).json()
    assert sub["summary"]["frame_count"] == 3 and sub["summary"]["source_dates"] == ["2020-03-01", "2020-03-03", "2020-03-05"]
    assert sub["summary"]["excluded_scene_ids"] == ["20200302_PlanetScope"]
    toofew = client.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 2, "exclude_scene_ids": ["20200301_PlanetScope", "20200302_PlanetScope"]})
    assert toofew.status_code == 422 and "제외 2개" in toofew.json()["detail"]
