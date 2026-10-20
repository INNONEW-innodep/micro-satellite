"""위성 보정 수위·수면적 조회의 기간 필터(start·end) — 3세부 연계 요구 "지점·기간을 지정한 조회"."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

REAL_REGIONS_DIR = Path(__file__).resolve().parents[2] / "regions"
REGION = "nakdong_busan"
LEVELS = f"/api/v1/regions/{REGION}/satellite/levels"
AREA = f"/api/v1/regions/{REGION}/satellite/area-series"

pytestmark = pytest.mark.skipif(
    not (REAL_REGIONS_DIR / REGION / "satellite" / "levels.json").is_file(), reason="저장소 내장 낙동강 하구 지역이 없다"
)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = Settings(result_dir=tmp_path / "results", regions_dir=REAL_REGIONS_DIR, regions_user_dir=tmp_path / "user")
    return TestClient(create_app(settings))


def _dates(records: list[dict]) -> list[str]:
    return [r["date"] for r in records]


# --- levels ----------------------------------------------------------------------------


def test_levels_without_filter_returns_everything_and_echoes_open_period(client: TestClient) -> None:
    payload = client.get(LEVELS).json()
    assert payload["n"] == len(payload["records"]) == 30
    assert payload["period"] == {"start": None, "end": None}
    assert min(_dates(payload["records"])) == "2020-02-18" and max(_dates(payload["records"])) == "2020-04-16"


def test_levels_start_only_is_inclusive(client: TestClient) -> None:
    payload = client.get(LEVELS, params={"start": "2020-04-14"}).json()
    assert payload["period"] == {"start": "2020-04-14", "end": None}
    assert payload["n"] == len(payload["records"]) == 11  # 04-14(4) + 04-15(3) + 04-16(4)
    assert all(d >= "2020-04-14" for d in _dates(payload["records"]))
    assert "2020-04-14" in _dates(payload["records"])


def test_levels_end_only_is_inclusive(client: TestClient) -> None:
    payload = client.get(LEVELS, params={"end": "2020-03-02"}).json()
    assert payload["period"] == {"start": None, "end": "2020-03-02"}
    assert payload["n"] == 8  # 02-18(4) + 03-02(4)
    assert set(_dates(payload["records"])) == {"2020-02-18", "2020-03-02"}


def test_levels_start_and_end_together(client: TestClient) -> None:
    payload = client.get(LEVELS, params={"start": "2020-03-12", "end": "2020-03-30"}).json()
    assert payload["period"] == {"start": "2020-03-12", "end": "2020-03-30"}
    assert set(_dates(payload["records"])) == {"2020-03-12", "2020-03-25", "2020-03-30"}
    assert payload["n"] == 11  # 4 + 4 + 3
    # 같은 날에 값이 하나뿐인 날짜(경계)도 포함
    same_day = client.get(LEVELS, params={"start": "2020-03-30", "end": "2020-03-30"}).json()
    assert set(_dates(same_day["records"])) == {"2020-03-30"} and same_day["n"] == 3


def test_levels_accepts_compact_yyyymmdd_and_normalises_echo(client: TestClient) -> None:
    compact = client.get(LEVELS, params={"start": "20200312", "end": "20200330"}).json()
    dashed = client.get(LEVELS, params={"start": "2020-03-12", "end": "2020-03-30"}).json()
    assert compact["period"] == {"start": "2020-03-12", "end": "2020-03-30"}
    assert compact["records"] == dashed["records"] and compact["n"] == dashed["n"] == 11


def test_levels_rejects_reversed_or_malformed_period(client: TestClient) -> None:
    reversed_ = client.get(LEVELS, params={"start": "2020-04-01", "end": "2020-03-01"})
    assert reversed_.status_code == 422 and "start" in reversed_.json()["detail"]
    assert client.get(LEVELS, params={"start": "2020/03/01"}).status_code == 422
    assert client.get(LEVELS, params={"end": "March"}).status_code == 422
    assert client.get(LEVELS, params={"start": "2020-13-01"}).status_code == 422


def test_levels_station_and_period_compose(client: TestClient) -> None:
    station_only = client.get(LEVELS, params={"station_id": "구포대교"}).json()
    assert station_only["loc_id"] == "gupo" and station_only["n"] == 8 and station_only["period"] == {"start": None, "end": None}

    both = client.get(LEVELS, params={"station_id": "구포대교", "start": "2020-03-01", "end": "2020-03-31"}).json()
    assert both["station_id"] == "구포대교" and both["loc_id"] == "gupo"
    assert both["location"] and both["location"]["gauge_station"] == "구포대교"
    assert both["period"] == {"start": "2020-03-01", "end": "2020-03-31"}
    assert all(r["loc_id"] == "gupo" for r in both["records"])
    assert set(_dates(both["records"])) == {"2020-03-02", "2020-03-12", "2020-03-25", "2020-03-30"} and both["n"] == 4
    # 대조 요약(n·rmse)은 거른 뒤의 기록으로 센다
    assert both["comparison"] is None or both["comparison"]["n"] <= both["n"]
    # 지점 미연결이면 기간과 무관하게 빈 결과이되 기간은 그대로 돌려준다
    unlinked = client.get(LEVELS, params={"station_id": "없는지점", "start": "2020-03-01"}).json()
    assert unlinked["n"] == 0 and unlinked["loc_id"] is None and unlinked["period"] == {"start": "2020-03-01", "end": None}


# --- area-series -----------------------------------------------------------------------


def test_area_series_without_filter_keeps_all_rows(client: TestClient) -> None:
    payload = client.get(AREA).json()
    assert payload["available"] is True and payload["period"] == {"start": None, "end": None}
    assert set(payload["sensors"]) == {"ICEYE", "PlanetScope"}
    for block in payload["sensors"].values():
        assert len(block["rows"]) == block["n_scenes"] == 4


def test_area_series_filters_each_sensor_by_date_and_echoes_period(client: TestClient) -> None:
    full = client.get(AREA).json()
    payload = client.get(AREA, params={"start": "2020-03-01", "end": "2020-03-31"}).json()
    assert payload["available"] is True
    assert payload["period"] == {"start": "2020-03-01", "end": "2020-03-31"}
    assert _dates(payload["sensors"]["ICEYE"]["rows"]) == ["2020-03-02", "2020-03-30"]
    assert _dates(payload["sensors"]["PlanetScope"]["rows"]) == ["2020-03-12", "2020-03-25"]
    # 요약값(상관계수·n_scenes·공통 격자 면적)은 다시 계산하지 않고 전체 기준 그대로
    for sensor in ("ICEYE", "PlanetScope"):
        for key in ("n_scenes", "correlations", "common_valid_footprint_km2", "frame"):
            assert payload["sensors"][sensor][key] == full["sensors"][sensor][key]
    for key in ("schema", "source_scenes", "water_threshold", "notes_ko"):
        assert payload[key] == full[key]

    compact = client.get(AREA, params={"start": "20200301", "end": "20200331"}).json()
    assert compact["sensors"] == payload["sensors"] and compact["period"] == payload["period"]

    start_only = client.get(AREA, params={"start": "2020-04-15"}).json()
    assert _dates(start_only["sensors"]["ICEYE"]["rows"]) == ["2020-04-15", "2020-04-16"]
    assert start_only["sensors"]["PlanetScope"]["rows"] == []

    end_only = client.get(AREA, params={"end": "20200218"}).json()
    assert end_only["sensors"]["ICEYE"]["rows"] == []
    assert _dates(end_only["sensors"]["PlanetScope"]["rows"]) == ["2020-02-18"]


def test_area_series_rejects_reversed_period(client: TestClient) -> None:
    assert client.get(AREA, params={"start": "2020-04-01", "end": "2020-03-01"}).status_code == 422
    assert client.get(AREA, params={"start": "2020.03.01"}).status_code == 422


# --- 저장 기록의 날짜 형식이 섞여 있어도 -------------------------------------------------


def test_mixed_stored_date_formats_and_dateless_rows(tmp_path: Path) -> None:
    """파일 기록이 YYYYMMDD·ISO·datetime 문자열·정수로 섞여 있어도 같은 기준으로 거르고, date 없는 기록은 기간 조회에 안 든다."""

    import json

    region = tmp_path / "builtin" / "toy"
    (region / "satellite").mkdir(parents=True)
    (region / "region.json").write_text(json.dumps({
        "schema": "watercast.region/1", "id": "toy", "name": "toy", "kind": "river", "forecast": {"horizon_max_days": 7},
    }), encoding="utf-8")
    (region / "satellite" / "levels.json").write_text(json.dumps({"records": [
        {"date": "20230101", "loc_id": "a"}, {"date": "2023-01-02", "loc_id": "a"},
        {"date": "2023-01-03T09:00:00", "loc_id": "a"}, {"date": 20230104, "loc_id": "a"},
        {"loc_id": "a"}, {"date": None, "loc_id": "a"},
    ]}), encoding="utf-8")
    (region / "satellite" / "area_series.json").write_text(json.dumps({"schema": "x", "region_id": "toy", "sensors": {
        "S1": {"n_scenes": 3, "rows": [{"date": "20230101"}, {"date": "2023-01-02"}, {"date": "2023-01-03"}],
               "scenes": [{"date": "2023-01-01"}, {"date": "2023-01-03"}]},
        "S2": "not-a-block",
    }}), encoding="utf-8")
    settings = Settings(result_dir=tmp_path / "results", regions_dir=tmp_path / "builtin", regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    levels, area = "/api/v1/regions/toy/satellite/levels", "/api/v1/regions/toy/satellite/area-series"

    assert client.get(levels).json()["n"] == 6  # 필터 없으면 date 없는 기록도 그대로
    assert _dates(client.get(levels, params={"start": "2023-01-02"}).json()["records"]) == ["2023-01-02", "2023-01-03T09:00:00", 20230104]
    assert _dates(client.get(levels, params={"end": "2023-01-02"}).json()["records"]) == ["2023-01-02", "20230101"]
    assert _dates(client.get(levels, params={"start": "20230103", "end": "2023-01-04"}).json()["records"]) == ["2023-01-03T09:00:00", 20230104]

    payload = client.get(area, params={"start": "2023-01-02"}).json()
    assert _dates(payload["sensors"]["S1"]["rows"]) == ["2023-01-02", "2023-01-03"]
    assert _dates(payload["sensors"]["S1"]["scenes"]) == ["2023-01-03"]
    assert payload["sensors"]["S1"]["n_scenes"] == 3 and payload["sensors"]["S2"] == "not-a-block"
