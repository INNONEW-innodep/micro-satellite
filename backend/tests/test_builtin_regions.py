"""저장소에 커밋된 내장 지역(regions/<id>)이 레지스트리 규약을 지키는지 검사한다.

새 지역을 디렉터리로 추가할 때(예: WAMIS 수위계만 있는 gauge-only 지역) 파일 형식이
어긋나면 화면에서 조용히 빠지므로, 여기서 잡는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.gauge.regions import REGION_SCHEMA, SERIES_SCHEMA, RegionRegistry

ROOT = Path(__file__).resolve().parents[2]
REGIONS_DIR = ROOT / "regions"
GAUGE_ONLY = ["han_river", "yeongsan_river", "seomjin_river", "geum_mainstem", "nakdong_upper"]


def _registry(tmp_path: Path) -> RegionRegistry:
    return RegionRegistry(REGIONS_DIR, tmp_path / "user")


def test_builtin_registry_lists_expected_regions(tmp_path: Path) -> None:
    ids = {record.id for record in _registry(tmp_path).list()}
    assert {"nakdong_busan", "geum_daecheong", *GAUGE_ONLY} <= ids


@pytest.mark.parametrize("region_id", GAUGE_ONLY)
def test_gauge_only_region_manifest_and_files(region_id: str, tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    record = registry.get(region_id)
    manifest = record.manifest
    assert manifest["schema"] == REGION_SCHEMA
    assert manifest["satellite"]["available"] is False
    assert manifest["gauge"]["available"] is True
    assert record.gauge_available
    assert manifest["weather"]["provider"] == "KMA_ASOS" and manifest["weather"]["station_id"]
    assert len(manifest["bbox_wgs84"]) == 4
    split = record.split
    assert split["train_end"] < split["val_start"] <= split["val_end"] < split["test_start"] <= split["test_end"]

    payload = json.loads(record.series_path.read_text(encoding="utf-8"))
    assert payload["schema"] == SERIES_SCHEMA
    declared = {s["id"] for s in payload["stations"]}
    assert declared == set(payload["series"]) and len(declared) >= 2
    for station in payload["stations"]:
        assert station["source"] == "WAMIS" and station["source_code"]
        assert station["terrain"] and station["terrain_note"], station["id"]

    series_set = registry.load_series(region_id)
    assert series_set.meta["weather_available"] is True
    weather = registry.load_weather_rows(region_id)
    assert weather[0]["date"] <= split["train_end"] and weather[-1]["date"] >= split["test_end"]
    for station in series_set.stations:
        daily = series_set.series[station.id]
        assert "rain_mm" in daily.exog and "temp_c" in daily.exog
    # 수집 기준: 2015~2024 원자료 일 커버리지 85% 이상 (fetch_wamis_levels.py --min-coverage 규칙).
    # 레지스트리 QC(평탄 구간·스파이크 제거)로 줄어든 n_days 는 여기서 보지 않는다.
    for station in payload["stations"]:
        assert len(payload["series"][station["id"]]) >= 0.85 * 3653, (region_id, station["id"])


@pytest.mark.parametrize("region_id", GAUGE_ONLY)
def test_gauge_only_region_has_evaluation_artifacts(region_id: str, tmp_path: Path) -> None:
    record = _registry(tmp_path).get(region_id)
    evaluation_path = record.artifacts_dir / "evaluation.json"
    assert evaluation_path.is_file(), f"{region_id}: scripts/train_gauge_models.py 미실행"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["rows"] and evaluation["model_summary"]
    assert "cells_beating_persistence" in evaluation
    stations = {s["id"] for s in json.loads(record.series_path.read_text(encoding="utf-8"))["stations"]}
    assert {row["station_id"] for row in evaluation["rows"]} == stations
    for station_id in stations:
        assert (record.artifacts_dir / f"{station_id}.json").is_file()


ALL_BUILTIN = ["nakdong_busan", "geum_daecheong", *GAUGE_ONLY]


@pytest.mark.parametrize("region_id", ALL_BUILTIN)
def test_station_coordinates_lie_inside_region_bbox(region_id: str, tmp_path: Path) -> None:
    """WAMIS 지점 좌표(fetch_wamis_station_info.py)가 지역 bbox_wgs84(±0.1°) 안에 있어야 지도가 지역을 벗어나지 않는다."""

    record = _registry(tmp_path).get(region_id)
    min_lon, min_lat, max_lon, max_lat = record.manifest["bbox_wgs84"]
    payload = json.loads(record.series_path.read_text(encoding="utf-8"))
    for station in payload["stations"]:
        lon, lat = station.get("lon"), station.get("lat")
        assert isinstance(lon, float) and isinstance(lat, float), (region_id, station["id"], "좌표 없음")
        assert min_lon - 0.1 <= lon <= max_lon + 0.1 and min_lat - 0.1 <= lat <= max_lat + 0.1, (region_id, station["id"], lon, lat)


@pytest.mark.parametrize("region_id", ["nakdong_busan", "geum_daecheong"])
def test_satellite_locations_sit_on_their_gauge_anchor(region_id: str, tmp_path: Path) -> None:
    """AOI 중심은 스냅 원점(anchor_lonlat)에서 1.2 km 안(snap_distance_m 와 일치)이고, bbox_lonlat 은 bbox_epsg32652 의 재투영이다."""

    pytest.importorskip("rasterio")
    from rasterio.warp import transform

    record = _registry(tmp_path).get(region_id)
    stations = {s["id"]: s for s in json.loads(record.series_path.read_text(encoding="utf-8"))["stations"]}
    links = record.manifest["gauge"].get("satellite_location_links") or {}
    for loc_id, location in record.satellite_locations().items():
        minx, miny, maxx, maxy = location["bbox_epsg32652"]
        lons, lats = transform("EPSG:32652", "EPSG:4326", [minx, maxx, maxx, minx], [maxy, maxy, miny, miny])
        for (lon, lat), corner in zip(location["bbox_lonlat"], zip(lons, lats)):
            assert abs(lon - corner[0]) < 1e-5 and abs(lat - corner[1]) < 1e-5, (loc_id, "bbox_lonlat 불일치")
        anchor = location["anchor_lonlat"]
        assert anchor, loc_id
        xs, ys = transform("EPSG:4326", "EPSG:32652", [location["center_lonlat"][0], anchor[0]], [location["center_lonlat"][1], anchor[1]])
        distance = ((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2) ** 0.5
        assert distance <= 1200.0, (loc_id, distance)
        if location.get("snap_distance_m") is not None:
            assert abs(distance - location["snap_distance_m"]) < 5.0, (loc_id, distance, location["snap_distance_m"])
        gauge = location.get("gauge_station")
        if gauge is not None:
            assert links.get(gauge) == loc_id, (loc_id, gauge)
            station = stations[gauge]
            xs, ys = transform("EPSG:4326", "EPSG:32652", [station["lon"], anchor[0]], [station["lat"], anchor[1]])
            assert ((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2) ** 0.5 < 5.0, (loc_id, "수위계 좌표 ≠ anchor")


def test_reset_removes_user_copy_left_by_retraining(tmp_path: Path) -> None:
    """재학습이 만든 사용자 사본(region.json·series·weather·models 사본 + 새 산출물)은 되돌리기로 통째로 걷힌다.

    사본의 models/gauge/experiments/ 는 내장에서 그대로 복사된 디렉터리라 '사용자가 더한 것'이 아니다 —
    이것 때문에 사본이 남아 origin=user 로 내장 지역을 계속 가리던 결함(2026-11-08)의 회귀 테스트.
    """

    registry = _registry(tmp_path)
    record = registry.get("nakdong_busan")
    user_region = registry._materialize_user_copy(record)
    user_models = user_region / "models" / "gauge"
    assert (user_models / "experiments").is_dir()  # 내장 experiments 가 사본에 복사돼 있다
    (user_models / "구포대교.json").write_text("{}", encoding="utf-8")  # 재학습 산출물 흉내
    (user_models / "evaluation.json").write_text("{}", encoding="utf-8")

    removed = registry.reset_user_artifacts("nakdong_busan")

    assert "구포대교.json" in removed and "evaluation.json" in removed
    assert not user_region.exists(), "내장 복사본뿐인 사본은 전체가 제거되어야 한다"
    assert RegionRegistry(REGIONS_DIR, tmp_path / "user").get("nakdong_busan").origin == "builtin"


def test_reset_keeps_user_copy_that_has_user_scenes(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    user_region = registry._materialize_user_copy(registry.get("nakdong_busan"))
    (user_region / "satellite").mkdir()
    (user_region / "satellite" / "scenes.json").write_text('{"scenes": []}', encoding="utf-8")
    (user_region / "models" / "gauge" / "evaluation.json").write_text("{}", encoding="utf-8")

    registry.reset_user_artifacts("nakdong_busan")

    assert user_region.is_dir() and (user_region / "satellite" / "scenes.json").is_file()
