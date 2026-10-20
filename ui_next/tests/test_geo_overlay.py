"""위성 수위 산출 구역(AOI) ↔ 수위계 위치 그리기(ui_next/geo_overlay.py) — 순수 도우미와 슬라이드 3 AppTest.

픽스처는 백엔드/데이터 계약(region.json satellite.locations · scenes.json quicklook · stations lat/lon)을 그대로 따른 값이다.
부산 gupo 상자와 구포대교 좌표는 인수 패키지 AOI 정본(aoi_busan.geojson)에서 가져왔다.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from ui_next.geo_overlay import (
    auto_zoom,
    bbox_to_pixel_rect,
    distance_label,
    link_stations,
    location_table_fields,
    lonlat_offset_m,
    map_figure,
    overlay_figure,
    short_note,
    station_distance_m,
    station_xy,
    to_pixel,
)

UI_DIR = Path(__file__).resolve().parents[1]

# 부산 ICEYE 2020-03-02 퀵룩(787×1000) — 원본 Pre GeoTIFF 의 EPSG:32652 범위
QUICKLOOK = {
    "crs": "EPSG:32652",
    "bounds_epsg32652": [473010.36, 3847364.66, 531723.36, 3921935.66],
    "width": 787,
    "height": 1000,
    "footprint_lonlat": [[128.702627, 35.440504], [129.349529, 35.440364], [129.346670, 34.767949], [128.705059, 34.768085]],
}

LOCATIONS = {
    "gupo": {
        "name_ko": "구포대교 부근 본류 하도", "gauge_station": "구포대교", "obscd": "2022680", "river": "낙동강", "crs": "EPSG:32652",
        "bbox_epsg32652": [499186.885, 3895504.01, 499486.885, 3895804.01],
        "bbox_lonlat": [[128.991067, 35.205242], [128.994363, 35.205242], [128.994363, 35.202537], [128.991067, 35.202537]],
        "center_lonlat": [128.992715, 35.203889], "anchor_lonlat": [128.993333, 35.203611], "snap_distance_m": 63.8,
        "note_ko": "OK", "source": "aoi_busan.geojson",
    },
    "jeongcheon": {
        "name_ko": "정천교 부근 조만강", "gauge_station": "정천교", "obscd": "2022685", "river": "조만강", "crs": "EPSG:32652",
        "bbox_epsg32652": [484750.885, 3895936.01, 485050.885, 3896236.01],
        "bbox_lonlat": [[128.832462, 35.209022], [128.835758, 35.209026], [128.835764, 35.206321], [128.832468, 35.206316]],
        "center_lonlat": [128.834113, 35.207671], "anchor_lonlat": [128.833611, 35.205556], "snap_distance_m": 240.0,
        "note_ko": "⚠️ 요건 미충족", "source": "aoi_busan.geojson",
    },
    "gimhae": {
        "name_ko": "김해교 하류 본류 하도", "gauge_station": None, "obscd": "2022640", "river": "낙동강", "crs": "EPSG:32652",
        "bbox_epsg32652": [500188.885, 3904672.01, 500488.885, 3904972.01],
        "bbox_lonlat": [[129.002077, 35.287911], [129.005376, 35.28791], [129.005376, 35.285205], [129.002077, 35.285205]],
        "center_lonlat": [129.003727, 35.286558], "anchor_lonlat": [128.994167, 35.289444], "snap_distance_m": 930.5,
        "note_ko": "재설정 2026-08-18 — 구 AOI(관측소 직상)는 서측 급경사 구릉·교량부 혼재로 경계표고 std 0.50~0.93 m. 신 AOI=하류 930 m 본류 하도",
        "source": "aoi_busan.geojson",
    },
}

STATIONS = [
    {"id": "구포대교", "name": "구포대교", "source": "WAMIS", "source_code": "2022680", "terrain": "estuary", "terrain_note": "하구부 본류·감조",
     "unit": "m", "n_days": 3653, "period": ["2015-01-01", "2024-12-31"], "satellite_location": "gupo", "model_trained": True, "qc": None,
     "lat": 35.203611, "lon": 128.993333, "address": "부산광역시 북구 구포동", "river": "낙동강"},
    {"id": "정천교", "name": "정천교", "source": "WAMIS", "source_code": "2022685", "terrain": "tributary", "terrain_note": "조만강 지류",
     "unit": "m", "n_days": 3000, "period": ["2015-01-01", "2024-12-31"], "satellite_location": "jeongcheon", "model_trained": True, "qc": None,
     "lat": 35.205556, "lon": 128.833611, "address": "경상남도 김해시", "river": "조만강"},
    {"id": "낙동강하구언(내)", "name": "낙동강하구언(내)", "source": "WAMIS", "source_code": "2022696", "terrain": "estuary",
     "unit": "m", "n_days": 3246, "period": ["2015-01-01", "2024-12-31"], "satellite_location": None, "model_trained": True, "qc": None,
     "lat": 35.106, "lon": 128.962, "address": "부산광역시 사하구", "river": "낙동강"},
    {"id": "좌표없음", "name": "좌표없음", "source": "WAMIS", "source_code": "0000000", "terrain": "estuary",
     "unit": "m", "n_days": 100, "period": ["2015-01-01", "2024-12-31"], "satellite_location": None, "model_trained": False, "qc": None,
     "lat": None, "lon": None, "address": None, "river": None},
]

REGION = {
    "id": "nakdong_busan", "name": "낙동강 하구 (부산·김해)", "description": "픽스처", "crs": "EPSG:32652",
    "bbox_wgs84": [128.85, 35.05, 129.05, 35.35], "weather": {"provider": "KMA_ASOS", "station_id": "159", "station_name": "부산"},
    "satellite_available": True, "scenes_available": True, "gt_labels_available": True, "satellite_levels_available": True,
    "gauge_available": True, "n_stations": 4, "n_stations_trained": 3,
    "satellite": {"available": True, "sensors": ["ICEYE"], "scenes": {"ICEYE": ["2020-03-02"]}},
    "satellite_locations": LOCATIONS,
    "stations": STATIONS,
    "forecast": {"split": {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}},
}


def _png(width: int = 787, height: int = 1000) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (20, 30, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# --- 좌표 변환 ---------------------------------------------------------------------------------


def test_to_pixel_maps_corners_and_center() -> None:
    bounds = QUICKLOOK["bounds_epsg32652"]
    w, h = QUICKLOOK["width"], QUICKLOOK["height"]
    assert to_pixel(bounds[0], bounds[3], bounds, w, h) == (0.0, 0.0)  # 왼쪽 위
    assert to_pixel(bounds[2], bounds[1], bounds, w, h) == (w, h)  # 오른쪽 아래
    px, py = to_pixel((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, bounds, w, h)
    assert abs(px - w / 2) < 1e-9 and abs(py - h / 2) < 1e-9
    # y 는 아래로 커진다: 북쪽(큰 y)일수록 작은 py
    _, north = to_pixel(bounds[0], bounds[3] - 1000, bounds, w, h)
    _, south = to_pixel(bounds[0], bounds[1] + 1000, bounds, w, h)
    assert north < south
    with pytest.raises(ValueError):
        to_pixel(0, 0, [0, 0, 0, 0], w, h)


def test_bbox_to_pixel_rect_is_ordered_and_tiny_on_full_scene() -> None:
    bounds = QUICKLOOK["bounds_epsg32652"]
    x0, y0, x1, y1 = bbox_to_pixel_rect(LOCATIONS["gupo"]["bbox_epsg32652"], bounds, 787, 1000)
    assert 0 < x0 < x1 < 787 and 0 < y0 < y1 < 1000
    # 300 m 상자 ÷ (58.7 km / 787 px ≈ 74.6 m/px) ≈ 4 px
    assert 3.5 < (x1 - x0) < 4.6 and 3.5 < (y1 - y0) < 4.6
    # 픽셀 좌표 → 미터로 되돌리면 상자 중심과 맞는다
    cx = bounds[0] + (x0 + x1) / 2 / 787 * (bounds[2] - bounds[0])
    assert abs(cx - 499336.885) < 1.0


def test_station_offset_matches_snap_distance() -> None:
    # 구포대교 anchor [128.993333, 35.203611] vs gupo 상자 중심 [499336.885, 3895654.01] → 정본 snap 63.8 m
    location = LOCATIONS["gupo"]
    distance = station_distance_m(location, STATIONS[0])
    assert distance is not None and 40 <= distance <= 120
    assert abs(distance - 63.8) < 3.0
    xy = station_xy(location, STATIONS[0])
    assert xy is not None and abs(xy[0] - 499336.885) < 120 and abs(xy[1] - 3895654.01) < 120
    # 지점 좌표가 없으면 anchor_lonlat 로, 그것도 없으면 snap_distance_m 로 떨어진다
    assert abs(station_distance_m(location, {"id": "구포대교"}) - distance) < 1e-6
    bare = {k: v for k, v in location.items() if k != "anchor_lonlat"}
    assert station_distance_m(bare, None) == 63.8
    assert station_xy(bare, None) is None
    dx, dy = lonlat_offset_m([128.993333, 35.203611], [128.992715, 35.203889])
    assert 50 < dx < 62 and -35 < dy < -27
    assert math.isclose(math.hypot(dx, dy), distance)


def test_labels_and_links() -> None:
    assert distance_label(63.8) == "64 m" and distance_label(1234.0) == "1.2 km" and distance_label(None) == "–"
    assert short_note(LOCATIONS["gimhae"]["note_ko"]) == "재설정 2026-08-18"
    assert short_note("") == "–"
    assert len(short_note("가" * 80)) == 40
    linked = link_stations(LOCATIONS, STATIONS)
    assert linked["gupo"]["id"] == "구포대교" and linked["jeongcheon"]["id"] == "정천교" and linked["gimhae"] is None
    # 지점 쪽 satellite_location 이 없어도 구역의 gauge_station 으로 이어 준다
    linked2 = link_stations(LOCATIONS, [dict(STATIONS[0], satellite_location=None)])
    assert linked2["gupo"]["id"] == "구포대교"
    fields = location_table_fields(dict(LOCATIONS["gupo"], loc_id="gupo"), STATIONS[0])
    assert fields == {"위성 산출 구역": "gupo · 구포대교 부근 본류 하도", "수위계까지 거리": "64 m", "구역 설정 메모": "OK"}
    assert location_table_fields(None, None)["위성 산출 구역"] == "–"
    assert 9 < auto_zoom([128.83, 129.01], [35.2, 35.29]) < 12
    assert auto_zoom([129.0], [35.2]) == 12.0


# --- 그림 ------------------------------------------------------------------------------------


def test_overlay_figure_draws_boxes_stations_and_distance() -> None:
    figure = overlay_figure(_png(), QUICKLOOK, LOCATIONS, STATIONS, title="t")
    assert len(figure.layout.images) == 1 and figure.layout.images[0].source.startswith("data:image/png;base64,")
    assert len(figure.layout.shapes) == 3  # 상자 3개
    ring = next(t for t in figure.data if t.name == "위성 산출 구역")
    assert list(ring.text) == ["gupo · 구포대교 부근 본류 하도", "jeongcheon · 정천교 부근 조만강", "gimhae · 김해교 하류 본류 하도"]
    station_traces = [t for t in figure.data if t.mode == "markers+text" and t.text and "수위계" in t.text[0]]
    # gimhae 는 공개 수위계가 없어 AOI 를 스냅한 기준 지점만 속 빈 마커로 표시한다
    assert [t.text[0] for t in station_traces] == ["구포대교 · 수위계", "정천교 · 수위계", "기준점 · 수위계 자료 없음"]
    assert "부산광역시 북구 구포동" in station_traces[0].hovertemplate and station_traces[2].marker.color == "rgba(0,0,0,0)"
    dashed = [t for t in figure.data if t.mode == "lines"]
    assert len(dashed) == 3 and dashed[0].line.dash == "dash"
    labels = [a.text for a in figure.layout.annotations]
    # 등장방형 근사 거리 라벨: 정본 snap 63.8 / 240.0 / 930.5 m 와 1 % 안에서 맞는다
    assert "64 m" in labels and any(l in labels for l in ("238 m", "239 m", "240 m", "241 m")) and any(l in labels for l in ("924 m", "925 m", "926 m", "0.9 km"))
    assert list(figure.layout.yaxis.range) == [1000, 0] and figure.layout.yaxis.scaleanchor == "x"
    # 확대: 구역 주변만 보이도록 축 범위가 잘린다
    zoomed = overlay_figure(_png(), QUICKLOOK, LOCATIONS, STATIONS, zoom_to_locations=True)
    x0, x1 = zoomed.layout.xaxis.range
    y1, y0 = zoomed.layout.yaxis.range
    assert 0 < x0 < x1 < 787 and 0 < y0 < y1 < 1000 and (x1 - x0) >= 240
    # 이미지가 없어도(다운로드 실패) 상자·수위계는 그린다
    assert len(overlay_figure(None, QUICKLOOK, LOCATIONS, STATIONS).layout.images) == 0
    with pytest.raises(ValueError):
        overlay_figure(_png(), {"width": 0, "height": 0}, LOCATIONS, STATIONS)


def test_overlay_figure_skips_boxes_outside_the_scene() -> None:
    far = {"daecheong": dict(LOCATIONS["gupo"], bbox_epsg32652=[364365.0, 4038324.0, 364515.0, 4038474.0], center_lonlat=[127.486714, 36.481306], anchor_lonlat=None)}
    figure = overlay_figure(_png(), QUICKLOOK, far, [])
    assert len(figure.layout.shapes) == 0 and all(t.name != "위성 산출 구역" for t in figure.data)


def test_map_figure_builds_for_satellite_and_gauge_only_regions() -> None:
    figure = map_figure(REGION, LOCATIONS, STATIONS, QUICKLOOK["footprint_lonlat"], title="지도")
    kinds = [t.name for t in figure.data]
    assert "위성 장면 범위" in kinds and "지상 수위계" in kinds
    assert sum(1 for t in figure.data if t.fill == "toself" and t.name != "위성 장면 범위") == 3
    gauges = next(t for t in figure.data if t.name == "지상 수위계")
    assert list(gauges.text) == ["구포대교", "정천교", "낙동강하구언(내)"]  # 좌표 없는 지점은 빠지고 각주로 남는다
    assert gauges.customdata[0][0] == "부산광역시 북구 구포동" and "gupo" in gauges.customdata[0][3]
    assert any("좌표없음" in (a.text or "") for a in figure.layout.annotations)
    assert figure.layout.map.style == "carto-darkmatter" and 8 < figure.layout.map.zoom < 13
    # 수위계만 있는 지역: 상자 없이 점만
    gauge_only = map_figure({"id": "g", "bbox_wgs84": [126.8, 35.0, 127.4, 35.4]}, {}, STATIONS[2:3])
    assert [t.name for t in gauge_only.data] == ["지상 수위계"]
    # 아무 좌표도 없으면 지역 bbox 중심으로 빈 지도
    empty = map_figure({"id": "g", "bbox_wgs84": [126.8, 35.0, 127.4, 35.4]}, {}, [])
    assert len(empty.data) == 0 and abs(empty.layout.map.center.lon - 127.1) < 1e-9


# --- AppTest: 슬라이드 3 ---------------------------------------------------------------------------


def _slide3_app(ui_dir: str, fixture: dict, with_locations: bool) -> None:  # AppTest.from_function 이 본문을 스크립트로 실행한다
    import io as _io
    import sys

    import streamlit as st
    from PIL import Image as _Image

    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    from slides import render_slides

    region = dict(fixture["region"])
    if not with_locations:
        region["satellite_locations"] = {}
    scene = fixture["scene"]

    def png() -> bytes:
        buffer = _io.BytesIO()
        _Image.new("RGB", (787, 1000), (20, 30, 40)).save(buffer, format="PNG")
        return buffer.getvalue()

    class FakeClient:
        def list_regions(self):
            return [region]

        def get_region(self, rid):
            return region

        def satellite_scenes(self, rid):
            return {"region_id": rid, "available": True, "sensors": ["ICEYE"], "n_scenes": 1, "n_with_masks": 1, "scenes": [scene]}

        def fetch_bytes(self, url):
            return png()

        def satellite_levels(self, rid, station_id=None):
            loc_id = next((s["satellite_location"] for s in region["stations"] if s["id"] == station_id), None)
            return {"region_id": rid, "station_id": station_id, "loc_id": loc_id, "n": 1,
                    "location": (region.get("satellite_locations") or {}).get(loc_id),
                    "records": [{"date": "2020-03-02", "satellite": "ICEYE", "water_level_m": 1.0, "corrected_water_level_m": 1.2, "gauge_level_m": 1.1, "corrected_minus_gauge_m": 0.1}],
                    "comparison": {"n": 1, "rmse_m": 0.1, "mae_m": 0.1}}

        def segmenters(self):
            return {"items": []}

        def wbms_status(self):
            return {"available": False}

        def __getattr__(self, name):  # 그 밖의 호출은 빈 값
            return lambda *a, **k: {}

    st.session_state.slide = 2
    st.session_state.slide_region = region["id"]
    render_slides(client_factory=lambda: FakeClient(), execute_quick_sample=lambda *a, **k: None)


SCENE = {
    "scene_id": "20200302_ICEYE", "date": "2020-03-02", "sensor": "ICEYE", "sensor_type": "SAR",
    "images": {"input": "quicklooks/20200302_ICEYE_input.png", "overlay": "quicklooks/20200302_ICEYE_overlay.png"},
    "image_urls": {"input": "/api/v1/regions/nakdong_busan/satellite/files/quicklooks/20200302_ICEYE_input.png",
                   "overlay": "/api/v1/regions/nakdong_busan/satellite/files/quicklooks/20200302_ICEYE_overlay.png"},
    "detection_available": True, "water_area_km2": 40.1453, "gt_available": False, "metrics": {},
    "frame": {"width": 469, "height": 640, "bounds_epsg32652": [479038.7, 3847364.7, 528783.6, 3915252.0]},
    "quicklook": QUICKLOOK,
}


def _run_slide3(with_locations: bool) -> AppTest:
    app = AppTest.from_function(_slide3_app, kwargs={"ui_dir": str(UI_DIR), "fixture": {"region": REGION, "scene": SCENE}, "with_locations": with_locations}, default_timeout=60)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    return app


def test_apptest_slide3_shows_location_overlay_and_extended_table() -> None:
    app = _run_slide3(True)
    markdown = "\n".join(m.value for m in app.markdown)
    assert "어디를 잰 것인가 — 위성 수위 산출 구역 ↔ 수위계 위치" in markdown
    assert "구역 주변만 확대" in [t.label for t in app.toggle]
    # 대조표에 새 열이 붙었다
    tables = [t.value for t in app.table]
    joined = "\n".join(str(t.columns.tolist()) for t in tables)
    assert "위성 산출 구역" in joined and "수위계까지 거리" in joined and "구역 설정 메모" in joined
    compare = next(t for t in tables if "수위계까지 거리" in t.columns)
    assert compare["수위계까지 거리"].tolist()[0] == "64 m" and compare["수위계까지 거리"].tolist()[1] in ("238 m", "239 m", "240 m", "241 m")
    assert compare["위성 산출 구역"].tolist() == ["gupo · 구포대교 부근 본류 하도", "jeongcheon · 정천교 부근 조만강"]


def test_apptest_slide3_says_when_locations_are_missing() -> None:
    app = _run_slide3(False)
    markdown = "\n".join(m.value for m in app.markdown)
    assert "어디를 잰 것인가" in markdown
    assert "위성 수위 산출 구역(AOI)</b>이 등록되어 있지 않습니다" in markdown
    assert not app.toggle
