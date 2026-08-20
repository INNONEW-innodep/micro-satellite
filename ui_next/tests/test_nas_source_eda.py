from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import FrozenInstanceError

import plotly.graph_objects as go
import pytest

from ui_next.nas_source_eda import (
    NAS_SOURCE_PROFILE_PATH,
    available_source_group_ids,
    build_iceye_footprint_map,
    build_nas_source_eda,
    build_nas_source_figure,
    load_nas_source_profile,
    source_metric_rows,
    source_table_rows,
)


def _raw_profile() -> dict[str, object]:
    return json.loads(NAS_SOURCE_PROFILE_PATH.read_text(encoding="utf-8"))


def _write_profile(tmp_path, payload: object):
    target = tmp_path / "profile.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def test_profile_loader_validates_and_freezes_the_sanitized_audit() -> None:
    profile = load_nas_source_profile()

    assert profile.schema_version == "1.0"
    assert profile.source_label == "NAS 읽기 전용 원본 감사"
    assert profile.audited_at_utc == "2026-08-10T04:41:02Z"
    assert tuple(profile.groups) == (
        "planetscope_raw",
        "iceye_raw",
        "university_single_date",
        "giheung_worldview",
        "hoedong_worldview",
    )
    assert available_source_group_ids() == (
        "planetscope-raw",
        "iceye-raw",
        "university-single-date",
        "giheung-optical",
        "hoedong-optical",
    )

    with pytest.raises(TypeError):
        profile.groups["new"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.groups["planetscope_raw"]["logical_root"] = "/tmp"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        profile.profile_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["groups"]["iceye_raw"].update(
                {"debug_note": "/mnt/private/source.tif"}
            ),
            "absolute path",
        ),
        (
            lambda payload: payload["groups"]["iceye_raw"].update(
                {"debug_note": "smb://nas/share"}
            ),
            "connection or credential",
        ),
        (
            lambda payload: payload["groups"]["iceye_raw"].update(
                {"password": "should-not-exist"}
            ),
            "connection or credential",
        ),
        (
            lambda payload: payload["audit"].update({"credentials_included": True}),
            "credentials_included=false",
        ),
    ),
)
def test_profile_loader_rejects_paths_connections_and_credentials(
    tmp_path,
    mutate,
    message: str,
) -> None:
    payload = copy.deepcopy(_raw_profile())
    mutate(payload)
    changed = _write_profile(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_nas_source_profile(changed)


def test_planetscope_source_eda_exposes_actual_udm2_quality_not_water_labels() -> None:
    view = build_nas_source_eda("planetscope-raw")

    assert view.profile_group_key == "planetscope_raw"
    assert len(view.metric_rows) == 4
    assert {row["label"]: row["value"] for row in view.metric_rows} == {
        "고유 관측일": "4일",
        "원본 Scene": "8개",
        "UDM2 usable": "68.0~69.3%",
        "유효영역 Cloud": "최대 0.01051%",
    }
    assert len(view.table_rows) == 8
    assert view.table_rows[0]["크기 (H×W)"] == "7572 × 9844"
    assert view.table_rows[0]["UDM2 usable (%)"] == pytest.approx(69.082)
    assert view.table_rows[-1]["usable 중 cloud (%)"] == 0.0
    assert [trace.name for trace in view.figure.data] == [
        "usable 영역",
        "usable 중 clear",
        "usable 중 cloud",
    ]
    assert max(view.figure.data[2].y) == pytest.approx(0.01051)
    assert "수체 라벨이 아닙니다" in view.figure.layout.title.text
    assert view.readiness.daily_timeseries_prediction_available is False
    assert "전처리" in view.readiness.headline_ko


def test_iceye_source_eda_keeps_incidence_and_look_side_visible() -> None:
    view = build_nas_source_eda("iceye-raw")

    metrics = {row["label"]: row["value"] for row in view.metric_rows}
    assert metrics["원본 획득"] == "5회"
    assert metrics["중심 입사각"] == "11.61~33.17°"
    assert metrics["Look side"] == "right 4 · left 1"
    assert metrics["라벨 없는 원본"] == "1일"
    assert len(view.table_rows) == 5
    assert view.table_rows[1]["Look side"] == "left"
    assert view.table_rows[1]["GRD GCP"] == 276
    assert view.table_rows[3]["GRD 크기 (H×W)"] == "26305 × 18802"
    assert tuple(view.figure.data[1].marker.symbol) == (
        "circle",
        "diamond",
        "circle",
        "circle",
        "circle",
    )
    assert min(view.figure.data[1].y) == pytest.approx(11.605521)
    assert max(view.figure.data[1].y) == pytest.approx(33.168668)
    assert "입사각" in " ".join(view.readiness.cautions_ko)


def _trace_lat_lon(trace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read coordinates from a Scattermap trace or its Scatter fallback."""

    lats = getattr(trace, "lat", None)
    lons = getattr(trace, "lon", None)
    if lats is None or lons is None:
        return tuple(trace.y), tuple(trace.x)
    return tuple(lats), tuple(lons)


def test_iceye_footprint_map_places_all_layers_near_busan() -> None:
    view = build_nas_source_eda("iceye-raw")

    assert view.map_figure is not None
    figure = view.map_figure
    names = [trace.name for trace in figure.data]
    assert "대표 장면 footprint (2020-03-02)" in names
    assert "획득 중심 · 주황=left-looking" in names
    if importlib.util.find_spec("rasterio") is not None:
        # 라벨 커버리지 레이어는 rasterio(선택 의존성) 있을 때만 그려지는 계약
        assert any(name.startswith("라벨 커버리지") for name in names)

    for trace in figure.data:
        lats, lons = _trace_lat_lon(trace)
        assert lats and lons
        assert all(34.0 <= lat <= 36.0 for lat in lats), trace.name
        assert all(128.0 <= lon <= 130.0 for lon in lons), trace.name

    footprint = figure.data[names.index("대표 장면 footprint (2020-03-02)")]
    lats, lons = _trace_lat_lon(footprint)
    assert len(lats) == 5  # closed 4-corner ring
    assert lats[0] == lats[-1] and lons[0] == lons[-1]

    centers = figure.data[names.index("획득 중심 · 주황=left-looking")]
    lats, lons = _trace_lat_lon(centers)
    assert len(lats) == 5  # one marker per profiled acquisition
    assert "left" in " ".join(centers.text)

    # 지도는 ICEYE 전용이며 다른 그룹의 기존 뷰 구조를 바꾸지 않습니다.
    assert build_nas_source_eda("planetscope-raw").map_figure is None


def test_iceye_footprint_map_degrades_when_optional_assets_are_missing(
    tmp_path,
) -> None:
    profile = load_nas_source_profile()
    group = profile.groups["iceye_raw"]

    figure = build_iceye_footprint_map(
        group,
        meta_path=tmp_path / "missing_meta.json",
        manifest_path=tmp_path / "missing_manifest.json",
    )

    assert figure is not None  # 프로파일 센터 마커만으로도 지도가 유지됩니다.
    names = [trace.name for trace in figure.data]
    assert names == ["획득 중심 · 주황=left-looking"]

    invalid_meta = tmp_path / "broken_meta.json"
    invalid_meta.write_text("{not json", encoding="utf-8")
    figure = build_iceye_footprint_map(
        group,
        meta_path=invalid_meta,
        manifest_path=tmp_path / "missing_manifest.json",
    )
    assert figure is not None
    assert all(not trace.name.startswith("대표 장면") for trace in figure.data)

    assert (
        build_iceye_footprint_map(
            {"acquisitions": ()},
            meta_path=tmp_path / "missing_meta.json",
            manifest_path=tmp_path / "missing_manifest.json",
        )
        is None
    )


def test_university_source_eda_separates_area_level_and_json_validity() -> None:
    view = build_nas_source_eda("university-single-date")

    metrics = {row["label"]: row["value"] for row in view.metric_rows}
    assert metrics["관측시점"] == "1회"
    assert metrics["보고 지점"] == "2곳"
    assert metrics["정상 JSON"] == "2/4"
    assert metrics["영상↔결과 정합"] == "(-0.5, +0.5) px"
    assert metrics["수체 Polygon"] == "5,833개"
    assert [row["지점"] for row in view.table_rows] == ["Andong", "Daecheong"]
    assert [row["WB JSON"] for row in view.table_rows] == [
        "형식 오류",
        "형식 오류",
    ]
    assert [row["WL JSON"] for row in view.table_rows] == ["정상", "정상"]
    assert view.table_rows[0]["수체면적 (km²)"] == pytest.approx(23.17)
    assert view.table_rows[1]["수위 (EL.m)"] == pytest.approx(71.6348549987)
    assert [trace.name for trace in view.figure.data] == [
        "보고 수체면적",
        "보고 수위",
    ]
    assert view.readiness.status == "single_date_reference"
    assert view.readiness.source_eda_available is True
    assert view.readiness.scientific_validation_available is False
    assert "trailing comma" in view.limitations_ko[-1]


def test_worldview_source_eda_keeps_pairs_sensors_dates_and_dimensions_separate() -> (
    None
):
    giheung = build_nas_source_eda("giheung-optical")
    hoedong = build_nas_source_eda("hoedong-optical")

    giheung_metrics = {row["label"]: row["value"] for row in giheung.metric_rows}
    assert giheung_metrics == {
        "고유 관측일": "3일",
        "PAN·MUL Pair": "3쌍",
        "센서": "WV03 · WV02",
        "분석 밴드": "MUL 8 · PAN 1",
    }
    assert [row["센서"] for row in giheung.table_rows] == [
        "WV03",
        "WV02",
        "WV03",
    ]
    assert giheung.table_rows[1]["MUL 크기 (H×W)"] == "2749 × 2339"
    assert giheung.table_rows[1]["PAN 크기 (H×W)"] == "10998 × 9356"

    hoedong_metrics = {row["label"]: row["value"] for row in hoedong.metric_rows}
    assert hoedong_metrics["고유 관측일"] == "2일"
    assert hoedong_metrics["PAN·MUL Pair"] == "3쌍"
    assert [row["관측시각 (UTC)"][:10] for row in hoedong.table_rows].count(
        "2023-11-08"
    ) == 2
    assert hoedong.table_rows[0]["PAN 크기 (H×W)"] == "10208 × 8272"
    assert len(hoedong.figure.data) == 2
    assert max(hoedong.figure.data[1].y) == pytest.approx(100.85504)
    assert "11.55초" in " ".join(hoedong.readiness.cautions_ko)
    assert "독립" in " ".join(hoedong.readiness.cautions_ko)


def test_convenience_payloads_are_plotly_ready_and_do_not_expose_nas_access() -> None:
    for group_id in available_source_group_ids():
        metrics = source_metric_rows(group_id)
        rows = source_table_rows(group_id)
        figure = build_nas_source_figure(group_id)
        view = build_nas_source_eda(group_id)

        assert metrics
        assert rows
        assert isinstance(figure, go.Figure)
        assert all(set(row) == {"label", "value", "help"} for row in metrics)
        rendered = repr(
            (
                [dict(row) for row in view.metric_rows],
                [dict(row) for row in view.table_rows],
                view.provenance_ko,
                view.readiness,
            )
        )
        assert "/mnt/" not in rendered
        assert "172." not in rendered
        assert "smb://" not in rendered.lower()
        assert "admin@" not in rendered.lower()
        assert view.readiness.daily_timeseries_prediction_available is False
        assert view.readiness.scientific_validation_available is False

    with pytest.raises(KeyError, match="no source EDA"):
        build_nas_source_eda("support-assets")
