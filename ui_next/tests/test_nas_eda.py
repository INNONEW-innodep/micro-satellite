from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from ui_next.nas_catalog import get_nas_dataset_group
from ui_next.nas_eda import (
    BUSAN_SOURCE_MANIFEST_PATH,
    build_busan_resampling_error_figure,
    build_busan_source_area_figure,
    build_inventory_size_figure,
    build_inventory_timeline_figure,
    build_readiness_matrix_figure,
    busan_source_rows,
    evaluate_busan_area_holdout,
    explain_readiness,
    holdout_rows,
    inventory_group_rows,
    readiness_rows,
    summarize_busan_source_eda,
    summarize_inventory,
    summarize_readiness,
)


def test_inventory_eda_covers_the_complete_classified_catalog() -> None:
    summary = summarize_inventory()

    assert summary.file_count == 300
    assert summary.total_bytes == 57_270_553_908
    assert summary.total_gib == pytest.approx(53.3373597)
    assert summary.group_count == 8
    assert summary.dated_group_count == 7
    assert summary.timeline_point_count == 23
    assert summary.acquisition_count == 28
    assert summary.matched_pair_count == 8
    assert summary.materialized_group_count == 2
    assert summary.direct_candidate_group_count == 2
    assert (
        sum(bucket.total_bytes for bucket in summary.modality_buckets)
        == summary.total_bytes
    )
    assert (
        sum(bucket.file_count for bucket in summary.readiness_buckets)
        == summary.file_count
    )

    largest = max(summary.groups, key=lambda group: group.total_bytes)
    assert largest.group_id == "iceye-raw"
    assert largest.total_gib == pytest.approx(21.9119471)
    assert largest.size_share_pct == pytest.approx(41.0817994)


def test_inventory_chart_data_keeps_sensors_and_groups_separate() -> None:
    summary = summarize_inventory()
    rows = inventory_group_rows(summary)
    size_figure = build_inventory_size_figure(summary)
    timeline_figure = build_inventory_timeline_figure(summary)

    assert len(rows) == 8
    assert sum(row["파일"] for row in rows) == 300
    assert len(size_figure.data) == 1
    assert len(size_figure.data[0].x) == 8
    assert len(timeline_figure.data) == 7
    assert sum(len(trace.x) for trace in timeline_figure.data) == 23
    assert "서로 다른 센서/지역" in timeline_figure.layout.title.text


def test_inventory_summary_rejects_catalog_interval_drift() -> None:
    source = get_nas_dataset_group("busan-water-labels")
    from dataclasses import replace

    changed = replace(source, interval_days=(1, 2, 3))
    with pytest.raises(ValueError, match="interval_days"):
        summarize_inventory((changed,))


def test_busan_source_eda_uses_original_common_grid_statistics() -> None:
    summary = summarize_busan_source_eda()

    assert summary.dataset_id == "nas-busan-planetscope-water-labels-2020"
    assert summary.crs == "EPSG:32652"
    assert summary.source_pixel_width_m == 3.0
    assert summary.source_pixel_height_m == 3.0
    assert summary.source_pixel_area_m2 == 9.0
    assert summary.preview_width == summary.preview_height == 512
    assert summary.common_grid_width == 9459
    assert summary.common_grid_height == 8897
    assert summary.common_grid_pixel_count == 84_156_723
    assert summary.frame_count == 4
    assert summary.period_days == 56
    assert summary.interval_days == (23, 13, 20)
    assert summary.interval_min_days == 13
    assert summary.interval_median_days == 20.0
    assert summary.interval_max_days == 23

    assert [frame.source_water_pixels for frame in summary.frames] == [
        3_181_939,
        3_075_039,
        3_363_699,
        3_278_274,
    ]
    assert [frame.source_water_area_km2 for frame in summary.frames] == pytest.approx(
        [28.637451, 27.675351, 30.273291, 29.504466]
    )
    assert summary.source_area_change_pct == pytest.approx(3.02755647)
    assert summary.max_abs_preview_area_error_pct == pytest.approx(0.79935853)
    assert all(frame.pair_grid_equal for frame in summary.frames)
    assert [
        frame.source_water_fraction_pct for frame in summary.frames
    ] == pytest.approx([3.7809683, 3.6539434, 3.9969463, 3.8954392])


def test_busan_source_rows_and_figures_label_preview_as_a_separate_series() -> None:
    summary = summarize_busan_source_eda()
    scores = evaluate_busan_area_holdout(summary)
    rows = busan_source_rows(summary)
    area_figure = build_busan_source_area_figure(summary, scores)
    error_figure = build_busan_resampling_error_figure(summary)

    assert len(rows) == 4
    assert rows[0]["원본 공통격자 면적 (km²)"] == 28.637451
    assert rows[1]["직전 대비 (%)"] == pytest.approx(-3.36)
    assert [trace.name for trace in area_figure.data[:2]] == [
        "원본 공통격자 계산",
        "512 파생 자산",
    ]
    assert len(area_figure.data) == 4
    assert tuple(error_figure.data[0].y) == pytest.approx(
        (-0.2076508, -0.2985126, -0.7993585, -0.1925270)
    )


def test_one_point_holdout_baselines_are_explicitly_not_validation_metrics() -> None:
    scores = evaluate_busan_area_holdout()
    persistence, linear = scores

    assert persistence.model_id == "area-persistence"
    assert persistence.predicted_area_km2 == pytest.approx(30.273291)
    assert persistence.observed_area_km2 == pytest.approx(29.504466)
    assert persistence.mae_km2 == persistence.rmse_km2 == pytest.approx(0.768825)
    assert persistence.absolute_percentage_error_pct == pytest.approx(2.60579195)
    assert linear.model_id == "irregular-linear-area"
    assert linear.predicted_area_km2 == pytest.approx(30.1472754734)
    assert linear.mae_km2 == linear.rmse_km2 == pytest.approx(0.6428094734)
    assert linear.absolute_percentage_error_pct == pytest.approx(2.1786853333)
    assert all(score.evaluation_count == 1 for score in scores)
    assert all(not score.scientific_validation_allowed for score in scores)
    assert all("일반화 성능" in score.note_ko for score in scores)

    rows = holdout_rows(scores)
    assert all(row["평가 표본"] == 1 for row in rows)
    assert all(row["검증 성능 주장 가능"] is False for row in rows)


def test_busan_manifest_validation_catches_area_tampering(tmp_path) -> None:
    payload = json.loads(BUSAN_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["frames"][0]["source_common_grid_water_area_m2"] += 9.0
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source water area"):
        summarize_busan_source_eda(changed)


def test_source_eda_never_exposes_the_absolute_nas_root() -> None:
    summary = summarize_busan_source_eda()
    serialized = repr(summary)

    assert "/mnt/" not in serialized
    assert "172." not in serialized
    assert all(
        not frame.input_relative_path.startswith("/") for frame in summary.frames
    )
    assert all(
        not frame.label_relative_path.startswith("/") for frame in summary.frames
    )


def test_readiness_is_capability_specific_and_current_api_is_only_busan() -> None:
    summaries = summarize_readiness()
    by_id = {summary.group_id: summary for summary in summaries}

    busan = by_id["busan-water-labels"]
    iceye = by_id["iceye-water-labels"]
    support = by_id["support-assets"]
    assert busan.can_execute_current_api is True
    assert iceye.can_execute_current_api is True
    assert support.can_execute_current_api is False
    assert {gate.key: gate.state for gate in busan.gates} == {
        "catalog_eda": "ready",
        "source_raster_eda": "limited",
        "mask_forecast_demo": "ready",
        "learned_timeseries": "blocked",
        "water_level_forecast": "blocked",
        "future_truth_evaluation": "blocked",
    }
    assert {gate.key: gate.state for gate in iceye.gates}[
        "mask_forecast_demo"
    ] == "ready"
    assert "보관" in support.headline_ko

    rows = readiness_rows(summaries)
    figure = build_readiness_matrix_figure(summaries)
    assert len(rows) == 8 * 6
    assert len(figure.data) == 1
    assert len(figure.data[0].z) == 8
    assert len(figure.data[0].z[0]) == 6


def test_profiles_and_readiness_records_are_frozen() -> None:
    inventory = summarize_inventory()
    source = summarize_busan_source_eda()
    readiness = explain_readiness(get_nas_dataset_group("busan-water-labels"))

    with pytest.raises(FrozenInstanceError):
        inventory.file_count = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        source.period_days = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        readiness.can_execute_current_api = False  # type: ignore[misc]
