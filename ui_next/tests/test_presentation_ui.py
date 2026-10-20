"""지역 발표 화면의 순수 도우미와 단계 배선."""

from __future__ import annotations

from ui_next.presentation_ui import (
    data_status_badges,
    detection_rows,
    detection_summary,
    forecast_headline,
    region_caveats,
)
from ui_next.state import PHASES, phase_index

REGION = {
    "id": "r", "name": "R", "satellite_available": True, "gt_labels_available": False, "satellite_levels_available": True,
    "gauge_available": True, "n_stations": 3, "n_stations_trained": 3,
    "satellite": {"available": True, "sensors": ["PlanetScope"], "scenes": {"PlanetScope": ["2019-03-23", "2022-05-23"]}, "notes": "②③만"},
}
DETECTION = {"available": True, "sensors": {"ICEYE": {"device": "cpu", "pooled": {"water_iou": 0.927, "f1": 0.962},
             "per_scene": [{"date": "2020-03-02", "water_iou": 0.925, "f1": 0.96}, {"date": "2020-04-16", "water_iou": 0.881, "f1": 0.94}]}}}
EVALUATION = {"split": {"test_start": "2024-01-01"}, "rows": [
    {"station_name": "A", "horizon": 7, "skill_vs_persistence": 0.5, "test_rmse": 0.08, "regime_shift_flag": False},
    {"station_name": "B", "horizon": 7, "skill_vs_persistence": -0.05, "test_rmse": 0.14, "regime_shift_flag": True},
    {"station_name": "A", "horizon": 1, "skill_vs_persistence": 0.3, "test_rmse": 0.06},
]}


def test_badges_reflect_availability() -> None:
    badges = dict(data_status_badges(REGION))
    assert badges["위성 장면 · PlanetScope 2시점"] is True
    assert badges["GT 수체 라벨"] is False
    assert badges["수위계 3지점"] is True
    assert dict(data_status_badges({"satellite": {}}))["위성 장면 · 없음"] is False


def test_detection_tables() -> None:
    rows = detection_rows(DETECTION)
    assert [r["관측일"] for r in rows] == ["2020-03-02", "2020-04-16"]
    summary = detection_summary(DETECTION)[0]
    assert summary["씬 수"] == 2 and summary["최저 IoU"] == 0.881
    assert abs(summary["씬 평균 IoU"] - 0.903) < 1e-9
    assert detection_summary({"available": False}) == []


def test_forecast_headline_picks_horizon_rows_only() -> None:
    head = forecast_headline(EVALUATION, 7)
    assert head["n"] == 2 and head["best"][0] == "A" and head["worst"][0] == "B"
    assert abs(head["mean_skill"] - 0.225) < 1e-9
    assert forecast_headline(EVALUATION, 30)["n"] == 0


def test_caveats_are_honest_about_missing_labels_and_baseline() -> None:
    lines = region_caveats(REGION, {"available": False}, EVALUATION)
    assert any("GT" in line for line in lines)
    assert any("persistence" in line for line in lines)
    assert any("B" in line and "1σ" in line for line in lines)
    assert any("②③" in line for line in lines)


def test_presentation_phase_is_wired() -> None:
    assert phase_index("지역 발표") == phase_index("수위 예측") + 1
    assert len(PHASES) == 10
