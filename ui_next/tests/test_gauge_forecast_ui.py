"""수위 예측 화면의 순수 도우미와 배선(단계·클라이언트·따라하기 5단계)을 고정한다.

화면 자체는 Streamlit이 필요하지만, 평가표 변환·업로드 파싱·라벨링은 순수
함수라 여기서 직접 검증한다. 규율: persistence 를 예측 모델로 부르지 않는다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ui_next.api_client import APIClient, Endpoints
from ui_next.gauge_forecast_ui import (
    FORECASTER_KO,
    evaluation_pivot,
    evaluation_rows_for_station,
    forecaster_label,
    parse_uploaded_series,
    skill_text,
    terrain_label,
    winners_table,
)
from ui_next.state import PHASES, phase_index
from ui_next.walkthrough import STEP_KEYS, STEP_TITLES, load_detection_results

UI_DIR = Path(__file__).resolve().parents[1]
REPO = UI_DIR.parent

EVALUATION = {
    "rows": [
        {"station_id": "A", "station_name": "A교", "terrain": "estuary", "horizon": 7, "forecaster_id": "mean_reversion",
         "val_rmse": 0.14, "test_n": 366, "test_mse": 0.0061, "test_rmse": 0.078, "test_mae": 0.038,
         "persistence_test_rmse": 0.111, "climatology_test_rmse": 0.077, "skill_vs_persistence": 0.508, "regime_shift_flag": False},
        {"station_id": "A", "station_name": "A교", "terrain": "estuary", "horizon": 1, "forecaster_id": "ensemble_mean",
         "val_rmse": 0.097, "test_n": 366, "test_mse": 0.0041, "test_rmse": 0.064, "test_mae": 0.031,
         "persistence_test_rmse": 0.077, "climatology_test_rmse": 0.077, "skill_vs_persistence": 0.308, "regime_shift_flag": False},
        {"station_id": "B", "station_name": "B교", "terrain": "mainstem", "horizon": 7, "forecaster_id": "ridge_ar",
         "val_rmse": 0.16, "test_n": 366, "test_mse": 0.019, "test_rmse": 0.137, "test_mae": 0.08,
         "persistence_test_rmse": 0.134, "climatology_test_rmse": 0.41, "skill_vs_persistence": -0.05, "regime_shift_flag": True,
         "test_mean_shift_sigma": 1.8},
    ],
    "winners_by_terrain": {"estuary": {"mean_reversion": 4, "ensemble_mean": 2}, "mainstem": {"ridge_ar": 6}},
}


def test_labels_never_call_persistence_a_forecast() -> None:
    assert "기준선" in forecaster_label("persistence")
    assert forecaster_label("unknown_model") == "unknown_model"
    assert forecaster_label(None) == "-"
    assert terrain_label("estuary") == "감조(하구)"
    assert skill_text(None) == "-" and skill_text(0.508) == "+50.8 %" and skill_text(-0.05) == "-5.0 %"
    assert all("예측" not in v or k != "persistence" for k, v in FORECASTER_KO.items())


def test_station_rows_are_sorted_by_horizon_and_carry_baselines() -> None:
    rows = evaluation_rows_for_station(EVALUATION, "A")
    assert [r["예측일수"] for r in rows] == ["D+1", "D+7"]
    assert rows[1]["선택 모델"] == forecaster_label("mean_reversion")
    assert rows[1]["persistence RMSE (m)"] == 0.111
    assert rows[1]["skill vs persistence"] == pytest.approx(0.508)
    assert evaluation_rows_for_station(EVALUATION, "없음") == []


def test_pivot_and_winners_tables() -> None:
    pivot = evaluation_pivot(EVALUATION)
    by_station = {row["지점"]: row for row in pivot}
    assert "D+7" in by_station["A교"] and "0.078" in by_station["A교"]["D+7"]
    assert by_station["B교"].get("주의") == "체계 이동"
    assert "주의" not in by_station["A교"]
    winners = winners_table(EVALUATION)
    assert winners[0]["지형"] == "감조(하구)"
    assert winners[0]["가장 많이 선택된 모델"] == forecaster_label("mean_reversion")


def test_parse_uploaded_series_csv_and_json() -> None:
    csv_rows = parse_uploaded_series(b"date,water_level_m\n2020-01-01,1.5\n2020-01-02,\n2020-01-03,1.7\n")
    assert csv_rows == [{"date": "2020-01-01", "water_level_m": 1.5}, {"date": "2020-01-03", "water_level_m": 1.7}]
    json_rows = parse_uploaded_series(json.dumps({"rows": [{"date": "2020-01-01", "level": 2}]}).encode())
    assert json_rows == [{"date": "2020-01-01", "level": 2}]
    assert parse_uploaded_series(b"   ") == []
    with pytest.raises(ValueError):
        parse_uploaded_series(b"foo,bar\n1,2\n")


def test_phase_and_walkthrough_wiring() -> None:
    assert phase_index("수위 예측") == phase_index("결과") + 1
    assert STEP_KEYS[-1] == "gauge" and "5단계" in STEP_TITLES["gauge"]
    app_source = (UI_DIR / "app.py").read_text(encoding="utf-8")
    assert "render_gauge_forecast_page" in app_source
    # readiness 튜플은 PHASES 길이와 같아야 한다 — 단계를 끼워 넣을 때 가장 흔한 사고
    match = re.search(r"readiness = \((.*?)\n    \)", app_source, re.S)
    assert match is not None
    entries = [line for line in match.group(1).splitlines() if line.strip().startswith(("True", "bool"))]
    assert len(entries) == len(PHASES)


def test_api_client_gauge_paths_quote_station_ids() -> None:
    client = APIClient("http://example")
    path = client._path(Endpoints.GAUGE_FORECAST, region_id="nakdong_busan", station_id="구포대교")
    assert path == "/api/v1/regions/nakdong_busan/gauge/%EA%B5%AC%ED%8F%AC%EB%8C%80%EA%B5%90/forecast"
    assert client._path(Endpoints.REGION, region_id="a/b") == "/api/v1/regions/a%2Fb"


@pytest.mark.skipif(not (REPO / "data" / "wbms_runs" / "wb" / "2020").is_dir(), reason="로컬 WBMS 산출이 없다")
def test_detection_results_join_iou_from_valreport() -> None:
    """로컬에는 체인 배치 JSON 대신 VALREPORT(per_scene)만 있어도 IoU가 채워져야 한다."""

    rows = load_detection_results()
    assert rows, "WB 마스크 산출이 있으면 표가 비면 안 된다"
    with_iou = [r for r in rows if isinstance(r.get("IoU"), (int, float))]
    assert with_iou, "IoU 조인이 전부 비면 근거 없는 표가 된다"
    assert all(0.0 < r["IoU"] <= 1.0 for r in with_iou)
