from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _result_fixture() -> dict:
    steps = []
    pairs = []
    weather = []
    for horizon in range(1, 31):
        target_date = f"2026-08-{horizon + 1:02d}" if horizon < 30 else "2026-09-01"
        predicted = 2.0 + horizon * 0.01
        reference = 2.0 + horizon * 0.009
        residual = predicted - reference
        steps.append(
            {
                "horizon": horizon,
                "target_date": target_date,
                "water_area_pixels": 100 + horizon,
                "water_area_km2": 0.001 + horizon / 100_000,
                "water_level_m": predicted,
                "water_level_source": "adapter_output",
                "reference_water_level_m": reference,
                "water_level_error_m": residual,
                "area_change_pct": horizon / 10,
                "risk": "normal",
            }
        )
        pairs.append(
            {
                "horizon": horizon,
                "target_date": target_date,
                "predicted_water_level_m": predicted,
                "reference_water_level_m": reference,
                "residual_m": residual,
                "absolute_error_m": abs(residual),
                "percentage_error_pct": abs(residual / reference) * 100,
            }
        )
        weather.append(
            {
                "date": target_date,
                "kind": "scenario",
                "precipitation_mm": horizon % 5,
            }
        )
    return {
        "id": "result-smoke-test",
        "status": "completed",
        "model_id": "trained-demo",
        "created_at": "2026-08-01T00:00:00Z",
        "input": {
            "source_dates": [
                "2026-07-29",
                "2026-07-30",
                "2026-07-31",
                "2026-08-01",
            ]
        },
        "parameters": {
            "historical_water_levels_m": [1.9, 1.95, 1.98, 2.0],
            "weather": weather,
        },
        "steps": steps,
        "artifacts": [],
        "warnings": [],
        "evaluation": {
            "status": "available",
            "kind": "holdout",
            "label": "홀드아웃 백테스트",
            "truth_provenance": "test fixture",
            "sample_count": 30,
            "mae_m": 0.0155,
            "rmse_m": 0.018,
            "mape_pct": 0.7,
            "r2": 0.98,
            "bias_m": 0.0155,
            "pairs": pairs,
        },
    }


def test_daily_horizons_and_dynamic_evaluation_render_without_error() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20)
    app.session_state["service_mode"] = "수체 시계열 예측"
    app.session_state["phase"] = 3
    app.session_state["prediction_result"] = _result_fixture()
    app.session_state["connection_attempted"] = True
    app.session_state["connection_status"] = {"status": "error"}

    app.run(timeout=20)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["D+7 예측 수위"] == "2.070 m"
    assert metrics["D+14 예측 수위"] == "2.140 m"
    assert metrics["D+30 예측 수위"] == "2.300 m"
    assert metrics["RMSE"] == "0.018 m"
    assert metrics["평가 표본"] == "30일"
    assert any(
        radio.label == "그래프 전망 범위" and radio.value == 30
        for radio in app.radio
    )
    assert len(app.get("plotly_chart")) >= 3


def test_irregular_nas_area_result_shows_multi_frame_evidence_and_area_horizons() -> None:
    targets = []
    steps = []
    for horizon in range(1, 31):
        target_date = f"2020-05-{horizon:02d}"
        pixels = 10_000 + horizon * 10
        targets.append(
            {
                "horizon": horizon,
                "target_date": target_date,
                "days_ahead": horizon,
                "trend_target_pixels": pixels - 3.5,
                "precipitation_mm": 2.0 if horizon % 4 == 0 else 0.0,
                "rainfall_memory_mm": 2.0,
                "weather_adjustment_pixels": 3.5,
                "raw_target_pixels": float(pixels),
                "capped_target_pixels": pixels,
                "was_capped": False,
            }
        )
        steps.append(
            {
                "horizon": horizon,
                "target_date": target_date,
                "water_area_pixels": pixels,
                "water_area_km2": pixels / 1000.0,
                "water_level_m": None,
                "water_level_source": "unavailable",
                "reference_water_level_m": None,
                "water_level_error_m": None,
                "area_change_pct": horizon / 10.0,
                "risk": "normal",
            }
        )
    result = {
        "id": "nas-irregular-result",
        "status": "completed",
        "model_id": "irregular-area-trend",
        "created_at": "2020-04-30T00:00:00Z",
        "input": {
            "source_dates": [
                "2020-02-18",
                "2020-03-12",
                "2020-03-25",
                "2020-04-14",
            ]
        },
        "parameters": {"historical_water_levels_m": [], "weather": []},
        "steps": steps,
        "artifacts": [],
        "warnings": [],
        "adapter_metadata": {
            "strategy": "irregular_log_area_trend",
            "uses_all_input_frames": True,
            "observation_count": 4,
            "source_dates": [
                "2020-02-18",
                "2020-03-12",
                "2020-03-25",
                "2020-04-14",
            ],
            "observed_area_pixels": [9891, 9550, 10394, 10192],
            "fitted_area_change_pct_per_day": 0.0833,
            "matched_target_weather_rows": 30,
            "targets": targets,
        },
        "evaluation": {"status": "unavailable", "pairs": []},
    }

    app = AppTest.from_file(str(APP_PATH), default_timeout=20)
    app.session_state["service_mode"] = "수체 시계열 예측"
    app.session_state["phase"] = 3
    app.session_state["prediction_result"] = result
    app.session_state["prediction_context"] = {"pixel_area_placeholder": False}
    app.session_state["connection_attempted"] = True
    app.session_state["connection_status"] = {"status": "error"}

    app.run(timeout=20)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["계산에 사용한 과거"] == "4프레임"
    assert metrics["D+7 수체"] == "10.0700 km²"
    assert metrics["D+14 수체"] == "10.1400 km²"
    assert metrics["D+30 수체"] == "10.3000 km²"
    assert any("일별 수체면적 전망" in item.value for item in app.markdown)
    assert len(app.get("plotly_chart")) >= 2
