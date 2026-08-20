from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.store import FileResultStore


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(result_dir=tmp_path / "results")
    return TestClient(create_app(settings))


def _png(mask: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return stream.getvalue()


def test_health_and_models(tmp_path: Path) -> None:
    client = _client(tmp_path)
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["models_loaded"] == 3
    models = client.get("/api/v1/models").json()
    assert {item["id"] for item in models} == {
        "irregular-area-trend",
        "persistence",
        "weather-morphology",
    }
    persistence = next(item for item in models if item["id"] == "persistence")
    assert persistence["produces_water_level"] is True


def test_sample_weather_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)
    stations = client.get("/api/v1/weather/stations")
    assert stations.status_code == 200
    assert any(item["id"] == "159" for item in stations.json()["items"])
    response = client.post(
        "/api/v1/weather/observations",
        json={
            "station_id": "159",
            "start_date": "2024-08-01",
            "end_date": "2024-08-03",
            "source": "sample",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["station_id"] == "159"
    assert payload["station_name"] == "부산"
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["date"] == "2024-08-01"
    assert payload["rows"][0]["source"] == "sample"
    assert payload["metadata"]["is_forecast"] is False


def test_prediction_preserves_non_secret_source_metadata(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("mask.png", _png(mask), "image/png"))],
        data={
            "input_metadata_json": (
                '{"sample_id":"busan-nas-water-labels",'
                '"sensor":"PlanetScope family inferred",'
                '"source_manifest_path":"ui_next/assets/nas/busan_2020/manifest.json"}'
            )
        },
    )

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["input"]["source_metadata"]["sample_id"] == "busan-nas-water-labels"
    loaded = client.get(f"/api/v1/predictions/{result['id']}")
    assert loaded.status_code == 200
    assert loaded.json()["input"]["source_metadata"] == result["input"][
        "source_metadata"
    ]


def test_weather_status_and_missing_kma_key_are_actionable(tmp_path: Path) -> None:
    with patch.dict("os.environ", {}, clear=True):
        client = _client(tmp_path)
        status = client.get("/api/v1/weather/status")
        assert status.status_code == 200
        status_payload = status.json()
        assert status_payload["server_key_configured"] is False
        assert status_payload["asos_available"] is False
        assert status_payload["sample_available"] is True
        assert status_payload["per_request_key_supported"] is True

        response = client.post(
            "/api/v1/weather/observations",
            json={
                "station_id": "159",
                "start_date": "2024-08-01",
                "end_date": "2024-08-03",
                "source": "asos",
            },
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "KMA_API_KEY_REQUIRED"
    assert detail["sample_fallback_available"] is True
    assert detail["setup_env_var"] == "KMA_API_KEY"


def test_prediction_round_trip_files_csv_bundle_and_reload(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = np.zeros((6, 6), dtype=np.uint8)
    first[2:4, 2:4] = 1
    second = first.copy()
    response = client.post(
        "/api/v1/predictions",
        files=[
            ("files", ("2026-08-01.png", _png(first), "image/png")),
            ("files", ("2026-08-02.png", _png(second), "image/png")),
        ],
        data={
            "model_id": "persistence",
            "horizon_steps": "2",
            "pixel_area_m2": "9",
            "source_dates_json": '["2026-08-01","2026-08-02"]',
            "historical_water_levels_json": "[1.1,1.25]",
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["steps"][0]["target_date"] == "2026-08-03"
    assert result["steps"][0]["water_area_m2"] == 36
    assert result["steps"][0]["water_level_m"] == 1.25
    assert result["steps"][0]["risk"] == "normal"
    prediction_id = result["id"]

    detail = client.get(f"/api/v1/predictions/{prediction_id}")
    assert detail.status_code == 200
    files = client.get(f"/api/v1/predictions/{prediction_id}/files").json()["items"]
    names = {item["name"] for item in files}
    assert {
        "result.json",
        "results.csv",
        "mask_001.npy",
        "mask_001.png",
        "mask_001.tif",
    } <= names
    assert (
        client.get(f"/api/v1/predictions/{prediction_id}/files/result.json").status_code
        == 200
    )
    csv_response = client.get(f"/api/v1/predictions/{prediction_id}/files/results.csv")
    assert csv_response.status_code == 200
    assert "water_level_m" in csv_response.text

    bundle_response = client.get(f"/api/v1/predictions/{prediction_id}/bundle")
    assert bundle_response.status_code == 200
    with ZipFile(BytesIO(bundle_response.content)) as archive:
        assert {"result.json", "results.csv", "mask_002.npy"} <= set(archive.namelist())

    reloaded = FileResultStore(tmp_path / "results")
    assert reloaded.get(prediction_id).steps[0].water_level_m == 1.25


def test_prediction_evaluates_supplied_future_truth_dynamically(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("2026-08-01.png", _png(mask), "image/png"))],
        data={
            "model_id": "persistence",
            "horizon_steps": "3",
            "source_dates_json": '["2026-08-01"]',
            "historical_water_levels_json": "[2.0]",
            "reference_water_levels_json": "[1.0,2.0,4.0]",
            "evaluation_kind": "measured",
            "evaluation_truth_provenance": "Gauge QA fixture",
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    evaluation = result["evaluation"]
    assert evaluation["status"] == "available"
    assert evaluation["kind"] == "measured"
    assert evaluation["label"] == "실측 사후 평가"
    assert evaluation["truth_provenance"] == "Gauge QA fixture"
    assert evaluation["sample_count"] == 3
    assert evaluation["mae_m"] == pytest.approx(1.0)
    assert evaluation["rmse_m"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert evaluation["mape_pct"] == pytest.approx(50.0)
    assert evaluation["r2"] == pytest.approx(-1.0 / 14.0)
    assert evaluation["bias_m"] == pytest.approx(-1.0 / 3.0)
    assert [pair["residual_m"] for pair in evaluation["pairs"]] == [1.0, 0.0, -2.0]
    assert [step["water_level_source"] for step in result["steps"]] == [
        "adapter_output",
        "adapter_output",
        "adapter_output",
    ]
    assert [step["reference_water_level_m"] for step in result["steps"]] == [
        1.0,
        2.0,
        4.0,
    ]
    assert [step["water_level_error_m"] for step in result["steps"]] == [
        1.0,
        0.0,
        -2.0,
    ]

    csv_response = client.get(
        f"/api/v1/predictions/{result['id']}/files/results.csv"
    )
    assert csv_response.status_code == 200
    assert "reference_water_level_m" in csv_response.text
    assert "water_level_error_m" in csv_response.text


def test_reference_truth_is_not_a_predictor_input(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((3, 3), dtype=np.uint8)
    common_data = {
        "model_id": "persistence",
        "horizon_steps": "2",
        "historical_water_levels_json": "[3.0]",
    }
    first = client.post(
        "/api/v1/predictions",
        files=[("files", ("mask.png", _png(mask), "image/png"))],
        data={**common_data, "reference_water_levels_json": "[1.0,2.0]"},
    )
    second = client.post(
        "/api/v1/predictions",
        files=[("files", ("mask.png", _png(mask), "image/png"))],
        data={**common_data, "reference_water_levels_json": "[100.0,200.0]"},
    )
    assert first.status_code == second.status_code == 201
    first_result = first.json()
    second_result = second.json()
    assert [step["water_level_m"] for step in first_result["steps"]] == [3.0, 3.0]
    assert [step["water_level_m"] for step in second_result["steps"]] == [3.0, 3.0]
    assert [step["water_area_pixels"] for step in first_result["steps"]] == [0, 0]
    assert [step["water_area_pixels"] for step in second_result["steps"]] == [0, 0]
    assert first_result["evaluation"]["rmse_m"] != second_result["evaluation"][
        "rmse_m"
    ]


def test_weather_model_and_calibrated_level(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("mask.png", _png(mask), "image/png"))],
        data={
            "model_id": "weather-morphology",
            "horizon_steps": "1",
            "pixel_area_m2": "4",
            "weather_json": '[{"date":"2026-08-10","precipitation_mm":20}]',
            "water_level_config_json": (
                '{"baseline_level_m":2.0,"meters_per_area_ratio":0.5,"baseline_area_m2":4}'
            ),
        },
    )
    assert response.status_code == 201, response.text
    step = response.json()["steps"][0]
    assert step["water_area_pixels"] == 9
    assert step["water_level_m"] == 6.0
    assert step["risk"] == "flood_risk"
    assert step["rule_based"] is True


def test_irregular_area_trend_api_uses_all_dated_frames(tmp_path: Path) -> None:
    client = _client(tmp_path)
    masks = []
    for radius in (1, 2, 3, 3):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8 - radius : 8 + radius, 8 - radius : 8 + radius] = 1
        masks.append(mask)
    response = client.post(
        "/api/v1/predictions",
        files=[
            ("files", (f"mask_{index}.png", _png(mask), "image/png"))
            for index, mask in enumerate(masks)
        ],
        data={
            "model_id": "irregular-area-trend",
            "horizon_steps": "7",
            "source_dates_json": (
                '["2020-02-18","2020-03-12","2020-03-25","2020-04-14"]'
            ),
            "target_dates_json": (
                '["2020-04-15","2020-04-16","2020-04-17",'
                '"2020-04-18","2020-04-19","2020-04-20","2020-04-21"]'
            ),
            "weather_json": (
                '[{"date":"2020-04-15","precipitation_mm":20.0},'
                '{"date":"2020-04-16","precipitation_mm":0.0}]'
            ),
        },
    )

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["model_id"] == "irregular-area-trend"
    assert len(result["steps"]) == 7
    metadata = result["adapter_metadata"]
    assert metadata["uses_all_input_frames"] is True
    assert metadata["observation_count"] == 4
    assert metadata["elapsed_days"] == [0.0, 23.0, 36.0, 56.0]
    assert metadata["observed_area_pixels"] == [4, 16, 36, 36]
    assert metadata["matched_target_weather_rows"] == 2
    assert len(metadata["targets"]) == 7


def test_rejects_misaligned_history(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((3, 3), dtype=np.uint8)
    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("mask.png", _png(mask), "image/png"))],
        data={"historical_water_levels_json": "[1.0,2.0]"},
    )
    assert response.status_code == 422
    assert "length must match" in response.json()["detail"]


def test_rejects_misaligned_reference_truth(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((3, 3), dtype=np.uint8)
    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("mask.png", _png(mask), "image/png"))],
        data={
            "horizon_steps": "2",
            "reference_water_levels_json": "[1.0]",
        },
    )
    assert response.status_code == 422
    assert "reference_water_levels_m length must match horizon_steps" in response.json()[
        "detail"
    ]


def test_rejects_unsorted_or_nonfuture_dates(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((3, 3), dtype=np.uint8)
    response = client.post(
        "/api/v1/predictions",
        files=[
            ("files", ("one.png", _png(mask), "image/png")),
            ("files", ("two.png", _png(mask), "image/png")),
        ],
        data={"source_dates_json": '["2026-08-02","2026-08-01"]'},
    )
    assert response.status_code == 422
    assert "strictly increasing" in response.json()["detail"]

    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("one.png", _png(mask), "image/png"))],
        data={
            "horizon_steps": "1",
            "source_dates_json": '["2026-08-02"]',
            "target_dates_json": '["2026-08-02"]',
        },
    )
    assert response.status_code == 422
    assert "after the latest source" in response.json()["detail"]


def test_rejects_invalid_weather_domain(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mask = np.zeros((3, 3), dtype=np.uint8)
    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("one.png", _png(mask), "image/png"))],
        data={"weather_json": '[{"precipitation_mm":-1}]'},
    )
    assert response.status_code == 422
    assert "greater than or equal to 0" in response.json()["detail"]
