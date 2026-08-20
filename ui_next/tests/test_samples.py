from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import date
from io import BytesIO

import pytest
from PIL import Image

from backend.app.schemas import WeatherRow
from backend.app.services.inputs import UploadedBytes, load_mask_sequence
from ui_next.samples import (
    MASK_SIZE,
    NAS_BUSAN_MANIFEST_PATH,
    NAS_BUSAN_MASK_SIZE,
    NAS_BUSAN_SAMPLE_ID,
    NAS_ICEYE_MANIFEST_PATH,
    NAS_ICEYE_MASK_SIZE,
    NAS_ICEYE_SAMPLE_ID,
    RECOVERED_CROP_BOXES,
    RECOVERED_FIGURE_PATH,
    RECOVERED_SOURCE_DATES,
    RECOVERED_TARGET_DATES,
    SOURCE_DATES,
    SUPPORTED_DAILY_HORIZONS,
    SYNTHETIC_REFERENCE_DISCLAIMER_KO,
    TARGET_DATES,
    frame_preview_bytes,
    get_sample,
    list_samples,
    sample_catalog,
)

EXPECTED_IDS = (
    "busan-doc-recovered",
    "busan-nas-water-labels",
    "iceye-nas-water-labels",
    "nakdong_stable",
    "han_drought",
    "gwangju_flood",
)
EXPECTED_NAMES = (
    "부산 문서 그림 복원",
    "부산 NAS 수체 라벨 2020",
    "ICEYE NAS SAR 수체 라벨 2020",
    "낙동강 안정 예시",
    "한강 가뭄 예시",
    "광주 홍수 예시",
)
SYNTHETIC_IDS = ("nakdong_stable", "han_drought", "gwangju_flood")


def test_catalog_list_and_get_helpers_are_stable() -> None:
    samples = list_samples()
    assert tuple(sample.sample_id for sample in samples) == EXPECTED_IDS
    assert tuple(sample.display_name for sample in samples) == EXPECTED_NAMES
    assert tuple(sample_catalog()) == EXPECTED_IDS
    assert all(get_sample(sample.sample_id) is sample for sample in samples)
    recommended = [sample.sample_id for sample in samples if sample.recommended]
    assert recommended == [NAS_BUSAN_SAMPLE_ID]
    with pytest.raises(KeyError, match="available"):
        get_sample("missing")


def test_sample_api_and_nested_metadata_are_immutable() -> None:
    sample = get_sample("nakdong_stable")
    with pytest.raises(FrozenInstanceError):
        sample.display_name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        sample.model_options["rain_mm_per_dilation"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        sample.weather_rows[0]["source"] = "changed"  # type: ignore[index]
    assert sample.synthetic_masks is True
    assert sample.rule_based_baseline is True
    assert sample.measured_data is False
    assert sample.trained_model is False
    assert "실측 자료" in sample.disclaimer_ko


@pytest.mark.parametrize("sample_id", EXPECTED_IDS)
def test_binary_png_frames_are_backend_compatible(sample_id: str) -> None:
    sample = get_sample(sample_id)
    assert len(sample.frames) == 4
    expected_size = {
        "busan-doc-recovered": (533, 500),
        NAS_BUSAN_SAMPLE_ID: NAS_BUSAN_MASK_SIZE,
        NAS_ICEYE_SAMPLE_ID: NAS_ICEYE_MASK_SIZE,
    }.get(sample_id, MASK_SIZE)
    uploaded = []
    for frame in sample.frames:
        assert frame.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert frame.content_type == "image/png"
        with Image.open(BytesIO(frame.png_bytes)) as image:
            assert image.format == "PNG"
            assert image.mode == "L"
            assert image.size == expected_size
            assert set(image.tobytes()).issubset({0, 255})
        uploaded.append(UploadedBytes(frame.name, frame.png_bytes))

    loaded = load_mask_sequence(uploaded, threshold=sample.threshold)
    assert loaded.frames.shape == (4, expected_size[1], expected_size[0])
    assert {int(value) for value in set(loaded.frames.ravel())} <= {0, 1}


def test_frame_bytes_and_helpers_are_deterministic() -> None:
    first_pass = {
        sample.sample_id: tuple(frame.sha256 for frame in sample.frames)
        for sample in list_samples()
    }
    second_pass = {
        sample.sample_id: tuple(frame.sha256 for frame in get_sample(sample.sample_id).frames)
        for sample in list_samples()
    }
    assert first_pass == second_pass
    assert len({digest for digests in first_pass.values() for digest in digests}) == 24
    assert frame_preview_bytes("gwangju_flood") == get_sample("gwangju_flood").frames[-1].png_bytes
    assert frame_preview_bytes("gwangju_flood", 0) == get_sample("gwangju_flood").frames[0].png_bytes
    with pytest.raises(IndexError, match="outside"):
        frame_preview_bytes("gwangju_flood", 99)


@pytest.mark.parametrize("sample_id", SYNTHETIC_IDS)
def test_dates_levels_and_weather_are_aligned(sample_id: str) -> None:
    sample = get_sample(sample_id)
    assert sample.source_dates == SOURCE_DATES
    assert sample.target_dates == TARGET_DATES
    assert len(sample.historical_water_levels_m) == len(SOURCE_DATES)
    assert len(sample.weather_rows) == len(SOURCE_DATES) + len(TARGET_DATES)

    source_rows = sample.weather_rows[: len(SOURCE_DATES)]
    target_rows = sample.weather_rows[len(SOURCE_DATES) :]
    assert tuple(row["date"] for row in source_rows) == tuple(
        value.isoformat() for value in SOURCE_DATES
    )
    assert tuple(row["date"] for row in target_rows) == tuple(
        value.isoformat() for value in TARGET_DATES
    )
    assert all(row["kind"] == "observed" for row in source_rows)
    assert all(row["kind"] == "scenario" for row in target_rows)
    assert all(row["source"] == "built_in_demo" for row in target_rows)
    assert all(row["is_measured"] is False for row in sample.weather_rows)
    assert all(
        row["weather_provenance"]
        == "repository_example_csv_or_code_fallback_with_manual_scenario_override"
        for row in sample.weather_rows
    )


@pytest.mark.parametrize("horizon_days", SUPPORTED_DAILY_HORIZONS)
@pytest.mark.parametrize("sample_id", SYNTHETIC_IDS)
def test_daily_forecast_calendar_and_weather_scenario_are_aligned_and_explicit(
    sample_id: str, horizon_days: int
) -> None:
    sample = get_sample(sample_id)
    dates = sample.forecast_dates(horizon_days)
    weather = sample.forecast_weather_payload(horizon_days)

    assert len(dates) == horizon_days
    assert dates[0] > sample.source_dates[-1]
    assert tuple(
        (dates[index + 1] - dates[index]).days
        for index in range(horizon_days - 1)
    ) == (1,) * (horizon_days - 1)
    assert sample.forecast_date_strings(horizon_days) == tuple(
        value.isoformat() for value in dates
    )

    observed = weather[: len(sample.source_dates)]
    scenario = weather[len(sample.source_dates) :]
    assert len(weather) == len(sample.source_dates) + horizon_days
    assert tuple(row["date"] for row in observed) == sample.source_date_strings
    assert tuple(row["date"] for row in scenario) == sample.forecast_date_strings(
        horizon_days
    )
    assert all(row["kind"] == "observed" for row in observed)
    assert all(row["kind"] == "scenario" for row in scenario)
    assert all(row["source"] == "built_in_demo" for row in scenario)
    assert all(row["is_measured"] is False for row in scenario)
    assert all(row["is_trained_output"] is False for row in scenario)
    assert all(row["scenario_description_ko"] for row in scenario)

    # Every call is deterministic but returns mutable copies for the UI.
    repeated = sample.forecast_weather_payload(horizon_days)
    assert weather == repeated
    weather[-1]["precipitation_mm"] = 999.0
    assert sample.forecast_weather_payload(horizon_days) == repeated


@pytest.mark.parametrize("horizon_days", SUPPORTED_DAILY_HORIZONS)
def test_synthetic_reference_levels_are_deterministic_and_scenario_shaped(
    horizon_days: int,
) -> None:
    stable_sample = get_sample("nakdong_stable")
    drought_sample = get_sample("han_drought")
    flood_sample = get_sample("gwangju_flood")

    stable = stable_sample.synthetic_reference_levels(horizon_days)
    drought = drought_sample.synthetic_reference_levels(horizon_days)
    flood = flood_sample.synthetic_reference_levels(horizon_days)

    assert len(stable) == len(drought) == len(flood) == horizon_days
    assert stable == stable_sample.synthetic_reference_levels(horizon_days)
    assert drought == drought_sample.synthetic_reference_water_levels_m(horizon_days)
    assert flood == flood_sample.synthetic_reference_levels(horizon_days)
    assert max(stable) - min(stable) < 0.05
    assert drought == tuple(sorted(drought, reverse=True))
    assert drought[-1] < drought[0]
    assert flood[0] > flood_sample.water_level_config.baseline_level_m  # type: ignore[union-attr]
    assert max(flood) - min(flood) > 0.35

    for sample in (stable_sample, drought_sample, flood_sample):
        assert sample.water_level_validation_allowed is False
        assert "실측 정답" in sample.synthetic_reference_provenance_ko
        assert "모델 검증" in sample.synthetic_reference_provenance_ko
        assert (
            sample.metadata["synthetic_reference_provenance_ko"]
            == SYNTHETIC_REFERENCE_DISCLAIMER_KO
        )


def test_dynamic_helpers_preserve_legacy_three_day_payload_and_absent_sources() -> None:
    for sample_id in SYNTHETIC_IDS:
        sample = get_sample(sample_id)
        assert sample.forecast_dates(3) == sample.target_dates
        assert sample.forecast_weather_payload(3) == sample.weather_payload()

    recovered = get_sample("busan-doc-recovered")
    assert recovered.forecast_dates(3) == recovered.target_dates
    assert recovered.forecast_weather_payload(30) == ()
    assert recovered.synthetic_reference_levels(30) == ()

    nas = get_sample(NAS_BUSAN_SAMPLE_ID)
    assert nas.forecast_dates(3) == nas.target_dates
    assert nas.forecast_weather_payload(30) == ()
    assert nas.synthetic_reference_levels(30) == ()


@pytest.mark.parametrize("horizon_days", (0, 366))
def test_dynamic_helpers_reject_out_of_contract_horizons(horizon_days: int) -> None:
    sample = get_sample("nakdong_stable")
    with pytest.raises(ValueError, match="between 1 and 365"):
        sample.forecast_dates(horizon_days)
    with pytest.raises(ValueError, match="between 1 and 365"):
        sample.forecast_weather_payload(horizon_days)
    with pytest.raises(ValueError, match="between 1 and 365"):
        sample.synthetic_reference_levels(horizon_days)


@pytest.mark.parametrize("sample_id", SYNTHETIC_IDS)
def test_synthetic_scenarios_make_origin_and_non_event_status_explicit(sample_id: str) -> None:
    sample = get_sample(sample_id)
    metadata = sample.metadata

    assert sample.data_classification == "deterministic_synthetic_scenario"
    assert sample.data_classification_ko == "코드 생성 합성 시나리오"
    assert sample.actual_event_data is False
    assert sample.region_verified is False
    assert sample.observation_dates_verified is False
    assert "기존에 제공받은 지역 실측 데이터가 아닙니다" in sample.source_description_ko
    assert "_mask_png" in sample.mask_provenance_ko
    assert "weather_ex.csv" in sample.weather_provenance_ko
    assert "예시값" in sample.water_level_provenance_ko
    assert "시나리오 라벨" in sample.region_label_note_ko
    assert "실제 홍수·가뭄 사실 확인" in sample.not_suitable_for_ko
    assert "ui_next/samples.py::_mask_png" in sample.source_assets
    assert metadata["actual_event_data"] is False
    assert metadata["data_classification"] == sample.data_classification
    assert metadata["source_assets"] == sample.source_assets
    assert "실제" in sample.disclaimer_ko


@pytest.mark.parametrize("sample_id", SYNTHETIC_IDS)
def test_weather_and_prediction_defaults_match_backend_contract(sample_id: str) -> None:
    sample = get_sample(sample_id)
    validated = [WeatherRow.model_validate(dict(row)) for row in sample.weather_rows]
    assert len(validated) == 7
    assert all(row.precipitation_mm is not None and row.precipitation_mm >= 0 for row in validated)
    assert all(row.humidity_pct is not None and 0 <= row.humidity_pct <= 100 for row in validated)

    config = sample.water_level_config
    assert config is not None
    payload = config.as_api_dict()
    assert config.illustrative is True
    assert config.measured is False
    assert payload["baseline_area_m2"] > 0
    assert set(payload) == {
        "baseline_level_m",
        "meters_per_area_ratio",
        "baseline_area_m2",
    }
    assert sample.recommended_model_id == "weather-morphology"
    assert sample.input_rows()[0]["content"] == sample.frames[0].png_bytes
    assert sample.weather_payload()[0] == dict(sample.weather_rows[0])


def test_recovered_sample_has_exact_provenance_and_forbids_metric_validation() -> None:
    sample = get_sample("busan-doc-recovered")
    assert sample.source_dates == RECOVERED_SOURCE_DATES
    assert sample.target_dates == RECOVERED_TARGET_DATES
    assert sample.historical_water_levels_m == (None, None, None, None)
    assert sample.weather_rows == ()
    assert sample.water_level_config is None
    assert sample.pixel_area_m2 is None
    assert sample.recommended_model_id == "persistence"
    assert sample.synthetic_masks is False
    assert sample.derived_demo is True
    assert sample.raw_source_available is False
    assert sample.raw_data_provenance_known is False
    assert sample.georeferenced is False
    assert sample.pixel_area_known is False
    assert sample.area_validation_allowed is False
    assert sample.water_level_validation_allowed is False
    assert "면적·수위 정확도 검증" in sample.disclaimer_ko
    assert "02_converted_images.png" in sample.provenance
    assert sample.data_classification == "derived_document_figure_demo"
    assert sample.data_classification_ko == "저장소 문서 그림 파생 데모"
    assert sample.actual_event_data is False
    assert sample.region_verified is False
    assert sample.observation_dates_verified is False
    assert "원본 부산 래스터는 현재 저장소에 없어" in sample.source_description_ko
    assert "밝기 128" in sample.mask_provenance_ko
    assert sample.weather_provenance_ko == "이 샘플에는 기상 행을 연결하지 않았습니다."
    assert "02_converted_images.png" in sample.source_assets[0]

    with Image.open(RECOVERED_FIGURE_PATH) as figure:
        grayscale = figure.convert("L")
        for frame, crop_box in zip(sample.frames, RECOVERED_CROP_BOXES, strict=True):
            expected = grayscale.crop(crop_box).point(
                lambda value: 255 if value >= 128 else 0, mode="L"
            )
            with Image.open(BytesIO(frame.png_bytes)) as recovered:
                assert recovered.size == (533, 500)
                assert recovered.tobytes() == expected.tobytes()


def test_nas_busan_sample_uses_manifest_assets_and_keeps_missing_levels_null() -> None:
    manifest = json.loads(NAS_BUSAN_MANIFEST_PATH.read_text(encoding="utf-8"))
    sample = get_sample(NAS_BUSAN_SAMPLE_ID)

    assert sample.source_dates == (
        date(2020, 2, 18),
        date(2020, 3, 12),
        date(2020, 3, 25),
        date(2020, 4, 14),
    )
    assert sample.target_dates == (
        date(2020, 4, 15),
        date(2020, 4, 16),
        date(2020, 4, 17),
    )
    assert sample.historical_water_levels_m == (None, None, None, None)
    assert sample.water_level_config is None
    assert sample.synthetic_reference_levels(30) == ()
    assert sample.weather_rows == ()
    assert sample.forecast_weather_payload(30) == ()
    assert sample.recommended_model_id == "irregular-area-trend"
    assert sample.recommended is True

    assert sample.synthetic_masks is False
    assert sample.derived_demo is False
    assert sample.measured_data is False
    assert sample.raw_source_available is True
    assert sample.raw_data_provenance_known is True
    assert sample.observation_dates_verified is True
    assert sample.pixel_area_known is True
    assert sample.area_validation_allowed is False
    assert sample.water_level_validation_allowed is False
    assert sample.data_classification == manifest["classification"]
    assert sample.classification_badge_ko == "NAS 실자료 · 파생 수체 라벨"
    assert sample.sensor_name_ko == "PlanetScope 계열 추정"
    assert sample.source_manifest_path == "ui_next/assets/nas/busan_2020/manifest.json"
    assert sample.pixel_area_m2 == pytest.approx(manifest["preview_pixel_area_m2"])
    assert manifest["source_pixel_area_m2"] == 9.0
    assert manifest["label_semantics"] == {"0": "non-water", "1": "water"}
    assert manifest["crs"] == "EPSG:32652"
    assert manifest["preview_shape"] == [512, 512]
    assert "PlanetScope 계열 추정" in sample.source_description_ko
    assert "리샘플링" in sample.disclaimer_ko
    assert manifest["dataset_id"] in sample.provenance

    uploaded = []
    for frame, frame_manifest in zip(sample.frames, manifest["frames"], strict=True):
        assert frame.observed_on.isoformat() == frame_manifest["date"]
        assert frame.sha256 == frame_manifest["mask_sha256"]
        assert frame.context_sha256 == frame_manifest["preview_sha256"]
        assert frame.context_png_bytes is not None

        with Image.open(BytesIO(frame.png_bytes)) as mask:
            assert mask.format == "PNG"
            assert mask.mode == "L"
            assert mask.size == NAS_BUSAN_MASK_SIZE
            values = mask.tobytes()
            assert set(values).issubset({0, 255})
            water_pixels = sum(value == 255 for value in values)
        assert water_pixels == frame_manifest["preview_water_pixels"]
        assert water_pixels * sample.pixel_area_m2 == pytest.approx(
            frame_manifest["preview_water_area_m2"]
        )

        with Image.open(BytesIO(frame.context_png_bytes)) as context:
            assert context.format == "PNG"
            assert context.mode == "RGB"
            assert context.size == NAS_BUSAN_MASK_SIZE

        assert hashlib.sha256(frame.png_bytes).hexdigest() == frame_manifest[
            "mask_sha256"
        ]
        assert hashlib.sha256(frame.context_png_bytes).hexdigest() == frame_manifest[
            "preview_sha256"
        ]
        uploaded.append(UploadedBytes(frame.name, frame.png_bytes))

        for source_key in ("input_source", "label_source"):
            relative_path = frame_manifest[source_key]["relative_path"]
            assert any(relative_path in source for source in sample.source_assets)

    loaded = load_mask_sequence(uploaded, threshold=sample.threshold)
    assert loaded.frames.shape == (4, 512, 512)
    assert {int(value) for value in loaded.frames.ravel()} <= {0, 1}
    assert loaded.geo_reference is None

    metadata = sample.metadata
    assert metadata["classification_badge_ko"] == sample.classification_badge_ko
    assert metadata["sensor_name_ko"] == sample.sensor_name_ko
    assert metadata["source_manifest_path"] == sample.source_manifest_path


def test_nas_iceye_sample_uses_actual_common_grid_assets_and_browse_stays_separate() -> None:
    manifest = json.loads(NAS_ICEYE_MANIFEST_PATH.read_text(encoding="utf-8"))
    sample = get_sample(NAS_ICEYE_SAMPLE_ID)

    assert sample.source_dates == (
        date(2020, 3, 2),
        date(2020, 3, 30),
        date(2020, 4, 15),
        date(2020, 4, 16),
    )
    assert sample.target_dates == (
        date(2020, 4, 17),
        date(2020, 4, 18),
        date(2020, 4, 19),
    )
    assert sample.recommended_model_id == "irregular-area-trend"
    assert sample.historical_water_levels_m == (None, None, None, None)
    assert sample.sensor_name_ko == "ICEYE-X5 Stripmap VV"
    assert sample.source_manifest_path == "ui_next/assets/nas/iceye_2020/manifest.json"
    assert sample.pixel_area_m2 == pytest.approx(
        abs(
            manifest["common_grid"]["transform"][0]
            * manifest["common_grid"]["transform"][4]
        )
    )
    assert manifest["cross_date_alignment"]["direct_array_stack_allowed"] is False
    assert manifest["cross_date_alignment"]["warp_to_common_grid_required"] is True

    observed_water = []
    uploads = []
    for frame, item in zip(sample.frames, manifest["items"], strict=True):
        assert frame.context_png_bytes is None
        assert frame.sha256 == item["assets"]["mask"]["sha256"]
        with Image.open(BytesIO(frame.png_bytes)) as image:
            assert image.mode == "L"
            assert image.size == NAS_ICEYE_MASK_SIZE
            values = image.tobytes()
            assert set(values).issubset({0, 255})
            observed_water.append(sum(value == 255 for value in values))
        assert observed_water[-1] == item["counts"]["preview_nearest"][
            "water_pixel_count"
        ]
        assert item["source"]["input"]["sha256"] is None
        assert item["native"]["pair_alignment"]["same_pixel_grid"] is True
        assert item["label_qa"]["actual_unique_values"] == [0, 1]
        assert item["label_qa"]["declared_nodata_occurs_in_pixels"] is False
        uploads.append(UploadedBytes(frame.name, frame.png_bytes))

    assert observed_water == [4627, 4236, 4319, 5350]
    loaded = load_mask_sequence(uploads, threshold=sample.threshold)
    assert loaded.frames.shape == (4, 699, 512)
    assert loaded.geo_reference is None


def test_scenarios_have_expected_directional_mask_area() -> None:
    def wet_pixels(sample_id: str) -> list[int]:
        output = []
        for frame in get_sample(sample_id).frames:
            with Image.open(BytesIO(frame.png_bytes)) as image:
                output.append(sum(value > 0 for value in image.tobytes()))
        return output

    stable = wet_pixels("nakdong_stable")
    drought = wet_pixels("han_drought")
    flood = wet_pixels("gwangju_flood")
    assert max(stable) - min(stable) < max(stable) * 0.15
    assert drought == sorted(drought, reverse=True)
    assert flood == sorted(flood)
