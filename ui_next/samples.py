"""Deterministic demonstration datasets for UI/API smoke tests.

Three scenarios use generated binary masks.  One representative scenario is
recovered at runtime from a repository documentation figure; it is useful for
visual/API demonstrations but is not a substitute for the unavailable raw
raster.  Nothing in this catalog represents a verified water level, a trained
model output, or a scientifically validated forecast.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

from PIL import Image, ImageDraw

MASK_SIZE = (192, 192)
SOURCE_DATES = tuple(date(2025, 8, day) for day in range(1, 5))
TARGET_DATES = tuple(date(2025, 8, day) for day in range(5, 8))
SUPPORTED_DAILY_HORIZONS = (7, 14, 30)
RECOMMENDED_MODEL_ID = "weather-morphology"
RECOVERED_SAMPLE_ID = "busan-doc-recovered"
RECOVERED_FIGURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2_time_series_prediction"
    / "docs"
    / "examples"
    / "02_converted_images.png"
)
RECOVERED_CROP_BOXES = (
    (129, 256, 662, 756),
    (1018, 256, 1551, 756),
    (129, 1059, 662, 1559),
    (1018, 1059, 1551, 1559),
)
RECOVERED_SOURCE_DATES = (
    date(2020, 2, 18),
    date(2020, 3, 12),
    date(2020, 3, 25),
    date(2020, 4, 14),
)
RECOVERED_TARGET_DATES = tuple(date(2020, 4, day) for day in range(15, 18))
# 부산 게이지 실측에서 잰 면적-수위 민감도. data/wbms_runs/fused 의 김해 지점
# 관측 8쌍(면적 km² vs 위성 수위 m)을 상대 면적 변화에 회귀해 얻었다(r=+0.919).
# 지점 ROI 에서 잰 민감도를 전체 AOI 면적에 적용한 근사이므로 검증된 환산식이
# 아니다. 그래도 임의값은 아니어서 measured=True 로 표시한다.
BUSAN_GAUGE_LEVEL_BASELINE_M = 2.0179
BUSAN_GAUGE_LEVEL_PER_AREA_RATIO = 4.6584
BUSAN_GAUGE_SOURCE_KO = (
    "김해 게이지 관측 8쌍(면적-수위) 회귀 · r=+0.919 · data/wbms_runs/fused"
)

NAS_BUSAN_SAMPLE_ID = "busan-nas-water-labels"
NAS_BUSAN_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "nas" / "busan_2020"
NAS_BUSAN_MANIFEST_PATH = NAS_BUSAN_ASSET_DIR / "manifest.json"
NAS_BUSAN_MASK_SIZE = (512, 512)
NAS_ICEYE_SAMPLE_ID = "iceye-nas-water-labels"
NAS_ICEYE_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "nas" / "iceye_2020"
NAS_ICEYE_MANIFEST_PATH = NAS_ICEYE_ASSET_DIR / "manifest.json"
NAS_ICEYE_MASK_SIZE = (512, 699)
DISCLAIMER_KO = (
    "이 화면에서 코드로 만든 합성 마스크·수위·기상 시나리오입니다. 지역명은 "
    "시나리오를 구분하기 위한 이름일 뿐 해당 지역에서 발생한 실제 사건이나 "
    "실측 자료를 뜻하지 않습니다. 학습 모델 출력 또는 운영 예측 결과도 아닙니다."
)
SYNTHETIC_REFERENCE_DISCLAIMER_KO = (
    "일별 기준 수위는 화면과 오차 계산 흐름을 재현하려고 코드로 만든 결정론적 "
    "합성 비교선입니다. 실측 정답이 아니며 모델 검증, 정확도 인증 또는 운영 판단의 "
    "근거로 사용할 수 없습니다."
)


@dataclass(frozen=True, slots=True)
class SampleFrame:
    """One immutable, backend-ready binary PNG observation."""

    name: str
    observed_on: date
    png_bytes: bytes
    water_level_m: float | None
    context_png_bytes: bytes | None = None

    @property
    def content_type(self) -> str:
        return "image/png"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png_bytes).hexdigest()

    @property
    def context_sha256(self) -> str | None:
        if self.context_png_bytes is None:
            return None
        return hashlib.sha256(self.context_png_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class IllustrativeWaterLevelConfig:
    """Area-to-level coefficients supplied only to demonstrate API wiring."""

    baseline_level_m: float
    meters_per_area_ratio: float
    baseline_area_m2: float
    illustrative: bool = True
    measured: bool = False

    def as_api_dict(self) -> dict[str, float]:
        """Return only fields accepted by backend ``WaterLevelConfig``."""

        return {
            "baseline_level_m": self.baseline_level_m,
            "meters_per_area_ratio": self.meters_per_area_ratio,
            "baseline_area_m2": self.baseline_area_m2,
        }


@dataclass(frozen=True, slots=True)
class SampleDataset:
    """Immutable description and payload for one quick-start scenario."""

    sample_id: str
    display_name: str
    region_name: str
    scenario_name: str
    summary_ko: str
    frames: tuple[SampleFrame, ...]
    target_dates: tuple[date, ...]
    weather_rows: tuple[Mapping[str, Any], ...]
    water_level_config: IllustrativeWaterLevelConfig | None
    model_options: Mapping[str, Any]
    recommended_model_id: str = RECOMMENDED_MODEL_ID
    pixel_area_m2: float | None = 9.0
    threshold: float = 0.5
    caution_pct: float = 5.0
    risk_pct: float = 15.0
    synthetic_masks: bool = True
    derived_demo: bool = False
    rule_based_baseline: bool = True
    measured_data: bool = False
    trained_model: bool = False
    recommended: bool = False
    raw_source_available: bool = False
    raw_data_provenance_known: bool = False
    georeferenced: bool = False
    pixel_area_known: bool = False
    area_validation_allowed: bool = False
    water_level_validation_allowed: bool = False
    provenance: str = "generated_in_ui_next.samples"
    weather_provenance: str = "repo_scenario"
    disclaimer_ko: str = DISCLAIMER_KO
    data_classification: str = "deterministic_synthetic_scenario"
    data_classification_ko: str = "코드 생성 합성 시나리오"
    source_description_ko: str = (
        "광주 홍수·한강 가뭄·낙동강 안정이라는 이름과 시나리오 개념은 기존 "
        "pipeline_ui/app.py의 발표용 합성 데모에도 있었습니다. 다만 현재 선택되는 "
        "픽셀 데이터는 ui_next/samples.py가 새로 그리는 결정론적 데모이며, 기존에 "
        "제공받은 지역 실측 파일을 읽은 것이 아닙니다. 기존에 제공받은 지역 실측 "
        "데이터가 아닙니다."
    )
    mask_provenance_ko: str = (
        "ui_next/samples.py의 _mask_png가 PIL 도형으로 192×192 이진 마스크 4장을 "
        "생성합니다. 위성영상이나 1세부 제공 래스터에서 추출한 마스크가 아닙니다."
    )
    weather_provenance_ko: str = (
        "저장소의 2_time_series_prediction/data/weather_ex.csv 예시값 중 7일을 "
        "기본 모양으로 사용하고, 선택한 일별 전망 기간의 미래 강수량을 시나리오별 "
        "값으로 만듭니다. 파일이 없으면 코드 내 고정값을 사용하며 모두 비실측으로 "
        "표시합니다."
    )
    water_level_provenance_ko: str = (
        "수위 숫자와 면적-수위 변환계수는 API 연결과 증감 표시를 확인하려고 코드에 "
        "직접 적은 예시값입니다. 관측소 실측 수위나 검교정된 환산식이 아닙니다."
    )
    synthetic_reference_provenance_ko: str = SYNTHETIC_REFERENCE_DISCLAIMER_KO
    region_label_note_ko: str = (
        "지역명은 안정·가뭄·홍수 형태를 쉽게 구분하기 위한 시나리오 라벨입니다. "
        "그 지역의 실제 과거 사건, 위치 또는 공간 범위를 주장하지 않습니다."
    )
    intended_use_ko: str = (
        "화면 조작, 업로드 형식, API 요청·응답, 그래프와 위험 후처리의 방향성을 "
        "재현 가능하게 시연하는 용도입니다."
    )
    not_suitable_for_ko: str = (
        "실제 홍수·가뭄 사실 확인, 모델 정확도 평가, 지역별 위험판단, 보고서 수치나 "
        "운영 의사결정의 근거로 사용하면 안 됩니다."
    )
    source_assets: tuple[str, ...] = (
        "pipeline_ui/app.py (기존 합성 시나리오 이름·개념)",
        "ui_next/samples.py::_mask_png",
        "2_time_series_prediction/data/weather_ex.csv",
    )
    actual_event_data: bool = False
    region_verified: bool = False
    observation_dates_verified: bool = False
    classification_badge_ko: str = "합성 시나리오"
    sensor_name_ko: str = "센서 없음 · 코드 생성"
    source_manifest_path: str | None = None

    @property
    def source_dates(self) -> tuple[date, ...]:
        return tuple(frame.observed_on for frame in self.frames)

    @property
    def source_date_strings(self) -> tuple[str, ...]:
        return tuple(value.isoformat() for value in self.source_dates)

    @property
    def target_date_strings(self) -> tuple[str, ...]:
        return tuple(value.isoformat() for value in self.target_dates)

    @property
    def historical_water_levels_m(self) -> tuple[float | None, ...]:
        return tuple(frame.water_level_m for frame in self.frames)

    @property
    def frame_preview_bytes(self) -> tuple[bytes, ...]:
        """All source frames as browser-displayable PNG bytes."""

        return tuple(frame.png_bytes for frame in self.frames)

    def preview_bytes(self, frame_index: int = -1) -> bytes:
        """Return one source-frame preview, defaulting to the latest frame."""

        try:
            return self.frames[frame_index].png_bytes
        except IndexError as exc:
            raise IndexError(
                f"frame_index {frame_index} is outside 0..{len(self.frames) - 1}"
            ) from exc

    def input_rows(self) -> tuple[dict[str, Any], ...]:
        """Return fresh row dictionaries matching ``ui_next.app`` input state."""

        return tuple(
            {
                "name": frame.name,
                "date": frame.observed_on.isoformat(),
                "water_level_m": frame.water_level_m,
                "sha256": frame.sha256,
                "size": len(frame.png_bytes),
                "content_type": frame.content_type,
                "content": frame.png_bytes,
            }
            for frame in self.frames
        )

    def weather_payload(self) -> tuple[dict[str, Any], ...]:
        """Return mutable copies suitable for JSON serialization/editing."""

        return tuple(dict(row) for row in self.weather_rows)

    def forecast_dates(self, horizon_days: int) -> tuple[date, ...]:
        """Return a daily forecast calendar beginning after the last input frame.

        The stored three-day ``target_dates`` remain available for compatibility;
        this helper supplies the presentation presets (7, 14, and 30 days) and
        also accepts any positive backend-compatible horizon up to 365 days.
        """

        horizon = _validate_daily_horizon(horizon_days)
        latest = self.source_dates[-1]
        return tuple(latest + timedelta(days=offset) for offset in range(1, horizon + 1))

    def forecast_date_strings(self, horizon_days: int) -> tuple[str, ...]:
        """Return :meth:`forecast_dates` in backend-ready ISO date form."""

        return tuple(value.isoformat() for value in self.forecast_dates(horizon_days))

    def forecast_weather_payload(self, horizon_days: int) -> tuple[dict[str, Any], ...]:
        """Return observed rows plus a deterministic daily future weather scenario.

        The document-derived sample deliberately has no weather provenance and
        therefore returns no rows.  Generated scenarios always mark future rows
        as ``scenario`` / ``built_in_demo`` and never as measurements.
        """

        horizon = _validate_daily_horizon(horizon_days)
        if not self.weather_rows:
            return ()
        rows = _scenario_weather(
            self.sample_id,
            horizon_days=horizon,
            source_dates=self.source_dates,
        )
        return tuple(dict(row) for row in rows)

    def synthetic_reference_levels(self, horizon_days: int) -> tuple[float, ...]:
        """Return a deterministic *demonstration-only* water-level comparison line.

        These values are intentionally separate from model input and are not
        measurements or validation truth.  Samples without an illustrative
        water-level setup return an empty tuple.
        """

        horizon = _validate_daily_horizon(horizon_days)
        config = self.water_level_config
        # 예시 시나리오에만 붙인다. 실자료 샘플이 계수를 갖게 되면서 이 조건을
        # config 존재 여부로 두면 **실제 관측에 가짜 정답 수위**가 생기고 평가
        # 지표가 그 위조값과 비교돼 버린다. 예측 수위 산출과는 별개의 문제다.
        if config is None or not config.illustrative:
            return ()
        return _synthetic_reference_levels(
            self.sample_id,
            horizon,
            baseline_level_m=self.water_level_config.baseline_level_m,
        )

    def synthetic_reference_water_levels_m(
        self, horizon_days: int
    ) -> tuple[float, ...]:
        """Explicit-unit alias for :meth:`synthetic_reference_levels`."""

        return self.synthetic_reference_levels(horizon_days)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "synthetic_masks": self.synthetic_masks,
                "derived_demo": self.derived_demo,
                "rule_based_baseline": self.rule_based_baseline,
                "measured_data": self.measured_data,
                "trained_model": self.trained_model,
                "recommended": self.recommended,
                "raw_source_available": self.raw_source_available,
                "raw_data_provenance_known": self.raw_data_provenance_known,
                "georeferenced": self.georeferenced,
                "pixel_area_known": self.pixel_area_known,
                "area_validation_allowed": self.area_validation_allowed,
                "water_level_validation_allowed": self.water_level_validation_allowed,
                "provenance": self.provenance,
                "weather_provenance": self.weather_provenance,
                "disclaimer_ko": self.disclaimer_ko,
                "data_classification": self.data_classification,
                "data_classification_ko": self.data_classification_ko,
                "source_description_ko": self.source_description_ko,
                "mask_provenance_ko": self.mask_provenance_ko,
                "weather_provenance_ko": self.weather_provenance_ko,
                "water_level_provenance_ko": self.water_level_provenance_ko,
                "synthetic_reference_provenance_ko": (
                    self.synthetic_reference_provenance_ko
                ),
                "region_label_note_ko": self.region_label_note_ko,
                "intended_use_ko": self.intended_use_ko,
                "not_suitable_for_ko": self.not_suitable_for_ko,
                "source_assets": self.source_assets,
                "actual_event_data": self.actual_event_data,
                "region_verified": self.region_verified,
                "observation_dates_verified": self.observation_dates_verified,
                "classification_badge_ko": self.classification_badge_ko,
                "sensor_name_ko": self.sensor_name_ko,
                "source_manifest_path": self.source_manifest_path,
            }
        )


def _mask_png(sample_id: str, frame_index: int) -> bytes:
    image = Image.new("L", MASK_SIZE, color=0)
    draw = ImageDraw.Draw(image)

    if sample_id == "nakdong_stable":
        widths = (15, 16, 15, 16)
        offset = (0, 1, -1, 0)[frame_index]
        main = [
            (x + offset, y)
            for x, y in ((69, -8), (76, 24), (67, 53), (73, 82), (65, 112), (72, 145), (66, 200))
        ]
        branch = [
            (x + offset, y)
            for x, y in ((70, 88), (91, 99), (112, 93), (133, 101))
        ]
        draw.line(main, fill=255, width=widths[frame_index], joint="curve")
        draw.line(branch, fill=255, width=max(5, widths[frame_index] // 2), joint="curve")
        draw.ellipse((56, 127, 82, 151), fill=255)
        draw.ellipse((61, 132, 77, 146), fill=0)
    elif sample_id == "han_drought":
        widths = (34, 29, 24, 19)
        main = [(-8, 104), (26, 97), (59, 105), (92, 91), (126, 99), (158, 88), (200, 93)]
        draw.line(main, fill=255, width=widths[frame_index], joint="curve")
        # Fixed islands make the decreasing channel visually legible.
        draw.ellipse((65, 91, 84, 101), fill=0)
        draw.ellipse((128, 86, 145, 95), fill=0)
    elif sample_id == "gwangju_flood":
        widths = (11, 15, 21, 29)
        main = [(102, -8), (94, 27), (101, 57), (92, 89), (99, 121), (88, 154), (92, 200)]
        tributary = [(18, 69), (48, 73), (70, 86), (96, 91)]
        draw.line(main, fill=255, width=widths[frame_index], joint="curve")
        draw.line(tributary, fill=255, width=max(6, widths[frame_index] // 2), joint="curve")
        if frame_index >= 2:
            radius = 10 + (frame_index - 2) * 9
            draw.ellipse((96 - radius, 112 - radius, 96 + radius, 112 + radius), fill=255)
        if frame_index == 3:
            draw.polygon([(55, 126), (88, 105), (127, 116), (143, 148), (91, 158)], fill=255)
    else:  # pragma: no cover - construction uses only fixed catalog IDs
        raise KeyError(f"unknown sample mask generator: {sample_id}")

    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _recovered_mask_pngs() -> tuple[bytes, ...]:
    """Recover the four plot interiors from the repository documentation image."""

    if not RECOVERED_FIGURE_PATH.is_file():
        raise FileNotFoundError(
            "representative demo figure is missing: "
            f"{RECOVERED_FIGURE_PATH.as_posix()}"
        )
    output: list[bytes] = []
    with Image.open(RECOVERED_FIGURE_PATH) as figure:
        grayscale = figure.convert("L")
        for crop_box in RECOVERED_CROP_BOXES:
            crop = grayscale.crop(crop_box)
            if crop.size != (533, 500):
                raise ValueError(
                    f"recovered crop is {crop.size}, expected the documented (533, 500)"
                )
            binary = crop.point(lambda value: 255 if value >= 128 else 0, mode="L")
            stream = BytesIO()
            binary.save(stream, format="PNG", optimize=False, compress_level=9)
            output.append(stream.getvalue())
    return tuple(output)


def _parse_example_date(raw: str) -> date:
    normalized = raw.strip().replace(" ", "").rstrip(".")
    year, month, day = (int(value) for value in normalized.split(".") if value)
    return date(year, month, day)


def _validate_daily_horizon(horizon_days: int) -> int:
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
        raise TypeError("horizon_days must be an integer")
    if not 1 <= horizon_days <= 365:
        raise ValueError("horizon_days must be between 1 and 365")
    return horizon_days


def _weather_example_rows() -> tuple[dict[str, Any], ...]:
    path = (
        Path(__file__).resolve().parents[1]
        / "2_time_series_prediction"
        / "data"
        / "weather_ex.csv"
    )
    fallback = (
        (1, 55.0, 0.3, 17.0, 27.0, 1020.0, 3.0),
        (2, 58.0, 2.4, 18.0, 28.0, 1020.0, 3.0),
        (3, 75.0, 5.5, 17.0, 27.0, 1018.0, 5.3),
        (4, 70.0, 2.5, 15.0, 25.0, 1015.0, 5.0),
        (5, 78.0, 4.8, 15.0, 26.0, 1010.0, 6.0),
        (6, 85.0, 15.0, 14.0, 25.0, 1005.0, 8.0),
        (7, 78.0, 2.0, 13.0, 25.0, 1010.0, 6.0),
    )
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            parsed = []
            for raw in csv.DictReader(handle):
                observed_on = _parse_example_date(raw["일자(date)"])
                if observed_on > TARGET_DATES[-1]:
                    continue
                minimum = float(raw["tmp_min (°C)"])
                maximum = float(raw["tmp_max (°C)"])
                parsed.append(
                    {
                        "date": observed_on.isoformat(),
                        "timestamp": observed_on.isoformat(),
                        "precipitation_mm": float(raw["precipitation (mm)"]),
                        "temperature_c": (minimum + maximum) / 2.0,
                        "min_temperature_c": minimum,
                        "max_temperature_c": maximum,
                        "humidity_pct": float(raw["humidity (%)"]),
                        "avg_local_pressure_hpa": float(raw["pressure (hPa)"]),
                        "wind_speed_mps": float(raw["wind_speed (m/s)"]),
                    }
                )
            if len(parsed) == 7:
                return tuple(parsed)

    return tuple(
        {
            "date": date(2025, 8, day).isoformat(),
            "timestamp": date(2025, 8, day).isoformat(),
            "precipitation_mm": rain,
            "temperature_c": (minimum + maximum) / 2.0,
            "min_temperature_c": minimum,
            "max_temperature_c": maximum,
            "humidity_pct": humidity,
            "avg_local_pressure_hpa": pressure,
            "wind_speed_mps": wind,
        }
        for day, humidity, rain, minimum, maximum, pressure, wind in fallback
    )


_SCENARIO_RAIN_PATTERNS_MM: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        # The first three values preserve the original quick-start scenario.
        "nakdong_stable": (4.8, 15.0, 2.0, 0.0, 6.5, 1.2, 8.0),
        "han_drought": (0.0,) * 30,
        "gwangju_flood": (
            24.0,
            48.0,
            30.0,
            4.0,
            0.0,
            2.0,
            14.0,
            36.0,
            18.0,
            0.0,
            1.0,
            8.0,
            28.0,
            12.0,
            0.0,
            3.0,
            6.0,
            42.0,
            20.0,
            5.0,
            0.0,
            0.0,
            16.0,
            30.0,
            10.0,
            2.0,
            0.0,
            5.0,
            22.0,
            12.0,
        ),
    }
)

_SCENARIO_DESCRIPTION_KO: Mapping[str, str] = MappingProxyType(
    {
        "nakdong_stable": "주기적 약한 비와 간헐적 보통 비를 가정한 안정 시나리오",
        "han_drought": "전 기간 무강수를 가정한 가뭄 시나리오",
        "gwangju_flood": "초기 집중호우와 후속 강우 펄스를 가정한 홍수 시나리오",
    }
)


def _scenario_rainfall(sample_id: str, horizon_days: int) -> tuple[float, ...]:
    try:
        pattern = _SCENARIO_RAIN_PATTERNS_MM[sample_id]
    except KeyError as exc:
        raise KeyError(f"no generated weather scenario for {sample_id!r}") from exc
    return tuple(pattern[index % len(pattern)] for index in range(horizon_days))


def _scenario_weather(
    sample_id: str,
    horizon_days: int = len(TARGET_DATES),
    source_dates: tuple[date, ...] = SOURCE_DATES,
) -> tuple[Mapping[str, Any], ...]:
    horizon = _validate_daily_horizon(horizon_days)
    rows = _weather_example_rows()
    if not source_dates:
        raise ValueError("source_dates must contain at least one observation date")
    target_dates = tuple(
        source_dates[-1] + timedelta(days=offset)
        for offset in range(1, horizon + 1)
    )
    target_rain = _scenario_rainfall(sample_id, horizon)
    output: list[Mapping[str, Any]] = []
    for index, observed_on in enumerate(source_dates):
        base = rows[index % len(rows)]
        row = dict(base)
        row["date"] = observed_on.isoformat()
        row["timestamp"] = observed_on.isoformat()
        row.update(
            {
                "kind": "observed",
                "source": "built_in_demo",
                "is_measured": False,
                "is_trained_output": False,
                "scenario_id": sample_id,
                "weather_provenance": (
                    "repository_example_csv_or_code_fallback_with_manual_scenario_override"
                ),
            }
        )
        output.append(MappingProxyType(row))

    future_templates = rows[len(source_dates) :] or rows
    for index, (forecast_on, rain_mm) in enumerate(
        zip(target_dates, target_rain, strict=True)
    ):
        base = future_templates[index % len(future_templates)]
        row = dict(base)
        row["date"] = forecast_on.isoformat()
        row["timestamp"] = forecast_on.isoformat()
        row["base_example_precipitation_mm"] = row["precipitation_mm"]
        row["precipitation_mm"] = rain_mm
        row.update(
            {
                "kind": "scenario",
                "source": "built_in_demo",
                "is_measured": False,
                "is_trained_output": False,
                "scenario_id": sample_id,
                "scenario_description_ko": _SCENARIO_DESCRIPTION_KO[sample_id],
                "weather_provenance": (
                    "repository_example_csv_or_code_fallback_with_manual_scenario_override"
                ),
            }
        )
        output.append(MappingProxyType(row))
    return tuple(output)


def _synthetic_reference_levels(
    sample_id: str,
    horizon_days: int,
    *,
    baseline_level_m: float,
) -> tuple[float, ...]:
    """Build a reproducible illustration line, never a validation target."""

    horizon = _validate_daily_horizon(horizon_days)
    if sample_id == "nakdong_stable":
        return tuple(
            round(
                baseline_level_m
                + 0.014 * math.sin((2.0 * math.pi * day) / 7.0)
                + 0.0001 * day,
                3,
            )
            for day in range(1, horizon + 1)
        )
    if sample_id == "han_drought":
        return tuple(
            round(
                max(
                    0.0,
                    baseline_level_m
                    - 0.0085 * day
                    - 0.035 * (1.0 - math.exp(-day / 7.0)),
                ),
                3,
            )
            for day in range(1, horizon + 1)
        )
    if sample_id == "gwangju_flood":
        storage = 0.0
        levels: list[float] = []
        for day, rain_mm in enumerate(_scenario_rainfall(sample_id, horizon), start=1):
            storage = 0.70 * storage + rain_mm / 40.0
            level = baseline_level_m + 0.42 * storage - 0.0025 * day
            levels.append(round(max(0.0, level), 3))
        return tuple(levels)
    raise KeyError(f"no synthetic water-level reference for {sample_id!r}")


def _water_area_m2(png_bytes: bytes, pixel_area_m2: float = 9.0) -> float:
    with Image.open(BytesIO(png_bytes)) as image:
        histogram = image.convert("L").histogram()
    return float(sum(histogram[1:])) * pixel_area_m2


def _dataset(
    sample_id: str,
    display_name: str,
    region_name: str,
    scenario_name: str,
    summary_ko: str,
    levels: tuple[float | None, ...],
    meters_per_area_ratio: float,
    model_options: Mapping[str, Any],
) -> SampleDataset:
    frames = tuple(
        SampleFrame(
            name=f"{sample_id}_{observed_on.isoformat().replace('-', '')}.png",
            observed_on=observed_on,
            png_bytes=_mask_png(sample_id, index),
            water_level_m=levels[index],
        )
        for index, observed_on in enumerate(SOURCE_DATES)
    )
    last_level = next(value for value in reversed(levels) if value is not None)
    return SampleDataset(
        sample_id=sample_id,
        display_name=display_name,
        region_name=region_name,
        scenario_name=scenario_name,
        summary_ko=summary_ko,
        frames=frames,
        target_dates=TARGET_DATES,
        weather_rows=_scenario_weather(sample_id),
        water_level_config=IllustrativeWaterLevelConfig(
            baseline_level_m=last_level,
            meters_per_area_ratio=meters_per_area_ratio,
            baseline_area_m2=_water_area_m2(frames[-1].png_bytes),
        ),
        model_options=MappingProxyType(dict(model_options)),
        provenance=(
            "scenario labels/concepts inherited from synthetic pipeline_ui/app.py demo; "
            "current masks=deterministic_synthetic_scenario:ui_next/samples.py::_mask_png; "
            "weather base=2_time_series_prediction/data/weather_ex.csv or fixed fallback; "
            "future precipitation and illustrative levels are manually configured"
        ),
        weather_provenance=(
            "repository_example_csv_or_code_fallback_with_manual_scenario_override"
        ),
    )


def _nas_busan_manifest() -> dict[str, Any]:
    if not NAS_BUSAN_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "NAS Busan demo manifest is missing: "
            f"{NAS_BUSAN_MANIFEST_PATH.as_posix()}"
        )
    value = json.loads(NAS_BUSAN_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("NAS Busan demo manifest must contain one JSON object")
    return value


def _nas_busan_asset(relative_path: str, expected_sha256: str) -> bytes:
    candidate = (NAS_BUSAN_ASSET_DIR / relative_path).resolve()
    try:
        candidate.relative_to(NAS_BUSAN_ASSET_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"NAS Busan asset path leaves its package directory: {relative_path!r}"
        ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"NAS Busan demo asset is missing: {relative_path}")
    payload = candidate.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"NAS Busan demo asset checksum mismatch for {relative_path!r}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return payload


def _nas_busan_dataset() -> SampleDataset:
    manifest = _nas_busan_manifest()
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != 4:
        raise ValueError("NAS Busan demo manifest must describe exactly four frames")

    manifest_dates = manifest.get("dates")
    if not isinstance(manifest_dates, list):
        raise TypeError("NAS Busan demo manifest dates must be a JSON array")
    declared_dates = tuple(str(value) for value in manifest_dates)
    frames: list[SampleFrame] = []
    source_assets = ["ui_next/assets/nas/busan_2020/manifest.json"]
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise TypeError("NAS Busan demo frame metadata must be JSON objects")
        observed_on = date.fromisoformat(str(raw_frame["date"]))
        mask_path = str(raw_frame["mask"])
        preview_path = str(raw_frame["preview"])
        mask_bytes = _nas_busan_asset(mask_path, str(raw_frame["mask_sha256"]))
        preview_bytes = _nas_busan_asset(
            preview_path, str(raw_frame["preview_sha256"])
        )
        frames.append(
            SampleFrame(
                name=Path(mask_path).name,
                observed_on=observed_on,
                png_bytes=mask_bytes,
                water_level_m=None,
                context_png_bytes=preview_bytes,
            )
        )
        for source_key in ("input_source", "label_source"):
            source = raw_frame.get(source_key)
            if isinstance(source, dict) and source.get("relative_path"):
                source_assets.append(
                    "신규데이터_위성영상_라벨링/0_Busan/"
                    + str(source["relative_path"])
                )

    frame_dates = tuple(frame.observed_on.strftime("%Y%m%d") for frame in frames)
    if frame_dates != declared_dates:
        raise ValueError(
            "NAS Busan demo manifest frame dates do not match its declared dates"
        )
    if tuple(frame.observed_on for frame in frames) != tuple(
        sorted(frame.observed_on for frame in frames)
    ):
        raise ValueError("NAS Busan demo frames must be in ascending date order")

    preview_shape = manifest.get("preview_shape")
    if preview_shape != [NAS_BUSAN_MASK_SIZE[1], NAS_BUSAN_MASK_SIZE[0]]:
        raise ValueError(
            "NAS Busan demo manifest preview_shape must be [512, 512]"
        )
    pixel_area_m2 = float(manifest["preview_pixel_area_m2"])
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("NAS Busan demo preview_pixel_area_m2 must be positive")

    last_date = frames[-1].observed_on
    common_bounds = manifest.get("common_bounds_utm")
    return SampleDataset(
        sample_id=NAS_BUSAN_SAMPLE_ID,
        display_name="부산 NAS 수체 라벨 2020",
        region_name="부산",
        scenario_name="실제 위성 관측 라벨",
        summary_ko=(
            "NAS에 제공된 2020년 부산 4시점 광학영상과 이진 수체 라벨을 같은 "
            "512×512 공통 격자로 만든 연동 샘플입니다."
        ),
        frames=tuple(frames),
        target_dates=tuple(last_date + timedelta(days=day) for day in range(1, 4)),
        weather_rows=(),
        # 수위는 백엔드 기준선이 스스로 내지 않는다. 실측 게이지에서 잰 계수를
        # 주어야 결과 화면의 '예측 수위'가 미산출로 비지 않는다.
        water_level_config=IllustrativeWaterLevelConfig(
            baseline_level_m=BUSAN_GAUGE_LEVEL_BASELINE_M,
            meters_per_area_ratio=BUSAN_GAUGE_LEVEL_PER_AREA_RATIO,
            baseline_area_m2=sum(
                _water_area_m2(frame.png_bytes, pixel_area_m2) for frame in frames
            ) / len(frames),
            illustrative=False,
            measured=True,
        ),
        model_options=MappingProxyType(
            {
                "max_daily_area_change_pct": 1.0,
                "max_total_area_change_pct": 25.0,
                "rainfall_response_pct_per_20mm": 0.2,
                "rainfall_memory_decay": 0.65,
            }
        ),
        recommended_model_id="irregular-area-trend",
        pixel_area_m2=pixel_area_m2,
        threshold=0.5,
        synthetic_masks=False,
        derived_demo=False,
        measured_data=False,
        trained_model=False,
        recommended=True,
        raw_source_available=True,
        raw_data_provenance_known=True,
        georeferenced=False,
        pixel_area_known=True,
        area_validation_allowed=False,
        water_level_validation_allowed=False,
        provenance=(
            f"manifest={manifest.get('dataset_id')}; "
            f"classification={manifest.get('classification')}; "
            f"source_crs={manifest.get('crs')}; "
            f"source_pixel_area_m2={manifest.get('source_pixel_area_m2')}; "
            f"preview_pixel_area_m2={pixel_area_m2}; common_bounds={common_bounds}"
        ),
        weather_provenance="runtime_kma_asos_historical_replay",
        disclaimer_ko=(
            "실제 제공 위성영상과 대응 라벨에서 만든 샘플이지만 512×512로 공간 "
            "리샘플링한 발표·연동용 자산입니다. 면적은 manifest의 리샘플 격자 픽셀 "
            "면적으로 계산한 근사값이며, 실측 수위·미래 정답·검증된 예측 성능은 없습니다."
        ),
        data_classification=str(manifest.get("classification")),
        data_classification_ko="실제 위성영상 대응 파생 이진 수체 라벨",
        source_description_ko=(
            "NAS의 신규데이터_위성영상_라벨링/0_Busan에서 날짜가 일치하는 input_busan "
            "광학영상과 label_busan 수체 라벨을 가져왔습니다. 센서명은 원본 파일명과 "
            "날짜 대응으로 판단한 'PlanetScope 계열 추정'이며 확정 식별값이 아닙니다."
        ),
        mask_provenance_ko=(
            "원본 공통영역의 0=비수체, 1=수체 라벨을 최근접 방식으로 512×512에 "
            "리샘플링한 PNG입니다. API에는 화면에 보이는 동일한 이진 PNG 네 장이 "
            "날짜순으로 전달됩니다."
        ),
        weather_provenance_ko=(
            "빠른 실행 시 부산 ASOS 159 지점에서 입력 관측일과 목표일의 과거 일자료를 "
            "조회합니다. 이는 2020년 실제 관측 재현이며 미래 기상예보가 아닙니다. "
            "조회가 실패하면 기상 보정 없이 면적 추세만 사용합니다."
        ),
        water_level_provenance_ko=(
            "관측소 실측 수위와 면적-수위 보정식이 없으므로 네 입력 프레임의 수위는 "
            "모두 null이며 합성 수위도 만들지 않습니다."
        ),
        region_label_note_ko=(
            "부산 표기는 제공 폴더명과 EPSG:32652 공통영역 계보를 따릅니다. 특정 홍수 "
            "사건이나 관측소 수위 자료를 뜻하지 않습니다."
        ),
        intended_use_ko=(
            "실제 제공 영상·라벨의 날짜 정렬, API 업로드, 수체 픽셀·근사 면적 변화와 "
            "모델 교체 흐름을 시연하는 용도입니다."
        ),
        not_suitable_for_ko=(
            "원본 해상도 학습, 경계 정확도 평가, 절대 수위 산출, 미래 정답 기반 RMSE "
            "또는 운영 홍수 판단의 근거로 사용하면 안 됩니다."
        ),
        source_assets=tuple(source_assets),
        actual_event_data=False,
        region_verified=True,
        observation_dates_verified=True,
        classification_badge_ko="NAS 실자료 · 파생 수체 라벨",
        sensor_name_ko="PlanetScope 계열 추정",
        source_manifest_path="ui_next/assets/nas/busan_2020/manifest.json",
    )


def _nas_iceye_asset(relative_path: str, expected_sha256: str) -> bytes:
    candidate = (NAS_ICEYE_ASSET_DIR / relative_path).resolve()
    try:
        candidate.relative_to(NAS_ICEYE_ASSET_DIR.resolve())
    except ValueError as exc:
        raise ValueError("NAS ICEYE asset path must remain inside its asset directory") from exc
    payload = candidate.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"NAS ICEYE asset checksum mismatch for {relative_path!r}: {actual}"
        )
    return payload


def _nas_iceye_dataset() -> SampleDataset:
    """Load the actual, common-grid ICEYE label sequence materialization."""

    if not NAS_ICEYE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"NAS ICEYE demo manifest is missing: {NAS_ICEYE_MANIFEST_PATH.as_posix()}"
        )
    manifest = json.loads(NAS_ICEYE_MANIFEST_PATH.read_text(encoding="utf-8"))
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 4:
        raise ValueError("NAS ICEYE demo manifest must contain exactly four items")
    common_grid = manifest.get("common_grid")
    if not isinstance(common_grid, dict):
        raise TypeError("NAS ICEYE demo common_grid must be a JSON object")
    preview_width = int(common_grid.get("preview_width", 0))
    preview_height = int(common_grid.get("preview_height", 0))
    if (preview_width, preview_height) != NAS_ICEYE_MASK_SIZE:
        raise ValueError("NAS ICEYE demo preview grid must be 512×699")
    transform = common_grid.get("transform")
    if not isinstance(transform, list) or len(transform) != 6:
        raise ValueError("NAS ICEYE demo transform must contain six values")
    pixel_area_m2 = abs(float(transform[0]) * float(transform[4]))
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("NAS ICEYE demo pixel area must be positive")

    frames: list[SampleFrame] = []
    source_assets = ["ui_next/assets/nas/iceye_2020/manifest.json"]
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            raise TypeError(f"NAS ICEYE item {index} must be a JSON object")
        observed_on = date.fromisoformat(str(raw_item["date"]))
        assets = raw_item.get("assets")
        source = raw_item.get("source")
        if not isinstance(assets, dict) or not isinstance(source, dict):
            raise TypeError("NAS ICEYE item assets/source must be JSON objects")
        mask = assets.get("mask")
        if not isinstance(mask, dict):
            raise TypeError("NAS ICEYE mask asset must be a JSON object")
        mask_file = str(mask["file"])
        mask_bytes = _nas_iceye_asset(mask_file, str(mask["sha256"]))
        frames.append(
            SampleFrame(
                name=mask_file,
                observed_on=observed_on,
                png_bytes=mask_bytes,
                water_level_m=None,
                context_png_bytes=None,
            )
        )
        for source_key in ("input", "label", "raw_quicklook", "raw_grd_metadata"):
            entry = source.get(source_key)
            if isinstance(entry, dict) and entry.get("logical_path"):
                source_assets.append(str(entry["logical_path"]))

    dates = tuple(frame.observed_on for frame in frames)
    if dates != tuple(sorted(dates)) or len(dates) != len(set(dates)):
        raise ValueError("NAS ICEYE demo dates must be sorted and unique")
    temporal = manifest.get("temporal_coverage")
    declared_dates = tuple(
        date.fromisoformat(str(value))
        for value in (temporal.get("dates", []) if isinstance(temporal, dict) else [])
    )
    if declared_dates != dates:
        raise ValueError("NAS ICEYE item dates do not match temporal_coverage")

    last_date = dates[-1]
    return SampleDataset(
        sample_id=NAS_ICEYE_SAMPLE_ID,
        display_name="ICEYE NAS SAR 수체 라벨 2020",
        region_name="부산권",
        scenario_name="실제 SAR 대응 수체 라벨",
        summary_ko=(
            "NAS ICEYE_WB의 날짜별 입력–라벨 4쌍을 공통 지도영역의 512×699 "
            "격자로 정렬한 실제 이진 수체 라벨 연동 샘플입니다."
        ),
        frames=tuple(frames),
        target_dates=tuple(last_date + timedelta(days=day) for day in range(1, 4)),
        weather_rows=(),
        water_level_config=None,
        model_options=MappingProxyType(
            {
                "max_daily_area_change_pct": 1.0,
                "max_total_area_change_pct": 25.0,
                "rainfall_response_pct_per_20mm": 0.2,
                "rainfall_memory_decay": 0.65,
            }
        ),
        recommended_model_id="irregular-area-trend",
        pixel_area_m2=pixel_area_m2,
        threshold=0.5,
        synthetic_masks=False,
        derived_demo=False,
        measured_data=False,
        trained_model=False,
        recommended=False,
        raw_source_available=True,
        raw_data_provenance_known=True,
        georeferenced=False,
        pixel_area_known=True,
        area_validation_allowed=False,
        water_level_validation_allowed=False,
        provenance=(
            f"manifest={manifest.get('dataset_id')}; classification={manifest.get('classification')}; "
            f"source_crs={common_grid.get('crs')}; common_bounds={common_grid.get('bounds')}; "
            f"preview_pixel_area_m2={pixel_area_m2}"
        ),
        weather_provenance="runtime_kma_asos_historical_replay",
        disclaimer_ko=(
            "실제 ICEYE 대응 수체 라벨에서 만든 공통격자 시연셋이지만 4시점뿐이고 "
            "간격이 28·16·1일로 불규칙합니다. browse 이미지는 마스크와 정렬되지 않은 "
            "표시용이며, 실측 수위·미래 수체 정답·검증된 예측 성능은 없습니다."
        ),
        data_classification=str(manifest.get("classification")),
        data_classification_ko="실제 ICEYE SAR 대응 파생 이진 수체 라벨",
        source_description_ko=(
            "NAS ICEYE_WB의 2020-03-02, 03-30, 04-15, 04-16 input–label pair를 "
            "읽기 전용으로 검사했습니다. 각 날짜 pair는 EPSG:32652·3 m에서 일치하지만 "
            "날짜 간 footprint와 원점이 달라 map-space 공통격자로 변환했습니다."
        ),
        mask_provenance_ko=(
            "원본 uint8 라벨의 실제 전체 값 0=배경, 1=수체를 확인하고 네 날짜의 "
            "공통 물리영역을 최근접 방식으로 512×699 PNG(0/255)에 샘플링했습니다."
        ),
        weather_provenance_ko=(
            "빠른 실행 시 부산 ASOS 159의 과거 일자료를 역사 재현 forcing으로 조회합니다. "
            "실제 과거 관측이지만 미래 기상예보가 아닙니다."
        ),
        water_level_provenance_ko=(
            "같은 AOI의 관측소 수위나 검교정된 면적–수위 관계가 없어 모든 수위는 null입니다."
        ),
        region_label_note_ko=(
            "ICEYE 원천 메타데이터의 중심이 부산권에 있으나 특정 홍수 사건이나 수위 "
            "관측소를 의미하지 않습니다."
        ),
        intended_use_ko=(
            "SAR pair·라벨 값·공통격자·면적 변화를 EDA하고, 여러 날짜를 실제로 쓰는 "
            "교체형 기준선 API를 시연하는 용도입니다."
        ),
        not_suitable_for_ko=(
            "일별 학습 모델 성능, 절대 수위, 원시 SAR 밝기 변화, 운영 홍수 경보의 "
            "근거로 사용하면 안 됩니다."
        ),
        source_assets=tuple(source_assets),
        actual_event_data=False,
        region_verified=True,
        observation_dates_verified=True,
        classification_badge_ko="NAS 실자료 · ICEYE 파생 수체 라벨",
        sensor_name_ko="ICEYE-X5 Stripmap VV",
        source_manifest_path="ui_next/assets/nas/iceye_2020/manifest.json",
    )


def _recovered_dataset() -> SampleDataset:
    frames = tuple(
        SampleFrame(
            name=f"busan_doc_recovered_{observed_on.isoformat().replace('-', '')}.png",
            observed_on=observed_on,
            png_bytes=png_bytes,
            water_level_m=None,
        )
        for observed_on, png_bytes in zip(
            RECOVERED_SOURCE_DATES, _recovered_mask_pngs(), strict=True
        )
    )
    return SampleDataset(
        sample_id=RECOVERED_SAMPLE_ID,
        display_name="부산 문서 그림 복원",
        region_name="부산",
        scenario_name="문서 그림 복원",
        summary_ko=(
            "저장소 문서의 4개 수체 예시 패널을 복원한 대표 연동 샘플입니다. "
            "원본 래스터를 대체하지 않습니다."
        ),
        frames=frames,
        target_dates=RECOVERED_TARGET_DATES,
        weather_rows=(),
        water_level_config=None,
        model_options=MappingProxyType({}),
        recommended_model_id="irregular-area-trend",
        pixel_area_m2=None,
        synthetic_masks=False,
        derived_demo=True,
        recommended=False,
        raw_source_available=False,
        raw_data_provenance_known=False,
        georeferenced=False,
        pixel_area_known=False,
        area_validation_allowed=False,
        water_level_validation_allowed=False,
        provenance=(
            "derived_demo:2_time_series_prediction/docs/examples/"
            "02_converted_images.png; threshold=128; raw raster provenance unknown"
        ),
        weather_provenance="none",
        disclaimer_ko=(
            "문서 그림에서 threshold 128로 복원한 derived_demo입니다. 원본 래스터, "
            "지리참조, 픽셀 면적, 실측 수위가 없으므로 면적·수위 정확도 검증에 "
            "사용하면 안 됩니다. 학습 모델 결과도 아닙니다."
        ),
        data_classification="derived_document_figure_demo",
        data_classification_ko="저장소 문서 그림 파생 데모",
        source_description_ko=(
            "기존 저장소 문서에 있던 02_converted_images.png의 그래프 내부 4개를 "
            "잘라 UI 입력으로 복원했습니다. 원본 부산 래스터는 현재 저장소에 없어 "
            "그림의 생성 원천과 공간 정확도를 이 UI에서 검증할 수 없습니다."
        ),
        mask_provenance_ko=(
            "2_time_series_prediction/docs/examples/02_converted_images.png의 고정 좌표를 "
            "533×500으로 자른 뒤 밝기 128을 기준으로 흑백 이진화했습니다. 원본 픽셀 "
            "배열을 읽은 것이 아니라 문서용 그림에서 재추출한 값입니다."
        ),
        weather_provenance_ko="이 샘플에는 기상 행을 연결하지 않았습니다.",
        water_level_provenance_ko="실측 또는 예시 수위를 넣지 않았으며 네 프레임 모두 값이 없습니다.",
        region_label_note_ko=(
            "문서와 파일명이 부산 데이터셋을 지칭하지만, 원본 래스터·메타데이터가 "
            "없으므로 실제 위치·공간 범위·관측 품질은 검증되지 않았습니다."
        ),
        intended_use_ko=(
            "저장소에 남아 있는 대표적인 수체 모양으로 업로드·미리보기·persistence "
            "API 흐름을 시연하는 용도입니다."
        ),
        not_suitable_for_ko=(
            "면적·수위 정확도, 지리적 위치, 원본 전처리 품질 또는 학습 모델 성능을 "
            "검증하는 자료로 사용하면 안 됩니다."
        ),
        source_assets=(
            "2_time_series_prediction/docs/examples/02_converted_images.png",
            "2_time_series_prediction/generate_readme_examples.py",
        ),
        actual_event_data=False,
        region_verified=False,
        observation_dates_verified=False,
        classification_badge_ko="문서 그림 파생 데모",
        sensor_name_ko="원 센서 확인 불가",
    )


_SAMPLES = (
    _recovered_dataset(),
    _nas_busan_dataset(),
    _nas_iceye_dataset(),
    _dataset(
        "nakdong_stable",
        "낙동강 안정 예시",
        "낙동강",
        "안정",
        "코드로 그린 폭 변화가 작은 하천입니다. 낙동강 실측이 아닌 합성 정상 시나리오로 연동 흐름을 확인합니다.",
        (2.10, 2.11, None, 2.12),
        0.20,
        {
            "rain_mm_per_dilation": 20.0,
            "dry_threshold_mm": 0.1,
            "dry_days_per_erosion": 3,
            "max_iterations_per_step": 2,
        },
    ),
    _dataset(
        "han_drought",
        "한강 가뭄 예시",
        "한강",
        "가뭄",
        "코드로 그린 점차 좁아지는 수체와 무강수 조건입니다. 한강의 실제 가뭄 자료가 아닌 합성 감소 시나리오입니다.",
        (1.82, 1.72, 1.61, 1.48),
        0.35,
        {
            "rain_mm_per_dilation": 20.0,
            "dry_threshold_mm": 0.1,
            "dry_days_per_erosion": 1,
            "max_iterations_per_step": 2,
        },
    ),
    _dataset(
        "gwangju_flood",
        "광주 홍수 예시",
        "광주",
        "홍수",
        "코드로 그린 확장 수체와 고강수 조건입니다. 광주의 실제 홍수 자료가 아닌 합성 증가·위험 후처리 시나리오입니다.",
        (0.92, 1.03, 1.28, 1.66),
        0.55,
        {
            "rain_mm_per_dilation": 12.0,
            "dry_threshold_mm": 0.1,
            "dry_days_per_erosion": 3,
            "max_iterations_per_step": 3,
        },
    ),
)
_CATALOG: Mapping[str, SampleDataset] = MappingProxyType(
    {sample.sample_id: sample for sample in _SAMPLES}
)


def sample_catalog() -> Mapping[str, SampleDataset]:
    """Return the immutable sample-id-to-dataset catalog."""

    return _CATALOG


def list_samples() -> tuple[SampleDataset, ...]:
    """Return samples in stable product-card order."""

    return _SAMPLES


def get_sample(sample_id: str) -> SampleDataset:
    """Look up one sample with an actionable error for unknown IDs."""

    try:
        return _CATALOG[sample_id]
    except KeyError as exc:
        available = ", ".join(_CATALOG)
        raise KeyError(f"unknown sample_id {sample_id!r}; available: {available}") from exc


def frame_preview_bytes(sample_id: str, frame_index: int = -1) -> bytes:
    """Convenience helper used by product cards and frame selectors."""

    return get_sample(sample_id).preview_bytes(frame_index)


__all__ = [
    "DISCLAIMER_KO",
    "MASK_SIZE",
    "NAS_BUSAN_ASSET_DIR",
    "NAS_BUSAN_MANIFEST_PATH",
    "NAS_BUSAN_MASK_SIZE",
    "NAS_BUSAN_SAMPLE_ID",
    "NAS_ICEYE_ASSET_DIR",
    "NAS_ICEYE_MANIFEST_PATH",
    "NAS_ICEYE_MASK_SIZE",
    "NAS_ICEYE_SAMPLE_ID",
    "RECOMMENDED_MODEL_ID",
    "RECOVERED_CROP_BOXES",
    "RECOVERED_FIGURE_PATH",
    "RECOVERED_SAMPLE_ID",
    "RECOVERED_SOURCE_DATES",
    "RECOVERED_TARGET_DATES",
    "SOURCE_DATES",
    "SUPPORTED_DAILY_HORIZONS",
    "SYNTHETIC_REFERENCE_DISCLAIMER_KO",
    "TARGET_DATES",
    "IllustrativeWaterLevelConfig",
    "SampleDataset",
    "SampleFrame",
    "frame_preview_bytes",
    "get_sample",
    "list_samples",
    "sample_catalog",
]
