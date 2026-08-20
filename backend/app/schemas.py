from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WeatherRow(BaseModel):
    """Normalized weather row accepted by every predictor adapter.

    Extra KMA columns are retained so a custom model can consume them without
    requiring changes to the API contract.
    """

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    date: Date | None = None
    kind: Literal["observed", "scenario"] | None = Field(
        default=None,
        description="Historical observation or caller-supplied future scenario",
    )
    station_id: str | None = None
    station_name: str | None = None
    precipitation_mm: float | None = Field(
        default=None, ge=0, description="Daily precipitation in mm"
    )
    temperature_c: float | None = None
    avg_temperature_c: float | None = None
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    humidity_pct: float | None = Field(default=None, ge=0, le=100)
    avg_humidity_percent: float | None = Field(default=None, ge=0, le=100)
    min_humidity_percent: float | None = Field(default=None, ge=0, le=100)
    wind_speed_mps: float | None = Field(default=None, ge=0)
    avg_wind_speed_m_s: float | None = Field(default=None, ge=0)
    max_wind_speed_m_s: float | None = Field(default=None, ge=0)
    max_instant_wind_speed_m_s: float | None = Field(default=None, ge=0)
    avg_local_pressure_hpa: float | None = None
    avg_sea_level_pressure_hpa: float | None = None
    sunshine_hours: float | None = Field(default=None, ge=0)
    solar_radiation_mj_m2: float | None = Field(default=None, ge=0)
    new_snow_cm: float | None = Field(default=None, ge=0)
    weather_description: str | None = None
    source: str | None = None


class WaterLevelConfig(BaseModel):
    """Optional area-to-level calibration used only when a model returns no level.

    ``meters_per_area_ratio`` expresses level change for a 100% area change:
    ``level = baseline_level + coefficient * (area / baseline_area - 1)``.
    """

    baseline_level_m: float
    meters_per_area_ratio: float
    baseline_area_m2: float | None = Field(default=None, gt=0)


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    description: str
    min_frames: int = Field(ge=1)
    supports_weather: bool = False
    produces_water_level: bool = False
    built_in: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class InputSummary(BaseModel):
    frame_count: int
    height: int
    width: int
    source_files: list[str]
    source_dates: list[Date] = Field(default_factory=list)
    threshold: float
    pixel_area_m2: float
    georeferenced: bool = False
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Caller-supplied non-secret provenance such as sample ID, sensor, "
            "processing class, and logical source manifest."
        ),
    )


class PredictionParameters(BaseModel):
    horizon_steps: int
    target_dates: list[Date] = Field(default_factory=list)
    weather: list[WeatherRow] = Field(default_factory=list)
    historical_water_levels_m: list[float | None] = Field(
        default_factory=list,
        description="Optional level observation aligned one-to-one with input frames.",
    )
    reference_water_levels_m: list[float | None] = Field(
        default_factory=list,
        description=(
            "Optional future truth aligned with target_dates. It is used only for "
            "post-prediction evaluation and is never passed to the predictor."
        ),
    )
    evaluation_kind: str | None = None
    evaluation_truth_provenance: str | None = None
    model_options: dict[str, Any] = Field(default_factory=dict)
    water_level_config: WaterLevelConfig | None = None
    caution_pct: float = 5.0
    risk_pct: float = 15.0


class RiskLevel(str, Enum):
    normal = "normal"
    caution = "caution"
    flood_risk = "flood_risk"
    drought_risk = "drought_risk"


class ForecastStep(BaseModel):
    horizon: int = Field(ge=1)
    target_date: Date | None = None
    water_area_pixels: int = Field(ge=0)
    water_area_m2: float = Field(ge=0)
    water_area_km2: float = Field(ge=0)
    area_change_pct: float
    water_level_m: float | None = None
    water_level_change_m: float | None = None
    water_level_source: Literal[
        "adapter_output", "area_level_calibration", "unavailable"
    ] = "unavailable"
    reference_water_level_m: float | None = None
    water_level_error_m: float | None = None
    risk: RiskLevel
    rule_based: Literal[True] = Field(
        default=True,
        description="The risk label is derived from configured area-change rules.",
    )
    mask_npy_url: str
    preview_png_url: str
    mask_tif_url: str


class ArtifactInfo(BaseModel):
    name: str
    kind: Literal["mask", "preview", "raster", "metadata", "bundle"]
    media_type: str
    size_bytes: int = Field(ge=0)
    url: str
    horizon: int | None = None


class WaterLevelEvaluationPair(BaseModel):
    horizon: int = Field(ge=1)
    target_date: Date | None = None
    predicted_water_level_m: float
    reference_water_level_m: float
    residual_m: float
    absolute_error_m: float = Field(ge=0)
    percentage_error_pct: float | None = Field(default=None, ge=0)


class WaterLevelEvaluation(BaseModel):
    status: Literal["available", "unavailable"]
    kind: str
    label: str
    truth_provenance: str
    sample_count: int = Field(ge=0)
    mae_m: float | None = Field(default=None, ge=0)
    rmse_m: float | None = Field(default=None, ge=0)
    mape_pct: float | None = Field(default=None, ge=0)
    r2: float | None = None
    bias_m: float | None = None
    pairs: list[WaterLevelEvaluationPair] = Field(default_factory=list)
    unavailable_reason: str | None = None
    warning: str | None = None


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: Literal["completed", "failed"]
    model_id: str
    model_version: str
    created_at: datetime
    completed_at: datetime
    input: InputSummary
    parameters: PredictionParameters
    steps: list[ForecastStep] = Field(default_factory=list)
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    result_url: str
    artifacts_url: str
    bundle_url: str
    warnings: list[str] = Field(default_factory=list)
    adapter_metadata: dict[str, Any] = Field(default_factory=dict)
    evaluation: WaterLevelEvaluation | None = None
    error: str | None = None


class PredictionList(BaseModel):
    items: list[PredictionResult]
    total: int


class ArtifactList(BaseModel):
    prediction_id: str
    items: list[ArtifactInfo]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str
    models_loaded: int
    result_store: str
    plugin_errors: list[str] = Field(default_factory=list)
    store_errors: list[str] = Field(default_factory=list)


class WeatherObservationRequest(BaseModel):
    station_id: str = Field(min_length=1)
    start_date: Date
    end_date: Date
    source: Literal["asos", "sample"] = "sample"
    service_key: str | None = Field(
        default=None,
        description="Optional per-request KMA key; KMA_API_KEY is preferred for deployments.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> WeatherObservationRequest:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class WeatherObservationResponse(BaseModel):
    station_id: str
    station_name: str | None = None
    source: str
    start_date: Date
    end_date: Date
    rows: list[WeatherRow]
    missing_dates: list[Date] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class WeatherStatusResponse(BaseModel):
    """Public weather capability flags; never exposes an authentication key."""

    asos_available: bool
    server_key_configured: bool
    per_request_key_supported: bool = True
    sample_available: bool = True
    observation_availability: str = "D-1까지의 과거 일 관측"
    documentation_url: str
    setup_env_var: str = "KMA_API_KEY"
