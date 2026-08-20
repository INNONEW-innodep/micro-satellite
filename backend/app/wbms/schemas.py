from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class WbmsSegmentationJobRequest(BaseModel):
    """② detect_water scene request (asynchronous job)."""

    sensor: Literal["iceye", "planet"] = "iceye"
    testbed: Literal["busan", "dcd"] = "busan"
    image_date: str = Field(pattern=r"^\d{8}$", description="Scene date, YYYYMMDD")
    input_tif: str | None = Field(
        default=None,
        description=(
            "Optional host path of the ① preprocessed Pre_*.tif. Defaults to the "
            "handover 05_l1_pre scene for the sensor and date."
        ),
    )
    force: bool = Field(
        default=False,
        description="Skip the cached-mask fast path and rerun the container.",
    )


class WbmsSegmentationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: Literal["cache", "run"]
    output_dir: str
    base: str
    mask_tif: str
    meta_json: str | None = None
    qc_json: str
    products: list[str] = Field(default_factory=list)
    elapsed_s: float | None = None


class WbmsSegmentationJob(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    model_id: str = "wbms-detect-water"
    module: str = "detect_water"
    status: JobStatus
    cached: bool = False
    request: dict[str, Any] = Field(default_factory=dict)
    command: list[str] = Field(default_factory=list)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: WbmsSegmentationResult | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class WbmsSegmentationJobList(BaseModel):
    items: list[WbmsSegmentationJob]
    total: int


class WbmsObservation(BaseModel):
    """One ③ calc_wlwa satellite water-level record to correct."""

    date: str = Field(description="Observation date, YYYYMMDD or YYYY-MM-DD")
    satellite: str = Field(description="iceye|planet (ICEYE/PlanetScope accepted)")
    water_level_m: float = Field(description="③ satellite-derived water level (m)")
    water_area_km2: float | None = Field(default=None, ge=0)

    @field_validator("date")
    @classmethod
    def _normalize_date(cls, value: str) -> str:
        raw = value.strip().replace("-", "")
        if len(raw) != 8 or not raw.isdigit():
            raise ValueError("date must be YYYYMMDD or YYYY-MM-DD")
        return raw

    @field_validator("satellite")
    @classmethod
    def _normalize_satellite(cls, value: str) -> str:
        key = value.strip().lower()
        aliases = {"iceye": "iceye", "planet": "planet", "planetscope": "planet"}
        if key not in aliases:
            raise ValueError("satellite must be one of iceye, planet (ICEYE/PlanetScope)")
        return aliases[key]


class WbmsFusionRequest(BaseModel):
    """④ fusion-LSTM correction request (synchronous).

    Semantics: this corrects the water level **on the observed dates**. The
    bundled model is a same-day corrector, not a forecaster. Optional future
    steps are honest persistence of the last corrected level.
    """

    loc_id: str = Field(min_length=1, description="Gauge id from the bundle loc_index")
    testbed: Literal["busan"] = "busan"
    observations: list[WbmsObservation] = Field(min_length=1, max_length=64)
    persistence_horizon_days: int = Field(
        default=0,
        ge=0,
        le=30,
        description=(
            "Optional number of future days to extend by persistence of the last "
            "corrected level. 0 disables the extension."
        ),
    )

    @field_validator("loc_id")
    @classmethod
    def _lower(cls, value: str) -> str:
        return value.strip().lower()


class WbmsFusionRecord(BaseModel):
    date: str
    satellite: str
    satellite_water_level_m: float
    corrected_water_level_m: float
    offset_m: float
    correction_mode: str
    wl_out_of_train_range: bool = False
    pair_gap_days: int | None = None
    paired: bool | None = None
    water_area_km2: float | None = None


class WbmsPersistenceStep(BaseModel):
    date: str
    corrected_water_level_m: float
    method: Literal["persistence"] = "persistence"


class WbmsWeatherWindow(BaseModel):
    date: str
    window_days: int
    window_start: str
    window_end: str
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    temperature_out_of_train_range: bool = False
    calendar_gaps: int = 0


class WbmsBundleInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    schema_id: str | None = None
    head: str | None = None
    days: int
    wl_valid_range_m: list[float]
    weather_channels: list[str] = Field(default_factory=list)
    train_temperature_range_c: list[float] = Field(default_factory=list)
    lodo_rmse_m: float | None = None
    baseline_rmse_m: float | None = None
    discriminative: bool | None = None
    model_sha256: str | None = None


class WbmsFusionResponse(BaseModel):
    model_id: str = "wbms-fusion-lstm"
    semantics: Literal["same_day_correction"] = "same_day_correction"
    loc_id: str
    testbed: str
    records: list[WbmsFusionRecord]
    persistence_forecast: list[WbmsPersistenceStep] = Field(default_factory=list)
    weather_windows: list[WbmsWeatherWindow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    bundle: WbmsBundleInfo
    runtime: dict[str, Any] = Field(default_factory=dict)


class WbmsStatusResponse(BaseModel):
    docker_image: str
    image_available: bool
    handover_dir: str
    handover_available: bool
    bundle_dir: str
    bundle_available: bool
    aws_csv: str
    aws_csv_available: bool
    jobs_dir: str
    wb_output_dir: str
    cached_scenes: list[dict[str, Any]] = Field(default_factory=list)
