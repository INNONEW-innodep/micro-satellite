from __future__ import annotations

from datetime import date as Date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ForecasterDescription(BaseModel):
    id: str
    name: str
    description: str
    name_ko: str | None = None  # 모델 안내문(MODEL_GUIDE)에 한국어 이름이 있으면 그것
    is_baseline: bool
    is_reference: bool
    hyperparameter_grid: list[dict[str, Any]]
    min_history_days: int
    built_in: bool = True
    uses_future_exog: bool = False


class ExperimentsResponse(BaseModel):
    region_id: str
    items: dict[str, dict[str, Any] | None]
    sources: dict[str, str | None] = {}
    note_ko: str = ""


class StationInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    source: str = ""
    source_code: str = ""
    terrain: str = ""
    terrain_note: str = ""
    unit: str = "m"
    n_days: int = 0
    period: list[str] | None = None
    satellite_location: str | None = None
    model_trained: bool = False


class RegionSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    origin: str
    crs: str = ""
    weather: dict[str, Any] = Field(default_factory=dict)
    satellite_available: bool
    satellite: dict[str, Any] = Field(default_factory=dict)
    gauge_available: bool
    stations: list[StationInfo] = Field(default_factory=list)
    n_stations: int = 0
    n_stations_trained: int = 0
    forecast: dict[str, Any] = Field(default_factory=dict)


class RegionList(BaseModel):
    items: list[RegionSummary]
    total: int
    builtin_dir: str
    user_dir: str


class RegionCreateRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64, description="영문·숫자·'_-.' 만")
    name: str = Field(min_length=1)
    description: str = ""
    crs: str = ""
    bbox_wgs84: list[float] = Field(default_factory=list)
    weather: dict[str, Any] = Field(default_factory=dict)
    satellite: dict[str, Any] = Field(default_factory=dict)
    gauge: dict[str, Any] = Field(default_factory=dict)
    forecast: dict[str, Any] = Field(default_factory=dict)
    overwrite: bool = False


class SeriesRow(BaseModel):
    date: str
    water_level_m: float


class SeriesUploadRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(description="[{date, water_level_m}] — 열 이름은 date/obs_date/ymd, water_level_m/level/wl/value 허용")
    replace: bool = Field(default=False, description="true면 기존 지점 시계열을 버리고 새 행만 남긴다")
    station: dict[str, Any] = Field(default_factory=dict, description="name·source·source_code·terrain 등 지점 메타")


class SeriesResponse(BaseModel):
    region_id: str
    station: StationInfo
    rows: list[SeriesRow]
    n: int


class SplitSpec(BaseModel):
    """시간 분할 명세 — 학습(≤ train_end) → 검증 → 시험. 날짜는 ISO(YYYY-MM-DD).

    순서 `train_end < val_start ≤ val_end < test_start ≤ test_end` 와 최소 길이(학습 730일·검증 90일·
    시험 30일), 지역 시계열 범위 안쪽인지의 검증은 요청을 받은 라우터에서 한다(422 + 한국어 사유).
    """

    model_config = ConfigDict(extra="forbid")

    train_end: Date = Field(description="학습 구간 마지막 날(포함)")
    val_start: Date = Field(description="검증 구간 첫 날")
    val_end: Date = Field(description="검증 구간 마지막 날(포함)")
    test_start: Date = Field(description="시험 구간 첫 날")
    test_end: Date = Field(description="시험 구간 마지막 날(포함)")

    def to_mapping(self) -> dict[str, str]:
        return {
            "train_end": self.train_end.isoformat(),
            "val_start": self.val_start.isoformat(),
            "val_end": self.val_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


class TrainRequest(BaseModel):
    stations: list[str] | None = Field(default=None, description="비우면 지역의 모든 지점")
    forecasters: list[str] | None = Field(
        default=None,
        description="후보로 허용할 예측기 id. 비우면 기준선(persistence)을 뺀 전부. 모델 교체 실험에 쓴다.",
    )
    horizon_max: int | None = Field(default=None, ge=1, le=90)
    split: SplitSpec | None = Field(
        default=None,
        description="사용자 지정 시간 분할. 비우면 지역 기본값(region.json forecast.split, 없으면 마지막 2년을 검증·시험).",
    )
    async_run: bool = Field(
        default=False,
        description="true 면 즉시 202 + job_id 를 돌려주고 백그라운드 스레드에서 학습한다. GET .../gauge/train/jobs/{job_id} 로 진행률·결과 조회.",
    )


class TrainJobProgress(BaseModel):
    stations_total: int = 0
    stations_done: int = 0
    current_station: str | None = None


class TrainJobAccepted(BaseModel):
    job_id: str
    status: str
    region_id: str
    split_used: dict[str, str]
    stations: list[str]
    poll_url: str = ""


class TrainJobStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    region_id: str
    status: str = Field(description="queued | running | done | failed")
    progress: TrainJobProgress = Field(default_factory=TrainJobProgress)
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float | None = None
    split_used: dict[str, str] = Field(default_factory=dict)
    custom_split: bool = False
    stations: list[str] = Field(default_factory=list)
    forecasters: list[str] | None = None
    horizon_max: int | None = None
    error: str | None = None
    evaluation: dict[str, Any] | None = None


class TrainJobList(BaseModel):
    items: list[TrainJobStatus]


class ModelsResetResponse(BaseModel):
    region_id: str
    removed_files: list[str]
    active: str = Field(description="되돌린 뒤 활성 산출물의 출처 — 항상 'builtin'")
    artifacts_dir: str = ""


class EvaluationRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    station_id: str
    station_name: str
    terrain: str = ""
    horizon: int
    forecaster_id: str
    forecaster_name: str | None = None
    hparams: dict[str, Any] | None = None
    val_rmse: float | None = None
    test_n: int | None = None
    test_mse: float | None = None
    test_rmse: float | None = None
    test_mae: float | None = None
    persistence_test_rmse: float | None = None
    persistence_test_mse: float | None = None
    climatology_test_rmse: float | None = None
    skill_vs_persistence: float | None = None


class EvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_: str = Field(alias="schema", default="")
    region_id: str
    generated_at: str = ""
    split: dict[str, Any] = Field(default_factory=dict)
    n_stations: int = 0
    rows: list[EvaluationRow] = Field(default_factory=list)
    winners_by_terrain: dict[str, dict[str, int]] = Field(default_factory=dict)
    cells_total: int = 0
    cells_beating_persistence: int = 0
    note_ko: str = ""
    trained_now: list[str] = Field(default_factory=list)
    failures: dict[str, str] = Field(default_factory=dict)
    allowed_forecasters: list[str] | None = None
    custom_split: bool = Field(default=False, description="사용자 지정 분할로 학습한 산출물이면 true")
    trained_at: str | None = Field(default=None, description="evaluation.json 의 generated_at")
    artifacts_origin: str = Field(default="", description="활성 산출물 위치 — 'user' 또는 'builtin'")


class ForecastStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    horizon: int
    target_date: str
    forecast_m: float | None
    forecaster_id: str
    persistence_reference_m: float
    sigma_m: float | None = None
    lower_1sigma_m: float | None = None
    upper_1sigma_m: float | None = None
    lower_2sigma_m: float | None = None
    upper_2sigma_m: float | None = None
    q05_m: float | None = None
    q95_m: float | None = None
    observed_m: float | None = None
    error_m: float | None = None


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    region_id: str
    station: StationInfo
    as_of: str
    requested_as_of: str | None = Field(default=None, description="요청한 원점(as_of). 비우면 None")
    as_of_clamped: bool = Field(default=False, description="요청한 원점이 자료 마지막 날보다 뒤라 마지막 날로 맞췄는지")
    data_period: list[str] | None = Field(default=None, description="이 지점 수위 자료 기간 [시작, 끝]")
    observed_days_after: int = Field(default=0, description="원점 뒤 실제 관측이 있어 비교한 날 수")
    origin_note_ko: str | None = Field(default=None, description="원점을 맞춘 사정(한국어). 맞추지 않았으면 None")
    origin_level_m: float
    origin_observed: bool
    horizon: int
    params_used: str
    params_note_ko: str
    band_note_ko: str
    steps: list[ForecastStep]
    history: list[dict[str, Any]]
    verification: dict[str, Any] | None = None
    models_by_horizon: dict[str, str] = Field(default_factory=dict)


class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    region_id: str
    station: dict[str, Any]
    horizon: int
    split: dict[str, Any]
    forecaster_id: str
    forecaster_name: str | None = None
    hparams: dict[str, Any] | None = None
    test: dict[str, Any]
    test_baselines: dict[str, Any]
    skill_vs_persistence: float | None = None
    trace: dict[str, list[Any]]


class ForecastQuery(BaseModel):
    as_of: Date | None = None
    horizon: int | None = Field(default=None, ge=1, le=90)
    history_days: int = Field(default=120, ge=7, le=3660)
