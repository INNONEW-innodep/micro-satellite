"""장면 파이프라인 API 응답 모델.

화면(UI)이 이 이름들에 맞춰 만들어지므로 키 이름은 계약이다 — 바꾸지 말고 더하기만 한다.
``extra="allow"`` 를 둔 모델은 내부에서 붙인 부가 정보(note·testbed·options 등)를 그대로 흘려보낸다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StepStatus = Literal["pending", "running", "done", "skipped", "failed"]
PipelineStatus = Literal["queued", "running", "done", "failed"]


class SceneStep(BaseModel):
    key: str = Field(description="register|detect|area|levels|predict|deliver (고정 순서)")
    label: str = Field(description="화면 표시용 한국어 이름")
    status: StepStatus
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float | None = None
    detail: str | None = Field(default=None, description="짧은 결과 설명(산출 파일명·면적 등) 또는 건너뛴/실패한 사유")


class SceneOutputs(BaseModel):
    model_config = ConfigDict(extra="allow")

    quicklook_url: str | None = None
    overlay_url: str | None = None
    mask_url: str | None = Field(default=None, description="원해상도 수체 마스크 GeoTIFF(WB_*.tif)")
    product_url: str | None = Field(default=None, description="규약 파일명 {product_type}_{product_id}.tif")
    product_name: str | None = None
    water_area_km2: float | None = None
    area_series_url: str | None = None
    levels_url: str | None = None
    prediction_id: str | None = None
    prediction_url: str | None = None
    folder: str | None = Field(default=None, description="장면 산출 폴더(backend/data 기준 상대경로)")


class SceneLiaison(BaseModel):
    order_id: str | None = None
    scene_id: str | None = None
    work_id: str | None = None
    algm_id: str | None = None
    product_type: str | None = None
    product_id: str | None = None


class SceneStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    scene_id: str
    region_id: str
    sensor: str
    image_date: str = Field(description="YYYY-MM-DD")
    pipeline_id: str | None = Field(default=None, description="파이프라인 기록이 없는 기존 장면은 null")
    status: PipelineStatus
    progress: float = Field(ge=0, le=1)
    steps: list[SceneStep]
    outputs: SceneOutputs
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str | None = None
    finished_at: str | None = None
    liaison: SceneLiaison


class SceneSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    scene_id: str
    region_id: str
    sensor: str
    image_date: str
    status: PipelineStatus
    progress: float
    created_at: str | None = None
    finished_at: str | None = None
    water_area_km2: float | None = None
    product_name: str | None = None
    pipeline_id: str | None = None


class SceneList(BaseModel):
    items: list[SceneSummary]
    total: int


class SceneAccepted(BaseModel):
    scene_id: str
    pipeline_id: str
    status_url: str


class SceneOptionRegion(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    testbed: str | None = None
    sensors: list[str] = Field(default_factory=list)
    has_gauge: bool = False


class SceneOptionSensor(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    label: str
    type: str
    needs_mask: bool


class SceneOptions(BaseModel):
    regions: list[SceneOptionRegion]
    sensors: list[SceneOptionSensor]
    note_ko: str | None = None


__all__ = [
    "SceneAccepted",
    "SceneLiaison",
    "SceneList",
    "SceneOptions",
    "SceneOutputs",
    "SceneStatus",
    "SceneStep",
    "SceneSummary",
]
