"""인도받은 WBMS 도커 모델을 SegmentationAdapter 로 감싼 내장 어댑터."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapters.wbms_segmentation import WbmsInputError, WbmsSegmentationJobAdapter
from .base import (
    SceneRequest,
    SegmentationAdapter,
    SegmentationSubmission,
    SegmenterInfo,
)


class WbmsDockerSegmenter(SegmentationAdapter):
    def __init__(self, job_adapter: WbmsSegmentationJobAdapter) -> None:
        self._jobs = job_adapter

    @property
    def info(self) -> SegmenterInfo:
        return SegmenterInfo(
            id="wbms-detect-water",
            name="WBMS U-Net 수체 탐지 (인도 모델, 도커)",
            description=(
                "WBMS_SAR_ICEYE.h5(U-Net+ASPP) / WBMS_Optic_PlanetScope.h5(Attention U-Net)를 "
                f"컨테이너 {self._jobs.settings.docker_image} 로 실행한다. 호스트 docker CLI 필요."
            ),
            version=str(self._jobs.settings.docker_image),
            execution="docker",
            input_kind="Pre_*.tif + _meta.json (+ SAR: 3-class mask)",
            built_in=True,
        )

    def capabilities(self) -> dict[str, Any]:
        settings = self._jobs.settings
        payload = self._jobs.catalog.capabilities()
        payload.update(
            {
                "docker_image": settings.docker_image,
                "image_available": self._jobs.runner.image_available(settings.docker_image),
                "handover_available": settings.handover_dir.is_dir(),
                "weights_available": {
                    key: (settings.model_dir / profile["weights"]).is_file()
                    for key, profile in self._jobs.catalog.sensors.items()
                },
                "gpu_mode": settings.gpu_mode,
                "input_contract_ko": (
                    "05_l1_pre/<pre_dir>/<yyyy>/<yyyymmdd>/Pre_<TB>_<Sensor>_<yyyymmdd>T*.tif 와 같은 이름의 "
                    "_meta.json 사이드카. SAR 는 06_aux/Masks/Img_DEM_3class_utm/<testbed>_<date>_mask3_utm.tif 필수. "
                    "input_path 를 주면 임의 경로의 Pre tif 도 받는다(사이드카는 여전히 필요)."
                ),
            }
        )
        return payload

    def supports(self, sensor: str, testbed: str) -> bool:
        return self._jobs.catalog.supports(sensor, testbed)

    def submit(self, request: SceneRequest) -> SegmentationSubmission:
        body = {
            "sensor": request.sensor,
            "testbed": request.testbed,
            "image_date": request.image_date,
            "input_tif": str(request.input_path) if request.input_path else None,
            "force": bool(request.force),
        }
        # 연계 파라미터(order_id 등)는 options 로 들어온다 — 잡 요청에 그대로 싣는다
        for key in ("order_id", "scene_id", "work_id", "algm_id", "product_type", "product_id"):
            value = (request.options or {}).get(key)
            if value not in (None, ""):
                body[key] = str(value)
        job = self._jobs.submit(body)
        return SegmentationSubmission(
            job_id=str(job["job_id"]),
            status=str(job["status"]),
            cached=bool(job.get("cached")),
            result=job.get("result"),
            warnings=list(job.get("warnings") or []),
            raw=job,
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._jobs.get(job_id)

    def list_cached(self) -> list[dict[str, Any]]:
        return self._jobs.list_cached()


__all__ = ["WbmsDockerSegmenter", "WbmsInputError", "Path"]
