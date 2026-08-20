from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from ..adapters.wbms_fusion import WbmsFusionAdapter
from ..adapters.wbms_segmentation import (
    WbmsExecutionError,
    WbmsInputError,
    WbmsSegmentationJobAdapter,
)
from .jobs import JobNotFoundError
from .schemas import (
    WbmsFusionRequest,
    WbmsFusionResponse,
    WbmsSegmentationJob,
    WbmsSegmentationJobList,
    WbmsSegmentationJobRequest,
    WbmsStatusResponse,
)

router = APIRouter(prefix="/wbms", tags=["wbms"])


def _segmentation(request: Request) -> WbmsSegmentationJobAdapter:
    return request.app.state.wbms_segmentation


def _fusion(request: Request) -> WbmsFusionAdapter:
    return request.app.state.wbms_fusion


@router.get(
    "/status",
    response_model=WbmsStatusResponse,
    summary="Report wbms container-image and handover readiness",
)
async def wbms_status(request: Request) -> WbmsStatusResponse:
    adapter = _segmentation(request)
    fusion = _fusion(request)
    settings = adapter.settings
    image_available = await run_in_threadpool(
        adapter.runner.image_available, settings.docker_image
    )
    cached = await run_in_threadpool(adapter.list_cached)
    return WbmsStatusResponse(
        docker_image=settings.docker_image,
        image_available=image_available,
        handover_dir=str(settings.handover_dir),
        handover_available=settings.handover_dir.is_dir(),
        bundle_dir=str(settings.bundle_dir),
        bundle_available=(
            (settings.bundle_dir / "meta.json").is_file()
            and (settings.bundle_dir / "model.keras").is_file()
        ),
        aws_csv=str(fusion.settings.aws_csv),
        aws_csv_available=fusion.settings.aws_csv.is_file(),
        jobs_dir=str(settings.jobs_dir),
        wb_output_dir=str(settings.wb_output_dir),
        cached_scenes=cached,
    )


@router.post(
    "/segmentation/jobs",
    response_model=WbmsSegmentationJob,
    status_code=202,
    summary="Start ② detect_water as an asynchronous container job",
    description=(
        "CPU inference is ~610 s per scene, so this endpoint returns a job id "
        "immediately; poll GET /wbms/segmentation/jobs/{job_id}. If a COMPLETE "
        "mask for the scene already exists under the output root, the job is "
        "returned as succeeded instantly (cached=true) without running docker."
    ),
)
async def create_segmentation_job(
    payload: WbmsSegmentationJobRequest, request: Request
) -> WbmsSegmentationJob:
    adapter = _segmentation(request)
    try:
        job = await run_in_threadpool(adapter.submit, payload.model_dump())
    except WbmsInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return WbmsSegmentationJob.model_validate(job)


@router.get(
    "/segmentation/jobs",
    response_model=WbmsSegmentationJobList,
    summary="List segmentation jobs (newest first)",
)
async def list_segmentation_jobs(
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> WbmsSegmentationJobList:
    jobs = await run_in_threadpool(_segmentation(request).list)
    items = [WbmsSegmentationJob.model_validate(job) for job in jobs[offset : offset + limit]]
    return WbmsSegmentationJobList(items=items, total=len(jobs))


@router.get(
    "/segmentation/jobs/{job_id}",
    response_model=WbmsSegmentationJob,
    summary="Get one segmentation job's state and result",
)
async def get_segmentation_job(job_id: str, request: Request) -> WbmsSegmentationJob:
    try:
        job = await run_in_threadpool(_segmentation(request).get, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return WbmsSegmentationJob.model_validate(job)


@router.post(
    "/fusion/corrections",
    response_model=WbmsFusionResponse,
    summary="Correct ③ satellite water levels with the ④ fusion LSTM (synchronous)",
    description=(
        "Runs Correct.py inside the wbms container (~seconds on CPU). The model "
        "corrects the water level for the observed dates; it does not forecast. "
        "persistence_horizon_days>0 adds an explicitly-labeled persistence "
        "extension of the last corrected level."
    ),
)
async def create_fusion_correction(
    payload: WbmsFusionRequest, request: Request
) -> WbmsFusionResponse:
    adapter = _fusion(request)
    body = payload.model_dump()
    try:
        result = await run_in_threadpool(adapter.correct, body)
    except WbmsInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WbmsExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return WbmsFusionResponse.model_validate(result)
