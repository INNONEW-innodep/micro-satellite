from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..adapters.wbms_fusion import WbmsFusionAdapter
from ..adapters.wbms_segmentation import (
    WbmsExecutionError,
    WbmsInputError,
    WbmsSegmentationJobAdapter,
)
from .jobs import JobNotFoundError
from ..api_docs import API_DOCS
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
    description=API_DOCS["wbms_status"],
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
    description=API_DOCS["wbms_segmentation_create"],
)
async def create_segmentation_job(
    payload: WbmsSegmentationJobRequest, request: Request
) -> WbmsSegmentationJob:
    adapter = _segmentation(request)
    try:
        job = await run_in_threadpool(adapter.submit, payload.model_dump())
    except WbmsInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        # GPU 프리플라이트(gpu 강제인데 VRAM 부족 등) — 요청 자체가 지금 실행 불가
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        # 잡 디렉터리가 읽기전용 마운트(:ro) 아래에 있으면 캐시 히트조차 기록에 실패한다.
        raise HTTPException(
            status_code=503,
            detail=(
                f"cannot write job state under {adapter.settings.jobs_dir}: {exc}. "
                "Set BACKEND_WBMS_JOBS_DIR to a writable path (docker-compose mounts backend/data rw)."
            ),
        ) from exc
    return WbmsSegmentationJob.model_validate(job)


@router.get(
    "/segmentation/jobs",
    response_model=WbmsSegmentationJobList,
    summary="List segmentation jobs (newest first)",
    description=API_DOCS["wbms_segmentation_list"],
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
    description=API_DOCS["wbms_segmentation_detail"],
)
async def get_segmentation_job(job_id: str, request: Request) -> WbmsSegmentationJob:
    try:
        job = await run_in_threadpool(_segmentation(request).get, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return WbmsSegmentationJob.model_validate(job)


# ── 산출물 다운로드 (3세부 연계: 원해상도 마스크 GeoTIFF 내려받기) ────────────────

_MEDIA_TYPES = {
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".json": "application/json",
    ".txt": "text/plain",
}


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


async def _succeeded_job_result(request: Request, job_id: str) -> dict[str, Any]:
    """잡을 찾아 성공한 잡의 result 를 돌려준다. 없으면 404, 아직 성공 전이면 409."""

    try:
        job = await run_in_threadpool(_segmentation(request).get, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    status = job.get("status")
    result = job.get("result")
    if status != "succeeded" or not isinstance(result, dict):
        raise HTTPException(
            status_code=409,
            detail=(
                f"job {job_id} has not succeeded (status={status}); "
                "products are only available for succeeded jobs"
            ),
        )
    return result


def _result_dir(result: dict[str, Any]) -> Path | None:
    """산출물 디렉터리 — output_dir 이 없으면 mask_tif 의 부모."""

    output_dir = result.get("output_dir")
    if output_dir:
        return Path(str(output_dir))
    mask_tif = result.get("mask_tif")
    return Path(str(mask_tif)).parent if mask_tif else None


def _product_names(result: dict[str, Any]) -> set[str]:
    """내려받을 수 있는 파일 이름(basename) 집합.

    detect_water 의 ``products`` 목록에 더해 잡 결과가 직접 가리키는 마스크·메타·QC·규약명
    사본(``product_tif`` — 다른 담당자가 추가할 PT000003_{product_id}.tif)을 허용한다.
    ``_qc.json`` 은 detect_water 가 products 에 넣지 않으므로 여기서 보탠다.
    """

    names = {Path(str(name)).name for name in (result.get("products") or []) if name}
    for key in ("mask_tif", "meta_json", "qc_json", "product_tif"):
        value = result.get(key)
        if value:
            names.add(Path(str(value)).name)
    return names


@router.get(
    "/segmentation/jobs/{job_id}/mask",
    summary="Download the job's water-mask GeoTIFF (original resolution)",
    description=(
        "성공한 ② detect_water 잡의 수체 마스크 GeoTIFF 를 원해상도 그대로 내려준다 "
        "(`image/tiff`, `Content-Disposition: attachment`). 3세부(세명소프트)가 마스크 파일을 "
        "직접 받아 가는 경로다.\n\n"
        "- 규약 파일명 사본(`result.product_tif`, `PT000003_{product_id}.tif`)이 있으면 그것을, "
        "없으면 `result.mask_tif`(`WB_…tif`)를 보낸다.\n"
        "- 404: 잡이 없거나 파일이 디스크에서 사라짐. 409: 잡이 아직 `succeeded` 가 아님 "
        "(queued/running/failed).\n"
        "- 잡 상세(`GET /segmentation/jobs/{job_id}`)의 `result.products` 에 있는 다른 파일은 "
        "`/segmentation/jobs/{job_id}/files/{name}` 으로 받는다."
    ),
    response_class=FileResponse,
    responses={
        200: {"content": {"image/tiff": {}}, "description": "수체 마스크 GeoTIFF"},
        404: {"description": "잡이 없거나 마스크 파일이 디스크에 없음"},
        409: {"description": "잡이 아직 성공 상태가 아님"},
    },
)
async def download_segmentation_mask(job_id: str, request: Request) -> FileResponse:
    result = await _succeeded_job_result(request, job_id)
    # 서버가 직접 적은 경로지만, 규약명 사본은 산출 디렉터리 안·원본은 마스크 캐시 안에 있을 때만 연다
    settings = request.app.state.wbms_segmentation.settings
    allowed_roots = [Path(settings.product_dir).resolve(), Path(settings.wb_output_dir).resolve()]

    def _contained(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(root == resolved or root in resolved.parents for root in allowed_roots)

    candidates = [Path(str(v)) for v in (result.get("product_tif"), result.get("mask_tif")) if v]
    if not candidates:
        raise HTTPException(status_code=404, detail="job result has no mask file")
    # 규약명 사본이 사라졌으면 원본 WB_…tif 로 되돌아간다 — 둘 다 없을 때만 404
    path = next((c for c in candidates if c.is_file() and _contained(c)), None)
    if path is None:
        raise HTTPException(status_code=404, detail=f"mask file missing on disk: {candidates[0].name}")
    return FileResponse(
        path,
        media_type="image/tiff",
        filename=path.name,
        content_disposition_type="attachment",
    )


@router.get(
    "/segmentation/jobs/{job_id}/files/{name}",
    summary="Download one product file of a succeeded job (mask, _meta.json, _qc.json)",
    description=(
        "성공한 잡의 산출물 하나를 이름으로 내려준다. `name` 은 잡 상세의 `result.products` "
        "(또는 `mask_tif`·`meta_json`·`qc_json`·`product_tif`)에 적힌 파일명(basename)과 "
        "정확히 같아야 하며, 경로 구분자나 `..` 이 섞인 이름은 404 로 거절한다 "
        "(산출물 디렉터리 밖은 절대 열지 않는다).\n\n"
        "매체 유형: `.tif` → `image/tiff`, `.json` → `application/json`, 그 밖은 "
        "`application/octet-stream`. 404: 잡·파일 없음. 409: 잡이 아직 성공 전."
    ),
    response_class=FileResponse,
    responses={
        200: {"description": "요청한 산출물 파일"},
        404: {"description": "잡이 없거나 이름이 산출물 목록에 없거나 파일이 디스크에 없음"},
        409: {"description": "잡이 아직 성공 상태가 아님"},
    },
)
async def download_segmentation_file(job_id: str, name: str, request: Request) -> FileResponse:
    result = await _succeeded_job_result(request, job_id)
    # 경로 탈출 방지: 이름은 산출물 목록의 basename 과 정확히 같아야 한다
    if name != Path(name).name or name in {"", ".", ".."} or name not in _product_names(result):
        raise HTTPException(status_code=404, detail=f"{name!r} is not a product of job {job_id}")
    base = _result_dir(result)
    if base is None:
        raise HTTPException(status_code=404, detail="job result has no output directory")
    base = base.resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.is_file():
        raise HTTPException(status_code=404, detail=f"product file missing on disk: {name}")
    return FileResponse(
        path,
        media_type=_media_type(path),
        filename=path.name,
        content_disposition_type="attachment",
    )


@router.post(
    "/fusion/corrections",
    response_model=WbmsFusionResponse,
    summary="Correct ③ satellite water levels with the ④ fusion LSTM (synchronous)",
    description=API_DOCS["wbms_fusion"],
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
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"cannot write fusion workspace under {adapter.settings.fusion_work_dir}: {exc}. "
                "Set BACKEND_WBMS_FUSION_DIR to a writable path."
            ),
        ) from exc
    return WbmsFusionResponse.model_validate(result)
