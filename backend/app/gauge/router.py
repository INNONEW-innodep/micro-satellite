from __future__ import annotations

import os
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from ..api_docs import API_DOCS
from .forecasters import ForecasterRegistry
from .jobs import GaugeTrainJobs, JobConflictError
from .regions import (
    RegionError,
    RegionNotFoundError,
    RegionRegistry,
    StationNotFoundError,
    parse_series_rows,
)
from .schemas import (
    BacktestResponse,
    EvaluationResponse,
    ExperimentsResponse,
    ForecasterDescription,
    ForecastResponse,
    ModelsResetResponse,
    RegionCreateRequest,
    RegionList,
    RegionSummary,
    SeriesResponse,
    SeriesUploadRequest,
    StationInfo,
    TrainJobAccepted,
    TrainJobList,
    TrainJobStatus,
    TrainRequest,
)
from .service import ArtifactNotFoundError, GaugeForecastService
from .training import Split, TrainingError, train_region

router = APIRouter(tags=["regions"])


def _service(request: Request) -> GaugeForecastService:
    return request.app.state.gauge_service


def _regions(request: Request) -> RegionRegistry:
    return request.app.state.regions


def _forecasters(request: Request) -> ForecasterRegistry:
    return request.app.state.gauge_forecasters


def _jobs(request: Request) -> GaugeTrainJobs:
    """학습 잡 저장소 — main.py 가 app.state 에 두지 않았어도(테스트 앱 등) 게으르게 만든다."""

    jobs = getattr(request.app.state, "gauge_train_jobs", None)
    if jobs is None:
        jobs = GaugeTrainJobs()
        request.app.state.gauge_train_jobs = jobs
    return jobs


def _region_or_404(request: Request, region_id: str):
    try:
        return _regions(request).get(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from exc


@router.get(
    "/gauge/forecasters",
    response_model=list[ForecasterDescription],
    summary="List swappable gauge forecasters",
    description=API_DOCS["gauge_forecasters"],
)
async def list_forecasters(request: Request) -> list[ForecasterDescription]:
    from .model_guide import MODEL_GUIDE

    registry = _forecasters(request)
    built_in_ids = {"persistence", "climatology", "ridge_ar", "ridge_arx", "ridge_arx_future", "seasonal_anomaly_ar", "mean_reversion", "analog_knn", "ensemble_mean"}
    names_ko = {str(g.get("id")): str(g.get("name_ko")) for g in MODEL_GUIDE.get("forecasters") or [] if g.get("name_ko")}
    return [
        ForecasterDescription(
            id=item.info.id,
            name=item.info.name,
            description=item.info.description,
            name_ko=names_ko.get(item.info.id),
            is_baseline=item.info.is_baseline,
            is_reference=item.info.is_reference,
            hyperparameter_grid=[dict(h) for h in item.info.hyperparameter_grid],
            min_history_days=item.info.min_history_days,
            built_in=item.info.id in built_in_ids,
            uses_future_exog=item.info.uses_future_exog,
        )
        for item in registry.list()
    ]


@router.get(
    "/regions",
    response_model=RegionList,
    summary="List regions (built-in + user)",
    description=API_DOCS["regions_list"],
)
async def list_regions(request: Request) -> RegionList:
    service = _service(request)
    regions = _regions(request)
    items = [RegionSummary(**service.region_summary(record)) for record in regions.list()]
    return RegionList(items=items, total=len(items), builtin_dir=str(regions.builtin_dir), user_dir=str(regions.user_dir))


@router.post(
    "/regions",
    response_model=RegionSummary,
    status_code=201,
    summary="Create a region (user directory)",
    description=API_DOCS["regions_create"],
)
async def create_region(payload: RegionCreateRequest, request: Request) -> RegionSummary:
    regions = _regions(request)
    try:
        record = regions.create_region(payload.model_dump(exclude={"overwrite"}), overwrite=payload.overwrite)
    except RegionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RegionSummary(**_service(request).region_summary(record))


@router.get(
    "/regions/{region_id}",
    response_model=RegionSummary,
    summary="Describe one region",
    description=API_DOCS["regions_detail"],
)
async def region_detail(region_id: str, request: Request) -> RegionSummary:
    record = _region_or_404(request, region_id)
    return RegionSummary(**_service(request).region_summary(record))


@router.get(
    "/regions/{region_id}/gauge/stations",
    response_model=list[StationInfo],
    summary="List gauge stations of a region",
    description=API_DOCS["gauge_stations"],
)
async def list_stations(region_id: str, request: Request) -> list[StationInfo]:
    record = _region_or_404(request, region_id)
    summary = _service(request).region_summary(record)
    return [StationInfo(**item) for item in summary["stations"]]


@router.get(
    "/regions/{region_id}/satellite/scenes",
    summary="Satellite scenes of the region with quicklook/overlay image URLs",
    description=API_DOCS["satellite_scenes"],
)
async def satellite_scenes(region_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    prefix = request.app.state.settings.api_prefix
    return _service(request).scenes(region_id, prefix)



def _scene_path_roots(request: Request) -> list[Path]:
    """서버 경로로 장면을 등록할 때 허용하는 루트. SCENE_PATH_ROOTS(콜론 구분)로 더 열 수 있다."""

    settings = request.app.state.settings
    repo_root = Path(__file__).resolve().parents[3]
    roots = [repo_root / "data", repo_root / "regions", Path(settings.regions_dir), Path(settings.regions_user_dir)]
    for extra in (os.getenv("SCENE_PATH_ROOTS") or "").split(":"):
        if extra.strip():
            roots.append(Path(extra.strip()).expanduser())
    for attr in ("wbms_jobs_dir", "wbms_fusion_dir"):
        value = getattr(settings, attr, None)
        if value:
            roots.append(Path(value))
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in out:
            out.append(resolved)
    return out


@router.post(
    "/regions/{region_id}/satellite/scenes",
    status_code=201,
    summary="Register a new satellite scene (host path or upload) → quicklooks, mask frame, forecast sample",
    description=API_DOCS["satellite_scene_register"],
)
async def register_satellite_scene(
    region_id: str,
    request: Request,
    sensor: Annotated[str, Form(description="센서 이름. 예: ICEYE, PlanetScope, Sentinel-1")],
    image_date: Annotated[str, Form(description="YYYYMMDD 또는 YYYY-MM-DD")],
    sensor_type: Annotated[str, Form(description="SAR 또는 OPTIC")] = "OPTIC",
    pre_path: Annotated[str | None, Form(description="서버(호스트) 경로의 전처리 GeoTIFF")] = None,
    mask_path: Annotated[str | None, Form(description="서버 경로의 탐지 마스크 GeoTIFF(0=육지,1=수체). 없으면 사진만 등록")] = None,
    gt_path: Annotated[str | None, Form(description="서버 경로의 GT 라벨 GeoTIFF(선택)")] = None,
    pre_file: Annotated[UploadFile | None, File(description="전처리 GeoTIFF 업로드(경로 대신)")] = None,
    mask_file: Annotated[UploadFile | None, File(description="탐지 마스크 GeoTIFF 업로드(선택)")] = None,
    gt_file: Annotated[UploadFile | None, File(description="GT 라벨 GeoTIFF 업로드(선택)")] = None,
    run_detection: Annotated[bool, Form(description="마스크가 없을 때 등록된 탐지 어댑터(WBMS)로 탐지 잡을 제출")] = False,
    testbed: Annotated[str | None, Form(description="탐지 잡용 테스트베드 키(카탈로그). 비우면 지역의 wbms_testbed")] = None,
    note: Annotated[str | None, Form()] = None,
    # ── 과제 통합 플랫폼 연계 파라미터(선택). run_detection=true 일 때 탐지 잡으로 그대로 전달된다.
    order_id: Annotated[str | None, Form(description="통합 플랫폼 주문 ID")] = None,
    scene_id: Annotated[str | None, Form(description="통합 플랫폼 Scene ID")] = None,
    work_id: Annotated[str | None, Form(description="통합 플랫폼 작업 ID")] = None,
    algm_id: Annotated[str | None, Form(description="알고리즘 ID. 예: AG000003")] = None,
    product_type: Annotated[str | None, Form(description="산출물 타입 코드. 수체탐지 PT000003")] = None,
    product_id: Annotated[str | None, Form(description="원본 영상 제품 ID. 비우면 전처리 사이드카에서 파생")] = None,
) -> dict[str, Any]:
    import re

    record = _region_or_404(request, region_id)
    liaison_options = {
        k: v.strip() for k, v in (
            ("order_id", order_id), ("scene_id", scene_id), ("work_id", work_id),
            ("algm_id", algm_id), ("product_type", product_type), ("product_id", product_id),
        ) if v is not None and v.strip()
    }
    if "product_type" in liaison_options and not re.match(r"^PT\d{6}$", liaison_options["product_type"]):
        raise HTTPException(status_code=422, detail="product_type 은 PT000003 형식이어야 합니다")
    if "product_id" in liaison_options and not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,199}$", liaison_options["product_id"]):
        raise HTTPException(status_code=422, detail="product_id 는 영문·숫자·_-. 만 허용됩니다")
    date_digits = re.sub(r"[^0-9]", "", image_date)[:8]
    if len(date_digits) != 8:
        raise HTTPException(status_code=422, detail="image_date 는 YYYYMMDD 형식이어야 합니다")
    try:
        datetime.strptime(date_digits, "%Y%m%d")
    except ValueError:
        raise HTTPException(status_code=422, detail=f"image_date 가 달력에 없는 날짜입니다: {date_digits}") from None
    sensor = sensor.strip()
    if not re.match(r"^[A-Za-z0-9_-]{1,32}$", sensor):
        raise HTTPException(status_code=422, detail="sensor 는 영문·숫자·_- 1~32자여야 합니다")
    sensor_type = sensor_type.strip().upper()
    if sensor_type not in ("SAR", "OPTIC"):
        raise HTTPException(status_code=422, detail="sensor_type 은 SAR 또는 OPTIC")

    if note is not None and len(note) > 500:
        raise HTTPException(status_code=422, detail="note 는 500자 이하여야 합니다")

    settings = request.app.state.settings
    # sensor·date 는 위에서 [A-Za-z0-9_-]·숫자만 통과했으므로 디렉터리 이름에 경로 문자가 섞이지 않는다.
    # 업로드 파일명은 클라이언트 이름을 쓰지 않고 pre/mask 로 고정한다(경로 탈출 방지).
    upload_root = Path(settings.regions_user_dir) / record.id / "uploads" / f"{date_digits}_{sensor}"
    max_bytes = int(getattr(settings, "scene_upload_max_bytes", 0) or 0)

    async def materialize(upload: UploadFile | None, path: str | None, name: str) -> Path | None:
        if upload is not None:
            if upload.size is not None and max_bytes and int(upload.size) > max_bytes:
                raise HTTPException(status_code=413, detail=f"{name}: 업로드 크기 {upload.size} B 가 한도 {max_bytes} B 를 넘습니다")
            upload_root.mkdir(parents=True, exist_ok=True)
            dest = upload_root / f"{name}.tif"
            if dest.is_file():
                # 같은 센서·촬영일 재등록: 이번 업로드가 거부되면 이미 등록된 장면의 원본을 되돌릴 수 있게 잠시 보관한다
                backup = dest.with_name(f"{name}.tif.prev")
                dest.replace(backup)
                replaced[dest] = backup
            written = 0
            with dest.open("wb") as handle:
                while True:
                    chunk = await upload.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if max_bytes and written > max_bytes:
                        handle.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail=f"{name}: 업로드가 한도 {max_bytes} B 를 넘습니다")
                    handle.write(chunk)
            if written == 0:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=422, detail=f"{name}: 빈 파일입니다")
            return dest
        if path:
            candidate = Path(path).expanduser().resolve()
            # 서버 경로 등록은 자료 루트(저장소 data/·regions/·사용자 지역 디렉터리·환경변수 추가 루트) 안으로 제한 —
            # 임의 호스트 경로의 존재 여부가 새어 나가지 않게 한다(감사 지적). 업로드 경로는 이 제한과 무관.
            roots = _scene_path_roots(request)
            if not any(candidate == root or root in candidate.parents for root in roots):
                raise HTTPException(
                    status_code=422,
                    detail=f"{name}: 서버 경로 등록은 허용 루트 안의 파일만 됩니다 ({', '.join(str(x) for x in roots)}). 파일을 직접 업로드하세요.",
                )
            if not candidate.is_file():
                raise HTTPException(status_code=422, detail=f"{name}: 파일이 없습니다: {candidate}")
            return candidate
        return None

    uploaded_paths: list[Path] = []
    replaced: dict[Path, Path] = {}

    def discard_uploads() -> None:
        # 업로드로 받은 파일은 GeoTIFF 가 아니었거나 뒤 파일이 거부됐을 수 있다 — 실패한 업로드를 남겨 두지 않는다
        for uploaded in uploaded_paths:
            if uploaded.parent == upload_root:
                uploaded.unlink(missing_ok=True)
        for dest, backup in replaced.items():  # 덮어쓴 기존 장면 원본은 되돌린다
            if backup.is_file():
                backup.replace(dest)
        replaced.clear()
        if upload_root.is_dir() and not any(upload_root.iterdir()):
            upload_root.rmdir()

    def commit_uploads() -> None:
        for backup in replaced.values():
            backup.unlink(missing_ok=True)
        replaced.clear()

    try:
        pre = await materialize(pre_file, pre_path, "pre")
        if pre is not None and pre_file is not None:
            uploaded_paths.append(pre)
        if pre is None:
            raise HTTPException(status_code=422, detail="pre_path 또는 pre_file 이 필요합니다(전처리 GeoTIFF)")
        mask = await materialize(mask_file, mask_path, "mask")
        if mask is not None and mask_file is not None:
            uploaded_paths.append(mask)
        gt = await materialize(gt_file, gt_path, "gt")
        if gt is not None and gt_file is not None:
            uploaded_paths.append(gt)
    except HTTPException:
        discard_uploads()  # pre 는 저장됐는데 mask/gt 가 413·422 로 거부된 경우
        raise

    service = _service(request)
    try:
        result = await run_in_threadpool(
            service.register_scene, region_id, sensor=sensor, sensor_type=sensor_type, image_date=date_digits,
            pre_tif=pre, mask_tif=mask, gt_tif=gt, source_note=note,
        )
    except Exception as exc:  # noqa: BLE001 - rasterio 오류 등을 422 로
        discard_uploads()
        raise HTTPException(status_code=422, detail=f"장면 자산 생성 실패: {type(exc).__name__}: {exc}") from exc
    commit_uploads()

    job: dict[str, Any] | None = None
    if mask is None and run_detection:
        segmenters = request.app.state.segmenters
        tb = (testbed or (record.manifest.get("satellite") or {}).get("wbms_testbed") or "").lower()
        sensor_key = {"iceye": "iceye", "planetscope": "planet", "planet": "planet", "umbra": "umbra"}.get(sensor.lower(), sensor.lower())
        adapter = segmenters.resolve(sensor_key, tb) if tb else None
        if adapter is None:
            result["detection_note_ko"] = f"'{sensor}' × '{tb or '?'}' 를 받는 탐지 어댑터가 없어 잡을 제출하지 않았습니다(카탈로그 등록 필요)."
        else:
            from ..segmentation.base import SceneRequest

            try:
                submission = await run_in_threadpool(
                    adapter.submit, SceneRequest(sensor=sensor_key, testbed=tb, image_date=date_digits, input_path=pre, options=liaison_options)
                )
                job = {"job_id": submission.job_id, "status": submission.status, "cached": submission.cached, "warnings": submission.warnings}
            except Exception as exc:  # noqa: BLE001
                result["detection_note_ko"] = f"탐지 잡 제출 실패: {exc}"
    result["detection_job"] = job
    result["next_ko"] = (
        "마스크가 등록되어 퀵룩·오버레이·면적·공통 격자 프레임이 만들어졌고, 면적 예측 샘플에 자동 반영됩니다(UI 재시작 시 목록 갱신)."
        if mask is not None
        else "사진만 등록되었습니다. 탐지 마스크가 나오면 같은 sensor·image_date 로 다시 등록해 갱신하세요."
    )
    return result


@router.get(
    "/regions/{region_id}/satellite/files/{relative_path:path}",
    summary="Serve a satellite asset file (quicklook PNG, mask frame PNG)",
    description=API_DOCS["satellite_files"],
)
async def satellite_file(region_id: str, relative_path: str, request: Request) -> FileResponse:
    record = _region_or_404(request, region_id)
    try:
        path = record.satellite_file(relative_path)
    except RegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"file not found: {relative_path}") from exc
    media = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    if path.suffix.lower() == ".json":
        media = "application/json"
    return FileResponse(path, media_type=media)


@router.get(
    "/regions/{region_id}/satellite/scenes/{scene_id}/mask",
    summary="Download the scene's source water-mask GeoTIFF (original resolution)",
    description=(
        "장면 목록(`GET /regions/{region_id}/satellite/scenes`)의 한 장면에 붙은 원본 수체 마스크 "
        "GeoTIFF(`source_mask`, 탐지 산출 `WB_…tif` 또는 업로드한 마스크)를 원해상도 그대로 내려준다 "
        "(`image/tiff`, `Content-Disposition: attachment`). 퀵룩 PNG(`satellite/files/…`)와 달리 "
        "지리참조·nodata 가 그대로 살아 있어 3세부(세명소프트) 같은 외부 시스템이 받아 쓰는 용도다.\n\n"
        "- `source_mask` 가 상대경로면 저장소 루트 기준으로 푼다(내장 지역), 절대경로면 그대로 쓴다(업로드 장면).\n"
        "- 404: 지역·장면이 없거나, 장면에 마스크가 없거나(`detection_available=false`), 파일이 디스크에서 사라짐."
    ),
    response_class=FileResponse,
    responses={
        200: {"content": {"image/tiff": {}}, "description": "장면의 원본 수체 마스크 GeoTIFF"},
        404: {"description": "지역·장면·마스크 파일 없음"},
    },
)
async def satellite_scene_mask(region_id: str, scene_id: str, request: Request) -> FileResponse:
    _region_or_404(request, region_id)
    prefix = request.app.state.settings.api_prefix
    payload = _service(request).scenes(region_id, prefix)
    scene = next((s for s in payload.get("scenes") or [] if s.get("scene_id") == scene_id), None)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"unknown scene {scene_id!r} in region {region_id!r}")
    source_mask = scene.get("source_mask")
    if not source_mask:
        raise HTTPException(status_code=404, detail=f"scene {scene_id!r} has no water mask (detection_available=false)")
    path = Path(str(source_mask)).expanduser()
    if not path.is_absolute():  # 내장 지역의 scenes.json 은 저장소 루트 기준 상대경로를 쓴다
        path = Path(__file__).resolve().parents[3] / path
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"mask file missing on disk: {path.name}")
    return FileResponse(path, media_type="image/tiff", filename=path.name, content_disposition_type="attachment")


@router.get(
    "/regions/{region_id}/satellite/timeseries",
    summary="Satellite-based level/area time-series metrics on record (m and cm)",
    description=API_DOCS["satellite_timeseries"],
)
async def satellite_timeseries(region_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    return _service(request).timeseries_eval(region_id)


_PERIOD_DOC = "포함 구간 — YYYY-MM-DD 또는 YYYYMMDD. 생략하면 그쪽 끝은 무제한"


def _period_or_422(start: str | None, end: str | None) -> tuple[Date | None, Date | None]:
    """start·end 질의값을 날짜로 바꾼다. 형식이 틀리거나 start > end 면 422 — 빈 결과로 얼버무리지 않는다."""

    def parse(name: str, raw: str | None) -> Date | None:
        text = (raw or "").strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        raise HTTPException(status_code=422, detail=f"{name}: 날짜는 YYYY-MM-DD 또는 YYYYMMDD 여야 한다 — 받은 값 {text!r}")

    lo, hi = parse("start", start), parse("end", end)
    if lo and hi and lo > hi:
        raise HTTPException(status_code=422, detail=f"start({lo.isoformat()}) 가 end({hi.isoformat()}) 보다 늦다")
    return lo, hi


@router.get(
    "/regions/{region_id}/satellite/area-series",
    summary="Comparable water-area series on the per-sensor common frame (with the old per-footprint number next to it)",
    description=API_DOCS["satellite_area_series"] + (
        "\n\n**기간 필터** — `start`·`end`(포함, YYYY-MM-DD 또는 YYYYMMDD)를 주면 센서별 `rows`(있으면 `scenes` 도)를 관측일로 거른다. "
        "상관계수·`n_scenes` 등 요약값은 전체 기록 기준 그대로이며 다시 계산하지 않는다. 요청한 기간은 `period`로 되돌려 준다. "
        "start > end 는 422."
    ),
)
async def satellite_area_series(
    region_id: str,
    request: Request,
    start: Annotated[str | None, Query(description=_PERIOD_DOC)] = None,
    end: Annotated[str | None, Query(description=_PERIOD_DOC)] = None,
) -> dict[str, Any]:
    _region_or_404(request, region_id)
    lo, hi = _period_or_422(start, end)
    return _service(request).area_series(region_id, start=lo, end=hi)


@router.get(
    "/regions/{region_id}/weather",
    summary="Stored KMA ASOS daily weather for the region (same rows the forecasters use)",
    description=API_DOCS["region_weather"],
)
async def region_weather(
    region_id: str,
    request: Request,
    start: Annotated[Date | None, Query(description="포함 시작일 YYYY-MM-DD")] = None,
    end: Annotated[Date | None, Query(description="포함 종료일 YYYY-MM-DD")] = None,
) -> dict[str, Any]:
    _region_or_404(request, region_id)
    return _service(request).region_weather(region_id, start=start, end=end)


@router.get(
    "/gauge/model-guide",
    summary="Plain-language guide to the forecasters, the skill metric, and why no LSTM",
    description=API_DOCS["gauge_model_guide"],
)
async def gauge_model_guide(request: Request) -> dict[str, Any]:
    from .model_guide import MODEL_GUIDE

    registry = _forecasters(request)
    known = {item.info.id for item in registry.list()}
    return {**MODEL_GUIDE, "registered_ids": sorted(known)}


@router.get(
    "/regions/{region_id}/satellite/detection",
    summary="Water-body detection accuracy on record for the region (per scene)",
    description=API_DOCS["satellite_detection"],
)
async def satellite_detection(region_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    return _service(request).detection_eval(region_id)


@router.get(
    "/regions/{region_id}/satellite/levels",
    summary="Satellite-derived water levels (WBMS) with same-day gauge comparison",
    description=API_DOCS["satellite_levels"] + (
        "\n\n**기간 필터** — `start`·`end`(포함, YYYY-MM-DD 또는 YYYYMMDD)로 관측일을 좁힌다. `station_id` 와 함께 쓸 수 있고, "
        "`n`·`comparison` 은 거른 뒤의 기록으로 센다. 요청한 기간은 `period`로 되돌려 준다. start > end 는 422."
    ),
)
async def satellite_levels(
    region_id: str,
    request: Request,
    station_id: Annotated[str | None, Query(description="수위계 지점 id — satellite_location_links 로 위성 산출 위치를 찾는다")] = None,
    start: Annotated[str | None, Query(description=_PERIOD_DOC)] = None,
    end: Annotated[str | None, Query(description=_PERIOD_DOC)] = None,
) -> dict[str, Any]:
    _region_or_404(request, region_id)
    lo, hi = _period_or_422(start, end)
    return _service(request).satellite_levels(region_id, station_id, start=lo, end=hi)


@router.get(
    "/regions/{region_id}/gauge/evaluation",
    response_model=EvaluationResponse,
    summary="Station × horizon evaluation table",
    description=API_DOCS["gauge_evaluation"],
)
async def gauge_evaluation(region_id: str, request: Request) -> EvaluationResponse:
    _region_or_404(request, region_id)
    payload = _service(request).evaluation(region_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="학습 산출물이 없습니다 — POST /gauge/train 을 먼저 실행하세요")
    return EvaluationResponse(**payload)


@router.get(
    "/regions/{region_id}/gauge/experiments",
    response_model=ExperimentsResponse,
    summary="Side experiments on record for the region (future-rain bound, satellite area-model LOO comparison)",
    description=API_DOCS["gauge_experiments"],
)
async def gauge_experiments(region_id: str, request: Request) -> ExperimentsResponse:
    _region_or_404(request, region_id)
    return ExperimentsResponse(**_service(request).experiments(region_id))


@router.post(
    "/regions/{region_id}/gauge/train",
    response_model=EvaluationResponse,
    responses={
        202: {"model": TrainJobAccepted, "description": "async_run=true — 잡 접수(백그라운드 학습)"},
        409: {"description": "같은 지역에 진행 중인 학습 잡이 있다"},
        422: {"description": "지점·예측기·분할 검증 실패(한국어 사유)"},
    },
    summary="Train / re-train gauge forecasters (sync, or async_run=true → 202 job)",
    description=API_DOCS["gauge_train"],
)
async def gauge_train(region_id: str, payload: TrainRequest, request: Request):
    record = _region_or_404(request, region_id)
    registry = _regions(request)
    forecasters = _forecasters(request)
    service = _service(request)
    jobs = _jobs(request)
    if payload.forecasters:
        known = {item.info.id for item in forecasters.list()}
        unknown = sorted(set(payload.forecasters) - known)
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown forecaster(s): {unknown}; known: {sorted(known)}")
        if all(forecasters.get(fid).info.is_baseline for fid in payload.forecasters):
            raise HTTPException(status_code=422, detail="기준선(persistence)만으로는 학습할 수 없습니다 — 예측이 아닙니다")

    # 같은 지역에 백그라운드 잡이 돌고 있으면 동기 학습도 막는다 — 같은 디렉터리를 동시에 쓰면 평가표가 깨진다.
    active = jobs.active_for(region_id)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"'{region_id}' 에 진행 중인 학습 잡이 있습니다 (job_id={active['job_id']}, {active['status']}) — 끝난 뒤 다시 시도하세요",
        )

    # 사용자 지정 분할은 학습 전에 검사한다(순서·시계열 범위·최소 길이). 비동기 요청도 여기서 422 를 받는다.
    split: Split | None = None
    if payload.split is not None:
        try:
            split = await run_in_threadpool(service.validate_split, region_id, payload.split.to_mapping())
        except TrainingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RegionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not payload.async_run:
        def train_sync() -> dict[str, Any]:
            with jobs.hold(region_id):  # 동기 학습도 지역 잠금 — 그 사이 비동기 잡·초기화는 409
                return train_region(
                    registry,
                    forecasters,
                    region_id,
                    stations=payload.stations,
                    allowed_forecasters=payload.forecasters,
                    horizon_max=payload.horizon_max,
                    split=split,
                    fallback_artifacts_dir=registry.builtin_artifacts_dir(region_id),
                )

        try:
            evaluation = await run_in_threadpool(train_sync)
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrainingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RegionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EvaluationResponse(**service.decorate_evaluation(region_id, evaluation, "user"))

    # --- 비동기: 지점 목록을 지금 확정하고(없는 지점은 422) 잡을 접수한다 -----------------------------
    try:
        series_set = await run_in_threadpool(registry.load_series, region_id)
    except RegionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    available = [s.id for s in series_set.stations]
    if payload.stations:
        missing = sorted(set(payload.stations) - set(available))
        if missing:
            raise HTTPException(status_code=422, detail=f"지점을 찾을 수 없습니다: {missing}")
        stations = [s for s in available if s in set(payload.stations)]
    else:
        stations = available
    if not stations:
        raise HTTPException(status_code=422, detail="학습할 지점이 없습니다 — 수위 시계열을 먼저 올리세요")
    split_used = split.to_dict() if split is not None else service.split_default(record, series_set)
    fallback_dir = registry.builtin_artifacts_dir(region_id)

    def run(on_progress):  # noqa: ANN001 - 잡 저장소가 넘겨 주는 진행률 콜백
        return train_region(
            registry,
            forecasters,
            region_id,
            stations=stations,
            allowed_forecasters=payload.forecasters,
            horizon_max=payload.horizon_max,
            split=split,
            fallback_artifacts_dir=fallback_dir,
            on_progress=on_progress,
        )

    def finalize(evaluation: dict[str, Any]) -> dict[str, Any]:
        # GET .../gauge/evaluation 과 같은 응답(custom_split·trained_at·artifacts_origin 포함)
        return service.decorate_evaluation(region_id, evaluation, "user")

    try:
        snapshot = jobs.submit(
            region_id,
            run,
            split_used=split_used,
            custom_split=split is not None,
            stations=stations,
            forecasters=payload.forecasters,
            horizon_max=payload.horizon_max,
            finalize=finalize,
        )
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    prefix = request.app.state.settings.api_prefix
    accepted = TrainJobAccepted(
        job_id=snapshot["job_id"],
        status=snapshot["status"],
        region_id=region_id,
        split_used=split_used,
        stations=stations,
        poll_url=f"{prefix}/regions/{region_id}/gauge/train/jobs/{snapshot['job_id']}",
    )
    return JSONResponse(status_code=202, content=accepted.model_dump())


@router.get(
    "/regions/{region_id}/gauge/train/jobs",
    response_model=TrainJobList,
    summary="List background training jobs of a region (newest first)",
    description=API_DOCS["gauge_train_jobs"],
)
async def gauge_train_jobs(region_id: str, request: Request) -> TrainJobList:
    _region_or_404(request, region_id)
    return TrainJobList(items=[TrainJobStatus(**item) for item in _jobs(request).list(region_id)])


@router.get(
    "/regions/{region_id}/gauge/train/jobs/{job_id}",
    response_model=TrainJobStatus,
    summary="Poll one background training job (progress, result when done)",
    description=API_DOCS["gauge_train_job"],
)
async def gauge_train_job(region_id: str, job_id: str, request: Request) -> TrainJobStatus:
    _region_or_404(request, region_id)
    item = _jobs(request).get(job_id)
    if item is None or item["region_id"] != region_id:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r} for region {region_id!r}")
    return TrainJobStatus(**item)


@router.delete(
    "/regions/{region_id}/gauge/models",
    response_model=ModelsResetResponse,
    responses={409: {"description": "되돌릴 내장 산출물이 없거나 학습 잡이 진행 중"}},
    summary="Reset user-trained artifacts → built-in artifacts become active again",
    description=API_DOCS["gauge_models_reset"],
)
async def gauge_models_reset(region_id: str, request: Request) -> ModelsResetResponse:
    _region_or_404(request, region_id)
    active = _jobs(request).active_for(region_id)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"'{region_id}' 에 진행 중인 학습 잡이 있어 되돌릴 수 없습니다 (job_id={active['job_id']}) — 끝난 뒤 다시 시도하세요",
        )
    registry = _regions(request)
    try:
        removed = registry.reset_user_artifacts(region_id)
    except RegionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    artifacts_dir, origin = _service(request).active_artifacts(region_id)
    return ModelsResetResponse(region_id=region_id, removed_files=removed, active=origin, artifacts_dir=str(artifacts_dir))


@router.get(
    "/regions/{region_id}/gauge/{station_id}/series",
    response_model=SeriesResponse,
    summary="Daily gauge series of a station",
    description=API_DOCS["gauge_series"],
)
async def gauge_series(
    region_id: str,
    station_id: str,
    request: Request,
    start: Annotated[Date | None, Query()] = None,
    end: Annotated[Date | None, Query()] = None,
) -> SeriesResponse:
    _region_or_404(request, region_id)
    try:
        payload = _service(request).series_rows(region_id, station_id, start=start, end=end)
    except StationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown station {station_id!r}") from exc
    return SeriesResponse(**payload)


@router.post(
    "/regions/{region_id}/gauge/{station_id}/series",
    response_model=StationInfo,
    summary="Upload / merge a station's daily series (JSON)",
    description=API_DOCS["gauge_series_upload"],
)
async def gauge_series_upload(region_id: str, station_id: str, payload: SeriesUploadRequest, request: Request) -> StationInfo:
    _region_or_404(request, region_id)
    try:
        station = _regions(request).upsert_station_series(
            region_id, station_id, payload.rows, replace=payload.replace, station_meta=payload.station
        )
    except RegionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StationInfo(**station.to_dict())


@router.post(
    "/regions/{region_id}/gauge/{station_id}/series/file",
    response_model=StationInfo,
    summary="Upload a station's daily series as CSV/JSON file",
    description=API_DOCS["gauge_series_upload_file"],
)
async def gauge_series_upload_file(
    region_id: str,
    station_id: str,
    request: Request,
    file: Annotated[UploadFile, File(description="CSV(date,water_level_m) 또는 JSON 행 목록")],
    replace: Annotated[bool, Form()] = False,
    station_name: Annotated[str | None, Form()] = None,
    terrain: Annotated[str | None, Form()] = None,
    source: Annotated[str | None, Form()] = None,
) -> StationInfo:
    _region_or_404(request, region_id)
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일이 50 MB를 넘습니다")
    try:
        rows = parse_series_rows(raw)
        station = _regions(request).upsert_station_series(
            region_id,
            station_id,
            rows,
            replace=replace,
            station_meta={"name": station_name, "terrain": terrain, "source": source or "user_upload"},
        )
    except (RegionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StationInfo(**station.to_dict())


@router.get(
    "/regions/{region_id}/gauge/{station_id}/model",
    summary="Selected model, candidates and metrics per horizon",
    description=API_DOCS["gauge_model"],
)
async def gauge_model(region_id: str, station_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    try:
        return _service(request).station_detail(region_id, station_id)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"{station_id!r} 학습 산출물이 없습니다") from exc


@router.get(
    "/regions/{region_id}/gauge/{station_id}/forecast",
    response_model=ForecastResponse,
    summary="Forecast water level D+1..D+H from an origin date",
    description=API_DOCS["gauge_forecast"],
)
async def gauge_forecast(
    region_id: str,
    station_id: str,
    request: Request,
    as_of: Annotated[Date | None, Query(description="전망 원점(포함). 비우면 마지막 관측일")] = None,
    horizon: Annotated[int | None, Query(ge=1, le=90)] = None,
    history_days: Annotated[int, Query(ge=7, le=3660)] = 120,
    forecaster: Annotated[str | None, Query(description="예측기 id 로 모델을 지정(즉석 적합). 비우면 검증에서 고른 모델")] = None,
) -> ForecastResponse:
    _region_or_404(request, region_id)
    service = _service(request)
    try:
        payload = await run_in_threadpool(
            service.forecast, region_id, station_id, as_of=as_of, horizon=horizon, history_days=history_days,
            forecaster_id=forecaster,
        )
    except StationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown station {station_id!r}") from exc
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"{station_id!r} 학습 산출물이 없습니다 — POST /gauge/train") from exc
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc.args[0] if exc.args else exc)) from exc
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ForecastResponse(**payload)


@router.get(
    "/regions/{region_id}/gauge/{station_id}/forecast/compare",
    summary="Forecasts from every model family at once, with differences and per-model metrics",
    description=API_DOCS["gauge_forecast_compare"],
)
async def gauge_forecast_compare(
    region_id: str,
    station_id: str,
    request: Request,
    as_of: Annotated[Date | None, Query()] = None,
    horizon: Annotated[int | None, Query(ge=1, le=30)] = None,
) -> dict[str, Any]:
    _region_or_404(request, region_id)
    service = _service(request)
    try:
        return await run_in_threadpool(service.forecast_compare, region_id, station_id, as_of=as_of, horizon=horizon)
    except StationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown station {station_id!r}") from exc
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"{station_id!r} 학습 산출물이 없습니다") from exc
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/regions/{region_id}/gauge/{station_id}/backtest",
    response_model=BacktestResponse,
    summary="Test-year backtest trace for one horizon",
    description=API_DOCS["gauge_backtest"],
)
async def gauge_backtest(
    region_id: str,
    station_id: str,
    request: Request,
    horizon: Annotated[int, Query(ge=1, le=90)] = 7,
) -> BacktestResponse:
    _region_or_404(request, region_id)
    try:
        payload = _service(request).backtest(region_id, station_id, horizon)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"{station_id!r} 학습 산출물이 없습니다") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0] if exc.args else exc)) from exc
    return BacktestResponse(**payload)
