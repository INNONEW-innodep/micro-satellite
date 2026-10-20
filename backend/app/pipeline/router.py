"""장면 파이프라인 API — 위성 장면 하나를 올리면 등록→탐지→면적→수위→예측→연계를 한 번에.

* ``POST /scenes``                 접수(202) — multipart: region_id, sensor, image_date, pre_file|pre_path, …
* ``GET  /scenes``                 목록(최신순)
* ``GET  /scenes/options``         접수 화면 선택지(지역·센서)
* ``GET  /scenes/{scene_id}``      단계별 상태·산출 URL
* ``GET  /scenes/{scene_id}/files/{name}``  장면 산출 폴더의 파일
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..gauge.regions import RegionNotFoundError
from ..gauge.router import _scene_path_roots
from ..wbms.liaison import LiaisonError, liaison_params
from .jobs import SceneConflictError, ScenePipelineJobs
from .schemas import SceneAccepted, SceneList, SceneOptions, SceneStatus

router = APIRouter(prefix="/scenes", tags=["scenes"])

_MEDIA_TYPES = {
    ".tif": "image/tiff", ".tiff": "image/tiff", ".png": "image/png", ".json": "application/json",
    ".txt": "text/plain", ".csv": "text/csv", ".npy": "application/x-npy",
}
_UPLOAD_CHUNK = 8 * 1024 * 1024


def _jobs(request: Request) -> ScenePipelineJobs:
    jobs = getattr(request.app.state, "scene_pipeline", None)
    if jobs is None:
        raise HTTPException(status_code=503, detail="scene pipeline is not installed (install_pipeline(app))")
    return jobs


def resolve_sensor(catalog: Any, raw: str) -> tuple[str, str, str, str] | None:
    """입력 센서 이름을 카탈로그 키·표시명·종류·파일 라벨로 — 대소문자 무시.

    'planet' 도 PlanetScope, 'UMBRA'/'Umbra'/'umbra' 모두 Umbra(파일 라벨 UMBRA).
    """

    text = str(raw or "").strip().lower()
    if not text:
        return None
    for key, profile in catalog.sensors.items():
        label = str(profile.get("file_label") or profile.get("name", ""))
        if text in (str(key).lower(), str(profile.get("name", "")).lower(), label.lower()):
            return str(key), str(profile["name"]), str(profile.get("type", "OPTIC")).upper(), label
    return None


def _date_digits(image_date: str) -> str:
    digits = re.sub(r"[^0-9]", "", str(image_date or ""))[:8]
    if len(digits) != 8:
        raise HTTPException(status_code=422, detail="image_date 는 YYYYMMDD 또는 YYYY-MM-DD 형식이어야 합니다")
    try:
        datetime.strptime(digits, "%Y%m%d")  # noqa: DTZ007 - 달력 검증만, 시간대 무관
    except ValueError:
        raise HTTPException(status_code=422, detail=f"image_date 가 달력에 없는 날짜입니다: {digits}") from None
    return digits


def _time_digits(acquisition_time: str | None) -> str:
    text = str(acquisition_time or "").strip()
    if not text:
        return "000000"
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) == 4:
        digits += "00"
    if len(digits) != 6:
        raise HTTPException(status_code=422, detail="acquisition_time 은 HH:MM:SS 또는 HHMMSS 형식이어야 합니다")
    try:
        datetime.strptime(digits, "%H%M%S")  # noqa: DTZ007 - 형식 검증만
    except ValueError:
        raise HTTPException(status_code=422, detail=f"acquisition_time 이 시각으로 맞지 않습니다: {text}") from None
    return digits


def _testbed_code(catalog: Any, testbed: str | None, region_id: str) -> str:
    if testbed and testbed in catalog.testbeds:
        return str(catalog.testbeds[testbed].get("code") or testbed.capitalize())
    if testbed:
        return testbed.capitalize()
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", region_id) if part) or "Region"


async def _write_upload(upload: UploadFile, dest: Path, max_bytes: int, label: str) -> Path:
    if upload.size is not None and max_bytes and int(upload.size) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{label}: 업로드 크기 {upload.size} B 가 한도 {max_bytes} B 를 넘습니다")
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as handle:
        while True:
            chunk = await upload.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes and written > max_bytes:
                handle.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"{label}: 업로드가 한도 {max_bytes} B 를 넘습니다")
            handle.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"{label}: 빈 파일입니다")
    return dest


def _server_path(request: Request, raw: str, label: str) -> Path:
    candidate = Path(raw).expanduser().resolve()
    roots = _scene_path_roots(request)
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise HTTPException(
            status_code=422,
            detail=f"{label}: 서버 경로 등록은 허용 루트 안의 파일만 됩니다 ({', '.join(str(x) for x in roots)}). 파일을 직접 업로드하세요.",
        )
    if not candidate.is_file():
        raise HTTPException(status_code=422, detail=f"{label}: 파일이 없습니다: {candidate}")
    return candidate


@router.get(
    "/options",
    response_model=SceneOptions,
    summary="Choices for the scene submission form (regions, sensors)",
    description=(
        "접수 화면이 보여 줄 지역·센서 목록. `regions[]` 는 지역 레지스트리(region.json 의 `satellite.wbms_testbed`·"
        "`sensors`, 수위계 보유 여부), `sensors[]` 는 WBMS 카탈로그(`backend/app/wbms/catalog.json`)에서 만든다. "
        "`needs_mask=true`(SAR) 센서는 3-class 마스크가 필요하며 날짜별 마스크가 없으면 인접 날짜 것을, 그것도 없으면 영상 유효 범위로 "
        "생성한 마스크(바다 없음 가정)를 쓴다. `sensors[].id` 는 파일 라벨(ICEYE·UMBRA·PlanetScope)이고 `regions[].sensors` 도 같은 표기다."
    ),
)
async def scene_options(request: Request) -> dict[str, Any]:
    return await run_in_threadpool(_jobs(request).options)


@router.get(
    "",
    response_model=SceneList,
    summary="List scene pipelines (newest first) plus known scenes without a pipeline record",
    description=(
        "파이프라인 기록(장면 폴더 status.json)을 최신순으로 돌려준다. 파이프라인을 거치지 않은 기존 장면"
        "(내장 지역 자산 등)도 `pipeline_id=null`·`status=done` 으로 합성해 함께 보인다. `region_id` 로 거를 수 있다."
    ),
)
async def list_scenes(
    request: Request,
    region_id: Annotated[str | None, Query(description="이 지역의 장면만")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> dict[str, Any]:
    jobs = _jobs(request)
    items = await run_in_threadpool(jobs.list, region_id)
    items += await run_in_threadpool(jobs.synthesize_all, region_id)
    items = sorted(items, key=lambda item: str(item.get("created_at") or ""))[::-1]  # 최신순, 동률은 나중 접수가 앞
    return {"items": items[:limit], "total": len(items)}


@router.post(
    "",
    response_model=SceneAccepted,
    status_code=202,
    summary="Submit one satellite scene — register → detect → area → levels → predict → deliver",
    description=(
        "장면 하나의 전체 체인을 백그라운드로 돌린다. 응답은 202 와 `status_url`; 진행은 `GET /scenes/{scene_id}` 로 본다.\n\n"
        "- `sensor`: ICEYE | Umbra | PlanetScope (대소문자 무시, `planet`·`UMBRA` 허용). Umbra 는 전용 모델이 없어 ICEYE SAR 모델로 "
        "탐지하며 컨테이너에는 `--sensor iceye` 로 넘긴다(산출물 이름 `WB_<TB>_ICEYE_*`). `image_date`: YYYY-MM-DD 또는 YYYYMMDD. "
        "`acquisition_time`(선택): HH:MM:SS/HHMMSS, 기본 000000 — 전처리 파일명 `Pre_<TB>_<센서라벨>_<yyyymmddTHHMMSS>.tif`(Umbra 는 UMBRA) 와 "
        "①사이드카 `AcquisitionDateTime` 에 쓴다.\n"
        "- `pre_file`(업로드) 또는 `pre_path`(서버 경로, 허용 루트 안만). `mask_file` 을 주면 탐지를 건너뛰고 그 마스크를 쓴다. "
        "`mask3_file`(선택, SAR): 3-class 마스크(0=유효,1=제외,2=바다; 영상과 같은 격자). 없으면 날짜별 인수 마스크 → 인접 날짜 마스크 → "
        "영상 유효 범위로 생성(바다 없음 가정, 경고) 순으로 쓴다.\n"
        "- 연계 파라미터 6종(order_id·scene_id·work_id·algm_id·product_type·product_id)은 탐지 잡·규약 파일명·완료 로그에 그대로 쓰인다.\n"
        "- 같은 장면(`<yyyymmdd>_<센서>`)의 파이프라인이 돌고 있으면 409. 잘못된 입력은 422, 모르는 지역은 404."
    ),
)
async def submit_scene(
    request: Request,
    image_date: Annotated[str, Form(description="YYYY-MM-DD 또는 YYYYMMDD")],
    region_id: Annotated[str, Form(description="지역 id (GET /scenes/options)")] = "nakdong_busan",
    sensor: Annotated[str, Form(description="ICEYE | Umbra | PlanetScope (planet·UMBRA 허용, 대소문자 무시)")] = "ICEYE",
    acquisition_time: Annotated[str | None, Form(description="HH:MM:SS 또는 HHMMSS (UTC), 기본 000000")] = None,
    pre_file: Annotated[UploadFile | None, File(description="전처리 GeoTIFF 업로드")] = None,
    pre_path: Annotated[str | None, Form(description="서버(호스트) 경로의 전처리 GeoTIFF — 허용 루트 안만")] = None,
    mask_file: Annotated[UploadFile | None, File(description="수체 마스크 GeoTIFF(0=육지,1=수체) — 주면 탐지 생략")] = None,
    mask_path: Annotated[str | None, Form(description="서버 경로의 수체 마스크 GeoTIFF(선택)")] = None,
    mask3_file: Annotated[UploadFile | None, File(description="SAR 3-class 마스크 GeoTIFF(0=유효,1=제외,2=바다; 영상과 같은 격자) — 주면 인수 마스크·자동 생성 대신 쓴다")] = None,
    gt_file: Annotated[UploadFile | None, File(description="GT 라벨 GeoTIFF(선택)")] = None,
    gt_path: Annotated[str | None, Form(description="서버 경로의 GT 라벨 GeoTIFF(선택)")] = None,
    order_id: Annotated[str | None, Form(description="통합 플랫폼 주문 ID")] = None,
    scene_id: Annotated[str | None, Form(description="통합 플랫폼 Scene ID (연계 파라미터; 우리 scene_id 와 별개)")] = None,
    work_id: Annotated[str | None, Form(description="통합 플랫폼 작업 ID")] = None,
    algm_id: Annotated[str | None, Form(description="알고리즘 ID. 예: AG000003")] = None,
    product_type: Annotated[str | None, Form(description="산출물 타입 코드 PT\\d{6}. 수체탐지 PT000003")] = None,
    product_id: Annotated[str | None, Form(description="원본 영상 제품 ID. 비우면 업로드 파일명에서 파생")] = None,
    detect: Annotated[bool, Form(description="마스크가 없을 때 WBMS 탐지를 돌린다")] = True,
    predict: Annotated[bool, Form(description="공통 격자 프레임이 3개 이상이면 면적 시계열 예측을 돌린다")] = True,
    note: Annotated[str | None, Form(description="메모(500자 이하)")] = None,
) -> dict[str, Any]:
    jobs = _jobs(request)
    services = jobs.services
    try:
        region = services.regions.get(region_id)
    except RegionNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from None
    catalog = services.wbms.catalog
    resolved = resolve_sensor(catalog, sensor)
    if resolved is None:
        known = sorted(str(p["name"]) for p in catalog.sensors.values())
        raise HTTPException(status_code=422, detail=f"sensor {sensor!r} 를 모릅니다 — 가능: {known}")
    sensor_key, sensor_name, sensor_type, sensor_label = resolved
    date_digits = _date_digits(image_date)
    time_digits = _time_digits(acquisition_time)
    try:
        liaison = liaison_params({
            "order_id": order_id, "scene_id": scene_id, "work_id": work_id,
            "algm_id": algm_id, "product_type": product_type, "product_id": product_id,
        })
    except LiaisonError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if note is not None and len(note) > 500:
        raise HTTPException(status_code=422, detail="note 는 500자 이하여야 합니다")
    if pre_file is None and not (pre_path or "").strip():
        raise HTTPException(status_code=422, detail="pre_file 또는 pre_path 가 필요합니다(전처리 GeoTIFF)")

    our_scene_id = f"{date_digits}_{sensor_name}"
    if jobs.active(our_scene_id):
        raise HTTPException(status_code=409, detail=f"장면 {our_scene_id} 의 파이프라인이 아직 진행 중입니다 — 끝난 뒤 다시 제출하세요")

    satellite = region.manifest.get("satellite") or {}
    testbed = (str(satellite.get("wbms_testbed") or "").strip().lower()) or None
    testbed_code = _testbed_code(catalog, testbed, region.id)
    has_mask = mask_file is not None or bool((mask_path or "").strip())
    if not has_mask and detect:
        adapter = services.segmenters.resolve(sensor_key, testbed) if testbed else None
        if adapter is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"지역 '{region.id}'(테스트베드 {testbed or '없음'}) × '{sensor_name}' 를 받는 탐지 어댑터가 없습니다 — "
                    "mask_file 을 함께 올리거나 detect=false 로 제출하세요"
                ),
            )

    settings = services.settings
    stamp = f"{date_digits}T{time_digits}"
    pre_stem = f"Pre_{testbed_code}_{sensor_label}_{stamp}"  # 파일 라벨(Umbra → UMBRA)은 표시명과 다를 수 있다
    work_dir = Path(settings.regions_user_dir) / region.id / "uploads" / f"{date_digits}_{sensor_name}"
    max_bytes = int(getattr(settings, "scene_upload_max_bytes", 0) or 0)
    written: list[Path] = []

    def discard() -> None:
        for path in written:
            path.unlink(missing_ok=True)

    try:
        if pre_file is not None:
            pre_source = await _write_upload(pre_file, work_dir / f"{pre_stem}.tif", max_bytes, "pre_file")
            written.append(pre_source)
            pre_is_upload = True
            pre_original_name = Path(pre_file.filename or "").name or None
        else:
            pre_source = _server_path(request, str(pre_path), "pre_path")
            pre_is_upload = False
            pre_original_name = pre_source.name
        mask_source: Path | None = None
        if mask_file is not None:
            mask_source = await _write_upload(mask_file, work_dir / f"WB_{testbed_code}_{sensor_label}_{stamp}.tif", max_bytes, "mask_file")
            written.append(mask_source)
        elif (mask_path or "").strip():
            mask_source = _server_path(request, str(mask_path), "mask_path")
        mask3_source: Path | None = None
        if mask3_file is not None:
            # SAR 3-class 마스크(0=유효 SAR, 1=제외, 2=바다). 탐지 단계가 인수 마스크·자동 생성보다 먼저 쓴다.
            mask3_source = await _write_upload(
                mask3_file, work_dir / f"{testbed or 'region'}_{date_digits}_mask3_utm.tif", max_bytes, "mask3_file",
            )
            written.append(mask3_source)
        gt_source: Path | None = None
        if gt_file is not None:
            gt_source = await _write_upload(gt_file, work_dir / f"GT_{testbed_code}_{sensor_label}_{stamp}.tif", max_bytes, "gt_file")
            written.append(gt_source)
        elif (gt_path or "").strip():
            gt_source = _server_path(request, str(gt_path), "gt_path")
    except HTTPException:
        discard()
        raise
    finally:
        for upload in (pre_file, mask_file, mask3_file, gt_file):
            if upload is not None:
                await upload.close()

    spec = {
        "scene_id": our_scene_id, "region_id": region.id, "sensor": sensor_name, "sensor_key": sensor_key,
        "sensor_type": sensor_type, "date_digits": date_digits, "acquisition_stamp": stamp,
        "testbed": testbed, "testbed_code": testbed_code, "work_dir": work_dir, "pre_stem": pre_stem,
        "pre_source": pre_source, "pre_is_upload": pre_is_upload, "pre_original_name": pre_original_name,
        "mask_source": mask_source, "mask3_source": mask3_source, "gt_source": gt_source, "liaison": liaison,
        "detect": detect, "predict": predict, "note": note,
    }
    try:
        snapshot = await run_in_threadpool(jobs.submit, spec)
    except SceneConflictError as exc:
        discard()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        discard()
        raise HTTPException(status_code=503, detail=f"cannot write scene state under {jobs.scenes_root}: {exc}") from exc
    prefix = services.api_prefix
    return {
        "scene_id": our_scene_id,
        "pipeline_id": snapshot["pipeline_id"],
        "status_url": f"{prefix}/scenes/{our_scene_id}",
    }


@router.get(
    "/{scene_id}",
    response_model=SceneStatus,
    summary="Scene pipeline status — six steps in fixed order, outputs, warnings",
    description=(
        "`steps[]` 는 register(장면 등록)·detect(수체 탐지)·area(면적 계산)·levels(수위 산출)·predict(시계열 예측)·"
        "deliver(규약 파일·로그) 고정 순서. `status` 는 queued|running|done|failed, `progress` 0..1. "
        "파이프라인 기록이 없어도 지역 장면 기록에 있는 scene_id 면 `pipeline_id=null` 로 합성해 돌려준다. 404: 모르는 장면."
    ),
)
async def scene_status(scene_id: str, request: Request) -> dict[str, Any]:
    jobs = _jobs(request)
    record = jobs.get(scene_id)
    if record is None:
        record = await run_in_threadpool(jobs.synthesize, scene_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown scene {scene_id!r}")
    return record


@router.get(
    "/{scene_id}/files/{name}",
    summary="Download one file from the scene output folder",
    description=(
        "장면 산출 폴더(`outputs.folder`)의 파일 하나 — 규약 파일(PT…tif)·마스크(WB_…tif)·quicklook.png·overlay.png·"
        "area.json·levels.json·prediction.json·status.json. `name` 은 파일명(basename)만 허용하며 경로 구분자·`..` 은 404."
    ),
    response_class=FileResponse,
)
async def scene_file(scene_id: str, name: str, request: Request) -> FileResponse:
    jobs = _jobs(request)
    folder = jobs.folder_for(scene_id)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"unknown scene {scene_id!r} (no pipeline record)")
    if name != Path(name).name or name in {"", ".", ".."} or name.startswith("."):
        raise HTTPException(status_code=404, detail=f"{name!r} is not a file of scene {scene_id}")
    base = folder.resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {name}")
    media = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    attachment = path.suffix.lower() in (".tif", ".tiff", ".npy")
    return FileResponse(
        path, media_type=media, filename=path.name if attachment else None,
        content_disposition_type="attachment" if attachment else "inline",
    )
