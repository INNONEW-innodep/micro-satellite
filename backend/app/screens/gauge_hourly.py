"""시 단위 수위 — 수집 잡(``POST /regions/{id}/gauge/collect-hourly``)·자료 현황·전망(``…/forecast-hourly``). 2026-10-20.

일 수위 수집(``wamis.py``)과 같은 잡 꼴(``ScreenJobs``, 지역당 동시 하나)이고 알고리즘·저장은 ``gauge/hourly.py`` 에 있다.
지점의 WAMIS 코드는 지역 시계열의 지점 메타(``source_code``)에서 가져오므로 보통은 지점 id 만 보내면 된다.
저장 위치는 지역 사용자 사본 ``gauge/hourly/<지점>.json`` — 내장 지역도 사용자 디렉터리에만 쓴다(내장 자료는 건드리지 않는다).
"""

from __future__ import annotations

import threading
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from ..gauge.hourly import (
    HORIZONS_HOURS,
    MAX_HORIZON_HOURS,
    MIN_HOURS_TO_FIT,
    HourlyForecaster,
    holdout_eval,
    hourly_path,
    load_hourly,
    merge_hourly,
    save_hourly,
    wamis_fetch_hourly,
)
from ..gauge.regions import RegionNotFoundError, RegionRegistry, StationNotFoundError, _safe_station_dirname
from .jobs import JobConflictError, ProgressUpdate, ScreenJobs

router = APIRouter(tags=["regions"])

JOB_KIND = "gauge_hourly_collect"


def _regions(request: Request) -> RegionRegistry:
    return request.app.state.regions


def _region_or_404(request: Request, region_id: str):
    try:
        return _regions(request).get(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from exc


def _jobs(request: Request) -> ScreenJobs:
    jobs = getattr(request.app.state, "gauge_hourly_jobs", None)
    if jobs is None:
        jobs = ScreenJobs(JOB_KIND)
        request.app.state.gauge_hourly_jobs = jobs
    return jobs


def fetch_hourly_rows(code: str, start: Date, end: Date) -> tuple[list[tuple[str, float | None]], list[str]]:
    """인터넷 필요. 테스트가 바꿔 끼우는 지점."""

    return wamis_fetch_hourly(code, start, end)


def _station_file(registry: RegionRegistry, region_id: str, station_id: str) -> Path:
    return hourly_path(registry._user_region_dir(region_id), _safe_station_dirname(station_id))


class _ForecasterCache:
    """지점 파일(mtime)·원점 인덱스가 같으면 학습한 모델을 다시 쓴다 — 화면이 예측시간만 바꿔 다시 물을 때 수 초를 아낀다."""

    def __init__(self, limit: int = 16) -> None:
        self._lock = threading.Lock()
        self._items: dict[tuple[str, str, float], HourlyForecaster] = {}
        self._limit = limit

    def get(self, region_id: str, station_id: str, path: Path) -> HourlyForecaster | None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        key = (region_id, station_id, mtime)
        with self._lock:
            fc = self._items.get(key)
        if fc is not None:
            return fc
        doc = load_hourly(path)
        if doc is None:
            return None
        fc = HourlyForecaster.from_doc(doc)
        with self._lock:
            if len(self._items) >= self._limit:
                self._items.pop(next(iter(self._items)))
            self._items[key] = fc
        return fc


def _cache(request: Request) -> _ForecasterCache:
    cache = getattr(request.app.state, "gauge_hourly_cache", None)
    if cache is None:
        cache = _ForecasterCache()
        request.app.state.gauge_hourly_cache = cache
    return cache


# ── 수집 잡 ─────────────────────────────────────────────────────────────────────────


class HourlyStation(BaseModel):
    id: str = Field(max_length=80, description="지역 시계열의 지점 id(한글 가능)")
    code: str | None = Field(default=None, pattern=r"^\d{7}$", description="WAMIS 관측소 코드(7자리). 비우면 지점 메타 source_code")

    @model_validator(mode="after")
    def clean(self) -> HourlyStation:
        self.id = self.id.strip()
        if not self.id or "/" in self.id or "\\" in self.id:
            raise ValueError(f"지점 id 가 올바르지 않습니다: {self.id!r}")
        return self


class HourlyCollectRequest(BaseModel):
    stations: list[HourlyStation] = Field(min_length=1, max_length=20)
    start: Date = Field(description="수집 시작일(포함)")
    end: Date | None = Field(default=None, description="수집 종료일(포함). 비우면 오늘")
    async_run: bool = Field(default=True)

    @model_validator(mode="after")
    def check_range(self) -> HourlyCollectRequest:
        end = self.end or Date.today()
        if end < self.start:
            raise ValueError("end 가 start 보다 앞입니다")
        if (end - self.start).days > 366 * 6:
            raise ValueError("한 번에 6년까지만 받을 수 있습니다 — 나눠서 받으세요")
        self.end = end
        return self


def build_hourly_collect_run(registry: RegionRegistry, region_id: str, payload: HourlyCollectRequest):
    series_set = registry.load_series(region_id)
    specs: list[tuple[HourlyStation, str, str]] = []
    for spec in payload.stations:
        code = spec.code
        name = spec.id
        try:
            st = series_set.station(spec.id)
            name = st.name or spec.id
            code = code or (st.source_code or None)
        except StationNotFoundError:
            if code is None:
                raise HTTPException(status_code=422, detail=f"지점 {spec.id!r} 는 지역 시계열에 없고 code 도 없습니다 — 7자리 WAMIS 코드를 함께 보내세요")
        if not code:
            raise HTTPException(status_code=422, detail=f"지점 {spec.id!r} 의 WAMIS 코드를 모릅니다(지점 메타 source_code 없음) — code 를 보내세요")
        specs.append((spec, name, code))

    def run(update: ProgressUpdate) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        total = len(specs)
        for index, (spec, name, code) in enumerate(specs):
            update(progress=index / total, current=spec.id)
            entry: dict[str, Any] = {"id": spec.id, "name": name, "code": code, "status": "running", "n_rows": 0, "n_new": 0, "n_total": 0, "period": None, "failures": [], "note": None}
            try:
                rows, failures = fetch_hourly_rows(code, payload.start, payload.end)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                entry.update(status="failed", note=f"수집 실패: {exc}")
                results.append(entry)
                continue
            entry["n_rows"] = len(rows)
            entry["failures"] = failures[:24]
            if not rows:
                entry.update(status="empty", note="받은 시 자료가 없습니다 — 코드·기간을 확인하세요")
                results.append(entry)
                continue
            path = _station_file(registry, region_id, spec.id)
            doc = merge_hourly(load_hourly(path), rows, station={"id": spec.id, "name": name, "code": code})
            save_hourly(path, doc)
            entry.update(status="done", n_new=doc.get("n_new", 0), n_total=doc["n"], period=doc["period"])
            if failures:
                entry["note"] = f"{len(failures)}개 달은 받지 못했습니다(기존 자료는 유지)"
            results.append(entry)
        return {"stations": results, "current": None, "n_done": sum(1 for r in results if r["status"] == "done"), "n_failed": sum(1 for r in results if r["status"] in ("failed", "empty"))}

    return run


@router.post(
    "/regions/{region_id}/gauge/collect-hourly",
    status_code=202,
    responses={200: {"description": "async_run=false — 완료 상태"}, 202: {"description": "잡 접수 {job_id, status_url}"}, 404: {"description": "모르는 지역"}, 409: {"description": "진행 중인 시 단위 수집 잡"}, 422: {"description": "지점·기간 검증 실패"}},
    summary="Collect WAMIS hourly levels (wl_hrdata) for stations into the region user copy (background job)",
    description=(
        "`stations[]` {id, code?} 마다 WAMIS 시 수위를 월 단위로 받아 `gauge/hourly/<지점>.json` 에 병합한다(같은 시각은 새 값, 결측은 기존 유지). "
        "code 를 비우면 지역 시계열의 지점 메타 source_code 를 쓴다. 진행은 `GET …/gauge/collect-hourly/jobs/{job_id}`. 지역당 동시에 하나(409). "
        "WAMIS 는 무인증이며 약 2시간 지연으로 최신 시각까지 준다."
    ),
)
async def collect_hourly(region_id: str, payload: HourlyCollectRequest, request: Request):
    _region_or_404(request, region_id)
    registry = _regions(request)
    jobs = _jobs(request)
    run = build_hourly_collect_run(registry, region_id, payload)
    data = {
        "start": payload.start.isoformat(), "end": payload.end.isoformat() if payload.end else None,
        "stations": [{"id": s.id, "code": s.code, "status": "queued", "n_rows": 0, "n_new": 0, "n_total": 0, "period": None, "note": None} for s in payload.stations],
        "n_stations": len(payload.stations), "current": None,
    }
    prefix = str(request.app.state.settings.api_prefix)
    try:
        if not payload.async_run:
            return JSONResponse(status_code=200, content=await run_in_threadpool(jobs.run_sync, region_id, run, data=data))
        snapshot = jobs.submit(region_id, run, data=data)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content={"job_id": snapshot["job_id"], "status": snapshot["status"], "region_id": region_id, "status_url": f"{prefix}/regions/{region_id}/gauge/collect-hourly/jobs/{snapshot['job_id']}"})


@router.get("/regions/{region_id}/gauge/collect-hourly/jobs", summary="List hourly collection jobs of a region (newest first)")
async def collect_hourly_jobs(region_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    items = _jobs(request).list(region_id)
    return {"region_id": region_id, "items": items, "total": len(items)}


@router.get("/regions/{region_id}/gauge/collect-hourly/jobs/{job_id}", summary="Poll one hourly collection job")
async def collect_hourly_job(region_id: str, job_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    item = _jobs(request).get(job_id)
    if item is None or item.get("region_id") != region_id:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r} for region {region_id!r}")
    return item


# ── 자료 현황 · 전망 ─────────────────────────────────────────────────────────────────


def _hourly_status_item(registry: RegionRegistry, region_id: str, station_id: str, *, with_eval: bool) -> dict[str, Any]:
    path = _station_file(registry, region_id, station_id)
    doc = load_hourly(path)
    if doc is None:
        return {"station_id": station_id, "available": False, "n_hours": 0, "period": None, "note_ko": "시 단위 자료가 아직 없어요 — collect-hourly 로 받으세요"}
    item: dict[str, Any] = {
        "station_id": station_id, "available": True, "n_hours": int(doc.get("n") or 0), "period": doc.get("period"),
        "source": doc.get("source"), "source_code": doc.get("source_code"), "updated_at": doc.get("updated_at"),
        "enough_to_forecast": int(doc.get("n") or 0) >= MIN_HOURS_TO_FIT,
    }
    if with_eval:
        item["holdout"] = holdout_eval(doc)
    return item


@router.get(
    "/regions/{region_id}/gauge/hourly",
    summary="Hourly-level data status per station of a region (what has been collected, optional holdout evaluation)",
    description="`stations[]` {station_id, available, n_hours, period[YYYYMMDDHH], enough_to_forecast, holdout?}. `with_eval=true` 면 앞 80 %/뒤 20 % 홀드아웃(예측시간 1·3·6·12·24h RMSE·직전값 대비 개선)을 계산한다(지점당 ~0.5초).",
)
async def hourly_status(region_id: str, request: Request, with_eval: Annotated[bool, Query()] = False) -> dict[str, Any]:
    _region_or_404(request, region_id)
    registry = _regions(request)
    series_set = registry.load_series(region_id)
    items = await run_in_threadpool(lambda: [_hourly_status_item(registry, region_id, st.id, with_eval=with_eval) for st in series_set.stations])
    return {
        "region_id": region_id, "stations": items, "n_available": sum(1 for i in items if i["available"]),
        "horizons_hours": list(HORIZONS_HOURS), "max_horizon_hours": MAX_HORIZON_HOURS,
        "note_ko": "시 단위 자료는 지역 사용자 사본 gauge/hourly/<지점>.json 에 있고 WAMIS wl_hrdata(무인증)로 받는다.",
    }


@router.get(
    "/regions/{region_id}/gauge/{station_id}/hourly",
    summary="Hourly-level data status of one station (+ holdout evaluation with with_eval=true)",
)
async def hourly_station_status(region_id: str, station_id: str, request: Request, with_eval: Annotated[bool, Query()] = True) -> dict[str, Any]:
    _region_or_404(request, region_id)
    registry = _regions(request)
    item = await run_in_threadpool(_hourly_status_item, registry, region_id, station_id, with_eval=with_eval)
    if not item["available"]:
        raise HTTPException(status_code=404, detail=f"지점 {station_id!r} 의 시 단위 자료가 없습니다 — POST /regions/{region_id}/gauge/collect-hourly 로 먼저 받으세요")
    return {"region_id": region_id, **item}


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:16] if fmt == "%Y-%m-%dT%H:%M" else text, fmt)
        except ValueError:
            continue
    raise HTTPException(status_code=422, detail=f"as_of 는 YYYY-MM-DD, YYYY-MM-DDTHH 또는 YYYY-MM-DDTHH:MM 이어야 합니다: {value!r}")


@router.get(
    "/regions/{region_id}/gauge/{station_id}/forecast-hourly",
    summary="Hourly level forecast from an origin hour (ridge AR fitted on the fly; persistence when data is short)",
    description=(
        "`as_of` 원점 시각(비우면 자료의 마지막 시각; 날짜만 주면 그날 0시 이전 마지막 관측), `horizon_hours` 1..48(기본 24), `history_hours` 최근 실측(기본 72). "
        "응답 `steps[]` {horizon_hours, target_time, forecast_m, lower/upper_1sigma_m, observed_m(있으면), method}, `verification`(원점 뒤 실측이 있을 때), "
        "`model`(λ·검증 RMSE·직전값 대비 개선). 모델은 원점까지의 자료로 호출 시점에 학습하므로 미래를 보지 않는다. 지상 수위계(WAMIS 시 자료) 전용."
    ),
)
async def forecast_hourly(
    region_id: str, station_id: str, request: Request,
    as_of: Annotated[str | None, Query(description="원점 시각 YYYY-MM-DDTHH:MM (비우면 자료 끝)")] = None,
    horizon_hours: Annotated[int, Query(ge=1, le=MAX_HORIZON_HOURS)] = 24,
    history_hours: Annotated[int, Query(ge=1, le=24 * 30)] = 72,
) -> dict[str, Any]:
    _region_or_404(request, region_id)
    registry = _regions(request)
    path = _station_file(registry, region_id, station_id)
    fc = _cache(request).get(region_id, station_id, path)
    if fc is None:
        raise HTTPException(status_code=404, detail=f"지점 {station_id!r} 의 시 단위 자료가 없습니다 — POST /regions/{region_id}/gauge/collect-hourly 로 먼저 받으세요")
    origin = _parse_as_of(as_of)
    clamped = False
    note = None
    if origin is not None and fc.times and origin > fc.times[-1]:
        note = f"요청한 원점 {origin:%Y-%m-%d %H시} 는 자료 끝({fc.times[-1]:%Y-%m-%d %H시}) 뒤라 자료 끝에서 출발합니다"
        origin, clamped = None, True
    try:
        result = await run_in_threadpool(fc.forecast, as_of=origin, horizon_hours=horizon_hours, history_hours=history_hours)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result.update({"region_id": region_id, "station_id": station_id, "as_of_requested": as_of, "as_of_clamped": clamped, "origin_note_ko": note})
    return result


__all__ = ["HourlyCollectRequest", "HourlyStation", "build_hourly_collect_run", "fetch_hourly_rows", "router"]
