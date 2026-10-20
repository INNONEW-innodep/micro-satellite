"""외생 입력 피드(2026-10-20) — 기상청 단기예보 수집·조회, 댐 운영 일자료(WAMIS mn_dtdata) 수집 잡·조회.

- ``POST /regions/{id}/weather/forecast-collect`` : 지점(또는 위경도)의 단기예보를 받아 ``weather/kma_forecast_daily.json`` 에 저장.
  키는 서버 ``KMA_API_KEY``(ASOS 와 같은 키, 단기예보 서비스 활용신청 필요). 없으면 422. 예측기 ``ridge_arx_future``·
  ``ridge_arx_season_future`` 가 자료 끝 원점에서 이 파일의 강수를 원점 이후 입력으로 쓴다.
- ``GET  /regions/{id}/weather/forecast``         : 저장된 예보 행과 발표 시각.
- ``POST /regions/{id}/dam/collect``              : WAMIS 댐 일 운영 자료(저수위·유입·총방류·여수로·강우)를 월 단위로 받아
  ``weather/dam_daily.json`` 에 병합(배경 잡, 지역당 하나). 무인증·인터넷 필요. 예측기 ``ridge_arx_season`` 의 댐 입력.
- ``GET  /regions/{id}/dam``                      : 저장된 댐 자료 현황.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from ..gauge import kma_forecast
from ..gauge.regions import RegionNotFoundError, RegionRegistry
from .jobs import JobConflictError, ProgressUpdate, ScreenJobs

router = APIRouter(tags=["regions"])

DAM_SCHEMA = "watercast.dam.daily/1"
DAM_URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkd/mn_dtdata?damcd={code}&startdt={start}&enddt={end}&output=json"
DAM_FIELDS = {"rwl": "dam_level_m", "iqty": "inflow_cms", "tdqty": "outflow_cms", "edqty": "power_cms", "spdqty": "spillway_cms", "otltdqty": "other_cms", "itqty": "intake_cms", "rf": "rain_mm"}
_OBJ = re.compile(r"\{[^{}]*\"obsymd\":\"(\d{8})\"[^{}]*\}")
_KV = re.compile(r'"(\w+)":"([^"]*)"')
_LOCK = threading.Lock()


def _regions(request: Request) -> RegionRegistry:
    return request.app.state.regions


def _region_or_404(request: Request, region_id: str):
    try:
        return _regions(request).get(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from exc


def _jobs(request: Request) -> ScreenJobs:
    jobs = getattr(request.app.state, "dam_collect_jobs", None)
    if jobs is None:
        jobs = ScreenJobs("dam_collect")
        request.app.state.dam_collect_jobs = jobs
    return jobs


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with _LOCK:
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# ── 기상청 단기예보 ───────────────────────────────────────────────────────────────────


def kma_collect(service_key: str, lat: float, lon: float) -> dict[str, Any]:
    """인터넷 필요. 테스트가 바꿔 끼우는 지점."""

    return kma_forecast.collect(service_key, lat, lon)


class ForecastCollectRequest(BaseModel):
    station_id: str | None = Field(default=None, max_length=80, description="좌표를 가져올 지점 id(비우면 좌표가 있는 첫 지점)")
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def check(self) -> ForecastCollectRequest:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat 과 lon 은 함께 보내야 합니다")
        return self


def _pick_point(registry: RegionRegistry, region_id: str, payload: ForecastCollectRequest) -> tuple[float, float, str]:
    if payload.lat is not None and payload.lon is not None:
        return float(payload.lat), float(payload.lon), "request"
    series_set = registry.load_series(region_id)
    stations = series_set.stations
    if payload.station_id:
        stations = [s for s in stations if s.id == payload.station_id]
        if not stations:
            raise HTTPException(status_code=422, detail=f"지점 {payload.station_id!r} 가 지역 시계열에 없습니다")
    for st in stations:
        if st.lat is not None and st.lon is not None:
            return float(st.lat), float(st.lon), st.id
    record = registry.get(region_id)
    bbox = record.manifest.get("bbox_wgs84")
    if isinstance(bbox, list) and len(bbox) == 4:
        return (float(bbox[1]) + float(bbox[3])) / 2, (float(bbox[0]) + float(bbox[2])) / 2, "bbox_center"
    raise HTTPException(status_code=422, detail="좌표를 아는 지점이 없습니다 — lat/lon 을 함께 보내세요")


@router.post(
    "/regions/{region_id}/weather/forecast-collect",
    responses={200: {"description": "저장된 예보 문서 요약"}, 404: {"description": "모르는 지역"}, 422: {"description": "키 없음·좌표 없음"}, 502: {"description": "기상청 응답 오류"}},
    summary="Collect the latest KMA short-range forecast (getVilageFcst) for the region point and store daily rain/temperature rows",
    description=(
        "서버 `KMA_API_KEY` 로 기상청 단기예보(최근 발표분, +3일)를 받아 날짜별 강수 합·기온 평균·강수확률 최대로 집계해 "
        "`weather/kma_forecast_daily.json` 에 저장한다. 좌표는 `station_id`(비우면 좌표 있는 첫 지점) 또는 `lat/lon`. "
        "이후 `GET …/gauge/{station}/forecast?forecaster=ridge_arx_future|ridge_arx_season_future` 가 자료 끝 원점에서 이 예보를 쓴다 "
        "(응답 `future_exog_feed`). 과거 발표분은 이 API 로 받을 수 없다."
    ),
)
async def weather_forecast_collect(region_id: str, payload: ForecastCollectRequest, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    registry = _regions(request)
    key = os.getenv("KMA_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=422, detail="기상청 단기예보 수집에는 서버의 KMA_API_KEY 가 필요합니다 — 키를 설정하고(단기예보 서비스 활용신청 포함) 다시 시도하세요")
    lat, lon, origin = _pick_point(registry, region_id, payload)
    try:
        doc = await run_in_threadpool(kma_collect, key, lat, lon)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"기상청 단기예보를 받지 못했습니다: {exc}") from exc
    doc.update({"region_id": region_id, "point_source": origin})
    path = registry.weather_forecast_write_path(region_id)
    _write_json(path, doc)
    return {"region_id": region_id, "stored": str(path.name), **{k: v for k, v in doc.items() if k != "rows"}, "rows": doc["rows"], "n_days": len(doc["rows"])}


@router.get("/regions/{region_id}/weather/forecast", summary="Stored KMA short-range forecast rows of the region (daily rain/temperature) and its issue time")
async def weather_forecast_status(region_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    rows, meta = _regions(request).load_weather_forecast(region_id)
    return {"region_id": region_id, "available": bool(rows), "n_days": len(rows), "rows": rows, **{k: v for k, v in meta.items() if k in ("provider", "issued_at", "base_date", "base_time", "nx", "ny", "lat", "lon", "fetched_at", "point_source", "note_ko")},
            "note_ko": meta.get("note_ko") or "아직 받은 단기예보가 없어요 — POST …/weather/forecast-collect"}


# ── 댐 운영 일자료 ─────────────────────────────────────────────────────────────────


def parse_dam_payload(payload: str) -> list[dict[str, Any]]:
    """정상 JSON 이든 잘린 응답이든 객체 단위로 건진다 → [{date, dam_level_m, inflow_cms, outflow_cms, …}]."""

    out = []
    for m in _OBJ.finditer(payload or ""):
        kv = dict(_KV.findall(m.group(0)))
        ymd = kv.get("obsymd") or ""
        row: dict[str, Any] = {"date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"}
        for src, dst in DAM_FIELDS.items():
            raw = (kv.get(src) or "").strip()
            try:
                row[dst] = float(raw) if raw else None
            except ValueError:
                row[dst] = None
        out.append(row)
    return out


def fetch_dam_rows(code: str, start: Date, end: Date) -> tuple[list[dict[str, Any]], list[str]]:
    """월 단위로 WAMIS 를 부른다(연 단위는 잘린다). 인터넷 필요. 테스트가 바꿔 끼우는 지점."""

    from .wamis import load_wamis_module  # 일 수위 스크립트의 재시도 fetch 재사용

    module = load_wamis_module()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        hi = min(end, nxt - timedelta(days=1))
        text = module.fetch(DAM_URL.format(code=code, start=cur.strftime("%Y%m%d"), end=hi.strftime("%Y%m%d")), attempts=3, timeout=60.0)
        got = parse_dam_payload(text) if text and not str(text).startswith("__error__") else []
        if not got:
            failures.append(f"{cur:%Y-%m}: {str(text)[:60] if text else 'empty'}")
        rows.extend(got)
        cur = nxt
    return rows, failures


class DamCollectRequest(BaseModel):
    dam_code: str = Field(pattern=r"^\d{7}$", description="WAMIS 댐 코드(7자리). 대청 3008110")
    dam_name: str | None = Field(default=None, max_length=80)
    start: Date
    end: Date | None = None
    async_run: bool = True

    @model_validator(mode="after")
    def check(self) -> DamCollectRequest:
        end = self.end or Date.today()
        if end < self.start:
            raise ValueError("end 가 start 보다 앞입니다")
        if (end - self.start).days > 366 * 12:
            raise ValueError("한 번에 12년까지만")
        self.end = end
        return self


def build_dam_collect_run(registry: RegionRegistry, region_id: str, payload: DamCollectRequest):
    def run(update: ProgressUpdate) -> dict[str, Any]:
        update(progress=0.05, current="WAMIS 요청 중")
        rows, failures = fetch_dam_rows(payload.dam_code, payload.start, payload.end)  # type: ignore[arg-type]
        if not rows:
            raise RuntimeError("받은 댐 자료가 없습니다 — 댐 코드·기간을 확인하세요" + (f" ({failures[0]})" if failures else ""))
        update(progress=0.8, current="병합 중")
        path = registry.dam_write_path(region_id)
        existing: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                for r in json.loads(path.read_text(encoding="utf-8")).get("rows") or []:
                    existing[str(r.get("date"))] = dict(r)
            except (OSError, ValueError):
                existing = {}
        n_new = 0
        for r in rows:
            if r["date"] not in existing:
                n_new += 1
            existing[r["date"]] = r
        keys = sorted(existing)
        doc = {
            "schema": DAM_SCHEMA, "region_id": region_id, "dam_code": payload.dam_code, "dam_name": payload.dam_name,
            "source": "WAMIS wkd/mn_dtdata (개방 API, 무인증)", "fields_ko": {"dam_level_m": "저수위 EL.m", "inflow_cms": "유입량 m³/s", "outflow_cms": "총방류량 m³/s", "power_cms": "발전방류 m³/s", "spillway_cms": "여수로 방류 m³/s", "other_cms": "기타 방류 m³/s", "intake_cms": "취수량 m³/s", "rain_mm": "댐 강우 mm"},
            "n": len(keys), "period": [keys[0], keys[-1]] if keys else None, "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "rows": [existing[k] for k in keys],
            "note_ko": "예측기 ridge_arx_season 의 댐 입력(원점 이전 총방류·유입·저수위). 방류 계획은 공개 API 가 없어 들어가지 않는다.",
        }
        _write_json(path, doc)
        registry.update_manifest_section(region_id, "dam", {"code": payload.dam_code, "name": payload.dam_name, "daily_file": "weather/dam_daily.json"})
        return {"n_rows": len(rows), "n_new": n_new, "n_total": len(keys), "period": doc["period"], "failures": failures[:24], "current": None,
                "note": (f"{len(failures)}개 달은 받지 못했습니다(기존 자료는 유지)" if failures else None)}

    return run


@router.post(
    "/regions/{region_id}/dam/collect",
    status_code=202,
    responses={200: {"description": "async_run=false — 완료 상태"}, 202: {"description": "잡 접수"}, 404: {"description": "모르는 지역"}, 409: {"description": "진행 중"}, 422: {"description": "검증 실패"}},
    summary="Collect WAMIS daily dam operations (level, inflow, outflow, spillway, rain) into the region (background job)",
    description="`dam_code`(7자리) 의 일 운영 자료를 월 단위로 받아 `weather/dam_daily.json` 에 병합하고 region.json `dam` 섹션을 갱신한다. 진행은 `GET …/dam/collect/jobs/{job_id}`.",
)
async def dam_collect(region_id: str, payload: DamCollectRequest, request: Request):
    _region_or_404(request, region_id)
    registry = _regions(request)
    jobs = _jobs(request)
    run = build_dam_collect_run(registry, region_id, payload)
    data = {"dam_code": payload.dam_code, "dam_name": payload.dam_name, "start": payload.start.isoformat(), "end": payload.end.isoformat() if payload.end else None, "current": None}
    prefix = str(request.app.state.settings.api_prefix)
    try:
        if not payload.async_run:
            return JSONResponse(status_code=200, content=await run_in_threadpool(jobs.run_sync, region_id, run, data=data))
        snapshot = jobs.submit(region_id, run, data=data)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content={"job_id": snapshot["job_id"], "status": snapshot["status"], "region_id": region_id, "status_url": f"{prefix}/regions/{region_id}/dam/collect/jobs/{snapshot['job_id']}"})


@router.get("/regions/{region_id}/dam/collect/jobs/{job_id}", summary="Poll one dam collection job")
async def dam_collect_job(region_id: str, job_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    item = _jobs(request).get(job_id)
    if item is None or item.get("region_id") != region_id:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r} for region {region_id!r}")
    return item


@router.get("/regions/{region_id}/dam", summary="Stored dam daily data status of the region (code, period, last rows)")
async def dam_status(region_id: str, request: Request, tail: Annotated[int, Query(ge=0, le=400)] = 7) -> dict[str, Any]:
    _region_or_404(request, region_id)
    rows, meta = _regions(request).load_dam_rows(region_id)
    return {"region_id": region_id, "available": bool(rows), "n": len(rows), "period": meta.get("period"), "dam_code": meta.get("dam_code"), "dam_name": meta.get("dam_name"),
            "source": meta.get("source"), "fields_ko": meta.get("fields_ko"), "updated_at": meta.get("updated_at"), "last_rows": rows[-tail:] if tail else [],
            "note_ko": meta.get("note_ko") or "아직 받은 댐 자료가 없어요 — POST …/dam/collect"}


__all__ = ["DamCollectRequest", "ForecastCollectRequest", "build_dam_collect_run", "fetch_dam_rows", "kma_collect", "parse_dam_payload", "router"]
