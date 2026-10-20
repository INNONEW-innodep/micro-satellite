"""기상청 ASOS 일자료 수집 잡(``POST /regions/{id}/weather/collect``) → 지역 사본 ``weather/asos_daily.json``.

``POST /weather/observations`` 가 쓰는 것과 **같은** 내부 서비스(``weather_service`` 패키지의 ``AsosDailyClient`` /
``SampleWeatherProvider``)를 직접 부른다(자기 자신에게 HTTP 를 보내지 않는다). ``scripts/fetch_region_weather.py`` 처럼
연 단위로 받고 같은 ``normalise`` 로 행을 정규화해, 예측기(``RegionRegistry.load_weather_rows``)가 읽는 스키마
``watercast.weather.daily/1`` 로 저장한다. 서버에 KMA 키가 없으면 합성 샘플 소스로 떨어지며 — 그 사실을 잡·파일·
매니페스트에 그대로 적는다(합성값을 실측처럼 두지 않는다).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib.util
import json
import os
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from ..gauge.regions import RegionNotFoundError, RegionRegistry
from ..weather import _record_to_weather_row, _weather_service
from .jobs import JobConflictError, ProgressUpdate, ScreenJobs

router = APIRouter(tags=["weather"])

REPO_ROOT = Path(__file__).resolve().parents[3]
WEATHER_SCRIPT = REPO_ROOT / "scripts" / "fetch_region_weather.py"
PROVIDER_ASOS = "KMA_ASOS"
PROVIDER_SAMPLE = "SAMPLE_SYNTHETIC"
SAMPLE_WARNING = (
    "서버에 KMA_API_KEY 가 없어(또는 source=sample 요청) 합성 샘플 기상으로 저장했습니다 — 실측이 아니므로 성능 수치의 근거로 쓰지 "
    "마세요. 실측(asos)으로 다시 수집하면 덮어씁니다."
)

_MODULE: Any = None
_MODULE_LOCK = threading.Lock()


def load_weather_script() -> Any:
    """``scripts/fetch_region_weather.py`` 를 한 번만 모듈로 올린다 — ``normalise``·``SCHEMA`` 를 그대로 쓴다."""

    global _MODULE
    with _MODULE_LOCK:
        if _MODULE is None:
            spec = importlib.util.spec_from_file_location("watercast_fetch_region_weather", WEATHER_SCRIPT)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {WEATHER_SCRIPT}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULE = module
    return _MODULE


def _jobs(request: Request) -> ScreenJobs:
    jobs = getattr(request.app.state, "weather_collect_jobs", None)
    if jobs is None:
        jobs = ScreenJobs("weather_collect")
        request.app.state.weather_collect_jobs = jobs
    return jobs


def _regions(request: Request) -> RegionRegistry:
    return request.app.state.regions


def _region_or_404(request: Request, region_id: str):
    try:
        return _regions(request).get(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from exc


class WeatherCollectRequest(BaseModel):
    station_id: str = Field(min_length=1, max_length=16, description="기상청 ASOS 지점 번호 (예: 133 대전, 159 부산)")
    start: dt.date
    end: dt.date
    source: Literal["auto", "asos", "sample"] = Field(
        default="auto", description="auto: KMA_API_KEY 가 있으면 asos, 없으면 sample(경고). asos 는 키가 없으면 422."
    )
    async_run: bool = True

    @model_validator(mode="after")
    def check_period(self) -> WeatherCollectRequest:
        if self.end < self.start:
            raise ValueError(f"start({self.start}) 가 end({self.end}) 보다 늦습니다")
        return self


def choose_provider(service: Any, source: str) -> tuple[str, Any]:
    """(소스 이름, 제공자). ``POST /weather/observations`` 와 같은 클래스를 쓴다."""

    key_configured = bool(os.getenv("KMA_API_KEY", "").strip())
    if source == "sample" or (source == "auto" and not key_configured):
        return "sample", service.SampleWeatherProvider()
    return "asos", service.AsosDailyClient()  # 키가 없으면 WeatherConfigurationError


def build_weather_run(registry: RegionRegistry, region_id: str, payload: WeatherCollectRequest, service: Any, source: str, provider: Any):
    script = load_weather_script()
    station = service.resolve_station(payload.station_id)
    first, last = payload.start, payload.end
    years_all = list(range(first.year, last.year + 1))

    def run(update: ProgressUpdate) -> dict[str, Any]:
        years: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []
        warnings: list[str] = [SAMPLE_WARNING] if source == "sample" else []
        station_name = getattr(station, "name", None)
        for index, year in enumerate(years_all):
            lo = max(first, dt.date(year, 1, 1))
            hi = min(last, dt.date(year, 12, 31))
            entry: dict[str, Any] = {"year": year, "start": lo.isoformat(), "end": hi.isoformat(), "n_rows": 0, "source": source, "status": "running", "note": None}
            update(progress=index / len(years_all), years=years + [entry], n_rows=len(raw_rows))
            try:
                series = asyncio.run(provider.get_daily_observations(start_date=lo, end_date=hi, station=station))
                rows = [_record_to_weather_row(r, source).model_dump(mode="json") for r in series.records]
            except Exception as exc:  # noqa: BLE001 - 한 해의 실패는 기록하고 계속(전부 실패면 잡 실패)
                entry.update(status="failed", note=f"{type(exc).__name__}: {exc}")
                years.append(entry)
                continue
            station_name = station_name or getattr(series, "station", None) and getattr(series.station, "name", None)
            raw_rows.extend(rows)
            entry.update(n_rows=len(rows), status="done")
            years.append(entry)
        rows = script.normalise(raw_rows)
        if not rows:
            failed = [f"{y['year']}: {y.get('note')}" for y in years if y["status"] == "failed"]
            raise RuntimeError("수집된 기상 행이 없습니다" + (" — " + "; ".join(failed[:3]) if failed else ""))
        for year in years:
            if year["status"] == "failed":
                warnings.append(f"{year['year']}년 수집 실패: {year.get('note')}")
        provider_name = PROVIDER_ASOS if source == "asos" else PROVIDER_SAMPLE
        # 기존 파일과 병합(같은 제공자·같은 지점일 때만 날짜 기준 덮어쓰기). 다르면 교체 — 합성·실측이 섞이지 않게.
        path = registry.weather_write_path(region_id)
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
        merged: dict[str, dict[str, Any]] = {}
        same_origin = (
            isinstance(existing, dict)
            and str(existing.get("provider") or "") == provider_name
            and str(existing.get("station_id") or "") == str(station.id)
        )
        if same_origin:
            for row in existing.get("rows") or []:
                if isinstance(row, dict) and row.get("date"):
                    merged[str(row["date"])] = dict(row)
        elif existing:
            warnings.append(
                f"기존 기상 파일(provider={existing.get('provider')}, station={existing.get('station_id')})과 출처가 달라 교체했습니다"
            )
        for row in rows:
            merged[row["date"]] = row
        all_rows = [merged[k] for k in sorted(merged)]
        have = {r["date"] for r in all_rows}
        missing = [d.isoformat() for d in (first + dt.timedelta(days=i) for i in range((last - first).days + 1)) if d.isoformat() not in have]
        period = [all_rows[0]["date"], all_rows[-1]["date"]]
        document = {
            "schema": script.SCHEMA,
            "region_id": region_id,
            "provider": provider_name,
            "source": source,
            "station_id": str(station.id),
            "station_name": station_name,
            "fetched_range": period,
            "collected_range": [first.isoformat(), last.isoformat()],
            "fetched_at": dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "note_ko": (
                "기상청 ASOS 일자료(AsosDalyInfoService). 강수 빈칸은 0으로 둠. 예측기의 외생 입력(원점 이전 강수·기온)으로 쓴다."
                if source == "asos" else
                "합성 샘플 기상(weather_service.SampleWeatherProvider) — 실측이 아니다. 기능 확인용이며 성능 근거로 쓰지 않는다."
            ),
            "missing_dates": missing,
            "rows": all_rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        rel = (registry.get(region_id).manifest.get("weather") or {}).get("daily_file") or "weather/asos_daily.json"
        registry.update_manifest_section(region_id, "weather", {
            "provider": provider_name, "station_id": str(station.id), "station_name": station_name,
            "source": source, "daily_file": rel, "period": period,
            "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        })
        return {
            "years": years, "n_rows": len(all_rows), "n_rows_new": len(rows), "period": period, "missing_dates_n": len(missing),
            "source": source, "provider": provider_name, "station_id": str(station.id), "station_name": station_name,
            "file": str(path), "warnings": warnings,
        }

    return run


@router.post(
    "/regions/{region_id}/weather/collect",
    status_code=202,
    responses={
        200: {"description": "async_run=false — 완료 상태(잡 본문)"},
        202: {"description": "async_run=true — 잡 접수 {job_id, status_url}"},
        404: {"description": "모르는 지역"},
        409: {"description": "같은 지역에 진행 중인 기상 수집 잡이 있다"},
        422: {"description": "지점·기간 검증 실패, 또는 source=asos 인데 KMA 키가 없음"},
        503: {"description": "weather_service 패키지가 없다"},
    },
    summary="Collect KMA ASOS daily weather year by year into the region (background job) and update the manifest",
    description=(
        "모델 평가 화면 ③ 용. `POST /weather/observations` 와 같은 내부 서비스를 연 단위로 불러 "
        "`regions/<id>/weather/asos_daily.json`(사용자 사본, 예측기가 읽는 스키마)에 저장하고 지역 정의의 `weather` "
        "{provider, station_id, station_name} 를 갱신한다. 서버에 KMA_API_KEY 가 없으면(`source=auto`) 합성 샘플로 떨어지며 "
        "잡의 `source=sample`·`warnings` 와 파일·매니페스트의 `provider=SAMPLE_SYNTHETIC` 으로 그 사실을 남긴다. "
        "진행은 `GET …/weather/collect/jobs/{job_id}` — `years[]` {year, n_rows, source, status}, `n_rows`."
    ),
)
async def weather_collect(region_id: str, payload: WeatherCollectRequest, request: Request):
    _region_or_404(request, region_id)
    service = _weather_service()
    if service is None:
        raise HTTPException(status_code=503, detail="weather_service package is not available")
    try:
        source, provider = choose_provider(service, payload.source)
        service.resolve_station(payload.station_id)
    except Exception as exc:
        configuration_error = getattr(service, "WeatherConfigurationError", ())
        if isinstance(exc, configuration_error):
            raise HTTPException(
                status_code=422,
                detail="기상청 ASOS 실관측 수집에는 서버의 KMA_API_KEY 가 필요합니다 — 키를 설정하거나 source=sample(합성, 기능 확인용)로 요청하세요",
            ) from exc
        raise HTTPException(status_code=422, detail=f"기상 지점 확인 실패: {exc}") from exc
    # 오늘 이후 종료일은 제공자가 거부한다(ASOS 는 D-1 까지) — 잡으로 넘기기 전에 422 로 알린다
    latest = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date() - dt.timedelta(days=1)
    if payload.end > latest:
        raise HTTPException(status_code=422, detail=f"end 는 D-1({latest.isoformat()} KST) 이전이어야 합니다 — ASOS 는 과거 관측이지 예보가 아닙니다")
    registry = _regions(request)
    jobs = _jobs(request)
    run = build_weather_run(registry, region_id, payload, service, source, provider)
    data = {
        "station_id": payload.station_id, "start": payload.start.isoformat(), "end": payload.end.isoformat(),
        "source": source, "years": [], "n_rows": 0, "warnings": [SAMPLE_WARNING] if source == "sample" else [],
    }
    prefix = str(request.app.state.settings.api_prefix)
    try:
        if not payload.async_run:
            return JSONResponse(status_code=200, content=await run_in_threadpool(jobs.run_sync, region_id, run, data=data))
        snapshot = jobs.submit(region_id, run, data=data)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(
        status_code=202,
        content={
            "job_id": snapshot["job_id"], "status": snapshot["status"], "region_id": region_id, "source": source,
            "status_url": f"{prefix}/regions/{region_id}/weather/collect/jobs/{snapshot['job_id']}",
        },
    )


@router.get("/regions/{region_id}/weather/collect/jobs", summary="List weather collection jobs of a region (newest first)")
async def weather_collect_jobs(region_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    items = _jobs(request).list(region_id)
    return {"region_id": region_id, "items": items, "total": len(items)}


@router.get(
    "/regions/{region_id}/weather/collect/jobs/{job_id}",
    summary="Poll one weather collection job — status, progress, years[], n_rows, warnings",
)
async def weather_collect_job(region_id: str, job_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    item = _jobs(request).get(job_id)
    if item is None or item.get("region_id") != region_id:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r} for region {region_id!r}")
    return item


__all__ = [
    "PROVIDER_ASOS",
    "PROVIDER_SAMPLE",
    "SAMPLE_WARNING",
    "WeatherCollectRequest",
    "build_weather_run",
    "choose_provider",
    "load_weather_script",
    "router",
]
