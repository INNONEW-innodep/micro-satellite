"""WAMIS 유역 관측소 목록(``GET /wamis/stations``)과 일 수위 수집 잡(``POST /regions/{id}/gauge/collect``).

수집 자체는 ``scripts/fetch_wamis_levels.py`` 를 모듈로 올려 그대로 쓴다(연 단위 재시도·잘린 JSON 회수 포함).
받은 행은 ``POST /regions/{id}/gauge/{station}/series`` 와 같은 경로(``RegionRegistry.upsert_station_series``,
replace=false)로 지역 시계열에 병합한다 — 사용자 사본 생성·QC·지점 메타(source WAMIS·source_code·terrain)까지 같다.
WAMIS 는 인증키가 없고 인터넷이 필요하다. 테스트는 ``wamis_list_basin_stations``·``wamis_collect`` 를 바꿔 끼운다.
"""

from __future__ import annotations

import importlib.util
import json
import re
import threading
from datetime import date as Date
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from ..gauge.regions import RegionError, RegionNotFoundError, RegionRegistry
from .jobs import JobConflictError, ProgressUpdate, ScreenJobs

router = APIRouter(tags=["regions"])

REPO_ROOT = Path(__file__).resolve().parents[3]
WAMIS_SCRIPT = REPO_ROOT / "scripts" / "fetch_wamis_levels.py"
BASINS: dict[int, str] = {1: "한강", 2: "낙동강", 3: "금강", 4: "섬진강", 5: "영산강"}
LIST_ATTEMPTS = 4  # 스크립트 기본(12회, 실패 시 2·4·…초 대기)은 API 응답으로는 너무 길다
# WAMIS 목록 응답은 한강·낙동강처럼 큰 유역에서 자주 잘린다(2026-10-16 실측: 6회 중 0건). 저장소에 둔 사본으로 받친다.
CACHE_DIR = Path(__file__).resolve().parents[1] / "wamis_cache"


def cached_basin_rows(basin: int) -> list[dict[str, Any]]:
    """`backend/app/wamis_cache/stations_basin_<b>.json` 의 관측소 행(스크립트 원형 키). 없으면 []."""

    path = CACHE_DIR / f"stations_basin_{int(basin)}.json"
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in (doc.get("rows") or []) if isinstance(r, dict) and r.get("obscd")]

_MODULE: Any = None
_MODULE_LOCK = threading.Lock()


def load_wamis_module() -> Any:
    """``scripts/fetch_wamis_levels.py`` 를 한 번만 모듈로 올린다(scripts/ 는 패키지가 아니다)."""

    global _MODULE
    with _MODULE_LOCK:
        if _MODULE is None:
            spec = importlib.util.spec_from_file_location("watercast_fetch_wamis_levels", WAMIS_SCRIPT)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {WAMIS_SCRIPT}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULE = module
    return _MODULE


def wamis_list_basin_stations(basin: int) -> list[dict[str, str]]:
    """유역 관측소 목록(인터넷 필요). 테스트가 바꿔 끼우는 지점."""

    return load_wamis_module().list_basin_stations(int(basin), attempts=LIST_ATTEMPTS)


def wamis_collect(stations: dict[str, str], start_digits: str, end_digits: str) -> tuple[dict[str, list], list[str]]:
    """일 수위 수집(인터넷 필요) → ({지점 id: [{date(YYYYMMDD), water_level_m, obscd}]}, 수집 실패 목록). 테스트가 바꿔 끼우는 지점."""

    module = load_wamis_module()
    before = len(module.FETCH_FAILURES)
    result = module.collect(dict(stations), str(start_digits), str(end_digits))
    return result, [str(f) for f in module.FETCH_FAILURES[before:]]


class WamisStationCache:
    """유역별 관측소 목록 — 프로세스 수명 동안 메모리에 둔다(WAMIS 가 자주 끊기므로 한 번 받으면 다시 묻지 않는다)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[int, list[dict[str, Any]]] = {}

    def get(self, basin: int) -> list[dict[str, Any]] | None:
        with self._lock:
            items = self._items.get(int(basin))
            return [dict(i) for i in items] if items is not None else None

    def put(self, basin: int, items: list[dict[str, Any]]) -> None:
        with self._lock:
            self._items[int(basin)] = [dict(i) for i in items]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def _cache(request: Request) -> WamisStationCache:
    cache = getattr(request.app.state, "wamis_station_cache", None)
    if cache is None:
        cache = WamisStationCache()
        request.app.state.wamis_station_cache = cache
    return cache


def _jobs(request: Request) -> ScreenJobs:
    jobs = getattr(request.app.state, "gauge_collect_jobs", None)
    if jobs is None:
        jobs = ScreenJobs("gauge_collect")
        request.app.state.gauge_collect_jobs = jobs
    return jobs


def _regions(request: Request) -> RegionRegistry:
    return request.app.state.regions


def _region_or_404(request: Request, region_id: str):
    try:
        return _regions(request).get(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from exc


def _station_item(row: dict[str, Any]) -> dict[str, Any]:
    """WAMIS wl_dubwlobs 행 → 화면용 항목. 목록 API 에는 주소가 없어 address 는 null(제원은 wl_obsinfo 가 따로 준다)."""

    return {
        "code": str(row.get("obscd") or ""),
        "name": str(row.get("obsnm") or ""),
        "river": str(row.get("bbsnnm") or "") or None,  # 목록에는 대권역명만 있다
        "address": row.get("addr") or row.get("address") or None,
        "status": str(row.get("clsyn") or "") or None,  # 운영 | 폐쇄 …
        "kind": str(row.get("obsknd") or "") or None,
        "org": str(row.get("mngorg") or "") or None,
        "sub_basin": str(row.get("sbsncd") or "") or None,
    }


@router.get(
    "/wamis/stations",
    summary="WAMIS gauge stations of a basin (cached per process), optional name regex filter",
    description=(
        "`basin` 1=한강 2=낙동강 3=금강 4=섬진강 5=영산강. `match` 는 관측소 이름 정규식(선택), `operating_only=true` 면 "
        "운영 중(clsyn=운영)만. 항목 {code, name, river(대권역명), address(null — 목록 API 에 없음), status, kind, org}. "
        "유역별 결과는 프로세스가 살아 있는 동안 메모리에 캐시한다. WAMIS 는 인터넷이 필요하며 못 받으면 502."
    ),
)
async def wamis_stations(
    request: Request,
    basin: Annotated[int, Query(ge=1, le=5, description="유역 코드 1=한강 2=낙동강 3=금강 4=섬진강 5=영산강")],
    match: Annotated[str | None, Query(description="관측소 이름 정규식(선택). 예: 나주|담양")] = None,
    operating_only: Annotated[bool, Query(description="운영 중인 관측소만")] = False,
) -> dict[str, Any]:
    pattern = None
    if match and match.strip():
        try:
            pattern = re.compile(match.strip())
        except re.error as exc:
            raise HTTPException(status_code=422, detail=f"match 정규식이 올바르지 않습니다: {exc}") from exc
    cache = _cache(request)
    items = cache.get(basin)
    cached = items is not None
    source = "memory" if cached else "live"
    if items is None:
        live_error: str | None = None
        try:
            rows = await run_in_threadpool(wamis_list_basin_stations, basin)
        except Exception as exc:  # 잘린 응답·연결 실패 → 저장소 사본으로
            rows, live_error = [], f"{type(exc).__name__}: {exc}"
        items = [_station_item(r) for r in rows if isinstance(r, dict) and r.get("obscd")]
        if not items:
            fallback = [_station_item(r) for r in cached_basin_rows(basin)]
            if fallback:
                items, source = fallback, "cache"
        if not items:
            raise HTTPException(
                status_code=502,
                detail=f"WAMIS 유역 {basin}({BASINS.get(basin, '')}) 관측소 목록을 받지 못했고(비어 있음) 저장된 사본도 없습니다 — 인터넷 연결과 WAMIS 서버 상태를 확인하세요"
                + (f": {live_error}" if live_error else ""),
            )
        cache.put(basin, items)
    picked = [
        i for i in items
        if (pattern is None or pattern.search(i["name"])) and (not operating_only or (i.get("status") or "운영") == "운영")
    ]
    return {
        "basin": basin, "basin_name": BASINS.get(basin), "match": match, "operating_only": operating_only,
        "items": picked, "n": len(picked), "n_total": len(items), "cached": cached, "source": source,
        "note_ko": "WAMIS 개방 API(wl_dubwlobs, 무인증) 관측소 목록. 좌표·주소는 지점 제원(wl_obsinfo)에 있어 목록에는 없다."
                   + (" 이번 응답은 WAMIS 가 답하지 않아 저장소 사본(backend/app/wamis_cache)으로 채웠다." if source == "cache" else ""),
    }


# ── 수집 잡 ─────────────────────────────────────────────────────────────────────


class CollectStation(BaseModel):
    id: str | None = Field(default=None, max_length=80, description="지역 시계열의 지점 id. 비우면 name")
    name: str | None = Field(default=None, max_length=80, description="표시 이름. 비우면 id")
    code: str = Field(pattern=r"^\d{7}$", description="WAMIS 관측소 코드(7자리)")
    terrain: str = Field(default="", max_length=80)
    terrain_note: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def fill_names(self) -> CollectStation:
        sid = (self.id or "").strip() or (self.name or "").strip()
        if not sid:
            raise ValueError("stations[]: id 또는 name 이 필요합니다")
        if "/" in sid or "\\" in sid:
            raise ValueError(f"지점 id 에 경로 구분자를 쓸 수 없습니다: {sid!r}")
        self.id = sid
        self.name = (self.name or "").strip() or sid
        return self


class GaugeCollectRequest(BaseModel):
    stations: list[CollectStation] = Field(min_length=1, max_length=50)
    start: Date = Field(description="수집 시작일(포함) YYYY-MM-DD")
    end: Date = Field(description="수집 종료일(포함) YYYY-MM-DD")
    min_coverage: float = Field(default=0.85, ge=0.0, le=1.0, description="기간 대비 유효 일수 비율 하한 — 미달 지점도 병합하되 note 로 표시")
    async_run: bool = Field(default=True, description="true: 202 + 잡 접수(배경), false: 끝날 때까지 기다려 완료 상태를 돌려준다")

    @model_validator(mode="after")
    def check_period(self) -> GaugeCollectRequest:
        if self.end < self.start:
            raise ValueError(f"start({self.start}) 가 end({self.end}) 보다 늦습니다")
        ids = [s.id for s in self.stations]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"지점 id 가 중복됩니다: {dupes}")
        return self


def _digits(day: Date) -> str:
    return day.strftime("%Y%m%d")


def _iso(digits: str) -> str:
    text = str(digits)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) >= 8 and text[:8].isdigit() else text[:10]


def build_collect_run(registry: RegionRegistry, region_id: str, payload: GaugeCollectRequest):
    """잡 본문 — 지점마다 WAMIS 수집 → 커버리지 → 지역 시계열 병합(replace=false, 메타 source=WAMIS)."""

    start8, end8 = _digits(payload.start), _digits(payload.end)
    expected = (payload.end - payload.start).days + 1
    specs = list(payload.stations)

    def pending(spec: CollectStation, status: str) -> dict[str, Any]:
        return {
            "id": spec.id, "name": spec.name, "code": spec.code, "terrain": spec.terrain,
            "n_days": 0, "expected_days": expected, "coverage": None, "period": None, "status": status, "note": None,
        }

    def run(update: ProgressUpdate) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for index, spec in enumerate(specs):
            entry = pending(spec, "running")
            # 진행 중 조회에도 전체 지점이 보이게 — 끝난 것 + 지금 것 + 아직 안 한 것(queued)
            update(progress=index / len(specs), stations=results + [entry] + [pending(s, "queued") for s in specs[index + 1:]], current=spec.id)
            notes: list[str] = []
            try:
                fetched, failures = wamis_collect({spec.id: spec.code}, start8, end8)
            except Exception as exc:  # noqa: BLE001 - 지점 하나의 실패가 나머지를 막지 않는다
                entry.update(status="failed", note=f"수집 실패: {type(exc).__name__}: {exc}")
                results.append(entry)
                continue
            rows = [r for r in (fetched.get(spec.id) or []) if r.get("water_level_m") is not None]
            if failures:
                notes.append(f"연도별 수집 실패 {len(failures)}건(결측과 구분): " + "; ".join(failures[:5]))
                entry["fetch_failures"] = failures
            coverage = round(len(rows) / expected, 4) if expected else 0.0
            entry.update(
                n_days=len(rows), coverage=coverage,
                period=[_iso(rows[0]["date"]), _iso(rows[-1]["date"])] if rows else None,
            )
            if not rows:
                notes.insert(0, "수집된 행이 없습니다 — 관측소 코드·기간(WAMIS 는 일 수위가 있는 관측소만)을 확인하세요")
                entry.update(status="empty", note=" / ".join(notes))
                results.append(entry)
                continue
            try:
                station = registry.upsert_station_series(
                    region_id, spec.id, rows, replace=False,
                    station_meta={"name": spec.name, "source": "WAMIS", "source_code": spec.code, "terrain": spec.terrain, "terrain_note": spec.terrain_note},
                )
            except RegionError as exc:
                entry.update(status="failed", note=f"병합 실패: {exc}")
                results.append(entry)
                continue
            entry["n_days_total"] = station.n_days  # 기존 자료와 합친 뒤 지역에 있는 유효 일수
            entry["period_total"] = list(station.period) if station.period else None
            if coverage < payload.min_coverage:
                notes.insert(0, f"커버리지 {coverage:.1%} 가 기준 {payload.min_coverage:.0%} 미만 — 병합은 했으나 학습 전 자료를 확인하세요")
                entry["status"] = "low_coverage"
            else:
                entry["status"] = "done"
            entry["note"] = " / ".join(notes) if notes else None
            results.append(entry)
        n_merged = sum(1 for r in results if r["status"] in ("done", "low_coverage"))
        return {
            "stations": results, "current": None, "n_stations": len(results), "n_merged": n_merged,
            "n_low_coverage": sum(1 for r in results if r["status"] == "low_coverage"),
            "n_failed": sum(1 for r in results if r["status"] in ("failed", "empty")),
        }

    return run


@router.post(
    "/regions/{region_id}/gauge/collect",
    status_code=202,
    responses={
        200: {"description": "async_run=false — 수집이 끝난 뒤 완료 상태(잡 본문)"},
        202: {"description": "async_run=true — 잡 접수 {job_id, status_url}"},
        404: {"description": "모르는 지역"},
        409: {"description": "같은 지역에 진행 중인 수집 잡이 있다"},
        422: {"description": "지점·기간 검증 실패(한국어 사유)"},
    },
    summary="Collect WAMIS daily levels for stations and merge them into the region series (background job)",
    description=(
        "모델 평가 화면 ② 용. `stations[]` {id, name, code(7자리), terrain, terrain_note} 마다 WAMIS 일 수위를 받아(연 단위 재시도) "
        "`POST /regions/{id}/gauge/{station}/series` 와 같은 경로로 병합한다(replace=false, 사용자 사본·QC·지점 메타 source=WAMIS·"
        "source_code 동일). `min_coverage` 미달 지점도 병합하되 `status=low_coverage` 와 note 로 표시한다. 진행은 "
        "`GET …/gauge/collect/jobs/{job_id}`. 지역당 동시에 하나(409)."
    ),
)
async def gauge_collect(region_id: str, payload: GaugeCollectRequest, request: Request):
    _region_or_404(request, region_id)
    registry = _regions(request)
    jobs = _jobs(request)
    run = build_collect_run(registry, region_id, payload)
    data = {
        "start": payload.start.isoformat(), "end": payload.end.isoformat(), "min_coverage": payload.min_coverage,
        "stations": [
            {"id": s.id, "name": s.name, "code": s.code, "terrain": s.terrain, "n_days": 0, "coverage": None, "period": None, "status": "queued", "note": None}
            for s in payload.stations
        ],
        "n_stations": len(payload.stations), "current": None,
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
            "job_id": snapshot["job_id"], "status": snapshot["status"], "region_id": region_id,
            "status_url": f"{prefix}/regions/{region_id}/gauge/collect/jobs/{snapshot['job_id']}",
        },
    )


@router.get(
    "/regions/{region_id}/gauge/collect/jobs",
    summary="List WAMIS collection jobs of a region (newest first)",
)
async def gauge_collect_jobs(region_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    items = _jobs(request).list(region_id)
    return {"region_id": region_id, "items": items, "total": len(items)}


@router.get(
    "/regions/{region_id}/gauge/collect/jobs/{job_id}",
    summary="Poll one WAMIS collection job — status, progress 0..1, per-station coverage/status/note",
    description=(
        "`status` queued|running|done|failed, `progress` 0..1, `stations[]` {id, name, code, n_days, coverage, period, status"
        "(queued|running|done|low_coverage|empty|failed), note}, `started_at`, `finished_at`, `error`."
    ),
)
async def gauge_collect_job(region_id: str, job_id: str, request: Request) -> dict[str, Any]:
    _region_or_404(request, region_id)
    item = _jobs(request).get(job_id)
    if item is None or item.get("region_id") != region_id:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r} for region {region_id!r}")
    return item


__all__ = [
    "BASINS",
    "CollectStation",
    "GaugeCollectRequest",
    "WamisStationCache",
    "build_collect_run",
    "load_wamis_module",
    "router",
    "wamis_collect",
    "wamis_list_basin_stations",
]
