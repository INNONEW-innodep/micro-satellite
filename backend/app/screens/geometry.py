"""지역 지도 기하(``GET /regions/{id}/geometry``)와 준비 상태(``GET /regions/{id}/readiness``).

화면은 외부 타일 없이 인라인 SVG(등장방형 투영)로 지도를 그리므로, 경계 상자·수위계 지점·위성 수위
산출 위치(AOI)·장면 촬영범위를 전부 **경위도(lon, lat)** 로 맞춰 한 번에 준다. 투영좌표(EPSG:32652 등)만
있는 상자는 rasterio.warp.transform 으로 바꾼다 — 새 좌표를 지어내지 않고 지역 정의·장면 기록에 있는 값만 쓴다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..gauge.regions import RegionNotFoundError, RegionRecord, RegionRegistry
from ..gauge.service import GaugeForecastService

router = APIRouter(tags=["regions"])

_BBOX_KEY_RE = re.compile(r"^bbox_epsg(\d+)$", re.IGNORECASE)
_KOREA_LON = (124.0, 132.0)
_KOREA_LAT = (33.0, 39.0)


def _service(request: Request) -> GaugeForecastService:
    return request.app.state.gauge_service


def _regions(request: Request) -> RegionRegistry:
    return request.app.state.regions


def _region_or_404(request: Request, region_id: str) -> RegionRecord:
    try:
        return _regions(request).get(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from exc


# ── 좌표 변환 ──────────────────────────────────────────────────────────────────


def to_lonlat(crs: str, xs: Sequence[float], ys: Sequence[float]) -> list[list[float]] | None:
    """투영좌표 → WGS84 경위도. rasterio 가 없거나 CRS 가 틀리면 None(지어내지 않는다)."""

    try:
        from rasterio.warp import (
            transform,  # 지연 import — 모듈 로드 시점에는 필요 없다
        )
    except ImportError:
        return None
    try:
        lons, lats = transform(str(crs), "EPSG:4326", [float(x) for x in xs], [float(y) for y in ys])
    except Exception:  # noqa: BLE001 - 잘못된 CRS·좌표는 기하 없음으로
        return None
    return [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in zip(lons, lats)]


def bbox_polygon_lonlat(crs: str, bbox: Sequence[float]) -> list[list[float]] | None:
    """[minx, miny, maxx, maxy](투영) → 경위도 모서리 4개(좌상→우상→우하→좌하, 지역 정의의 bbox_lonlat 과 같은 순서)."""

    if len(bbox) != 4:
        return None
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    return to_lonlat(crs, [minx, maxx, maxx, minx], [maxy, maxy, miny, miny])


def _lonlat_pairs(value: Any) -> list[list[float]] | None:
    """[[lon, lat], ...] 꼴만 통과(숫자 2개짜리 점 목록). 아니면 None."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    out: list[list[float]] = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            return None
        try:
            out.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return out or None


def _lonlat_point(value: Any) -> list[float] | None:
    pairs = _lonlat_pairs([value]) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else None
    return pairs[0] if pairs else None


def aoi_polygon_lonlat(location: Mapping[str, Any]) -> list[list[float]] | None:
    """AOI(satellite.locations 의 한 항목)의 경위도 다각형 — ``bbox_lonlat`` 이 있으면 그것, 없으면 투영 상자를 변환."""

    polygon = _lonlat_pairs(location.get("bbox_lonlat")) or _lonlat_pairs(location.get("polygon_lonlat"))
    if polygon:
        return polygon
    crs = str(location.get("crs") or "")
    for key, value in location.items():
        found = _BBOX_KEY_RE.match(str(key))
        if found and isinstance(value, Sequence) and len(value) == 4:
            return bbox_polygon_lonlat(crs or f"EPSG:{found.group(1)}", value)
    return None


def scene_footprint_lonlat(scene: Mapping[str, Any]) -> list[list[float]] | None:
    """장면 촬영범위 — ``quicklook.footprint_lonlat`` 이 있으면 그것, 없으면 ``quicklook.bounds_epsg32652`` 를 변환."""

    quicklook = scene.get("quicklook") or {}
    if not isinstance(quicklook, Mapping):
        return None
    footprint = _lonlat_pairs(quicklook.get("footprint_lonlat"))
    if footprint:
        return footprint
    crs = str(quicklook.get("crs") or quicklook.get("source_crs") or "")
    for key, value in quicklook.items():
        found = _BBOX_KEY_RE.match(str(key))
        if found and isinstance(value, Sequence) and len(value) == 4:
            return bbox_polygon_lonlat(crs or f"EPSG:{found.group(1)}", value)
    return None


def union_bbox(points: Sequence[Sequence[float]], *, margin_ratio: float = 0.05, min_margin_deg: float = 0.005) -> list[float] | None:
    """점들의 경계 상자 + 여백(변의 5%, 최소 0.005°). 점이 없으면 None."""

    lons = [float(p[0]) for p in points]
    lats = [float(p[1]) for p in points]
    if not lons:
        return None
    min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
    pad_lon = max(min_margin_deg, (max_lon - min_lon) * margin_ratio)
    pad_lat = max(min_margin_deg, (max_lat - min_lat) * margin_ratio)
    return [round(min_lon - pad_lon, 6), round(min_lat - pad_lat, 6), round(max_lon + pad_lon, 6), round(max_lat + pad_lat, 6)]


# ── 기하 ─────────────────────────────────────────────────────────────────────


def region_geometry(service: GaugeForecastService, record: RegionRecord, api_prefix: str) -> dict[str, Any]:
    summary = service.region_summary(record)
    stations = [
        {
            "id": s["id"], "name": s.get("name") or s["id"], "lat": s.get("lat"), "lon": s.get("lon"),
            "terrain": s.get("terrain") or "", "satellite_location": s.get("satellite_location"),
            "n_days": s.get("n_days"), "period": s.get("period"), "model_trained": bool(s.get("model_trained")),
        }
        for s in summary.get("stations") or []
    ]
    links = record.satellite_links()  # station → loc
    station_by_loc = {str(loc): str(station) for station, loc in links.items()}
    aois = []
    for loc_id, location in record.satellite_locations().items():
        station_id = station_by_loc.get(loc_id) or (str(location["gauge_station"]) if location.get("gauge_station") else None)
        aois.append({
            "loc_id": loc_id,
            "station_id": station_id,
            "name": location.get("name_ko") or loc_id,
            "polygon_lonlat": aoi_polygon_lonlat(location),
            "anchor_lonlat": _lonlat_point(location.get("anchor_lonlat")),
            "center_lonlat": _lonlat_point(location.get("center_lonlat")),
            "river": location.get("river"),
        })
    scenes = []
    for scene in service.scenes(record.id, api_prefix).get("scenes") or []:
        urls = scene.get("image_urls") or {}
        scenes.append({
            "scene_id": scene.get("scene_id"),
            "sensor": scene.get("sensor"),
            "sensor_type": scene.get("sensor_type"),
            "date": str(scene.get("date") or "")[:10] or None,
            "footprint_lonlat": scene_footprint_lonlat(scene),
            "quicklook_url": urls.get("input"),
            "overlay_url": urls.get("overlay"),
            "detection_available": bool(scene.get("detection_available")),
            "water_area_km2": scene.get("water_area_km2"),
            "area_common_frame_km2": scene.get("area_common_frame_km2"),
        })
    scenes.sort(key=lambda s: (str(s.get("date") or ""), str(s.get("sensor") or "")))

    manifest_bbox = list(record.manifest.get("bbox_wgs84") or [])
    bbox: list[float] | None
    bbox_source: str
    if len(manifest_bbox) == 4:
        bbox = [float(v) for v in manifest_bbox]
        bbox_source = "manifest"
    else:
        points: list[Sequence[float]] = [(s["lon"], s["lat"]) for s in stations if s.get("lon") is not None and s.get("lat") is not None]
        for aoi in aois:
            points.extend(aoi["polygon_lonlat"] or [])
            if aoi["anchor_lonlat"]:
                points.append(aoi["anchor_lonlat"])
        for scene in scenes:
            points.extend(scene["footprint_lonlat"] or [])
        bbox = union_bbox(points)
        bbox_source = "union" if bbox else "none"
    return {
        "region_id": record.id,
        "name": record.name,
        "crs": str(record.manifest.get("crs") or ""),
        "bbox_wgs84": bbox,
        "bbox_source": bbox_source,
        "stations": stations,
        "aois": aois,
        "scenes": scenes,
        "n_stations": len(stations),
        "n_aois": len(aois),
        "n_scenes": len(scenes),
        "note_ko": (
            "좌표는 모두 WGS84 경위도(lon, lat). bbox_wgs84 는 지역 정의값(manifest)이거나, 없으면 지점·AOI·장면 촬영범위의 "
            "합집합에 여백을 둔 것(union). AOI 다각형은 지역 정의의 bbox_lonlat 또는 투영 상자(EPSG:32652)를 변환한 것이며, "
            "장면 촬영범위는 퀵룩 지리참조(footprint_lonlat)다. 좌표가 없는 항목은 null 로 둔다(지어내지 않음)."
        ),
    }


@router.get(
    "/regions/{region_id}/geometry",
    summary="Map geometry of a region — bbox, gauge stations, satellite AOIs, scene footprints (all WGS84 lon/lat)",
    description=(
        "서비스 화면 ① 의 위치 지도용. `bbox_wgs84` [minlon, minlat, maxlon, maxlat], `stations[]` {id, name, lat, lon, terrain, "
        "satellite_location}, `aois[]` {loc_id, station_id(연결 수위계 또는 null), polygon_lonlat[[lon,lat]…], anchor_lonlat}, "
        "`scenes[]` {scene_id, sensor, date, footprint_lonlat(4점 또는 null), quicklook_url, overlay_url, detection_available, "
        "water_area_km2}. 404: 모르는 지역."
    ),
)
async def region_geometry_endpoint(region_id: str, request: Request) -> dict[str, Any]:
    record = _region_or_404(request, region_id)
    prefix = str(request.app.state.settings.api_prefix)
    return await run_in_threadpool(region_geometry, _service(request), record, prefix)


# ── 준비 상태 ─────────────────────────────────────────────────────────────────


def region_readiness(service: GaugeForecastService, regions: RegionRegistry, record: RegionRecord) -> dict[str, Any]:
    # 수위 자료
    stations: list[dict[str, Any]] = []
    if record.gauge_available:
        series_set = regions.load_series(record.id)
        stations = [s.to_dict() for s in series_set.stations]
    periods = [s["period"] for s in stations if s.get("period")]
    gauge_period = [min(p[0] for p in periods), max(p[1] for p in periods)] if periods else None
    gauge = {
        "available": bool(stations),
        "n_stations": len(stations),
        "period": gauge_period,
        "n_days": max((int(s.get("n_days") or 0) for s in stations), default=0),
        "stations": [{"id": s["id"], "name": s.get("name"), "n_days": s.get("n_days"), "period": s.get("period"), "source": s.get("source")} for s in stations],
    }
    # 기상 자료
    weather_rows = regions.load_weather_rows(record.id)
    weather_meta = dict(record.manifest.get("weather") or {})
    weather = {
        "available": bool(weather_rows),
        "station": str(weather_meta.get("station_id")) if weather_meta.get("station_id") not in (None, "") else None,
        "station_name": weather_meta.get("station_name"),
        "provider": weather_meta.get("provider"),
        "period": [str(weather_rows[0].get("date")), str(weather_rows[-1].get("date"))] if weather_rows else None,
        "n_days": len(weather_rows),
    }
    # 학습 산출물
    evaluation = service.evaluation(record.id)
    artifacts_dir, origin = service.active_artifacts(record.id)
    n_trained = len([p for p in artifacts_dir.glob("*.json") if p.name != "evaluation.json"]) if artifacts_dir.is_dir() else 0
    trained = evaluation is not None
    models = {
        "trained": trained,
        "trained_at": (evaluation or {}).get("trained_at"),
        "split": dict((evaluation or {}).get("split") or {}) or None,
        "custom_split": bool((evaluation or {}).get("custom_split")),
        "artifacts_origin": origin if trained else None,
        "n_stations_trained": n_trained if trained else 0,
    }
    # 위성 장면
    scenes = (record.scenes() or {}).get("scenes") or []
    satellite = {
        "available": bool(scenes),
        "n_scenes": len(scenes),
        "n_with_masks": sum(1 for s in scenes if s.get("detection_available")),
        "sensors": sorted({str(s.get("sensor")) for s in scenes if s.get("sensor")}),
        "levels_available": bool(record.satellite_levels()),
    }
    # 다음 할 일 — 화면의 안내문(한국어). 막힌 단계 순서대로.
    next_steps: list[str] = []
    if not stations:
        next_steps.append("수위 자료가 없습니다 — ② 수위 자료 수집(WAMIS 지점 검색·수집 또는 CSV 업로드)")
    if not weather_rows:
        next_steps.append("기상 자료가 없습니다 — ③ 기상 자료 수집(기상청 ASOS 지점·기간)")
    if not trained:
        next_steps.append("학습되지 않았습니다 — ④ 학습(구간 선택 후 실행)" if stations else "학습되지 않았습니다 — ④ 학습은 수위 자료가 있어야 합니다")
    elif stations and n_trained < len(stations):
        missing = len(stations) - n_trained
        next_steps.append(f"학습 산출물이 없는 지점이 {missing}개 있습니다 — ④ 학습을 다시 실행하면 전 지점이 갱신됩니다")
    if not scenes:
        next_steps.append("위성 장면이 없습니다 — 서비스 화면 ① 에서 영상을 올리면 탐지·면적 계열이 생깁니다(평가에는 필수가 아님)")
    if stations and weather_rows and trained:
        next_steps.append("준비 완료 — ⑤ 성능 평가·⑥ 시험을 볼 수 있습니다")
    ready = bool(stations) and trained
    return {
        "region_id": record.id,
        "name": record.name,
        "origin": record.origin,
        "ready": ready,
        "gauge": gauge,
        "weather": weather,
        "models": models,
        "satellite": satellite,
        "next_steps": next_steps,
    }


@router.get(
    "/regions/{region_id}/readiness",
    summary="Readiness panel of the evaluation screen — gauge/weather/models/satellite status + next steps (Korean)",
    description=(
        "모델 평가 화면 상단 '준비 상태' 패널용. `gauge` {n_stations, period, n_days}, `weather` {available, station, period}, "
        "`models` {trained, trained_at, split}, `satellite` {n_scenes, n_with_masks}, `next_steps[]`(한국어 안내, 막힌 단계 순서). "
        "`gauge.n_days` 는 가장 긴 지점의 유효 관측일수. 404: 모르는 지역."
    ),
)
async def region_readiness_endpoint(region_id: str, request: Request) -> dict[str, Any]:
    record = _region_or_404(request, region_id)
    return await run_in_threadpool(region_readiness, _service(request), _regions(request), record)


__all__ = [
    "aoi_polygon_lonlat",
    "bbox_polygon_lonlat",
    "region_geometry",
    "region_readiness",
    "router",
    "scene_footprint_lonlat",
    "to_lonlat",
    "union_bbox",
]
