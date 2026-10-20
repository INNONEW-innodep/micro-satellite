"""``GET /regions/{region_id}/scenarios`` — 자료가 **언제** 있는지와, 그 안에서 고를 수 있는 전망 **시나리오**.

서비스 화면 ④ 에서 쓴다. 영상 날짜(예: 2026-09-20)에 지상 수위계 자료가 없으면 지금까지는 전망 원점이 조용히
자료 마지막 날로 옮겨졌다. 이 계약은 그 사실을 화면에 먼저 알려 주고, 자료가 있는 기간 안에서 출발일(원점)과
어디까지 내다볼지(예측일수·끝 날짜)를 고르게 한다.

새 알고리즘은 없다. 지역 레지스트리(수위 시계열·기상·위성 장면)와 학습 산출물(분할·최대 예측일수)을 읽어
**기간·창·추천 원점**으로 엮기만 한다.

응답
    gauge      {available, period, n_days, stations[{id,name,period,n_days,model_trained}]}
    weather    {available, station, station_name, period, n_days}
    satellite  {available, n_scenes, sensors[], scenes[{scene_id,sensor,date,has_frame,water_area_km2,in_gauge_period}]}
    models     {trained, split{train_end,val_start,val_end,test_start,test_end}, horizon_max}
    forecast_window {origin_min, origin_max, horizon_max, verifiable_until, min_history_days}
    scenarios  [{id, kind, as_of, label_ko, note_ko, available, observed_days_after, in_sample, scene_id, sensor}]
    requested  (as_of 를 줬을 때) {as_of, available, resolved_as_of, clamped, observed_days_after, in_sample, note_ko,
                horizon, target_date}
    expansion  다지역 확장 경로 판정 {mode: satellite_series(경로 A) | gauge_first(경로 B), label_ko, rule_ko, reasons_ko[],
                checks{scenes_per_sensor, years, flood_season, labels_or_pairs, gauge, dem}} — 요청서(2026-10-16) 초안 규칙
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date as Date
from datetime import timedelta
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from ..gauge.regions import RegionNotFoundError, RegionRecord, RegionRegistry
from ..gauge.service import GaugeForecastService

router = APIRouter(tags=["regions"])

# 원점 이전에 있어야 하는 관측 일수(안내용). 예측기 입력 창(최근 수십 일) + 계절 특징을 만들 수 있는 최소치.
MIN_HISTORY_DAYS = 60

# 다지역 확장 경로 판정 규칙(요청서 2026-10-16 초안, 조정 가능): 센서당 장면 ≥ 36 · 기간 ≥ 2년 · 홍수기(6~9월) 포함 ·
# (정답 라벨 또는 위성-수위계 같은 날 대조 ≥ 30건) 이면 경로 A(satellite_series: 위성 시계열 직접 학습), 아니면
# 경로 B(gauge_first: 지상 수위계로 먼저 예측, 위성은 검증). 판정은 자료 보유량만 보며 모델을 바꾸지 않는다.
EXPANSION_RULE: dict[str, Any] = {"min_scenes_per_sensor": 36, "min_years": 2.0, "flood_months": [6, 7, 8, 9], "min_pairs": 30}


def _service(request: Request) -> GaugeForecastService:
    return request.app.state.gauge_service


def _regions(request: Request) -> RegionRegistry:
    return request.app.state.regions


def _region_or_404(request: Request, region_id: str) -> RegionRecord:
    try:
        return _regions(request).get(region_id)
    except RegionNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown region {region_id!r}") from None


def _iso(value: Any) -> Date | None:
    if value in (None, ""):
        return None
    try:
        return Date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _satellite_pairs(record: RegionRecord) -> int:
    """위성 수위 ↔ 지상 수위계 같은 날 대조 건수 — levels.json 기록 중 보정 수위가 있고 수위계에 연결된 위치(loc)의 것."""

    path = record.satellite_levels_path
    if path is None or not path.is_file():
        return 0
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    records = doc.get("records") if isinstance(doc, dict) else doc
    if not isinstance(records, list):
        return 0
    linked = {str(v) for v in record.satellite_links().values()}
    n = 0
    for row in records:
        if not isinstance(row, dict) or row.get("corrected_water_level_m") is None:
            continue
        if linked and str(row.get("loc_id")) not in linked:
            continue
        n += 1
    return n


def expansion_mode(record: RegionRecord, scenes: list[dict[str, Any]], *, gauge_available: bool) -> dict[str, Any]:
    """EXPANSION_RULE 로 경로 A/B 를 판정하고 사유를 한국어로 돌려준다(장면 목록은 region_scenarios 가 만든 것)."""

    rule = EXPANSION_RULE
    per_sensor = Counter(str(s.get("sensor")) for s in scenes if s.get("sensor"))
    dates = sorted(d for d in (_iso(s.get("date")) for s in scenes) if d)
    years = round((dates[-1] - dates[0]).days / 365.25, 2) if len(dates) >= 2 else 0.0
    flood = any(d.month in rule["flood_months"] for d in dates)
    manifest_sat = record.manifest.get("satellite") or {}
    labels = bool(manifest_sat.get("gt_labels_available"))
    pairs = _satellite_pairs(record)
    dem = bool(manifest_sat.get("wbms_testbed"))
    max_scenes = max(per_sensor.values()) if per_sensor else 0
    checks = {
        "scenes_per_sensor": {"value": dict(per_sensor), "min": rule["min_scenes_per_sensor"], "ok": max_scenes >= rule["min_scenes_per_sensor"]},
        "years": {"value": years, "min": rule["min_years"], "ok": years >= rule["min_years"]},
        "flood_season": {"value": flood, "months": list(rule["flood_months"]), "ok": flood},
        "labels_or_pairs": {"labels": labels, "pairs": pairs, "min_pairs": rule["min_pairs"], "ok": labels or pairs >= rule["min_pairs"]},
        "gauge": {"value": gauge_available, "ok": gauge_available},
        "dem": {"value": dem, "ok": dem},
    }
    path_a = all(checks[k]["ok"] for k in ("scenes_per_sensor", "years", "flood_season", "labels_or_pairs"))
    mode = "satellite_series" if path_a else "gauge_first"
    reasons: list[str] = []
    if not scenes:
        reasons.append("위성 장면 없음")
    else:
        reasons.append(f"센서당 장면 최대 {max_scenes}장 ({'≥' if checks['scenes_per_sensor']['ok'] else '<'} {rule['min_scenes_per_sensor']}장)")
        reasons.append(f"관측 기간 {years}년 ({'≥' if checks['years']['ok'] else '<'} {rule['min_years']:g}년)")
        reasons.append("홍수기(6~9월) 장면 " + ("포함" if flood else "없음"))
        reasons.append(f"정답 라벨 {'있음' if labels else '없음'} · 위성-수위계 대조 {pairs}건")
    reasons.append("지상 수위계 있음 → 경로 B 가동" if gauge_available else "지상 수위계 없음 — 경로 B 도 수집 뒤에 가능")
    return {
        "mode": mode,
        "label_ko": "경로 A — 위성 시계열 직접 학습(경로 B 병기)" if path_a else "경로 B — 지상 수위계 우선, 위성은 검증",
        "rule_ko": f"경로 A 조건: 센서당 장면 ≥ {rule['min_scenes_per_sensor']} · 기간 ≥ {rule['min_years']:g}년 · 홍수기 포함 · (정답 라벨 또는 대조 ≥ {rule['min_pairs']}건). 넘으면 경로 A 를 추가로 돌려 둘을 병기(승격)",
        "reasons_ko": reasons,
        "checks": checks,
    }


def _days_between(a: Date, b: Date) -> int:
    return int((b - a).days)


def region_scenarios(
    service: GaugeForecastService,
    regions: RegionRegistry,
    record: RegionRecord,
    *,
    as_of: Date | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    # ── 수위 자료 ─────────────────────────────────────────────────────────────
    stations: list[dict[str, Any]] = []
    gauge_first: Date | None = None
    gauge_last: Date | None = None
    if record.gauge_available:
        series_set = regions.load_series(record.id)
        artifacts_dir, _origin = service.active_artifacts(record.id)
        for station in series_set.stations:
            series = series_set.series.get(station.id)
            if series is None:
                continue
            valid = np.flatnonzero(series.valid)
            if valid.size == 0:
                continue
            first, last = _iso(str(series.dates[valid[0]])), _iso(str(series.dates[valid[-1]]))
            trained = regions.station_artifact_path(artifacts_dir, station.id).is_file()
            stations.append({
                "id": station.id, "name": station.name, "terrain": station.terrain,
                "period": [first.isoformat(), last.isoformat()] if first and last else None,
                "n_days": int(valid.size), "model_trained": bool(trained),
            })
            if first and (gauge_first is None or first < gauge_first):
                gauge_first = first
            if last and (gauge_last is None or last > gauge_last):
                gauge_last = last
    gauge = {
        "available": bool(stations),
        "period": [gauge_first.isoformat(), gauge_last.isoformat()] if gauge_first and gauge_last else None,
        "n_days": max((int(s["n_days"]) for s in stations), default=0),
        "n_stations": len(stations),
        "stations": stations,
    }

    # ── 기상 자료 ─────────────────────────────────────────────────────────────
    weather_rows = regions.load_weather_rows(record.id)
    weather_meta = dict(record.manifest.get("weather") or {})
    weather = {
        "available": bool(weather_rows),
        "station": str(weather_meta.get("station_id")) if weather_meta.get("station_id") not in (None, "") else None,
        "station_name": weather_meta.get("station_name"),
        "period": [str(weather_rows[0].get("date"))[:10], str(weather_rows[-1].get("date"))[:10]] if weather_rows else None,
        "n_days": len(weather_rows),
    }

    # ── 학습 산출물(분할·최대 예측일수) ───────────────────────────────────────
    evaluation = service.evaluation(record.id)
    split = dict((evaluation or {}).get("split") or record.split or {})
    horizon_max = int(record.horizon_max or 30)
    if stations:
        try:
            artifact = service.load_station_artifact(record.id, stations[0]["id"])
            horizon_max = int(artifact.get("horizon_max") or horizon_max)
            split = dict(artifact.get("split") or split)
        except Exception:  # noqa: BLE001 - 산출물이 없으면 지역 기본값
            pass
    models = {
        "trained": evaluation is not None,
        "trained_at": (evaluation or {}).get("trained_at"),
        "split": split or None,
        "horizon_max": horizon_max,
        "n_stations_trained": sum(1 for s in stations if s["model_trained"]),
    }
    val_end = _iso(split.get("val_end"))
    test_start = _iso(split.get("test_start"))

    # ── 위성 장면 ─────────────────────────────────────────────────────────────
    raw_scenes = (record.scenes() or {}).get("scenes") or []
    area_rows: dict[str, float] = {}
    for sensor_block in ((record.area_series() or {}).get("sensors") or {}).values():
        for row in (sensor_block or {}).get("rows") or []:
            if row.get("scene_id") and row.get("area_km2_common_frame") is not None:
                area_rows[str(row["scene_id"])] = float(row["area_km2_common_frame"])
    scenes: list[dict[str, Any]] = []
    for scene in raw_scenes:
        day = _iso(scene.get("date"))
        if day is None:
            continue
        scene_id = str(scene.get("scene_id"))
        scenes.append({
            "scene_id": scene_id, "sensor": str(scene.get("sensor") or ""), "date": day.isoformat(),
            "has_frame": bool((scene.get("frame") or {}).get("path")),
            "water_area_km2": scene.get("water_area_km2"),
            "area_km2_common_frame": area_rows.get(scene_id),
            "in_gauge_period": bool(gauge_first and gauge_last and gauge_first <= day <= gauge_last),
        })
    scenes.sort(key=lambda s: (s["date"], s["sensor"]))
    satellite = {
        "available": bool(scenes),
        "n_scenes": len(scenes),
        "sensors": sorted({s["sensor"] for s in scenes if s["sensor"]}),
        "period": [scenes[0]["date"], scenes[-1]["date"]] if scenes else None,
        "scenes": scenes,
    }
    expansion = expansion_mode(record, scenes, gauge_available=bool(gauge_first and gauge_last))

    # ── 전망 창 ───────────────────────────────────────────────────────────────
    window: dict[str, Any] | None = None
    if gauge_first and gauge_last:
        origin_min = min(gauge_first + timedelta(days=MIN_HISTORY_DAYS), gauge_last)
        window = {
            "origin_min": origin_min.isoformat(),
            "origin_max": gauge_last.isoformat(),
            "horizon_max": horizon_max,
            "verifiable_until": gauge_last.isoformat(),
            "min_history_days": MIN_HISTORY_DAYS,
        }

    def observed_after(origin: Date, h: int) -> int:
        if gauge_last is None:
            return 0
        return max(0, min(int(h), _days_between(origin, gauge_last)))

    def in_sample(origin: Date) -> bool | None:
        return (origin <= val_end) if val_end else None

    # ── 추천 시나리오 ─────────────────────────────────────────────────────────
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, origin: Date | None, label: str, note: str, **extra: Any) -> None:
        available = bool(origin and gauge_first and gauge_last and gauge_first <= origin <= gauge_last)
        key = f"{kind}:{origin.isoformat() if origin else ''}:{extra.get('scene_id') or ''}"
        if key in seen:
            return
        seen.add(key)
        scenarios.append({
            "id": key, "kind": kind, "as_of": origin.isoformat() if origin else None,
            "label_ko": label, "note_ko": note, "available": available,
            "observed_days_after": observed_after(origin, horizon_max) if available and origin else 0,
            "in_sample": in_sample(origin) if available and origin else None,
            "scene_id": extra.get("scene_id"), "sensor": extra.get("sensor"),
        })

    if gauge_first and gauge_last:
        for scene in scenes:
            day = _iso(scene["date"])
            assert day is not None
            if gauge_first <= day <= gauge_last:
                n_after = observed_after(day, horizon_max)
                add(
                    "scene", day,
                    f"영상 날짜 {day.isoformat()} ({scene['sensor']})",
                    (f"이 영상이 찍힌 날을 출발일로 — 이후 실제 수위 {n_after}일과 비교할 수 있어요"
                     if n_after else "이 영상이 찍힌 날을 출발일로 — 이후 실측이 없어 비교는 못 해요"),
                    scene_id=scene["scene_id"], sensor=scene["sensor"],
                )
            else:
                add(
                    "scene_no_gauge", None,
                    f"영상 날짜 {day.isoformat()} ({scene['sensor']}) — 수위 자료 없음",
                    f"이 날짜에는 지상 수위계 자료가 없어요 (자료 {gauge_first.isoformat()} ~ {gauge_last.isoformat()})",
                    scene_id=scene["scene_id"], sensor=scene["sensor"],
                )
        if test_start and gauge_first <= test_start <= gauge_last:
            add(
                "test_start", test_start,
                f"시험 연도 첫날 {test_start.isoformat()}",
                "모델이 학습 때 보지 않은 구간이라 공정한 비교예요",
            )
        verify_origin = gauge_last - timedelta(days=horizon_max)
        if verify_origin >= gauge_first:
            add(
                "verify", verify_origin,
                f"자료 끝 {horizon_max}일 전 {verify_origin.isoformat()}",
                f"{horizon_max}일 전망 전부를 실제 수위와 비교할 수 있는 마지막 출발일이에요",
            )
        add(
            "latest", gauge_last,
            f"자료 마지막 날 {gauge_last.isoformat()}",
            "그 뒤 실측이 없어 순수한 미래 전망이에요 (비교 불가)",
        )

    # ── 요청한 원점 풀이 ──────────────────────────────────────────────────────
    requested: dict[str, Any] | None = None
    if as_of is not None:
        h = int(horizon or 7)
        if h > horizon_max:
            h = horizon_max
        if gauge_first is None or gauge_last is None:
            requested = {
                "as_of": as_of.isoformat(), "available": False, "resolved_as_of": None, "clamped": False,
                "observed_days_after": 0, "in_sample": None, "horizon": h, "target_date": None,
                "note_ko": "이 지역에는 지상 수위계 자료가 없어 수위 전망을 만들 수 없어요",
            }
        else:
            resolved = as_of
            clamped = False
            note: str | None = None
            if as_of > gauge_last:
                # 자료 뒤의 날짜 → '자료 끝 H일 전'(전망 전부를 실측과 비교할 수 있는 마지막 출발일)로 맞춘다.
                # forecast API 자체는 마지막 날로 자르지만, 화면이 기본으로 고르는 출발일은 비교가 되는 이 날짜다.
                verify_origin = gauge_last - timedelta(days=horizon_max)
                resolved, clamped = (verify_origin if verify_origin >= gauge_first else gauge_last), True
                note = (f"{as_of.isoformat()} 에는 수위 자료가 없어요 (자료 {gauge_first.isoformat()} ~ {gauge_last.isoformat()}). "
                        f"출발일을 자료 끝 {horizon_max}일 전인 {resolved.isoformat()} 로 맞췄어요 — 전망 전부를 실제 수위와 비교할 수 있어요. "
                        "아래에서 바꿀 수 있어요.")
            elif as_of < gauge_first:
                resolved, clamped = min(gauge_first + timedelta(days=MIN_HISTORY_DAYS), gauge_last), True
                note = (f"{as_of.isoformat()} 은 수위 자료가 시작되기 전이에요 (자료 {gauge_first.isoformat()} ~ {gauge_last.isoformat()}). "
                        f"자료 시작 {MIN_HISTORY_DAYS}일 뒤인 {resolved.isoformat()} 을 출발일로 맞췄어요.")
            elif _days_between(gauge_first, as_of) < MIN_HISTORY_DAYS:
                note = (f"출발일 이전 관측이 {_days_between(gauge_first, as_of)}일뿐이라 전망이 비어 있을 수 있어요 — "
                        f"{(gauge_first + timedelta(days=MIN_HISTORY_DAYS)).isoformat()} 이후를 권해요.")
            n_after = observed_after(resolved, h)
            requested = {
                "as_of": as_of.isoformat(), "available": not clamped, "resolved_as_of": resolved.isoformat(),
                "clamped": clamped, "observed_days_after": n_after, "in_sample": in_sample(resolved),
                "horizon": h, "target_date": (resolved + timedelta(days=h)).isoformat(), "note_ko": note,
            }

    note_ko = (
        "지상 수위계 자료가 없어 수위 전망 시나리오를 만들 수 없어요 — 모델 평가 화면 ② 에서 수위 자료를 먼저 모으세요"
        if not stations
        else (
            "학습된 모델이 없어요 — 모델 평가 화면 ④ 에서 학습한 뒤 전망을 볼 수 있어요"
            if not models["trained"]
            else f"출발일은 {window['origin_min']} ~ {window['origin_max']} 사이에서 고르세요. "
                 f"{window['verifiable_until']} 까지는 실제 수위가 있어 전망과 비교할 수 있어요."
        )
    )
    return {
        "region_id": record.id,
        "name": record.name,
        "gauge": gauge,
        "weather": weather,
        "models": models,
        "satellite": satellite,
        "forecast_window": window,
        "scenarios": scenarios,
        "requested": requested,
        "note_ko": note_ko,
        "expansion": expansion,
    }


@router.get(
    "/regions/{region_id}/scenarios",
    summary="Data availability + forecast scenarios of a region (which dates have data, where an origin can be placed)",
    description=(
        "서비스 화면 ④ 의 '자료가 있는 기간·시나리오' 패널용. `gauge.period`(지상 수위계 자료 기간)·`weather.period`·"
        "`satellite.scenes[]`(영상 날짜, 수위 기간 안인지)·`models.split`(학습/검증/시험)·`forecast_window`"
        "(원점 최소/최대, 최대 예측일수, 실측 비교 가능 마지막 날)·`scenarios[]`(추천 출발일: 영상 날짜·시험 연도 첫날·"
        "자료 끝 H일 전·자료 마지막 날; `available=false` 는 수위 자료 밖의 영상 날짜). `as_of` 를 주면 `requested` 에 "
        "그 날짜를 어디로 맞췄는지(`clamped`, `resolved_as_of`)와 이후 실측 일수·목표일을 돌려준다. `expansion` 은 다지역 확장 경로 판정(경로 A `satellite_series` / 경로 B `gauge_first`)과 사유·검사값. 404: 모르는 지역."
    ),
)
async def region_scenarios_endpoint(
    region_id: str,
    request: Request,
    as_of: Annotated[Date | None, Query(description="확인할 출발일(영상 날짜 등). 자료 밖이면 어디로 맞추는지 알려 준다")] = None,
    horizon: Annotated[int | None, Query(ge=1, le=90, description="예측일수(기본 7) — requested.target_date 계산용")] = None,
) -> dict[str, Any]:
    record = _region_or_404(request, region_id)
    return await run_in_threadpool(
        region_scenarios, _service(request), _regions(request), record, as_of=as_of, horizon=horizon,
    )


__all__ = ["EXPANSION_RULE", "MIN_HISTORY_DAYS", "expansion_mode", "region_scenarios", "router"]
