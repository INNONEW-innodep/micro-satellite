#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""수체 면적 시계열을 '비교 가능한' 공통 격자 기준으로 다시 세어 지역 자산으로 굳힌다.

왜 필요한가
-----------
`regions/<id>/satellite/scenes.json` 의 `water_area_km2` 는 **씬마다 자기 footprint 안에서**
센 값이다. 촬영 범위가 날짜마다 다르면(부산 광학 2020-03-25 는 03-12 보다 남쪽으로 8.5 km
치우쳐 하구/바다를 더 담는다) 면적이 커져도 물이 는 것이 아니다. 즉 **원계열은 날짜 간
비교가 불가능**하다.

반면 `regions/<id>/satellite/masks/<센서>/<날짜>.png` 는 이미 센서별 **공통 격자**(같은
EPSG:32652 범위·같은 화소 크기)로 잘라 맞춰 둔 이진 마스크다. 이 PNG 의 물 화소를 세고
`frame.pixel_area_m2` 를 곱하면 **같은 창으로 본 면적**, 곧 시계열로 비교 가능한 계열이 된다.

이 스크립트가 만드는 것
-----------------------
1. `regions/<id>/satellite/area_series.json` (schema `watercast.satellite.area-series/1`)
   — 센서별 frame 정의 + 날짜별 공통격자/원footprint 면적·GT 면적·같은 날 수위계 수위.
2. `scenes.json` 각 씬에 `area_common_frame_km2` 추가(기존 `water_area_km2` 는 그대로 둔다).

원본 GeoTIFF(`source_mask`, GT 라벨)가 있으면 프레임 안 유효자료 비율·GT 면적·모든 씬의
유효영역 교집합 면적까지 함께 기록한다. 없으면 그 항목만 null 로 두고 PNG 기반 면적은 낸다.

멱등(idempotent): 같은 입력이면 `generated_at` 외 내용이 같다. 여러 번 돌려도 안전하다.

사용:
    .venv/bin/python scripts/build_common_frame_area.py                # 모든 지역
    .venv/bin/python scripts/build_common_frame_area.py nakdong_busan  # 일부 지역
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = os.path.join(REPO, "regions")
SCHEMA = "watercast.satellite.area-series/1"
WATER_THRESHOLD = 127  # 마스크 PNG 는 0/255 이진. 리샘플링 여지를 두고 중간값으로 자른다.

# GT 라벨 원본(씬 footprint 격자) — 있으면 공통 격자로 옮겨 GT 면적도 같은 창으로 센다.
GT_DIRS: dict[tuple[str, str], str] = {
    ("nakdong_busan", "ICEYE"): "data/incoming/handover/06_aux/Labels_GT",
    ("nakdong_busan", "PlanetScope"): "data/incoming/handover/06_aux/Labels_GT_planet",
}

NOTES_KO = [
    "area_km2_common_frame 이 시계열로 비교 가능한 면적이다. 센서별 공통 격자(frame.bounds_epsg32652)"
    " 라는 '같은 창'으로 모든 날짜를 세기 때문이다.",
    "area_km2_scene_footprint 은 scenes.json 의 기존 water_area_km2 — 씬마다 자기 촬영범위 안에서 센 값이라"
    " 날짜 간 비교에 쓰면 안 된다. 촬영범위가 넓어지거나 옮겨가면 물이 늘지 않아도 면적이 커진다.",
    "scene_footprint_km2 는 그 씬 원본의 외곽 상자 넓이(km²), valid_area_km2 는 그 안에서 실제 자료가 있는"
    " 넓이(km²)다. 두 값이 날짜마다 다르다는 것이 원계열을 못 믿는 이유 그 자체다.",
    "frame_valid_fraction 은 공통 격자 안에서 그 씬에 실제 자료가 있는 화소 비율이다. 1 보다 작으면 그 날짜의"
    " 공통격자 면적도 '덜 본 만큼' 과소평가된다 — 계열 해석 전에 반드시 확인할 것.",
    "area_km2_common_valid_footprint 는 그 센서 모든 씬의 유효영역 교집합(진짜로 고정된 창)에서 센 면적이다."
    " frame_valid_fraction 이 날짜마다 다를 때는 이 값이 가장 공정한 비교 대상이다.",
    "gt_area_km2 는 사람이 그린 GT 라벨을 같은 공통 격자로 옮겨(탐지 유효영역으로 한정해) 센 면적이다."
    " 탐지 면적과 GT 면적을 같은 창에서 비교해야 '탐지가 틀렸나, 창이 달랐나'를 가릴 수 있다.",
    "gauge_levels 는 같은 날짜 지상 수위계(regions/<id>/gauge/series.json) 실측 수위(m)다. 면적과 수위를"
    " 같은 행에 두면 면적 변화가 수문학적으로 말이 되는지 바로 확인할 수 있다.",
]


# ── 작은 유틸 ────────────────────────────────────────────────────────────────
def _round(value: Any, digits: int = 3) -> Any:
    return None if value is None else round(float(value), digits)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """표본 상관계수. 표본이 3개 미만이거나 한쪽이 상수면 None(계산이 의미 없다)."""

    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return round(sxy / math.sqrt(sxx * syy), 4)


def _rasterio():
    try:
        import rasterio  # noqa: PLC0415 - 선택 의존(원본 GeoTIFF 가 있을 때만 쓴다)

        return rasterio
    except Exception:  # noqa: BLE001
        return None


def read_on_frame(tif: str, frame: dict) -> np.ndarray | None:
    """씬 격자 래스터를 공통 격자로 최근접 재투영한다. 값 1=물, 255=자료없음."""

    rasterio = _rasterio()
    if rasterio is None or not os.path.isfile(tif):
        return None
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    bounds = frame["bounds_epsg32652"]
    width, height = int(frame["width"]), int(frame["height"])
    dst = np.full((height, width), 255, np.uint8)
    with rasterio.open(tif) as ds:
        nodata = ds.nodata if ds.nodata is not None else 255
        reproject(
            ds.read(1), dst,
            src_transform=ds.transform, src_crs=ds.crs,
            dst_transform=from_bounds(*bounds, width, height), dst_crs="EPSG:32652",
            src_nodata=nodata, dst_nodata=255, resampling=Resampling.nearest,
        )
    return dst


def gt_raster_for(region_id: str, sensor: str, date_iso: str) -> str | None:
    base = GT_DIRS.get((region_id, sensor))
    if not base:
        return None
    hits = sorted(glob.glob(os.path.join(REPO, base, f"*{date_iso.replace('-', '')}*.tif")))
    return hits[0] if hits else None


def gauge_levels_by_date(region_dir: str) -> dict[str, dict[str, float]]:
    """date → {수위계 지점: 수위 m}. gauge/series.json 이 없으면 빈 dict."""

    path = os.path.join(region_dir, "gauge", "series.json")
    if not os.path.isfile(path):
        return {}
    try:
        payload = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, dict[str, float]] = {}
    for station, rows in (payload.get("series") or {}).items():
        for row in rows or []:
            level = row.get("water_level_m")
            if level is None:
                continue
            out.setdefault(str(row["date"]), {})[str(station)] = float(level)
    return out


# ── 지역 하나 ────────────────────────────────────────────────────────────────
def build_region(region_id: str, *, verbose: bool = True, region_dir: str | None = None) -> dict[str, Any] | None:
    """지역 하나의 area_series.json 을 (다시) 만든다.

    ``region_dir`` 를 주면 그 디렉터리(예: 운영 중 장면을 올려 생긴 사용자 사본
    ``backend/data/regions/<id>``)의 ``satellite/scenes.json`` 을 읽고 같은 곳에 쓴다. 생략하면
    저장소 내장 ``regions/<id>`` 다. 장면 파이프라인(backend/app/pipeline)이 새 장면을 등록한 뒤
    이 함수를 사용자 사본 경로로 호출해 공통 격자 면적 계열을 갱신한다.
    """

    region_dir = region_dir or os.path.join(REGIONS, region_id)
    sat_dir = os.path.join(region_dir, "satellite")
    scenes_path = os.path.join(sat_dir, "scenes.json")
    if not os.path.isfile(scenes_path):
        return None
    payload = json.load(open(scenes_path, encoding="utf-8"))
    scenes = [s for s in payload.get("scenes") or [] if s.get("frame")]
    if not scenes:
        return None

    levels = gauge_levels_by_date(region_dir)
    by_sensor: dict[str, list[dict]] = {}
    for scene in scenes:
        by_sensor.setdefault(scene["sensor"], []).append(scene)

    sensors_out: dict[str, Any] = {}
    common_frame_area: dict[str, float] = {}   # scene_id → 공통격자 면적
    for sensor, group in sorted(by_sensor.items()):
        group.sort(key=lambda s: s["date"])
        frame = group[0]["frame"]
        pixel_area = float(frame["pixel_area_m2"])
        width, height = int(frame["width"]), int(frame["height"])

        # 1) PNG(이미 공통 격자로 맞춰 둔 마스크) — 이것이 비교 가능한 면적의 근거다.
        water: dict[str, np.ndarray] = {}
        for scene in group:
            png = os.path.join(sat_dir, scene["frame"]["path"])
            water[scene["scene_id"]] = np.asarray(Image.open(png).convert("L")) > WATER_THRESHOLD

        # 2) 원본 마스크가 있으면 프레임 안 유효자료와 그 교집합(진짜 고정 창)도 구한다.
        valid: dict[str, np.ndarray] = {}
        for scene in group:
            arr = read_on_frame(os.path.join(REPO, scene["source_mask"]), scene["frame"]) if scene.get("source_mask") else None
            if arr is not None:
                valid[scene["scene_id"]] = arr != 255
        intersection = (np.logical_and.reduce(list(valid.values()))
                        if len(valid) == len(group) and valid else None)

        rows = []
        for scene in group:
            sid, date_iso = scene["scene_id"], scene["date"]
            mask = water[sid]
            area_cf = float(mask.sum()) * pixel_area / 1e6
            common_frame_area[sid] = area_cf

            quicklook = scene.get("quicklook") or {}
            fp_bounds = quicklook.get("bounds_epsg32652")
            fp_km2 = ((fp_bounds[2] - fp_bounds[0]) * (fp_bounds[3] - fp_bounds[1]) / 1e6) if fp_bounds else None

            valid_mask = valid.get(sid)
            gt_km2 = None
            gt_tif = gt_raster_for(region_id, sensor, date_iso)
            if gt_tif and scene.get("gt_available"):
                gt_arr = read_on_frame(gt_tif, scene["frame"])
                if gt_arr is not None:
                    gt_water = gt_arr == 1
                    if valid_mask is not None:   # GT 는 탐지 유효영역 밖까지 그려져 있어 그대로 세면 과대평가
                        gt_water = gt_water & valid_mask
                    gt_km2 = float(gt_water.sum()) * pixel_area / 1e6

            rows.append({
                "date": date_iso,
                "scene_id": sid,
                "area_km2_common_frame": _round(area_cf),
                "area_km2_scene_footprint": _round(scene.get("water_area_km2"), 4),
                "scene_footprint_km2": _round(fp_km2, 1),
                "scene_footprint_bounds_epsg32652": fp_bounds,
                "valid_area_km2": _round(scene.get("valid_area_km2"), 2),
                "frame_valid_fraction": (_round(float(valid_mask.mean()), 4) if valid_mask is not None else None),
                "area_km2_common_valid_footprint": (
                    _round(float((mask & intersection).sum()) * pixel_area / 1e6) if intersection is not None else None),
                "gt_area_km2": _round(gt_km2),
                "gauge_levels": levels.get(date_iso, {}),
            })

        cf = [r["area_km2_common_frame"] for r in rows]
        raw = [r["area_km2_scene_footprint"] for r in rows]
        fps = [r["scene_footprint_km2"] for r in rows]
        vfr = [r["frame_valid_fraction"] for r in rows]
        civ = [r["area_km2_common_valid_footprint"] for r in rows]
        sensors_out[sensor] = {
            "frame": {
                "bounds_epsg32652": frame["bounds_epsg32652"],
                "width": width, "height": height,
                "pixel_area_m2": pixel_area,
                "coverage_km2": _round(width * height * pixel_area / 1e6, 2),
            },
            "common_valid_footprint_km2": (_round(float(intersection.sum()) * pixel_area / 1e6, 2)
                                           if intersection is not None else None),
            "n_scenes": len(rows),
            "correlations": {
                "raw_area_vs_footprint_km2": pearson(raw, fps),
                "common_frame_area_vs_footprint_km2": pearson(cf, fps),
                "common_valid_area_vs_footprint_km2": pearson(civ, fps) if intersection is not None else None,
                "common_frame_area_vs_frame_valid_fraction": pearson(cf, vfr) if valid else None,
                "note_ko": "원면적이 촬영범위와 함께 움직이면(상관 높음) 그 계열은 수문 신호가 아니라 촬영 기하를 보고 있는 것이다.",
            },
            "rows": rows,
        }
        if verbose:
            print(f"  [{region_id}/{sensor}] {len(rows)}씬 · 공통격자 {width}×{height} "
                  f"({sensors_out[sensor]['frame']['coverage_km2']} km²) · "
                  f"면적 {min(cf):.3f}~{max(cf):.3f} km² (원계열 {min(raw):.3f}~{max(raw):.3f})", flush=True)

    out = {
        "schema": SCHEMA,
        "region_id": region_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/build_common_frame_area.py",
        "source_scenes": "satellite/scenes.json",
        "water_threshold": WATER_THRESHOLD,
        "sensors": sensors_out,
        "notes_ko": NOTES_KO,
    }
    with open(os.path.join(sat_dir, "area_series.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # scenes.json 갱신 — 기존 water_area_km2 는 손대지 않고 비교 가능한 값을 옆에 붙인다.
    changed = False
    for scene in payload.get("scenes") or []:
        area = common_frame_area.get(scene.get("scene_id"))
        if area is None:
            continue
        value = _round(area)
        if scene.get("area_common_frame_km2") != value:
            scene["area_common_frame_km2"] = value
            changed = True
    if changed:
        with open(scenes_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return out


def main(argv: list[str]) -> int:
    wanted = argv[1:] or sorted(
        d for d in os.listdir(REGIONS)
        if os.path.isfile(os.path.join(REGIONS, d, "satellite", "scenes.json"))
    )
    built = 0
    for region_id in wanted:
        print(f"[{region_id}]", flush=True)
        if build_region(region_id) is None:
            print("  위성 장면 자산이 없어 건너뜀", flush=True)
            continue
        built += 1
    print(f"완료 — 지역 {built}곳에 area_series.json 기록", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
