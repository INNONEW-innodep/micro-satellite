#!/usr/bin/env python3
"""WBMS AOI GeoJSON(EPSG:32652) → `regions/<id>/region.json` 의 `satellite.locations` 로 옮긴다.

인수 패키지의 `aoi_busan.geojson`·`aoi_dcd.geojson`(③ calc_wlwa 의 AOI 정본)에 있는 위성 수위 산출 위치를
지역 정의에 넣어, 화면·API가 "위성이 어디를 재는지"를 지도에 그릴 수 있게 한다.

- 좌표는 투영좌표(EPSG:32652, m) 그대로 두고(`bbox_epsg32652`) 경위도 모서리(`bbox_lonlat`)·중심(`center_lonlat`)을
  rasterio.warp.transform 으로 덧붙인다(pyproj 없음).
- `gauge_station` 은 `gauge.satellite_location_links`(지점 → loc_id)를 뒤집어 채운다.
- 인수 노트(`properties.note`)는 `note_ko` 에 원문 그대로 둔다.
- 멱등: 같은 입력이면 파일이 바뀌지 않는다(다른 키는 손대지 않는다).

    .venv/bin/python scripts/import_satellite_locations.py            # 두 지역 모두
    .venv/bin/python scripts/import_satellite_locations.py --dry-run  # 결과만 출력
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AOI_DIR = ROOT / "data" / "incoming" / "handover" / "02_package" / "deploy_package" / "config"
REGIONS_DIR = ROOT / "regions"

# 지역 ↔ AOI 파일 ↔ 화면 이름. 이름은 인수 노트(지점거리·재설정 사유)를 요약한 것이다.
SOURCES: dict[str, dict[str, Any]] = {
    "nakdong_busan": {
        "file": "aoi_busan.geojson",
        "names_ko": {
            "gupo": "구포대교 부근 본류 하도(64 m)",
            "gimhae": "김해교 하류 본류 하도(재설정, 930 m)",
            "jeongcheon": "정천교 부근 조만강",
            "hupo": "호포대교 부근 양산천",
        },
    },
    "geum_daecheong": {
        "file": "aoi_dcd.geojson",
        "names_ko": {
            "daecheong": "대청댐 본댐 저수위 타일(627 m)",
            "jojeongji": "대청조정지댐 타일(750 m)",
            "bangsuro": "대청댐방수로 타일(269 m)",
        },
    },
}


def to_lonlat(xs: list[float], ys: list[float], crs: str = "EPSG:32652") -> list[list[float]]:
    from rasterio.warp import transform

    lons, lats = transform(crs, "EPSG:4326", xs, ys)
    return [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in zip(lons, lats)]


def polygon_bbox(coords: list[list[float]]) -> list[float]:
    xs = [float(p[0]) for p in coords]
    ys = [float(p[1]) for p in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def build_location(feature: dict[str, Any], *, names_ko: dict[str, str], station_by_loc: dict[str, str], source: str) -> tuple[str, dict[str, Any]]:
    props = dict(feature.get("properties") or {})
    loc_id = str(props.get("loc_id") or "").strip()
    if not loc_id:
        raise ValueError("feature 에 loc_id 가 없습니다")
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Polygon":
        raise ValueError(f"{loc_id}: Polygon 만 지원합니다 ({geometry.get('type')})")
    ring = geometry["coordinates"][0]
    minx, miny, maxx, maxy = polygon_bbox(ring)
    # 시계 방향: 좌상 → 우상 → 우하 → 좌하
    corners_x = [minx, maxx, maxx, minx]
    corners_y = [maxy, maxy, miny, miny]
    bbox_lonlat = to_lonlat(corners_x, corners_y)
    center = to_lonlat([(minx + maxx) / 2], [(miny + maxy) / 2])[0]
    anchor = props.get("anchor_lonlat")
    anchor_lonlat = [round(float(anchor[0]), 6), round(float(anchor[1]), 6)] if isinstance(anchor, (list, tuple)) and len(anchor) == 2 else None
    snap = props.get("snap_distance_m")
    obscd = str(props.get("obscd")).strip() if props.get("obscd") not in (None, "") else None
    river = str(props.get("river")).strip() if props.get("river") not in (None, "") else None
    return loc_id, {
        "name_ko": names_ko.get(loc_id, loc_id),
        "gauge_station": station_by_loc.get(loc_id),
        "obscd": obscd,
        "river": river,
        "crs": "EPSG:32652",
        "bbox_epsg32652": [round(v, 3) for v in (minx, miny, maxx, maxy)],
        "bbox_lonlat": bbox_lonlat,
        "center_lonlat": center,
        "anchor_lonlat": anchor_lonlat,
        "snap_distance_m": float(snap) if isinstance(snap, (int, float)) else None,
        "note_ko": str(props.get("note") or ""),
        "source": source,
    }


def import_region(region_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    spec = SOURCES[region_id]
    aoi_path = AOI_DIR / spec["file"]
    manifest_path = REGIONS_DIR / region_id / "region.json"
    aoi = json.loads(aoi_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    links = dict((manifest.get("gauge") or {}).get("satellite_location_links") or {})
    station_by_loc = {str(loc): str(station) for station, loc in links.items()}
    locations: dict[str, dict[str, Any]] = {}
    for feature in aoi.get("features") or []:
        loc_id, item = build_location(feature, names_ko=spec["names_ko"], station_by_loc=station_by_loc, source=spec["file"])
        locations[loc_id] = item
    satellite = manifest.setdefault("satellite", {})
    changed = satellite.get("locations") != locations
    satellite["locations"] = locations
    if changed and not dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"region_id": region_id, "n_locations": len(locations), "changed": changed, "locations": locations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", action="append", choices=sorted(SOURCES), help="특정 지역만(기본: 전부)")
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 결과만 출력")
    args = parser.parse_args()
    for region_id in args.region or sorted(SOURCES):
        result = import_region(region_id, dry_run=args.dry_run)
        state = "변경 없음" if not result["changed"] else ("(dry-run) 변경 예정" if args.dry_run else "기록")
        print(f"[{region_id}] 위성 산출 위치 {result['n_locations']}곳 — {state}")
        for loc_id, item in result["locations"].items():
            print(f"  {loc_id:<11} {item['name_ko']:<28} 지점={item['gauge_station'] or '-':<8} 중심={item['center_lonlat']} 거리={item['snap_distance_m']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
