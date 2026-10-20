#!/usr/bin/env python3
"""기존 `regions/*/satellite/scenes.json` 장면에 퀵룩 지리참조(`quicklook`)를 채운다.

`POST /regions/{id}/satellite/scenes` 로 새로 등록되는 장면은 `build_scene_record` 가 바로 넣지만, 그 전에
`scripts/make_region_scene_assets.py` 로 만든 18개 장면(부산 8·대청 10)에는 없다. 이 스크립트가 각 항목의
`source_pre`(원본 전처리 GeoTIFF) 를 열어 범위(EPSG:32652)를 읽고, 저장된 퀵룩 PNG 크기와 함께 적는다.

    quicklook = {"crs": "EPSG:32652", "source_crs", "bounds_epsg32652": [minx,miny,maxx,maxy],
                 "width", "height" (퀵룩 PNG 화소), "footprint_lonlat": [[lon,lat] x4 시계 방향]}

- `source_pre` 가 파일명만이면 `data/incoming/handover/05_l1_pre/{iceye_pre,planet_pre,dcd_pre}` 아래에서 찾는다.
- 원본이 없으면 그 장면은 건너뛰고 보고한다(값을 지어내지 않는다). 멱등: 이미 같은 값이면 파일을 다시 쓰지 않는다.

    .venv/bin/python scripts/backfill_scene_georef.py             # 전 지역
    .venv/bin/python scripts/backfill_scene_georef.py --region nakdong_busan --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.gauge.scene_assets import quicklook_georef  # noqa: E402

REGIONS_DIR = ROOT / "regions"
PRE_ROOT = ROOT / "data" / "incoming" / "handover" / "05_l1_pre"
PRE_DIRS = ("iceye_pre", "planet_pre", "dcd_pre")


def resolve_source_pre(value: str) -> Path | None:
    """scenes.json 의 source_pre → 실제 파일. 절대/상대 경로 → 그대로, 파일명만 → 인수 전처리 디렉터리 검색."""

    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute() and (ROOT / candidate).is_file():
        return ROOT / candidate
    name = candidate.name
    for sub in PRE_DIRS:
        base = PRE_ROOT / sub
        if not base.is_dir():
            continue
        hits = sorted(base.rglob(name))
        if hits:
            return hits[0]
    return None


def png_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.width, image.height


def backfill_region(region_dir: Path, *, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    region_id = region_dir.name
    scenes_path = region_dir / "satellite" / "scenes.json"
    report: dict[str, Any] = {"region_id": region_id, "filled": [], "kept": [], "missing": [], "changed": False}
    if not scenes_path.is_file():
        return report
    payload = json.loads(scenes_path.read_text(encoding="utf-8"))
    changed = False
    for scene in payload.get("scenes") or []:
        scene_id = str(scene.get("scene_id"))
        if scene.get("quicklook") and not force:
            report["kept"].append(scene_id)
            continue
        pre = resolve_source_pre(str(scene.get("source_pre") or ""))
        input_rel = (scene.get("images") or {}).get("input")
        input_png = region_dir / "satellite" / input_rel if input_rel else None
        if pre is None or input_png is None or not input_png.is_file():
            report["missing"].append(f"{scene_id}: source_pre={scene.get('source_pre')!r} 원본={'있음' if pre else '없음'} 퀵룩={'있음' if input_png and input_png.is_file() else '없음'}")
            continue
        width, height = png_size(input_png)
        georef = quicklook_georef(pre, width, height)
        if georef is None:
            report["missing"].append(f"{scene_id}: 원본에 좌표계가 없음 ({pre})")
            continue
        if scene.get("quicklook") != georef:
            scene["quicklook"] = georef
            changed = True
        report["filled"].append(f"{scene_id} {width}x{height} bounds={georef['bounds_epsg32652']} lonlat[0]={georef['footprint_lonlat'][0]}")
    report["changed"] = changed
    if changed and not dry_run:
        scenes_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", action="append", help="특정 지역 id 만(기본: regions/ 전부)")
    parser.add_argument("--regions-dir", type=Path, default=REGIONS_DIR)
    parser.add_argument("--force", action="store_true", help="이미 quicklook 이 있어도 다시 계산")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    region_dirs = [args.regions_dir / r for r in args.region] if args.region else sorted(p for p in args.regions_dir.iterdir() if (p / "region.json").is_file())
    exit_code = 0
    for region_dir in region_dirs:
        report = backfill_region(region_dir, force=args.force, dry_run=args.dry_run)
        if not (report["filled"] or report["kept"] or report["missing"]):
            continue
        state = "변경 없음" if not report["changed"] else ("(dry-run) 변경 예정" if args.dry_run else "기록")
        print(f"[{report['region_id']}] 채움 {len(report['filled'])} / 유지 {len(report['kept'])} / 원본 없음 {len(report['missing'])} — {state}")
        for line in report["filled"]:
            print(f"  [채움] {line}")
        for line in report["missing"]:
            print(f"  [건너뜀] {line}")
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
