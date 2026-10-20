#!/usr/bin/env python3
"""지역별 위성 장면 자산(퀵룩·탐지 오버레이·공통 격자 마스크)을 만든다.

화면은 수 GB 원본 GeoTIFF를 읽을 수 없으니(서버 이미지에 없다), 지역 레지스트리
``regions/<id>/satellite/`` 아래에 작은 PNG와 목록(``scenes.json``)을 두고 API가
그것만 서빙한다. 산출:

- ``quicklooks/<scene_id>_input.png``   입력 영상 퀵룩 (SAR: dB 스트레치, 광학: RGB)
- ``quicklooks/<scene_id>_overlay.png`` 입력 + 탐지 마스크(청록) 오버레이
- ``quicklooks/<scene_id>_gt.png``      입력 + GT 라벨(노랑) 오버레이 — GT 있을 때만
- ``masks/<sensor>/<scene_id>.png``     센서별 공통 격자로 잘라 내린 이진 마스크(0/255)
  → 수체 면적 시계열 예측 샘플의 프레임이 된다 (같은 센서 프레임은 같은 H×W).
- ``scenes.json``                        장면 목록·면적·IoU·이미지 파일명

    python3 scripts/make_region_scene_assets.py --region nakdong_busan
    python3 scripts/make_region_scene_assets.py --region geum_daecheong
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
HANDOVER = ROOT / "data" / "incoming" / "handover"
WB_ROOT = ROOT / "data" / "wbms_runs" / "wb"

# 지역 → (센서 키, 센서 표시명, Pre 디렉터리, 마스크 base 접두, GT 디렉터리·패턴)
REGION_SOURCES = {
    "nakdong_busan": [
        {"sensor": "ICEYE", "type": "SAR", "pre_dir": "iceye_pre", "wb_prefix": "WB_Busan_ICEYE_",
         "gt_dir": HANDOVER / "06_aux" / "Labels_GT", "gt_pattern": "{date}_Label.tif", "gt_water_value": 1},
        {"sensor": "PlanetScope", "type": "OPTIC", "pre_dir": "planet_pre", "wb_prefix": "WB_Busan_PlanetScope_",
         "wb_root": ROOT / "data" / "wbms_runs" / "wb_planet",
         "gt_dir": HANDOVER / "06_aux" / "Labels_GT_planet", "gt_pattern": "{date}_Mosaic_TOA_label.tif", "gt_water_value": 1},
    ],
    "geum_daecheong": [
        {"sensor": "PlanetScope", "type": "OPTIC", "pre_dir": "dcd_pre", "wb_prefix": "WB_DCD_PlanetScope_",
         "gt_dir": None, "gt_pattern": None, "gt_water_value": None},
    ],
}
QUICKLOOK_MAX = 1000  # 긴 변 픽셀
MASK_MAX = 640


def _out_shape(width: int, height: int, longest: int) -> tuple[int, int]:
    scale = longest / max(width, height)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def read_quicklook(path: Path, sensor_type: str) -> tuple[np.ndarray, np.ndarray]:
    """(RGB uint8 HxWx3, valid mask HxW) — 긴 변 QUICKLOOK_MAX."""

    with rasterio.open(path) as ds:
        h, w = _out_shape(ds.width, ds.height, QUICKLOOK_MAX)
        if sensor_type == "SAR":
            band = ds.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
            nodata = ds.nodata if ds.nodata is not None else -9999.0
            valid = np.isfinite(band) & (band > nodata + 1e-3) & (band != 0)
            values = band[valid]
            lo, hi = (np.percentile(values, [2, 98]) if values.size else (-25.0, 5.0))
            gray = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
            gray[~valid] = 0
            rgb = np.stack([gray] * 3, axis=-1)
        else:
            # PlanetScope Pre: B,G,R,NIR 반사도. nodata 6.5535
            data = ds.read([3, 2, 1], out_shape=(3, h, w), resampling=Resampling.average).astype(np.float32)
            nodata = ds.nodata if ds.nodata is not None else 6.5535
            valid = np.all(np.isfinite(data), axis=0) & np.all(data < nodata - 1e-3, axis=0) & np.any(data > 0, axis=0)
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(3):
                chan = data[i]
                vals = chan[valid]
                lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0.0, 0.3))
                rgb[..., i] = np.clip((chan - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb[~valid] = 0
        return (rgb * 255).astype(np.uint8), valid


def read_mask_like(path: Path, ref_path: Path, longest: int, water_value: int, bounds: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """마스크를 참조 영상과 같은 범위·격자 비율로 읽는다. (water bool HxW, valid HxW)"""

    with rasterio.open(ref_path) as ref:
        ref_bounds = bounds or tuple(ref.bounds)
        h, w = _out_shape(ref.width, ref.height, longest)
    with rasterio.open(path) as ds:
        window = from_bounds(*ref_bounds, transform=ds.transform)
        if bounds is not None:
            cols = max(1, int(round(window.width)))
            rows = max(1, int(round(window.height)))
            h, w = _out_shape(cols, rows, longest)
        arr = ds.read(1, window=window, out_shape=(h, w), resampling=Resampling.nearest, boundless=True, fill_value=255)
        nodata = int(ds.nodata) if ds.nodata is not None else 255
        valid = arr != nodata
        water = valid & (arr == water_value)
    return water, valid


def blend(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1 - alpha) + col * alpha
    return out.astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, optimize=True)


def scene_area_km2(mask_path: Path) -> tuple[float, float]:
    """전체 해상도 마스크에서 수체 면적·유효 면적(km²)."""

    with rasterio.open(mask_path) as ds:
        px = abs(ds.transform.a * ds.transform.e)
        water = 0
        valid = 0
        nodata = int(ds.nodata) if ds.nodata is not None else 255
        for _, window in ds.block_windows(1):
            block = ds.read(1, window=window)
            valid += int((block != nodata).sum())
            water += int((block == 1).sum())
    return water * px / 1e6, valid * px / 1e6


def common_bounds(paths: list[Path]) -> tuple[float, float, float, float]:
    lefts, bottoms, rights, tops = [], [], [], []
    for p in paths:
        with rasterio.open(p) as ds:
            lefts.append(ds.bounds.left); bottoms.append(ds.bounds.bottom)
            rights.append(ds.bounds.right); tops.append(ds.bounds.top)
    return max(lefts), max(bottoms), min(rights), min(tops)


def build_region(region_id: str, regions_dir: Path) -> dict:
    sources = REGION_SOURCES[region_id]
    region_dir = regions_dir / region_id
    manifest_path = region_dir / "region.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    detection = {}
    det_path = region_dir / "satellite" / "detection_eval.json"
    if det_path.is_file():
        payload = json.loads(det_path.read_text(encoding="utf-8"))
        for sensor, block in payload.get("sensors", {}).items():
            for scene in block.get("per_scene", []):
                detection[(sensor, scene["date"])] = scene
    sat_dir = region_dir / "satellite"
    scenes_out: list[dict] = []
    for source in sources:
        sensor = source["sensor"]
        pre_root = HANDOVER / "05_l1_pre" / source["pre_dir"]
        pre_files = sorted(p for p in pre_root.glob("*/*/Pre_*.tif") if not p.name.endswith("_lsmap.tif"))
        if not pre_files:
            print(f"[{region_id}/{sensor}] Pre 장면 없음: {pre_root}", file=sys.stderr)
            continue
        # 마스크가 있는 장면만 공통 격자 계산에 넣는다
        entries = []
        for pre in pre_files:
            date = pre.parent.name
            wb_dir = source.get("wb_root", WB_ROOT) / date[:4] / date
            masks = sorted(wb_dir.glob(f"{source['wb_prefix']}{date}T*.tif")) if wb_dir.is_dir() else []
            entries.append((date, pre, masks[0] if masks else None))
        mask_paths = [m for _, _, m in entries if m]
        bounds = common_bounds(mask_paths) if len(mask_paths) >= 2 else None
        for date, pre, mask in entries:
            scene_id = f"{date}_{sensor}"
            iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            print(f"[{region_id}] {scene_id} …", flush=True)
            rgb, valid = read_quicklook(pre, source["type"])
            images = {"input": f"quicklooks/{scene_id}_input.png"}
            save_png(rgb, sat_dir / images["input"])
            record: dict = {
                "scene_id": scene_id, "date": iso, "sensor": sensor, "sensor_type": source["type"],
                "source_pre": str(pre.relative_to(ROOT)), "images": images,
                "detection_available": mask is not None,
            }
            if mask is not None:
                water, _ = read_mask_like(mask, pre, QUICKLOOK_MAX, 1)
                if water.shape != rgb.shape[:2]:
                    water = np.array(Image.fromarray(water.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 0
                images["overlay"] = f"quicklooks/{scene_id}_overlay.png"
                save_png(blend(rgb, water, (34, 211, 238)), sat_dir / images["overlay"])
                mask_only = np.zeros_like(rgb)
                mask_only[water] = (34, 211, 238)
                images["mask"] = f"quicklooks/{scene_id}_mask.png"
                save_png(mask_only, sat_dir / images["mask"])
                area, valid_area = scene_area_km2(mask)
                record.update({"water_area_km2": round(area, 4), "valid_area_km2": round(valid_area, 2), "source_mask": str(mask.relative_to(ROOT))})
                if bounds is not None:
                    frame, _ = read_mask_like(mask, pre, MASK_MAX, 1, bounds=bounds)
                    frame_path = sat_dir / "masks" / sensor / f"{scene_id}.png"
                    save_png((frame.astype(np.uint8) * 255), frame_path)
                    with rasterio.open(mask) as ds:
                        px = abs(ds.transform.a)
                    scale = (bounds[2] - bounds[0]) / frame.shape[1]
                    record["frame"] = {"path": f"masks/{sensor}/{scene_id}.png", "height": int(frame.shape[0]), "width": int(frame.shape[1]),
                                       "pixel_size_m": round(scale, 3), "pixel_area_m2": round(scale * scale, 3), "native_pixel_m": px,
                                       "bounds_epsg32652": [round(b, 1) for b in bounds]}
            if source["gt_dir"] is not None:
                gt_path = source["gt_dir"] / source["gt_pattern"].format(date=date)
                if gt_path.is_file():
                    gt_water, _ = read_mask_like(gt_path, pre, QUICKLOOK_MAX, source["gt_water_value"])
                    if gt_water.shape != rgb.shape[:2]:
                        gt_water = np.array(Image.fromarray(gt_water.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 0
                    images["gt"] = f"quicklooks/{scene_id}_gt.png"
                    save_png(blend(rgb, gt_water, (250, 204, 21)), sat_dir / images["gt"])
                    record["gt_available"] = True
            metric = detection.get((sensor, iso))
            if metric:
                record["metrics"] = {k: metric[k] for k in ("water_iou", "f1", "precision", "recall") if k in metric}
            scenes_out.append(record)
    scenes_out.sort(key=lambda r: (r["sensor"], r["date"]))
    payload = {
        "schema": "watercast.satellite.scenes/1",
        "region_id": region_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quicklook_max_px": QUICKLOOK_MAX,
        "mask_frame_max_px": MASK_MAX,
        "note_ko": (
            "퀵룩은 원본을 긴 변 1000px로 내린 것. overlay 청록 = 탐지 마스크(WBMS), gt 노랑 = GT 라벨. "
            "masks/<센서>/ 는 센서별 공통 범위로 잘라 같은 크기로 맞춘 이진 마스크로, 수체 면적 시계열 예측의 입력 프레임이다."
        ),
        "scenes": scenes_out,
    }
    (sat_dir / "scenes.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest.setdefault("satellite", {})["scenes_file"] = "satellite/scenes.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", action="append", required=True, choices=sorted(REGION_SOURCES))
    parser.add_argument("--regions-dir", default=str(ROOT / "regions"))
    args = parser.parse_args()
    for region_id in args.region:
        payload = build_region(region_id, Path(args.regions_dir))
        print(f"{region_id}: {len(payload['scenes'])} scenes, "
              f"{sum(1 for s in payload['scenes'] if s.get('detection_available'))} with masks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
