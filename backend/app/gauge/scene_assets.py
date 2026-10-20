"""위성 장면 하나를 지역 자산(퀵룩·오버레이·공통 격자 마스크·scenes.json 항목)으로 만든다.

`scripts/make_region_scene_assets.py`(일괄)와 `POST /regions/{id}/satellite/scenes`(장면 하나 등록)가
같은 코드를 쓴다. 새 위성 데이터가 들어오면 이 함수 하나로 화면·면적 예측 샘플까지 이어진다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

QUICKLOOK_MAX = 1000
MASK_MAX = 640
WATER_COLOR = (34, 211, 238)
GT_COLOR = (250, 204, 21)


def _rasterio():
    import rasterio  # 지연 import — 백엔드 기동 시 필수 의존이 아니게

    return rasterio


def _out_shape(width: int, height: int, longest: int) -> tuple[int, int]:
    scale = longest / max(width, height)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def read_quicklook(path: Path, sensor_type: str) -> np.ndarray:
    """RGB uint8 (H, W, 3), 긴 변 QUICKLOOK_MAX. SAR: dB 2~98% 스트레치, 광학: R,G,B 밴드."""

    rasterio = _rasterio()
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        h, w = _out_shape(ds.width, ds.height, QUICKLOOK_MAX)
        if sensor_type.upper() == "SAR" or ds.count == 1:
            band = ds.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
            nodata = ds.nodata if ds.nodata is not None else -9999.0
            valid = np.isfinite(band) & (band > nodata + 1e-3) & (band != 0)
            values = band[valid]
            lo, hi = (np.percentile(values, [2, 98]) if values.size else (-25.0, 5.0))
            gray = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
            gray[~valid] = 0
            rgb = np.stack([gray] * 3, axis=-1)
        else:
            bands = [3, 2, 1] if ds.count >= 3 else [1, 1, 1]
            data = ds.read(bands, out_shape=(3, h, w), resampling=Resampling.average).astype(np.float32)
            nodata = ds.nodata if ds.nodata is not None else 6.5535
            valid = np.all(np.isfinite(data), axis=0) & np.all(data < nodata - 1e-3, axis=0) & np.any(data > 0, axis=0)
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(3):
                chan = data[i]
                vals = chan[valid]
                lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0.0, 0.3))
                rgb[..., i] = np.clip((chan - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb[~valid] = 0
    return (rgb * 255).astype(np.uint8)


def read_mask_like(path: Path, ref_path: Path, longest: int, water_value: int = 1, bounds: tuple | None = None) -> np.ndarray:
    """마스크를 참조 영상 범위(또는 bounds)로 잘라 긴 변 longest 로 읽는다. bool (H, W)."""

    rasterio = _rasterio()
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

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
        return (arr != nodata) & (arr == water_value)


def blend(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    if mask.shape != rgb.shape[:2]:
        from PIL import Image

        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 0
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return out.astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, optimize=True)


def scene_area_km2(mask_path: Path) -> tuple[float, float]:
    rasterio = _rasterio()
    with rasterio.open(mask_path) as ds:
        px = abs(ds.transform.a * ds.transform.e)
        nodata = int(ds.nodata) if ds.nodata is not None else 255
        water = valid = 0
        for _, window in ds.block_windows(1):
            block = ds.read(1, window=window)
            valid += int((block != nodata).sum())
            water += int((block == 1).sum())
    return water * px / 1e6, valid * px / 1e6


def raster_bounds(path: Path) -> tuple[float, float, float, float]:
    rasterio = _rasterio()
    with rasterio.open(path) as ds:
        return tuple(ds.bounds)


def native_pixel_m(path: Path) -> float:
    rasterio = _rasterio()
    with rasterio.open(path) as ds:
        return abs(ds.transform.a)


def bounds_to_lonlat(bounds: tuple[float, float, float, float] | Sequence[float], crs: Any) -> list[list[float]]:
    """상자 네 모서리(좌상→우상→우하→좌하, 시계 방향)를 경위도로. pyproj 없이 rasterio.warp.transform 을 쓴다."""

    from rasterio.warp import transform

    minx, miny, maxx, maxy = (float(v) for v in bounds)
    xs = [minx, maxx, maxx, minx]
    ys = [maxy, maxy, miny, miny]
    lons, lats = transform(crs, "EPSG:4326", xs, ys)
    return [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in zip(lons, lats)]


def quicklook_georef(pre_tif: Path, width: int, height: int) -> dict[str, Any] | None:
    """퀵룩 PNG(원본을 긴 변 QUICKLOOK_MAX 로 내린 것)의 지리참조 — 원본 GeoTIFF 의 범위 그대로, 크기만 PNG 화소.

    화면은 `bounds_epsg32652`→`(width, height)` 의 아핀 매핑으로 경위도 점(수위계·AOI)을 화소로 찍는다(ui_next/geo_overlay.py).
    원본이 EPSG:32652 가 아니면 범위를 EPSG:32652 로 옮겨 적고 `source_crs` 에 원 좌표계를 남긴다. 좌표계가 없는 원본이면 None.
    """

    rasterio = _rasterio()
    from rasterio.warp import transform_bounds

    with rasterio.open(pre_tif) as ds:
        if ds.crs is None:
            return None
        crs = ds.crs
        bounds = tuple(float(v) for v in ds.bounds)
    source_crs = str(crs)
    if source_crs.upper() != "EPSG:32652":
        bounds = tuple(float(v) for v in transform_bounds(crs, "EPSG:32652", *bounds))
    return {
        "crs": "EPSG:32652",
        "source_crs": source_crs,
        "bounds_epsg32652": [round(v, 3) for v in bounds],
        "width": int(width),
        "height": int(height),
        "footprint_lonlat": bounds_to_lonlat(bounds, "EPSG:32652"),
    }


def build_scene_record(
    *,
    sat_dir: Path,
    scene_id: str,
    date_iso: str,
    sensor: str,
    sensor_type: str,
    pre_tif: Path,
    mask_tif: Path | None,
    gt_tif: Path | None = None,
    gt_water_value: int = 1,
    frame_bounds: tuple[float, float, float, float] | None = None,
    metrics: Mapping[str, Any] | None = None,
    source_note: str | None = None,
) -> dict[str, Any]:
    """퀵룩·오버레이·(GT)·공통 격자 마스크 PNG를 만들고 scenes.json 항목을 돌려준다."""

    rgb = read_quicklook(pre_tif, sensor_type)
    images = {"input": f"quicklooks/{scene_id}_input.png"}
    save_png(rgb, sat_dir / images["input"])
    rasterio = _rasterio()
    try:  # 장면 자체의 좌표계·해상도 — 지역 기본값과 다를 수 있다(업로드 장면)
        with rasterio.open(pre_tif) as ds:
            crs_text = str(ds.crs) if ds.crs else None
            pixel_m = [round(float(abs(ds.transform.a)), 3), round(float(abs(ds.transform.e)), 3)]
            size_px = [int(ds.width), int(ds.height)]
    except Exception:  # noqa: BLE001 - 메타 없이도 등록은 된다
        crs_text, pixel_m, size_px = None, None, None
    try:  # 퀵룩 지리참조 — 화면에서 수위계·AOI 를 사진 위에 찍는 데 쓴다
        quicklook = quicklook_georef(pre_tif, rgb.shape[1], rgb.shape[0])
    except Exception:  # noqa: BLE001
        quicklook = None
    record: dict[str, Any] = {
        "scene_id": scene_id, "date": date_iso, "sensor": sensor, "sensor_type": sensor_type,
        "source_pre": pre_tif.name, "images": images, "detection_available": mask_tif is not None,
        "crs": crs_text, "pixel_size_m": pixel_m, "size_px": size_px, "quicklook": quicklook,
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if source_note:
        record["source_note"] = source_note
    if mask_tif is not None:
        water = read_mask_like(mask_tif, pre_tif, QUICKLOOK_MAX, 1)
        images["overlay"] = f"quicklooks/{scene_id}_overlay.png"
        save_png(blend(rgb, water, WATER_COLOR), sat_dir / images["overlay"])
        mask_only = np.zeros_like(rgb)
        if water.shape == rgb.shape[:2]:
            mask_only[water] = WATER_COLOR
        images["mask"] = f"quicklooks/{scene_id}_mask.png"
        save_png(mask_only, sat_dir / images["mask"])
        area, valid_area = scene_area_km2(mask_tif)
        record.update({"water_area_km2": round(area, 4), "valid_area_km2": round(valid_area, 2), "source_mask": str(mask_tif)})
        bounds = frame_bounds or raster_bounds(mask_tif)
        frame = read_mask_like(mask_tif, pre_tif, MASK_MAX, 1, bounds=bounds)
        frame_path = sat_dir / "masks" / sensor / f"{scene_id}.png"
        save_png(frame.astype(np.uint8) * 255, frame_path)
        scale = (bounds[2] - bounds[0]) / frame.shape[1]
        record["frame"] = {
            "path": f"masks/{sensor}/{scene_id}.png", "height": int(frame.shape[0]), "width": int(frame.shape[1]),
            "pixel_size_m": round(scale, 3), "pixel_area_m2": round(scale * scale, 3), "native_pixel_m": native_pixel_m(mask_tif),
            "bounds_epsg32652": [round(b, 1) for b in bounds],
        }
    if gt_tif is not None and gt_tif.is_file():
        gt_water = read_mask_like(gt_tif, pre_tif, QUICKLOOK_MAX, gt_water_value)
        images["gt"] = f"quicklooks/{scene_id}_gt.png"
        save_png(blend(rgb, gt_water, GT_COLOR), sat_dir / images["gt"])
        record["gt_available"] = True
    if metrics:
        record["metrics"] = dict(metrics)
    return record


def upsert_scene(sat_dir: Path, record: Mapping[str, Any], region_id: str) -> dict[str, Any]:
    """scenes.json 에 항목을 넣거나(같은 scene_id) 바꾼다."""

    path = sat_dir / "scenes.json"
    payload: dict[str, Any]
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"schema": "watercast.satellite.scenes/1", "region_id": region_id, "quicklook_max_px": QUICKLOOK_MAX,
                   "mask_frame_max_px": MASK_MAX, "scenes": [],
                   "note_ko": "퀵룩은 원본을 긴 변 1000px로 내린 것. overlay 청록 = 탐지 마스크, gt 노랑 = GT 라벨. masks/<센서>/ 는 센서별 공통 범위 이진 마스크."}
    scenes = [s for s in payload.get("scenes", []) if s.get("scene_id") != record["scene_id"]]
    scenes.append(dict(record))
    scenes.sort(key=lambda s: (s["sensor"], s["date"]))
    payload["scenes"] = scenes
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def existing_frame_bounds(sat_dir: Path, sensor: str) -> tuple[float, float, float, float] | None:
    """같은 센서의 기존 프레임 범위 — 새 장면도 같은 격자로 잘라야 시계열 프레임 크기가 맞는다."""

    path = sat_dir / "scenes.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for scene in payload.get("scenes", []):
        frame = scene.get("frame") or {}
        if scene.get("sensor") == sensor and frame.get("bounds_epsg32652"):
            b = frame["bounds_epsg32652"]
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return None
