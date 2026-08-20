#!/usr/bin/env python3
"""ICEYE SLC(h5) → sigma0(dB) ingest.

1_water_body_detection/README.md 절차 중 1(intensity)·4(sigma0)·5(median
filter)·6(dB) 단계의 실구현. 산출물은 slant-range 기하(지리참조 없음)이며
코너 GCP만 부착된다 — 정사보정(2·7단계)은 scripts/iceye_geocode.py가 담당.

산출물
  data/ingest/<date>_ICEYE_sigma0db_slant.tif   Float32, nodata -9999, GCP 5점
  ui_next/assets/nas/iceye_2020/<date>_slc_quicklook.png
  pipeline_ui/assets/iceye_<date>_busan.png
  ui_next/assets/nas/iceye_2020/iceye_meta.json (+ pipeline_ui/assets/ 복사본)

통계(p2/p50/p98 등)는 슬라이스가 아닌 멀티룩 전체 장면 기준으로 계산한다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_H5 = REPO / "ICEYE_X5_SLC_SM_23467_20200302T183857-008.h5"
NODATA = -9999.0

SCALAR_KEYS = [
    "product_name", "satellite_name", "product_type", "product_level",
    "acquisition_mode", "polarization", "look_side", "orbit_direction",
    "orbit_absolute_number", "orbit_repeat_cycle", "heading",
    "acquisition_start_utc", "acquisition_end_utc", "calibration_factor",
    "carrier_frequency", "chirp_bandwidth", "slant_range_spacing",
    "azimuth_ground_spacing", "satellite_look_angle", "mean_orbit_altitude",
    "avg_scene_height", "range_looks", "azimuth_looks",
]


def _scalar(ds):
    v = ds[()]
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_meta(f: h5py.File) -> dict:
    meta = {k: _scalar(f[k]) for k in SCALAR_KEYS if k in f}
    inc = f["local_incidence_angle"][:]
    meta["incidence_deg"] = {"min": float(inc.min()), "max": float(inc.max())}
    # coord_*: [range_px(col), azimuth_line(row), lat, lon], 1-기반
    corners = {}
    for k in ("coord_first_near", "coord_first_far", "coord_last_near",
              "coord_last_far", "coord_center"):
        c = f[k][:]
        corners[k] = {"col1": float(c[0]), "row1": float(c[1]),
                      "lat": float(c[2]), "lon": float(c[3])}
    fn = corners["coord_first_near"]
    if not (fn["col1"] == 1.0 and fn["row1"] == 1.0):
        raise AssertionError(f"coord_* 축 순서/오프셋 가정 위배: first_near={fn}")
    meta["corners"] = corners
    return meta


def _haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def stream_sigma0_multilook(f: h5py.File, cal: float, looks_az: int,
                            looks_rg: int, block_rows: int) -> np.ndarray:
    s_i, s_q = f["s_i"], f["s_q"]
    n_az, n_rg = s_i.shape
    az_out, rg_out = n_az // looks_az, n_rg // looks_rg
    n_az_use, n_rg_use = az_out * looks_az, rg_out * looks_rg  # 홀수 여분 crop
    out = np.empty((az_out, rg_out), dtype=np.float32)
    block_rows -= block_rows % looks_az
    for r0 in range(0, n_az_use, block_rows):
        r1 = min(r0 + block_rows, n_az_use)
        i = s_i[r0:r1, :n_rg_use].astype(np.float32)
        q = s_q[r0:r1, :n_rg_use].astype(np.float32)
        sig = cal * (i * i + q * q)
        rows = r1 - r0
        ml = sig.reshape(rows // looks_az, looks_az, rg_out, looks_rg).mean(axis=(1, 3))
        out[r0 // looks_az:r1 // looks_az] = ml
        print(f"  sigma0/multilook {r1}/{n_az_use} lines", flush=True)
    return out


def to_db(sig_linear: np.ndarray) -> tuple[np.ndarray, float]:
    valid = sig_linear > 0
    db = np.full(sig_linear.shape, NODATA, dtype=np.float32)
    db[valid] = 10.0 * np.log10(sig_linear[valid])
    return db, float(1.0 - valid.mean())


def make_quicklook(db: np.ndarray, max_dim: int) -> tuple[np.ndarray, dict]:
    stride = max(1, math.ceil(max(db.shape) / max_dim))
    ql = db[::stride, ::stride]
    valid = ql > NODATA
    lo, hi = np.percentile(ql[valid], [2, 98])
    img = np.zeros(ql.shape, dtype=np.uint8)
    img[valid] = np.clip((ql[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return img, {"stretch_db": [float(lo), float(hi)], "stride": stride}


def write_gcp_geotiff(path: Path, db: np.ndarray, corners: dict,
                      looks_az: int, looks_rg: int, meta: dict):
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS

    gcps = []
    for k, c in corners.items():
        # 1-기반 [col,row] → 0-기반 풀해상도 → 멀티룩 픽셀 좌표
        gcps.append(GroundControlPoint(
            row=(c["row1"] - 1.0) / looks_az, col=(c["col1"] - 1.0) / looks_rg,
            x=c["lon"], y=c["lat"], id=k))
    profile = dict(driver="GTiff", width=db.shape[1], height=db.shape[0],
                   count=1, dtype="float32", nodata=NODATA,
                   compress="deflate", tiled=True,
                   blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(db, 1)
        dst.gcps = (gcps, CRS.from_epsg(4326))
        dst.update_tags(
            GEOMETRY="slant-range (not orthorectified)",
            PRODUCT=meta.get("product_name", ""),
            PROCESSING=f"sigma0=cal*(i^2+q^2), multilook {looks_az}x{looks_rg}, "
                       f"median 3x3, dB",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument("--looks", type=int, nargs=2, default=(2, 2),
                    metavar=("AZ", "RG"))
    ap.add_argument("--median", type=int, default=3,
                    help="median filter 윈도우 (0=생략)")
    ap.add_argument("--block-rows", type=int, default=2048)
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "ingest")
    args = ap.parse_args()

    looks_az, looks_rg = args.looks
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        meta = read_meta(f)
        date = meta["acquisition_start_utc"][:10].replace("-", "")
        cal = float(meta["calibration_factor"])
        print(f"[1/5] sigma0 + multilook {looks_az}x{looks_rg} 스트리밍")
        sig = stream_sigma0_multilook(f, cal, looks_az, looks_rg, args.block_rows)

    if args.median:
        print(f"[2/5] median filter {args.median}x{args.median}")
        from scipy.ndimage import median_filter
        sig = median_filter(sig, size=args.median)

    print("[3/5] dB 변환 + 전장면 통계")
    db, zero_frac = to_db(sig)
    valid = db > NODATA
    p2, p50, p98 = (float(v) for v in np.percentile(db[valid], [2, 50, 98]))
    mean_db_of_linear = float(10.0 * np.log10(sig[valid].mean()))
    del sig

    c = meta["corners"]
    az_km = _haversine_km(c["coord_first_near"], c["coord_last_near"])
    rg_km = _haversine_km(c["coord_first_near"], c["coord_first_far"])

    print("[4/5] GeoTIFF + 퀵룩 PNG")
    tif_path = args.out_dir / f"{date}_ICEYE_sigma0db_slant.tif"
    write_gcp_geotiff(tif_path, db, c, looks_az, looks_rg, meta)

    from PIL import Image

    def atomic_png(arr: np.ndarray, path: Path):
        # UI가 부분 기록 파일을 읽지 않도록 임시파일 → 원자 교체
        tmp = path.with_name(path.name + ".tmp")
        Image.fromarray(arr).save(tmp, format="PNG")
        tmp.replace(path)

    ql_ui, ql_info = make_quicklook(db, 1024)
    ql_pipe, _ = make_quicklook(db, 768)
    ui_png = REPO / "ui_next/assets/nas/iceye_2020" / f"{date}_slc_quicklook.png"
    pipe_png = REPO / "pipeline_ui/assets" / f"iceye_{date}_busan.png"
    pipe_png.parent.mkdir(parents=True, exist_ok=True)
    atomic_png(ql_ui, ui_png)
    atomic_png(ql_pipe, pipe_png)

    print("[5/5] iceye_meta.json")
    order = ["coord_first_near", "coord_first_far", "coord_last_far", "coord_last_near"]
    meta_out = {
        "schema_version": "1.0",
        "generated_by": "scripts/iceye_slc_ingest.py",
        "acquisition_date": meta["acquisition_start_utc"][:10],
        "satellite": meta["satellite_name"],
        "product_name": meta["product_name"],
        "product_type": meta["product_type"],
        "product_level": meta["product_level"],
        "polarization": meta["polarization"],
        "look_side": meta["look_side"],
        "orbit_direction": meta["orbit_direction"],
        "orbit_absolute_number": meta["orbit_absolute_number"],
        # h5가 반복주기 미제공 시 99999 센티널을 씀 — 그대로 노출하지 않는다
        "orbit_repeat_cycle_days": (meta["orbit_repeat_cycle"]
                                    if 0 < meta["orbit_repeat_cycle"] <= 60
                                    else None),
        "acquisition_start_utc": meta["acquisition_start_utc"],
        "acquisition_end_utc": meta["acquisition_end_utc"],
        "heading_deg": meta["heading"],
        "incidence_deg": meta["incidence_deg"],
        "carrier_frequency_ghz": meta["carrier_frequency"] / 1e9,
        "chirp_bandwidth_mhz": meta["chirp_bandwidth"] / 1e6,
        "slant_range_spacing_m": meta["slant_range_spacing"],
        "azimuth_ground_spacing_m": meta["azimuth_ground_spacing"],
        "center_latlon": [c["coord_center"]["lat"], c["coord_center"]["lon"]],
        "footprint_latlon": [[c[k]["lat"], c[k]["lon"]] for k in order],
        "scene_size_km": {"azimuth": round(az_km, 1), "range": round(rg_km, 1)},
        "calibration_factor": meta["calibration_factor"],
        "processing": {
            "geometry": "slant-range (not orthorectified)",
            "looks": [looks_az, looks_rg],
            "median_filter": args.median,
            "nodata": NODATA,
            "stats_db_full_scene": {
                "p2": p2, "p50": p50, "p98": p98,
                "mean_db_of_linear_mean": mean_db_of_linear,
                "zero_pixel_fraction": zero_frac,
            },
            "quicklook_stretch_db": ql_info["stretch_db"],
        },
        "quicklook": {
            "ui_next": str(ui_png.relative_to(REPO)),
            "pipeline_ui": str(pipe_png.relative_to(REPO)),
        },
        "sigma0_geotiff": str(tif_path.relative_to(REPO)),
    }
    for dst in (ui_png.parent / "iceye_meta.json",
                pipe_png.parent / "iceye_meta.json"):
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2))
        tmp.replace(dst)

    print(json.dumps(meta_out["processing"], ensure_ascii=False, indent=2))
    print(f"done: {tif_path} ({tif_path.stat().st_size/1e6:.0f} MB), "
          f"{ui_png.name}, {pipe_png.name}")


if __name__ == "__main__":
    main()
