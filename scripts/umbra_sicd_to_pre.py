#!/usr/bin/env python3
"""Umbra SICD(복소 SAR, NITF) → WBMS ① 입력 규격(Pre_<TB>_UMBRA_<stamp>.tif) 변환.

ICEYE 경로는 GAMMA 로 γ⁰ 지오코딩을 했지만 Umbra 는 SICD 에 방사 보정 다항식(GammaZeroSFPoly)과
정확한 투영 모델이 들어 있어 sarpy 만으로 같은 물리량을 만들 수 있다.

    .venv/bin/python scripts/umbra_sicd_to_pre.py --input <scene.nitf> --testbed dcd \
        --dem data/incoming/handover/06_aux/DEM_daecheong_ngii3m/daecheong_ngii_32652_3m_topofilled.tif \
        --output_dir data/incoming/kwater_202610/umbra_pre

흐름
  1. 복소 영상을 행 블록으로 읽어 |z|² 에 γ⁰ 보정 다항식을 곱하고(선형), 행·열 방향으로 묶어 ~3 m 로
     다중관측(multilook)한다 — 스포틀라이트 0.1~0.3 m 화소라 수백 룩이 되어 스페클이 크게 줄어든다.
  2. 출력 격자: EPSG:32652 · 3 m · 기존 대청 PlanetScope 격자 원점(357867, 4043247)에 맞춰 스냅.
  3. 역방향 지오코딩: 출력 화소마다 경위도 + 높이(DEM, 지오이드 차는 SCP 높이로 보정)를 SICD 투영 모델로
     영상 좌표(row, col)로 보내 다중관측 영상을 쌍선형 보간한다.
  4. 10·log10 → dB, 촬영범위 밖 -9999. 사이드카 _meta.json(ICEYE 규격 동일 키) · 3-class 마스크
     (0=유효, 1=제외 — 내륙이라 해양 2 없음) · 퀵룩 PNG.

출력 규격은 인수 매뉴얼 IO_SPEC §6.2(ICEYE 1밴드 3.0 m dB, nodata -9999, EPSG:32652)와 같다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NODATA = -9999.0
PLANET_DCD_ORIGIN = (357867.0, 4043247.0)  # 기존 대청 PlanetScope 격자 좌상단 — 공통 격자 정렬용
EPSG_OUT = "EPSG:32652"


def log(msg: str) -> None:
    print(f"[umbra→pre] {msg}", flush=True)


def snap(value: float, origin: float, step: float, *, up: bool) -> float:
    k = (value - origin) / step
    k = math.ceil(k) if up else math.floor(k)
    return origin + k * step


def multilook_gamma0(reader, sicd, fr: int, fc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """|z|²·GammaZeroSF 를 (fr×fc) 블록 평균 → 선형 γ⁰ 다중관측 영상과 블록 중심의 영상 좌표."""

    nrows, ncols = int(sicd.ImageData.NumRows), int(sicd.ImageData.NumCols)
    R, C = nrows // fr, ncols // fc
    out = np.empty((R, C), dtype=np.float32)
    scp_r, scp_c = float(sicd.ImageData.SCPPixel.Row), float(sicd.ImageData.SCPPixel.Col)
    row_ss, col_ss = float(sicd.Grid.Row.SS), float(sicd.Grid.Col.SS)
    first_r, first_c = int(sicd.ImageData.FirstRow), int(sicd.ImageData.FirstCol)
    poly = sicd.Radiometric.GammaZeroSFPoly if sicd.Radiometric is not None else None
    if poly is None:
        log("경고: GammaZeroSFPoly 없음 — 보정 없이 |z|² 사용 (상대값)")
    col_centers = first_c + np.arange(C) * fc + (fc - 1) / 2.0
    ycol = (col_centers - scp_c) * col_ss
    rows_per_block = max(fr, (fr * max(1, 64 // fr)))  # 64행 단위로 읽는다
    t0 = time.time()
    r = 0
    while r < R:
        rb = min(R - r, rows_per_block // fr)
        r0, r1 = r * fr, (r + rb) * fr
        z = reader[r0:r1, :C * fc]
        power = (z.real.astype(np.float32) ** 2 + z.imag.astype(np.float32) ** 2)
        blk = power.reshape(rb, fr, C, fc).mean(axis=(1, 3))
        if poly is not None:
            row_centers = first_r + (np.arange(r, r + rb) * fr + (fr - 1) / 2.0)
            xrow = (row_centers - scp_r) * row_ss
            # sarpy Poly2D 는 같은 모양의 x·y 배열을 요구한다(브로드캐스트 불가)
            X = np.repeat(xrow[:, None], C, axis=1); Y = np.repeat(ycol[None, :], rb, axis=0)
            blk *= poly(X, Y).astype(np.float32)
        out[r:r + rb] = blk
        r += rb
        if r % (rb * 40) == 0 or r >= R:
            log(f"  multilook {r}/{R} 행 ({time.time() - t0:.0f}s)")
    row_centers_all = first_r + np.arange(R) * fr + (fr - 1) / 2.0
    return out, row_centers_all, col_centers


def bilinear(img: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """img[rr, cc] 쌍선형 보간. 범위 밖은 NaN."""

    H, W = img.shape
    out = np.full(rr.shape, np.nan, dtype=np.float32)
    ok = (rr >= 0) & (rr <= H - 1) & (cc >= 0) & (cc <= W - 1)
    r = rr[ok]; c = cc[ok]
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, H - 1); c1 = np.minimum(c0 + 1, W - 1)
    wr = (r - r0).astype(np.float32); wc = (c - c0).astype(np.float32)
    v = (img[r0, c0] * (1 - wr) * (1 - wc) + img[r1, c0] * wr * (1 - wc)
         + img[r0, c1] * (1 - wr) * wc + img[r1, c1] * wr * wc)
    out[ok] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Umbra SICD .nitf")
    ap.add_argument("--testbed", default="dcd", help="카탈로그 테스트베드 키 (dcd|busan)")
    ap.add_argument("--dem", required=True, help="EPSG:32652 DEM GeoTIFF (정표고)")
    ap.add_argument("--output_dir", required=True, help="<root>/<yyyy>/<yyyymmdd>/ 아래에 쓴다 (05_l1_pre 규약)")
    ap.add_argument("--res", type=float, default=3.0)
    ap.add_argument("--db_floor", type=float, default=-40.0, help="γ⁰ dB 하한(0 전력 보호)")
    ap.add_argument("--edge_db", type=float, default=2.0, help="가장자리 테이퍼 판정: 중앙 중앙값보다 이만큼 낮은 경계 행·열 제거")
    a = ap.parse_args()

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform
    from sarpy.io.complex.converter import open_complex

    src = Path(a.input)
    reader = open_complex(str(src))
    sicd = reader.sicd_meta
    info = sicd.CollectionInfo
    start = sicd.Timeline.CollectStart
    start_dt = datetime.fromisoformat(str(start).replace("Z", "")).replace(tzinfo=timezone.utc) if start is not None else None
    stamp = start_dt.strftime("%Y%m%dT%H%M%S") if start_dt else "00000000T000000"
    date8 = stamp[:8]
    tb_code = {"dcd": "DCD", "busan": "Busan"}.get(a.testbed, a.testbed.upper())
    stem = f"Pre_{tb_code}_UMBRA_{stamp}"
    out_dir = Path(a.output_dir) / date8[:4] / date8
    out_dir.mkdir(parents=True, exist_ok=True)

    row_ss, col_ss = float(sicd.Grid.Row.SS), float(sicd.Grid.Col.SS)
    fr, fc = max(1, round(a.res / row_ss)), max(1, round(a.res / col_ss))
    log(f"{src.name}: {info.CollectorName} {info.RadarMode.ModeType} {int(sicd.ImageData.NumRows)}x{int(sicd.ImageData.NumCols)} "
        f"SS {row_ss:.3f}x{col_ss:.3f} m → 룩 {fr}x{fc} (≈{fr * row_ss:.2f}x{fc * col_ss:.2f} m)")

    # 1. 다중관측 γ⁰ (선형)
    ml, ml_rows, ml_cols = multilook_gamma0(reader, sicd, fr, fc)

    # 1b. 가장자리 테이퍼 제거 — 영상 경계 쪽에서 평균 dB 가 중앙 평탄부보다 edge_db 이상 낮은 행·열은
    #     안테나 패턴·처리 경계의 어두운 띠라 물로 오인된다(09-28 실측). 연속된 경계 구간만 잘라낸다.
    with np.errstate(divide="ignore"):
        ml_db = 10.0 * np.log10(np.maximum(ml, 1e-12))
    row_prof = np.nanmedian(np.where(ml > 0, ml_db, np.nan), axis=1)
    col_prof = np.nanmedian(np.where(ml > 0, ml_db, np.nan), axis=0)

    def trim(prof: np.ndarray, edge_db: float) -> tuple[int, int]:
        n = prof.size; core = np.nanmedian(prof[n // 4: 3 * n // 4]); lo = core - edge_db
        a = 0
        while a < n // 4 and (np.isnan(prof[a]) or prof[a] < lo):
            a += 1
        b = n
        while b > 3 * n // 4 and (np.isnan(prof[b - 1]) or prof[b - 1] < lo):
            b -= 1
        return a, n - b

    r_lo, r_hi = trim(row_prof, a.edge_db); c_lo, c_hi = trim(col_prof, a.edge_db)
    # 여유 2룩 더 잘라낸다
    r_lo += 2; r_hi += 2; c_lo += 2; c_hi += 2
    ml[:r_lo, :] = 0; ml[ml.shape[0] - r_hi:, :] = 0; ml[:, :c_lo] = 0; ml[:, ml.shape[1] - c_hi:] = 0
    log(f"가장자리 제거: 행 앞/뒤 {r_lo}/{r_hi}, 열 앞/뒤 {c_lo}/{c_hi} 룩 (기준 중앙 중앙값 −{a.edge_db} dB)")

    # 2. 출력 격자 — 모서리 경위도 → UTM, 기존 대청 격자에 스냅
    corners = [(float(c.Lat), float(c.Lon)) for c in sicd.GeoData.ImageCorners.get_array()]
    xs, ys = warp_transform("EPSG:4326", EPSG_OUT, [c[1] for c in corners], [c[0] for c in corners])
    res = a.res
    ox, oy = PLANET_DCD_ORIGIN
    x0 = snap(min(xs) - 2 * res, ox, res, up=False); x1 = snap(max(xs) + 2 * res, ox, res, up=True)
    y1 = snap(max(ys) + 2 * res, oy, res, up=True); y0 = snap(min(ys) - 2 * res, oy, res, up=False)
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    transform = from_origin(x0, y1, res, res)
    log(f"출력 격자 {W}x{H} @ {res} m, UL=({x0:.0f},{y1:.0f}) EPSG:32652")

    # 3. DEM → 출력 격자 (같은 CRS·해상도이므로 창 읽기 + 재표본) + 지오이드 차 보정
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    with rasterio.open(a.dem) as dem_ds:
        win = from_bounds(x0, y0, x1, y1, dem_ds.transform)
        dem = dem_ds.read(1, window=win, out_shape=(H, W), resampling=Resampling.bilinear, boundless=True,
                          fill_value=dem_ds.nodata if dem_ds.nodata is not None else NODATA).astype(np.float64)
        dem_nodata = dem_ds.nodata if dem_ds.nodata is not None else NODATA
        # SCP 위치의 DEM 값과 SICD 의 SCP HAE 차이를 지오이드·기준면 차로 보고 전체에 더한다
        scp = sicd.GeoData.SCP.LLH
        sx, sy = warp_transform("EPSG:4326", EPSG_OUT, [float(scp.Lon)], [float(scp.Lat)])
        try:
            scp_dem = next(dem_ds.sample([(sx[0], sy[0])]))[0]
        except Exception:  # noqa: BLE001
            scp_dem = np.nan
    valid_dem = np.isfinite(dem) & (dem != dem_nodata)
    scp_hae = float(scp.HAE)
    offset = (scp_hae - float(scp_dem)) if np.isfinite(scp_dem) and scp_dem != dem_nodata else 0.0
    if not valid_dem.any():
        log("경고: DEM 이 촬영범위를 덮지 않음 — SCP 높이 평면으로 지오코딩")
    fill_h = float(np.nanmean(dem[valid_dem])) if valid_dem.any() else (scp_hae - offset)
    dem = np.where(valid_dem, dem, fill_h) + offset
    log(f"높이: DEM 유효 {valid_dem.mean() * 100:.1f}% · SCP HAE {scp_hae:.1f} m vs DEM {scp_dem:.1f} m → 보정 {offset:+.1f} m")

    # 4. 역방향 지오코딩 (행 블록)
    out = np.full((H, W), NODATA, dtype=np.float32)
    xs_col = x0 + (np.arange(W) + 0.5) * res
    fr_off, fc_off = (fr - 1) / 2.0, (fc - 1) / 2.0
    first_r, first_c = float(sicd.ImageData.FirstRow), float(sicd.ImageData.FirstCol)
    t0 = time.time()
    block = 128
    for r in range(0, H, block):
        rb = min(block, H - r)
        yy = y1 - (np.arange(r, r + rb) + 0.5) * res
        gx, gy = np.meshgrid(xs_col, yy)
        lon, lat = warp_transform(EPSG_OUT, "EPSG:4326", gx.ravel().tolist(), gy.ravel().tolist())
        coords = np.column_stack([np.asarray(lat), np.asarray(lon), dem[r:r + rb].ravel()])
        img_pts, _delta, _iters = sicd.project_ground_to_image_geo(coords, ordering="latlong")
        rows_full = img_pts[:, 0] - first_r
        cols_full = img_pts[:, 1] - first_c
        # 다중관측 격자 좌표: 블록 중심 = k*f + (f-1)/2
        rr = (rows_full - fr_off) / fr
        cc = (cols_full - fc_off) / fc
        val = bilinear(ml, rr, cc).reshape(rb, W)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(np.maximum(val, 1e-12))
        db = np.where(np.isfinite(val) & (val > 0), np.maximum(db, a.db_floor), NODATA).astype(np.float32)  # 0 전력 = 제거 구역
        out[r:r + rb] = db
        if (r // block) % 10 == 0:
            log(f"  geocode {min(r + rb, H)}/{H} 행 ({time.time() - t0:.0f}s)")

    valid = out != NODATA
    cov = valid.mean() * 100
    v = out[valid]
    log(f"유효 {cov:.1f}% · γ⁰ dB p5/p50/p95 = {np.percentile(v, 5):.1f}/{np.percentile(v, 50):.1f}/{np.percentile(v, 95):.1f}")

    # 5. 쓰기 — Pre tif, 사이드카, mask3, 퀵룩
    pre_path = out_dir / f"{stem}.tif"
    profile = {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "float32", "crs": EPSG_OUT,
               "transform": transform, "nodata": NODATA, "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512}
    with rasterio.open(pre_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(AREA_OR_POINT="Area", SOURCE="Umbra SICD via sarpy", LOOKS=f"{fr}x{fc}")
    mask_dir = out_dir
    mask_path = mask_dir / f"{a.testbed}_{date8}_mask3_utm.tif"
    with rasterio.open(mask_path, "w", **{**profile, "dtype": "uint8", "nodata": None}) as dst:
        dst.write(np.where(valid, 0, 1).astype(np.uint8), 1)
    scpcoa = sicd.SCPCOA
    rcv = sicd.RadarCollection
    pol = None
    try:
        pol = rcv.RcvChannels[0].TxRcvPolarization
    except Exception:  # noqa: BLE001
        pass
    meta = {
        "Document": "Processed", "TestbedName": tb_code, "SatelliteName": "Umbra",
        "OrbitNumber": str(getattr(info, "CoreName", "") or ""),
        "OrbitDirection": "DESCENDING" if (getattr(sicd.SCPCOA, "ARPVel", None) is not None and float(sicd.SCPCOA.ARPVel.Z) < 0) else "ASCENDING",
        "ImageDate": date8,
        "Entry": {"date": date8, "AcquisitionDateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if start_dt else None,
                  "PixelSize_m": str(res), "Bands": "1", "BandOrder": "amplitude", "ValueType": "dB", "Normalized": "false",
                  "CRS": EPSG_OUT, "Scene_UL": {"x": str(x0), "y": str(y1)}, "Scene_Posting": {"x": str(res), "y": str(res)}},
        "Provenance": {"stage": "umbra_sicd_to_pre", "input_file": src.name, "meta_source": f"{src.name}.sicd.xml",
                       "software": "sarpy " + __import__("sarpy").__version__},
        "Umbra": {"collector": str(info.CollectorName), "mode": str(info.RadarMode.ModeType), "polarization": str(pol),
                  "incidence_deg": float(scpcoa.IncidenceAng), "graze_deg": float(scpcoa.GrazeAng), "side": str(scpcoa.SideOfTrack),
                  "looks_row_col": [fr, fc], "sample_spacing_m": [row_ss, col_ss], "radiometric": "GammaZeroSFPoly",
                  "height_offset_m": offset, "valid_fraction": round(float(cov) / 100, 4),
                  "db_p5_p50_p95": [round(float(np.percentile(v, q)), 2) for q in (5, 50, 95)]},
    }
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # 퀵룩 PNG
    try:
        from PIL import Image
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        q = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1); q[~valid] = 0
        scale = max(1, int(math.ceil(max(W, H) / 1200)))
        Image.fromarray((q[::scale, ::scale] * 255).astype(np.uint8)).save(out_dir / f"{stem}_quicklook.png")
    except Exception as exc:  # noqa: BLE001
        log(f"퀵룩 생략: {exc}")
    log(f"완료 → {pre_path}  (+ _meta.json, {mask_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
