#!/usr/bin/env python3
"""pipeline_ui STEP2 '실모델(WBMS U-Net) 산출' 자산 생성기.

체인 러너 ②(detect_water) 산출 WB 마스크(uint8 0=land/1=water/255=nodata,
EPSG:32652 3 m)와 같은 격자의 정사 입력(Pre γ0 dB float32, nodata=-9999)을
읽어 발표 UI용 PNG 3장 + 메타 1장을 out-dir에 원자적으로(tmp→os.replace) 저장.

산출 규약 (기본 --tag 20200302):
  wbms_<tag>_pre_crop.png      관심영역 Pre γ0 (유효픽셀 p2–98 스트레치, 8bit RGB)
  wbms_<tag>_overlay_crop.png  관심영역 Pre + 수체 마스크 오버레이 합성본
  wbms_<tag>_overlay_full.png  전체 씬 축소 오버레이
  wbms_<tag>_meta.json         crop 창·UTM 범위·수체 비율 등 (GT 대조 수치 없음)

관심영역(정사각, 기본 4096 px = 12.3 km)은 라벨 격자에서 수체×육지 혼합
밀도가 최대인 창을 자동 선택한다 — 낙동강 하구(수로·삼각주 혼재)가 뽑히고
순수 외해(수체 100%)나 내륙(수체 0%)은 배제되는 점수 함수.

사용:
  python scripts/make_pipeline_ui_wbms_assets.py                 # 20200302 기본 경로
  python scripts/make_pipeline_ui_wbms_assets.py --pre P.tif --wb W.tif \
      --tag 20200416 --out-dir /tmp/check                        # 다른 씬 검증용

WB 마스크가 아직 없으면(체인 러너 ② 미완) 안내만 출력하고 종료 코드 0으로
스킵한다 — 기존 UI 동작은 자산이 생길 때까지 변하지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEF_PRE = ROOT / ("data/incoming/handover/05_l1_pre/iceye_pre/2020/20200302/"
                  "Pre_Busan_ICEYE_20200302T183857.tif")
DEF_WB = ROOT / "data/wbms_runs/wb/2020/20200302/WB_Busan_ICEYE_20200302T183857.tif"
DEF_OUT = ROOT / "pipeline_ui/assets"

# pipeline_ui/app.py 오버레이 팔레트와 동일 (overlay_mask / mask_to_rgb)
WATER_RGB = np.array([56, 189, 248], dtype=np.float32)
BG_RGB = np.array([15, 27, 46], dtype=np.uint8)
WATER_ALPHA = 0.45


def _pad_block(arr: np.ndarray, factor: int, cw: int, fill) -> np.ndarray:
    """(r, w) 배열을 (crows*factor, cw*factor)로 fill 패딩."""
    r, w = arr.shape
    crows = math.ceil(r / factor)
    out = np.full((crows * factor, cw * factor), fill, dtype=arr.dtype)
    out[:r, :w] = arr
    return out


def _block_stats(pre: np.ndarray, wb: np.ndarray, nodata: float, factor: int, cw: int):
    """청크 한 장의 블록별 (γ0 합, γ0 유효수, 수체수, WB 유효수) 반환."""
    crows = math.ceil(pre.shape[0] / factor)
    p = _pad_block(pre.astype(np.float32), factor, cw, np.nan)
    m = _pad_block(wb, factor, cw, np.uint8(255))
    valid = np.isfinite(p) & (p != nodata)
    pv = np.where(valid, p, 0.0)
    shp = (crows, factor, cw, factor)
    s_db = pv.reshape(shp).sum(axis=(1, 3), dtype=np.float64)
    n_db = valid.reshape(shp).sum(axis=(1, 3), dtype=np.int64)
    n_wat = (m == 1).reshape(shp).sum(axis=(1, 3), dtype=np.int64)
    n_wbv = (m != 255).reshape(shp).sum(axis=(1, 3), dtype=np.int64)
    return s_db, n_db, n_wat, n_wbv


def stream_scene(pre_ds, wb_ds, factor: int, chunk_coarse_rows: int = 16):
    """전체 씬을 행 스트립으로 스트리밍하며 factor×factor 블록 통계 집계."""
    from rasterio.windows import Window

    H, W = pre_ds.height, pre_ds.width
    ch, cw = math.ceil(H / factor), math.ceil(W / factor)
    sum_db = np.zeros((ch, cw), np.float64)
    cnt_db = np.zeros((ch, cw), np.int64)
    cnt_wat = np.zeros((ch, cw), np.int64)
    cnt_wbv = np.zeros((ch, cw), np.int64)
    nodata = pre_ds.nodata if pre_ds.nodata is not None else -9999.0
    for c0 in range(0, ch, chunk_coarse_rows):
        r0 = c0 * factor
        r1 = min(H, (c0 + chunk_coarse_rows) * factor)
        win = Window(0, r0, W, r1 - r0)
        s, n, w, v = _block_stats(pre_ds.read(1, window=win),
                                  wb_ds.read(1, window=win), nodata, factor, cw)
        c1 = c0 + s.shape[0]
        sum_db[c0:c1] += s
        cnt_db[c0:c1] += n
        cnt_wat[c0:c1] += w
        cnt_wbv[c0:c1] += v
    # 블록별 전체 픽셀 수 (가장자리 부분 블록 보정)
    row_px = np.full(ch, factor, np.int64); row_px[-1] = H - factor * (ch - 1)
    col_px = np.full(cw, factor, np.int64); col_px[-1] = W - factor * (cw - 1)
    tot_px = np.outer(row_px, col_px)
    return sum_db, cnt_db, cnt_wat, cnt_wbv, tot_px


def pick_roi(cnt_wat, cnt_wbv, tot_px, win_cells: int):
    """수체×육지 혼합 밀도 최대의 win_cells 정사각 창 좌상단(coarse) 선택."""
    def integral(a):
        c = np.cumsum(np.cumsum(a.astype(np.float64), 0), 1)
        return np.pad(c, ((1, 0), (1, 0)))

    ch, cw = cnt_wat.shape
    k = min(win_cells, ch, cw)
    Iw, Iv, It = integral(cnt_wat), integral(cnt_wbv), integral(tot_px)

    def wsum(I):
        return I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]

    Sw, Sv, St = wsum(Iw), wsum(Iv), wsum(It)
    valid_frac = Sv / np.maximum(St, 1)
    wf = Sw / np.maximum(Sv, 1)
    score = wf * (1.0 - wf)              # 하구형 혼합 영역에서 최대
    score = np.where(valid_frac >= 0.55, score, -1.0)
    if score.max() <= 0:                 # 폴백: 수체 비율 최대 창
        score = np.where(valid_frac >= 0.30, wf, -1.0)
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    return r, c, k, float(wf[r, c]), float(valid_frac[r, c])


def render_rgb(gray01: np.ndarray, water_frac: np.ndarray | None, valid: np.ndarray):
    """스트레치된 [0,1] 그레이 + (선택) 수체 오버레이 → uint8 RGB."""
    rgb = np.repeat((np.clip(gray01, 0, 1) * 255.0)[..., None], 3, axis=-1)
    if water_frac is not None:
        a = (WATER_ALPHA * np.clip(water_frac, 0, 1))[..., None]
        rgb = np.clip(rgb * (1.0 - a) + WATER_RGB[None, None, :] * a, 0, 255)
    out = rgb.astype(np.uint8)
    out[~valid] = BG_RGB
    return out


def stretch(gray_db: np.ndarray, valid: np.ndarray, p_lo=2.0, p_hi=98.0,
            ref: np.ndarray | None = None):
    """유효픽셀 percentile 스트레치. ref가 있으면 그 분포로 p2/p98 산정."""
    src = ref if ref is not None else gray_db[valid]
    if src.size == 0:
        return np.zeros_like(gray_db), (0.0, 1.0)
    lo, hi = np.percentile(src, [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((gray_db - lo) / (hi - lo), 0, 1), (float(lo), float(hi))


def atomic_save_png(rgb: np.ndarray, path: Path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(rgb).save(tmp, format="PNG")
    os.replace(tmp, path)


def atomic_save_json(obj: dict, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre", type=Path, default=DEF_PRE, help="정사 입력 γ0 dB GeoTIFF")
    ap.add_argument("--wb", type=Path, default=DEF_WB, help="② WB 마스크 GeoTIFF (0/1/255)")
    ap.add_argument("--out-dir", type=Path, default=DEF_OUT)
    ap.add_argument("--tag", default="20200302", help="자산 파일명 태그")
    ap.add_argument("--crop-size", type=int, default=4096,
                    help="관심영역 한 변(라벨 격자 px, crop-out의 배수)")
    ap.add_argument("--crop-out", type=int, default=1024, help="crop PNG 한 변(px)")
    ap.add_argument("--full-max", type=int, default=1200, help="전체 씬 PNG 긴 변 상한(px)")
    args = ap.parse_args()

    if not args.wb.exists():
        print(f"[skip] WB 마스크가 아직 없습니다: {args.wb}")
        print("       체인 러너 ②(detect_water) 완료 후 이 스크립트를 다시 실행하세요."
              " UI는 자산이 생길 때까지 기존 동작을 유지합니다.")
        return 0
    if not args.pre.exists():
        print(f"[error] Pre 입력이 없습니다: {args.pre}", file=sys.stderr)
        return 2
    if args.crop_size % args.crop_out != 0:
        print("[error] --crop-size는 --crop-out의 배수여야 합니다.", file=sys.stderr)
        return 2

    import rasterio
    from rasterio.windows import Window

    with rasterio.open(args.pre) as pre_ds, rasterio.open(args.wb) as wb_ds:
        if (pre_ds.width, pre_ds.height) != (wb_ds.width, wb_ds.height) or \
           pre_ds.crs != wb_ds.crs or \
           not np.allclose(tuple(pre_ds.transform)[:6], tuple(wb_ds.transform)[:6], atol=0.5):
            print("[error] Pre/WB 격자가 다릅니다 — 같은 라벨 격자 산출물이어야 합니다.",
                  file=sys.stderr)
            print(f"  pre: {pre_ds.width}x{pre_ds.height} {pre_ds.crs} {pre_ds.transform}",
                  file=sys.stderr)
            print(f"  wb : {wb_ds.width}x{wb_ds.height} {wb_ds.crs} {wb_ds.transform}",
                  file=sys.stderr)
            return 2

        H, W = pre_ds.height, pre_ds.width
        px_m = abs(pre_ds.transform.a)
        nodata = pre_ds.nodata if pre_ds.nodata is not None else -9999.0

        # ---- 1) 전체 씬 스트리밍 블록 통계 (축소 오버레이 + ROI 선택 재료) ----
        factor = math.ceil(max(H, W) / args.full_max)
        print(f"[info] 전체 씬 {W}x{H} → 1/{factor} 블록 집계 중...")
        sum_db, cnt_db, cnt_wat, cnt_wbv, tot_px = stream_scene(pre_ds, wb_ds, factor)

        gray_db = np.where(cnt_db > 0, sum_db / np.maximum(cnt_db, 1), np.nan)
        wfrac = cnt_wat / np.maximum(cnt_wbv, 1)
        valid_full = (cnt_wbv >= 0.5 * tot_px) & (cnt_db > 0)
        g01, (lo_f, hi_f) = stretch(gray_db, valid_full)
        full_rgb = render_rgb(np.nan_to_num(g01), wfrac, valid_full)

        scene_water = int(cnt_wat.sum())
        scene_valid = int(cnt_wbv.sum())

        # ---- 2) 관심영역 자동 선택 (라벨 격자 기준 정사각) ----
        S = min(args.crop_size, H, W)
        S -= S % args.crop_out  # crop-out 배수 유지
        r, c, k, roi_wf, roi_vf = pick_roi(cnt_wat, cnt_wbv, tot_px, math.ceil(S / factor))
        row0 = min(max(0, r * factor), H - S)
        col0 = min(max(0, c * factor), W - S)
        print(f"[info] 관심영역: offset=({col0},{row0}) size={S}px "
              f"(수체 {roi_wf * 100:.1f}% · 유효 {roi_vf * 100:.1f}%)")

        # ---- 3) 관심영역 full-res 읽기 → nan-aware 블록 평균 → PNG 쌍 ----
        win = Window(col0, row0, S, S)
        pre_c = pre_ds.read(1, window=win)
        wb_c = wb_ds.read(1, window=win)
        f2 = S // args.crop_out
        cw2 = args.crop_out
        s2, n2, w2, v2 = _block_stats(pre_c, wb_c, nodata, f2, cw2)
        gray_c = np.where(n2 > 0, s2 / np.maximum(n2, 1), np.nan)
        wfrac_c = w2 / np.maximum(v2, 1)
        tot2 = f2 * f2
        valid_c = (v2 >= 0.5 * tot2) & (n2 > 0)
        # 스트레치는 full-res 유효픽셀 분포 기준 (블록평균보다 대비 정확)
        ref = pre_c[np.isfinite(pre_c) & (pre_c != nodata)]
        g01_c, (lo_c, hi_c) = stretch(gray_c, valid_c, ref=ref)
        g01_c = np.nan_to_num(g01_c)
        pre_valid_c = n2 > 0
        crop_pre_rgb = render_rgb(g01_c, None, pre_valid_c)
        crop_ovl_rgb = render_rgb(g01_c, wfrac_c, valid_c)
        crop_water = int(w2.sum())
        crop_valid = int(v2.sum())
        bounds = rasterio.windows.bounds(win, pre_ds.transform)

        # ---- 4) 원자적 저장 ----
        out = args.out_dir
        p_pre = out / f"wbms_{args.tag}_pre_crop.png"
        p_ovl = out / f"wbms_{args.tag}_overlay_crop.png"
        p_full = out / f"wbms_{args.tag}_overlay_full.png"
        p_meta = out / f"wbms_{args.tag}_meta.json"
        atomic_save_png(crop_pre_rgb, p_pre)
        atomic_save_png(crop_ovl_rgb, p_ovl)
        atomic_save_png(full_rgb, p_full)
        meta = {
            "tag": args.tag,
            "generated_utc": dt.datetime.now(dt.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": {"pre": str(args.pre), "wb": str(args.wb)},
            "grid": {"width": W, "height": H, "crs": str(pre_ds.crs),
                     "pixel_m": px_m},
            "scene_water": {
                "water_px": scene_water, "valid_px": scene_valid,
                "water_pct_of_valid": round(100.0 * scene_water / max(scene_valid, 1), 2),
            },
            "crop": {
                "col_off": int(col0), "row_off": int(row0), "size_px": int(S),
                "size_km": round(S * px_m / 1000.0, 2),
                "utm_bounds": [round(v, 1) for v in bounds],
                "water_pct_of_valid": round(100.0 * crop_water / max(crop_valid, 1), 2),
                "stretch_db": [round(lo_c, 2), round(hi_c, 2)],
            },
            "full": {"png_size": [int(full_rgb.shape[1]), int(full_rgb.shape[0])],
                     "stretch_db": [round(lo_f, 2), round(hi_f, 2)]},
            "model": {"file": "WBMS_SAR_ICEYE.h5", "arch": "U-Net",
                      "protocol_iou": 0.8786,
                      "protocol_note": "배포 프로토콜, 20200416 시험씬 기준",
                      "device": "CPU"},
            "note": "GT 대조 수치 없음 — 실모델 산출 표기 전용",
        }
        atomic_save_json(meta, p_meta)

    for p in (p_pre, p_ovl, p_full, p_meta):
        print(f"[ok] {p} ({p.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
