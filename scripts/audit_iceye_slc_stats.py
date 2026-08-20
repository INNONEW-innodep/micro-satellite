#!/usr/bin/env python3
"""ICEYE SLC 도메인 갭·전처리 스펙 검증 통계 스크립트.

ICEYE-X5 SLC(H5)의 s_i/s_q를 행 블록 스트리밍으로 읽어 sigma0(dB) 분포,
스펙클(ENL), 저후방산란(수체 프록시) 분리도 통계를 산출한다.
1_water_body_detection Module0(TSX ASC/DSC min-max 정규화) 대비
ICEYE 정규화 '잠정 사양'의 근거 수치를 만드는 것이 목적.

계산 규약(검증 리포트에서 확정):
- sigma0 = calibration_factor * (s_i^2 + s_q^2)  [선형 파워]
- 진폭 0 픽셀은 log 변환 전 제외, 제외 비율 기록
- '선형 파워 평균의 dB 변환'과 '픽셀별 dB의 평균'을 모두 산출(키 이름으로 구분)
- 통계는 전장면 기준. 출력 히스토그램은 dB -35~+25, 0.5dB 빈
- ENL = mean^2/var 를 512x512 타일 중 변동계수(CV) 하위 5% 타일 평균으로 추정
  (싱글룩 원본 vs 2x2 블록평균 멀티룩 비교)
- 수체/육지 분리도: slant-range 기하라 지리 지정 불가 → dB 히스토그램 Otsu
  임계값 기반 저산란 클래스('수체 프록시')의 비율/평균/표준편차/Fisher ratio

출력:
- data/audit/iceye_slc_stats_20200302.json
- data/audit/iceye_slc_stats_20200302.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5 = REPO_ROOT / "ICEYE_X5_SLC_SM_23467_20200302T183857-008.h5"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "audit"

BLOCK_ROWS = 2048          # 행 블록 스트리밍 크기
TILE = 512                 # ENL 균질영역 추정용 타일 한 변(싱글룩)
CV_LOW_FRAC = 0.05         # 변동계수 하위 5% 타일을 균질 영역으로 간주

# 세밀 히스토그램(퍼센타일/Otsu용): 0.01 dB 빈.
# 최소 비영 파워=1 → 10*log10(cal) ≈ -27.71 dB, 최대 ≈ 65.6 dB이므로 [-30, 70)이면 전 범위 포함.
FINE_LO, FINE_HI, FINE_STEP = -30.0, 70.0, 0.01
# 출력용 히스토그램: -35~+25 dB, 0.5 dB 빈 (규약)
COARSE_LO, COARSE_HI, COARSE_STEP = -35.0, 25.0, 0.5

PCTS = [1, 2, 5, 25, 50, 75, 95, 98, 99]


def log(msg: str) -> None:
    print(f"[audit_iceye] {msg}", flush=True)


def percentiles_from_hist(counts: np.ndarray, edges: np.ndarray, pcts) -> dict:
    """균일 빈 히스토그램에서 선형 보간 퍼센타일."""
    total = counts.sum()
    cum = np.cumsum(counts)
    out = {}
    for p in pcts:
        target = total * (p / 100.0)
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = min(idx, len(counts) - 1)
        prev = cum[idx - 1] if idx > 0 else 0
        in_bin = counts[idx]
        frac = (target - prev) / in_bin if in_bin > 0 else 0.5
        out[f"p{p}"] = float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))
    return out


def otsu_from_hist(counts: np.ndarray, edges: np.ndarray) -> dict:
    """히스토그램 기반 Otsu 임계값 + 클래스 통계(저산란=class0)."""
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = counts.astype(np.float64)
    total = p.sum()
    p /= total
    w0 = np.cumsum(p)
    mu_cum = np.cumsum(p * centers)
    mu_total = mu_cum[-1]
    valid = (w0 > 1e-12) & (w0 < 1.0 - 1e-12)
    sigma_b = np.full_like(w0, -1.0)
    sigma_b[valid] = (mu_total * w0[valid] - mu_cum[valid]) ** 2 / (
        w0[valid] * (1.0 - w0[valid])
    )
    t_idx = int(np.argmax(sigma_b))
    threshold = float(edges[t_idx + 1])  # class0: bins [0..t_idx]

    def class_stats(sel_counts, sel_centers):
        n = sel_counts.sum()
        if n == 0:
            return 0.0, float("nan"), float("nan")
        mean = float((sel_counts * sel_centers).sum() / n)
        var = float((sel_counts * (sel_centers - mean) ** 2).sum() / n)
        return float(n / total), mean, float(np.sqrt(var))

    frac0, mu0, sd0 = class_stats(counts[: t_idx + 1], centers[: t_idx + 1])
    frac1, mu1, sd1 = class_stats(counts[t_idx + 1 :], centers[t_idx + 1 :])
    fisher = float((mu0 - mu1) ** 2 / (sd0 ** 2 + sd1 ** 2))
    return {
        "otsu_threshold_db": threshold,
        "note": "slant-range 기하로 지리 지정 불가 → 저후방산란 클래스를 '수체 프록시'로 사용 (실제 수체와 다를 수 있음)",
        "water_proxy_low_backscatter": {
            "fraction_of_valid_pixels": frac0,
            "mean_db": mu0,
            "std_db": sd0,
        },
        "land_proxy_high_backscatter": {
            "fraction_of_valid_pixels": frac1,
            "mean_db": mu1,
            "std_db": sd1,
        },
        "fisher_ratio": fisher,
    }


def enl_from_tiles(tile_sum, tile_sumsq, n_per_tile, low_frac):
    """타일별 mean/var → CV 하위 low_frac 타일의 ENL(mean^2/var) 통계."""
    mean = tile_sum / n_per_tile
    var = tile_sumsq / n_per_tile - mean ** 2
    ok = (mean > 0) & (var > 0)
    mean, var = mean[ok], var[ok]
    cv = np.sqrt(var) / mean
    n_sel = max(1, int(np.floor(cv.size * low_frac)))
    sel = np.argsort(cv)[:n_sel]
    enl_tiles = mean[sel] ** 2 / var[sel]
    return {
        "n_tiles_total": int(cv.size),
        "n_tiles_selected_low_cv": int(n_sel),
        "cv_selection_percentile": low_frac * 100.0,
        "enl_mean_of_selected_tiles": float(enl_tiles.mean()),
        "enl_median_of_selected_tiles": float(np.median(enl_tiles)),
        "enl_fullscene_reference": float(
            (tile_sum.sum() / (n_per_tile * tile_sum.size)) ** 2
            / (
                tile_sumsq.sum() / (n_per_tile * tile_sum.size)
                - (tile_sum.sum() / (n_per_tile * tile_sum.size)) ** 2
            )
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--block-rows", type=int, default=BLOCK_ROWS)
    args = ap.parse_args()

    t_start = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    log(f"open {args.h5}")
    with h5py.File(args.h5, "r") as f:
        cal = float(f["calibration_factor"][()])
        dsi, dsq = f["s_i"], f["s_q"]
        n_rows, n_cols = dsi.shape

        def scalar_str(name):
            v = f[name][()]
            return v.decode() if isinstance(v, bytes) else str(v)

        meta = {
            "product_name": scalar_str("product_name"),
            "satellite_name": scalar_str("satellite_name"),
            "product_level": scalar_str("product_level"),
            "acquisition_mode": scalar_str("acquisition_mode"),
            "acquisition_start_utc": scalar_str("acquisition_start_utc"),
            "orbit_direction": scalar_str("orbit_direction"),
            "polarization": scalar_str("polarization"),
            "look_side": scalar_str("look_side"),
            "sample_precision": scalar_str("sample_precision"),
            "azimuth_looks": int(f["azimuth_looks"][()]),
            "range_looks": int(f["range_looks"][()]),
            "calibration_factor": cal,
            "shape_rows_cols": [int(n_rows), int(n_cols)],
            "slant_range_spacing_m": float(f["slant_range_spacing"][()]),
            "azimuth_ground_spacing_m": float(f["azimuth_ground_spacing"][()]),
            "local_incidence_angle_deg_min": float(np.min(f["local_incidence_angle"][:])),
            "local_incidence_angle_deg_max": float(np.max(f["local_incidence_angle"][:])),
        }
        log(f"scene {n_rows}x{n_cols}, cal={cal:.9g}, "
            f"{meta['orbit_direction']}/{meta['polarization']}")

        # --- 누산기 준비 ---
        fine_edges = np.arange(FINE_LO, FINE_HI + FINE_STEP / 2, FINE_STEP)
        fine_counts = np.zeros(len(fine_edges) - 1, dtype=np.int64)

        n_total = n_rows * n_cols
        n_zero = 0
        sum_lin = 0.0            # 선형 sigma0 합 (0 픽셀은 0 기여)
        sum_db = 0.0             # 픽셀별 dB 합 (비영 픽셀만)
        sum_db2 = 0.0
        db_min, db_max = np.inf, -np.inf

        # ENL 타일 (싱글룩 512x512 / 2x2 멀티룩 후 256x256, 동일 타일 격자)
        n_tr, n_tc = n_rows // TILE, n_cols // TILE
        H_t, W_t = n_tr * TILE, n_tc * TILE
        tile_sum = np.zeros((n_tr, n_tc))
        tile_sumsq = np.zeros((n_tr, n_tc))
        mtile_sum = np.zeros((n_tr, n_tc))
        mtile_sumsq = np.zeros((n_tr, n_tc))
        mt = TILE // 2

        block = args.block_rows
        assert block % TILE == 0, "block-rows는 512의 배수여야 타일 정렬이 유지됨"
        n_blocks = (n_rows + block - 1) // block

        for bi, r0 in enumerate(range(0, n_rows, block)):
            r1 = min(r0 + block, n_rows)
            si = dsi[r0:r1].astype(np.float64)
            sq = dsq[r0:r1].astype(np.float64)
            power = si * si + sq * sq          # int16^2 합은 float64에서 정확
            del si, sq
            sigma0 = cal * power

            zero_mask = power == 0
            n_zero += int(zero_mask.sum())
            sum_lin += float(sigma0.sum())

            db = 10.0 * np.log10(sigma0[~zero_mask])
            del power
            sum_db += float(db.sum())
            sum_db2 += float((db * db).sum())
            if db.size:
                db_min = min(db_min, float(db.min()))
                db_max = max(db_max, float(db.max()))
            fine_counts += np.histogram(db, bins=fine_edges)[0]
            del db

            # 타일 누산 (블록 시작이 512 배수 → 타일 경계 정렬)
            n_seg = max(0, (min(r1, H_t) - r0)) // TILE
            if n_seg > 0:
                tr0 = r0 // TILE
                part = sigma0[: n_seg * TILE, :W_t]
                seg = part.reshape(n_seg, TILE, n_tc, TILE)
                tile_sum[tr0 : tr0 + n_seg] += seg.sum(axis=(1, 3))
                tile_sumsq[tr0 : tr0 + n_seg] += (seg * seg).sum(axis=(1, 3))
                # 2x2 블록평균 멀티룩
                ml = part.reshape(n_seg * mt, 2, W_t // 2, 2).mean(axis=(1, 3))
                mseg = ml.reshape(n_seg, mt, n_tc, mt)
                mtile_sum[tr0 : tr0 + n_seg] += mseg.sum(axis=(1, 3))
                mtile_sumsq[tr0 : tr0 + n_seg] += (mseg * mseg).sum(axis=(1, 3))
                del part, seg, ml, mseg
            del sigma0

            log(f"block {bi + 1}/{n_blocks} rows[{r0}:{r1}] done "
                f"({time.time() - t_start:.1f}s elapsed)")

        # --- 집계 ---
        n_valid = n_total - n_zero
        mean_lin = sum_lin / n_valid           # 비영 픽셀 기준 선형 평균
        mean_db_of_linear_mean = 10.0 * np.log10(mean_lin)
        mean_of_pixel_db = sum_db / n_valid
        std_of_pixel_db = float(np.sqrt(max(0.0, sum_db2 / n_valid - mean_of_pixel_db ** 2)))

        pct = percentiles_from_hist(fine_counts, fine_edges, PCTS)
        otsu = otsu_from_hist(fine_counts, fine_edges)

        # 출력용 0.5dB 히스토그램: 세밀 빈(0.01)을 50개씩 합산 (경계 정렬됨)
        coarse_edges = np.arange(COARSE_LO, COARSE_HI + COARSE_STEP / 2, COARSE_STEP)
        n_coarse = len(coarse_edges) - 1
        coarse_counts = np.zeros(n_coarse, dtype=np.int64)
        ratio = round(COARSE_STEP / FINE_STEP)
        for ci in range(n_coarse):
            lo, hi = coarse_edges[ci], coarse_edges[ci + 1]
            if hi <= FINE_LO or lo >= FINE_HI:
                continue
            fi0 = int(round((lo - FINE_LO) / FINE_STEP))
            coarse_counts[ci] = fine_counts[max(0, fi0) : fi0 + ratio].sum()
        below = int(fine_counts[: max(0, int(round((COARSE_LO - FINE_LO) / FINE_STEP)))].sum())
        above = int(fine_counts[int(round((COARSE_HI - FINE_LO) / FINE_STEP)) :].sum())

        enl_single = enl_from_tiles(tile_sum, tile_sumsq, TILE * TILE, CV_LOW_FRAC)
        enl_multi = enl_from_tiles(mtile_sum, mtile_sumsq, mt * mt, CV_LOW_FRAC)

    elapsed = time.time() - t_start
    stats = {
        "provisional": True,
        "purpose": "ICEYE 정규화 '잠정 사양' 근거 수치 (TSX Module0 min-max 정규화 대비 도메인 갭 검증)",
        "limitations": [
            "단일 장면(2020-03-02)·단일 날짜 통계 — 시계열 변동 미반영",
            "DESCENDING 궤도만 존재 — ASC 통계 산출 불가 (Module0의 ASC/DSC별 min-max 전제와 비대칭)",
            "VV 단일 편파",
            "slant-range 기하(비지오코딩) — 수체/육지의 지리적 판별 불가, 저후방산란 Otsu 클래스를 '수체 프록시'로만 사용",
            "ENL은 저CV 타일 기반 균질영역 근사 추정치",
        ],
        "input_file": str(args.h5),
        "metadata": meta,
        "convention": {
            "sigma0_definition": "sigma0 = calibration_factor * (s_i^2 + s_q^2)",
            "db_definition": "10*log10(sigma0), 진폭 0 픽셀은 log 변환 전 제외",
            "percentile_estimation": f"{FINE_STEP} dB 빈 히스토그램 선형 보간",
        },
        "pixel_counts": {
            "n_total": int(n_total),
            "n_zero_amplitude_excluded": int(n_zero),
            "zero_amplitude_fraction_percent": 100.0 * n_zero / n_total,
            "n_valid": int(n_valid),
        },
        "sigma0_db_stats": {
            "mean_sigma0_linear": mean_lin,
            "mean_db_of_linear_mean": float(mean_db_of_linear_mean),
            "mean_of_pixel_db": float(mean_of_pixel_db),
            "std_of_pixel_db": std_of_pixel_db,
            "min_db": db_min,
            "max_db": db_max,
            "percentiles_db": pct,
            "note": "mean_db_of_linear_mean(선형 파워 평균의 dB 변환)과 mean_of_pixel_db(픽셀별 dB의 산술평균)는 정의가 다른 값임",
        },
        "histogram_db": {
            "range_db": [COARSE_LO, COARSE_HI],
            "bin_width_db": COARSE_STEP,
            "bin_edges_db": [round(float(e), 3) for e in coarse_edges],
            "counts": coarse_counts.tolist(),
            "counts_below_range": below,
            "counts_above_range": above,
        },
        "enl": {
            "definition": "ENL = mean(sigma0_linear)^2 / var(sigma0_linear), 512x512 타일 중 CV 하위 5% 타일(균질영역 근사)의 평균",
            "single_look": enl_single,
            "multilook_2x2_boxcar": enl_multi,
        },
        "water_land_separability_proxy": otsu,
        "runtime_sec": round(elapsed, 1),
    }

    json_path = args.outdir / "iceye_slc_stats_20200302.json"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {json_path}")

    md_path = args.outdir / "iceye_slc_stats_20200302.md"
    md_path.write_text(build_markdown(stats), encoding="utf-8")
    log(f"wrote {md_path}")
    log(f"total elapsed {elapsed:.1f}s")


def build_markdown(s: dict) -> str:
    m = s["metadata"]
    d = s["sigma0_db_stats"]
    p = d["percentiles_db"]
    z = s["pixel_counts"]
    e1 = s["enl"]["single_look"]
    e2 = s["enl"]["multilook_2x2_boxcar"]
    o = s["water_land_separability_proxy"]
    w = o["water_proxy_low_backscatter"]
    l = o["land_proxy_high_backscatter"]
    lim = "\n".join(f"- {x}" for x in s["limitations"])
    return f"""# ICEYE SLC sigma0 통계 감사 (잠정 사양 근거) — {m['product_name']}

> **provisional: true** — 단일 장면 기반 잠정 수치. 아래 '한계' 참조.
> 생성: `scripts/audit_iceye_slc_stats.py`, 수치 전체: `iceye_slc_stats_20200302.json`

## 장면 개요

| 항목 | 값 |
|---|---|
| 위성/모드 | {m['satellite_name']} / {m['acquisition_mode']} ({m['product_level']}) |
| 취득 시각 (UTC) | {m['acquisition_start_utc']} |
| 궤도 / 편파 | **{m['orbit_direction']}** / **{m['polarization']}** |
| 크기 (az × rg) | {m['shape_rows_cols'][0]} × {m['shape_rows_cols'][1]}, int16 I/Q |
| 룩 수 (az × rg) | {m['azimuth_looks']} × {m['range_looks']} (싱글룩 SLC) |
| calibration_factor | {m['calibration_factor']:.9g} |
| 국지 입사각 | {m['local_incidence_angle_deg_min']:.2f}° ~ {m['local_incidence_angle_deg_max']:.2f}° |

## sigma0 dB 분포 (전장면, 진폭 0 픽셀 제외)

| 항목 | 값 |
|---|---|
| 진폭 0 픽셀 제외 비율 | {z['zero_amplitude_fraction_percent']:.4f}% ({z['n_zero_amplitude_excluded']:,} / {z['n_total']:,}) |
| **선형 파워 평균의 dB 변환** (mean_db_of_linear_mean) | **{d['mean_db_of_linear_mean']:.2f} dB** |
| **픽셀별 dB의 평균** (mean_of_pixel_db) | **{d['mean_of_pixel_db']:.2f} dB** (std {d['std_of_pixel_db']:.2f} dB) |
| min / max | {d['min_db']:.2f} / {d['max_db']:.2f} dB |

두 평균은 정의가 다른 값이다(Jensen 부등식): 선형 평균은 고산란 스펙클 꼬리에 지배되고, 픽셀별 dB 평균은 분포 중심을 반영한다. 정규화 상수로는 반드시 어느 쪽인지 명시해 사용할 것.

### 퍼센타일 (dB)

| p1 | p2 | p5 | p25 | p50 | p75 | p95 | p98 | p99 |
|---|---|---|---|---|---|---|---|---|
| {p['p1']:.2f} | {p['p2']:.2f} | {p['p5']:.2f} | {p['p25']:.2f} | {p['p50']:.2f} | {p['p75']:.2f} | {p['p95']:.2f} | {p['p98']:.2f} | {p['p99']:.2f} |

## 스펙클 — ENL (CV 하위 {e1['cv_selection_percentile']:.0f}% 타일 평균)

| 처리 | 선택 타일 | ENL(평균) | ENL(중앙값) | 전장면 참고치 |
|---|---|---|---|---|
| 싱글룩 원본 | {e1['n_tiles_selected_low_cv']}/{e1['n_tiles_total']} | {e1['enl_mean_of_selected_tiles']:.2f} | {e1['enl_median_of_selected_tiles']:.2f} | {e1['enl_fullscene_reference']:.3f} |
| 2×2 블록평균 멀티룩 | {e2['n_tiles_selected_low_cv']}/{e2['n_tiles_total']} | {e2['enl_mean_of_selected_tiles']:.2f} | {e2['enl_median_of_selected_tiles']:.2f} | {e2['enl_fullscene_reference']:.3f} |

## 수체/육지 분리도 — 수체 프록시(저후방산란, Otsu)

slant-range 기하라 지리 지정이 안 되므로 **저후방산란 클래스를 '수체 프록시'로만** 해석한다.

| 항목 | 값 |
|---|---|
| Otsu 임계값 | {o['otsu_threshold_db']:.2f} dB |
| 수체 프록시 비율 | {100 * w['fraction_of_valid_pixels']:.2f}% |
| 수체 프록시 평균/표준편차 | {w['mean_db']:.2f} / {w['std_db']:.2f} dB |
| 육지 프록시 평균/표준편차 | {l['mean_db']:.2f} / {l['std_db']:.2f} dB |
| Fisher ratio | {o['fisher_ratio']:.3f} |

## TSX Module0 min-max 정규화 대비 시사점

Module0(`1_water_body_detection/Module0/Module0_ImportingData.md`)은 TSX Float32 GeoTIFF에 대해
궤도(ASC/DSC)별 전체 min-max를 동적으로 구해 `[0,1]` 정규화하고, 결측값 `-9999`를 제외/KNN 보간하는 전제다.
이 ICEYE 장면과의 대비점:

1. **min-max를 선형 sigma0에 그대로 쓰면 붕괴** — 선형 최대치가 {d['max_db']:.1f} dB(≈{10 ** (d['max_db'] / 10):,.0f})로
   p99({p['p99']:.2f} dB) 대비 수십 dB 위의 극단 스펙클 꼬리에 min-max가 지배되어, 유효 다이나믹레인지
   (p2~p98: {p['p2']:.2f}~{p['p98']:.2f} dB)가 `[0,1]`의 극히 일부로 압착된다. **dB 변환 후 퍼센타일 클리핑
   (예: p2~p98) 정규화**를 잠정 사양으로 제안.
2. **결측 표현이 다름** — TSX의 `-9999` 대신 ICEYE SLC는 진폭 0 픽셀({z['zero_amplitude_fraction_percent']:.3f}%)이
   결측/무효에 해당. log 변환 전 마스킹 필요(KNN 보간 전제도 재검토 대상).
3. **스펙클 수준이 다름** — 싱글룩 SLC(ENL≈{e1['enl_mean_of_selected_tiles']:.1f})는 TSX 학습 데이터
   (오소보정·멀티룩 전처리본)보다 스펙클이 강함. 2×2 멀티룩만으로 ENL {e2['enl_mean_of_selected_tiles']:.1f}로
   개선되므로 Module0 투입 전 멀티룩(또는 스펙클 필터) 단계를 잠정 사양에 포함.
4. **ASC/DSC별 상수 산출 불가** — 본 장면은 DESCENDING 단일이므로 Module0의 궤도별 min-max 전제 중
   ASC 쪽 상수를 만들 수 없음. DSC 잠정 상수로만 기록하고 ASC 장면 확보 시 갱신 필요.
5. **두 종류 평균의 구분** — 선형 평균의 dB({d['mean_db_of_linear_mean']:.2f} dB)와 픽셀 dB 평균
   ({d['mean_of_pixel_db']:.2f} dB)의 차이가 약 {d['mean_db_of_linear_mean'] - d['mean_of_pixel_db']:.1f} dB.
   TSX 쪽 통계와 비교할 때 동일 정의끼리 비교해야 함.
6. **분리도 근거** — Otsu {o['otsu_threshold_db']:.2f} dB 기준 저산란(수체 프록시) {100 * w['fraction_of_valid_pixels']:.1f}%,
   Fisher ratio {o['fisher_ratio']:.2f}. 기하보정 전 원시 dB 분포에서도 저산란 클래스가 분리 가능함을 시사하나,
   지리 검증 전까지는 프록시 수치로만 취급.

## 한계

{lim}
"""


if __name__ == "__main__":
    main()
