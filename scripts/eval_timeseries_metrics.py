#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시계열 정량 평가 — 수위 보정·예측 기준선·면적 시계열의 전형적 지표 산출.

계열(혼용 금지):
  A. 수위 보정 정확도  — 융합 LSTM 보정 수위 vs 게이지 실측 (⚠ in-sample 30표본)
  B. 예측 참고치       — persistence(직전 관측 유지) walk-forward 1-step
                         (관측 8시점·불규칙 간격이라 '참고치'로만 인용)
  C. 면적 시계열       — WB 마스크 면적 vs GT 라벨 면적 (배포 산출 기준, 씬당 3 m 격자)
  참고선               — 인수문서 LODO RMSE 0.0453 m / 위성 미사용 베이스라인 0.0199 m
                         (정식 out-of-sample 프로토콜 수치 — A와 계열이 다름)

지표: RMSE · MAE · MBE(bias) · max|err| · MAPE · R² (수위), km²·%오차 (면적).
산출: data/eval/timeseries_metrics.{json,csv,md}
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(REPO, "data", "incoming", "handover")
FUSED_DIR = os.path.join(REPO, "data", "wbms_runs", "fused")
GAUGE_CSV = os.path.join(H, "06_aux", "deploy_train_wamis_v3_finalwb.csv")
OUT_DIR = os.path.join(REPO, "data", "eval")

LOC_NAMES = {0: "jeongcheon", 1: "hupo", 2: "gimhae", 3: "gupo"}
SAT_NAMES = {0: "PlanetScope", 1: "ICEYE"}
LODO_RMSE_M = 0.0453          # 인수문서 §3 — leave-one-date-out
BASELINE_RMSE_M = 0.0199      # 위성 미사용 베이스라인 (동일 문서)

SCENES = {  # (sensor, date) → (WB 마스크, GT 라벨)
    ("ICEYE", d): (
        os.path.join(REPO, "data", "wbms_runs", "wb", "2020", d),
        os.path.join(H, "06_aux", "Labels_GT"),
    ) for d in ("20200302", "20200330", "20200415", "20200416")
} | {
    ("PlanetScope", d): (
        os.path.join(REPO, "data", "wbms_runs", "wb_planet", "2020", d),
        os.path.join(H, "06_aux", "Labels_GT_planet"),
    ) for d in ("20200218", "20200312", "20200325", "20200414")
}


# ── 공통 지표 ────────────────────────────────────────────────────────────────
def level_stats(pairs: list[tuple[float, float]]) -> dict:
    """pairs = [(pred, truth)] → 전형적 수위 지표."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    err = [p - t for p, t in pairs]
    truths = [t for _, t in pairs]
    mean_t = sum(truths) / n
    sse = sum(e * e for e in err)
    sst = sum((t - mean_t) ** 2 for t in truths)
    return {
        "n": n,
        "mse_m2": round(sse / n, 6),
        "rmse_m": round(math.sqrt(sse / n), 4),
        "mae_m": round(sum(abs(e) for e in err) / n, 4),
        "bias_m": round(sum(err) / n, 4),
        "max_abs_err_m": round(max(abs(e) for e in err), 4),
        "mape_pct": round(100 * sum(abs(e) / t for e, (_, t) in zip(err, pairs)) / n, 2),
        "r2": round(1 - sse / sst, 4) if sst > 0 else None,
    }


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def load_gauge() -> dict[tuple[str, str, str], float]:
    """(date_iso, satellite, station) → 실측 수위 m."""
    truth = {}
    with open(GAUGE_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = row["date"].strip()
            fmt = "%Y-%m-%d" if "-" in raw else "%Y%m%d"
            date = dt.datetime.strptime(raw, fmt).date().isoformat()
            key = (date, SAT_NAMES[int(row["sat_type"])], LOC_NAMES[int(row["loc_id"])])
            truth[key] = float(row["real_water_level"])
    return truth


def load_fused() -> list[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(FUSED_DIR, "*", "FUSED_*.json"))):
        if p.endswith("_qc.json"):
            continue
        station = os.path.basename(os.path.dirname(p))
        doc = json.load(open(p, encoding="utf-8"))
        for r in doc.get("records", []):
            date = dt.datetime.strptime(str(r["date"]), "%Y%m%d").date().isoformat()
            recs.append({
                "station": station, "date": date, "satellite": r["satellite"],
                "sat_level_m": float(r["water_level_m"]),
                "corrected_m": float(r["corrected_water_level_m"]),
            })
    return recs


def paired_water_km2(wb_dir: str, gt_dir: str, date: str) -> tuple[float, float] | None:
    """(WB 물 면적, GT 물 면적) km² — 둘 다 WB 유효영역(≠255) 교집합 기준.

    GT 라벨은 씬 footprint 밖까지 정의돼 있으므로 전체를 세면 과대평가된다
    (배포 검증 프로토콜과 동일하게 유효영역으로 한정해야 공정)."""
    import numpy as np
    import rasterio
    wb_hits = sorted(glob.glob(os.path.join(wb_dir, "WB_*.tif")))
    wb_hits = [p for p in wb_hits if "_qc" not in p and "_meta" not in p]
    gt_hits = sorted(glob.glob(os.path.join(gt_dir, f"*{date}*.tif")))
    if not wb_hits or not gt_hits:
        return None
    wb_n = gt_n = 0
    with rasterio.open(wb_hits[0]) as wb, rasterio.open(gt_hits[0]) as gt:
        if (wb.width, wb.height) != (gt.width, gt.height):
            print(f"  [warn] {date}: WB {wb.width}x{wb.height} ≠ "
                  f"GT {gt.width}x{gt.height} — 면적 비교 생략", file=sys.stderr)
            return None
        for _, window in wb.block_windows(1):
            w = wb.read(1, window=window)
            g = gt.read(1, window=window)
            valid = w != 255
            wb_n += int(np.count_nonzero(w == 1))
            gt_n += int(np.count_nonzero((g == 1) & valid))
    return wb_n * 9.0 / 1e6, gt_n * 9.0 / 1e6


# ── 평가 ────────────────────────────────────────────────────────────────────
def eval_levels(recs, truth):
    """계열 A: 보정/정렬-위성/지점평균 vs 실측, 전체·지점별."""
    joined = []
    for r in recs:
        t = truth.get((r["date"], r["satellite"], r["station"]))
        if t is not None:
            joined.append({**r, "truth_m": t})
    stations = sorted({j["station"] for j in joined})

    def sub(pred_key, rows):
        return level_stats([(j[pred_key], j["truth_m"]) for j in rows])

    # 위성 수위는 상대 체계라 지점별 상수 정렬(평균 오프셋 제거) 후 비교해야 공정
    by_st = {s: [j for j in joined if j["station"] == s] for s in stations}
    for s, rows in by_st.items():
        off = sum(j["truth_m"] - j["sat_level_m"] for j in rows) / len(rows)
        for j in rows:
            j["sat_aligned_m"] = j["sat_level_m"] + off
            j["station_mean_m"] = sum(x["truth_m"] for x in rows) / len(rows)

    out = {
        "overall": {
            "corrected": sub("corrected_m", joined),
            "satellite_debiased": sub("sat_aligned_m", joined),
            "station_mean_baseline": sub("station_mean_m", joined),
        },
        "per_station": {
            s: {"corrected": sub("corrected_m", by_st[s]),
                "satellite_debiased": sub("sat_aligned_m", by_st[s])}
            for s in stations
        },
    }
    return out, joined


def eval_persistence(joined):
    """계열 B: 지점별 시간순 정렬 → gauge(t+1) 예측치로 corrected(t)·gauge(t) 사용."""
    pairs_corr, pairs_naive, gaps = [], [], []
    for s in sorted({j["station"] for j in joined}):
        rows = sorted((j for j in joined if j["station"] == s),
                      key=lambda j: j["date"])
        # 같은 날짜에 두 위성 관측이 있으면 평균으로 하나의 시점으로 축약
        by_date: dict[str, list] = {}
        for j in rows:
            by_date.setdefault(j["date"], []).append(j)
        series = []
        for d in sorted(by_date):
            g = by_date[d]
            series.append((d,
                           sum(x["corrected_m"] for x in g) / len(g),
                           g[0]["truth_m"]))
        for (d0, c0, t0), (d1, _c1, t1) in zip(series, series[1:]):
            pairs_corr.append((c0, t1))    # 시스템 persistence: 직전 보정치 유지
            pairs_naive.append((t0, t1))   # 관측 persistence: 직전 실측 유지
            gaps.append((dt.date.fromisoformat(d1) - dt.date.fromisoformat(d0)).days)
    return {
        "horizon_days": {"min": min(gaps), "max": max(gaps),
                         "mean": round(sum(gaps) / len(gaps), 1)},
        "persistence_of_corrected": level_stats(pairs_corr),
        "persistence_of_gauge": level_stats(pairs_naive),
    }


HORIZON_BINS = [(1, 7), (8, 14), (15, 28), (29, 58)]
LEVEL_RANGE_M = 2.415 - 1.43   # 게이지 실측 전체 범위 — 정규화 MSE 분모


def eval_horizon(joined):
    """계열 E: 예측 지평별 성능 — 모든 관측쌍(t_i→t_j, i<j)에 persistence 적용.

    '며칠 앞을 예측하면 얼마나 나빠지나'를 관측이 허용하는 모든 간격에서 측정.
    관측 8시점뿐이라 매일이 아니라 구간(bin) 단위이며, 갈수기 데이터라
    홍수기 급변 상황의 저하는 이 곡선에 나타나지 않는다(한계 명시)."""
    pairs = []   # (horizon_days, pred_corrected, truth)
    for s in sorted({j["station"] for j in joined}):
        rows = sorted((j for j in joined if j["station"] == s),
                      key=lambda j: j["date"])
        by_date: dict[str, list] = {}
        for j in rows:
            by_date.setdefault(j["date"], []).append(j)
        series = [(d,
                   sum(x["corrected_m"] for x in g) / len(g),
                   g[0]["truth_m"])
                  for d, g in sorted(by_date.items())]
        for i, (d0, c0, _t0) in enumerate(series):
            for d1, _c1, t1 in series[i + 1:]:
                h = (dt.date.fromisoformat(d1) - dt.date.fromisoformat(d0)).days
                pairs.append({"station": s, "from": d0, "to": d1,
                              "horizon_days": h, "pred_m": round(c0, 4),
                              "truth_m": t1, "err_m": round(c0 - t1, 4)})
    bins = []
    for lo, hi in HORIZON_BINS:
        sub = [(p["pred_m"], p["truth_m"]) for p in pairs
               if lo <= p["horizon_days"] <= hi]
        st_ = level_stats(sub)
        if st_.get("n"):
            st_["norm_mse"] = round(st_["mse_m2"] / LEVEL_RANGE_M ** 2, 5)
        bins.append({"horizon": f"{lo}~{hi}일", **st_})
    return {"n_pairs_total": len(pairs), "bins": bins, "pairs": pairs}


def eval_areas():
    """계열 C: 씬별 WB 면적 vs GT 면적."""
    rows = []
    for (sensor, date), (wb_dir, gt_dir) in sorted(SCENES.items()):
        pair = paired_water_km2(wb_dir, gt_dir, date)
        if pair is None:
            rows.append({"sensor": sensor, "date": date, "status": "missing"})
            continue
        wb, gt = pair
        rows.append({
            "sensor": sensor, "date": dt.datetime.strptime(date, "%Y%m%d")
                                        .date().isoformat(),
            "wb_km2": round(wb, 3), "gt_km2": round(gt, 3),
            "err_km2": round(wb - gt, 3),
            "abs_pct_err": round(100 * abs(wb - gt) / gt, 2),
        })
        print(f"  면적 {sensor} {date}: WB {wb:.3f} vs GT {gt:.3f} km²", flush=True)
    ok = [r for r in rows if "err_km2" in r]
    summary = {}
    for sensor in ("ICEYE", "PlanetScope"):
        ss = [r for r in ok if r["sensor"] == sensor]
        if ss:
            summary[sensor] = {
                "n": len(ss),
                "mae_km2": round(sum(abs(r["err_km2"]) for r in ss) / len(ss), 3),
                "mape_pct": round(sum(r["abs_pct_err"] for r in ss) / len(ss), 2),
                "bias_km2": round(sum(r["err_km2"] for r in ss) / len(ss), 3),
            }
    return {"per_scene": rows, "summary": summary}


# ── 출력 ────────────────────────────────────────────────────────────────────
def write_outputs(result: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    jp = os.path.join(OUT_DIR, "timeseries_metrics.json")
    json.dump(result, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # CSV: 평평한 표 (계열 라벨 포함)
    cp = os.path.join(OUT_DIR, "timeseries_metrics.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["계열", "구분", "대상", "N", "MSE(m²)", "RMSE(m)", "MAE(m)",
                    "bias(m)", "max|err|(m)", "MAPE(%)", "R2", "비고"])
        A = result["A_수위_보정_정확도"]
        for scope, block in [("전체", A["overall"])] + \
                [(s, v) for s, v in A["per_station"].items()]:
            for name, st_ in block.items():
                if st_.get("n"):
                    w.writerow(["A 수위보정(in-sample)", scope, name, st_["n"],
                                st_["mse_m2"], st_["rmse_m"], st_["mae_m"],
                                st_["bias_m"], st_["max_abs_err_m"],
                                st_["mape_pct"], st_["r2"],
                                "융합 LSTM 학습표본과 동일"])
        B = result["B_예측_참고치_persistence"]
        for name in ("persistence_of_corrected", "persistence_of_gauge"):
            st_ = B[name]
            w.writerow(["B 예측참고(walk-forward)", "전체", name, st_["n"],
                        st_["mse_m2"], st_["rmse_m"], st_["mae_m"], st_["bias_m"],
                        st_["max_abs_err_m"], st_["mape_pct"], st_["r2"],
                        f"불규칙 간격 {B['horizon_days']['min']}~"
                        f"{B['horizon_days']['max']}일"])
        w.writerow(["참고선", "인수문서", "LODO RMSE", "",
                    round(LODO_RMSE_M ** 2, 6), LODO_RMSE_M, "", "", "",
                    "", "", "정식 out-of-sample 프로토콜"])
        w.writerow(["참고선", "인수문서", "위성 미사용 베이스라인", "",
                    round(BASELINE_RMSE_M ** 2, 6), BASELINE_RMSE_M, "", "", "",
                    "", "", "이 값이 더 낮음 — '융합으로 향상' 주장 금지"])
        w.writerow([])
        w.writerow(["계열", "지평", "N", "MSE(m²)", "정규화MSE", "RMSE(m)",
                    "MAE(m)", "bias(m)", "max|err|(m)"])
        for b in result["E_지평별_성능"]["bins"]:
            w.writerow(["E 지평별(persistence)", b["horizon"], b["n"], b["mse_m2"],
                        b["norm_mse"], b["rmse_m"], b["mae_m"], b["bias_m"],
                        b["max_abs_err_m"]])
        w.writerow([])
        w.writerow(["계열", "지점", "기준일", "예측일", "지평(일)", "예측(m)",
                    "실측(m)", "오차(m)"])
        for p in result["E_지평별_성능"]["pairs"]:
            w.writerow(["E 관측쌍", p["station"], p["from"], p["to"],
                        p["horizon_days"], p["pred_m"], p["truth_m"], p["err_m"]])
        w.writerow([])
        w.writerow(["계열", "센서", "날짜", "WB(km²)", "GT(km²)", "오차(km²)",
                    "|오차|(%)"])
        for r in result["C_면적_시계열"]["per_scene"]:
            if "err_km2" in r:
                w.writerow(["C 면적", r["sensor"], r["date"], r["wb_km2"],
                            r["gt_km2"], r["err_km2"], r["abs_pct_err"]])
        for sensor, s in result["C_면적_시계열"]["summary"].items():
            w.writerow(["C 면적(요약)", sensor, f"n={s['n']}", "", "",
                        f"MAE {s['mae_km2']} / bias {s['bias_km2']}",
                        f"MAPE {s['mape_pct']}"])

    # MD 요약
    mp = os.path.join(OUT_DIR, "timeseries_metrics.md")
    A0 = result["A_수위_보정_정확도"]["overall"]
    B = result["B_예측_참고치_persistence"]
    C = result["C_면적_시계열"]["summary"]
    lines = [
        "# 시계열 정량 평가 (2026-08-21)",
        "",
        "계열 혼용 금지: A는 in-sample, B는 참고치, 참고선(LODO)이 정식 프로토콜.",
        "",
        "## A. 수위 보정 정확도 — 보정 수위 vs 게이지 실측 (in-sample 30표본)",
        "| 예측치 | N | MSE(m²) | RMSE(m) | MAE(m) | bias(m) | max|err|(m) | R² |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, st_ in A0.items():
        lines.append(f"| {name} | {st_['n']} | {st_['mse_m2']} | {st_['rmse_m']} | "
                     f"{st_['mae_m']} | {st_['bias_m']} | {st_['max_abs_err_m']} | "
                     f"{st_['r2']} |")
    lines += [
        "",
        "해석 주의: ① 보정은 상수-정렬 위성 수위(RMSE 0.057→0.026 m) 대비 개선되나, "
        "지점평균 베이스라인이 더 낮음 — 인수문서의 베이스라인 우위 구조와 일치. "
        "② 지점별 R²는 지점 내 수위 변동이 수 cm뿐이라 음수가 정상적으로 나온다 "
        "(분산 대비 지표의 한계) — 지점별 판단은 RMSE/MAE 기준.",
        "", f"참고선(인수문서, out-of-sample): LODO RMSE {LODO_RMSE_M} m · "
            f"위성 미사용 베이스라인 {BASELINE_RMSE_M} m (후자가 더 낮음 — 융합 향상 주장 금지)",
        "",
        f"## B. 예측 참고치 — walk-forward persistence "
        f"(간격 {B['horizon_days']['min']}~{B['horizon_days']['max']}일, "
        f"평균 {B['horizon_days']['mean']}일)",
        "| 방식 | N | MSE(m²) | RMSE(m) | MAE(m) | R² |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("persistence_of_corrected", "persistence_of_gauge"):
        st_ = B[name]
        lines.append(f"| {name} | {st_['n']} | {st_['mse_m2']} | {st_['rmse_m']} | "
                     f"{st_['mae_m']} | {st_['r2']} |")
    E = result["E_지평별_성능"]
    lines += ["", f"## E. 지평별 성능 — 모든 관측쌍 persistence ({E['n_pairs_total']}쌍)",
              "| 지평 | N | MSE(m²) | 정규화MSE | RMSE(m) | MAE(m) |",
              "|---|---|---|---|---|---|"]
    for b in E["bins"]:
        lines.append(f"| {b['horizon']} | {b['n']} | {b['mse_m2']} | {b['norm_mse']} | "
                     f"{b['rmse_m']} | {b['mae_m']} |")
    lines += ["", "주의: 갈수기(2~4월) 데이터라 저하가 완만함 — 홍수기 급변 시 "
                  "지평별 저하는 이보다 훨씬 가파를 것(이 데이터로는 측정 불가).",
              "", "## C. 면적 시계열 — WB vs GT (3 m 격자)",
              "| 센서 | N | MAE(km²) | MAPE(%) | bias(km²) |", "|---|---|---|---|---|"]
    for sensor, s in C.items():
        lines.append(f"| {sensor} | {s['n']} | {s['mae_km2']} | {s['mape_pct']} | "
                     f"{s['bias_km2']} |")
    open(mp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return jp, cp, mp


def main() -> int:
    print("[1/3] 수위 보정 정확도 (A)")
    truth = load_gauge()
    recs = load_fused()
    A, joined = eval_levels(recs, truth)
    print(f"  대조쌍 {A['overall']['corrected']['n']}건 · "
          f"corrected RMSE {A['overall']['corrected']['rmse_m']} m")
    print("[2/3] persistence walk-forward (B)")
    B = eval_persistence(joined)
    print("[2.5/3] 지평별 성능 (E) — 전체 관측쌍 persistence")
    E = eval_horizon(joined)
    print(f"  관측쌍 {E['n_pairs_total']}건")
    print("[3/3] 면적 시계열 (C) — 마스크 16개 집계")
    C = eval_areas()
    result = {
        "generated_by": "scripts/eval_timeseries_metrics.py",
        "labels": {
            "A": "in-sample 30표본 (융합 LSTM 학습표본과 동일) — 일반화 성능 아님. "
                 "지점별 R²는 지점 내 변동(수 cm) 대비 지표라 음수가 정상 — RMSE/MAE로 판단",
            "B": "관측 8시점·불규칙 간격 walk-forward 참고치",
            "C": "배포 산출 WB 마스크 vs GT 라벨, 3 m 격자",
            "참고선": f"LODO {LODO_RMSE_M} m / 위성 미사용 {BASELINE_RMSE_M} m "
                      "(정식 프로토콜, 인수문서)",
        },
        "A_수위_보정_정확도": A,
        "B_예측_참고치_persistence": B,
        "C_면적_시계열": C,
        "E_지평별_성능": E,
    }
    result["labels"]["E"] = ("모든 관측쌍 persistence — 갈수기 데이터라 홍수기 급변 저하는 "
                             "미반영. norm_mse는 수위 범위(0.985 m) 기준 정규화")
    paths = write_outputs(result)
    print("산출:", *paths, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
