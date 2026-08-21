#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""지평별 마스크 persistence IoU — 발표 덱의 하드코딩 IoU(0.86/0.62) 실측 대체.

정의: 씬 A의 산출 수체 마스크(WB_A)를 '예측'으로 삼아, h일 뒤 씬 B의 GT와
비교한 IoU. 함께 IoU(GT_A, GT_B)도 산출한다 — 수체 자체가 얼마나 변했는지의
상한선(어떤 persistence 예측도 이보다 좋을 수 없음).

씬별 3 m 격자 원점이 서로 달라(cross_date_alignment: warp 필요) 씬 A를
씬 B 격자로 nearest 재투영 후, 두 씬 모두 관측된 유효영역 교집합에서만 계산.
산출: data/eval/mask_persistence_iou.{json,md} (+ csv 행)
"""
from __future__ import annotations

import datetime as dt
import glob
import itertools
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(REPO, "data", "incoming", "handover")
OUT_DIR = os.path.join(REPO, "data", "eval")
NODATA = 255

SENSORS = {
    "ICEYE": {
        "wb_root": os.path.join(REPO, "data", "wbms_runs", "wb", "2020"),
        "gt_dir": os.path.join(H, "06_aux", "Labels_GT"),
        "dates": ["20200302", "20200330", "20200415", "20200416"],
    },
    "PlanetScope": {
        "wb_root": os.path.join(REPO, "data", "wbms_runs", "wb_planet", "2020"),
        "gt_dir": os.path.join(H, "06_aux", "Labels_GT_planet"),
        "dates": ["20200218", "20200312", "20200325", "20200414"],
    },
}
# 발표 덱 관례(단기 1~22일 / 장기)를 따르되 실제 최장 간격(56일)까지 커버
BINS = [("단기(1~22일)", 1, 22), ("장기(23~56일)", 23, 56)]


def open_scene(sensor: str, date: str):
    sc = SENSORS[sensor]
    wb = sorted(p for p in glob.glob(os.path.join(sc["wb_root"], date, "WB_*.tif")))
    gt = sorted(glob.glob(os.path.join(sc["gt_dir"], f"*{date}*.tif")))
    if not wb or not gt:
        raise FileNotFoundError(f"{sensor} {date}: WB 또는 GT 없음")
    return wb[0], gt[0]


def read_all(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        return ds.read(1), {"crs": ds.crs, "transform": ds.transform,
                            "width": ds.width, "height": ds.height}


def warp_to(src: np.ndarray, src_meta: dict, dst_meta: dict,
            fill: int = NODATA) -> np.ndarray:
    out = np.full((dst_meta["height"], dst_meta["width"]), fill, dtype=np.uint8)
    reproject(src, out,
              src_transform=src_meta["transform"], src_crs=src_meta["crs"],
              dst_transform=dst_meta["transform"], dst_crs=dst_meta["crs"],
              src_nodata=None, dst_nodata=fill,
              resampling=Resampling.nearest)
    return out


def iou(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    p = (pred == 1) & valid
    t = (truth == 1) & valid
    union = np.count_nonzero(p | t)
    return float(np.count_nonzero(p & t) / union) if union else float("nan")


def main() -> int:
    rows = []
    for sensor, sc in SENSORS.items():
        for da, db in itertools.combinations(sc["dates"], 2):
            h = (dt.datetime.strptime(db, "%Y%m%d")
                 - dt.datetime.strptime(da, "%Y%m%d")).days
            wb_a_p, gt_a_p = open_scene(sensor, da)
            wb_b_p, gt_b_p = open_scene(sensor, db)
            wb_b, meta_b = read_all(wb_b_p)
            gt_b, _ = read_all(gt_b_p)
            wb_a_src, meta_a = read_all(wb_a_p)
            gt_a_src, meta_ga = read_all(gt_a_p)
            wb_a = warp_to(wb_a_src, meta_a, meta_b)
            gt_a = warp_to(gt_a_src, meta_ga, meta_b)
            del wb_a_src, gt_a_src
            valid = (wb_b != NODATA) & (wb_a != NODATA)
            row = {
                "sensor": sensor, "from": da, "to": db, "horizon_days": h,
                "iou_forecast": round(iou(wb_a, gt_b, valid), 4),
                "iou_change_ceiling": round(iou(gt_a, gt_b, valid), 4),
                "valid_px": int(valid.sum()),
            }
            rows.append(row)
            print(f"{sensor} {da}→{db} ({h:2d}일): 예측 IoU {row['iou_forecast']:.4f} "
                  f"/ 상한 {row['iou_change_ceiling']:.4f}", flush=True)

    summary = {}
    for label, lo, hi in BINS:
        for sensor in SENSORS:
            sub = [r for r in rows
                   if r["sensor"] == sensor and lo <= r["horizon_days"] <= hi]
            if sub:
                summary[f"{sensor} {label}"] = {
                    "n": len(sub),
                    "iou_forecast_mean": round(
                        sum(r["iou_forecast"] for r in sub) / len(sub), 4),
                    "iou_forecast_min": min(r["iou_forecast"] for r in sub),
                    "iou_change_ceiling_mean": round(
                        sum(r["iou_change_ceiling"] for r in sub) / len(sub), 4),
                }

    os.makedirs(OUT_DIR, exist_ok=True)
    result = {
        "definition": "IoU(WB_A→B격자 재투영, GT_B) — 관측 마스크 persistence를 "
                      "예측으로 간주한 실측. ceiling=IoU(GT_A, GT_B)=수체 자체 변화 상한. "
                      "갈수기 4씬이라 홍수기 급변 저하는 미반영.",
        "pairs": rows, "summary": summary,
    }
    json.dump(result, open(os.path.join(OUT_DIR, "mask_persistence_iou.json"),
                           "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    lines = ["# 지평별 마스크 persistence IoU (실측, 2026-08-21)", "",
             "| 센서 | 구간 | N | 예측 IoU(평균) | 최저 | 변화상한 IoU |",
             "|---|---|---|---|---|---|"]
    for key, s in summary.items():
        sensor, label = key.split(" ", 1)
        lines.append(f"| {sensor} | {label} | {s['n']} | {s['iou_forecast_mean']} | "
                     f"{s['iou_forecast_min']} | {s['iou_change_ceiling_mean']} |")
    lines += ["", "쌍별:", "| 센서 | A→B | 지평(일) | 예측 IoU | 상한 IoU |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sensor']} | {r['from']}→{r['to']} | {r['horizon_days']} | "
                     f"{r['iou_forecast']} | {r['iou_change_ceiling']} |")
    open(os.path.join(OUT_DIR, "mask_persistence_iou.md"), "w",
         encoding="utf-8").write("\n".join(lines) + "\n")
    print("산출: data/eval/mask_persistence_iou.{json,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
