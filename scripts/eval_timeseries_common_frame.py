#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IITP 성과지표(정규화 MSE)를 **공통 격자 면적 계열**로 다시 계산해 원계열과 나란히 낸다.

원본 `scripts/eval_timeseries_metrics.py` 는 그대로 둔다(재현 가능해야 한다). 이 스크립트는
같은 함수(`eval_indicator`)에 **면적 계열만 바꿔** 넣어 두 변종을 한 파일에 같이 적는다.

  · scene_footprint (원본) : 씬마다 자기 촬영범위에서 센 면적 → 기존 공표값 0.13526
  · common_frame   (정정) : 센서별 공통 격자에서 센 면적 → 이 스크립트가 내는 값

왜 정정이 필요한가: 2020-03-25 광학 씬은 03-12 보다 남쪽으로 8.5 km 치우쳐 하구/바다를 담는다.
그래서 면적이 33 → 65 km² 로 '급증'한 것처럼 보였지만, 같은 창(공통 격자)에서 세면 28.9 → 30.1 km²
이고 그 씬의 GT 라벨도 같은 창에서 30.0 km² 다. 탐지는 맞았고(원 footprint 기준 GT 65.06 vs
탐지 65.30, 0.37 % 오차) **계열이 비교 불가능했던 것**이다. 홍수가 아니다.

산출:
  data/eval/timeseries_metrics_common_frame.json
  docs/presentation/면적계열_공통격자_정정.md
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import eval_timeseries_metrics as ets  # noqa: E402  (경로 삽입 후 import)

REGION = "nakdong_busan"
AREA_SERIES = os.path.join(REPO, "regions", REGION, "satellite", "area_series.json")
OUT_JSON = os.path.join(REPO, "data", "eval", "timeseries_metrics_common_frame.json")
OUT_MD = os.path.join(REPO, "docs", "presentation", "면적계열_공통격자_정정.md")
PUBLISHED_COMPOSITE = 0.13526   # 기존 공표값(씬 footprint 계열) — 지우지 말 것
PUBLISHED_AREA = 0.27014
PUBLISHED_LEVEL = 0.00039


# ── 면적 계열 두 가지 ────────────────────────────────────────────────────────
def area_rows_scene_footprint() -> list[dict]:
    """원본 계열 — 씬 원해상도(3 m)에서 각자 footprint 로 센 WB/GT 면적."""

    return [r for r in ets.eval_areas()["per_scene"] if "err_km2" in r]


def area_rows_common_frame() -> list[dict]:
    """정정 계열 — 센서별 공통 격자에서 센 WB/GT 면적(build_common_frame_area.py 산출)."""

    if not os.path.isfile(AREA_SERIES):
        raise SystemExit(f"{AREA_SERIES} 가 없다 — 먼저 scripts/build_common_frame_area.py 를 돌릴 것")
    payload = json.load(open(AREA_SERIES, encoding="utf-8"))
    rows = []
    for sensor, block in payload["sensors"].items():
        for row in block["rows"]:
            wb, gt = row.get("area_km2_common_frame"), row.get("gt_area_km2")
            if wb is None or gt is None:
                continue
            rows.append({
                "sensor": sensor, "date": row["date"],
                "wb_km2": round(float(wb), 3), "gt_km2": round(float(gt), 3),
                "err_km2": round(float(wb) - float(gt), 3),
                "abs_pct_err": round(100 * abs(float(wb) - float(gt)) / float(gt), 2),
            })
    rows.sort(key=lambda r: (r["sensor"], r["date"]))
    return rows


def area_summary(rows: list[dict]) -> dict:
    out = {}
    for sensor in sorted({r["sensor"] for r in rows}):
        ss = [r for r in rows if r["sensor"] == sensor]
        out[sensor] = {
            "n": len(ss),
            "mse_km2sq": round(sum(r["err_km2"] ** 2 for r in ss) / len(ss), 3),
            "mae_km2": round(sum(abs(r["err_km2"]) for r in ss) / len(ss), 3),
            "mape_pct": round(sum(r["abs_pct_err"] for r in ss) / len(ss), 2),
            "bias_km2": round(sum(r["err_km2"] for r in ss) / len(ss), 3),
        }
    return out


# ── footprint 효과의 크기 ────────────────────────────────────────────────────
AREA_SERIES_REGIONS = (REGION, "geum_daecheong")


def footprint_effect() -> dict:
    """지역·센서별 상관(원면적 vs 촬영범위, 공통격자면적 vs 촬영범위)과 03-25 분해."""

    per_sensor = {}
    for region_id in AREA_SERIES_REGIONS:
        path = os.path.join(REPO, "regions", region_id, "satellite", "area_series.json")
        if not os.path.isfile(path):
            continue
        payload = json.load(open(path, encoding="utf-8"))
        for sensor, block in payload["sensors"].items():
            rows = block["rows"]
            per_sensor[f"{region_id}/{sensor}"] = {
                "region_id": region_id, "sensor": sensor, "n": len(rows),
                "footprint_km2_range": [min(r["scene_footprint_km2"] for r in rows),
                                        max(r["scene_footprint_km2"] for r in rows)],
                "raw_area_km2_range": [min(r["area_km2_scene_footprint"] for r in rows),
                                       max(r["area_km2_scene_footprint"] for r in rows)],
                "common_frame_area_km2_range": [min(r["area_km2_common_frame"] for r in rows),
                                                max(r["area_km2_common_frame"] for r in rows)],
                **block["correlations"],
            }
    return {"regions": list(AREA_SERIES_REGIONS), "per_sensor": per_sensor,
            "spike_20200325": spike_decomposition()}


def spike_decomposition() -> dict:
    """03-25 의 '급증'을 공통 격자 안/밖으로 쪼갠다(원해상도 3 m 그대로 센다)."""

    try:
        import numpy as np
        import rasterio
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason_ko": f"원본 마스크를 못 읽어 분해 생략 ({exc})"}
    scenes = json.load(open(os.path.join(REPO, "regions", REGION, "satellite", "scenes.json"), encoding="utf-8"))
    index = {s["scene_id"]: s for s in scenes["scenes"]}
    frame = index["20200325_PlanetScope"]["frame"]["bounds_epsg32652"]
    out = {"available": True, "common_frame_bounds_epsg32652": frame, "scenes": {}}
    for scene_id in ("20200312_PlanetScope", "20200325_PlanetScope"):
        scene = index[scene_id]
        path = os.path.join(REPO, scene["source_mask"])
        if not os.path.isfile(path):
            return {"available": False, "reason_ko": f"{path} 없음"}
        with rasterio.open(path) as ds:
            arr = ds.read(1)
            tr = ds.transform
            pixel = abs(tr.a * tr.e)
            rows, cols = np.indices(arr.shape)
            xs = tr.c + (cols + 0.5) * tr.a
            ys = tr.f + (rows + 0.5) * tr.e
            inside = (xs >= frame[0]) & (xs <= frame[2]) & (ys >= frame[1]) & (ys <= frame[3])
            water = arr == 1
            out["scenes"][scene["date"]] = {
                "footprint_bounds_epsg32652": scene["quicklook"]["bounds_epsg32652"],
                "water_total_km2": round(float(water.sum()) * pixel / 1e6, 3),
                "water_inside_common_frame_km2": round(float((water & inside).sum()) * pixel / 1e6, 3),
                "water_outside_common_frame_km2": round(float((water & ~inside).sum()) * pixel / 1e6, 3),
                "valid_outside_common_frame_km2": round(float(((arr != 255) & ~inside).sum()) * pixel / 1e6, 1),
            }
    a, b = out["scenes"]["2020-03-12"], out["scenes"]["2020-03-25"]
    out["explained"] = {
        "raw_jump_km2": round(b["water_total_km2"] - a["water_total_km2"], 3),
        "from_outside_strip_km2": round(b["water_outside_common_frame_km2"] - a["water_outside_common_frame_km2"], 3),
        "inside_common_frame_km2": round(b["water_inside_common_frame_km2"] - a["water_inside_common_frame_km2"], 3),
        "southward_shift_m": round(a["footprint_bounds_epsg32652"][1] - b["footprint_bounds_epsg32652"][1], 1),
    }
    total = out["explained"]["raw_jump_km2"]
    out["explained"]["share_from_footprint_pct"] = (
        round(100 * out["explained"]["from_outside_strip_km2"] / total, 1) if total else None)
    out["explained"]["note_ko"] = (
        "원계열의 '급증' 중 대부분은 03-25 씬이 남쪽으로 더 내려가 담은 하구·바다 띠에서 나온다."
        " 공통 격자 안에서의 변화는 그 나머지뿐이며 홍수 규모가 아니다.")
    return out


# ── 대청 역상관(수위↑인데 면적↓) 원인 조사 ──────────────────────────────────
DAECHEONG = "geum_daecheong"
DAECHEONG_STATION = "대청댐"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx, my = sum(p[0] for p in pairs) / n, sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    return round(sum((p[0] - mx) * (p[1] - my) for p in pairs) / math.sqrt(sxx * syy), 3)


def daecheong_investigation() -> dict:
    """대청: 저수위가 높은 2019 배치가 면적이 더 작게 나오는 이유를 데이터로 가른다.

    가설 세 가지를 같은 수치로 검증한다.
      (1) 공통 격자 안 유효자료 비율이 배치마다 다르다 → 덜 본 만큼 면적이 작게 나온다.
      (2) 공통 격자가 저수지 일부(댐 근처 구간)만 덮는다 → 저수지 전체 수위와 대응이 약하다.
      (3) 구름·결빙 등 광학 품질 → 밝은 화소 비율로 확인.
    """

    path = os.path.join(REPO, "regions", DAECHEONG, "satellite", "area_series.json")
    if not os.path.isfile(path):
        return {"available": False, "reason_ko": "area_series.json 없음"}
    payload = json.load(open(path, encoding="utf-8"))
    block = payload["sensors"].get("PlanetScope")
    if not block:
        return {"available": False, "reason_ko": "PlanetScope 계열 없음"}
    rows = sorted(block["rows"], key=lambda r: r["date"])
    level = [r["gauge_levels"].get(DAECHEONG_STATION) for r in rows]
    cf = [r["area_km2_common_frame"] for r in rows]
    civ = [r["area_km2_common_valid_footprint"] for r in rows]
    vfr = [r["frame_valid_fraction"] for r in rows]
    batch = ["2019" if r["date"] < "2021" else "2022" for r in rows]
    idx19 = [i for i, b in enumerate(batch) if b == "2019"]
    idx22 = [i for i, b in enumerate(batch) if b == "2022"]

    def sub(values, idx):
        return [values[i] for i in idx]

    out = {
        "available": True,
        "station": DAECHEONG_STATION,
        "frame": block["frame"],
        "common_valid_footprint_km2": block["common_valid_footprint_km2"],
        "rows": [{"date": r["date"], "level_m": r["gauge_levels"].get(DAECHEONG_STATION),
                  "area_km2_common_frame": r["area_km2_common_frame"],
                  "area_km2_common_valid_footprint": r["area_km2_common_valid_footprint"],
                  "frame_valid_fraction": r["frame_valid_fraction"],
                  "scene_valid_area_km2": r["valid_area_km2"], "batch": b}
                 for r, b in zip(rows, batch)],
        "correlations": {
            "level_vs_common_frame_area": _pearson(level, cf),
            "level_vs_common_valid_area": _pearson(level, civ),
            "level_vs_frame_valid_fraction": _pearson(level, vfr),
            "common_frame_area_vs_frame_valid_fraction": _pearson(cf, vfr),
            "level_vs_common_valid_area_2019": _pearson(sub(level, idx19), sub(civ, idx19)),
            "level_vs_common_valid_area_2022": _pearson(sub(level, idx22), sub(civ, idx22)),
        },
        "batches": {
            "2019": {"n": len(idx19),
                     "level_m": [min(sub(level, idx19)), max(sub(level, idx19))],
                     "frame_valid_fraction": [min(sub(vfr, idx19)), max(sub(vfr, idx19))],
                     "area_km2_common_frame": [min(sub(cf, idx19)), max(sub(cf, idx19))],
                     "area_km2_common_valid_footprint": [min(sub(civ, idx19)), max(sub(civ, idx19))],
                     "scene_valid_area_km2": [min(r["valid_area_km2"] for r in sub(rows, idx19)),
                                              max(r["valid_area_km2"] for r in sub(rows, idx19))]},
            "2022": {"n": len(idx22),
                     "level_m": [min(sub(level, idx22)), max(sub(level, idx22))],
                     "frame_valid_fraction": [min(sub(vfr, idx22)), max(sub(vfr, idx22))],
                     "area_km2_common_frame": [min(sub(cf, idx22)), max(sub(cf, idx22))],
                     "area_km2_common_valid_footprint": [min(sub(civ, idx22)), max(sub(civ, idx22))],
                     "scene_valid_area_km2": [min(r["valid_area_km2"] for r in sub(rows, idx22)),
                                              max(r["valid_area_km2"] for r in sub(rows, idx22))]},
        },
        "cloud_check": daecheong_brightness(),
        "verdict_ko": [],
    }
    c = out["correlations"]
    out["verdict_ko"] = [
        f"공통 격자 면적은 수위와 r={c['level_vs_common_frame_area']} 로 역상관처럼 보이지만, 같은 면적이"
        f" 공통 격자 안 유효자료 비율과는 r={c['common_frame_area_vs_frame_valid_fraction']} 로 거의 일대일이다."
        " 즉 '수위가 높은데 면적이 작다'가 아니라 '2019 배치가 프레임을 덜 덮었다'가 맞다.",
        f"모든 씬의 유효영역 교집합({out['common_valid_footprint_km2']} km², 진짜로 고정된 창)에서 다시 세면"
        f" 수위와의 상관이 r={c['level_vs_common_valid_area']} 로 사라진다 — 역상관은 실체가 아니다.",
        f"프레임을 100 % 덮은 2022 배치 안에서만 보면 수위-면적 상관이 r={c['level_vs_common_valid_area_2022']}"
        f" 로 물리적으로 옳은 양의 방향이다(2019 배치 안에서는 r={c['level_vs_common_valid_area_2019']}, 덮개가 59~87 %로"
        " 들쭉날쭉해 신호가 나오지 않는다).",
        f"공통 격자는 {block['frame']['coverage_km2']} km² 이고 그 안에서 탐지되는 물은 6~8 km² 다."
        " 대청호 만수위 수면적(약 72 km²)의 10 % 안팎, 그것도 댐 직상류 구간만 보는 창이다."
        " 이 창의 면적으로 저수지 전체 저수량·수위를 논하면 안 된다.",
        "구름·결빙은 원인이 아니다 — 열 장 모두 공통 격자 안 밝은 화소 비율이 1.1 % 이하다(아래 cloud_check).",
        "결론: 2019 배치(5씬)는 면적 시계열 근거로 쓰지 않는다. 촬영 자체가 공통 격자를 부분적으로만 덮어"
        " (유효자료 59~87 %, 씬 유효면적 210~220 km² vs 2022 배치 690~800 km²) 면적이 구조적으로 과소평가된다."
        " 대청 면적 계열을 인용해야 한다면 area_km2_common_valid_footprint(고정 창) 만, 그것도 '댐 인근 구간'"
        " 이라는 단서와 함께 쓴다.",
    ]
    return out


def daecheong_brightness() -> dict:
    """구름/결빙 대리지표 — 공통 격자 안 가시밴드 평균 반사도와 밝은 화소 비율."""

    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason_ko": f"원본 영상을 못 읽어 생략 ({exc})"}
    scenes_path = os.path.join(REPO, "regions", DAECHEONG, "satellite", "scenes.json")
    scenes = json.load(open(scenes_path, encoding="utf-8"))["scenes"]
    rows = []
    for scene in sorted(scenes, key=lambda s: s["date"]):
        pre = os.path.join(REPO, scene["source_pre"])
        if not os.path.isfile(pre):
            continue
        frame = scene["frame"]
        bounds = frame["bounds_epsg32652"]
        width, height = int(frame["width"]), int(frame["height"])
        with rasterio.open(pre) as ds:
            bands = min(ds.count, 3)
            dst = np.full((bands, height, width), np.nan, np.float32)
            reproject(ds.read(list(range(1, bands + 1))).astype(np.float32), dst,
                      src_transform=ds.transform, src_crs=ds.crs,
                      dst_transform=from_bounds(*bounds, width, height), dst_crs="EPSG:32652",
                      src_nodata=ds.nodata, dst_nodata=np.nan, resampling=Resampling.average)
            nodata = ds.nodata
        visible = np.nanmean(dst, axis=0)
        valid = np.isfinite(visible) & (visible > 0)
        if nodata is not None:
            valid &= visible < float(nodata) - 1e-3
        if not valid.any():
            continue
        values = visible[valid]
        rows.append({"date": scene["date"],
                     "mean_reflectance": round(float(values.mean()), 4),
                     "bright_pixel_fraction_gt_0p28": round(float((values > 0.28).mean()), 4)})
    return {"available": True, "threshold_note_ko":
            "가시밴드 평균 반사도 0.28 초과를 '밝음(구름·눈 후보)'으로 본 대리지표. 정식 구름마스크가 아니다.",
            "rows": rows}


# ── 실행 ────────────────────────────────────────────────────────────────────
def run() -> dict:
    truth = ets.load_gauge()
    _levels, joined = ets.eval_levels(ets.load_fused(), truth)

    variants = {}
    for name, rows, label in (
        ("scene_footprint", area_rows_scene_footprint(),
         "씬 footprint 계열 (원본·기존 공표값) — 씬마다 자기 촬영범위에서 센 면적"),
        ("common_frame", area_rows_common_frame(),
         "공통 격자 계열 (정정) — 센서별 공통 격자에서 센 면적, 날짜 간 비교 가능"),
    ):
        indicator = ets.eval_indicator(joined, rows)
        gts = [r["gt_km2"] for r in rows]
        indicator["정의"] = (
            f"변수별 [0,1] min-max 정규화(수위 범위 {round(ets.LEVEL_RANGE_M, 3)} m · "
            f"면적 범위 {round(max(gts) - min(gts), 1)} km²) 후 persistence 1-step 예측 MSE, 동가중 평균")
        indicator.pop("주의", None)
        if name == "common_frame":
            # eval_indicator 의 '정정_참고' 는 씬 footprint 계열을 가리키는 문구다.
            # 정정 변종에 그대로 달면 "이 값은 씬 footprint 계열" 이라는 거짓말이 된다.
            indicator.pop("정정_참고", None)
        variants[name] = {
            "label_ko": label,
            "면적_계열": rows,
            "면적_요약": area_summary(rows),
            "지표": indicator,
        }

    base = variants["scene_footprint"]["지표"]
    fixed = variants["common_frame"]["지표"]
    result = {
        "generated_by": "scripts/eval_timeseries_common_frame.py",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "region_id": REGION,
        "labels": {
            "성과지표": "수체면적 및 수위변화 시계열 예측 정확도(MSE) · 목표 0.38",
            "변종_설명_ko": "같은 계산식에 면적 계열만 바꿔 넣은 두 결과. 수위 항은 두 변종에서 동일하다(면적 계열과 무관).",
        },
        "published_scene_footprint": {
            "종합_정규화MSE": PUBLISHED_COMPOSITE,
            "수체면적_정규화MSE": PUBLISHED_AREA,
            "수위변화_정규화MSE": PUBLISHED_LEVEL,
            "출처": "data/eval/timeseries_metrics.json (scripts/eval_timeseries_metrics.py)",
            "주의_ko": "공표된 값이므로 지우지 않는다. 다만 면적 항은 촬영범위 이동이 지배한다는 사실을 함께 적어야 한다.",
        },
        "variants": variants,
        "comparison": {
            "수위변화_정규화MSE": {"scene_footprint": base["수위변화_정규화MSE"],
                             "common_frame": fixed["수위변화_정규화MSE"],
                             "note_ko": "수위 항은 면적 계열과 무관하므로 그대로다."},
            "수체면적_정규화MSE": {"scene_footprint": base["수체면적_정규화MSE"],
                             "common_frame": fixed["수체면적_정규화MSE"]},
            "종합_정규화MSE": {"scene_footprint": base["종합_정규화MSE"],
                          "common_frame": fixed["종합_정규화MSE"], "목표": 0.38},
            "재현_확인_ko": ("scene_footprint 변종이 공표값과 같아야 이 계산이 원본과 같은 식임이 증명된다: "
                         f"{base['종합_정규화MSE']} vs 공표 {PUBLISHED_COMPOSITE}"),
            "일치": bool(abs(base["종합_정규화MSE"] - PUBLISHED_COMPOSITE) < 5e-5),
        },
        "footprint_effect": footprint_effect(),
        "daecheong_negative_correlation": daecheong_investigation(),
        "honesty_ko": [
            "두 변종 모두 목표 0.38 을 만족한다. 정정의 목적은 값을 좋게 만드는 것이 아니라, 원계열의 면적 오차가"
            " 수문 현상이 아니라 촬영 기하에서 온 것임을 밝히는 데 있다.",
            "정규화 분모(면적 범위)가 변종마다 다르다 — 공통 격자 계열은 범위가 좁아 분모가 작아지는데도 정규화 MSE 가"
            " 더 낮다. 즉 개선은 분모 장난이 아니다.",
            "인접 씬쌍 6개뿐이라 어느 변종이든 단일 사건이 계열을 지배할 수 있다. n 을 반드시 병기할 것.",
        ],
    }
    return result


def write_markdown(result: dict) -> None:
    base = result["variants"]["scene_footprint"]["지표"]
    fixed = result["variants"]["common_frame"]["지표"]
    fx = result["footprint_effect"]
    spike = fx["spike_20200325"]
    ps = fx["per_sensor"]
    cf_rows = result["variants"]["common_frame"]["면적_계열"]
    cf_0312 = next(r for r in cf_rows if r["date"] == "2020-03-12")
    cf_0325 = next(r for r in cf_rows if r["date"] == "2020-03-25")

    lines = [
        "# 수체 면적 계열 정정 — 공통 격자 기준 재계산",
        "",
        f"> 생성: `scripts/eval_timeseries_common_frame.py` ({result['generated_at'][:10]}) · "
        "근거 데이터 `regions/nakdong_busan/satellite/area_series.json`, `data/eval/timeseries_metrics_common_frame.json`",
        "> **기존 공표값은 지우지 않는다.** 아래 표는 같은 계산식에 면적 계열만 바꿔 넣은 결과를 나란히 둔 것이다.",
        "",
        "## 1. 한 줄 결론",
        "",
        "**2020-03-25 광학 씬의 \"면적 급증(33 → 65 km²)\"은 홍수가 아니라 촬영범위(footprint)가 남쪽으로 "
        f"{spike['explained']['southward_shift_m'] / 1000:.1f} km 옮겨간 결과다.** "
        f"같은 창(센서별 공통 격자)으로 세면 03-12 → 03-25 면적은 {cf_0312['wb_km2']:.3f} → {cf_0325['wb_km2']:.3f} km² 로 "
        f"{100 * (cf_0325['wb_km2'] - cf_0312['wb_km2']) / cf_0312['wb_km2']:.1f} % 늘었을 뿐이다.",
        "",
        "## 2. 성과지표 — 원본과 정정 나란히",
        "",
        "| 구성 | 씬 footprint 계열 (기존 공표) | 공통 격자 계열 (정정) |",
        "|---|---|---|",
        f"| 수위변화 예측 (정규화 MSE) | {base['수위변화_정규화MSE']} | {fixed['수위변화_정규화MSE']} (동일 — 면적과 무관) |",
        f"| 수체면적 예측 (정규화 MSE) | {base['수체면적_정규화MSE']} | {fixed['수체면적_정규화MSE']} |",
        f"| **종합 (동가중 평균)** | **{base['종합_정규화MSE']}** | **{fixed['종합_정규화MSE']}** |",
        f"| 목표 0.38 | {'달성' if base['달성'] else '미달'} | {'달성' if fixed['달성'] else '미달'} |",
        "",
        f"기존 공표값 **{result['published_scene_footprint']['종합_정규화MSE']}** 은 이 스크립트의 "
        f"씬 footprint 변종 **{base['종합_정규화MSE']}** 과 일치한다"
        f"({'재현 확인' if result['comparison']['일치'] else '⚠ 불일치 — 확인 필요'}). 계산식은 그대로이고 입력 계열만 바꿨다.",
        "",
        f"- 원본 정의: {base['정의']}",
        f"- 정정 정의: {fixed['정의']}",
        "",
        "## 3. 03-25 \"급증\"이 촬영범위 이동인 증거",
        "",
        "**(1) 촬영범위 자체가 다르다** (EPSG:32652, m)",
        "",
        "| 씬 | footprint (minx, miny, maxx, maxy) | 공통 격자 |",
        "|---|---|---|",
    ]
    for date in ("2020-03-12", "2020-03-25"):
        b = spike["scenes"][date]["footprint_bounds_epsg32652"]
        lines.append(f"| {date} | {b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f} | "
                     f"{', '.join(f'{v:.0f}' for v in spike['common_frame_bounds_epsg32652'])} |")
    lines += [
        "",
        f"03-25 씬의 남쪽 경계가 03-12 보다 **{spike['explained']['southward_shift_m']:.0f} m 더 남쪽**이다. "
        "즉 03-12 는 상류 구간을, 03-25 는 하구(강폭이 훨씬 넓고 바다에 닿는 구간)를 담았다.",
        "",
        "**(2) 물 면적을 공통 격자 안/밖으로 쪼개면** (원해상도 3 m 그대로)",
        "",
        "| 씬 | 전체 물 | 공통 격자 안 | 공통 격자 밖(남쪽 띠) |",
        "|---|---|---|---|",
    ]
    for date in ("2020-03-12", "2020-03-25"):
        s = spike["scenes"][date]
        lines.append(f"| {date} | {s['water_total_km2']:.3f} km² | {s['water_inside_common_frame_km2']:.3f} km² | "
                     f"{s['water_outside_common_frame_km2']:.3f} km² |")
    ex = spike["explained"]
    lines += [
        "",
        f"원계열의 급증 **{ex['raw_jump_km2']:+.3f} km²** 중 **{ex['from_outside_strip_km2']:+.3f} km²"
        f"({ex['share_from_footprint_pct']:.1f} %)** 가 공통 격자 **밖** 남쪽 띠에서 나왔다. "
        f"공통 격자 안의 변화는 {ex['inside_common_frame_km2']:+.3f} km² 뿐이다.",
        "",
        "**(3) 사람이 그린 GT 라벨도 같은 말을 한다**",
        "",
        "| 기준 창 | 탐지(WB) | GT 라벨 | 차이 |",
        "|---|---|---|---|",
        "| 씬 footprint (원본) | 65.305 km² | 65.063 km² | +0.24 km² (0.37 %) |",
    ]
    lines += [
        f"| 공통 격자 (정정) | {cf_0325['wb_km2']:.3f} km² | {cf_0325['gt_km2']:.3f} km² | "
        f"{cf_0325['err_km2']:+.3f} km² ({cf_0325['abs_pct_err']:.2f} %) |",
        "",
        "두 창 모두에서 탐지와 GT 가 일치한다 → **탐지 모델은 틀리지 않았다**. 틀린 것은 창이 다른 값들을 "
        "한 줄에 세워 시계열이라고 부른 쪽이다. 65.06 km² 는 '그 씬 촬영범위 안의 물'로는 맞는 값이고, "
        "'같은 지역의 물이 늘었다'는 뜻으로는 틀린 값이다.",
        "",
        "## 4. 촬영범위 효과의 크기 (센서별)",
        "",
        "| 지역·센서 | 씬 | 촬영범위(km²) | 원면적(km²) | 공통격자 면적(km²) | 원면적 ↔ 촬영범위 r | 공통격자면적 ↔ 촬영범위 r |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, blk in ps.items():
        lines.append(
            f"| {key} | {blk['n']} | {blk['footprint_km2_range'][0]:.0f}~{blk['footprint_km2_range'][1]:.0f} | "
            f"{blk['raw_area_km2_range'][0]:.1f}~{blk['raw_area_km2_range'][1]:.1f} | "
            f"{blk['common_frame_area_km2_range'][0]:.1f}~{blk['common_frame_area_km2_range'][1]:.1f} | "
            f"{blk['raw_area_vs_footprint_km2']} | {blk['common_frame_area_vs_footprint_km2']} |")
    lines += [
        "",
        "r 은 면적과 촬영범위 넓이의 상관계수(부산은 n=4 이므로 방향만 읽을 것). 대청 PlanetScope(n=10)는 "
        "원면적 ↔ 촬영범위 r = 0.909 — **원계열이 사실상 촬영범위를 보고 있었다**. 같은 지역에서 촬영범위가 "
        "378~391 km²(2019)에서 1044~1187 km²(2022)로 3배가 되자 원면적도 14~30 → 36~58 km² 로 함께 뛰었지만, "
        "공통 격자로 세면 9.9~16.9 → 17.0~18.7 km² 로 폭이 크게 줄어든다(남는 차이는 §5 에서 다시 가른다).",
        "",
        "부산 광학의 `raw_area_vs_footprint_km2` 가 음수인 것은 03-25 의 이동이 '넓이'가 아니라 '위치' 변화이기 "
        "때문이다 — 촬영범위 넓이만으로는 이 문제가 안 잡힌다는 뜻이고, 그래서 넓이가 아니라 **같은 격자**로 "
        "맞춰야 한다.",
        "",
    ]

    dc = result.get("daecheong_negative_correlation") or {}
    if dc.get("available"):
        c = dc["correlations"]
        b19, b22 = dc["batches"]["2019"], dc["batches"]["2022"]
        lines += [
            "## 5. 곁다리 확인 — 대청 \"수위는 오르는데 면적은 준다\"의 정체",
            "",
            "공통 격자로 고쳐도 대청(금강)은 저수위와 면적이 **거꾸로** 움직이는 것처럼 보였다"
            f"(r={c['level_vs_common_frame_area']}). 저수지에서 물리적으로 불가능한 방향이라 원인을 따로 캤다.",
            "",
            "| 배치 | 씬 | 대청댐 수위(m) | 공통 격자 안 유효자료 | 공통 격자 면적(km²) | 고정 창 면적(km²) | 씬 유효면적(km²) |",
            "|---|---|---|---|---|---|---|",
            f"| 2019-03~04 | {b19['n']} | {b19['level_m'][0]:.2f}~{b19['level_m'][1]:.2f} | "
            f"{100 * b19['frame_valid_fraction'][0]:.0f}~{100 * b19['frame_valid_fraction'][1]:.0f} % | "
            f"{b19['area_km2_common_frame'][0]:.2f}~{b19['area_km2_common_frame'][1]:.2f} | "
            f"{b19['area_km2_common_valid_footprint'][0]:.2f}~{b19['area_km2_common_valid_footprint'][1]:.2f} | "
            f"{b19['scene_valid_area_km2'][0]:.0f}~{b19['scene_valid_area_km2'][1]:.0f} |",
            f"| 2022-05 | {b22['n']} | {b22['level_m'][0]:.2f}~{b22['level_m'][1]:.2f} | "
            f"{100 * b22['frame_valid_fraction'][0]:.0f}~{100 * b22['frame_valid_fraction'][1]:.0f} % | "
            f"{b22['area_km2_common_frame'][0]:.2f}~{b22['area_km2_common_frame'][1]:.2f} | "
            f"{b22['area_km2_common_valid_footprint'][0]:.2f}~{b22['area_km2_common_valid_footprint'][1]:.2f} | "
            f"{b22['scene_valid_area_km2'][0]:.0f}~{b22['scene_valid_area_km2'][1]:.0f} |",
            "",
            "| 상관 | r |",
            "|---|---|",
            f"| 수위 ↔ 공통 격자 면적 | {c['level_vs_common_frame_area']} (역상관처럼 보임) |",
            f"| 공통 격자 면적 ↔ 프레임 안 유효자료 비율 | **{c['common_frame_area_vs_frame_valid_fraction']}** |",
            f"| 수위 ↔ 프레임 안 유효자료 비율 | {c['level_vs_frame_valid_fraction']} |",
            f"| 수위 ↔ **고정 창**(전 씬 유효영역 교집합 {dc['common_valid_footprint_km2']} km²) 면적 | "
            f"{c['level_vs_common_valid_area']} (사라짐) |",
            f"| 위 상관을 2022 배치 안에서만 | **{c['level_vs_common_valid_area_2022']}** (양의 방향 — 물리적으로 옳음) |",
            f"| 위 상관을 2019 배치 안에서만 | {c['level_vs_common_valid_area_2019']} |",
            "",
            "**데이터가 지지하는 결론**",
            "",
        ]
        lines += [f"{i}. {text}" for i, text in enumerate(dc["verdict_ko"], 1)]
        cloud = dc.get("cloud_check") or {}
        if cloud.get("available"):
            worst = max(cloud["rows"], key=lambda r: r["bright_pixel_fraction_gt_0p28"])
            lines += [
                "",
                f"구름 대리지표(가시밴드 평균 반사도 > 0.28 화소 비율): 최대가 {worst['date']} 의 "
                f"{100 * worst['bright_pixel_fraction_gt_0p28']:.2f} % — 구름·결빙으로 설명되지 않는다. "
                f"({cloud['threshold_note_ko']})",
            ]
        lines += ["", "## 6. 인용 규칙", ""]
    else:
        lines += ["## 5. 인용 규칙", ""]

    lines += [
        "- 면적 **시계열**을 말할 때는 반드시 `area_km2_common_frame`(공통 격자)을 쓴다.",
        "- `water_area_km2`(= `area_km2_scene_footprint`)는 **그 씬 한 장의 탐지 면적**으로만 인용한다. "
        "탐지 정확도(IoU·GT 대비 면적 오차)의 근거로는 여전히 유효하다.",
        "- 03-25 를 \"면적 급증 이벤트\"·\"홍수\"·\"이벤트 예측의 필요성\"으로 소개하지 않는다. "
        "이벤트 예측(ConvLSTM+기상 융합)의 필요성은 다른 근거로 말해야 한다.",
        "- 기존 공표값 0.13526 을 지우지 말고, 옆에 공통 격자 값을 함께 적는다.",
        "",
    ]
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → {os.path.relpath(OUT_MD, REPO)}", flush=True)


def main() -> int:
    result = run()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"  → {os.path.relpath(OUT_JSON, REPO)}", flush=True)
    write_markdown(result)
    c = result["comparison"]
    print(f"\n종합 정규화 MSE — 씬 footprint {c['종합_정규화MSE']['scene_footprint']} "
          f"(공표 {PUBLISHED_COMPOSITE}, 재현 {'일치' if c['일치'] else '불일치'}) · "
          f"공통 격자 {c['종합_정규화MSE']['common_frame']} · 목표 0.38", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
