#!/usr/bin/env python3
"""수체 면적 시계열 모델 직접 비교 실험 — persistence 말고 다른 모델을 쓰고 검증할 수 있나.

두 갈래로 답한다.

A. **공통 격자 면적 계열 위의 직접 모델** — 장면별 자기 footprint 로 잰 ``scenes.json``
   ``water_area_km2`` 는 날짜마다 촬영 범위가 달라 계열로 비교할 수 없다. 센서별 **공통 격자**
   마스크(``regions/<id>/satellite/masks/<sensor>/<date>.png``, 127 초과 화소 × ``frame.pixel_area_m2``)
   로 다시 재면 비교 가능한 계열이 된다. 그 계열에서 후보 7종(+지점별 수위 회귀)을
   **LOO(leave-one-out)** 로 돌려 km² 단위 RMSE·MAE 와 persistence 대비 skill 을 낸다.
   표본이 부산 센서당 4개, 대청 10개뿐이라 **모델을 가릴 수 없다** — 수치와 함께 그 판정을 적는다.

B. **수위→면적 다리(rating curve)** — 면적이 같은 날 수위의 함수로 설명되면, 10개년 **일별**
   수위로 파생 면적 일계열을 만들어 배포와 같은 13종 예측 파이프라인·같은 분할
   (학습 ≤2018 · 검증 2019~2023 · 시험 2024)로 돌리고, 위성 관측일에서 예보 파생 면적 대
   실측 공통격자 면적의 오차를 (검량 오차)+(수위 예보 오차)로 분해한다. 다리가 서지 않는
   조합(대청 음의 상관, 부산 n=4·수위 변동폭 1~4 cm)에는 **모델을 내지 않는다**.

산출:
  regions/<id>/models/gauge/experiments/area_models.json   (schema watercast.satellite.experiment.area-models/1)
  docs/presentation/면적모델_실험.md

    .venv/bin/python scripts/eval_area_models_experiment.py
    .venv/bin/python scripts/eval_area_models_experiment.py --region nakdong_busan
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.gauge.forecasters import DailySeries, ForecasterRegistry, build_forecaster_registry  # noqa: E402
from backend.app.gauge.regions import RegionRegistry  # noqa: E402
from backend.app.gauge.training import Split, _clean, metrics, origins_for_targets, train_station  # noqa: E402

EXPERIMENT_SCHEMA = "watercast.satellite.experiment.area-models/1"
EXPERIMENT_FILENAME = "area_models.json"
DOCS_DEFAULT = ROOT / "docs" / "presentation" / "면적모델_실험.md"
MASK_WATER_THRESHOLD = 127  # PNG 0/255 이진 마스크 — 127 초과가 물
DEFAULT_PLUGINS = (
    "backend.app.gauge.plugins.neural:NeuralMLP,backend.app.gauge.plugins.neural:NeuralLSTM,"
    "backend.app.gauge.plugins.neural:NeuralGRU,backend.app.gauge.plugins.neural:NeuralTCN,"
    "backend.app.gauge.plugins.prophet_hybrid:ProphetHybrid"
)
BRIDGE_HORIZONS: tuple[int, ...] = (1, 3, 5, 7, 10, 14, 30)
BRIDGE_END_TO_END_HORIZONS: tuple[int, ...] = (1, 7)
REPEAT_GAP_DAYS = 2  # 이 간격 이하의 인접 관측쌍 = 관측 잡음 대용(수위가 거의 안 변한 날)

# 후보 이름 — UI·문서가 그대로 쓴다.
CANDIDATE_KO: dict[str, str] = {
    "persistence": "직전 관측 유지 (기준선)",
    "series_mean": "계열 평균",
    "linear_trend": "시간 선형추세",
    "doy_climatology": "연중 시기 평년값 (day-of-year)",
    "ar1_mean_reversion": "AR(1) 평균회귀",
    "level_linear": "같은 날 수위 선형회귀",
    "level_isotonic": "같은 날 수위 단조회귀 (PAVA)",
}

# 다리(rating curve) 를 공개해도 되는지 판정하는 문턱 — 전부 통과해야 한다.
GATE = {
    "min_pairs": 6,
    "min_pearson_r": 0.6,
    "require_positive_slope": True,
    "min_level_range_m": 0.30,
    "min_loo_skill_vs_mean": 0.0,
    "max_implied_range_frac_of_frame": 0.5,
    "max_implied_range_over_observed": 3.0,
}
GATE_KO = {
    "min_pairs": f"짝지은 (면적, 수위) 관측 ≥ {GATE['min_pairs']}개",
    "min_pearson_r": f"피어슨 상관 r ≥ {GATE['min_pearson_r']:+.2f}",
    "require_positive_slope": "기울기 > 0 (수위가 오르면 면적도 넓어져야 한다)",
    "min_level_range_m": f"관측일 수위 변동폭 ≥ {GATE['min_level_range_m']:.2f} m (일평균 수위 잡음보다 커야 한다)",
    "min_loo_skill_vs_mean": "검량곡선의 LOO 오차가 '면적 평균'보다 작다",
    "max_implied_range_frac_of_frame": f"10개년 수위폭을 넣었을 때 파생 면적폭 ≤ 공통격자 면적의 {GATE['max_implied_range_frac_of_frame'] * 100:.0f} %",
    "max_implied_range_over_observed": f"그 파생 면적폭 ≤ 실측 면적폭의 {GATE['max_implied_range_over_observed']:.0f}배",
}


# --- 공통 격자 면적 계열 -------------------------------------------------------------------------


def _area_from_mask(path: Path, pixel_area_m2: float) -> float:
    from PIL import Image

    arr = np.array(Image.open(path).convert("L"))
    return float(int((arr > MASK_WATER_THRESHOLD).sum()) * float(pixel_area_m2) / 1e6)


def _masks_for_sensor(sat_dir: Path, scenes: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    from PIL import Image

    out: dict[str, np.ndarray] = {}
    for scene in scenes:
        frame = scene.get("frame") or {}
        path = sat_dir / str(frame.get("path") or "")
        if path.is_file():
            out[str(scene["date"])[:10]] = np.array(Image.open(path).convert("L")) > MASK_WATER_THRESHOLD
    return out


def _load_area_series_file(sat_dir: Path) -> dict[str, dict[str, float]] | None:
    """다른 도구가 남긴 ``satellite/area_series.json`` 을 관대하게 읽는다 → {sensor: {date: km²}}.

    스키마가 확정되지 않았으므로 흔한 모양을 모두 받는다: 최상위 ``series``/``sensors``/``records``/``rows``,
    센서별 사전 또는 ``sensor`` 열을 가진 평면 목록, 면적 키는 ``area_km2``/``water_area_km2``/
    ``common_area_km2``/``area``. 읽을 수 없으면 None 을 돌려주고 호출자가 마스크에서 다시 잰다.
    """

    path = sat_dir / "area_series.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # 공통 격자 면적 열 이름 후보 — 앞쪽이 우선. `area_km2_common_frame` 이 build_common_frame_area.py 의 이름이다.
    area_keys = ("area_km2_common_frame", "common_frame_area_km2", "area_km2", "water_area_km2", "common_area_km2", "area")
    out: dict[str, dict[str, float]] = {}

    def add(sensor: Any, row: Mapping[str, Any]) -> None:
        date = str(row.get("date") or row.get("scene_date") or "")[:10]
        value = next((row[k] for k in area_keys if isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool)), None)
        sensor_id = str(sensor or row.get("sensor") or "").strip()
        if date and sensor_id and value is not None:
            out.setdefault(sensor_id, {})[date] = float(value)

    def rows_of(value: Any) -> list[Mapping[str, Any]]:
        """센서 블록이 목록이든 {rows|points|records|series: [...]} 사전이든 행 목록으로."""

        if isinstance(value, list):
            return [r for r in value if isinstance(r, Mapping)]
        if isinstance(value, Mapping):
            for key in ("rows", "points", "records", "series"):
                inner = value.get(key)
                if isinstance(inner, list):
                    return [r for r in inner if isinstance(r, Mapping)]
            if any(k in value for k in area_keys):
                return [value]
        return []

    block = payload if isinstance(payload, Mapping) else {}
    containers = [block.get(k) for k in ("sensors", "series", "records", "rows", "points")] + [block]
    for container in containers:
        if isinstance(container, list):
            for row in container:
                if isinstance(row, Mapping):
                    add(None, row)
        elif isinstance(container, Mapping):
            for sensor, value in container.items():
                for row in rows_of(value):
                    add(sensor, row)
        if out:
            break
    return out or None


def build_area_series(region_dir: Path) -> dict[str, dict[str, Any]]:
    """센서별 공통 격자 면적 계열. ``satellite/area_series.json`` 이 있으면 그 값을 쓰고 없으면 마스크에서 잰다."""

    sat_dir = region_dir / "satellite"
    scenes_path = sat_dir / "scenes.json"
    if not scenes_path.is_file():
        return {}
    scenes = (json.loads(scenes_path.read_text(encoding="utf-8")) or {}).get("scenes") or []
    from_file = _load_area_series_file(sat_dir)
    out: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        frame = scene.get("frame") or {}
        if not frame.get("path"):
            continue
        sensor = str(scene.get("sensor") or "")
        date = str(scene.get("date") or "")[:10]
        mask_path = sat_dir / str(frame["path"])
        if not mask_path.is_file():
            continue
        prior = (from_file or {}).get(sensor, {}).get(date)
        recomputed = _area_from_mask(mask_path, float(frame["pixel_area_m2"]))
        block = out.setdefault(sensor, {"sensor": sensor, "points": [], "frame": None, "source": "masks"})
        block["points"].append({
            "date": date,
            "area_km2": round(float(prior) if prior is not None else recomputed, 4),
            "area_km2_recomputed": round(recomputed, 4),
            "scene_own_footprint_area_km2": scene.get("water_area_km2"),
            "scene_valid_area_km2": scene.get("valid_area_km2"),
        })
        if block["frame"] is None:
            block["frame"] = {
                "width": int(frame["width"]), "height": int(frame["height"]),
                "pixel_area_m2": float(frame["pixel_area_m2"]),
                "bounds_epsg32652": list(frame.get("bounds_epsg32652") or []),
                "frame_area_km2": round(int(frame["width"]) * int(frame["height"]) * float(frame["pixel_area_m2"]) / 1e6, 2),
                "water_threshold": MASK_WATER_THRESHOLD,
            }
        if prior is not None:
            block["source"] = "satellite/area_series.json"
    for block in out.values():
        block["points"].sort(key=lambda p: p["date"])
        block["n"] = len(block["points"])
        diffs = [abs(float(p["area_km2"]) - float(p["area_km2_recomputed"])) for p in block["points"]]
        block["max_abs_diff_vs_recompute_km2"] = round(max(diffs), 4) if diffs else 0.0
    return out


def mask_consistency(sat_dir: Path, scenes: Sequence[Mapping[str, Any]], sensor: str) -> dict[str, Any]:
    """같은 센서 마스크들의 겹침 진단 — 계열의 '변동' 이 탐지 흔들림보다 큰지 보는 잣대."""

    subset = [s for s in scenes if str(s.get("sensor")) == sensor]
    masks = _masks_for_sensor(sat_dir, subset)
    if len(masks) < 2:
        return {}
    px = float((subset[0].get("frame") or {}).get("pixel_area_m2") or 0.0)
    dates = sorted(masks)
    stack = np.stack([masks[d] for d in dates])
    km2 = lambda m: float(m.sum()) * px / 1e6  # noqa: E731
    ious = [float((masks[a] & masks[b]).sum()) / max(1.0, float((masks[a] | masks[b]).sum()))
            for i, a in enumerate(dates) for b in dates[i + 1:]]
    core, union = km2(stack.all(axis=0)), km2(stack.any(axis=0))
    batches: dict[str, list[str]] = {}
    for d in dates:
        batches.setdefault(d[:4], []).append(d)
    by_batch = {}
    for year, group in batches.items():
        if len(group) < 2:
            continue
        by_batch[year] = round(float(np.mean([
            float((masks[a] & masks[b]).sum()) / max(1.0, float((masks[a] | masks[b]).sum()))
            for i, a in enumerate(group) for b in group[i + 1:]
        ])), 4)
    return {
        "mean_pairwise_iou": round(float(np.mean(ious)), 4),
        "min_pairwise_iou": round(float(np.min(ious)), 4),
        "mean_pairwise_iou_by_batch_year": by_batch,
        "always_water_km2": round(core, 3),
        "ever_water_km2": round(union, 3),
        "churn_km2": round(union - core, 3),
        "note_ko": (
            "always_water = 모든 날짜에 물인 화소, ever_water = 한 번이라도 물인 화소. 그 차(churn)가 계열의 "
            "면적 변동폭보다 크면, 날짜별 면적 차이는 수위 변화가 아니라 탐지 흔들림일 수 있다."
        ),
    }


# --- 순수 수치 도우미 ---------------------------------------------------------------------------


def _ptp(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values)) if values.size else 0.0


def pava_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """단조 증가 등장성(isotonic) 회귀 — Pool Adjacent Violators. (정렬된 x, 같은 길이의 적합값)."""

    order = np.argsort(np.asarray(x, dtype=float))
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    values: list[float] = []
    weights: list[float] = []
    for value in ys:
        values.append(float(value))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2 = values.pop(), weights.pop()
            v1, w1 = values.pop(), weights.pop()
            values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
    fit: list[float] = []
    for value, weight in zip(values, weights):
        fit.extend([value] * int(round(weight)))
    return xs, np.asarray(fit, dtype=float)


def pava_predict(xs: np.ndarray, fit: np.ndarray, x0: float) -> float:
    """구간 밖은 양 끝 값으로 고정(단조 회귀의 표준 외삽)."""

    if xs.size == 0:
        return float("nan")
    if xs.size == 1:
        return float(fit[0])
    return float(np.interp(float(x0), xs, fit))


def sign_test(model_err: np.ndarray, base_err: np.ndarray) -> dict[str, Any]:
    """|오차| 가 기준선보다 작은 점의 수에 대한 양측 부호검정(정확 이항). n 이 작으면 최소 p 도 크다."""

    diff = np.abs(model_err) - np.abs(base_err)
    wins = int((diff < -1e-12).sum())
    losses = int((diff > 1e-12).sum())
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "ties": int(diff.shape[0]), "p_two_sided": None,
                "min_attainable_p": None, "significant_at_0_05": False}
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    p = min(1.0, 2.0 * tail)
    min_p = min(1.0, 2.0 * (1.0 / (2.0 ** n)))
    return {
        "wins": wins, "losses": losses, "ties": int(diff.shape[0]) - n,
        "p_two_sided": round(p, 4), "min_attainable_p": round(min_p, 4),
        "significant_at_0_05": bool(p < 0.05),
    }


def loo_predictions(
    dates: Sequence[str],
    areas: Sequence[float],
    levels: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    """후보마다 LOO 예측값 배열. 각 i 에 대해 i 를 뺀 나머지로만 적합/계산한다."""

    n = len(areas)
    a = np.asarray(areas, dtype=float)
    d0 = np.datetime64(dates[0], "D")
    t = np.asarray([float((np.datetime64(d, "D") - d0).astype(int)) for d in dates])
    doy = np.asarray([float(np.datetime64(d, "D").astype(object).timetuple().tm_yday) for d in dates])
    out: dict[str, np.ndarray] = {}

    def put(name: str, i: int, value: float) -> None:
        out.setdefault(name, np.full(n, np.nan))[i] = float(value)

    for i in range(n):
        keep = np.asarray([j for j in range(n) if j != i], dtype=int)
        ta, aa, da = t[keep], a[keep], doy[keep]
        earlier = keep[t[keep] < t[i]]
        anchor = int(earlier[-1]) if earlier.size else int(keep[0])
        put("persistence", i, a[anchor])
        put("series_mean", i, float(aa.mean()))
        if keep.size >= 2 and _ptp(ta) > 0:
            slope, intercept = np.polyfit(ta, aa, 1)
            put("linear_trend", i, slope * t[i] + intercept)
        gap_doy = np.abs(da - doy[i])
        gap_doy = np.minimum(gap_doy, 365.0 - gap_doy)
        weight = np.exp(-0.5 * (gap_doy / 30.0) ** 2)
        put("doy_climatology", i, float((weight * aa).sum() / weight.sum()) if weight.sum() > 1e-9 else float(aa.mean()))
        mu = float(aa.mean())
        num = den = 0.0
        for k in range(keep.size - 1):
            gap = max(1.0, float(ta[k + 1] - ta[k]))
            num += (aa[k] - mu) * (aa[k + 1] - mu) / gap
            den += ((aa[k] - mu) ** 2) / gap
        phi = float(np.clip(num / den, 0.0, 1.0)) if den > 1e-12 else 0.0
        gap = max(1.0, abs(float(t[i] - t[anchor])))
        put("ar1_mean_reversion", i, mu + (phi ** gap) * (a[anchor] - mu))
        for station, values in levels.items():
            lv = np.asarray(values, dtype=float)
            if np.isnan(lv).any() or keep.size < 2 or _ptp(lv[keep]) <= 1e-9:
                continue
            slope, intercept = np.polyfit(lv[keep], aa, 1)
            put(f"level_linear:{station}", i, slope * lv[i] + intercept)
            xs, fit = pava_fit(lv[keep], aa)
            put(f"level_isotonic:{station}", i, pava_predict(xs, fit, float(lv[i])))
    return out


def score_loo(preds: Mapping[str, np.ndarray], areas: Sequence[float], dates: Sequence[str]) -> dict[str, Any]:
    a = np.asarray(areas, dtype=float)
    base = preds["persistence"]
    base_mse = float(np.mean((base - a) ** 2))
    mean_pred = preds.get("series_mean")
    mean_mse = float(np.mean((mean_pred - a) ** 2)) if mean_pred is not None else None
    out: dict[str, Any] = {}
    for name, pred in preds.items():
        err = pred - a
        mse = float(np.mean(err ** 2))
        out[name] = {
            "family": name.split(":", 1)[0],
            "name_ko": CANDIDATE_KO.get(name.split(":", 1)[0], name),
            "level_station": name.split(":", 1)[1] if ":" in name else None,
            "n": int(a.shape[0]),
            "rmse_km2": round(math.sqrt(mse), 4),
            "mae_km2": round(float(np.mean(np.abs(err))), 4),
            "mse": round(mse, 6),
            "skill_vs_persistence": (round(1.0 - mse / base_mse, 4) if base_mse > 0 else None),
            "skill_vs_series_mean": (round(1.0 - mse / mean_mse, 4) if mean_mse else None),
            "sign_test_vs_persistence": sign_test(err, base - a) if name != "persistence" else None,
            "per_point": [
                {"date": d, "truth_km2": round(float(t_), 4), "pred_km2": round(float(p_), 4), "err_km2": round(float(p_ - t_), 4)}
                for d, t_, p_ in zip(dates, a, pred)
            ],
        }
    return out


def rating_curve(levels: Sequence[float], areas: Sequence[float]) -> dict[str, Any]:
    """전체 점으로 적합한 선형 검량곡선 + LOO 불확실도."""

    lv = np.asarray(levels, dtype=float)
    ar = np.asarray(areas, dtype=float)
    n = int(lv.shape[0])
    slope, intercept = (np.polyfit(lv, ar, 1) if n >= 2 and _ptp(lv) > 1e-12 else (float("nan"), float("nan")))
    r = float(np.corrcoef(lv, ar)[0, 1]) if n >= 2 and _ptp(lv) > 1e-12 and _ptp(ar) > 1e-12 else float("nan")
    loo_err: list[float] = []
    for i in range(n):
        keep = np.asarray([j for j in range(n) if j != i], dtype=int)
        if keep.size < 2 or _ptp(lv[keep]) <= 1e-12:
            continue
        s_, b_ = np.polyfit(lv[keep], ar[keep], 1)
        loo_err.append(float(s_ * lv[i] + b_ - ar[i]))
    errs = np.asarray(loo_err, dtype=float)
    return {
        "n_pairs": n,
        "slope_km2_per_m": round(float(slope), 4) if math.isfinite(slope) else None,
        "intercept_km2": round(float(intercept), 4) if math.isfinite(intercept) else None,
        "pearson_r": round(r, 4) if math.isfinite(r) else None,
        "level_range_m": round(_ptp(lv), 4),
        "area_range_km2": round(_ptp(ar), 4),
        "loo": {
            "n": int(errs.shape[0]),
            "rmse_km2": round(float(np.sqrt(np.mean(errs ** 2))), 4) if errs.size else None,
            "mae_km2": round(float(np.mean(np.abs(errs))), 4) if errs.size else None,
            "max_abs_km2": round(float(np.max(np.abs(errs))), 4) if errs.size else None,
        },
    }


def repeat_pairs(points: Sequence[Mapping[str, Any]], levels: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    """며칠 안 떨어진 인접 관측쌍 — 수위가 거의 그대로인데 면적이 얼마나 달라지나(관측 잡음 대용)."""

    out: list[dict[str, Any]] = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        gap = int((np.datetime64(b["date"], "D") - np.datetime64(a["date"], "D")).astype(int))
        if gap > REPEAT_GAP_DAYS:
            continue
        out.append({
            "from": a["date"], "to": b["date"], "gap_days": gap,
            "delta_area_km2": round(float(b["area_km2"]) - float(a["area_km2"]), 4),
            "delta_level_m": {
                st: round(float(v[i + 1]) - float(v[i]), 4)
                for st, v in levels.items() if not (math.isnan(float(v[i])) or math.isnan(float(v[i + 1])))
            },
        })
    return out


# --- 지역 실행 ---------------------------------------------------------------------------------


def station_levels_on(rows: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> list[float]:
    by_date: dict[str, float] = {}
    for row in rows:
        raw = str(row.get("date") or "")
        if len(raw) == 8 and raw.isdigit():
            raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        value = row.get("water_level_m")
        if value is not None:
            try:
                by_date[raw[:10]] = float(value)
            except (TypeError, ValueError):
                continue
    return [by_date.get(d, float("nan")) for d in dates]


def stations_in_frame(series_set: Any, frame: Mapping[str, Any]) -> list[str]:
    """공통 격자 상자 안에 들어오는 수위 지점만 — 화면 밖 지점으로 면적을 설명하지 않는다."""

    bounds = list(frame.get("bounds_epsg32652") or [])
    if len(bounds) != 4:
        return [s.id for s in series_set.stations]
    try:
        from rasterio.warp import transform
    except ImportError:
        return [s.id for s in series_set.stations]
    out: list[str] = []
    for station in series_set.stations:
        if station.lat is None or station.lon is None:
            continue
        xs, ys = transform("EPSG:4326", "EPSG:32652", [float(station.lon)], [float(station.lat)])
        if bounds[0] <= xs[0] <= bounds[2] and bounds[1] <= ys[0] <= bounds[3]:
            out.append(station.id)
    return out


def evaluate_gate(curve: Mapping[str, Any], frame: Mapping[str, Any], full_level_range_m: float) -> dict[str, Any]:
    slope = curve.get("slope_km2_per_m")
    r = curve.get("pearson_r")
    loo_rmse = (curve.get("loo") or {}).get("rmse_km2")
    area_range = float(curve.get("area_range_km2") or 0.0)
    frame_area = float(frame.get("frame_area_km2") or 0.0)
    implied = abs(float(slope)) * float(full_level_range_m) if slope is not None else None
    # '면적 평균' 대비: 검량곡선의 LOO RMSE 가 면적 표준편차보다 작아야 의미가 있다
    passes = {
        "min_pairs": int(curve["n_pairs"]) >= GATE["min_pairs"],
        "min_pearson_r": (r is not None and r >= GATE["min_pearson_r"]),
        "require_positive_slope": (slope is not None and slope > 0),
        "min_level_range_m": float(curve.get("level_range_m") or 0.0) >= GATE["min_level_range_m"],
        "min_loo_skill_vs_mean": (loo_rmse is not None and area_range > 0 and loo_rmse < area_range / 2.0),
        "max_implied_range_frac_of_frame": (implied is not None and frame_area > 0 and implied <= GATE["max_implied_range_frac_of_frame"] * frame_area),
        "max_implied_range_over_observed": (implied is not None and area_range > 0 and implied <= GATE["max_implied_range_over_observed"] * area_range),
    }
    reasons = [GATE_KO[k] for k, ok in passes.items() if not ok]
    return {
        "passes": passes,
        "defensible": all(passes.values()),
        "failed_criteria_ko": reasons,
        "implied_area_range_km2": round(implied, 2) if implied is not None else None,
        "implied_over_frame_pct": round(100.0 * implied / frame_area, 1) if implied is not None and frame_area > 0 else None,
        "gauge_level_range_10y_m": round(float(full_level_range_m), 3),
    }


def _derive_series(level_series: DailySeries, slope: float, intercept: float) -> DailySeries:
    y = intercept + slope * level_series.y
    derived = DailySeries.from_arrays(level_series.dates, y)
    if level_series.exog:
        derived = DailySeries(dates=derived.dates, y=derived.y, filled=derived.filled, doy=derived.doy,
                              exog=dict(level_series.exog))
    return derived


def run_bridge_demo(
    series_set: Any,
    forecasters: ForecasterRegistry,
    *,
    sensor: str,
    station_id: str,
    curve: Mapping[str, Any],
    gate: Mapping[str, Any],
    split: Split,
    points: Sequence[Mapping[str, Any]],
    sat_levels: Sequence[float],
    horizon_max: int,
) -> dict[str, Any]:
    """파생 면적 일계열에 배포와 같은 파이프라인을 돌린다. 판정이 '방어 불가' 여도 수치는 남긴다."""

    slope = float(curve["slope_km2_per_m"])
    intercept = float(curve["intercept_km2"])
    level_series = series_set.series[station_id]
    derived = _derive_series(level_series, slope, intercept)
    report = tuple(h for h in BRIDGE_HORIZONS if h <= horizon_max)
    area_art = train_station(derived, split, forecasters, horizon_max=horizon_max, report_horizons=report)
    level_art = train_station(level_series, split, forecasters, horizon_max=horizon_max, report_horizons=report)

    horizons: dict[str, Any] = {}
    for horizon in report:
        block = area_art["horizons"].get(str(horizon))
        lblock = level_art["horizons"].get(str(horizon))
        if not block:
            continue
        horizons[str(horizon)] = {
            "forecaster_id": block["forecaster_id"],
            "forecaster_name": block.get("forecaster_name"),
            "hparams": block.get("hparams"),
            "n_test": block.get("n_test"),
            "test_rmse_km2": round(float(block["test"]["rmse"]), 4),
            "test_mae_km2": round(float(block["test"]["mae"]), 4),
            "persistence_test_rmse_km2": round(float((block.get("test_baselines") or {}).get("persistence", {}).get("rmse", float("nan"))), 4),
            "skill_vs_persistence": block.get("skill_vs_persistence"),
            "level_equivalent": None if not lblock else {
                "forecaster_id": lblock["forecaster_id"],
                "test_rmse_m": round(float(lblock["test"]["rmse"]), 4),
                "skill_vs_persistence": lblock.get("skill_vs_persistence"),
                "test_rmse_m_times_slope_km2": round(float(lblock["test"]["rmse"]) * abs(slope), 4),
            },
        }
    errored = sorted({c["forecaster_id"] for c in (area_art["horizons"].get(str(report[0])) or {}).get("candidates", []) if "error" in c})
    families = sorted({c["forecaster_id"] for c in (area_art["horizons"].get(str(report[0])) or {}).get("candidates", [])})

    # 끝단 대조: 위성 관측일의 예보 파생 면적 vs 실측 공통격자 면적. 파라미터는 **학습 구간만**으로 적합한다.
    rows: list[dict[str, Any]] = []
    for horizon in BRIDGE_END_TO_END_HORIZONS:
        block = area_art["horizons"].get(str(horizon))
        if not block:
            continue
        forecaster = forecasters.get(block["forecaster_id"])
        train_origins = origins_for_targets(derived, horizon, None, split.train_end)
        try:
            params = forecaster.fit(derived, train_origins, horizon, block.get("hparams") or {})
        except Exception as exc:  # noqa: BLE001 — 학습 구간만으로는 못 맞출 수도 있다
            rows.append({"horizon": horizon, "error": str(exc)})
            continue
        for point, level_obs in zip(points, sat_levels):
            if math.isnan(float(level_obs)):
                continue
            try:
                target = derived.index_of(np.datetime64(point["date"], "D"))
            except (ValueError, IndexError):
                continue
            origin = target - horizon
            if origin < 0:
                continue
            pred = forecaster.predict(derived, np.array([origin]), horizon, params)
            forecast_area = float(pred[0]) if pred.shape[0] else float("nan")
            rating_area = intercept + slope * float(level_obs)
            sat_area = float(point["area_km2"])
            level_forecast = (forecast_area - intercept) / slope if abs(slope) > 1e-12 else float("nan")
            rows.append({
                "date": point["date"], "horizon": horizon,
                "origin_date": str(derived.dates[origin]),
                "satellite_area_km2": round(sat_area, 4),
                "rating_area_km2": round(rating_area, 4),
                "forecast_area_km2": round(forecast_area, 4) if math.isfinite(forecast_area) else None,
                "level_observed_m": round(float(level_obs), 4),
                "level_forecast_m": round(level_forecast, 4) if math.isfinite(level_forecast) else None,
                "rating_error_km2": round(rating_area - sat_area, 4),
                "level_forecast_error_km2": round(forecast_area - rating_area, 4) if math.isfinite(forecast_area) else None,
                "total_error_km2": round(forecast_area - sat_area, 4) if math.isfinite(forecast_area) else None,
                "in_split": ("test" if np.datetime64(point["date"], "D") >= split.test_start
                             else "val" if np.datetime64(point["date"], "D") >= split.val_start else "train"),
            })
    # 요약: (1) 파생 면적 skill 이 수위 skill 과 같은가 — 같다면 다리는 예측력을 보태지 않고 단위만 바꾼 것이다
    diffs = [abs(float(e["skill_vs_persistence"]) - float((e.get("level_equivalent") or {}).get("skill_vs_persistence")))
             for e in horizons.values()
             if isinstance(e.get("skill_vs_persistence"), (int, float)) and isinstance((e.get("level_equivalent") or {}).get("skill_vs_persistence"), (int, float))]
    by_horizon: dict[str, Any] = {}
    for horizon in BRIDGE_END_TO_END_HORIZONS:
        subset = [r for r in rows if r.get("horizon") == horizon and r.get("total_error_km2") is not None]
        if not subset:
            continue
        rms = lambda key: round(float(np.sqrt(np.mean([float(r[key]) ** 2 for r in subset]))), 4)  # noqa: E731
        by_horizon[f"D+{horizon}"] = {
            "n": len(subset),
            "rating_error_rmse_km2": rms("rating_error_km2"),
            "level_forecast_error_rmse_km2": rms("level_forecast_error_km2"),
            "total_error_rmse_km2": rms("total_error_km2"),
        }
    rating_loo = (curve.get("loo") or {}).get("rmse_km2")
    warnings_ko = [
        "검량곡선은 바로 이 위성 관측일 4~10점으로 적합했으므로 위 표의 '검량 오차' 는 **내부 표본(in-sample)** 값이라 낙관적이다. "
        f"같은 곡선의 LOO RMSE 는 {rating_loo} km² 이고 이 계열의 전체 면적 변동폭은 {curve.get('area_range_km2')} km² 다 — "
        "즉 검량 오차만으로 이미 신호 크기에 맞먹는다.",
    ]
    if diffs:
        warnings_ko.append(
            f"파생 면적의 시험 skill 은 같은 분할 **수위** 모델의 skill 과 최대 {max(diffs):.4f} 차이다(사실상 동일). "
            "파생 면적 예측은 수위 예측을 아핀 변환으로 다시 쓴 것일 뿐이며, 면적이 예측 가능하다는 증거가 **아니다**."
        )
    derived_min = float(np.nanmin(derived.y))
    if derived_min < 0:
        warnings_ko.append(
            f"10개년 수위를 이 곡선에 넣으면 파생 면적이 최저 {derived_min:.1f} km² (음수)까지 내려간다 — 곡선이 관측 구간 밖에서 물리적으로 무의미하다는 증거다."
        )
    return {
        "published": False,
        "defensible": bool(gate.get("defensible")),
        "demonstration_only": True,
        "summary": {
            "skill_max_abs_diff_vs_level": (round(max(diffs), 6) if diffs else None),
            "end_to_end_rmse_km2": by_horizon,
            "rating_loo_rmse_km2": rating_loo,
            "warnings_ko": warnings_ko,
        },
        "sensor": sensor,
        "level_station": station_id,
        "rating_curve": dict(curve),
        "gate": dict(gate),
        "derived_series": {
            "formula_ko": f"파생면적(km²) = {intercept:.3f} + {slope:.3f} × 수위(m) @ {station_id}",
            "period": [str(derived.dates[0]), str(derived.dates[-1])],
            "n_days": int(derived.n),
            "n_valid": int(derived.valid.sum()),
            "min_km2": round(float(np.nanmin(derived.y)), 3),
            "max_km2": round(float(np.nanmax(derived.y)), 3),
        },
        "split": split.to_dict(),
        "forecasters_tried": families,
        "forecasters_unavailable": errored,
        "horizons": horizons,
        "end_to_end": {
            "params_fit_on_ko": f"학습 구간만(≤ {split.train_end}) — 위성 관측일이 검증 구간이라 누수를 막기 위해",
            "decomposition_ko": "총오차 = (검량 오차: 실측 수위를 검량곡선에 넣은 면적 − 위성 실측 면적) + (수위 예보 오차: 예보 수위를 넣은 면적 − 실측 수위를 넣은 면적)",
            "rows": rows,
        },
    }


def run_region(
    registry: RegionRegistry,
    forecasters: ForecasterRegistry,
    region_id: str,
    *,
    horizon_max: int = 30,
    write: bool = True,
    skip_bridge: bool = False,
) -> dict[str, Any]:
    record = registry.get(region_id)
    started = time.time()
    area_blocks = build_area_series(record.path)
    if not area_blocks:
        raise RuntimeError(f"{region_id}: 공통 격자 마스크가 없어 면적 계열을 만들 수 없습니다")
    scenes = (json.loads((record.path / "satellite" / "scenes.json").read_text(encoding="utf-8")) or {}).get("scenes") or []
    series_set = registry.load_series(region_id)
    split = Split.from_mapping(record.split) if record.split else None

    series_out: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []
    best_bridge: tuple[float, dict[str, Any]] | None = None

    for sensor, block in sorted(area_blocks.items()):
        points = block["points"]
        dates = [p["date"] for p in points]
        areas = [float(p["area_km2"]) for p in points]
        frame = block["frame"] or {}
        in_frame = stations_in_frame(series_set, frame)
        levels = {sid: station_levels_on(series_set.raw_rows.get(sid) or [], dates) for sid in in_frame}
        levels = {sid: v for sid, v in levels.items() if not any(math.isnan(x) for x in v)}
        for index, point in enumerate(points):
            point["levels_m"] = {sid: round(float(values[index]), 4) for sid, values in levels.items()}

        preds = loo_predictions(dates, areas, levels)
        candidates = score_loo(preds, areas, dates)
        ranked = sorted(candidates, key=lambda k: candidates[k]["rmse_km2"])
        curves: dict[str, Any] = {}
        for sid, values in levels.items():
            curve = rating_curve(values, areas)
            full = series_set.series[sid].y
            gate = evaluate_gate(curve, frame, _ptp(full[~np.isnan(full)]))
            curves[sid] = {**curve, "gate": gate}
            gate_rows.append({"sensor": sensor, "level_station": sid, **{k: curve[k] for k in ("n_pairs", "pearson_r", "slope_km2_per_m", "level_range_m")},
                              "loo_rmse_km2": (curve["loo"] or {}).get("rmse_km2"), **gate})
            if curve["slope_km2_per_m"] and curve["slope_km2_per_m"] > 0 and curve["pearson_r"] is not None:
                score = (1 if gate["defensible"] else 0, curve["pearson_r"], curve["n_pairs"])
                if best_bridge is None or score > best_bridge[0]:
                    best_bridge = (score, {"sensor": sensor, "station": sid, "curve": curve, "gate": gate,
                                           "points": points, "levels": values})

        area_range = _ptp(np.asarray(areas))
        consistency = mask_consistency(record.path / "satellite", scenes, sensor)
        churn = float(consistency.get("churn_km2") or 0.0)
        reps = repeat_pairs(points, levels)
        max_rep = max((abs(float(r["delta_area_km2"])) for r in reps), default=None)
        verdict: list[str] = []
        if len(areas) < 6:
            verdict.append(f"관측 {len(areas)}개 — LOO 시험점이 {len(areas)}개뿐이라 후보 간 차이가 통계적으로 구분되지 않는다(부호검정 최소 p = {2 / 2 ** len(areas):.3f} > 0.05).")
        if churn and churn > area_range:
            verdict.append(f"탐지 흔들림(ever−always = {churn:.2f} km²)이 계열의 면적 변동폭({area_range:.2f} km²)보다 크다 — 날짜별 차이를 수위 변화로 읽으면 안 된다.")
        if max_rep is not None and max_rep >= 0.5 * area_range:
            rep = max((r for r in reps), key=lambda r: abs(float(r["delta_area_km2"])))
            verdict.append(
                f"{rep['gap_days']}일 간격 재방문({rep['from']}→{rep['to']})에서 면적이 {abs(float(rep['delta_area_km2'])):.2f} km² 달라졌다 — "
                f"계열 전체 변동폭 {area_range:.2f} km² 의 {abs(float(rep['delta_area_km2'])) / area_range * 100:.0f} %. 이만큼이 관측 잡음 하한이다."
            )
        series_out[sensor] = {
            "source": block["source"],
            "max_abs_diff_vs_recompute_km2": block.get("max_abs_diff_vs_recompute_km2"),
            "frame": frame,
            "n": len(areas),
            "period": [dates[0], dates[-1]],
            "points": points,
            "area_range_km2": round(area_range, 4),
            "area_mean_km2": round(float(np.mean(areas)), 4),
            "gauge_stations_in_frame": sorted(levels),
            "mask_consistency": consistency,
            "repeat_pairs": reps,
            "observation_noise_floor_km2": round(max_rep, 4) if max_rep is not None else None,
            "rating_curves": curves,
            "candidates": candidates,
            "ranking_by_loo_rmse": ranked,
            "best_by_loo": ranked[0] if ranked else None,
            "any_candidate_significant_vs_persistence": any(
                (c.get("sign_test_vs_persistence") or {}).get("significant_at_0_05") for c in candidates.values()
            ),
            "verdict_ko": verdict,
        }

    bridge: dict[str, Any] = {
        "gate_criteria_ko": dict(GATE_KO),
        "gate_thresholds": dict(GATE),
        "evaluated": gate_rows,
        "any_defensible": any(row["defensible"] for row in gate_rows),
        "demonstration": None,
        "note_ko": (
            "다리가 하나도 문턱을 넘지 못하면 파생 면적 예측 모델을 공개하지 않는다. 아래 demonstration 은 "
            "'문턱을 넘었다면 이렇게 나온다' 를 보여 주는 **시연**일 뿐이며 성과 수치로 인용하면 안 된다."
        ),
    }
    if best_bridge is not None and split is not None and not skip_bridge:
        pick = best_bridge[1]
        bridge["demonstration"] = run_bridge_demo(
            series_set, forecasters, sensor=pick["sensor"], station_id=pick["station"],
            curve=pick["curve"], gate=pick["gate"], split=split,
            points=pick["points"], sat_levels=pick["levels"], horizon_max=horizon_max,
        )

    payload = _clean({
        "schema": EXPERIMENT_SCHEMA,
        "region_id": region_id,
        "region_name": record.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/eval_area_models_experiment.py",
        "frame_note_ko": (
            "면적은 센서별 **공통 격자**(같은 EPSG:32652 상자·같은 H×W) 마스크에서 다시 잰 값이다. "
            "scenes.json 의 water_area_km2 는 장면마다 자기 footprint 로 잰 값이라 날짜 간 비교가 되지 않는다."
        ),
        "series": series_out,
        "bridge": bridge,
        "notes_ko": build_notes(region_id, series_out, bridge),
        "verdict_ko": build_verdict(series_out, bridge),
        "elapsed_s": round(time.time() - started, 1),
    })
    if write:
        builtin = registry.builtin_record(region_id) or record
        target = builtin.artifacts_dir / "experiments" / EXPERIMENT_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        payload["written_to"] = str(target)
    return payload


def build_notes(region_id: str, series_out: Mapping[str, Any], bridge: Mapping[str, Any]) -> list[str]:
    notes = [
        "면적 계열은 센서별 공통 격자 마스크에서 다시 잰 값이다. scenes.json 의 장면별 water_area_km2 는 촬영 범위가 날짜마다 달라 계열 비교에 쓸 수 없다 "
        "(부산 PlanetScope 2020-03-25 는 03-12 보다 약 8.5 km 남쪽을 찍어 하구·바다를 더 담았고 그래서 raw 65.3 km² 다. 그 장면의 GT 수동라벨은 65.063 km² 로 **탐지는 맞았다** — 틀린 것은 계열이다).",
        "LOO(leave-one-out) = 관측 하나를 빼고 나머지로만 적합해 그 하나를 맞히기를 관측 수만큼 반복한 것이다. 따라서 시험점 수 = 관측 수이며, 표본이 4~10개인 여기서는 **순위가 우연히 뒤집힐 수 있다**.",
        "부호검정(sign test) 최소 달성 가능 p 는 n=4 에서 0.125, n=10 에서 0.002 다. 즉 **n=4 계열에서는 어떤 모델도 persistence 보다 유의하게 낫다고 말할 수 없다** — 표의 skill 은 순위 참고용이다.",
        "같은 날 수위 회귀(level_linear·level_isotonic)의 수위는 **공통 격자 상자 안에 들어오는 수위계 지점**만 썼다. 상자 밖 지점으로 면적을 설명하지 않는다.",
        "LOO 의 '직전 관측 유지' 는 뺀 날짜보다 앞선 관측 중 가장 가까운 값을 쓴다. 계열의 **첫 관측**은 앞선 관측이 없어 예외로 그 다음 관측을 쓰므로, 그 한 점에서는 기준선이 미래를 본다(기준선에 유리한 쪽 예외다).",
        "**LOO 는 내삽(interpolation) 평가이지 예보(forecast) 평가가 아니다.** 기준선을 뺀 모든 후보"
        "(계열 평균·시간 선형추세·연중 시기 평년값·AR(1)·수위 회귀)는 뺀 날짜보다 **나중** 관측까지 포함한 나머지 전부로 적합한다. "
        "따라서 표의 skill 은 '그날을 앞에서부터 예보했을 때' 의 성적이 아니다. 실제로 목표일 **이전** 관측만 쓰도록 바꿔(전진연쇄) 다시 재면 "
        "대청 1위였던 시간 선형추세는 LOO 2.414 km²(skill +23.2 %) 에서 26.6 km²(persistence 2.51 km² 대비 skill 음수)로 무너지고, "
        "대청에서는 어떤 후보도 persistence 를 이기지 못한다. 이 표는 '표본이 모델을 가르지 못한다' 의 근거로만 읽고, "
        "예보 성능의 근거로 인용하면 안 된다.",
        "이 실험은 공식 성적표(evaluation.json·화면의 선택 모델·개선율)에 들어가지 않는다. 배포 면적 예측 모델을 새로 만들지도 않는다.",
    ]
    for sensor, block in series_out.items():
        for line in block.get("verdict_ko") or []:
            notes.append(f"{sensor}: {line}")
    if not bridge.get("any_defensible"):
        notes.append(
            "수위→면적 다리는 어느 조합도 문턱을 넘지 못했다. 따라서 파생 면적 예측 모델을 **공개하지 않는다**. "
            "bridge.demonstration 이 있다면 그것은 파이프라인이 도는지 보이는 시연이며 성과 수치가 아니다."
        )
    demo = bridge.get("demonstration") or {}
    notes += list(((demo.get("summary") or {}).get("warnings_ko")) or [])
    if demo.get("forecasters_unavailable"):
        notes.append(
            f"이 실험 환경에는 torch·prophet 이 없어 {', '.join(demo['forecasters_unavailable'])} 는 후보에서 빠졌다 "
            f"(시도 {len(demo.get('forecasters_tried') or [])}종 중 {len(demo['forecasters_unavailable'])}종 제외). "
            "배포 학습은 이 계열이 아니라 실제 수위 계열에서 그 모델들까지 포함해 돌린다."
        )
    if region_id == "geum_daecheong":
        notes.append(
            "대청: 공통 격자 면적이 대청댐 수위와 **음의 상관**(r≈−0.66)이다. 저수지에서 물리적으로 거꾸로다. "
            "원인은 수문이 아니라 배치별 탐지 안정도다 — 2019년 3~4월 5씬의 상호 IoU 평균이 0.61 인 데 비해 2022년 5월 5씬은 0.93 이고, "
            "2022 코어(5씬 모두 물)에서 2019 합집합을 뺀 넓이는 0.04 km² 에 불과하다. 즉 같은 수면을 2019 씬들이 제각기 덜 잡은 것이지 물이 늘어난 것이 아니다.",
        )
        notes.append(
            "대청 공통 격자는 134 km² 상자이고 최대 탐지 면적이 약 20 km² 다. 대청호 만수면적(약 72 km²)의 일부(댐 인근 구간)만 보고 있으므로 "
            "이 면적을 저수지 저수면적으로 읽어서는 안 된다.",
        )
    return notes


def build_verdict(series_out: Mapping[str, Any], bridge: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for sensor, block in sorted(series_out.items()):
        best = block.get("best_by_loo")
        cand = (block.get("candidates") or {}).get(best) or {}
        n = block.get("n")
        sig = "유의" if block.get("any_candidate_significant_vs_persistence") else "유의하지 않음"
        label = str(cand.get("name_ko", best))
        station = cand.get("level_station")
        caveat = ""
        if station:
            label = f"{label}({station})"
            curve = (block.get("rating_curves") or {}).get(station) or {}
            slope = curve.get("slope_km2_per_m")
            if isinstance(slope, (int, float)) and slope < 0:
                caveat = (f" 그 1위 후보의 검량 기울기는 {slope:+.1f} km²/m 로 **음수**다 — 수위가 오를수록 면적이 준다는 뜻이라 "
                          "물리적으로 거꾸로이고, 표본이 작아 잡음에 맞춘 결과로 봐야 한다.")
        lines.append(
            f"{sensor} (n={n}): LOO 최소 RMSE 는 {label} {cand.get('rmse_km2')} km² "
            f"(persistence {(block.get('candidates') or {}).get('persistence', {}).get('rmse_km2')} km², skill {cand.get('skill_vs_persistence')}). "
            f"persistence 대비 부호검정은 전 후보 {sig} — 이 표본으로는 모델을 가릴 수 없다.{caveat}"
        )
    lines.append(
        "결론: 위성 면적 계열 자체로는 어떤 모델도 **검증되지 않는다**(표본 부족). 검증 가능한 것은 10개년 일별 수위 계열이고, "
        "그 결과를 면적으로 옮기려면 검량곡선이 서야 하는데 "
        + ("여기서는 서지 않는다." if not bridge.get("any_defensible") else "그 조건을 만족하는 조합이 있다.")
    )
    return lines


# --- 문서 -------------------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".3f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) and not isinstance(value, bool) else "—"


def write_docs(results: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# 수체 면적 시계열 모델 — 직접 비교 실험 (표본이 적을 때 무엇을 말할 수 있나)",
        "",
        f"생성: {datetime.now(timezone.utc).isoformat(timespec='seconds')} · 스크립트 `scripts/eval_area_models_experiment.py` · "
        f"원본 JSON `regions/<id>/models/gauge/experiments/area_models.json`",
        "",
        "## 왜 이 실험인가",
        "",
        "\"persistence 말고 다른 모델을 쓰고 검증은 못하나\" 에 대한 답이다. 세 문장으로 요약하면:",
        "",
        "1. 위성 수체 면적 계열은 지역·센서당 관측이 4~10개뿐이라 **모델을 가릴 수 없다**. 아래 LOO 표는 수치를 다 보여 주지만, 그 수치로 우열을 주장할 수는 없다.",
        "2. 검증이 실제로 되는 것은 10개년 **일별 지상 수위** 계열이다(배포 성적표가 그것이다).",
        "3. 수위 성적을 면적으로 옮기는 다리(검량곡선)는 이 자료에서 서지 않는다 — 아래 '다리 판정' 표가 그 근거다.",
        "",
        "## 공통 격자 (읽기 전 주의)",
        "",
        "`scenes.json` 의 `water_area_km2` 는 **장면마다 자기 촬영 범위**로 잰 값이라 날짜 간 비교가 되지 않는다.",
        "부산 PlanetScope 2020-03-25 는 03-12 보다 약 8.5 km 남쪽을 찍어 하구·바다를 더 담았고, 그래서 raw 면적이 65.3 km² 로 튄다.",
        "그 장면의 GT 수동라벨이 65.063 km² 이므로 **탐지는 0.4 % 오차로 맞았다** — 틀린 것은 계열이다.",
        "센서별 공통 격자 마스크(`masks/<sensor>/<date>.png`, 127 초과 화소 × `frame.pixel_area_m2`)로 다시 재면 그 급증은 사라진다.",
        "",
    ]
    for region_id, payload in results.items():
        name = payload.get("region_name") or region_id
        lines += [f"## {name} (`{region_id}`)", ""]
        for sensor, block in sorted((payload.get("series") or {}).items()):
            frame = block.get("frame") or {}
            lines += [
                f"### {sensor} — 관측 {block.get('n')}개 ({block['period'][0]} ~ {block['period'][1]})",
                "",
                f"공통 격자 {frame.get('width')}×{frame.get('height')} · {frame.get('frame_area_km2')} km² · "
                f"면적 평균 {block.get('area_mean_km2')} km² · 변동폭 {block.get('area_range_km2')} km² · 계열 출처 `{block.get('source')}`"
                + (f" (마스크 재계산과 최대 차이 {block.get('max_abs_diff_vs_recompute_km2')} km²)" if block.get("max_abs_diff_vs_recompute_km2") is not None else ""),
                "",
                "| 날짜 | 공통격자 면적 (km²) | 장면 자기 footprint 면적 (km²) | 장면 유효 범위 (km²) |",
                "|---|---:|---:|---:|",
            ]
            for point in block.get("points") or []:
                lines.append(
                    f"| {point['date']} | {_fmt(point.get('area_km2'))} | {_fmt(point.get('scene_own_footprint_area_km2'))} | "
                    f"{_fmt(point.get('scene_valid_area_km2'), '.1f')} |"
                )
            consistency = block.get("mask_consistency") or {}
            lines += [
                "",
                f"마스크 일관성: 상호 IoU 평균 {consistency.get('mean_pairwise_iou')} (최소 {consistency.get('min_pairwise_iou')})"
                + (f" · 배치별 {consistency.get('mean_pairwise_iou_by_batch_year')}" if consistency.get("mean_pairwise_iou_by_batch_year") else "")
                + f" · 항상 물 {consistency.get('always_water_km2')} km² / 한 번이라도 물 {consistency.get('ever_water_km2')} km² "
                f"(흔들림 {consistency.get('churn_km2')} km²)",
                "",
                "| 후보 | LOO RMSE (km²) | LOO MAE (km²) | persistence 대비 skill | 부호검정 p (양측) |",
                "|---|---:|---:|---:|---:|",
            ]
            candidates = block.get("candidates") or {}
            for key in block.get("ranking_by_loo_rmse") or []:
                c = candidates[key]
                sign = c.get("sign_test_vs_persistence") or {}
                label = c["name_ko"] + (f" — {c['level_station']}" if c.get("level_station") else "")
                skill = c.get("skill_vs_persistence")
                lines.append(
                    f"| {label} | {_fmt(c.get('rmse_km2'))} | {_fmt(c.get('mae_km2'))} | "
                    f"{(f'{skill * 100:+.1f} %' if isinstance(skill, (int, float)) else '—')} | "
                    f"{_fmt(sign.get('p_two_sided'), '.3f')} |"
                )
            lines += [""]
            for line in block.get("verdict_ko") or []:
                lines.append(f"> {line}")
            lines.append("")
        bridge = payload.get("bridge") or {}
        lines += [
            "### 수위 → 면적 다리 판정",
            "",
            "| 센서 | 수위 지점 | n | r | 기울기 (km²/m) | 관측일 수위폭 (m) | 10년 수위폭을 넣으면 (km²) | 격자 대비 | 판정 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in bridge.get("evaluated") or []:
            lines.append(
                f"| {row['sensor']} | {row['level_station']} | {row['n_pairs']} | {_fmt(row.get('pearson_r'), '+.3f')} | "
                f"{_fmt(row.get('slope_km2_per_m'), '+.1f')} | {_fmt(row.get('level_range_m'), '.4f')} | "
                f"{_fmt(row.get('implied_area_range_km2'), '.0f')} | {_fmt(row.get('implied_over_frame_pct'), '.0f')} % | "
                f"{'통과' if row.get('defensible') else '탈락 — ' + '; '.join(row.get('failed_criteria_ko') or [])} |"
            )
        lines += ["", f"문턱: {' · '.join(GATE_KO.values())}", ""]
        demo = bridge.get("demonstration")
        if demo:
            lines += [
                f"#### 시연(공개 아님): {demo['sensor']} × {demo['level_station']}",
                "",
                f"`{demo['derived_series']['formula_ko']}` 로 만든 일별 파생 면적({demo['derived_series']['period'][0]} ~ "
                f"{demo['derived_series']['period'][1]}, {demo['derived_series']['n_valid']}일, "
                f"{demo['derived_series']['min_km2']} ~ {demo['derived_series']['max_km2']} km²)에 배포와 같은 분할·후보로 예측을 돌린 결과다.",
                "",
                "| 지평 | 선택 모델 | 시험 RMSE (km²) | persistence RMSE (km²) | skill | 같은 분할 수위 모델 skill |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for horizon, entry in sorted((demo.get("horizons") or {}).items(), key=lambda kv: int(kv[0])):
                level_eq = entry.get("level_equivalent") or {}
                lines.append(
                    f"| D+{horizon} | {entry.get('forecaster_id')} | {_fmt(entry.get('test_rmse_km2'))} | "
                    f"{_fmt(entry.get('persistence_test_rmse_km2'))} | {_fmt(entry.get('skill_vs_persistence'), '+.3f')} | "
                    f"{_fmt(level_eq.get('skill_vs_persistence'), '+.3f')} |"
                )
            rows = [r for r in (demo.get("end_to_end") or {}).get("rows") or [] if "error" not in r]
            if rows:
                lines += [
                    "",
                    "끝단 대조 — 위성 관측일의 예보 파생 면적 vs 실측 공통격자 면적 (총오차 = 검량 오차 + 수위 예보 오차):",
                    "",
                    "| 날짜 | 지평 | 위성 실측 (km²) | 검량 (km²) | 예보 (km²) | 검량 오차 | 수위 예보 오차 | 총오차 | 분할 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
                for row in rows:
                    lines.append(
                        f"| {row['date']} | D+{row['horizon']} | {_fmt(row.get('satellite_area_km2'))} | {_fmt(row.get('rating_area_km2'))} | "
                        f"{_fmt(row.get('forecast_area_km2'))} | {_fmt(row.get('rating_error_km2'), '+.3f')} | "
                        f"{_fmt(row.get('level_forecast_error_km2'), '+.3f')} | {_fmt(row.get('total_error_km2'), '+.3f')} | {row.get('in_split')} |"
                    )
            lines += ["", f"> {demo['end_to_end']['params_fit_on_ko']}", ""]
            summary = demo.get("summary") or {}
            if summary.get("end_to_end_rmse_km2"):
                lines += [
                    "| 지평 | 검량 오차 RMSE (km²) | 수위 예보 오차 RMSE (km²) | 총오차 RMSE (km²) |",
                    "|---|---:|---:|---:|",
                ]
                for horizon, entry in summary["end_to_end_rmse_km2"].items():
                    lines.append(
                        f"| {horizon} | {_fmt(entry.get('rating_error_rmse_km2'))} | "
                        f"{_fmt(entry.get('level_forecast_error_rmse_km2'))} | {_fmt(entry.get('total_error_rmse_km2'))} |"
                    )
                lines.append("")
            for warning in summary.get("warnings_ko") or []:
                lines.append(f"> ⚠️ {warning}")
            lines.append("")
        lines += ["### 정직 고지", ""]
        lines += [f"- {note}" for note in payload.get("notes_ko") or []]
        lines += [""]
        lines += [f"**판정**: {line}" for line in payload.get("verdict_ko") or []]
        lines += [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions-dir", default=str(ROOT / "regions"))
    parser.add_argument("--user-dir", default=str(ROOT / "backend" / "data" / "regions"))
    parser.add_argument("--region", action="append", help="지역 id (반복 가능). 비우면 위성 마스크가 있는 지역 전부")
    parser.add_argument("--plugins", default=DEFAULT_PLUGINS, help="GAUGE_FORECASTER_PLUGINS 형식 (없는 라이브러리는 후보에서 자동 제외)")
    parser.add_argument("--horizon-max", type=int, default=30)
    parser.add_argument("--skip-bridge", action="store_true", help="다리 시연(파생 면적 학습)을 건너뛴다")
    parser.add_argument("--docs-out", default=str(DOCS_DEFAULT))
    parser.add_argument("--no-docs", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="JSON 을 지역 디렉터리에 쓰지 않는다")
    args = parser.parse_args()

    registry = RegionRegistry(Path(args.regions_dir), Path(args.user_dir))
    forecasters = build_forecaster_registry(args.plugins)
    region_ids = args.region or [
        r.id for r in registry.list()
        if r.origin == "builtin" and (r.path / "satellite" / "masks").is_dir()
    ]
    results: dict[str, dict[str, Any]] = {}
    exit_code = 0
    for region_id in region_ids:
        print(f"== {region_id}")
        try:
            payload = run_region(registry, forecasters, region_id, horizon_max=args.horizon_max,
                                 write=not args.no_write, skip_bridge=args.skip_bridge)
        except Exception as exc:  # noqa: BLE001
            print(f"   건너뜀: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        results[region_id] = payload
        for sensor, block in sorted((payload.get("series") or {}).items()):
            best = block.get("best_by_loo")
            cand = (block.get("candidates") or {}).get(best) or {}
            print(f"   {sensor}: n={block['n']} 최저 LOO RMSE {cand.get('rmse_km2')} km² ({best}) · "
                  f"persistence {(block.get('candidates') or {}).get('persistence', {}).get('rmse_km2')} km²")
        bridge = payload.get("bridge") or {}
        print(f"   다리: 방어 가능 {bridge.get('any_defensible')} · 후보 {len(bridge.get('evaluated') or [])}개")
        print(f"   {payload.get('written_to')} · {payload.get('elapsed_s')}s")
    if results and not args.no_docs:
        for record in registry.list():
            if record.id in results:
                continue
            path = record.artifacts_dir / "experiments" / EXPERIMENT_FILENAME
            if path.is_file():
                try:
                    results[record.id] = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    pass
        write_docs(dict(sorted(results.items())), Path(args.docs_out))
        print(f"== 문서: {args.docs_out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
