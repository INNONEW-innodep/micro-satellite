#!/usr/bin/env python3
"""면적·수위 전망의 오차를 walk-forward로 실측한다.

MSE 하나만 보면 안 되는 자료다. 이 시계열들은 변동이 작아서 **아무 값이나
직전 관측을 그대로 내놓아도 MSE가 낮게 나온다.** 그래서 같은 분할에서 단순
기준선(직전값 유지·평균)도 함께 재고, 그 대비 개선율(skill)을 같이 낸다.
skill ≤ 0 이면 "MSE가 낮다"는 말이 모델의 공로가 아니라는 뜻이다.

평가 방식: 1-step walk-forward. 앞의 k개로 적합해 바로 다음 한 점을 맞히고,
k를 늘려 가며 반복한다. 관측이 4~8개뿐이라 시험점은 2~6개다 — 성능 주장이
아니라 자릿수 확인용이다.
"""

from __future__ import annotations

import datetime as dt
import glob
import io
import json
import math
import os
import sys
from collections.abc import Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ui_next"))

from forecast_bands import project_trend  # noqa: E402

MIN_TRAIN = 2


def _metrics(truth: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    n = len(truth)
    errors = [p - t for t, p in zip(truth, pred)]
    mse = sum(e * e for e in errors) / n
    mae = sum(abs(e) for e in errors) / n
    mape = sum(abs(e) / t for e, t in zip(errors, truth) if t) / n * 100.0
    return {"n": n, "MSE": mse, "RMSE": math.sqrt(mse), "MAE": mae, "MAPE_pct": mape}


def walk_forward(values: Sequence[float], days: Sequence[float]) -> dict[str, dict]:
    """앞의 k개로 다음 1점을 예측. 추세 / 직전값 / 평균 세 가지를 같은 분할로."""

    truth: list[float] = []
    trend: list[float] = []
    last: list[float] = []
    mean: list[float] = []
    for k in range(MIN_TRAIN, len(values)):
        train_v, train_d = values[:k], days[:k]
        ahead = days[k] - days[k - 1]
        projected = project_trend(train_v, train_d, [ahead])
        if not projected:
            continue
        truth.append(values[k])
        trend.append(projected[0])
        last.append(train_v[-1])
        mean.append(sum(train_v) / len(train_v))
    if not truth:
        return {}
    out = {
        "추세 적합": _metrics(truth, trend),
        "직전값 유지": _metrics(truth, last),
        "평균": _metrics(truth, mean),
    }
    base = out["직전값 유지"]["MSE"]
    for name, item in out.items():
        item["skill_vs_persistence_pct"] = (
            (1.0 - item["MSE"] / base) * 100.0 if base > 0 else float("nan")
        )
    return out


def _print(title: str, unit: str, result: dict[str, dict], spread: str = "") -> None:
    if not result:
        print(f"\n### {title} — 시험점 부족, 생략")
        return
    print(f"\n### {title}  (단위 {unit}){spread}")
    print(f"{'방법':12} {'시험점':>5} {'MSE':>14} {'RMSE':>12} {'MAE':>12} {'MAPE%':>8} {'skill%':>8}")
    for name, m in result.items():
        print(
            f"{name:12} {m['n']:>5} {m['MSE']:>14.6g} {m['RMSE']:>12.6g} "
            f"{m['MAE']:>12.6g} {m['MAPE_pct']:>8.2f} {m['skill_vs_persistence_pct']:>8.1f}"
        )


def area_series() -> tuple[list[float], list[float]]:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, os.path.join(REPO, "ui_next"))
    from samples import get_sample

    sample = get_sample("busan-nas-water-labels")
    areas: list[float] = []
    for frame in sample.frames:
        with Image.open(io.BytesIO(frame.png_bytes)) as image:
            areas.append(float((np.asarray(image.convert("L")) > 127).sum()))
    base = dt.date.fromisoformat(sample.source_date_strings[0])
    days = [
        float((dt.date.fromisoformat(d) - base).days) for d in sample.source_date_strings
    ]
    return areas, days


def level_series() -> dict[str, tuple[list[float], list[float]]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for path in sorted(glob.glob(os.path.join(REPO, "data/wbms_runs/fused/*/FUSED_Busan_*.json"))):
        if path.endswith("_qc.json"):
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get("records", []):
            level = record.get("water_level_m")
            if isinstance(level, (int, float)) and not isinstance(level, bool) and level > 0:
                grouped.setdefault(str(record["loc_id"]), []).append(
                    (str(record["date"]), float(level))
                )
    out: dict[str, tuple[list[float], list[float]]] = {}
    for station, items in grouped.items():
        items.sort()
        base = dt.datetime.strptime(items[0][0], "%Y%m%d").date()
        days = [
            float((dt.datetime.strptime(d, "%Y%m%d").date() - base).days) for d, _ in items
        ]
        out[station] = ([v for _, v in items], days)
    return out


def main() -> int:
    print("=" * 78)
    print("전망 오차 실측 · 1-step walk-forward")
    print("=" * 78)
    print(
        "주의: 관측 표본이 매우 적고 시계열 변동이 작다. MSE 절대값이 낮은 것은\n"
        "모델 성능의 근거가 아니며, 아래 skill%(직전값 유지 대비 개선율)를 함께 보라.\n"
        "skill% 가 0 이하이면 단순히 직전값을 반복하는 것보다 낫지 않다는 뜻이다."
    )

    areas, days = area_series()
    print(f"\n[면적] 관측 {len(areas)}시점 {[int(a) for a in areas]} px")
    _print("수체 면적 (픽셀)", "px²", walk_forward(areas, days))

    for station, (values, sdays) in sorted(level_series().items()):
        print(f"\n[수위·{station}] 관측 {len(values)}시점 "
              f"{min(values):.3f}~{max(values):.3f} m")
        _print(f"수위 · {station}", "m²", walk_forward(values, sdays))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
