#!/usr/bin/env python3
"""지상 수위계 일 자료로 시계열 예측을 **정상적으로** 검증한다.

위성 6~8시점으로는 학습·검증 구간을 날짜로 가를 수 없어, 앞서의 평가는 모든
관측을 순차 재사용하는 walk-forward 뿐이었다. 일 자료는 표본이 수백 배라
"앞 구간으로 학습하고 뒤 구간은 아예 보지 않는" 홀드아웃이 가능하다.

비교 대상을 반드시 함께 낸다. 변동이 작은 시계열은 아무 예측기나 MSE가 낮게
나오므로, 단순 기준선(직전값 유지 = persistence, 학습구간 평균)을 같은 분할에서
재고 그 대비 개선율(skill)을 본다. skill ≤ 0 이면 그 모델은 쓸 이유가 없다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gauge"


def _parse(series: Sequence[dict]) -> tuple[list[dt.date], list[float]]:
    days, values = [], []
    for row in series:
        try:
            days.append(dt.datetime.strptime(row["date"], "%Y%m%d").date())
            values.append(float(row["water_level_m"]))
        except (KeyError, ValueError, TypeError):
            continue
    return days, values


def _metrics(truth: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    n = len(truth)
    mse = sum((p - t) ** 2 for t, p in zip(truth, pred)) / n
    mae = sum(abs(p - t) for t, p in zip(truth, pred)) / n
    mape = sum(abs(p - t) / t for t, p in zip(truth, pred) if t) / n * 100.0
    return {"n": n, "mse": mse, "rmse": math.sqrt(mse), "mae": mae, "mape": mape}


def forecast_persistence(train: Sequence[float], horizon: int) -> list[float]:
    """직전 관측 유지."""
    return [train[-1]] * horizon


def forecast_mean(train: Sequence[float], horizon: int) -> list[float]:
    """학습구간 평균."""
    value = sum(train) / len(train)
    return [value] * horizon


def forecast_drift(train: Sequence[float], horizon: int) -> list[float]:
    """마지막 값에서 학습구간 평균 기울기만큼 외삽 (Theta/drift 기준선)."""
    if len(train) < 2:
        return forecast_persistence(train, horizon)
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return [train[-1] + slope * (h + 1) for h in range(horizon)]


def forecast_seasonal_naive(train: Sequence[float], horizon: int, period: int = 365) -> list[float]:
    """1년 전 같은 날 값. 계절성이 있으면 강력한 기준선이다."""
    if len(train) < period:
        return forecast_persistence(train, horizon)
    return [train[-period + h] for h in range(horizon)]


def forecast_ar1(train: Sequence[float], horizon: int) -> list[float]:
    """평균 회귀 AR(1). 이 자료가 추세가 아니라 평균 회귀라는 관찰을 모델로 옮긴 것."""
    n = len(train)
    if n < 3:
        return forecast_persistence(train, horizon)
    mean = sum(train) / n
    num = sum((train[i] - mean) * (train[i - 1] - mean) for i in range(1, n))
    den = sum((train[i] - mean) ** 2 for i in range(n))
    phi = max(-0.99, min(0.99, num / den)) if den > 0 else 0.0
    out, last = [], train[-1]
    for _ in range(horizon):
        last = mean + phi * (last - mean)
        out.append(last)
    return out


MODELS = {
    "직전값 유지": forecast_persistence,
    "학습구간 평균": forecast_mean,
    "드리프트": forecast_drift,
    "계절 naive(1년전)": forecast_seasonal_naive,
    "AR(1) 평균회귀": forecast_ar1,
}


def evaluate(values: Sequence[float], horizon: int) -> dict[str, dict[str, float]]:
    """마지막 horizon일을 통째로 감춰 두고 예측한다 (진짜 홀드아웃)."""
    if len(values) <= horizon + 10:
        return {}
    train, test = list(values[:-horizon]), list(values[-horizon:])
    out: dict[str, dict[str, float]] = {}
    for name, fn in MODELS.items():
        out[name] = _metrics(test, fn(train, horizon))
    base = out["직전값 유지"]["mse"]
    for item in out.values():
        item["skill"] = (1.0 - item["mse"] / base) * 100.0 if base > 0 else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None)
    parser.add_argument("--horizons", default="7,14,30")
    args = parser.parse_args()

    path = Path(args.file) if args.file else sorted(DATA_DIR.glob("wamis_daily_*.json"))[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    horizons = [int(h) for h in args.horizons.split(",")]

    print("=" * 88)
    print(f"지상 수위계 일 자료 · 홀드아웃 예측 검증   ({path.name})")
    print("=" * 88)
    print("마지막 N일을 학습에서 완전히 제외하고 예측 → 그 구간과 대조한다.")
    print("skill = '직전값 유지' 대비 MSE 개선율(%). 0 이하면 그 모델은 쓸 이유가 없다.\n")

    for station, series in sorted(payload["series"].items()):
        days, values = _parse(series)
        if len(values) < 60:
            print(f"[{station}] 자료 {len(values)}일 — 건너뜀\n")
            continue
        print(f"[{station}] 관측 {len(values):,}일 · {days[0]} ~ {days[-1]} · "
              f"수위 {min(values):.3f}~{max(values):.3f} m")
        for horizon in horizons:
            table = evaluate(values, horizon)
            if not table:
                continue
            print(f"  ── D+{horizon} 홀드아웃 (학습 {len(values) - horizon:,}일 / 검증 {horizon}일)")
            print(f"     {'모델':20} {'RMSE(m)':>9} {'MAE(m)':>9} {'MAPE%':>7} {'skill%':>8}")
            for name, m in table.items():
                print(f"     {name:20} {m['rmse']:>9.4f} {m['mae']:>9.4f} "
                      f"{m['mape']:>7.2f} {m['skill']:>8.1f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
