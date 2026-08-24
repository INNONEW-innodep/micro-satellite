#!/usr/bin/env python3
"""지점별·예측일수별 오차를 rolling-origin 방식으로 낸다.

홀드아웃을 한 번만 자르면 그 구간이 우연히 잔잔했는지 요동쳤는지에 따라 값이
크게 흔들린다. 그래서 검증 구간을 여러 번 미끄러뜨리며(rolling origin) 반복
평가하고 평균을 낸다. 각 회차에서 검증 구간은 학습에 **전혀** 쓰지 않는다.

지형마다 잘 맞는 모델이 다르다는 것이 이 스크립트의 확인 대상이다. 그래서
모델을 고정하지 않고 여러 기준선을 같은 분할로 돌려 지점×예측일수마다 승자를
고른다. 'skill'은 직전값 유지(persistence) 대비 MSE 개선율이며, 0 이하면 그
모델을 쓸 이유가 없다는 뜻이다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gauge"
Model = Callable[[Sequence[float], int], list[float]]


# --- 예측 기준선 ---------------------------------------------------------------


def persistence(train: Sequence[float], horizon: int) -> list[float]:
    return [train[-1]] * horizon


def train_mean(train: Sequence[float], horizon: int) -> list[float]:
    value = sum(train) / len(train)
    return [value] * horizon


def recent_mean(train: Sequence[float], horizon: int, window: int = 30) -> list[float]:
    tail = train[-window:]
    value = sum(tail) / len(tail)
    return [value] * horizon


def drift(train: Sequence[float], horizon: int) -> list[float]:
    if len(train) < 2:
        return persistence(train, horizon)
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return [train[-1] + slope * (h + 1) for h in range(horizon)]


def seasonal_naive(train: Sequence[float], horizon: int, period: int = 365) -> list[float]:
    if len(train) < period + horizon:
        return persistence(train, horizon)
    return [train[-period + h] for h in range(horizon)]


def ar1(train: Sequence[float], horizon: int) -> list[float]:
    """평균 회귀 AR(1). 관측이 추세보다 평균으로 되돌아가는 지점에서 강하다."""
    n = len(train)
    if n < 10:
        return persistence(train, horizon)
    mean = sum(train) / n
    num = sum((train[i] - mean) * (train[i - 1] - mean) for i in range(1, n))
    den = sum((value - mean) ** 2 for value in train)
    phi = max(-0.99, min(0.99, num / den)) if den > 0 else 0.0
    out, last = [], train[-1]
    for _ in range(horizon):
        last = mean + phi * (last - mean)
        out.append(last)
    return out


def damped_ar1_recent(train: Sequence[float], horizon: int, window: int = 90) -> list[float]:
    """최근 구간 평균으로 회귀. 장기 평균이 현재 수위와 동떨어진 지점 대비."""
    tail = list(train[-window:]) if len(train) >= window else list(train)
    return ar1(tail, horizon) if len(tail) >= 10 else persistence(train, horizon)


MODELS: dict[str, Model] = {
    "persistence(직전값)": persistence,
    "전체평균": train_mean,
    "최근30일평균": recent_mean,
    "드리프트": drift,
    "계절naive(1년전)": seasonal_naive,
    "AR(1)전체": ar1,
    "AR(1)최근90일": damped_ar1_recent,
}


# --- 평가 ----------------------------------------------------------------------


def rolling_origin(
    values: Sequence[float], horizon: int, folds: int, min_train: int
) -> dict[str, dict[str, float]]:
    """검증 원점을 뒤에서부터 horizon 간격으로 옮기며 반복 평가한다."""

    cuts: list[int] = []
    end = len(values)
    for _ in range(folds):
        start = end - horizon
        if start < min_train:
            break
        cuts.append(start)
        end = start
    if not cuts:
        return {}

    squared: dict[str, list[float]] = {name: [] for name in MODELS}
    absolute: dict[str, list[float]] = {name: [] for name in MODELS}
    relative: dict[str, list[float]] = {name: [] for name in MODELS}
    for cut in cuts:
        train, test = list(values[:cut]), list(values[cut : cut + horizon])
        for name, model in MODELS.items():
            pred = model(train, horizon)
            for truth, guess in zip(test, pred):
                squared[name].append((guess - truth) ** 2)
                absolute[name].append(abs(guess - truth))
                if truth:
                    relative[name].append(abs(guess - truth) / abs(truth))

    out: dict[str, dict[str, float]] = {}
    for name in MODELS:
        if not squared[name]:
            continue
        mse = sum(squared[name]) / len(squared[name])
        out[name] = {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": sum(absolute[name]) / len(absolute[name]),
            "mape": (sum(relative[name]) / len(relative[name]) * 100.0) if relative[name] else float("nan"),
            "n_points": len(squared[name]),
            "n_folds": len(cuts),
        }
    base = out.get("persistence(직전값)", {}).get("mse", 0.0)
    for item in out.values():
        item["skill"] = (1.0 - item["mse"] / base) * 100.0 if base > 0 else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None)
    parser.add_argument("--horizons", default="1,3,5,7,14,30")
    parser.add_argument("--folds", type=int, default=12, help="검증 반복 횟수")
    parser.add_argument("--min-train", type=int, default=400, help="최소 학습 일수")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    path = Path(args.file) if args.file else sorted(DATA_DIR.glob("wamis_daily_*.json"))[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    horizons = [int(h) for h in args.horizons.split(",")]

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fetch_wamis_levels import TERRAIN
    except ImportError:
        TERRAIN = {}

    print("=" * 96)
    print(f"지점별 · 예측일수별 오차   (자료: {path.name})")
    print("=" * 96)
    print(f"방식: rolling-origin {args.folds}회 반복 · 검증 구간은 매 회차 학습에서 완전 제외")
    print("skill = persistence 대비 MSE 개선율(%). 0 이하면 그 모델은 쓸 이유가 없다.\n")

    summary: dict = {"source": path.name, "method": "rolling-origin", "stations": {}}
    for station, series in sorted(payload["series"].items()):
        rows = [(r["date"], float(r["water_level_m"])) for r in series]
        if len(rows) < args.min_train + max(horizons):
            print(f"[{station}] 자료 {len(rows)}일 — 최소 요구 미달, 건너뜀\n")
            continue
        dates = [dt.datetime.strptime(d, "%Y%m%d").date() for d, _ in rows]
        values = [v for _, v in rows]

        print(f"┌ {station}  ({TERRAIN.get(station, '지형 미분류')})")
        print(f"│ 자료 규모 : {len(values):,}일 · {dates[0]} ~ {dates[-1]}")
        print(f"│ 수위 범위 : {min(values):.3f} ~ {max(values):.3f} m "
              f"(평균 {statistics.fmean(values):.3f}, 표준편차 {statistics.pstdev(values):.3f})")

        station_out: dict = {
            "n_days": len(values),
            "period": [str(dates[0]), str(dates[-1])],
            "terrain": TERRAIN.get(station, ""),
            "horizons": {},
        }
        for horizon in horizons:
            table = rolling_origin(values, horizon, args.folds, args.min_train)
            if not table:
                continue
            best = min(table.items(), key=lambda kv: kv[1]["mse"])
            first_cut = len(values) - horizon * table[best[0]]["n_folds"]
            print(f"│")
            print(f"│ ── D+{horizon} · 검증 {table[best[0]]['n_folds']}회 × {horizon}일 "
                  f"= {table[best[0]]['n_points']}점 · 검증기간 {dates[first_cut]} ~ {dates[-1]}")
            print(f"│    {'모델':22} {'MSE(m²)':>12} {'RMSE(m)':>9} {'MAE(m)':>8} {'MAPE%':>7} {'skill%':>8}")
            for name, m in sorted(table.items(), key=lambda kv: kv[1]["mse"]):
                mark = " ★" if name == best[0] else "  "
                print(f"│  {mark}{name:22} {m['mse']:>12.6f} {m['rmse']:>9.4f} "
                      f"{m['mae']:>8.4f} {m['mape']:>7.2f} {m['skill']:>8.1f}")
            station_out["horizons"][horizon] = {
                "best_model": best[0],
                "eval_start": str(dates[first_cut]),
                "eval_end": str(dates[-1]),
                "metrics": table,
            }
        summary["stations"][station] = station_out
        print("└" + "─" * 94 + "\n")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"JSON 저장: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
