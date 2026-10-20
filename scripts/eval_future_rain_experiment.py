#!/usr/bin/env python3
"""미래 강수(완전 예보 가정) 상한 실험 — 강수 예보를 넣으면 수위 예측이 얼마나 좋아지는가.

기상 자료가 있는 지역의 모든 지점을 지역 기본 분할로 ``ridge_arx``(원점 이전 강수만) 와
``ridge_arx_future``(원점 이후 **실측** 강수를 예보로 간주) 두 후보만으로 학습하고, 시험 연도에서
persistence / AR-X / +3일 예보 / +전 구간 예보 의 오차를 지점×지평별로 비교한다.

- 산출물은 **스크래치 디렉터리**에 쓰고 지역의 배포 산출물(``regions/<id>/models/gauge/*.json``)은
  건드리지 않는다. 결과 요약만 ``regions/<id>/models/gauge/experiments/future_rain.json`` 에 남긴다.
- ``ridge_arx_future`` 는 ``uses_future_exog`` 라 기본 후보 풀에서 제외되며, 여기서만 명시적으로 허용한다.
- **완전 예보 가정**: 실측 미래 강수를 오차 없는 예보로 본 것이므로 결과는 이득의 **상한**이다. 실시간에는
  기상청 예보가 필요하고(아직 미배선), 예보 오차만큼 이득이 줄어든다.

    .venv/bin/python scripts/eval_future_rain_experiment.py                  # 기상 있는 지역 전부
    .venv/bin/python scripts/eval_future_rain_experiment.py --region nakdong_busan
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.gauge.forecasters import DailySeries, ForecasterRegistry, build_forecaster_registry  # noqa: E402
from backend.app.gauge.regions import RegionRegistry  # noqa: E402
from backend.app.gauge.training import (  # noqa: E402
    Split,
    TrainingError,
    _clean,
    _skill,
    load_artifacts,
    metrics,
    origins_for_targets,
    train_region,
)

EXPERIMENT_SCHEMA = "watercast.gauge.experiment.future_rain/1"
EXPERIMENT_FILENAME = "future_rain.json"
ALLOWED = ("ridge_arx", "ridge_arx_future")
VARIANTS: tuple[tuple[str, Any], ...] = (("future_3d", 3), ("future_all", "all"))
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 7, 10, 14, 30)
N_EVENTS = 5
BEAT_MARGIN = 0.005  # skill 차이 0.5 %p

NOTES_KO = [
    "완전 예보 가정(perfect prognosis): 시험 연도의 **실측** 미래 강수를 오차 없는 예보로 간주해 넣었다. 따라서 이 표의 이득은 상한이다.",
    "실시간 전망에서는 원점 이후 강수가 관측에 없으므로 기상청 단기(3일)·중기 예보가 입력이 되어야 하며, 그 연결은 아직 배선되지 않았다. 예보 오차만큼 이득이 줄어든다.",
    "future_3d = 원점 이후 3일치 강수만 안다(그 뒤는 0). future_all = 목표일까지 전부 안다. D+1·D+3 에서는 두 변형의 입력이 같아 결과도 같다.",
    "ridge_arx 는 배포 학습과 같은 격자(mode·p·lam 12개) 중 검증 최상을, ridge_arx_future 는 p=7·delta 고정에 lam∈{10,100} 중 검증 최상을 썼다. 격자가 더 큰 쪽은 AR-X 라 미래 강수 쪽에 유리하게 고른 비교가 아니다.",
    "공식 성능 수치(evaluation.json·화면의 선택 모델·개선율)에는 이 실험이 들어가지 않는다. 배포 후보 풀은 uses_future_exog 예측기를 제외한다.",
    "skill = 1 − MSE/MSE_persistence. delta_rmse_cm = RMSE(AR-X) − RMSE(+예보) (양수면 예보가 오차를 줄임).",
    "강수는 지역 대표 ASOS 1지점 값이다. 상류 유역의 비로 오르는 하류 지점(낙동강 하구 구포대교 2024-07-10~11 급상승 때 부산 강수 32 mm·0 mm)에서는 지역 강수 예보의 이득이 거의 없고, 유역 평균 강수(또는 상류 지점 수위)가 있어야 한다.",
]


def _best_hparams(candidates: Sequence[Mapping[str, Any]], forecaster_id: str, future_days: Any = None) -> Mapping[str, Any] | None:
    """검증 persistence 대비 비율이 가장 작은 후보의 하이퍼파라미터. ``future_days`` 를 주면 그 변형만."""

    best: Mapping[str, Any] | None = None
    best_ratio = math.inf
    for item in candidates:
        if item.get("forecaster_id") != forecaster_id or "val" not in item:
            continue
        hp = item.get("hparams") or {}
        if future_days is not None and str(hp.get("future_days")) != str(future_days):
            continue
        ratio = float(item["val"]["ratio_vs_persistence_mean"])
        if ratio < best_ratio:
            best, best_ratio = hp, ratio
    return best


def _abs_error(pred: np.ndarray, truth: float) -> float | None:
    value = float(pred[0]) if pred.shape[0] else math.nan
    return abs(truth - value) if math.isfinite(value) else None


def evaluate_station(
    series: DailySeries,
    split: Split,
    artifact: Mapping[str, Any],
    forecasters: ForecasterRegistry,
    *,
    report_horizons: Iterable[int],
) -> dict[str, Any]:
    """스크래치 산출물의 후보 검증 순위로 변형별 최상 하이퍼파라미터를 고르고, 학습+검증 재적합 → 시험 오차."""

    persistence = forecasters.get("persistence")
    arx = forecasters.get("ridge_arx")
    future = forecasters.get("ridge_arx_future")
    horizons: dict[str, Any] = {}
    fitted: dict[int, dict[str, Any]] = {}
    for horizon in sorted({int(h) for h in report_horizons}):
        block = artifact["horizons"].get(str(horizon))
        if not block:
            continue
        candidates = block.get("candidates") or []
        test_origins = origins_for_targets(series, horizon, split.test_start, split.test_end)
        trainval_origins = origins_for_targets(series, horizon, None, split.val_end)
        if test_origins.shape[0] == 0:
            continue
        truth = series.y[test_origins + horizon]
        preds: dict[str, np.ndarray] = {"persistence": persistence.predict(series, test_origins, horizon, {})}
        chosen: dict[str, Any] = {}
        params_h: dict[str, Any] = {}
        arx_hp = _best_hparams(candidates, "ridge_arx")
        if arx_hp is not None:
            p_arx = arx.fit(series, trainval_origins, horizon, arx_hp)
            preds["ridge_arx"] = arx.predict(series, test_origins, horizon, p_arx)
            chosen["ridge_arx"] = dict(arx_hp)
            params_h["ridge_arx"] = p_arx
        for key, future_days in VARIANTS:
            hp = _best_hparams(candidates, "ridge_arx_future", future_days)
            if hp is None:
                continue
            p_future = future.fit(series, trainval_origins, horizon, hp)
            preds[key] = future.predict(series, test_origins, horizon, p_future)
            chosen[key] = dict(hp)
            params_h[key] = p_future
        fitted[horizon] = params_h
        scored = {name: metrics(pred, truth) for name, pred in preds.items()}
        base_mse = scored["persistence"]["mse"]
        entry: dict[str, Any] = {"n_test": int(test_origins.shape[0]), "hparams": chosen, "models": {}}
        for name, score in scored.items():
            entry["models"][name] = {
                "rmse": score["rmse"], "mse": score["mse"], "mae": score["mae"],
                "skill_vs_persistence": _skill(score["mse"], base_mse) if name != "persistence" else 0.0,
            }
        arx_rmse = scored.get("ridge_arx", {}).get("rmse", math.nan)
        for key, _ in VARIANTS:
            if key in scored:
                entry["models"][key]["delta_rmse_cm"] = (arx_rmse - scored[key]["rmse"]) * 100.0 if math.isfinite(arx_rmse) else None
                s_future = entry["models"][key]["skill_vs_persistence"]
                s_arx = entry["models"].get("ridge_arx", {}).get("skill_vs_persistence")
                entry["models"][key]["skill_gain_pp"] = (s_future - s_arx) * 100.0 if (s_future is not None and s_arx is not None) else None
        horizons[str(horizon)] = entry

    # 이벤트: 시험 연도 1일 상승폭 상위 N — 그 날짜를 목표로 하는 D+1·D+3 절대오차
    events: list[dict[str, Any]] = []
    test_mask = (series.dates >= split.test_start) & (series.dates <= split.test_end)
    index = np.arange(1, series.n)
    index = index[test_mask[index] & series.valid[index] & series.valid[index - 1]]
    if index.shape[0]:
        rises = series.y[index] - series.y[index - 1]
        order = np.argsort(-rises)[:N_EVENTS]
        for pos in order:
            t = int(index[pos])
            event: dict[str, Any] = {"date": str(series.dates[t]), "rise_m": float(rises[pos]), "level_m": float(series.y[t]), "errors": {}}
            for horizon in (1, 3):
                origin = t - horizon
                if origin < 0 or horizon not in fitted:
                    continue
                origins = np.array([origin])
                truth_t = float(series.y[t])
                row: dict[str, Any] = {"persistence": _abs_error(persistence.predict(series, origins, horizon, {}), truth_t)}
                params_h = fitted[horizon]
                if "ridge_arx" in params_h:
                    row["ridge_arx"] = _abs_error(arx.predict(series, origins, horizon, params_h["ridge_arx"]), truth_t)
                if "future_all" in params_h:
                    row["future_all"] = _abs_error(future.predict(series, origins, horizon, params_h["future_all"]), truth_t)
                # 원점 이후 실측 강수(그 창에서 '예보'로 간주한 값)
                if series.has_exog("rain_mm"):
                    rain = np.nan_to_num(series.exog["rain_mm"], nan=0.0)
                    row["future_rain_mm"] = float(rain[origin + 1 : t + 1].sum())
                event["errors"][f"D+{horizon}"] = row
            events.append(event)
    return {"horizons": horizons, "events": events}


def summarize_region(stations: Mapping[str, Mapping[str, Any]], report_horizons: Iterable[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_horizon": {}, "cells_future_beats_arx": {key: 0 for key, _ in VARIANTS}, "cells_total": 0}
    for horizon in sorted({int(h) for h in report_horizons}):
        acc: dict[str, list[float]] = {}
        n = 0
        for payload in stations.values():
            entry = (payload.get("horizons") or {}).get(str(horizon))
            if not entry:
                continue
            n += 1
            out["cells_total"] += 1
            for name, model in entry["models"].items():
                if model.get("skill_vs_persistence") is not None:
                    acc.setdefault(f"skill:{name}", []).append(model["skill_vs_persistence"])
                if model.get("rmse") is not None:
                    acc.setdefault(f"rmse:{name}", []).append(model["rmse"])
                if model.get("delta_rmse_cm") is not None:
                    acc.setdefault(f"delta_rmse_cm:{name}", []).append(model["delta_rmse_cm"])
                if model.get("skill_gain_pp") is not None and model["skill_gain_pp"] > BEAT_MARGIN * 100.0:
                    out["cells_future_beats_arx"][name] = out["cells_future_beats_arx"].get(name, 0) + 1
        out["by_horizon"][str(horizon)] = {
            "n_stations": n,
            "mean_skill": {k.split(":", 1)[1]: float(np.mean(v)) for k, v in acc.items() if k.startswith("skill:")},
            "mean_rmse_m": {k.split(":", 1)[1]: float(np.mean(v)) for k, v in acc.items() if k.startswith("rmse:")},
            "mean_delta_rmse_cm": {k.split(":", 1)[1]: float(np.mean(v)) for k, v in acc.items() if k.startswith("delta_rmse_cm:")},
        }
    return out


def run_region(
    registry: RegionRegistry,
    forecasters: ForecasterRegistry,
    region_id: str,
    *,
    scratch_dir: Path,
    horizon_max: int = 30,
    report_horizons: Sequence[int] = DEFAULT_HORIZONS,
    write: bool = True,
) -> dict[str, Any]:
    """지역 하나를 실험하고(스크래치 학습) 요약 JSON 을 ``<artifacts_dir>/experiments/future_rain.json`` 에 쓴다."""

    record = registry.get(region_id)
    series_set = registry.load_series(region_id)
    if not series_set.meta.get("weather_available"):
        raise TrainingError(f"{region_id}: 기상 자료가 없어 미래 강수 실험을 할 수 없습니다")
    scratch = Path(scratch_dir) / region_id / "models" / "gauge"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_region(
        registry, forecasters, region_id,
        allowed_forecasters=list(ALLOWED), horizon_max=horizon_max, write_dir=scratch,
    )
    artifacts = load_artifacts(scratch)
    stations_out: dict[str, Any] = {}
    for station in series_set.stations:
        artifact = artifacts.get(station.id)
        if artifact is None:
            continue
        series = series_set.series[station.id]
        split = Split.from_mapping(artifact["split"])
        result = evaluate_station(series, split, artifact, forecasters, report_horizons=report_horizons)
        result["station"] = {"id": station.id, "name": station.name, "terrain": station.terrain}
        stations_out[station.id] = result
    if not stations_out:
        raise TrainingError(f"{region_id}: 학습된 지점이 없습니다")
    split_used = next(iter(artifacts.values()))["split"]
    payload = _clean({
        "schema": EXPERIMENT_SCHEMA,
        "region_id": region_id,
        "region_name": record.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split_used,
        "horizon_max": int(horizon_max),
        "report_horizons": sorted({int(h) for h in report_horizons}),
        "forecasters": list(ALLOWED),
        "variants": {key: {"future_days": fd} for key, fd in VARIANTS},
        "n_stations": len(stations_out),
        "stations": stations_out,
        "summary": summarize_region(stations_out, report_horizons),
        "notes_ko": list(NOTES_KO),
        "elapsed_s": round(time.time() - started, 1),
    })
    if write:
        # 내장 지역이면 저장소의 산출물 디렉터리(사용자 사본이 있어도 원본) 아래 experiments/ 에 쓴다
        builtin = registry.builtin_record(region_id)
        target = (builtin or record).artifacts_dir / "experiments" / EXPERIMENT_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        payload["written_to"] = str(target)
    return payload


def _fmt_skill(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.1f}%"


def write_docs(results: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    """지평마다 지역 간 비교표 하나 — docs/presentation/미래강수_실험.md."""

    horizons: list[int] = sorted({int(h) for r in results.values() for h in r["report_horizons"]})
    lines = [
        "# 미래 강수(예보)를 넣으면 얼마나 좋아지나 — 완전 예보 가정 상한 실험",
        "",
        f"생성: {datetime.now(timezone.utc).isoformat(timespec='seconds')} · 스크립트 `scripts/eval_future_rain_experiment.py` · "
        f"원본 JSON `regions/<id>/models/gauge/experiments/future_rain.json`",
        "",
        "## 읽는 법",
        "",
    ]
    lines += [f"- {note}" for note in NOTES_KO]
    lines += ["", "skill 은 지점 평균(지역 안), 전체 행은 지점 가중 평균. ΔRMSE 는 RMSE(AR-X) − RMSE(+예보), cm.", ""]
    for horizon in horizons:
        lines += [f"## D+{horizon}", "", "| 지역 | 지점 수 | AR-X(과거 강수) skill | +3일 예보 skill | +전 구간 예보 skill | ΔRMSE 3일 (cm) | ΔRMSE 전 구간 (cm) |", "|---|---:|---:|---:|---:|---:|---:|"]
        pooled: dict[str, list[float]] = {}
        for region_id, payload in results.items():
            row = (payload.get("summary") or {}).get("by_horizon", {}).get(str(horizon))
            if not row:
                continue
            ms, md = row["mean_skill"], row["mean_delta_rmse_cm"]
            lines.append(
                f"| {payload.get('region_name') or region_id} | {row['n_stations']} | {_fmt_skill(ms.get('ridge_arx'))} | "
                f"{_fmt_skill(ms.get('future_3d'))} | {_fmt_skill(ms.get('future_all'))} | "
                f"{md.get('future_3d', float('nan')):+.1f} | {md.get('future_all', float('nan')):+.1f} |"
            )
            for station in payload["stations"].values():
                entry = station["horizons"].get(str(horizon))
                if not entry:
                    continue
                for name, model in entry["models"].items():
                    if model.get("skill_vs_persistence") is not None:
                        pooled.setdefault(f"skill:{name}", []).append(model["skill_vs_persistence"])
                    if model.get("delta_rmse_cm") is not None:
                        pooled.setdefault(f"delta:{name}", []).append(model["delta_rmse_cm"])
        n_all = len(pooled.get("skill:ridge_arx", []))
        mean = lambda key: float(np.mean(pooled[key])) if pooled.get(key) else None  # noqa: E731
        d3, da = mean("delta:future_3d"), mean("delta:future_all")
        lines.append(
            f"| **전체(지점 가중)** | {n_all} | {_fmt_skill(mean('skill:ridge_arx'))} | {_fmt_skill(mean('skill:future_3d'))} | "
            f"{_fmt_skill(mean('skill:future_all'))} | {(d3 if d3 is not None else float('nan')):+.1f} | {(da if da is not None else float('nan')):+.1f} |"
        )
        lines.append("")
    # 이벤트: 지역별 최대 상승일 1개씩
    lines += ["## 급상승일 오차 비교 (지역별 시험 연도 최대 1일 상승)", "", "| 지역 | 지점 | 날짜 | 상승 (m) | 원점 이후 강수 D+1 (mm) | D+1 persistence | D+1 AR-X | D+1 +예보 | D+3 persistence | D+3 AR-X | D+3 +예보 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for region_id, payload in results.items():
        best: tuple[float, str, dict[str, Any]] | None = None
        for station in payload["stations"].values():
            for event in station.get("events") or []:
                if best is None or event["rise_m"] > best[0]:
                    best = (event["rise_m"], station["station"]["name"], event)
        if best is None:
            continue
        _, station_name, event = best
        e1, e3 = event["errors"].get("D+1", {}), event["errors"].get("D+3", {})
        cm = lambda v: "—" if v is None else f"{v * 100:.1f} cm"  # noqa: E731
        lines.append(
            f"| {payload.get('region_name') or region_id} | {station_name} | {event['date']} | {event['rise_m']:.2f} | "
            f"{e1.get('future_rain_mm', float('nan')):.0f} | {cm(e1.get('persistence'))} | {cm(e1.get('ridge_arx'))} | {cm(e1.get('future_all'))} | "
            f"{cm(e3.get('persistence'))} | {cm(e3.get('ridge_arx'))} | {cm(e3.get('future_all'))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions-dir", default=str(ROOT / "regions"))
    parser.add_argument("--user-dir", default=str(ROOT / "backend" / "data" / "regions"))
    parser.add_argument("--region", action="append", help="지역 id (반복 가능). 비우면 기상 자료가 있는 지역 전부")
    parser.add_argument("--scratch-dir", default=None, help="스크래치 학습 산출물 위치(기본: 임시 디렉터리)")
    parser.add_argument("--horizon-max", type=int, default=30)
    parser.add_argument("--docs-out", default=str(ROOT / "docs" / "presentation" / "미래강수_실험.md"))
    parser.add_argument("--no-docs", action="store_true")
    args = parser.parse_args()

    registry = RegionRegistry(Path(args.regions_dir), Path(args.user_dir))
    forecasters = build_forecaster_registry()
    scratch = Path(args.scratch_dir) if args.scratch_dir else Path(tempfile.mkdtemp(prefix="future_rain_"))
    region_ids = args.region or [r.id for r in registry.list() if r.gauge_available and r.origin == "builtin"]

    results: dict[str, dict[str, Any]] = {}
    exit_code = 0
    for region_id in region_ids:
        print(f"== {region_id} → scratch {scratch / region_id}")
        try:
            payload = run_region(registry, forecasters, region_id, scratch_dir=scratch, horizon_max=args.horizon_max)
        except TrainingError as exc:
            print(f"   건너뜀: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        results[region_id] = payload
        summary = payload["summary"]["by_horizon"]
        print(f"   지점 {payload['n_stations']}개 · {payload['elapsed_s']}s · {payload.get('written_to')}")
        for h, row in summary.items():
            ms, md = row["mean_skill"], row["mean_delta_rmse_cm"]
            print(f"   D+{h:>2}: arx {_fmt_skill(ms.get('ridge_arx')):>7}  +3d {_fmt_skill(ms.get('future_3d')):>7}  "
                  f"+all {_fmt_skill(ms.get('future_all')):>7}  ΔRMSE 3d {md.get('future_3d', float('nan')):+.1f} cm  all {md.get('future_all', float('nan')):+.1f} cm")
    if results and not args.no_docs:
        # 기존 실험 JSON 이 있는 지역(이번에 돌리지 않은)도 표에 합친다
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
