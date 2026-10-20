"""시간 분할 학습·모델 선택·시험 평가·산출물 저장.

규율:

1. **시간으로 가른다.** 학습(≤ train_end) → 검증(val) → 시험(test). 검증 구간은
   학습에, 시험 구간은 학습·선택 어디에도 쓰지 않는다.
2. **선택은 검증에서, 보고는 시험에서.** 지점×예측일수마다 후보(모델×하이퍼파라미터)를
   검증 MSE로 고르고, 고른 하나만 학습+검증으로 다시 적합해 시험 구간 오차를 보고한다.
   시험 구간에서 승자를 고르면 보고값이 낙관적으로 치우친다.
3. **persistence 는 기준선이지 후보가 아니다.** skill = 1 − MSE_model / MSE_persistence.
4. 배포용 파라미터는 전체 자료로 한 번 더 적합한다(``params_deploy``). 시험 연도 안의
   날짜를 원점으로 전망할 때는 시험을 보지 않은 ``params_eval`` 을 써야 누수가 없다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .forecasters import DailySeries, ForecasterRegistry
from .regions import GaugeSeriesSet, RegionRecord, RegionRegistry, StationRecord

ARTIFACT_SCHEMA = "watercast.gauge.forecast-artifact/1"
EVALUATION_SCHEMA = "watercast.gauge.evaluation/1"
MIN_VAL_POINTS = 30
MIN_TRAIN_POINTS = 730
# 사용자 지정 분할의 최소 길이(일). 학습은 폴드당 MIN_TRAIN_POINTS 와 같은 2년, 검증은 한 분기, 시험은 한 달.
MIN_TRAIN_DAYS = 730
MIN_VAL_DAYS = 90
MIN_TEST_DAYS = 30
SPLIT_KEYS = ("train_end", "val_start", "val_end", "test_start", "test_end")


class TrainingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Split:
    train_end: np.datetime64
    val_start: np.datetime64
    val_end: np.datetime64
    test_start: np.datetime64
    test_end: np.datetime64

    @classmethod
    def from_mapping(cls, raw: Mapping[str, str], series_end: np.datetime64 | None = None) -> Split:
        def get(key: str) -> np.datetime64:
            value = raw.get(key)
            if not value:
                raise TrainingError(f"forecast.split.{key} 가 없습니다")
            return np.datetime64(str(value)[:10], "D")

        split = cls(get("train_end"), get("val_start"), get("val_end"), get("test_start"), get("test_end"))
        if not (split.train_end < split.val_start <= split.val_end < split.test_start <= split.test_end):
            raise TrainingError("split 은 train_end < val_start ≤ val_end < test_start ≤ test_end 이어야 합니다")
        return split

    @classmethod
    def default_for(cls, series: DailySeries) -> Split:
        """분할이 없을 때: 마지막 2년을 검증·시험으로 쓴다."""

        end = series.dates[-1]
        year = np.timedelta64(365, "D")
        return cls(
            train_end=end - 2 * year,
            val_start=end - 2 * year + np.timedelta64(1, "D"),
            val_end=end - year,
            test_start=end - year + np.timedelta64(1, "D"),
            test_end=end,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "train_end": str(self.train_end),
            "val_start": str(self.val_start),
            "val_end": str(self.val_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }


def series_range(series_set: GaugeSeriesSet) -> tuple[np.datetime64, np.datetime64] | None:
    """지역 시계열 범위 — 가장 이른 지점 시작일 .. 가장 늦은 지점 종료일. 지점이 없으면 None."""

    starts = [s.dates[0] for s in series_set.series.values() if s.n > 0]
    ends = [s.dates[-1] for s in series_set.series.values() if s.n > 0]
    if not starts:
        return None
    return min(starts), max(ends)


def validate_split(
    raw: Mapping[str, Any],
    series_range_: tuple[np.datetime64, np.datetime64] | None,
    *,
    min_train_days: int = MIN_TRAIN_DAYS,
    min_val_days: int = MIN_VAL_DAYS,
    min_test_days: int = MIN_TEST_DAYS,
) -> Split:
    """사용자 지정 분할을 검사해 ``Split`` 으로 만든다. 어긋나면 한국어 사유의 ``TrainingError``.

    규칙: 순서 train_end < val_start ≤ val_end < test_start ≤ test_end · 모든 날짜가 지역 시계열 범위
    안 · 학습(시계열 시작..train_end) ≥ 730일 · 검증 ≥ 90일 · 시험 ≥ 30일. 오류 문구에 지역 범위를
    함께 넣어 화면이 그대로 보여 줄 수 있게 한다.
    """

    def day(value: Any, key: str) -> np.datetime64:
        text = str(value)[:10] if value is not None else ""
        try:
            return np.datetime64(text, "D")
        except ValueError as exc:
            raise TrainingError(f"분할 {key} 의 날짜 형식이 잘못됐습니다: {value!r} (YYYY-MM-DD)") from exc

    missing = [k for k in SPLIT_KEYS if not raw.get(k)]
    if missing:
        raise TrainingError(f"분할에 빠진 항목이 있습니다: {missing} — {list(SPLIT_KEYS)} 전부 필요합니다")
    dates = {k: day(raw[k], k) for k in SPLIT_KEYS}
    range_text = f"{series_range_[0]} ~ {series_range_[1]}" if series_range_ else "(시계열 없음)"
    if not (dates["train_end"] < dates["val_start"] <= dates["val_end"] < dates["test_start"] <= dates["test_end"]):
        raise TrainingError(
            "분할 순서가 어긋났습니다 — train_end < val_start ≤ val_end < test_start ≤ test_end 이어야 합니다. "
            f"(입력: 학습 종료 {dates['train_end']}, 검증 {dates['val_start']}~{dates['val_end']}, "
            f"시험 {dates['test_start']}~{dates['test_end']}; 지역 시계열 범위 {range_text})"
        )
    if series_range_ is None:
        raise TrainingError("이 지역에는 수위 시계열이 없어 분할을 검사할 수 없습니다 — 시계열을 먼저 올리세요")
    lo, hi = series_range_
    outside = [k for k in SPLIT_KEYS if not (lo <= dates[k] <= hi)]
    if outside:
        raise TrainingError(
            f"분할 날짜가 지역 시계열 범위({range_text}) 밖입니다: "
            + ", ".join(f"{k}={dates[k]}" for k in outside)
        )
    one = np.timedelta64(1, "D")
    train_days = int((dates["train_end"] - lo) / one) + 1
    val_days = int((dates["val_end"] - dates["val_start"]) / one) + 1
    test_days = int((dates["test_end"] - dates["test_start"]) / one) + 1
    if train_days < min_train_days:
        raise TrainingError(
            f"학습 구간이 너무 짧습니다 — 시계열 시작({lo})부터 train_end({dates['train_end']})까지 {train_days}일, "
            f"최소 {min_train_days}일 필요 (지역 시계열 범위 {range_text})"
        )
    if val_days < min_val_days:
        raise TrainingError(
            f"검증 구간이 너무 짧습니다 — {dates['val_start']}~{dates['val_end']} {val_days}일, 최소 {min_val_days}일 필요 "
            f"(지역 시계열 범위 {range_text})"
        )
    if test_days < min_test_days:
        raise TrainingError(
            f"시험 구간이 너무 짧습니다 — {dates['test_start']}~{dates['test_end']} {test_days}일, 최소 {min_test_days}일 필요 "
            f"(지역 시계열 범위 {range_text})"
        )
    return Split(dates["train_end"], dates["val_start"], dates["val_end"], dates["test_start"], dates["test_end"])


def origins_for_targets(
    series: DailySeries, horizon: int, start: np.datetime64 | None, end: np.datetime64 | None
) -> np.ndarray:
    """목표일 ``t+h`` 가 [start, end] 안에 있고 정답이 있는 원점 ``t``."""

    index = np.arange(series.n - horizon)
    target_dates = series.dates[index + horizon]
    keep = series.valid[index + horizon]
    if start is not None:
        keep &= target_dates >= start
    if end is not None:
        keep &= target_dates <= end
    return index[keep]


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    ok = ~(np.isnan(pred) | np.isnan(truth))
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "mse": math.nan, "rmse": math.nan, "mae": math.nan, "bias": math.nan}
    residual = truth[ok] - pred[ok]
    mse = float(np.mean(residual**2))
    return {
        "n": n,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def residual_profile(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    ok = ~(np.isnan(pred) | np.isnan(truth))
    if ok.sum() < 5:
        return {"sigma": math.nan, "q05": math.nan, "q25": math.nan, "q50": math.nan, "q75": math.nan, "q95": math.nan}
    residual = truth[ok] - pred[ok]
    q = np.quantile(residual, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "sigma": float(np.std(residual)),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q50": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
    }


def _skill(model_mse: float, base_mse: float) -> float | None:
    if not (math.isfinite(model_mse) and math.isfinite(base_mse)) or base_mse <= 0:
        return None
    return 1.0 - model_mse / base_mse


def _clean(value: Any) -> Any:
    """JSON에 NaN을 쓰지 않는다 — 파서마다 취급이 달라 조용히 깨진다."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def validation_folds(series: DailySeries, split: Split, horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """검증 구간을 **연도별 폴드**로 나눈다 (rolling origin).

    폴드 = 검증 연도 하나. 학습 원점은 그 해 이전 전부(목표일 < 폴드 시작).
    한 해가 홍수로 뒤덮이면 그 해 MSE는 모든 후보가 비슷하게 나빠 1~2% 차이로
    승자가 갈리고 다음 해에 뒤집힌다. 여러 해를 따로 재서 persistence 대비
    비율을 평균하면 특정 해의 극단값이 선택을 좌우하지 못한다.
    """

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    start_year = int(str(split.val_start)[:4])
    end_year = int(str(split.val_end)[:4])
    for year in range(start_year, end_year + 1):
        fold_start = max(split.val_start, np.datetime64(f"{year}-01-01", "D"))
        fold_end = min(split.val_end, np.datetime64(f"{year}-12-31", "D"))
        train_origins = origins_for_targets(series, horizon, None, fold_start - np.timedelta64(1, "D"))
        val_origins = origins_for_targets(series, horizon, fold_start, fold_end)
        if train_origins.shape[0] < MIN_TRAIN_POINTS or val_origins.shape[0] < MIN_VAL_POINTS:
            continue
        folds.append((train_origins, val_origins))
    return folds


def train_station(
    series: DailySeries,
    split: Split,
    registry: ForecasterRegistry,
    *,
    horizon_max: int = 30,
    report_horizons: Sequence[int] = (1, 3, 5, 7, 14, 30),
    allowed_forecasters: Iterable[str] | None = None,
    station: StationRecord | None = None,
) -> dict[str, Any]:
    """지점 하나를 예측일수 1..horizon_max 전부 학습·선택·평가한다."""

    candidates_pool = registry.candidates(allowed_forecasters)
    if not candidates_pool:
        raise TrainingError("선택 가능한 예측기(기준선 제외)가 없습니다")
    references = registry.references()
    persistence = registry.get("persistence")

    horizons_out: dict[str, Any] = {}
    backtest: dict[str, Any] = {}
    report_set = {int(h) for h in report_horizons}

    for horizon in range(1, int(horizon_max) + 1):
        folds = validation_folds(series, split, horizon)
        if not folds:
            raise TrainingError(f"D+{horizon}: 검증 폴드를 만들 수 없습니다 — 자료 기간이나 분할을 확인하세요")
        test_origins = origins_for_targets(series, horizon, split.test_start, split.test_end)
        trainval_origins = origins_for_targets(series, horizon, None, split.val_end)
        all_origins = origins_for_targets(series, horizon, None, None)

        # 1) 폴드마다 후보 전부를 적합하고, persistence 대비 MSE 비율을 폴드 평균한다.
        candidate_specs = [
            (forecaster, dict(hparams))
            for forecaster in candidates_pool
            for hparams in forecaster.info.hyperparameter_grid
        ]
        ratios: list[list[float]] = [[] for _ in candidate_specs]
        pooled_sq: list[list[float]] = [[] for _ in candidate_specs]
        pooled_residual: list[list[float]] = [[] for _ in candidate_specs]
        errors: list[str | None] = [None] * len(candidate_specs)
        reference_pooled: dict[str, list[float]] = {ref.info.id: [] for ref in references}
        n_val_total = 0
        for train_origins, val_origins in folds:
            val_truth = series.y[val_origins + horizon]
            n_val_total += int(val_origins.shape[0])
            persistence_mse = metrics(persistence.predict(series, val_origins, horizon, {}), val_truth)["mse"]
            for ref in references:
                ref_params = ref.fit(series, train_origins, horizon, {})
                ref_pred = ref.predict(series, val_origins, horizon, ref_params)
                ok = ~(np.isnan(ref_pred) | np.isnan(val_truth))
                reference_pooled[ref.info.id].extend(((val_truth - ref_pred)[ok] ** 2).tolist())
            for index, (forecaster, hparams) in enumerate(candidate_specs):
                if errors[index] is not None:
                    continue
                try:
                    params = forecaster.fit(series, train_origins, horizon, hparams)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    errors[index] = str(exc)
                    continue
                pred = forecaster.predict(series, val_origins, horizon, params)
                score = metrics(pred, val_truth)
                if score["n"] < MIN_VAL_POINTS or not math.isfinite(score["mse"]):
                    errors[index] = "검증 예측이 부족합니다"
                    continue
                ratios[index].append(score["mse"] / persistence_mse if persistence_mse > 0 else 1.0)
                ok = ~(np.isnan(pred) | np.isnan(val_truth))
                pooled_sq[index].extend(((val_truth - pred)[ok] ** 2).tolist())
                pooled_residual[index].extend((val_truth - pred)[ok].tolist())

        params_deploy_cache: dict[str, Any] = {}
        candidates: list[dict[str, Any]] = []
        for index, (forecaster, hparams) in enumerate(candidate_specs):
            entry: dict[str, Any] = {"forecaster_id": forecaster.info.id, "hparams": hparams}
            if errors[index] is not None or not ratios[index]:
                entry["error"] = errors[index] or "검증 실패"
            else:
                mse = float(np.mean(pooled_sq[index]))
                entry["val"] = {
                    "n": len(pooled_sq[index]),
                    "n_folds": len(ratios[index]),
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "ratio_vs_persistence_mean": float(np.mean(ratios[index])),
                    "ratio_vs_persistence_by_fold": [float(r) for r in ratios[index]],
                }
            candidates.append(entry)
        usable = [i for i, c in enumerate(candidates) if "val" in c]
        if not usable:
            raise TrainingError(f"D+{horizon}: 검증 가능한 후보가 없습니다")
        best_index = min(usable, key=lambda i: candidates[i]["val"]["ratio_vs_persistence_mean"])
        best_forecaster, best_hparams = candidate_specs[best_index]
        candidates[best_index]["selected"] = True
        residuals_val = residual_profile(
            np.zeros(len(pooled_residual[best_index])), np.asarray(pooled_residual[best_index], dtype=np.float64)
        )
        baseline_val: dict[str, Any] = {}
        for ref_id, squares in reference_pooled.items():
            if squares:
                ref_mse = float(np.mean(squares))
                baseline_val[ref_id] = {"n": len(squares), "mse": ref_mse, "rmse": math.sqrt(ref_mse)}

        # 2) 고른 하나만 학습+검증으로 다시 적합해 시험 구간을 본다.
        params_eval = best_forecaster.fit(series, trainval_origins, horizon, best_hparams)
        test_block: dict[str, Any] = {"n": 0}
        baseline_test: dict[str, Any] = {}
        skill_vs_persistence: float | None = None
        test_truth = series.y[test_origins + horizon] if test_origins.shape[0] > 0 else None
        if test_origins.shape[0] > 0:
            test_pred = best_forecaster.predict(series, test_origins, horizon, params_eval)
            test_block = metrics(test_pred, test_truth)
            test_block["residual"] = residual_profile(test_pred, test_truth)
            for baseline in references:
                bparams = baseline.fit(series, trainval_origins, horizon, {})
                baseline_test[baseline.info.id] = metrics(
                    baseline.predict(series, test_origins, horizon, bparams), test_truth
                )
            skill_vs_persistence = _skill(test_block["mse"], baseline_test["persistence"]["mse"])
            if horizon in report_set:
                persistence_pred = persistence.predict(series, test_origins, horizon, {})
                backtest[str(horizon)] = {
                    "dates": [str(d) for d in series.dates[test_origins + horizon]],
                    "truth": [float(v) for v in test_truth],
                    "pred": _clean([float(v) for v in test_pred]),
                    "persistence": [float(v) for v in persistence_pred],
                }

        # 2-1) **모델(계열)별 성능지표**: 계열마다 검증 최상 하이퍼파라미터로 학습+검증 재적합 → 시험 오차,
        #      그리고 배포 파라미터. 선택되지 않은 모델도 같은 시험 구간에서 채점해 비교표를 만든다.
        family_best: dict[str, int] = {}
        for index in usable:
            fid = candidates[index]["forecaster_id"]
            if fid not in family_best or candidates[index]["val"]["ratio_vs_persistence_mean"] < candidates[family_best[fid]]["val"]["ratio_vs_persistence_mean"]:
                family_best[fid] = index
        families: dict[str, Any] = {}
        family_params: dict[str, Any] = {}
        for fid, index in family_best.items():
            forecaster, hparams = candidate_specs[index]
            entry: dict[str, Any] = {"hparams": hparams, "val": candidates[index]["val"], "selected": index == best_index}
            try:
                if index == best_index:
                    p_eval = params_eval
                else:
                    p_eval = forecaster.fit(series, trainval_origins, horizon, hparams)
                if test_truth is not None:
                    fam_pred = forecaster.predict(series, test_origins, horizon, p_eval)
                    fam_test = metrics(fam_pred, test_truth)
                    entry["test"] = fam_test
                    entry["skill_vs_persistence"] = _skill(fam_test["mse"], baseline_test["persistence"]["mse"]) if baseline_test else None
                p_deploy = params_deploy_cache.get(fid)
                if p_deploy is None:
                    p_deploy = forecaster.fit(series, all_origins, horizon, hparams)
                family_params[fid] = {"params_eval": p_eval, "params_deploy": p_deploy}
            except (ValueError, np.linalg.LinAlgError) as exc:
                entry["error"] = str(exc)
            families[fid] = entry
        for ref in references:
            if ref.info.id in baseline_test and ref.info.id not in families:
                families[ref.info.id] = {"hparams": {}, "test": baseline_test[ref.info.id], "reference": True,
                                          "skill_vs_persistence": _skill(baseline_test[ref.info.id]["mse"], baseline_test["persistence"]["mse"])}

        # 3) 배포용: 전체 자료로 적합.
        params_deploy = family_params.get(best_forecaster.info.id, {}).get("params_deploy") or best_forecaster.fit(
            series, all_origins, horizon, best_hparams
        )

        horizons_out[str(horizon)] = {
            "forecaster_id": best_forecaster.info.id,
            "forecaster_name": best_forecaster.info.name,
            "hparams": dict(best_hparams),
            "val": candidates[best_index]["val"],
            "val_baselines": baseline_val,
            "test": test_block,
            "test_baselines": baseline_test,
            "skill_vs_persistence": skill_vs_persistence,
            "residual_val": residuals_val,
            "params_eval": params_eval,
            "params_deploy": params_deploy,
            "families": families,
            "family_params": family_params,
            "candidates": candidates,
            "n_train_first_fold": int(folds[0][0].shape[0]),
            "n_val": n_val_total,
            "n_val_folds": len(folds),
            "n_test": int(test_origins.shape[0]),
        }

    # 다중 지평 공유 파라미터(신경망 가중치·Prophet 궤적)를 선택된 지평의 참조만 모아 저장한다.
    shared: dict[str, Any] = {}
    refs_by_forecaster: dict[str, set[str]] = {}
    for block in horizons_out.values():
        for key in ("params_eval", "params_deploy"):
            ref = (block.get(key) or {}).get("shared_ref")
            if ref:
                refs_by_forecaster.setdefault(block["forecaster_id"], set()).add(ref)
        for fid, fp in (block.get("family_params") or {}).items():
            for key in ("params_eval", "params_deploy"):
                ref = (fp.get(key) or {}).get("shared_ref")
                if ref:
                    refs_by_forecaster.setdefault(fid, set()).add(ref)
    for forecaster_id, refs in refs_by_forecaster.items():
        try:
            shared.update(registry.get(forecaster_id).export_shared(sorted(refs)))
        except Exception as exc:  # noqa: BLE001 - 공유 파라미터 실패는 산출물에 기록
            shared[f"__error__{forecaster_id}"] = str(exc)

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "station": station.to_dict() if station else {"id": "station"},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series_period": [str(series.dates[0]), str(series.dates[-1])],
        "n_valid_days": int(series.valid.sum()),
        "weather_available": bool(series.has_exog("rain_mm")),
        "diagnostics": series_diagnostics(series, split),
        "split": split.to_dict(),
        "horizon_max": int(horizon_max),
        "report_horizons": sorted(report_set),
        "selection_rule": (
            "검증 연도별 폴드(rolling origin)에서 persistence 대비 MSE 비율을 평균해 최소인 후보 선택"
            "(기준선 제외) → 학습+검증 재적합 → 시험 구간 보고"
        ),
        "horizons": horizons_out,
        "backtest": backtest,
        "shared_params": shared,
    }
    return _clean(artifact)


def series_diagnostics(series: DailySeries, split: Split) -> dict[str, Any]:
    """학습·검증 구간 대비 시험 구간의 수위 분포 이동. 체계 이동(기준점 변경·구조물)을 드러낸다."""

    trainval = series.valid & (series.dates <= split.val_end)
    test = series.valid & (series.dates >= split.test_start) & (series.dates <= split.test_end)
    out: dict[str, Any] = {
        "trainval_mean_m": float(np.mean(series.y[trainval])) if trainval.any() else None,
        "trainval_std_m": float(np.std(series.y[trainval])) if trainval.any() else None,
        "test_mean_m": float(np.mean(series.y[test])) if test.any() else None,
        "missing_days": int((~series.valid).sum()),
        "missing_ratio": float((~series.valid).mean()),
    }
    if trainval.any() and test.any() and out["trainval_std_m"]:
        shift = (out["test_mean_m"] - out["trainval_mean_m"]) / out["trainval_std_m"]
        out["test_mean_shift_sigma"] = float(shift)
        out["regime_shift_flag"] = bool(abs(shift) >= 1.0)
        out["note_ko"] = (
            "시험 연도 평균이 학습·검증 평균에서 1σ 이상 벗어남 — 기준점 변경·구조물 등 체계 이동 가능성. "
            "평균 회귀 계열은 이런 해에 불리하다."
            if abs(shift) >= 1.0
            else "시험 연도 분포가 학습·검증 분포와 유사"
        )
    return _clean(out)


def summarize_evaluation(artifacts: Mapping[str, Mapping[str, Any]], region: RegionRecord) -> dict[str, Any]:
    """지점×보고 지평 표와 지형별 승자 집계."""

    rows: list[dict[str, Any]] = []
    winners: dict[str, dict[str, int]] = {}
    for station_id, artifact in sorted(artifacts.items()):
        station = artifact.get("station", {})
        terrain = str(station.get("terrain") or "")
        diagnostics = artifact.get("diagnostics") or {}
        for horizon in artifact.get("report_horizons", []):
            block = artifact["horizons"].get(str(horizon))
            if not block:
                continue
            test = block.get("test") or {}
            base = (block.get("test_baselines") or {}).get("persistence") or {}
            clim = (block.get("test_baselines") or {}).get("climatology") or {}
            # 기상 효과: 같은 검증 폴드에서 기상 없는 최상 후보 vs 기상 포함 최상 후보의 persistence 대비 비율
            ratios_plain = [c["val"]["ratio_vs_persistence_mean"] for c in block.get("candidates", [])
                            if "val" in c and c["forecaster_id"] in ("ridge_ar",)]
            ratios_wx = [c["val"]["ratio_vs_persistence_mean"] for c in block.get("candidates", [])
                         if "val" in c and c["forecaster_id"] in ("ridge_arx",)]
            weather_gain = None
            if ratios_plain and ratios_wx:
                weather_gain = float(min(ratios_plain) - min(ratios_wx))  # 양수면 기상 포함이 더 낫다(비율 감소)
            rows.append(
                {
                    "station_id": station_id,
                    "station_name": station.get("name") or station_id,
                    "terrain": terrain,
                    "horizon": int(horizon),
                    "forecaster_id": block["forecaster_id"],
                    "forecaster_name": block.get("forecaster_name"),
                    "hparams": block.get("hparams"),
                    "val_rmse": (block.get("val") or {}).get("rmse"),
                    "test_n": test.get("n"),
                    "test_mse": test.get("mse"),
                    "test_rmse": test.get("rmse"),
                    "test_mae": test.get("mae"),
                    "persistence_test_rmse": base.get("rmse"),
                    "persistence_test_mse": base.get("mse"),
                    "climatology_test_rmse": clim.get("rmse"),
                    "skill_vs_persistence": block.get("skill_vs_persistence"),
                    "regime_shift_flag": bool(diagnostics.get("regime_shift_flag")),
                    "test_mean_shift_sigma": diagnostics.get("test_mean_shift_sigma"),
                    # 기상 사용: AR-X, AR-X 를 품은 앙상블, 그리고 강수·기온 채널을 입력으로 받는 신경망(지역에 기상 자료가 있을 때)
                    "uses_weather": block["forecaster_id"] in ("ridge_arx",) or (
                        block["forecaster_id"] == "ensemble_mean"
                        and any(m.get("forecaster_id") == "ridge_arx" for m in (block.get("params_eval") or {}).get("members", []))
                    ) or (
                        str(block["forecaster_id"]).startswith("neural_") and bool(artifact.get("weather_available"))
                    ),
                    "weather_gain_ratio": weather_gain,
                }
            )
            winners.setdefault(terrain or "미분류", {})
            winners[terrain or "미분류"][block["forecaster_id"]] = winners[terrain or "미분류"].get(block["forecaster_id"], 0) + 1
    # 모델별 성능지표 요약(지점 평균): 보고 지평마다 계열별 시험 RMSE/MSE/MAE·개선율
    per_model: dict[str, dict[str, Any]] = {}
    for station_id, artifact in sorted(artifacts.items()):
        for horizon in artifact.get("report_horizons", []):
            block = artifact["horizons"].get(str(horizon))
            if not block:
                continue
            for fid, fam in (block.get("families") or {}).items():
                test = fam.get("test") or {}
                if not isinstance(test.get("mse"), (int, float)):
                    continue
                slot = per_model.setdefault(fid, {}).setdefault(str(horizon), {"mse": [], "rmse": [], "mae": [], "skill": [], "selected": 0, "n_stations": 0})
                slot["mse"].append(test["mse"]); slot["rmse"].append(test["rmse"]); slot["mae"].append(test["mae"])
                if fam.get("skill_vs_persistence") is not None:
                    slot["skill"].append(fam["skill_vs_persistence"])
                slot["n_stations"] += 1
                slot["selected"] += int(bool(fam.get("selected")))
    model_summary = {
        fid: {
            h: {
                "n_stations": v["n_stations"], "selected_count": v["selected"],
                "mean_rmse": float(np.mean(v["rmse"])), "median_rmse": float(np.median(v["rmse"])),
                "mean_mse": float(np.mean(v["mse"])), "mean_mae": float(np.mean(v["mae"])),
                "mean_skill": float(np.mean(v["skill"])) if v["skill"] else None,
                "median_skill": float(np.median(v["skill"])) if v["skill"] else None,
            }
            for h, v in hs.items()
        }
        for fid, hs in per_model.items()
    }
    positive = [r for r in rows if r["skill_vs_persistence"] is not None and r["skill_vs_persistence"] > 0]
    return _clean(
        {
            "schema": EVALUATION_SCHEMA,
            "region_id": region.id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": next(iter(artifacts.values()))["split"] if artifacts else region.split,
            "n_stations": len(artifacts),
            "rows": rows,
            "winners_by_terrain": winners,
            "model_summary": model_summary,
            "cells_total": len(rows),
            "cells_beating_persistence": len(positive),
            "cells_weather_helps": sum(1 for r in rows if (r.get("weather_gain_ratio") or 0) > 0.005),
            "cells_with_weather_candidate": sum(1 for r in rows if r.get("weather_gain_ratio") is not None),
            "note_ko": (
                "test_* 는 학습·선택에 쓰지 않은 시험 연도 오차. skill_vs_persistence = 1 − MSE/MSE_persistence "
                "(양수면 직전값 유지보다 낫다). 모델은 지점×예측일수마다 검증 연도별 폴드에서 persistence 대비 "
                "MSE 비율 평균이 최소인 후보로 골랐다."
            ),
        }
    )


def train_region(
    registry: RegionRegistry,
    forecasters: ForecasterRegistry,
    region_id: str,
    *,
    stations: Iterable[str] | None = None,
    allowed_forecasters: Iterable[str] | None = None,
    horizon_max: int | None = None,
    write_dir: Path | None = None,
    split: Split | Mapping[str, Any] | None = None,
    fallback_artifacts_dir: Path | None = None,
    on_progress: Callable[[str | None, int, int], None] | None = None,
) -> dict[str, Any]:
    """지역의 지점들을 학습하고 산출물을 ``write_dir`` 에 저장한다.

    ``split`` 을 주면(사용자 지정 분할) 모든 지점에 그 분할을 쓰고 ``evaluation.json`` 에
    ``custom_split: true`` 로 기록한다. 비우면 지점마다 지역 기본 분할(``region.json``
    forecast.split, 없으면 마지막 2년)을 쓴다 — 기존 동작 그대로.
    ``on_progress(current_station, stations_done, stations_total)`` 은 지점을 시작할 때와 전부 끝났을 때 불린다.
    ``fallback_artifacts_dir`` 는 이번에 학습하지 않은 지점의 산출물을 평가표에 채울 때 ``write_dir`` 보다 먼저 읽는 곳(내장 산출물).
    """

    record = registry.get(region_id)
    series_set: GaugeSeriesSet = registry.load_series(region_id)
    custom_split: Split | None = None
    if split is not None:
        custom_split = split if isinstance(split, Split) else validate_split(split, series_range(series_set))
    wanted = {str(s) for s in stations} if stations else None
    targets = [s for s in series_set.stations if wanted is None or s.id in wanted]
    if wanted:
        missing = wanted - {s.id for s in targets}
        if missing:
            raise TrainingError(f"지점을 찾을 수 없습니다: {sorted(missing)}")
    if not targets:
        raise TrainingError("학습할 지점이 없습니다 — 수위 시계열을 먼저 올리세요")

    write_dir = write_dir or registry.artifacts_write_dir(region_id)
    write_dir.mkdir(parents=True, exist_ok=True)
    horizon_max = int(horizon_max or record.horizon_max)

    artifacts: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for index, station in enumerate(targets):
        if on_progress is not None:
            on_progress(station.id, index, len(targets))
        series = series_set.series[station.id]
        try:
            if custom_split is not None:
                station_split = custom_split
            else:
                station_split = Split.from_mapping(record.split, series.dates[-1]) if record.split else Split.default_for(series)
            artifact = train_station(
                series,
                station_split,
                forecasters,
                horizon_max=horizon_max,
                report_horizons=record.report_horizons,
                allowed_forecasters=allowed_forecasters,
                station=station,
            )
        except TrainingError as exc:
            failures[station.id] = str(exc)
            continue
        artifacts[station.id] = artifact
        path = RegionRegistry.station_artifact_path(write_dir, station.id)
        path.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    if on_progress is not None:
        on_progress(None, len(targets), len(targets))

    if not artifacts:
        # 한 지점도 학습되지 않았는데 평가표를 쓰면 (특히 사용자 지정 분할일 때) 그 분할로 학습된 산출물이
        # 하나도 없는 채로 split·custom_split 만 새로 기록돼 화면이 잘못된 구간을 보여 준다 — 실패로 알린다.
        detail = "; ".join(f"{sid}: {msg}" for sid, msg in failures.items())
        raise TrainingError(f"학습된 지점이 없습니다 — {detail}")

    # 기존 산출물(이번에 학습하지 않은 지점)도 평가표에 포함해 표가 비지 않게 한다.
    merged: dict[str, dict[str, Any]] = {}
    if fallback_artifacts_dir is not None and fallback_artifacts_dir.resolve() != write_dir.resolve():
        merged.update(load_artifacts(fallback_artifacts_dir))
    merged.update(load_artifacts(write_dir))
    merged.update(artifacts)
    evaluation = summarize_evaluation(merged, record)
    evaluation["trained_now"] = sorted(artifacts)
    evaluation["failures"] = failures
    evaluation["allowed_forecasters"] = sorted(allowed_forecasters) if allowed_forecasters else None
    # 분할 기록: 사용자 지정이면 요청한 분할을 그대로 적고, 지점 산출물 중 다른 분할로 학습된 것을 표시한다.
    evaluation["custom_split"] = custom_split is not None
    evaluation["split_source"] = "request" if custom_split is not None else ("region" if record.split else "default_last_2y")
    if custom_split is not None:
        evaluation["split"] = custom_split.to_dict()
    evaluation["stations_with_other_split"] = sorted(
        sid for sid, art in merged.items() if (art.get("split") or {}) != evaluation["split"]
    )
    (write_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return evaluation


def load_artifacts(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not artifacts_dir.is_dir():
        return out
    for path in sorted(artifacts_dir.glob("*.json")):
        if path.name == "evaluation.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("schema") != ARTIFACT_SCHEMA:
            continue
        station_id = str(payload.get("station", {}).get("id") or path.stem)
        out[station_id] = payload
    return out
