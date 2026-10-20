"""학습 산출물로 전망·백테스트를 제공하는 서비스 계층."""

from __future__ import annotations

import math
from datetime import date as Date
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np

from .forecasters import DailySeries, ForecasterRegistry
from .regions import GaugeSeriesSet, RegionRecord, RegionRegistry, StationNotFoundError
from .training import Split, load_artifacts, origins_for_targets, series_range, validate_split
import json  # noqa: E402

SIGMA_Z = (1.0, 2.0)


def _iso_date(value: Any) -> str:
    """기록의 date 를 비교용 ISO(YYYY-MM-DD)로 맞춘다 — 파일마다 YYYYMMDD 와 YYYY-MM-DD 가 섞여 있을 수 있다."""

    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def _in_period(value: Any, lo: str | None, hi: str | None) -> bool:
    """포함 구간 검사. lo·hi 는 ISO 문자열 또는 None(무제한). date 가 비어 있는 기록은 어느 구간에도 들지 않는다."""

    d = _iso_date(value)
    if not d:
        return False
    return (lo is None or d >= lo) and (hi is None or d <= hi)


def _period_echo(start: Date | None, end: Date | None) -> dict[str, str | None]:
    """요청한 기간을 응답에 그대로 돌려준다 — 화면·연계 쪽이 무엇으로 걸렀는지 확인할 수 있게."""

    return {"start": start.isoformat() if start else None, "end": end.isoformat() if end else None}


class ArtifactNotFoundError(KeyError):
    pass


class GaugeForecastService:
    def __init__(self, regions: RegionRegistry, forecasters: ForecasterRegistry) -> None:
        self.regions = regions
        self.forecasters = forecasters
        # 사용자가 모델을 지정해 전망할 때 즉석 적합한 파라미터. (region, station, forecaster, key, trained_at) → {h: params}
        self._adhoc_params: dict[tuple[str, str, str, str, str], dict[int, dict[str, Any]]] = {}

    def _adhoc_fit(
        self,
        region_id: str,
        station_id: str,
        forecaster_id: str,
        artifact: Mapping[str, Any],
        full: DailySeries,
        params_key: str,
        horizons: range,
    ) -> dict[int, dict[str, Any]]:
        """지정 예측기를 지평마다 즉석 적합한다. 하이퍼파라미터는 산출물의 검증 순위에서 가장 좋은 것."""

        cache_key = (region_id, station_id, forecaster_id, params_key, str(artifact.get("trained_at")))
        cached = self._adhoc_params.get(cache_key)
        if cached and all(h in cached for h in horizons):
            return cached
        forecaster = self.forecasters.get(forecaster_id)
        split = Split.from_mapping(artifact["split"])
        out: dict[int, dict[str, Any]] = dict(cached or {})
        for h in horizons:
            if h in out:
                continue
            block = artifact["horizons"].get(str(h))
            if not block:
                continue
            ranked = sorted(
                (c for c in block.get("candidates", []) if c.get("forecaster_id") == forecaster_id and "val" in c),
                key=lambda c: c["val"].get("ratio_vs_persistence_mean", 1e9),
            )
            hparams = dict(ranked[0]["hparams"]) if ranked else dict(forecaster.info.hyperparameter_grid[0])
            origins = origins_for_targets(full, h, None, split.val_end if params_key == "params_eval" else None)
            try:
                out[h] = forecaster.fit(full, origins, h, hparams) | {"_hparams": hparams}
            except ValueError:
                continue
        self._adhoc_params[cache_key] = out
        if len(self._adhoc_params) > 64:
            self._adhoc_params.pop(next(iter(self._adhoc_params)))
        return out

    # --- 조회 --------------------------------------------------------------------

    def active_artifacts(self, region_id: str) -> tuple[Path, str]:
        """활성 산출물 디렉터리와 출처. 사용자 디렉터리에 지점 산출물이 있으면 그것("user"),
        없으면 내장 산출물("builtin"). 내장 지역의 사용자 사본이 비어 있어도(되돌리기 뒤) 내장으로 떨어진다."""

        record = self.regions.get(region_id)
        user_dir = self.regions.user_artifacts_dir(record.id)
        if user_dir.is_dir() and any(user_dir.glob("*.json")):
            return user_dir, "user"
        builtin_dir = self.regions.builtin_artifacts_dir(record.id)
        if builtin_dir is not None:
            return builtin_dir, "builtin"
        return record.artifacts_dir, ("builtin" if record.origin == "builtin" else "user")

    def artifacts_dir(self, region_id: str) -> Path:
        return self.active_artifacts(region_id)[0]

    def load_station_artifact(self, region_id: str, station_id: str) -> dict[str, Any]:
        artifacts = load_artifacts(self.artifacts_dir(region_id))
        try:
            return artifacts[station_id]
        except KeyError as exc:
            raise ArtifactNotFoundError(station_id) from exc

    def split_default(self, record: RegionRecord, series_set: GaugeSeriesSet | None = None) -> dict[str, str]:
        """지역 기본 분할 — 매니페스트 forecast.split, 없으면 가장 늦게 끝나는 지점 기준 '마지막 2년'."""

        if record.split:
            return dict(record.split)
        if series_set is None and record.gauge_available:
            series_set = self.regions.load_series(record.id)
        if series_set and series_set.series:
            latest = max(series_set.series.values(), key=lambda s: s.dates[-1])
            return Split.default_for(latest).to_dict()
        return {}

    def series_range(self, region_id: str, series_set: GaugeSeriesSet | None = None) -> list[str] | None:
        """지역 시계열 범위 [시작, 끝] — 분할 검사와 화면 안내에 쓴다."""

        if series_set is None:
            series_set = self.regions.load_series(region_id)
        rng = series_range(series_set)
        return [str(rng[0]), str(rng[1])] if rng else None

    def validate_split(self, region_id: str, raw: Mapping[str, Any]) -> Split:
        """사용자 지정 분할 검사(순서·범위·최소 길이). 어긋나면 한국어 사유의 ``TrainingError``."""

        return validate_split(raw, series_range(self.regions.load_series(region_id)))

    def decorate_evaluation(self, region_id: str, payload: dict[str, Any], origin: str) -> dict[str, Any]:
        record = self.regions.get(region_id)
        split_default = self.split_default(record)
        recorded = payload.get("custom_split")
        if recorded is None:
            # 예전 산출물(플래그 없음): 사용자 디렉터리에 있고 분할이 기본과 다르면 사용자 지정으로 본다
            recorded = origin == "user" and bool(payload.get("split")) and payload.get("split") != split_default
        payload["custom_split"] = bool(recorded)
        payload["trained_at"] = payload.get("generated_at") or None
        payload["artifacts_origin"] = origin
        payload["split_default"] = split_default
        return payload

    def evaluation(self, region_id: str) -> dict[str, Any] | None:
        """활성 ``evaluation.json`` + ``custom_split``·``trained_at``·``artifacts_origin``."""

        artifacts_dir, origin = self.active_artifacts(region_id)
        path = artifacts_dir / "evaluation.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.decorate_evaluation(region_id, payload, origin)

    EXPERIMENT_FILES: dict[str, str] = {"future_rain": "future_rain.json", "area_models": "area_models.json"}

    def experiments(self, region_id: str) -> dict[str, Any]:
        """``<산출물 디렉터리>/experiments/*.json`` 의 실험 결과(미래 강수 상한 실험 등). 없으면 항목이 null.

        실험은 저장소 스크립트가 **내장** 산출물 아래에만 쓰므로 내장 디렉터리를 먼저 읽고, 없으면 활성(사용자)
        디렉터리의 사본으로 떨어진다. 공식 평가(evaluation.json)와는 별개의 파일이다.
        """

        record = self.regions.get(region_id)
        active_dir, origin = self.active_artifacts(region_id)
        roots: list[tuple[Path, str]] = []
        builtin_record = self.regions.builtin_record(record.id)
        if builtin_record is not None:
            roots.append((builtin_record.artifacts_dir, "builtin"))
        if not roots or roots[0][0].resolve() != active_dir.resolve():
            roots.append((active_dir, origin))
        items: dict[str, Any] = {}
        sources: dict[str, str | None] = {}
        for key, filename in self.EXPERIMENT_FILES.items():
            items[key] = None
            sources[key] = None
            for root, root_origin in roots:
                path = root / "experiments" / filename
                if not path.is_file():
                    continue
                try:
                    items[key] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                sources[key] = root_origin
                break
        return {
            "region_id": region_id,
            "items": items,
            "sources": sources,
            "note_ko": (
                "실험 결과는 공식 평가표(evaluation.json)와 별개다. future_rain 은 실측 미래 강수를 예보로 간주한 "
                "완전 예보 가정의 상한 실험이며 배포 모델·개선율에는 들어가지 않는다. area_models 는 공통 격자 위성 "
                "수체 면적 계열(지역·센서당 4~10점)에서 후보들을 LOO 로 비교한 실험이며, 표본이 작아 모델 우열을 "
                "가릴 수 없다는 판정과 수위→면적 검량곡선의 방어 가능 여부를 함께 담는다."
            ),
        }

    def region_weather(self, region_id: str, *, start: Date | None = None, end: Date | None = None) -> dict[str, Any]:
        """지역에 저장된 기상청 ASOS 일자료(예측기의 외생 입력과 같은 파일). 화면은 이것을 먼저 쓰고, 범위 밖만 실시간 조회한다."""

        record = self.regions.get(region_id)
        rows = self.regions.load_weather_rows(region_id)
        lo = start.isoformat() if start else None
        hi = end.isoformat() if end else None
        picked = [r for r in rows if (lo is None or str(r.get("date")) >= lo) and (hi is None or str(r.get("date")) <= hi)]
        meta = dict(record.manifest.get("weather") or {})
        return {
            "region_id": region_id, "provider": meta.get("provider"), "station_id": meta.get("station_id"), "station_name": meta.get("station_name"),
            "source": "asos_stored" if rows else "none", "range": [str(rows[0].get("date")), str(rows[-1].get("date"))] if rows else None,
            "rows": picked, "n": len(picked),
            "note_ko": "지역 레지스트리에 저장된 기상청 ASOS 일 관측(학습 때 쓴 외생 입력과 동일). 실관측이며 예보가 아니다." if rows else "이 지역에는 저장된 기상 자료가 없다.",
        }

    def region_summary(self, record: RegionRecord) -> dict[str, Any]:
        series_set = self.regions.load_series(record.id) if record.gauge_available else None
        artifacts_dir, artifacts_origin = self.active_artifacts(record.id)
        artifacts = load_artifacts(artifacts_dir)
        evaluation = self.evaluation(record.id) if artifacts else None
        split_default = self.split_default(record, series_set)
        stations = []
        if series_set:
            for station in series_set.stations:
                stations.append(station.to_dict() | {"model_trained": station.id in artifacts})
        satellite = dict(record.manifest.get("satellite") or {})
        return {
            "id": record.id,
            "name": record.name,
            "description": str(record.manifest.get("description") or ""),
            "origin": record.origin,
            "crs": str(record.manifest.get("crs") or ""),
            "bbox_wgs84": list(record.manifest.get("bbox_wgs84") or []) or None,
            "weather": dict(record.manifest.get("weather") or {}),
            "satellite_available": bool(satellite.get("available")),
            "satellite": satellite,
            "satellite_locations": record.satellite_locations(),
            "satellite_levels_available": bool(record.satellite_levels()),
            "detection_eval_available": record.detection_eval() is not None,
            "timeseries_eval_available": record.timeseries_eval() is not None,
            "weather_daily_available": bool(self.regions.load_weather_rows(record.id)),
            "scenes_available": record.scenes() is not None,
            "area_series_available": record.area_series() is not None,
            "gt_labels_available": bool(satellite.get("gt_labels_available")),
            "gauge_available": record.gauge_available,
            "stations": stations,
            "n_stations": len(stations),
            "n_stations_trained": sum(1 for s in stations if s["model_trained"]),
            "forecast": {
                "horizon_max": record.horizon_max,
                "horizons_report": record.report_horizons,
                "split": record.split,
                "split_default": split_default,
                "split_active": dict((evaluation or {}).get("split") or {}) or split_default,
                "custom_split": bool((evaluation or {}).get("custom_split")),
                "trained_at": (evaluation or {}).get("trained_at"),
                "artifacts_dir": str(artifacts_dir),
                "artifacts_origin": artifacts_origin if artifacts else None,
                "reset_available": artifacts_origin == "user" and self.regions.builtin_artifacts_dir(record.id) is not None,
                "series_range": self.series_range(record.id, series_set) if series_set else None,
            },
        }

    # --- 전망 --------------------------------------------------------------------

    def forecast(
        self,
        region_id: str,
        station_id: str,
        *,
        as_of: Date | None = None,
        horizon: int | None = None,
        history_days: int = 120,
        forecaster_id: str | None = None,
    ) -> dict[str, Any]:
        record = self.regions.get(region_id)
        series_set: GaugeSeriesSet = self.regions.load_series(region_id)
        try:
            station = series_set.station(station_id)
        except StationNotFoundError:
            raise
        full = series_set.series[station_id]
        artifact = self.load_station_artifact(region_id, station_id)
        horizon_max = int(artifact.get("horizon_max") or record.horizon_max)
        horizon = min(int(horizon or horizon_max), horizon_max)

        # 원점: 지정한 날짜(포함)까지의 관측만 쓴다. 미래를 본 파라미터도 쓰지 않는다.
        # 자료 마지막 날보다 뒤를 요청하면 마지막 날로 맞추되, 그 사실을 응답(as_of_clamped·origin_note_ko)에 남긴다.
        requested_origin = None if as_of is None else np.datetime64(as_of.isoformat(), "D")
        if requested_origin is not None and requested_origin < full.dates[0]:
            raise ValueError(
                f"원점 {requested_origin} 이전에는 {station_id} 수위 자료가 없습니다 — "
                f"자료 기간 {full.dates[0]} ~ {full.dates[-1]} 안에서 고르세요"
            )
        as_of_clamped = requested_origin is not None and requested_origin > full.dates[-1]
        origin_date = full.dates[-1] if (requested_origin is None or as_of_clamped) else requested_origin
        series = full.truncated(origin_date) if origin_date < full.dates[-1] else full
        origin_index = series.n - 1
        test_end = np.datetime64(str(artifact["split"]["test_end"])[:10], "D")
        # 원점 뒤에 실측이 남아 있고 그 실측이 시험 구간이면, 시험을 본 배포 파라미터는 누수다.
        use_eval_params = origin_date < test_end and origin_date < full.dates[-1]
        params_key = "params_eval" if use_eval_params else "params_deploy"
        val_end = np.datetime64(str(artifact["split"]["val_end"])[:10], "D")
        in_sample = origin_date <= val_end  # 학습·검증 구간 안의 원점 — 모델이 이 시기를 이미 봤다

        # 모델 지정: 산출물이 고른 모델 대신 사용자가 고른 예측기를 즉석 적합해 쓴다(기준선은 거부).
        chosen = (forecaster_id or "").strip() or None
        if chosen in ("auto", "best", "selected"):
            chosen = None
        # 공유 파라미터(신경망·Prophet)를 예측기에 다시 올린다 — 서비스 프로세스는 학습 캐시가 없다.
        shared_blobs = artifact.get("shared_params") or {}
        if shared_blobs:
            holders = {b["forecaster_id"] for b in artifact["horizons"].values()}
            if chosen:
                holders.add(chosen)  # 사용자가 고른 계열(선택되지 않은 신경망·Prophet)도 가중치를 올린다
            for fid in holders:
                try:
                    self.forecasters.get(fid).import_shared(shared_blobs)
                except KeyError:
                    continue
        adhoc: dict[int, dict[str, Any]] = {}
        if chosen:
            info = self.forecasters.get(chosen).info  # KeyError → 라우터가 422
            if info.is_baseline:
                raise ValueError(f"{chosen} 는 기준선이라 예측 모델로 쓸 수 없습니다")
            # 산출물에 계열별 파라미터가 있으면 그것을(학습 때 같은 규칙으로 적합), 없으면 즉석 적합
            stored = {
                h: artifact["horizons"][str(h)]["family_params"][chosen][params_key]
                for h in range(1, horizon + 1)
                if str(h) in artifact["horizons"] and chosen in (artifact["horizons"][str(h)].get("family_params") or {})
            }
            if stored:
                adhoc = stored
            else:
                adhoc = self._adhoc_fit(region_id, station_id, chosen, artifact, full, params_key, range(1, horizon + 1))
            if not adhoc:
                raise ValueError(
                    f"{chosen} 를 이 지점에 적합할 수 없습니다 — 필요한 입력(기상 외생 자료 등)이 없거나 자료가 부족합니다. "
                    "다른 예측기를 고르거나 forecaster 를 비워 검증에서 고른 모델을 쓰세요."
                )

        steps: list[dict[str, Any]] = []
        for h in range(1, horizon + 1):
            block = artifact["horizons"].get(str(h))
            if not block:
                continue
            if chosen:
                if h not in adhoc:
                    continue
                forecaster = self.forecasters.get(chosen)
                params = adhoc[h]
                model_used = chosen
            else:
                forecaster = self.forecasters.get(block["forecaster_id"])
                params = block[params_key]
                model_used = block["forecaster_id"]
            pred = float(forecaster.predict(series, np.array([origin_index]), h, params)[0])
            residual = block.get("residual_val") or {}
            sigma = residual.get("sigma")
            target_date = origin_date + np.timedelta64(h, "D")
            truth: float | None = None
            if target_date <= full.dates[-1]:
                value = full.y[full.index_of(target_date)]
                truth = None if math.isnan(value) else float(value)
            step = {
                "horizon": h,
                "target_date": str(target_date),
                "forecast_m": pred if math.isfinite(pred) else None,
                "forecaster_id": model_used,
                "persistence_reference_m": float(series.filled[origin_index]),
                "sigma_m": sigma,
                "q05_m": (pred + residual["q05"]) if residual.get("q05") is not None and math.isfinite(pred) else None,
                "q95_m": (pred + residual["q95"]) if residual.get("q95") is not None and math.isfinite(pred) else None,
                "observed_m": truth,
                "error_m": (truth - pred) if truth is not None and math.isfinite(pred) else None,
            }
            for z in SIGMA_Z:
                if sigma is not None and math.isfinite(pred):
                    step[f"lower_{int(z)}sigma_m"] = pred - z * sigma
                    step[f"upper_{int(z)}sigma_m"] = pred + z * sigma
                else:
                    step[f"lower_{int(z)}sigma_m"] = None
                    step[f"upper_{int(z)}sigma_m"] = None
            steps.append(step)

        history_start = max(0, origin_index - int(history_days) + 1)
        history = [
            {"date": str(series.dates[i]), "water_level_m": (None if math.isnan(series.y[i]) else float(series.y[i]))}
            for i in range(history_start, origin_index + 1)
        ]
        errors = [s["error_m"] for s in steps if s["error_m"] is not None]
        persistence_errors = [
            s["observed_m"] - s["persistence_reference_m"] for s in steps if s["observed_m"] is not None
        ]
        verification = None
        if errors:
            mse = float(np.mean(np.square(errors)))
            mse_p = float(np.mean(np.square(persistence_errors))) if persistence_errors else None
            verification = {
                "n": len(errors),
                "rmse_m": math.sqrt(mse),
                "mae_m": float(np.mean(np.abs(errors))),
                "persistence_rmse_m": math.sqrt(mse_p) if mse_p is not None else None,
                "skill_vs_persistence": (1.0 - mse / mse_p) if mse_p else None,
                "note_ko": "원점 이후 실제 관측이 있는 날짜만 비교. 원점 이전 자료로만 만든 전망이다.",
            }
        # 2026-10-20: 원점 이후 강수(예보)를 쓰는 모델이면 어떤 예보가 붙었는지 알린다(실시간: 기상청 단기예보 파일 → exog_tail)
        future_exog_feed: dict[str, Any] | None = None
        used_ids = {s["forecaster_id"] for s in steps}
        if any(getattr(self.forecasters.get(fid).info, "uses_future_exog", False) for fid in used_ids if fid):
            tail = series.exog_tail.get("rain_mm") if origin_date >= full.dates[-1] else None
            _, fmeta = self.regions.load_weather_forecast(region_id)
            n_tail = int(tail.shape[0]) if tail is not None else 0
            future_exog_feed = {
                "n_days": n_tail,
                "issued_at": fmeta.get("issued_at") if n_tail else None,
                "source": fmeta.get("provider") if n_tail else None,
                "note_ko": (
                    f"자료 끝 뒤 {n_tail}일의 강수는 기상청 단기예보(발표 {fmeta.get('issued_at')})에서 가져왔습니다."
                    if n_tail else
                    ("과거 원점이라 원점 이후 강수는 0(비 없음)으로 두었습니다 — 그때의 예보는 보관돼 있지 않습니다."
                     if origin_date < full.dates[-1] else
                     "지역에 기상청 단기예보 파일이 없어 원점 이후 강수를 0(비 없음)으로 두었습니다 — POST /regions/{id}/weather/forecast-collect 로 받으세요.")
                ),
            }
        return {
            "region_id": region_id,
            "station": station.to_dict() | {"model_trained": True},
            "as_of": str(origin_date),
            "requested_as_of": None if requested_origin is None else str(requested_origin),
            "as_of_clamped": bool(as_of_clamped),
            "data_period": [str(full.dates[0]), str(full.dates[-1])],
            "observed_days_after": sum(1 for s in steps if s["observed_m"] is not None),
            "origin_note_ko": (
                f"요청한 출발일 {requested_origin} 에는 수위 자료가 없어 자료 마지막 날 {origin_date} 을 출발일로 썼습니다 "
                f"(자료 기간 {full.dates[0]} ~ {full.dates[-1]})."
                if as_of_clamped
                else None
            ),
            "origin_level_m": float(series.filled[origin_index]),
            "origin_observed": bool(series.valid[origin_index]),
            "horizon": horizon,
            "params_used": params_key,
            "in_sample": bool(in_sample),
            "params_note_ko": (
                "원점이 학습·검증 구간 안이라 표본 내(in-sample) 전망입니다 — 모델이 이 시기를 학습에 썼으므로 성능 근거로 쓰지 말고, 성능은 시험 연도 평가표를 보세요."
                if in_sample
                else (
                    "시험 연도 안의 원점 — 시험 구간을 보지 않은 파라미터(params_eval)로 전망한 표본 외(out-of-sample) 결과"
                    if use_eval_params
                    else "전체 자료로 적합한 배포 파라미터(params_deploy)로 전망"
                )
            ),
            "band_note_ko": "σ·분위 밴드는 검증 연도 잔차에서 계산한 경험적 산포이며 모델 신뢰구간이 아니다.",
            "forecaster_override": chosen,
            "forecaster_note_ko": (
                f"사용자가 고른 예측기({chosen})를 씀 — 학습 때 계열별 검증 최상 하이퍼파라미터로 같은 규칙으로 적합한 파라미터. 밴드는 검증에서 고른 모델의 잔차 산포를 그대로 쓴다."
                if chosen
                else "예측일수마다 검증 연도별 폴드에서 고른 모델을 쓴다."
            ),
            "steps": steps,
            "history": history,
            "verification": verification,
            "models_by_horizon": {str(s["horizon"]): s["forecaster_id"] for s in steps},
            "future_exog_feed": future_exog_feed,
            "coverage_note_ko": (
                None
                if any(s.get("forecast_m") is not None for s in steps)
                else (
                    f"원점({origin_date}) 이전 관측이 {int(series.valid.sum())}일뿐이라 모델 입력(최근 수십 일 + 계절 특징)을 만들 수 없어 "
                    "예측값이 비어 있습니다. 더 뒤의 날짜를 원점으로 고르세요."
                )
            ),
        }

    def forecast_compare(
        self, region_id: str, station_id: str, *, as_of: Date | None = None, horizon: int | None = None
    ) -> dict[str, Any]:
        """모든 계열의 전망을 한 번에: 지평별 예측·선택 모델 대비 차이·(원점 이후 실측이 있으면) 모델별 검증 지표."""

        base = self.forecast(region_id, station_id, as_of=as_of, horizon=horizon, history_days=90)
        artifact = self.load_station_artifact(region_id, station_id)
        families = sorted({fid for b in artifact["horizons"].values() for fid in (b.get("family_params") or {})})
        runs: dict[str, dict[str, Any]] = {}
        for fid in families:
            try:
                runs[fid] = self.forecast(region_id, station_id, as_of=as_of, horizon=horizon, history_days=7, forecaster_id=fid)
            except (KeyError, ValueError):
                continue
        table = []
        for step in base["steps"]:
            h = step["horizon"]
            row = {"horizon": h, "target_date": step["target_date"], "selected_model": step["forecaster_id"], "selected_m": step["forecast_m"],
                   "persistence_m": step["persistence_reference_m"], "observed_m": step["observed_m"], "models": {}}
            for fid, run in runs.items():
                st = next((x for x in run["steps"] if x["horizon"] == h), None)
                if st and st.get("forecast_m") is not None:
                    row["models"][fid] = {"forecast_m": st["forecast_m"], "diff_vs_selected_m": st["forecast_m"] - step["forecast_m"] if step.get("forecast_m") is not None else None}
            table.append(row)
        # 모델별 검증(원점 이후 실측 있을 때)과 시험 연도 성능(산출물)
        per_model = {}
        for fid, run in runs.items():
            v = run.get("verification")
            hz = str(min(int(horizon or 7), 30))
            fam = ((artifact["horizons"].get(hz) or {}).get("families") or {}).get(fid) or {}
            per_model[fid] = {
                "verification": v,
                "test_at_horizon": {"horizon": int(hz), **(fam.get("test") or {}), "skill_vs_persistence": fam.get("skill_vs_persistence"), "selected": bool(fam.get("selected"))},
            }
        return {
            "region_id": region_id, "station": base["station"], "as_of": base["as_of"], "horizon": base["horizon"],
            "in_sample": base.get("in_sample"), "params_note_ko": base.get("params_note_ko"),
            "origin_level_m": base["origin_level_m"], "history": base["history"], "selected": base["steps"],
            "models": sorted(runs), "table": table, "per_model": per_model,
            "note_ko": "모든 계열은 학습 때 검증 최상 하이퍼파라미터로 같은 규칙으로 적합된 파라미터를 쓴다. 선택 모델은 검증 연도별 폴드 성적이 가장 좋은 것.",
        }

    def backtest(self, region_id: str, station_id: str, horizon: int) -> dict[str, Any]:
        artifact = self.load_station_artifact(region_id, station_id)
        block = artifact["horizons"].get(str(int(horizon)))
        trace = artifact.get("backtest", {}).get(str(int(horizon)))
        if not block or not trace:
            available = sorted(int(k) for k in artifact.get("backtest", {}))
            raise KeyError(f"D+{horizon} 백테스트 궤적이 없습니다. 보고 지평: {available}")
        return {
            "region_id": region_id,
            "station": artifact.get("station"),
            "horizon": int(horizon),
            "split": artifact.get("split"),
            "forecaster_id": block["forecaster_id"],
            "forecaster_name": block.get("forecaster_name"),
            "hparams": block.get("hparams"),
            "test": block.get("test"),
            "test_baselines": block.get("test_baselines"),
            "skill_vs_persistence": block.get("skill_vs_persistence"),
            "trace": trace,
        }

    def station_detail(self, region_id: str, station_id: str) -> dict[str, Any]:
        artifact = self.load_station_artifact(region_id, station_id)
        horizons = {}
        for key, block in artifact["horizons"].items():
            horizons[key] = {
                k: v
                for k, v in block.items()
                if k not in {"params_eval", "params_deploy", "candidates", "family_params"}
            }
            horizons[key]["candidates"] = [
                {"forecaster_id": c["forecaster_id"], "hparams": c.get("hparams"), "val_rmse": (c.get("val") or {}).get("rmse"), "selected": bool(c.get("selected")), "error": c.get("error")}
                for c in block.get("candidates", [])
            ]
        return {k: v for k, v in artifact.items() if k not in {"horizons", "backtest"}} | {"horizons": horizons}

    def register_scene(
        self,
        region_id: str,
        *,
        sensor: str,
        sensor_type: str,
        image_date: str,
        pre_tif: Path,
        mask_tif: Path | None,
        gt_tif: Path | None = None,
        source_note: str | None = None,
    ) -> dict[str, Any]:
        """새 위성 장면을 지역 자산으로 등록한다(퀵룩·오버레이·공통 격자 마스크·scenes.json).

        마스크가 없으면 '탐지 대기' 장면으로 등록되고(사진만), 마스크가 오면 같은 scene_id 로 다시 등록해 갱신한다.
        """

        from .scene_assets import build_scene_record, existing_frame_bounds, upsert_scene

        record = self.regions.get(region_id)
        date_iso = f"{image_date[:4]}-{image_date[4:6]}-{image_date[6:8]}" if len(image_date) == 8 and image_date.isdigit() else str(image_date)[:10]
        scene_id = f"{date_iso.replace('-', '')}_{sensor}"
        sat_dir = self.regions.satellite_write_dir(region_id)
        bounds = existing_frame_bounds(sat_dir, sensor)
        scene = build_scene_record(
            sat_dir=sat_dir, scene_id=scene_id, date_iso=date_iso, sensor=sensor, sensor_type=sensor_type,
            pre_tif=pre_tif, mask_tif=mask_tif, gt_tif=gt_tif, frame_bounds=bounds, source_note=source_note,
        )
        upsert_scene(sat_dir, scene, region_id)
        # 지역 정의에 위성 가용·센서·장면 날짜를 반영
        target = sat_dir.parent
        manifest_path = target / "region.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        satellite = manifest.setdefault("satellite", {})
        satellite["available"] = True
        sensors = list(satellite.get("sensors") or [])
        if sensor not in sensors:
            sensors.append(sensor)
        satellite["sensors"] = sensors
        scenes_by_sensor = dict(satellite.get("scenes") or {})
        dates = sorted(set(scenes_by_sensor.get(sensor, [])) | {date_iso})
        scenes_by_sensor[sensor] = dates
        satellite["scenes"] = scenes_by_sensor
        satellite["scenes_file"] = "satellite/scenes.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"region_id": region_id, "scene": scene, "assets_dir": str(sat_dir), "frame_grid_reused": bounds is not None}

    def scenes(self, region_id: str, api_prefix: str = "/api/v1") -> dict[str, Any]:
        record = self.regions.get(region_id)
        payload = record.scenes()
        if payload is None:
            return {"region_id": region_id, "available": False, "scenes": [], "note_ko": "이 지역에는 위성 장면 자산이 없다."}
        scenes = []
        for scene in payload.get("scenes") or []:
            item = dict(scene)
            images = {}
            for kind, rel in (scene.get("images") or {}).items():
                images[kind] = f"{api_prefix}/regions/{region_id}/satellite/files/{rel}"
            item["image_urls"] = images
            frame = scene.get("frame")
            if frame:
                item["frame_url"] = f"{api_prefix}/regions/{region_id}/satellite/files/{frame['path']}"
            item.setdefault("area_common_frame_km2", None)
            scenes.append(item)
        sensors = sorted({s["sensor"] for s in scenes})
        return {
            "region_id": region_id,
            "available": bool(scenes),
            "generated_at": payload.get("generated_at"),
            "note_ko": payload.get("note_ko"),
            "sensors": sensors,
            "scenes": scenes,
            "n_scenes": len(scenes),
            "n_with_masks": sum(1 for s in scenes if s.get("detection_available")),
            "area_series_available": record.area_series() is not None,
            "area_note_ko": (
                "water_area_km2 는 씬마다 자기 촬영범위에서 센 값이라 날짜 간 비교 불가. 날짜 간 비교에는"
                " area_common_frame_km2(센서별 공통 격자로 다시 센 면적)를 쓸 것 —"
                f" 자세한 계열은 {api_prefix}/regions/{region_id}/satellite/area-series."
            ),
        }

    def area_series(self, region_id: str, *, start: Date | None = None, end: Date | None = None) -> dict[str, Any]:
        """공통 격자 면적 시계열. 파일이 없으면 available=false·series=null 로 안전하게 답한다.

        start·end(포함)를 주면 센서 블록마다 `rows`(있으면 `scenes` 도)를 date 로 거른다. 그 밖의 값(상관계수·n_scenes 등)은
        전체 기록 기준 그대로 두고 다시 계산하지 않는다 — 기간 조회는 열람용이지 지표 재산출이 아니다.
        """

        record = self.regions.get(region_id)
        payload = record.area_series()
        period = _period_echo(start, end)
        if payload is None:
            return {
                "region_id": region_id, "available": False, "sensors": None, "period": period,
                "note_ko": ("이 지역에는 공통 격자 면적 시계열이 없다 —"
                            " scripts/build_common_frame_area.py 를 돌리면 생긴다(위성 마스크가 있어야 한다)."),
            }
        body = {k: v for k, v in payload.items() if k != "region_id"}
        if (start or end) and isinstance(body.get("sensors"), Mapping):
            lo, hi = period["start"], period["end"]
            sensors: dict[str, Any] = {}
            for sensor, block in body["sensors"].items():
                if not isinstance(block, Mapping):
                    sensors[sensor] = block
                    continue
                filtered = dict(block)
                for key in ("rows", "scenes"):
                    items = block.get(key)
                    if isinstance(items, list):
                        filtered[key] = [r for r in items if not isinstance(r, Mapping) or _in_period(r.get("date"), lo, hi)]
                sensors[sensor] = filtered
            body["sensors"] = sensors
        return {"region_id": region_id, "available": True, "period": period, **body}

    def timeseries_eval(self, region_id: str) -> dict[str, Any]:
        record = self.regions.get(region_id)
        payload = record.timeseries_eval()
        if payload is None:
            return {"region_id": region_id, "available": False, "series": {},
                    "note_ko": "이 지역에는 위성 기반 시계열 지표 기록이 없다(위성 시점이 없거나 수위 산출·GT가 없다)."}
        return {"region_id": region_id, "available": True, **{k: v for k, v in payload.items() if k != "region_id"}}

    def detection_eval(self, region_id: str) -> dict[str, Any]:
        record = self.regions.get(region_id)
        payload = record.detection_eval()
        if payload is None:
            has_scenes = bool((record.scenes() or {}).get("scenes"))
            return {"region_id": region_id, "available": False, "sensors": {}, "has_scenes": has_scenes,
                    "note_ko": (
                        "이 지역에는 GT 라벨이 없어 탐지 정확도를 계산할 수 없다 — 탐지 결과는 있어도 IoU 를 주장하지 않는다."
                        if has_scenes else
                        "이 지역에는 아직 위성 장면이 없어 탐지 정확도를 계산할 대상이 없다 — 장면을 등록하면 탐지·대조 기능이 열린다."
                    )}
        return {"region_id": region_id, "available": True, **{k: v for k, v in payload.items() if k != "region_id"}}

    def satellite_levels(
        self, region_id: str, station_id: str | None = None, *, start: Date | None = None, end: Date | None = None,
    ) -> dict[str, Any]:
        """위성 산출 수위 기록. station_id 로 지점을, start·end(포함)로 기간을 좁힌다 — 둘 다 없으면 전체."""

        record = self.regions.get(region_id)
        records = record.satellite_levels()
        links = record.satellite_links()
        period = _period_echo(start, end)
        loc = links.get(station_id) if station_id else None
        if station_id and not loc:
            return {"region_id": region_id, "station_id": station_id, "loc_id": None, "location": None, "records": [], "n": 0,
                    "period": period, "note_ko": "이 지점에는 연결된 위성 수위 산출 위치가 없습니다"}
        location = record.satellite_locations().get(str(loc)) if loc else None
        if loc:
            records = [r for r in records if str(r.get("loc_id")) == str(loc)]
        if start or end:
            records = [r for r in records if _in_period(r.get("date"), period["start"], period["end"])]
        # 같은 날 수위계 값이 있으면 붙여 준다 — 대조표가 화면·API 양쪽에서 같아야 한다
        gauge_by_date: dict[str, float] = {}
        if station_id and record.gauge_available:
            series_set = self.regions.load_series(region_id)
            if station_id in series_set.raw_rows:
                gauge_by_date = {str(r["date"]): float(r["water_level_m"]) for r in series_set.raw_rows[station_id]}
        out = []
        for r in sorted(records, key=lambda x: (str(x.get("date")), str(x.get("loc_id")))):
            item = dict(r)
            gauge = gauge_by_date.get(str(r.get("date")))
            item["gauge_level_m"] = gauge
            corrected = r.get("corrected_water_level_m")
            item["corrected_minus_gauge_m"] = (corrected - gauge) if isinstance(corrected, (int, float)) and gauge is not None else None
            out.append(item)
        errors = [abs(x["corrected_minus_gauge_m"]) for x in out if x["corrected_minus_gauge_m"] is not None]
        summary = None
        if errors:
            summary = {"n": len(errors), "rmse_m": float(np.sqrt(np.mean(np.square(errors)))), "mae_m": float(np.mean(errors))}
        return {"region_id": region_id, "station_id": station_id, "loc_id": loc, "location": location, "records": out, "n": len(out),
                "period": period, "comparison": summary,
                "note_ko": "위성 수위는 관측일 산출이며 예측이 아니다. corrected_* 는 지역별 보정(부산: 융합 LSTM 당일 보정, 대청: 수위대 상수)."}

    def series_rows(
        self, region_id: str, station_id: str, *, start: Date | None = None, end: Date | None = None
    ) -> dict[str, Any]:
        series_set = self.regions.load_series(region_id)
        station = series_set.station(station_id)
        rows = series_set.raw_rows[station_id]
        if start:
            rows = [r for r in rows if str(r["date"]) >= start.isoformat()]
        if end:
            rows = [r for r in rows if str(r["date"]) <= end.isoformat()]
        trained = (self.artifacts_dir(region_id) / f"{station.id}.json").is_file()
        return {"region_id": region_id, "station": station.to_dict() | {"model_trained": trained}, "rows": rows, "n": len(rows)}
