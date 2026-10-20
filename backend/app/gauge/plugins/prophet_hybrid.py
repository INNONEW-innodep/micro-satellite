"""Prophet(구 Facebook) 하이브리드 예측기 플러그인.

Prophet 은 '추세(변화점) + 연 주기 푸리에 계절성' 을 곡선으로 적합하는 모델이라 원점
상태를 보지 않는다. 그래서 여기서는 Prophet 이 준 곡선 위에 **잔차의 자기회귀**를 얹는다:

    ŷ[t+h] = prophet(t+h) + AR_h( y[t]−prophet(t), y[t−1]−prophet(t−1), … )

Prophet 은 폴드마다 한 번만 적합(수 초)하고, 곡선을 시계열 범위 + 60일까지 배열로
저장하므로 서버(prophet 없음)에서도 numpy 만으로 추론한다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..forecasters import DailySeries, ForecasterInfo, GaugeForecaster, ridge_fit, ridge_predict, rolling_mean

EXTRA_DAYS = 60


class ProphetHybrid(GaugeForecaster):
    def __init__(self) -> None:
        self._shared: dict[str, dict[str, Any]] = {}

    @property
    def info(self) -> ForecasterInfo:
        return ForecasterInfo(
            id="prophet_hybrid",
            name="Prophet + 잔차 자기회귀",
            description=(
                "Prophet(추세 변화점 + 연 주기 계절성)으로 곡선을 적합하고, 그 곡선 대비 잔차의 최근 p일로 h일 뒤 잔차를 릿지 회귀. "
                "Prophet 은 폴드마다 한 번 적합하고 곡선을 저장해 numpy 로 추론한다."
            ),
            hyperparameter_grid=({"p": 7, "lam": 10.0, "changepoint_prior": 0.05}, {"p": 14, "lam": 10.0, "changepoint_prior": 0.2}),
            min_history_days=730,
        )

    @staticmethod
    def _key(series: DailySeries, last_index: int, hparams: Mapping[str, Any]) -> str:
        h = hashlib.sha1()
        h.update(str(series.dates[0]).encode()); h.update(str(series.n).encode())
        h.update(np.ascontiguousarray(series.filled[::7]).tobytes()[:20000])
        h.update(str(int(last_index)).encode()); h.update(json.dumps({"cp": hparams.get("changepoint_prior", 0.05)}).encode())
        return f"prophet:{h.hexdigest()[:16]}"

    def _curve(self, series: DailySeries, last_index: int, hparams: Mapping[str, Any]) -> dict[str, Any]:
        key = self._key(series, last_index, hparams)
        if key in self._shared:
            return self._shared[key]
        try:
            import pandas as pd
            from prophet import Prophet
        except ImportError:
            raise ValueError("prophet 이 없어 학습할 수 없습니다 (추론은 numpy 로 가능)") from None
        import logging

        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        idx = np.arange(last_index + 1)
        valid = series.valid[idx]
        frame = pd.DataFrame({"ds": pd.to_datetime(series.dates[idx][valid].astype("datetime64[D]")), "y": series.y[idx][valid]})
        model = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            changepoint_prior_scale=float(hparams.get("changepoint_prior", 0.05)),
        )
        model.fit(frame)
        future_dates = np.arange(series.dates[0], series.dates[-1] + np.timedelta64(EXTRA_DAYS + 1, "D"), dtype="datetime64[D]")
        future = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
        yhat = model.predict(future)["yhat"].to_numpy(dtype=np.float64)
        blob = {"start": str(series.dates[0]), "yhat": yhat, "last_index": int(last_index), "n_train": int(valid.sum())}
        self._shared[key] = blob
        return blob

    @staticmethod
    def _residual_features(series: DailySeries, yhat: np.ndarray, origins: np.ndarray, p: int) -> np.ndarray:
        resid = series.filled - yhat[: series.n]
        lags = np.stack(
            [np.where(origins - i >= 0, resid[np.maximum(origins - i, 0)], np.nan) for i in range(p)],
            axis=1,
        )
        roll30 = rolling_mean(resid, 30, origins)[:, None]
        return np.concatenate([lags, roll30], axis=1)

    def fit(self, series, origins, horizon, hparams) -> dict[str, Any]:
        p = int(hparams.get("p", 7))
        lam = float(hparams.get("lam", 10.0))
        origins = self._usable_origins(series, origins, horizon)
        last_index = int(origins.max() + horizon)
        blob = self._curve(series, last_index, hparams)
        yhat = blob["yhat"]
        X = self._residual_features(series, yhat, origins, p)
        ok = ~np.isnan(X).any(axis=1)
        origins, X = origins[ok], X[ok]
        if origins.shape[0] < X.shape[1] + 5:
            raise ValueError("Prophet 잔차 회귀 원점이 부족합니다")
        # 잔차의 **변화량**을 학습한다: Prophet 곡선은 적합 구간 밖에서 표류하므로 잔차 절대값을
        # 맞추면 검증 구간에서 크게 틀린다. 변화량이면 벌점이 커질수록 '현재 잔차 유지'로 수렴한다.
        resid_now = series.filled[origins] - yhat[origins]
        target = (series.y[origins + horizon] - yhat[origins + horizon]) - resid_now
        params = ridge_fit(X, target, lam)
        params.update({"p": p, "lam": lam, "shared_ref": self._key(series, last_index, hparams), "hparams": dict(hparams)})
        return params

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        out = np.full(origins.shape[0], np.nan, dtype=np.float64)
        blob = self._shared.get(str(params.get("shared_ref")))
        if blob is None:
            return out
        yhat = np.asarray(blob["yhat"], dtype=np.float64)
        offset = int((series.dates[0] - np.datetime64(str(blob["start"]), "D")).astype(int))
        if offset < 0:
            return out
        aligned = yhat[offset:]
        if aligned.shape[0] < series.n:
            return out
        X = self._residual_features(series, aligned, origins, int(params["p"]))
        ok = ~np.isnan(X).any(axis=1)
        target_idx = origins + int(horizon)
        ok &= target_idx < aligned.shape[0]
        if ok.any():
            resid_now = series.filled[origins[ok]] - aligned[origins[ok]]
            out[ok] = aligned[target_idx[ok]] + resid_now + ridge_predict(X[ok], params)
        return out

    def export_shared(self, refs: Iterable[str]) -> dict[str, Any]:
        out = {}
        for ref in refs:
            blob = self._shared.get(ref)
            if blob:
                out[ref] = {**{k: v for k, v in blob.items() if k != "yhat"}, "yhat": [round(float(v), 4) for v in blob["yhat"]]}
        return out

    def import_shared(self, blobs: Mapping[str, Any]) -> None:
        for ref, blob in blobs.items():
            if isinstance(blob, Mapping) and str(ref).startswith("prophet:") and "yhat" in blob:
                self._shared[ref] = {**blob, "yhat": np.asarray(blob["yhat"], dtype=np.float64)}
