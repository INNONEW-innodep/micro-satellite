"""교체 가능한 수위 시계열 예측기.

모든 예측기는 **직접(direct) 다중 지평** 방식이다: 예측일수 ``h`` 마다 따로
학습하고, 원점 ``t`` 의 특징으로 ``y[t+h]`` 를 바로 낸다. 한 걸음씩 재귀하면
오차가 누적되고 모델마다 누적 방식이 달라 비교가 흐려진다.

새 모델은 :class:`GaugeForecaster` 를 구현하고 ``GAUGE_FORECASTER_PLUGINS``
(``package.module:ClassOrFactory``, 쉼표 구분) 로 등록하면 화면·API 변경 없이
후보에 들어간다. 학습 파라미터는 JSON으로 직렬화 가능해야 한다 — 산출물이
지역 디렉터리에 파일로 남고 다른 프로세스가 그것만 읽어 전망하기 때문이다.

``is_baseline`` 예측기(직전값 유지, 계절 평년값)는 skill 기준선으로만 계산되고
**배포 모델로 선택되지 않는다.** 직전값을 유지하는 것은 예측이 아니다.
"""

from __future__ import annotations

import importlib
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Params = dict[str, Any]
MAX_EXOG_TAIL = 45  # 달력 끝 뒤에 붙일 수 있는 외생 입력(예보) 일수 상한
FLOOD_MONTHS = (6, 7, 8, 9)  # 홍수기 — 계절 분리 예측기의 그룹


# --- 시계열 컨테이너 -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DailySeries:
    """연속 일 달력 위에 놓인 수위 시계열.

    ``y`` 는 결측일에 NaN, ``filled`` 는 특징 구성용 선형 보간값이다. 학습·평가의
    **정답은 항상 ``y``** 에서만 읽고, 보간값을 정답으로 쓰지 않는다.
    """

    dates: np.ndarray  # datetime64[D], 길이 n, 하루 간격
    y: np.ndarray  # float64, NaN = 결측
    filled: np.ndarray  # float64, 결측을 선형 보간
    doy: np.ndarray  # int, 1..366
    # 외생 입력(같은 달력에 정렬): 예) rain_mm, temp_c. 없으면 빈 dict. 결측은 NaN.
    exog: dict[str, np.ndarray] = field(default_factory=dict)
    # 2026-10-20: 달력 **끝 다음 날부터**의 외생 입력(기상청 단기예보 강수 등, 최대 MAX_EXOG_TAIL 일). 원점이 자료 끝일 때
    # ``future_rain_features`` 가 이어 붙여 읽는다. ``truncated`` 로 과거 원점을 만들면 비워진다(그때의 예보는 모른다).
    exog_tail: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.dates.shape[0])

    def has_exog(self, *names: str) -> bool:
        return all(name in self.exog for name in names)

    def with_exog(self, rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]) -> DailySeries:
        """``rows`` (date + 열)에서 ``mapping`` {열: exog 이름} 으로 외생 배열을 붙인다."""

        arrays = {name: np.full(self.n, np.nan, dtype=np.float64) for name in mapping.values()}
        tails = {name: np.full(MAX_EXOG_TAIL, np.nan, dtype=np.float64) for name in mapping.values()}
        tail_len = 0
        start = self.dates[0]
        for row in rows:
            raw = str(row.get("date") or "")[:10]
            if not raw:
                continue
            try:
                offset = int((np.datetime64(raw, "D") - start).astype(int))
            except ValueError:
                continue
            if offset < 0 or offset >= self.n + MAX_EXOG_TAIL:
                continue
            target = arrays if offset < self.n else tails
            pos = offset if offset < self.n else offset - self.n
            for column, name in mapping.items():
                value = row.get(column)
                if value is None:
                    continue
                try:
                    target[name][pos] = float(value)
                except (TypeError, ValueError):
                    continue
                if offset >= self.n:
                    tail_len = max(tail_len, pos + 1)
        new_tail = dict(self.exog_tail)
        if tail_len:
            new_tail.update({name: arr[:tail_len] for name, arr in tails.items()})
        return DailySeries(dates=self.dates, y=self.y, filled=self.filled, doy=self.doy, exog={**self.exog, **arrays}, exog_tail=new_tail)

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.y)

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> DailySeries:
        """``[{"date": "YYYY-MM-DD", "water_level_m": float}, ...]`` 에서 만든다."""

        pairs: dict[np.datetime64, float] = {}
        for row in rows:
            raw_date = str(row.get("date") or "").strip()
            if not raw_date:
                continue
            if len(raw_date) == 8 and raw_date.isdigit():
                raw_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            value = row.get("water_level_m")
            if value is None or value == "":
                continue
            level = float(value)
            if not math.isfinite(level):
                continue
            pairs[np.datetime64(raw_date[:10], "D")] = level
        if not pairs:
            raise ValueError("시계열에 유효한 (date, water_level_m) 행이 없습니다")
        start = min(pairs)
        end = max(pairs)
        dates = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
        y = np.full(dates.shape[0], np.nan, dtype=np.float64)
        offsets = ((np.array(list(pairs)) - start).astype("timedelta64[D]")).astype(int)
        y[offsets] = np.array(list(pairs.values()), dtype=np.float64)
        return cls.from_arrays(dates, y)

    @classmethod
    def from_arrays(cls, dates: np.ndarray, y: np.ndarray) -> DailySeries:
        dates = np.asarray(dates, dtype="datetime64[D]")
        y = np.asarray(y, dtype=np.float64)
        if dates.shape != y.shape:
            raise ValueError("dates와 y의 길이가 다릅니다")
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            raise ValueError("유효 관측이 2개 미만입니다")
        index = np.arange(y.shape[0])
        filled = np.interp(index, index[valid], y[valid])
        return cls(dates=dates, y=y, filled=filled, doy=day_of_year(dates))

    def truncated(self, end: np.datetime64) -> DailySeries:
        """``end`` (포함) 이후를 잘라낸다. 전망 원점을 과거로 옮길 때 쓴다."""

        end = np.datetime64(end, "D")
        keep = self.dates <= end
        if keep.sum() < 2:
            raise ValueError("잘라낸 뒤 남는 관측이 2개 미만입니다")
        cut = DailySeries.from_arrays(self.dates[keep], self.y[keep])
        if self.exog:
            cut = DailySeries(dates=cut.dates, y=cut.y, filled=cut.filled, doy=cut.doy,
                              exog={k: v[keep] for k, v in self.exog.items()})
        return cut

    def index_of(self, date: np.datetime64) -> int:
        offset = int((np.datetime64(date, "D") - self.dates[0]).astype(int))
        if offset < 0 or offset >= self.n:
            raise IndexError(f"{date} 는 시계열 범위 밖입니다")
        return offset


def day_of_year(dates: np.ndarray) -> np.ndarray:
    dates = np.asarray(dates, dtype="datetime64[D]")
    year_start = dates.astype("datetime64[Y]").astype("datetime64[D]")
    return (dates - year_start).astype(int) + 1


def target_doy(series: DailySeries, origins: np.ndarray, horizon: int) -> np.ndarray:
    """원점에서 ``horizon`` 일 뒤 날짜의 연중일. 시계열 끝을 넘어도 계산된다."""

    target_dates = series.dates[origins] + np.timedelta64(int(horizon), "D")
    return day_of_year(target_dates)


def rolling_mean(values: np.ndarray, window: int, origins: np.ndarray) -> np.ndarray:
    """원점을 포함해 뒤로 ``window`` 일의 평균. 창이 시계열 앞을 벗어나면 NaN."""

    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    upper = origins + 1
    lower = upper - window
    out = np.full(origins.shape[0], np.nan, dtype=np.float64)
    ok = lower >= 0
    out[ok] = (cumsum[upper[ok]] - cumsum[lower[ok]]) / float(window)
    return out


def seasonal_features(doy: np.ndarray, harmonics: int = 2) -> np.ndarray:
    angle = 2.0 * np.pi * doy.astype(np.float64) / 365.25
    columns = []
    for k in range(1, harmonics + 1):
        columns.append(np.sin(k * angle))
        columns.append(np.cos(k * angle))
    return np.stack(columns, axis=1)


def climatology(series: DailySeries, origins_mask: np.ndarray, smooth_days: int = 15) -> np.ndarray:
    """연중일별 평년 수위(366칸). 학습 구간의 유효 관측만 쓰고 ±smooth_days 원형 평활."""

    sums = np.zeros(367, dtype=np.float64)
    counts = np.zeros(367, dtype=np.float64)
    use = origins_mask & series.valid
    np.add.at(sums, series.doy[use], series.y[use])
    np.add.at(counts, series.doy[use], 1.0)
    sums, counts = sums[1:], counts[1:]  # 0번 칸 제거 → 인덱스 = doy-1
    kernel = np.ones(2 * smooth_days + 1)
    padded_sum = np.concatenate([sums[-smooth_days:], sums, sums[:smooth_days]])
    padded_cnt = np.concatenate([counts[-smooth_days:], counts, counts[:smooth_days]])
    smoothed_sum = np.convolve(padded_sum, kernel, mode="valid")
    smoothed_cnt = np.convolve(padded_cnt, kernel, mode="valid")
    overall = float(np.nanmean(series.y[use])) if use.any() else float(np.nanmean(series.y))
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(smoothed_cnt > 0, smoothed_sum / smoothed_cnt, overall)
    return clim


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> Params:
    """절편은 벌점 없이, 표준화한 특징에 L2 벌점을 주는 릿지 회귀(닫힌 해)."""

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd
    n, k = Xs.shape
    A = np.concatenate([np.ones((n, 1)), Xs], axis=1)
    penalty = lam * np.eye(k + 1)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(A.T @ A + penalty, A.T @ y)
    return {
        "intercept": float(weights[0]),
        "coef": [float(v) for v in weights[1:]],
        "mu": [float(v) for v in mu],
        "sd": [float(v) for v in sd],
    }


def ridge_predict(X: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    mu = np.asarray(params["mu"], dtype=np.float64)
    sd = np.asarray(params["sd"], dtype=np.float64)
    coef = np.asarray(params["coef"], dtype=np.float64)
    return float(params["intercept"]) + ((X - mu) / sd) @ coef


def exog_features(series: DailySeries, origins: np.ndarray) -> np.ndarray | None:
    """원점까지의 강수·기온 요약 특징. 외생 자료가 없으면 None.

    미래 기상(목표일까지의 비)은 원점에서 알 수 없으므로 **원점 이전** 값만 쓴다:
    강수 1·3·7·30일 합(log1p), 7일 평균기온·기온 편차. 결측 강수는 0으로 둔다.
    """

    if not series.has_exog("rain_mm"):
        return None
    origins = np.asarray(origins, dtype=int)
    rain = np.nan_to_num(series.exog["rain_mm"], nan=0.0)
    cols = []
    for window in (1, 3, 7, 30):
        cols.append(np.log1p(np.maximum(rolling_mean(rain, window, origins) * window, 0.0)))
    if series.has_exog("temp_c"):
        temp = series.exog["temp_c"]
        temp_filled = np.where(np.isnan(temp), np.nanmean(temp) if np.isfinite(np.nanmean(temp)) else 0.0, temp)
        cols.append(rolling_mean(temp_filled, 7, origins))
        cols.append(temp_filled[origins] - rolling_mean(temp_filled, 30, origins))
    return np.stack(cols, axis=1)


def dam_features(series: DailySeries, origins: np.ndarray) -> np.ndarray | None:
    """원점까지의 댐 운영 요약 특징(2026-10-20). 외생 ``dam_out``(총방류량 m³/s)이 없으면 None.

    총방류 1·3·7일 합(log1p), 유입 1·3일 합(log1p, 있으면), 저수위 30일 평균 대비 편차(있으면). 결측 방류는 0.
    원점 **이전** 값만 쓴다 — 방류 계획(예고)은 공개 API 가 없어 입력하지 않는다(docs/presentation/댐방류_D+1_실험_2026-10-16.md).
    """

    if not series.has_exog("dam_out"):
        return None
    origins = np.asarray(origins, dtype=int)
    out = np.nan_to_num(series.exog["dam_out"], nan=0.0)
    cols = [np.log1p(np.maximum(rolling_mean(out, w, origins) * w, 0.0)) for w in (1, 3, 7)]
    if series.has_exog("dam_in"):
        inflow = np.nan_to_num(series.exog["dam_in"], nan=0.0)
        cols += [np.log1p(np.maximum(rolling_mean(inflow, w, origins) * w, 0.0)) for w in (1, 3)]
    if series.has_exog("dam_level"):
        level = series.exog["dam_level"]
        mean = np.nanmean(level) if np.isfinite(np.nanmean(level)) else 0.0
        level_filled = np.where(np.isnan(level), mean, level)
        cols.append(level_filled[origins] - rolling_mean(level_filled, 30, origins))
    return np.stack(cols, axis=1)


def future_rain_features(
    series: DailySeries, origins: np.ndarray, horizon: int, future_days: int | str
) -> tuple[np.ndarray, int]:
    """원점 **이후** ``origin+1 .. origin+min(h, K)`` 의 강수 합(log1p)·일 최대(log1p). ``K`` = ``future_days``
    (정수) 또는 ``h`` (``"all"``).

    ``h > K`` 이면 K일 뒤의 강수는 아무것도 기여하지 않는다 — "3일 예보는 알고, 그 뒤는 모른다"를 흉내 낸다.
    창이 외생 배열 끝을 넘으면(실시간 전망) 넘은 날은 0으로 두고, 그런 원점 수를 두 번째 값으로 돌려준다.
    """

    origins = np.asarray(origins, dtype=int)
    rain = np.nan_to_num(series.exog["rain_mm"], nan=0.0)
    tail = series.exog_tail.get("rain_mm")
    if tail is not None and tail.shape[0]:
        rain = np.concatenate([rain, np.nan_to_num(np.asarray(tail, dtype=np.float64), nan=0.0)])  # 자료 끝 뒤의 예보 강수
    n = rain.shape[0]
    k = int(horizon) if str(future_days) == "all" else max(1, min(int(horizon), int(future_days)))
    cumsum = np.concatenate([[0.0], np.cumsum(rain)])
    lo = origins + 1
    hi = np.minimum(origins + k + 1, n)  # exclusive
    total = np.where(hi > lo, cumsum[hi] - cumsum[np.minimum(lo, n)], 0.0)
    peak = np.zeros(origins.shape[0], dtype=np.float64)
    for step in range(1, k + 1):
        idx = origins + step
        peak = np.maximum(peak, np.where(idx < n, rain[np.minimum(idx, n - 1)], 0.0))
    truncated = int((origins + k >= n).sum())
    return np.stack([np.log1p(np.maximum(total, 0.0)), np.log1p(np.maximum(peak, 0.0))], axis=1), truncated


# --- 인터페이스 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForecasterInfo:
    id: str
    name: str
    description: str
    is_baseline: bool = False  # True면 skill 기준선으로만 쓰고 배포 후보에서 제외
    is_reference: bool = False  # True면 평가표에 항상 참고치로 함께 보고
    hyperparameter_grid: tuple[Mapping[str, Any], ...] = field(default_factory=lambda: ({},))
    min_history_days: int = 30
    # True면 원점 **이후**의 외생 입력(미래 강수 등)을 특징으로 쓴다. 실측을 예보로 간주하는 '완전 예보 가정'
    # 실험용이며, 명시적으로 허용하지 않는 한 배포 후보 풀(ForecasterRegistry.candidates)에서 제외된다.
    uses_future_exog: bool = False


class GaugeForecaster(ABC):
    """지점 하나·예측일수 하나에 대한 직접 예측기.

    다중 지평을 한 번에 학습하는 모델(신경망·Prophet 등)은 ``fit`` 이 지평마다 불려도
    내부 캐시로 한 번만 학습하고, 파라미터에는 ``{"shared_ref": key, ...}`` 만 남긴다.
    실제 가중치는 ``export_shared``/``import_shared`` 로 산출물과 서비스 사이를 오간다.
    """

    @property
    @abstractmethod
    def info(self) -> ForecasterInfo:
        raise NotImplementedError

    def export_shared(self, refs: Iterable[str]) -> dict[str, Any]:
        """``refs`` 에 해당하는 공유 파라미터(JSON 직렬화 가능)를 돌려준다. 기본: 없음."""

        return {}

    def import_shared(self, blobs: Mapping[str, Any]) -> None:
        """산출물의 공유 파라미터를 다시 올린다. 기본: 무시."""

        return None

    @abstractmethod
    def fit(
        self,
        series: DailySeries,
        origins: np.ndarray,
        horizon: int,
        hparams: Mapping[str, Any],
    ) -> Params:
        """``origins`` 의 ``y[t+h]`` 를 정답으로 학습해 JSON 직렬화 가능한 파라미터를 낸다."""

        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        series: DailySeries,
        origins: np.ndarray,
        horizon: int,
        params: Mapping[str, Any],
    ) -> np.ndarray:
        """원점별 예측값. 특징을 만들 수 없는 원점은 NaN."""

        raise NotImplementedError

    # 편의: 학습 가능한 원점(정답 있음 + 특징 가능)만 남긴다.
    def _usable_origins(self, series: DailySeries, origins: np.ndarray, horizon: int) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        targets = origins + horizon
        inside = targets < series.n
        keep = np.zeros(origins.shape[0], dtype=bool)
        keep[inside] = series.valid[targets[inside]]
        return origins[keep]


# --- 기준선 (선택 불가) -----------------------------------------------------------


class PersistenceForecaster(GaugeForecaster):
    """직전값 유지. skill 기준선일 뿐 예측이 아니므로 배포 후보에서 제외된다."""

    info = ForecasterInfo(
        id="persistence",
        name="직전값 유지 (기준선)",
        description="원점의 관측값을 그대로 미래값으로 둔다. skill 계산의 분모.",
        is_baseline=True,
        is_reference=True,
        min_history_days=1,
    )

    def fit(self, series, origins, horizon, hparams) -> Params:
        return {}

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        return series.filled[origins].copy()


class ClimatologyForecaster(GaugeForecaster):
    """연중일 평년값. 학습 구간에서 배우는 가장 단순한 계절 모델.

    현재 상태를 보지 않으므로 단기에는 약하지만, 장기 지평에서 자기회귀가
    평년으로 수렴하는 지점이라면 이것이 이길 수 있다. 참고치로 항상 보고하고
    후보에도 넣는다 — 이기는데 감추면 평가가 아니다.
    """

    info = ForecasterInfo(
        id="climatology",
        name="계절 평년값",
        description="학습 구간의 연중일별 평균 수위(±15일 평활)를 목표일에 맞춰 예측값으로 쓴다.",
        is_reference=True,
        min_history_days=365,
    )

    def fit(self, series, origins, horizon, hparams) -> Params:
        origins = np.asarray(origins, dtype=int)
        mask = np.zeros(series.n, dtype=bool)
        mask[origins] = True
        return {"clim": [float(v) for v in climatology(series, mask)]}

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        clim = np.asarray(params["clim"], dtype=np.float64)
        return clim[target_doy(series, origins, horizon) - 1]


# --- 학습 모델 --------------------------------------------------------------------


class RidgeARForecaster(GaugeForecaster):
    """최근 p일 수위·30/365일 이동평균·계절 조화항의 릿지 자기회귀.

    ``mode="delta"`` 는 변화량 ``y[t+h]-y[t]`` 를 학습해 벌점이 커질수록
    직전값 유지로 수렴하고, ``mode="level"`` 은 수위 자체를 학습해 평균으로
    수렴한다. 어느 쪽이 맞는지는 지점·지평마다 검증 구간이 고른다.
    """

    info = ForecasterInfo(
        id="ridge_ar",
        name="릿지 자기회귀 (AR)",
        description=(
            "최근 p일 수위, 30·365일 이동평균, 연중 조화항(2차)을 특징으로 하는 릿지 회귀. "
            "delta 모드는 변화량을, level 모드는 수위를 직접 학습한다."
        ),
        hyperparameter_grid=tuple(
            {"p": p, "lam": lam, "mode": mode}
            for mode in ("delta", "level")
            for p in (7, 30)
            for lam in (1.0, 10.0, 100.0)
        ),
        min_history_days=365,
    )

    @staticmethod
    def _features(series: DailySeries, origins: np.ndarray, horizon: int, p: int) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        lags = np.stack(
            [np.where(origins - i >= 0, series.filled[np.maximum(origins - i, 0)], np.nan) for i in range(p)],
            axis=1,
        )
        roll30 = rolling_mean(series.filled, 30, origins)[:, None]
        roll365 = rolling_mean(series.filled, 365, origins)[:, None]
        season = seasonal_features(target_doy(series, origins, horizon))
        return np.concatenate([lags, roll30, roll365, season], axis=1)

    def fit(self, series, origins, horizon, hparams) -> Params:
        p = int(hparams.get("p", 7))
        lam = float(hparams.get("lam", 10.0))
        mode = str(hparams.get("mode", "delta"))
        origins = self._usable_origins(series, origins, horizon)
        X = self._features(series, origins, horizon, p)
        ok = ~np.isnan(X).any(axis=1)
        origins, X = origins[ok], X[ok]
        if origins.shape[0] < X.shape[1] + 5:
            raise ValueError("릿지 자기회귀 학습에 필요한 원점이 부족합니다")
        target = series.y[origins + horizon]
        if mode == "delta":
            target = target - series.filled[origins]
        params = ridge_fit(X, target, lam)
        params.update({"p": p, "lam": lam, "mode": mode})
        return params

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        X = self._features(series, origins, horizon, int(params["p"]))
        out = np.full(origins.shape[0], np.nan, dtype=np.float64)
        ok = ~np.isnan(X).any(axis=1)
        if ok.any():
            pred = ridge_predict(X[ok], params)
            if params.get("mode") == "delta":
                pred = pred + series.filled[origins[ok]]
            out[ok] = pred
        return out


class RidgeARXForecaster(RidgeARForecaster):
    """릿지 자기회귀 + **기상 외생 입력**(원점 이전 강수 합·기온).

    ``ridge_ar`` 특징에 강수 1·3·7·30일 합(log)과 7일 평균기온·기온 편차를 더한다.
    비가 온 다음 날 수위가 오르는 반응을 단기 지평에서 잡기 위한 모델이며, 목표일까지의
    미래 강수는 쓰지 않는다(원점에서 알 수 없다). 지역 기상 자료가 없으면 학습을 거른다.
    """

    info = ForecasterInfo(
        id="ridge_arx",
        name="릿지 자기회귀 + 기상 (AR-X)",
        description=(
            "ridge_ar 특징 + 원점 이전 강수 1·3·7·30일 합(log1p)·7일 평균기온·기온 편차. delta/level 모드. "
            "기상청 ASOS 일자료가 지역에 있어야 후보가 된다."
        ),
        hyperparameter_grid=tuple(
            {"p": p, "lam": lam, "mode": mode}
            for mode in ("delta", "level")
            for p in (7, 30)
            for lam in (1.0, 10.0, 100.0)
        ),
        min_history_days=365,
    )

    @staticmethod
    def _features(series: DailySeries, origins: np.ndarray, horizon: int, p: int) -> np.ndarray:
        base = RidgeARForecaster._features(series, origins, horizon, p)
        extra = exog_features(series, np.asarray(origins, dtype=int))
        if extra is None:
            raise ValueError("기상 외생 자료(rain_mm)가 없어 ridge_arx 를 쓸 수 없습니다")
        return np.concatenate([base, extra], axis=1)


class RidgeARXFutureForecaster(RidgeARXForecaster):
    """릿지 AR-X + **원점 이후 강수(완전 예보 가정)**. 실험 전용 — 기본 후보 풀에서 제외된다.

    ``ridge_arx`` 특징(원점 이전 강수·기온)에 ``origin+1 .. origin+min(h, K)`` 의 강수 합(log1p)·일 최대(log1p)를
    더한다. ``K`` 는 하이퍼파라미터 ``future_days`` (3 또는 ``"all"``=h). ``h > K`` 이면 K일 뒤 강수는 기여하지
    않는다("3일 예보는 알고, 그 뒤는 모른다").

    학습·평가에서는 **실측** 강수(``series.exog["rain_mm"]``)의 원점 이후 값을 예보로 간주한다 — 즉 오차 없는
    완전 예보(perfect prognosis)를 가정한 **상한**이다. 실시간 전망에서는 원점 이후 강수가 관측에 없으므로
    이 입력은 기상청 단기·중기 예보에서 와야 하는데, 그 연결은 **아직 배선되지 않았다**. 창이 외생 배열 끝을
    넘으면 넘은 날을 0(비 없음)으로 두고 ``last_note_ko`` 에 사유를 남긴다. 파라미터에 ``future_days`` 를
    기록하므로 서비스 시점에도 같은 규칙으로 ``series.exog`` 에서 미래 강수를 읽는다.
    """

    info = ForecasterInfo(
        id="ridge_arx_future",
        name="릿지 AR + 강수 예보 (완전 예보 가정)",
        description=(
            "ridge_arx 특징 + 원점 이후 강수 합·일 최대(log1p). future_days=3 은 3일치, 'all' 은 목표일까지 전부. "
            "실측 미래 강수를 예보로 간주하는 상한 실험용 — 기본 후보 풀에서 제외되며 실시간 예보 연결은 미배선."
        ),
        hyperparameter_grid=tuple(
            {"p": 7, "lam": lam, "mode": "delta", "future_days": fd}
            for fd in (3, "all")
            for lam in (10.0, 100.0)
        ),
        min_history_days=365,
        uses_future_exog=True,
    )

    def __init__(self) -> None:
        self.last_note_ko: str | None = None

    @staticmethod
    def _features_future(series: DailySeries, origins: np.ndarray, horizon: int, p: int, future_days: int | str) -> tuple[np.ndarray, int]:
        base = RidgeARXForecaster._features(series, origins, horizon, p)
        future, truncated = future_rain_features(series, np.asarray(origins, dtype=int), horizon, future_days)
        return np.concatenate([base, future], axis=1), truncated

    def fit(self, series, origins, horizon, hparams) -> Params:
        p = int(hparams.get("p", 7))
        lam = float(hparams.get("lam", 10.0))
        mode = str(hparams.get("mode", "delta"))
        future_days = hparams.get("future_days", "all")
        future_days = "all" if str(future_days) == "all" else int(future_days)
        origins = self._usable_origins(series, origins, horizon)
        X, _ = self._features_future(series, origins, horizon, p, future_days)
        ok = ~np.isnan(X).any(axis=1)
        origins, X = origins[ok], X[ok]
        if origins.shape[0] < X.shape[1] + 5:
            raise ValueError("릿지 AR-X(미래 강수) 학습에 필요한 원점이 부족합니다")
        target = series.y[origins + horizon]
        if mode == "delta":
            target = target - series.filled[origins]
        params = ridge_fit(X, target, lam)
        params.update({"p": p, "lam": lam, "mode": mode, "future_days": future_days})
        return params

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        future_days = params.get("future_days", "all")
        future_days = "all" if str(future_days) == "all" else int(future_days)
        X, truncated = self._features_future(series, origins, horizon, int(params["p"]), future_days)
        self.last_note_ko = (
            f"원점 이후 강수 창이 관측 범위를 넘는 원점 {truncated}개 — 넘은 날은 강수 0으로 두었다. "
            "실시간 전망에서는 이 입력이 기상청 예보에서 와야 하며 아직 연결되지 않았다."
            if truncated else None
        )
        out = np.full(origins.shape[0], np.nan, dtype=np.float64)
        ok = ~np.isnan(X).any(axis=1)
        if ok.any():
            pred = ridge_predict(X[ok], params)
            if params.get("mode") == "delta":
                pred = pred + series.filled[origins[ok]]
            out[ok] = pred
        return out


class RidgeARXSeasonForecaster(RidgeARXForecaster):
    """릿지 AR-X **계절 분리**(홍수기 6~9월 전용 모델 + 비홍수기 모델) + 댐 운영 입력(있으면). 2026-10-20.

    ``scripts/eval_realtime_forecast_ideas.py`` ③(오차 큰 달 전용 모델)과 ``eval_dam_discharge_d1.py``(댐 관측 입력)를
    서비스 예측기로 옮긴 것. 특징은 ``ridge_arx``(원점 이전 강수·기온) + ``dam_features``(총방류·유입·저수위, 지역에
    ``weather/dam_daily.json`` 이 있을 때). 원점의 달이 홍수기면 홍수기 모델, 아니면 비홍수기 모델을 쓰고, 그룹 표본이
    모자라면 전체 모델(pooled)로 돌아간다. 미래 강수는 쓰지 않으므로 기본 후보 풀에 들어간다.
    """

    info = ForecasterInfo(
        id="ridge_arx_season",
        name="릿지 AR-X 계절 분리 (홍수기 전용 + 댐 운영)",
        description=(
            "ridge_arx 특징 + 댐 총방류·유입·저수위(지역에 댐 일자료가 있을 때)를 홍수기(6~9월)·비홍수기 두 모델로 나눠 적합. "
            "원점의 달로 모델을 고르고 표본이 모자라면 전체 모델. 기상청 ASOS 일자료가 지역에 있어야 후보가 된다."
        ),
        hyperparameter_grid=tuple(
            {"p": p, "lam": lam, "mode": "delta"}
            for p in (7, 30)
            for lam in (10.0, 100.0)
        ),
        min_history_days=365 * 2,
    )

    @staticmethod
    def _season_features(series: DailySeries, origins: np.ndarray, horizon: int, p: int, use_dam: bool) -> np.ndarray:
        base = RidgeARXForecaster._features(series, origins, horizon, p)
        if not use_dam:
            return base
        extra = dam_features(series, np.asarray(origins, dtype=int))
        if extra is None:
            raise ValueError("학습 때 쓴 댐 운영 외생 자료(dam_out)가 없어 ridge_arx_season 전망을 만들 수 없습니다")
        return np.concatenate([base, extra], axis=1)

    def _features_for(self, series: DailySeries, origins: np.ndarray, horizon: int, p: int, use_dam: bool) -> np.ndarray:
        return self._season_features(series, origins, horizon, p, use_dam)

    @staticmethod
    def _is_flood(series: DailySeries, origins: np.ndarray) -> np.ndarray:
        months = (series.dates[np.asarray(origins, dtype=int)].astype("datetime64[M]").astype(int) % 12) + 1
        return np.isin(months, FLOOD_MONTHS)

    def fit(self, series, origins, horizon, hparams) -> Params:
        p = int(hparams.get("p", 7))
        lam = float(hparams.get("lam", 10.0))
        mode = str(hparams.get("mode", "delta"))
        use_dam = series.has_exog("dam_out")
        origins = self._usable_origins(series, origins, horizon)
        X = self._features_for(series, origins, horizon, p, use_dam)
        ok = ~np.isnan(X).any(axis=1)
        origins, X = origins[ok], X[ok]
        if origins.shape[0] < X.shape[1] + 5:
            raise ValueError("릿지 AR-X(계절 분리) 학습에 필요한 원점이 부족합니다")
        target = series.y[origins + horizon]
        if mode == "delta":
            target = target - series.filled[origins]
        flood = self._is_flood(series, origins)
        pooled = ridge_fit(X, target, lam)
        groups: dict[str, Params | None] = {}
        for key, mask in (("flood", flood), ("other", ~flood)):
            groups[key] = ridge_fit(X[mask], target[mask], lam) if int(mask.sum()) >= X.shape[1] + 5 else None
        return {
            "p": p, "lam": lam, "mode": mode, "dam_used": bool(use_dam), "flood_months": list(FLOOD_MONTHS),
            "pooled": pooled, "flood": groups["flood"], "other": groups["other"],
            "n_flood": int(flood.sum()), "n_other": int((~flood).sum()),
        }

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        X = self._features_for(series, origins, horizon, int(params["p"]), bool(params.get("dam_used")))
        out = np.full(origins.shape[0], np.nan, dtype=np.float64)
        ok = ~np.isnan(X).any(axis=1)
        if not ok.any():
            return out
        flood = self._is_flood(series, origins)
        pred = np.full(origins.shape[0], np.nan, dtype=np.float64)
        for key, mask in (("flood", flood & ok), ("other", (~flood) & ok)):
            if not mask.any():
                continue
            group_params = params.get(key) or params["pooled"]
            pred[mask] = ridge_predict(X[mask], group_params)
        if params.get("mode") == "delta":
            pred = pred + series.filled[origins]
        out[ok] = pred[ok]
        return out


class RidgeARXSeasonFutureForecaster(RidgeARXSeasonForecaster):
    """계절 분리 + **원점 이후 3일 강수(예보)**. 2026-10-20.

    실시간에는 지역의 기상청 단기예보 파일(``weather/kma_forecast_daily.json`` → ``DailySeries.exog_tail``)에서, 학습·평가에서는
    실측 강수를 예보로 간주(완전 예보 상한)한다. ``uses_future_exog`` 라 ``forecasters`` 에 명시했을 때만 후보가 된다.
    """

    info = ForecasterInfo(
        id="ridge_arx_season_future",
        name="릿지 AR-X 계절 분리 + 3일 강수 예보",
        description=(
            "ridge_arx_season 특징 + 원점 이후 3일 강수 합·일 최대(log1p). 학습·평가는 실측 미래 강수(완전 예보 상한), "
            "실시간은 지역의 기상청 단기예보 파일. 후보 풀에는 명시적으로 허용했을 때만 들어간다."
        ),
        hyperparameter_grid=tuple({"p": p, "lam": lam, "mode": "delta"} for p in (7, 30) for lam in (10.0, 100.0)),
        min_history_days=365 * 2,
        uses_future_exog=True,
    )

    def __init__(self) -> None:
        self.last_note_ko: str | None = None

    def _features_for(self, series: DailySeries, origins: np.ndarray, horizon: int, p: int, use_dam: bool) -> np.ndarray:
        base = self._season_features(series, origins, horizon, p, use_dam)
        future, truncated = future_rain_features(series, np.asarray(origins, dtype=int), horizon, 3)
        self.last_note_ko = (
            f"원점 {truncated}개의 3일 강수 창이 외생 자료 끝을 넘어 0(비 없음)으로 두었습니다 — 기상청 단기예보를 수집하면 채워집니다"
            if truncated else None
        )
        return np.concatenate([base, future], axis=1)


class SeasonalAnomalyARForecaster(GaugeForecaster):
    """평년값 대비 편차(anomaly)의 릿지 자기회귀.

    수위를 ``평년값(연중일) + 편차`` 로 나누고 편차만 학습한다. 계절 순환이
    뚜렷하고 이벤트 후 평년으로 되돌아가는 지점에서 유리하다.
    """

    info = ForecasterInfo(
        id="seasonal_anomaly_ar",
        name="계절편차 자기회귀",
        description=(
            "학습 구간 연중일 평년값을 빼서 편차를 만들고, 편차의 최근 p일과 30일 이동평균으로 "
            "h일 뒤 편차를 릿지 회귀한다. 예측 = 목표일 평년값 + 예측 편차."
        ),
        hyperparameter_grid=tuple({"p": p, "lam": lam} for p in (7, 30) for lam in (1.0, 10.0, 100.0)),
        min_history_days=730,
    )

    @staticmethod
    def _anomaly_features(anomaly: np.ndarray, origins: np.ndarray, p: int) -> np.ndarray:
        lags = np.stack(
            [np.where(origins - i >= 0, anomaly[np.maximum(origins - i, 0)], np.nan) for i in range(p)],
            axis=1,
        )
        roll30 = rolling_mean(anomaly, 30, origins)[:, None]
        return np.concatenate([lags, roll30], axis=1)

    def fit(self, series, origins, horizon, hparams) -> Params:
        p = int(hparams.get("p", 7))
        lam = float(hparams.get("lam", 10.0))
        origins = self._usable_origins(series, origins, horizon)
        mask = np.zeros(series.n, dtype=bool)
        mask[origins] = True
        clim = climatology(series, mask)
        anomaly = series.filled - clim[series.doy - 1]
        X = self._anomaly_features(anomaly, origins, p)
        ok = ~np.isnan(X).any(axis=1)
        origins, X = origins[ok], X[ok]
        if origins.shape[0] < X.shape[1] + 5:
            raise ValueError("계절편차 자기회귀 학습에 필요한 원점이 부족합니다")
        target_index = origins + horizon
        target = series.y[target_index] - clim[series.doy[target_index] - 1]
        params = ridge_fit(X, target, lam)
        params.update({"p": p, "lam": lam, "clim": [float(v) for v in clim]})
        return params

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        clim = np.asarray(params["clim"], dtype=np.float64)
        anomaly = series.filled - clim[series.doy - 1]
        X = self._anomaly_features(anomaly, origins, int(params["p"]))
        out = np.full(origins.shape[0], np.nan, dtype=np.float64)
        ok = ~np.isnan(X).any(axis=1)
        if ok.any():
            out[ok] = clim[target_doy(series, origins[ok], horizon) - 1] + ridge_predict(X[ok], params)
        return out


class MeanReversionForecaster(GaugeForecaster):
    """기준 수위로의 회귀 계수 하나를 학습하는 AR(1)형 모델.

    기준 수위 ``m`` 은 학습 구간 평균(``window=None``) 또는 원점 직전 ``window``
    일 평균이며, ``y[t+h]-m = phi_h * (y[t]-m)`` 의 ``phi_h`` 를 최소제곱으로
    직접 구한다. 파라미터가 하나라 표본이 적은 지점에서도 안정적이다.
    """

    info = ForecasterInfo(
        id="mean_reversion",
        name="평균 회귀 AR(1)",
        description=(
            "기준 수위(학습 평균 또는 최근 30/90/365일 평균)로 되돌아가는 비율 phi를 "
            "예측일수마다 직접 학습한다."
        ),
        hyperparameter_grid=({"window": None}, {"window": 30}, {"window": 90}, {"window": 365}),
        min_history_days=30,
    )

    @staticmethod
    def _reference(series: DailySeries, origins: np.ndarray, window: int | None, train_mean: float | None) -> np.ndarray:
        if window is None:
            return np.full(origins.shape[0], float(train_mean), dtype=np.float64)
        return rolling_mean(series.filled, int(window), origins)

    def fit(self, series, origins, horizon, hparams) -> Params:
        window = hparams.get("window")
        origins = self._usable_origins(series, origins, horizon)
        train_mean = float(np.nanmean(series.y[origins])) if window is None else None
        m = self._reference(series, origins, window, train_mean)
        ok = ~np.isnan(m)
        origins, m = origins[ok], m[ok]
        if origins.shape[0] < 10:
            raise ValueError("평균 회귀 학습에 필요한 원점이 부족합니다")
        x = series.filled[origins] - m
        target = series.y[origins + horizon] - m
        denominator = float(np.dot(x, x))
        phi = float(np.dot(x, target) / denominator) if denominator > 1e-12 else 0.0
        return {"phi": max(-1.0, min(1.0, phi)), "window": window, "train_mean": train_mean}

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        m = self._reference(series, origins, params.get("window"), params.get("train_mean"))
        return m + float(params["phi"]) * (series.filled[origins] - m)


class AnalogKNNForecaster(GaugeForecaster):
    """유사사례(analog) k-최근접 이웃 예측.

    원점 직전 p일의 수위 **형태**(마지막 값을 뺀 편차)와 가장 닮은 과거 창 k개를
    찾아, 그 뒤 h일 동안 실제로 일어난 변화량의 평균을 현재 수위에 더한다.
    선형 모델이 놓치는 비선형 거동(급상승 뒤 감쇠 등)을 사례로 흉내 낸다.
    파라미터는 사례 탐색 범위(``train_end``)뿐이라 과적합 위험이 낮다.
    """

    info = ForecasterInfo(
        id="analog_knn",
        name="유사사례 k-NN",
        description=(
            "직전 p일 수위 형태와 가장 비슷한 과거 창 k개의 이후 h일 변화량 평균을 현재 수위에 더한다."
        ),
        hyperparameter_grid=tuple({"p": p, "k": k} for p in (7, 14) for k in (20, 50)),
        min_history_days=365,
    )

    @staticmethod
    def _windows(series: DailySeries, origins: np.ndarray, p: int) -> np.ndarray:
        """각 원점의 직전 p일 창을 마지막 값 기준 편차로 만든다. 창이 앞을 벗어나면 NaN."""

        origins = np.asarray(origins, dtype=int)
        cols = []
        for i in range(p):
            idx = origins - i
            cols.append(np.where(idx >= 0, series.filled[np.maximum(idx, 0)], np.nan))
        window = np.stack(cols, axis=1)
        return window - window[:, :1]

    def fit(self, series, origins, horizon, hparams) -> Params:
        p = int(hparams.get("p", 7))
        k = int(hparams.get("k", 20))
        origins = self._usable_origins(series, origins, horizon)
        if origins.shape[0] < k + p:
            raise ValueError("유사사례 탐색에 필요한 과거 창이 부족합니다")
        last_target = series.dates[int(origins.max()) + horizon]
        return {"p": p, "k": k, "library_end": str(last_target)}

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        p, k = int(params["p"]), int(params["k"])
        origins = np.asarray(origins, dtype=int)
        library_end = np.datetime64(str(params["library_end"]), "D")
        # 사례 창: 목표일이 library_end 이하이고 정답이 있는 원점 전부
        lib = np.arange(series.n - horizon)
        lib = lib[(series.dates[lib + horizon] <= library_end) & series.valid[lib + horizon]]
        lib_windows = self._windows(series, lib, p)
        ok_lib = ~np.isnan(lib_windows).any(axis=1)
        lib, lib_windows = lib[ok_lib], lib_windows[ok_lib]
        out = np.full(origins.shape[0], np.nan, dtype=np.float64)
        if lib.shape[0] < k:
            return out
        lib_change = series.y[lib + horizon] - series.filled[lib]
        query = self._windows(series, origins, p)
        ok_q = ~np.isnan(query).any(axis=1)
        if not ok_q.any():
            return out
        q = query[ok_q]
        # 자기 자신(또는 미래 정보를 담은 창)은 사례에서 제외: 사례 원점 + h 가 질의 원점보다 뒤면 안 된다.
        d2 = (q**2).sum(axis=1)[:, None] + (lib_windows**2).sum(axis=1)[None, :] - 2.0 * q @ lib_windows.T
        forbidden = (lib[None, :] + horizon) > origins[ok_q][:, None]
        d2 = np.where(forbidden, np.inf, d2)
        neighbours = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
        changes = lib_change[neighbours]
        weights = np.where(np.isinf(np.take_along_axis(d2, neighbours, axis=1)), 0.0, 1.0)
        counts = weights.sum(axis=1)
        mean_change = np.where(counts > 0, (changes * weights).sum(axis=1) / np.maximum(counts, 1.0), np.nan)
        out[ok_q] = series.filled[origins[ok_q]] + mean_change
        return out


class EnsembleForecaster(GaugeForecaster):
    """서로 다른 계열의 예측기 평균.

    검증 연도가 홍수 등 이벤트에 지배되면 단일 후보 선택이 1~2% 차이로 갈려
    다음 해에 뒤집힌다. 계열이 다른 모델(자기회귀·계절편차·평균회귀·유사사례)의
    평균은 어느 한 해에 과적합되기 어렵다. 구성원은 하이퍼파라미터를 고정한다.
    """

    MEMBERS: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("ridge_ar", {"p": 7, "lam": 10.0, "mode": "level"}),
        ("ridge_ar", {"p": 7, "lam": 10.0, "mode": "delta"}),
        ("ridge_arx", {"p": 7, "lam": 10.0, "mode": "delta"}),
        ("seasonal_anomaly_ar", {"p": 7, "lam": 10.0}),
        ("mean_reversion", {"window": 365}),
        ("analog_knn", {"p": 7, "k": 50}),
    )

    info = ForecasterInfo(
        id="ensemble_mean",
        name="계열 앙상블 평균",
        description=(
            "릿지 자기회귀(level·delta)·계절편차 자기회귀·평균 회귀(365일)·유사사례 k-NN 다섯 구성원의 "
            "예측 평균. 구성원 하이퍼파라미터는 고정."
        ),
        hyperparameter_grid=({"members": "all"}, {"members": "linear"}),
        min_history_days=730,
    )

    def __init__(self) -> None:
        self._members: dict[str, GaugeForecaster] = {
            "ridge_ar": RidgeARForecaster(),
            "ridge_arx": RidgeARXForecaster(),
            "seasonal_anomaly_ar": SeasonalAnomalyARForecaster(),
            "mean_reversion": MeanReversionForecaster(),
            "analog_knn": AnalogKNNForecaster(),
        }

    def _selected(self, hparams: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        if hparams.get("members") == "linear":
            return tuple(m for m in self.MEMBERS if m[0] != "analog_knn")
        return self.MEMBERS

    def fit(self, series, origins, horizon, hparams) -> Params:
        fitted = []
        for member_id, member_hparams in self._selected(hparams):
            try:
                params = self._members[member_id].fit(series, origins, horizon, member_hparams)
            except ValueError:
                continue
            fitted.append({"forecaster_id": member_id, "hparams": dict(member_hparams), "params": params})
        if len(fitted) < 2:
            raise ValueError("앙상블 구성원이 2개 미만입니다")
        return {"members": fitted, "selection": str(hparams.get("members", "all"))}

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        origins = np.asarray(origins, dtype=int)
        stack = []
        for member in params["members"]:
            stack.append(self._members[member["forecaster_id"]].predict(series, origins, horizon, member["params"]))
        matrix = np.stack(stack, axis=0)
        with np.errstate(invalid="ignore"):
            return np.nanmean(matrix, axis=0)


# --- 레지스트리 ------------------------------------------------------------------


class ForecasterRegistry:
    def __init__(self) -> None:
        self._items: dict[str, GaugeForecaster] = {}
        self.load_errors: list[str] = []

    def register(self, forecaster: GaugeForecaster) -> None:
        info = forecaster.info
        if not isinstance(info, ForecasterInfo):
            raise TypeError("forecaster.info 는 ForecasterInfo 여야 합니다")
        if info.id in self._items:
            raise ValueError(f"forecaster id {info.id!r} 가 이미 등록되어 있습니다")
        self._items[info.id] = forecaster

    def get(self, forecaster_id: str) -> GaugeForecaster:
        try:
            return self._items[forecaster_id]
        except KeyError as exc:
            raise KeyError(f"unknown forecaster {forecaster_id!r}; available: {', '.join(sorted(self._items))}") from exc

    def list(self) -> list[GaugeForecaster]:
        return [self._items[key] for key in sorted(self._items)]

    def candidates(self, allowed: Iterable[str] | None = None) -> list[GaugeForecaster]:
        """배포 후보(기준선 제외). ``allowed`` 가 주어지면 그 id만.

        **누수 방지**: 원점 이후 외생 입력을 쓰는 예측기(``uses_future_exog``)는 ``allowed`` 에 명시적으로
        적었을 때만 후보가 된다. 기본 학습(train_region/train_station)이 실측 미래 강수를 본 모델을 배포 모델로
        고르는 일이 없도록 한다.
        """

        allowed_set = {item for item in allowed} if allowed is not None else None
        out = []
        for item in self.list():
            if item.info.is_baseline:
                continue
            if allowed_set is not None and item.info.id not in allowed_set:
                continue
            if item.info.uses_future_exog and (allowed_set is None or item.info.id not in allowed_set):
                continue
            out.append(item)
        return out

    def references(self) -> list[GaugeForecaster]:
        """평가표에 항상 함께 싣는 참고 예측기(직전값 유지·계절 평년값)."""

        return [item for item in self.list() if item.info.is_reference]

    def load_plugins(self, specs: str | Iterable[str]) -> None:
        if isinstance(specs, str):
            plugin_specs = [item.strip() for item in specs.split(",") if item.strip()]
        else:
            plugin_specs = [item.strip() for item in specs if item.strip()]
        for spec in plugin_specs:
            try:
                self.register(_load_plugin(spec))
            except Exception as exc:  # noqa: BLE001 - 외부 플러그인 실패를 격리한다
                self.load_errors.append(f"{spec}: {exc}")

    def __len__(self) -> int:
        return len(self._items)


def _load_plugin(spec: str) -> GaugeForecaster:
    if ":" not in spec:
        raise ValueError("plugin must use 'package.module:ClassOrFactory' syntax")
    module_name, attribute_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute_name)
    if inspect.isclass(candidate) or callable(candidate):
        candidate = candidate()
    if not isinstance(candidate, GaugeForecaster):
        raise TypeError("plugin must produce a GaugeForecaster instance")
    return candidate


def build_forecaster_registry(plugin_specs: str = "") -> ForecasterRegistry:
    registry = ForecasterRegistry()
    registry.register(PersistenceForecaster())
    registry.register(ClimatologyForecaster())
    registry.register(RidgeARForecaster())
    registry.register(RidgeARXForecaster())
    registry.register(RidgeARXFutureForecaster())
    registry.register(RidgeARXSeasonForecaster())
    registry.register(RidgeARXSeasonFutureForecaster())
    registry.register(SeasonalAnomalyARForecaster())
    registry.register(MeanReversionForecaster())
    registry.register(AnalogKNNForecaster())
    registry.register(EnsembleForecaster())
    registry.load_plugins(plugin_specs)
    return registry
