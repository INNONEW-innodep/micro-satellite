"""관측 변동성에서 예측 산포 밴드를 만든다.

기준선 모델의 예측선 하나만 그리면 화면이 밋밋하고, 더 중요하게는 **이 예측이
얼마나 불확실한지**가 전혀 드러나지 않는다. 여기서는 입력 관측들이 실제로 얼마나
출렁였는지(σ)를 재서 예측 궤적 둘레에 ±1σ·±2σ·±3σ 범위를 씌운다.

용어를 분명히 한다. 이 밴드는 **모델이 산출한 신뢰구간이 아니다.** 과거 관측의
변동 폭을 미래로 확장한 산포 범위이며, 모델의 예측 오차를 측정한 값이 아니다.
표본이 네 시점뿐이라 σ 자체의 불확실성도 크다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

SIGMA_LEVELS: Final[tuple[int, ...]] = (-3, -2, -1, 0, 1, 2, 3)
MIN_OBSERVATIONS: Final[int] = 3

BAND_DISCLAIMER_KO: Final[str] = (
    "이 범위는 **입력 관측이 실제로 변동한 폭(σ)** 을 미래로 확장한 산포이며, "
    "모델이 산출한 신뢰구간이 아닙니다. 관측 표본이 적어 σ 자체의 불확실성도 큽니다."
)


@dataclass(frozen=True, slots=True)
class VariabilityProfile:
    """입력 관측에서 잰 변동성."""

    sigma_per_interval: float  # 관측 간격 1회당 상대 변화율의 표준편차
    sigma_per_day: float  # 하루당으로 환산한 값
    mean_interval_days: float
    observation_count: int
    relative_changes: tuple[float, ...]

    @property
    def usable(self) -> bool:
        return self.observation_count >= MIN_OBSERVATIONS and self.sigma_per_day > 0.0


def measure_variability(
    areas: Sequence[float], dates_days: Sequence[float] | None = None
) -> VariabilityProfile:
    """관측 면적 시계열에서 상대 변화율의 표준편차를 잰다.

    관측 간격이 불규칙하므로 간격당 σ를 그대로 쓰면 안 된다. 하루당으로 환산해
    두어야 나중에 임의의 horizon에 확장할 수 있다. 확산은 무작위 보행을 가정해
    시간의 제곱근에 비례한다고 본다.
    """

    values = [float(area) for area in areas if area is not None and float(area) > 0.0]
    if len(values) < MIN_OBSERVATIONS:
        return VariabilityProfile(0.0, 0.0, 0.0, len(values), ())

    changes = tuple(
        (values[i + 1] - values[i]) / values[i] for i in range(len(values) - 1)
    )
    if len(changes) < 2:
        return VariabilityProfile(0.0, 0.0, 0.0, len(values), changes)

    mean = sum(changes) / len(changes)
    variance = sum((item - mean) ** 2 for item in changes) / (len(changes) - 1)
    sigma_interval = math.sqrt(variance)

    if dates_days is not None and len(dates_days) == len(values):
        spans = [
            float(dates_days[i + 1]) - float(dates_days[i]) for i in range(len(values) - 1)
        ]
        positive = [span for span in spans if span > 0.0]
        mean_interval = sum(positive) / len(positive) if positive else 1.0
    else:
        mean_interval = 1.0

    sigma_day = sigma_interval / math.sqrt(mean_interval) if mean_interval > 0 else 0.0
    return VariabilityProfile(
        sigma_per_interval=sigma_interval,
        sigma_per_day=sigma_day,
        mean_interval_days=mean_interval,
        observation_count=len(values),
        relative_changes=changes,
    )


def band_multiplier(profile: VariabilityProfile, days_ahead: float, sigma_level: int) -> float:
    """h일 뒤 sigma_level배 지점의 배율. 1.0이면 예측값 그대로."""

    if not profile.usable or days_ahead <= 0:
        return 1.0
    spread = profile.sigma_per_day * math.sqrt(days_ahead) * sigma_level
    # 면적은 음수가 될 수 없다. -3σ가 예측값을 넘어서면 0에서 자른다.
    return max(0.0, 1.0 + spread)


def build_bands(
    steps: Sequence[Mapping[str, Any]],
    profile: VariabilityProfile,
    *,
    value_key: str = "water_area_pixels",
    levels: Sequence[int] = SIGMA_LEVELS,
) -> tuple[Mapping[str, Any], ...]:
    """예측 단계마다 σ 단계별 값을 붙인 행을 만든다.

    3D 표현이 목적이므로 (날짜 × σ단계 × 값)의 격자를 그대로 낼 수 있게 한 행에
    모든 σ 단계를 담는다.
    """

    rows: list[Mapping[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        base = step.get(value_key)
        if not isinstance(base, (int, float)) or isinstance(base, bool):
            continue
        days_ahead = float(step.get("horizon", index) or index)
        row: dict[str, Any] = {
            "horizon": int(days_ahead),
            "target_date": step.get("target_date"),
            "예측값": float(base),
        }
        for level in levels:
            row[f"{level:+d}σ" if level else "0σ"] = float(base) * band_multiplier(
                profile, days_ahead, level
            )
        rows.append(row)
    return tuple(rows)


def band_surface(
    band_rows: Sequence[Mapping[str, Any]], levels: Sequence[int] = SIGMA_LEVELS
) -> tuple[list[str], list[int], list[list[float]]]:
    """3D surface용 (x=날짜, y=σ단계, z=값 격자)를 만든다."""

    dates = [str(row.get("target_date") or row.get("horizon")) for row in band_rows]
    grid: list[list[float]] = []
    for level in levels:
        key = f"{level:+d}σ" if level else "0σ"
        grid.append([float(row.get(key, row["예측값"])) for row in band_rows])
    return dates, list(levels), grid


def summarize_spread(
    band_rows: Sequence[Mapping[str, Any]], profile: VariabilityProfile
) -> Mapping[str, Any]:
    """마지막 시점에서 밴드가 얼마나 벌어졌는지 요약한다."""

    if not band_rows or not profile.usable:
        return {"usable": False}
    last = band_rows[-1]
    base = float(last["예측값"])
    low = float(last.get("-3σ", base))
    high = float(last.get("+3σ", base))
    return {
        "usable": True,
        "horizon": last["horizon"],
        "target_date": last.get("target_date"),
        "center": base,
        "low_3sigma": low,
        "high_3sigma": high,
        "spread_pct": (high - low) / base * 100.0 if base else 0.0,
        "sigma_per_day_pct": profile.sigma_per_day * 100.0,
        "sigma_per_interval_pct": profile.sigma_per_interval * 100.0,
        "mean_interval_days": profile.mean_interval_days,
        "observation_count": profile.observation_count,
    }


# walk-forward 실측 결과(scripts/eval_forecast_error.py): 이 관측들에서는 추세를
# 조금이라도 외삽할수록 MSE가 단조 증가했다. 면적·수위 다섯 계열 모두 φ=0, 즉
# "마지막 관측 유지"가 최적이었다. 변동이 추세가 아니라 평균 회귀 잡음이기
# 때문이다. 그래서 기본 감쇠를 0으로 두고, 근거가 생기면 올리도록 인자로 뺐다.
DEFAULT_TREND_DAMPING: Final[float] = 0.0


def project_trend(
    values: Sequence[float],
    days: Sequence[float],
    horizon_days: Sequence[float],
    *,
    damping: float = DEFAULT_TREND_DAMPING,
) -> tuple[float, ...]:
    """관측 시계열에 로그선형 추세를 적합해 미래 값을 만든다.

    수위는 백엔드 기준선이 산출하지 않는다(``produces_water_level=False``).
    그렇다고 면적에서 환산해 버리면 검증되지 않은 면적-수위 관계를 만들어 내는
    셈이라, 수위는 **수위 관측 자체의 추세**로 따로 전망한다. 면적과 수위는
    별개의 관측량이므로 각자의 σ 도 따로 잰다.

    양수 값에만 쓴다. 로그 공간에서 직선을 맞춰 상대 변화율이 일정하다고 본다.
    """

    pairs = [
        (float(day), float(value))
        for day, value in zip(days, values)
        if value is not None and float(value) > 0.0
    ]
    if len(pairs) < 2:
        return tuple(float(pairs[0][1]) for _ in horizon_days) if pairs else ()

    xs = [item[0] for item in pairs]
    ys = [math.log(item[1]) for item in pairs]
    n = len(pairs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return tuple(float(pairs[-1][1]) for _ in horizon_days)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    # 감쇠 추세: 마지막 관측에서 출발해 기울기를 damping 배만 반영한다.
    # damping=0 이면 마지막 관측을 그대로 유지하고, 1 이면 완전한 로그선형 외삽이다.
    last_day, last_value = xs[-1], pairs[-1][1]
    factor = max(0.0, float(damping))
    return tuple(
        last_value * math.exp(factor * slope * float(ahead)) for ahead in horizon_days
    )


def forecast_with_bands(
    values: Sequence[float],
    days: Sequence[float],
    horizon_days: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    damping: float = DEFAULT_TREND_DAMPING,
) -> tuple[tuple[Mapping[str, Any], ...], VariabilityProfile]:
    """한 관측 시계열을 추세 전망 + σ 밴드로 만든다.

    면적이든 수위든 같은 절차를 쓰되, σ 는 **그 시계열 자신의 변동**에서 잰다.
    """

    profile = measure_variability(values, days)
    projected = project_trend(values, days, horizon_days, damping=damping)
    if not projected:
        return (), profile
    steps = [
        {
            "horizon": float(ahead),
            "target_date": (labels[i] if labels and i < len(labels) else f"D+{ahead:g}"),
            "value": projected[i],
        }
        for i, ahead in enumerate(horizon_days)
    ]
    rows = build_bands(steps, profile, value_key="value")
    return rows, profile


def walk_forward_error(
    values: Sequence[float],
    days: Sequence[float],
    *,
    damping: float = DEFAULT_TREND_DAMPING,
    min_train: int = 2,
) -> Mapping[str, Any]:
    """이 시계열에서 전망이 실제로 얼마나 틀렸는지 1-step walk-forward로 잰다.

    MSE 절대값만으로는 판단할 수 없다. 변동이 작은 시계열은 아무 예측기나 MSE가
    낮게 나오므로, 같은 분할에서 '직전값 유지'도 함께 재고 그 대비 개선율을
    돌려준다. skill 이 0 이하이면 낮은 MSE 가 모델의 공로가 아니라는 뜻이다.
    """

    truth: list[float] = []
    predicted: list[float] = []
    naive: list[float] = []
    for k in range(min_train, len(values)):
        ahead = float(days[k]) - float(days[k - 1])
        projection = project_trend(values[:k], days[:k], [ahead], damping=damping)
        if not projection:
            continue
        truth.append(float(values[k]))
        predicted.append(projection[0])
        naive.append(float(values[k - 1]))
    if not truth:
        return {"usable": False}

    def _mse(pred: Sequence[float]) -> float:
        return sum((p - t) ** 2 for t, p in zip(truth, pred)) / len(truth)

    mse = _mse(predicted)
    naive_mse = _mse(naive)
    mae = sum(abs(p - t) for t, p in zip(truth, predicted)) / len(truth)
    mape = (
        sum(abs(p - t) / t for t, p in zip(truth, predicted) if t) / len(truth) * 100.0
    )
    return {
        "usable": True,
        "n_test": len(truth),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "mape_pct": mape,
        "naive_mse": naive_mse,
        "skill_vs_naive_pct": (1.0 - mse / naive_mse) * 100.0 if naive_mse > 0 else 0.0,
    }
