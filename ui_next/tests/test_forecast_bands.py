"""관측 변동성 기반 σ 밴드의 계산을 고정한다.

시연을 다이나믹하게 만들려고 넣은 기능이지만, 숫자가 근거 없이 커지면 그게 더
나쁘다. 그래서 (1) 관측이 없으면 밴드를 만들지 않고, (2) 불규칙한 관측 간격을
하루 단위로 환산하며, (3) 면적이 음수로 내려가지 않는다는 세 가지를 못박는다.
"""

from __future__ import annotations

import math

import pytest

from ui_next.forecast_bands import (
    SIGMA_LEVELS,
    band_multiplier,
    band_surface,
    build_bands,
    measure_variability,
    summarize_spread,
)

# 부산 NAS 수체 라벨 4시점의 실제 픽셀 수와 경과일.
BUSAN_AREAS = [9891.0, 9550.0, 10394.0, 10192.0]
BUSAN_DAYS = [0.0, 23.0, 36.0, 56.0]


def _steps(count: int, base: float = 10200.0) -> list[dict]:
    return [
        {"horizon": h, "target_date": f"2020-04-{h:02d}", "water_area_pixels": base}
        for h in range(1, count + 1)
    ]


def test_measures_real_observed_variability() -> None:
    profile = measure_variability(BUSAN_AREAS, BUSAN_DAYS)
    assert profile.usable
    assert profile.observation_count == 4
    # 간격당 σ 는 약 6.7%, 평균 간격은 약 18.7일이다.
    assert 0.06 < profile.sigma_per_interval < 0.08
    assert 18.0 < profile.mean_interval_days < 19.5
    # 하루당 σ 는 간격당 σ 를 √간격으로 나눈 값이어야 한다.
    assert profile.sigma_per_day == pytest.approx(
        profile.sigma_per_interval / math.sqrt(profile.mean_interval_days), rel=1e-9
    )


@pytest.mark.parametrize(
    "areas", [[], [100.0], [100.0, 110.0], [0.0, 0.0, 0.0]]
)
def test_too_few_observations_produce_no_band(areas: list[float]) -> None:
    """관측이 모자라면 변동성을 지어내지 않는다."""

    profile = measure_variability(areas)
    assert not profile.usable
    assert build_bands(_steps(5), profile) == () or all(
        row["+3σ"] == row["예측값"] for row in build_bands(_steps(5), profile)
    )


def test_band_widens_with_the_square_root_of_time() -> None:
    profile = measure_variability(BUSAN_AREAS, BUSAN_DAYS)
    one = band_multiplier(profile, 1, 1) - 1.0
    four = band_multiplier(profile, 4, 1) - 1.0
    assert four == pytest.approx(one * 2.0, rel=1e-9), "4배 시간이면 폭은 2배여야 한다"


def test_band_never_goes_negative() -> None:
    """-3σ 가 예측값을 넘어서도 면적이 음수가 되면 안 된다."""

    profile = measure_variability([100.0, 10.0, 100.0, 10.0], [0.0, 1.0, 2.0, 3.0])
    rows = build_bands(_steps(60), profile)
    assert rows
    assert all(row["-3σ"] >= 0.0 for row in rows)


def test_every_sigma_level_is_present_and_ordered() -> None:
    profile = measure_variability(BUSAN_AREAS, BUSAN_DAYS)
    rows = build_bands(_steps(10), profile)
    assert rows
    for row in rows:
        values = [row[f"{level:+d}σ" if level else "0σ"] for level in SIGMA_LEVELS]
        assert values == sorted(values), "σ 단계가 커질수록 값도 커져야 한다"
        assert row["0σ"] == pytest.approx(row["예측값"])


def test_surface_grid_matches_dates_and_levels() -> None:
    profile = measure_variability(BUSAN_AREAS, BUSAN_DAYS)
    rows = build_bands(_steps(12), profile)
    dates, levels, grid = band_surface(rows)
    assert len(dates) == len(rows)
    assert list(levels) == list(SIGMA_LEVELS)
    assert len(grid) == len(SIGMA_LEVELS)
    assert all(len(line) == len(rows) for line in grid)


def test_spread_summary_reports_a_visible_range() -> None:
    """실제 관측으로 만든 밴드는 시연에서 눈에 보일 만큼 벌어져야 한다."""

    profile = measure_variability(BUSAN_AREAS, BUSAN_DAYS)
    rows = build_bands(_steps(30), profile)
    summary = summarize_spread(rows, profile)
    assert summary["usable"]
    assert summary["spread_pct"] > 20.0, "3σ 폭이 20% 미만이면 3D에서 밋밋하다"
    assert summary["low_3sigma"] < summary["center"] < summary["high_3sigma"]


def test_damping_default_matches_measured_optimum() -> None:
    """walk-forward 실측에서 추세 외삽이 항상 손해였다 — 기본 감쇠는 0이어야 한다."""

    from ui_next.forecast_bands import DEFAULT_TREND_DAMPING, project_trend

    assert DEFAULT_TREND_DAMPING == 0.0
    values, days = BUSAN_AREAS, BUSAN_DAYS
    flat = project_trend(values, days, [7.0, 30.0])
    assert flat == pytest.approx((values[-1], values[-1])), (
        "감쇠 0이면 마지막 관측을 그대로 유지해야 한다"
    )


def test_walk_forward_error_reports_skill_against_naive() -> None:
    """MSE 만으로 판단하지 않도록 naive 대비 skill 을 반드시 함께 낸다."""

    from ui_next.forecast_bands import walk_forward_error

    result = walk_forward_error(BUSAN_AREAS, BUSAN_DAYS)
    assert result["usable"]
    assert {"mse", "rmse", "mae", "mape_pct", "naive_mse", "skill_vs_naive_pct"} <= set(result)
    # 기본 감쇠(0)에서는 naive 와 동일한 예측이므로 skill 이 0 이어야 한다.
    assert result["skill_vs_naive_pct"] == pytest.approx(0.0, abs=1e-9)


def test_trend_extrapolation_is_worse_than_naive_on_this_data() -> None:
    """추세를 켜면 오히려 나빠진다는 사실을 고정해 둔다.

    이 전제가 깨지면(자료가 늘어 추세가 유효해지면) 기본 감쇠를 다시 정해야 한다.
    """

    from ui_next.forecast_bands import walk_forward_error

    damped = walk_forward_error(BUSAN_AREAS, BUSAN_DAYS, damping=1.0)
    assert damped["skill_vs_naive_pct"] < 0.0
