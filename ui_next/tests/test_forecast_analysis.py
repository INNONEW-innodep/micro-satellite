from ui_next.forecast_analysis import (
    cumulative_rmse,
    forecast_window_options,
    horizon_row,
    precipitation_by_date,
)


def test_forecast_window_options_keep_standard_and_full_ranges() -> None:
    assert forecast_window_options(3) == (3,)
    assert forecast_window_options(14) == (7, 14)
    assert forecast_window_options(20) == (7, 14, 20)
    assert forecast_window_options(30) == (7, 14, 30)


def test_horizon_and_chart_helpers() -> None:
    rows = [{"frame": 7, "water_level_m": 2.1}, {"horizon": 14}]
    assert horizon_row(rows, 7)["water_level_m"] == 2.1
    assert horizon_row(rows, 30) is None
    assert cumulative_rmse([{"residual_m": 1}, {"residual_m": -1}]) == [1.0, 1.0]
    assert precipitation_by_date(
        [{"timestamp": "2026-08-01T00:00:00", "rainfall_mm": 3.5}]
    ) == {"2026-08-01": 3.5}
