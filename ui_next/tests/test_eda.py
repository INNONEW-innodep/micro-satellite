from __future__ import annotations

import datetime as dt
import math

import pytest

from ui_next.eda import (
    build_input_water_figure,
    build_result_summary_figure,
    build_weather_figure,
    render_eda_overview,
    render_input_eda,
    render_result_eda,
    render_weather_eda,
    summarize_eda,
    summarize_input_eda,
    summarize_result_eda,
    summarize_weather_eda,
)
from ui_next.samples import get_sample


def test_sample_input_summary_has_dates_intervals_resolution_and_pixels() -> None:
    sample = get_sample("busan-doc-recovered")
    summary = summarize_input_eda(sample)

    assert summary.frame_count == 4
    assert summary.decoded_frame_count == 4
    assert summary.unique_date_count == 4
    assert summary.duplicate_date_count == 0
    assert summary.period_days == 56
    assert summary.interval_min_days == 13
    assert summary.interval_median_days == 20.0
    assert summary.interval_max_days == 23
    assert summary.resolutions == ((533, 500),)
    assert summary.resolution_label == "533 × 500 px"
    assert [point.water_pixels for point in summary.points] == [9446, 8802, 9734, 9521]
    assert summary.pixel_area_known is False
    assert summary.pixel_area_m2 is None
    assert summary.water_area_first_km2 is None
    assert all(point.water_area_km2 is None for point in summary.points)
    assert "km²로 환산하지 않고" in " ".join(summary.notes)
    assert "입력 변화 요약" in summary.interpretation


def test_known_pixel_area_is_explicit_and_duplicate_dates_are_robust() -> None:
    rows = [
        {
            "name": "one.png",
            "date": "2026-08-01",
            "water_pixels": 100,
            "width": 10,
            "height": 10,
        },
        {
            "name": "two.png",
            "date": "2026-08-01",
            "water_pixels": 125,
            "width": 10,
            "height": 10,
        },
        {
            "name": "three.png",
            "date": "2026-08-04",
            "water_pixels": 150,
            "width": 10,
            "height": 10,
        },
    ]
    summary = summarize_input_eda(
        input_rows=rows, pixel_area_m2=4.0, pixel_area_known=True
    )

    assert summary.frame_count == 3
    assert summary.unique_date_count == 2
    assert summary.duplicate_date_count == 1
    assert summary.period_days == 3
    assert summary.interval_min_days == 3
    assert summary.interval_median_days == 3
    assert summary.interval_max_days == 3
    assert summary.water_pixel_change_pct == 50.0
    assert summary.water_area_first_km2 == pytest.approx(0.0004)
    assert summary.water_area_last_km2 == pytest.approx(0.0006)
    assert summary.pixel_area_known is True
    assert any("중복 날짜" in note for note in summary.notes)


def test_sample_placeholder_pixel_size_never_becomes_physical_area() -> None:
    sample = get_sample("gwangju_flood")
    assert sample.pixel_area_m2 == 9.0
    assert sample.pixel_area_known is False

    summary = summarize_input_eda(sample)
    assert summary.pixel_area_known is False
    assert summary.pixel_area_m2 is None
    assert summary.water_area_last_km2 is None
    figure = build_input_water_figure(summary)
    assert figure is not None
    assert figure.layout.yaxis.title.text == "수체 픽셀 수"
    assert "km²" not in figure.layout.title.text


def test_weather_summary_separates_kinds_and_calculates_missingness() -> None:
    rows = [
        {
            "date": "2026-08-01",
            "kind": "observed",
            "precipitation_mm": 10,
            "temperature_c": 20,
            "humidity_pct": 60,
        },
        {
            "date": "2026-08-01",
            "kind": "observed",
            "precipitation_mm": None,
            "min_temperature_c": 18,
            "max_temperature_c": 22,
            "humidity_pct": None,
        },
        {
            "date": "2026-08-03",
            "kind": "scenario",
            "precipitation_mm": 30,
            "temperature_c": 25,
            "humidity_pct": 80,
        },
        {
            "date": "not-a-date",
            "precipitation_mm": -1,
            "temperature_c": None,
            "humidity_pct": None,
        },
    ]
    summary = summarize_weather_eda(rows)

    assert summary.row_count == 4
    assert summary.dated_row_count == 3
    assert summary.unique_date_count == 2
    assert summary.duplicate_date_count == 1
    assert summary.invalid_date_count == 1
    assert summary.observed_count == 2
    assert summary.scenario_count == 1
    assert summary.unknown_kind_count == 1
    assert summary.period_days == 2
    assert summary.precipitation_count == 2
    assert summary.precipitation_sum_mm == 40
    assert summary.precipitation_mean_mm == 20
    assert summary.precipitation_max_mm == 30
    assert summary.precipitation_max_date == dt.date(2026, 8, 3)
    assert summary.precipitation_missing_rate_pct == 50
    assert summary.feature_missing_rate_pct == pytest.approx(100 * 5 / 12)
    assert summary.temperature_count == 3
    assert summary.humidity_count == 2
    assert "가정값" in summary.interpretation

    figure = build_weather_figure(summary)
    assert figure is not None
    assert {trace.name for trace in figure.data} == {
        "관측 강수",
        "시나리오 강수",
        "구분 미상 강수",
        "평균 기온",
        "평균 습도",
    }
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"


RESULT_STEPS = [
    {
        "horizon": 1,
        "target_date": "2026-08-05",
        "water_area_pixels": 100,
        "water_area_km2": 0.10,
        "area_change_pct": 0,
        "water_level_m": 1.0,
        "risk": "normal",
    },
    {
        "horizon": 2,
        "target_date": "2026-08-06",
        "water_area_pixels": 120,
        "water_area_km2": 0.12,
        "area_change_pct": 20,
        "water_level_m": 1.2,
        "risk": "caution",
    },
    {
        "horizon": 3,
        "target_date": None,
        "water_area_pixels": 150,
        "water_area_km2": 0.15,
        "area_change_pct": 50,
        "water_level_m": None,
        "risk": "flood_risk",
    },
]


def test_result_summary_uses_pixels_by_default_and_reports_risk() -> None:
    summary = summarize_result_eda(RESULT_STEPS)

    assert summary.step_count == 3
    assert summary.unique_horizon_count == 3
    assert (summary.horizon_min, summary.horizon_max) == (1, 3)
    assert summary.period_days == 1
    assert summary.area_unit == "px"
    assert summary.area_value_first == 100
    assert summary.area_value_last == 150
    assert summary.area_net_change == 50
    assert summary.area_net_change_pct == 50
    assert summary.latest_area_change_from_input_pct == 50
    assert summary.water_level_count == 2
    assert summary.water_level_net_change_m == pytest.approx(0.2)
    assert dict(summary.risk_counts) == {
        "normal": 1,
        "caution": 1,
        "flood_risk": 1,
    }
    assert "홍수 위험" in summary.interpretation
    assert any("목표 날짜" in note for note in summary.notes)

    figure = build_result_summary_figure(summary)
    assert figure is not None
    assert figure.layout.yaxis.title.text == "수체 픽셀 수"


def test_result_physical_area_requires_positive_known_flag() -> None:
    summary = summarize_result_eda(RESULT_STEPS, sample={"pixel_area_known": True})
    assert summary.pixel_area_known is True
    assert summary.area_unit == "km²"
    assert summary.area_value_first == 0.10
    assert summary.area_value_last == 0.15
    assert summary.area_net_change == pytest.approx(0.05)
    assert summary.area_net_change_pct == pytest.approx(50)
    figure = build_result_summary_figure(summary)
    assert figure is not None
    assert figure.layout.yaxis.title.text == "수체 면적 (km²)"


def test_empty_and_invalid_inputs_return_explanations_not_exceptions() -> None:
    input_summary = summarize_input_eda(input_rows=[])
    weather_summary = summarize_weather_eda([])
    result_summary = summarize_result_eda([])

    assert input_summary.frame_count == 0
    assert input_summary.period_days is None
    assert "없어" in input_summary.interpretation
    assert weather_summary.row_count == 0
    assert weather_summary.precipitation_missing_rate_pct == 0
    assert weather_summary.feature_missing_rate_pct == 0
    assert result_summary.step_count == 0
    assert result_summary.risk_counts == ()
    assert build_input_water_figure(input_summary) is None
    assert build_weather_figure(weather_summary) is None
    assert build_result_summary_figure(result_summary) is None


def test_combined_summary_uses_sample_weather_when_rows_are_omitted() -> None:
    sample = get_sample("han_drought")
    overview = summarize_eda(sample=sample, result_steps=RESULT_STEPS)

    assert overview.input.frame_count == 4
    assert overview.weather.row_count == 7
    assert overview.weather.observed_count == 4
    assert overview.weather.scenario_count == 3
    assert overview.result.step_count == 3
    assert overview.result.pixel_area_known is False


class _FakeColumn:
    def __init__(self) -> None:
        self.metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def metric(self, *args: object, **kwargs: object) -> None:
        self.metrics.append((args, kwargs))


class _FakeStreamlit:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.figures: list[object] = []
        self.metric_columns: list[_FakeColumn] = []

    def markdown(self, value: str) -> None:
        self.messages.append(value)

    def columns(self, count: int) -> list[_FakeColumn]:
        columns = [_FakeColumn() for _ in range(count)]
        self.metric_columns.extend(columns)
        return columns

    def caption(self, value: str) -> None:
        self.messages.append(value)

    def info(self, value: str) -> None:
        self.messages.append(value)

    def plotly_chart(self, figure: object, **_: object) -> None:
        self.figures.append(figure)


def test_streamlit_renderers_return_their_pure_summaries() -> None:
    sample = get_sample("nakdong_stable")
    fake = _FakeStreamlit()

    input_summary = render_input_eda(sample, st_module=fake)
    weather_summary = render_weather_eda(sample.weather_rows, st_module=fake)
    result_summary = render_result_eda(
        RESULT_STEPS, sample=sample, st_module=fake, show_chart=True
    )

    assert input_summary.frame_count == 4
    assert weather_summary.row_count == 7
    assert result_summary.step_count == 3
    assert len(fake.metric_columns) == 15
    assert len(fake.figures) == 3
    assert any("입력 데이터 요약" in message for message in fake.messages)


def test_overview_renderer_uses_catalog_weather_by_default() -> None:
    fake = _FakeStreamlit()
    overview = render_eda_overview(
        sample=get_sample("han_drought"),
        result_steps=RESULT_STEPS,
        st_module=fake,
    )

    assert overview.input.frame_count == 4
    assert overview.weather.row_count == 7
    assert overview.result.step_count == 3


def test_no_non_finite_number_leaks_into_weather_statistics() -> None:
    summary = summarize_weather_eda(
        [
            {
                "date": "2026-08-01",
                "kind": "observed",
                "precipitation_mm": math.nan,
                "temperature_c": math.inf,
                "humidity_pct": "bad",
            }
        ]
    )
    assert summary.precipitation_count == 0
    assert summary.temperature_count == 0
    assert summary.humidity_count == 0
    assert summary.precipitation_missing_rate_pct == 100
    assert summary.feature_missing_rate_pct == 100
