from __future__ import annotations

from ui_next.sample_weather import (
    KMA_BUSAN_STATION_ID,
    build_kma_historical_replay,
    resolve_sample_weather,
)
from ui_next.samples import get_sample


def test_nas_sample_requests_busan_asos_and_marks_target_as_historical_replay() -> None:
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return {
            "station_id": "159",
            "station_name": "부산",
            "source": "asos",
            "rows": [
                {"date": "2020-02-18", "precipitation_mm": None, "source": "asos"},
                {"date": "2020-03-12", "precipitation_mm": 1.0, "source": "asos"},
                {"date": "2020-03-25", "precipitation_mm": 2.0, "source": "asos"},
                {"date": "2020-04-14", "precipitation_mm": None, "source": "asos"},
                {"date": "2020-04-15", "precipitation_mm": None, "source": "asos"},
                {"date": "2020-04-16", "precipitation_mm": None, "source": "asos"},
                {"date": "2020-04-17", "precipitation_mm": 32.6, "source": "asos"},
            ],
        }

    plan = resolve_sample_weather(
        get_sample("busan-nas-water-labels"),
        3,
        observation_loader=loader,
    )

    assert calls == [
        {
            "source": "asos",
            "station_id": KMA_BUSAN_STATION_ID,
            "start_date": "2020-02-18",
            "end_date": "2020-04-17",
        }
    ]
    assert len(plan.rows) == 7
    assert tuple(row["kind"] for row in plan.rows[:4]) == ("observed",) * 4
    assert tuple(row["kind"] for row in plan.rows[4:]) == ("scenario",) * 3
    assert all(row["is_measured"] is True for row in plan.rows)
    assert all(row["is_forecast"] is False for row in plan.rows)
    assert all(row["historical_replay"] is True for row in plan.rows)
    assert all(row["source"] == "kma_asos_historical_replay" for row in plan.rows)
    assert plan.rows[-1]["precipitation_mm"] == 32.6
    assert plan.historical_replay is True
    assert plan.missing_dates == ()
    assert "미래예보" in plan.note_ko


def test_replay_reports_missing_dates_without_fabricating_rows() -> None:
    sample = get_sample("busan-nas-water-labels")
    plan = build_kma_historical_replay(
        {"rows": [{"date": "2020-04-15", "precipitation_mm": 0.0}]},
        source_dates=sample.source_dates,
        target_dates=sample.forecast_dates(2),
    )

    assert len(plan.rows) == 1
    assert plan.rows[0]["date"] == "2020-04-15"
    assert plan.missing_dates == (
        "2020-02-18",
        "2020-03-12",
        "2020-03-25",
        "2020-04-14",
        "2020-04-16",
    )


def test_synthetic_samples_keep_their_existing_deterministic_weather() -> None:
    sample = get_sample("gwangju_flood")
    plan = resolve_sample_weather(sample, 7)

    assert plan.rows == sample.forecast_weather_payload(7)
    assert plan.source == "built_in_demo"
    assert plan.historical_replay is False


def test_nas_sample_without_loader_is_explicitly_empty() -> None:
    plan = resolve_sample_weather(get_sample("busan-nas-water-labels"), 7)

    assert plan.rows == ()
    assert plan.source == "none"
    assert plan.historical_replay is True
    assert len(plan.missing_dates) == 11


def test_iceye_materialized_sample_uses_the_same_busan_asos_replay_contract() -> None:
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return {"station_id": "159", "station_name": "부산", "rows": []}

    sample = get_sample("iceye-nas-water-labels")
    plan = resolve_sample_weather(sample, 7, observation_loader=loader)

    assert calls == [
        {
            "source": "asos",
            "station_id": KMA_BUSAN_STATION_ID,
            "start_date": "2020-03-02",
            "end_date": "2020-04-23",
        }
    ]
    assert plan.historical_replay is True
    assert len(plan.missing_dates) == 11
