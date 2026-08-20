"""Internal helpers shared by weather providers."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

from .errors import WeatherValidationError
from .schemas import DailyWeatherObservation, DateLike, WeatherMetadata, WeatherSeries, parse_date
from .stations import resolve_station


OFFICIAL_DOCUMENTATION_URL = "https://www.data.go.kr/data/15059093/openapi.do"
KMA_TIMEZONE = "Asia/Seoul"
OBSERVATION_AVAILABILITY = "Daily ASOS observations are provided through D-1 (KST)."
OBSERVATION_DISCLAIMER_KO = (
    "이 데이터는 기상청 ASOS 과거 일 관측자료이며 미래 예보 데이터가 아닙니다."
)
SAMPLE_DISCLAIMER_KO = (
    "이 데이터는 UI 및 연동 시험용 합성 샘플이며 기상청 실관측 또는 미래 예보 데이터가 아닙니다."
)


def today_in_korea() -> date:
    return datetime.now(ZoneInfo(KMA_TIMEZONE)).date()


def validate_observation_period(
    start_date: DateLike,
    end_date: DateLike,
    *,
    today_provider: Callable[[], date] = today_in_korea,
) -> tuple[date, date]:
    start = parse_date(start_date, field_name="start_date")
    end = parse_date(end_date, field_name="end_date")
    if start > end:
        raise WeatherValidationError("start_date must be on or before end_date")

    latest = today_provider() - timedelta(days=1)
    if end > latest:
        raise WeatherValidationError(
            "ASOS daily data is historical observation data, not a forecast; "
            f"end_date must be D-1 or earlier ({latest.isoformat()} KST)"
        )
    return start, end


def missing_dates(
    start: date,
    end: date,
    records: Iterable[DailyWeatherObservation],
) -> tuple[date, ...]:
    observed = {record.observed_on for record in records}
    days = (end - start).days + 1
    return tuple(start + timedelta(days=offset) for offset in range(days) if start + timedelta(days=offset) not in observed)


def build_series(
    *,
    station_value: object,
    start: date,
    end: date,
    records: Iterable[DailyWeatherObservation],
    metadata: WeatherMetadata,
) -> WeatherSeries:
    station = resolve_station(station_value)  # type: ignore[arg-type]
    by_date = {record.observed_on: record for record in records}
    ordered = tuple(by_date[value] for value in sorted(by_date))
    return WeatherSeries(
        station=station,
        requested_start=start,
        requested_end=end,
        records=ordered,
        missing_dates=missing_dates(start, end, ordered),
        metadata=metadata,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

