"""Deterministic synthetic weather provider for UI and integration demos."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Callable

from ._common import (
    KMA_TIMEZONE,
    OBSERVATION_AVAILABILITY,
    OFFICIAL_DOCUMENTATION_URL,
    SAMPLE_DISCLAIMER_KO,
    build_series,
    today_in_korea,
    utc_now,
    validate_observation_period,
)
from .schemas import DailyWeatherObservation, DateLike, Station, WeatherMetadata, WeatherSeries
from .stations import resolve_station


class SampleWeatherProvider:
    """Generate reproducible sample-shaped data without calling KMA.

    This provider deliberately enforces the same D-1 window as the live client
    and marks every response as synthetic, never as an observation or forecast.
    """

    def __init__(
        self,
        *,
        seed: int = 20260809,
        today_provider: Callable[[], date] = today_in_korea,
        now_provider=utc_now,
    ) -> None:
        self.seed = int(seed)
        self._today_provider = today_provider
        self._now_provider = now_provider

    async def get_daily_observations(
        self,
        start_date: DateLike,
        end_date: DateLike,
        station: Station | str | int,
    ) -> WeatherSeries:
        start, end = validate_observation_period(
            start_date,
            end_date,
            today_provider=self._today_provider,
        )
        resolved_station = resolve_station(station)
        records = tuple(
            self._make_record(start + timedelta(days=offset), resolved_station)
            for offset in range((end - start).days + 1)
        )
        metadata = WeatherMetadata(
            provider="Built-in deterministic sample provider",
            mode="sample",
            dataset="synthetic ASOS-shaped daily data",
            data_category="synthetic_sample",
            is_forecast=False,
            is_sample=True,
            timezone=KMA_TIMEZONE,
            availability=OBSERVATION_AVAILABILITY,
            disclaimer_ko=SAMPLE_DISCLAIMER_KO,
            documentation_url=OFFICIAL_DOCUMENTATION_URL,
            source_url=None,
            endpoint_url=None,
            fetched_at=self._now_provider(),
        )
        return build_series(
            station_value=resolved_station,
            start=start,
            end=end,
            records=records,
            metadata=metadata,
        )

    def _make_record(self, observed_on: date, station: Station) -> DailyWeatherObservation:
        randomizer = random.Random(self.seed + observed_on.toordinal() * 1009 + int(station.id))
        seasonal = 13.0 + 12.0 * math.sin(2.0 * math.pi * (observed_on.timetuple().tm_yday - 105) / 365.25)
        avg_temp = seasonal + randomizer.uniform(-2.2, 2.2)
        spread = randomizer.uniform(5.0, 9.0)
        rainy = randomizer.random() < 0.32
        precipitation = round(randomizer.gammavariate(1.5, 7.0), 1) if rainy else 0.0
        humidity = min(96.0, max(30.0, 56.0 + precipitation * 1.1 + randomizer.uniform(-10, 10)))
        pressure = 1013.0 + randomizer.uniform(-7.0, 7.0)
        wind = max(0.2, randomizer.gauss(2.6, 1.0))
        sunshine = max(0.0, randomizer.uniform(4.0, 10.0) - precipitation * 0.25)

        raw = {
            "sample": True,
            "generator": "weather_service.SampleWeatherProvider",
            "tm": observed_on.isoformat(),
            "stnId": station.id,
        }
        return DailyWeatherObservation(
            observed_on=observed_on,
            station_id=station.id,
            station_name=station.name,
            precipitation_mm=precipitation,
            avg_temperature_c=round(avg_temp, 1),
            min_temperature_c=round(avg_temp - spread / 2, 1),
            max_temperature_c=round(avg_temp + spread / 2, 1),
            avg_humidity_percent=round(humidity, 1),
            min_humidity_percent=round(max(20.0, humidity - randomizer.uniform(8, 22)), 1),
            avg_local_pressure_hpa=round(pressure, 1),
            avg_sea_level_pressure_hpa=round(pressure + randomizer.uniform(0.0, 4.0), 1),
            avg_wind_speed_m_s=round(wind, 1),
            max_wind_speed_m_s=round(wind * randomizer.uniform(1.4, 2.2), 1),
            max_instant_wind_speed_m_s=round(wind * randomizer.uniform(1.8, 2.8), 1),
            sunshine_hours=round(sunshine, 1),
            solar_radiation_mj_m2=round(sunshine * randomizer.uniform(1.1, 1.8), 2),
            new_snow_cm=0.0,
            weather_description="합성 강수 샘플" if rainy else "합성 맑음 샘플",
            raw=raw,
        )

