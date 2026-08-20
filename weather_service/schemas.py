"""Dependency-free schemas shared by live and sample weather providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping

from .errors import WeatherResponseFormatError, WeatherValidationError


DateLike = date | datetime | str


def parse_date(value: DateLike, *, field_name: str = "date") -> date:
    """Parse a date accepted by the backend-facing API.

    Both ``YYYYMMDD`` (the KMA request format) and ISO ``YYYY-MM-DD`` are
    accepted. A datetime is reduced to its calendar date.
    """

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise WeatherValidationError(f"{field_name} must be a date or date string")

    cleaned = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass
    raise WeatherValidationError(
        f"{field_name} must use YYYYMMDD or YYYY-MM-DD format: {value!r}"
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True, slots=True)
class Station:
    """A KMA ASOS station used by the daily-observation endpoint."""

    id: str
    name: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        station_id = str(self.id).strip()
        if not station_id.isdigit() or not 1 <= int(station_id) <= 999:
            raise WeatherValidationError(f"invalid ASOS station id: {self.id!r}")
        object.__setattr__(self, "id", station_id)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "aliases": list(self.aliases)}


@dataclass(frozen=True, slots=True)
class DailyWeatherObservation:
    """Model-friendly subset of one KMA ASOS daily record.

    Missing upstream values remain ``None`` rather than being silently changed
    to zero. The full API item is retained in ``raw`` for future model adapters.
    """

    observed_on: date
    station_id: str
    station_name: str | None = None
    precipitation_mm: float | None = None
    avg_temperature_c: float | None = None
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    avg_humidity_percent: float | None = None
    min_humidity_percent: float | None = None
    avg_local_pressure_hpa: float | None = None
    avg_sea_level_pressure_hpa: float | None = None
    avg_wind_speed_m_s: float | None = None
    max_wind_speed_m_s: float | None = None
    max_instant_wind_speed_m_s: float | None = None
    sunshine_hours: float | None = None
    solar_radiation_mj_m2: float | None = None
    new_snow_cm: float | None = None
    weather_description: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_api_item(
        cls,
        item: Mapping[str, Any],
        *,
        station_name: str | None = None,
    ) -> "DailyWeatherObservation":
        """Parse the documented JSON item returned by data.go.kr."""

        if not isinstance(item, Mapping):
            raise WeatherResponseFormatError("ASOS item must be a JSON object")
        if not item.get("tm"):
            raise WeatherResponseFormatError("ASOS item is missing required field 'tm'")
        if not item.get("stnId"):
            raise WeatherResponseFormatError("ASOS item is missing required field 'stnId'")

        try:
            observed_on = parse_date(str(item["tm"]), field_name="tm")
        except WeatherValidationError as exc:
            raise WeatherResponseFormatError(f"invalid ASOS field 'tm': {item['tm']!r}") from exc

        return cls(
            observed_on=observed_on,
            station_id=str(item["stnId"]).strip(),
            station_name=_optional_text(item.get("stnNm")) or station_name,
            precipitation_mm=_optional_float(item.get("sumRn")),
            avg_temperature_c=_optional_float(item.get("avgTa")),
            min_temperature_c=_optional_float(item.get("minTa")),
            max_temperature_c=_optional_float(item.get("maxTa")),
            avg_humidity_percent=_optional_float(item.get("avgRhm")),
            min_humidity_percent=_optional_float(item.get("minRhm")),
            avg_local_pressure_hpa=_optional_float(item.get("avgPa")),
            avg_sea_level_pressure_hpa=_optional_float(item.get("avgPs")),
            avg_wind_speed_m_s=_optional_float(item.get("avgWs")),
            max_wind_speed_m_s=_optional_float(item.get("maxWs")),
            max_instant_wind_speed_m_s=_optional_float(item.get("maxInsWs")),
            sunshine_hours=_optional_float(item.get("sumSsHr")),
            solar_radiation_mj_m2=_optional_float(item.get("sumGsr")),
            new_snow_cm=_optional_float(item.get("sumDpthFhsc")),
            weather_description=_optional_text(item.get("iscs")),
            raw=dict(item),
        )

    def model_features(self) -> dict[str, float | None]:
        """Return stable feature names suitable for a replaceable model adapter."""

        return {
            "precipitation_mm": self.precipitation_mm,
            "avg_temperature_c": self.avg_temperature_c,
            "min_temperature_c": self.min_temperature_c,
            "max_temperature_c": self.max_temperature_c,
            "avg_humidity_percent": self.avg_humidity_percent,
            "avg_local_pressure_hpa": self.avg_local_pressure_hpa,
            "avg_wind_speed_m_s": self.avg_wind_speed_m_s,
        }

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        result = {
            "date": self.observed_on.isoformat(),
            "station_id": self.station_id,
            "station_name": self.station_name,
            **self.model_features(),
            "min_humidity_percent": self.min_humidity_percent,
            "avg_sea_level_pressure_hpa": self.avg_sea_level_pressure_hpa,
            "max_wind_speed_m_s": self.max_wind_speed_m_s,
            "max_instant_wind_speed_m_s": self.max_instant_wind_speed_m_s,
            "sunshine_hours": self.sunshine_hours,
            "solar_radiation_mj_m2": self.solar_radiation_mj_m2,
            "new_snow_cm": self.new_snow_cm,
            "weather_description": self.weather_description,
        }
        if include_raw:
            result["raw"] = dict(self.raw)
        return result


@dataclass(frozen=True, slots=True)
class WeatherMetadata:
    """Provenance that prevents observations or samples being labelled forecasts."""

    provider: str
    mode: str
    dataset: str
    data_category: str
    is_forecast: bool
    is_sample: bool
    timezone: str
    availability: str
    disclaimer_ko: str
    documentation_url: str
    source_url: str | None
    endpoint_url: str | None
    fetched_at: datetime

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fetched_at"] = self.fetched_at.astimezone(timezone.utc).isoformat()
        return result


@dataclass(frozen=True, slots=True)
class WeatherSeries:
    """A complete provider response with request and provenance metadata."""

    station: Station
    requested_start: date
    requested_end: date
    records: tuple[DailyWeatherObservation, ...]
    missing_dates: tuple[date, ...]
    metadata: WeatherMetadata

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        return {
            "station": self.station.to_dict(),
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "record_count": len(self.records),
            "missing_dates": [value.isoformat() for value in self.missing_dates],
            "metadata": self.metadata.to_dict(),
            "records": [record.to_dict(include_raw=include_raw) for record in self.records],
        }

