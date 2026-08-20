"""Backend-friendly access to KMA ASOS historical daily weather data."""

from .client import (
    DEFAULT_HTTP_ENDPOINT,
    DEFAULT_HTTPS_ENDPOINT,
    AsosDailyClient,
    AsyncHttpTransport,
    HttpResponse,
    UrllibAsyncTransport,
    WeatherProvider,
)
from .errors import (
    WeatherApiError,
    WeatherConfigurationError,
    WeatherResponseFormatError,
    WeatherServiceError,
    WeatherTransportError,
    WeatherValidationError,
)
from .sample import SampleWeatherProvider
from .schemas import (
    DailyWeatherObservation,
    DateLike,
    Station,
    WeatherMetadata,
    WeatherSeries,
    parse_date,
)
from .stations import ASOS_STATIONS, list_stations, resolve_station

__all__ = [
    "ASOS_STATIONS",
    "DEFAULT_HTTP_ENDPOINT",
    "DEFAULT_HTTPS_ENDPOINT",
    "AsosDailyClient",
    "AsyncHttpTransport",
    "DailyWeatherObservation",
    "DateLike",
    "HttpResponse",
    "SampleWeatherProvider",
    "Station",
    "UrllibAsyncTransport",
    "WeatherApiError",
    "WeatherConfigurationError",
    "WeatherMetadata",
    "WeatherProvider",
    "WeatherResponseFormatError",
    "WeatherSeries",
    "WeatherServiceError",
    "WeatherTransportError",
    "WeatherValidationError",
    "list_stations",
    "parse_date",
    "resolve_station",
]

