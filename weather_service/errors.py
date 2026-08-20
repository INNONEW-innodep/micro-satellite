"""Typed errors exposed by :mod:`weather_service`.

The exception messages intentionally never include the request URL because the
public-data service key is a query-string parameter.
"""

from __future__ import annotations


class WeatherServiceError(Exception):
    """Base class for all weather-service failures."""


class WeatherConfigurationError(WeatherServiceError):
    """The provider is not configured, most commonly because its key is absent."""


class WeatherValidationError(WeatherServiceError, ValueError):
    """A station or date range is invalid for the selected provider."""


class WeatherTransportError(WeatherServiceError):
    """The upstream service could not be reached or returned an HTTP failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class WeatherApiError(WeatherServiceError):
    """The upstream API returned a documented, non-success result code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.api_message = str(message)
        super().__init__(f"KMA ASOS API error {self.code}: {self.api_message}")


class WeatherResponseFormatError(WeatherServiceError):
    """The upstream payload did not match the documented response shape."""

