"""Async client for KMA ASOS daily historical observations.

The public API is asynchronous. The default transport uses ``urllib`` inside
``asyncio.to_thread`` so the package does not impose an HTTP-library dependency
on the backend. A transport can be injected for tests or an application's own
connection-pooling implementation.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping, Protocol

from ._common import (
    KMA_TIMEZONE,
    OBSERVATION_AVAILABILITY,
    OBSERVATION_DISCLAIMER_KO,
    OFFICIAL_DOCUMENTATION_URL,
    build_series,
    today_in_korea,
    utc_now,
    validate_observation_period,
)
from .errors import (
    WeatherApiError,
    WeatherConfigurationError,
    WeatherResponseFormatError,
    WeatherTransportError,
    WeatherValidationError,
)
from .schemas import DailyWeatherObservation, DateLike, Station, WeatherMetadata, WeatherSeries
from .stations import resolve_station


DEFAULT_HTTPS_ENDPOINT = (
    "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
)
DEFAULT_HTTP_ENDPOINT = (
    "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
)


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str]


class AsyncHttpTransport(Protocol):
    """Minimal injectable transport contract used by :class:`AsosDailyClient`."""

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse: ...


class UrllibAsyncTransport:
    """Standard-library transport that keeps blocking I/O off the event loop."""

    user_agent = "micro-satellite-weather-service/1.0"

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        return await asyncio.to_thread(self._get_sync, url, params, timeout)

    def _get_sync(
        self,
        url: str,
        params: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{url}?{query}",
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HttpResponse(
                    status_code=response.status,
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            # Preserve the response body for a useful upstream error while never
            # propagating the URL (which contains ServiceKey).
            return HttpResponse(
                status_code=exc.code,
                body=exc.read(),
                headers=dict(exc.headers.items()) if exc.headers else {},
            )


class WeatherProvider(Protocol):
    """Common interface implemented by the live and sample providers."""

    async def get_daily_observations(
        self,
        start_date: DateLike,
        end_date: DateLike,
        station: Station | str | int,
    ) -> WeatherSeries: ...


class AsosDailyClient:
    """Client for data.go.kr's KMA ASOS daily-observation service.

    ``api_key`` takes precedence over ``env_var``. Public-data keys copied in
    URL-encoded form are decoded once and safely encoded by the transport.
    HTTP fallback is opt-in and only occurs after an HTTPS transport/HTTP
    failure; KMA API validation and authentication errors never trigger it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_var: str = "KMA_API_KEY",
        https_endpoint: str = DEFAULT_HTTPS_ENDPOINT,
        http_endpoint: str = DEFAULT_HTTP_ENDPOINT,
        allow_http_fallback: bool = False,
        timeout: float = 15.0,
        page_size: int = 999,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.2,
        transport: AsyncHttpTransport | None = None,
        today_provider: Callable[[], date] = today_in_korea,
        now_provider=utc_now,
    ) -> None:
        configured_key = api_key if api_key is not None else os.getenv(env_var)
        if not configured_key or not configured_key.strip():
            raise WeatherConfigurationError(
                f"KMA ASOS API key is required; pass api_key or set {env_var}"
            )
        if timeout <= 0:
            raise WeatherValidationError("timeout must be positive")
        if not 1 <= page_size <= 9999:
            raise WeatherValidationError("page_size must be between 1 and 9999")
        if max_retries < 0:
            raise WeatherValidationError("max_retries cannot be negative")
        if retry_backoff_seconds < 0:
            raise WeatherValidationError("retry_backoff_seconds cannot be negative")
        if allow_http_fallback and not http_endpoint.startswith("http://"):
            raise WeatherValidationError("http_endpoint must use http:// when fallback is enabled")

        # data.go.kr exposes encoded and decoded key variants. Decode exactly
        # once here; urlencode in the transport then produces one valid encoding.
        self._api_key = urllib.parse.unquote(configured_key.strip())
        self.https_endpoint = https_endpoint
        self.http_endpoint = http_endpoint
        self.allow_http_fallback = allow_http_fallback
        self.timeout = timeout
        self.page_size = page_size
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.transport = transport or UrllibAsyncTransport()
        self._today_provider = today_provider
        self._now_provider = now_provider

    async def get_daily_observations(
        self,
        start_date: DateLike,
        end_date: DateLike,
        station: Station | str | int,
    ) -> WeatherSeries:
        """Fetch and parse all pages for one station and inclusive date range."""

        start, end = validate_observation_period(
            start_date,
            end_date,
            today_provider=self._today_provider,
        )
        resolved_station = resolve_station(station)
        records: list[DailyWeatherObservation] = []
        page = 1
        total_count: int | None = None
        endpoint_used: str | None = None

        while True:
            params = {
                "ServiceKey": self._api_key,
                "pageNo": str(page),
                "numOfRows": str(self.page_size),
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "DAY",
                "startDt": start.strftime("%Y%m%d"),
                "endDt": end.strftime("%Y%m%d"),
                "stnIds": resolved_station.id,
            }
            payload, endpoint_used = await self._request_json(params)
            page_records, page_total = self._parse_page(payload, resolved_station)
            records.extend(page_records)
            total_count = page_total if total_count is None else total_count

            if not page_records:
                break
            if total_count is not None and len(records) >= total_count:
                break
            if len(page_records) < self.page_size:
                break
            page += 1
            if page > 10000:
                raise WeatherResponseFormatError("ASOS pagination exceeded safety limit")

        metadata = WeatherMetadata(
            provider="Korea Meteorological Administration (KMA)",
            mode="live",
            dataset="ASOS daily observations",
            data_category="historical_observation",
            is_forecast=False,
            is_sample=False,
            timezone=KMA_TIMEZONE,
            availability=OBSERVATION_AVAILABILITY,
            disclaimer_ko=OBSERVATION_DISCLAIMER_KO,
            documentation_url=OFFICIAL_DOCUMENTATION_URL,
            source_url=OFFICIAL_DOCUMENTATION_URL,
            endpoint_url=endpoint_used or self.https_endpoint,
            fetched_at=self._now_provider(),
        )
        return build_series(
            station_value=resolved_station,
            start=start,
            end=end,
            records=records,
            metadata=metadata,
        )

    async def _request_json(
        self,
        params: Mapping[str, str],
    ) -> tuple[Mapping[str, Any], str]:
        endpoints = [self.https_endpoint]
        if self.allow_http_fallback:
            endpoints.append(self.http_endpoint)

        last_error: WeatherTransportError | None = None
        for endpoint_index, endpoint in enumerate(endpoints):
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.transport.get(
                        endpoint,
                        params=params,
                        timeout=self.timeout,
                    )
                except (OSError, TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                    last_error = WeatherTransportError(
                        "KMA ASOS service could not be reached"
                    )
                else:
                    if 200 <= response.status_code < 300:
                        return self._decode_json_or_error(response.body), endpoint
                    last_error = WeatherTransportError(
                        f"KMA ASOS service returned HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                    if response.status_code < 500 and response.status_code != 429:
                        raise last_error

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))

            # Reaching the next endpoint is the explicit HTTPS -> HTTP fallback.
            if endpoint_index + 1 < len(endpoints):
                continue

        raise last_error or WeatherTransportError("KMA ASOS request failed")

    @staticmethod
    def _decode_json_or_error(body: bytes) -> Mapping[str, Any]:
        try:
            decoded = body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise WeatherResponseFormatError("KMA ASOS response is not UTF-8") from exc
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            xml_error = AsosDailyClient._parse_xml_error(decoded)
            if xml_error is not None:
                raise xml_error
            raise WeatherResponseFormatError("KMA ASOS response is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise WeatherResponseFormatError("KMA ASOS JSON root must be an object")
        return payload

    @staticmethod
    def _parse_xml_error(text: str) -> WeatherApiError | None:
        try:
            root = ET.fromstring(text.strip())
        except (ET.ParseError, ValueError):
            return None

        def first_text(*tags: str) -> str | None:
            for tag in tags:
                element = root.find(f".//{tag}")
                if element is not None and element.text:
                    return element.text.strip()
            return None

        code = first_text("resultCode", "returnReasonCode")
        message = first_text("resultMsg", "errMsg", "returnAuthMsg")
        if code or message:
            return WeatherApiError(code or "UNKNOWN", message or "Unknown upstream error")
        return None

    @staticmethod
    def _parse_page(
        payload: Mapping[str, Any],
        station: Station,
    ) -> tuple[list[DailyWeatherObservation], int]:
        response = payload.get("response")
        if not isinstance(response, Mapping):
            raise WeatherResponseFormatError("KMA ASOS response is missing object 'response'")
        header = response.get("header")
        if not isinstance(header, Mapping):
            raise WeatherResponseFormatError("KMA ASOS response is missing object 'header'")

        code = str(header.get("resultCode", "")).strip()
        message = str(header.get("resultMsg", "")).strip()
        if code in {"03", "NODATA_ERROR"} or message == "NODATA_ERROR":
            return [], 0
        if code != "00":
            raise WeatherApiError(code or "UNKNOWN", message or "Unknown upstream error")

        body = response.get("body")
        if not isinstance(body, Mapping):
            raise WeatherResponseFormatError("KMA ASOS response is missing object 'body'")
        try:
            total_count = int(body.get("totalCount", 0))
        except (TypeError, ValueError) as exc:
            raise WeatherResponseFormatError("KMA ASOS 'totalCount' must be an integer") from exc

        items_container = body.get("items")
        if items_container in (None, ""):
            raw_items: list[Any] = []
        elif isinstance(items_container, Mapping):
            item_value = items_container.get("item", [])
            if item_value in (None, ""):
                raw_items = []
            elif isinstance(item_value, list):
                raw_items = item_value
            elif isinstance(item_value, Mapping):
                raw_items = [item_value]
            else:
                raise WeatherResponseFormatError("KMA ASOS 'items.item' has an invalid type")
        else:
            raise WeatherResponseFormatError("KMA ASOS 'items' has an invalid type")

        parsed: list[DailyWeatherObservation] = []
        for index, item in enumerate(raw_items):
            try:
                record = DailyWeatherObservation.from_api_item(
                    item,
                    station_name=station.name,
                )
            except WeatherResponseFormatError as exc:
                raise WeatherResponseFormatError(f"invalid ASOS item at index {index}: {exc}") from exc
            if record.station_id != station.id:
                raise WeatherResponseFormatError(
                    f"ASOS item station {record.station_id!r} does not match requested station {station.id!r}"
                )
            parsed.append(record)
        return parsed, total_count

