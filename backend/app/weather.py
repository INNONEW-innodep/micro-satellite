from __future__ import annotations

import inspect
import os
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException

from .api_docs import API_DOCS
from .schemas import (
    WeatherObservationRequest,
    WeatherObservationResponse,
    WeatherRow,
    WeatherStatusResponse,
)

router = APIRouter(prefix="/weather", tags=["weather"])
KMA_DOCUMENTATION_URL = "https://www.data.go.kr/data/15059093/openapi.do"


@router.get(
    "/status",
    response_model=WeatherStatusResponse,
    summary="Check live ASOS and sample weather capabilities",
    description=API_DOCS["weather_status"],
)
async def weather_status() -> WeatherStatusResponse:
    service = _weather_service()
    server_key_configured = bool(os.getenv("KMA_API_KEY", "").strip())
    return WeatherStatusResponse(
        asos_available=service is not None and server_key_configured,
        server_key_configured=server_key_configured,
        sample_available=service is not None,
        documentation_url=KMA_DOCUMENTATION_URL,
    )


@router.get(
    "/stations",
    summary="List commonly used KMA ASOS stations",
    description=API_DOCS["weather_stations"],
)
async def list_weather_stations() -> dict[str, Any]:
    service = _weather_service()
    if service is None:
        raise HTTPException(
            status_code=503, detail="weather_service package is not available"
        )
    if hasattr(service, "list_stations"):
        stations = service.list_stations()
        if inspect.isawaitable(stations):
            stations = await stations
        return {"items": _to_jsonable(stations)}

    # The weather package deliberately exposes station resolution as its stable
    # contract. These common IDs keep the UI useful without coupling to internals.
    common = (
        ("108", "Seoul"),
        ("159", "Busan"),
        ("133", "Daejeon"),
        ("143", "Daegu"),
        ("156", "Gwangju"),
        ("184", "Jeju"),
    )
    items: list[dict[str, Any]] = []
    for station_id, fallback_name in common:
        try:
            station = service.resolve_station(station_id)
            items.append(_to_jsonable(station))
        except Exception:  # noqa: BLE001 - tolerate older weather_service station catalogs
            items.append({"id": station_id, "name": fallback_name})
    return {"items": items}


@router.post(
    "/observations",
    response_model=WeatherObservationResponse,
    summary="Fetch KMA ASOS daily observations",
    description=API_DOCS["weather_observations"],
)
async def weather_observations(
    request: WeatherObservationRequest,
) -> WeatherObservationResponse:
    service = _weather_service()
    if service is None:
        raise HTTPException(
            status_code=503, detail="weather_service package is not available"
        )
    try:
        station = service.resolve_station(request.station_id)
        if request.source == "asos":
            provider = service.AsosDailyClient(api_key=request.service_key)
        else:
            provider = service.SampleWeatherProvider()
        series = await provider.get_daily_observations(
            start_date=request.start_date,
            end_date=request.end_date,
            station=station,
        )
        rows = [
            _record_to_weather_row(record, request.source) for record in series.records
        ]
    except Exception as exc:
        configuration_error = getattr(service, "WeatherConfigurationError", ())
        if request.source == "asos" and isinstance(exc, configuration_error):
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "KMA_API_KEY_REQUIRED",
                    "message_ko": (
                        "기상청 ASOS 실관측 조회에는 공공데이터포털 인증키가 필요합니다. "
                        "서버의 KMA_API_KEY를 설정하거나 이번 요청에 service_key를 전달하세요."
                    ),
                    "documentation_url": KMA_DOCUMENTATION_URL,
                    "setup_env_var": "KMA_API_KEY",
                    "sample_fallback_available": True,
                },
            ) from exc
        status_code = 502 if request.source == "asos" else 422
        raise HTTPException(
            status_code=status_code, detail=f"weather request failed: {exc}"
        ) from exc
    note = "ASOS observations are historical data through D-1, not a future forecast."
    return WeatherObservationResponse(
        station_id=str(
            getattr(station, "station_id", getattr(station, "id", request.station_id))
        ),
        station_name=getattr(station, "name", None),
        source=request.source,
        start_date=request.start_date,
        end_date=request.end_date,
        rows=rows,
        missing_dates=list(getattr(series, "missing_dates", [])),
        metadata=_to_jsonable(getattr(series, "metadata", {})),
        note=note,
    )


def _weather_service() -> Any | None:
    try:
        import weather_service
    except ImportError:
        return None
    return weather_service


def _record_to_weather_row(record: Any, source: str) -> WeatherRow:
    if hasattr(record, "model_features"):
        values = dict(record.model_features())
    elif hasattr(record, "model_dump"):
        values = dict(record.model_dump(mode="json"))
    elif isinstance(record, dict):
        values = dict(record)
    else:
        values = dict(vars(record))

    for attribute in ("date", "observation_date", "observed_on", "tm"):
        if values.get("date") is None and getattr(record, attribute, None) is not None:
            values["date"] = getattr(record, attribute)
    for attribute in ("station_id", "station_name"):
        if values.get(attribute) is None and getattr(record, attribute, None) is not None:
            values[attribute] = getattr(record, attribute)
    aliases = {
        "rainfall_mm": "precipitation_mm",
        "daily_precipitation_mm": "precipitation_mm",
        "avg_temperature_c": "temperature_c",
        "mean_temperature_c": "temperature_c",
        "avg_humidity_pct": "humidity_pct",
        "mean_humidity_pct": "humidity_pct",
        "avg_humidity_percent": "humidity_pct",
        "avg_wind_speed_mps": "wind_speed_mps",
        "mean_wind_speed_mps": "wind_speed_mps",
        "avg_wind_speed_m_s": "wind_speed_mps",
    }
    for source_key, target in aliases.items():
        if values.get(target) is None and values.get(source_key) is not None:
            values[target] = values[source_key]
    values.setdefault("source", source)
    return WeatherRow.model_validate(values)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)
