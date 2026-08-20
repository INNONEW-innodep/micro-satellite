"""Small, replaceable HTTP client for the FastAPI water prediction service."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import quote, urljoin

import requests


class Endpoints:
    HEALTH = "/api/v1/health"
    MODELS = "/api/v1/models"
    WEATHER_STATIONS = "/api/v1/weather/stations"
    WEATHER_STATUS = "/api/v1/weather/status"
    WEATHER_OBSERVATIONS = "/api/v1/weather/observations"
    PREDICTIONS = "/api/v1/predictions"
    PREDICTION = "/api/v1/predictions/{prediction_id}"
    FILES = "/api/v1/predictions/{prediction_id}/files"
    FILE = "/api/v1/predictions/{prediction_id}/files/{name}"
    BUNDLE = "/api/v1/predictions/{prediction_id}/bundle"
    OPENAPI = "/openapi.json"
    DOCS = "/docs"
    REDOC = "/redoc"


@dataclass(frozen=True)
class UploadPart:
    name: str
    content: bytes | BinaryIO
    content_type: str = "application/octet-stream"


class APIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class APIClient:
    """Synchronous requests client with all server paths kept in ``Endpoints``."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def absolute_url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self.session.request(method, self.absolute_url(path), **kwargs)
        except requests.RequestException as exc:
            raise APIError(f"백엔드에 연결할 수 없습니다: {exc}") from exc

        if not 200 <= response.status_code < 300:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text[:1000]
            message = "API 요청이 실패했습니다"
            if isinstance(detail, Mapping):
                nested = detail.get("detail")
                if isinstance(nested, Mapping):
                    message = str(
                        nested.get("message_ko")
                        or nested.get("message")
                        or nested.get("code")
                        or message
                    )
                else:
                    message = str(nested or detail.get("message") or message)
            elif detail:
                message = str(detail)
            raise APIError(message, status_code=response.status_code, detail=detail)

        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            return response.json()
        return response.content

    def health(self) -> Mapping[str, Any]:
        result = self._request("GET", Endpoints.HEALTH)
        return result if isinstance(result, Mapping) else {"status": "ok", "response": result}

    def list_models(self) -> list[dict[str, Any]]:
        return normalize_collection(self._request("GET", Endpoints.MODELS), ("models", "items", "data"))

    def list_predictions(self, *, offset: int = 0, limit: int = 10) -> Mapping[str, Any]:
        result = self._request(
            "GET",
            Endpoints.PREDICTIONS,
            params={"offset": int(offset), "limit": int(limit)},
        )
        if not isinstance(result, Mapping):
            raise APIError("예측 목록 API가 JSON 객체가 아닌 응답을 반환했습니다.")
        return result

    def list_weather_stations(self) -> list[dict[str, Any]]:
        return normalize_collection(
            self._request("GET", Endpoints.WEATHER_STATIONS), ("stations", "items", "data")
        )

    def weather_status(self) -> Mapping[str, Any]:
        result = self._request("GET", Endpoints.WEATHER_STATUS)
        if not isinstance(result, Mapping):
            raise APIError("기상 설정 상태 API가 JSON 객체가 아닌 응답을 반환했습니다.")
        return result

    def weather_observations(
        self,
        *,
        source: str,
        station_id: str,
        start_date: str,
        end_date: str,
        service_key: str | None = None,
    ) -> Mapping[str, Any]:
        payload = {
            "source": source,
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
        }
        if service_key and service_key.strip():
            payload["service_key"] = service_key.strip()
        result = self._request(
            "POST",
            Endpoints.WEATHER_OBSERVATIONS,
            json=payload,
        )
        return result if isinstance(result, Mapping) else {"records": result}

    def create_prediction(
        self,
        *,
        files: Sequence[UploadPart],
        model_id: str,
        horizon_steps: int,
        threshold: float,
        source_dates: Sequence[str],
        weather: Sequence[Mapping[str, Any]] | None,
        pixel_area_m2: float,
        target_dates: Sequence[str] | None = None,
        input_metadata: Mapping[str, Any] | None = None,
        historical_water_levels: Sequence[float | None] | None = None,
        reference_water_levels: Sequence[float | None] | None = None,
        evaluation_kind: str | None = None,
        evaluation_truth_provenance: str | None = None,
        water_level_config: Mapping[str, Any] | None = None,
        model_options: Mapping[str, Any] | None = None,
        caution_pct: float = 5.0,
        risk_pct: float = 15.0,
    ) -> Mapping[str, Any]:
        multipart: list[tuple[str, tuple[str, bytes | BinaryIO, str]]] = [
            ("files", (part.name, part.content, part.content_type)) for part in files
        ]
        normalized_weather = []
        for raw_row in weather or []:
            row = dict(raw_row)
            if row.get("date") is None and row.get("timestamp") is not None:
                row["date"] = row["timestamp"]
            normalized_weather.append(row)
        form = {
            "model_id": model_id,
            "horizon_steps": str(horizon_steps),
            "threshold": str(threshold),
            "source_dates_json": json.dumps(list(source_dates), ensure_ascii=False),
            "target_dates_json": json.dumps(list(target_dates or []), ensure_ascii=False),
            "input_metadata_json": json.dumps(
                dict(input_metadata or {}), ensure_ascii=False
            ),
            "weather_json": json.dumps(normalized_weather, ensure_ascii=False),
            "historical_water_levels_json": json.dumps(
                list(historical_water_levels or []), ensure_ascii=False
            ),
            "reference_water_levels_json": json.dumps(
                list(reference_water_levels or []), ensure_ascii=False
            ),
            "evaluation_kind": evaluation_kind or "",
            "evaluation_truth_provenance": evaluation_truth_provenance or "",
            "pixel_area_m2": str(pixel_area_m2),
            "water_level_config_json": json.dumps(water_level_config, ensure_ascii=False)
            if water_level_config
            else "",
            "model_options_json": json.dumps(dict(model_options or {}), ensure_ascii=False),
            "caution_pct": str(caution_pct),
            "risk_pct": str(risk_pct),
        }
        result = self._request("POST", Endpoints.PREDICTIONS, files=multipart, data=form)
        if not isinstance(result, Mapping):
            raise APIError("예측 API가 JSON 객체가 아닌 응답을 반환했습니다.")
        return result

    def get_prediction(self, prediction_id: str) -> Mapping[str, Any]:
        path = Endpoints.PREDICTION.format(prediction_id=quote(str(prediction_id), safe=""))
        result = self._request("GET", path)
        if not isinstance(result, Mapping):
            raise APIError("예측 조회 API가 JSON 객체가 아닌 응답을 반환했습니다.")
        return result

    def list_files(self, prediction_id: str) -> list[dict[str, Any]]:
        """Return every downloadable artifact, including the result manifest."""

        path = Endpoints.FILES.format(
            prediction_id=quote(str(prediction_id), safe="")
        )
        return normalize_collection(
            self._request("GET", path), ("items", "artifacts", "files", "data")
        )

    def wait_for_prediction(
        self,
        prediction_id: str,
        *,
        max_wait_seconds: float = 20.0,
        poll_interval_seconds: float = 0.8,
    ) -> Mapping[str, Any]:
        deadline = time.monotonic() + max_wait_seconds
        latest: Mapping[str, Any] = {"id": prediction_id, "status": "queued"}
        while time.monotonic() < deadline:
            latest = self.get_prediction(prediction_id)
            status = str(latest.get("status", "")).lower()
            if status in {"completed", "complete", "succeeded", "failed", "error"}:
                return latest
            time.sleep(poll_interval_seconds)
        return latest

    def download_artifact(self, prediction_id: str, name: str) -> bytes:
        path = Endpoints.FILE.format(
            prediction_id=quote(str(prediction_id), safe=""),
            name=quote(str(name), safe=""),
        )
        result = self._request("GET", path)
        if not isinstance(result, bytes):
            raise APIError("산출물 API가 파일 바이트를 반환하지 않았습니다.")
        return result

    def download_bundle(self, prediction_id: str) -> bytes:
        path = Endpoints.BUNDLE.format(prediction_id=quote(str(prediction_id), safe=""))
        result = self._request("GET", path)
        if not isinstance(result, bytes):
            raise APIError("번들 API가 ZIP 파일 바이트를 반환하지 않았습니다.")
        return result

    def get_openapi(self) -> Mapping[str, Any]:
        result = self._request("GET", Endpoints.OPENAPI)
        if not isinstance(result, Mapping):
            raise APIError("OpenAPI 문서가 JSON 객체가 아닙니다.")
        return result


def normalize_collection(payload: Any, keys: Iterable[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = next((payload[key] for key in keys if isinstance(payload.get(key), list)), [])
    else:
        values = []
    result: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            result.append(dict(value))
        else:
            result.append({"id": str(value), "name": str(value)})
    return result


def weather_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract record arrays while leaving field normalization to ``state``."""

    for key in ("rows", "records", "observations", "items", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            nested = value.get("records") or value.get("items")
            if isinstance(nested, list):
                return [dict(item) for item in nested if isinstance(item, Mapping)]
    return []
