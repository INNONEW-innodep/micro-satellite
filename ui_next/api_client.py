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
    # 지역 레지스트리 · 지상 수위계 시계열 예측
    GAUGE_FORECASTERS = "/api/v1/gauge/forecasters"
    REGIONS = "/api/v1/regions"
    REGION = "/api/v1/regions/{region_id}"
    GAUGE_STATIONS = "/api/v1/regions/{region_id}/gauge/stations"
    GAUGE_EVALUATION = "/api/v1/regions/{region_id}/gauge/evaluation"
    GAUGE_EXPERIMENTS = "/api/v1/regions/{region_id}/gauge/experiments"
    GAUGE_TRAIN = "/api/v1/regions/{region_id}/gauge/train"
    GAUGE_TRAIN_JOBS = "/api/v1/regions/{region_id}/gauge/train/jobs"
    GAUGE_TRAIN_JOB = "/api/v1/regions/{region_id}/gauge/train/jobs/{job_id}"
    GAUGE_MODELS = "/api/v1/regions/{region_id}/gauge/models"
    GAUGE_SERIES = "/api/v1/regions/{region_id}/gauge/{station_id}/series"
    GAUGE_SERIES_FILE = "/api/v1/regions/{region_id}/gauge/{station_id}/series/file"
    GAUGE_MODEL = "/api/v1/regions/{region_id}/gauge/{station_id}/model"
    GAUGE_FORECAST = "/api/v1/regions/{region_id}/gauge/{station_id}/forecast"
    GAUGE_BACKTEST = "/api/v1/regions/{region_id}/gauge/{station_id}/backtest"
    GAUGE_FORECAST_COMPARE = "/api/v1/regions/{region_id}/gauge/{station_id}/forecast/compare"
    SATELLITE_LEVELS = "/api/v1/regions/{region_id}/satellite/levels"
    SATELLITE_DETECTION = "/api/v1/regions/{region_id}/satellite/detection"
    SATELLITE_SCENES = "/api/v1/regions/{region_id}/satellite/scenes"
    SATELLITE_TIMESERIES = "/api/v1/regions/{region_id}/satellite/timeseries"
    SATELLITE_AREA_SERIES = "/api/v1/regions/{region_id}/satellite/area-series"
    GAUGE_MODEL_GUIDE = "/api/v1/gauge/model-guide"
    REGION_WEATHER = "/api/v1/regions/{region_id}/weather"
    SEGMENTERS = "/api/v1/segmenters"
    WBMS_STATUS = "/api/v1/wbms/status"


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


_PYDANTIC_KO = {
    "string_too_short": "너무 짧습니다",
    "string_too_long": "너무 깁니다",
    "missing": "필수 항목이 비어 있습니다",
    "value_error": "값이 올바르지 않습니다",
    "greater_than_equal": "너무 작습니다",
    "less_than_equal": "너무 큽니다",
    "greater_than": "너무 작습니다",
    "less_than": "너무 큽니다",
    "int_parsing": "정수여야 합니다",
    "float_parsing": "숫자여야 합니다",
    "literal_error": "허용되지 않는 값입니다",
    "json_invalid": "JSON 형식이 아닙니다",
}


def _format_validation_errors(items: list[Any]) -> str:
    """FastAPI 422 의 pydantic 오류 목록을 사람이 읽는 한 줄로."""
    parts = []
    for item in items:
        if not isinstance(item, Mapping):
            parts.append(str(item))
            continue
        loc = [str(x) for x in (item.get("loc") or []) if str(x) not in ("body", "query", "path", "form")]
        field = ".".join(loc) or "입력"
        kind = str(item.get("type") or "")
        ko = _PYDANTIC_KO.get(kind)
        msg = str(item.get("msg") or "")
        parts.append(f"'{field}': {ko or msg}" + (f" ({msg})" if ko and msg else ""))
    return "입력값 검증 실패 — " + "; ".join(parts) if parts else ""


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
                elif isinstance(nested, list):
                    message = _format_validation_errors(nested) or message
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

    # --- 지역 레지스트리 · 수위계 시계열 예측 ---------------------------------------

    @staticmethod
    def _path(template: str, **parts: str) -> str:
        return template.format(**{key: quote(str(value), safe="") for key, value in parts.items()})

    def _mapping(self, result: Any, what: str) -> Mapping[str, Any]:
        if not isinstance(result, Mapping):
            raise APIError(f"{what} API가 JSON 객체가 아닌 응답을 반환했습니다.")
        return result

    def list_gauge_forecasters(self) -> list[dict[str, Any]]:
        return normalize_collection(self._request("GET", Endpoints.GAUGE_FORECASTERS), ("items",))

    def list_regions(self) -> list[dict[str, Any]]:
        return normalize_collection(self._request("GET", Endpoints.REGIONS), ("items",))

    def get_region(self, region_id: str) -> Mapping[str, Any]:
        return self._mapping(self._request("GET", self._path(Endpoints.REGION, region_id=region_id)), "지역")

    def create_region(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._mapping(self._request("POST", Endpoints.REGIONS, json=dict(payload)), "지역 생성")

    def gauge_evaluation(self, region_id: str) -> Mapping[str, Any]:
        return self._mapping(
            self._request("GET", self._path(Endpoints.GAUGE_EVALUATION, region_id=region_id)), "수위 평가"
        )

    def gauge_experiments(self, region_id: str) -> Mapping[str, Any]:
        """부수 실험 산출(미래 강수 상한 실험 등): {items: {future_rain: <json | null>}}. 공식 평가표와 별개."""
        return self._mapping(
            self._request("GET", self._path(Endpoints.GAUGE_EXPERIMENTS, region_id=region_id)), "수위 실험"
        )

    def gauge_train(
        self,
        region_id: str,
        *,
        stations: Sequence[str] | None = None,
        forecasters: Sequence[str] | None = None,
        horizon_max: int | None = None,
        split: Mapping[str, Any] | None = None,
        async_run: bool = False,
    ) -> Mapping[str, Any]:
        """지역 수위 모델 학습. ``split`` 은 {train_end, val_start, val_end, test_start, test_end} ISO 날짜.

        ``async_run=True`` 면 서버가 202 ``{job_id, status:"queued", ...}`` 를 바로 돌려주고 뒤에서 학습한다 —
        진행은 :meth:`gauge_train_job` 으로 본다. 기본(False)은 예전처럼 학습이 끝날 때까지 기다린다.
        """
        body: dict[str, Any] = {}
        if stations:
            body["stations"] = list(stations)
        if forecasters:
            body["forecasters"] = list(forecasters)
        if horizon_max:
            body["horizon_max"] = int(horizon_max)
        if split:
            body["split"] = {key: str(value)[:10] for key, value in dict(split).items() if value}
        if async_run:
            body["async_run"] = True
        return self._mapping(
            self._request("POST", self._path(Endpoints.GAUGE_TRAIN, region_id=region_id), json=body, timeout=600.0),
            "수위 모델 학습",
        )

    def gauge_train_job(self, region_id: str, job_id: str) -> Mapping[str, Any]:
        """비동기 학습 작업 하나의 상태 — status(queued|running|done|failed)·progress·elapsed_s·evaluation(완료 시)."""
        return self._mapping(
            self._request("GET", self._path(Endpoints.GAUGE_TRAIN_JOB, region_id=region_id, job_id=job_id)),
            "수위 모델 학습 작업",
        )

    def gauge_train_jobs(self, region_id: str) -> list[dict[str, Any]]:
        """지역의 학습 작업 목록(최신순)."""
        return normalize_collection(
            self._request("GET", self._path(Endpoints.GAUGE_TRAIN_JOBS, region_id=region_id)), ("items",)
        )

    def gauge_reset_models(self, region_id: str) -> Mapping[str, Any]:
        """사용자 디렉터리의 학습 산출물을 지워 내장(기본 구간) 산출물로 되돌린다."""
        return self._mapping(
            self._request("DELETE", self._path(Endpoints.GAUGE_MODELS, region_id=region_id)),
            "수위 모델 초기화",
        )

    def gauge_series(
        self, region_id: str, station_id: str, *, start: str | None = None, end: str | None = None
    ) -> Mapping[str, Any]:
        params = {key: value for key, value in (("start", start), ("end", end)) if value}
        return self._mapping(
            self._request(
                "GET", self._path(Endpoints.GAUGE_SERIES, region_id=region_id, station_id=station_id), params=params
            ),
            "수위 시계열",
        )

    def upload_gauge_series(
        self,
        region_id: str,
        station_id: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        replace: bool = False,
        station: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        body = {"rows": [dict(row) for row in rows], "replace": bool(replace), "station": dict(station or {})}
        return self._mapping(
            self._request(
                "POST", self._path(Endpoints.GAUGE_SERIES, region_id=region_id, station_id=station_id), json=body
            ),
            "수위 시계열 업로드",
        )

    def gauge_model(self, region_id: str, station_id: str) -> Mapping[str, Any]:
        return self._mapping(
            self._request("GET", self._path(Endpoints.GAUGE_MODEL, region_id=region_id, station_id=station_id)),
            "수위 모델",
        )

    def gauge_forecast(
        self,
        region_id: str,
        station_id: str,
        *,
        as_of: str | None = None,
        horizon: int | None = None,
        history_days: int = 120,
        forecaster: str | None = None,
    ) -> Mapping[str, Any]:
        params: dict[str, Any] = {"history_days": int(history_days)}
        if as_of:
            params["as_of"] = as_of
        if horizon:
            params["horizon"] = int(horizon)
        if forecaster:
            params["forecaster"] = forecaster
        return self._mapping(
            self._request(
                "GET", self._path(Endpoints.GAUGE_FORECAST, region_id=region_id, station_id=station_id), params=params
            ),
            "수위 전망",
        )

    def satellite_timeseries(self, region_id: str) -> Mapping[str, Any]:
        return self._mapping(
            self._request("GET", self._path(Endpoints.SATELLITE_TIMESERIES, region_id=region_id)), "위성 시계열 지표"
        )

    def satellite_area_series(self, region_id: str) -> Mapping[str, Any]:
        """공통 격자 기준 수체 면적 시계열 — 날짜 간 비교가 가능한 유일한 면적 계열.

        `satellite_scenes` 의 `water_area_km2` 는 씬마다 촬영범위가 달라 계열로 쓰면 안 된다.
        """

        return self._mapping(
            self._request("GET", self._path(Endpoints.SATELLITE_AREA_SERIES, region_id=region_id)), "공통 격자 면적 계열"
        )

    def gauge_model_guide(self) -> Mapping[str, Any]:
        return self._mapping(self._request("GET", Endpoints.GAUGE_MODEL_GUIDE), "모델 안내")

    def region_weather(self, region_id: str, *, start: str | None = None, end: str | None = None) -> Mapping[str, Any]:
        """지역에 저장된 기상청 ASOS 일자료(학습 외생 입력과 동일). 실시간 KMA 조회보다 먼저 쓴다."""
        params = {k: v for k, v in (("start", start), ("end", end)) if v}
        return self._mapping(
            self._request("GET", self._path(Endpoints.REGION_WEATHER, region_id=region_id), params=params), "지역 기상"
        )

    def satellite_scenes(self, region_id: str) -> Mapping[str, Any]:
        return self._mapping(
            self._request("GET", self._path(Endpoints.SATELLITE_SCENES, region_id=region_id)), "위성 장면"
        )

    def register_scene(
        self,
        region_id: str,
        *,
        sensor: str,
        sensor_type: str,
        image_date: str,
        pre_file: tuple[str, bytes] | None = None,
        mask_file: tuple[str, bytes] | None = None,
        gt_file: tuple[str, bytes] | None = None,
        pre_path: str | None = None,
        mask_path: str | None = None,
        gt_path: str | None = None,
        run_detection: bool = False,
        testbed: str | None = None,
        note: str | None = None,
        timeout: float = 600.0,
    ) -> Mapping[str, Any]:
        """새 위성 장면 등록(POST /regions/{id}/satellite/scenes, multipart).

        ``*_file`` 은 ``(파일명, 바이트)`` 업로드, ``*_path`` 는 서버(호스트) 경로. 전처리 GeoTIFF 는 둘 중 하나가 필요하다.
        """

        if not (pre_file or pre_path):
            raise APIError("전처리 GeoTIFF(업로드 또는 서버 경로)가 필요합니다")
        form: dict[str, str] = {
            "sensor": sensor.strip(),
            "sensor_type": sensor_type.strip().upper(),
            "image_date": image_date.replace("-", "").strip(),
            "run_detection": "true" if run_detection else "false",
        }
        for key, value in (("pre_path", pre_path), ("mask_path", mask_path), ("gt_path", gt_path), ("testbed", testbed), ("note", note)):
            if value:
                form[key] = str(value).strip()
        multipart: list[tuple[str, tuple[str, bytes, str]]] = []
        for field, part in (("pre_file", pre_file), ("mask_file", mask_file), ("gt_file", gt_file)):
            if part:
                name, content = part
                multipart.append((field, (name or f"{field}.tif", content, "image/tiff")))
        return self._mapping(
            self._request(
                "POST", self._path(Endpoints.SATELLITE_SCENES, region_id=region_id),
                data=form, files=multipart or None, timeout=timeout,
            ),
            "위성 장면 등록",
        )

    def segmenters(self) -> Mapping[str, Any]:
        """등록된 수체 탐지 어댑터 목록(GET /segmenters)."""
        return self._mapping(self._request("GET", Endpoints.SEGMENTERS), "탐지 어댑터 목록")

    def wbms_status(self) -> Mapping[str, Any]:
        """WBMS 인도 모델·도커 이미지 가용 상태(GET /wbms/status)."""
        return self._mapping(self._request("GET", Endpoints.WBMS_STATUS), "WBMS 상태")

    def fetch_bytes(self, path_or_url: str) -> bytes:
        """API가 준 상대 URL(또는 절대 URL)의 바이트를 받는다 — 퀵룩 PNG 등."""

        url = path_or_url if path_or_url.startswith("http") else self.absolute_url(path_or_url)
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise APIError(f"파일을 받지 못했습니다: {exc}") from exc
        if response.status_code != 200:
            raise APIError(f"파일 응답 {response.status_code}: {path_or_url}", status_code=response.status_code)
        return response.content

    def satellite_detection(self, region_id: str) -> Mapping[str, Any]:
        return self._mapping(
            self._request("GET", self._path(Endpoints.SATELLITE_DETECTION, region_id=region_id)), "탐지 정확도"
        )

    def satellite_levels(self, region_id: str, station_id: str | None = None) -> Mapping[str, Any]:
        params = {"station_id": station_id} if station_id else {}
        return self._mapping(
            self._request("GET", self._path(Endpoints.SATELLITE_LEVELS, region_id=region_id), params=params),
            "위성 수위",
        )

    def gauge_forecast_compare(self, region_id: str, station_id: str, *, as_of: str | None = None, horizon: int | None = None) -> Mapping[str, Any]:
        params: dict[str, Any] = {}
        if as_of:
            params["as_of"] = as_of
        if horizon:
            params["horizon"] = int(horizon)
        return self._mapping(
            self._request("GET", self._path(Endpoints.GAUGE_FORECAST_COMPARE, region_id=region_id, station_id=station_id), params=params, timeout=180.0),
            "모델별 전망 비교",
        )

    def gauge_backtest(self, region_id: str, station_id: str, *, horizon: int = 7) -> Mapping[str, Any]:
        return self._mapping(
            self._request(
                "GET",
                self._path(Endpoints.GAUGE_BACKTEST, region_id=region_id, station_id=station_id),
                params={"horizon": int(horizon)},
            ),
            "수위 백테스트",
        )


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
