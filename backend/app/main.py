from __future__ import annotations

import logging
import os

import json
from datetime import date
from typing import Annotated, Any, TypeVar

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ValidationError

from .adapters.registry import AdapterRegistry, build_registry
from .api_docs import API_DOCS, SERVICE_DESCRIPTION, TAG_DOCS
from .config import Settings
from .schemas import (
    ArtifactInfo,
    ArtifactList,
    HealthResponse,
    ModelInfo,
    PredictionList,
    PredictionResult,
    WaterLevelConfig,
    WeatherRow,
)
from .adapters.gauge_level import GaugeLevelForecastAdapter
from .adapters.wbms_fusion import WbmsFusionAdapter
from .adapters.wbms_segmentation import WbmsSegmentationJobAdapter
from .gauge.forecasters import build_forecaster_registry
from .gauge.jobs import GaugeTrainJobs
from .gauge.regions import RegionRegistry
from .gauge.router import router as gauge_router
from .gauge.service import GaugeForecastService
from .segmentation.registry import SegmenterRegistry
from .segmentation.router import router as segmenters_router
from .segmentation.wbms_plugin import WbmsDockerSegmenter
from .services.inputs import UploadedBytes, load_mask_sequence
from .services.prediction import PredictionService
from .services.store import ArtifactNotFoundError, FileResultStore, ResultNotFoundError
from .weather import router as weather_router
from .wbms.config import WbmsSettings
from .wbms.router import router as wbms_router
from .wbms.runner import DockerRunner

T = TypeVar("T")


def _ensure_logging() -> None:
    """루트 로거에 핸들러가 없으면(uvicorn 은 자기 로거만 꾸민다) 시각·수준·이름이 붙는 핸들러를 하나 단다 — 오류 로그가 컨테이너 로그에 남게."""

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=os.getenv("BACKEND_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(
    settings: Settings | None = None, wbms_settings: WbmsSettings | None = None
) -> FastAPI:
    _ensure_logging()
    settings = settings or Settings.from_env()
    wbms_settings = wbms_settings or WbmsSettings.from_env()
    registry = build_registry(settings.predictor_plugins)
    store = FileResultStore(settings.result_dir)
    prediction_service = PredictionService(registry, store, settings.api_prefix)
    wbms_runner = DockerRunner(wbms_settings.docker_bin)

    app = FastAPI(
        title=settings.title,
        version=settings.version,
        description=SERVICE_DESCRIPTION,
        openapi_tags=[
            {"name": name, "description": TAG_DOCS[name]}
            for name in ("system", "scenes", "predictions", "weather", "wbms", "regions")
        ],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    # 2026-10-20: 오류 응답 통일(한국어 사유·필드별 안내·요청 id·최근 오류 조회) + 오류 로그
    from .errors import install_error_handlers

    install_error_handlers(app)
    app.state.registry = registry
    app.state.store = store
    app.state.prediction_service = prediction_service
    app.state.wbms_settings = wbms_settings
    app.state.wbms_segmentation = WbmsSegmentationJobAdapter(wbms_settings, wbms_runner)
    app.state.wbms_fusion = WbmsFusionAdapter(wbms_settings, wbms_runner)
    # 수체 탐지 어댑터 레지스트리 — 인도 WBMS 도커 모델이 첫 내장 어댑터
    segmenters = SegmenterRegistry()
    segmenters.register(WbmsDockerSegmenter(app.state.wbms_segmentation))
    segmenters.load_plugins(settings.segmenter_plugins)
    app.state.segmenters = segmenters
    # 지역 레지스트리 + 수위계 시계열 예측기 (플러그인 교체 가능)
    app.state.regions = RegionRegistry(settings.regions_dir, settings.regions_user_dir)
    app.state.gauge_forecasters = build_forecaster_registry(settings.gauge_forecaster_plugins)
    app.state.gauge_service = GaugeForecastService(app.state.regions, app.state.gauge_forecasters)
    # 백그라운드 학습 잡(async_run=true) — 프로세스 메모리, 작업자 스레드 1개
    app.state.gauge_train_jobs = GaugeTrainJobs()
    # 기존 예측 API에서도 학습 수위 모델을 고를 수 있게 한다 — persistence 만 수위를 내던 문제의 해법
    registry.register(GaugeLevelForecastAdapter(app.state.gauge_service), built_in=True)
    # 면적을 수위 전망에 종속시키는 다리 모델 (2026-10-15): 면적 = 마지막 탐지 면적 + 기울기 × 수위 변화
    from .adapters.level_area_bridge import LevelAreaBridgeAdapter

    registry.register(LevelAreaBridgeAdapter(app.state.gauge_service), built_in=True)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        # 서비스 화면이 정문. API 문서는 /docs, 쉬운 화면 /easy, 모델 평가 /eval.
        return RedirectResponse(url="/app")

    @app.get(
        f"{settings.api_prefix}/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Check backend readiness",
        description=API_DOCS["health"],
    )
    async def health(request: Request) -> HealthResponse:
        current_registry: AdapterRegistry = request.app.state.registry
        current_store: FileResultStore = request.app.state.store
        return HealthResponse(
            service=settings.title,
            version=settings.version,
            models_loaded=len(current_registry),
            result_store=str(current_store.root),
            plugin_errors=current_registry.load_errors,
            store_errors=current_store.load_errors,
        )

    @app.get(
        f"{settings.api_prefix}/models",
        response_model=list[ModelInfo],
        tags=["system"],
        summary="List selectable predictor adapters",
        description=API_DOCS["models_list"],
    )
    async def models(request: Request) -> list[ModelInfo]:
        return request.app.state.registry.list()

    @app.get(
        f"{settings.api_prefix}/models/{{model_id}}",
        response_model=ModelInfo,
        tags=["system"],
        summary="Describe one predictor adapter",
        description=API_DOCS["models_detail"],
    )
    async def model_detail(model_id: str, request: Request) -> ModelInfo:
        try:
            return request.app.state.registry.describe(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        f"{settings.api_prefix}/predictions",
        response_model=PredictionResult,
        status_code=201,
        tags=["predictions"],
        summary="Run a time-series prediction",
        description=API_DOCS["predictions_create"],
    )
    async def create_prediction(
        request: Request,
        files: Annotated[
            list[UploadFile],
            File(description="Ordered NPY, PNG, TIFF, or GeoTIFF masks"),
        ],
        model_id: Annotated[str, Form()] = "persistence",
        horizon_steps: Annotated[int, Form(ge=1, le=365)] = 3,
        threshold: Annotated[float, Form(ge=0, le=1)] = 0.5,
        pixel_area_m2: Annotated[float, Form(gt=0)] = 9.0,
        source_dates_json: Annotated[str | None, Form()] = None,
        target_dates_json: Annotated[str | None, Form()] = None,
        input_metadata_json: Annotated[str | None, Form()] = None,
        weather_json: Annotated[str | None, Form()] = None,
        historical_water_levels_json: Annotated[str | None, Form()] = None,
        reference_water_levels_json: Annotated[str | None, Form()] = None,
        evaluation_kind: Annotated[str | None, Form()] = None,
        evaluation_truth_provenance: Annotated[str | None, Form()] = None,
        water_level_config_json: Annotated[str | None, Form()] = None,
        model_options_json: Annotated[str | None, Form()] = None,
        caution_pct: Annotated[float, Form(ge=0)] = 5.0,
        risk_pct: Annotated[float, Form(gt=0)] = 15.0,
    ) -> PredictionResult:
        try:
            request.app.state.registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            uploads = [
                UploadedBytes(name=item.filename or "upload", content=await item.read())
                for item in files
            ]
            loaded = await run_in_threadpool(load_mask_sequence, uploads, threshold)
            source_dates = _parse_dates(source_dates_json, "source_dates_json")
            target_dates = _parse_dates(target_dates_json, "target_dates_json")
            input_metadata = _parse_object(
                input_metadata_json, "input_metadata_json"
            )
            weather = _parse_models(weather_json, "weather_json", WeatherRow)
            historical_levels = _parse_optional_floats(
                historical_water_levels_json, "historical_water_levels_json"
            )
            reference_levels = _parse_optional_floats(
                reference_water_levels_json, "reference_water_levels_json"
            )
            water_level_config = _parse_optional_model(
                water_level_config_json, "water_level_config_json", WaterLevelConfig
            )
            model_options = _parse_object(model_options_json, "model_options_json")
            return await run_in_threadpool(
                request.app.state.prediction_service.run,
                loaded=loaded,
                model_id=model_id,
                horizon_steps=horizon_steps,
                threshold=threshold,
                source_dates=source_dates,
                target_dates=target_dates,
                input_metadata=input_metadata,
                weather=weather,
                historical_water_levels_m=historical_levels,
                reference_water_levels_m=reference_levels,
                evaluation_kind=evaluation_kind,
                evaluation_truth_provenance=evaluation_truth_provenance,
                pixel_area_m2=pixel_area_m2,
                water_level_config=water_level_config,
                caution_pct=caution_pct,
                risk_pct=risk_pct,
                model_options=model_options,
            )
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            for item in files:
                await item.close()

    @app.get(
        f"{settings.api_prefix}/predictions",
        response_model=PredictionList,
        tags=["predictions"],
        summary="List persisted predictions",
        description=API_DOCS["predictions_list"],
    )
    async def list_predictions(
        request: Request,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> PredictionList:
        items = request.app.state.store.list()
        return PredictionList(items=items[offset : offset + limit], total=len(items))

    @app.get(
        f"{settings.api_prefix}/predictions/{{prediction_id}}",
        response_model=PredictionResult,
        tags=["predictions"],
        summary="Get one complete prediction result",
        description=API_DOCS["predictions_detail"],
    )
    async def prediction_detail(
        prediction_id: str, request: Request
    ) -> PredictionResult:
        try:
            return request.app.state.store.get(prediction_id)
        except ResultNotFoundError as exc:
            raise HTTPException(status_code=404, detail="prediction not found") from exc

    @app.get(
        f"{settings.api_prefix}/predictions/{{prediction_id}}/files",
        response_model=ArtifactList,
        tags=["predictions"],
        summary="List generated mask artifacts",
        description=API_DOCS["predictions_files"],
    )
    async def prediction_files(prediction_id: str, request: Request) -> ArtifactList:
        try:
            result = request.app.state.store.get(prediction_id)
        except ResultNotFoundError as exc:
            raise HTTPException(status_code=404, detail="prediction not found") from exc
        manifest = request.app.state.store.root / prediction_id / "result.json"
        manifest_info = ArtifactInfo(
            name="result.json",
            kind="metadata",
            media_type="application/json",
            size_bytes=manifest.stat().st_size if manifest.is_file() else 0,
            url=f"{settings.api_prefix}/predictions/{prediction_id}/files/result.json",
        )
        return ArtifactList(
            prediction_id=prediction_id, items=[manifest_info, *result.artifacts]
        )

    @app.get(
        f"{settings.api_prefix}/predictions/{{prediction_id}}/files/{{filename}}",
        tags=["predictions"],
        summary="View or download one generated file",
        description=API_DOCS["predictions_file"],
    )
    async def prediction_file(
        prediction_id: str,
        filename: str,
        request: Request,
        download: bool = False,
    ) -> FileResponse:
        try:
            path, media_type = request.app.state.store.artifact_path(
                prediction_id, filename
            )
        except (ResultNotFoundError, ArtifactNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        return FileResponse(
            path, media_type=media_type, filename=path.name if download else None
        )

    @app.get(
        f"{settings.api_prefix}/predictions/{{prediction_id}}/bundle",
        tags=["predictions"],
        summary="Download metadata and every output file as ZIP",
        description=API_DOCS["predictions_bundle"],
    )
    async def prediction_bundle(prediction_id: str, request: Request) -> FileResponse:
        try:
            path = request.app.state.store.bundle_path(prediction_id)
        except ResultNotFoundError as exc:
            raise HTTPException(status_code=404, detail="prediction not found") from exc
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.get(
        f"{settings.api_prefix}/predictions/{{prediction_id}}/artifacts",
        response_model=ArtifactList,
        tags=["predictions"],
        summary="Alias for the generated file list",
        include_in_schema=False,
    )
    async def prediction_artifacts_alias(
        prediction_id: str, request: Request
    ) -> ArtifactList:
        return await prediction_files(prediction_id, request)

    @app.get(
        f"{settings.api_prefix}/predictions/{{prediction_id}}/artifacts/{{filename}}",
        tags=["predictions"],
        summary="Alias for viewing or downloading a generated file",
        include_in_schema=False,
    )
    async def prediction_artifact_alias(
        prediction_id: str,
        filename: str,
        request: Request,
        download: bool = False,
    ) -> FileResponse:
        return await prediction_file(prediction_id, filename, request, download)

    app.include_router(weather_router, prefix=settings.api_prefix)
    app.include_router(wbms_router, prefix=settings.api_prefix)
    app.include_router(gauge_router, prefix=settings.api_prefix)
    app.include_router(segmenters_router, prefix=settings.api_prefix)
    # 쉬운 화면 — 위성 영상 1장을 올려 결과를 받는 단일 페이지 (/easy). 외부 의존 없음.
    from .easy_ui import mount_easy_ui
    from .pipeline import install_pipeline

    # 입구 하나짜리 장면 파이프라인 — 등록→탐지→면적→수위→예측→규약 파일·로그 (/api/v1/scenes)
    install_pipeline(app)
    # 서비스·평가 화면용 API — 지오메트리·준비상태·WAMIS 지점·수위/기상 수집 잡·장면 재예측 (파이프라인 뒤에)
    from .screens_api import install_screens_api

    install_screens_api(app)
    mount_easy_ui(app)
    # 서비스 화면 (/app) — 자료 넣기·위치 확인 → 탐지 → 시각화 → 예측
    from .service_ui import mount_service_ui

    mount_service_ui(app)
    # 모델 평가 화면 (/eval) — 지역 → 수위·기상 자료 수집 → 학습 → 성능 평가 → 시험
    from .eval_ui import mount_eval_ui

    mount_eval_ui(app)
    # 간단 화면 /simple — 지역·날짜 고르고 버튼 하나로 그림·수위·면적 (2026-10-15)
    from .simple_ui import mount_simple_ui

    mount_simple_ui(app)
    return app


def _load_json(raw: str | None, label: str, default: T) -> Any | T:
    if raw is None or not raw.strip():
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON: {exc.msg}") from exc


def _parse_dates(raw: str | None, label: str) -> list[date]:
    values = _load_json(raw, label, [])
    if not isinstance(values, list):
        raise TypeError(f"{label} must be a JSON array")
    try:
        return [date.fromisoformat(str(value)) for value in values]
    except ValueError as exc:
        raise ValueError(f"{label} values must use YYYY-MM-DD") from exc


def _parse_models(raw: str | None, label: str, model: type[BaseModel]) -> list[Any]:
    values = _load_json(raw, label, [])
    if not isinstance(values, list):
        raise TypeError(f"{label} must be a JSON array")
    return [model.model_validate(value) for value in values]


def _parse_optional_floats(raw: str | None, label: str) -> list[float | None]:
    values = _load_json(raw, label, [])
    if not isinstance(values, list):
        raise TypeError(f"{label} must be a JSON array")
    parsed: list[float | None] = []
    for value in values:
        if value is None:
            parsed.append(None)
        elif isinstance(value, bool):
            raise ValueError(f"{label} must contain numbers or null")
        else:
            try:
                parsed.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} must contain numbers or null") from exc
    return parsed


def _parse_optional_model(
    raw: str | None, label: str, model: type[BaseModel]
) -> Any | None:
    value = _load_json(raw, label, None)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return model.model_validate(value)


def _parse_object(raw: str | None, label: str) -> dict[str, Any]:
    value = _load_json(raw, label, {})
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


app = create_app()
