from __future__ import annotations

import json
from datetime import date
from typing import Annotated, Any, TypeVar

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ValidationError

from .adapters.registry import AdapterRegistry, build_registry
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
from .adapters.wbms_fusion import WbmsFusionAdapter
from .adapters.wbms_segmentation import WbmsSegmentationJobAdapter
from .services.inputs import UploadedBytes, load_mask_sequence
from .services.prediction import PredictionService
from .services.store import ArtifactNotFoundError, FileResultStore, ResultNotFoundError
from .weather import router as weather_router
from .wbms.config import WbmsSettings
from .wbms.router import router as wbms_router
from .wbms.runner import DockerRunner

T = TypeVar("T")


def create_app(
    settings: Settings | None = None, wbms_settings: WbmsSettings | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    wbms_settings = wbms_settings or WbmsSettings.from_env()
    registry = build_registry(settings.predictor_plugins)
    store = FileResultStore(settings.result_dir)
    prediction_service = PredictionService(registry, store, settings.api_prefix)
    wbms_runner = DockerRunner(wbms_settings.docker_bin)

    app = FastAPI(
        title=settings.title,
        version=settings.version,
        description=(
            "Model-independent API for water-mask time-series forecasting. "
            "Upload upstream masks, select an adapter, and export every result as JSON or files."
        ),
        openapi_tags=[
            {"name": "system", "description": "Service readiness and model discovery."},
            {
                "name": "predictions",
                "description": "Run, inspect, and export predictions.",
            },
            {
                "name": "weather",
                "description": "KMA ASOS observations and offline sample data.",
            },
            {
                "name": "wbms",
                "description": (
                    "Delivered WBMS real models run inside the wbms container via "
                    "host-runner subprocess: ② detect_water as async jobs, "
                    "④ fusion-LSTM same-day water-level correction."
                ),
            },
        ],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.store = store
    app.state.prediction_service = prediction_service
    app.state.wbms_settings = wbms_settings
    app.state.wbms_segmentation = WbmsSegmentationJobAdapter(wbms_settings, wbms_runner)
    app.state.wbms_fusion = WbmsFusionAdapter(wbms_settings, wbms_runner)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get(
        f"{settings.api_prefix}/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Check backend readiness",
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
    )
    async def models(request: Request) -> list[ModelInfo]:
        return request.app.state.registry.list()

    @app.get(
        f"{settings.api_prefix}/models/{{model_id}}",
        response_model=ModelInfo,
        tags=["system"],
        summary="Describe one predictor adapter",
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
        description=(
            "Files are interpreted in multipart order. A single NPY may contain [T,H,W]. "
            "weather_json is a JSON array of rows; historical levels and dates align with decoded frames. "
            "input_metadata_json may carry non-secret source and preprocessing provenance. "
            "Optional reference_water_levels_json aligns with forecast horizons and is used only "
            "after inference to calculate evaluation metrics; it is never passed to the predictor."
        ),
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
