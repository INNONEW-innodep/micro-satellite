from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
from typing import Any

from ..schemas import ModelInfo
from .base import PredictionAdapter
from .irregular_area_trend import IrregularAreaTrendAdapter
from .persistence import PersistenceAdapter
from .weather_morphology import WeatherMorphologyAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, PredictionAdapter] = {}
        self._built_in: dict[str, bool] = {}
        self.load_errors: list[str] = []

    def register(self, adapter: PredictionAdapter, *, built_in: bool = False) -> None:
        self._validate_adapter(adapter)
        model_id = adapter.info.id
        if model_id in self._adapters:
            raise ValueError(f"predictor id {model_id!r} is already registered")
        self._adapters[model_id] = adapter
        self._built_in[model_id] = built_in

    def get(self, model_id: str) -> PredictionAdapter:
        try:
            return self._adapters[model_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters))
            raise KeyError(
                f"unknown model {model_id!r}; available: {available}"
            ) from exc

    def list(self) -> list[ModelInfo]:
        return [self.describe(model_id) for model_id in sorted(self._adapters)]

    def describe(self, model_id: str) -> ModelInfo:
        adapter = self.get(model_id)
        return adapter.info.model_copy(update={"built_in": self._built_in[model_id]})

    def __len__(self) -> int:
        return len(self._adapters)

    def load_plugins(self, specs: str | Iterable[str]) -> None:
        if isinstance(specs, str):
            plugin_specs = [item.strip() for item in specs.split(",") if item.strip()]
        else:
            plugin_specs = [item.strip() for item in specs if item.strip()]
        for spec in plugin_specs:
            try:
                self.register(_load_plugin(spec), built_in=False)
            except Exception as exc:  # noqa: BLE001 - isolate third-party plugin failures
                self.load_errors.append(f"{spec}: {exc}")

    @staticmethod
    def _validate_adapter(adapter: Any) -> None:
        if not hasattr(adapter, "info") or not isinstance(adapter.info, ModelInfo):
            raise TypeError("adapter.info must be a ModelInfo instance")
        if not callable(getattr(adapter, "predict", None)):
            raise TypeError("adapter.predict must be callable")


def _load_plugin(spec: str) -> PredictionAdapter:
    if ":" not in spec:
        raise ValueError("plugin must use 'package.module:ClassOrFactory' syntax")
    module_name, attribute_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute_name)
    if inspect.isclass(candidate) or callable(candidate):
        candidate = candidate()
    return candidate


def build_registry(plugin_specs: str = "") -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(PersistenceAdapter(), built_in=True)
    registry.register(IrregularAreaTrendAdapter(), built_in=True)
    registry.register(WeatherMorphologyAdapter(), built_in=True)
    registry.load_plugins(plugin_specs)
    return registry
