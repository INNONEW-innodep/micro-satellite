from __future__ import annotations

import sys
import types
from datetime import date

import numpy as np

from backend.app.adapters.base import PredictionContext, PredictionOutput
from backend.app.adapters.persistence import PersistenceAdapter
from backend.app.adapters.registry import build_registry
from backend.app.adapters.weather_morphology import WeatherMorphologyAdapter
from backend.app.schemas import ModelInfo


def test_persistence_repeats_mask_and_latest_valid_level() -> None:
    frames = np.zeros((2, 5, 5), dtype=np.float32)
    frames[-1, 2, 2] = 1
    output = PersistenceAdapter().predict(
        frames,
        [],
        PredictionContext(
            horizon_steps=3, threshold=0.5, historical_water_levels_m=(1.2, 1.4)
        ),
    )
    assert output.masks.shape == (3, 5, 5)
    assert output.masks[:, 2, 2].tolist() == [1, 1, 1]
    assert output.water_levels_m == [1.4, 1.4, 1.4]


def test_weather_morphology_expands_after_rain() -> None:
    frames = np.zeros((1, 5, 5), dtype=np.float32)
    frames[0, 2, 2] = 1
    output = WeatherMorphologyAdapter().predict(
        frames,
        [{"date": "2026-08-10", "precipitation_mm": 20}],
        PredictionContext(horizon_steps=1, threshold=0.5),
    )
    assert int(output.masks[0].sum()) == 9
    assert output.metadata["operations"][0]["operation"] == "dilate"


def test_weather_morphology_does_not_reuse_dated_history_for_future() -> None:
    frames = np.zeros((1, 5, 5), dtype=np.float32)
    frames[0, 2, 2] = 1
    output = WeatherMorphologyAdapter().predict(
        frames,
        [{"date": "2026-08-01", "precipitation_mm": 100}],
        PredictionContext(
            horizon_steps=1,
            threshold=0.5,
            target_dates=(date(2026, 8, 10),),
        ),
    )
    assert int(output.masks[0].sum()) == 1
    assert output.metadata["operations"][0]["precipitation_mm"] is None
    assert "No weather row matches target date" in output.warnings[0]


def test_weather_morphology_can_cap_boundary_jump_on_coarse_grids() -> None:
    frames = np.zeros((1, 100, 100), dtype=np.float32)
    frames[0, 40:60, 49:51] = 1
    before = int(frames[0].sum())
    output = WeatherMorphologyAdapter().predict(
        frames,
        [{"date": "2026-08-10", "precipitation_mm": 40}],
        PredictionContext(
            horizon_steps=1,
            threshold=0.5,
            options={"max_area_change_pct_per_step": 2.5},
        ),
    )

    assert int(output.masks[0].sum()) == before + 1
    operation = output.metadata["operations"][0]
    assert operation["operation"] == "dilate"
    assert operation["iterations"] == 2
    assert operation["area_change_capped"] is True
    assert operation["uncapped_area_pixels"] > int(output.masks[0].sum())


def test_external_plugin_registry(monkeypatch) -> None:
    module = types.ModuleType("fake_predictor_plugin")

    class Plugin:
        @property
        def info(self):
            return ModelInfo(
                id="fake", name="Fake", version="1", description="test", min_frames=1
            )

        def predict(self, frames, weather, request):
            return PredictionOutput(
                np.repeat(frames[-1][None], request.horizon_steps, axis=0)
            )

    module.Plugin = Plugin
    monkeypatch.setitem(sys.modules, "fake_predictor_plugin", module)
    registry = build_registry("fake_predictor_plugin:Plugin")
    assert registry.get("fake").info.name == "Fake"
    assert next(item for item in registry.list() if item.id == "fake").built_in is False
    assert registry.describe("fake").built_in is False
