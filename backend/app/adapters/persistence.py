from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..schemas import ModelInfo
from .base import PredictionAdapter, PredictionContext, PredictionOutput


class PersistenceAdapter(PredictionAdapter):
    """Strong sanity-check baseline: repeat the latest observed mask."""

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            id="persistence",
            name="Persistence baseline",
            version="1.0.0",
            description=(
                "Repeats the latest input mask and, when supplied, the latest valid observed "
                "water level for every forecast horizon."
            ),
            min_frames=1,
            supports_weather=False,
            produces_water_level=True,
            options={},
        )

    def predict(
        self,
        frames: np.ndarray,
        weather: Sequence[Mapping[str, Any]],
        request: PredictionContext,
    ) -> PredictionOutput:
        masks = np.repeat(frames[-1][None, ...], request.horizon_steps, axis=0)
        last_level = next(
            (
                value
                for value in reversed(request.historical_water_levels_m)
                if value is not None
            ),
            None,
        )
        return PredictionOutput(
            masks=masks,
            water_levels_m=[last_level] * request.horizon_steps
            if last_level is not None
            else None,
            metadata={"strategy": "repeat_last_observation"},
        )
