from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from ..schemas import ModelInfo


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Model-independent request values passed to a predictor adapter."""

    horizon_steps: int
    threshold: float
    source_dates: Sequence[date] = field(default_factory=tuple)
    target_dates: Sequence[date] = field(default_factory=tuple)
    historical_water_levels_m: Sequence[float | None] = field(default_factory=tuple)
    pixel_area_m2: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionOutput:
    """Canonical adapter result.

    Masks may be probabilities or binary values. The service applies the request
    threshold and is responsible for artifact generation and post-processing.
    """

    masks: np.ndarray
    water_levels_m: Sequence[float | None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class PredictionAdapter(ABC):
    """Small interface custom models implement to become API-selectable."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        frames: np.ndarray,
        weather: Sequence[Mapping[str, Any]],
        request: PredictionContext,
    ) -> PredictionOutput:
        """Return ``[horizon, height, width]`` masks and optional level values."""

        raise NotImplementedError
