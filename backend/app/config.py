from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings kept intentionally small and environment-friendly."""

    result_dir: Path
    predictor_plugins: str = ""
    api_prefix: str = "/api/v1"
    title: str = "Water Time-series Prediction API"
    version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> Settings:
        default_result_dir = Path(__file__).resolve().parents[1] / "data" / "results"
        return cls(
            result_dir=Path(
                os.getenv("BACKEND_RESULT_DIR", str(default_result_dir))
            ).resolve(),
            predictor_plugins=os.getenv("PREDICTOR_PLUGINS", ""),
        )
