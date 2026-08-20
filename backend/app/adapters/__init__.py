from .base import PredictionAdapter, PredictionContext, PredictionOutput
from .irregular_area_trend import IrregularAreaTrendAdapter
from .persistence import PersistenceAdapter
from .registry import AdapterRegistry, build_registry
from .weather_morphology import WeatherMorphologyAdapter

__all__ = [
    "AdapterRegistry",
    "IrregularAreaTrendAdapter",
    "PersistenceAdapter",
    "PredictionAdapter",
    "PredictionContext",
    "PredictionOutput",
    "WeatherMorphologyAdapter",
    "build_registry",
]
