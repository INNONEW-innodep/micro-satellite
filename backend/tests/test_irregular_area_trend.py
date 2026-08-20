from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from backend.app.adapters.base import PredictionContext
from backend.app.adapters.irregular_area_trend import IrregularAreaTrendAdapter


def _square(size: int, radius: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    mask[center - radius : center + radius, center - radius : center + radius] = 1
    return mask


def test_irregular_area_trend_uses_every_dated_frame_and_returns_exact_targets() -> None:
    adapter = IrregularAreaTrendAdapter()
    frames = np.stack((_square(32, 3), _square(32, 4), _square(32, 5)))
    request = PredictionContext(
        horizon_steps=3,
        threshold=0.5,
        source_dates=(date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 20)),
        target_dates=(date(2020, 1, 21), date(2020, 1, 22), date(2020, 1, 23)),
    )

    output = adapter.predict(frames, (), request)

    assert output.masks.shape == (3, 32, 32)
    assert output.metadata["uses_all_input_frames"] is True
    assert output.metadata["observation_count"] == 3
    assert output.metadata["elapsed_days"] == [0.0, 9.0, 19.0]
    assert output.metadata["observed_area_pixels"] == [36, 64, 100]
    targets = [item["capped_target_pixels"] for item in output.metadata["targets"]]
    assert [int(mask.sum()) for mask in output.masks] == targets
    assert targets == sorted(targets)
    assert any("not a trained" in warning for warning in output.warnings)


def test_earlier_frames_change_prediction_even_when_latest_mask_is_identical() -> None:
    adapter = IrregularAreaTrendAdapter()
    latest = _square(48, 5)
    growing = np.stack((_square(48, 3), _square(48, 4), latest))
    shrinking = np.stack((_square(48, 7), _square(48, 6), latest))
    request = PredictionContext(
        horizon_steps=5,
        threshold=0.5,
        source_dates=(date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 20)),
        target_dates=tuple(date(2020, 1, day) for day in range(21, 26)),
    )

    growing_output = adapter.predict(growing, (), request)
    shrinking_output = adapter.predict(shrinking, (), request)

    assert growing_output.metadata["fitted_area_change_pct_per_day"] > 0
    assert shrinking_output.metadata["fitted_area_change_pct_per_day"] < 0
    assert growing_output.masks[-1].sum() > latest.sum()
    assert shrinking_output.masks[-1].sum() < latest.sum()


def test_trend_caps_sparse_extrapolation_and_rejects_empty_observation() -> None:
    adapter = IrregularAreaTrendAdapter()
    frames = np.stack((_square(24, 1), _square(24, 6)))
    request = PredictionContext(
        horizon_steps=1,
        threshold=0.5,
        source_dates=(date(2020, 1, 1), date(2020, 1, 2)),
        target_dates=(date(2020, 1, 12),),
        options={
            "max_daily_area_change_pct": 0.1,
            "max_total_area_change_pct": 50.0,
        },
    )

    output = adapter.predict(frames, (), request)
    latest_area = int(frames[-1].sum())
    assert output.metadata["targets"][0]["was_capped"] is True
    assert int(output.masks[0].sum()) == round(latest_area * 1.01)

    empty = frames.copy()
    empty[0] = 0
    with pytest.raises(ValueError, match="at least one water pixel"):
        adapter.predict(empty, (), request)


def test_target_dated_rainfall_changes_area_and_is_auditable() -> None:
    adapter = IrregularAreaTrendAdapter()
    frames = np.stack((_square(40, 4), _square(40, 4)))
    request = PredictionContext(
        horizon_steps=2,
        threshold=0.5,
        source_dates=(date(2020, 1, 1), date(2020, 1, 10)),
        target_dates=(date(2020, 1, 11), date(2020, 1, 12)),
        options={"rainfall_response_pct_per_20mm": 1.0},
    )
    weather = (
        {"date": "2020-01-11", "precipitation_mm": 20.0},
        {"date": "2020-01-12", "precipitation_mm": 0.0},
    )

    output = adapter.predict(frames, weather, request)

    assert output.metadata["matched_target_weather_rows"] == 2
    assert output.metadata["targets"][0]["weather_adjustment_pixels"] > 0
    assert output.metadata["targets"][1]["rainfall_memory_mm"] == pytest.approx(13.0)
    assert output.masks[0].sum() > frames[-1].sum()
