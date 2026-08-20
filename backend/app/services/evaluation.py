"""Water-level evaluation that never feeds reference values into prediction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np

from ..schemas import WaterLevelEvaluation, WaterLevelEvaluationPair


def evaluate_water_levels(
    predicted: Sequence[float | None],
    reference: Sequence[float | None],
    target_dates: Sequence[date],
    *,
    kind: str,
    truth_provenance: str,
) -> WaterLevelEvaluation | None:
    """Calculate metrics on finite aligned pairs, or ``None`` without truth."""

    if not reference:
        return None

    pairs: list[WaterLevelEvaluationPair] = []
    for index, (predicted_value, reference_value) in enumerate(
        zip(predicted, reference, strict=True), start=1
    ):
        if predicted_value is None or reference_value is None:
            continue
        predicted_float = float(predicted_value)
        reference_float = float(reference_value)
        residual = predicted_float - reference_float
        pairs.append(
            WaterLevelEvaluationPair(
                horizon=index,
                target_date=(
                    target_dates[index - 1] if index <= len(target_dates) else None
                ),
                predicted_water_level_m=predicted_float,
                reference_water_level_m=reference_float,
                residual_m=residual,
                absolute_error_m=abs(residual),
                percentage_error_pct=(
                    abs(residual / reference_float) * 100.0
                    if not math.isclose(reference_float, 0.0, abs_tol=1e-12)
                    else None
                ),
            )
        )

    label = _evaluation_label(kind)
    warning = (
        "Synthetic demonstration truth; these metrics are not measured model performance."
        if kind == "synthetic_demo"
        else None
    )
    if not pairs:
        return WaterLevelEvaluation(
            status="unavailable",
            kind=kind,
            label=label,
            truth_provenance=truth_provenance,
            sample_count=0,
            unavailable_reason=(
                "No horizon has both a predicted and reference water level."
            ),
            warning=warning,
        )

    predicted_values = np.asarray(
        [pair.predicted_water_level_m for pair in pairs], dtype=np.float64
    )
    reference_values = np.asarray(
        [pair.reference_water_level_m for pair in pairs], dtype=np.float64
    )
    residuals = predicted_values - reference_values
    absolute = np.abs(residuals)
    percentage_values = [
        pair.percentage_error_pct
        for pair in pairs
        if pair.percentage_error_pct is not None
    ]
    total_variance = float(
        np.sum((reference_values - float(reference_values.mean())) ** 2)
    )
    r2 = (
        1.0 - float(np.sum(residuals**2)) / total_variance
        if len(pairs) >= 2 and total_variance > 0.0
        else None
    )
    return WaterLevelEvaluation(
        status="available",
        kind=kind,
        label=label,
        truth_provenance=truth_provenance,
        sample_count=len(pairs),
        mae_m=float(absolute.mean()),
        rmse_m=float(np.sqrt(np.mean(residuals**2))),
        mape_pct=(
            float(np.mean(percentage_values)) if percentage_values else None
        ),
        r2=r2,
        bias_m=float(residuals.mean()),
        pairs=pairs,
        warning=warning,
    )


def _evaluation_label(kind: str) -> str:
    return {
        "synthetic_demo": "합성 시연 백테스트",
        "holdout": "홀드아웃 백테스트",
        "measured": "실측 사후 평가",
        "user_supplied": "사용자 정답 평가",
    }.get(kind, kind)


__all__ = ["evaluate_water_levels"]
