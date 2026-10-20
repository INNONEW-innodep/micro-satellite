"""화면 공통 한국어 라벨 — 백엔드 모델 id/영문 이름을 사람이 읽는 이름으로."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AREA_MODEL_KO: dict[str, str] = {
    "irregular-area-trend": "수체 면적 추세 기준선 (불규칙 관측 + 강수 보정)",
    "weather-morphology": "강수 형태학 기준선 (비 오면 팽창·마르면 수축)",
    "persistence": "직전 마스크 유지 (기준선)",
    "gauge-level-forecast": "수위계 학습 모델 수위 예측 (지점 시계열 모델)",
}


def area_model_label(model: Mapping[str, Any] | str) -> str:
    """면적 예측 모델(adapter) 라벨. 등록된 한국어 이름이 없으면 백엔드 이름을 그대로."""
    if isinstance(model, str):
        return AREA_MODEL_KO.get(model, model)
    model_id = str(model.get("id") or "")
    return AREA_MODEL_KO.get(model_id) or str(model.get("name_ko") or model.get("name") or model_id)
