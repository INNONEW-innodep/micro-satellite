from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui_next.state import phase_index

from ui_next.model_eval import GAUGE_CSV_PATH, WEATHER_CSV_PATH

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.skipif(
    not (GAUGE_CSV_PATH.is_file() and WEATHER_CSV_PATH.is_file()),
    reason="handover evidence package not present",
)
def test_model_eval_phase_renders_all_three_panels_with_series_labels() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20)
    app.session_state["phase"] = phase_index("정량 평가")

    app.run(timeout=20)

    assert not app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert "① 수체 분할 정량 평가" in tab_labels
    assert "② 게이지 실측 수위" in tab_labels
    assert "③ 기상 근거 데이터" in tab_labels

    metric_labels = [metric.label for metric in app.metric]
    assert "water IoU" in metric_labels  # 스모크 재현 지표
    assert "실측 표본" in metric_labels  # 게이지 in-sample 패널
    assert "관측 일수" in metric_labels  # 기상 근거 데이터

    rendered_markdown = " ".join(str(item.value) for item in app.markdown)
    assert "수치 계열 규율" in rendered_markdown
    assert "IN-SAMPLE" in rendered_markdown
