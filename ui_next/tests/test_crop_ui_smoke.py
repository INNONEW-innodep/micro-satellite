from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_crop_workspace_renders_results_without_touching_water_state() -> None:
    water_input_state = [{"name": "existing-water-frame"}]
    app = AppTest.from_file(str(APP_PATH), default_timeout=15)
    app.session_state["service_mode"] = "작물 탐지"
    app.session_state["input_bundle"] = water_input_state

    app.run(timeout=15)

    assert not app.exception
    assert app.radio[0].label == "분석 업무"
    assert app.radio[0].value == "작물 탐지"
    assert [metric.label for metric in app.metric] == [
        "처리 해상도",
        "후보 픽셀",
        "전체 픽셀",
        "후보 비율",
        "실제 면적",
    ]
    assert app.metric[0].value == "1200×511"
    assert app.metric[4].value == "미산출"
    assert len(app.get("image")) == 4
    assert [tab.label for tab in app.tabs] == [
        "원본 · 오버레이",
        "점수 · 이진 마스크",
        "수치 · 내보내기",
        "출처 · 한계",
    ]
    assert app.session_state["input_bundle"] == water_input_state

    app.selectbox[0].set_value("esa_saudi_west").run(timeout=15)
    assert not app.exception
    assert app.metric[0].value == "750×639"
    assert app.metric[3].value == "18.26%"
    assert app.session_state["input_bundle"] == water_input_state
