from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ui_next import learning
from ui_next.learning import (
    DETECTION_VS_LEVEL,
    FRAME_CONCEPTS,
    GLOSSARY,
    HANDOFF_CHECKLIST,
    INPUT_FIELDS,
    INPUT_FORMATS,
    LEARNING_TOPICS,
    MODEL_SWAP_CHECKLIST,
    PREPROCESS_STEPS,
    PRESENTATION_NARRATION,
    PROJECT_PIPELINE_NOTES,
    PROJECT_PIPELINE_STEPS,
    RUNTIME_MODELS,
    SUPPLIED_MODEL_MATERIAL,
    frame_timeline_svg,
    input_formats_table_html,
    item_grid_html,
    learning_card_html,
    model_cards_html,
    preprocess_table_html,
    process_storyboard_svg,
    project_pipeline_table_html,
    render_learning_help,
    render_learning_page,
)


def _all_copy() -> str:
    values: list[str] = []
    for collection in (
        FRAME_CONCEPTS,
        INPUT_FIELDS,
        DETECTION_VS_LEVEL,
        MODEL_SWAP_CHECKLIST,
        HANDOFF_CHECKLIST,
        GLOSSARY,
        PRESENTATION_NARRATION,
        PROJECT_PIPELINE_NOTES,
    ):
        for item in collection:
            values.extend((item.label, item.explanation))
    for spec in INPUT_FORMATS:
        values.extend(
            (spec.extension, spec.accepted_shape, spec.recommended_use, spec.caution)
        )
    for step in PREPROCESS_STEPS:
        values.extend(
            (
                step.title,
                step.input_value,
                step.action,
                step.output_value,
                step.quality_check,
            )
        )
    for step in PROJECT_PIPELINE_STEPS:
        values.extend(
            (
                step.title,
                step.input_value,
                step.action,
                step.output_value,
                step.quality_check,
            )
        )
    for model in (*RUNTIME_MODELS, *SUPPLIED_MODEL_MATERIAL):
        values.extend(
            (
                model.model_id,
                model.name,
                model.status,
                model.role,
                model.input_value,
                model.behavior,
                model.output_value,
                model.limitation,
            )
        )
    for topic in LEARNING_TOPICS.values():
        values.extend((topic.topic_id, topic.title, topic.short_description))
    return "\n".join(values)


def test_topics_cover_the_complete_beginner_story() -> None:
    assert tuple(LEARNING_TOPICS) == (
        "frame",
        "input",
        "preprocess",
        "detection",
        "models",
        "swap",
        "terms",
        "presentation",
    )
    assert all(topic.title and topic.short_description for topic in LEARNING_TOPICS.values())


def test_frame_is_defined_as_a_dated_mask_not_video_fps() -> None:
    copy = "\n".join(item.explanation for item in FRAME_CONCEPTS)
    assert "특정 관측 날짜" in copy
    assert "2차원 마스크 한 장" in copy
    assert "동영상의 초당 프레임이 아니라" in copy
    assert "입력 프레임" in "\n".join(item.label for item in FRAME_CONCEPTS)
    assert "horizon=3" in copy
    assert "[T,H,W]" in copy


def test_current_runtime_input_contract_is_exact_and_beginner_friendly() -> None:
    assert tuple(spec.extension for spec in INPUT_FORMATS) == (
        ".png",
        ".tif / .tiff / GeoTIFF",
        ".npy",
    )
    npy = INPUT_FORMATS[-1]
    assert "[H,W]" in npy.accepted_shape
    assert "[T,H,W]" in npy.accepted_shape
    assert "[H,W,1]" in npy.accepted_shape
    assert "[T,H,W,1]" in npy.accepted_shape
    copy = _all_copy()
    for required in (
        "source date",
        "target date",
        "water_level_m",
        "pixel_area_m2",
        "observed",
        "scenario",
        "YYYY-MM-DD",
        "엄격히 증가",
        "원본 위성영상",
        "2D 마스크",
    ):
        assert required in copy
    assert "PNG만으로 km²를 알 수 없습니다" in copy


def test_upstream_preprocessing_runs_from_inventory_to_forecast_qc() -> None:
    assert tuple(step.number for step in PREPROCESS_STEPS) == tuple(range(1, 8))
    assert tuple(step.title for step in PREPROCESS_STEPS) == (
        "인수·목록화",
        "센서별 보정",
        "동일 격자 정렬",
        "수체 감지",
        "마스크 정제·검수",
        "시계열 묶기",
        "예측·결과 검수",
    )
    assert all(
        step.input_value and step.action and step.output_value and step.quality_check
        for step in PREPROCESS_STEPS
    )
    copy = _all_copy()
    for required in (
        "1세부",
        "Sigma0",
        "정사보정",
        "CRS",
        "리샘플링",
        "이진 수체 마스크",
        "품질정보",
        "WATERCAST 업로드/API 입력 묶음",
    ):
        assert required in copy
    assert len(HANDOFF_CHECKLIST) >= 7


def test_delivery_deck_pipeline_is_explained_without_claiming_it_is_runtime() -> None:
    assert tuple(step.number for step in PROJECT_PIPELINE_STEPS) == (1, 2, 3, 4)
    assert tuple(step.title for step in PROJECT_PIPELINE_STEPS) == (
        "전처리 · preprocess.py",
        "수체 탐지 · detect_water.py",
        "수위·면적 산출 · calc_wlwa.py",
        "이종 센서 퓨전 보정 · Correct.py",
    )
    copy = _all_copy()
    for required in (
        "ICEYE(SAR)",
        "PlanetScope(광학)",
        "Processed_*.tif",
        "WB_*.tif",
        "0=비수체, 1=수체, 255=NoData",
        "DEM",
        "수계선",
        "WLWA_*.csv",
        "직전 60일 AWS",
        "CWLWA_*.csv",
        "absolute·relative·none",
    ):
        assert required in copy
    assert "현재 작업공간에서는 동일 이름의 실행 스크립트·학습 가중치를 확인하지 못했으므로" in copy


def test_project_pipeline_copy_separates_nodata_correction_and_forecast_roles() -> None:
    notes = "\n".join(item.explanation for item in PROJECT_PIPELINE_NOTES)
    assert "일반 PNG에서는 255를 흰색 수체" in notes
    assert "relative를 현장 절대 수위처럼 발표하면 안 됩니다" in notes
    assert "현재 수위 보정 입력이며 미래 기상 예보와 같은 뜻이 아닙니다" in _all_copy()
    assert "그 뒤 WATERCAST" in notes


def test_water_detection_area_level_and_forecast_are_not_conflated() -> None:
    labels = tuple(item.label for item in DETECTION_VS_LEVEL)
    assert labels == ("수체 감지", "수체 면적", "수위 추정", "시계열 예측")
    copy = "\n".join(item.explanation for item in DETECTION_VS_LEVEL)
    assert "어디까지 물인가" in copy
    assert "물 표면의 높이" in copy
    assert "마스크만으로 자동 확정되지 않으며" in copy
    assert "픽셀 수까지만" in copy


def test_model_copy_matches_registry_and_distinguishes_supplied_material() -> None:
    assert tuple(model.model_id for model in RUNTIME_MODELS) == (
        "persistence",
        "irregular-area-trend",
        "weather-morphology",
    )
    assert all("현재 백엔드에 등록" in model.status for model in RUNTIME_MODELS)

    supplied = {model.model_id: model for model in SUPPLIED_MODEL_MATERIAL}
    assert set(supplied) == {
        "tsx-unet-material",
        "convlstm-material",
        "external-adapter",
    }
    assert "현재 API 미등록" in supplied["tsx-unet-material"].status
    assert "체크포인트" in supplied["tsx-unet-material"].status
    assert "현재 API 미등록" in supplied["convlstm-material"].status
    assert ".weights.h5" in supplied["convlstm-material"].status
    assert "수위를 직접 반환하지 않음" in supplied["convlstm-material"].output_value

    weather = RUNTIME_MODELS[2]
    for actual_default in ("20mm", "0.1mm", "3일", "최대 3회"):
        assert actual_default in weather.behavior
    assert "데모 규칙" in weather.limitation
    assert "재난 확률이 아닙니다" in weather.limitation


def test_model_swap_contract_includes_adapter_validation_and_rollback() -> None:
    copy = "\n".join(
        f"{item.label} {item.explanation}" for item in MODEL_SWAP_CHECKLIST
    )
    for required in (
        "PredictionAdapter",
        "[horizon,H,W]",
        "PREDICTOR_PLUGINS",
        "plugin_errors",
        "persistence",
        "체크포인트 해시",
        "되돌리기",
    ):
        assert required in copy
    assert len(MODEL_SWAP_CHECKLIST) == 9


def test_copy_has_no_unverified_performance_claims() -> None:
    copy = _all_copy()
    assert "운영 성능으로 인용하면 안 됩니다" in copy
    assert "재현·교차검증" in copy
    assert "정확도가 생기는 것은 아닙니다" in copy
    for fabricated_or_unverified_metric in ("IoU 0.84", "R² 0.97", "36cm"):
        assert fabricated_or_unverified_metric not in copy


def test_glossary_and_presentation_are_substantial() -> None:
    assert len(GLOSSARY) >= 20
    assert len(PRESENTATION_NARRATION) == 6
    glossary_labels = {item.label for item in GLOSSARY}
    assert {"CRS", "NoData", "정사보정", "이진 마스크", "checkpoint", "adapter"} <= glossary_labels
    talk = "\n".join(item.explanation for item in PRESENTATION_NARRATION)
    assert "1세부" in talk
    assert "프레임" in talk
    assert "수체 감지" in talk and "수위" in talk
    assert "PredictionAdapter" in talk


def test_html_helpers_escape_dynamic_content_and_reuse_theme_classes() -> None:
    rendered = learning_card_html(
        '<script>alert("x")</script>',
        '<img src=x onerror="bad()">',
        kicker="<unsafe>",
        status='<svg onload="bad()">',
    )
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "<svg" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img" in rendered
    assert "&lt;svg" in rendered
    assert "wc-panel" in rendered
    assert "wc-learning-card" in rendered

    grid = item_grid_html((FRAME_CONCEPTS[0],))
    assert "wc-learning-grid" in grid
    assert "프레임(frame)" in grid


def test_tables_are_complete_and_use_dark_theme_classes() -> None:
    formats = input_formats_table_html()
    assert formats.count("<tbody>") == 1
    assert formats.count("<tr>") == len(INPUT_FORMATS) + 1
    assert "wc-learning-table" in formats
    assert ".png" in formats and "GeoTIFF" in formats and ".npy" in formats

    process = preprocess_table_html()
    assert process.count("<tr>") == len(PREPROCESS_STEPS) + 1
    assert "1. 인수·목록화" in process
    assert "7. 예측·결과 검수" in process

    project = project_pipeline_table_html()
    assert project.count("<tr>") == len(PROJECT_PIPELINE_STEPS) + 1
    assert "STEP 1" in project and "STEP 4" in project
    assert "Processed_*.tif" in project and "CWLWA_*.csv" in project
    assert "wc-learning-table" in project

    models = model_cards_html(RUNTIME_MODELS)
    assert "persistence" in models and "irregular-area-trend" in models
    assert "weather-morphology" in models
    assert "wc-learning-status" in models


def test_original_visuals_are_accessible_self_contained_and_unambiguous() -> None:
    storyboard = process_storyboard_svg()
    assert storyboard.startswith('<figure class="wc-panel wc-learning-visual">')
    assert 'role="img"' in storyboard
    assert "<title" in storyboard and "<desc" in storyboard
    assert storyboard.count("data-panel=") == 4
    assert "1세부 전달" in storyboard
    assert "개념 그림" in storyboard
    assert "href=" not in storyboard and "<script" not in storyboard

    timeline = frame_timeline_svg()
    assert 'role="img"' in timeline
    assert "입력 1" in timeline and "예측 3" in timeline
    assert "horizon = 3" in timeline
    assert "날짜는 설명용 예시" in timeline
    assert "href=" not in timeline and "<script" not in timeline


def test_renderer_api_is_simple_and_unknown_topics_fail_before_rendering() -> None:
    assert len(inspect.signature(render_learning_page).parameters) == 0
    signature = inspect.signature(render_learning_help)
    assert tuple(signature.parameters) == ("topic_id", "expanded")
    assert signature.parameters["expanded"].default is False
    with pytest.raises(KeyError, match="available: frame, input"):
        render_learning_help("does-not-exist")


def test_learning_page_visual_prefers_comic_and_has_svg_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class StreamlitStub:
        def __init__(self) -> None:
            self.images: list[tuple[str, str, str]] = []
            self.markdowns: list[tuple[str, bool]] = []

        def image(self, path: str, *, caption: str, width: str) -> None:
            self.images.append((path, caption, width))

        def markdown(self, body: str, *, unsafe_allow_html: bool) -> None:
            self.markdowns.append((body, unsafe_allow_html))

    stub = StreamlitStub()
    comic = tmp_path / "comic.png"
    comic.write_bytes(b"test image placeholder")
    monkeypatch.setattr(learning, "LEARNING_COMIC_PATH", comic)
    learning._render_process_visual(stub)
    assert stub.images == [
        (
            str(comic),
            "개념 설명용 자체 제작 4컷 · 1세부 원천자료가 전처리·수체 감지·시계열 예측으로 이어지는 흐름이며 실제 관측·예측 결과가 아닙니다.",
            "stretch",
        )
    ]
    assert stub.markdowns == []

    stub = StreamlitStub()
    monkeypatch.setattr(learning, "LEARNING_COMIC_PATH", tmp_path / "missing.png")
    learning._render_process_visual(stub)
    assert stub.images == []
    assert len(stub.markdowns) == 1
    assert "data-panel=" in stub.markdowns[0][0]
    assert stub.markdowns[0][1] is True
