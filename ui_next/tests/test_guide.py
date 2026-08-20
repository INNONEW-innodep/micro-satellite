from __future__ import annotations

import inspect

from ui_next.guide import (
    API_ENDPOINTS,
    FAQ_ITEMS,
    GLOSSARY,
    GUIDE_LIMITATIONS,
    INPUT_CONCEPTS,
    MENU_GUIDES,
    MODEL_CONCEPTS,
    PAGE_HELP,
    PRESENTATION_SCRIPT,
    RESULT_CONCEPTS,
    WORKFLOW_STEPS,
    flow_diagram_html,
    guide_card_html,
    item_list_html,
    render_guide_page,
    render_page_help,
)


EXPECTED_MENUS = ("데이터", "기상", "예측 실행", "결과", "API 가이드")


def _all_copy() -> str:
    values: list[str] = []
    for step in WORKFLOW_STEPS:
        values.extend((step.menu, step.title, step.action, step.check))
    for menu in MENU_GUIDES:
        values.extend((menu.menu, menu.purpose, menu.when_to_use, *menu.reading))
        for item in menu.controls:
            values.extend((item.label, item.explanation))
    for collection in (INPUT_CONCEPTS, MODEL_CONCEPTS, RESULT_CONCEPTS, API_ENDPOINTS, PRESENTATION_SCRIPT):
        for item in collection:
            values.extend((item.label, item.explanation))
    for item in FAQ_ITEMS:
        values.extend((item.question, item.answer))
    for item in GLOSSARY:
        values.extend((item.term, item.definition))
    values.extend(GUIDE_LIMITATIONS)
    return "\n".join(values)


def test_workflow_and_every_top_menu_are_covered_once() -> None:
    assert len(WORKFLOW_STEPS) == 5
    assert tuple(step.number for step in WORKFLOW_STEPS) == (1, 2, 3, 4, 5)
    assert tuple(step.menu for step in WORKFLOW_STEPS) == EXPECTED_MENUS
    assert tuple(menu.menu for menu in MENU_GUIDES) == EXPECTED_MENUS
    assert all(menu.controls and menu.reading for menu in MENU_GUIDES)


def test_required_beginner_terms_and_result_fields_are_present() -> None:
    copy = _all_copy().lower()
    required = (
        "mask",
        "date",
        "pixel_area_m2",
        "water_level_m",
        "observed",
        "scenario",
        "asos",
        "d-1",
        "kma_api_key",
        "adapter",
        "persistence",
        "weather-morphology",
        "학습 모델",
        "horizon",
        "threshold",
        "risk",
        "area_pixels",
        "area_km2",
        "change",
        "level",
        "null",
        "artifact",
        "openapi",
        "prediction id",
    )
    assert all(term in copy for term in required)


def test_copy_explicitly_rejects_overclaiming_baselines_and_samples() -> None:
    copy = _all_copy()
    assert "학습 모델이 아니라" in copy
    assert "실측·성능 검증 자료가 아닙니다" in copy
    assert "미래 예보" in copy
    assert "발생 확률이 아닙니다" in copy
    assert "면적·수위 정확도 검증" in copy
    assert any("기준선" in item for item in GUIDE_LIMITATIONS)
    assert any("샘플" in item and "실측" in item for item in GUIDE_LIMITATIONS)


def test_api_external_integration_covers_result_and_artifact_retrieval() -> None:
    labels = {item.label for item in API_ENDPOINTS}
    assert "GET /api/v1/health" in labels
    assert "GET /api/v1/models" in labels
    assert "POST /api/v1/predictions" in labels
    assert "GET /api/v1/predictions/{id}" in labels
    assert "GET /api/v1/predictions/{id}/files/{name}" in labels
    assert "GET /api/v1/predictions/{id}/bundle" in labels


def test_presentation_faq_and_glossary_are_substantial() -> None:
    assert len(PRESENTATION_SCRIPT) == 5
    assert len(FAQ_ITEMS) >= 8
    assert len(GLOSSARY) >= 18
    assert any("발표" not in item.label and "WATERCAST" in item.explanation for item in PRESENTATION_SCRIPT)


def test_html_helpers_escape_content_and_use_theme_classes() -> None:
    rendered = guide_card_html(
        '<script>alert("x")</script>',
        '<img src=x onerror="bad()">',
        kicker="<unsafe>",
        extra_class='wc-guide-script onclick="bad()"',
    )
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img" in rendered
    assert "&lt;unsafe&gt;" in rendered
    assert "onclick=" not in rendered
    assert "wc-panel" in rendered
    assert "wc-guide-card" in rendered

    listed = item_list_html((INPUT_CONCEPTS[0],))
    assert "wc-guide-list" in listed
    assert "<b>mask</b>" in listed


def test_flow_svg_is_self_contained_accessible_and_has_five_steps() -> None:
    rendered = flow_diagram_html()
    assert rendered.startswith('<div class="wc-panel wc-guide-flow">')
    assert "<svg" in rendered and "</svg>" in rendered
    assert 'role="img"' in rendered
    assert "aria-label=" in rendered
    assert rendered.count("data-step=") == 5
    for menu in EXPECTED_MENUS:
        assert menu in rendered


def test_public_renderer_has_no_required_arguments() -> None:
    assert callable(render_guide_page)
    assert len(inspect.signature(render_guide_page).parameters) == 0


def test_page_local_help_covers_each_route_and_has_presentation_copy() -> None:
    assert tuple(PAGE_HELP) == ("data", "weather", "prediction", "result", "api")
    assert tuple(item.menu for item in PAGE_HELP.values()) == EXPECTED_MENUS
    assert all(len(item.use_order) >= 3 for item in PAGE_HELP.values())
    assert all(item.presentation_line for item in PAGE_HELP.values())
    signature = inspect.signature(render_page_help)
    assert tuple(signature.parameters) == ("page_id", "expanded")
    assert signature.parameters["expanded"].default is False
