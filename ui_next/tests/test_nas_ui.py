from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from ui_next.nas_catalog import (
    NAS_CATALOG_SUMMARY,
    get_nas_dataset_group,
    list_nas_dataset_groups,
)
from ui_next.nas_ui import (
    binary_mask_metrics,
    build_observation_interval_figure,
    cyan_overlay_png,
    filter_nas_groups,
    format_binary_size,
    representative_asset,
    role_labels,
)
from ui_next.samples import get_sample

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_catalog_filters_and_summary_labels_preserve_source_groups() -> None:
    groups = list_nas_dataset_groups()

    assert len(groups) == NAS_CATALOG_SUMMARY.group_count == 8
    assert format_binary_size(NAS_CATALOG_SUMMARY.total_bytes) == "53.3 GiB"
    assert len(filter_nas_groups(groups, readiness="demo_ready")) == 2
    assert tuple(
        group.group_id for group in filter_nas_groups(groups, modality="sar")
    ) == ("iceye-water-labels", "iceye-raw", "university-single-date")
    assert tuple(
        group.group_id
        for group in filter_nas_groups(
            groups, readiness="preprocess_required", modality="optical"
        )
    ) == ("planetscope-raw", "giheung-optical", "hoedong-optical")

    with pytest.raises(ValueError, match="unknown readiness"):
        filter_nas_groups(groups, readiness="ready-ish")
    with pytest.raises(ValueError, match="unknown modality"):
        filter_nas_groups(groups, modality="thermal")


def test_roles_and_interval_plot_explain_quality_masks_and_irregular_dates() -> None:
    planetscope = get_nas_dataset_group("planetscope-raw")
    assert "UDM2 품질 마스크" in role_labels(planetscope)
    assert "수체 이진 라벨" not in role_labels(planetscope)

    busan = get_nas_dataset_group("busan-water-labels")
    figure = build_observation_interval_figure(busan)
    assert figure is not None
    assert tuple(figure.data[0].y) == (23, 13, 20)
    assert "프레임 수와 일수" in figure.layout.title.text
    assert build_observation_interval_figure(
        get_nas_dataset_group("university-single-date")
    ) is None


def test_cyan_overlay_uses_the_exact_context_and_backend_mask_pair() -> None:
    frame = get_sample("busan-nas-water-labels").frames[0]
    assert frame.context_png_bytes is not None

    payload = cyan_overlay_png(frame.context_png_bytes, frame.png_bytes)
    with Image.open(io.BytesIO(frame.context_png_bytes)) as source:
        context = source.convert("RGB")
    with Image.open(io.BytesIO(frame.png_bytes)) as source:
        mask = source.convert("L")
    with Image.open(io.BytesIO(payload)) as source:
        overlay = source.convert("RGB")

    assert overlay.size == context.size == mask.size == (512, 512)
    mask_values = mask.tobytes()
    context_values = context.tobytes()
    overlay_values = overlay.tobytes()
    dry_pixel = next(index for index, value in enumerate(mask_values) if value == 0)
    water_pixel = next(index for index, value in enumerate(mask_values) if value == 255)
    dry_slice = slice(dry_pixel * 3, dry_pixel * 3 + 3)
    water_slice = slice(water_pixel * 3, water_pixel * 3 + 3)
    assert overlay_values[dry_slice] == context_values[dry_slice]
    assert overlay_values[water_slice] != context_values[water_slice]
    assert payload == cyan_overlay_png(frame.context_png_bytes, frame.png_bytes)

    other = io.BytesIO()
    Image.new("L", (8, 8), color=255).save(other, format="PNG")
    with pytest.raises(ValueError, match="share one size"):
        cyan_overlay_png(frame.context_png_bytes, other.getvalue())


def test_persistence_holdout_is_computed_from_third_and_fourth_nas_masks() -> None:
    sample = get_sample("busan-nas-water-labels")
    metrics = binary_mask_metrics(
        sample.frames[2].png_bytes,
        sample.frames[3].png_bytes,
    )

    assert metrics.true_positive == 9549
    assert metrics.false_positive == 845
    assert metrics.false_negative == 643
    assert metrics.true_negative == 251107
    assert metrics.iou == pytest.approx(0.8651808)
    assert metrics.dice == pytest.approx(0.9277179)
    assert metrics.precision == pytest.approx(0.9187031)
    assert metrics.recall == pytest.approx(0.9369113)


@pytest.mark.parametrize(
    "group_id",
    (
        "iceye-raw",
        "planetscope-raw",
        "university-single-date",
        "giheung-optical",
        "hoedong-optical",
    ),
)
def test_display_only_representative_assets_exist(group_id: str) -> None:
    representative = representative_asset(group_id)
    assert representative is not None
    path, caption = representative
    assert path.is_file()
    assert path.stat().st_size > 0
    assert "아닙니다" in caption or "않습니다" in caption


def test_nas_input_mode_renders_materialized_busan_demo_without_api_call() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["service_mode"] = "수체 시계열 예측"
    app.session_state["phase"] = 0
    app.session_state["input_mode"] = "NAS 전달자료"
    app.session_state["connection_attempted"] = True
    app.session_state["connection_status"] = {"status": "error"}

    app.run(timeout=30)

    assert not app.exception
    assert any(
        radio.label == "입력 방식" and radio.value == "NAS 전달자료"
        for radio in app.radio
    )
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["NAS 파일"] == "300개"
    assert metrics["총 용량"] == "53.3 GiB"
    assert metrics["분류 그룹"] == "8개"
    assert metrics["직접 후보"] == "2개"
    assert metrics["로컬 시연셋"] == "2개"
    assert metrics["IoU"] == "0.865"
    assert metrics["Dice"] == "0.928"
    assert any(item.label == "예측 준비 상태" for item in app.selectbox)
    assert any(item.label == "센서 유형" for item in app.selectbox)
    assert any(item.label == "데이터 그룹" for item in app.selectbox)

    buttons = {button.label: button for button in app.button}
    assert buttons["데이터 불러오기"].disabled is False
    assert buttons["EDA 데이터로 바로 예측"].disabled is False
    assert len(app.get("plotly_chart")) >= 2
    assert any("실제 NAS 입력" in item.value for item in app.markdown)
