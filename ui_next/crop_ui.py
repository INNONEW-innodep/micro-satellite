"""Standalone Streamlit page for the crop-candidate detection POC."""

from __future__ import annotations

import hashlib
import html
import json
from typing import Any

import streamlit as st
from crop_detection import (
    ESA_LICENSE_URL,
    CropDetectionResult,
    CropSample,
    DetectorMode,
    decode_crop_image,
    get_crop_sample,
    list_crop_samples,
    load_crop_sample,
    run_crop_detector,
)

CROP_CSS = """
<style>
.crop-hero { margin:4px 0 14px; padding:20px 22px; border-color:rgba(74,222,128,.34)!important;
  background:linear-gradient(120deg,rgba(20,83,45,.30),rgba(15,23,42,.90))!important; }
.crop-hero h2 { margin:4px 0 7px; color:var(--wc-text); }
.crop-hero p { margin:0; color:var(--wc-muted); line-height:1.65; max-width:980px; }
.crop-kicker { color:#86efac; font-size:.65rem; font-weight:850; letter-spacing:.14em; }
.crop-truth { margin:8px 0 14px; padding:12px 14px; border-left:3px solid #4ade80;
  background:rgba(20,83,45,.18); color:#bbf7d0; line-height:1.6; }
.crop-source { margin:8px 0 13px; padding:13px 15px; }
.crop-source b { color:#bbf7d0; }
.crop-source p { margin:4px 0 0; color:var(--wc-muted); line-height:1.6; }
.crop-result-title { margin:15px 0 8px; color:#dcfce7; }
.crop-swatch { display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:3px; }
</style>
"""

MODE_LABELS: dict[DetectorMode, str] = {
    "false_color_red": "False-color · 붉은 식생 신호",
    "natural_rgb_green": "Natural RGB · 녹색 식생 신호",
}


def render_crop_detection_page() -> None:
    st.markdown(CROP_CSS, unsafe_allow_html=True)
    st.markdown(
        '<section class="wc-panel crop-hero">'
        '<div class="crop-kicker">STANDALONE CROP DETECTION POC</div>'
        '<h2>작물 재배지 후보 탐지</h2>'
        '<p>공개 위성 샘플 또는 사용자가 올린 RGB 이미지를 대상으로 작물로 보이는 식생 '
        '후보 픽셀을 찾고, 이진 마스크와 오버레이 결과까지 보여주는 독립 화면입니다.</p>'
        "</section>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="crop-truth"><b>현재 상태 · RULE-BASED VEGETATION DEMO</b><br/>'
        "학습된 작물 Segmentation 모델이 아니라 색상지수 기반 POC입니다. 작물 종류 분류, "
        "정확도 평가, 실제 필지 면적 산출은 하지 않습니다.</div>",
        unsafe_allow_html=True,
    )
    _render_crop_help()

    input_source = st.radio(
        "입력 방식",
        ("ESA 공개 샘플", "내 이미지 업로드"),
        horizontal=True,
        key="crop_input_source",
    )

    sample: CropSample | None = None
    source_name: str
    source_key: str
    if input_source == "ESA 공개 샘플":
        sample_ids = [item.sample_id for item in list_crop_samples()]
        selected_id = st.selectbox(
            "작물 탐지 샘플",
            sample_ids,
            format_func=lambda sample_id: get_crop_sample(sample_id).display_name,
            key="crop_sample_id",
        )
        sample = get_crop_sample(selected_id)
        image = load_crop_sample(selected_id)
        source_name = sample.display_name
        source_key = f"sample:{sample.sample_id}"
        default_mode = sample.detector_mode
        _render_sample_source(sample)
    else:
        uploaded = st.file_uploader(
            "RGB 이미지",
            type=("png", "jpg", "jpeg"),
            key="crop_uploaded_image",
            help="GeoTIFF와 다중분광 밴드는 아직 이 UI POC의 입력이 아닙니다. PNG/JPEG RGB만 받습니다.",
        )
        if uploaded is None:
            st.info("PNG 또는 JPEG 항공·위성 이미지를 선택하세요. 바로 보려면 ESA 공개 샘플을 사용하세요.")
            return
        content = uploaded.getvalue()
        try:
            image = decode_crop_image(content)
        except ValueError as exc:
            st.error(str(exc))
            return
        source_name = uploaded.name
        source_key = f"upload:{hashlib.sha256(content).hexdigest()}"
        default_mode = "natural_rgb_green"
        st.warning(
            "사용자 업로드는 출처·촬영일·좌표계·공간 해상도를 화면이 검증하지 않습니다. "
            "자연색 RGB에서는 녹색 식생 후보가 풀·나무까지 포함할 수 있습니다."
        )

    setting_columns = st.columns(3)
    mode_options: tuple[DetectorMode, ...] = (
        "false_color_red",
        "natural_rgb_green",
    )
    mode = setting_columns[0].selectbox(
        "탐지 영상 유형",
        mode_options,
        index=mode_options.index(default_mode),
        format_func=lambda value: MODE_LABELS[value],
        key=f"crop_detector_mode_{source_key[:24]}",
        help="ESA 내장 false-color 샘플은 붉은 식생, 일반 사진은 녹색 식생 모드를 사용합니다.",
    )
    threshold = setting_columns[1].slider(
        "후보 판정 임계값",
        min_value=0.15,
        max_value=0.75,
        value=0.38,
        step=0.01,
        key="crop_threshold",
        help="낮추면 후보가 넓어지고 오탐이 늘 수 있으며, 높이면 강한 색상 신호만 남습니다.",
    )
    overlay_alpha = setting_columns[2].slider(
        "오버레이 불투명도",
        min_value=0.10,
        max_value=0.80,
        value=0.48,
        step=0.02,
        key="crop_overlay_alpha",
    )

    settings_signature = hashlib.sha256(
        f"{source_key}|{mode}|{threshold:.3f}|{overlay_alpha:.3f}".encode()
    ).hexdigest()
    force_run = st.button(
        "작물 후보 탐지 실행",
        type="primary",
        icon=":material/agriculture:",
        width="stretch",
    )
    result = st.session_state.get("crop_detection_result")
    cached_signature = st.session_state.get("crop_detection_signature")
    if force_run or not isinstance(result, CropDetectionResult) or cached_signature != settings_signature:
        with st.spinner("색상지수 계산 → 잡음 정리 → 마스크·오버레이 생성 중입니다..."):
            result = run_crop_detector(
                image,
                detector_mode=mode,
                threshold=float(threshold),
                overlay_alpha=float(overlay_alpha),
            )
        st.session_state.crop_detection_result = result
        st.session_state.crop_detection_signature = settings_signature
        st.session_state.crop_detection_source_name = source_name

    _render_crop_result(result, source_name=source_name, sample=sample)


def _render_sample_source(sample: CropSample) -> None:
    st.markdown(
        '<div class="wc-panel crop-source">'
        f"<b>{_escape(sample.display_name)}</b> · {_escape(sample.credit)}"
        f"<p>{_escape(sample.summary_ko)} {_escape(sample.imagery_description_ko)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    source_columns = st.columns([1, 1, 2])
    source_columns[0].link_button(
        "ESA 원천 페이지",
        sample.source_url,
        icon=":material/open_in_new:",
        width="stretch",
    )
    source_columns[1].link_button(
        "라이선스",
        sample.license_url,
        icon=":material/license:",
        width="stretch",
    )
    source_columns[2].caption(
        f"{sample.license_name} · ESA 공개 장면 한 장에서 파생된 보기용 ROI · 생성형 샘플 아님"
    )


def _render_crop_help() -> None:
    with st.expander("설명 보기 · 작물 탐지는 무엇을 하는 화면인가요?", expanded=False):
        st.markdown(
            """
1. **입력**: 작물이 보이는 RGB/false-color 항공·위성 이미지 한 장을 받습니다.
2. **색상지수**: false-color에서는 붉은 신호, 자연색에서는 녹색 신호가 주변 색보다 얼마나 강한지 0~1 점수로 계산합니다.
3. **이진화**: 임계값 이상을 `작물/식생 후보`, 미만을 `그 외`로 나눕니다.
4. **잡음 정리**: 주변 픽셀과 연결되지 않은 작은 점을 줄이고 마스크 경계를 정리합니다.
5. **결과**: 원본, 후보 점수 지도, 이진 마스크, 원본 위 오버레이를 표시합니다.

여기서 **탐지**는 “작물일 가능성이 있어 보이는 위치”를 픽셀로 표시한다는 뜻입니다.
현재 방식은 밭과 숲·잔디를 구분하는 학습 모델이 아니며, 확률이나 정확도를 만들지 않습니다.
실제 작물 Segmentation으로 교체할 때는 RGB/다중분광 입력, 정답 라벨, 학습 가중치와
독립 검증셋이 필요합니다.
"""
        )


def _render_crop_result(
    result: CropDetectionResult,
    *,
    source_name: str,
    sample: CropSample | None,
) -> None:
    st.markdown("### 탐지 결과", help="설정이 바뀌면 동일한 결정론적 탐지기를 다시 실행합니다.")
    metrics = st.columns(5)
    metrics[0].metric("처리 해상도", f"{result.width}×{result.height}")
    metrics[1].metric("후보 픽셀", f"{result.candidate_pixels:,}")
    metrics[2].metric("전체 픽셀", f"{result.valid_pixels:,}")
    metrics[3].metric("후보 비율", f"{result.candidate_ratio_pct:.2f}%")
    metrics[4].metric("실제 면적", "미산출", help="GeoTIFF 지리참조와 픽셀 면적이 없어 계산하지 않습니다.")
    st.caption(
        "후보 비율은 이 이미지 안에서 색상 규칙을 통과한 픽셀 비율입니다. 작물 정확도·신뢰도·점유율이 아닙니다."
    )

    overlay_tab, mask_tab, export_tab, provenance_tab = st.tabs(
        ["원본 · 오버레이", "점수 · 이진 마스크", "수치 · 내보내기", "출처 · 한계"]
    )
    with overlay_tab:
        image_columns = st.columns(2)
        image_columns[0].image(
            result.source_png,
            caption=f"원본 · {source_name}",
            width="stretch",
        )
        image_columns[1].image(
            result.overlay_png,
            caption="탐지 오버레이 · 초록=후보, 노랑=경계",
            width="stretch",
        )
    with mask_tab:
        mask_columns = st.columns(2)
        mask_columns[0].image(
            result.score_png,
            caption="색상지수 점수 · 밝을수록 후보 규칙에 가까움 (확률 아님)",
            width="stretch",
        )
        mask_columns[1].image(
            result.mask_png,
            caption="이진 마스크 · 초록=작물/식생 후보, 남색=그 외",
            width="stretch",
        )
    with export_tab:
        summary = result.summary()
        summary.update(
            {
                "source_name": source_name,
                "sample_id": sample.sample_id if sample else None,
                "source_url": sample.source_url if sample else None,
                "credit": sample.credit if sample else "user_upload",
                "license": sample.license_name if sample else "user_supplied",
            }
        )
        st.json(summary)
        downloads = st.columns(3)
        downloads[0].download_button(
            "오버레이 PNG",
            result.overlay_png,
            file_name="crop_candidate_overlay.png",
            mime="image/png",
            width="stretch",
        )
        downloads[1].download_button(
            "마스크 PNG",
            result.mask_png,
            file_name="crop_candidate_mask.png",
            mime="image/png",
            width="stretch",
        )
        downloads[2].download_button(
            "요약 JSON",
            json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="crop_detection_summary.json",
            mime="application/json",
            width="stretch",
        )
    with provenance_tab:
        if sample is not None:
            st.markdown(f"**원천** · [{sample.source_title}]({sample.source_url})")
            st.markdown(f"**크레디트** · `{sample.credit}`")
            st.markdown(
                f"**라이선스** · [{sample.license_name}]({ESA_LICENSE_URL})"
            )
            st.success("허용 용도 · " + sample.intended_use_ko)
            st.warning("해석 한계 · " + sample.limitation_ko)
        else:
            st.warning(
                "사용자 업로드이므로 원천·라이선스·촬영 조건·정답 라벨을 화면이 알 수 없습니다. "
                "결과를 외부에 사용할 때 업로드 제공자가 계보를 별도로 기록해야 합니다."
            )
        st.info(
            "교체 지점 · 향후 검증된 segmentation 모델이 생기면 현재 색상지수 계산 부분을 "
            "모델 추론으로 바꾸고, UI의 원본/마스크/오버레이/내보내기 구조는 그대로 사용할 수 있습니다."
        )


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


__all__ = [
    "CROP_CSS",
    "MODE_LABELS",
    "render_crop_detection_page",
]
