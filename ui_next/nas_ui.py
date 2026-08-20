"""Streamlit renderer for the classified NAS delivery-data catalog.

The catalog and the executable sample deliberately remain separate: every NAS
group can be inspected, but only a group with a materialized, backend-ready
binary-mask sequence can activate the existing prediction callbacks.
"""

from __future__ import annotations

import html
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:  # package imports used by pytest and other Python callers
    from .api_client import APIError
    from .eda import render_input_eda
    from .nas_catalog import (
        NAS_CATALOG_SUMMARY,
        NasDatasetGroup,
        get_nas_dataset_group,
        list_nas_dataset_groups,
    )
    from .nas_eda import (
        build_busan_resampling_error_figure,
        build_busan_source_area_figure,
        build_inventory_size_figure,
        build_inventory_timeline_figure,
        build_readiness_matrix_figure,
        busan_source_rows,
        evaluate_busan_area_holdout,
        holdout_rows,
        inventory_group_rows,
        readiness_rows,
        summarize_busan_source_eda,
        summarize_inventory,
        summarize_readiness,
    )
    from .model_eval_ui import render_gauge_levels_panel
    from .nas_source_eda import available_source_group_ids, build_nas_source_eda
    from .samples import SampleDataset, get_sample
except ImportError:  # direct Streamlit execution adds ui_next/ to sys.path
    from api_client import APIError
    from eda import render_input_eda
    from nas_catalog import (
        NAS_CATALOG_SUMMARY,
        NasDatasetGroup,
        get_nas_dataset_group,
        list_nas_dataset_groups,
    )
    from nas_eda import (
        build_busan_resampling_error_figure,
        build_busan_source_area_figure,
        build_inventory_size_figure,
        build_inventory_timeline_figure,
        build_readiness_matrix_figure,
        busan_source_rows,
        evaluate_busan_area_holdout,
        holdout_rows,
        inventory_group_rows,
        readiness_rows,
        summarize_busan_source_eda,
        summarize_inventory,
        summarize_readiness,
    )
    from model_eval_ui import render_gauge_levels_panel
    from nas_source_eda import available_source_group_ids, build_nas_source_eda
    from samples import SampleDataset, get_sample


ALL_FILTER = "all"
REPRESENTATIVE_ASSET_DIR = (
    Path(__file__).resolve().parent / "assets" / "nas" / "representatives"
)
ICEYE_DEMO_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "nas" / "iceye_2020"

REPRESENTATIVE_ASSETS = {
    "iceye-raw": (
        "iceye_sar_quicklook_20200302.jpg",
        "ICEYE SAR quicklook · 화면 탐색용이며 GRD/SLC 분석 입력을 대신하지 않습니다.",
    ),
    "planetscope-raw": (
        "planetscope_udm2_20200218.png",
        "PlanetScope UDM2 품질 마스크 · 색은 품질 범주이며 수체 라벨이 아닙니다.",
    ),
    "university-single-date": (
        "university_pred_wb_20210411.png",
        "시립대 Pred_WB 파생 수체 · 단일 날짜 참고 결과로 시계열 입력이 아닙니다.",
    ),
    "giheung-optical": (
        "giheung_browse_20230730.jpg",
        "기흥호수 상용 광학 browse · 표시 전용이며 원본 PAN/MUL 분석값이 아닙니다.",
    ),
    "hoedong-optical": (
        "hoedong_browse_20240823.jpg",
        "회동저수지 상용 광학 browse · 표시 전용이며 수체 마스크가 아닙니다.",
    ),
}

READINESS_LABELS = {
    ALL_FILTER: "전체 준비 상태",
    "demo_ready": "직접 예측 후보",
    "preprocess_required": "전처리 필요",
    "single_date_reference": "단일 시점 참고",
    "support_only": "보조·보관 자료",
}

MODALITY_LABELS = {
    ALL_FILTER: "전체 센서 유형",
    "optical": "광학",
    "sar": "SAR 레이더",
    "none": "센서 없음",
}

ROLE_LABELS = {
    "segmentation_input": "수체 탐지 입력영상",
    "water_mask_label": "수체 이진 라벨",
    "water_mask_timeseries": "수체 마스크 시계열",
    "sar_grd": "SAR GRD",
    "sar_slc": "SAR SLC",
    "quicklook": "Quicklook 미리보기",
    "metadata": "메타데이터",
    "surface_reflectance_imagery": "지표반사도 영상",
    "quality_mask_udm2": "UDM2 품질 마스크",
    "scene_metadata": "장면 메타데이터",
    "sar_source_imagery": "SAR 원천영상",
    "dem": "DEM 고도",
    "derived_water_body": "파생 수체 결과",
    "derived_water_level": "파생 수위 결과",
    "geojson": "공간 벡터",
    "presentation": "발표자료",
    "panchromatic_imagery": "PAN 전정색 영상",
    "multispectral_imagery": "MUL 다중분광 영상",
    "browse_preview": "Browse 미리보기",
    "footprint": "촬영영역",
    "delivery_archive": "전달 ZIP",
    "duplicate_candidate": "중복 가능 보관본",
    "support_material": "보조자료",
}

NAS_UI_CSS = """
<style>
.nas-hero { margin:4px 0 14px; padding:19px 21px; border-color:rgba(34,211,238,.30)!important;
  background:linear-gradient(120deg,rgba(8,47,73,.34),rgba(15,23,42,.92))!important; }
.nas-hero h3 { margin:4px 0 7px!important; color:#ecfeff!important; }
.nas-hero p { margin:0; color:#a5b4c7; line-height:1.65; max-width:1040px; }
.nas-kicker { color:#67e8f9; font-size:.66rem; font-weight:850; letter-spacing:.13em; }
.nas-group { margin:12px 0 13px; padding:15px 17px; border:1px solid #334155;
  border-left:4px solid #22d3ee; border-radius:8px; background:#0b1324; }
.nas-group b { display:block; margin:3px 0 4px; color:#e0f2fe; font-size:1.02rem; }
.nas-group p { margin:0; color:#94a3b8; line-height:1.55; }
.nas-badge { display:inline-flex; align-items:center; border:1px solid rgba(34,211,238,.34);
  border-radius:999px; padding:3px 8px; color:#67e8f9; background:rgba(8,145,178,.14);
  font-size:.66rem; font-weight:800; letter-spacing:.05em; }
.nas-badge--blocked { border-color:rgba(249,115,22,.38); color:#fdba74;
  background:rgba(154,52,18,.16); }
.nas-truth { margin:8px 0 15px; padding:13px 15px; border-left:3px solid #f59e0b;
  background:rgba(120,53,15,.16); color:#fde68a; line-height:1.62; }
</style>
"""


@dataclass(frozen=True, slots=True)
class BinaryMaskMetrics:
    """Pixel metrics for one explicitly defined binary-mask comparison."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    iou: float | None
    dice: float | None
    precision: float | None
    recall: float | None


def format_binary_size(byte_count: int) -> str:
    """Return a compact IEC size label for catalog cards and metrics."""

    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            decimals = 0 if unit == "B" else 1
            return f"{value:.{decimals}f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable size unit")


def filter_nas_groups(
    groups: Sequence[NasDatasetGroup],
    *,
    readiness: str = ALL_FILTER,
    modality: str = ALL_FILTER,
) -> tuple[NasDatasetGroup, ...]:
    """Filter catalog groups without mutating their stable source order."""

    if readiness not in READINESS_LABELS:
        raise ValueError(f"unknown readiness filter: {readiness}")
    if modality not in MODALITY_LABELS:
        raise ValueError(f"unknown modality filter: {modality}")
    return tuple(
        group
        for group in groups
        if (readiness == ALL_FILTER or group.readiness == readiness)
        and (modality == ALL_FILTER or group.sensor_modality == modality)
    )


def role_labels(group: NasDatasetGroup) -> tuple[str, ...]:
    """Translate machine-readable role identifiers for presentation."""

    return tuple(ROLE_LABELS.get(role, role) for role in group.roles)


def representative_asset(group_id: str) -> tuple[Path, str] | None:
    """Return one local display-only thumbnail and its explicit limitation."""

    item = REPRESENTATIVE_ASSETS.get(group_id)
    if item is None:
        return None
    filename, caption = item
    return REPRESENTATIVE_ASSET_DIR / filename, caption


def binary_mask_metrics(prediction_png: bytes, truth_png: bytes) -> BinaryMaskMetrics:
    """Calculate binary overlap metrics directly from two same-grid PNG masks."""

    with Image.open(io.BytesIO(prediction_png)) as source:
        prediction = source.convert("L")
    with Image.open(io.BytesIO(truth_png)) as source:
        truth = source.convert("L")
    if prediction.size != truth.size:
        raise ValueError(
            "prediction and truth masks must share one size; "
            f"{prediction.size} != {truth.size}"
        )

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for predicted_value, truth_value in zip(
        prediction.tobytes(), truth.tobytes(), strict=True
    ):
        predicted_water = predicted_value >= 128
        truth_water = truth_value >= 128
        if predicted_water and truth_water:
            true_positive += 1
        elif predicted_water:
            false_positive += 1
        elif truth_water:
            false_negative += 1
        else:
            true_negative += 1

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return BinaryMaskMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        iou=ratio(true_positive, true_positive + false_positive + false_negative),
        dice=ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        precision=ratio(true_positive, true_positive + false_positive),
        recall=ratio(true_positive, true_positive + false_negative),
    )


@lru_cache(maxsize=16)
def cyan_overlay_png(
    context_png_bytes: bytes,
    mask_png_bytes: bytes,
    opacity: float = 0.56,
) -> bytes:
    """Blend a cyan water-mask overlay over one RGB context preview."""

    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")
    with Image.open(io.BytesIO(context_png_bytes)) as source:
        context = source.convert("RGB")
    with Image.open(io.BytesIO(mask_png_bytes)) as source:
        mask = source.convert("L")
    if context.size != mask.size:
        raise ValueError(
            f"context and mask must share one size; {context.size} != {mask.size}"
        )
    binary_mask = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
    cyan = Image.new("RGB", context.size, color=(34, 211, 238))
    tinted = Image.blend(context, cyan, opacity)
    overlay = Image.composite(tinted, context, binary_mask)
    stream = io.BytesIO()
    overlay.save(stream, format="PNG", optimize=False, compress_level=7)
    return stream.getvalue()


def build_observation_interval_figure(group: NasDatasetGroup) -> go.Figure | None:
    """Build a compact plot of actual gaps between successive observations."""

    if not group.interval_days or len(group.observation_dates) < 2:
        return None
    labels = [
        f"{left[5:]} → {right[5:]}"
        for left, right in zip(
            group.observation_dates[:-1],
            group.observation_dates[1:],
            strict=True,
        )
    ]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=group.interval_days,
            marker={"color": ["#22d3ee", "#38bdf8", "#0ea5e9", "#0284c7"]},
            text=[f"{days}일" for days in group.interval_days],
            textposition="outside",
            hovertemplate="%{x}<br>관측 간격 %{y}일<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,.42)",
        font={"color": "#cbd5e1"},
        height=285,
        title={"text": "실제 관측 간격 · 프레임 수와 일수는 다릅니다", "x": 0.01},
        margin={"l": 18, "r": 18, "t": 48, "b": 28},
        showlegend=False,
    )
    figure.update_xaxes(title_text="연속 관측일", gridcolor="#1e293b")
    figure.update_yaxes(title_text="간격 (일)", gridcolor="#1e293b", rangemode="tozero")
    return figure


def _readiness_badge(group: NasDatasetGroup) -> tuple[str, str]:
    label = READINESS_LABELS[group.readiness]
    css_class = "nas-badge" if group.readiness == "demo_ready" else "nas-badge nas-badge--blocked"
    return label, css_class


def _group_option_label(group_id: str) -> str:
    group = get_nas_dataset_group(group_id)
    direct = "실행 가능" if group.materialized else READINESS_LABELS[group.readiness]
    return f"{group.display_name} · {direct}"


def _render_inventory_summary() -> None:
    metrics = st.columns(5)
    metrics[0].metric("NAS 파일", f"{NAS_CATALOG_SUMMARY.file_count:,}개")
    metrics[1].metric("총 용량", format_binary_size(NAS_CATALOG_SUMMARY.total_bytes))
    metrics[2].metric("분류 그룹", f"{NAS_CATALOG_SUMMARY.group_count}개")
    metrics[3].metric(
        "직접 후보", f"{NAS_CATALOG_SUMMARY.direct_prediction_group_count}개"
    )
    metrics[4].metric(
        "로컬 시연셋", f"{NAS_CATALOG_SUMMARY.materialized_sample_count}개"
    )


def _render_complete_inventory_eda() -> None:
    """Render whole-tree EDA before a user drills into one source group."""

    inventory = summarize_inventory()
    readiness = summarize_readiness()
    st.markdown("### 전체 NAS EDA · 무엇을 서로 분리했나요?")
    st.write(
        "파일 300개를 지역·센서·제품 단계·날짜 역할이 같은 자료끼리 8개 그룹으로 "
        "나눴습니다. 아래 용량은 보관 ZIP도 포함하지만, 타임라인은 각 그룹의 고유 "
        "관측일만 표시합니다. 서로 다른 지역이나 센서의 점을 한 시계열로 이어 붙이지 않습니다."
    )
    metrics = st.columns(5)
    metrics[0].metric("날짜가 있는 그룹", f"{inventory.dated_group_count} / {inventory.group_count}")
    metrics[1].metric("그룹별 날짜점", f"{inventory.timeline_point_count}개")
    metrics[2].metric("Acquisition", f"{inventory.acquisition_count}건")
    metrics[3].metric("입력–라벨 쌍", f"{inventory.matched_pair_count}쌍")
    metrics[4].metric("현재 실행 가능", f"{inventory.materialized_group_count}그룹")

    size_tab, timeline_tab, readiness_tab = st.tabs(
        ("① 분류·용량", "② 관측일·간격", "③ 분석·예측 준비도")
    )
    with size_tab:
        st.plotly_chart(
            build_inventory_size_figure(inventory),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "막대 길이는 원천 파일 용량입니다. 용량이 크다고 학습 표본이 많은 것은 아닙니다. "
            "예를 들어 SLC 한 장과 같은 날의 quicklook은 서로 다른 시간 프레임이 아닙니다."
        )
        st.dataframe(inventory_group_rows(inventory), width="stretch", hide_index=True)
    with timeline_tab:
        st.plotly_chart(
            build_inventory_timeline_figure(inventory),
            width="stretch",
            config={"displaylogo": False},
        )
        st.info(
            "점 하나는 해당 그룹의 고유 관측 날짜입니다. Acquisition은 같은 날 여러 "
            "scene이나 부분 촬영을 포함할 수 있으므로, 날짜점 수와 파일 수가 다릅니다."
        )
    with readiness_tab:
        st.plotly_chart(
            build_readiness_matrix_figure(readiness),
            width="stretch",
            config={"displaylogo": False},
        )
        st.write(
            "‘카탈로그 EDA 가능’과 ‘미래 수위 검증 가능’은 다른 조건입니다. 원천영상만 "
            "있으면 파일·센서 EDA는 가능하지만, 수체 탐지·동일 격자·여러 날짜·미래 정답이 "
            "없으면 학습 예측이나 RMSE 검증은 불가능합니다."
        )
        with st.expander("그룹별 준비도 판정 근거 전체표", expanded=False):
            st.dataframe(readiness_rows(readiness), width="stretch", hide_index=True)

    with st.expander("EDA 집계 범위와 주의사항", expanded=False):
        for note in inventory.notes_ko:
            st.write(f"• {note}")
        st.write(
            "이 전체 현황은 정적 인벤토리 EDA입니다. 픽셀값을 실제로 읽은 원본 수치 EDA는 "
            "아래에서 데이터 그룹을 선택했을 때 별도로 표시합니다."
        )


def _render_group_detail(group: NasDatasetGroup) -> None:
    readiness_label, badge_class = _readiness_badge(group)
    st.markdown(
        '<section class="nas-group">'
        f'<span class="{badge_class}">{html.escape(readiness_label)}</span>'
        f"<b>{html.escape(group.display_name)}</b>"
        f"<p>{html.escape(group.direct_prediction_reason_ko)}</p>"
        "</section>",
        unsafe_allow_html=True,
    )

    intervals = " · ".join(f"{days}일" for days in group.interval_days) or "해당 없음"
    date_range = (
        f"{group.observation_dates[0]} ~ {group.observation_dates[-1]}"
        if group.observation_dates
        else "날짜 없음"
    )
    metrics = st.columns(6)
    metrics[0].metric("센서 유형", MODALITY_LABELS.get(group.sensor_modality, group.sensor_modality))
    metrics[1].metric("고유 관측일", f"{group.unique_date_count}일")
    metrics[2].metric("Acquisition", f"{group.acquisition_count}건")
    metrics[3].metric("입력–라벨 쌍", f"{group.matched_pair_count}쌍")
    metrics[4].metric("파일", f"{group.file_count:,}개")
    metrics[5].metric("원천 용량", format_binary_size(group.total_bytes))
    st.caption(
        f"센서 · {group.sensor_name} ({group.sensor_confidence})  |  "
        f"관측 기간 · {date_range}  |  관측 간격 · {intervals}"
    )
    st.markdown("**데이터 역할** · " + " · ".join(role_labels(group)))
    st.caption("논리 경로 · " + "  |  ".join(group.relative_paths))

    if group.observation_dates:
        rows: list[dict[str, Any]] = []
        for index, observed_on in enumerate(group.observation_dates):
            rows.append(
                {
                    "순서": index + 1,
                    "관측 날짜": observed_on,
                    "이전 관측과 간격": (
                        "첫 관측" if index == 0 else f"{group.interval_days[index - 1]}일"
                    ),
                    "입력–라벨 대응": (
                        "대응 쌍" if index < group.matched_pair_count else "해당 없음"
                    ),
                }
            )
        with st.expander("날짜·간격 상세표", expanded=False):
            st.dataframe(rows, width="stretch", hide_index=True)

    st.warning("\n".join(f"• {message}" for message in group.cautions_ko))
    if group.direct_prediction and not group.materialized:
        st.info(
            "원천 구성상 직접 예측 후보이지만 아직 로컬 경량 마스크로 materialize되지 않았습니다. "
            "동일 격자·NoData 검수와 변환이 끝난 뒤 버튼이 활성화됩니다."
        )


def _render_representative_preview(group: NasDatasetGroup) -> None:
    representative = representative_asset(group.group_id)
    if representative is None:
        return
    asset_path, caption = representative
    if not asset_path.is_file():
        st.warning(f"대표 썸네일 파일을 찾지 못했습니다: {asset_path.name}")
        return
    st.markdown("#### 대표 썸네일 · 표시 전용")
    columns = st.columns([1.15, 1])
    columns[0].image(str(asset_path), caption=caption, width="stretch")
    columns[1].warning(
        "이 이미지는 카탈로그에서 자료 성격을 빠르게 구분하기 위한 축소본입니다. "
        "픽셀값·해상도·지리정보가 원본 분석 계약을 보존하지 않으므로 예측 API, "
        "학습, 면적 계산 입력으로 사용하면 안 됩니다."
    )
    if group.group_id == "planetscope-raw":
        columns[1].error(
            "UDM2는 구름·그림자·유효 픽셀 등의 품질 범주입니다. 흰색이나 특정 색을 "
            "물로 해석해 수체 라벨처럼 넣으면 안 됩니다."
        )


def _render_verified_source_profile(group: NasDatasetGroup) -> None:
    """Render per-scene pixel/vendor metadata from the read-only NAS audit."""

    if group.group_id not in available_source_group_ids():
        return
    try:
        view = build_nas_source_eda(group.group_id)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        st.error(f"원본 EDA 프로파일을 표시하지 못했습니다: {exc}")
        return
    st.markdown("#### 실제 NAS 원본 EDA · 픽셀·센서 메타데이터 감사")
    st.caption(f"{view.provenance_ko} · 감사시각 UTC {view.audited_at_utc}")
    columns = st.columns(len(view.metric_rows))
    for column, metric in zip(columns, view.metric_rows, strict=True):
        column.metric(
            str(metric["label"]),
            str(metric["value"]),
            help=str(metric["help"]),
        )
    st.plotly_chart(view.figure, width="stretch", config={"displaylogo": False})
    if view.map_figure is not None:
        st.plotly_chart(
            view.map_figure, width="stretch", config={"displaylogo": False}
        )
        st.caption(
            "footprint 폴리곤은 로컬 SLC 1개(2020-03-02)에서 계산한 대표 장면이며, "
            "다른 날짜는 획득 중심점과 라벨 커버리지 사각형으로만 표시합니다."
        )
    st.dataframe(view.table_rows, width="stretch", hide_index=True)
    if group.group_id == "iceye-raw":
        _render_iceye_sigma0_stats()
    st.info(f"{view.readiness.headline_ko} · {view.readiness.explanation_ko}")
    with st.expander("설명 보기 · 이 원본을 예측 입력으로 만들려면", expanded=False):
        st.markdown("**필수 다음 단계**")
        for item in view.readiness.next_steps_ko:
            st.write(f"• {item}")
        st.markdown("**이 데이터에서 특히 조심할 점**")
        for item in view.readiness.cautions_ko:
            st.write(f"• {item}")
        st.markdown("**원본 감사 공통 한계**")
        for item in view.limitations_ko:
            st.write(f"• {item}")


def _render_materialized_gallery(sample: SampleDataset) -> None:
    if sample.sample_id == "iceye-nas-water-labels":
        _render_iceye_materialized_gallery(sample)
        return
    st.markdown("#### 날짜별 원영상 context · 이진 수체 마스크 · 오버레이")
    st.caption(
        "광학 context는 원본 공통영역을 발표용 RGB로 늘린 미리보기이고, API에는 가운데의 "
        "0/1 이진 마스크만 전달됩니다. 청록 영역은 두 자료의 위치 관계를 확인하는 오버레이입니다."
    )
    tabs = st.tabs(list(sample.source_date_strings))
    for tab, frame in zip(tabs, sample.frames, strict=True):
        with tab:
            columns = st.columns(3)
            if frame.context_png_bytes is None:
                columns[0].info("이 프레임에는 context 미리보기가 없습니다.")
            else:
                columns[0].image(
                    frame.context_png_bytes,
                    caption="광학 context · RGB stretch",
                    width="stretch",
                )
            columns[1].image(
                frame.png_bytes,
                caption="이진 수체 마스크 · 흰색=수체",
                width="stretch",
            )
            if frame.context_png_bytes is None:
                columns[2].info("context가 없어 오버레이를 만들지 않았습니다.")
            else:
                columns[2].image(
                    cyan_overlay_png(frame.context_png_bytes, frame.png_bytes),
                    caption="청록 오버레이 · 시각 검수용",
                    width="stretch",
                )


def _render_iceye_materialized_gallery(sample: SampleDataset) -> None:
    """Keep browse quicklooks visibly separate from geocoded label masks."""

    manifest_path = ICEYE_DEMO_ASSET_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or len(items) != len(sample.frames):
        st.error("ICEYE browse manifest와 로컬 마스크 프레임 수가 일치하지 않습니다.")
        return
    st.markdown("#### 날짜별 ICEYE browse · 공통격자 수체 마스크")
    st.warning(
        "왼쪽 quicklook은 원천 SAR 장면을 눈으로 확인하는 browse 전용 축소본입니다. "
        "오른쪽 마스크와 같은 격자·방향·범위가 아니므로 겹쳐 보거나 픽셀값을 분석하지 "
        "않습니다. 예측 API에는 오른쪽 이진 마스크만 전달됩니다."
    )
    tabs = st.tabs(list(sample.source_date_strings))
    for tab, frame, item in zip(tabs, sample.frames, items, strict=True):
        with tab:
            assets = item.get("assets") if isinstance(item, dict) else None
            browse = assets.get("browse") if isinstance(assets, dict) else None
            browse_file = browse.get("file") if isinstance(browse, dict) else None
            columns = st.columns(2)
            if isinstance(browse_file, str):
                candidate = (ICEYE_DEMO_ASSET_DIR / browse_file).resolve()
                try:
                    candidate.relative_to(ICEYE_DEMO_ASSET_DIR.resolve())
                except ValueError:
                    columns[0].error("browse 경로가 자산 폴더를 벗어났습니다.")
                else:
                    columns[0].image(
                        str(candidate),
                        caption="ICEYE raw quicklook · 비정렬 browse 전용",
                        width="stretch",
                    )
            else:
                columns[0].info("이 날짜의 browse 자산이 없습니다.")
            columns[1].image(
                frame.png_bytes,
                caption="EPSG:32652 공통격자 라벨 · 흰색=수체",
                width="stretch",
            )
    st.caption(
        "label-validity PNG는 선언된 라벨 NoData 15가 실제 픽셀에 없는지 확인한 QA "
        "자산입니다. 모두 유효하다는 뜻은 SAR input footprint가 모두 유효하다는 뜻이 "
        "아니며, 학습에서는 input != 0 유효영역을 별도로 적용해야 합니다."
    )


def _render_busan_source_eda() -> None:
    """Show source-grid evidence separately from the lightweight API masks."""

    source = summarize_busan_source_eda()
    scores = evaluate_busan_area_holdout(source)
    st.markdown("#### 원본 GeoTIFF 기반 EDA · 부산 4시점")
    st.write(
        "아래 면적은 화면 PNG를 다시 센 값이 아니라, NAS 원본 라벨 네 장의 공통 "
        "교집합을 3 m 격자에서 읽어 `값 1=수체`를 집계한 결과입니다. 날짜마다 원본 "
        "footprint가 달라 공통영역으로 자른 뒤 비교했습니다."
    )
    metrics = st.columns(6)
    metrics[0].metric("관측 기간", f"{source.period_days}일")
    metrics[1].metric("관측 프레임", f"{source.frame_count}개")
    metrics[2].metric("관측 간격", " / ".join(map(str, source.interval_days)) + "일")
    metrics[3].metric("최소–최대 면적", f"{source.source_area_min_km2:.2f}–{source.source_area_max_km2:.2f} km²")
    metrics[4].metric("첫→마지막", f"{source.source_area_change_pct:+.2f}%")
    metrics[5].metric("512 최대 오차", f"{source.max_abs_preview_area_error_pct:.2f}%")

    st.plotly_chart(
        build_busan_source_area_figure(source, scores),
        width="stretch",
        config={"displaylogo": False},
    )
    st.dataframe(busan_source_rows(source), width="stretch", hide_index=True)
    st.caption(
        "원본 수체면적은 28.637451 → 27.675351 → 30.273291 → 29.504466 km²입니다. "
        "막대가 아니라 날짜축의 실제 간격으로 읽어야 하며, 네 값만으로 계절성이나 "
        "홍수 원인을 판단할 수는 없습니다."
    )

    error_column, holdout_column = st.columns([1, 1.25])
    with error_column:
        st.plotly_chart(
            build_busan_resampling_error_figure(source),
            width="stretch",
            config={"displaylogo": False},
        )
    with holdout_column:
        st.markdown("##### 마지막 날짜 1회 홀드아웃 · 면적 기준선")
        st.dataframe(holdout_rows(scores), width="stretch", hide_index=True)
        st.warning(
            "첫 3시점으로 2020-04-14 한 점만 맞혀 본 연결 점검입니다. 표본이 1개라 "
            "MAE와 RMSE가 같은 값이 되며, 7·14·30일 일반화 성능이나 수위 RMSE가 "
            "아닙니다. 직전 면적 유지 오차는 0.768825 km², 불규칙 날짜 선형추세는 "
            "0.642809 km²입니다."
        )

    with st.expander("원본 격자·처리·근거 상세", expanded=False):
        xmin, ymin, xmax, ymax = source.common_bounds_utm
        st.write(
            f"CRS `{source.crs}` · 원본 픽셀 {source.source_pixel_width_m:g}×"
            f"{source.source_pixel_height_m:g} m · 공통격자 "
            f"{source.common_grid_width:,}×{source.common_grid_height:,} · "
            f"bounds ({xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f})"
        )
        st.markdown("**실제 처리 근거**")
        for item in source.evidence_ko:
            st.write(f"• {item}")
        st.markdown("**파생 자산 생성 단계**")
        for item in source.processing:
            st.write(f"• {item}")
        st.markdown("**해석 제한**")
        for item in source.limitations:
            st.write(f"• {item}")


ICEYE_GEOCODED_OVERLAY = "20200302_geocoded_label_overlay.png"
ICEYE_SIGMA0_STATS = "slc_sigma0_stats_20200302.json"


def _render_iceye_geocoded_overlay() -> None:
    """지도 좌표로 지오코딩한 실제 SAR 위에 배포 라벨을 겹친 유일한 장면.

    다른 ICEYE 화면은 slant-range 퀵룩이거나 마스크만 따로 보여주기 때문에,
    영상과 라벨이 실제로 같은 곳을 가리키는지는 이 그림에서만 확인된다.
    """

    overlay_path = ICEYE_DEMO_ASSET_DIR / ICEYE_GEOCODED_OVERLAY
    if not overlay_path.exists():
        return
    st.markdown("#### 지오코딩 검증 · 실제 SAR 위에 배포 수체 라벨 중첩 (2020-03-02)")
    st.write(
        "SLC를 EPSG:32652·3 m 지도 격자로 지오코딩한 실제 후방산란 영상에 같은 날짜의 "
        "배포 수체 라벨을 파랑으로 겹쳤습니다. 낙동강 본류와 수영강이 영상의 어두운 "
        "저후방산란 물길과 겹치는지가 영상–라벨 정합의 육안 근거입니다."
    )
    st.image(
        str(overlay_path),
        caption=(
            "ICEYE-X5 SLC → sigma0 → 지오코딩(EPSG:32652, 3 m) · 파랑 = 배포 수체 라벨 · "
            "scripts/iceye_geocode.py 산출"
        ),
        width="stretch",
    )
    st.caption(
        "검은 여백은 회전한 촬영 footprint 바깥이며 관측이 없는 영역입니다. "
        "육안 정합 확인용이고 기하 정확도 수치가 아닙니다."
    )


def _render_iceye_sigma0_stats() -> None:
    """SLC 원본 전 장면에서 실측한 sigma0 분포. 다른 화면에는 dB 수치가 없다."""

    stats_path = ICEYE_DEMO_ASSET_DIR / ICEYE_SIGMA0_STATS
    if not stats_path.exists():
        return
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        db_stats = stats["sigma0_db_stats"]
        percentiles = db_stats["percentiles_db"]
        counts = stats["pixel_counts"]
    except (OSError, ValueError, KeyError, TypeError):
        return

    st.markdown("#### 원본 SLC sigma0 실측 분포 · 전 장면 (2020-03-02)")
    st.write(
        "지오코딩·리샘플 이전의 SLC 전 픽셀에서 직접 계산한 후방산란 통계입니다. "
        "정규화 기준을 정할 때 쓰는 값이며, 화면의 퀵룩 스트레치와는 별개입니다."
    )
    columns = st.columns(5)
    columns[0].metric(
        "유효 픽셀", f"{int(counts['n_valid']):,}",
        help="진폭 0 픽셀은 dB 변환 전 제외했습니다.",
    )
    columns[1].metric("중앙값", f"{percentiles['p50']:.2f} dB")
    columns[2].metric("p2 – p98", f"{percentiles['p2']:.1f} – {percentiles['p98']:.1f} dB")
    columns[3].metric("픽셀 dB 표준편차", f"{db_stats['std_of_pixel_db']:.2f} dB")
    columns[4].metric(
        "진폭 0 비율",
        f"{float(counts['zero_amplitude_fraction_percent']):.3f} %",
    )
    st.caption(
        f"{stats['metadata']['product_name']} · {stats['convention']['sigma0_definition']} · "
        f"입사각 {stats['metadata']['local_incidence_angle_deg_min']:.1f}–"
        f"{stats['metadata']['local_incidence_angle_deg_max']:.1f}° · "
        "단일 장면·단일 날짜 통계이며 시계열 변동은 반영하지 않습니다."
    )


def _render_iceye_source_eda(sample: SampleDataset) -> None:
    """Render audited source/common-grid ICEYE label statistics and holdout."""

    manifest = json.loads(
        (ICEYE_DEMO_ASSET_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or len(items) != len(sample.frames):
        st.error("ICEYE manifest의 날짜 항목과 API 마스크 프레임이 일치하지 않습니다.")
        return
    temporal = manifest["temporal_coverage"]
    common_grid = manifest["common_grid"]
    source_areas: list[float] = []
    preview_areas: list[float] = []
    dates: list[str] = []
    rows: list[dict[str, Any]] = []
    previous_area: float | None = None
    for item in items:
        observed_on = str(item["date"])
        common_counts = item["counts"]["common_bounds_native_centers"]
        preview_counts = item["counts"]["preview_nearest"]
        area = float(common_counts["water_area_km2"])
        preview_area = float(preview_counts["water_area_km2"])
        raw_fields = item["source"]["raw_grd_metadata"]["fields"]
        native_input = item["native"]["input"]
        native_label = item["native"]["label"]
        dates.append(observed_on)
        source_areas.append(area)
        preview_areas.append(preview_area)
        rows.append(
            {
                "관측 날짜": observed_on,
                "이전 대비 (%)": (
                    None if previous_area is None else (area - previous_area) / previous_area * 100.0
                ),
                "원본 공통영역 면적 (km²)": area,
                "512×699 면적 (km²)": preview_area,
                "축소 면적 오차 (%)": (preview_area - area) / area * 100.0,
                "공통영역 수체 비율 (%)": float(common_counts["water_ratio_of_valid"]) * 100.0,
                "입력/라벨 크기": f"{native_input['height']}×{native_input['width']}",
                "Pair 격자 일치": bool(item["native"]["pair_alignment"]["same_pixel_grid"]),
                "라벨 값": "/".join(map(str, item["label_qa"]["actual_unique_values"])),
                "NoData 태그/실제 개수": f"{native_label['nodata']} / {common_counts['declared_nodata_pixel_count']}",
                "Look side": raw_fields["look_side"],
                "중심 입사각 (°)": float(raw_fields["incidence_center"]),
            }
        )
        previous_area = area

    st.markdown("#### 원본 GeoTIFF 기반 EDA · ICEYE SAR 라벨 4시점")
    st.write(
        "4개 input–label pair의 전체 TIFF 헤더와 라벨 픽셀을 읽어 날짜별 정합, "
        "0/1 값, NoData, 공통 물리영역 면적을 확인했습니다. 날짜별 배열 크기와 원점이 "
        "달라 지도 좌표에서 공통영역으로 변환한 뒤 비교했습니다."
    )
    changes = [
        (right - left) / left * 100.0
        for left, right in pairwise(source_areas)
    ]
    metrics = st.columns(6)
    metrics[0].metric("관측 기간", f"{temporal['span_days']}일")
    metrics[1].metric("관측 간격", " / ".join(map(str, temporal["interval_days"])) + "일")
    metrics[2].metric("공통영역", f"{float(common_grid['bbox_area_km2']):,.1f} km²")
    metrics[3].metric("수체 최소–최대", f"{min(source_areas):.2f}–{max(source_areas):.2f} km²")
    metrics[4].metric("첫→마지막", f"{(source_areas[-1] / source_areas[0] - 1) * 100:+.2f}%")
    metrics[5].metric("마지막 1일", f"{changes[-1]:+.2f}%")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=source_areas,
            name="원본 공통영역 계산",
            mode="lines+markers",
            line={"color": "#22d3ee", "width": 3},
            marker={"size": 9},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=preview_areas,
            name="512×699 시연 마스크",
            mode="lines+markers",
            line={"color": "#f59e0b", "width": 2, "dash": "dot"},
            marker={"size": 7},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=390,
        title={"text": "ICEYE 공통영역 수체면적 · 원본 집계와 경량 마스크 분리", "x": 0.01},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,.42)",
        font={"color": "#cbd5e1"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 18, "r": 18, "t": 58, "b": 28},
    )
    figure.update_xaxes(title_text="관측 날짜", gridcolor="#1e293b")
    figure.update_yaxes(title_text="수체 면적 (km²)", gridcolor="#1e293b")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.dataframe(rows, width="stretch", hide_index=True)

    overlap_rows = []
    for left, right in zip(sample.frames[:-1], sample.frames[1:], strict=True):
        score = binary_mask_metrics(left.png_bytes, right.png_bytes)
        overlap_rows.append(
            {
                "날짜 구간": f"{left.observed_on.isoformat()} → {right.observed_on.isoformat()}",
                "간격 (일)": (right.observed_on - left.observed_on).days,
                "IoU": score.iou,
                "Dice": score.dice,
            }
        )
    st.markdown("##### 연속 마스크 공간 겹침")
    st.dataframe(overlap_rows, width="stretch", hide_index=True)

    training_dates = [date.fromisoformat(value) for value in dates[:-1]]
    first_date = training_dates[0]
    x_values = [(value - first_date).days for value in training_dates]
    y_values = source_areas[:-1]
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    denominator = sum((value - mean_x) ** 2 for value in x_values)
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator
    holdout_day = (date.fromisoformat(dates[-1]) - first_date).days
    linear_prediction = max(0.0, mean_y + slope * (holdout_day - mean_x))
    actual = source_areas[-1]
    persistence_error = abs(source_areas[-2] - actual)
    linear_error = abs(linear_prediction - actual)
    holdout_columns = st.columns(2)
    holdout_columns[0].metric(
        "직전 면적 유지 · 1회 RMSE",
        f"{persistence_error:.3f} km²",
        delta=f"예측 {source_areas[-2]:.3f} km²",
        delta_color="off",
    )
    holdout_columns[1].metric(
        "불규칙 날짜 선형추세 · 1회 RMSE",
        f"{linear_error:.3f} km²",
        delta=f"예측 {linear_prediction:.3f} km²",
        delta_color="off",
    )
    st.error(
        f"2020-04-16 원본 공통영역 면적은 {actual:.3f} km²로 하루 전보다 "
        f"{changes[-1]:+.2f}% 변했습니다. 직전 유지와 단순 추세 모두 이 변화를 크게 "
        "놓칩니다. 단 한 점의 RMSE이며 모델 성능이 아닙니다. 입사각·look side가 날짜마다 "
        "달라 실제 수문 변화, 관측기하, 라벨 생성 차이를 분리하기 전에는 비 때문이라고 "
        "단정할 수 없습니다."
    )
    _render_iceye_geocoded_overlay()
    with st.expander("ICEYE 전처리와 모델링 판단 근거", expanded=False):
        st.write(
            "• 네 날짜 pair 자체는 같은 3 m 격자이지만 날짜 간 footprint와 sub-pixel "
            "origin이 달라 direct array stack은 금지했습니다."
        )
        st.write(
            "• 라벨 NoData 태그는 15지만 전체 픽셀에서 15는 0개였습니다. 반면 input=0 "
            "외곽영역은 약 31–38% 이상이어서 학습용 valid mask로 따로 제외해야 합니다."
        )
        st.write(
            "• 2020-03-30은 유일한 left-looking 장면이며 중심 입사각 범위가 약 "
            "15.7°–33.2°입니다. scene-wise robust normalization과 관측기하 변수가 필요합니다."
        )
        st.write(
            "• 2020-04-17 raw SAR는 있지만 대응 정답 라벨이 없습니다. 탐지 모델을 붙이면 "
            "D+1 inference 데모는 가능하지만 정확도 평가 정답으로 쓸 수 없습니다."
        )


_GAUGE_SATELLITE_BY_GROUP = {
    "busan-water-labels": "PlanetScope",
    "iceye-water-labels": "ICEYE",
}


def _render_materialized_eda(group: NasDatasetGroup, sample: SampleDataset) -> None:
    if group.group_id == "busan-water-labels":
        _render_busan_source_eda()
    elif group.group_id == "iceye-water-labels":
        _render_iceye_source_eda(sample)

    st.markdown("#### API 입력용 경량 마스크 EDA")
    interval_figure = build_observation_interval_figure(group)
    if interval_figure is not None:
        st.plotly_chart(
            interval_figure,
            width="stretch",
            config={"displaylogo": False},
        )
    render_input_eda(sample)
    grid_label = (
        "512×699" if group.group_id == "iceye-water-labels" else "512×512"
    )
    st.caption(
        f"표시 면적은 공통 {grid_label} 리샘플 격자와 manifest 픽셀 면적으로 계산한 "
        "근사값입니다. 바로 위 원본 공통격자 면적과 비교할 수 있지만, 경계 정확도나 "
        "검증된 수문 면적 지표로 해석하지 마세요."
    )
    if len(sample.frames) >= 4:
        holdout = binary_mask_metrics(
            sample.frames[-2].png_bytes,
            sample.frames[-1].png_bytes,
        )
        st.markdown("##### Persistence 단일 홀드아웃 · 마스크 겹침")
        metrics = st.columns(4)
        metrics[0].metric("IoU", f"{holdout.iou:.3f}" if holdout.iou is not None else "-")
        metrics[1].metric(
            "Dice", f"{holdout.dice:.3f}" if holdout.dice is not None else "-"
        )
        metrics[2].metric(
            "Precision",
            f"{holdout.precision:.3f}" if holdout.precision is not None else "-",
        )
        metrics[3].metric(
            "Recall", f"{holdout.recall:.3f}" if holdout.recall is not None else "-"
        )
        st.warning(
            f"{sample.source_date_strings[-2]} 마스크를 아무 변화 없이 유지해 "
            f"{sample.source_date_strings[-1]} 라벨과 단 한 번 비교한 persistence 기준선입니다. "
            "학습 모델의 일반화 성능, 일별 미래 예측 성능 또는 수위 RMSE가 아닙니다."
        )
        with st.expander("단일 비교 픽셀 집계", expanded=False):
            st.write(
                f"TP {holdout.true_positive:,} · FP {holdout.false_positive:,} · "
                f"FN {holdout.false_negative:,} · TN {holdout.true_negative:,}"
            )

    gauge_satellite = _GAUGE_SATELLITE_BY_GROUP.get(group.group_id)
    if gauge_satellite is not None:
        st.markdown("#### 같은 날짜 게이지 실측 수위 · 인수 패키지")
        render_gauge_levels_panel(satellite=gauge_satellite, compact=True)


def render_nas_catalog(
    *,
    activate_sample: Callable[[SampleDataset, int], None],
    execute_quick_sample: Callable[[SampleDataset, int], None],
    go_to_phase: Callable[[int], None],
    format_api_error: Callable[[APIError], str],
) -> NasDatasetGroup | None:
    """Render the NAS inventory and bridge one materialized set to existing flow."""

    st.markdown(NAS_UI_CSS, unsafe_allow_html=True)
    st.markdown(
        '<section class="wc-panel nas-hero">'
        '<div class="nas-kicker">DELIVERED DATA INVENTORY · CLASSIFIED</div>'
        '<h3>NAS 전달자료 분류와 시연 세트</h3>'
        '<p>원천영상, 품질 마스크, 수체 라벨, DEM·수위 결과와 보관본을 센서와 역할별로 '
        '분리했습니다. 서로 다른 지역·센서·제품 단계를 한 시계열로 섞지 않습니다.</p>'
        "</section>",
        unsafe_allow_html=True,
    )
    _render_inventory_summary()
    st.markdown(
        '<div class="nas-truth"><b>중요 · 실제 NAS 입력 ≠ 검증된 학습 모델</b><br/>'
        '부산·ICEYE 로컬 세트는 실제 제공 영상·라벨에서 파생됐지만, 현재 백엔드의 내장 모델은 '
        '설명 가능한 기준선입니다. 같은 날짜 게이지 실측 수위는 인수 패키지 30표본(in-sample, '
        '융합 LSTM 학습표본과 동일)으로 표시만 가능하며, 미래 정답이 없어 일반화 수위 '
        'MAE·RMSE는 여전히 산출할 수 없습니다. 실모델 검증 수치는 ‘정량 평가’ 화면에서 '
        '계열 라벨과 함께 확인하세요.</div>',
        unsafe_allow_html=True,
    )
    _render_complete_inventory_eda()
    st.divider()
    st.markdown("### 데이터 그룹 상세 · 하나씩 검토하고 실행합니다")

    filter_columns = st.columns(2)
    readiness = filter_columns[0].selectbox(
        "예측 준비 상태",
        tuple(READINESS_LABELS),
        format_func=lambda value: READINESS_LABELS[value],
        key="nas_readiness_filter",
    )
    modality = filter_columns[1].selectbox(
        "센서 유형",
        tuple(MODALITY_LABELS),
        format_func=lambda value: MODALITY_LABELS[value],
        key="nas_modality_filter",
    )
    groups = filter_nas_groups(
        list_nas_dataset_groups(), readiness=readiness, modality=modality
    )
    if not groups:
        st.info("선택한 준비 상태와 센서 유형에 해당하는 NAS 그룹이 없습니다.")
        return None

    group_ids = tuple(group.group_id for group in groups)
    if st.session_state.get("nas_group_id") not in group_ids:
        st.session_state.nas_group_id = group_ids[0]
    selected_id = st.selectbox(
        "데이터 그룹",
        group_ids,
        format_func=_group_option_label,
        key="nas_group_id",
        help="그룹은 지역·센서·제품 단계와 라벨 의미가 같은 자료끼리만 묶었습니다.",
    )
    group = get_nas_dataset_group(selected_id)
    _render_group_detail(group)
    _render_representative_preview(group)
    _render_verified_source_profile(group)

    sample: SampleDataset | None = None
    if group.materialized_sample_id:
        try:
            sample = get_sample(group.materialized_sample_id)
        except KeyError as exc:
            st.error(f"로컬 시연 샘플 연결을 찾을 수 없습니다: {exc}")

    forecast_days = 30
    if sample is not None:
        forecast_days = int(
            st.radio(
                "일별 시연 전망 기간",
                (7, 14, 30),
                index=2,
                horizontal=True,
                format_func=lambda days: {
                    7: "1주 · 7일",
                    14: "2주 · 14일",
                    30: "1개월 · 30일",
                }[days],
                key="nas_forecast_days",
                help="마지막 관측 다음 날부터 하루 1개 프레임을 요청합니다. 미래 정답이 없어 RMSE는 계산되지 않습니다.",
            )
        )
        st.info(
            f"로컬 시연셋 연결됨 · {sample.display_name} · {len(sample.frames)}프레임 · "
            f"권장 기준선 {sample.recommended_model_id}"
        )
        if sample.recommended_model_id == "irregular-area-trend":
            st.success(
                "이 기준선은 마지막 장만 반복하지 않습니다. 네 프레임의 수체면적과 "
                "실제 날짜 간격을 모두 적합하고, 목표 날짜의 강수를 감쇠 규칙으로 보정한 "
                "뒤 마지막 경계를 목표 면적까지 확장·축소합니다."
            )
            with st.expander("설명 보기 · 여러 프레임이 예측에 어떻게 쓰이나요?", expanded=False):
                st.write(
                    "1) 각 입력 마스크에서 수체 픽셀 수를 셉니다. 2) 첫 날짜로부터 지난 "
                    "실제 일수에 로그 면적 추세를 맞춥니다. 3) 목표 날짜별 KMA 강수가 있으면 "
                    "사용자가 확인할 수 있는 작은 규칙 보정을 더합니다. 4) 일 변화율과 전체 "
                    "변화율 상한을 적용합니다. 5) 마지막 수체 경계를 목표 픽셀 수로 만듭니다."
                )
                st.warning(
                    "면적 추세와 경계 렌더링 기준선이지 학습된 유역·유량·수문 모델이 아닙니다. "
                    "네 시점에서는 불확실성이나 계절성을 신뢰성 있게 학습할 수 없습니다."
                )

    can_execute = bool(sample is not None and group.direct_prediction)
    actions = st.columns([1, 2])
    load_clicked = actions[0].button(
        "데이터 불러오기",
        icon=":material/input:",
        disabled=not can_execute,
        width="stretch",
        help=(
            "로컬 변환이 끝난 날짜별 이진 마스크를 입력·기상 단계에 채웁니다."
            if can_execute
            else "이 그룹은 아직 백엔드 입력용 로컬 마스크 세트가 없습니다."
        ),
    )
    predict_clicked = actions[1].button(
        "EDA 데이터로 바로 예측",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not can_execute,
        width="stretch",
        help=(
            "화면에서 검토한 동일한 날짜별 마스크를 FastAPI 다중시점 기준선에 전송합니다."
            if can_execute
            else "전처리·동일 격자·NoData 검수와 로컬 materialize가 먼저 필요합니다."
        ),
    )

    if load_clicked and sample is not None:
        activate_sample(sample, forecast_days)
        weather_plan = st.session_state.get("sample_weather_plan") or {}
        weather_source = weather_plan.get("source")
        st.session_state.sample_loaded_toast = (
            f"{sample.display_name} 마스크와 부산 ASOS 역사 재현 관측을 불러왔습니다. "
            "기상 단계에서 측정값·대상 기간과 미래예보가 아니라는 점을 검토하세요."
            if weather_source == "kma_asos_historical_replay"
            else f"{sample.display_name} 마스크를 불러왔지만 ASOS 관측은 연결되지 않았습니다. "
            "기상 단계에서 입력을 확인하세요."
        )
        go_to_phase(1)
    if predict_clicked and sample is not None:
        try:
            with st.spinner("NAS 4시점과 날짜·기상을 다중시점 기준선에 전송하는 중입니다..."):
                execute_quick_sample(sample, forecast_days)
        except APIError as exc:
            st.error(format_api_error(exc))

    if sample is not None:
        st.warning(sample.disclaimer_ko)
        _render_materialized_gallery(sample)
        _render_materialized_eda(group, sample)

    return group


__all__ = [
    "ALL_FILTER",
    "MODALITY_LABELS",
    "NAS_UI_CSS",
    "READINESS_LABELS",
    "REPRESENTATIVE_ASSETS",
    "ROLE_LABELS",
    "BinaryMaskMetrics",
    "binary_mask_metrics",
    "build_observation_interval_figure",
    "cyan_overlay_png",
    "filter_nas_groups",
    "format_binary_size",
    "render_nas_catalog",
    "representative_asset",
    "role_labels",
]
