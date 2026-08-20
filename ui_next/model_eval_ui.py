"""Streamlit renderer for the real-model quantitative evaluation dashboard.

All numbers come from the best-effort loaders in ``model_eval``; a missing
file renders an explicit degraded notice instead of an empty chart.  The
three metric series (연구 프로토콜 / 배포 프로토콜 / 스모크 재현) are always
labelled, and pooled·in-sample values carry their own warnings so they cannot
be quoted as representative generalization performance.
"""

from __future__ import annotations

import streamlit as st

try:  # package imports used by pytest and other Python callers
    from .model_eval import (
        FUSED_BADGE_KO,
        FUSED_NOTICE_LINES_KO,
        FUSED_SECTION_HEADER_KO,
        FUSED_WAITING_MESSAGE_KO,
        GaugeLevelTable,
        build_gauge_level_figure,
        build_scene_iou_figure,
        build_weather_figure,
        fused_correction_rows,
        gauge_detail_rows,
        gauge_pivot_rows,
        load_batch_evals,
        load_gauge_levels,
        load_smoke_eval,
        load_valreports,
        load_weather_series,
        per_scene_rows,
        pooled_rows,
        protocol_comparison_rows,
    )
except ImportError:  # direct Streamlit execution adds ui_next/ to sys.path
    from model_eval import (
        FUSED_BADGE_KO,
        FUSED_NOTICE_LINES_KO,
        FUSED_SECTION_HEADER_KO,
        FUSED_WAITING_MESSAGE_KO,
        GaugeLevelTable,
        build_gauge_level_figure,
        build_scene_iou_figure,
        build_weather_figure,
        fused_correction_rows,
        gauge_detail_rows,
        gauge_pivot_rows,
        load_batch_evals,
        load_gauge_levels,
        load_smoke_eval,
        load_valreports,
        load_weather_series,
        per_scene_rows,
        pooled_rows,
        protocol_comparison_rows,
    )

_IN_SAMPLE_BADGE_HTML = (
    '<span style="display:inline-flex;align-items:center;border:1px solid '
    "rgba(249,115,22,.45);border-radius:999px;padding:3px 10px;color:#fdba74;"
    "background:rgba(154,52,18,.2);font-size:.7rem;font-weight:800;"
    'letter-spacing:.05em;">IN-SAMPLE · 융합 LSTM 학습표본과 동일</span>'
)


_FUSED_BADGE_HTML = (
    '<span style="display:inline-flex;align-items:center;border:1px solid '
    "rgba(34,211,238,.45);border-radius:999px;padding:3px 10px;color:#67e8f9;"
    "background:rgba(8,145,178,.18);font-size:.7rem;font-weight:800;"
    'letter-spacing:.05em;margin-right:6px;">{label}</span>'
)


def _render_in_sample_badge() -> None:
    st.markdown(_IN_SAMPLE_BADGE_HTML, unsafe_allow_html=True)


def render_fused_correction_panel(table: GaugeLevelTable) -> None:
    """당일 보정치 계열. 예측 성능으로 읽히지 않게 고지 3줄을 항상 함께 낸다."""

    st.markdown(f"##### {FUSED_SECTION_HEADER_KO}")
    st.markdown(
        _FUSED_BADGE_HTML.format(label=FUSED_BADGE_KO) + _IN_SAMPLE_BADGE_HTML,
        unsafe_allow_html=True,
    )
    if not table.has_fused:
        st.info(FUSED_WAITING_MESSAGE_KO)
        return
    rows = fused_correction_rows(table)
    if not rows:
        st.info(FUSED_WAITING_MESSAGE_KO)
        return
    # 캔버스 dataframe이 일부 환경에서 빈 칸으로 보이는 문제를 피해 정적 렌더한다.
    st.table(rows)
    flagged = sum(1 for row in rows if row["비고"])
    if flagged:
        st.warning(
            f"{flagged}건은 교차센서 쌍이 없거나 학습범위를 벗어난 보정입니다. "
            "'비고' 열을 확인하세요."
        )
    for line in FUSED_NOTICE_LINES_KO:
        st.caption(line)
    st.caption(f"출처 · {table.fused_provenance} · {table.fused_message_ko}")


def render_gauge_levels_panel(
    *, satellite: str | None = None, compact: bool = False
) -> GaugeLevelTable:
    """Render the measured gauge-level panel; reusable from the NAS catalog."""

    table = load_gauge_levels()
    if table.status != "ok":
        st.info(table.message_ko + f" (근거 경로: {table.provenance})")
        return table

    _render_in_sample_badge()
    st.warning(
        "이 게이지 실측 수위 30표본은 융합 LSTM(v3_finalwb) 학습에 사용된 표본과 "
        "동일합니다(in-sample). 같은 날짜의 수위 표시는 가능하지만, 이 값과의 오차를 "
        "일반화 성능이나 미래 예측 검증처럼 인용하면 안 됩니다."
    )

    if compact:
        rows = gauge_detail_rows(table, satellite=satellite)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption(
                f"출처 · {table.provenance} (WAMIS 게이지 실측, "
                f"{table.level_min_m:g}~{table.level_max_m:g} m) · 전체 계열과 "
                "융합 보정 대조는 '정량 평가' 화면에서 확인하세요."
            )
        return table

    metrics = st.columns(4)
    metrics[0].metric("실측 표본", f"{table.n_samples}개")
    metrics[1].metric("게이지 지점", "4곳")
    metrics[2].metric("수위 범위", f"{table.level_min_m:g}–{table.level_max_m:g} m")
    metrics[3].metric("관측 기간", f"{table.date_min} ~ {table.date_max}")

    st.markdown("##### 날짜×지점 실측 수위 (m)")
    # 핵심 표는 정적 렌더(st.table) — 캔버스 기반 dataframe이 화면 캡처·일부
    # 환경에서 빈 칸으로 보이는 문제를 피한다 (행 수가 적어 정적으로 충분)
    st.table(gauge_pivot_rows(table))
    st.caption(
        "빈 칸은 해당 날짜·위성에 그 지점 표본이 없다는 뜻입니다 "
        "(jeongcheon은 ICEYE 2일만 존재). 지점 이름은 FUSED 증적의 "
        "water_level_m 대조로 loc_id 0→jeongcheon · 1→hupo · 2→gimhae · 3→gupo로 "
        "확인했습니다."
    )

    figure = build_gauge_level_figure(table)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

    render_fused_correction_panel(table)

    st.markdown("##### 표본 상세")
    if table.has_fused:
        st.success(table.fused_message_ko + f" (출처: {table.fused_provenance})")
    else:
        st.info(FUSED_WAITING_MESSAGE_KO)
    st.dataframe(gauge_detail_rows(table), width="stretch", hide_index=True)
    st.caption(f"출처 · {table.provenance} · WAMIS 게이지 실측 수위")
    return table


def _render_segmentation_tab() -> None:
    valreports = load_valreports()
    smoke = load_smoke_eval()

    st.markdown("#### 시험씬 계열 대조 · 같은 씬이라도 계열마다 수치가 다릅니다")
    # 핵심 표는 정적 렌더 — 캔버스 dataframe의 빈 칸 렌더 문제 회피
    st.table(protocol_comparison_rows(valreports, smoke))
    st.caption(
        "연구 프로토콜은 인수문서 기재값(연구 격자·경로), 배포 프로토콜은 배포 UTM "
        "그리드의 VALREPORT 실측, 스모크 재현은 로컬 재추론입니다. 세 계열을 한 "
        "숫자로 합치거나 서로 바꿔 인용하지 마세요."
    )

    st.plotly_chart(
        build_scene_iou_figure(valreports, smoke),
        width="stretch",
        config={"displaylogo": False},
    )

    for report in valreports:
        if report.status != "ok":
            st.error(report.message_ko + f" (근거 경로: {report.provenance})")
            continue
        st.markdown(
            f"##### {report.sensor_name} ({report.sensor_type}) · 씬별 지표 — "
            f"{report.protocol_label_ko}"
        )
        st.dataframe(per_scene_rows(report), width="stretch", hide_index=True)

    pooled = pooled_rows(valreports)
    if pooled:
        with st.expander("pooled(4씬 합산) 지표 · 대표값 아님", expanded=False):
            st.dataframe(pooled, width="stretch", hide_index=True)
            st.warning(
                "pooled water IoU(SAR 0.925887 · 광학 0.941484)는 4개 씬 픽셀을 "
                "합산한 값입니다. 씬별 최저(SAR 0416 0.878575)와 함께 보아야 하며 "
                "대표 성능처럼 단독 인용하면 안 됩니다."
            )

    st.markdown("#### 스모크 재현 · 로컬 실추론")
    if smoke.status == "ok" and smoke.metrics is not None:
        metrics = st.columns(4)
        metrics[0].metric(
            "water IoU", f"{smoke.metrics.water_iou:.6f}", help=smoke.protocol_label_ko
        )
        metrics[1].metric("F1", f"{smoke.metrics.f1:.6f}")
        metrics[2].metric("씬", f"{smoke.sensor_name} {smoke.scene_date}")
        metrics[3].metric("유효 픽셀", f"{smoke.valid_pixels:,}")
        st.caption(
            f"출처 · {smoke.provenance} · 배포 프로토콜과 같은 씬(2020-04-16)이지만 "
            "로컬 재추론 산출이므로 별도 계열로 표기합니다 (0.881026 vs 0.878575)."
        )
    else:
        st.info(smoke.message_ko + f" (근거 경로: {smoke.provenance})")

    st.markdown("#### 배치 산출 · data/eval")
    batch = load_batch_evals()
    if batch.status == "waiting":
        st.info(batch.message_ko)
    else:
        st.dataframe(
            [
                {
                    "수치 계열": batch.protocol_label_ko,
                    "산출": report.name,
                    "센서": report.sensor_name or "미상",
                    "씬 날짜": report.scene_date or "미상",
                    "water IoU": report.water_iou,
                    "F1": report.f1,
                }
                for report in batch.reports
            ],
            width="stretch",
            hide_index=True,
        )
        if batch.skipped:
            st.warning(
                "형식이 계약과 달라 건너뛴 파일 · " + ", ".join(batch.skipped)
            )
        st.caption(f"출처 · {batch.provenance}")

    with st.expander("데이터 소스와 프로토콜 · provenance", expanded=False):
        for report in valreports:
            if report.status != "ok":
                st.write(f"• {report.provenance} — {report.message_ko}")
                continue
            st.write(
                f"• {report.provenance} — {report.protocol_label_ko} · "
                f"모델 {report.model_name} (sha256 {report.model_sha256[:12]}…) · "
                f"GT {report.testset_id} (md5 {report.gt_md5[:12]}…) · "
                f"threshold {report.threshold:g} · {report.n_evaluated}씬 평가"
            )
        st.write(f"• {smoke.provenance} — {smoke.protocol_label_ko}")
        st.write(f"• {batch.provenance} — {batch.protocol_label_ko} ({batch.status})")
        st.write(
            "• GT 라벨 원본 — data/incoming/handover/06_aux/Labels_GT (ICEYE 4장) · "
            "Labels_GT_planet (PlanetScope 4장)"
        )


def _render_weather_tab() -> None:
    series = load_weather_series()
    if series.status != "ok":
        st.info(series.message_ko + f" (근거 경로: {series.provenance})")
        return
    metrics = st.columns(4)
    metrics[0].metric("관측 기간", f"{series.date_min} ~ {series.date_max}")
    metrics[1].metric("관측 일수", f"{series.n_days}일")
    metrics[2].metric("AWS 지점", " · ".join(series.station_ids))
    metrics[3].metric("강수 빈 칸", f"{series.missing_precipitation}일")
    figure = build_weather_figure(series)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(
        "융합 LSTM 보정(FUSED, days=60)이 참조하는 60일 기상 창의 근거 데이터입니다. "
        "평균기온·평균 상대습도·일강수량 3채널을 그대로 표시하며, 강수 빈 칸은 0으로 "
        "채우지 않고 표시에서 제외했습니다."
    )
    if series.sha256_matches_fused_evidence:
        st.caption(
            f"무결성 · {series.provenance} sha256이 FUSED 증적(aws_sha256) 기록과 "
            f"일치합니다 ({series.encoding.upper()} 인코딩으로 파싱)."
        )
    else:
        st.warning(
            "현재 기상 CSV의 sha256이 FUSED 증적에 기록된 값과 다릅니다. "
            "파일이 교체되었는지 확인하세요."
        )


def render_model_eval_page() -> None:
    """Top-level 정량 평가 phase: real-model metrics, gauge truth, weather."""

    st.subheader("정량 평가 · 실모델 검증과 실측 근거")
    st.caption(
        "인수 패키지 검증 리포트(배포 프로토콜)와 로컬 스모크 재현, 게이지 실측 수위, "
        "기상 근거 데이터를 계열 구분 라벨과 함께 보여줍니다."
    )
    st.markdown(
        '<div class="truth-note"><b>수치 계열 규율:</b> 연구 프로토콜(인수문서 기재) · '
        "배포 프로토콜(VALREPORT 실측) · 스모크 재현(로컬 실추론)은 격자·경로가 달라 "
        "같은 씬에서도 수치가 다릅니다. pooled·in-sample 수치는 대표값이 아닙니다.</div>",
        unsafe_allow_html=True,
    )
    segmentation_tab, gauge_tab, weather_tab = st.tabs(
        ("① 수체 분할 정량 평가", "② 게이지 실측 수위", "③ 기상 근거 데이터")
    )
    with segmentation_tab:
        _render_segmentation_tab()
    with gauge_tab:
        render_gauge_levels_panel()
    with weather_tab:
        _render_weather_tab()


__all__ = ["render_gauge_levels_panel", "render_model_eval_page"]
