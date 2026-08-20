"""Streamlit UI for a swappable water time-series prediction backend."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from api_client import APIClient, APIError, Endpoints, UploadPart, weather_records
from crop_ui import render_crop_detection_page
from eda import render_input_eda, render_result_eda, render_weather_eda
from forecast_analysis import (
    cumulative_rmse,
    forecast_window_options,
    horizon_row,
    precipitation_by_date,
)
from guide import render_guide_page, render_page_help
from learning import render_learning_help, render_learning_page
from model_eval_ui import render_model_eval_page
from nas_ui import render_nas_catalog
from sample_weather import resolve_sample_weather
from samples import SampleDataset, get_sample, list_samples
from state import (
    FINGERPRINTS_KEY,
    apply_configuration,
    extract_prediction_steps,
    fingerprint,
    normalize_weather_rows,
    openapi_operation_rows,
    validate_input_rows,
)
from theme import compact_header_html, inject_theme, transition_overlay_html

PHASES = (
    ("데이터", "마스크·날짜·빠른 테스트", ":material/database:"),
    ("기상", "ASOS 관측·시나리오", ":material/cloud:"),
    ("예측 실행", "모델 선택·실행", ":material/model_training:"),
    ("결과", "마스크·면적·수위", ":material/monitoring:"),
    ("정량 평가", "실모델 검증·실측 수위·기상", ":material/fact_check:"),
    ("이해 가이드", "원천자료·모델·메뉴 설명", ":material/menu_book:"),
    ("API 가이드", "연계 명세·예제", ":material/api:"),
)

SERVICE_MODES = ("수체 시계열 예측", "작물 탐지")
SERVICE_MODE_LABELS = {
    "수체 시계열 예측": "💧 수체 시계열 예측",
    "작물 탐지": "🌾 작물 탐지",
}

RISK_LABELS = {
    "normal": "정상",
    "caution": "주의",
    "flood_risk": "홍수 위험",
    "drought_risk": "가뭄 위험",
}
RISK_COLORS = {
    "normal": "#22c55e",
    "caution": "#f97316",
    "flood_risk": "#ef4444",
    "drought_risk": "#a855f7",
}
KMA_ASOS_DOCUMENTATION_URL = "https://www.data.go.kr/data/15059093/openapi.do"
LOCAL_TIMEZONE = ZoneInfo("Asia/Seoul")

FALLBACK_OPERATIONS = [
    {"method": "GET", "path": Endpoints.HEALTH, "summary": "서비스 상태 및 버전 확인"},
    {"method": "GET", "path": Endpoints.MODELS, "summary": "교체 가능한 시계열 모델 목록"},
    {"method": "GET", "path": Endpoints.WEATHER_STATUS, "summary": "ASOS 키 설정·샘플 사용 가능 여부"},
    {"method": "GET", "path": Endpoints.WEATHER_STATIONS, "summary": "기상청 ASOS 관측소 목록"},
    {"method": "POST", "path": Endpoints.WEATHER_OBSERVATIONS, "summary": "ASOS 과거 관측/샘플 조회"},
    {"method": "POST", "path": Endpoints.PREDICTIONS, "summary": "마스크 시계열 예측 생성"},
    {"method": "GET", "path": Endpoints.PREDICTIONS, "summary": "저장된 예측 목록"},
    {"method": "GET", "path": Endpoints.PREDICTION, "summary": "예측 상태와 프레임별 결과 조회"},
    {"method": "GET", "path": Endpoints.FILES, "summary": "예측 산출물 목록"},
    {"method": "GET", "path": Endpoints.FILE, "summary": "예측 마스크 등 산출물 다운로드"},
    {"method": "GET", "path": Endpoints.BUNDLE, "summary": "결과 전체 ZIP 다운로드"},
]


def configure_page() -> None:
    st.set_page_config(
        page_title="WATERCAST · 위성영상 분석 콘솔",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_theme()


def sync_config(domain: str, value: Any) -> tuple[bool, tuple[str, ...]]:
    updated, removed, changed = apply_configuration(dict(st.session_state), domain, value)
    st.session_state[FINGERPRINTS_KEY] = updated[FINGERPRINTS_KEY]
    for key in removed:
        st.session_state.pop(key, None)
    return changed, removed


def api_client() -> APIClient:
    return APIClient(st.session_state.api_base_url, timeout=float(st.session_state.api_timeout))


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## WATERCAST")
        st.caption("SATELLITE ANALYTICS · 업무 콘솔")
        if st.session_state.get("service_mode") == "작물 탐지":
            st.divider()
            st.markdown("### 작물 탐지 POC")
            st.success("로컬 탐지 화면 준비됨")
            st.caption(
                "공개 ESA 샘플 또는 사용자 RGB 이미지에서 색상지수 기반 식생·경작 후보를 "
                "표시합니다. 수위 예측 백엔드는 호출하지 않습니다."
            )
            crop_result = st.session_state.get("crop_detection_result")
            if crop_result is not None:
                st.divider()
                st.markdown("### 최근 탐지")
                st.caption(
                    f"후보 비율: {float(crop_result.candidate_ratio_pct):.2f}%"
                )
                st.caption(
                    f"처리 크기: {crop_result.width}×{crop_result.height}px"
                )
                st.caption("판정 방식: 규칙 기반 색상지수")
            st.divider()
            st.link_button(
                "ESA 샘플 원천 보기",
                "https://www.esa.int/ESA_Multimedia/Images/2015/07/Desert_fields",
                icon=":material/open_in_new:",
                width="stretch",
            )
            return
        st.divider()
        st.markdown("### 백엔드 연결")
        st.text_input(
            "FastAPI 주소",
            value=os.getenv("WATER_API_URL", "http://localhost:8000"),
            key="api_base_url",
            help="예: http://localhost:8000 (끝의 /는 생략 가능)",
        )
        st.slider("요청 제한 시간(초)", 3, 120, 30, key="api_timeout")
        changed, _ = sync_config(
            "connection",
            {"base_url": st.session_state.api_base_url.rstrip("/"), "timeout": st.session_state.api_timeout},
        )
        if changed:
            st.caption("연결 설정이 바뀌어 서버에서 받은 결과를 초기화했습니다.")

        check_connection = st.button(
            "연결 확인",
            icon=":material/sync:",
            width="stretch",
        )
        if check_connection or "connection_attempted" not in st.session_state:
            st.session_state.connection_attempted = True
            try:
                short_client = APIClient(
                    st.session_state.api_base_url,
                    timeout=min(float(st.session_state.api_timeout), 2.0),
                )
                st.session_state.connection_status = dict(short_client.health())
            except APIError as exc:
                st.session_state.connection_status = {"status": "error", "message": str(exc)}

        health = st.session_state.get("connection_status")
        if health:
            status = str(health.get("status", "unknown")).lower()
            if status in {"ok", "healthy", "ready"}:
                st.success(f"연결됨 · {status}")
            else:
                st.error(health.get("message") or f"상태: {status}")
        else:
            st.info("연결 확인 전입니다. 입력 작성은 오프라인에서도 가능합니다.")

        st.divider()
        st.markdown("### 현재 작업")
        st.caption(f"입력: {'확정' if st.session_state.get('input_bundle') else '미확정'}")
        st.caption(f"기상: {'확정' if st.session_state.get('weather_confirmed') else '미확정'}")
        prediction = st.session_state.get("prediction_result") or {}
        st.caption(f"예측: {prediction.get('status', '미실행')}")

        st.divider()
        recent_header = st.columns([3, 1])
        recent_header[0].markdown("### 최근 실행")
        refresh_recent = recent_header[1].button(
            "",
            key="refresh_recent_predictions",
            icon=":material/refresh:",
            help="최근 실행 새로고침",
        )
        is_connected = str((health or {}).get("status", "")).lower() in {
            "ok",
            "healthy",
            "ready",
        }
        if is_connected and (refresh_recent or "recent_predictions" not in st.session_state):
            try:
                payload = api_client().list_predictions(limit=6)
                st.session_state.recent_predictions = list(payload.get("items") or [])
                st.session_state.pop("recent_predictions_error", None)
            except APIError as exc:
                st.session_state.recent_predictions_error = format_api_error(exc)

        recent = st.session_state.get("recent_predictions", [])
        for index, item in enumerate(recent):
            prediction_id = str(item.get("id") or item.get("prediction_id") or "")
            created = str(item.get("created_at") or "")[:16].replace("T", " ")
            label = f"{item.get('status', '-')} · {item.get('model_id', '-')}\n\n{created or prediction_id[:10]}"
            if st.button(
                label,
                key=f"recent_prediction_{index}_{prediction_id}",
                width="stretch",
            ):
                try:
                    st.session_state.prediction_result = dict(
                        api_client().get_prediction(prediction_id)
                    )
                    st.session_state.pop("prediction_context", None)
                    st.session_state.artifact_cache = {}
                    st.session_state.pop("bundle_cache", None)
                    go_to_phase(3)
                except APIError as exc:
                    st.error(format_api_error(exc))
        if not recent:
            if st.session_state.get("recent_predictions_error"):
                st.caption(st.session_state.recent_predictions_error)
            else:
                st.caption("저장된 실행이 없습니다.")


def render_header() -> None:
    if st.session_state.get("service_mode") == "작물 탐지":
        st.markdown(
            compact_header_html(
                "현재 화면 · 작물 탐지",
                "로컬 POC · 색상지수 기준선",
                connected=True,
            ),
            unsafe_allow_html=True,
        )
        return
    phase = int(st.session_state.get("phase", 0))
    phase_title = PHASES[phase][0]
    health = st.session_state.get("connection_status") or {}
    connected = str(health.get("status", "")).lower() in {"ok", "healthy", "ready"}
    api_label = "API 정상" if connected else "API 연결 안 됨"
    prediction = st.session_state.get("prediction_result") or {}
    model_label = str(prediction.get("model_id") or "모델 대기")
    st.markdown(
        compact_header_html(
            f"현재 화면 · {phase_title}",
            f"{api_label} · {model_label}",
            connected=connected,
        ),
        unsafe_allow_html=True,
    )


def render_transition_overlay() -> None:
    target = st.session_state.pop("transition_phase", None)
    if target is None:
        return
    try:
        phase_title = PHASES[int(target)][0]
    except (IndexError, TypeError, ValueError):
        return
    st.markdown(transition_overlay_html(phase_title), unsafe_allow_html=True)


def render_service_switch() -> str:
    """Render the top-level task switch without mixing either task's state."""

    st.markdown(
        '<div class="wc-nav-kicker">ANALYSIS WORKSPACE · 분석 업무 전환</div>',
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "분석 업무",
        SERVICE_MODES,
        format_func=lambda value: SERVICE_MODE_LABELS[value],
        horizontal=True,
        key="service_mode",
        label_visibility="collapsed",
        help="수체 시계열 예측과 작물 후보 탐지는 입력·결과 상태를 서로 섞지 않는 별도 화면입니다.",
    )
    st.markdown('<div class="wc-divider"></div>', unsafe_allow_html=True)
    return str(mode)


def render_phase_navigation() -> None:
    if "phase" not in st.session_state:
        st.session_state.phase = 0
    readiness = (
        True,
        bool(st.session_state.get("input_bundle")),
        bool(st.session_state.get("input_bundle") and st.session_state.get("weather_confirmed")),
        bool(st.session_state.get("prediction_result")),
        True,
        True,
        True,
    )
    st.markdown('<div class="wc-nav-kicker">WORKFLOW</div>', unsafe_allow_html=True)
    columns = st.columns(len(PHASES), gap="small")
    for index, ((title, subtitle, icon), column) in enumerate(zip(PHASES, columns)):
        with column:
            if st.button(
                title,
                key=f"phase_button_{index}",
                disabled=not readiness[index],
                width="stretch",
                type="primary" if st.session_state.phase == index else "secondary",
                icon=icon,
                help=subtitle,
            ):
                go_to_phase(index)
    st.markdown('<div class="wc-divider"></div>', unsafe_allow_html=True)


def file_signature(uploaded: Any, index: int) -> str:
    digest = hashlib.sha256(uploaded.getvalue()).hexdigest()[:10]
    return f"{index}_{digest}"


def handoff_template_zip() -> bytes:
    """Build a small, self-describing handoff package for an upstream team."""

    frames_csv = (
        "date,filename,water_level_m,quality_flag\n"
        "2026-08-01,masks/20260801_water_mask.tif,,ok\n"
        "2026-08-11,masks/20260811_water_mask.tif,2.315,ok\n"
    )
    metadata = {
        "dataset_id": "replace-with-project-id",
        "data_stage": "binary_water_mask_after_detection",
        "sensor": "replace-with-sensor-name",
        "source_product_level": "replace-with-product-level",
        "crs": "EPSG:replace",
        "transform": ["pixel_width", 0, "origin_x", 0, "pixel_height_negative", "origin_y"],
        "resolution_m": ["x_resolution", "y_resolution"],
        "shape_hw": ["height", "width"],
        "nodata": "declare explicitly; do not confuse with non-water=0",
        "mask_values": {"0": "non-water", "1": "water"},
        "detection_model": {
            "id": "replace-with-model-id",
            "checkpoint_sha256": "replace-with-checkpoint-hash",
            "threshold": 0.5,
            "preprocess_version": "replace-with-version",
        },
        "provenance": "who created the masks, from which source files, and when",
    }
    readme = (
        "WATERCAST 1세부 전달 예시\n\n"
        "1) 현재 예측 UI에 넣는 것은 원본 SAR/광학영상이 아니라 수체 감지를 끝낸 날짜별 2D 마스크입니다.\n"
        "2) masks/ 아래 파일은 모두 같은 H×W, CRS, transform, 해상도와 공간 범위를 사용해야 합니다.\n"
        "3) frames.csv의 한 행은 한 프레임(한 관측 날짜)이며 날짜는 오름차순이어야 합니다.\n"
        "4) 권장 마스크 값은 0=비수체, 1=수체입니다. NoData는 반드시 별도로 선언하십시오.\n"
        "5) water_level_m은 같은 날짜의 검증된 수위가 있을 때만 쓰고, 없으면 비워 둡니다.\n"
        "6) metadata.json의 placeholder를 실제 계보·좌표·탐지 모델 정보로 교체하십시오.\n"
        "7) 이 ZIP 자체를 UI에 업로드하는 것이 아니라, 규격을 맞춘 mask 파일들을 직접 업로드합니다.\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", readme.encode("utf-8-sig"))
        archive.writestr("frames.csv", frames_csv.encode("utf-8-sig"))
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return buffer.getvalue()


def go_to_phase(index: int) -> None:
    st.session_state.phase = index
    st.session_state.transition_phase = index
    st.rerun()


def activate_sample(sample: SampleDataset, forecast_days: int = 30) -> None:
    """Load one deterministic sample into the same state used by direct uploads."""

    rows = list(sample.input_rows())
    draft = [{key: value for key, value in row.items() if key != "content"} for row in rows]
    sync_config(
        "input",
        {"input_source": "sample", "sample_id": sample.sample_id, "rows": draft},
    )
    st.session_state.input_bundle = rows

    weather_error: str | None = None
    try:
        weather_plan = resolve_sample_weather(
            sample,
            forecast_days,
            observation_loader=api_client().weather_observations,
        )
    except APIError as exc:
        weather_error = format_api_error(exc)
        weather_plan = resolve_sample_weather(sample, forecast_days)
    weather = list(weather_plan.rows)
    sync_config("weather_data", weather)
    st.session_state.weather_data = weather
    st.session_state.weather_confirmed = bool(weather)
    st.session_state.weather_query_result = weather
    st.session_state.weather_editor_revision = fingerprint(weather)[:12]
    st.session_state.weather_query_notice = {
        "source": weather_plan.source,
        "note": weather_plan.note_ko
        + (f" 조회 오류: {weather_error}" if weather_error else ""),
        "metadata": {
            "is_sample": True,
            "is_forecast": False,
            "historical_replay": weather_plan.historical_replay,
            "is_measured": weather_plan.historical_replay and bool(weather),
            "scenario_id": sample.sample_id,
            "forecast_days": forecast_days,
            "disclaimer_ko": sample.disclaimer_ko,
        },
        "missing_dates": list(weather_plan.missing_dates),
    }
    st.session_state.sample_weather_plan = {
        "source": weather_plan.source,
        "historical_replay": weather_plan.historical_replay,
        "missing_dates": list(weather_plan.missing_dates),
        "error": weather_error,
    }
    st.session_state.active_sample_id = sample.sample_id
    st.session_state.active_sample_forecast_days = forecast_days
    st.session_state.water_forecast_period = {
        7: "1주 · 7일",
        14: "2주 · 14일",
        30: "1개월 · 30일",
    }.get(forecast_days, "직접 설정")
    st.session_state.active_sample_metadata = dict(sample.metadata)
    st.session_state.active_sample_disclaimer = sample.disclaimer_ko


def execute_quick_sample(sample: SampleDataset, forecast_days: int = 30) -> None:
    """Run a catalog item through the live multipart prediction endpoint."""

    activate_sample(sample, forecast_days)
    client = api_client()
    health = dict(client.health())
    st.session_state.connection_status = health
    models = client.list_models()
    st.session_state.models = models
    selected_model = next(
        (item for item in models if model_identity(item) == sample.recommended_model_id),
        None,
    )
    if selected_model is None:
        raise APIError(
            f"샘플 권장 모델 {sample.recommended_model_id!r}이 백엔드에 등록되어 있지 않습니다."
        )
    if not bool(selected_model.get("available", selected_model.get("ready", True))):
        raise APIError(f"샘플 권장 모델 {sample.recommended_model_id!r}을 현재 사용할 수 없습니다.")

    # The recovered documentation panels have no geospatial pixel area.  The
    # backend still requires a positive scale, so 1 m²/pixel is deliberately
    # used as an integration placeholder and prominently labelled in the UI.
    pixel_area_m2 = float(sample.pixel_area_m2 or 1.0)
    water_level_config = (
        sample.water_level_config.as_api_dict() if sample.water_level_config else None
    )
    target_dates = list(sample.forecast_date_strings(forecast_days))
    weather = list(st.session_state.get("weather_data") or [])
    reference_levels = list(sample.synthetic_reference_water_levels_m(forecast_days))
    source_metadata = {
        "sample_id": sample.sample_id,
        "data_classification": sample.data_classification,
        "data_classification_ko": sample.data_classification_ko,
        "classification_badge_ko": sample.classification_badge_ko,
        "sensor_name_ko": sample.sensor_name_ko,
        "source_manifest_path": sample.source_manifest_path,
        "raw_data_provenance_known": sample.raw_data_provenance_known,
        "observation_dates_verified": sample.observation_dates_verified,
        "pixel_area_known": sample.pixel_area_known,
        "area_validation_allowed": sample.area_validation_allowed,
        "weather_source": st.session_state.get("sample_weather_plan", {}).get(
            "source"
        ),
        "historical_weather_replay": bool(
            st.session_state.get("sample_weather_plan", {}).get(
                "historical_replay"
            )
        ),
        "source_assets": list(sample.source_assets),
        "limitations_ko": sample.disclaimer_ko,
    }
    settings = {
        "model_id": sample.recommended_model_id,
        "horizon_steps": forecast_days,
        "target_dates": target_dates,
        "threshold": sample.threshold,
        "pixel_area_m2": pixel_area_m2,
        "historical_water_levels": list(sample.historical_water_levels_m),
        "reference_water_levels": reference_levels,
        "evaluation_kind": "synthetic_demo" if reference_levels else None,
        "evaluation_truth_provenance": (
            sample.synthetic_reference_provenance_ko if reference_levels else None
        ),
        "water_level_config": water_level_config,
        "model_options": dict(sample.model_options),
        "caution_pct": sample.caution_pct,
        "risk_pct": sample.risk_pct,
    }
    sync_config("model", settings)
    parts = [
        UploadPart(row["name"], row["content"], row["content_type"])
        for row in st.session_state.input_bundle
    ]
    result = client.create_prediction(
        files=parts,
        model_id=sample.recommended_model_id,
        horizon_steps=forecast_days,
        threshold=sample.threshold,
        source_dates=list(sample.source_date_strings),
        input_metadata=source_metadata,
        weather=weather,
        pixel_area_m2=pixel_area_m2,
        target_dates=target_dates,
        historical_water_levels=list(sample.historical_water_levels_m),
        reference_water_levels=reference_levels,
        evaluation_kind="synthetic_demo" if reference_levels else None,
        evaluation_truth_provenance=(
            sample.synthetic_reference_provenance_ko if reference_levels else None
        ),
        water_level_config=water_level_config,
        model_options=dict(sample.model_options),
        caution_pct=sample.caution_pct,
        risk_pct=sample.risk_pct,
    )
    status = str(result.get("status", "")).lower()
    prediction_id = result.get("id") or result.get("prediction_id")
    if prediction_id and status in {"queued", "pending", "running", "accepted"}:
        result = client.wait_for_prediction(str(prediction_id))
    st.session_state.prediction_result = dict(result)
    st.session_state.prediction_context = {
        "input_source": "sample",
        "sample_id": sample.sample_id,
        "forecast_days": forecast_days,
        "evaluation_kind": "synthetic_demo" if reference_levels else None,
        "pixel_area_placeholder": not sample.pixel_area_known,
        "weather_source": (
            st.session_state.get("sample_weather_plan", {}).get("source")
        ),
        "historical_weather_replay": bool(
            st.session_state.get("sample_weather_plan", {}).get(
                "historical_replay"
            )
        ),
        "limitations": sample.disclaimer_ko,
    }
    st.session_state.artifact_cache = {}
    st.session_state.pop("bundle_cache", None)
    st.session_state.pop("recent_predictions", None)
    go_to_phase(3)


def sample_label(sample: SampleDataset) -> str:
    if sample.raw_data_provenance_known and not sample.synthetic_masks:
        kind = "NAS 실자료"
    elif sample.derived_demo:
        kind = "문서 복원"
    else:
        kind = "합성 시나리오"
    recommended = " · 대표" if sample.recommended else ""
    return f"{sample.display_name} · {kind}{recommended}"


def render_sample_catalog() -> None:
    samples = list_samples()
    selected_id = st.selectbox(
        "테스트 데이터셋",
        [sample.sample_id for sample in samples],
        index=1,
        format_func=lambda sample_id: sample_label(get_sample(sample_id)),
        key="selected_sample_id",
        help="모든 샘플은 재현 가능하며 실제 운영·성능 검증 데이터와 구분됩니다.",
    )
    sample = get_sample(selected_id)
    forecast_days = int(
        st.radio(
            "일별 테스트 전망 기간",
            (7, 14, 30),
            index=2,
            horizontal=True,
            format_func=lambda days: {7: "1주 · 7일", 14: "2주 · 14일", 30: "1개월 · 30일"}[days],
            key="water_quick_forecast_days",
            help="하루에 한 프레임씩 생성합니다. 한 번 실행한 30일 결과는 결과 화면에서 7·14·30일로 나눠 볼 수 있습니다.",
        )
    )
    if sample.raw_data_provenance_known and not sample.synthetic_masks:
        classification = "NAS SOURCE-DERIVED"
    elif sample.derived_demo:
        classification = "DERIVED DEMO"
    else:
        classification = "SYNTHETIC"
    st.markdown(
        f'<div class="wc-sample-summary"><span>{classification}</span><b>{sample.display_name}</b>'
        f'<p>{sample.summary_ko}</p></div>',
        unsafe_allow_html=True,
    )

    metrics = st.columns(5)
    metrics[0].metric("입력 프레임", f"{len(sample.frames)}개")
    metrics[1].metric("관측 시작", sample.source_date_strings[0])
    metrics[2].metric("최근 관측", sample.source_date_strings[-1])
    metrics[3].metric("권장 모델", sample.recommended_model_id)
    if sample.pixel_area_known and sample.pixel_area_m2:
        pixel_area_label = f"{sample.pixel_area_m2:g} m²"
    elif sample.pixel_area_m2:
        pixel_area_label = f"{sample.pixel_area_m2:g} m² · 가정"
    else:
        pixel_area_label = "미상 · API 1 m²"
    metrics[4].metric("픽셀 면적", pixel_area_label)
    st.caption(
        f"빠른 실행 설정 · 마지막 관측 다음 날부터 {forecast_days}일 동안 하루 1개 프레임 · "
        f"목표 {sample.forecast_date_strings(forecast_days)[0]} ~ {sample.forecast_date_strings(forecast_days)[-1]}"
    )

    st.warning(sample.disclaimer_ko)
    if not sample.pixel_area_known:
        integration_pixel_area = float(sample.pixel_area_m2 or 1.0)
        st.caption(
            "빠른 API 테스트에서는 필수 파라미터를 충족하기 위해 "
            f"{integration_pixel_area:g} m²/px를 가정값으로 전송합니다. "
            "이 샘플의 화면·EDA·CSV에서는 검증되지 않은 km²를 숨기고 픽셀 수로 설명합니다."
        )
    if sample.water_level_config:
        st.caption(
            "표시 수위와 면적-수위 계수는 API 연결을 보여주는 예시값입니다. 현장 보정값이 아닙니다."
        )

    actions = st.columns([1, 2])
    if actions[0].button(
        "데이터만 불러오기",
        icon=":material/input:",
        width="stretch",
    ):
        activate_sample(sample, forecast_days)
        st.session_state.sample_loaded_toast = (
            "입력과 기상 시나리오를 불러왔습니다. 기상 단계에서 전송값을 검토하세요."
        )
        go_to_phase(1)
    if actions[1].button(
        "선택 샘플로 바로 예측",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    ):
        try:
            with st.spinner("샘플을 FastAPI에 전송하고 결과 산출물을 만드는 중입니다..."):
                execute_quick_sample(sample, forecast_days)
        except APIError as exc:
            st.error(format_api_error(exc))

    render_input_eda(sample)

    st.markdown("#### 입력 프레임 비교")
    previews = st.columns(2)
    previews[0].image(
        sample.frames[0].png_bytes,
        caption=f"첫 관측 · {sample.source_date_strings[0]}",
        width="stretch",
    )
    previews[1].image(
        sample.frames[-1].png_bytes,
        caption=f"최근 관측 · {sample.source_date_strings[-1]}",
        width="stretch",
    )

    with st.expander("설명 보기 · 이 샘플은 어디서 왔나요?", expanded=False):
        if sample.actual_event_data:
            st.success("실제 사건 자료로 표시된 샘플입니다. 아래 원천·검증 범위를 확인하세요.")
        elif sample.raw_data_provenance_known and not sample.synthetic_masks:
            st.success(
                "실제 NAS 위성영상 대응 라벨에서 만든 파생 샘플입니다. 다만 특정 "
                "홍수·가뭄 사건이나 실측 수위 정답을 뜻하지는 않습니다."
            )
        elif sample.derived_demo:
            st.error(
                "실측 사건 데이터가 아닙니다. 저장소 문서 그림을 잘라 만든 파생 데모이며 "
                "원본 래스터와 위치 정확도는 확인할 수 없습니다."
            )
        else:
            st.error(
                f"실제 {sample.region_name} {sample.scenario_name} 사건 자료가 아닙니다. "
                "지역명은 합성 변화 패턴을 구분하기 위한 시나리오 이름입니다."
            )

        st.markdown(f"**자료 분류** · {sample.data_classification_ko}")
        st.write(sample.source_description_ko)
        provenance_columns = st.columns(3)
        with provenance_columns[0]:
            st.markdown("##### 수체 마스크")
            st.write(sample.mask_provenance_ko)
        with provenance_columns[1]:
            st.markdown("##### 기상 데이터")
            st.write(sample.weather_provenance_ko)
        with provenance_columns[2]:
            st.markdown("##### 수위 데이터")
            st.write(sample.water_level_provenance_ko)

        st.markdown("##### 지역·날짜 표기의 의미")
        st.write(sample.region_label_note_ko)
        use_columns = st.columns(2)
        with use_columns[0]:
            st.success("사용 가능 · " + sample.intended_use_ko)
        with use_columns[1]:
            st.warning("사용 금지 · " + sample.not_suitable_for_ko)

        st.markdown("##### 화면이 참고한 저장소 자산")
        st.code("\n".join(sample.source_assets), language="text")
        st.caption(
            "아래 JSON은 발표용 설명이 아니라 재현·외부 연동 점검을 위한 개발자용 전송 요약입니다."
        )
        st.json(
            {
                "sample_id": sample.sample_id,
                "classification": sample.data_classification,
                "actual_event_data": sample.actual_event_data,
                "region_verified": sample.region_verified,
                "observation_dates_verified": sample.observation_dates_verified,
                "source_dates": sample.source_date_strings,
                "target_dates": sample.forecast_date_strings(forecast_days),
                "weather_rows": len(sample.forecast_weather_payload(forecast_days)),
                "synthetic_reference_rows": len(
                    sample.synthetic_reference_water_levels_m(forecast_days)
                ),
                "recommended_model_id": sample.recommended_model_id,
                "model_options": dict(sample.model_options),
                "metadata": dict(sample.metadata),
            }
        )


def render_direct_upload() -> None:
    st.write(
        "이노뎁 전처리 단계가 만든 **수체 마스크**를 시간순 관측으로 등록합니다. "
        "원본 SAR 영상이 아니라 모델 입력 규격에 맞춘 마스크를 받습니다."
    )
    template_columns = st.columns([1, 2])
    template_columns[0].download_button(
        "1세부 전달 규격 템플릿 ZIP",
        data=handoff_template_zip(),
        file_name="watercast_handoff_template.zip",
        mime="application/zip",
        icon=":material/download:",
        width="stretch",
        help="frames.csv, metadata.json과 초보자용 전달 체크리스트가 들어 있습니다.",
    )
    template_columns[1].caption(
        "1세부 담당자에게 전달할 파일명·날짜·CRS·NoData·탐지 모델 계보 예시입니다. "
        "ZIP을 업로드하는 것이 아니라 이 규격으로 만든 마스크 파일을 아래에서 선택합니다."
    )
    uploaded_files = st.file_uploader(
        "마스크 파일 (최소 2개)",
        type=["npy", "tif", "tiff", "png"],
        accept_multiple_files=True,
        help=(
            "같은 좌표계·해상도·격자의 파일당 단일 2D 마스크만 받습니다. "
            "[T,H,W] NPY는 UI가 아닌 API에서 decoded frame 수만큼 source_dates를 전달하세요."
        ),
    )
    st.caption("UI 입력 계약: 파일 1개 = 날짜 1개 = 2D 마스크 1개. 3D [T,H,W] NPY는 직접 API 호출에서만 사용하세요.")
    include_levels = st.checkbox(
        "관측 수위(m)를 함께 입력",
        value=False,
        help="수위가 없으면 면적/마스크 예측은 가능하지만 수위 결과는 null일 수 있습니다.",
    )

    rows: list[dict[str, Any]] = []
    if uploaded_files:
        st.markdown("#### 관측 메타데이터")
        header = st.columns([3, 2, 2, 1])
        for col, label in zip(header, ("파일", "관측 날짜", "관측 수위(m)", "크기")):
            col.caption(label)
        base_date = dt.datetime.now(LOCAL_TIMEZONE).date() - dt.timedelta(
            days=len(uploaded_files)
        )
        for index, uploaded in enumerate(uploaded_files):
            signature = file_signature(uploaded, index)
            cols = st.columns([3, 2, 2, 1])
            cols[0].write(uploaded.name)
            observed_date = cols[1].date_input(
                "관측 날짜",
                value=base_date + dt.timedelta(days=index),
                key=f"input_date_{signature}",
                label_visibility="collapsed",
            )
            level: float | None = None
            if include_levels:
                level = cols[2].number_input(
                    "수위",
                    value=None,
                    step=0.01,
                    format="%.3f",
                    placeholder="미입력",
                    key=f"input_level_{signature}",
                    label_visibility="collapsed",
                )
            else:
                cols[2].caption("미입력")
            cols[3].caption(f"{uploaded.size / 1024:.1f} KB")
            rows.append(
                {
                    "name": uploaded.name,
                    "date": observed_date.isoformat(),
                    "water_level_m": level,
                    "sha256": hashlib.sha256(uploaded.getvalue()).hexdigest(),
                    "size": uploaded.size,
                    "content_type": uploaded.type or "application/octet-stream",
                    "content": uploaded.getvalue(),
                }
            )

    draft = [
        {key: value for key, value in row.items() if key != "content"}
        for row in rows
    ]
    changed, removed = sync_config(
        "input",
        {"input_source": "upload", "include_levels": include_levels, "rows": draft},
    )
    if changed and removed:
        st.warning("입력 파일 또는 날짜/수위가 바뀌어 기존 기상·예측 결과를 초기화했습니다.")

    errors = validate_input_rows(rows)
    if rows:
        for error in errors:
            st.error(error)
    else:
        st.info("날짜가 다른 2D 마스크 파일을 2개 이상 선택하세요.")
    if rows and not errors:
        sorted_names = [row["name"] for row in sorted(rows, key=lambda row: row["date"])]
        st.info("API 전송 순서: " + " → ".join(sorted_names))
        render_input_eda(input_rows=rows)

    if st.button(
        "입력 확정하고 기상 단계로",
        type="primary",
        icon=":material/arrow_forward:",
        disabled=bool(errors),
        width="stretch",
    ):
        st.session_state.input_bundle = sorted(rows, key=lambda row: row["date"])
        st.session_state.pop("active_sample_id", None)
        st.session_state.pop("active_sample_metadata", None)
        st.session_state.pop("active_sample_disclaimer", None)
        go_to_phase(1)


def render_input_phase() -> None:
    st.subheader("데이터 · 입력 시계열")
    st.caption("저장소 샘플로 백엔드 전체 흐름을 즉시 확인하거나, 앞단에서 만든 마스크를 직접 등록합니다.")
    render_page_help("data", expanded=False)
    render_learning_help("frame", expanded=False)
    render_learning_help("detection", expanded=False)
    render_learning_help("input", expanded=False)
    render_learning_help("preprocess", expanded=False)
    mode = st.radio(
        "입력 방식",
        ("NAS 전달자료", "빠른 테스트", "직접 업로드"),
        horizontal=True,
        key="input_mode",
    )
    if mode == "NAS 전달자료":
        render_nas_catalog(
            activate_sample=activate_sample,
            execute_quick_sample=execute_quick_sample,
            go_to_phase=go_to_phase,
            format_api_error=format_api_error,
        )
    elif mode == "빠른 테스트":
        render_sample_catalog()
    else:
        render_direct_upload()


def station_label(station: Mapping[str, Any]) -> str:
    station_id = station.get("id") or station.get("station_id") or station.get("stn_id") or "?"
    name = station.get("name") or station.get("station_name") or station.get("stn_ko") or "관측소"
    return f"{name} ({station_id})"


def api_error_detail(exc: APIError) -> Mapping[str, Any]:
    detail: Any = exc.detail
    if isinstance(detail, Mapping) and isinstance(detail.get("detail"), Mapping):
        detail = detail["detail"]
    return detail if isinstance(detail, Mapping) else {}


def apply_weather_response(
    response: Mapping[str, Any], *, requested_source: str
) -> tuple[bool, int]:
    """Validate one provider response and put it in the editable weather state."""

    raw = weather_records(response)
    prepared: list[dict[str, Any]] = []
    for item in raw:
        row = dict(item)
        row.setdefault("kind", "observed")
        if not row.get("source"):
            row["source"] = requested_source
        prepared.append(row)
    normalized, errors = normalize_weather_rows(prepared)
    if errors:
        for error in errors:
            st.error(error)
        return False, 0

    sync_config("weather_data", normalized)
    st.session_state.pop("weather_data", None)
    st.session_state.weather_confirmed = False
    st.session_state.weather_query_result = normalized
    st.session_state.weather_editor_revision = fingerprint(normalized)[:12]
    st.session_state.weather_query_notice = {
        "source": response.get("source", requested_source),
        "note": response.get("note"),
        "metadata": response.get("metadata"),
        "missing_dates": response.get("missing_dates", []),
    }
    st.session_state.weather_effective_source = response.get(
        "source", requested_source
    )
    st.session_state.pop("kma_key_required", None)
    return True, len(normalized)


def request_weather_rows(
    *,
    source: str,
    station_id: str,
    start_date: dt.date,
    end_date: dt.date,
    service_key: str | None = None,
) -> tuple[bool, int]:
    response = api_client().weather_observations(
        source=source,
        station_id=station_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        service_key=service_key,
    )
    return apply_weather_response(response, requested_source=source)


def render_weather_chart(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return
    sample_mask = frame["source"].astype(str).str.contains("sample", case=False, na=False)
    observed = frame[(frame["kind"] == "observed") & ~sample_mask]
    sample = frame[(frame["kind"] == "observed") & sample_mask]
    scenario = frame[frame["kind"] == "scenario"]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for subset, name, color in ((observed, "관측 강수", "#087f8c"), (scenario, "시나리오 강수", "#f0a202")):
        if not subset.empty:
            figure.add_trace(
                go.Bar(x=subset["timestamp"], y=subset["precipitation_mm"], name=name, marker_color=color, opacity=0.7),
                secondary_y=False,
            )
    if not sample.empty:
        figure.add_trace(
            go.Bar(
                x=sample["timestamp"],
                y=sample["precipitation_mm"],
                name="합성 샘플 강수",
                marker_color="#8b5cf6",
                opacity=0.65,
                marker_pattern_shape="/",
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(x=frame["timestamp"], y=frame["temperature_c"], name="기온 °C", mode="lines+markers", line={"color": "#e4572e"}),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(x=frame["timestamp"], y=frame["humidity_pct"], name="습도 %", mode="lines+markers", line={"color": "#4763d2", "dash": "dot"}),
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_dark",
        height=390,
        margin={"l": 15, "r": 15, "t": 28, "b": 15},
        legend={"orientation": "h", "y": 1.08},
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,.35)",
        font={"color": "#cbd5e1"},
    )
    figure.update_yaxes(title_text="강수량 (mm)", gridcolor="#1e293b", secondary_y=False)
    figure.update_yaxes(
        title_text="기온 (°C) / 습도 (%)",
        gridcolor="#1e293b",
        secondary_y=True,
    )
    figure.update_xaxes(gridcolor="#1e293b")
    st.plotly_chart(figure, width="stretch")


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "timestamp",
        "precipitation_mm",
        "temperature_c",
        "min_temperature_c",
        "max_temperature_c",
        "humidity_pct",
        "wind_speed_mps",
        "avg_local_pressure_hpa",
        "kind",
        "source",
    ]
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame:
            frame[column] = None
    extra_columns = [column for column in frame.columns if column not in columns and column != "date"]
    frame = frame[[*columns, *extra_columns]]
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.date
    return frame


def render_weather_phase() -> None:
    st.subheader("기상 · 관측 및 외생변수")
    render_page_help("weather", expanded=False)
    if not st.session_state.get("input_bundle"):
        st.warning("먼저 입력 시계열을 확정하세요.")
        return
    st.markdown(
        '<div class="truth-note"><b>자료 의미:</b> 기상청 ASOS는 오늘 기준 D-1까지의 과거 관측입니다. '
        "미래 예보가 아니며, 미래 외생변수는 아래 표에 <code>scenario</code> 행으로 직접 추가하거나 CSV로 교체해야 합니다.</div>",
        unsafe_allow_html=True,
    )

    if "weather_capabilities_attempted" not in st.session_state:
        st.session_state.weather_capabilities_attempted = True
        try:
            short_client = APIClient(
                st.session_state.api_base_url,
                timeout=min(float(st.session_state.api_timeout), 3.0),
            )
            st.session_state.weather_capabilities = dict(short_client.weather_status())
            st.session_state.pop("weather_capabilities_error", None)
        except APIError as exc:
            st.session_state.weather_capabilities_error = format_api_error(exc)

    capabilities = st.session_state.get("weather_capabilities") or {}
    status_known = "server_key_configured" in capabilities
    server_key_configured = capabilities.get("server_key_configured") is True
    status_columns = st.columns([3, 1])
    if server_key_configured:
        status_columns[0].success(
            "기상청 ASOS 사용 가능 · 서버에 KMA_API_KEY가 설정되어 있습니다."
        )
    elif status_known:
        status_columns[0].warning(
            "기상청 인증키 미설정 · 합성 샘플은 즉시 사용할 수 있고, 실관측은 아래에서 키를 입력해야 합니다."
        )
    else:
        status_columns[0].info(
            "기상 설정 상태를 확인하지 못했습니다. 실관측 조회가 실패하면 샘플로 즉시 전환할 수 있습니다."
        )
    if status_columns[1].button(
        "설정 다시 확인",
        icon=":material/refresh:",
        width="stretch",
    ):
        try:
            st.session_state.weather_capabilities = dict(api_client().weather_status())
            st.session_state.pop("weather_capabilities_error", None)
            st.rerun()
        except APIError as exc:
            st.session_state.weather_capabilities_error = format_api_error(exc)
    if st.session_state.get("weather_capabilities_error"):
        st.caption(st.session_state.weather_capabilities_error)

    source_options = (
        "샘플 시나리오 · 키 없이 즉시",
        "기상청 ASOS 실관측",
    )
    source_label = st.radio(
        "조회 원천",
        source_options,
        index=1 if server_key_configured else 0,
        horizontal=True,
        key="weather_source_choice_v3",
        help="샘플은 화면/API 연동 확인용이며 실제 관측이 아닙니다. ASOS는 D-1까지의 공식 과거 관측입니다.",
    )
    source = "asos" if source_label == "기상청 ASOS 실관측" else "sample"

    request_service_key: str | None = None
    if source == "asos":
        request_service_key = st.text_input(
            "이번 조회에 사용할 공공데이터포털 인증키 (서버 키가 없을 때만)",
            type="password",
            key="kma_request_service_key",
            help=(
                "공공데이터포털에서 발급받은 일반 인증키(Encoding 또는 Decoding)를 붙여 넣으세요. "
                "이 값은 이번 UI 세션의 요청에만 포함됩니다."
            ),
        ).strip() or None
        with st.expander("기상청 인증키 발급·설정 방법", expanded=not server_key_configured):
            st.markdown(
                "1. 공공데이터포털에 본인 계정으로 로그인합니다.\n"
                "2. **기상청 지상(종관, ASOS) 일자료 조회서비스**에서 `활용신청`을 누릅니다.\n"
                "3. 발급된 일반 인증키를 위 칸에 붙여 넣거나 서버 실행 전에 "
                "`export KMA_API_KEY='발급키'`를 설정합니다.\n"
                "4. 자료는 미래예보가 아니라 **전일(D-1)까지의 과거 일 관측**입니다."
            )
            st.link_button(
                "공공데이터포털 ASOS 활용신청 열기",
                KMA_ASOS_DOCUMENTATION_URL,
                icon=":material/open_in_new:",
                width="stretch",
            )
    else:
        st.info(
            "샘플 원천은 키 없이 바로 조회됩니다. 날짜와 관측소 흐름을 시험하기 위한 결정적 합성 자료이며 "
            "ASOS 실관측이나 미래예보가 아닙니다."
        )

    utility_buttons = st.columns(2)
    if utility_buttons[0].button(
        "기상 입력 없이 모델 단계로",
        help="기상 입력을 사용하지 않는 모델을 선택할 때 빈 배열을 API에 전달합니다.",
        icon=":material/skip_next:",
        width="stretch",
    ):
        sync_config("weather_data", [])
        st.session_state.weather_data = []
        st.session_state.weather_confirmed = True
        go_to_phase(2)

    if utility_buttons[1].button(
        "관측소 목록 새로고침",
        icon=":material/location_on:",
        width="stretch",
    ):
        try:
            st.session_state.stations = api_client().list_weather_stations()
        except APIError as exc:
            st.error(format_api_error(exc))

    stations = st.session_state.get("stations", [])
    if stations:
        selected_station = st.selectbox(
            "ASOS 관측소",
            stations,
            format_func=station_label,
            help="수체 위치와 가장 가까우면서 대표성이 있는 관측소를 선택하세요.",
        )
        station_id = str(
            selected_station.get("id")
            or selected_station.get("station_id")
            or selected_station.get("stn_id")
        )
    else:
        station_id = st.text_input(
            "ASOS 관측소 ID",
            value="159",
            help="159: 부산. 관측소 목록을 불러오면 이름으로 선택할 수 있습니다.",
        )

    input_dates = [dt.date.fromisoformat(row["date"]) for row in st.session_state.input_bundle]
    latest_observation = dt.datetime.now(LOCAL_TIMEZONE).date() - dt.timedelta(days=1)
    default_start = min(min(input_dates), latest_observation)
    default_end = min(max(input_dates), latest_observation)
    dates = st.columns(2)
    start_date = dates[0].date_input("조회 시작", value=default_start, max_value=latest_observation)
    end_date = dates[1].date_input("조회 종료", value=max(default_start, default_end), max_value=latest_observation)
    query_settings = {
        "source": source,
        "station_id": station_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "per_request_key": bool(request_service_key),
    }
    changed, removed = sync_config("weather_query", query_settings)
    if changed and removed:
        st.warning("기상 조회 조건이 바뀌어 편집 데이터와 예측 결과를 초기화했습니다.")

    if start_date > end_date:
        st.error("조회 시작일은 종료일보다 늦을 수 없습니다.")

    live_key_ready = (
        source != "asos"
        or server_key_configured
        or bool(request_service_key)
        or not status_known
    )
    if source == "asos" and not live_key_ready:
        st.warning(
            "실관측 조회 버튼은 인증키를 입력하면 활성화됩니다. 지금 결과 흐름을 보고 싶다면 아래 샘플 조회를 사용하세요."
        )
    query_buttons = st.columns(2 if source == "asos" else 1)
    query_clicked = query_buttons[0].button(
        "ASOS 과거 관측 조회" if source == "asos" else "합성 샘플 조회",
        type="primary",
        icon=":material/cloud_download:" if source == "asos" else ":material/science:",
        disabled=start_date > end_date or not live_key_ready,
        width="stretch",
    )
    sample_fallback_clicked = False
    if source == "asos":
        sample_fallback_clicked = query_buttons[1].button(
            "키 없이 같은 기간 샘플 조회",
            icon=":material/bolt:",
            disabled=start_date > end_date,
            width="stretch",
            help="ASOS 대신 동일 관측소·기간의 합성 자료를 불러옵니다. 결과에 sample 출처가 표시됩니다.",
        )

    if query_clicked:
        try:
            with st.spinner("기상 API를 조회하는 중입니다..."):
                applied, row_count = request_weather_rows(
                    source=source,
                    station_id=station_id,
                    start_date=start_date,
                    end_date=end_date,
                    service_key=request_service_key,
                )
            if applied:
                st.success(f"{row_count}개 기상 행을 불러왔습니다.")
        except APIError as exc:
            detail = api_error_detail(exc)
            if detail.get("code") == "KMA_API_KEY_REQUIRED":
                st.session_state.kma_key_required = dict(detail)
                st.error(str(detail.get("message_ko") or format_api_error(exc)))
            else:
                st.error(format_api_error(exc))

    if sample_fallback_clicked:
        try:
            with st.spinner("동일 기간의 합성 샘플을 준비하는 중입니다..."):
                applied, row_count = request_weather_rows(
                    source="sample",
                    station_id=station_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            if applied:
                st.success(
                    f"{row_count}개 합성 기상 행을 불러왔습니다. 표의 source=sample 표시를 확인하세요."
                )
        except APIError as exc:
            st.error(format_api_error(exc))

    if st.session_state.get("kma_key_required"):
        st.markdown(
            '<div class="wc-explain"><b>왜 조회가 안 됐나요?</b><p>실제 ASOS는 개인 인증키가 필요한 외부 서비스입니다. '
            "키를 발급받아 붙여 넣거나, 위의 ‘키 없이 같은 기간 샘플 조회’로 전체 예측 흐름을 먼저 확인하세요.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 예측에 전달할 기상 표")
    st.caption(
        "한 행이 하루를 뜻합니다. 조회값을 수정하거나 행을 추가할 수 있으며, 오늘/미래 가정은 kind=scenario로 지정하세요."
    )
    csv_upload = st.file_uploader("사용자 기상 CSV로 교체 (선택)", type=["csv"], key="weather_csv")
    if csv_upload and st.button("업로드 CSV를 편집표에 적용"):
        try:
            uploaded_frame = pd.read_csv(csv_upload)
            normalized, errors = normalize_weather_rows(uploaded_frame.to_dict("records"))
            if errors:
                for error in errors:
                    st.error(error)
            else:
                sync_config("weather_data", normalized)
                st.session_state.pop("weather_data", None)
                st.session_state.weather_confirmed = False
                st.session_state.weather_query_result = normalized
                st.session_state.weather_editor_revision = fingerprint(normalized)[:12]
                st.session_state.weather_query_notice = {
                    "source": "user_csv",
                    "note": "사용자가 업로드한 CSV로 서버 조회 결과를 교체했습니다.",
                    "metadata": {"is_sample": False, "is_forecast": None},
                    "missing_dates": [],
                }
                st.rerun()
        except Exception as exc:  # noqa: BLE001 - surface any uploaded CSV parser failure in the UI
            st.error(f"CSV를 읽지 못했습니다: {exc}")

    template_csv = (
        "timestamp,precipitation_mm,temperature_c,humidity_pct,kind,source\n"
        f"{dt.datetime.now(LOCAL_TIMEZONE).date().isoformat()},0,24,70,scenario,user_scenario\n"
    )
    st.download_button("CSV 양식 받기", template_csv.encode("utf-8-sig"), "weather_scenario_template.csv", "text/csv")

    base_records = st.session_state.get("weather_query_result") or st.session_state.get("weather_data") or []
    if not base_records:
        st.info("먼저 관측을 조회하거나 CSV를 적용하세요.")
        return
    notice = st.session_state.get("weather_query_notice", {})
    metadata = notice.get("metadata") or {}
    if notice.get("source") == "sample" or metadata.get("is_sample") is True:
        st.warning("현재 표는 명시적으로 선택한 결정적 합성 샘플입니다. 실제 ASOS 관측이나 모델 성능 근거로 사용하지 마세요.")
    if metadata.get("is_forecast") is False:
        st.caption("제공자 메타데이터: 과거 관측 자료이며 미래 예보가 아닙니다.")
    elif metadata.get("is_forecast") is True:
        st.info("제공자 메타데이터가 이 자료를 예보로 표시합니다.")
    if metadata.get("disclaimer_ko"):
        st.caption(str(metadata["disclaimer_ko"]))
    if notice.get("note"):
        st.caption(str(notice["note"]))
    if notice.get("missing_dates"):
        st.warning("누락 날짜: " + ", ".join(map(str, notice["missing_dates"])))
    if notice.get("metadata"):
        with st.expander("기상 제공자 메타데이터"):
            st.json(notice["metadata"])
    editor_key = f"weather_editor_{st.session_state.get('weather_editor_revision', 'initial')}"
    edited = st.data_editor(
        records_to_frame(base_records),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DateColumn("날짜", format="YYYY-MM-DD", required=True),
            "precipitation_mm": st.column_config.NumberColumn("강수량(mm)", min_value=0.0),
            "temperature_c": st.column_config.NumberColumn("기온(°C)"),
            "humidity_pct": st.column_config.NumberColumn("습도(%)", min_value=0.0, max_value=100.0),
            "wind_speed_mps": st.column_config.NumberColumn("평균 풍속(m/s)"),
            "avg_local_pressure_hpa": st.column_config.NumberColumn("평균 현지기압(hPa)"),
            "min_temperature_c": st.column_config.NumberColumn("최저기온(°C)"),
            "max_temperature_c": st.column_config.NumberColumn("최고기온(°C)"),
            "kind": st.column_config.SelectboxColumn("구분", options=["observed", "scenario"], required=True),
            "source": st.column_config.TextColumn("출처"),
        },
        key=editor_key,
    )
    candidate, weather_errors = normalize_weather_rows(edited.to_dict("records"))
    for error in weather_errors:
        st.error(error)
    if candidate:
        render_weather_eda(candidate)

    if st.button(
        "기상 데이터 확정하고 모델 단계로",
        type="primary",
        icon=":material/arrow_forward:",
        disabled=bool(weather_errors),
        width="stretch",
    ):
        changed, _ = sync_config("weather_data", candidate)
        st.session_state.weather_data = candidate
        st.session_state.weather_confirmed = True
        if changed:
            st.toast("편집한 기상 데이터가 바뀌어 이전 예측을 초기화했습니다.")
        go_to_phase(2)


def model_identity(model: Mapping[str, Any]) -> str:
    return str(model.get("id") or model.get("model_id") or model.get("name") or "unknown")


def model_label(model: Mapping[str, Any]) -> str:
    identity = model_identity(model)
    name = model.get("display_name") or model.get("name") or identity
    version = model.get("version")
    available = model.get("available", model.get("ready", True))
    suffix = f" · v{version}" if version else ""
    suffix += " · 내장 기준선" if model.get("built_in") else " · 플러그인"
    suffix += " · 수위 출력" if model.get("produces_water_level") else " · 수위 미출력"
    suffix += " · 사용 불가" if not available else ""
    return f"{name} ({identity}){suffix}"


def render_prediction_phase() -> None:
    st.subheader("예측 실행 · 모델과 추론 설정")
    render_page_help("prediction", expanded=False)
    render_learning_help("models", expanded=False)
    render_learning_help("swap", expanded=False)
    if not st.session_state.get("input_bundle") or not st.session_state.get("weather_confirmed"):
        st.warning("입력과 기상 데이터를 먼저 확정하세요.")
        return

    toolbar = st.columns([3, 1])
    if "models_attempted" not in st.session_state:
        st.session_state.models_attempted = True
        try:
            st.session_state.models = APIClient(
                st.session_state.api_base_url,
                timeout=min(float(st.session_state.api_timeout), 4.0),
            ).list_models()
            st.session_state.pop("models_error", None)
        except APIError as exc:
            st.session_state.models_error = format_api_error(exc)
    if toolbar[1].button("모델 목록 새로고침", width="stretch"):
        try:
            st.session_state.models = api_client().list_models()
            st.session_state.models_attempted = True
            st.session_state.pop("models_error", None)
        except APIError as exc:
            st.session_state.models_error = format_api_error(exc)
    models = st.session_state.get("models", [])
    if models:
        preferred_model_id: str | None = None
        active_sample_id = st.session_state.get("active_sample_id")
        if active_sample_id:
            try:
                preferred_model_id = get_sample(
                    str(active_sample_id)
                ).recommended_model_id
            except KeyError:
                preferred_model_id = None
        preferred_index = next(
            (
                index
                for index, item in enumerate(models)
                if model_identity(item) == preferred_model_id
            ),
            0,
        )
        selected_model = toolbar[0].selectbox(
            "백엔드 등록 모델",
            models,
            index=preferred_index,
            format_func=model_label,
        )
        model_id = model_identity(selected_model)
        description = selected_model.get("description")
        if description:
            st.caption(str(description))
        badges = []
        badges.append("내장 기준선" if selected_model.get("built_in") else "외부 모델 플러그인")
        badges.append("기상 입력 사용" if selected_model.get("supports_weather") else "기상 입력 미사용")
        badges.append("모델 수위 출력" if selected_model.get("produces_water_level") else "모델 수위 미출력")
        st.info(" · ".join(badges))
    else:
        selected_model = {}
        model_id = toolbar[0].text_input(
            "모델 ID",
            value="persistence",
            help="서버 목록을 받지 못한 경우에도 등록된 ID를 직접 입력할 수 있습니다.",
        )
        if st.session_state.get("models_error"):
            st.caption("모델 목록 자동 조회 실패 · 수동 ID를 사용합니다: " + st.session_state.models_error)

    default_options = (
        selected_model.get("options")
        if isinstance(selected_model.get("options"), Mapping)
        else {}
    )
    active_model_sample_id = st.session_state.get("active_sample_id")
    active_model_sample: SampleDataset | None = None
    if active_model_sample_id:
        try:
            active_model_sample = get_sample(str(active_model_sample_id))
        except KeyError:
            active_model_sample = None
        if (
            active_model_sample is not None
            and model_id == active_model_sample.recommended_model_id
        ):
            default_options = dict(active_model_sample.model_options)
    with st.expander("모델 플러그인 옵션과 위험 임계값", expanded=False):
        options_text = st.text_area(
            "model_options_json",
            value=json.dumps(default_options, ensure_ascii=False, indent=2),
            key=f"model_options_{hashlib.sha256(model_id.encode('utf-8')).hexdigest()[:10]}",
            help="모델 레지스트리가 제공한 기본 옵션입니다. JSON 객체로 수정하면 어댑터에 그대로 전달됩니다.",
        )
        model_options: dict[str, Any] = {}
        model_options_error: str | None = None
        try:
            parsed_options = json.loads(options_text or "{}")
            if not isinstance(parsed_options, dict):
                raise TypeError("JSON 최상위 값은 객체여야 합니다.")
            model_options = parsed_options
        except (json.JSONDecodeError, TypeError) as exc:
            model_options_error = f"모델 옵션 JSON을 확인하세요: {exc}"
            st.error(model_options_error)
        risk_cols = st.columns(2)
        caution_pct = risk_cols[0].number_input("주의 면적 변화율(%)", min_value=0.0, value=5.0)
        risk_pct = risk_cols[1].number_input("위험 면적 변화율(%)", min_value=0.01, value=15.0)
        risk_threshold_error = risk_pct <= caution_pct
        if risk_threshold_error:
            st.error("위험 임계값은 주의 임계값보다 커야 합니다.")

    st.markdown("#### 일별 전망 기간")
    period = st.radio(
        "예측 기간",
        ("1주 · 7일", "2주 · 14일", "1개월 · 30일", "직접 설정"),
        index=2,
        horizontal=True,
        key="water_forecast_period",
        help="1주·2주·1개월은 하루에 예측 프레임 1개를 만듭니다. 직접 설정에서는 프레임 수와 날짜 간격을 바꿀 수 있습니다.",
    )
    period_horizons = {"1주 · 7일": 7, "2주 · 14일": 14, "1개월 · 30일": 30}
    if period == "직접 설정":
        period_cols = st.columns(2)
        horizon_steps = period_cols[0].number_input(
            "예측 프레임 수",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
        )
        cadence_days = period_cols[1].number_input(
            "예측 프레임 간격(일)",
            min_value=1,
            max_value=365,
            value=1,
            step=1,
            help="예: 30프레임 × 1일은 30일 일별 전망, 6프레임 × 5일은 30일 간격표본입니다.",
        )
    else:
        horizon_steps = period_horizons[str(period)]
        cadence_days = 1
        st.caption(
            f"선택값 · {horizon_steps}프레임 × 1일 간격 = 마지막 관측 이후 {horizon_steps}일 일별 전망"
        )

    cols = st.columns(2)
    threshold = cols[0].slider(
        "마스크 이진화 임계값",
        0.0,
        1.0,
        float(active_model_sample.threshold if active_model_sample else 0.5),
        0.01,
    )
    pixel_area_m2 = cols[1].number_input(
        "픽셀 면적(m²)",
        min_value=0.0001,
        value=float(
            active_model_sample.pixel_area_m2
            if active_model_sample and active_model_sample.pixel_area_m2
            else 9.0
        ),
        step=1.0,
    )
    last_source_date = max(dt.date.fromisoformat(row["date"]) for row in st.session_state.input_bundle)
    target_dates = [
        (last_source_date + dt.timedelta(days=int(cadence_days) * step)).isoformat()
        for step in range(1, int(horizon_steps) + 1)
    ]
    st.caption(
        f"목표 날짜 · {target_dates[0]} ~ {target_dates[-1]} · 총 {len(target_dates)}개"
    )
    with st.expander("목표 날짜 전체 보기"):
        st.write(" → ".join(target_dates))
    if selected_model.get("supports_weather"):
        supplied_scenarios = {
            str(row.get("timestamp") or row.get("date"))[:10]
            for row in st.session_state.weather_data
            if row.get("kind") == "scenario"
        }
        missing_scenarios = [date for date in target_dates if date not in supplied_scenarios]
        if missing_scenarios:
            st.warning(
                "기상 사용 모델이지만 다음 목표 날짜의 scenario 행이 없습니다: "
                + ", ".join(missing_scenarios)
                + ". 내장 기준선은 해당 날짜를 결측으로 처리하며, 외부 모델의 결측 정책은 어댑터 문서를 확인하세요."
            )

    historical_levels = [row.get("water_level_m") for row in st.session_state.input_bundle]
    reference_levels: list[float | None] = []
    evaluation_kind: str | None = None
    evaluation_truth_provenance: str | None = None
    active_sample_id = st.session_state.get("active_sample_id")
    if active_sample_id:
        try:
            active_sample = get_sample(str(active_sample_id))
            if int(cadence_days) == 1:
                reference_levels = list(
                    active_sample.synthetic_reference_water_levels_m(
                        int(horizon_steps)
                    )
                )
            if reference_levels:
                evaluation_kind = "synthetic_demo"
                evaluation_truth_provenance = (
                    active_sample.synthetic_reference_provenance_ko
                )
                st.warning(
                    "이 샘플의 미래 기준 수위는 합성 비교선입니다. 오차 그래프 작동을 보여줄 뿐 실측 정답이나 모델 검증 결과가 아닙니다."
                )
        except KeyError:
            pass

    with st.expander("선택 · 미래 기준 수위 CSV로 오차 평가", expanded=False):
        st.write(
            "예측이 끝난 날짜와 같은 날짜의 정답 수위를 넣으면 MAE·RMSE·MAPE·R²·Bias를 계산합니다. "
            "기준 수위는 모델에 전달되지 않고 추론 후 평가에만 사용됩니다."
        )
        st.download_button(
            "기준 수위 CSV 양식",
            pd.DataFrame(
                {"date": target_dates, "water_level_m": [None] * len(target_dates)}
            ).to_csv(index=False).encode("utf-8-sig"),
            file_name="future_reference_water_levels.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        reference_upload = st.file_uploader(
            "기준 수위 CSV",
            type=["csv"],
            key="water_reference_level_csv",
            help="필수 열: date(YYYY-MM-DD), water_level_m. 일부 날짜만 값이 있어도 해당 날짜만 평가합니다.",
        )
        if reference_upload is not None:
            try:
                reference_upload.seek(0)
                reference_frame = pd.read_csv(reference_upload)
                required = {"date", "water_level_m"}
                if not required.issubset(reference_frame.columns):
                    raise ValueError("date, water_level_m 열이 필요합니다.")
                reference_frame["date"] = reference_frame["date"].astype(str).str[:10]
                if reference_frame["date"].duplicated().any():
                    raise ValueError("date가 중복되었습니다.")
                values_by_date: dict[str, float | None] = {}
                for _, reference_row in reference_frame.iterrows():
                    raw_value = reference_row["water_level_m"]
                    values_by_date[str(reference_row["date"])] = (
                        None if pd.isna(raw_value) else float(raw_value)
                    )
                reference_levels = [values_by_date.get(day) for day in target_dates]
                if not any(value is not None for value in reference_levels):
                    raise ValueError("목표 날짜와 일치하는 유효 수위가 없습니다.")
                evaluation_kind = st.selectbox(
                    "평가 자료 유형",
                    ("measured", "holdout", "user_supplied"),
                    format_func=lambda value: {
                        "measured": "실측 사후 평가",
                        "holdout": "홀드아웃 백테스트",
                        "user_supplied": "사용자 기준값 평가",
                    }[value],
                )
                evaluation_truth_provenance = st.text_input(
                    "정답 자료 출처",
                    value=f"사용자 업로드 · {reference_upload.name}",
                    help="관측소, 기관, 분할 기간 등 재현에 필요한 출처를 적으세요.",
                ).strip()
                st.success(
                    f"목표 날짜와 일치한 기준 수위 {sum(value is not None for value in reference_levels)}개를 평가에 사용합니다."
                )
            except (TypeError, ValueError, UnicodeDecodeError) as exc:
                reference_levels = []
                evaluation_kind = None
                evaluation_truth_provenance = None
                st.error(f"기준 수위 CSV를 확인하세요: {exc}")
    water_level_config: dict[str, Any] | None = None
    level_config_error: str | None = None
    if any(value is not None for value in historical_levels):
        st.caption("입력 프레임과 정렬된 관측 수위: " + json.dumps(historical_levels, ensure_ascii=False))
    else:
        st.info("관측 수위가 없습니다. 선택 모델이 수위를 직접 출력하지 않으면 결과 수위는 null입니다.")

    use_level_calibration = st.checkbox(
        "면적 변화율로 수위 후처리 보정",
        value=False,
        help="학습 모델이 수위를 출력하지 않을 때만 사용합니다. 검증된 현장 계수를 직접 입력해야 합니다.",
    )
    if use_level_calibration:
        st.warning("이 계수는 UI가 추정하지 않습니다. 대상 수계에서 검증한 면적-수위 관계를 입력하세요.")
        if selected_model.get("produces_water_level"):
            st.caption("모델이 수위를 반환하면 모델 출력이 우선하며, 이 보정식은 수위 출력이 없을 때만 적용됩니다.")
        calibration_cols = st.columns(3)
        baseline_default = next(
            (str(value) for value in reversed(historical_levels) if value is not None), ""
        )
        baseline_text = calibration_cols[0].text_input("기준 수위(m)", value=baseline_default)
        ratio_text = calibration_cols[1].text_input("100% 면적 변화당 수위 변화(m)", value="")
        area_text = calibration_cols[2].text_input("기준 면적(m², 선택)", value="")
        try:
            baseline_level_m = float(baseline_text)
            meters_per_area_ratio = float(ratio_text)
            baseline_area_m2 = float(area_text) if area_text.strip() else None
            if baseline_area_m2 is not None and baseline_area_m2 <= 0:
                raise ValueError("기준 면적은 0보다 커야 합니다.")
            water_level_config = {
                "baseline_level_m": baseline_level_m,
                "meters_per_area_ratio": meters_per_area_ratio,
                **({"baseline_area_m2": baseline_area_m2} if baseline_area_m2 is not None else {}),
            }
        except ValueError as exc:
            level_config_error = f"수위 보정값을 확인하세요: {exc}"
            st.error(level_config_error)

    minimum_frames = int(selected_model.get("min_frames", 1) or 1)
    if len(st.session_state.input_bundle) < minimum_frames:
        st.error(f"선택 모델은 최소 {minimum_frames}개 입력 프레임이 필요합니다.")

    prediction_settings = {
        "model_id": model_id,
        "horizon_steps": int(horizon_steps),
        "target_dates": target_dates,
        "threshold": float(threshold),
        "pixel_area_m2": float(pixel_area_m2),
        "historical_water_levels": historical_levels,
        "reference_water_levels": reference_levels,
        "evaluation_kind": evaluation_kind,
        "evaluation_truth_provenance": evaluation_truth_provenance,
        "water_level_config": water_level_config,
        "model_options": model_options,
        "caution_pct": float(caution_pct),
        "risk_pct": float(risk_pct),
    }
    changed, removed = sync_config("model", prediction_settings)
    if changed and removed:
        st.warning("모델 또는 추론 설정이 바뀌어 이전 예측 결과를 초기화했습니다.")

    with st.expander("API 전송 내용 미리보기"):
        st.json(
            {
                **prediction_settings,
                "files": [row["name"] for row in st.session_state.input_bundle],
                "source_dates": [row["date"] for row in st.session_state.input_bundle],
                "weather_rows": len(st.session_state.weather_data),
            }
        )

    disabled = (
        not model_id.strip()
        or level_config_error is not None
        or model_options_error is not None
        or risk_threshold_error
        or len(st.session_state.input_bundle) < minimum_frames
        or not bool(selected_model.get("available", selected_model.get("ready", True)))
    )
    if st.button("예측 실행", type="primary", disabled=disabled, width="stretch"):
        bundle = st.session_state.input_bundle
        parts = [UploadPart(row["name"], row["content"], row["content_type"]) for row in bundle]
        try:
            with st.spinner("모델 서버가 시계열을 처리하는 중입니다..."):
                result = api_client().create_prediction(
                    files=parts,
                    model_id=model_id,
                    horizon_steps=int(horizon_steps),
                    threshold=float(threshold),
                    source_dates=[row["date"] for row in bundle],
                    input_metadata=(
                        dict(st.session_state.get("active_sample_metadata") or {})
                        if st.session_state.get("active_sample_id")
                        else {
                            "input_source": "direct_upload",
                            "provenance_supplied": False,
                        }
                    ),
                    weather=st.session_state.weather_data,
                    pixel_area_m2=float(pixel_area_m2),
                    target_dates=target_dates,
                    historical_water_levels=historical_levels,
                    reference_water_levels=reference_levels,
                    evaluation_kind=evaluation_kind,
                    evaluation_truth_provenance=evaluation_truth_provenance,
                    water_level_config=water_level_config,
                    model_options=model_options,
                    caution_pct=float(caution_pct),
                    risk_pct=float(risk_pct),
                )
                status = str(result.get("status", "")).lower()
                prediction_id = result.get("id") or result.get("prediction_id")
                if prediction_id and status in {"queued", "pending", "running", "accepted"}:
                    result = api_client().wait_for_prediction(str(prediction_id))
            st.session_state.prediction_result = dict(result)
            active_sample_metadata = st.session_state.get("active_sample_metadata") or {}
            st.session_state.prediction_context = {
                "input_source": "sample"
                if st.session_state.get("active_sample_id")
                else "upload",
                "sample_id": st.session_state.get("active_sample_id"),
                "pixel_area_placeholder": bool(
                    st.session_state.get("active_sample_id")
                    and not active_sample_metadata.get("pixel_area_known", False)
                ),
                "limitations": st.session_state.get("active_sample_disclaimer"),
            }
            st.session_state.artifact_cache = {}
            st.session_state.pop("bundle_cache", None)
            go_to_phase(3)
        except APIError as exc:
            st.error(format_api_error(exc))


def artifact_entries(result: Mapping[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    raw_artifacts = result.get("artifacts", [])
    if isinstance(raw_artifacts, Mapping):
        raw_artifacts = [{"name": key, "url": value} for key, value in raw_artifacts.items()]
    if isinstance(raw_artifacts, list):
        for item in raw_artifacts:
            if isinstance(item, str):
                entries.append({"name": Path(item).name, "label": Path(item).name})
            elif isinstance(item, Mapping):
                name = item.get("name") or item.get("filename") or item.get("artifact_name")
                if name:
                    entries.append({"name": str(name), "label": str(item.get("label") or name)})
    for index, step in enumerate(steps, start=1):
        urls = (
            step.get("preview_png_url"),
            step.get("mask_tif_url"),
            step.get("mask_npy_url"),
            step.get("mask_url"),
            step.get("artifact_url"),
        )
        names = [
            step.get("artifact_name") or step.get("mask_artifact") or step.get("filename"),
            *(Path(str(url)).name for url in urls if url),
        ]
        for name in (value for value in names if value):
            entries.append({"name": str(name), "label": f"프레임 {index} · {name}"})
    deduplicated: dict[str, dict[str, str]] = {}
    for entry in entries:
        deduplicated.setdefault(entry["name"], entry)
    preview_priority = {".png": 0, ".jpg": 0, ".jpeg": 0, ".tif": 1, ".tiff": 1, ".npy": 2}
    return sorted(
        deduplicated.values(),
        key=lambda entry: (preview_priority.get(Path(entry["name"]).suffix.lower(), 3), entry["name"]),
    )


def first_not_none(values: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Select the first present value without treating numeric zero as missing."""

    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return default


def risk_label(value: Any) -> str:
    key = str(value or "normal")
    return RISK_LABELS.get(key, key)


WATER_LEVEL_SOURCE_LABELS = {
    "adapter_output": "모델 어댑터 직접 출력",
    "area_level_calibration": "면적–수위 보정식 후처리",
    "unavailable": "수위 산출 없음",
}


def _level_text(value: Any, *, missing: str = "미산출") -> str:
    try:
        return f"{float(value):.3f} m"
    except (TypeError, ValueError):
        return missing


def render_horizon_level_cards(
    result: Mapping[str, Any], table_rows: list[dict[str, Any]]
) -> None:
    """Show the current level and exact D+7/D+14/D+30 snapshots."""

    parameters = result.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    history = parameters.get("historical_water_levels_m")
    if not isinstance(history, list):
        history = []
    current = next((value for value in reversed(history) if value is not None), None)
    cards = st.columns(4)
    cards[0].metric("최근 관측 수위", _level_text(current, missing="관측 없음"))
    for column, horizon in zip(cards[1:], (7, 14, 30)):
        row = horizon_row(table_rows, horizon)
        value = row.get("water_level_m") if row else None
        delta = None
        if value is not None and current is not None:
            delta = f"{float(value) - float(current):+.3f} m"
        column.metric(
            f"D+{horizon} 예측 수위",
            _level_text(value, missing="범위 밖" if row is None else "미산출"),
            delta=delta,
        )


def render_daily_water_level_chart(
    result: Mapping[str, Any],
    table_rows: list[dict[str, Any]],
    *,
    prediction_id: str,
) -> None:
    """Join input observations and daily forecasts in one presentation chart."""

    if not table_rows:
        return
    options = forecast_window_options(len(table_rows))
    if not options:
        return
    selected_days = st.radio(
        "그래프 전망 범위",
        options,
        index=len(options) - 1,
        horizontal=True,
        format_func=lambda days: f"{days}일" if days in {7, 14, 30} else f"전체 {days}일",
        key=f"water_result_window_days_{prediction_id or 'current'}",
        help="한 번 산출한 일별 결과에서 D+7·D+14·D+30 구간을 바꿔 봅니다. 모델을 다시 실행하지 않습니다.",
    )
    visible_rows = table_rows[: int(selected_days)]
    parameters = result.get("parameters")
    input_summary = result.get("input")
    if not isinstance(parameters, Mapping):
        parameters = {}
    if not isinstance(input_summary, Mapping):
        input_summary = {}

    raw_dates = input_summary.get("source_dates")
    raw_history = parameters.get("historical_water_levels_m")
    source_dates = list(raw_dates) if isinstance(raw_dates, list) else []
    historical_levels = list(raw_history) if isinstance(raw_history, list) else []
    observed = [
        (str(day)[:10], level)
        for day, level in zip(source_dates, historical_levels)
        if level is not None
    ]
    predicted = [row for row in visible_rows if row.get("water_level_m") is not None]
    references = [
        row for row in visible_rows if row.get("reference_water_level_m") is not None
    ]
    if not observed and not predicted and not references:
        st.info(
            "수위 시계열을 그릴 값이 없습니다. 수위를 직접 출력하는 모델을 사용하거나 "
            "검증된 면적–수위 보정식을 설정하세요. 수체 면적 그래프는 아래에서 확인할 수 있습니다."
        )
        return

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.76, 0.24],
        subplot_titles=("관측 수위 → 미래 일별 예측", "예측에 전달된 일 강수 시나리오"),
    )
    if observed:
        figure.add_trace(
            go.Scatter(
                x=[item[0] for item in observed],
                y=[item[1] for item in observed],
                name="관측 수위",
                mode="lines+markers",
                line={"color": "#22d3ee", "width": 3},
                marker={"size": 7},
                hovertemplate="%{x}<br>관측 수위 %{y:.3f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if predicted:
        forecast_x = [str(row["target_date"])[:10] for row in predicted]
        forecast_y = [row["water_level_m"] for row in predicted]
        if observed:
            forecast_x.insert(0, observed[-1][0])
            forecast_y.insert(0, observed[-1][1])
        figure.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                name="예측 수위",
                mode="lines+markers",
                line={"color": "#f59e0b", "width": 3},
                marker={"size": 6},
                hovertemplate="%{x}<br>예측 수위 %{y:.3f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if references:
        figure.add_trace(
            go.Scatter(
                x=[str(row["target_date"])[:10] for row in references],
                y=[row["reference_water_level_m"] for row in references],
                name="평가 기준 수위",
                mode="lines+markers",
                line={"color": "#cbd5e1", "width": 2, "dash": "dot"},
                marker={"size": 5, "symbol": "diamond"},
                hovertemplate="%{x}<br>기준 수위 %{y:.3f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )

    weather_rows = parameters.get("weather")
    precipitation = precipitation_by_date(
        weather_rows if isinstance(weather_rows, list) else []
    )
    forecast_dates = [str(row["target_date"])[:10] for row in visible_rows]
    figure.add_trace(
        go.Bar(
            x=forecast_dates,
            y=[precipitation.get(day, 0.0) for day in forecast_dates],
            name="강수 시나리오",
            marker_color="rgba(56,189,248,.55)",
            hovertemplate="%{x}<br>강수 %{y:.1f} mm<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if source_dates and forecast_dates:
        boundary = dt.datetime.fromisoformat(str(source_dates[-1])[:10])
        figure.add_vline(
            x=boundary,
            line={"color": "#64748b", "dash": "dash", "width": 1},
            row=1,
            col=1,
        )
        figure.add_annotation(
            x=boundary,
            y=1,
            xref="x",
            yref="paper",
            text="관측 종료 / 예측 시작",
            showarrow=False,
            xanchor="left",
            font={"color": "#94a3b8", "size": 11},
        )
    figure.update_layout(
        template="plotly_dark",
        height=570,
        margin={"l": 16, "r": 16, "t": 70, "b": 20},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,.35)",
        font={"color": "#cbd5e1"},
        bargap=0.25,
    )
    figure.update_xaxes(gridcolor="#1e293b", type="date")
    figure.update_yaxes(title_text="수위 (m)", gridcolor="#1e293b", row=1, col=1)
    figure.update_yaxes(title_text="강수 (mm)", gridcolor="#1e293b", row=2, col=1)
    st.plotly_chart(figure, width="stretch")

    source_values = {
        str(row.get("water_level_source") or "unavailable")
        for row in visible_rows
    }
    source_text = ", ".join(
        WATER_LEVEL_SOURCE_LABELS.get(value, value) for value in sorted(source_values)
    )
    st.caption(
        f"수위 산출 출처 · {source_text}. 회색 기준선은 정답 수위가 요청에 별도로 제공된 경우에만 표시되며 모델 입력에는 사용되지 않습니다."
    )


def render_water_level_evaluation(result: Mapping[str, Any]) -> None:
    """Render measured or explicitly labelled demonstration metrics."""

    st.markdown("#### 수위 예측 오차 · 정확도 검토")
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("status") != "available":
        cards = st.columns(5)
        for column, label in zip(cards, ("MAE", "RMSE", "MAPE", "R²", "Bias")):
            column.metric(label, "산출 불가")
        reason = (
            evaluation.get("unavailable_reason")
            if isinstance(evaluation, Mapping)
            else "미래 시점의 기준(정답) 수위가 제공되지 않았습니다."
        )
        st.info(
            f"{reason} 예측값만으로 RMSE를 만들 수는 없습니다. 같은 날짜의 실측·홀드아웃 기준값을 제공하면 백엔드가 동적으로 계산합니다."
        )
        return

    if evaluation.get("kind") == "synthetic_demo":
        st.warning(
            "합성 시연 백테스트 · 모델 성능 검증 아님. 아래 수치는 화면과 API의 평가 기능을 확인하기 위한 합성 기준선 비교입니다."
        )
    elif evaluation.get("warning"):
        st.warning(str(evaluation["warning"]))

    metric_specs = (
        ("MAE", "mae_m", "{:.3f} m"),
        ("RMSE", "rmse_m", "{:.3f} m"),
        ("MAPE", "mape_pct", "{:.2f}%"),
        ("R²", "r2", "{:.3f}"),
        ("Bias", "bias_m", "{:+.3f} m"),
        ("평가 표본", "sample_count", "{:d}일"),
    )
    cards = st.columns(len(metric_specs))
    for column, (label, key, pattern) in zip(cards, metric_specs):
        value = evaluation.get(key)
        if value is None:
            rendered = "산출 불가"
        elif key == "sample_count":
            rendered = pattern.format(int(value))
        else:
            rendered = pattern.format(float(value))
        column.metric(label, rendered)
    st.caption(
        f"평가 유형 · {evaluation.get('label', evaluation.get('kind', '-'))} · "
        f"정답 출처 · {evaluation.get('truth_provenance', '-')}"
    )

    raw_pairs = evaluation.get("pairs")
    pairs = [dict(item) for item in raw_pairs if isinstance(item, Mapping)] if isinstance(raw_pairs, list) else []
    if pairs:
        horizons = [int(pair["horizon"]) for pair in pairs]
        references = [float(pair["reference_water_level_m"]) for pair in pairs]
        predictions = [float(pair["predicted_water_level_m"]) for pair in pairs]
        residuals = [float(pair["residual_m"]) for pair in pairs]
        running_rmse = cumulative_rmse(pairs)
        minimum = min(references + predictions)
        maximum = max(references + predictions)
        padding = max((maximum - minimum) * 0.08, 0.01)
        figure = make_subplots(
            rows=1,
            cols=3,
            horizontal_spacing=0.1,
            subplot_titles=("실제–예측 비교", "D+별 잔차", "누적 RMSE"),
        )
        figure.add_trace(
            go.Scatter(
                x=[minimum - padding, maximum + padding],
                y=[minimum - padding, maximum + padding],
                name="완전 일치선",
                mode="lines",
                line={"color": "#64748b", "dash": "dash"},
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=references,
                y=predictions,
                name="일별 비교",
                mode="markers",
                marker={"color": "#22d3ee", "size": 8},
                customdata=horizons,
                hovertemplate="D+%{customdata}<br>기준 %{x:.3f} m<br>예측 %{y:.3f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=[f"D+{value}" for value in horizons],
                y=residuals,
                name="잔차",
                marker_color=["#f97316" if value >= 0 else "#38bdf8" for value in residuals],
                hovertemplate="%{x}<br>예측−기준 %{y:+.3f} m<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=[f"D+{value}" for value in horizons[: len(running_rmse)]],
                y=running_rmse,
                name="누적 RMSE",
                mode="lines+markers",
                line={"color": "#f59e0b", "width": 3},
                hovertemplate="%{x}<br>누적 RMSE %{y:.3f} m<extra></extra>",
            ),
            row=1,
            col=3,
        )
        figure.update_layout(
            template="plotly_dark",
            height=390,
            margin={"l": 16, "r": 16, "t": 60, "b": 24},
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(2,6,23,.35)",
            font={"color": "#cbd5e1"},
        )
        figure.update_xaxes(gridcolor="#1e293b")
        figure.update_yaxes(gridcolor="#1e293b")
        figure.update_xaxes(title_text="기준 수위 (m)", row=1, col=1)
        figure.update_yaxes(title_text="예측 수위 (m)", row=1, col=1)
        figure.update_yaxes(title_text="잔차 (m)", zeroline=True, row=1, col=2)
        figure.update_yaxes(title_text="RMSE (m)", rangemode="tozero", row=1, col=3)
        st.plotly_chart(figure, width="stretch")

    with st.expander("설명 보기 · MAE, RMSE, MAPE, R², Bias는 무엇인가요?"):
        st.markdown(
            "- **MAE**: 일별 절대 오차의 평균입니다. 0에 가까울수록 좋습니다.\n"
            "- **RMSE**: 큰 오차에 더 큰 벌점을 주는 오차입니다. 0에 가까울수록 좋습니다.\n"
            "- **MAPE**: 기준 수위 대비 오차 비율입니다. 기준값이 0인 날짜는 제외됩니다.\n"
            "- **R²**: 기준 수위 변동을 얼마나 설명하는지 봅니다. 1에 가까울수록 좋고 음수도 가능합니다.\n"
            "- **Bias**: `예측−기준` 평균입니다. 양수면 평균적으로 높게, 음수면 낮게 예측했습니다.\n\n"
            "지표는 평가 자료의 기간·관측소·결측 처리와 함께 읽어야 하며, 합성 시연값은 실제 모델 성능으로 인용하면 안 됩니다."
        )


def render_horizon_area_cards(
    table_rows: list[dict[str, Any]], *, pixel_area_placeholder: bool
) -> None:
    """Show the latest input baseline and exact D+7/D+14/D+30 area outputs."""

    if not table_rows:
        return
    area_key = "water_area_pixels" if pixel_area_placeholder else "water_area_km2"

    def area_text(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "미산출"
        if pixel_area_placeholder:
            return f"{round(number):,} px"
        return f"{number:,.4f} km²"

    first = table_rows[0]
    first_value = first.get(area_key)
    first_change = first.get("area_change_pct")
    baseline: float | None = None
    try:
        divisor = 1.0 + float(first_change) / 100.0
        if divisor != 0:
            baseline = float(first_value) / divisor
    except (TypeError, ValueError):
        baseline = None

    cards = st.columns(4)
    cards[0].metric("최근 입력 수체", area_text(baseline))
    for column, horizon in zip(cards[1:], (7, 14, 30)):
        row = horizon_row(table_rows, horizon)
        value = row.get(area_key) if row else None
        change = row.get("area_change_pct") if row else None
        delta = None
        try:
            delta = f"{float(change):+.2f}%"
        except (TypeError, ValueError):
            pass
        column.metric(
            f"D+{horizon} 수체",
            "범위 밖" if row is None else area_text(value),
            delta=delta,
        )


def render_prediction_chart(
    table_rows: list[dict[str, Any]], *, pixel_area_placeholder: bool
) -> None:
    if not table_rows:
        return
    frame = pd.DataFrame(table_rows)
    x_values = frame["target_date"]
    area_column = "water_area_pixels" if pixel_area_placeholder else "water_area_km2"
    area_title = "수체 픽셀 수" if pixel_area_placeholder else "수체 면적 (km²)"
    colors = [RISK_COLORS.get(str(value), "#38bdf8") for value in frame["risk"]]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=frame[area_column],
            name=area_title,
            mode="lines+markers",
            line={"color": "#38bdf8", "width": 3},
            marker={"size": 10, "color": colors, "line": {"color": "#e0f2fe", "width": 1}},
            fill="tozeroy",
            fillcolor="rgba(14,165,233,.12)",
            customdata=[[risk_label(value)] for value in frame["risk"]],
            hovertemplate="%{x}<br>" + area_title + ": %{y:,.4f}<br>위험도: %{customdata[0]}<extra></extra>",
        ),
        secondary_y=False,
    )
    has_level = frame["water_level_m"].notna().any()
    if has_level:
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["water_level_m"],
                name="수위 (m)",
                mode="lines+markers",
                line={"color": "#fbbf24", "width": 2, "dash": "dot"},
                marker={"size": 7},
                hovertemplate="%{x}<br>수위: %{y:.3f} m<extra></extra>",
            ),
            secondary_y=True,
        )
    figure.update_layout(
        template="plotly_dark",
        height=390,
        margin={"l": 16, "r": 16, "t": 38, "b": 18},
        legend={"orientation": "h", "y": 1.1},
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,.35)",
        font={"color": "#cbd5e1"},
    )
    figure.update_xaxes(gridcolor="#1e293b", title_text="목표 날짜")
    figure.update_yaxes(gridcolor="#1e293b", title_text=area_title, secondary_y=False)
    figure.update_yaxes(
        gridcolor="#1e293b",
        title_text="수위 (m)" if has_level else "",
        showgrid=False,
        secondary_y=True,
    )
    st.plotly_chart(figure, width="stretch")


def render_irregular_trend_diagnostics(result: Mapping[str, Any]) -> None:
    """Explain the multi-frame area baseline from persisted adapter metadata."""

    metadata = result.get("adapter_metadata")
    if not isinstance(metadata, Mapping):
        return
    if metadata.get("strategy") != "irregular_log_area_trend":
        return
    raw_dates = metadata.get("source_dates") or []
    raw_areas = metadata.get("observed_area_pixels") or []
    raw_targets = metadata.get("targets") or []
    if not isinstance(raw_dates, list) or not isinstance(raw_areas, list):
        return
    targets = [item for item in raw_targets if isinstance(item, Mapping)]
    try:
        daily_pct = float(metadata.get("fitted_area_change_pct_per_day", 0.0))
    except (TypeError, ValueError):
        daily_pct = 0.0
    matched_weather = int(metadata.get("matched_target_weather_rows") or 0)
    capped_count = sum(bool(item.get("was_capped")) for item in targets)

    st.markdown("#### 예측 근거 · NAS 다중시점 면적 추세와 강수 보정")
    st.write(
        "청록 점은 API에 실제로 전달된 모든 과거 마스크의 수체 픽셀입니다. 주황선은 "
        "불규칙한 관측일 간격을 반영한 면적 추세에 목표일 강수 메모리를 더한 뒤, "
        "변화율 상한을 적용해 실제 출력 마스크가 갖도록 만든 목표 픽셀입니다."
    )
    metrics = st.columns(4)
    metrics[0].metric("계산에 사용한 과거", f"{len(raw_areas)}프레임")
    metrics[1].metric("적합 일 변화율", f"{daily_pct:+.4f}% / 일")
    metrics[2].metric("목표일 기상 행", f"{matched_weather} / {len(targets)}일")
    metrics[3].metric("상한 적용", f"{capped_count}일")
    precipitation_values = int(
        metadata.get("target_precipitation_value_count") or 0
    )
    st.caption(
        f"목표일 기상 행 {matched_weather}개 중 강수 숫자가 명시된 날은 "
        f"{precipitation_values}일입니다. ASOS 무강수일은 강수 필드가 null로 올 수 있으며 "
        "모델은 그 날 추가 강수 보정을 하지 않습니다."
    )

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=[str(value) for value in raw_dates],
            y=[float(value) for value in raw_areas],
            name="관측 마스크 수체 픽셀",
            mode="lines+markers",
            line={"color": "#22d3ee", "width": 3},
            marker={"size": 9},
            hovertemplate="%{x}<br>관측 %{y:,.0f} px<extra></extra>",
        ),
        secondary_y=False,
    )
    target_dates = [str(item.get("target_date") or f"D+{index + 1}") for index, item in enumerate(targets)]
    trend_values = [float(item.get("trend_target_pixels") or 0.0) for item in targets]
    final_values = [float(item.get("capped_target_pixels") or 0.0) for item in targets]
    precipitation = [float(item.get("precipitation_mm") or 0.0) for item in targets]
    figure.add_trace(
        go.Scatter(
            x=target_dates,
            y=trend_values,
            name="날짜 추세만",
            mode="lines",
            line={"color": "#a78bfa", "width": 2, "dash": "dot"},
            hovertemplate="%{x}<br>추세 %{y:,.1f} px<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=target_dates,
            y=final_values,
            name="강수·상한 반영 출력",
            mode="lines+markers",
            line={"color": "#f59e0b", "width": 3},
            marker={"size": 6},
            hovertemplate="%{x}<br>출력 목표 %{y:,.0f} px<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=target_dates,
            y=precipitation,
            name="목표일 강수",
            marker={"color": "rgba(56,189,248,.28)"},
            hovertemplate="%{x}<br>강수 %{y:.1f} mm<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_dark",
        height=410,
        margin={"l": 16, "r": 16, "t": 38, "b": 18},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,.35)",
        font={"color": "#cbd5e1"},
        barmode="overlay",
    )
    figure.update_xaxes(title_text="관측일 / 목표일", gridcolor="#1e293b")
    figure.update_yaxes(
        title_text="수체 픽셀 수", gridcolor="#1e293b", secondary_y=False
    )
    figure.update_yaxes(
        title_text="강수 (mm)", showgrid=False, rangemode="tozero", secondary_y=True
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.warning(
        "이 그래프는 계산 근거를 공개하는 기준선 진단입니다. 주황선이 매끄럽다고 정확한 "
        "수문 예측이라는 뜻은 아니며, SAR/광학 경계의 공간 변화와 유량 물리는 학습하지 않습니다."
    )
    with st.expander("일별 추세·강수·상한 계산표", expanded=False):
        st.dataframe(
            [
                {
                    "목표 날짜": item.get("target_date"),
                    "D+일": item.get("days_ahead"),
                    "추세 픽셀": item.get("trend_target_pixels"),
                    "강수 (mm)": item.get("precipitation_mm"),
                    "강수 메모리 (mm)": item.get("rainfall_memory_mm"),
                    "강수 보정 픽셀": item.get("weather_adjustment_pixels"),
                    "상한 전 픽셀": item.get("raw_target_pixels"),
                    "최종 목표 픽셀": item.get("capped_target_pixels"),
                    "상한 적용": bool(item.get("was_capped")),
                }
                for item in targets
            ],
            width="stretch",
            hide_index=True,
        )


def render_result_phase() -> None:
    st.subheader("결과 · 시계열 예측")
    render_page_help("result", expanded=False)
    render_learning_help("detection", expanded=False)
    result = st.session_state.get("prediction_result")
    if not result:
        st.warning("아직 예측 결과가 없습니다.")
        return

    prediction_id = str(result.get("id") or result.get("prediction_id") or "")
    controls = st.columns([4, 1])
    controls[0].caption(f"예측 ID: {prediction_id or '-'}")
    if controls[1].button(
        "상태 새로고침",
        icon=":material/refresh:",
        disabled=not prediction_id,
        width="stretch",
    ):
        try:
            st.session_state.prediction_result = dict(api_client().get_prediction(prediction_id))
            st.rerun()
        except APIError as exc:
            st.error(format_api_error(exc))

    status = str(result.get("status", "unknown"))
    steps = extract_prediction_steps(result)
    context = st.session_state.get("prediction_context") or {}
    pixel_area_placeholder = bool(context.get("pixel_area_placeholder"))

    table_rows: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        table_rows.append(
            {
                "frame": first_not_none(step, "step", "horizon", default=index),
                "target_date": first_not_none(step, "target_date", "date", default="-"),
                "water_area_pixels": step.get("water_area_pixels"),
                "water_area_km2": first_not_none(step, "water_area_km2", "area_km2"),
                "water_level_m": step.get("water_level_m"),
                "water_level_source": step.get("water_level_source", "unavailable"),
                "reference_water_level_m": step.get("reference_water_level_m"),
                "water_level_error_m": step.get("water_level_error_m"),
                "area_change_pct": first_not_none(step, "area_change_pct", "change_pct"),
                "risk": first_not_none(step, "risk", "risk_level", default="normal"),
                "mask": first_not_none(
                    step,
                    "preview_png_url",
                    "mask_tif_url",
                    "mask_npy_url",
                    "artifact_name",
                    "mask_artifact",
                    "mask_url",
                    default="-",
                ),
            }
        )

    latest = table_rows[-1] if table_rows else {}
    latest_change = latest.get("area_change_pct")
    if pixel_area_placeholder:
        latest_area = (
            f"{int(latest['water_area_pixels']):,} px"
            if latest.get("water_area_pixels") is not None
            else "-"
        )
        latest_area_label = "수체 픽셀"
    else:
        latest_area = (
            f"{float(latest['water_area_km2']):,.4f} km²"
            if latest.get("water_area_km2") is not None
            else "-"
        )
        latest_area_label = "최종 수체 면적"
    latest_level = (
        f"{float(latest['water_level_m']):.3f} m"
        if latest.get("water_level_m") is not None
        else "미산출"
    )
    metrics = st.columns(5)
    metrics[0].metric("실행 상태", {"completed": "완료", "failed": "실패"}.get(status, status))
    metrics[1].metric("모델", result.get("model_id", "-"))
    metrics[2].metric(
        latest_area_label,
        latest_area,
        delta=f"{float(latest_change):+.2f}%" if latest_change is not None else None,
    )
    metrics[3].metric("예측 수위", latest_level)
    metrics[4].metric("위험도", risk_label(latest.get("risk", "-")))
    st.caption(
        f"생성 {str(result.get('created_at', '-'))[:19]} · 결과 {len(steps)}프레임"
    )

    if status.lower() in {"failed", "error"}:
        st.error(result.get("error") or "백엔드 예측이 실패했습니다.")
    if result.get("model_id") in {
        "persistence",
        "irregular-area-trend",
        "weather-morphology",
    }:
        st.info(
            "현재 결과는 연동 확인용 내장 기준선입니다. 학습된 수문 예측 모델의 성능 결과가 아닙니다. "
            "실제 모델 어댑터를 등록하면 같은 화면과 API 계약을 그대로 사용합니다."
        )
    if context.get("limitations"):
        st.warning(str(context["limitations"]))
    for warning in result.get("warnings", []):
        st.warning(str(warning))

    render_irregular_trend_diagnostics(result)

    if table_rows:
        parameters = result.get("parameters")
        history = (
            parameters.get("historical_water_levels_m", [])
            if isinstance(parameters, Mapping)
            else []
        )
        has_level_series = any(
            value is not None
            for value in [
                *history,
                *(row.get("water_level_m") for row in table_rows),
                *(row.get("reference_water_level_m") for row in table_rows),
            ]
        )
        if has_level_series:
            st.markdown("#### 일별 수위 전망 · 1주 / 2주 / 1개월")
            st.write(
                "마지막 관측 수위에서 이어지는 미래 일별 수위를 한 화면에서 봅니다. "
                "D+7·D+14·D+30 카드는 정확히 해당 날짜의 값이며, 그래프 버튼은 표시 범위만 바꿉니다."
            )
            render_horizon_level_cards(result, table_rows)
            render_daily_water_level_chart(
                result,
                table_rows,
                prediction_id=prediction_id,
            )
            render_water_level_evaluation(result)
        else:
            st.markdown("#### 일별 수체면적 전망 · 1주 / 2주 / 1개월")
            st.info(
                "이 NAS 세트에는 같은 AOI의 실측 수위와 검증된 면적–수위 환산식이 없어 "
                "수위를 임의 생성하지 않았습니다. 아래 그래프의 일별 수체 픽셀·면적과 "
                "D+7·D+14·D+30 결과를 예측 출력으로 해석하세요."
            )

    active_sample: SampleDataset | None = None
    if context.get("sample_id"):
        try:
            active_sample = get_sample(str(context["sample_id"]))
        except KeyError:
            active_sample = None
    render_result_eda(
        steps,
        sample=active_sample,
        pixel_area_known=not pixel_area_placeholder,
        show_chart=False,
    )

    if table_rows:
        st.markdown("#### 수체 범위 예측 · 마스크 면적 추이")
        if pixel_area_placeholder:
            st.caption(
                "픽셀 면적이 검증되지 않은 샘플이므로 차트는 물리 면적 대신 수체 픽셀 수를 표시합니다."
            )
        render_horizon_area_cards(
            table_rows, pixel_area_placeholder=pixel_area_placeholder
        )
        render_prediction_chart(table_rows, pixel_area_placeholder=pixel_area_placeholder)

        st.markdown("#### 프레임별 면적·수위")
        steps_frame = pd.DataFrame(table_rows)
        display_frame = steps_frame.drop(columns=["mask"]).copy()
        export_frame = steps_frame.copy()
        if pixel_area_placeholder:
            display_frame = display_frame.drop(columns=["water_area_km2"])
            export_frame = export_frame.drop(columns=["water_area_km2"])
            st.caption(
                "픽셀 면적이 검증되지 않아 표와 CSV에서 km² 열을 숨겼습니다. "
                "원본 API JSON에는 연동 가정값으로 계산된 필드가 남아 있으므로 물리량으로 해석하지 마세요."
            )
        display_frame["risk"] = display_frame["risk"].map(risk_label)
        st.dataframe(
            display_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "frame": "프레임",
                "target_date": "목표 날짜",
                "water_area_pixels": st.column_config.NumberColumn("수체 픽셀", format="%d"),
                "water_area_km2": st.column_config.NumberColumn("수체 면적(km²)", format="%.6f"),
                "water_level_m": st.column_config.NumberColumn("수위(m)", format="%.3f"),
                "water_level_source": "수위 산출 출처",
                "reference_water_level_m": st.column_config.NumberColumn(
                    "평가 기준 수위(m)", format="%.3f"
                ),
                "water_level_error_m": st.column_config.NumberColumn(
                    "수위 오차(m)", format="%+.3f"
                ),
                "area_change_pct": st.column_config.NumberColumn("면적 변화(%)", format="%.2f"),
                "risk": "위험도",
            },
        )
        st.download_button(
            "프레임 결과 CSV 다운로드",
            export_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"prediction_{prediction_id or 'result'}_steps.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        if all(row["water_level_m"] is None for row in table_rows):
            st.caption(
                "수위 값이 null입니다. 오류를 숨긴 값이 아니라 선택 모델/보정 설정이 수위를 산출하지 않았다는 뜻입니다."
            )
    else:
        st.info("완료된 프레임이 아직 없습니다. 비동기 작업이면 상태 새로고침을 누르세요.")

    st.markdown("#### 산출물 미리보기 및 다운로드")
    artifacts = artifact_entries(result, steps)
    if artifacts and prediction_id:
        preview_column, artifact_column = st.columns([3, 1])
        with preview_column:
            selected = st.selectbox("산출물", artifacts, format_func=lambda item: item["label"])
            cache = st.session_state.setdefault("artifact_cache", {})
            suffix = Path(selected["name"]).suffix.lower()
            error_key = f"artifact_error_{prediction_id}_{selected['name']}"
            if selected["name"] not in cache and suffix in {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
            }:
                try:
                    cache[selected["name"]] = api_client().download_artifact(
                        prediction_id, selected["name"]
                    )
                    st.session_state.pop(error_key, None)
                except APIError as exc:
                    st.session_state[error_key] = format_api_error(exc)
            content = cache.get(selected["name"])
            if content and suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                try:
                    st.image(content, caption=selected["label"], width="stretch")
                except Exception:  # noqa: BLE001 - unsupported browser preview falls back to guidance
                    st.info(
                        "브라우저 미리보기를 지원하지 않는 영상 형식입니다. 파일을 내려받아 GIS 도구에서 확인하세요."
                    )
            elif not content:
                st.info("이 산출물은 미리보기 대신 다운로드로 확인하세요.")

        with artifact_column:
            st.markdown("##### 선택 산출물")
            st.caption(selected["name"])
            if st.session_state.get(error_key):
                st.error(st.session_state[error_key])
            if not content and st.button(
                "파일 불러오기",
                icon=":material/refresh:",
                width="stretch",
            ):
                try:
                    cache[selected["name"]] = api_client().download_artifact(
                        prediction_id, selected["name"]
                    )
                    st.session_state.pop(error_key, None)
                    st.rerun()
                except APIError as exc:
                    st.error(format_api_error(exc))
            if content:
                st.download_button(
                    "선택 파일 다운로드",
                    content,
                    file_name=selected["name"],
                    mime="application/octet-stream",
                    icon=":material/download:",
                    width="stretch",
                )

            if st.button(
                "전체 ZIP 준비",
                icon=":material/folder_zip:",
                width="stretch",
            ):
                try:
                    st.session_state.bundle_cache = api_client().download_bundle(prediction_id)
                except APIError as exc:
                    st.error(format_api_error(exc))
            if st.session_state.get("bundle_cache"):
                st.download_button(
                    "전체 ZIP 다운로드",
                    st.session_state.bundle_cache,
                    file_name=f"prediction_{prediction_id}.zip",
                    mime="application/zip",
                    icon=":material/download:",
                    width="stretch",
                )
    else:
        st.caption("응답에 다운로드 가능한 artifact 이름이 없습니다.")

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "전체 결과 JSON 다운로드",
        json.dumps(result, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=f"prediction_{prediction_id or 'result'}.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
    )
    weather_frame = records_to_frame(st.session_state.get("weather_data", []))
    download_cols[1].download_button(
        "사용 기상 CSV 다운로드",
        weather_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"prediction_{prediction_id or 'result'}_weather.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )


def flow_diagram_html() -> str:
    return """
    <div style="font-family:system-ui,sans-serif;background:#080f1f;border:1px solid #1e293b;border-radius:8px;padding:18px;overflow-x:auto">
      <svg width="1120" height="210" viewBox="0 0 1120 210" style="max-width:100%;height:auto" role="img" aria-label="API 데이터 흐름">
        <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>
        <style>.box{fill:#0f172a;stroke:#334155;stroke-width:2}.accent{fill:#082f49;stroke:#0ea5e9;stroke-width:2}.title{font-weight:700;font-size:15px;fill:#e2e8f0}.sub{font-size:11px;fill:#94a3b8}.line{stroke:#475569;stroke-width:2;marker-end:url(#arrow)}</style>
        <rect class="box" x="10" y="48" rx="13" width="180" height="90"/><text class="title" x="100" y="80" text-anchor="middle">이노뎁 전처리</text><text class="sub" x="100" y="103" text-anchor="middle">mask files + dates</text><text class="sub" x="100" y="121" text-anchor="middle">optional water levels</text>
        <line class="line" x1="190" y1="93" x2="235" y2="93"/>
        <rect class="box" x="238" y="22" rx="13" width="190" height="142"/><text class="title" x="333" y="57" text-anchor="middle">Streamlit UI</text><text class="sub" x="333" y="82" text-anchor="middle">입력 검증 · 상태 무효화</text><text class="sub" x="333" y="101" text-anchor="middle">ASOS/시나리오 편집</text><text class="sub" x="333" y="120" text-anchor="middle">결과·산출물 다운로드</text>
        <line class="line" x1="428" y1="93" x2="473" y2="93"/>
        <rect class="accent" x="476" y="22" rx="13" width="205" height="142"/><text class="title" x="578" y="57" text-anchor="middle">FastAPI /api/v1</text><text class="sub" x="578" y="82" text-anchor="middle">weather/observations</text><text class="sub" x="578" y="101" text-anchor="middle">predictions/{id}</text><text class="sub" x="578" y="120" text-anchor="middle">files/{name} · bundle</text>
        <line class="line" x1="681" y1="72" x2="726" y2="72"/>
        <rect class="box" x="729" y="22" rx="13" width="170" height="90"/><text class="title" x="814" y="55" text-anchor="middle">모델 어댑터</text><text class="sub" x="814" y="80" text-anchor="middle">모델 ID로 교체</text><text class="sub" x="814" y="98" text-anchor="middle">mask → future frames</text>
        <line class="line" x1="681" y1="132" x2="726" y2="151"/>
        <rect class="box" x="729" y="125" rx="13" width="170" height="70"/><text class="title" x="814" y="155" text-anchor="middle">기상청 ASOS</text><text class="sub" x="814" y="178" text-anchor="middle">과거 관측 D-1까지</text>
        <line class="line" x1="899" y1="72" x2="944" y2="93"/>
        <rect class="box" x="947" y="48" rx="13" width="165" height="90"/><text class="title" x="1029" y="80" text-anchor="middle">예측·평가 결과</text><text class="sub" x="1029" y="103" text-anchor="middle">D+7·14·30 · RMSE</text><text class="sub" x="1029" y="121" text-anchor="middle">JSON + artifacts</text>
      </svg>
    </div>
    """


def render_api_guide_phase() -> None:
    st.subheader("API 가이드 · 외부 시스템 연계")
    render_page_help("api", expanded=False)
    st.write("외부 시스템은 UI를 거치지 않고 같은 REST API로 모델 목록, 예측 결과, 마스크 산출물을 가져갈 수 있습니다.")
    st.html(flow_diagram_html())

    if "openapi_spec" not in st.session_state and "openapi_error" not in st.session_state:
        try:
            short_client = APIClient(st.session_state.api_base_url, timeout=min(float(st.session_state.api_timeout), 4.0))
            st.session_state.openapi_spec = dict(short_client.get_openapi())
        except APIError as exc:
            st.session_state.openapi_error = str(exc)

    spec = st.session_state.get("openapi_spec")
    rows = openapi_operation_rows(spec) if spec else FALLBACK_OPERATIONS
    source = "백엔드 /openapi.json 실시간 명세" if spec else "UI 내장 fallback 명세"
    st.markdown(f"#### 엔드포인트 · {source}")
    if st.session_state.get("openapi_error"):
        st.caption(f"OpenAPI를 읽지 못해 fallback을 표시합니다: {st.session_state.openapi_error}")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    link_cols = st.columns(3)
    link_cols[0].link_button("Swagger UI", api_client().absolute_url(Endpoints.DOCS), width="stretch")
    link_cols[1].link_button("ReDoc", api_client().absolute_url(Endpoints.REDOC), width="stretch")
    link_cols[2].link_button("OpenAPI JSON", api_client().absolute_url(Endpoints.OPENAPI), width="stretch")
    if spec:
        st.download_button(
            "현재 OpenAPI 명세 다운로드",
            json.dumps(spec, ensure_ascii=False, indent=2).encode("utf-8"),
            "openapi.json",
            "application/json",
        )

    base = st.session_state.api_base_url.rstrip("/")
    st.markdown("#### 호출 순서")
    st.code(
        f"""# 1) 사용 가능한 모델 확인
curl -s '{base}{Endpoints.MODELS}'

# 2) 과거 ASOS 관측 조회 (미래 예보가 아님)
curl -X POST '{base}{Endpoints.WEATHER_OBSERVATIONS}' \\
  -H 'Content-Type: application/json' \\
  -d '{{"source":"asos","station_id":"159","start_date":"2026-07-01","end_date":"2026-07-31"}}'

# 3) 예측 생성: 동일 격자의 마스크를 날짜 순서대로 첨부
curl -X POST '{base}{Endpoints.PREDICTIONS}' \\
  -F 'files=@mask_20260701.tif' \\
  -F 'files=@mask_20260711.tif' \\
  -F 'model_id=persistence' \\
  -F 'horizon_steps=30' \\
  -F 'threshold=0.5' \\
  -F 'pixel_area_m2=9' \\
  -F 'source_dates_json=["2026-07-01","2026-07-11"]' \\
  -F 'input_metadata_json={{"dataset_id":"my-water-mask-series","sensor":"SAR","preprocess_version":"v1.2"}}' \\
  -F 'historical_water_levels_json=[null,null]' \\
  -F 'model_options_json={{}}' \\
  -F 'caution_pct=5' \\
  -F 'risk_pct=15' \\
  -F 'weather_json=[]'

# 4) 결과 조회 / 산출물 다운로드
curl -s '{base}/api/v1/predictions/<prediction_id>'
curl -OJ '{base}/api/v1/predictions/<prediction_id>/files/<name>'
curl -OJ '{base}/api/v1/predictions/<prediction_id>/bundle'""",
        language="bash",
    )

    with st.expander("Python requests 예제", expanded=False):
        st.code(
            f'''import json
import requests

base_url = "{base}"
files = [
    ("files", ("mask_1.tif", open("mask_1.tif", "rb"), "image/tiff")),
    ("files", ("mask_2.tif", open("mask_2.tif", "rb"), "image/tiff")),
]
form = {{
    "model_id": "persistence",
    "horizon_steps": "30",
    "threshold": "0.5",
    "pixel_area_m2": "9",
    "source_dates_json": json.dumps(["2026-07-01", "2026-07-11"]),
    "input_metadata_json": json.dumps({{
        "dataset_id": "my-water-mask-series",
        "sensor": "SAR",
        "preprocess_version": "v1.2",
    }}),
    "historical_water_levels_json": json.dumps([None, None]),
    "model_options_json": json.dumps({{}}),
    "caution_pct": "5",
    "risk_pct": "15",
    "weather_json": json.dumps([]),
}}
created = requests.post(f"{{base_url}}/api/v1/predictions", files=files, data=form).json()
result = requests.get(f"{{base_url}}/api/v1/predictions/{{created['id']}}").json()
artifact = requests.get(f"{{base_url}}/api/v1/predictions/{{created['id']}}/files/{{result['artifacts'][0]['name']}}")
artifact.raise_for_status()
open("predicted_mask.tif", "wb").write(artifact.content)''',
            language="python",
        )

    st.markdown("#### 응답 해석")
    st.markdown(
        "- `steps[]`: D+1~D+30 목표 날짜, 수체 면적, 예측 수위, 평가 기준 수위·오차, 위험도, 마스크 산출물 참조\n"
        "- `input.source_metadata`: 요청의 `input_metadata_json`을 그대로 보존한 데이터셋 ID·센서·전처리 계보입니다. 비밀번호나 API 키는 넣지 않습니다.\n"
        "- `water_level_m`: 모델 또는 면적-수위 보정이 없으면 `null`이 정상입니다. 임의 값으로 채우지 않습니다.\n"
        "- `reference_water_levels_json`: horizon과 같은 길이의 선택 정답 수위입니다. predictor 입력이 아니라 추론 후 평가에만 씁니다.\n"
        "- `evaluation`: 정답과 예측이 함께 있는 날짜의 MAE·RMSE·MAPE·R²·Bias와 일별 pair입니다. 정답이 없으면 `null`입니다.\n"
        "- `artifacts[]`: 외부 시스템이 별도 다운로드할 GeoTIFF/PNG/NPY 등 결과 파일\n"
        "- UI는 파일당 단일 2D 마스크만 받습니다. API에서 3D `[T,H,W]` NPY를 보내면 `source_dates_json`과 "
        "`historical_water_levels_json`은 업로드 파일 수가 아니라 디코딩된 T 프레임 수에 맞춰야 합니다.\n"
        "- 정확한 필드와 오류 코드는 실행 중인 서버의 OpenAPI/Swagger 문서를 기준으로 합니다."
    )


def format_api_error(exc: APIError) -> str:
    prefix = f"HTTP {exc.status_code} · " if exc.status_code else ""
    if exc.status_code == 422:
        return prefix + f"요청 형식 검증에 실패했습니다. 서버 OpenAPI와 입력값을 확인하세요. ({exc})"
    return prefix + str(exc)


def render_understanding_guide_phase() -> None:
    """Combine the technical learning center and the screen-by-screen manual."""

    st.subheader("이해 가이드 · 원천자료부터 결과 해석까지")
    st.caption(
        "처음 보는 사람은 ‘수체·전처리·모델 이해’부터, 화면 조작과 발표를 준비할 때는 "
        "‘버튼·그래프·발표 가이드’를 보세요."
    )
    learning_tab, usage_tab = st.tabs(
        ["수체·전처리·모델 이해", "버튼·그래프·발표 가이드"]
    )
    with learning_tab:
        render_learning_page()
    with usage_tab:
        render_guide_page()


def main() -> None:
    configure_page()
    if "service_mode" not in st.session_state:
        st.session_state.service_mode = SERVICE_MODES[0]
    if st.session_state.service_mode == "수체 시계열 예측":
        render_transition_overlay()
    else:
        st.session_state.pop("transition_phase", None)
    render_sidebar()
    render_header()
    service_mode = render_service_switch()
    if service_mode == "작물 탐지":
        render_crop_detection_page()
        return
    render_phase_navigation()
    if st.session_state.get("sample_loaded_toast"):
        st.toast(st.session_state.pop("sample_loaded_toast"))
    phase = int(st.session_state.phase)
    renderers = (
        render_input_phase,
        render_weather_phase,
        render_prediction_phase,
        render_result_phase,
        render_model_eval_page,
        render_understanding_guide_phase,
        render_api_guide_phase,
    )
    renderers[phase]()


if __name__ == "__main__":
    main()
