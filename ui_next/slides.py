"""슬라이드형 기본 화면 — 누구나 순서대로 넘기며 이해하는 여섯 장.

1 지역 선택 → 2 데이터 둘러보기 → 3 위성 수체 탐지(사진) → 4 시계열 예측(모델 선택·기상)
→ 5 결과 그래프 → 6 성능 평가. 기존 상세 화면(정량 평가·API 가이드·직접 업로드 등)은
"고급 화면"으로 그대로 남기고, 이 모듈은 같은 백엔드 API만 부른다. 숫자를 만들거나
고정하지 않는다 — 전부 지역 레지스트리·학습 산출물·예측 API에서 온다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api_client import APIClient, APIError
from eda import padded_range
from geo_overlay import location_table_fields, map_figure, overlay_figure
from area_models_ui import render_area_models_experiment
from future_rain_ui import render_future_rain_experiment
from gauge_forecast_ui import (
    DEFAULT_REGION_ID,
    _backtest_figure,
    _cached,
    _forecast_figure,
    _series_figure,
    evaluation_pivot,
    forecaster_label,
    skill_text,
    terrain_label,
    winners_table,
)
from model_compare_ui import (  # 기존 이름(_compare_figure 등)은 여기서 재수출한다
    MODEL_COLORS,  # noqa: F401
    _compare_figure,  # noqa: F401
    compare_difference_rows,  # noqa: F401
    compare_metric_rows,  # noqa: F401
    render_compare_payload,
    render_model_compare,
)
from presentation_ui import data_status_badges, detection_summary
from samples import SampleDataset, list_samples
from satellite_input_ui import render_satellite_input
from labels import area_model_label
from tables import show_table, static_table
from training_window_ui import active_split as _active_split
from training_window_ui import render_split_badge, render_training_window


SLIDES: tuple[tuple[str, str, str], ...] = (
    ("region", "지역 선택", "어느 지역을 볼까요?"),
    ("data", "데이터 둘러보기", "이 지역에는 어떤 자료가 있나요?"),
    ("detect", "위성 수체 탐지", "위성 영상에서 물을 찾은 결과"),
    ("forecast", "시계열 예측", "앞으로 수위와 수체 면적은?"),
    ("result", "결과 그래프", "예측 결과를 한눈에"),
    ("evaluate", "성능 평가", "예측이 얼마나 맞았나"),
)
SLIDE_KEYS = tuple(key for key, _, _ in SLIDES)
REPORT_HORIZONS = (1, 3, 5, 7, 10, 14, 30)
HIDDEN_AREA_MODELS = frozenset({"persistence"})

SLIDES_CSS = """
<style>
.sl-rail { display:flex; gap:6px; margin:4px 0 14px; flex-wrap:wrap; }
.sl-pill { flex:1 1 120px; text-align:center; border-radius:10px; padding:8px 10px; font-size:.82rem;
  font-weight:800; border:1px solid #334155; color:#7c8ea3; background:#0b1324; }
.sl-pill--done { border-color:rgba(52,211,153,.5); color:#6ee7b7; background:rgba(6,78,59,.2); }
.sl-pill--now { border-color:rgba(34,211,238,.7); color:#67e8f9; background:rgba(8,145,178,.22); box-shadow:0 0 0 2px rgba(34,211,238,.15); }
.sl-hero { padding:22px 26px; border:1px solid #334155; border-left:6px solid #22d3ee; border-radius:14px;
  background:linear-gradient(90deg, rgba(8,47,73,.5), rgba(2,6,23,.15)); margin:2px 0 16px; }
.sl-hero .k { color:#67e8f9; font-size:.8rem; font-weight:800; letter-spacing:.12em; }
.sl-hero h1 { margin:4px 0 6px!important; color:#f0f9ff!important; font-size:1.9rem!important; line-height:1.2; }
.sl-hero p { margin:0; color:#cbd5e1; font-size:1.05rem; line-height:1.6; }
.sl-explain { margin:10px 0 16px; padding:14px 18px; border-radius:12px; border:1px solid rgba(56,189,248,.35);
  background:rgba(8,47,73,.28); color:#e0f2fe; line-height:1.7; font-size:.98rem; }
.sl-explain b { color:#7dd3fc; }
.sl-explain ul { margin:6px 0 0 18px; padding:0; }
.sl-card { padding:16px 18px; border:1px solid #334155; border-radius:14px; background:#0b1324; min-height:210px; }
.sl-card h3 { margin:0 0 6px!important; color:#e0f2fe!important; font-size:1.15rem!important; }
.sl-card p { margin:0 0 10px; color:#94a3b8; line-height:1.55; font-size:.92rem; }
.sl-badges { display:flex; gap:6px; flex-wrap:wrap; }
.sl-badge { border-radius:999px; padding:3px 10px; font-size:.74rem; font-weight:700; border:1px solid #334155; color:#7c8ea3; }
.sl-badge--on { border-color:rgba(52,211,153,.55); color:#6ee7b7; background:rgba(6,78,59,.22); }
.sl-badge--off { border-color:rgba(248,113,113,.4); color:#fca5a5; background:rgba(127,29,29,.15); }
.sl-note { margin:8px 0 12px; padding:10px 14px; border-left:3px solid #f59e0b;
  background:rgba(120,53,15,.16); color:#fde68a; line-height:1.6; border-radius:0 6px 6px 0; font-size:.9rem; }
.sl-none { margin:8px 0 12px; padding:14px 16px; border:1px dashed #334155; border-radius:10px; color:#94a3b8; line-height:1.6; }
.sl-caption { color:#64748b; font-size:.8rem; }
.sl-big { font-size:2.2rem; font-weight:900; color:#67e8f9; line-height:1; }
.sl-big small { font-size:.9rem; color:#94a3b8; font-weight:600; margin-left:6px; }
</style>
"""

_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(2,6,23,.42)",
    "font": {"color": "#cbd5e1", "size": 13},
    "hovermode": "x unified",
    "margin": {"l": 18, "r": 18, "t": 52, "b": 22},
}


# --- 상태·내비게이션 ---------------------------------------------------------------------


def slide_index(key: str) -> int:
    return SLIDE_KEYS.index(key)


def current_slide() -> int:
    raw = st.session_state.get("slide", 0)
    try:
        index = int(raw)
    except (TypeError, ValueError):
        index = 0
    index = max(0, min(index, len(SLIDES) - 1))
    st.session_state.slide = index
    return index


def go_to_slide(index: int) -> None:
    st.session_state.slide = max(0, min(int(index), len(SLIDES) - 1))
    st.rerun()


def selected_region_id() -> str | None:
    return st.session_state.get("slide_region")


def _rail(current: int, *, skipped: set[str] | None = None) -> None:
    chips = []
    for index, (key, title, _sub) in enumerate(SLIDES):
        variant = "sl-pill--now" if index == current else ("sl-pill--done" if index < current else "")
        if skipped and key in skipped and index != current:
            variant, mark = "", "– "
            title = f"{title} (위성 없음)"
        else:
            mark = "✓ " if index < current else f"{index + 1}. "
        chips.append(f'<span class="sl-pill {variant}">{mark}{title}</span>')
    st.markdown(f'<div class="sl-rail">{"".join(chips)}</div>', unsafe_allow_html=True)


def _hero(current: int, region_name: str | None) -> None:
    _key, title, sub = SLIDES[current]
    kicker = f"STEP {current + 1} / {len(SLIDES)}" + (f" · {region_name}" if region_name else "")
    st.markdown(
        f'<div class="sl-hero"><div class="k">{kicker}</div><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )


def _explain(html: str) -> None:
    st.markdown(f'<div class="sl-explain">{html}</div>', unsafe_allow_html=True)


def _note(text: str) -> None:
    st.markdown(f'<div class="sl-note">{text}</div>', unsafe_allow_html=True)


def _none(text: str) -> None:
    st.markdown(f'<div class="sl-none">{text}</div>', unsafe_allow_html=True)


def _nav(current: int, *, next_enabled: bool = True, next_label: str | None = None) -> None:
    st.divider()
    cols = st.columns([1, 4, 1])
    if current > 0 and cols[0].button("◀ 이전", key=f"sl_prev_{current}", width="stretch"):
        go_to_slide(current - 1)
    if current < len(SLIDES) - 1:
        label = next_label or f"다음: {SLIDES[current + 1][1]} ▶"
        if cols[2].button(label, key=f"sl_next_{current}", type="primary", width="stretch", disabled=not next_enabled):
            go_to_slide(current + 1)
    else:
        if cols[2].button("처음으로 ↺", key=f"sl_restart_{current}", width="stretch"):
            go_to_slide(0)


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "-"


def _region(client: APIClient, region_id: str | None) -> Mapping[str, Any] | None:
    regions = _cached("regions", client.list_regions)
    return next((r for r in regions if r["id"] == region_id), None)


def _satellite_locations(client: APIClient, region: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """위성 수위 산출 구역(AOI) 사전 — 지역 요약에 없으면 상세 조회로 한 번 더 본다. 없으면 {}."""

    locations = region.get("satellite_locations")
    if not isinstance(locations, Mapping):
        try:
            detail = _cached(f"region_detail:{region['id']}", lambda: client.get_region(region["id"]))
        except APIError:
            detail = {}
        locations = detail.get("satellite_locations") if isinstance(detail, Mapping) else None
    return {str(k): v for k, v in (locations or {}).items() if isinstance(v, Mapping)}


def _station_map(client: APIClient, region: Mapping[str, Any], *, scene: Mapping[str, Any] | None = None, height: int = 400, title: str = "") -> None:
    """지역의 수위계 지점(주소·하천 hover)과 위성 수위 산출 구역, 선택 장면의 발자국을 지도에 그린다."""

    stations = region.get("stations") or []
    locations = _satellite_locations(client, region)
    footprint = ((scene or {}).get("quicklook") or {}).get("footprint_lonlat") if scene else None
    with_coords = [s for s in stations if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lon"), (int, float))]
    if not with_coords and not locations:
        _none("지도에 찍을 좌표가 아직 없습니다 — 수위계 좌표(WAMIS 지점정보)나 위성 산출 구역이 등록되면 여기에 지도가 나옵니다.")
        return
    st.plotly_chart(map_figure(region, locations, stations, footprint, height=height, title=title), width="stretch", config={"displaylogo": False, "scrollZoom": True})
    if any(s.get("satellite_location") for s in with_coords):
        bits = [f"주황·회색 점 = 지상 수위계 {len(with_coords)}지점(주황 = 위성 산출 구역과 연결)"]
    else:  # 위성 없는 지역 — 주황 점이 없는데 범례에 적으면 헷갈린다
        bits = [f"회색 점 = 지상 수위계 {len(with_coords)}지점"]
    if locations:
        bits.append(f"청록 상자 = 위성 수위 산출 구역 {len(locations)}곳(구역 크기 수백 m라 확대해야 상자로 보입니다)")
    if footprint:
        bits.append("분홍 테두리 = 선택한 위성 장면이 덮는 범위")
    st.markdown('<div class="sl-caption">' + " · ".join(bits) + "</div>", unsafe_allow_html=True)


def region_samples_for(region_id: str) -> list[SampleDataset]:
    return [s for s in list_samples() if getattr(s, "region_id", None) == region_id]


def _guide(client: APIClient) -> Mapping[str, Any]:
    try:
        return _cached("model_guide", client.gauge_model_guide)
    except APIError:
        return {}


def _model_label_with_weather(model_id: str, guide: Mapping[str, Any]) -> str:
    label = forecaster_label(model_id)
    entry = next((f for f in guide.get("forecasters") or [] if f.get("id") == model_id), None)
    if entry and entry.get("uses_weather"):
        label += " · 기상 사용"
    return label


def _cm(value: Any, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f} cm" if isinstance(value, (int, float)) else "-"


def _cm2(value: Any, digits: int = 1) -> str:
    return f"{value * 1e4:.{digits}f} cm²" if isinstance(value, (int, float)) else "-"


MODEL_FIGURES = {
    "ridge_ar": "model_ridge_ar.png", "ridge_arx": "model_ridge_arx.png", "seasonal_anomaly_ar": "model_seasonal_anomaly.png",
    "mean_reversion": "model_mean_reversion.png", "analog_knn": "model_analog_knn.png", "ensemble_mean": "model_ensemble.png",
    "neural_mlp": "model_neural.png", "neural_lstm": "model_neural.png", "neural_gru": "model_neural.png", "neural_tcn": "model_neural.png",
    "prophet_hybrid": "model_prophet_hybrid.png",
}
_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "models"


def _figure_bytes(name: str) -> bytes | None:
    path = _ASSET_DIR / name
    try:
        return path.read_bytes()
    except OSError:
        return None


def render_model_guide(guide: Mapping[str, Any], *, key: str) -> None:
    """모델·지표 설명 상자 — 문구는 /gauge/model-guide, 그림은 ui_next/assets/models."""

    if not guide:
        return
    with st.expander("모델은 어떤 것들이고, 딥러닝·GPU는 어떻게 썼나요?", expanded=False):
        skill = guide.get("skill") or {}
        st.markdown(f"**{skill.get('name_ko', '개선율')}** — `{skill.get('formula', '')}`")
        st.write(skill.get("plain_ko", ""))
        st.caption(skill.get("why_ko", ""))
        protocol = _figure_bytes("selection_protocol.png")
        if protocol:
            st.image(protocol, width="stretch")
        st.markdown("**모델은 어떻게 고르나요?**")
        st.write(guide.get("selection_ko", ""))
        seen_figures: set[str] = set()
        for f in guide.get("forecasters") or []:
            fid = str(f.get("id"))
            title = f"{f.get('name_ko')} (`{fid}`)" + (" · 기상 입력" if f.get("uses_weather") else "") + (" · 기준선(선택 안 함)" if f.get("baseline") else "")
            st.markdown(f"**{title}**")
            st.write(f.get("plain_ko", ""))
            figure = MODEL_FIGURES.get(fid)
            if figure and figure not in seen_figures:
                data = _figure_bytes(figure)
                if data:
                    st.image(data, width="stretch")
                seen_figures.add(figure)
        st.markdown("**기상(강수·기온)은 어떻게 쓰나요?**")
        st.write(guide.get("weather_ko", ""))
        st.markdown("**딥러닝(LSTM 등)과 GPU는?**")
        for line in guide.get("why_no_lstm_ko") or []:
            st.markdown(f"- {line}")
        gpu = _figure_bytes("gpu_vs_cpu.png")
        if gpu:
            st.image(gpu, width="content")


# --- 1. 지역 선택 -----------------------------------------------------------------------------


def _slide_region(client: APIClient) -> None:
    _explain(
        "<b>이 플랫폼은 지역 단위로 움직입니다.</b> 지역에 <b>위성 영상</b>이 있으면 물이 있는 곳(수체)을 찾아 면적과 수위를 "
        "재고, <b>지상 수위계 기록</b>이 있으면 그것으로 앞날의 수위를 예측하는 모델을 만듭니다. 둘 다 있으면 위성이 잰 수위와 "
        "수위계를 같은 날 맞대어 봅니다. 카드를 눌러 지역을 고르세요."
    )
    try:
        regions = _cached("regions", client.list_regions)
    except APIError as exc:
        st.error(str(exc))
        return
    if not regions:
        _none("등록된 지역이 없습니다.")
        return
    ordered = sorted(regions, key=lambda r: (not r.get("satellite_available"), r["id"] != DEFAULT_REGION_ID, r["name"]))
    # 지역이 늘어도 카드가 좁아지지 않게 한 줄에 최대 4장씩 배치한다
    per_row = 4 if len(ordered) > 4 else max(1, len(ordered))
    columns = [column for start in range(0, len(ordered), per_row) for column in st.columns(per_row)]
    for column, region in zip(columns, ordered):
        with column:
            badges = "".join(
                f'<span class="sl-badge {"sl-badge--on" if on else "sl-badge--off"}">{"✓" if on else "✗"} {label}</span>'
                for label, on in data_status_badges(region)
            )
            st.markdown(
                f'<div class="sl-card"><h3>{region["name"]}</h3><p>{region.get("description") or ""}</p>'
                f'<div class="sl-badges">{badges}</div></div>',
                unsafe_allow_html=True,
            )
            chosen = selected_region_id() == region["id"]
            if st.button(
                "선택됨 ✓" if chosen else "이 지역 보기",
                key=f"sl_pick_{region['id']}",
                type="primary" if not chosen else "secondary",
                width="stretch",
            ):
                st.session_state.slide_region = region["id"]
                for key in ("sl_level_forecast", "sl_area_forecast", "sl_area_sample"):
                    st.session_state.pop(key, None)
                go_to_slide(1)
    # 지도: bbox 중심점
    centers = []
    for region in ordered:
        bbox = None
        try:
            bbox = _cached(f"region_detail:{region['id']}", lambda rid=region["id"]: client.get_region(rid)).get("bbox_wgs84")
        except APIError:
            bbox = None
        if isinstance(bbox, list) and len(bbox) == 4:
            n_stations = int(region.get("n_stations") or 0)
            n_trained = int(region.get("n_stations_trained") or 0)
            sat = region.get("satellite") or {}
            n_scenes = sum(len(v) for v in (sat.get("scenes") or {}).values() if isinstance(v, list))
            hover = f"수위계 {n_stations}지점 (예측 모델 {n_trained}지점)" + (f"<br>위성 장면 {n_scenes}시점" if n_scenes else "<br>위성 장면 없음")
            centers.append((region["name"], (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bool(region.get("satellite_available")), hover))
    if centers:
        figure = go.Figure()
        figure.add_trace(
            go.Scattermap(
                lon=[c[1] for c in centers], lat=[c[2] for c in centers], mode="markers+text",
                text=[c[0] for c in centers], textposition="top center",
                marker={"size": [18 if c[3] else 12 for c in centers], "color": ["#22d3ee" if c[3] else "#94a3b8" for c in centers]},
                customdata=[c[4] for c in centers],
                hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            )
        )
        figure.update_layout(
            map={"style": "carto-darkmatter", "center": {"lon": 127.6, "lat": 36.4}, "zoom": 5.6},
            height=330, margin={"l": 0, "r": 0, "t": 6, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        )
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
        st.markdown('<div class="sl-caption">청록 = 위성 영상이 있는 지역, 회색 = 수위계만 있는 지역</div>', unsafe_allow_html=True)


# --- 2. 데이터 둘러보기 ---------------------------------------------------------------------------


def _scene_timeline(scenes: list[Mapping[str, Any]]) -> go.Figure:
    figure = go.Figure()
    sensors = sorted({s["sensor"] for s in scenes})
    colors = {"ICEYE": "#f472b6", "PlanetScope": "#fbbf24"}
    for sensor in sensors:
        items = [s for s in scenes if s["sensor"] == sensor]
        figure.add_trace(
            go.Scatter(
                x=[s["date"] for s in items], y=[sensor] * len(items), mode="markers",
                marker={"size": 16, "symbol": "diamond", "color": colors.get(sensor, "#38bdf8")},
                name=sensor, hovertemplate="%{x}<br>%{y}<extra></extra>",
            )
        )
    figure.update_layout(**_LAYOUT, height=190, title={"text": "위성 관측 시점", "x": 0.01}, showlegend=False)
    figure.update_yaxes(type="category", gridcolor="#1e293b")
    figure.update_xaxes(gridcolor="#1e293b")
    return figure


def _slide_data(client: APIClient, region: Mapping[str, Any]) -> None:
    _explain(
        "<b>무엇을 보나요?</b> 이 지역이 가진 자료를 종류별로 확인합니다. 위성 영상은 '몇 날짜의 사진이 있는지', 수위계는 "
        "'어느 지점에서 몇 년 치 수위가 있는지'입니다. 예측 모델은 수위계 자료가 길수록 잘 배웁니다."
    )
    badges = "".join(
        f'<span class="sl-badge {"sl-badge--on" if on else "sl-badge--off"}">{"✓" if on else "✗"} {label}</span>'
        for label, on in data_status_badges(region)
    )
    st.markdown(f'<div class="sl-badges">{badges}</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 위성 영상")
        try:
            scenes = _cached(f"scenes:{region['id']}", lambda: client.satellite_scenes(region["id"]))
        except APIError:
            scenes = {"available": False, "scenes": []}
        if scenes.get("available"):
            st.plotly_chart(_scene_timeline(scenes["scenes"]), width="stretch", config={"displaylogo": False})
            by_sensor: dict[str, list] = {}
            for s in scenes["scenes"]:
                by_sensor.setdefault(s["sensor"], []).append(s["date"])
            static_table([{"센서": k, "장면 수": len(v), "기간": f"{v[0]} ~ {v[-1]}"} for k, v in by_sensor.items()])
            st.markdown(
                f'<div class="sl-caption">탐지 마스크가 있는 장면 {scenes.get("n_with_masks", 0)}개 · '
                f'{"GT(정답) 라벨 있음 → 탐지 정확도 계산 가능" if region.get("gt_labels_available") else "GT 라벨 없음 → 탐지 정확도는 계산하지 않음"}</div>',
                unsafe_allow_html=True,
            )
        else:
            _none("이 지역에는 위성 영상이 없습니다. 수위계 자료로 예측만 합니다.")
    with right:
        st.markdown("#### 지상 수위계")
        stations = region.get("stations") or []
        if stations:
            static_table(
                [
                    {"지점": s["name"], "지형": terrain_label(s.get("terrain")), "기간": " ~ ".join(s.get("period") or []),
                     "관측일": f"{s.get('n_days', 0):,}", "위성 연결": "✓" if s.get("satellite_location") else "-"}
                    for s in stations
                ]
            )
            st.markdown("**수위계 위치 지도** <span class='sl-caption'>(점에 마우스를 올리면 주소·하천 · 청록 상자 = 위성 수위 산출 구역)</span>", unsafe_allow_html=True)
            _station_map(client, region, height=360)
            weather = region.get("weather") or {}
            if weather.get("station_id"):
                st.markdown(f'<div class="sl-caption">기상: 기상청 ASOS {weather.get("station_name", "")} ({weather["station_id"]}) 지점</div>', unsafe_allow_html=True)
            qc_notes = []
            for s in stations:
                qc = s.get("qc") or {}
                if qc.get("removed_days"):
                    runs = ", ".join(f"{r['start']}~{r['end']} {r['days']}일 ≈{r['level_m']} m" for r in qc.get("flat_runs") or [])
                    spikes = ", ".join(f"{x['date']} {x['level_m']} m" for x in qc.get("spikes") or [])
                    qc_notes.append(f"<b>{s['name']}</b>: {qc['removed_days']}일 제외" + (f" · 정체 구간 {runs}" if runs else "") + (f" · 단발 스파이크 {spikes}" if spikes else ""))
            if qc_notes:
                st.markdown(
                    '<div class="sl-caption">자료 정제(계기 오류로 판단해 결측 처리, 값은 고치지 않음): ' + " / ".join(qc_notes)
                    + " — 정상 범위 밖에서 2주 이상 멈춘 구간과 하루 만에 튀었다 돌아온 값만 뺀다. 홍수 봉우리는 남긴다.</div>",
                    unsafe_allow_html=True,
                )
        else:
            _none("수위계 자료가 없습니다.")
            if region.get("scenes_available"):
                st.markdown("**위성 수위 산출 구역 지도**")
                _station_map(client, region, height=360)
    if stations:
        st.markdown("#### 수위계 기록 살펴보기")
        ids = [s["id"] for s in stations]
        preferred = next((s["id"] for s in stations if s.get("satellite_location")), ids[0])
        pick = st.selectbox("지점", ids, index=ids.index(preferred), key="sl_data_station",
                            format_func=lambda sid: next(s["name"] for s in stations if s["id"] == sid))
        try:
            series = _cached(f"series:{region['id']}:{pick}", lambda: client.gauge_series(region["id"], pick))
        except APIError as exc:
            st.error(str(exc))
        else:
            rows = series.get("rows") or []
            st.plotly_chart(
                _series_figure(rows, split=_active_split(region), title=f"{series['station']['name']} 일 수위 · {len(rows):,}일"),
                width="stretch", config={"displaylogo": False},
            )
            _explain(
                "<b>색 띠 읽는 법:</b> 회색은 모델이 <b>배우는</b> 기간, 파랑은 어떤 모델이 나은지 <b>고르는</b> 기간, 노랑은 모델이 한 번도 보지 않은 "
                "<b>시험</b> 기간입니다. 성능 평가는 노랑 구간에서만 합니다."
            )


# --- 3. 위성 수체 탐지 ----------------------------------------------------------------------------


def _detect_locations(client: APIClient, region: Mapping[str, Any], scene: Mapping[str, Any]) -> None:
    """어디를 잰 것인가 — 선택 장면 위에 위성 수위 산출 구역(AOI)과 수위계 위치를 겹쳐 보이고, 옆에 지도를 놓는다."""

    st.markdown("#### 어디를 잰 것인가 — 위성 수위 산출 구역 ↔ 수위계 위치")
    locations = _satellite_locations(client, region)
    stations = region.get("stations") or []
    if not locations:
        _none("이 지역은 위성 장면은 있지만 <b>위성 수위 산출 구역(AOI)</b>이 등록되어 있지 않습니다 — 위성 수위를 재는 위치를 지정하면 "
              "이 그림에 구역 상자와 수위계 위치가 겹쳐 나옵니다.")
        if any(isinstance(s.get("lat"), (int, float)) for s in stations):
            _station_map(client, region, scene=scene, height=380)
        return
    _explain(
        "<b>위성 수위는 사진 전체가 아니라 작은 구역에서 잽니다.</b> 청록 상자가 위성 수위 산출 구역(AOI, 한 변 수백 m)이고 "
        "주황 ▲가 그 구역과 짝지은 <b>지상 수위계</b>(다리 이름)입니다. 점선과 숫자는 상자 중심에서 수위계까지의 거리입니다. "
        "왼쪽은 이 장면의 탐지 결과 위에 겹친 것(기본은 구역 주변 확대), 오른쪽은 같은 위치를 지도 위에 놓은 것입니다. "
        "상자는 사진 전체에서는 몇 픽셀 크기라 '구역 주변만 확대'를 끄면 고리 표시로만 보입니다. 두 그림 모두 마우스 휠로 더 확대할 수 있습니다."
    )
    urls = scene.get("image_urls") or {}
    quicklook = scene.get("quicklook") if isinstance(scene.get("quicklook"), Mapping) else None
    left, right = st.columns([1.1, 1])
    with left:
        if not quicklook or not quicklook.get("bounds_epsg32652"):
            _none("이 장면은 퀵룩의 좌표 범위(quicklook.bounds_epsg32652)가 아직 없어 사진 위에 구역을 겹칠 수 없습니다. 오른쪽 지도에서 위치를 확인하세요.")
        else:
            zoom = st.toggle("구역 주변만 확대", value=True, key=f"sl_loc_zoom_{scene.get('scene_id') or scene.get('date')}",
                             help="끄면 장면 전체가 보입니다(구역 상자는 몇 픽셀 크기라 고리 표시로만 보입니다)")
            image_url = urls.get("overlay") or urls.get("input")
            image = None
            if image_url:
                try:
                    image = _cached(f"img:{image_url}", lambda u=image_url: client.fetch_bytes(u))
                except APIError:
                    image = None
            try:
                figure = overlay_figure(
                    image, quicklook, locations, stations, zoom_to_locations=zoom, compact_labels=not zoom,
                    title=f"{scene['sensor']} {scene['date']} · {'탐지 결과' if urls.get('overlay') else '원본'} 위 산출 구역·수위계", height=560,
                )
            except ValueError as exc:
                _none(f"겹쳐 그리기 실패: {exc}")
            else:
                st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    with right:
        _station_map(client, region, scene=scene, height=560, title="지도에서 본 같은 위치")
    st.markdown(
        '<div class="sl-caption">수위계 위치는 WAMIS 지점 좌표(경위도)를 구역 중심 기준 등장방형 근사로 미터 좌표에 놓은 것입니다(1 km 안에서 오차 1 m 미만, 표시용). '
        "구역 좌표는 인수 패키지 AOI 정본(EPSG:32652)에서 그대로 가져옵니다.</div>",
        unsafe_allow_html=True,
    )


def _slide_detect(client: APIClient, region: Mapping[str, Any]) -> None:
    try:
        scenes = _cached(f"scenes:{region['id']}", lambda: client.satellite_scenes(region["id"]))
    except APIError:
        scenes = {"available": False, "scenes": []}
    if not scenes.get("available"):
        _none("이 지역은 아직 위성 장면이 없습니다 — 전처리 GeoTIFF 를 올리면 탐지·면적·위성 수위 대조가 열립니다. "
              "지금은 '다음'을 눌러 지점 수위 예측으로 바로 갈 수도 있습니다.")
        _explain(
            "<b>왜 비어 있나요?</b> 이 지역에는 지상 수위계 기록만 있고 위성 장면(전처리 GeoTIFF·탐지 마스크)이 등록되지 않았습니다. "
            "<b>무엇을 올리면 되나요?</b> 정사·방사보정된 위성 영상 GeoTIFF 한 장(ICEYE·Sentinel-1 같은 SAR 또는 PlanetScope 같은 광학). "
            "탐지 마스크(0=육지, 1=수체)가 있으면 함께, 없으면 사진만 등록되어 '탐지 대기' 상태가 되고 탐지 어댑터(WBMS)가 있는 서버에서는 탐지를 돌릴 수 있습니다. "
            "<b>열리는 것:</b> 이 화면의 원본/탐지 오버레이 비교, 장면별 수체 면적, 지점과 연결된 위성 수위 vs 수위계 대조표."
        )
        render_satellite_input(client, region, key=f"sl_satin_{region['id']}")
        if st.session_state.get(f"sl_satin_{region['id']}_last_result", {}).get("region_id") == region["id"]:
            if st.button("등록한 장면 보기 ↻", key="sl_detect_reload"):
                st.rerun()
        return
    _explain(
        "<b>무엇을 보나요?</b> 왼쪽은 위성이 찍은 원본, 오른쪽은 인공지능(WBMS U-Net)이 <b>물이라고 판단한 곳을 청록색</b>으로 칠한 것입니다. "
        "정답 라벨이 있으면 노란색 정답과 비교해 <b>IoU</b>(겹치는 정도, 1에 가까울수록 정확)를 함께 보여 줍니다. 아래 슬라이더로 날짜를 바꿔 보세요."
    )
    sensors = scenes.get("sensors") or []
    sensor = st.radio("센서", sensors, horizontal=True, key="sl_detect_sensor",
                      format_func=lambda s: {"ICEYE": "ICEYE (SAR 레이더)", "PlanetScope": "PlanetScope (광학)"}.get(s, s)) if len(sensors) > 1 else sensors[0]
    items = [s for s in scenes["scenes"] if s["sensor"] == sensor]
    dates = [s["date"] for s in items]
    pick = st.select_slider("관측 날짜", options=dates, value=dates[0], key=f"sl_detect_date_{sensor}")
    scene = next(s for s in items if s["date"] == pick)
    urls = scene.get("image_urls") or {}
    cols = st.columns(3 if urls.get("gt") else 2)
    with cols[0]:
        st.markdown("**① 위성 원본**")
        if urls.get("input"):
            st.image(_cached(f"img:{urls['input']}", lambda u=urls["input"]: client.fetch_bytes(u)), width="stretch")
    with cols[1]:
        st.markdown("**② 탐지 결과 (청록 = 물)**")
        if urls.get("overlay"):
            st.image(_cached(f"img:{urls['overlay']}", lambda u=urls["overlay"]: client.fetch_bytes(u)), width="stretch")
        else:
            _none("이 장면은 탐지 마스크가 없습니다.")
    if urls.get("gt"):
        with cols[2]:
            st.markdown("**③ 정답 라벨 (노랑)**")
            st.image(_cached(f"img:{urls['gt']}", lambda u=urls["gt"]: client.fetch_bytes(u)), width="stretch")
    tiles = st.columns(4)
    tiles[0].metric("관측일", scene["date"])
    tiles[1].metric(
        "탐지된 수체 면적", f"{scene['water_area_km2']:.2f} km²" if isinstance(scene.get("water_area_km2"), (int, float)) else "-",
        help=("이 장면의 촬영범위 안에서 센 면적입니다. 촬영범위는 날짜마다 달라서 이 값끼리 이어 붙이면 시계열이 되지 않습니다"
              + (f" — 날짜 간 비교용 공통 격자 면적은 {scene['area_common_frame_km2']:.2f} km²."
                 if isinstance(scene.get("area_common_frame_km2"), (int, float)) else ".")),
    )
    metrics = scene.get("metrics") or {}
    tiles[2].metric("IoU (정답 대비)", _fmt(metrics.get("water_iou"), 3) if metrics else "GT 없음", help="1이면 정답과 완전히 일치")
    tiles[3].metric("F1", _fmt(metrics.get("f1"), 3) if metrics else "GT 없음")
    if not scene.get("detection_available"):
        _note("이 장면은 아직 탐지 마스크가 없습니다(탐지 대기). 마스크 GeoTIFF 가 생기면 아래 '새 위성 장면 등록'에서 같은 센서·촬영일로 다시 올리면 오버레이·면적이 붙습니다.")
    elif not metrics:
        _note("이 지역은 정답(GT) 라벨이 없어 정확도 수치를 내지 않습니다. 사진으로만 확인합니다.")

    _detect_locations(client, region, scene)

    # 면적 변화 막대 — 공통 격자(비교 가능) vs 씬별 촬영범위(비교 불가)
    with_area = [s for s in items if isinstance(s.get("water_area_km2"), (int, float))]
    if len(with_area) >= 2:
        dates = [s["date"] for s in with_area]
        raw = [s["water_area_km2"] for s in with_area]
        common = [s.get("area_common_frame_km2") for s in with_area]
        has_common = all(isinstance(v, (int, float)) for v in common)
        figure = go.Figure()
        if has_common:
            figure.add_trace(go.Bar(x=dates, y=common, name="공통 격자 (비교 가능)", marker={"color": "#22d3ee"},
                                    hovertemplate="%{x}<br>%{y:.2f} km²<extra>공통 격자</extra>"))
        figure.add_trace(go.Bar(x=dates, y=raw, name="씬별 촬영범위 (비교 불가)",
                                marker={"color": "#64748b" if has_common else "#22d3ee"},
                                hovertemplate="%{x}<br>%{y:.2f} km²<extra>씬별 촬영범위</extra>"))
        figure.update_layout(**_LAYOUT, height=280, barmode="group",
                             title={"text": f"{sensor} 장면별 탐지 수체 면적 (km²)", "x": 0.01})
        figure.update_yaxes(gridcolor="#1e293b", range=padded_range(raw + ([v for v in common] if has_common else []), pad_ratio=0.3))
        figure.update_xaxes(type="category")
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
        if has_common:
            _explain(
                "<b>회색 막대(씬별 촬영범위)를 시계열로 읽지 마세요.</b> 위성이 날짜마다 다른 범위를 찍기 때문에, 물이 늘지 않아도 "
                "범위가 넓거나 옮겨가면 면적이 커집니다. 부산 광학 2020-03-25 가 그 예로, 03-12 보다 남쪽으로 8.6 km 내려가 하구를 "
                "담는 바람에 33 → 65 km² 로 보였습니다. 같은 창(청록, 공통 격자)으로 세면 28.9 → 30.1 km²(+4.4 %)이고, 그 장면의 "
                "정답 라벨도 같은 창에서 30.0 km² 라 <b>탐지는 정확했고 계열이 비교 불가능했던 것</b>입니다."
            )

    # 위성 수위 vs 수위계
    linked = [s for s in region.get("stations") or [] if s.get("satellite_location")]
    if linked and region.get("satellite_levels_available"):
        st.markdown("#### 위성이 잰 수위 vs 지상 수위계 (같은 날)")
        rows = []
        locations = _satellite_locations(client, region)
        for s in linked:
            try:
                payload = _cached(f"satlevels:{region['id']}:{s['id']}", lambda sid=s["id"]: client.satellite_levels(region["id"], sid))
            except APIError:
                continue
            comparison = payload.get("comparison") or {}
            records = payload.get("records") or []
            note = next((str(r.get("correction_note") or "") for r in records if r.get("correction_note")), "")
            why = "" if comparison else (note or "보정 수위 없음 — 원값(상대 수위)만 있어 수위계와 직접 비교할 수 없음")
            loc_id = str(payload.get("loc_id") or s.get("satellite_location") or "")
            location = payload.get("location") if isinstance(payload.get("location"), Mapping) else locations.get(loc_id)
            where = location_table_fields(dict(location, loc_id=loc_id) if location else None, s)
            rows.append({"지점": s["name"], "위성 산출 구역": where["위성 산출 구역"] if location else (loc_id or "–"),
                         "수위계까지 거리": where["수위계까지 거리"],
                         "위성 시점 수": payload.get("n"), "대조 일수": comparison.get("n") or 0,
                         "RMSE": f"{_fmt(comparison.get('rmse_m'))} m ({_cm(comparison.get('rmse_m'))})" if comparison.get("rmse_m") is not None else "–",
                         "MAE": f"{_fmt(comparison.get('mae_m'))} m ({_cm(comparison.get('mae_m'))})" if comparison.get("mae_m") is not None else "–",
                         "비고": why or "보정 수위 vs 수위계 같은 날",
                         "구역 설정 메모": where["구역 설정 메모"]})
        if rows:
            static_table(rows)
            _explain("<b>읽는 법:</b> 위성 사진의 물 경계와 지형 높이로 계산한 수위를, 같은 날 실제 수위계 값과 맞대어 본 오차입니다. 예측이 아니라 <b>관측끼리의 비교</b>입니다. "
                     "'비고'가 채워진 지점은 위성 산출은 있지만 수위계 기준면으로 보정한 값이 없어 오차를 내지 않은 곳입니다. "
                     "'위성 산출 구역'은 위 그림의 청록 상자, '수위계까지 거리'는 그 상자 중심에서 수위계까지의 거리입니다.")

    with st.expander("새 위성 장면 등록 (GeoTIFF 업로드)", expanded=False):
        st.caption("다른 날짜·다른 센서의 전처리 GeoTIFF 를 올리면 같은 격자로 잘려 이 화면과 면적 시계열에 붙습니다.")
        if render_satellite_input(client, region, key=f"sl_satin_{region['id']}") is not None:
            if st.button("목록 새로고침 ↻", key="sl_detect_reload_more"):
                st.rerun()


# --- 4. 시계열 예측 -------------------------------------------------------------------------------


def _weather_window(client: APIClient, region: Mapping[str, Any], end: Date, days: int = 30) -> list[dict[str, Any]]:
    weather = region.get("weather") or {}
    station = str(weather.get("station_id") or "")
    if not station:
        return []
    start = end - timedelta(days=days - 1)
    key = f"weather:{station}:{start}:{end}"

    def load() -> list[dict[str, Any]]:
        # 1) 지역에 저장된 ASOS(학습 때 쓴 것과 같은 실관측) → 2) 범위 밖이면 실시간 조회(서버 키 있을 때) → 3) 그것도 없으면 빈 목록
        try:
            stored = client.region_weather(region["id"], start=start.isoformat(), end=end.isoformat())
            rows = [dict(r, _source="asos_stored") for r in (stored.get("rows") or []) if isinstance(r, Mapping)]
            if len(rows) >= max(1, days // 2):
                return rows
        except APIError:
            rows = []
        try:
            status = _cached("weather_status", client.weather_status)
            if not status.get("asos_available"):
                return rows
            payload = client.weather_observations(source="asos", station_id=station, start_date=start.isoformat(), end_date=end.isoformat())
        except APIError:
            return rows
        live = [dict(r, _source=payload.get("source")) for r in (payload.get("rows") or []) if isinstance(r, Mapping)]
        return live or rows

    return _cached(key, load)


def _weather_figure(rows: list[dict[str, Any]], title: str) -> go.Figure:
    dates = [str(r.get("date")) for r in rows]
    rain = [r.get("precipitation_mm") or 0 for r in rows]
    temp = [r.get("avg_temperature_c") if r.get("avg_temperature_c") is not None else r.get("temperature_c") for r in rows]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=dates, y=rain, name="강수 (mm)", marker={"color": "#38bdf8"}, hovertemplate="%{x}<br>강수 %{y:.1f} mm<extra></extra>"))
    if any(v is not None for v in temp):
        figure.add_trace(go.Scatter(x=dates, y=temp, name="기온 (°C)", yaxis="y2", line={"color": "#fbbf24", "width": 2}, hovertemplate="%{x}<br>기온 %{y:.1f} °C<extra></extra>"))
    figure.update_layout(
        **_LAYOUT, height=240, title={"text": title, "x": 0.01},
        yaxis={"title": "강수 (mm)", "gridcolor": "#1e293b"},
        yaxis2={"title": "기온 (°C)", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "y": -0.42, "yanchor": "top"},
    )
    return figure


def _slide_forecast(client: APIClient, region: Mapping[str, Any], *, execute_quick_sample: Callable[..., None]) -> None:
    _explain(
        "<b>두 가지를 예측합니다.</b> ① <b>지점 수위</b>: 수위계 10년 기록으로 배운 모델이 오늘 이후 며칠 뒤 수위를 내놓습니다. "
        "② <b>수체 면적</b>: 위성 탐지 마스크 시계열과 기상(강수)으로 앞날의 물 면적을 추정합니다. "
        "모델은 직접 골라 볼 수 있고, 기상청 관측을 함께 보여 줍니다. 실행 뒤 '다음'을 누르면 결과 그래프 화면입니다."
    )
    tab_level, tab_area = st.tabs(["① 지점 수위 예측", "② 수체 면적 예측"])

    with tab_level:
        stations = [s for s in region.get("stations") or [] if s.get("model_trained")]
        if not stations:
            _none("이 지역에는 학습된 수위 예측 모델이 없습니다.")
            if region.get("gauge_available"):
                render_training_window(client, region, key="sl_tw")
        else:
            ids = [s["id"] for s in stations]
            preferred = next((s["id"] for s in stations if s.get("satellite_location")), ids[0])
            c1, c2, c3 = st.columns([2, 2, 2])
            station_id = c1.selectbox("지점", ids, index=ids.index(preferred), key="sl_lv_station",
                                      format_func=lambda sid: next(f"{s['name']} · {terrain_label(s.get('terrain'))}" for s in stations if s["id"] == sid))
            try:
                forecasters = _cached("forecasters", client.list_gauge_forecasters)
            except APIError:
                forecasters = []
            guide = _guide(client)
            # 기준선(직전값 유지)과 참고선(계절 평년값)은 예측 모델로 고를 수 없다 — 비교 화면에서 참고로만 보인다
            model_options = ["auto"] + [f["id"] for f in forecasters if not f.get("is_baseline") and not f.get("is_reference") and f["id"] != "climatology"]
            model = c2.selectbox(
                "모델", model_options, key="sl_lv_model",
                format_func=lambda m: "자동 (검증에서 고른 최적 모델)" if m == "auto" else _model_label_with_weather(m, guide),
                help="자동은 예측일수마다 검증에서 가장 잘 맞은 모델을 씁니다. 다른 모델을 고르면 그 모델로 즉석 예측합니다. '기상 사용' 모델은 원점 이전 강수·기온을 입력으로 씁니다.",
            )
            horizon = c3.select_slider("며칠 뒤까지", options=[1, 3, 5, 7, 10, 14, 21, 30], value=14, key="sl_lv_horizon")
            station = next(s for s in stations if s["id"] == station_id)
            period = station.get("period") or []
            last = Date.fromisoformat(str(period[1])[:10]) if len(period) == 2 else Date.today()
            first = Date.fromisoformat(str(period[0])[:10]) if len(period) == 2 else Date(2015, 1, 1)
            verify = st.toggle("과거 날짜에서 예측해 실제와 비교해 보기", value=True, key="sl_lv_verify",
                               help="켜면 시험 연도 안의 날짜를 원점으로 두고, 그 뒤 실제 관측을 함께 그립니다.")
            split = _active_split(region)
            default_origin = Date(int(str(split.get("test_start", "2024"))[:4]), 7, 1) if verify else last
            # 키에 시험 연도를 넣어, 구간을 바꿔 다시 학습하면 원점 기본값(시험 연도 7월 1일)이 따라오게 한다 —
            # 같은 키를 session_state 에서 지워도 브라우저가 이전 값을 되돌려 보내 위젯이 초기화되지 않는다
            as_of = st.date_input("예측 원점 (이 날까지의 관측만 사용)", value=min(default_origin, last), min_value=first, max_value=last,
                                  key=f"sl_lv_asof_{str(split.get('test_start', ''))[:4]}") if verify else last
            weather_rows = _weather_window(client, region, as_of, 30)
            if weather_rows:
                src = weather_rows[0].get("_source") or "asos"
                src_label = {"asos_stored": "저장된 ASOS 실관측", "asos": "ASOS 실시간 조회", "sample": "표본 자료(실관측 아님)"}.get(str(src), str(src))
                st.plotly_chart(_weather_figure(weather_rows, f"기상청 관측 · 원점 전 30일 · {src_label}"), width="stretch", config={"displaylogo": False})
                st.markdown('<div class="sl-caption">기상 사용 모델(AR-X·앙상블)은 원점 이전 강수 1·3·7·30일 합과 기온을 입력으로 씁니다. 목표일까지의 미래 강수(예보)는 아직 입력이 아닙니다.</div>', unsafe_allow_html=True)
            render_model_guide(guide, key="sl_guide_forecast")
            # 학습·검증·시험 구간을 바꿔 다시 학습(비동기 작업). 지역별 키라 지역을 바꾸면 초기화된다.
            render_training_window(client, region, key="sl_tw")
            if st.button("수위 예측 실행 ▶", type="primary", key="sl_lv_run", width="stretch"):
                with st.spinner("예측 중…"):
                    try:
                        payload = client.gauge_forecast(
                            region["id"], station_id, as_of=as_of.isoformat(), horizon=int(horizon),
                            history_days=90, forecaster=None if model == "auto" else model,
                        )
                    except APIError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.sl_level_forecast = {"payload": payload, "station": station, "region_id": region["id"], "model": model}
                        st.success(f"{station['name']} D+1~D+{horizon} 수위 예측 완료 — '다음'에서 그래프를 봅니다.")
            if st.session_state.get("sl_level_forecast", {}).get("region_id") == region["id"]:
                done = st.session_state.sl_level_forecast
                stale = done["station"]["id"] != station_id
                st.markdown(f'<div class="sl-caption">최근 실행: {done["station"]["name"]} · 원점 {done["payload"].get("as_of")} · 모델 {"자동" if done["model"] == "auto" else forecaster_label(done["model"])}'
                            + (" — <b>지금 고른 지점과 다릅니다. 다시 실행하세요.</b>" if stale else "") + "</div>", unsafe_allow_html=True)
            if st.toggle("모든 모델로 한 번에 비교하기 (그래프 + 차이 표 + 성능지표)", value=False, key="sl_lv_compare"):
                render_model_compare(client, region, station, as_of=as_of, horizon=int(horizon), key="sl_lv")

    with tab_area:
        samples = region_samples_for(region["id"])
        if not samples:
            if region.get("scenes_available"):
                _none("이 지역의 위성 장면은 있지만 면적 예측에 쓸 <b>탐지 마스크 시계열</b>(같은 센서 2장 이상, 마스크 포함)이 UI 시작 때 없었습니다. "
                      "마스크가 있는 장면을 등록한 뒤 UI 를 다시 시작하면 여기에 샘플이 나타납니다.")
            else:
                _none("이 지역에는 아직 위성 장면이 없어 수체 면적 예측을 건너뜁니다. 지점 수위 예측(① 탭)은 그대로 쓸 수 있습니다.")
            _explain(
                "<b>면적 예측을 열려면:</b> 전처리 GeoTIFF + 탐지 마스크(0=육지, 1=수체)를 같은 센서로 2장 이상 등록하세요. "
                "등록은 <b>3. 위성 수체 탐지</b> 화면의 업로드 폼(또는 위성 파이프라인 화면 맨 아래)에서 합니다."
            )
            if st.button("◀ 3. 위성 수체 탐지로 가서 장면 올리기", key="sl_area_goto_upload"):
                go_to_slide(slide_index("detect"))
        else:
            c1, c2, c3 = st.columns([2, 2, 2])
            sample_ids = [s.sample_id for s in samples]
            sample_id = c1.selectbox("위성 마스크 시계열", sample_ids, key="sl_area_sample_id",
                                     format_func=lambda sid: next(s.display_name for s in samples if s.sample_id == sid))
            try:
                models = _cached("area_models", client.list_models)
            except APIError:
                models = []
            visible = [m for m in models if m.get("id") not in HIDDEN_AREA_MODELS] or models
            model_ids = [m["id"] for m in visible]
            sample = next(s for s in samples if s.sample_id == sample_id)
            default_model = sample.recommended_model_id if sample.recommended_model_id in model_ids else (model_ids[0] if model_ids else None)
            area_model = c2.selectbox("모델", model_ids, index=model_ids.index(default_model) if default_model in model_ids else 0, key="sl_area_model",
                                      format_func=lambda mid: area_model_label(next((m for m in visible if m["id"] == mid), mid)),
                                      help="설명 가능한 기준선 모델입니다. 학습된 정확도 모델이 아니며 어댑터를 등록하면 교체됩니다.")
            area_days = c3.select_slider("며칠 뒤까지", options=[7, 14, 30], value=14, key="sl_area_days")
            st.markdown(f'<div class="sl-caption">{sample.summary_ko} 기상: {sample.weather_provenance_ko}</div>', unsafe_allow_html=True)
            if st.button("수체 면적 예측 실행 ▶", type="primary", key="sl_area_run", width="stretch"):
                st.session_state.model_id = area_model
                st.session_state.selected_model_id = area_model
                with st.spinner("마스크를 백엔드에 보내 예측하는 중…"):
                    try:
                        execute_quick_sample(sample, int(area_days), navigate=False, model_id=area_model)
                    except APIError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.sl_area_forecast = {"sample_id": sample_id, "model": area_model, "region_id": region["id"], "days": int(area_days)}
                        st.success("면적 예측 완료 — '다음'에서 그래프를 봅니다.")
            if st.session_state.get("sl_area_forecast", {}).get("region_id") == region["id"]:
                done = st.session_state.sl_area_forecast
                notice = st.session_state.get("weather_query_notice") or {}
                st.markdown(f'<div class="sl-caption">최근 실행: {done["sample_id"]} · 모델 {area_model_label(done["model"])} · {done["days"]}일'
                            + (f' · 기상: {notice.get("note") or notice.get("source")}' if notice else "") + "</div>", unsafe_allow_html=True)
                if notice and str(notice.get("source")) in ("none", "built_in_demo"):
                    st.warning("이번 실행에는 기상청 실관측이 붙지 않았습니다(" + str(notice.get("note") or notice.get("source")) + "). 강수 보정이 0이거나 시나리오 값입니다.")



# 모델별 비교 구성 요소는 model_compare_ui 에 있다(위 import 에서 기존 이름을 재수출).


# --- 5. 결과 그래프 -------------------------------------------------------------------------------


def _area_result_figure(result: Mapping[str, Any], weather: list[Mapping[str, Any]], title: str) -> go.Figure:
    steps = result.get("steps") or []
    dates = [str(s.get("target_date") or "") for s in steps]
    areas = [s.get("water_area_km2") for s in steps]
    figure = go.Figure()
    rain_by_date = {str(r.get("timestamp") or r.get("date"))[:10]: r.get("precipitation_mm") for r in weather}
    rain = [rain_by_date.get(d) or 0 for d in dates]
    if any(rain):
        figure.add_trace(go.Bar(x=dates, y=rain, name="강수 (mm)", yaxis="y2", marker={"color": "rgba(56,189,248,.35)"}, hovertemplate="%{x}<br>강수 %{y:.1f} mm<extra></extra>"))
    figure.add_trace(go.Scatter(x=dates, y=areas, mode="lines+markers", name="예측 수체 면적 (km²)", line={"color": "#22d3ee", "width": 3.5}, marker={"size": 7}, hovertemplate="%{x}<br>면적 %{y:.3f} km²<extra></extra>"))
    figure.update_layout(
        **_LAYOUT, height=420, title={"text": title, "x": 0.01},
        yaxis={"title": "수체 면적 (km²)", "gridcolor": "#1e293b", "range": padded_range([a for a in areas if a is not None], pad_ratio=0.3)},
        yaxis2={"title": "강수 (mm)", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "y": -0.22, "yanchor": "top"},
    )
    return figure


def _slide_result(client: APIClient, region: Mapping[str, Any]) -> None:
    level = st.session_state.get("sl_level_forecast")
    area = st.session_state.get("sl_area_forecast")
    if not (level and level.get("region_id") == region["id"]) and not (area and area.get("region_id") == region["id"]):
        _none("아직 실행한 예측이 없습니다. '이전'으로 돌아가 예측을 실행하세요.")
        return
    _explain(
        "<b>그래프 읽는 법:</b> 굵은 청록 선이 예측, 회색 선은 원점까지의 실제 관측, 노란 선은 (과거 원점일 때) 그 뒤 실제 값입니다. "
        "연한 띠는 검증 기간 잔차로 만든 <b>±1σ/±2σ 산포</b>이며, 점선은 '어제 값 그대로'라는 비교 기준선입니다."
    )
    if level and level.get("region_id") == region["id"]:
        payload = level["payload"]
        station = level["station"]
        st.markdown(f"### ① {station['name']} 수위 예측")
        render_split_badge(client, region)
        steps = payload.get("steps") or []
        origin = payload.get("origin_level_m")
        tiles = st.columns(4)
        tiles[0].metric("원점 수위", f"{origin:.3f} m" if isinstance(origin, (int, float)) else "-", help=f"원점 {payload.get('as_of')}")
        if steps:
            last_step = steps[-1]
            tiles[1].metric(f"D+{last_step['horizon']} 예측", _fmt(last_step.get("forecast_m"), 3, " m"),
                            delta=(f"{last_step['forecast_m'] - origin:+.3f} m" if isinstance(last_step.get("forecast_m"), (int, float)) and isinstance(origin, (int, float)) else None))
            tiles[2].metric("±1σ 산포", _fmt(last_step.get("sigma_m"), 3, " m"))
        used = sorted({forecaster_label(v) for v in (payload.get("models_by_horizon") or {}).values()})
        tiles[3].metric("사용 모델", (used[0] if len(used) == 1 else f"{len(used)}종 (지평별)") if used else "-", help=", ".join(used) or None)
        if len(used) > 1:
            tiles[3].caption("·".join(used))
        st.plotly_chart(_forecast_figure(payload, title=f"{station['name']} · {payload.get('as_of')} 원점 D+1~D+{payload.get('horizon')} 수위 예측"), width="stretch", config={"displaylogo": False})
        verification = payload.get("verification")
        if verification:
            st.success(
                f"실제와 비교: {verification['n']}일 RMSE {verification['rmse_m']:.3f} m ({verification['rmse_m'] * 100:.1f} cm) · "
                f"MAE {verification['mae_m']:.3f} m ({verification['mae_m'] * 100:.1f} cm) · "
                f"기준선(어제 값 유지) RMSE {verification['persistence_rmse_m']:.3f} m ({verification['persistence_rmse_m'] * 100:.1f} cm) · "
                f"개선율 {skill_text(verification.get('skill_vs_persistence'))}"
            )
        st.markdown(f'<div class="sl-caption">{payload.get("forecaster_note_ko", "")} {payload.get("params_note_ko", "")}</div>', unsafe_allow_html=True)
        compare = st.session_state.get("sl_lv_compare_payload")
        if (
            compare and compare.get("region_id") == region["id"] and compare.get("station", {}).get("id") == station["id"] and str(compare.get("as_of")) == str(payload.get("as_of"))
            and int(compare.get("horizon") or 0) == int(payload.get("horizon") or -1)
        ):
            st.markdown("#### 모델별 비교")
            render_compare_payload(compare, station["name"], key="sl_res")
        with st.expander("예측 표 · 내려받기", expanded=False):
            table = pd.DataFrame(
                [{"목표일": s["target_date"], "D+": s["horizon"], "예측 (m)": s.get("forecast_m"), "±1σ": s.get("sigma_m"),
                  "기준선(어제 값)": s.get("persistence_reference_m"), "실제 (m)": s.get("observed_m"), "모델": forecaster_label(s.get("forecaster_id"))} for s in steps]
            )
            show_table(table)
            st.download_button("CSV 내려받기", table.to_csv(index=False).encode("utf-8-sig"), f"level_forecast_{station['id']}_{payload.get('as_of')}.csv", "text/csv", key="sl_dl_level")

    if area and area.get("region_id") == region["id"]:
        result = st.session_state.get("prediction_result") or {}
        steps = result.get("steps") or []
        st.markdown("### ② 수체 면적 예측")
        if not steps:
            _none("면적 예측 결과가 없습니다.")
        else:
            weather = list(st.session_state.get("weather_data") or [])
            first, last_step = steps[0], steps[-1]
            tiles = st.columns(4)
            tiles[0].metric("예측 프레임", f"{len(steps)}일")
            tiles[1].metric("첫 예측일", str(first.get("target_date") or "-"))
            tiles[2].metric("마지막 예측 면적", f"{last_step.get('water_area_km2', 0):,.2f} km²")
            tiles[3].metric("면적 변화", f"{last_step.get('area_change_pct', 0):+.2f} %", help="마지막 실제 관측 대비")
            st.plotly_chart(_area_result_figure(result, weather, f"{area['sample_id'].split(':')[-1]} · 수체 면적 예측과 강수"), width="stretch", config={"displaylogo": False})
            risk = str(last_step.get("risk") or "")
            risk_ko = {"normal": "정상", "caution": "주의", "flood_risk": "홍수 위험", "drought_risk": "가뭄 위험"}.get(risk, risk)
            st.markdown(f'<div class="sl-caption">마지막 날 위험도(규칙 기반): <b>{risk_ko}</b> · 모델 {result.get("model_id")} · 면적은 같은 센서 장면들이 <b>공통으로 덮는 범위</b> 안에서 리샘플 격자로 센 근사값이라 3단계의 장면 전체 면적보다 작을 수 있습니다.</div>', unsafe_allow_html=True)
            _note("면적 예측 모델은 설명 가능한 기준선입니다(학습된 정확도 모델 아님). 미래 정답이 없어 이 값으로 성능을 주장하지 않습니다. 성능 평가는 수위 예측에 대해 다음 화면에서 합니다.")


# --- 6. 성능 평가 ---------------------------------------------------------------------------------


def _slide_evaluate(client: APIClient, region: Mapping[str, Any]) -> None:
    render_split_badge(client, region)
    _explain(
        "<b>어떻게 평가했나요?</b> 수위계 기록을 시간 순서로 셋으로 나눴습니다. 앞부분으로 <b>배우고</b>, 가운데 몇 해로 <b>모델을 고르고</b>, "
        "마지막 해(모델이 한 번도 못 본 해)로 <b>채점</b>합니다. 채점 지표는 RMSE(오차의 크기, m)·MSE·MAE와, "
        "'어제 값을 그대로 두는 것'보다 얼마나 나은지(<b>skill</b>)입니다. 예측일수(며칠 뒤)를 바꾸면 표가 바뀝니다."
    )
    try:
        evaluation = _cached(f"evaluation:{region['id']}", lambda: client.gauge_evaluation(region["id"]))
    except APIError as exc:
        _none(f"이 지역의 평가표가 없습니다: {exc}")
        return
    split = evaluation.get("split") or {}
    guide = _guide(client)
    skill_guide = guide.get("skill") or {}
    if skill_guide:
        _explain(
            f"<b>개선율(skill)이란?</b> <code>{skill_guide.get('formula', '')}</code> — {skill_guide.get('plain_ko', '')} "
            f"<br><small>{skill_guide.get('why_ko', '')}</small>"
        )
    available = sorted({int(r["horizon"]) for r in evaluation.get("rows") or []})
    horizon = st.select_slider("예측일수 (며칠 뒤를 맞추나)", options=available or list(REPORT_HORIZONS), value=(7 if 7 in available else (available[0] if available else 7)), key="sl_ev_horizon")
    rows = [r for r in evaluation.get("rows") or [] if int(r["horizon"]) == int(horizon)]
    skills = [r["skill_vs_persistence"] for r in rows if r.get("skill_vs_persistence") is not None]
    tiles = st.columns(4)
    tiles[0].metric("시험 기간", f"{str(split.get('test_start', ''))[:4]}년", help="학습·선택에 쓰지 않은 해")
    tiles[1].metric(f"D+{horizon} 평균 개선율", skill_text(sum(skills) / len(skills)) if skills else "-", help="어제 값 유지 대비 MSE 감소율의 지점 평균")
    tiles[2].metric("기준선보다 나은 지점", f"{sum(1 for s in skills if s > 0)} / {len(skills)}")
    rmses = [r["test_rmse"] for r in rows if isinstance(r.get("test_rmse"), (int, float))]
    tiles[3].metric(f"D+{horizon} 평균 RMSE", f"{sum(rmses) / len(rmses):.3f} m" if rmses else "-", help=f"{sum(rmses) / len(rmses) * 100:.1f} cm" if rmses else None)
    if rmses:
        tiles[3].caption(f"= {sum(rmses) / len(rmses) * 100:.1f} cm")
    table = [
        {
            "지점": r.get("station_name"), "지형": terrain_label(r.get("terrain")),
            "선택 모델": forecaster_label(r.get("forecaster_id")) + (" · 기상" if r.get("uses_weather") else ""),
            "RMSE": f"{_fmt(r.get('test_rmse'), 3)} m ({_cm(r.get('test_rmse'))})",
            "MSE": f"{_fmt(r.get('test_mse'), 5)} m² ({_cm2(r.get('test_mse'))})",
            "MAE": f"{_fmt(r.get('test_mae'), 3)} m ({_cm(r.get('test_mae'))})",
            "기준선 RMSE": f"{_fmt(r.get('persistence_test_rmse'), 3)} m ({_cm(r.get('persistence_test_rmse'))})",
            "개선율 (skill)": skill_text(r.get("skill_vs_persistence")),
            "기상 효과": (f"{r['weather_gain_ratio'] * 100:+.1f} %p" if isinstance(r.get("weather_gain_ratio"), (int, float)) else "-"),
            "표본(일)": r.get("test_n"), "주의": "체계 이동" if r.get("regime_shift_flag") else "",
        }
        for r in sorted(rows, key=lambda r: -(r.get("skill_vs_persistence") or -9))
    ]
    st.markdown(f"#### D+{horizon} 지점별 성적표 (시험 {str(split.get('test_start', ''))[:4]}년)")
    static_table(table)
    st.markdown(
        '<div class="sl-caption">단위: RMSE·MAE는 m와 cm 병기, MSE는 m²와 cm²(=m²×10,000) 병기. '
        '"기상 효과"는 같은 검증 폴드에서 기상 입력이 없는 최상 후보 대비 기상 포함 최상 후보의 개선폭(퍼센트포인트)이며, '
        '양수면 강수·기온 입력이 도움이 됐다는 뜻입니다.</div>',
        unsafe_allow_html=True,
    )
    wx_help = evaluation.get("cells_weather_helps")
    wx_total = evaluation.get("cells_with_weather_candidate")
    if isinstance(wx_help, int) and wx_total:
        _note(
            f"기상 입력이 뚜렷이(0.5%p 이상) 도움이 된 셀은 {wx_help}/{wx_total}개입니다. 원점의 수위에는 어제까지의 비가 이미 반영돼 있어 "
            "'어제 강수'만으로는 단기 정확도가 크게 오르지 않고, 조석이 지배하는 하구에서는 오히려 잡음이 됩니다. "
            "다음 단계는 목표일까지의 강수 예보를 입력으로 넣는 것입니다."
        )
    st.markdown("#### 예측일수별 한눈에 보기 (선택 모델 · RMSE · 개선율)")
    show_table(pd.DataFrame(evaluation_pivot(evaluation)))
    model_summary = evaluation.get("model_summary") or {}
    if model_summary:
        st.markdown(f"#### 모델별 성능지표 · D+{horizon} (지역 전체 지점 평균, 시험 연도)")
        rows = []
        for fid, hs in model_summary.items():
            m = hs.get(str(horizon))
            if not m:
                continue
            rows.append({
                "모델": forecaster_label(fid), "지점 수": m["n_stations"], "선택된 지점": m["selected_count"],
                "평균 RMSE": f"{m['mean_rmse']:.3f} m ({m['mean_rmse'] * 100:.1f} cm)", "평균 MSE": f"{m['mean_mse']:.5f} m² ({m['mean_mse'] * 1e4:.1f} cm²)",
                "평균 MAE": f"{m['mean_mae']:.3f} m ({m['mean_mae'] * 100:.1f} cm)", "평균 개선율": skill_text(m.get("mean_skill")), "중앙값 개선율": skill_text(m.get("median_skill")),
                "_s": m.get("mean_skill") if m.get("mean_skill") is not None else -9,
            })
        rows.sort(key=lambda r: -r["_s"])
        for r in rows:
            r.pop("_s")
        show_table(pd.DataFrame(rows))
        _explain(f"<b>읽는 법:</b> 같은 시험 연도({str(split.get('test_start', ''))[:4]})에서 모든 모델을 채점한 결과입니다. 선택 모델은 검증 폴드 성적으로 정했으므로 시험에서 1등이 아닐 수도 있습니다. 직전값 유지·계절 평년값은 참고 기준선입니다.")
        with st.expander("지점 × 모델 시험 RMSE 행렬 (cm) — 위 예측일수 기준", expanded=False):
            st.markdown(f"**D+{horizon}** · 행 = 지점, 열 = 모델 계열, 값 = 시험 연도 RMSE(cm), ★ = 검증에서 선택된 모델")
            matrix = []
            for s_ in region.get("stations") or []:
                if not s_.get("model_trained"):
                    continue
                try:
                    detail = _cached(f"model:{region['id']}:{s_['id']}", lambda sid=s_["id"]: client.gauge_model(region["id"], sid))
                except APIError:
                    continue
                fams = ((detail.get("horizons") or {}).get(str(horizon)) or {}).get("families") or {}
                row = {"지점": s_["name"]}
                for fid, fam in fams.items():
                    t = fam.get("test") or {}
                    row[forecaster_label(fid)] = (f"{t['rmse'] * 100:.1f}" + ("★" if fam.get("selected") else "")) if isinstance(t.get("rmse"), (int, float)) else "-"
                matrix.append(row)
            if matrix:
                show_table(pd.DataFrame(matrix))
                st.markdown('<div class="sl-caption">★ = 그 지점·지평에서 검증으로 선택된 모델. 값은 시험 연도 RMSE(cm).</div>', unsafe_allow_html=True)
    # 부수 실험: 미래 강수(완전 예보 가정) 상한 — 공식 성적표와 별개, 위 표의 수치는 바뀌지 않는다
    render_future_rain_experiment(client, region, int(horizon), key="sl_ev_fr")
    # 부수 실험: 위성 수체 면적 계열의 모델 직접 비교(LOO) — 표본이 4~10점이라 우열을 가릴 수 없다는 판정까지 함께 보여 준다
    render_area_models_experiment(client, region, key="sl_ev_am")
    winners = winners_table(evaluation)
    if winners:
        st.markdown("#### 지형별로 어떤 모델이 뽑혔나")
        static_table(winners)
        _explain("<b>왜 지형별로 다른가요?</b> 하구(조석 영향)·본류·지류·댐 저수지는 물이 움직이는 방식이 달라, 같은 데이터를 줘도 잘 맞는 모델이 다릅니다. 그래서 지점마다 검증에서 이긴 모델을 따로 씁니다.")

    st.markdown("#### 시험 연도 전체를 예측해 본 결과 (백테스트)")
    stations = [s for s in region.get("stations") or [] if s.get("model_trained")]
    if stations:
        ids = [s["id"] for s in stations]
        preferred = next((s["id"] for s in stations if s.get("satellite_location")), ids[0])
        pick = st.selectbox("지점", ids, index=ids.index(preferred), key="sl_ev_station", format_func=lambda sid: next(s["name"] for s in stations if s["id"] == sid))
        try:
            backtest = _cached(f"backtest:{region['id']}:{pick}:{horizon}", lambda: client.gauge_backtest(region["id"], pick, horizon=int(horizon)))
        except APIError as exc:
            st.info(str(exc))
        else:
            st.plotly_chart(_backtest_figure(backtest), width="stretch", config={"displaylogo": False})
            test = backtest.get("test") or {}
            refs = backtest.get("test_baselines") or {}
            st.markdown(
                f'<div class="sl-caption">{forecaster_label(backtest.get("forecaster_id"))} · 시험 RMSE {_fmt(test.get("rmse"), 4)} m ({_cm(test.get("rmse"))}) · '
                f'MSE {_fmt(test.get("mse"), 5)} m² ({_cm2(test.get("mse"))}) · '
                f'기준선 RMSE {_fmt((refs.get("persistence") or {}).get("rmse"), 4)} m ({_cm((refs.get("persistence") or {}).get("rmse"))}) · '
                f'개선율 {skill_text(backtest.get("skill_vs_persistence"))} · 표본 {test.get("n", 0)}일</div>',
                unsafe_allow_html=True,
            )

    # 위성 기반 시계열 지표 (m·cm)
    try:
        ts_eval = _cached(f"tseval:{region['id']}", lambda: client.satellite_timeseries(region["id"]))
    except APIError:
        ts_eval = {"available": False}
    if ts_eval.get("available"):
        st.markdown("#### 위성 데이터로 잰 수위·수체 시계열 지표 (m · cm 병기)")
        _explain(
            "<b>이 절의 표는 '미래 예측' 성적이 아닙니다.</b> 위성 시점이 지점당 8개뿐이라 위성만으로 예측 모델을 학습할 수 없어, "
            "여기서는 <b>위성으로 잰 값이 얼마나 정확한가</b>(관측 정확도)와 <b>위성 시점 사이에 수위가 얼마나 변하는가</b>(참고치)를 잽니다. "
            "위 예측 모델 성적(RMSE·MSE)은 수위계 10년 자료로 학습한 모델이 <b>앞날</b>을 맞힌 오차이고, 아래 표는 <b>같은 날</b> 위성 값과 수위계 값의 차이입니다. 두 수치를 같은 기준으로 비교하면 안 됩니다."
        )
        series = ts_eval.get("series") or {}
        a = (series.get("A_level_correction") or {})
        corr = a.get("corrected") or {}
        b = (series.get("B_persistence_reference") or {}).get("persistence_of_corrected") or {}
        f_level = (series.get("F_change") or {}).get("level") or {}
        c_summary = (series.get("C_area") or {}).get("summary") or {}
        st.markdown("**① 위성 수위 — 무엇을 재는 오차인가**")
        static_table(
            [
                {"항목": "위성 수위 당일 보정 정확도 (융합 LSTM, in-sample 30표본)", "무엇을 재나": "위성 사진에서 뽑은 수위를 수위계 기준면으로 보정한 값 vs 같은 날 수위계 실측 — 관측 정확도(예측 아님)",
                 "RMSE": f"{corr.get('rmse_m', float('nan')):.4f} m ({corr.get('rmse_cm', float('nan')):.2f} cm)", "MSE": f"{corr.get('mse_m2', float('nan')):.6f} m² ({corr.get('mse_cm2', float('nan')):.2f} cm²)", "MAE": f"{corr.get('mae_m', float('nan')):.4f} m ({corr.get('mae_cm', float('nan')):.2f} cm)", "표본": str(corr.get("n"))},
                {"항목": "위성 수위 보정 · 정식 프로토콜 LODO", "무엇을 재나": "같은 보정을 '한 날짜를 빼고 학습해 그 날짜를 맞히는' 교차검증으로 잰 값 — 보정기의 일반화 정확도",
                 "RMSE": f"{a.get('lodo_rmse_m', float('nan')):.4f} m ({a.get('lodo_rmse_cm', float('nan')):.2f} cm)", "MSE": "–", "MAE": "–", "표본": "날짜 제외 교차검증"},
                {"항목": "위성 시점 사이 '변화 없음' 참고치 (8시점, 예측 아님)", "무엇을 재나": "직전 위성 시점의 보정 수위를 다음 위성 시점(평균 9일 뒤)까지 그대로 뒀을 때 수위계와의 차이 — 위성 관측 사이에 수위가 얼마나 움직였는지의 크기",
                 "RMSE": f"{b.get('rmse_m', float('nan')):.4f} m ({b.get('rmse_cm', float('nan')):.2f} cm)", "MSE": f"{b.get('mse_m2', float('nan')):.6f} m² ({b.get('mse_cm2', float('nan')):.2f} cm²)", "MAE": f"{b.get('mae_m', float('nan')):.4f} m ({b.get('mae_cm', float('nan')):.2f} cm)", "표본": str(b.get("n"))},
                {"항목": "위성 수위 변화량 추적 (무변화 대비 skill)", "무엇을 재나": "두 위성 시점 사이의 수위 변화(Δ)를 위성이 얼마나 맞게 잡았나 vs 같은 기간 수위계 변화 — '변화 없음'으로 두는 것보다 얼마나 나은지(skill)",
                 "RMSE": f"{f_level.get('rmse_m', float('nan')):.4f} m ({f_level.get('rmse_cm', float('nan')):.2f} cm)", "MSE": f"{f_level.get('mse_m2', float('nan')):.6f} m² ({f_level.get('mse_cm2', float('nan')):.2f} cm²)", "MAE": f"{f_level.get('mae_m', float('nan')):.4f} m ({f_level.get('mae_cm', float('nan')):.2f} cm)", "표본": f"{f_level.get('n')} · skill {skill_text(f_level.get('skill_vs_no_change'))}"},
            ]
        )
        st.markdown('<div class="sl-caption">읽는 법: 첫 줄(2.6 cm)이 "위성으로 잰 수위가 수위계와 얼마나 다른가", 셋째 줄(2.4 cm)이 "위성 사이 9일 동안 수위가 평균 얼마나 움직이는가"입니다. 2020년 2~4월 갈수기 자료라 변동이 작고, 홍수기 급변은 들어 있지 않습니다.</div>', unsafe_allow_html=True)
        if c_summary:
            st.markdown("**② 위성 탐지 면적 vs 정답 라벨 면적 (씬별)** — 탐지된 물 면적(km²)이 정답 라벨의 물 면적과 얼마나 다른가. 면적 '예측'이 아니라 <b>탐지 결과의 면적 정확도</b>입니다.", unsafe_allow_html=True)
            static_table([{"센서": k, "씬 수": v.get("n"), "면적 MAE (km²)": _fmt(v.get("mae_km2"), 3), "면적 MSE (km²²)": _fmt(v.get("mse_km2sq"), 3), "MAPE (%)": _fmt(v.get("mape_pct"), 2), "편향 (km²)": _fmt(v.get("bias_km2"), 3)} for k, v in c_summary.items()])
            st.markdown('<div class="sl-caption">읽는 법: ICEYE 면적 MAE 0.74 km² 는 약 40 km² 수체의 1.9 %(MAPE) 차이. 편향(+)은 탐지가 라벨보다 조금 넓게 잡는다는 뜻.</div>', unsafe_allow_html=True)
        bins = (series.get("E_by_horizon") or {}).get("bins") or []
        if bins:
            st.markdown("**③ 관측 간격별 '변화 없음' 오차 (수위계 쌍 99건)** — 위성이 n일 간격으로 찍힌다고 할 때, 그 사이 수위를 '변하지 않았다'고 두면 생기는 오차의 크기. 위성 관측 주기가 길어질수록 무엇을 잃는지 보이는 참고표이며, 어떤 모델의 예측 성적도 아닙니다.", unsafe_allow_html=True)
            static_table([{"관측 간격": b.get("horizon"), "표본(쌍)": b.get("n"), "RMSE": f"{b.get('rmse_m', 0):.4f} m ({b.get('rmse_cm', 0):.2f} cm)", "MSE": f"{b.get('mse_m2', 0):.6f} m² ({b.get('mse_cm2', 0):.2f} cm²)"} for b in bins])
            st.markdown('<div class="sl-caption">읽는 법: 1~7일이나 29~58일이나 2.3~2.5 cm 로 비슷한 것은 갈수기(2~4월)라 수위가 거의 움직이지 않았기 때문입니다. 홍수기라면 간격이 길수록 커집니다.</div>', unsafe_allow_html=True)
        g = series.get("G_iitp") or {}
        if g:
            st.markdown(f'<div class="sl-caption">IITP 성과지표 정합값(정규화 MSE) · <b>씬별 촬영범위 계열(기존 공표)</b>: 수위 {g.get("level_norm_mse")} · 면적 {g.get("area_norm_mse")} · 종합 <b>{g.get("composite_norm_mse")}</b> (목표 {g.get("target")}, {"달성" if g.get("achieved") else "미달"}).</div>', unsafe_allow_html=True)
            fix = g.get("correction") or {}
            if fix:
                st.markdown(f'<div class="sl-caption">같은 계산식에 <b>공통 격자 면적 계열</b>만 넣은 정정값: 수위 {fix.get("level_norm_mse")}(동일) · 면적 {fix.get("area_norm_mse")} · 종합 <b>{fix.get("composite_norm_mse")}</b> (목표 {fix.get("target")}, {"달성" if fix.get("achieved") else "미달"}). 두 값 모두 목표를 만족하며, 공표값은 지우지 않고 병기합니다.</div>', unsafe_allow_html=True)
            if g.get("caveat_ko"):
                _note(g["caveat_ko"])
        for line in ts_eval.get("honesty_ko") or []:
            _note(line)

    # 탐지 정확도(있으면)
    try:
        detection = _cached(f"detection:{region['id']}", lambda: client.satellite_detection(region["id"]))
    except APIError:
        detection = {"available": False}
    if detection.get("available"):
        st.markdown("#### 위성 수체 탐지 정확도 (정답 라벨 대비)")
        summary = detection_summary(detection)
        static_table([{"센서": s["센서"], "씬 수": s["씬 수"], "pooled IoU": _fmt(s["pooled IoU"], 4), "씬 평균 IoU": _fmt(s["씬 평균 IoU"], 3), "최저 IoU": _fmt(s["최저 IoU"], 3), "F1": _fmt(s["F1 (pooled)"], 4)} for s in summary])
        st.markdown(f'<div class="sl-caption">{detection.get("series_ko", "")}</div>', unsafe_allow_html=True)
    elif detection.get("has_scenes") is False or not region.get("scenes_available"):
        _note("이 지역에는 아직 위성 장면이 없어 탐지 정확도를 낼 대상이 없습니다. 위성 수체 탐지 화면에서 장면을 등록하면 탐지·대조 기능이 열립니다.")
    else:
        _note("이 지역은 정답(GT) 라벨이 없어 탐지 정확도(IoU)를 계산하지 않았습니다.")

    # 다른 지역과 비교
    try:
        regions = _cached("regions", client.list_regions)
    except APIError:
        regions = []
    compare = []
    for other in regions:
        try:
            ev = _cached(f"evaluation:{other['id']}", lambda rid=other["id"]: client.gauge_evaluation(rid))
        except APIError:
            continue
        srows = [r["skill_vs_persistence"] for r in ev.get("rows") or [] if int(r["horizon"]) == int(horizon) and r.get("skill_vs_persistence") is not None]
        rr = [r["test_rmse"] for r in ev.get("rows") or [] if int(r["horizon"]) == int(horizon) and isinstance(r.get("test_rmse"), (int, float))]
        if not srows:
            continue  # 이 예측일수를 보고하지 않는 지역(예: 학습 지평이 짧은 사용자 지역)은 빈 행 대신 뺀다
        compare.append({"지역": other["name"], "지점 수": len(srows), f"D+{horizon} 평균 개선율": skill_text(sum(srows) / len(srows)) if srows else "-",
                        f"D+{horizon} 평균 RMSE": f"{sum(rr) / len(rr):.3f} m ({sum(rr) / len(rr) * 100:.1f} cm)" if rr else "-", "기준선보다 나은 지점": f"{sum(1 for s in srows if s > 0)}/{len(srows)}"})
    if compare:
        st.markdown("#### 지역별 비교")
        static_table(compare)
    _note("한계: 원점 이후의 강우·방류량은 아직 모델 입력이 아니어서 홍수기 급변에는 오차가 커집니다(위 '미래 강수(예보)' 실험이 그 상한을 보여 줍니다). 시험 성적은 그 해의 날씨에 좌우되므로 다른 해에 그대로 재현된다고 보장하지 않습니다.")


# --- 진입점 ------------------------------------------------------------------------------------------


def render_slides(*, client_factory: Callable[[], APIClient], execute_quick_sample: Callable[..., None]) -> None:
    st.markdown(SLIDES_CSS, unsafe_allow_html=True)
    client = client_factory()
    current = current_slide()
    region_id = selected_region_id()
    region = None
    if region_id:
        try:
            region = _region(client, region_id)
        except APIError as exc:
            st.error(str(exc))
    if current > 0 and region is None:
        st.session_state.slide = 0
        current = 0
    _rail(current, skipped={"detect"} if region is not None and not region.get("scenes_available") else None)
    _hero(current, region["name"] if region else None)
    key = SLIDE_KEYS[current]
    if key == "region":
        _slide_region(client)
        _nav(current, next_enabled=region is not None, next_label="다음: 데이터 둘러보기 ▶" if region else "먼저 지역을 고르세요")
        return
    assert region is not None
    if key == "data":
        _slide_data(client, region)
    elif key == "detect":
        _slide_detect(client, region)
    elif key == "forecast":
        _slide_forecast(client, region, execute_quick_sample=execute_quick_sample)
    elif key == "result":
        _slide_result(client, region)
    elif key == "evaluate":
        _slide_evaluate(client, region)
    _nav(current)
