"""위성 파이프라인 화면 — 위성 데이터가 들어와서 예측이 나오기까지를 한 장면으로 따라간다.

입력 장면 → 전처리 → 수체 탐지 → 수위·면적 산출 → 시계열 예측 → 평가. 각 단계는
지역 레지스트리의 실제 자산(퀵룩·마스크·위성 수위·학습 산출물)과 API 결과만 보여 준다.
마지막에 "새 장면 등록" 폼이 있어 다른 위성 데이터가 왔을 때 같은 흐름에 들어오는
방법을 보여 준다(POST /regions/{id}/satellite/scenes).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date as Date
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from labels import area_model_label
from tables import static_table
from api_client import APIClient, APIError
from eda import padded_range
from gauge_forecast_ui import (
    DEFAULT_REGION_ID,
    _cached,
    _forecast_figure,
    forecaster_label,
    skill_text,
)
from satellite_input_ui import render_satellite_input
from geo_overlay import overlay_figure
from slides import (
    _LAYOUT,
    _explain,
    _fmt,
    _none,
    _note,
    _satellite_locations,
    _station_map,
    _weather_figure,
    _weather_window,
    region_samples_for,
    render_model_compare,
    render_model_guide,
)

PIPELINE_CSS = """
<style>
.pl-steps { display:flex; gap:6px; margin:6px 0 14px; flex-wrap:wrap; }
.pl-step { flex:1 1 130px; padding:10px 12px; border-radius:12px; border:1px solid #334155; background:#0b1324; }
.pl-step .n { color:#67e8f9; font-weight:900; font-size:.8rem; letter-spacing:.08em; }
.pl-step .t { color:#e0f2fe; font-weight:800; margin-top:2px; }
.pl-step .d { color:#94a3b8; font-size:.78rem; margin-top:4px; line-height:1.45; }
.pl-step--on { border-color:rgba(34,211,238,.6); box-shadow:0 0 0 2px rgba(34,211,238,.12); }
.pl-h { margin:18px 0 6px; padding-bottom:6px; border-bottom:1px solid #1e293b; color:#e2e8f0; font-size:1.15rem; font-weight:800; }
.pl-h small { color:#64748b; font-weight:500; margin-left:10px; font-size:.8rem; }
</style>
"""

STEPS = (
    ("입력", "위성 장면이 들어옵니다", "SAR(ICEYE) 또는 광학(PlanetScope) 원본 + 촬영 정보"),
    ("전처리", "지도에 맞게 펴고 밝기를 맞춥니다", "정사보정·방사보정(γ⁰ dB / 반사도)·3 m 격자 정렬"),
    ("탐지", "AI가 물을 찾습니다", "WBMS U-Net → 수체 마스크(0/1)"),
    ("수위·면적", "마스크에서 수치를 냅니다", "면적(km²)·경계 화소×DEM → 지점 수위, 수위계와 대조"),
    ("예측", "시계열로 앞날을 봅니다", "면적: 마스크 시계열 + 기상 / 수위: 수위계 10년 학습 모델"),
    ("평가", "얼마나 맞았는지 채점합니다", "IoU · RMSE(m·cm) · 개선율"),
)


def _steps_bar(active: int) -> None:
    html = "".join(
        f'<div class="pl-step {"pl-step--on" if i == active else ""}"><div class="n">STEP {i + 1}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (t, _s, d) in enumerate(STEPS)
    )
    st.markdown(f'<div class="pl-steps">{html}</div>', unsafe_allow_html=True)


def _h(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="pl-h">{title}<small>{sub}</small></div>', unsafe_allow_html=True)


def _img(client: APIClient, url: str | None):
    if not url:
        return None
    return _cached(f"img:{url}", lambda: client.fetch_bytes(url))


def _step3_locations(client: APIClient, region: Mapping[str, Any], scene: Mapping[str, Any]) -> None:
    """STEP 3 아래 — 이 장면에서 위성 수위를 재는 구역(AOI)과 짝지은 수위계 위치(슬라이드 3과 같은 그림, 낮게)."""

    locations = _satellite_locations(client, region)
    if not locations:
        return
    st.markdown("**어디를 잰 것인가 — 위성 수위 산출 구역(청록 상자) ↔ 지상 수위계(주황 ▲)**")
    quicklook = scene.get("quicklook") if isinstance(scene.get("quicklook"), Mapping) else None
    urls = scene.get("image_urls") or {}
    left, right = st.columns([1.1, 1])
    with left:
        if not quicklook or not quicklook.get("bounds_epsg32652"):
            _none("이 장면은 퀵룩 좌표 범위가 아직 없어 사진 위에 구역을 겹칠 수 없습니다.")
        else:
            image_url = urls.get("overlay") or urls.get("input")
            try:
                image = _img(client, image_url) if image_url else None
            except APIError:
                image = None
            try:
                figure = overlay_figure(image, quicklook, locations, region.get("stations") or [], zoom_to_locations=True, height=380,
                                        title=f"{scene['sensor']} {scene['date']} · 구역 주변 확대")
            except ValueError as exc:
                _none(f"겹쳐 그리기 실패: {exc}")
            else:
                st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    with right:
        _station_map(client, region, scene=scene, height=380)


def render_pipeline_page(*, client_factory: Callable[[], APIClient], execute_quick_sample: Callable[..., None]) -> None:
    st.markdown(PIPELINE_CSS, unsafe_allow_html=True)
    client = client_factory()
    try:
        regions = _cached("regions", client.list_regions)
    except APIError as exc:
        st.error(str(exc))
        return
    sat_regions = [r for r in regions if r.get("satellite_available") and r.get("scenes_available")]
    other_regions = [r for r in regions if r not in sat_regions]
    if not sat_regions:
        _none("위성 장면 자산이 있는 지역이 없습니다. 아래에서 전처리 GeoTIFF 를 올려 첫 장면을 등록하면 이 화면이 열립니다.")
        if other_regions:
            ids_all = [r["id"] for r in other_regions]
            pick = st.selectbox("장면을 등록할 지역", ids_all, key="pl_region_empty",
                                format_func=lambda rid: next(r["name"] for r in other_regions if r["id"] == rid))
            target = next(r for r in other_regions if r["id"] == pick)
            if render_satellite_input(client, target, key=f"pl_satin_{pick}") is not None:
                if st.button("새로고침 ↻", key="pl_register_reload_empty"):
                    st.rerun()
        return
    ids = [r["id"] for r in sat_regions]
    top = st.columns([2, 2, 2])
    region_id = top[0].selectbox("지역", ids, index=ids.index(DEFAULT_REGION_ID) if DEFAULT_REGION_ID in ids else 0, key="pl_region",
                                 format_func=lambda rid: next(r["name"] for r in sat_regions if r["id"] == rid))
    region = next(r for r in sat_regions if r["id"] == region_id)
    if other_regions:
        top[0].caption("위성 장면이 없는 지역(" + ", ".join(str(r.get("name") or r["id"]) for r in other_regions) + ")은 맨 아래 등록 폼에서 첫 장면을 올리면 여기 나타납니다.")
    try:
        scenes = _cached(f"scenes:{region_id}", lambda: client.satellite_scenes(region_id))
    except APIError as exc:
        st.error(str(exc))
        return
    sensors = scenes.get("sensors") or []
    sensor = top[1].selectbox("센서", sensors, key="pl_sensor", format_func=lambda s: {"ICEYE": "ICEYE (SAR 레이더)", "PlanetScope": "PlanetScope (광학)"}.get(s, s))
    items = [s for s in scenes["scenes"] if s["sensor"] == sensor]
    dates = [s["date"] for s in items]
    picked = top[2].selectbox("장면 (촬영일)", dates, index=len(dates) - 1, key=f"pl_date_{sensor}")
    scene = next(s for s in items if s["date"] == picked)
    urls = scene.get("image_urls") or {}

    _explain(
        "<b>이 화면은 '위성 데이터 한 장이 들어와서 예측까지 가는 길'</b>을 그대로 보여 줍니다. 위 단계표를 따라 내려가면서 "
        "각 단계가 실제로 만든 결과(사진·숫자)를 확인하세요. 맨 아래에서 새 장면을 같은 길에 넣는 방법을 볼 수 있습니다."
    )

    # STEP 1 입력
    _steps_bar(0)
    _h("STEP 1 · 입력 장면", f"{scene['sensor']} · {scene['date']}")
    c1, c2 = st.columns([3, 2])
    with c1:
        image = _img(client, urls.get("input"))
        if image:
            st.image(image, width="stretch", caption=f"{scene['sensor']} 원본 퀵룩 (긴 변 1000px로 축소)")
    with c2:
        static_table([
            {"항목": "센서", "값": f"{scene['sensor']} ({scene.get('sensor_type')})"},
            {"항목": "촬영일", "값": scene["date"]},
            {"항목": "좌표계", "값": scene.get("crs") or region.get("crs") or "EPSG:32652"},
            {"항목": "해상도·크기", "값": (f"{scene['pixel_size_m'][0]} m · {scene['size_px'][0]}×{scene['size_px'][1]} px" if scene.get("pixel_size_m") and scene.get("size_px") else "–")},
            {"항목": "원본 파일", "값": str(scene.get("source_pre", "")).split("/")[-1]},
            {"항목": "정답 라벨", "값": "있음" if scene.get("gt_available") else "없음"},
        ])
        st.markdown('<div class="sl-caption">SAR는 밝기가 후방산란(γ⁰ dB), 광학은 반사도입니다. 물은 SAR에서 어둡고 광학에서는 NIR 반사가 낮습니다.</div>', unsafe_allow_html=True)

    # STEP 2 전처리
    _steps_bar(1)
    _h("STEP 2 · 전처리", "인수 규약 Pre_*.tif + _meta.json")
    pre_rows = [
        {"단계": "정사보정", "설명": "궤도·DEM으로 지형 왜곡을 펴서 지도 좌표(EPSG:32652, 3 m)에 맞춤"},
        {"단계": "방사보정", "설명": "SAR: 후방산란 γ⁰를 dB로 · 광학: 대기보정 반사도(ACOLITE)"},
        {"단계": "격자 정렬", "설명": "같은 지역의 모든 날짜를 같은 3 m 격자에 맞춰 시계열 비교가 되게 함"},
        {"단계": "보조자료", "설명": "SAR는 바다/내륙을 나누는 3-class 마스크, 수위 산출용 DEM"},
    ]
    static_table(pre_rows)
    st.markdown('<div class="sl-caption">부산 ICEYE는 인수팀 GAMMA 전처리본, 대청 PlanetScope는 정사+대기보정 후 Pre 변환본을 그대로 씁니다. 자체 SLC 파이프라인(scripts/iceye_slc_ingest.py)은 보정식 미적용 상태라 여기 쓰지 않습니다.</div>', unsafe_allow_html=True)

    # STEP 3 탐지
    _steps_bar(2)
    _h("STEP 3 · 수체 탐지", "WBMS U-Net (인도 모델)")
    cols = st.columns(3 if urls.get("gt") else 2)
    with cols[0]:
        st.markdown("**탐지 마스크 오버레이 (청록 = 물)**")
        image = _img(client, urls.get("overlay"))
        if image:
            st.image(image, width="stretch")
        else:
            _none("이 장면은 아직 탐지 마스크가 없습니다.")
    with cols[1]:
        st.markdown("**마스크만**")
        image = _img(client, urls.get("mask"))
        if image:
            st.image(image, width="stretch")
        else:
            _none("탐지 대기 — 마스크가 등록되면 여기에 보입니다.")
    if urls.get("gt"):
        with cols[2]:
            st.markdown("**정답 라벨 (노랑)**")
            image = _img(client, urls.get("gt"))
            if image:
                st.image(image, width="stretch")
    try:
        segmenters = _cached("segmenters", lambda: client._request("GET", "/api/v1/segmenters"))
    except APIError:
        segmenters = {}
    seg = (segmenters.get("items") or [{}])[0] if isinstance(segmenters, Mapping) else {}
    metrics = scene.get("metrics") or {}
    tiles = st.columns(4)
    seg_name = str(seg.get("name") or "WBMS U-Net") if seg else "WBMS U-Net"
    tiles[0].metric("탐지 모델", seg_name if len(seg_name) <= 18 else "WBMS U-Net", help=seg_name)
    tiles[1].metric(
        "탐지 수체 면적", f"{scene['water_area_km2']:.2f} km²" if isinstance(scene.get("water_area_km2"), (int, float)) else "-",
        help=("이 장면의 촬영범위 안에서 센 면적입니다. 촬영범위가 날짜마다 달라 이 값끼리 이어 붙이면 시계열이 되지 않습니다"
              + (f" — 날짜 간 비교용 공통 격자 면적은 {scene['area_common_frame_km2']:.2f} km²."
                 if isinstance(scene.get("area_common_frame_km2"), (int, float)) else ".")),
    )
    tiles[2].metric("IoU (정답 대비)", _fmt(metrics.get("water_iou"), 3))
    tiles[3].metric("Precision / Recall", f"{_fmt(metrics.get('precision'), 3)} / {_fmt(metrics.get('recall'), 3)}")
    if seg:
        caps = seg.get("capabilities") or {}
        st.markdown(f'<div class="sl-caption">어댑터 {seg.get("id")} · 실행 {seg.get("execution")} · 받는 센서 {", ".join((caps.get("sensors") or {}).keys())} · 다른 탐지 모델은 SEGMENTER_PLUGINS 로 등록해 교체합니다.</div>', unsafe_allow_html=True)
    _step3_locations(client, region, scene)

    # STEP 4 수위·면적
    _steps_bar(3)
    _h("STEP 4 · 수위·면적 산출", "면적 = 물 화소 × 화소 면적 · 수위 = 경계 화소 × DEM 높이")
    linked = [s for s in region.get("stations") or [] if s.get("satellite_location")]
    rows = []
    for s in linked:
        try:
            payload = _cached(f"satlevels:{region_id}:{s['id']}", lambda sid=s["id"]: client.satellite_levels(region_id, sid))
        except APIError:
            continue
        rec = next((r for r in payload.get("records") or [] if str(r.get("date")) == scene["date"] and str(r.get("satellite", "")).lower() == sensor.lower()), None)
        if rec is None:
            # 같은 날 다른 센서의 산출은 이 장면의 값이 아니다 — 섞어 보이지 않는다(감사 지적). 있으면 참고로만 알린다
            other = next((r for r in payload.get("records") or [] if str(r.get("date")) == scene["date"]), None)
            if other:
                rows.append({"수위계 지점": s["name"], "위성 산출 위치": payload.get("loc_id"), "위성 원값 (m)": "–", "보정 수위 (m)": "–", "지상 수위계 (m)": _fmt(other.get("gauge_level_m")),
                             "차이": f"이 센서 산출 없음 (같은 날 {other.get('satellite')} 산출만 있음)"})
            continue
        if rec:
            rows.append({
                "수위계 지점": s["name"], "위성 산출 위치": payload.get("loc_id"),
                "위성 원값 (m)": _fmt(rec.get("water_level_m")), "보정 수위 (m)": _fmt(rec.get("corrected_water_level_m")),
                "지상 수위계 (m)": _fmt(rec.get("gauge_level_m")),
                "차이": (f"{rec['corrected_minus_gauge_m']:+.3f} m ({rec['corrected_minus_gauge_m'] * 100:+.1f} cm)" if isinstance(rec.get("corrected_minus_gauge_m"), (int, float)) else "-"),
            })
    if rows:
        static_table(rows)
        _explain("<b>읽는 법:</b> 위성 원값은 물 경계의 지형 높이(상대값)이고, 보정 수위는 그것을 수위계 기준면에 맞춘 값입니다. 같은 날 수위계와의 차이가 곧 '위성으로 잰 수위의 오차'입니다.")
    else:
        _none("이 날짜에는 위성 수위 산출 기록이 없습니다(수위 산출은 인수 체인 ③이 만든 시점만 있음).")
    with_area = [s for s in items if isinstance(s.get("water_area_km2"), (int, float))]
    if len(with_area) >= 2:
        dates = [s["date"] for s in with_area]
        raw = [s["water_area_km2"] for s in with_area]
        common = [s.get("area_common_frame_km2") for s in with_area]
        has_common = all(isinstance(v, (int, float)) for v in common)
        marker_colors = ["#f472b6" if d == scene["date"] else "#22d3ee" for d in dates]
        figure = go.Figure()
        if has_common:
            figure.add_trace(go.Scatter(x=dates, y=common, mode="lines+markers", name="공통 격자 (비교 가능)",
                                        marker={"size": 9, "color": marker_colors},
                                        line={"color": "#22d3ee", "width": 2.5},
                                        hovertemplate="%{x}<br>%{y:.2f} km²<extra>공통 격자</extra>"))
            figure.add_trace(go.Scatter(x=dates, y=raw, mode="lines+markers", name="씬별 촬영범위 (비교 불가)",
                                        marker={"size": 7, "color": "#64748b"},
                                        line={"color": "#64748b", "width": 1.6, "dash": "dot"},
                                        hovertemplate="%{x}<br>%{y:.2f} km²<extra>씬별 촬영범위</extra>"))
        else:
            figure.add_trace(go.Scatter(x=dates, y=raw, mode="lines+markers", name="씬별 촬영범위",
                                        marker={"size": 9, "color": marker_colors},
                                        line={"color": "#22d3ee", "width": 2.5},
                                        hovertemplate="%{x}<br>%{y:.2f} km²<extra></extra>"))
        figure.update_layout(**_LAYOUT, height=280, title={"text": f"{sensor} 탐지 수체 면적 시계열 (분홍 = 선택 장면)", "x": 0.01})
        figure.update_yaxes(gridcolor="#1e293b", title_text="km²",
                            range=padded_range(raw + ([v for v in common] if has_common else []), pad_ratio=0.3))
        figure.update_xaxes(type="category")
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
        if has_common:
            _explain(
                "<b>점선(씬별 촬영범위)은 시계열로 읽으면 안 됩니다.</b> 날짜마다 찍힌 범위가 달라, 물이 늘지 않아도 범위가 "
                "옮겨가면 면적이 커집니다(부산 광학 2020-03-25: 03-12 보다 남쪽으로 8.6 km 내려가 하구를 담아 33 → 65 km²). "
                "실선(공통 격자)은 모든 날짜를 같은 창으로 세서 비교가 됩니다 — 같은 구간에서는 28.9 → 30.1 km²(+4.4 %)입니다."
            )

    # STEP 5 예측
    _steps_bar(4)
    _h("STEP 5 · 시계열 예측", "면적: 마스크 시계열 + 기상 · 수위: 수위계 10년 학습 모델")
    samples = [s for s in region_samples_for(region_id) if s.sample_id.endswith(f":{sensor}")]
    left, right = st.columns(2)
    with left:
        st.markdown("**① 수체 면적 예측 (이 센서의 마스크 시계열)**")
        if not samples:
            _none("이 센서의 마스크 시계열 샘플이 없습니다.")
        else:
            sample = samples[0]
            try:
                models = _cached("area_models", client.list_models)
            except APIError:
                models = []
            visible = [m for m in models if m.get("id") not in ("persistence",)] or models
            model_ids = [m["id"] for m in visible]
            area_model = st.selectbox("모델", model_ids, index=model_ids.index("irregular-area-trend") if "irregular-area-trend" in model_ids else 0, key="pl_area_model",
                                      format_func=lambda mid: area_model_label(next((m for m in visible if m["id"] == mid), mid)))
            days = st.select_slider("며칠 뒤까지", options=[7, 14, 30], value=14, key="pl_area_days")
            latest_mask = max((str(x.get("date")) for x in items if x.get("detection_available")), default=None)
            if latest_mask and latest_mask != scene["date"]:
                st.caption(f"면적 예측은 이 센서의 마스크 시계열 마지막 관측({latest_mask}) 다음 날부터 시작합니다 — 선택한 장면({scene['date']})이 아니라 시계열 끝이 원점입니다.")
            if st.button("면적 예측 실행 ▶", key="pl_area_run", type="primary", width="stretch"):
                with st.spinner("마스크 시계열을 백엔드에 보내는 중…"):
                    try:
                        execute_quick_sample(sample, int(days), navigate=False, model_id=area_model)
                    except APIError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.pl_area_done = {"region_id": region_id, "sensor": sensor}
            done = st.session_state.get("pl_area_done")
            result = st.session_state.get("prediction_result") or {}
            if done and done.get("region_id") == region_id and done.get("sensor") == sensor and result.get("steps"):
                steps = result["steps"]
                weather = list(st.session_state.get("weather_data") or [])
                x = [str(s.get("target_date")) for s in steps]
                figure = go.Figure()
                rain_by_date = {str(r.get("timestamp") or r.get("date"))[:10]: r.get("precipitation_mm") for r in weather}
                rain = [rain_by_date.get(d) or 0 for d in x]
                if any(rain):
                    figure.add_trace(go.Bar(x=x, y=rain, name="강수 (mm)", yaxis="y2", marker={"color": "rgba(56,189,248,.35)"}))
                figure.add_trace(go.Scatter(x=x, y=[s.get("water_area_km2") for s in steps], mode="lines+markers", name="예측 면적 (km²)", line={"color": "#22d3ee", "width": 3}))
                figure.update_layout(**_LAYOUT, height=300, title={"text": f"수체 면적 예측 D+1~D+{len(steps)} · {result.get('model_id')}", "x": 0.01},
                                     yaxis={"title": "km²", "gridcolor": "#1e293b", "range": padded_range([s.get("water_area_km2") for s in steps], pad_ratio=0.3)},
                                     yaxis2={"title": "강수 (mm)", "overlaying": "y", "side": "right", "showgrid": False}, legend={"orientation": "h", "y": 1.18})
                st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
                st.markdown('<div class="sl-caption">면적은 센서 공통 범위 안의 근사값이며, 모델은 설명 가능한 기준선입니다(학습 정확도 모델 아님).</div>', unsafe_allow_html=True)
    compare_request: dict | None = None
    guide: Mapping = {}
    with right:
        st.markdown("**② 지점 수위 예측 (수위계 10년 학습 모델)**")
        stations = [s for s in region.get("stations") or [] if s.get("model_trained")]
        if not stations:
            _none("학습된 지점이 없습니다.")
        else:
            sids = [s["id"] for s in stations]
            preferred = next((s["id"] for s in stations if s.get("satellite_location")), sids[0])
            station_id = st.selectbox("지점", sids, index=sids.index(preferred), key="pl_lv_station", format_func=lambda sid: next(s["name"] for s in stations if s["id"] == sid))
            try:
                forecasters = _cached("forecasters", client.list_gauge_forecasters)
            except APIError:
                forecasters = []
            guide = _cached("model_guide", client.gauge_model_guide) if forecasters else {}
            options = ["auto"] + [f["id"] for f in forecasters if not f.get("is_baseline") and not f.get("is_reference") and f["id"] != "climatology"]
            model = st.selectbox("모델", options, key="pl_lv_model", format_func=lambda m: "자동 (검증 최적)" if m == "auto" else forecaster_label(m))
            horizon = st.select_slider("며칠 뒤까지", options=[1, 3, 5, 7, 10, 14, 30], value=14, key="pl_lv_horizon")
            station = next(s for s in stations if s["id"] == station_id)
            period = station.get("period") or []
            last = Date.fromisoformat(str(period[1])[:10]) if len(period) == 2 else Date.today()
            first = Date.fromisoformat(str(period[0])[:10]) if len(period) == 2 else Date(2015, 1, 1)
            scene_day = Date.fromisoformat(scene["date"])
            default = scene_day if first <= scene_day <= last else last
            # 위젯 키에 장면·지점을 넣어 장면을 바꾸면 원점도 그 장면 촬영일로 따라온다(감사: 첫 값에 고정되던 문제)
            as_of = st.date_input("예측 원점 (기본: 이 장면의 촬영일)", value=default, min_value=first, max_value=last, key=f"pl_lv_asof:{region_id}:{station_id}:{scene['date']}")
            if st.button("수위 예측 실행 ▶", key="pl_lv_run", type="primary", width="stretch"):
                try:
                    payload = client.gauge_forecast(region_id, station_id, as_of=as_of.isoformat(), horizon=int(horizon), history_days=60, forecaster=None if model == "auto" else model)
                except APIError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pl_level_done = {"region_id": region_id, "payload": payload, "station": station}
            done = st.session_state.get("pl_level_done")
            if done and done.get("region_id") == region_id:
                payload = done["payload"]
                st.plotly_chart(_forecast_figure(payload, title=f"{done['station']['name']} · {payload.get('as_of')} 원점 D+1~D+{payload.get('horizon')}"), width="stretch", config={"displaylogo": False})
                verification = payload.get("verification")
                if verification:
                    (st.info if payload.get("in_sample") else st.success)(
                        f"실제 대비 RMSE {verification['rmse_m']:.3f} m ({verification['rmse_m'] * 100:.1f} cm) · 기준선 {verification['persistence_rmse_m']:.3f} m · 개선율 {skill_text(verification.get('skill_vs_persistence'))}"
                    )
                st.markdown(f'<div class="sl-caption">{payload.get("params_note_ko", "")}</div>', unsafe_allow_html=True)
                weather_rows = _weather_window(client, region, Date.fromisoformat(str(payload.get("as_of"))[:10]), 30)
                if weather_rows:
                    st.plotly_chart(_weather_figure(weather_rows, "기상청 관측 · 원점 전 30일"), width="stretch", config={"displaylogo": False})
                if st.toggle("모든 모델로 비교 (그래프 + 순위 + 차이 표 + 성능지표)", value=False, key="pl_lv_compare"):
                    # 12모델 표·그래프는 반폭 열에 맞지 않아 두 열 아래(전체 폭)에 그린다
                    compare_request = {"station": done["station"], "as_of": Date.fromisoformat(str(payload.get("as_of"))[:10]), "horizon": int(payload.get("horizon") or horizon)}
    if compare_request:
        st.markdown(f"**모든 모델로 비교 · {compare_request['station']['name']}**")
        render_model_compare(client, region, compare_request["station"], as_of=compare_request["as_of"], horizon=compare_request["horizon"], key="pl_lv")
    if guide:
        render_model_guide(guide, key="pl_guide")

    # STEP 6 평가
    _steps_bar(5)
    _h("STEP 6 · 평가 요약", "자세한 표는 '쉬운 화면 → 6. 성능 평가'")
    try:
        evaluation = _cached(f"evaluation:{region_id}", lambda: client.gauge_evaluation(region_id))
    except APIError:
        evaluation = {}
    try:
        ts_eval = _cached(f"tseval:{region_id}", lambda: client.satellite_timeseries(region_id))
    except APIError:
        ts_eval = {"available": False}
    tiles = st.columns(4)
    tiles[0].metric("탐지 IoU (이 장면)", _fmt(metrics.get("water_iou"), 3), help="정답 라벨 대비. 없으면 계산 안 함")
    rows7 = [r for r in evaluation.get("rows") or [] if int(r.get("horizon", 0)) == 7 and r.get("skill_vs_persistence") is not None]
    tiles[1].metric("수위 예측 D+7 평균 개선율", skill_text(sum(r["skill_vs_persistence"] for r in rows7) / len(rows7)) if rows7 else "-")
    rmse7 = [r["test_rmse"] for r in rows7 if isinstance(r.get("test_rmse"), (int, float))]
    tiles[2].metric("수위 예측 D+7 평균 RMSE", f"{sum(rmse7) / len(rmse7):.3f} m ({sum(rmse7) / len(rmse7) * 100:.1f} cm)" if rmse7 else "-")
    corr = ((ts_eval.get("series") or {}).get("A_level_correction") or {}).get("corrected") or {}
    tiles[3].metric("위성 수위 보정 RMSE", f"{corr['rmse_m']:.3f} m ({corr['rmse_cm']:.1f} cm)" if corr else "-", help="융합 LSTM 당일 보정, in-sample 30표본")
    _note("탐지 정확도는 정답 라벨이 있는 지역만, 위성 수위 지표는 위성 수위 산출·수위계가 모두 있는 지역만 계산됩니다. 수위 예측 성능은 모델이 보지 않은 시험 연도 기준입니다.")

    # 새 장면 등록
    _h("새 위성 데이터가 왔을 때", "POST /api/v1/regions/{id}/satellite/scenes")
    _explain(
        "다른 날짜·다른 센서의 장면이 오면 <b>전처리 GeoTIFF(+탐지 마스크)</b>를 서버 경로나 업로드로 등록합니다. "
        "그러면 ① 퀵룩·오버레이가 만들어지고 ② 같은 센서의 기존 프레임과 <b>같은 격자</b>로 마스크가 잘려 ③ 면적 시계열 예측 샘플에 자동으로 붙습니다. "
        "마스크가 없으면 사진만 등록되고, 탐지 어댑터(WBMS)가 있는 호스트에서는 탐지 잡을 제출할 수 있습니다. "
        "새 센서·새 지역은 카탈로그(backend/app/wbms/catalog.json)와 보조자료(DEM·3-class 마스크)를 먼저 갖춰야 탐지가 돕니다."
    )
    if render_satellite_input(client, region, key=f"pl_satin_{region_id}") is not None:
        if st.button("목록 새로고침 ↻ (센서·장면 선택에 새 장면 반영)", key="pl_register_reload"):
            st.rerun()
    if other_regions:
        names = ", ".join(str(r.get("name") or r["id"]) for r in other_regions)
        with st.expander(f"위성 장면이 없는 지역에 첫 장면 등록 — {names}", expanded=False):
            st.caption("수위계만 있는 지역은 위 지역 선택에 나오지 않습니다. 여기서 전처리 GeoTIFF 를 올리면 그 지역도 이 화면에 나타납니다.")
            ids_other = [r["id"] for r in other_regions]
            pick = st.selectbox("지역", ids_other, key="pl_region_other",
                                format_func=lambda rid: next(r["name"] for r in other_regions if r["id"] == rid))
            target = next(r for r in other_regions if r["id"] == pick)
            if render_satellite_input(client, target, key=f"pl_satin_{pick}") is not None:
                if st.button("새로고침 ↻", key="pl_register_reload_other"):
                    st.rerun()
