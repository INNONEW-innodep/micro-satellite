"""지역별 발표용 한 장 요약 화면.

시연에서 지역 하나를 고르면 그 지역의 (1) 데이터 보유 현황, (2) 수체 탐지 정확도,
(3) 위성 수위 vs 수위계 대조, (4) 수위 시계열 예측 성능, (5) 한계·각주가 한 화면에
나온다. 모든 숫자는 API(지역 레지스트리 파일·학습 산출물)에서 읽으며, 이 모듈이
값을 만들거나 고정하지 않는다. 계열이 다른 수치를 섞지 않도록 표마다 출처를 적는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date as Date
from typing import Any, Callable

import pandas as pd
import streamlit as st
from tables import show_table, static_table
from api_client import APIClient, APIError
from geo_overlay import map_figure
from gauge_forecast_ui import (
    DEFAULT_REGION_ID,
    _cached,
    _forecast_figure,
    evaluation_pivot,
    forecaster_label,
    skill_text,
    terrain_label,
    winners_table,
)

PRESENTATION_CSS = """
<style>
.pr-hero { padding:18px 22px; border:1px solid #334155; border-left:5px solid #22d3ee; border-radius:12px;
  background:linear-gradient(90deg, rgba(8,47,73,.45), rgba(2,6,23,.2)); margin:6px 0 14px; }
.pr-hero h2 { margin:0 0 6px!important; color:#e0f2fe!important; font-size:1.55rem!important; }
.pr-hero p { margin:0; color:#94a3b8; line-height:1.6; }
.pr-badges { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
.pr-badge { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 12px;
  font-size:.78rem; font-weight:700; border:1px solid #334155; color:#7c8ea3; background:#0b1324; }
.pr-badge--on { border-color:rgba(52,211,153,.55); color:#6ee7b7; background:rgba(6,78,59,.22); }
.pr-badge--off { border-color:rgba(248,113,113,.4); color:#fca5a5; background:rgba(127,29,29,.15); }
.pr-section { margin:22px 0 6px; padding-bottom:6px; border-bottom:1px solid #1e293b;
  color:#e2e8f0; font-size:1.15rem; font-weight:800; }
.pr-section small { color:#64748b; font-weight:500; margin-left:10px; font-size:.82rem; }
.pr-note { margin:8px 0 12px; padding:10px 14px; border-left:3px solid #f59e0b;
  background:rgba(120,53,15,.16); color:#fde68a; line-height:1.6; border-radius:0 6px 6px 0; font-size:.9rem; }
.pr-none { margin:8px 0 12px; padding:10px 14px; border:1px dashed #334155; border-radius:8px;
  color:#94a3b8; line-height:1.6; font-size:.92rem; }
</style>
"""


# --- 순수 도우미 (테스트 대상) --------------------------------------------------------


def data_status_badges(region: Mapping[str, Any]) -> list[tuple[str, bool]]:
    """(라벨, 보유 여부) — 발표 첫 줄의 데이터 보유 배지."""

    satellite = region.get("satellite") or {}
    scenes = satellite.get("scenes") or {}
    n_scenes = sum(len(v) for v in scenes.values() if isinstance(v, list))
    sensors = ", ".join(satellite.get("sensors") or []) or "없음"
    badges = [
        (f"위성 장면 · {sensors} {n_scenes}시점" if n_scenes else "위성 장면 · 없음", bool(region.get("satellite_available"))),
        ("GT 수체 라벨", bool(region.get("gt_labels_available"))),
        ("위성 수위 산출", bool(region.get("satellite_levels_available"))),
        (f"수위계 {region.get('n_stations', 0)}지점", bool(region.get("gauge_available"))),
        (f"수위 예측 모델 {region.get('n_stations_trained', 0)}지점", bool(region.get("n_stations_trained"))),
    ]
    return badges


def detection_rows(detection: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sensor, block in (detection.get("sensors") or {}).items():
        for scene in block.get("per_scene") or []:
            rows.append(
                {
                    "센서": sensor,
                    "관측일": scene.get("date"),
                    "water IoU": scene.get("water_iou"),
                    "F1": scene.get("f1"),
                    "Precision": scene.get("precision"),
                    "Recall": scene.get("recall"),
                }
            )
    return rows


def detection_summary(detection: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for sensor, block in (detection.get("sensors") or {}).items():
        pooled = block.get("pooled") or {}
        scenes = block.get("per_scene") or []
        ious = [s["water_iou"] for s in scenes if isinstance(s.get("water_iou"), (int, float))]
        out.append(
            {
                "센서": sensor,
                "씬 수": len(scenes),
                "pooled IoU": pooled.get("water_iou"),
                "씬 평균 IoU": (sum(ious) / len(ious)) if ious else None,
                "최저 IoU": min(ious) if ious else None,
                "F1 (pooled)": pooled.get("f1"),
                "장치": block.get("device"),
            }
        )
    return out


def forecast_headline(evaluation: Mapping[str, Any], horizon: int = 7) -> dict[str, Any]:
    """D+h 행만 모아 대표 수치를 만든다: 평균 skill, 최고/최저 지점."""

    rows = [r for r in evaluation.get("rows") or [] if int(r.get("horizon", 0)) == horizon]
    skills = [(r["station_name"], r.get("skill_vs_persistence"), r.get("test_rmse")) for r in rows if r.get("skill_vs_persistence") is not None]
    if not skills:
        return {"horizon": horizon, "n": 0}
    skills.sort(key=lambda x: x[1], reverse=True)
    return {
        "horizon": horizon,
        "n": len(skills),
        "mean_skill": sum(s[1] for s in skills) / len(skills),
        "best": skills[0],
        "worst": skills[-1],
        "mean_rmse": sum(s[2] for s in skills if s[2] is not None) / max(1, len([s for s in skills if s[2] is not None])),
    }


def region_caveats(region: Mapping[str, Any], detection: Mapping[str, Any], evaluation: Mapping[str, Any]) -> list[str]:
    caveats = []
    if not detection.get("available"):
        caveats.append("GT 수체 라벨이 없어 탐지 정확도(IoU)는 계산하지 않는다. 탐지 산출은 있어도 정확도를 주장하지 않는다.")
    if not (region.get("satellite") or {}).get("available"):
        caveats.append("위성 장면이 없는 지역이다 — 수위계 시계열 예측만 실증한다.")
    split = evaluation.get("split") or {}
    if split:
        caveats.append(
            f"예측 성능은 학습·선택에 쓰지 않은 시험 연도({str(split.get('test_start', ''))[:4]}) 기준이며 그 해 기상 조건에 종속적이다. "
            "강우·방류 입력은 아직 특징에 없는 통계적 외삽이라 홍수기 급변에는 오차가 커진다."
        )
    flagged = sorted({r["station_name"] for r in evaluation.get("rows") or [] if r.get("regime_shift_flag")})
    if flagged:
        caveats.append(f"{', '.join(flagged)}: 시험 연도 평균 수위가 학습·검증 분포에서 1σ 이상 벗어남(기준점 변경·구조물 추정) — 장기 지평 오차에 반영됨.")
    caveats.append("직전값 유지(persistence)는 비교 기준선일 뿐 예측 모델이 아니다. skill = 1 − MSE/MSE(직전값 유지).")
    caveats.append("위성 수위 대조는 관측일 산출과 실측의 비교이지 예측이 아니다. 밴드(±σ)는 검증 잔차의 경험적 산포다.")
    note = (region.get("satellite") or {}).get("notes")
    if note:
        caveats.append(str(note))
    return caveats


# --- 화면 --------------------------------------------------------------------------------


def _section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="pr-section">{title}<small>{sub}</small></div>', unsafe_allow_html=True)


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "-"


def satellite_locations_of(client: APIClient, region: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """위성 수위 산출 구역(AOI) 사전. 지역 목록 항목에 없으면 상세 조회로 본다. 없으면 {}."""

    locations = region.get("satellite_locations")
    if not isinstance(locations, Mapping):
        try:
            detail = _cached(f"region_detail:{region['id']}", lambda: client.get_region(region["id"]))
        except APIError:
            detail = {}
        locations = detail.get("satellite_locations") if isinstance(detail, Mapping) else None
    return {str(k): v for k, v in (locations or {}).items() if isinstance(v, Mapping)}


def _station_map(client: APIClient, region: Mapping[str, Any]) -> None:
    stations = region.get("stations") or []
    locations = satellite_locations_of(client, region)
    with_coords = [s for s in stations if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lon"), (int, float))]
    if not with_coords and not locations:
        return
    st.markdown("**수위계 위치 · 위성 수위 산출 구역**")
    st.plotly_chart(map_figure(region, locations, stations, height=360), width="stretch", config={"displaylogo": False, "scrollZoom": True})
    st.caption(
        f"점 = 지상 수위계 {len(with_coords)}지점(주황 = 위성 산출 구역과 연결, 마우스를 올리면 주소·하천)"
        + (f" · 청록 상자 = 위성 수위 산출 구역 {len(locations)}곳" if locations else "")
    )


def render_presentation_page(*, client_factory: Callable[[], APIClient]) -> None:
    st.markdown(PRESENTATION_CSS, unsafe_allow_html=True)
    client = client_factory()
    try:
        regions = _cached("regions", client.list_regions)
    except APIError as exc:
        st.error(str(exc))
        return
    if not regions:
        st.warning("등록된 지역이 없습니다.")
        return
    region_ids = [r["id"] for r in regions]
    default_index = region_ids.index(DEFAULT_REGION_ID) if DEFAULT_REGION_ID in region_ids else 0
    cols = st.columns([3, 1])
    region_id = cols[0].selectbox(
        "발표할 지역", region_ids, index=default_index, key="pr_region",
        format_func=lambda rid: next((f"{r['name']} ({rid})" for r in regions if r["id"] == rid), rid),
    )
    horizon = cols[1].select_slider("대표 예측일수", options=[1, 3, 5, 7, 10, 14, 30], value=7, key="pr_horizon")
    region = next(r for r in regions if r["id"] == region_id)

    badges = "".join(
        f'<span class="pr-badge {"pr-badge--on" if on else "pr-badge--off"}">{"✓" if on else "✗"} {label}</span>'
        for label, on in data_status_badges(region)
    )
    st.markdown(
        f'<div class="pr-hero"><h2>{region["name"]}</h2><p>{region.get("description") or ""}</p>'
        f'<div class="pr-badges">{badges}</div></div>',
        unsafe_allow_html=True,
    )

    # 1. 데이터 현황
    _section("1. 데이터 보유 현황", "지역 레지스트리 region.json · gauge/series.json")
    satellite = region.get("satellite") or {}
    scenes = satellite.get("scenes") or {}
    left, right = st.columns(2)
    with left:
        st.markdown("**위성 장면**")
        if scenes:
            static_table([{"센서": k, "시점 수": len(v), "관측일": ", ".join(v)} for k, v in scenes.items()])
        else:
            st.markdown('<div class="pr-none">위성 장면 없음 — 수위계 시계열만 보유</div>', unsafe_allow_html=True)
    with right:
        st.markdown("**지상 수위계**")
        stations = region.get("stations") or []
        if stations:
            static_table(
                [
                    {
                        "지점": s["name"],
                        "지형": terrain_label(s.get("terrain")),
                        "기간": " ~ ".join(s.get("period") or []),
                        "관측일": f"{s.get('n_days', 0):,}",
                        "위성 연결": s.get("satellite_location") or "-",
                    }
                    for s in stations
                ]
            )
        else:
            st.markdown('<div class="pr-none">수위계 시계열 없음</div>', unsafe_allow_html=True)
        _station_map(client, region)

    # 2. 탐지 정확도
    _section("2. 수체 탐지 정확도", "GET /regions/{id}/satellite/detection")
    try:
        detection = _cached(f"detection:{region_id}", lambda: client.satellite_detection(region_id))
    except APIError as exc:
        detection = {"available": False, "note_ko": str(exc)}
    if detection.get("available"):
        summary = detection_summary(detection)
        tiles = st.columns(max(1, len(summary)) * 2)
        for index, item in enumerate(summary):
            tiles[index * 2].metric(f"{item['센서']} pooled IoU", _fmt(item["pooled IoU"], 4), help="전 씬 픽셀을 합쳐 계산한 IoU")
            tiles[index * 2 + 1].metric(f"{item['센서']} 씬 평균 / 최저", f"{_fmt(item['씬 평균 IoU'], 3)} / {_fmt(item['최저 IoU'], 3)}")
        frame = pd.DataFrame(detection_rows(detection))
        for col in ("water IoU", "F1", "Precision", "Recall"):
            frame[col] = frame[col].map(lambda v: _fmt(v, 4))
        static_table(frame)
        st.caption(f"계열: {detection.get('series_ko', '')} · {detection.get('protocol_note_ko', '')}")
    else:
        st.markdown(f'<div class="pr-none">{detection.get("note_ko", "탐지 정확도 기록 없음")}</div>', unsafe_allow_html=True)

    # 3. 위성 수위 vs 수위계
    _section("3. 위성 수위 vs 지상 수위계 (같은 지점·같은 날)", "GET /regions/{id}/satellite/levels")
    linked = [s for s in region.get("stations") or [] if s.get("satellite_location")]
    if not linked or not region.get("satellite_levels_available"):
        st.markdown('<div class="pr-none">위성 수위 산출과 연결된 수위계 지점이 없다</div>', unsafe_allow_html=True)
    else:
        rows = []
        detail_frames: dict[str, list[dict[str, Any]]] = {}
        for s in linked:
            try:
                payload = _cached(f"satlevels:{region_id}:{s['id']}", lambda sid=s["id"]: client.satellite_levels(region_id, sid))
            except APIError:
                continue
            comparison = payload.get("comparison") or {}
            rows.append(
                {
                    "수위계 지점": s["name"],
                    "위성 산출 위치": payload.get("loc_id"),
                    "위성 시점 수": payload.get("n"),
                    "보정 수위 대조 일수": comparison.get("n") or 0,
                    "RMSE (m)": _fmt(comparison.get("rmse_m")),
                    "MAE (m)": _fmt(comparison.get("mae_m")),
                }
            )
            detail_frames[s["name"]] = [
                {
                    "관측일": r.get("date"),
                    "센서": r.get("satellite"),
                    "위성 원값 (m)": _fmt(r.get("water_level_m")),
                    "보정 수위 (m)": _fmt(r.get("corrected_water_level_m")),
                    "수위계 (m)": _fmt(r.get("gauge_level_m")),
                    "차이 (m)": f"{r['corrected_minus_gauge_m']:+.3f}" if isinstance(r.get("corrected_minus_gauge_m"), (int, float)) else "-",
                }
                for r in payload.get("records") or []
            ]
        if rows:
            static_table(rows)
            with st.expander("일자별 대조 표", expanded=False):
                for name, frame in detail_frames.items():
                    st.markdown(f"**{name}**")
                    static_table(frame)
        st.caption("보정 수위: 부산 = 융합 LSTM 당일 보정(관측일 절대수위), 대청 = 수위대 상수 보정(인수 증적). 예측이 아니라 두 관측 경로의 대조.")

    # 4. 수위 예측 성능
    _section("4. 수위 시계열 예측 성능", "GET /regions/{id}/gauge/evaluation · 시험 연도 = 학습·선택 미사용")
    try:
        evaluation = _cached(f"evaluation:{region_id}", lambda: client.gauge_evaluation(region_id))
    except APIError as exc:
        evaluation = {}
        st.markdown(f'<div class="pr-none">평가표 없음: {exc}</div>', unsafe_allow_html=True)
    if evaluation:
        split = evaluation.get("split") or {}
        headline = forecast_headline(evaluation, int(horizon))
        tiles = st.columns(4)
        tiles[0].metric("직전값 유지보다 나은 셀", f"{evaluation.get('cells_beating_persistence', 0)} / {evaluation.get('cells_total', 0)}")
        if headline.get("n"):
            tiles[1].metric(f"D+{horizon} 평균 skill", skill_text(headline["mean_skill"]), help="지점 평균, 1 − MSE/MSE(직전값 유지)")
            tiles[2].metric(f"D+{horizon} 최고 지점", f"{headline['best'][0]} {skill_text(headline['best'][1])}", help=f"시험 RMSE {_fmt(headline['best'][2])} m")
            tiles[3].metric(f"D+{horizon} 최저 지점", f"{headline['worst'][0]} {skill_text(headline['worst'][1])}", help=f"시험 RMSE {_fmt(headline['worst'][2])} m")
        st.caption(
            f"학습 ≤ {split.get('train_end')} → 검증 {split.get('val_start')} ∼ {split.get('val_end')} (연도별 폴드에서 모델 선택) → "
            f"시험 {split.get('test_start')} ∼ {split.get('test_end')} (여기 오차를 보고). 셀 = 지점 × 예측일수(D+1·3·5·7·10·14·30)."
        )
        pivot = evaluation_pivot(evaluation)
        if pivot:
            show_table(pd.DataFrame(pivot))
        winners = winners_table(evaluation)
        if winners:
            st.markdown("**지형별로 선택된 모델**")
            static_table(winners)

        # 대표 전망: 시험 연도 한가운데를 원점으로 실제와 대조
        stations = region.get("stations") or []
        trained = [s for s in stations if s.get("model_trained")]
        if trained:
            preferred = next((s for s in trained if s.get("satellite_location")), trained[0])
            default_as_of = None
            if split.get("test_start"):
                year = int(str(split["test_start"])[:4])
                default_as_of = Date(year, 7, 1)
            pick = st.selectbox(
                "대표 지점 (원점 = 시험 연도 7월 1일, 그 뒤 실제 관측과 대조)",
                [s["id"] for s in trained],
                index=[s["id"] for s in trained].index(preferred["id"]),
                key="pr_station",
                format_func=lambda sid: next((s["name"] for s in trained if s["id"] == sid), sid),
            )
            try:
                forecast = client.gauge_forecast(
                    region_id, pick, as_of=default_as_of.isoformat() if default_as_of else None, horizon=30, history_days=90
                )
            except APIError as exc:
                st.info(str(exc))
            else:
                st.plotly_chart(
                    _forecast_figure(forecast, title=f"{next(s['name'] for s in trained if s['id'] == pick)} · {forecast.get('as_of')} 원점 D+1~30 전망 vs 실제"),
                    width="stretch", config={"displaylogo": False},
                )
                verification = forecast.get("verification")
                if verification:
                    st.success(
                        f"원점 이후 30일 실측 대비 RMSE {verification['rmse_m']:.3f} m · MAE {verification['mae_m']:.3f} m · "
                        f"직전값 유지 RMSE {verification['persistence_rmse_m']:.3f} m · skill {skill_text(verification.get('skill_vs_persistence'))} "
                        f"({forecast.get('params_note_ko')})"
                    )
                used = {forecaster_label(v) for v in (forecast.get("models_by_horizon") or {}).values()}
                st.caption(f"사용 모델: {', '.join(sorted(used))} · {forecast.get('band_note_ko')}")

    # 5. 한계
    _section("5. 한계 · 발표 시 함께 말할 것")
    for line in region_caveats(region, detection, evaluation or {}):
        st.markdown(f'<div class="pr-note">{line}</div>', unsafe_allow_html=True)
