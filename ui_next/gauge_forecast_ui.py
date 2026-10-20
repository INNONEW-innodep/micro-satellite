"""지점별 수위 시계열 예측 화면 — 지상 수위계 10개년 자료로 학습·평가·전망.

위성 관측만으로는 시계열 예측을 검증할 표본이 되지 않는다. 이 화면은 지역
레지스트리(`/api/v1/regions`)의 **수위계 일 시계열**로 지점×예측일수마다
모델을 고르고, 학습·선택에 쓰지 않은 시험 연도 오차를 보여 주며, 아무 날짜를
원점으로 전망해 실제와 대조한다. persistence(직전값 유지)는 기준선일 뿐이며
선택 대상이 아니다 — 화면 어디에서도 그것을 예측이라 부르지 않는다.

이 모듈은 API 클라이언트만 쓰고 파일을 직접 읽지 않는다. 위성 수위 대조 표만
walkthrough의 FUSED 로더를 재사용한다.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from datetime import date as Date
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tables import show_table, static_table
from api_client import APIClient, APIError
from eda import padded_range
from future_rain_ui import render_future_rain_experiment
from training_window_ui import active_split as _active_split
from training_window_ui import render_training_window

DEFAULT_REGION_ID = "nakdong_busan"
REPORT_HORIZONS = (1, 3, 5, 7, 14, 30)
TERRAIN_KO = {
    "estuary": "감조(하구)",
    "mainstem": "본류",
    "mainstem_tidal": "본류·감조",
    "tributary": "지류",
    "reservoir": "저수지(댐)",
    "dam": "댐",
    "han": "한강",
}
FORECASTER_KO = {
    "persistence": "직전값 유지 (기준선)",
    "climatology": "계절 평년값",
    "ridge_ar": "릿지 자기회귀 (AR)",
    "seasonal_anomaly_ar": "계절편차 자기회귀",
    "mean_reversion": "평균 회귀 AR(1)",
    "analog_knn": "유사사례 k-NN",
    "ensemble_mean": "계열 앙상블 평균",
    "ridge_arx": "릿지 자기회귀 + 기상 (AR-X)",
    "neural_mlp": "MLP 신경망",
    "neural_lstm": "LSTM 신경망",
    "neural_gru": "GRU 신경망",
    "neural_tcn": "TCN 신경망",
    "prophet_hybrid": "Prophet + 잔차 AR",
}
_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(2,6,23,.42)",
    "font": {"color": "#cbd5e1"},
    "hovermode": "x unified",
    "margin": {"l": 18, "r": 18, "t": 48, "b": 22},
}


# --- 순수 도우미 (테스트 대상) --------------------------------------------------------


def forecaster_label(forecaster_id: str | None) -> str:
    if not forecaster_id:
        return "-"
    return FORECASTER_KO.get(str(forecaster_id), str(forecaster_id))


def terrain_label(terrain: str | None) -> str:
    return TERRAIN_KO.get(str(terrain or ""), str(terrain or "미분류"))


def skill_text(skill: float | None) -> str:
    if skill is None:
        return "-"
    return f"{skill * 100:+.1f} %"


def evaluation_rows_for_station(evaluation: Mapping[str, Any], station_id: str) -> list[dict[str, Any]]:
    """평가표에서 한 지점의 지평별 행을 화면용 열 이름으로 바꾼다."""

    rows: list[dict[str, Any]] = []
    for row in evaluation.get("rows") or []:
        if str(row.get("station_id")) != station_id:
            continue
        rows.append(
            {
                "예측일수": f"D+{int(row['horizon'])}",
                "선택 모델": forecaster_label(row.get("forecaster_id")),
                "검증 RMSE (m)": row.get("val_rmse"),
                "시험 RMSE (m)": row.get("test_rmse"),
                "시험 MSE (m²)": row.get("test_mse"),
                "시험 MAE (m)": row.get("test_mae"),
                "persistence RMSE (m)": row.get("persistence_test_rmse"),
                "평년값 RMSE (m)": row.get("climatology_test_rmse"),
                "skill vs persistence": row.get("skill_vs_persistence"),
                "시험 표본": row.get("test_n"),
            }
        )
    rows.sort(key=lambda item: int(item["예측일수"][2:]))
    return rows


def evaluation_pivot(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """지점 × 지평 표 — 셀은 '모델 · 시험 RMSE · skill'."""

    stations: dict[str, dict[str, Any]] = {}
    for row in evaluation.get("rows") or []:
        entry = stations.setdefault(
            str(row["station_id"]),
            {"지점": row.get("station_name") or row["station_id"], "지형": terrain_label(row.get("terrain"))},
        )
        rmse = row.get("test_rmse")
        skill = row.get("skill_vs_persistence")
        entry[f"D+{int(row['horizon'])}"] = (
            f"{forecaster_label(row.get('forecaster_id'))} · {rmse:.3f} m · {skill_text(skill)}"
            if isinstance(rmse, (int, float))
            else forecaster_label(row.get("forecaster_id"))
        )
        if row.get("regime_shift_flag"):
            entry["주의"] = "체계 이동"
    return list(stations.values())


def winners_table(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for terrain, counts in (evaluation.get("winners_by_terrain") or {}).items():
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        out.append(
            {
                "지형": terrain_label(terrain),
                "가장 많이 선택된 모델": forecaster_label(ordered[0][0]) if ordered else "-",
                "선택 분포": ", ".join(f"{forecaster_label(k)} {v}" for k, v in ordered),
            }
        )
    return out


def parse_uploaded_series(raw: bytes) -> list[dict[str, Any]]:
    """CSV/JSON 업로드를 [{date, water_level_m}] 로 바꾼다. 서버가 다시 검증한다."""

    text = raw.decode("utf-8-sig", errors="replace").strip()
    if not text:
        return []
    if text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            payload = payload.get("rows") or payload.get("series") or []
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(c).strip() for c in frame.columns]
    date_col = next((c for c in frame.columns if c.lower() in {"date", "obs_date", "ymd", "day", "datetime", "timestamp"}), None)
    level_col = next(
        (c for c in frame.columns if c.lower() in {"water_level_m", "water_level", "level_m", "level", "wl", "value"}),
        None,
    )
    if date_col is None or level_col is None:
        raise ValueError("CSV에 date 열과 water_level_m 열이 필요합니다")
    rows = []
    for _, item in frame.iterrows():
        if pd.isna(item[level_col]):
            continue
        rows.append({"date": str(item[date_col])[:10], "water_level_m": float(item[level_col])})
    return rows


# --- 그림 ------------------------------------------------------------------------------


def _series_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: Mapping[str, Any] | None,
    title: str,
    satellite_points: Sequence[Mapping[str, Any]] = (),
) -> go.Figure:
    dates = [str(r["date"]) for r in rows]
    values = [r.get("water_level_m") for r in rows]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates, y=values, mode="lines", name="지상 수위계 (일)",
            line={"color": "#38bdf8", "width": 1.4},
            hovertemplate="%{x}<br>수위 %{y:.3f} m<extra></extra>",
        )
    )
    if satellite_points:
        figure.add_trace(
            go.Scatter(
                x=[p["date"] for p in satellite_points],
                y=[p["level"] for p in satellite_points],
                mode="markers", name="위성 산출 수위 (WBMS · 당일 보정)",
                marker={"color": "#fbbf24", "size": 10, "symbol": "diamond", "line": {"color": "#0f172a", "width": 1}},
                hovertemplate="%{x}<br>위성 수위 %{y:.3f} m<extra></extra>",
            )
        )
    if split:
        bands = (
            ("학습", str(dates[0]) if dates else None, split.get("train_end"), "rgba(148,163,184,.08)"),
            ("검증(연도별 폴드)", split.get("val_start"), split.get("val_end"), "rgba(56,189,248,.10)"),
            ("시험", split.get("test_start"), split.get("test_end"), "rgba(251,191,36,.12)"),
        )
        for label, start, end, color in bands:
            if start and end:
                figure.add_vrect(
                    x0=start, x1=end, fillcolor=color, line_width=0,
                    annotation_text=label, annotation_position="top left",
                    annotation={"font": {"size": 11, "color": "#94a3b8"}},
                )
    figure.update_layout(
        **_LAYOUT, height=360, title={"text": title, "x": 0.01},
        legend={"orientation": "h", "y": -0.22, "yanchor": "top"},
    )
    figure.update_xaxes(gridcolor="#1e293b")
    figure.update_yaxes(title_text="수위 (m)", gridcolor="#1e293b", range=padded_range([v for v in values if v is not None] + [p["level"] for p in satellite_points]))
    return figure


def _forecast_figure(payload: Mapping[str, Any], *, title: str) -> go.Figure:
    history = payload.get("history") or []
    steps = payload.get("steps") or []
    figure = go.Figure()
    h_dates = [str(r["date"]) for r in history]
    h_values = [r.get("water_level_m") for r in history]
    figure.add_trace(
        go.Scatter(
            x=h_dates, y=h_values, mode="lines", name="관측 (원점까지)",
            line={"color": "#94a3b8", "width": 1.6},
            hovertemplate="%{x}<br>관측 %{y:.3f} m<extra></extra>",
        )
    )
    f_dates = [str(s["target_date"]) for s in steps]
    f_values = [s.get("forecast_m") for s in steps]
    upper2 = [s.get("upper_2sigma_m") for s in steps]
    lower2 = [s.get("lower_2sigma_m") for s in steps]
    upper1 = [s.get("upper_1sigma_m") for s in steps]
    lower1 = [s.get("lower_1sigma_m") for s in steps]
    if any(v is not None for v in upper2):
        figure.add_trace(
            go.Scatter(
                x=f_dates + f_dates[::-1], y=upper2 + lower2[::-1], fill="toself",
                fillcolor="rgba(56,189,248,.10)", line={"width": 0}, name="±2σ (검증 잔차)",
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=f_dates + f_dates[::-1], y=upper1 + lower1[::-1], fill="toself",
                fillcolor="rgba(56,189,248,.20)", line={"width": 0}, name="±1σ (검증 잔차)",
                hoverinfo="skip",
            )
        )
    origin_date = str(payload.get("as_of"))
    origin_level = payload.get("origin_level_m")
    figure.add_trace(
        go.Scatter(
            x=[origin_date] + f_dates, y=[origin_level] + f_values, mode="lines+markers",
            name="선택 모델 전망", line={"color": "#38bdf8", "width": 3}, marker={"size": 6},
            hovertemplate="%{x}<br>전망 %{y:.3f} m<extra></extra>",
        )
    )
    persistence = [s.get("persistence_reference_m") for s in steps]
    figure.add_trace(
        go.Scatter(
            x=[origin_date] + f_dates, y=[origin_level] + persistence, mode="lines",
            name="직전값 유지 (기준선 · 예측 아님)", line={"color": "#64748b", "width": 1.5, "dash": "dot"},
            hovertemplate="%{x}<br>직전값 %{y:.3f} m<extra></extra>",
        )
    )
    observed = [s.get("observed_m") for s in steps]
    if any(v is not None for v in observed):
        figure.add_trace(
            go.Scatter(
                x=f_dates, y=observed, mode="lines+markers", name="실제 관측 (원점 이후)",
                line={"color": "#fbbf24", "width": 2}, marker={"size": 6, "symbol": "circle-open"},
                hovertemplate="%{x}<br>실제 %{y:.3f} m<extra></extra>",
            )
        )
    figure.add_vline(x=origin_date, line={"color": "#f472b6", "width": 1, "dash": "dash"})
    figure.add_annotation(x=origin_date, y=1.02, yref="paper", text="전망 원점", showarrow=False, font={"color": "#f472b6", "size": 11})
    all_values = [v for v in h_values + f_values + upper2 + lower2 + observed if v is not None]
    figure.update_layout(**_LAYOUT, height=420, title={"text": title, "x": 0.01}, legend={"orientation": "h", "y": -0.22, "yanchor": "top"})
    figure.update_xaxes(gridcolor="#1e293b")
    figure.update_yaxes(title_text="수위 (m)", gridcolor="#1e293b", range=padded_range(all_values))
    return figure


def _backtest_figure(payload: Mapping[str, Any]) -> go.Figure:
    trace = payload.get("trace") or {}
    dates = trace.get("dates") or []
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=dates, y=trace.get("truth"), mode="lines", name="실측", line={"color": "#fbbf24", "width": 2},
                   hovertemplate="%{x}<br>실측 %{y:.3f} m<extra></extra>")
    )
    figure.add_trace(
        go.Scatter(x=dates, y=trace.get("pred"), mode="lines", name=f"예측 · {forecaster_label(payload.get('forecaster_id'))}",
                   line={"color": "#38bdf8", "width": 2}, hovertemplate="%{x}<br>예측 %{y:.3f} m<extra></extra>")
    )
    figure.add_trace(
        go.Scatter(x=dates, y=trace.get("persistence"), mode="lines", name="직전값 유지 (기준선)",
                   line={"color": "#64748b", "width": 1.2, "dash": "dot"}, hovertemplate="%{x}<br>직전값 %{y:.3f} m<extra></extra>")
    )
    values = [v for v in (trace.get("truth") or []) + (trace.get("pred") or []) if v is not None]
    figure.update_layout(
        **_LAYOUT, height=380,
        title={"text": f"{(payload.get('station') or {}).get('name') or ''} · 시험 연도 백테스트 · D+{payload.get('horizon')} (학습·선택에 쓰지 않은 구간)".strip(" ·"), "x": 0.01},
        legend={"orientation": "h", "y": -0.22, "yanchor": "top"},
    )
    figure.update_xaxes(gridcolor="#1e293b")
    figure.update_yaxes(title_text="수위 (m)", gridcolor="#1e293b", range=padded_range(values))
    return figure


# --- 세션 캐시 ---------------------------------------------------------------------------


def _cache() -> dict[str, Any]:
    if "gauge_cache" not in st.session_state:
        st.session_state.gauge_cache = {}
    return st.session_state.gauge_cache


def invalidate_gauge_cache(prefix: str | None = None) -> None:
    cache = _cache()
    for key in list(cache):
        if prefix is None or key.startswith(prefix):
            cache.pop(key, None)


def _cached(key: str, loader: Callable[[], Any]) -> Any:
    cache = _cache()
    if key not in cache:
        cache[key] = loader()
    return cache[key]


def _satellite_points_for(client: APIClient, region_id: str, station: Mapping[str, Any]) -> list[dict[str, Any]]:
    """같은 지점의 WBMS 위성 산출 수위(지역 파일, API 경유). 없으면 빈 목록."""

    if not station.get("satellite_location"):
        return []
    try:
        payload = _cached(f"satlevels:{region_id}:{station['id']}", lambda: client.satellite_levels(region_id, str(station["id"])))
    except APIError:
        return []
    points = []
    for r in payload.get("records") or []:
        raw = r.get("water_level_m")
        corrected = r.get("corrected_water_level_m")
        # 차트에는 수위계와 같은 기준면인 보정 수위를 올린다. 원값은 지역에 따라 상대 수위(부산)라
        # 같은 축에 두면 축이 눌린다. 보정치가 없으면 원값(대청 조정지·방수로는 EL.m 절대값).
        level = corrected if isinstance(corrected, (int, float)) else raw
        if isinstance(level, (int, float)):
            points.append({"date": r.get("date"), "level": float(level), "raw": raw, "corrected": corrected,
                           "gauge": r.get("gauge_level_m"), "diff": r.get("corrected_minus_gauge_m"),
                           "satellite": r.get("satellite"), "note": r.get("correction_note")})
    _cache()[f"satcomparison:{region_id}:{station['id']}"] = payload.get("comparison")
    return points


# --- 화면 --------------------------------------------------------------------------------


def _region_and_station_controls(client: APIClient, *, key_prefix: str, region_lock: str | None = None):
    regions = _cached("regions", client.list_regions)
    if not regions:
        st.warning("등록된 지역이 없습니다. `regions/` 디렉터리나 `POST /api/v1/regions`로 지역을 추가하세요.")
        return None, None, None
    region_ids = [r["id"] for r in regions]
    if region_lock and region_lock in region_ids:
        region_id = region_lock
    else:
        default_index = region_ids.index(DEFAULT_REGION_ID) if DEFAULT_REGION_ID in region_ids else 0
        region_id = st.selectbox(
            "지역", region_ids, index=default_index, key=f"{key_prefix}_region",
            format_func=lambda rid: next((f"{r['name']} ({rid})" for r in regions if r["id"] == rid), rid),
        )
    region = next(r for r in regions if r["id"] == region_id)
    stations = [s for s in region.get("stations") or []]
    if not stations:
        st.info("이 지역에는 수위계 시계열이 없습니다. 아래 '새 지역·지점 추가'에서 올릴 수 있습니다.")
        return region, None, stations
    station_ids = [s["id"] for s in stations]
    preferred = next((sid for sid in station_ids if sid == "구포대교"), station_ids[0])
    station_id = st.selectbox(
        "지점", station_ids, index=station_ids.index(preferred), key=f"{key_prefix}_station",
        format_func=lambda sid: next(
            (f"{s['name']} · {terrain_label(s.get('terrain'))}" + (" · 위성 연결" if s.get("satellite_location") else "") for s in stations if s["id"] == sid),
            sid,
        ),
    )
    station = next(s for s in stations if s["id"] == station_id)
    return region, station, stations


def _region_badges(region: Mapping[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("위성 장면", "있음" if region.get("satellite_available") else "없음",
                   help="있으면 수체 탐지·위성 수위 산출(WBMS) 경로가 열립니다.")
    cols[1].metric("수위계 시계열", "있음" if region.get("gauge_available") else "없음",
                   help="있으면 지점별 예측 모델 학습·전망이 열립니다.")
    cols[2].metric("지점 수", f"{region.get('n_stations', 0)}개")
    cols[3].metric("학습된 지점", f"{region.get('n_stations_trained', 0)}개")


def render_station_evaluation(evaluation: Mapping[str, Any], station: Mapping[str, Any], *, compact: bool = False) -> None:
    rows = evaluation_rows_for_station(evaluation, str(station["id"]))
    if not rows:
        st.info("이 지점의 학습 산출물이 없습니다. '모델 갈아끼우기' 탭에서 학습을 실행하세요.")
        return
    frame = pd.DataFrame(rows)
    if compact:
        frame = frame[["예측일수", "선택 모델", "시험 RMSE (m)", "persistence RMSE (m)", "skill vs persistence"]]
    display = frame.copy()
    for col in display.columns:
        if col == "skill vs persistence":
            display[col] = display[col].map(skill_text)
        elif "MSE" in col:
            display[col] = display[col].map(lambda v: f"{v:.5f}" if isinstance(v, (int, float)) else "-")
        elif "RMSE" in col or "MAE" in col:
            display[col] = display[col].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-")
    static_table(display)
    regime = next((r for r in evaluation.get("rows") or [] if str(r.get("station_id")) == str(station["id"])), {})
    if regime.get("regime_shift_flag"):
        shift = regime.get("test_mean_shift_sigma")
        st.warning(
            f"이 지점은 시험 연도 평균 수위가 학습·검증 평균에서 {shift:+.1f}σ 벗어나 있습니다 "
            "(기준점 변경·구조물 등 체계 이동 가능성). 장기 지평의 시험 오차는 그 영향을 받습니다."
        )


def _render_overview_tab(client: APIClient, region: Mapping[str, Any], station: Mapping[str, Any]) -> None:
    series = _cached(f"series:{region['id']}:{station['id']}", lambda: client.gauge_series(region["id"], station["id"]))
    rows = series.get("rows") or []
    split = _active_split(region)
    points = _satellite_points_for(client, region["id"], station)
    st.plotly_chart(
        _series_figure(rows, split=split, title=f"{station['name']} · 지상 수위계 일 수위 {len(rows):,}일", satellite_points=points),
        width="stretch", config={"displaylogo": False},
    )
    cols = st.columns(4)
    cols[0].metric("자료 기간", f"{rows[0]['date']} ~ {rows[-1]['date']}" if rows else "-")
    cols[1].metric("관측일 수", f"{len(rows):,}일")
    cols[2].metric("지형", terrain_label(station.get("terrain")))
    cols[3].metric("출처", f"{station.get('source') or '-'} {station.get('source_code') or ''}".strip())
    if station.get("terrain_note"):
        st.caption(f"지형 설명: {station['terrain_note']}")
    if points:
        st.markdown("##### 위성 산출 수위 vs 지상 수위계 (같은 지점 · 같은 날)")
        table = []
        for p in points:
            table.append(
                {
                    "관측일": p["date"],
                    "센서": p.get("satellite") or "-",
                    "위성 산출 원값 (m)": f"{p['raw']:.3f}" if isinstance(p.get("raw"), (int, float)) else "-",
                    "보정 수위 (m)": f"{p['corrected']:.3f}" if isinstance(p.get("corrected"), (int, float)) else "-",
                    "지상 수위계 (m)": f"{p['gauge']:.3f}" if isinstance(p.get("gauge"), (int, float)) else "-",
                    "차이 (보정−수위계)": f"{p['diff']:+.3f}" if isinstance(p.get("diff"), (int, float)) else "-",
                }
            )
        static_table(table)
        comparison = _cache().get(f"satcomparison:{region['id']}:{station['id']}")
        if comparison:
            st.caption(
                f"보정 수위 vs 수위계 · {comparison['n']}일 · RMSE {comparison['rmse_m']:.3f} m · MAE {comparison['mae_m']:.3f} m"
            )
        st.caption(
            "위성 원값은 WBMS 체인(수체 마스크→경계 화소×DEM)의 관측일 산출이고, 보정 수위는 지역별 보정"
            "(부산: 융합 LSTM 당일 보정, 대청: 수위대 상수)으로 수위계 기준면에 맞춘 값입니다. 예측이 아니라 "
            "**같은 날의 두 관측 경로를 대조**하는 표입니다."
        )


def _render_evaluation_tab(client: APIClient, region: Mapping[str, Any], station: Mapping[str, Any]) -> None:
    try:
        evaluation = _cached(f"evaluation:{region['id']}", lambda: client.gauge_evaluation(region["id"]))
    except APIError as exc:
        st.info(f"평가표가 없습니다: {exc}")
        return
    split = evaluation.get("split") or {}
    st.markdown(f"##### {station['name']} · 예측일수별 선택 모델과 오차")
    st.caption(
        f"학습 ≤ {split.get('train_end')} → 검증 {split.get('val_start')} ∼ {split.get('val_end')} (연도별 폴드, 여기서 모델 선택) "
        f"→ 시험 {split.get('test_start')} ∼ {split.get('test_end')} (학습·선택에 미사용, 여기 오차를 보고). "
        "skill = 1 − MSE/MSE(직전값 유지). 양수면 직전값 유지보다 낫습니다."
    )
    render_station_evaluation(evaluation, station)

    st.markdown("##### 지역 전체 · 지점 × 예측일수 (선택 모델 · 시험 RMSE · skill)")
    pivot = evaluation_pivot(evaluation)
    if pivot:
        show_table(pd.DataFrame(pivot))
    total = evaluation.get("cells_total") or 0
    beating = evaluation.get("cells_beating_persistence") or 0
    st.caption(f"전체 {total}셀 중 {beating}셀이 직전값 유지보다 낫습니다 (시험 연도 기준).")
    winners = winners_table(evaluation)
    if winners:
        st.markdown("##### 지형별로 어떤 모델이 선택됐나")
        static_table(winners)
        st.caption("지형(감조·본류·지류·저수지)마다 선택된 모델이 다릅니다 — 단일 모델을 강요하지 않는 근거입니다.")
    # 부수 실험(완전 예보 가정) — 위 성적표의 수치와 별개, 슬라이드 6장과 같은 요소
    available = sorted({int(r["horizon"]) for r in evaluation.get("rows") or []}) or list(REPORT_HORIZONS)
    fr_h = st.select_slider("미래 강수 실험 예측일수", options=available, value=7 if 7 in available else available[0], key="gf_fr_horizon")
    render_future_rain_experiment(client, region, int(fr_h), key="gf_fr")


def _render_forecast_tab(client: APIClient, region: Mapping[str, Any], station: Mapping[str, Any], *, key_prefix: str) -> None:
    period = station.get("period") or []
    last = Date.fromisoformat(str(period[1])[:10]) if len(period) == 2 else Date.today()
    first = Date.fromisoformat(str(period[0])[:10]) if len(period) == 2 else Date(2015, 1, 1)
    split = _active_split(region)
    default_origin = last
    cols = st.columns([2, 2, 3])
    as_of = cols[0].date_input(
        "전망 원점 (이 날짜까지의 관측만 사용)", value=default_origin, min_value=first, max_value=last, key=f"{key_prefix}_asof",
        help="과거 날짜를 고르면 그 뒤 실제 관측과 나란히 그려 맞았는지 바로 볼 수 있습니다.",
    )
    horizon = cols[1].slider("예측일수", min_value=7, max_value=30, value=30, step=1, key=f"{key_prefix}_horizon")
    if split.get("test_start"):
        cols[2].caption(
            f"시험 연도({split.get('test_start')[:4]}) 안의 날짜를 원점으로 두면 시험 구간을 보지 않은 파라미터로 전망합니다. "
            "마지막 관측일이 원점이면 전체 자료로 적합한 배포 파라미터를 씁니다."
        )
    try:
        payload = client.gauge_forecast(region["id"], station["id"], as_of=as_of.isoformat(), horizon=int(horizon))
    except APIError as exc:
        st.error(str(exc))
        return
    st.plotly_chart(
        _forecast_figure(payload, title=f"{station['name']} · {payload.get('as_of')} 원점 D+1~D+{horizon} 전망"),
        width="stretch", config={"displaylogo": False},
    )
    steps = payload.get("steps") or []
    if steps:
        origin = payload.get("origin_level_m")
        last_step = steps[-1]
        metrics = st.columns(4)
        metrics[0].metric("원점 수위", f"{origin:.3f} m" if isinstance(origin, (int, float)) else "-")
        metrics[1].metric(
            f"D+{horizon} 전망", f"{last_step['forecast_m']:.3f} m" if isinstance(last_step.get("forecast_m"), (int, float)) else "-",
            delta=(f"{last_step['forecast_m'] - origin:+.3f} m" if isinstance(last_step.get("forecast_m"), (int, float)) and isinstance(origin, (int, float)) else None),
        )
        sigma = last_step.get("sigma_m")
        metrics[2].metric(f"D+{horizon} ±1σ", f"±{sigma:.3f} m" if isinstance(sigma, (int, float)) else "-", help=payload.get("band_note_ko"))
        used = sorted({forecaster_label(v) for v in (payload.get("models_by_horizon") or {}).values()})
        metrics[3].metric("사용 모델", (used[0] if len(used) == 1 else f"{len(used)}종 (지평별)") if used else "-",
                          help=("예측일수마다 검증에서 고른 모델이 다를 수 있습니다. " + ", ".join(used)) if used else None)
        if len(used) > 1:
            metrics[3].caption("·".join(used))
    if payload.get("coverage_note_ko"):
        st.warning(payload["coverage_note_ko"])
    verification = payload.get("verification")
    if verification:
        st.success(
            f"원점 이후 실제 관측 {verification['n']}일과 비교: RMSE {verification['rmse_m']:.3f} m · "
            f"MAE {verification['mae_m']:.3f} m · 직전값 유지 RMSE {verification['persistence_rmse_m']:.3f} m · "
            f"skill {skill_text(verification.get('skill_vs_persistence'))}"
        )
    st.caption(f"{payload.get('params_note_ko')} · {payload.get('band_note_ko')}")
    with st.expander("전망 표", expanded=False):
        table = [
            {
                "목표일": s["target_date"],
                "D+": s["horizon"],
                "전망 (m)": f"{s['forecast_m']:.3f}" if isinstance(s.get("forecast_m"), (int, float)) else "-",
                "±1σ": f"{s['sigma_m']:.3f}" if isinstance(s.get("sigma_m"), (int, float)) else "-",
                "직전값 (기준선)": f"{s['persistence_reference_m']:.3f}",
                "실제 (m)": f"{s['observed_m']:.3f}" if isinstance(s.get("observed_m"), (int, float)) else "-",
                "오차 (실제−전망)": f"{s['error_m']:+.3f}" if isinstance(s.get("error_m"), (int, float)) else "-",
                "모델": forecaster_label(s.get("forecaster_id")),
            }
            for s in steps
        ]
        show_table(pd.DataFrame(table))


def _render_backtest_tab(client: APIClient, region: Mapping[str, Any], station: Mapping[str, Any], *, key_prefix: str) -> None:
    options = list(REPORT_HORIZONS)
    try:  # 산출물이 보고하는 지평만 고르게 한다(짧게 학습한 사용자 지역은 D+7까지일 수 있다)
        detail = _cached(f"model:{region['id']}:{station['id']}", lambda: client.gauge_model(region["id"], station["id"]))
        reported = [int(h) for h in (detail.get("report_horizons") or []) if str(h) in (detail.get("horizons") or {})]
        if reported:
            options = sorted(set(reported))
    except APIError:
        pass
    horizon = st.select_slider("예측일수", options=options, value=7 if 7 in options else options[-1], key=f"{key_prefix}_bt_h")
    try:
        payload = _cached(f"backtest:{region['id']}:{station['id']}:{horizon}", lambda: client.gauge_backtest(region["id"], station["id"], horizon=int(horizon)))
    except APIError as exc:
        st.info(str(exc))
        return
    st.plotly_chart(_backtest_figure(payload), width="stretch", config={"displaylogo": False})
    test = payload.get("test") or {}
    refs = payload.get("test_baselines") or {}
    cols = st.columns(5)
    cols[0].metric("선택 모델", forecaster_label(payload.get("forecaster_id")))
    cols[1].metric("시험 RMSE", f"{test.get('rmse', float('nan')):.4f} m")
    cols[2].metric("시험 MSE", f"{test.get('mse', float('nan')):.5f} m²")
    cols[3].metric("직전값 유지 RMSE", f"{(refs.get('persistence') or {}).get('rmse', float('nan')):.4f} m")
    cols[4].metric("skill", skill_text(payload.get("skill_vs_persistence")))
    st.caption(
        f"시험 표본 {test.get('n', 0)}일 · 하이퍼파라미터 {json.dumps(payload.get('hparams') or {}, ensure_ascii=False)} · "
        "이 구간은 학습·선택 어디에도 쓰이지 않았습니다."
    )


def _render_swap_tab(client: APIClient, region: Mapping[str, Any], station: Mapping[str, Any], *, key_prefix: str) -> None:
    st.markdown("##### 예측기 목록 (교체 가능)")
    try:
        forecasters = _cached("forecasters", client.list_gauge_forecasters)
    except APIError as exc:
        st.error(str(exc))
        return
    table = [
        {
            "id": f["id"],
            "이름": f.get("name"),
            "역할": "기준선(선택 불가)" if f.get("is_baseline") else ("참고치+후보" if f.get("is_reference") else "후보"),
            "하이퍼파라미터 후보 수": len(f.get("hyperparameter_grid") or []),
            "설명": f.get("description"),
        }
        for f in forecasters
    ]
    show_table(pd.DataFrame(table))
    st.caption(
        "새 모델은 `GaugeForecaster`를 구현해 환경변수 `GAUGE_FORECASTER_PLUGINS=패키지.모듈:클래스`로 등록하면 "
        "화면·API 변경 없이 이 목록과 후보에 들어갑니다. 직전값 유지(persistence)는 어떤 경우에도 선택되지 않습니다."
    )
    st.markdown("##### 이 지점만 다시 학습 — 후보를 제한해 모델을 갈아끼워 보기")
    selectable = [f["id"] for f in forecasters if not f.get("is_baseline")]
    chosen = st.multiselect(
        "후보로 허용할 예측기", selectable, default=selectable, key=f"{key_prefix}_swap_models",
        format_func=forecaster_label,
        help="예: 유사사례 k-NN만 남기면 그 모델이 같은 시험 구간에서 얼마나 나오는지 바로 비교됩니다.",
    )
    scope = st.radio("학습 범위", ["이 지점만", "지역 전체"], horizontal=True, key=f"{key_prefix}_swap_scope")
    if st.button("다시 학습하고 평가", type="primary", icon=":material/model_training:", key=f"{key_prefix}_swap_btn"):
        if not chosen:
            st.error("후보를 하나 이상 고르세요.")
            return
        try:
            before = client.gauge_evaluation(region["id"])
        except APIError:
            before = {"rows": []}
        with st.spinner("검증 연도별 폴드로 후보를 고르고 시험 연도 오차를 계산하는 중… (지점당 수 초)"):
            try:
                after = client.gauge_train(
                    region["id"],
                    stations=[station["id"]] if scope == "이 지점만" else None,
                    forecasters=chosen,
                )
            except APIError as exc:
                st.error(str(exc))
                return
        invalidate_gauge_cache("evaluation:")
        invalidate_gauge_cache("backtest:")
        invalidate_gauge_cache("regions")
        st.session_state[f"{key_prefix}_swap_result"] = {"before": before, "after": after}
    result = st.session_state.get(f"{key_prefix}_swap_result")
    if result:
        before_rows = {int(r["horizon"]): r for r in result["before"].get("rows", []) if str(r.get("station_id")) == station["id"]}
        after_rows = {int(r["horizon"]): r for r in result["after"].get("rows", []) if str(r.get("station_id")) == station["id"]}
        compare = []
        for h in sorted(after_rows):
            b, a = before_rows.get(h), after_rows[h]
            compare.append(
                {
                    "예측일수": f"D+{h}",
                    "이전 모델": forecaster_label(b.get("forecaster_id")) if b else "-",
                    "이전 시험 RMSE": f"{b['test_rmse']:.4f}" if b and isinstance(b.get("test_rmse"), (int, float)) else "-",
                    "새 모델": forecaster_label(a.get("forecaster_id")),
                    "새 시험 RMSE": f"{a['test_rmse']:.4f}" if isinstance(a.get("test_rmse"), (int, float)) else "-",
                    "새 skill": skill_text(a.get("skill_vs_persistence")),
                }
            )
        st.markdown("##### 교체 전후 비교 (같은 시험 구간)")
        static_table(compare)
        st.caption(
            f"허용 후보: {', '.join(forecaster_label(x) for x in (result['after'].get('allowed_forecasters') or [])) or '전부'} · "
            "산출물은 사용자 디렉터리에 저장되어 이후 전망·백테스트가 새 모델을 씁니다."
        )


def _render_ingest_tab(client: APIClient, *, key_prefix: str) -> None:
    st.markdown("##### 새 지역 만들기")
    st.caption("지역은 위성 장면과 수위계 시계열 중 있는 것만 넣으면 됩니다. 여기서는 정의만 만들고, 지점 시계열은 아래에서 올립니다.")
    with st.form(f"{key_prefix}_region_form"):
        rid = st.text_input("지역 id (영문·숫자·_-.)", value="", placeholder="예: geum_river")
        rname = st.text_input("지역 이름", value="", placeholder="예: 금강 중류")
        rdesc = st.text_input("설명", value="")
        submitted = st.form_submit_button("지역 만들기", icon=":material/add_location:")
    if submitted:
        try:
            created = client.create_region({"id": rid.strip(), "name": rname.strip() or rid.strip(), "description": rdesc})
        except APIError as exc:
            st.error(str(exc))
        else:
            invalidate_gauge_cache("regions")
            st.success(f"지역 {created['name']} ({created['id']}) 를 만들었습니다. 아래에서 지점 시계열을 올리세요.")

    st.markdown("##### 지점 수위 시계열 올리기 (CSV/JSON)")
    regions = _cached("regions", client.list_regions)
    if not regions:
        return
    target_region = st.selectbox("대상 지역", [r["id"] for r in regions], key=f"{key_prefix}_ingest_region",
                                 format_func=lambda rid: next((f"{r['name']} ({rid})" for r in regions if r["id"] == rid), rid))
    station_id = st.text_input("지점 id (한글 가능)", value="", key=f"{key_prefix}_ingest_station", placeholder="예: 공주대교")
    terrain = st.selectbox("지형", ["", "estuary", "mainstem", "mainstem_tidal", "tributary", "reservoir"], key=f"{key_prefix}_ingest_terrain", format_func=lambda t: terrain_label(t) if t else "(미지정)")
    replace = st.checkbox("기존 시계열을 버리고 교체", value=False, key=f"{key_prefix}_ingest_replace")
    upload = st.file_uploader("CSV(date,water_level_m) 또는 JSON 행 목록", type=["csv", "json", "txt"], key=f"{key_prefix}_ingest_file")
    st.code("date,water_level_m\n2015-01-01,2.2758\n2015-01-02,2.3063", language="text")
    if st.button("올리기", icon=":material/upload:", key=f"{key_prefix}_ingest_btn"):
        if not station_id.strip() or upload is None:
            st.error("지점 id와 파일이 필요합니다.")
            return
        try:
            rows = parse_uploaded_series(upload.getvalue())
            station = client.upload_gauge_series(
                target_region, station_id.strip(), rows, replace=replace,
                station={"name": station_id.strip(), "terrain": terrain, "source": "user_upload"},
            )
        except (APIError, ValueError) as exc:
            st.error(str(exc))
            return
        invalidate_gauge_cache()
        st.success(f"{station['name']} · {station.get('n_days', 0):,}일 저장. 이제 학습을 실행하세요.")
    if st.button("이 지역 전체 학습 실행", type="primary", icon=":material/model_training:", key=f"{key_prefix}_ingest_train"):
        with st.spinner("학습·선택·평가 중…"):
            try:
                evaluation = client.gauge_train(target_region)
            except APIError as exc:
                st.error(str(exc))
                return
        invalidate_gauge_cache()
        st.success(
            f"지점 {evaluation.get('n_stations', 0)}개 학습 완료 · {evaluation.get('cells_beating_persistence', 0)}/{evaluation.get('cells_total', 0)}셀이 직전값 유지보다 낫습니다."
        )
        if evaluation.get("failures"):
            st.warning(f"실패: {evaluation['failures']}")
    target = next((r for r in regions if r["id"] == target_region), None)
    if target is not None:
        render_training_window(client, target, key=f"{key_prefix}_tw")


def render_gauge_forecast_page(*, client_factory: Callable[[], APIClient]) -> None:
    st.subheader("지점별 수위 시계열 예측 · 지상 수위계 10개년으로 학습·검증")
    st.write(
        "위성은 지점당 수 시점뿐이라 예측 모델을 학습·검증할 표본이 되지 않습니다. 그래서 같은 지점의 "
        "**지상 수위계 일 시계열(2015~2024)**로 지점×예측일수마다 모델을 고르고, 학습·선택에 쓰지 않은 "
        "시험 연도에서 오차를 보고합니다. 직전값 유지(persistence)는 비교 기준선일 뿐 예측 모델로 쓰지 않습니다."
    )
    client = client_factory()
    region, station, _stations = _region_and_station_controls(client, key_prefix="gf")
    if region is None:
        return
    _region_badges(region)
    if station is None:
        _render_ingest_tab(client, key_prefix="gf")
        return
    tabs = st.tabs(["관측 시계열", "모델 평가", "전망", "백테스트", "모델 갈아끼우기", "새 지역·지점 추가"])
    with tabs[0]:
        _render_overview_tab(client, region, station)
    with tabs[1]:
        _render_evaluation_tab(client, region, station)
    with tabs[2]:
        _render_forecast_tab(client, region, station, key_prefix="gf")
    with tabs[3]:
        _render_backtest_tab(client, region, station, key_prefix="gf")
    with tabs[4]:
        _render_swap_tab(client, region, station, key_prefix="gf")
    with tabs[5]:
        _render_ingest_tab(client, key_prefix="gf")


def render_gauge_forecast_step(*, client_factory: Callable[[], APIClient], go_to_phase: Callable[[str], None]) -> bool:
    """따라하기 화면용 축약판. 성공적으로 그리면 True."""

    client = client_factory()
    try:
        region, station, _ = _region_and_station_controls(client, key_prefix="wt_gf", region_lock=DEFAULT_REGION_ID)
    except APIError as exc:
        st.error(str(exc))
        return False
    if region is None or station is None:
        return False
    try:
        evaluation = _cached(f"evaluation:{region['id']}", lambda: client.gauge_evaluation(region["id"]))
    except APIError as exc:
        st.info(f"평가표가 없습니다: {exc}")
        return False
    st.markdown("**① 이 지점에서 어떤 모델이 얼마나 맞았나** (시험 연도 · 학습에 쓰지 않은 구간)")
    render_station_evaluation(evaluation, station, compact=True)
    st.markdown("**② 원점을 골라 전망하고 실제와 대조**")
    _render_forecast_tab(client, region, station, key_prefix="wt_gf")
    if st.button("수위 예측 화면에서 자세히", icon=":material/timeline:", key="wt_gf_more"):
        go_to_phase("수위 예측")
    return True
