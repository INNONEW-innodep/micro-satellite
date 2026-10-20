"""모델별 전망 비교 — 상호작용형 구성 요소 (쉬운 화면 4·5장, 위성 파이프라인 STEP 5 공용).

`GET /regions/{id}/gauge/{station}/forecast/compare` 응답(payload) 하나로 다음을 그린다.

* 모델 보이기/숨기기(다중 선택 + 빠른 버튼: 전체·상위 3·선택 모델만·딥러닝만·기준선 숨기기)
* 보기 조절: 예측 구간만 보기, 과거 N일, 축 단위 m/cm, 차이 보기(선택 모델 대비·실제 대비 cm)
* 순위표: 지표(이번 원점 검증 RMSE·시험 연도 RMSE/MSE/MAE/개선율) 기준 순위, 최고(🥇🥈🥉)·최하위(⚠) 표시
* 파고들기: 모델 하나를 골라 지평별 예측·실제·오차·선택 모델 대비 차이 표와 |오차| 막대
* 기존 성능지표 표와 지평별 차이 표(m·cm 병기)

숫자는 전부 payload(백엔드)에서 오며 여기서는 단위 변환(m→cm)과 정렬만 한다.
검증(verification)이 없는 지역(실제 관측이 아직 없는 원점)에서도 모든 요소가 동작해야 한다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date as Date
from datetime import timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api_client import APIClient, APIError
from eda import padded_range
from gauge_forecast_ui import _cached, forecaster_label, skill_text
from tables import show_table

_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(2,6,23,.42)",
    "font": {"color": "#cbd5e1", "size": 13},
    "hovermode": "x unified",
    "margin": {"l": 18, "r": 18, "t": 52, "b": 22},
}

# 색은 모델(개체)에 고정 — 필터로 모델 수가 바뀌어도 같은 모델은 같은 색
MODEL_COLORS = {
    "ensemble_mean": "#22d3ee", "ridge_ar": "#38bdf8", "ridge_arx": "#0ea5e9", "mean_reversion": "#a78bfa", "seasonal_anomaly_ar": "#c084fc",
    "analog_knn": "#34d399", "climatology": "#d6b98c", "neural_mlp": "#fb7185", "neural_lstm": "#f472b6", "neural_gru": "#e879f9", "neural_tcn": "#f97316",
    "prophet_hybrid": "#facc15",
}
FALLBACK_COLOR = "#64748b"
BASELINE_IDS = frozenset({"climatology", "persistence"})
DEEP_PREFIX = "neural_"
DEFAULT_HORIZONS = (1, 3, 5, 7, 10, 14, 30)
HISTORY_DAYS_DEFAULT = 30

# (지표 id, 표 제목, 이번 원점 검증 필요 여부, 낮을수록 좋은가)
METRICS: tuple[tuple[str, str, bool, bool], ...] = (
    ("verify_rmse", "이번 원점 검증 RMSE", True, True),
    ("verify_mae", "이번 원점 검증 MAE", True, True),
    ("verify_skill", "이번 원점 개선율", True, False),
    ("test_rmse", "시험 연도 RMSE", False, True),
    ("test_mse", "시험 연도 MSE", False, True),
    ("test_mae", "시험 연도 MAE", False, True),
    ("test_skill", "시험 연도 개선율", False, False),
)
PRESETS = ("전체", "상위 3", "선택 모델만", "딥러닝만", "기준선 숨기기")

# 쉬운 화면(SLIDES_CSS)과 파이프라인 화면 어디에 놓여도 같은 모양이 되도록 자체 스타일을 붙인다
COMPARE_CSS = """
<style>
.mc-caption { color:#64748b; font-size:.8rem; line-height:1.55; margin:2px 0 10px; }
.mc-explain { margin:8px 0 14px; padding:12px 16px; border-radius:12px; border:1px solid rgba(56,189,248,.35);
  background:rgba(8,47,73,.28); color:#e0f2fe; line-height:1.7; font-size:.96rem; }
.mc-explain b { color:#7dd3fc; }
.mc-note { margin:8px 0 12px; padding:10px 14px; border-left:3px solid #f59e0b;
  background:rgba(120,53,15,.16); color:#fde68a; line-height:1.6; border-radius:0 6px 6px 0; font-size:.9rem; }
.mc-none { margin:8px 0 12px; padding:14px 16px; border:1px dashed #334155; border-radius:10px; color:#94a3b8; line-height:1.6; }
</style>
"""


# --- 단위·서식 ------------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """숫자면 float, 아니면 None. NaN/inf 도 None — 정렬(순위)과 서식에 'nan' 이 섞이지 않게."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def fmt_m_cm(value: Any, digits_m: int = 3, digits_cm: int = 1) -> str:
    """0.123 → '0.123 m (12.3 cm)'."""
    v = _num(value)
    return f"{v:.{digits_m}f} m ({v * 100:.{digits_cm}f} cm)" if v is not None else "-"


def fmt_m2_cm2(value: Any) -> str:
    v = _num(value)
    return f"{v:.5f} m² ({v * 1e4:.1f} cm²)" if v is not None else "-"


def fmt_cm_signed(value_m: Any, digits: int = 1) -> str:
    v = _num(value_m)
    return f"{v * 100:+.{digits}f} cm" if v is not None else "-"


def fmt_diff_both(value_m: Any) -> str:
    v = _num(value_m)
    return f"{v * 100:+.1f} cm ({v:+.3f} m)" if v is not None else "-"


# --- 순위 ---------------------------------------------------------------------------------


def has_verification(payload: Mapping[str, Any]) -> bool:
    """이번 원점 이후 실제 관측이 있어 모델별 verification 이 채워졌는가."""
    return any(isinstance((item or {}).get("verification"), Mapping) and _num(item["verification"].get("rmse_m")) is not None
               for item in (payload.get("per_model") or {}).values())


def metric_options(payload: Mapping[str, Any]) -> list[str]:
    verified = has_verification(payload)
    return [mid for mid, _label, needs_verify, _lower in METRICS if verified or not needs_verify]


def metric_label(metric_id: str, horizon: int | None = None) -> str:
    label = next((lab for mid, lab, _v, _l in METRICS if mid == metric_id), metric_id)
    if metric_id.startswith("test_") and horizon:
        label += f" · D+{horizon}"
    return label


def _metric_pack(item: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    test = item.get("test_at_horizon") or {}
    ver = item.get("verification") or {}
    ver_rmse = _num(ver.get("rmse_m"))
    return {
        "test": {"rmse": _num(test.get("rmse")), "mse": _num(test.get("mse")), "mae": _num(test.get("mae")), "skill": _num(test.get("skill_vs_persistence")),
                 "n": _num(test.get("n")), "horizon": _num(test.get("horizon"))},
        "verify": {"rmse": ver_rmse, "mse": ver_rmse ** 2 if ver_rmse is not None else None, "mae": _num(ver.get("mae_m")),
                   "skill": _num(ver.get("skill_vs_persistence")), "n": _num(ver.get("n")), "persistence_rmse": _num(ver.get("persistence_rmse_m"))},
    }


def metric_value(entry: Mapping[str, Any], metric_id: str) -> float | None:
    group, name = metric_id.split("_", 1)
    return (entry.get("test" if group == "test" else "verify") or {}).get(name)


def rank_models(payload: Mapping[str, Any], metric_id: str) -> list[dict[str, Any]]:
    """지표 기준 순위. 값이 없는 모델은 순위 없이 뒤에 붙는다. best/worst 는 순위가 2개 이상일 때만 표시."""
    lower_better = next((low for mid, _lab, _v, low in METRICS if mid == metric_id), True)
    entries = []
    for fid in payload.get("models") or []:
        item = (payload.get("per_model") or {}).get(fid) or {}
        pack = _metric_pack(item)
        entries.append({"id": fid, "label": forecaster_label(fid), "selected": bool((item.get("test_at_horizon") or {}).get("selected")),
                        "test": pack["test"], "verify": pack["verify"], "value": None, "rank": None, "best": False, "worst": False})
    for e in entries:
        e["value"] = metric_value(e, metric_id)
    ranked = [e for e in entries if e["value"] is not None]
    unranked = [e for e in entries if e["value"] is None]
    ranked.sort(key=lambda e: (e["value"] if lower_better else -e["value"], e["label"]))
    for i, e in enumerate(ranked, start=1):
        e["rank"] = i
    if len(ranked) >= 2:
        ranked[0]["best"] = True
        ranked[-1]["worst"] = True
    elif ranked:
        ranked[0]["best"] = True
    return ranked + unranked


def rank_badge(entry: Mapping[str, Any]) -> str:
    rank = entry.get("rank")
    if rank is None:
        return "–"
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(int(rank), "")
    text = f"{medal} {rank}위".strip()
    if entry.get("worst"):
        text += " ⚠ 최하위"
    return text


def rank_suffix(entry: Mapping[str, Any] | None) -> str:
    """그래프 범례에 붙일 짧은 꼬리표: ' · 1위 🥇', ' · 12위 ⚠'."""
    if not entry or entry.get("rank") is None:
        return ""
    rank = int(entry["rank"])
    if entry.get("best"):
        return f" · {rank}위 🥇"
    if entry.get("worst"):
        return f" · {rank}위 ⚠"
    return f" · {rank}위"


def ranking_rows(ranking: Sequence[Mapping[str, Any]], metric_id: str) -> list[dict[str, Any]]:
    """순위표 행: 순위·모델·RMSE m(cm)·MSE m²(cm²)·MAE m(cm)·개선율·선택. 지표 그룹(시험/이번 원점)의 값을 보인다."""
    group = "test" if metric_id.startswith("test_") else "verify"
    prefix = "시험" if group == "test" else "이번 원점"
    rows = []
    for e in ranking:
        vals = e.get(group) or {}
        rows.append({
            "순위": rank_badge(e),
            "모델": e["label"] + (" ★선택" if e.get("selected") else ""),
            f"{prefix} RMSE": fmt_m_cm(vals.get("rmse")),
            f"{prefix} MSE": fmt_m2_cm2(vals.get("mse")),
            f"{prefix} MAE": fmt_m_cm(vals.get("mae")),
            f"{prefix} 개선율": skill_text(vals.get("skill")),
            "표본": f"{int(vals['n'])}" if _num(vals.get("n")) is not None else "-",
            "선택": "★ 검증 최적" if e.get("selected") else "",
        })
    return rows


# --- 보이기 필터 -----------------------------------------------------------------------------


def selected_model_ids(payload: Mapping[str, Any]) -> list[str]:
    """지평별 선택 모델 id(중복 제거, 등장 순)."""
    seen: list[str] = []
    for r in payload.get("table") or []:
        fid = r.get("selected_model")
        if fid and fid not in seen:
            seen.append(fid)
    return seen


def apply_preset(payload: Mapping[str, Any], ranking: Sequence[Mapping[str, Any]], preset: str) -> list[str]:
    """빠른 버튼 → 보일 모델 id 목록. 결과가 비면 전체로 되돌린다(아무것도 안 보이는 화면 방지)."""
    models = list(payload.get("models") or [])
    if preset == "전체":
        out = models
    elif preset == "상위 3":
        out = [e["id"] for e in ranking if e.get("rank") is not None][:3]
    elif preset == "선택 모델만":
        out = [fid for fid in selected_model_ids(payload) if fid in models]
    elif preset == "딥러닝만":
        out = [fid for fid in models if fid.startswith(DEEP_PREFIX)]
    elif preset == "기준선 숨기기":
        out = [fid for fid in models if fid not in BASELINE_IDS]
    else:
        out = models
    return out or models


def sanitize_visible(payload: Mapping[str, Any], visible: Sequence[str] | None) -> list[str]:
    models = list(payload.get("models") or [])
    if visible is None:
        return models
    return [fid for fid in models if fid in set(visible)]


# --- 차이·파고들기 자료 -----------------------------------------------------------------------


def difference_series(payload: Mapping[str, Any], visible: Sequence[str] | None = None) -> dict[str, Any]:
    """모델 − 선택 모델(cm), 모델 − 실제(cm), 실제 − 선택(cm) 계열. 실제가 없으면 vs_observed 는 빈 dict."""
    table = payload.get("table") or []
    ids = sanitize_visible(payload, visible)
    dates = [r["target_date"] for r in table]
    horizons = [int(r["horizon"]) for r in table]
    observed = [_num(r.get("observed_m")) for r in table]
    has_obs = any(v is not None for v in observed)
    vs_selected: dict[str, list[float | None]] = {}
    vs_observed: dict[str, list[float | None]] = {}
    for fid in ids:
        diffs, errs = [], []
        for r, obs in zip(table, observed):
            m = (r.get("models") or {}).get(fid) or {}
            fc = _num(m.get("forecast_m"))
            d = _num(m.get("diff_vs_selected_m"))
            if d is None and fc is not None and _num(r.get("selected_m")) is not None:
                d = fc - float(r["selected_m"])
            diffs.append(d * 100 if d is not None else None)
            errs.append((fc - obs) * 100 if has_obs and fc is not None and obs is not None else None)
        vs_selected[fid] = diffs
        if has_obs:
            vs_observed[fid] = errs
    obs_vs_sel = [(obs - float(r["selected_m"])) * 100 if obs is not None and _num(r.get("selected_m")) is not None else None
                  for r, obs in zip(table, observed)] if has_obs else None
    return {"dates": dates, "horizons": horizons, "vs_selected": vs_selected, "vs_observed": vs_observed, "observed_vs_selected": obs_vs_sel, "has_observed": has_obs}


def drilldown_rows(payload: Mapping[str, Any], fid: str) -> list[dict[str, Any]]:
    """한 모델의 지평별 예측·실제·오차·선택 모델 대비 차이(m·cm 병기)."""
    rows = []
    for r in payload.get("table") or []:
        m = (r.get("models") or {}).get(fid) or {}
        fc, obs, sel = _num(m.get("forecast_m")), _num(r.get("observed_m")), _num(r.get("selected_m"))
        err = fc - obs if fc is not None and obs is not None else None
        diff = _num(m.get("diff_vs_selected_m"))
        if diff is None and fc is not None and sel is not None:
            diff = fc - sel
        rows.append({
            "D+": int(r["horizon"]), "목표일": r["target_date"],
            "예측": fmt_m_cm(fc), "실제": fmt_m_cm(obs), "오차 (예측−실제)": fmt_diff_both(err),
            "선택 모델": forecaster_label(r.get("selected_model")), "선택 예측": fmt_m_cm(sel), "선택 대비 차이": fmt_diff_both(diff),
            "_abs_err_cm": abs(err) * 100 if err is not None else None, "_diff_cm": diff * 100 if diff is not None else None,
        })
    return rows


def drilldown_summary(payload: Mapping[str, Any], fid: str) -> dict[str, Any]:
    rows = drilldown_rows(payload, fid)
    errs = [r["_abs_err_cm"] for r in rows if r["_abs_err_cm"] is not None]
    diffs = [r["_diff_cm"] for r in rows if r["_diff_cm"] is not None]
    worst = max(rows, key=lambda r: r["_abs_err_cm"] if r["_abs_err_cm"] is not None else -1) if errs else None
    return {
        "n_observed": len(errs), "mean_abs_err_cm": sum(errs) / len(errs) if errs else None, "max_abs_err_cm": max(errs) if errs else None,
        "worst_horizon": worst["D+"] if worst else None, "mean_abs_diff_cm": sum(abs(d) for d in diffs) / len(diffs) if diffs else None,
        "max_abs_diff_cm": max(abs(d) for d in diffs) if diffs else None,
    }


def compare_difference_rows(payload: Mapping[str, Any], horizons: Sequence[int] = DEFAULT_HORIZONS, unit: str = "both") -> list[dict[str, Any]]:
    """지평별 모델 예측과 선택 모델 대비 차이 표. unit: 'both' → '2.279 m (+1.2 cm)', 'cm' → '227.9 cm (+1.2 cm)', 'm' → '2.279 m (+0.012 m)'."""

    def cell(fc: float | None, diff: float | None) -> str:
        if fc is None or diff is None:
            return "-"
        if unit == "cm":
            return f"{fc * 100:.1f} cm ({diff * 100:+.1f} cm)"
        if unit == "m":
            return f"{fc:.3f} m ({diff:+.3f} m)"
        return f"{fc:.3f} m ({diff * 100:+.1f} cm)"

    def level(v: float | None) -> str:
        if v is None:
            return "-"
        return f"{v * 100:.1f} cm" if unit == "cm" else (f"{v:.3f} m" if unit == "m" else fmt_m_cm(v))

    rows = []
    for r in payload.get("table") or []:
        if int(r["horizon"]) not in horizons:
            continue
        row = {"D+": r["horizon"], "목표일": r["target_date"], "선택 모델": forecaster_label(r["selected_model"]), "선택 예측": level(_num(r.get("selected_m"))),
               "실제": level(_num(r.get("observed_m"))), "직전값": level(_num(r.get("persistence_m")))}
        for fid in payload.get("models") or []:
            m = (r.get("models") or {}).get(fid) or {}
            row[forecaster_label(fid)] = cell(_num(m.get("forecast_m")), _num(m.get("diff_vs_selected_m")))
        rows.append(row)
    return rows


def compare_metric_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """모델별 성능지표: 시험 연도(산출물) + (있으면) 이번 원점 검증. 시험 개선율 내림차순."""

    rows = []
    for fid, item in (payload.get("per_model") or {}).items():
        test = item.get("test_at_horizon") or {}
        ver = item.get("verification") or {}
        rows.append({
            "모델": forecaster_label(fid) + (" ★선택" if test.get("selected") else ""),
            f"시험 RMSE D+{test.get('horizon', '')}": fmt_m_cm(test.get("rmse")),
            "시험 MSE": fmt_m2_cm2(test.get("mse")),
            "시험 MAE": fmt_m_cm(test.get("mae")),
            "시험 개선율": skill_text(test.get("skill_vs_persistence")),
            "이번 원점 RMSE": fmt_m_cm(ver.get("rmse_m")),
            "이번 원점 개선율": skill_text(ver.get("skill_vs_persistence")) if ver else "-",
            "_skill": test.get("skill_vs_persistence") if test.get("skill_vs_persistence") is not None else -9,
        })
    rows.sort(key=lambda r: -r["_skill"])
    for r in rows:
        r.pop("_skill", None)
    return rows


# --- 그래프 -------------------------------------------------------------------------------


def _hover_level(name: str) -> str:
    return "%{x}<br>" + name + " %{y:.3f} m (%{customdata:.1f} cm)<extra></extra>"


def _cm_data(values: Sequence[float | None]) -> list[float | None]:
    return [v * 100 if v is not None else None for v in values]


def _line_style(fid: str, entry: Mapping[str, Any] | None) -> dict[str, Any]:
    style = {"color": MODEL_COLORS.get(fid, FALLBACK_COLOR), "width": 1.6}
    if entry and entry.get("best"):
        style["width"] = 3.2
    elif entry and entry.get("worst"):
        style["width"] = 1.4
        style["dash"] = "dashdot"
    return style


def _compare_figure(
    payload: Mapping[str, Any], title: str, *, visible: Sequence[str] | None = None, ranking: Sequence[Mapping[str, Any]] | None = None,
    forecast_only: bool = False, history_days: int | None = None, show_persistence: bool = True, show_observed: bool = True, unit: str = "m",
) -> go.Figure:
    """모델별 전망을 한 그래프에: 굵은 청록 = 선택 모델, 색선 = 계열(1위 굵게·최하위 점쇄선), 노랑 = 실제.

    unit='cm' 이면 y 축을 cm 로 그린다(호버는 항상 m·cm 병기)."""

    scale = 100.0 if unit == "cm" else 1.0
    rank_by_id = {e["id"]: e for e in ranking or []}
    figure = go.Figure()
    table = payload.get("table") or []
    dates = [r["target_date"] for r in table]
    origin_date, origin_level = str(payload.get("as_of")), _num(payload.get("origin_level_m"))
    keep = int(history_days) if history_days else max(21, 2 * len(table))
    history = [] if forecast_only else (payload.get("history") or [])[-keep:]
    values: list[float] = []

    def add(x, y, name, **kw):
        ys = [_num(v) for v in y]
        values.extend(v for v in ys if v is not None)
        figure.add_trace(go.Scatter(x=x, y=[v * scale if v is not None else None for v in ys], name=name, customdata=_cm_data(ys), hovertemplate=_hover_level(name), **kw))

    if history:
        add([r["date"] for r in history], [r.get("water_level_m") for r in history], "관측 (원점까지)", mode="lines", line={"color": "#94a3b8", "width": 1.6})
    for fid in sanitize_visible(payload, visible):
        ys = [(r.get("models") or {}).get(fid, {}).get("forecast_m") for r in table]
        if all(v is None for v in ys):
            continue
        entry = rank_by_id.get(fid)
        add([origin_date] + dates, [origin_level] + ys, forecaster_label(fid) + rank_suffix(entry), mode="lines", line=_line_style(fid, entry), opacity=0.92)
    add([origin_date] + dates, [origin_level] + [r.get("selected_m") for r in table], "선택 모델 (검증 최적)", mode="lines+markers", line={"color": "#22d3ee", "width": 4}, marker={"size": 6})
    if show_persistence:
        add([origin_date] + dates, [origin_level] + [r.get("persistence_m") for r in table], "직전값 유지 (기준선)", mode="lines", line={"color": "#64748b", "width": 1.2, "dash": "dot"})
    observed = [_num(r.get("observed_m")) for r in table]
    if show_observed and any(v is not None for v in observed):
        add(dates, observed, "실제 관측", mode="lines+markers", line={"color": "#fbbf24", "width": 2.5}, marker={"size": 7, "symbol": "circle-open"})
    figure.add_vline(x=origin_date, line={"color": "#f472b6", "width": 1, "dash": "dash"})
    figure.add_annotation(x=origin_date, y=1.0, yref="paper", text="예측 원점", showarrow=False, font={"color": "#f472b6", "size": 11}, xanchor="left", yanchor="bottom")
    figure.update_layout(**_LAYOUT, height=520, title={"text": title, "x": 0.01}, legend={"orientation": "h", "y": -0.2})
    figure.update_xaxes(gridcolor="#1e293b")
    if forecast_only and dates:
        try:
            start = (Date.fromisoformat(origin_date[:10]) - timedelta(days=1)).isoformat()
        except ValueError:
            start = origin_date
        figure.update_xaxes(range=[start, dates[-1]])
    yrange = padded_range(values)
    figure.update_yaxes(title_text=f"수위 ({'cm' if unit == 'cm' else 'm'})", gridcolor="#1e293b", range=[v * scale for v in yrange] if yrange else None)
    return figure


def _difference_figure(
    payload: Mapping[str, Any], title: str, *, visible: Sequence[str] | None = None, ranking: Sequence[Mapping[str, Any]] | None = None, kind: str = "selected",
    show_observed: bool = True, show_persistence: bool = True,
) -> go.Figure:
    """차이 그래프(cm). kind='selected' → 모델 − 선택 모델(0 = 선택 모델), kind='observed' → 모델 − 실제(0 = 실제, 오차).

    실제 − 선택 선은 몇십 cm 라 모델 간 몇 cm 차이를 눌러 버리므로 '실제 관측' 체크를 끄면 y 축이 모델 차이에 맞춰진다."""
    series = difference_series(payload, visible)
    rank_by_id = {e["id"]: e for e in ranking or []}
    figure = go.Figure()
    dates = series["dates"]
    source = series["vs_observed"] if kind == "observed" else series["vs_selected"]
    values: list[float] = []
    for fid, ys in source.items():
        if all(v is None for v in ys):
            continue
        values.extend(v for v in ys if v is not None)
        entry = rank_by_id.get(fid)
        name = forecaster_label(fid) + rank_suffix(entry)
        figure.add_trace(go.Scatter(x=dates, y=ys, mode="lines+markers", name=name, line=_line_style(fid, entry), marker={"size": 5}, opacity=0.92,
                                    customdata=[v / 100 if v is not None else None for v in ys], hovertemplate="%{x}<br>" + name + " %{y:+.1f} cm (%{customdata:+.3f} m)<extra></extra>"))
    if kind == "selected" and show_observed and series["observed_vs_selected"] and any(v is not None for v in series["observed_vs_selected"]):
        ys = series["observed_vs_selected"]
        values.extend(v for v in ys if v is not None)
        figure.add_trace(go.Scatter(x=dates, y=ys, mode="lines+markers", name="실제 − 선택 모델", line={"color": "#fbbf24", "width": 2.5}, marker={"size": 7, "symbol": "circle-open"},
                                    customdata=[v / 100 if v is not None else None for v in ys], hovertemplate="%{x}<br>실제 − 선택 %{y:+.1f} cm (%{customdata:+.3f} m)<extra></extra>"))
    if kind == "selected" and show_persistence:
        persistence = [(_num(r.get("persistence_m")) - _num(r.get("selected_m"))) * 100 if _num(r.get("persistence_m")) is not None and _num(r.get("selected_m")) is not None else None
                       for r in payload.get("table") or []]
        if any(v is not None for v in persistence):
            values.extend(v for v in persistence if v is not None)
            figure.add_trace(go.Scatter(x=dates, y=persistence, mode="lines", name="직전값 유지 − 선택 모델", line={"color": "#64748b", "width": 1.2, "dash": "dot"},
                                        customdata=[v / 100 if v is not None else None for v in persistence], hovertemplate="%{x}<br>직전값 − 선택 %{y:+.1f} cm (%{customdata:+.3f} m)<extra></extra>"))
    zero_label = "0 = 실제 관측" if kind == "observed" else "0 = 선택 모델"
    figure.add_hline(y=0, line={"color": "#22d3ee" if kind == "selected" else "#fbbf24", "width": 2})
    figure.add_annotation(x=0.0, xref="paper", y=0, text=zero_label, showarrow=False, font={"color": "#22d3ee" if kind == "selected" else "#fbbf24", "size": 11}, xanchor="left", yanchor="bottom")
    figure.update_layout(**_LAYOUT, height=440, title={"text": title, "x": 0.01}, legend={"orientation": "h", "y": -0.25})
    figure.update_xaxes(gridcolor="#1e293b")
    yrange = padded_range(values + [0.0]) if values else None
    figure.update_yaxes(title_text="차이 (cm)", gridcolor="#1e293b", zeroline=False, range=yrange)
    return figure


def _drilldown_figure(payload: Mapping[str, Any], fid: str, title: str) -> go.Figure:
    """한 모델의 지평별 |오차|(실제 있을 때) 또는 선택 모델 대비 차이 막대(cm)."""
    rows = drilldown_rows(payload, fid)
    has_err = any(r["_abs_err_cm"] is not None for r in rows)
    x = [f"D+{r['D+']}" for r in rows]
    y = [r["_abs_err_cm"] if has_err else r["_diff_cm"] for r in rows]
    color = MODEL_COLORS.get(fid, FALLBACK_COLOR)
    figure = go.Figure(go.Bar(x=x, y=y, name="|오차|" if has_err else "선택 대비 차이", marker={"color": color, "line": {"color": "rgba(2,6,23,.9)", "width": 1}},
                              customdata=[[r["목표일"], (v or 0) / 100] for r, v in zip(rows, y)],
                              hovertemplate="%{x} (%{customdata[0]})<br>" + ("|오차| %{y:.1f} cm" if has_err else "차이 %{y:+.1f} cm") + " (%{customdata[1]:.3f} m)<extra></extra>"))
    layout = dict(_LAYOUT)
    layout["hovermode"] = "closest"
    figure.update_layout(**layout, height=260, title={"text": title, "x": 0.01}, showlegend=False, bargap=0.25)
    figure.update_xaxes(gridcolor="#1e293b")
    figure.update_yaxes(title_text="|오차| (cm)" if has_err else "차이 (cm)", gridcolor="#1e293b")
    return figure


# --- 화면 --------------------------------------------------------------------------------


def _state_key(key: str, name: str) -> str:
    return f"{key}_cmp_{name}"


def render_compare_payload(payload: Mapping[str, Any], station_name: str, *, key: str) -> None:
    """이미 받은 compare payload 로 상호작용형 비교 화면을 그린다(결과 장에서 재사용)."""

    st.markdown(COMPARE_CSS, unsafe_allow_html=True)
    models = list(payload.get("models") or [])
    if not models:
        st.markdown('<div class="mc-none">이 산출물에는 모델별 파라미터가 없습니다(재학습 필요).</div>', unsafe_allow_html=True)
        return
    horizon = int(payload.get("horizon") or len(payload.get("table") or []))
    verified = has_verification(payload)
    options = metric_options(payload)

    # ① 순위 지표
    c_metric, c_view = st.columns([1.2, 1])
    with c_metric:
        metric_id = st.radio("순위 기준 지표", options, format_func=lambda m: metric_label(m, horizon), horizontal=True, key=_state_key(key, "metric"),
                             help="이번 원점 검증 = 원점 이후 실제 관측과 비교(있을 때만). 시험 연도 = 학습·선택에 쓰지 않은 연도 성적.")
    ranking = rank_models(payload, metric_id)
    rank_by_id = {e["id"]: e for e in ranking}
    with c_view:
        view = st.radio("그래프", ["수위", "차이 (선택 모델 대비)", "오차 (실제 대비)"] if verified else ["수위", "차이 (선택 모델 대비)"], horizontal=True, key=_state_key(key, "view"))
        unit = st.radio("축 단위 (수위 그래프)", ["m", "cm"], horizontal=True, key=_state_key(key, "unit"), disabled=view != "수위",
                        help="차이·오차 그래프는 항상 cm. 호버에는 항상 m·cm 를 함께 보입니다.")

    # ② 보일 모델 — 빠른 버튼은 다중 선택 위젯이 만들어지기 전에 상태를 바꾼다
    vis_key = _state_key(key, "visible")
    stored = st.session_state.get(vis_key)
    if stored is not None:
        st.session_state[vis_key] = sanitize_visible(payload, stored)
    else:
        st.session_state[vis_key] = models
    cols = st.columns(len(PRESETS) + 2)
    for col, preset in zip(cols, PRESETS):
        if col.button(preset, key=_state_key(key, f"preset_{preset}"), width="stretch"):
            st.session_state[vis_key] = apply_preset(payload, ranking, preset)
    show_persistence = cols[len(PRESETS)].checkbox("직전값 기준선", value=True, key=_state_key(key, "persist"))
    show_observed = cols[len(PRESETS) + 1].checkbox("실제 관측", value=True, key=_state_key(key, "observed"), disabled=not verified,
                                                   help="차이 그래프에서 끄면 y 축이 모델 간 차이(몇 cm)에 맞춰집니다.")
    visible = st.multiselect("보일 모델 (선택 모델은 항상 표시)", models, key=vis_key, format_func=lambda fid: forecaster_label(fid) + rank_suffix(rank_by_id.get(fid)))

    # ③ 보기 조절
    v1, v2 = st.columns([1, 2])
    forecast_only = v1.toggle("예측 구간만 보기", value=False, key=_state_key(key, "forecast_only"), help="x 축을 원점~마지막 목표일로 좁히고 y 축을 보이는 선에 맞춥니다.")
    history_len = len(payload.get("history") or [])
    max_days = max(7, min(90, history_len or 90))
    if max_days > 7:
        # 원점을 옮겨 history 가 줄면 저장된 값이 새 max 를 넘을 수 있다 → 위젯 생성 전에 잘라 둔다
        hd_key = _state_key(key, "history_days")
        if isinstance(st.session_state.get(hd_key), int):
            st.session_state[hd_key] = max(7, min(int(st.session_state[hd_key]), max_days))
        else:
            st.session_state[hd_key] = min(HISTORY_DAYS_DEFAULT, max_days)
        history_days = v2.slider("과거 N일 (수위 그래프)", min_value=7, max_value=max_days, step=1, key=hd_key, disabled=forecast_only or view != "수위")
    else:
        # 원점 이전 관측이 7일 이하(계열 시작 직후 원점) — 슬라이더는 min<max 여야 하므로 있는 만큼만 그린다
        history_days = max_days
        v2.caption(f"과거 N일: 원점 이전 관측이 {history_len}일뿐이라 있는 만큼만 그립니다.")

    best = next((e for e in ranking if e.get("best")), None)
    worst = next((e for e in ranking if e.get("worst")), None)
    title = f"{station_name} · {payload.get('as_of')} 원점 · D+1~D+{horizon} 전망 비교 · 모델 {len(models)}종 (표시 {len(visible)}종)"
    if view == "수위":
        figure = _compare_figure(payload, title, visible=visible, ranking=ranking, forecast_only=forecast_only, history_days=history_days,
                                 show_persistence=show_persistence, show_observed=show_observed, unit=unit)
    elif view.startswith("차이"):
        figure = _difference_figure(payload, title + " · 선택 모델 대비 차이 (cm)", visible=visible, ranking=ranking, kind="selected", show_observed=show_observed, show_persistence=show_persistence)
    else:
        figure = _difference_figure(payload, title + " · 실제 대비 오차 (cm)", visible=visible, ranking=ranking, kind="observed")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False}, key=_state_key(key, "chart_" + ("lv" if view == "수위" else "df")))
    st.markdown(
        '<div class="mc-caption">읽는 법: 굵은 청록 = 지평별 검증 최적(선택) 모델, 노랑 동그라미 = 실제 관측, 회색 점선 = 직전값 유지 기준선. '
        f'범례의 순위는 <b>{metric_label(metric_id, horizon)}</b> 기준이며 🥇 1위는 굵게, ⚠ 최하위는 점쇄선으로 그립니다. '
        '선을 클릭하면 범례에서 숨기고, 드래그하면 확대됩니다. 호버에는 m 와 cm 를 함께 보입니다.</div>',
        unsafe_allow_html=True,
    )
    if best or worst:
        b = f"🥇 1위 <b>{best['label']}</b>" if best else ""
        w = f" · ⚠ 최하위 <b>{worst['label']}</b>" if worst else ""
        m_value = best["value"] if best else None
        vtxt = ""
        if m_value is not None:
            vtxt = f" ({skill_text(m_value)})" if metric_id.endswith("skill") else (f" ({fmt_m2_cm2(m_value)})" if metric_id.endswith("_mse") else f" ({fmt_m_cm(m_value)})")
        st.markdown(f'<div class="mc-explain">{metric_label(metric_id, horizon)} 기준 {b}{vtxt}{w}. 선택 모델(★)은 검증 연도 성적으로 고른 것이라 이번 원점에서 1위가 아닐 수 있습니다.</div>', unsafe_allow_html=True)

    # ④ 순위표
    st.markdown(f"**모델 순위** · {metric_label(metric_id, horizon)} 기준 · 단위 m(cm), m²(cm²)")
    show_table(pd.DataFrame(ranking_rows(ranking, metric_id)))
    if not verified:
        st.markdown('<div class="mc-note">이 원점 이후 실제 관측이 아직 없어 "이번 원점 검증" 지표는 계산되지 않았습니다. 순위는 시험 연도 성적으로만 매깁니다.</div>', unsafe_allow_html=True)

    # ⑤ 파고들기
    drill_ids = [e["id"] for e in ranking]
    drill = st.selectbox("자세히 볼 모델", drill_ids, key=_state_key(key, "drill"), format_func=lambda fid: forecaster_label(fid) + rank_suffix(rank_by_id.get(fid)))
    if drill:
        summary = drilldown_summary(payload, drill)
        entry = rank_by_id.get(drill) or {}
        tiles = st.columns(4)
        tiles[0].metric("순위", rank_badge(entry).replace(" ⚠ 최하위", " ⚠") if entry else "-", help=metric_label(metric_id, horizon))
        if summary["n_observed"]:
            tiles[1].metric("평균 |오차|", f"{summary['mean_abs_err_cm']:.1f} cm", help=f"{summary['mean_abs_err_cm'] / 100:.3f} m · 실제 관측 {summary['n_observed']}일")
            tiles[2].metric("최대 |오차|", f"{summary['max_abs_err_cm']:.1f} cm", help=f"{summary['max_abs_err_cm'] / 100:.3f} m", delta=f"D+{summary['worst_horizon']}", delta_color="off")
        else:
            tiles[1].metric("평균 |오차|", "-", help="실제 관측 없음")
            tiles[2].metric("최대 |오차|", "-", help="실제 관측 없음")
        tiles[3].metric("선택 모델과 평균 차이", f"{summary['mean_abs_diff_cm']:.1f} cm" if summary["mean_abs_diff_cm"] is not None else "-",
                        help=f"최대 {summary['max_abs_diff_cm']:.1f} cm" if summary["max_abs_diff_cm"] is not None else None)
        st.plotly_chart(_drilldown_figure(payload, drill, f"{forecaster_label(drill)} · 지평별 " + ("|오차| (cm)" if summary["n_observed"] else "선택 모델 대비 차이 (cm)")),
                        width="stretch", config={"displaylogo": False}, key=_state_key(key, "drill_chart"))
        rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in drilldown_rows(payload, drill)]
        show_table(pd.DataFrame(rows))

    # ⑥ 기존 표
    with st.expander("모델별 성능지표 표 (시험 연도 + 이번 원점)", expanded=False):
        show_table(pd.DataFrame(compare_metric_rows(payload)))
    with st.expander("지평별 모델 예측과 선택 모델 대비 차이", expanded=False):
        table_unit = st.radio("표 단위", ["m (cm 차이)", "cm", "m"], horizontal=True, key=_state_key(key, "table_unit"))
        unit_code = {"m (cm 차이)": "both", "cm": "cm", "m": "m"}[table_unit]
        show_table(pd.DataFrame(compare_difference_rows(payload, unit=unit_code)))
    st.markdown(f'<div class="mc-caption">{payload.get("note_ko", "")} {payload.get("params_note_ko", "")}</div>', unsafe_allow_html=True)


def render_model_compare(client: APIClient, region: Mapping[str, Any], station: Mapping[str, Any], *, as_of: Date, horizon: int, key: str) -> None:
    """compare API 호출 + 상호작용형 비교 화면. payload 는 session_state[f'{key}_compare_payload'] 에 남겨 결과 장에서 재사용한다."""

    try:
        payload = _cached(f"compare:{region['id']}:{station['id']}:{as_of}:{horizon}",
                          lambda: client.gauge_forecast_compare(region["id"], station["id"], as_of=as_of.isoformat(), horizon=int(horizon)))
    except APIError as exc:
        st.error(str(exc))
        return
    if not payload.get("models"):
        st.markdown('<div class="mc-none">이 산출물에는 모델별 파라미터가 없습니다(재학습 필요).</div>', unsafe_allow_html=True)
        return
    render_compare_payload(payload, station.get("name") or station.get("id") or "", key=key)
    st.session_state[f"{key}_compare_payload"] = payload
