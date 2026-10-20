"""수체 면적 시계열 모델 직접 비교 실험 — 공유 화면 요소와 순수 표 도우미.

`scripts/eval_area_models_experiment.py` 가 지역마다 남기는
``regions/<id>/models/gauge/experiments/area_models.json`` (스키마 ``watercast.satellite.experiment.area-models/1``)
을 ``GET /api/v1/regions/{id}/gauge/experiments`` 로 받아, 센서별 **공통 격자** 면적 계열과 후보별
LOO(leave-one-out) 성적표, 그리고 수위→면적 검량곡선의 문턱 판정을 그린다.

규율 — 이 화면이 반드시 말해야 하는 것:

* 면적은 **공통 격자**에서 다시 잰 값이다. ``scenes.json`` 의 장면별 면적은 촬영 범위가 날짜마다 달라 계열이 아니다.
* 표본이 센서당 4~10점이라 **모델 우열을 가릴 수 없다**(n=4 에서 부호검정 최소 p = 0.125).
* 검량곡선이 문턱을 넘지 못하면 파생 면적 예측 모델은 **공개하지 않는다**.

순수 도우미(`sensor_rows`·`candidate_table_rows`·`points_table_rows`·`bridge_table_rows`·`notes_ko`)는
Streamlit 없이도 쓰이고(발표자료 스크립트가 JSON 을 직접 읽어 같은 함수를 쓴다), 숫자는 만들지 않는다 —
전부 실험 JSON 에서 온다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

EXPANDER_TITLE = "수체 면적 시계열 모델 — 직접 비교(표본 적음)"
SIGN_TEST_MIN_P = {4: 0.125, 5: 0.0625, 6: 0.0313, 7: 0.0156}
FAMILY_KO: dict[str, str] = {
    "persistence": "직전 관측 유지 (기준선)",
    "series_mean": "계열 평균",
    "linear_trend": "시간 선형추세",
    "doy_climatology": "연중 시기 평년값",
    "ar1_mean_reversion": "AR(1) 평균회귀",
    "level_linear": "같은 날 수위 선형회귀",
    "level_isotonic": "같은 날 수위 단조회귀",
}


# --- 순수 도우미 ---------------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _fmt(value: Any, spec: str = ".3f", suffix: str = "") -> str:
    number = _num(value)
    return f"{format(number, spec)}{suffix}" if number is not None else "-"


def _pct(value: Any) -> str:
    number = _num(value)
    return f"{number * 100:+.1f} %" if number is not None else "-"


def series_blocks(payload: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    """{센서: 계열 블록}. 알 수 없는 모양이면 빈 사전."""

    if not isinstance(payload, Mapping):
        return {}
    series = payload.get("series") or payload.get("sensors")
    if not isinstance(series, Mapping):
        return {}
    return {str(k): v for k, v in series.items() if isinstance(v, Mapping)}


def sensors_available(payload: Mapping[str, Any] | None) -> list[str]:
    return sorted(series_blocks(payload))


def candidate_label(key: str, candidate: Mapping[str, Any] | None = None) -> str:
    """후보 id(`level_linear:구포대교`) → 화면 이름."""

    family, _, station = str(key).partition(":")
    name = str((candidate or {}).get("name_ko") or FAMILY_KO.get(family) or family)
    station = str((candidate or {}).get("level_station") or station or "")
    return f"{name} — {station}" if station else name


def sensor_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """센서별 한 줄 요약 — 관측 수·면적 범위·최저 LOO 후보·관측 잡음 하한."""

    out: list[dict[str, Any]] = []
    for sensor, block in sorted(series_blocks(payload).items()):
        candidates = block.get("candidates") or {}
        best = block.get("best_by_loo") or (min(candidates, key=lambda k: candidates[k].get("rmse_km2", 9e9)) if candidates else None)
        best_c = candidates.get(best) or {}
        base = candidates.get("persistence") or {}
        period = block.get("period") or ["", ""]
        out.append({
            "센서": sensor,
            "관측 수": block.get("n"),
            "기간": f"{str(period[0])[:10]} ~ {str(period[-1])[:10]}" if period else "-",
            "면적 평균": _fmt(block.get("area_mean_km2"), ".2f", " km²"),
            "면적 변동폭": _fmt(block.get("area_range_km2"), ".2f", " km²"),
            "직전값 유지 LOO RMSE": _fmt(base.get("rmse_km2"), ".3f", " km²"),
            "최저 LOO 후보": candidate_label(str(best), best_c) if best else "-",
            "그 RMSE": _fmt(best_c.get("rmse_km2"), ".3f", " km²"),
            "관측 잡음 하한": _fmt(block.get("observation_noise_floor_km2"), ".2f", " km²"),
        })
    return out


def candidate_table_rows(payload: Mapping[str, Any] | None, sensor: str) -> list[dict[str, Any]]:
    """한 센서의 후보별 LOO 표 (LOO RMSE 오름차순)."""

    block = series_blocks(payload).get(sensor)
    if not isinstance(block, Mapping):
        return []
    candidates = block.get("candidates") or {}
    order = block.get("ranking_by_loo_rmse")
    keys = [k for k in order if k in candidates] if isinstance(order, list) else sorted(
        candidates, key=lambda k: _num(candidates[k].get("rmse_km2")) if _num(candidates[k].get("rmse_km2")) is not None else 9e9
    )
    rows = []
    for key in keys:
        candidate = candidates[key] or {}
        sign = candidate.get("sign_test_vs_persistence") or {}
        rows.append({
            "후보": candidate_label(key, candidate),
            "LOO RMSE": _fmt(candidate.get("rmse_km2"), ".3f", " km²"),
            "LOO MAE": _fmt(candidate.get("mae_km2"), ".3f", " km²"),
            "직전값 유지 대비": _pct(candidate.get("skill_vs_persistence")),
            "이긴 점 / 진 점": (f"{sign.get('wins')} / {sign.get('losses')}" if sign else "-"),
            "부호검정 p": _fmt(sign.get("p_two_sided"), ".3f") if sign else "-",
        })
    return rows


def points_table_rows(payload: Mapping[str, Any] | None, sensor: str) -> list[dict[str, Any]]:
    """한 센서의 관측 목록 — 공통 격자 면적과 장면 자기 footprint 면적을 나란히."""

    block = series_blocks(payload).get(sensor)
    if not isinstance(block, Mapping):
        return []
    rows = []
    for point in block.get("points") or []:
        if not isinstance(point, Mapping):
            continue
        levels = point.get("levels_m") or {}
        row = {
            "날짜": str(point.get("date"))[:10],
            "공통 격자 면적": _fmt(point.get("area_km2"), ".3f", " km²"),
            "장면 자기 범위 면적": _fmt(point.get("scene_own_footprint_area_km2"), ".3f", " km²"),
        }
        for station, value in sorted(levels.items()):
            row[f"{station} 수위"] = _fmt(value, ".3f", " m")
        rows.append(row)
    return rows


def bridge_table_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """수위→면적 검량곡선 판정 표."""

    bridge = (payload or {}).get("bridge") if isinstance(payload, Mapping) else None
    if not isinstance(bridge, Mapping):
        return []
    rows = []
    for item in bridge.get("evaluated") or []:
        if not isinstance(item, Mapping):
            continue
        rows.append({
            "센서": item.get("sensor"),
            "수위 지점": item.get("level_station"),
            "짝 수": item.get("n_pairs"),
            "상관 r": _fmt(item.get("pearson_r"), "+.3f"),
            "기울기": _fmt(item.get("slope_km2_per_m"), "+.1f", " km²/m"),
            "관측일 수위폭": _fmt(item.get("level_range_m"), ".4f", " m"),
            "10년 수위폭 → 면적폭": _fmt(item.get("implied_area_range_km2"), ".0f", " km²"),
            "판정": "통과" if item.get("defensible") else "탈락",
            "탈락 사유": "; ".join(item.get("failed_criteria_ko") or []) or "-",
        })
    return rows


def bridge_defensible(payload: Mapping[str, Any] | None) -> bool:
    bridge = (payload or {}).get("bridge") if isinstance(payload, Mapping) else None
    return bool(isinstance(bridge, Mapping) and bridge.get("any_defensible"))


def bridge_demo_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """시연(공개 아님) 파생 면적 예측의 지평별 성적. 시연이 없으면 빈 목록."""

    bridge = (payload or {}).get("bridge") if isinstance(payload, Mapping) else None
    demo = bridge.get("demonstration") if isinstance(bridge, Mapping) else None
    if not isinstance(demo, Mapping):
        return []
    rows = []
    for horizon, entry in sorted((demo.get("horizons") or {}).items(), key=lambda kv: int(kv[0])):
        if not isinstance(entry, Mapping):
            continue
        level = entry.get("level_equivalent") or {}
        rows.append({
            "지평": f"D+{horizon}",
            "선택 모델": entry.get("forecaster_id"),
            "시험 RMSE": _fmt(entry.get("test_rmse_km2"), ".2f", " km²"),
            "직전값 유지 RMSE": _fmt(entry.get("persistence_test_rmse_km2"), ".2f", " km²"),
            "개선율": _pct(entry.get("skill_vs_persistence")),
            "같은 분할 수위 모델 개선율": _pct(level.get("skill_vs_persistence")),
        })
    return rows


def bridge_demo_meta(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    bridge = (payload or {}).get("bridge") if isinstance(payload, Mapping) else None
    demo = bridge.get("demonstration") if isinstance(bridge, Mapping) else None
    if not isinstance(demo, Mapping):
        return {}
    summary = demo.get("summary") or {}
    return {
        "sensor": demo.get("sensor"),
        "level_station": demo.get("level_station"),
        "formula_ko": (demo.get("derived_series") or {}).get("formula_ko"),
        "published": bool(demo.get("published")),
        "warnings_ko": [str(w) for w in (summary.get("warnings_ko") or [])],
        "rating_loo_rmse_km2": summary.get("rating_loo_rmse_km2"),
        "end_to_end_rmse_km2": summary.get("end_to_end_rmse_km2") or {},
    }


def end_to_end_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """위성 관측일의 (검량 오차)+(수위 예보 오차) 분해 표."""

    meta = bridge_demo_meta(payload)
    if not meta:
        return []
    bridge = payload["bridge"]  # type: ignore[index]
    rows_in = ((bridge.get("demonstration") or {}).get("end_to_end") or {}).get("rows") or []
    rows = []
    for row in rows_in:
        if not isinstance(row, Mapping) or row.get("total_error_km2") is None:
            continue
        rows.append({
            "날짜": str(row.get("date"))[:10],
            "지평": f"D+{row.get('horizon')}",
            "위성 실측": _fmt(row.get("satellite_area_km2"), ".2f", " km²"),
            "검량 면적": _fmt(row.get("rating_area_km2"), ".2f", " km²"),
            "예보 면적": _fmt(row.get("forecast_area_km2"), ".2f", " km²"),
            "검량 오차": _fmt(row.get("rating_error_km2"), "+.2f", " km²"),
            "수위 예보 오차": _fmt(row.get("level_forecast_error_km2"), "+.2f", " km²"),
            "총오차": _fmt(row.get("total_error_km2"), "+.2f", " km²"),
            "분할": row.get("in_split"),
        })
    return rows


def verdict_ko(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get("verdict_ko")
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if isinstance(value, str) else []


def notes_ko(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    for key in ("notes_ko", "honesty_ko", "caveats_ko"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
    return []


def region_summary_rows(experiments: Mapping[str, Mapping[str, Any] | None], names: Mapping[str, str]) -> list[list[str]]:
    """발표자료용: 지역 × 센서 한 줄 — 관측 수 · 최저 LOO 후보 · 직전값 유지 대비 · 다리 판정."""

    rows: list[list[str]] = []
    for region_id, payload in experiments.items():
        blocks = series_blocks(payload)
        if not blocks:
            continue
        defensible = "통과" if bridge_defensible(payload) else "탈락 (모델 미공개)"
        for sensor, block in sorted(blocks.items()):
            candidates = block.get("candidates") or {}
            best = block.get("best_by_loo")
            best_c = candidates.get(best) or {}
            base = candidates.get("persistence") or {}
            rows.append([
                names.get(region_id, region_id), sensor, str(block.get("n") or "-"),
                _fmt(base.get("rmse_km2"), ".2f"),
                candidate_label(str(best), best_c) if best else "-",
                _fmt(best_c.get("rmse_km2"), ".2f"),
                _pct(best_c.get("skill_vs_persistence")),
                defensible,
            ])
    return rows


# --- Streamlit 화면 ------------------------------------------------------------------------------


_BOX = (
    '<div style="margin:8px 0 12px;padding:12px 16px;border-radius:12px;border:1px solid rgba(56,189,248,.35);'
    'background:rgba(8,47,73,.28);color:#e0f2fe;line-height:1.7;font-size:.95rem">{html}</div>'
)
_WARN = (
    '<div style="margin:8px 0 12px;padding:10px 14px;border-left:3px solid #f59e0b;border-radius:0 6px 6px 0;'
    'background:rgba(120,53,15,.16);color:#fde68a;line-height:1.6;font-size:.9rem">{text}</div>'
)
_CAPTION = '<div style="color:#64748b;font-size:.8rem">{text}</div>'


def _session_cached(key: str, loader):
    import streamlit as st

    cache = st.session_state.setdefault("gauge_cache", {})
    if key not in cache:
        cache[key] = loader()
    return cache[key]


def load_area_models_experiment(client, region_id: str) -> Mapping[str, Any] | None:
    """API 에서 실험 JSON(없으면 None). 서버가 실험 API 를 모르면 안내 문구로 떨어진다."""

    from api_client import APIError

    def _load():
        try:
            payload = client.gauge_experiments(region_id)
        except APIError as exc:
            return {"_error": str(exc)}
        items = payload.get("items") if isinstance(payload, Mapping) else None
        block = (items or {}).get("area_models") if isinstance(items, Mapping) else None
        return block if isinstance(block, Mapping) else None

    return _session_cached(f"experiments:area_models:{region_id}", _load)


def _bold(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))


def render_area_models_experiment(client, region: Mapping[str, Any], key: str) -> None:
    """성능 평가 화면(슬라이드 6장) 공용 펼침 상자 — 면적 계열 모델 비교."""

    import streamlit as st
    from tables import static_table

    with st.expander(EXPANDER_TITLE, expanded=False):
        st.markdown(
            _BOX.format(
                html=(
                    "<b>무엇을 본 실험인가?</b> 위성 수체 <b>면적</b> 계열에 직전값 유지 말고 다른 모델을 써서 "
                    "검증할 수 있는지 직접 해 본 것입니다. 면적은 센서별 <b>공통 격자</b>(같은 상자·같은 화소)에서 다시 쟀습니다 — "
                    "장면마다 촬영 범위가 달라 원래의 장면별 면적은 날짜 간 비교가 되지 않기 때문입니다."
                    "<br>• 검증 방식은 <b>LOO</b>(관측 하나를 빼고 나머지로 맞히기)이고, 시험점 수 = 관측 수입니다. "
                    "<br>• 관측이 센서당 4~10개뿐이라 <b>모델 우열을 가릴 수 없습니다</b> — n=4 에서는 부호검정으로 얻을 수 있는 "
                    "가장 작은 p 가 0.125 라 애초에 유의할 수 없습니다. 아래 순위는 <b>참고용</b>입니다. "
                    "<br>• 공식 성적표(위 지점별 성적표·모델별 성능지표)는 이 실험을 <b>포함하지 않습니다</b>."
                )
            ),
            unsafe_allow_html=True,
        )
        payload = load_area_models_experiment(client, region["id"])
        if isinstance(payload, Mapping) and payload.get("_error"):
            st.markdown(_CAPTION.format(text=f"실험 결과를 불러오지 못했습니다: {payload['_error']}"), unsafe_allow_html=True)
            return
        sensors = sensors_available(payload)
        if not sensors:
            st.markdown(
                _CAPTION.format(text="이 지역에는 아직 면적 모델 실험 결과가 없습니다 — 위성 공통 격자 마스크가 있는 지역에만 만들어집니다 (scripts/eval_area_models_experiment.py 를 실행하면 regions/&lt;id&gt;/models/gauge/experiments/area_models.json 이 생깁니다)."),
                unsafe_allow_html=True,
            )
            return

        summary = sensor_rows(payload)
        st.markdown("**센서별 요약** (면적은 공통 격자 기준)")
        static_table(summary)
        st.markdown(_CAPTION.format(text="'관측 잡음 하한' = 이틀 이내 재방문한 두 장면의 면적 차이. 이 값이 계열 변동폭에 육박하면, 날짜별 면적 차이는 수위 변화가 아니라 탐지 흔들림으로 봐야 합니다."), unsafe_allow_html=True)

        sensor = sensors[0] if len(sensors) == 1 else st.selectbox("센서", sensors, key=f"{key}_sensor")
        block = series_blocks(payload).get(sensor) or {}
        tiles = st.columns(4)
        candidates = block.get("candidates") or {}
        best = block.get("best_by_loo")
        best_c = candidates.get(best) or {}
        tiles[0].metric("관측 수 (LOO 시험점)", str(block.get("n") or "-"), help="관측 하나를 빼고 나머지로 맞히기를 관측 수만큼 반복했습니다")
        tiles[1].metric("직전값 유지 LOO RMSE", _fmt((candidates.get("persistence") or {}).get("rmse_km2"), ".3f", " km²"))
        tiles[2].metric("최저 LOO RMSE", _fmt(best_c.get("rmse_km2"), ".3f", " km²"), help=candidate_label(str(best), best_c) if best else None)
        tiles[3].metric(
            "유의한 후보", "0 개" if not block.get("any_candidate_significant_vs_persistence") else "있음",
            help="직전값 유지 대비 부호검정 p < 0.05 인 후보 수",
        )

        st.markdown(f"**{sensor} 관측** (공통 격자 면적 vs 장면 자기 촬영 범위 면적)")
        static_table(points_table_rows(payload, sensor))
        st.markdown(f"**{sensor} 후보별 LOO 성적** (RMSE 오름차순 · 개선율 = 직전값 유지 대비 MSE 감소율)")
        static_table(candidate_table_rows(payload, sensor))
        consistency = block.get("mask_consistency") or {}
        if consistency:
            st.markdown(_CAPTION.format(text=(
                f"마스크 일관성: 상호 IoU 평균 {consistency.get('mean_pairwise_iou')} · "
                f"항상 물 {consistency.get('always_water_km2')} km² / 한 번이라도 물 {consistency.get('ever_water_km2')} km² "
                f"(흔들림 {consistency.get('churn_km2')} km²)."
            )), unsafe_allow_html=True)
        for line in block.get("verdict_ko") or []:
            st.markdown(_WARN.format(text=_bold(line)), unsafe_allow_html=True)

        bridge_rows = bridge_table_rows(payload)
        if bridge_rows:
            st.markdown("**수위 → 면적 검량곡선 판정** (이게 서야 10개년 일별 수위 예측을 면적으로 옮길 수 있습니다)")
            static_table(bridge_rows)
            if not bridge_defensible(payload):
                st.markdown(_WARN.format(text=_bold(
                    "어느 조합도 문턱을 넘지 못했습니다 → 파생 면적 예측 모델을 **공개하지 않습니다**. "
                    "아래 시연이 있다면 파이프라인이 도는지 보이는 용도이며 성과 수치로 인용하면 안 됩니다."
                )), unsafe_allow_html=True)
        demo_rows = bridge_demo_rows(payload)
        if demo_rows:
            meta = bridge_demo_meta(payload)
            st.markdown(f"**시연(공개 아님): {meta.get('sensor')} × {meta.get('level_station')}** — `{meta.get('formula_ko')}`")
            static_table(demo_rows)
            rows = end_to_end_rows(payload)
            if rows:
                st.markdown("**끝단 대조** — 위성 관측일의 예보 파생 면적 vs 실측 공통 격자 면적 (총오차 = 검량 오차 + 수위 예보 오차)")
                static_table(rows)
            for warning in meta.get("warnings_ko") or []:
                st.markdown(_WARN.format(text=_bold(warning)), unsafe_allow_html=True)
        for line in verdict_ko(payload):
            st.markdown(_WARN.format(text=_bold(line)), unsafe_allow_html=True)
        for line in notes_ko(payload):
            st.markdown(_CAPTION.format(text=_bold(line)), unsafe_allow_html=True)
        st.markdown(_CAPTION.format(text=f"산출: {payload.get('schema', 'area_models.json')} · {str(payload.get('generated_at', ''))[:19]} · scripts/eval_area_models_experiment.py"), unsafe_allow_html=True)
