"""미래 강수(예보) 상한 실험 — 공유 화면 요소와 순수 표 도우미.

`scripts/eval_future_rain_experiment.py` 가 지역마다 남기는 ``regions/<id>/models/gauge/experiments/future_rain.json``
을 ``GET /api/v1/regions/{id}/gauge/experiments`` 로 받아, 지점별 지평 표(직전값 유지 / 과거 강수 AR-X /
+3일 예보 / +전 구간 예보)와 급상승일 오차 표를 그린다. **완전 예보 가정**(perfect prognosis): 실측 미래
강수를 예보로 간주한 상한이며, 공식 성적표(evaluation.json)에는 넣지 않는다.

순수 도우미(`experiment_rows`·`station_table_rows`·`event_rows`·`summary_by_horizon`)는 Streamlit 없이도 쓰이고
(발표자료 스크립트가 JSON 을 직접 읽어 같은 함수를 쓴다), 산출 JSON 의 세부 키 이름이 조금 달라도
(중첩 `stations/horizons` 또는 평면 `rows`, 변형 이름 `ridge_arx_future_3`·`future_all` 등) 같은 표를 만든다.
숫자는 만들지 않는다 — 전부 실험 JSON 에서 온다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

# 변형(variant) 표준 id → 화면 이름. 순서 = 표 열 순서.
VARIANTS: tuple[tuple[str, str], ...] = (
    ("persistence", "직전값 유지 (기준선)"),
    ("ridge_arx", "과거 강수 AR-X"),
    ("future_3", "+3일 예보 (완전 예보)"),
    ("future_all", "+전 구간 예보 (완전 예보)"),
)
VARIANT_KO = dict(VARIANTS)
REPORT_HORIZONS = (1, 3, 5, 7, 10, 14, 30)
EVENT_HORIZONS = (1, 3)
BEATS_THRESHOLD = 0.005  # 0.5 %p — 평가표의 '기상이 뚜렷이 도운 셀' 과 같은 문턱
EXPANDER_TITLE = "미래 강수(예보)를 넣으면 얼마나 좋아지나 — 완전 예보 상한 실험"

_METRIC_KEYS = {
    "rmse": ("rmse", "test_rmse", "rmse_m"),
    "mse": ("mse", "test_mse", "mse_m2"),
    "mae": ("mae", "test_mae", "mae_m"),
    "skill": ("skill", "skill_vs_persistence", "test_skill"),
    "n": ("n", "test_n", "n_test"),
}
_METRIC_LEAVES = frozenset(k for keys in _METRIC_KEYS.values() for k in keys)


# --- 순수 도우미 ---------------------------------------------------------------------------------


def classify_variant(key: str) -> str | None:
    """키 이름(경로)을 표준 변형 id 로. 모르면 None."""

    k = str(key).lower()
    if "persist" in k:
        return "persistence"
    if "future" in k or "미래" in k or "forecast_rain" in k:
        if re.search(r"all|full|전", k):
            return "future_all"
        if re.search(r"(?<![0-9])3(?![0-9])", k):
            return "future_3"
        return "future_all"  # 변형 표기 없는 'future' 는 전 구간(이벤트 블록의 future(all))
    if "arx" in k:
        return "ridge_arx"
    return None


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _metric(mapping: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    nested = mapping.get("test") if isinstance(mapping.get("test"), Mapping) else None
    for name, aliases in _METRIC_KEYS.items():
        for alias in aliases:
            value = mapping.get(alias)
            if value is None and nested is not None:
                value = nested.get(alias)
            if value is not None:
                out[name] = _num(value) if name != "n" else (int(value) if isinstance(value, (int, float)) else None)
                break
    for extra in ("lam", "future_days", "params", "hparams", "note_ko", "delta_rmse_cm", "skill_gain_pp"):
        if extra in mapping:
            out[extra] = mapping[extra]
    return out


def _looks_like_metric(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        any(k in value for k in _METRIC_LEAVES) or (isinstance(value.get("test"), Mapping) and any(k in value["test"] for k in _METRIC_LEAVES))
    )


def _collect_variants(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    """행 안의 중첩 사전에서 변형별 지표를 찾는다 (`ridge_arx_future: {3: {...}, all: {...}}` 도 처리)."""

    found: dict[str, dict[str, Any]] = {}
    flat: dict[str, dict[str, Any]] = {}
    for key, value in mapping.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if _looks_like_metric(value):
            variant = classify_variant(path)
            if variant and variant not in found:
                found[variant] = _metric(value)
        elif isinstance(value, Mapping):
            for variant, metric in _collect_variants(value, path).items():
                found.setdefault(variant, metric)
        else:  # 평면 키: persistence_rmse, future_all_skill ...
            m = re.fullmatch(r"(.+?)_(rmse|mse|mae|skill|n)", str(key))
            if m and _num(value) is not None:
                variant = classify_variant(m.group(1))
                if variant:
                    flat.setdefault(variant, {})[m.group(2)] = _num(value)
    for variant, metric in flat.items():
        found.setdefault(variant, metric)
    return found


def _horizon_of(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    m = re.fullmatch(r"(?:d\+?|h|horizon[_-]?|day[_-]?)?(\d+)d?", text)
    return int(m.group(1)) if m else None


def _canonical_row(station: Mapping[str, Any], horizon: int, cell: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics = _collect_variants(cell)
    if not metrics:
        return None
    meta = station.get("station") if isinstance(station.get("station"), Mapping) else {}  # {"station": {id, name, terrain}} 형태
    sid = station.get("station_id") or meta.get("id") or station.get("id") or cell.get("station_id")
    name = station.get("station_name") or meta.get("name") or station.get("name") or cell.get("station_name") or sid
    n = cell.get("test_n") or cell.get("n_test") or cell.get("n") or next((m.get("n") for m in metrics.values() if m.get("n")), None)
    delta_block = cell.get("delta_rmse_cm") if isinstance(cell.get("delta_rmse_cm"), Mapping) else {}
    delta = {classify_variant(k) or k: _num(v) for k, v in delta_block.items()}
    for vid, m in metrics.items():  # 변형 안에 delta_rmse_cm 이 있는 형태
        if vid.startswith("future") and _num(m.get("delta_rmse_cm")) is not None and delta.get(vid) is None:
            delta[vid] = _num(m["delta_rmse_cm"])
    return {
        "station_id": str(sid) if sid is not None else str(name), "station_name": str(name),
        "terrain": station.get("terrain") or meta.get("terrain") or cell.get("terrain"),
        "horizon": int(horizon), "n": n, "metrics": metrics, "delta_rmse_cm": delta,
    }


def _station_items(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """`stations` 가 목록이든 지점 id 사전이든 지점 블록 목록으로 (사전 키는 station_id 로 붙인다)."""

    stations = payload.get("stations") or payload.get("by_station") or payload.get("per_station")
    if isinstance(stations, list):
        return [s for s in stations if isinstance(s, Mapping)]
    if isinstance(stations, Mapping):
        items = []
        for key, block in stations.items():
            if not isinstance(block, Mapping):
                continue
            meta = block.get("station") if isinstance(block.get("station"), Mapping) else {}
            items.append(dict(block, station_id=block.get("station_id") or meta.get("id") or key))
        return items
    return []


def experiment_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """실험 JSON → 지점×지평 표준 행 목록 (`metrics[variant] = {rmse, mse, mae, skill}`)."""

    if not isinstance(payload, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    flat = payload.get("rows") or payload.get("cells")
    if isinstance(flat, list):
        for cell in flat:
            if isinstance(cell, Mapping) and _horizon_of(cell.get("horizon")) is not None:
                row = _canonical_row(cell, _horizon_of(cell.get("horizon")), cell)
                if row:
                    rows.append(row)
    for station in _station_items(payload):
        hs = station.get("horizons") or station.get("by_horizon") or station.get("rows") or station.get("cells")
        if isinstance(hs, Mapping):
            pairs = [(_horizon_of(k), v) for k, v in hs.items()]
        elif isinstance(hs, list):
            pairs = [(_horizon_of((v or {}).get("horizon")), v) for v in hs]
        else:
            continue
        for horizon, cell in pairs:
            if horizon is None or not isinstance(cell, Mapping):
                continue
            row = _canonical_row(station, horizon, cell)
            if row:
                rows.append(row)
    seen: set[tuple[str, int]] = set()
    unique = []
    for row in rows:
        key = (row["station_id"], row["horizon"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    unique.sort(key=lambda r: (r["horizon"], r["station_name"]))
    return unique


def horizons_available(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({int(r["horizon"]) for r in rows})


def _fmt_m_cm(value: Any) -> str:
    v = _num(value)
    return f"{v:.3f} m ({v * 100:.1f} cm)" if v is not None else "-"


def skill_text(skill: Any) -> str:
    v = _num(skill)
    return f"{v * 100:+.1f} %" if v is not None else "-"


def station_table_rows(rows: Sequence[Mapping[str, Any]], horizon: int) -> list[dict[str, Any]]:
    """지점별 표 (한 지평): 변형마다 'RMSE m (cm) · 개선율', 마지막에 전 구간 예보 이득(cm)."""

    out = []
    for r in rows:
        if int(r["horizon"]) != int(horizon):
            continue
        metrics = r["metrics"]
        item: dict[str, Any] = {"지점": r["station_name"]}
        for vid, label in VARIANTS:
            m = metrics.get(vid) or {}
            if vid == "persistence":
                item[f"{label} RMSE"] = _fmt_m_cm(m.get("rmse"))
            else:
                item[label] = f"{_fmt_m_cm(m.get('rmse'))} · {skill_text(m.get('skill'))}" if m else "-"
        arx = _num((metrics.get("ridge_arx") or {}).get("rmse"))
        fut = _num((metrics.get("future_all") or {}).get("rmse"))
        delta = (r.get("delta_rmse_cm") or {}).get("future_all")
        if delta is None and arx is not None and fut is not None:
            delta = (arx - fut) * 100
        item["전 구간 예보 이득"] = f"{delta:+.1f} cm" if delta is not None else "-"
        out.append(item)
    out.sort(key=lambda it: it["지점"])
    return out


def summary_by_horizon(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """지평별 변형 평균 개선율(지점 평균)과 '전 구간/3일 예보가 AR-X 를 0.5%p 이상 앞선 셀' 수."""

    out: dict[int, dict[str, Any]] = {}
    for h in horizons_available(rows):
        subset = [r for r in rows if int(r["horizon"]) == h]
        entry: dict[str, Any] = {"n_stations": len(subset), "mean_skill": {}, "mean_rmse": {}, "beats_arx": {}}
        for vid, _label in VARIANTS:
            skills = [_num(r["metrics"].get(vid, {}).get("skill")) for r in subset]
            skills = [s for s in skills if s is not None]
            rmses = [_num(r["metrics"].get(vid, {}).get("rmse")) for r in subset]
            rmses = [v for v in rmses if v is not None]
            entry["mean_skill"][vid] = sum(skills) / len(skills) if skills else None
            entry["mean_rmse"][vid] = sum(rmses) / len(rmses) if rmses else None
        for vid in ("future_3", "future_all"):
            beats = 0
            for r in subset:
                a = _num(r["metrics"].get("ridge_arx", {}).get("skill"))
                f = _num(r["metrics"].get(vid, {}).get("skill"))
                if a is not None and f is not None and f - a > BEATS_THRESHOLD:
                    beats += 1
            entry["beats_arx"][vid] = beats
        out[h] = entry
    return out


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk(v, path + (str(i),))
    else:
        yield path, value


def _event_horizon(path: Sequence[str]) -> int | None:
    horizon = None
    for seg in path:
        s = seg.lower()
        if _horizon_of(s) is not None and re.fullmatch(r"(?:d\+?|h|horizon[_-]?|day[_-]?)?\d+d?", s):
            horizon = _horizon_of(s)
            continue
        m = re.search(r"(?:^|_)(?:d\+?|h)(\d+)(?:_|$)", s)
        if m:
            horizon = int(m.group(1))
    return horizon


def event_rows(payload: Mapping[str, Any] | None, station_id: str | None = None) -> list[dict[str, Any]]:
    """급상승일 표: 지점·날짜·1일 상승(m)·D+1/D+3 |오차|(cm) — 직전값 유지 / 과거 강수 AR-X / +전 구간 예보."""

    if not isinstance(payload, Mapping):
        return []
    events = payload.get("events")
    if isinstance(events, Mapping) and isinstance(events.get("items"), list):
        events = events["items"]
    elif isinstance(events, Mapping) and isinstance(events.get("by_station"), (list, Mapping)):
        events = events["by_station"]
    names = {r["station_id"]: r["station_name"] for r in experiment_rows(payload)}
    items: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(events, list):
        for ev in events:
            if isinstance(ev, Mapping):
                nested = ev.get("events") or ev.get("items") or ev.get("top_rises")
                if isinstance(nested, list):
                    sid = str(ev.get("station_id") or ev.get("id") or ev.get("station_name") or "")
                    items += [(sid, e) for e in nested if isinstance(e, Mapping)]
                else:
                    items.append((str(ev.get("station_id") or ev.get("station") or ev.get("station_name") or ""), ev))
    elif isinstance(events, Mapping):
        for sid, lst in events.items():
            if isinstance(lst, list):
                items += [(str(sid), e) for e in lst if isinstance(e, Mapping)]
            elif isinstance(lst, Mapping) and isinstance(lst.get("events") or lst.get("items"), list):
                items += [(str(sid), e) for e in (lst.get("events") or lst.get("items")) if isinstance(e, Mapping)]
    if not items:  # 지점 블록 안에 events 가 있는 형태
        for station in _station_items(payload):
            lst = station.get("events")
            if isinstance(lst, list):
                sid = str(station.get("station_id") or "")
                items += [(sid, e) for e in lst if isinstance(e, Mapping)]
    out = []
    for sid, ev in items:
        if station_id is not None and sid != str(station_id) and str(ev.get("station_id") or "") != str(station_id):
            continue
        date = ev.get("date") or ev.get("target_date") or ev.get("day") or ev.get("origin")
        rise = next((_num(ev.get(k)) for k in ("rise_m", "rise", "delta_m", "rise_1d_m", "jump_m", "rise_m_1d") if _num(ev.get(k)) is not None), None)
        errors: dict[int, dict[str, float]] = {}
        rain: dict[int, float] = {}
        for path, value in _walk({k: v for k, v in ev.items() if k not in ("date", "target_date", "day", "origin", "rise_m", "rise", "delta_m", "station_id", "station_name", "level_m", "level_before_m", "level_after_m")}):
            v = _num(value)
            if v is None:
                continue
            horizon = _event_horizon(path)
            if re.search(r"rain|precip|_mm$", path[-1].lower()):  # 원점~목표일 실측 강수(mm) — 오차가 아니다
                if horizon:
                    rain.setdefault(horizon, v)
                continue
            variant = classify_variant("/".join(path))
            if variant and horizon:
                errors.setdefault(horizon, {}).setdefault(variant, abs(v))
        name = ev.get("station_name") or names.get(sid) or sid
        row: dict[str, Any] = {"지점": name, "날짜": str(date)[:10] if date else "-", "1일 상승": f"{rise:+.2f} m" if rise is not None else "-"}
        for h in EVENT_HORIZONS:
            for vid, short in (("persistence", "직전값"), ("ridge_arx", "AR-X"), ("future_all", "+전 구간 예보")):
                e = errors.get(h, {}).get(vid)
                row[f"D+{h} |오차| {short}"] = f"{e * 100:.1f} cm" if e is not None else "-"
            if rain:
                row[f"~D+{h} 강수"] = f"{rain[h]:.1f} mm" if h in rain else "-"
        row["_rise"] = rise if rise is not None else -1e9
        out.append(row)
    out.sort(key=lambda r: (r["지점"], -r["_rise"]))
    for r in out:
        r.pop("_rise", None)
    return out


def notes_ko(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    for key in ("notes_ko", "honesty_ko", "notes", "caveats_ko"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
    return []


def region_summary_rows(experiments: Mapping[str, Mapping[str, Any] | None], names: Mapping[str, str], horizons: Sequence[int] = (1, 3, 7, 14, 30)) -> list[list[str]]:
    """발표자료용: 지역별 지점 평균 개선율 'AR-X / +3일 / +전 구간' (마지막 행 = 지점 가중 전체). 실험이 없는 지역은 뺀다."""

    rows: list[list[str]] = []
    pooled: dict[int, dict[str, list[float]]] = {h: {"ridge_arx": [], "future_3": [], "future_all": []} for h in horizons}
    for rid, payload in experiments.items():
        exp_rows = experiment_rows(payload)
        if not exp_rows:
            continue
        summary = summary_by_horizon(exp_rows)
        n_st = len({r["station_id"] for r in exp_rows})
        cells = []
        for h in horizons:
            entry = summary.get(h)
            if not entry:
                cells.append("-")
                continue
            parts = []
            for vid in ("ridge_arx", "future_3", "future_all"):
                s = entry["mean_skill"].get(vid)
                parts.append(f"{s * 100:+.0f}" if s is not None else "-")
                pooled[h][vid] += [_num(r["metrics"].get(vid, {}).get("skill")) for r in exp_rows if int(r["horizon"]) == h and _num(r["metrics"].get(vid, {}).get("skill")) is not None]
            cells.append(" / ".join(parts) + " %")
        rows.append([names.get(rid, rid), str(n_st)] + cells)
    if rows:
        total_cells = []
        for h in horizons:
            parts = [f"{sum(v) / len(v) * 100:+.0f}" if v else "-" for v in (pooled[h][vid] for vid in ("ridge_arx", "future_3", "future_all"))]
            total_cells.append(" / ".join(parts) + " %")
        rows.append(["전체 (지점 가중)", str(sum(int(r[1]) for r in rows))] + total_cells)
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


def load_future_rain_experiment(client, region_id: str) -> Mapping[str, Any] | None:
    """API 에서 실험 JSON(없으면 None). 서버가 실험 API 를 모르면(404 등) None 을 돌려주고 화면이 안내한다."""

    from api_client import APIError

    def _load():
        try:
            payload = client.gauge_experiments(region_id)
        except APIError as exc:
            return {"_error": str(exc)}
        items = payload.get("items") if isinstance(payload, Mapping) else None
        fr = (items or {}).get("future_rain") if isinstance(items, Mapping) else None
        return fr if isinstance(fr, Mapping) else None

    return _session_cached(f"experiments:future_rain:{region_id}", _load)


def render_future_rain_experiment(client, region: Mapping[str, Any], horizon: int, key: str) -> None:
    """성능 평가 화면(슬라이드 6장·수위 예측 모델 평가 탭) 공용 펼침 상자."""

    import streamlit as st
    from tables import static_table

    with st.expander(EXPANDER_TITLE, expanded=False):
        st.markdown(
            _BOX.format(
                html=(
                    "<b>완전 예보 가정(perfect prognosis)이란?</b> 원점 <b>이후</b>에 실제로 내린 비(실측)를 '완벽한 강수 예보' 로 간주해 "
                    "릿지 AR-X 에 추가 입력한 실험입니다. <b>+3일 예보</b>는 원점 다음 3일치만, <b>+전 구간 예보</b>는 목표일까지 전부 안다고 가정합니다. "
                    "<br>• 이 값은 예보 입력이 줄 수 있는 <b>이득의 상한</b>입니다. 실제 기상청 예보에는 오차가 있어 실시간 이득은 이보다 <b>작습니다</b>. "
                    "<br>• 공식 성적표(위 지점별 성적표·모델별 성능지표)는 이 실험을 <b>포함하지 않습니다</b> — 원점 이전 자료만 쓴 모델의 성적입니다. "
                    "<br>• 실시간 운영에는 기상청 단기 예보 연결이 필요하며, 아직 연결돼 있지 않습니다."
                )
            ),
            unsafe_allow_html=True,
        )
        payload = load_future_rain_experiment(client, region["id"])
        if isinstance(payload, Mapping) and payload.get("_error"):
            st.markdown(_CAPTION.format(text=f"실험 결과를 불러오지 못했습니다: {payload['_error']}"), unsafe_allow_html=True)
            return
        rows = experiment_rows(payload)
        if not rows:
            st.markdown(
                _CAPTION.format(text="이 지역에는 아직 미래 강수 실험 결과가 없습니다 (scripts/eval_future_rain_experiment.py 를 실행하면 regions/&lt;id&gt;/models/gauge/experiments/future_rain.json 이 만들어집니다)."),
                unsafe_allow_html=True,
            )
            return
        available = horizons_available(rows)
        h = int(horizon) if int(horizon) in available else (min(available, key=lambda a: abs(a - int(horizon))) if available else int(horizon))
        summary = summary_by_horizon(rows).get(h) or {}
        means = summary.get("mean_skill") or {}
        beats = summary.get("beats_arx") or {}
        n_st = summary.get("n_stations") or 0
        tiles = st.columns(4)
        tiles[0].metric(f"D+{h} 과거 강수 AR-X", skill_text(means.get("ridge_arx")), help="지점 평균 개선율 (직전값 유지 대비 MSE 감소율)")
        tiles[1].metric(f"D+{h} +3일 예보", skill_text(means.get("future_3")), help="원점 다음 3일 강수를 안다고 가정")
        tiles[2].metric(f"D+{h} +전 구간 예보", skill_text(means.get("future_all")), help="목표일까지 강수를 모두 안다고 가정")
        tiles[3].metric("예보가 AR-X 를 앞선 지점", f"{beats.get('future_all', 0)} / {n_st}", help="전 구간 예보 개선율이 과거 강수 AR-X 보다 0.5 %p 이상 높은 지점 수")
        split = payload.get("split") if isinstance(payload, Mapping) else None
        test_year = str((split or {}).get("test_start", ""))[:4]
        st.markdown(f"**D+{h} 지점별 비교** (시험 {test_year}년 · RMSE m(cm) · 개선율 = 직전값 유지 대비)" if test_year else f"**D+{h} 지점별 비교** (RMSE m(cm) · 개선율 = 직전값 유지 대비)")
        static_table(station_table_rows(rows, h))
        st.markdown(_CAPTION.format(text="'전 구간 예보 이득' = RMSE(과거 강수 AR-X) − RMSE(+전 구간 예보), cm. 양수면 미래 강수를 알 때 오차가 그만큼 줄어든다는 뜻입니다. 두 변형 모두 검증 폴드에서 정칙화 강도(λ)를 골랐습니다."), unsafe_allow_html=True)
        if h != int(horizon):
            st.markdown(_CAPTION.format(text=f"실험은 D+{h} 까지만 보고합니다 (선택한 D+{int(horizon)} 대신 가장 가까운 지평)."), unsafe_allow_html=True)
        events = event_rows(payload)
        if events:
            per_station = max(1, len(events) // max(1, n_st))
            st.markdown(f"**급상승일 오차 비교** (시험 {test_year}년 지점별 1일 상승 상위 {per_station}일 · |오차| cm)" if test_year else "**급상승일 오차 비교** (|오차| cm)")
            station_names = sorted({e["지점"] for e in events})
            if len(station_names) > 1:
                pick = st.selectbox("급상승일 표 지점", ["전체 지점", *station_names], key=f"{key}_ev_station")
                if pick != "전체 지점":
                    events = [e for e in events if e["지점"] == pick]
            static_table(events)
            st.markdown(_CAPTION.format(text="홍수기 급상승은 원점 이후 비 때문에 생기므로 과거 강수만 쓰는 모델은 이 날들을 놓칩니다. 미래 강수를 알면 D+1·D+3 오차가 얼마나 줄어드는지 보는 표입니다."), unsafe_allow_html=True)
        for line in notes_ko(payload):
            st.markdown(_WARN.format(text=re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)), unsafe_allow_html=True)  # 메모의 **굵게** 를 HTML 로
        st.markdown(_CAPTION.format(text=f"산출: {payload.get('schema', 'future_rain.json')} · {str(payload.get('generated_at', ''))[:19]} · scripts/eval_future_rain_experiment.py"), unsafe_allow_html=True)
