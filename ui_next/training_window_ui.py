"""학습·검증·시험 구간을 화면에서 직접 정해 다시 학습하는 구성 요소.

수위 예측 모델은 수위계 기록을 시간 순서로 셋으로 나눠 앞부분으로 배우고(학습), 가운데 몇 해로
모델을 고르고(검증), 마지막 해로 채점한다(시험). 기본 구간은 지역 정의(`region.json`
forecast.split)에 있고, 여기서는 연도 단위로 다른 구간을 골라 서버에 비동기 학습을 걸고
진행을 지켜본다. 순수 도우미(연도 → 날짜, 검증, 설명문)는 Streamlit 없이 쓸 수 있다.

서버 계약: POST .../gauge/train {split, async_run:true} → 202 {job_id}; GET .../gauge/train/jobs/{job_id};
DELETE .../gauge/models 로 사용자 산출물을 지워 내장 산출물(기본 구간)로 되돌린다.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date as Date
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st
from api_client import APIClient, APIError

SPLIT_KEYS: tuple[str, ...] = ("train_end", "val_start", "val_end", "test_start", "test_end")
MIN_TRAIN_DAYS = 730
MIN_VAL_DAYS = 90
MIN_TEST_DAYS = 30
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 20 * 60
EXPANDER_TITLE = "학습·검증·시험 구간 직접 정하기 (재학습)"
TRAIN_BUTTON = "이 구간으로 다시 학습 ▶"
RESET_BUTTON = "기본 구간으로 되돌리기 ↺"
# 학습이 끝나면 비워야 하는 세션 캐시(gauge_cache) 접두사와 화면 상태 키
CACHE_PREFIXES: tuple[str, ...] = ("region_detail:", "evaluation:", "model:", "forecast:", "backtest:", "compare:", "satlevels:", "regions")
SESSION_KEYS_TO_CLEAR: tuple[str, ...] = ("sl_level_forecast", "sl_lv_compare_payload")
FINAL_STATUSES = frozenset({"done", "failed"})


# --- 순수 도우미 -------------------------------------------------------------------------------------


def _date(value: Any) -> Date:
    return Date.fromisoformat(str(value)[:10])


def years_to_split(train_end_year: int, val_years: tuple[int, int], test_years: tuple[int, int]) -> dict[str, str]:
    """연도 세 묶음 → 5개 ISO 날짜. 학습 종료 = 12-31, 검증·시험은 1-1 ∼ 12-31."""

    val_start, val_end = int(val_years[0]), int(val_years[1])
    test_start, test_end = int(test_years[0]), int(test_years[1])
    return {
        "train_end": f"{int(train_end_year):04d}-12-31",
        "val_start": f"{val_start:04d}-01-01",
        "val_end": f"{val_end:04d}-12-31",
        "test_start": f"{test_start:04d}-01-01",
        "test_end": f"{test_end:04d}-12-31",
    }


def validate_split(split: Mapping[str, Any], series_range: tuple[str, str] | None) -> list[str]:
    """서버와 같은 규칙으로 미리 검사해 한글 문제 목록을 돌려준다(비어 있으면 통과)."""

    problems: list[str] = []
    try:
        dates = {key: _date(split[key]) for key in SPLIT_KEYS}
    except (KeyError, TypeError, ValueError):
        return ["구간 날짜 다섯 개(train_end, val_start, val_end, test_start, test_end)가 모두 있어야 합니다."]
    if not (dates["train_end"] < dates["val_start"] <= dates["val_end"] < dates["test_start"] <= dates["test_end"]):
        problems.append("순서가 맞지 않습니다 — 학습 종료 < 검증 시작 ≤ 검증 끝 < 시험 시작 ≤ 시험 끝 이어야 합니다.")
    start = end = None
    if series_range and all(series_range):
        start, end = _date(series_range[0]), _date(series_range[1])
        outside = [key for key in SPLIT_KEYS if not (start <= dates[key] <= end)]
        if outside:
            problems.append(f"수위계 기록 범위({start.isoformat()} ∼ {end.isoformat()}) 밖의 날짜가 있습니다: {', '.join(outside)}.")
        train_days = (dates["train_end"] - start).days + 1
        if train_days < MIN_TRAIN_DAYS:
            problems.append(f"학습 구간이 {train_days}일입니다 — 기록 시작({start.isoformat()})부터 최소 {MIN_TRAIN_DAYS}일(2년)이 필요합니다.")
    val_days = (dates["val_end"] - dates["val_start"]).days + 1
    if val_days < MIN_VAL_DAYS:
        problems.append(f"검증 구간이 {val_days}일입니다 — 최소 {MIN_VAL_DAYS}일이 필요합니다.")
    test_days = (dates["test_end"] - dates["test_start"]).days + 1
    if test_days < MIN_TEST_DAYS:
        problems.append(f"시험 구간이 {test_days}일입니다 — 최소 {MIN_TEST_DAYS}일이 필요합니다.")
    return problems


def split_days(split: Mapping[str, Any], series_range: tuple[str, str] | None) -> dict[str, int | None]:
    """구간별 일수(학습은 기록 시작을 알아야 계산된다)."""

    try:
        dates = {key: _date(split[key]) for key in SPLIT_KEYS}
    except (KeyError, TypeError, ValueError):
        return {"train": None, "val": None, "test": None}
    train = (dates["train_end"] - _date(series_range[0])).days + 1 if series_range and series_range[0] else None
    return {
        "train": train,
        "val": (dates["val_end"] - dates["val_start"]).days + 1,
        "test": (dates["test_end"] - dates["test_start"]).days + 1,
    }


def _is_year_aligned(split: Mapping[str, Any]) -> bool:
    text = {key: str(split.get(key) or "")[:10] for key in SPLIT_KEYS}
    return (
        text["train_end"].endswith("-12-31") and text["val_start"].endswith("-01-01") and text["val_end"].endswith("-12-31")
        and text["test_start"].endswith("-01-01") and text["test_end"].endswith("-12-31")
    )


def _year_span(start: str, end: str) -> str:
    a, b = str(start)[:4], str(end)[:4]
    return a if a == b else f"{a}∼{b}"  # '~' 는 마크다운 취소선이 되므로 '∼'(U+223C)를 쓴다


def describe_split(split: Mapping[str, Any] | None) -> str:
    """'학습 ≤2018 · 검증 2019∼2023 · 시험 2024'. 연도 경계가 아니면 날짜 그대로."""

    if not split or any(not split.get(key) for key in SPLIT_KEYS):
        return "구간 정보 없음"
    if _is_year_aligned(split):
        return (
            f"학습 ≤{str(split['train_end'])[:4]} · 검증 {_year_span(split['val_start'], split['val_end'])} · "
            f"시험 {_year_span(split['test_start'], split['test_end'])}"
        )
    d = {key: str(split[key])[:10] for key in SPLIT_KEYS}
    return f"학습 ≤{d['train_end']} · 검증 {d['val_start']}∼{d['val_end']} · 시험 {d['test_start']}∼{d['test_end']}"


def split_from_region(region_detail: Mapping[str, Any] | None) -> dict[str, Any]:
    """지역 응답의 forecast 블록에서 활성 구간·기본 구간·사용자 지정 여부·학습 시각을 뽑는다.

    예전 서버(split 만 주는)에서도 동작하도록 split_active → split → split_default 순으로 찾는다.
    """

    forecast = dict((region_detail or {}).get("forecast") or {})
    default = dict(forecast.get("split_default") or forecast.get("split") or {})
    active = dict(forecast.get("split_active") or forecast.get("split") or default)
    return {
        "active": active,
        "default": default,
        "custom": bool(forecast.get("custom_split")),
        "trained_at": forecast.get("trained_at"),
    }


def active_split(region: Mapping[str, Any] | None) -> dict[str, Any]:
    """지역 응답에서 지금 활성 산출물의 구간(사용자 지정이면 그 구간). 그래프 음영·전망 원점 기본값에 쓴다."""

    return split_from_region(region)["active"]


def trained_at_text(value: Any) -> str:
    """서버의 generated_at(UTC ISO) → 한국 시각 'YYYY-MM-DD HH:MM'. 시간대가 없으면 그대로."""

    if not value:
        return ""
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)[:16].replace("T", " ")
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(ZoneInfo("Asia/Seoul"))
    return stamp.strftime("%Y-%m-%d %H:%M")


def split_badge_text(region_detail: Mapping[str, Any] | None) -> str:
    """한 줄 배지: '학습 ≤2018 · 검증 2019∼2023 · 시험 2024 · 기본' (+ 학습 시각)."""

    info = split_from_region(region_detail)
    text = f"{describe_split(info['active'])} · {'사용자 지정' if info['custom'] else '기본'}"
    if info.get("trained_at"):
        text += f" · 학습 {trained_at_text(info['trained_at'])}"
    return text


def series_range_from_region(region: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """지역의 기록 범위(가장 이른 시작 ∼ 가장 늦은 끝). 서버가 forecast.series_range 를 주면 그것, 아니면 지점 period 로."""

    declared = ((region or {}).get("forecast") or {}).get("series_range")
    if isinstance(declared, (list, tuple)) and len(declared) == 2 and declared[0] and declared[1]:
        return str(declared[0])[:10], str(declared[1])[:10]
    starts, ends = [], []
    for station in (region or {}).get("stations") or []:
        period = station.get("period") or []
        if len(period) == 2 and period[0] and period[1]:
            starts.append(str(period[0])[:10])
            ends.append(str(period[1])[:10])
    if not starts:
        return None
    return min(starts), max(ends)


def series_years(region: Mapping[str, Any] | None, fallback_split: Mapping[str, Any] | None = None) -> tuple[int, int]:
    rng = series_range_from_region(region)
    if rng:
        return int(rng[0][:4]), int(rng[1][:4])
    if fallback_split and fallback_split.get("test_end") and fallback_split.get("train_end"):
        return int(str(fallback_split["train_end"])[:4]) - 4, int(str(fallback_split["test_end"])[:4])
    return 2015, 2024


def split_years(split: Mapping[str, Any]) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """활성 구간 → 컨트롤 기본값(학습 종료 연도, 검증 (시작, 끝), 시험 (시작, 끝))."""

    y = {key: int(str(split[key])[:4]) for key in SPLIT_KEYS}
    # 연도 경계가 아닌 구간(API 로 만든 사용자 지정, 예: 학습 ≤2021-06-30 · 검증 2021-07-01∼)은 연도만 읽으면
    # 학습 종료 연도와 검증 시작 연도가 겹쳐 기본값부터 순서 경고가 뜬다 — 연도 단위로 겹치지 않게 민다.
    train_end = y["train_end"]
    val_start = max(y["val_start"], train_end + 1)
    val_end = max(y["val_end"], val_start)
    test_start = max(y["test_start"], val_end + 1)
    test_end = max(y["test_end"], test_start)
    return train_end, (val_start, val_end), (test_start, test_end)


def _clamp(value: int, options: Sequence[int]) -> int:
    return min(max(int(value), options[0]), options[-1])


def elapsed_text(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(max(0.0, float(seconds)))
    return f"{total // 60}분 {total % 60}초" if total >= 60 else f"{total}초"


def progress_text(job: Mapping[str, Any]) -> str:
    progress = job.get("progress") or {}
    done = int(progress.get("stations_done") or 0)
    total = int(progress.get("stations_total") or 0)
    current = progress.get("current_station")
    status_ko = {"queued": "대기 중", "running": "학습 중", "done": "완료", "failed": "실패"}.get(str(job.get("status")), str(job.get("status")))
    text = f"{status_ko} · {done}/{total} 지점"
    if current:
        text += f" · 현재 {current}"
    return text + f" · {elapsed_text(job.get('elapsed_s'))} 경과"


# --- 세션 캐시 -------------------------------------------------------------------------------------


def _cache() -> dict[str, Any]:
    if "gauge_cache" not in st.session_state:
        st.session_state.gauge_cache = {}
    return st.session_state.gauge_cache


def _cached(key: str, loader: Callable[[], Any]) -> Any:
    cache = _cache()
    if key not in cache:
        cache[key] = loader()
    return cache[key]


def clear_forecast_caches() -> None:
    """학습 산출물이 바뀐 뒤 — 지역·평가·모델·전망·백테스트·비교·위성수위 캐시와 화면의 예측 결과를 비운다."""

    cache = _cache()
    for key in list(cache):
        if any(key.startswith(prefix) for prefix in CACHE_PREFIXES):
            cache.pop(key, None)
    for key in SESSION_KEYS_TO_CLEAR:
        st.session_state.pop(key, None)


def region_forecast_state(client: APIClient, region: Mapping[str, Any]) -> dict[str, Any]:
    """지역 상세(캐시)에서 구간 상태를 읽는다. 상세를 못 받으면 요약 응답으로 대신한다."""

    rid = str(region["id"])
    try:
        detail = _cached(f"region_detail:{rid}", lambda: client.get_region(rid))
    except APIError:
        detail = region
    return split_from_region(detail) | {"detail": detail}


def render_split_badge(client: APIClient, region: Mapping[str, Any]) -> None:
    """평가·결과 화면용 한 줄 배지."""

    state = region_forecast_state(client, region)
    st.caption(f"평가 구간 — {split_badge_text(state['detail'])}")


# --- 구성 요소 -------------------------------------------------------------------------------------


def _candidate_note(client: APIClient) -> str:
    try:
        forecasters = _cached("forecasters", client.list_gauge_forecasters)
    except APIError:
        return "서버의 예측기 목록을 읽지 못했습니다."
    candidates = [f for f in forecasters if not f.get("is_baseline")]
    plugins = [f["id"] for f in candidates if str(f.get("id", "")).startswith(("neural_", "prophet"))]
    text = f"지금 서버가 내놓는 후보 예측기 {len(candidates)}개(기준선 제외)."
    if plugins:
        text += f" 그중 딥러닝·Prophet 계열 {len(plugins)}개는 서버에 torch·prophet 이 없으면 적합에 실패해 후보에서 조용히 빠집니다."
    else:
        text += " 딥러닝·Prophet 계열은 이 서버에 없어 내장 7계열만으로 고릅니다."
    return text


def _adopt_running_job(client: APIClient, region_id: str) -> str | None:
    """409(이미 학습 중)일 때 실행 중인 작업을 찾아 이어서 지켜본다."""

    try:
        for job in client.gauge_train_jobs(region_id):
            if str(job.get("status")) in ("queued", "running") and job.get("job_id"):
                return str(job["job_id"])
    except APIError:
        return None
    return None


def _poll_job(client: APIClient, region_id: str, job_id: str) -> Mapping[str, Any]:
    """st.status 안에서 2초마다 작업 상태를 읽어 진행을 보여 준다. 최종 상태(또는 시간 초과) 작업을 돌려준다."""

    started = time.monotonic()
    with st.status("학습 요청을 보냈습니다…", expanded=True) as status:
        bar = st.progress(0.0, text="대기 중")
        note = st.empty()
        while True:
            try:
                job = client.gauge_train_job(region_id, job_id)
            except APIError as exc:
                status.update(label=f"작업 상태를 읽지 못했습니다: {exc}", state="error")
                return {"job_id": job_id, "status": "failed", "error": str(exc)}
            progress = job.get("progress") or {}
            total = int(progress.get("stations_total") or 0)
            done = int(progress.get("stations_done") or 0)
            label = progress_text(job)
            bar.progress(min(1.0, done / total) if total else 0.0, text=f"{done}/{total} 지점" + (f" · 현재 {progress.get('current_station')}" if progress.get("current_station") else ""))
            note.caption(f"작업 {job_id} · 구간 {describe_split(job.get('split_used') or {})}")
            state = str(job.get("status") or "")
            if state == "done":
                status.update(label=f"학습 완료 · {label}", state="complete", expanded=False)
                return job
            if state == "failed":
                status.update(label=f"학습 실패 · {label}", state="error")
                return job
            status.update(label=label, state="running")
            if time.monotonic() - started > POLL_TIMEOUT_S:
                status.update(label=f"20분이 지나도 끝나지 않았습니다 · {label}", state="error")
                return dict(job) | {"status": "timeout"}
            time.sleep(POLL_INTERVAL_S)  # 테스트에서는 time.sleep 을 monkeypatch 한다


def _finish_job(job: Mapping[str, Any], *, job_key: str, flash_key: str) -> None:
    state = str(job.get("status") or "")
    if state == "done":
        evaluation = job.get("evaluation") or {}
        message = (
            f"학습 완료 — 구간 {describe_split(job.get('split_used') or evaluation.get('split') or {})} · "
            f"지점 {evaluation.get('n_stations', (job.get('progress') or {}).get('stations_done', 0))}개"
        )
        if evaluation.get("cells_total"):
            message += f" · {evaluation.get('cells_beating_persistence', 0)}/{evaluation.get('cells_total', 0)}셀이 직전값 유지보다 낫습니다"
        if evaluation.get("failures"):
            message += f" · 실패 지점: {evaluation['failures']}"
        st.session_state.pop(job_key, None)
        st.session_state[flash_key] = ("success", message)
        clear_forecast_caches()
        st.rerun()
    elif state == "failed":
        st.session_state.pop(job_key, None)
        st.error(f"학습 실패: {job.get('error') or '원인 미상'}")
    else:  # timeout — 작업은 서버에서 계속 돈다. 키를 남겨 두면 다음 실행에서 이어서 지켜본다
        st.warning("학습이 아직 끝나지 않았습니다. 서버에서 계속 진행되며, 화면을 새로 고치면 이어서 진행을 봅니다.")


def render_training_window(client: APIClient, region: Mapping[str, Any], *, key: str) -> None:
    """학습·검증·시험 구간 컨트롤 + 재학습/되돌리기 버튼. 지역마다 키를 따로 둬 지역을 바꾸면 초기화된다."""

    rid = str(region["id"])
    prefix = f"{key}:{rid}"
    job_key = f"{prefix}:job"
    flash_key = f"{prefix}:flash"
    state = region_forecast_state(client, region)
    active, default, custom = state["active"], state["default"], state["custom"]
    series_range = series_range_from_region(state["detail"]) or series_range_from_region(region)
    first_year, last_year = series_years(state["detail"] if series_range else region, active or default)

    with st.expander(EXPANDER_TITLE, expanded=False):
        flash = st.session_state.pop(flash_key, None)
        if flash:
            (st.success if flash[0] == "success" else st.info)(flash[1])
        st.markdown(
            f"**현재 구간** {describe_split(active)} · "
            + ("`사용자 지정`" if custom else "`기본`")
            + (f" · 학습 {trained_at_text(state['trained_at'])}" if state.get("trained_at") else ""),
            help="기본 구간은 지역 정의(region.json)의 forecast.split 입니다. 사용자 지정은 이 화면에서 다시 학습한 산출물이 활성일 때입니다.",
        )
        if not active:
            st.info("이 지역에는 구간 정보가 없습니다. 지점 시계열을 올린 뒤 학습을 실행하세요.")
            return
        train_options = list(range(first_year, max(first_year, last_year - 1)))  # 검증·시험에 최소 두 해를 남긴다
        val_options = list(range(first_year + 1, max(first_year + 1, last_year)))
        test_options = list(range(first_year + 2, last_year + 1))
        if not train_options or not val_options or not test_options:
            st.info(f"수위계 기록이 {first_year}∼{last_year}년뿐이라 연도 단위로 세 구간을 나눌 수 없습니다.")
            return
        train_default, val_default, test_default = split_years(active)
        c1, c2, c3 = st.columns(3)
        train_end_year = c1.select_slider(
            "학습 종료 연도", options=train_options, value=_clamp(train_default, train_options), key=f"{prefix}:train_end",
            help="기록 시작부터 이 해 12월 31일까지로 모델을 배웁니다.",
        )
        val_years = c2.select_slider(
            "검증 연도 (모델 고르기)", options=val_options,
            value=(_clamp(val_default[0], val_options), _clamp(val_default[1], val_options)), key=f"{prefix}:val_years",
            help="연도별 폴드로 후보 모델·하이퍼파라미터를 고르는 해들입니다.",
        )
        test_years = c3.select_slider(
            "시험 연도 (채점)", options=test_options,
            value=(_clamp(test_default[0], test_options), _clamp(test_default[1], test_options)), key=f"{prefix}:test_years",
            help="학습·선택에 쓰지 않고 오차를 재는 해들입니다.",
        )
        split = years_to_split(int(train_end_year), tuple(val_years), tuple(test_years))
        days = split_days(split, series_range)
        st.markdown(
            f"학습 {series_range[0] if series_range else '기록 시작'} ∼ **{split['train_end']}** · "
            f"검증 **{split['val_start']} ∼ {split['val_end']}** · 시험 **{split['test_start']} ∼ {split['test_end']}**"
            + (f"  \n= 학습 {days['train']:,}일 · 검증 {days['val']:,}일 · 시험 {days['test']:,}일" if days["train"] is not None else "")
        )
        problems = validate_split(split, series_range)
        for problem in problems:
            st.warning(problem)
        same_as_active = all(str(active.get(k, ""))[:10] == split[k] for k in SPLIT_KEYS)
        if same_as_active and not problems:
            st.caption("지금 활성 산출물과 같은 구간입니다. 다시 학습하면 같은 구간으로 후보 선택을 새로 합니다.")
        st.caption(
            "시험 구간을 바꾸면 채점하는 해의 기상(홍수·갈수) 조건이 달라지므로 구간이 다른 결과끼리 오차를 직접 비교할 수 없습니다 — "
            "같은 구간 안에서 모델끼리 비교하세요. " + _candidate_note(client)
        )
        b1, b2 = st.columns([2, 1])
        job_running = bool(st.session_state.get(job_key))
        if b1.button(TRAIN_BUTTON, type="primary", key=f"{prefix}:train_btn", disabled=bool(problems) or job_running, width="stretch"):
            try:
                response = client.gauge_train(rid, split=split, async_run=True)
            except APIError as exc:
                adopted = _adopt_running_job(client, rid) if exc.status_code == 409 else None
                if adopted:
                    st.session_state[job_key] = adopted
                    st.info("이미 이 지역의 학습이 진행 중입니다 — 그 작업을 이어서 지켜봅니다.")
                else:
                    st.error(str(exc))
            else:
                if response.get("job_id"):
                    st.session_state[job_key] = str(response["job_id"])
                else:  # 동기 학습만 아는 예전 서버 — 응답이 곧 평가표다
                    _finish_job({"status": "done", "evaluation": response, "split_used": split}, job_key=job_key, flash_key=flash_key)
        if custom and b2.button(RESET_BUTTON, key=f"{prefix}:reset_btn", disabled=job_running, width="stretch",
                                help="사용자 디렉터리의 학습 산출물을 지우고 내장 산출물(기본 구간)로 되돌립니다."):
            try:
                result = client.gauge_reset_models(rid)
            except APIError as exc:
                st.error(str(exc))
            else:
                removed = result.get("removed_files")
                n_removed = len(removed) if isinstance(removed, list) else removed
                st.session_state[flash_key] = ("info", f"기본 구간 산출물로 되돌렸습니다 — 사용자 산출물 {n_removed or 0}개 삭제 · 구간 {describe_split(default)}")
                clear_forecast_caches()
                st.rerun()
        job_id = st.session_state.get(job_key)
        if job_id:
            job = _poll_job(client, rid, str(job_id))
            _finish_job(job, job_key=job_key, flash_key=flash_key)
