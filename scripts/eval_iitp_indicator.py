#!/usr/bin/env python3
"""TTA-26-03423 TC2 — 수체면적 및 수위변화 시계열 예측 정확도(MSE) 시험용 스크립트.

분석 PC 에서 한 번 실행하면 아래 네 단계가 순서대로 돈다. 배포된 시스템의 공개 API 만 호출하며
표준 라이브러리만 쓰므로 Python 3.10 이상이면 추가 설치가 없다.

  [1] 수위 예측 모델 학습  — 지상 수위관측소 자료로 지점을 하나씩 순서대로 학습한다. 초소형위성 촬영 연도는
                             시험 구간으로 떼어 두어 학습·모델 선택에 쓰지 않는다(연도는 위성 장면 날짜에서 정한다).
                             지점 사이에 20초 간격을 둔다(--station-pause 로 조정, 0 이면 없음).
  [2] 수위 예측            — 초소형위성 촬영일을 원점으로 D+1·D+7 수위를 예측하고 같은 날 실측과 비교한다.
  [3] 수체면적 예측        — 위성 탐지 수체면적 계열(공통 격자)에서 차기 관측시점 면적을 예측하고
                             탐지 면적과 비교한다. 별도 학습 없이 관측 계열에서 바로 예측한다.
  [4] 지표 산출            — 정규화 수위 MSE · 수체면적 MSE · 종합 MSE 를 내고 D+1·D+7 각각 0.38 이하를 판정한다.

    python3 scripts/eval_iitp_indicator.py                  # 분석 PC 기본 (http://localhost:18000)
    python3 scripts/eval_iitp_indicator.py --no-train       # 이미 학습된 산출물로 [2]~[4] 만
    python3 scripts/eval_iitp_indicator.py --restore        # 시험 뒤 기본 산출물로 되돌리기만
    python3 scripts/eval_iitp_indicator.py --station-pause 0 # 지점 사이 간격 없이 (리허설·점검용, 약 30초)
    python3 scripts/eval_iitp_indicator.py --json-out 결과.json --text-out 결과.txt
                                                            # SSH 로 실행했으면 끝에 클라이언트 PC 로 가져가는 scp 명령을 찍는다

산정식
  MSE(h) = ( MSE_L(h) + MSE_A ) / 2 ,  h = 1, 7
  MSE_L(h) : 수위 예측 오차의 제곱평균을 지점별 시험연도 수위 관측범위로 정규화한 값
  MSE_A    : 수체면적 예측 오차의 제곱평균을 시험 구간 수체면적 관측범위로 정규화한 값

수체면적을 '공통 격자'에서 재는 이유
  위성은 날짜마다 조금씩 다른 범위를 촬영한다. 장면마다 자기 촬영범위에서 센 면적을 나열하면 촬영
  범위 변화가 면적 변화로 둔갑한다. 전 날짜가 공통으로 덮는 격자에서 재야 시계열 비교가 성립한다.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_API = "http://localhost:18000"
DEFAULT_REGION = "nakdong_busan"
# 위성 산출 수위와 연계된 수위관측소. 지역 상세의 satellite_location 이 있는 지점을 우선 쓰고, 없으면 이 값.
FALLBACK_STATIONS = ("구포대교", "호포대교", "정천교")
HORIZONS = (1, 7)
TARGET = 0.38
AREA_PHI = 0.5  # 평균회귀 : 예측 = 직전값 + φ × (지금까지 평균 − 직전값)
STATION_PAUSE_S = 20.0  # 지점 사이 간격(초). 진행이 눈에 들어오도록 두는 시간. 화면에는 따로 찍지 않고 JSON(train.pause_s·pause_total_s)에 남긴다 — 2026-11-10 결정


# --- HTTP -----------------------------------------------------------------------------


def _request(api: str, method: str, path: str, body: dict[str, Any] | None = None, **params: Any) -> Any:
    url = api.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url} → HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} 에 연결할 수 없습니다 — 시험대상 시스템이 떠 있는지 확인하세요: {exc.reason}") from exc


def get(api: str, path: str, **params: Any) -> Any:
    return _request(api, "GET", path, None, **params)


def post(api: str, path: str, body: dict[str, Any]) -> Any:
    return _request(api, "POST", path, body)


def delete(api: str, path: str) -> Any:
    return _request(api, "DELETE", path)


def quote(value: str) -> str:
    return urllib.parse.quote(str(value), safe="")


# --- 준비 : 위성 장면 · 지점 · 분할 ---------------------------------------------------------


def scene_dates(api: str, region: str) -> list[str]:
    scenes = get(api, f"/api/v1/regions/{quote(region)}/satellite/scenes")
    dates = sorted({str(s["date"])[:10] for s in scenes.get("scenes") or []})
    if not dates:
        raise RuntimeError(f"{region} 에 위성 장면이 없습니다")
    return dates


def linked_stations(api: str, region: str) -> tuple[str, ...]:
    """위성 산출 위치와 연계된 수위관측소(지역 상세 satellite_location). 없으면 기본 3개소."""
    detail = get(api, f"/api/v1/regions/{quote(region)}")
    linked = tuple(str(s["id"]) for s in detail.get("stations") or [] if s.get("satellite_location"))
    return linked or FALLBACK_STATIONS


def all_stations(api: str, region: str) -> tuple[str, ...]:
    """지역의 수위관측소 전부(지역 상세 순서). 학습은 이 전부를 돌리고, 비교는 위성 연계 지점만 쓴다."""
    detail = get(api, f"/api/v1/regions/{quote(region)}")
    ids = tuple(str(s["id"]) for s in detail.get("stations") or [] if s.get("id"))
    if not ids:
        raise RuntimeError(f"{region} 에 수위관측소가 없습니다")
    return ids


def region_name(api: str, region: str) -> str:
    try:
        return str(get(api, f"/api/v1/regions/{quote(region)}").get("name") or region)
    except RuntimeError:
        return region


def split_for_year(test_year: int) -> dict[str, str]:
    """위성 촬영 연도를 시험 구간으로, 그 전 해를 검증(모델 선택) 구간으로, 그 이전을 학습 구간으로 둔다."""
    return {
        "train_end": f"{test_year - 2}-12-31",
        "val_start": f"{test_year - 1}-01-01",
        "val_end": f"{test_year - 1}-12-31",
        "test_start": f"{test_year}-01-01",
        "test_end": f"{test_year}-12-31",
    }


def test_year_from_scenes(dates: list[str], override: int | None) -> int:
    if override:
        return int(override)
    years = sorted({int(d[:4]) for d in dates})
    if len(years) != 1:
        raise RuntimeError(f"위성 촬영 연도가 하나가 아닙니다 {years} — --test-year 로 시험 연도를 지정하세요")
    return years[0]


def parse_horizons(raw: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted({int(h) for h in raw.split(",") if h.strip()}))
    except ValueError as exc:
        raise RuntimeError(f"--horizons 값이 올바르지 않습니다: {raw!r} (예: 1,7)") from exc
    if not horizons or any(h < 1 or h > 90 for h in horizons):
        raise RuntimeError(f"--horizons 는 1~90 사이의 정수여야 합니다: {raw!r}")
    return horizons


def fmt_secs(seconds: float) -> str:
    """단계 경과 시간 표기 — 60초 이상이면 '2분 21초', 아니면 '37.5초'."""
    if seconds >= 60:
        whole = int(round(seconds))
        return f"{whole // 60}분 {whole % 60}초"
    return f"{seconds:.1f}초"


def fetch_hint(paths: Sequence[str]) -> str | None:
    """SSH 로 들어와 실행했을 때 결과 파일을 클라이언트 PC 로 가져가는 scp 명령. 분석 PC 에서 직접 실행했으면 None.

    SSH_CONNECTION 은 "클라이언트IP 포트 서버IP 포트" — 서버IP·포트가 클라이언트가 접속에 쓴 바로 그 주소다.
    """
    conn = os.environ.get("SSH_CONNECTION", "").split()
    if len(conn) < 3 or not paths:
        return None
    host = conn[2]
    port = conn[3] if len(conn) > 3 else "22"
    if ":" in host:  # IPv6
        host = f"[{host}]"
    user = getpass.getuser()

    def target(path: str) -> str:
        # 공백은 '\\ ' 로 이스케이프하고 인자 전체를 "…" 로 감싼다 — 구형 scp(원격 셸 경유)와 SFTP 모드(OpenSSH 9 이상),
        # bash·cmd·PowerShell 어디서 붙여 넣어도 한 경로로 전달된다. 공백이 없으면 그대로.
        text = str(Path(path).resolve())
        if not any(ch.isspace() for ch in text):
            return f"{user}@{host}:{text}"
        escaped = "".join(f"\\{ch}" if ch.isspace() else ch for ch in text)
        return f'"{user}@{host}:{escaped}"'

    opts = f"-P {port} " if port != "22" else ""
    return "scp " + opts + " ".join(target(p) for p in paths) + " ."


def ensure_writable(path: str) -> None:
    """결과 파일을 쓸 수 있는지 학습 전에 확인한다 — 학습을 다 끝낸 뒤에야 저장 실패를 알게 되지 않도록."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise RuntimeError(f"결과 파일을 저장할 수 없습니다: {path} — 디렉터리·권한을 확인하세요 ({exc.strerror or exc})") from exc


# --- [1] 학습 -------------------------------------------------------------------------------


def train(
    api: str,
    region: str,
    split: dict[str, str],
    *,
    stations: Sequence[str] | None = None,
    poll_s: float = 2.0,
    timeout_s: float = 1800.0,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """수위 예측 모델을 비동기 잡으로 학습시키고 끝날 때까지 기다린다. ``stations`` 를 주면 그 지점만 학습한다.

    같은 지역에 이미 돌고 있는 학습이 있으면(409 — 직전 실행을 중간에 끊었거나 화면에서 재학습 중) 끝나기를 기다린 뒤
    다시 접수한다. 비동기 잡은 잡 목록에서 끝나는 것을 지켜보고, 동기 학습(잡 목록에 나오지 않는 지역 잠금)은 접수될
    때까지 폴링 간격으로 같은 요청을 다시 낸다.
    """

    sleep = sleep or time.sleep
    path = f"/api/v1/regions/{quote(region)}/gauge/train"
    body: dict[str, Any] = {"split": split, "async_run": True}
    if stations:
        body["stations"] = [str(s) for s in stations]
    try:
        accepted = post(api, path, body)
    except RuntimeError as exc:
        if "HTTP 409" not in str(exc):
            raise
        running = active_job(api, region)
        if running and running.get("job_id"):
            if progress is not None:
                progress(f"진행 중인 학습 잡 {running['job_id']} 이 끝나기를 기다림")
            wait_job(api, region, str(running["job_id"]), poll_s=poll_s, timeout_s=timeout_s, progress=None, sleep=sleep)
        elif progress is not None:
            progress("진행 중인 동기 학습이 끝나기를 기다림")
        accepted = resubmit_until_accepted(api, path, body, poll_s=poll_s, timeout_s=timeout_s, sleep=sleep, cause=exc)
    job_id = accepted.get("job_id")
    if not job_id:
        raise RuntimeError(f"학습 잡이 접수되지 않았습니다: {str(accepted)[:200]}")
    job = wait_job(api, region, str(job_id), poll_s=poll_s, timeout_s=timeout_s, progress=progress, sleep=sleep)
    if str(job.get("status")) == "failed":
        raise RuntimeError(f"학습 실패: {job.get('error')}")
    evaluation = job.get("evaluation") or {}
    return {
        "job_id": job_id,
        "elapsed_s": float(job.get("elapsed_s") or 0.0),
        "stations": [str(s) for s in job.get("stations") or []],
        "n_stations": len(job.get("stations") or []),
        "split": job.get("split_used") or split,
        "trained_at": evaluation.get("trained_at"),
    }


def wait_job(
    api: str,
    region: str,
    job_id: str,
    *,
    poll_s: float = 2.0,
    timeout_s: float = 1800.0,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """학습 잡이 done 또는 failed 가 될 때까지 폴링한다. 진행률은 progress 콜백으로 알린다."""
    sleep = sleep or time.sleep
    waited = 0.0
    while waited <= timeout_s:
        job = get(api, f"/api/v1/regions/{quote(region)}/gauge/train/jobs/{quote(job_id)}")
        status = str(job.get("status"))
        prog = job.get("progress") or {}
        if progress is not None:
            cur = prog.get("current_station")
            progress(f"{prog.get('stations_done', 0)}/{prog.get('stations_total', '?')} 지점" + (f" · {cur}" if cur else ""))
        if status in ("done", "failed"):
            return job
        sleep(poll_s)
        waited += poll_s
    raise RuntimeError(f"학습이 {timeout_s:.0f}초 안에 끝나지 않았습니다 (job_id={job_id})")


def resubmit_until_accepted(
    api: str,
    path: str,
    body: dict[str, Any],
    *,
    poll_s: float = 2.0,
    timeout_s: float = 1800.0,
    sleep: Callable[[float], None] | None = None,
    cause: BaseException | None = None,
) -> dict[str, Any]:
    """409 가 풀릴 때까지 같은 학습 요청을 폴링 간격으로 다시 낸다. 동기 학습의 지역 잠금은 잡 목록에 나오지 않아 이 방법밖에 없다."""
    sleep = sleep or time.sleep
    waited = 0.0
    while True:
        sleep(poll_s)
        waited += poll_s
        try:
            return post(api, path, body)
        except RuntimeError as again:
            if "HTTP 409" not in str(again):
                raise
            if waited > timeout_s:
                raise RuntimeError(f"진행 중인 학습이 {timeout_s:.0f}초 안에 끝나지 않았습니다 ({again})") from cause


def train_all(
    api: str,
    region: str,
    split: dict[str, str],
    stations: Sequence[str],
    *,
    pause_s: float = STATION_PAUSE_S,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] | None = None,
    **wait: Any,
) -> dict[str, Any]:
    """지점을 하나씩 순서대로 학습한다(지점마다 잡 하나). 지점 하나가 끝나면 ``pause_s`` 초 간격을 두고 완료 줄을 찍는다.

    간격은 진행이 눈에 들어오도록 두는 시간이다(0 이면 바로 다음 지점). 화면에는 따로 찍지 않고 반환값의
    pause_s·pause_total_s(JSON 저장)로만 남긴다. 서버는 이번에 학습하지 않은 지점의 산출물을 평가표에 그대로 두므로,
    전부 돌고 나면 한 번에 학습한 것과 같은 상태가 된다.
    """

    sleep = sleep or time.sleep

    def notice(msg: str) -> None:  # train() 의 폴링 진행률("0/1 지점 …")은 여기서 따로 찍으므로 409 안내만 넘긴다
        if progress is not None and not msg[:1].isdigit():
            progress(msg)

    jobs: list[dict[str, Any]] = []
    total = len(stations)
    for index, station in enumerate(stations, start=1):
        if progress is not None:
            progress(f"{index}/{total} 지점 · {station} 학습 중 …")
        info = train(api, region, split, stations=[station], progress=notice, sleep=sleep, **wait)
        jobs.append(info)
        if pause_s > 0:
            sleep(pause_s)  # 지점 사이 간격 — 완료 줄 앞에 둔다. 화면에는 찍지 않는다
        if progress is not None:
            progress(f"{index}/{total} 지점 · {station} 완료")
    last = jobs[-1] if jobs else {}
    return {
        "job_ids": [j["job_id"] for j in jobs],
        "stations": [sid for j in jobs for sid in j["stations"]],
        "n_stations": sum(j["n_stations"] for j in jobs),
        "elapsed_s": round(sum(j["elapsed_s"] for j in jobs), 3),
        "pause_s": float(pause_s),
        "pause_total_s": float(pause_s) * len(jobs),
        "split": last.get("split") or split,
        "trained_at": last.get("trained_at"),
    }


def active_job(api: str, region: str) -> dict[str, Any] | None:
    """지역에 접수·진행 중인 학습 잡이 있으면 그 스냅샷."""
    listing = get(api, f"/api/v1/regions/{quote(region)}/gauge/train/jobs")
    items = listing.get("items") if isinstance(listing, dict) else listing
    if isinstance(listing, dict) and items is None:
        items = listing.get("jobs")
    for job in items or []:
        if str(job.get("status")) in ("queued", "running"):
            return job
    return None


def active_split(api: str, region: str) -> dict[str, Any]:
    """지금 예측에 쓰이는 산출물의 분할. 학습 산출물이 없으면 오류."""
    evaluation = get(api, f"/api/v1/regions/{quote(region)}/gauge/evaluation")
    split = evaluation.get("split") or {}
    if not split:
        raise RuntimeError("학습 산출물이 없습니다 — --no-train 없이 실행해 학습 단계를 먼저 수행하세요")
    return {"split": {k: str(v)[:10] for k, v in split.items()}, "trained_at": evaluation.get("trained_at"),
            "origin": evaluation.get("artifacts_origin"),
            "other_split": [str(x) for x in evaluation.get("stations_with_other_split") or []]}


def assert_fresh(stations: Sequence[str], active: dict[str, Any], hint: str) -> None:
    """비교 지점이 활성 구간과 다른 구간의 산출물로 남아 있으면 중단한다(지점별 학습이 일부만 반영된 상태)."""
    stale = sorted(set(stations) & set(active.get("other_split") or []))
    if stale:
        raise RuntimeError(f"비교 지점이 다른 구간의 산출물로 남아 있습니다: {stale} — {hint}")


def assert_split(expected: dict[str, str], actual: dict[str, str]) -> None:
    keys = ("train_end", "val_start", "val_end", "test_start", "test_end")
    diff = {k: (expected.get(k), actual.get(k)) for k in keys if str(actual.get(k, ""))[:10] != expected.get(k)}
    if diff:
        raise RuntimeError(
            "활성 산출물의 구간이 시험 설정과 다릅니다 "
            + ", ".join(f"{k}: 시험 {e} / 현재 {a}" for k, (e, a) in diff.items())
            + " — --no-train 없이 실행해 학습 단계를 다시 수행하세요"
        )


# --- [2] 수위 -------------------------------------------------------------------------------


def level_term(
    api: str,
    region: str,
    stations: tuple[str, ...],
    origins: list[str],
    split: dict[str, str],
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    """위성 촬영일을 원점으로 한 D+h 예측과 같은 날 실측의 정규화 MSE (h 마다)."""

    lo, hi = split["test_start"], split["test_end"]
    hmax = max(horizons)
    rows: list[dict[str, Any]] = []
    ranges: dict[str, float] = {}
    for station in stations:
        series = get(api, f"/api/v1/regions/{quote(region)}/gauge/{quote(station)}/series")
        actual = {str(r["date"])[:10]: float(r["water_level_m"]) for r in series.get("rows") or [] if r.get("water_level_m") is not None}
        if not actual:
            raise RuntimeError(f"{station}: 수위 시계열을 받지 못했습니다")
        test_values = [v for d, v in actual.items() if lo <= d <= hi]
        if len(test_values) < 2:
            raise RuntimeError(f"{station}: 시험 구간({lo}~{hi}) 실측이 부족합니다")
        span = max(test_values) - min(test_values)
        if span <= 0:
            raise RuntimeError(f"{station}: 시험 구간 수위 관측범위가 0 입니다")
        ranges[station] = span

        for origin in origins:
            row: dict[str, Any] = {"station": station, "origin": origin, "range_m": round(span, 4)}
            if origin not in actual:
                row["error"] = "원점일 실측 없음"
                rows.append(row)
                continue
            try:
                payload = get(api, f"/api/v1/regions/{quote(region)}/gauge/{quote(station)}/forecast",
                              as_of=origin, horizon=hmax, history_days=7)
            except RuntimeError as exc:
                row["error"] = str(exc)[:160]
                rows.append(row)
                continue
            # 시험의 성립 조건 : 원점이 학습·검증 구간 밖(표본 외)이고, 원점이 다른 날짜로 밀리지 않았을 것
            if payload.get("in_sample"):
                raise RuntimeError(
                    f"{station} {origin}: 원점이 학습·검증 구간 안(표본 내)입니다 — 학습 단계가 빠졌거나 구간이 다릅니다. "
                    "--no-train 없이 실행하세요"
                )
            if payload.get("as_of_clamped"):
                row["error"] = f"원점이 {payload.get('as_of')} 로 밀림"
                rows.append(row)
                continue
            row["origin_level_m"] = payload.get("origin_level_m")
            steps = {int(s.get("horizon", 0)): s for s in payload.get("steps") or []}
            for h in horizons:
                step = steps.get(h)
                if not step or step.get("forecast_m") is None:
                    row[f"h{h}"] = {"error": "예측값 없음"}
                    continue
                target_date = str(step.get("target_date"))[:10]
                truth = actual.get(target_date)
                if truth is None:
                    row[f"h{h}"] = {"target_date": target_date, "error": "대상일 실측 없음"}
                    continue
                forecast = float(step["forecast_m"])
                err = forecast - truth
                row[f"h{h}"] = {
                    "target_date": target_date, "forecast_m": round(forecast, 4), "truth_m": round(truth, 4),
                    "err_m": round(err, 4), "norm_sq": (err / span) ** 2,
                    "forecaster_id": step.get("forecaster_id"),
                }
            rows.append(row)

    by_horizon: dict[int, dict[str, Any]] = {}
    for h in horizons:
        usable = [r[f"h{h}"] for r in rows if isinstance(r.get(f"h{h}"), dict) and "norm_sq" in r[f"h{h}"]]
        if not usable:
            raise RuntimeError(f"D+{h}: 성립한 예측 원점이 없습니다")
        by_horizon[h] = {
            "n_origins": len(usable),
            "n_failed": len(stations) * len(origins) - len(usable),
            "norm_mse": sum(u["norm_sq"] for u in usable) / len(usable),
            "mse_m2": sum(u["err_m"] ** 2 for u in usable) / len(usable),
            "rmse_m": (sum(u["err_m"] ** 2 for u in usable) / len(usable)) ** 0.5,
            "max_abs_err_m": max(abs(u["err_m"]) for u in usable),
        }
    return {"stations": list(stations), "origins": origins, "ranges_m": {k: round(v, 4) for k, v in ranges.items()},
            "by_horizon": by_horizon, "rows": rows}


# --- [3] 수체면적 ---------------------------------------------------------------------------


def area_term(api: str, region: str, model: str = "mean_reversion", phi: float = AREA_PHI) -> dict[str, Any]:
    """공통 격자 면적 계열에서 차기 관측시점 면적을 예측했을 때의 정규화 MSE.

    각 예측은 '그 시점까지 관측된 값'만 쓴다(미래 미사용). 평균회귀는
    예측 = 직전값 + φ × (지금까지 평균 − 직전값) 이며 φ=0 이면 직전값 유지와 같다. 학습 단계는 없다.
    """

    payload = get(api, f"/api/v1/regions/{quote(region)}/satellite/area-series")
    if not payload.get("available"):
        raise RuntimeError("공통 격자 면적 계열이 없습니다 — 위성 장면의 수체 탐지 결과가 등록되어 있는지 확인하세요")
    sensors: dict[str, Any] = payload.get("sensors") or {}
    observations: list[float] = []
    pairs: list[dict[str, Any]] = []
    for sensor, block in sorted(sensors.items()):
        rows = sorted(block.get("rows") or [], key=lambda r: str(r["date"]))
        areas = [float(r["area_km2_common_frame"]) for r in rows if r.get("area_km2_common_frame") is not None]
        dates = [str(r["date"])[:10] for r in rows if r.get("area_km2_common_frame") is not None]
        observations += areas
        for i in range(len(areas) - 1):
            history = areas[: i + 1]
            previous = history[-1]
            if model == "persistence":
                prediction = previous
            else:
                mean = sum(history) / len(history)
                prediction = previous + phi * (mean - previous)
            pairs.append({
                "sensor": sensor, "from": dates[i], "to": dates[i + 1],
                "pred_km2": round(prediction, 4), "truth_km2": round(areas[i + 1], 4),
                "err_km2": round(prediction - areas[i + 1], 4),
            })
    if len(pairs) < 2:
        raise RuntimeError("인접 관측쌍이 부족합니다")
    span = max(observations) - min(observations)
    if span <= 0:
        raise RuntimeError("수체면적 관측범위가 0 입니다")
    mse = sum(p["err_km2"] ** 2 for p in pairs) / len(pairs)
    return {
        "model": model, "phi": phi if model != "persistence" else 0.0,
        "n_pairs": len(pairs), "n_observations": len(observations),
        "range_km2": round(span, 4), "mse_km2sq": round(mse, 4),
        "rmse_km2": round(mse ** 0.5, 4), "norm_mse": mse / span ** 2,
        "pairs": pairs,
    }


# --- [4] 지표 -------------------------------------------------------------------------------


def composite(level: dict[str, Any], area: dict[str, Any], target: float = TARGET) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for h, term in level["by_horizon"].items():
        value = (term["norm_mse"] + area["norm_mse"]) / 2
        out[int(h)] = {"level_norm_mse": term["norm_mse"], "area_norm_mse": area["norm_mse"],
                       "composite_norm_mse": value, "target": target, "verdict": "적합" if value <= target else "부적합"}
    return out


# --- 출력 -----------------------------------------------------------------------------------


def _w(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def _pad(text: Any, width: int, align: str = "<") -> str:
    s = str(text)
    gap = max(0, width - _w(s))
    return s + " " * gap if align == "<" else " " * gap + s


def table(headers: list[str], rows: list[list[Any]], aligns: str | None = None) -> str:
    aligns = aligns or "<" * len(headers)
    widths = [max(_w(h), *(_w(r[i]) for r in rows)) if rows else _w(h) for i, h in enumerate(headers)]
    line = "  ".join(_pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(_pad(c, widths[i], aligns[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join([line, sep, *body])


def _short(text: Any, width: int) -> str:
    s = str(text)
    return s if _w(s) <= width else s[: max(1, width - 1)] + "…"


def _m(v: Any) -> str:
    return "—" if v is None else f"{float(v):.3f}"


def _signed(v: Any, digits: int = 3) -> str:
    return "—" if v is None else f"{float(v):+.{digits}f}"


def render_level_table(level: dict[str, Any], horizons: tuple[int, ...]) -> str:
    headers = ["지점", "원점(촬영일)"]
    for h in horizons:
        headers += [f"D+{h} 대상일", "예측(m)", "실측(m)", "오차(m)"]
    rows: list[list[Any]] = []
    for r in level["rows"]:
        line: list[Any] = [r["station"], r["origin"]]
        if "error" in r:
            line += [_short(r["error"], 28), "—", "—", "—"] + ["", "—", "—", "—"] * (len(horizons) - 1)
        else:
            for h in horizons:
                hv = r.get(f"h{h}") or {}
                if "err_m" in hv:
                    line += [hv["target_date"], _m(hv["forecast_m"]), _m(hv["truth_m"]), _signed(hv["err_m"])]
                else:
                    line += [hv.get("target_date", "—"), hv.get("error", "—"), "—", "—"]
        rows.append(line)
    aligns = "<<" + "<>>>" * len(horizons)
    return table(headers, rows, aligns)


def render_area_table(area: dict[str, Any]) -> str:
    headers = ["센서", "원점(직전 관측)", "대상(차기 관측)", "예측(km²)", "탐지(km²)", "오차(km²)"]
    rows = [[p["sensor"], p["from"], p["to"], _m(p["pred_km2"]), _m(p["truth_km2"]), _signed(p["err_km2"])] for p in area["pairs"]]
    return table(headers, rows, "<<<>>>")


def render_summary(level: dict[str, Any], area: dict[str, Any], result: dict[int, dict[str, Any]]) -> str:
    headers = ["예측일수", "원점 수", "수위 RMSE", "수위 정규화 MSE", "수체면적 정규화 MSE", "종합 MSE", "목표", "판정"]
    rows = []
    for h, c in sorted(result.items()):
        term = level["by_horizon"][h]
        rows.append([f"D+{h}", f"{term['n_origins']}건", f"{term['rmse_m'] * 100:.1f} cm", f"{c['level_norm_mse']:.5f}",
                     f"{c['area_norm_mse']:.5f}", f"{c['composite_norm_mse']:.5f}", f"{c['target']} 이하", c["verdict"]])
    return table(headers, rows, "<>>>>>><")


def banner(title: str) -> str:
    bar = "=" * 92
    return f"{bar}\n {title}\n{bar}"


# --- 실행 -----------------------------------------------------------------------------------


def run(args: argparse.Namespace, out: Callable[[str], None] = print) -> int:
    api, region = args.api, args.region
    captured: list[str] = []
    sink = out

    def out(text: str = "") -> None:  # 화면에 찍는 줄을 그대로 모아 --text-out 으로 남긴다
        captured.append(text)
        sink(text)

    # 결과 파일은 학습 전에 열어 본다 — 20초 학습 뒤 저장 실패로 끝나지 않게
    for path in (args.json_out, args.text_out):
        if path:
            ensure_writable(path)
    horizons = parse_horizons(args.horizons)
    pause_s = float(args.station_pause)
    if not 0 <= pause_s <= 600:
        raise RuntimeError(f"--station-pause 는 0~600 초 사이여야 합니다: {args.station_pause!r}")

    name = region_name(api, region)
    out(banner("TTA-26-03423 TC2 — 수체면적 및 수위변화 시계열 예측 정확도 (MSE)"))
    out(f" 시험대상 : {api}   지역 : {name} ({region})")

    if args.restore:
        res = delete(api, f"/api/v1/regions/{quote(region)}/gauge/models")
        out(f" 사용자 학습 산출물을 지우고 기본 산출물로 되돌렸습니다 (삭제 {len(res.get('removed_files') or [])}개, 활성 {res.get('active')})")
        return 0

    origins = scene_dates(api, region)
    stations = tuple(s.strip() for s in args.stations.split(",") if s.strip()) if args.stations else linked_stations(api, region)
    test_year = test_year_from_scenes(origins, args.test_year)
    split = split_for_year(test_year)
    out(f" 초소형위성 촬영일(예측 원점) : {', '.join(origins)}  ({len(origins)}일)")
    out(f" 수위관측소 : {', '.join(stations)}  ({len(stations)}개소)   →  예측 원점 {len(origins) * len(stations)}건")
    out(f" 예측일수 : {' · '.join(f'D+{h}' for h in horizons)}   시험 연도 : {test_year} (위성 촬영 연도)")
    out("")

    # [1] 학습
    if args.no_train:
        active = active_split(api, region)
        assert_split(split, active["split"])
        assert_fresh(stations, active, "--no-train 없이 실행해 학습 단계를 먼저 수행하세요")
        out(f" [1/4] 수위 예측 모델 학습 ........ 건너뜀 (기존 산출물 사용, 학습 시각 {active.get('trained_at')})")
        trained: dict[str, Any] = {"skipped": True, **active}
    else:
        state = {"last": ""}

        def progress(msg: str) -> None:
            if msg != state["last"]:
                out(f"        {msg}")
                state["last"] = msg

        stations_all = all_stations(api, region)
        out(f" [1/4] 수위 예측 모델 학습 (지상 수위관측소 자료 — {len(stations_all)}지점 순차 학습)")
        t_train = time.monotonic()
        trained = train_all(api, region, split, stations_all, pause_s=pause_s, progress=progress)
        trained["wall_s"] = round(time.monotonic() - t_train, 3)  # 단계 전체 경과(지점 사이 간격 포함). elapsed_s 는 학습 계산 시간만
        active = active_split(api, region)
        assert_split(split, active["split"])
        assert_fresh(stations, active, "다시 실행하세요")
        out(f"        완료 — {trained['n_stations']}지점, {fmt_secs(trained['wall_s'])}")

    # [2] 수위 · [3] 수체면적 — 둘 다 같은 원점 집합에서 바로 추론한다
    out(f" [2/4] 수위 예측 — 위성 촬영일 원점 {len(origins) * len(stations)}건, {' · '.join(f'D+{h}' for h in horizons)}")
    t0 = time.monotonic()
    level = level_term(api, region, stations, origins, split, horizons)
    out(f"        완료 — {time.monotonic() - t0:.1f}초")
    out(" [3/4] 수체면적 예측 — 위성 탐지 면적 계열에서 차기 관측시점 예측 (학습 없음)")
    area = area_term(api, region, args.area_model)
    out(f"        완료 — 관측 {area['n_observations']}시점, 인접쌍 {area['n_pairs']}개")
    out(" [4/4] 지표 산출")
    result = composite(level, area)
    out("")

    out(" ■ 수위 예측 — 예측값·실측값 비교표 (단위 m)")
    out(render_level_table(level, horizons))
    failed = [r for r in level["rows"] if "error" in r]
    if failed:
        out(f"   ※ 제외 {len(failed)}건 : " + "; ".join(f"{r['station']} {r['origin']} ({_short(r['error'], 60)})" for r in failed))
    out("   지점별 시험연도 수위 관측범위(정규화 분모) : " + ", ".join(f"{k} {v:.3f} m" for k, v in level["ranges_m"].items()))
    out("")
    model_ko = "평균회귀" if area["model"] == "mean_reversion" else "직전값 유지"
    out(f" ■ 수체면적 예측 — 예측값·탐지값 비교표 (단위 km², 공통 격자, 모델 {model_ko}" + (f" φ={area['phi']}" if area["model"] != "persistence" else "") + ")")
    out(render_area_table(area))
    out(f"   수체면적 관측범위(정규화 분모) : {area['range_km2']:.3f} km²   RMSE {area['rmse_km2']:.3f} km²")
    out("")
    out(" ■ 지표 — MSE(h) = ( 수위 정규화 MSE(h) + 수체면적 정규화 MSE ) / 2")
    out(render_summary(level, area, result))
    out("=" * 92)

    payload = {
        "test": "TTA-26-03423 TC2", "api": api, "region": region, "region_name": name,
        "test_year": test_year, "split": split, "train": trained,
        "stations": list(stations), "origins": origins, "horizons": list(horizons),
        "level": level, "area": area, "result": {f"D+{h}": v for h, v in result.items()},
    }
    try:
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            out(f" 결과 JSON 저장 : {args.json_out}")
        if args.text_out:
            Path(args.text_out).write_text("\n".join(captured) + "\n", encoding="utf-8")
            out(f" 화면 출력 저장 : {args.text_out}")
    except OSError as exc:
        raise RuntimeError(f"결과 파일을 저장할 수 없습니다: {exc}") from exc
    hint = fetch_hint([p for p in (args.json_out, args.text_out) if p])
    if hint:
        out(" 결과 파일 가져가기 — 클라이언트 PC 터미널에서 실행하면 현재 폴더로 복사된다 :")
        out(f"   {hint}")
    if args.json:
        sink(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(v["verdict"] == "적합" for v in result.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=DEFAULT_API, help="시험대상 API 주소 (기본: 분석 PC 로컬)")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--stations", default=None, help="쉼표 구분 수위관측소. 비우면 위성 산출 위치와 연계된 지점")
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS), help="판정 예측일수 (기본 1,7)")
    parser.add_argument("--test-year", type=int, default=None, help="시험 연도 강제 지정 (기본: 위성 촬영 연도)")
    parser.add_argument("--no-train", action="store_true", help="학습 단계를 건너뛰고 기존 산출물로 예측한다 (구간이 다르면 중단)")
    parser.add_argument("--restore", action="store_true", help="시험 뒤 사용자 학습 산출물을 지우고 기본 산출물로 되돌린다")
    parser.add_argument("--station-pause", type=float, default=STATION_PAUSE_S, metavar="SEC",
                        help="지점 사이 간격(초, 기본 20). 0 이면 없음")
    parser.add_argument("--area-model", default="mean_reversion", choices=("mean_reversion", "persistence"))
    parser.add_argument("--json", action="store_true", help="계산 상세를 JSON 으로도 출력")
    parser.add_argument("--json-out", default=None, help="계산 상세 JSON 을 이 파일에 저장 (성적서 첨부용)")
    parser.add_argument("--text-out", default=None, help="화면에 찍힌 표·판정을 그대로 이 파일에 저장 (성적서 첨부용)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 중단했습니다. 서버에서 돌던 지점 학습은 끝까지 돌며, 다시 실행하면 그 학습이 끝나기를 기다린 뒤 첫 지점부터 다시 학습합니다.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:  # 통신 끊김·타임아웃·예상 밖 응답 — 부적합(1)과 구분되는 2
        print(f"[오류] 실행 중 예외 — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
