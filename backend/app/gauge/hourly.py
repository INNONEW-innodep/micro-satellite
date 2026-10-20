"""시 단위 수위 — WAMIS ``wkw/wl_hrdata`` 수집·저장·릿지 자기회귀 전망 (2026-10-20, 실시간형 예측).

일 단위 예측기(``forecasters.py``)와 별개의 얇은 계층이다. 알고리즘은 ``scripts/eval_realtime_forecast_ideas.py`` 의
시 단위 실증(릿지 회귀: 최근 24시간 변화량 24개 + 24h·7일 평균 편차 + 시각·계절 조화항, 목표 = 원점 대비 변화량)을
그대로 서비스화한 것이다. 학습은 호출 시점에 즉석으로 한다(수 초, CPU) — 자료가 바뀌면 자동으로 새 모델이다.

저장: 지역 사용자 사본 ``<user_dir>/<region>/gauge/hourly/<station_id>.json`` (schema watercast.gauge.hourly/1,
rows [[YYYYMMDDHH, m], …] — WAMIS 의 HH 는 1..24 이고 24 는 다음날 0시다).
WAMIS 는 인증키가 없고 약 2시간 지연으로 최신 시각까지 준다. 합성값은 없다.
"""

from __future__ import annotations

import json
import math
import re
import threading
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

HOURLY_SCHEMA = "watercast.gauge.hourly/1"
WAMIS_HOURLY_URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkw/wl_hrdata?obscd={code}&startdt={start}&enddt={end}&output=json"
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
HORIZONS_HOURS = (1, 3, 6, 12, 24)
MIN_HOURS_TO_FIT = 24 * 30          # 30일은 있어야 전망한다(그 미만이면 직전값 유지만)
MAX_HORIZON_HOURS = 48
LAG_HOURS = 24
WARMUP_HOURS = 24 * 7 + 1           # 7일 평균 편차 특징을 만들려면 앞에 7일이 있어야 한다
_ROW = re.compile(r'\{"ymdh":"(\d{10})","wl":"([^"]*)"\}')
_LOCK = threading.Lock()


# ── WAMIS 수집 ────────────────────────────────────────────────────────────────────


def parse_hourly_payload(payload: str) -> list[tuple[str, float | None]]:
    """정상 JSON 이면 그대로, 잘렸으면 정규식으로 건진다 → [(YYYYMMDDHH, m|None)]."""

    try:
        data = json.loads(payload)
        out: list[tuple[str, float | None]] = []
        for row in data.get("list") or []:
            raw = str(row.get("wl", "")).strip()
            try:
                out.append((str(row["ymdh"]), float(raw) if raw else None))
            except (KeyError, ValueError):
                continue
        return out
    except (ValueError, AttributeError):
        return [(k, float(v) if v.strip() else None) for k, v in _ROW.findall(payload)]


def _month_chunks(start: Date, end: Date) -> list[tuple[Date, Date]]:
    out: list[tuple[Date, Date]] = []
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cur, min(end, nxt - timedelta(days=1))))
        cur = nxt
    return out


def wamis_fetch_hourly(code: str, start: Date, end: Date, *, fetch: Callable[[str], str] | None = None) -> tuple[list[tuple[str, float | None]], list[str]]:
    """월 단위로 나눠 받는다(한 번에 길게 물으면 WAMIS 응답이 잘린다). → (행, 실패한 달 목록). 테스트는 ``fetch`` 를 바꿔 끼운다."""

    if fetch is None:
        from ..screens.wamis import load_wamis_module  # 지연 import — 일 수위 스크립트의 재시도 fetch 를 그대로 쓴다

        module = load_wamis_module()
        fetch = lambda url: module.fetch(url, attempts=3, timeout=60.0)  # noqa: E731
    rows: list[tuple[str, float | None]] = []
    failures: list[str] = []
    for lo, hi in _month_chunks(start, end):
        url = WAMIS_HOURLY_URL.format(code=code, start=lo.strftime("%Y%m%d"), end=hi.strftime("%Y%m%d"))
        payload = fetch(url)
        if not payload or payload.startswith("__error__"):
            failures.append(f"{lo:%Y-%m}: {payload[:80] if payload else 'empty'}")
            continue
        got = parse_hourly_payload(payload)
        if not got:
            failures.append(f"{lo:%Y-%m}: 0 rows")
        rows.extend(got)
    return rows, failures


# ── 저장(지역 사용자 사본) ───────────────────────────────────────────────────────────


def hourly_path(region_user_dir: Path, station_id: str) -> Path:
    return Path(region_user_dir) / "gauge" / "hourly" / f"{station_id}.json"


def load_hourly(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list):
        return None
    return doc


def merge_hourly(existing: dict[str, Any] | None, rows: list[tuple[str, float | None]], *, station: dict[str, Any]) -> dict[str, Any]:
    """같은 시각은 새 값으로 바꾸고(None 은 기존 값을 지우지 않는다) 시각순으로 정렬한다."""

    table: dict[str, float] = {}
    for k, v in (existing or {}).get("rows") or []:
        if v is not None:
            table[str(k)] = float(v)
    n_new = 0
    for k, v in rows:
        if v is None or not math.isfinite(v):
            continue
        if str(k) not in table:
            n_new += 1
        table[str(k)] = float(v)
    keys = sorted(table)
    doc = {
        "schema": HOURLY_SCHEMA,
        "station_id": station.get("id"),
        "station_name": station.get("name"),
        "source": "WAMIS wkw/wl_hrdata (개방 API, 무인증)",
        "source_code": station.get("code"),
        "unit": "m",
        "n": len(keys),
        "period": [keys[0], keys[-1]] if keys else None,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rows": [[k, round(table[k], 3)] for k in keys],
        "note_ko": "YYYYMMDDHH, HH 는 1..24(24 는 다음날 0시). 결측 시각은 행이 없다.",
    }
    doc["n_new"] = n_new
    return doc


def save_hourly(path: Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with _LOCK:
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# ── 시계열 → 정규 격자 ───────────────────────────────────────────────────────────────


def parse_ymdh(key: str) -> datetime:
    y, mo, d, hh = int(key[:4]), int(key[4:6]), int(key[6:8]), int(key[8:10])
    return datetime(y, mo, d) + timedelta(hours=hh)


def to_grid(rows: list[list[Any]], *, max_gap_fill: int = 3) -> tuple[list[datetime], np.ndarray]:
    """시각순 행 → 빠진 시각이 없는 1시간 격자. 3시간 이하 결측은 선형 보간, 그 이상은 NaN."""

    pairs = [(parse_ymdh(str(k)), float(v)) for k, v in rows if v is not None]
    if not pairs:
        return [], np.zeros(0)
    t0 = min(t for t, _ in pairs)
    t1 = max(t for t, _ in pairs)
    n = int((t1 - t0).total_seconds() // 3600) + 1
    y = np.full(n, np.nan)
    for t, v in pairs:
        y[int((t - t0).total_seconds() // 3600)] = v
    isn = np.isnan(y)
    if isn.any() and (~isn).sum() >= 2:
        idx = np.arange(n)
        yi = np.interp(idx, idx[~isn], y[~isn])
        run = 0
        gap = np.zeros(n, dtype=int)
        for i in range(n):
            run = run + 1 if isn[i] else 0
            gap[i] = run
        # 결측 구간의 길이는 그 구간 끝의 run 값 — 구간 전체에 같은 길이를 적용한다
        i = 0
        while i < n:
            if isn[i]:
                j = i
                while j < n and isn[j]:
                    j += 1
                if j - i <= max_gap_fill:
                    y[i:j] = yi[i:j]
                i = j
            else:
                i += 1
    times = [t0 + timedelta(hours=int(i)) for i in range(n)]
    return times, y


# ── 릿지 자기회귀 ───────────────────────────────────────────────────────────────────


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    ym = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return w, mu, sd, ym


def _ridge_predict(m: tuple[np.ndarray, np.ndarray, np.ndarray, float], X: np.ndarray) -> np.ndarray:
    w, mu, sd, ym = m
    return ((X - mu) / sd) @ w + ym


def hourly_features(y: np.ndarray, times: list[datetime], t: np.ndarray) -> np.ndarray:
    cols = [y[t - i] - y[t - i - 1] for i in range(LAG_HOURS)]
    csum = np.nancumsum(np.nan_to_num(y))
    cnt = np.cumsum(np.isfinite(y).astype(float))

    def rmean(w: int) -> np.ndarray:
        hi = t
        lo = t - w
        s = csum[hi] - np.where(lo >= 0, csum[np.maximum(lo, 0)], 0.0)
        c = cnt[hi] - np.where(lo >= 0, cnt[np.maximum(lo, 0)], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / c, np.nan)

    cols.append(y[t] - rmean(24))
    cols.append(y[t] - rmean(24 * 7))
    hod = np.array([times[i].hour for i in t], dtype=float)
    doy = np.array([times[i].timetuple().tm_yday for i in t], dtype=float)
    cols += [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    return np.column_stack(cols)


def _rmse(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    return float(math.sqrt(np.mean(e * e))) if e.size else float("nan")


class HourlyForecaster:
    """한 지점의 시 단위 전망기 — 원점까지의 자료로 그 자리에서 학습한다(예측시간마다 릿지 하나)."""

    def __init__(self, times: list[datetime], y: np.ndarray) -> None:
        self.times = times
        self.y = y
        self.models: dict[int, dict[str, Any]] = {}

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "HourlyForecaster":
        times, y = to_grid(doc.get("rows") or [])
        return cls(times, y)

    def origin_index(self, as_of: datetime | None) -> int:
        """원점 = 지정 시각(포함) 이전의 마지막 유효 관측 시각 인덱스."""

        if not self.times:
            raise ValueError("시 단위 자료가 없습니다")
        valid = np.where(np.isfinite(self.y))[0]
        if valid.size == 0:
            raise ValueError("유효한 시 단위 관측이 없습니다")
        if as_of is None:
            return int(valid[-1])
        pos = int((as_of - self.times[0]).total_seconds() // 3600)
        pos = min(pos, len(self.times) - 1)
        if pos < 0:
            raise ValueError(f"원점 {as_of:%Y-%m-%d %H:%M} 이전에는 자료가 없습니다 — 자료 기간 {self.times[0]:%Y-%m-%d %H시} ~ {self.times[-1]:%Y-%m-%d %H시}")
        before = valid[valid <= pos]
        if before.size == 0:
            raise ValueError("원점 이전에 유효한 관측이 없습니다")
        return int(before[-1])

    def fit(self, h: int, end_index: int) -> dict[str, Any] | None:
        """``end_index`` 까지의 자료만으로 h시간 뒤 변화량 모델을 만든다. 마지막 20 % 를 검증으로 λ 를 고른다."""

        n_use = end_index + 1
        if n_use < MIN_HOURS_TO_FIT + WARMUP_HOURS:
            return None
        t_all = np.arange(WARMUP_HOURS, n_use - h)
        if t_all.size < 48:
            return None
        X = hourly_features(self.y, self.times, t_all)
        target = self.y[t_all + h] - self.y[t_all]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(target)
        X, target = X[ok], target[ok]
        if X.shape[0] < 48:
            return None
        cut = int(X.shape[0] * 0.8)
        best: tuple[float, float] | None = None
        for lam in LAMBDAS:
            m = _ridge_fit(X[:cut], target[:cut], lam)
            mse = float(np.mean((_ridge_predict(m, X[cut:]) - target[cut:]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, lam)
        assert best is not None
        pers = float(np.mean(target[cut:] ** 2))
        model = _ridge_fit(X, target, best[1])
        info = {
            "model": model, "lambda": best[1], "n_fit": int(X.shape[0]),
            "val_rmse_m": round(math.sqrt(best[0]), 4), "val_persistence_rmse_m": round(math.sqrt(pers), 4),
            "val_skill": round(1.0 - best[0] / pers, 3) if pers > 0 else None,
        }
        self.models[h] = info
        return info

    def forecast(self, *, as_of: datetime | None, horizon_hours: int, history_hours: int = 72) -> dict[str, Any]:
        o = self.origin_index(as_of)
        origin_time = self.times[o]
        origin_level = float(self.y[o])
        horizon_hours = max(1, min(int(horizon_hours), MAX_HORIZON_HOURS))
        feat = None
        if o >= WARMUP_HOURS:
            f = hourly_features(self.y, self.times, np.array([o]))
            feat = f if np.isfinite(f).all() else None
        steps: list[dict[str, Any]] = []
        n_obs = 0
        err: list[float] = []
        pers_err: list[float] = []
        for h in range(1, horizon_hours + 1):
            info = self.models.get(h) or (self.fit(h, o) if feat is not None else None)
            target_time = origin_time + timedelta(hours=h)
            if info is not None and feat is not None:
                delta = float(_ridge_predict(info["model"], feat)[0])
                value = origin_level + delta
                sigma = info["val_rmse_m"]
                method = "ridge_ar_hourly"
            else:
                value = origin_level
                sigma = None
                method = "persistence"
            idx = o + h
            observed = float(self.y[idx]) if idx < len(self.y) and np.isfinite(self.y[idx]) else None
            if observed is not None:
                n_obs += 1
                err.append(value - observed)
                pers_err.append(origin_level - observed)
            steps.append({
                "horizon_hours": h, "target_time": target_time.isoformat(timespec="minutes"), "forecast_m": round(value, 3),
                "lower_1sigma_m": round(value - sigma, 3) if sigma is not None else None,
                "upper_1sigma_m": round(value + sigma, 3) if sigma is not None else None,
                "observed_m": round(observed, 3) if observed is not None else None, "method": method,
            })
        hist_lo = max(0, o - int(history_hours) + 1)
        history = [
            {"time": self.times[i].isoformat(timespec="minutes"), "water_level_m": round(float(self.y[i]), 3)}
            for i in range(hist_lo, o + 1) if np.isfinite(self.y[i])
        ]
        fitted = {h: self.models[h] for h in range(1, horizon_hours + 1) if h in self.models}
        key = max(fitted) if fitted else None
        verification = None
        if n_obs:
            e = np.array(err)
            p = np.array(pers_err)
            verification = {"n": n_obs, "rmse_m": round(_rmse(e), 4), "mae_m": round(float(np.mean(np.abs(e))), 4), "persistence_rmse_m": round(_rmse(p), 4)}
        return {
            "unit": "hour",
            "as_of": origin_time.isoformat(timespec="minutes"),
            "origin_level_m": round(origin_level, 3),
            "horizon_hours": horizon_hours,
            "steps": steps,
            "history": history,
            "verification": verification,
            "data_period": [self.times[0].isoformat(timespec="minutes"), self.times[-1].isoformat(timespec="minutes")],
            "n_hours": int(np.isfinite(self.y).sum()),
            "model": {
                "id": "ridge_ar_hourly", "name_ko": "시 단위 릿지 자기회귀",
                "features_ko": "최근 24시간 변화량 24개 · 24시간/7일 평균 대비 편차 · 시각·계절 조화항",
                "fit_scope_ko": "원점까지의 자료로 호출 시점에 학습(예측시간마다 모델 하나, 마지막 20 % 로 λ 선택)",
                "lambda": fitted[key]["lambda"] if key else None,
                "n_fit": fitted[key]["n_fit"] if key else None,
                "val_rmse_m_by_h": {str(h): fitted[h]["val_rmse_m"] for h in fitted},
                "val_persistence_rmse_m_by_h": {str(h): fitted[h]["val_persistence_rmse_m"] for h in fitted},
                "val_skill_by_h": {str(h): fitted[h]["val_skill"] for h in fitted},
            } if fitted else {"id": "persistence", "name_ko": "직전값 유지", "reason_ko": f"학습에 필요한 자료({MIN_HOURS_TO_FIT // 24}일)가 원점 앞에 모자라요"},
            "note_ko": "시 단위 전망은 지상 수위계(WAMIS 시 자료)로만 냅니다. 띠는 검증 구간 RMSE(±1σ)이고 보증 구간이 아닙니다.",
        }


# ── 홀드아웃 평가(화면 안내용 수치) ──────────────────────────────────────────────────────


def holdout_eval(doc: dict[str, Any], *, horizons: tuple[int, ...] = HORIZONS_HOURS, test_fraction: float = 0.2) -> dict[str, Any]:
    """자료 앞 80 % 로 학습·λ 선택, 뒤 20 % 를 시험 — 예측시간별 RMSE 와 직전값 유지 대비 개선. 합성값 없음."""

    fc = HourlyForecaster.from_doc(doc)
    n = len(fc.y)
    if n < MIN_HOURS_TO_FIT * 2:
        return {"status": "insufficient", "n_hours": n, "note_ko": f"평가에는 최소 {MIN_HOURS_TO_FIT * 2 // 24}일이 필요해요"}
    split = int(n * (1 - test_fraction))
    out: dict[str, Any] = {"status": "ok", "n_hours": n, "test_start": fc.times[split].isoformat(timespec="minutes"), "test_end": fc.times[-1].isoformat(timespec="minutes"), "by_horizon": {}}
    for h in horizons:
        info = fc.fit(h, split - 1)
        if info is None:
            continue
        t = np.arange(split, n - h)
        X = hourly_features(fc.y, fc.times, t)
        target = fc.y[t + h] - fc.y[t]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(target)
        if ok.sum() < 24:
            continue
        pred = _ridge_predict(info["model"], X[ok])
        e = pred - target[ok]
        p = -target[ok]
        out["by_horizon"][str(h)] = {
            "n_test": int(ok.sum()), "rmse_m": round(_rmse(e), 4), "mae_m": round(float(np.mean(np.abs(e))), 4),
            "persistence_rmse_m": round(_rmse(p), 4), "skill": round(1.0 - (_rmse(e) / _rmse(p)) ** 2, 3) if _rmse(p) > 0 else None, "lambda": info["lambda"],
        }
    return out


__all__ = [
    "HORIZONS_HOURS", "HOURLY_SCHEMA", "MAX_HORIZON_HOURS", "MIN_HOURS_TO_FIT", "HourlyForecaster", "holdout_eval", "hourly_path",
    "load_hourly", "merge_hourly", "parse_hourly_payload", "parse_ymdh", "save_hourly", "to_grid", "wamis_fetch_hourly",
]
