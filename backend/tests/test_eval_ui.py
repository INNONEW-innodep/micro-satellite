"""모델 평가 화면(/eval) 테스트.

페이지가 기대하는 API 계약 전부(지역·준비 상태·WAMIS 검색·수위/기상 수집 잡·예측기·학습 잡·평가표·모델 비교·
백테스트·실험·전망)를 **인메모리 스텁 라우터**로 구현해 두고 그 위에 ``mount_eval_ui`` 를 붙여 검사한다.
스텁은 상태를 가진다 — 새 지역을 만들고 수집 잡을 돌리면 지점이 생기고, 학습 잡이 끝나면 평가표가 생기며
준비 상태(readiness)의 "다음 할 일"이 줄어든다. 잡은 GET 폴링 한 번마다 한 칸씩 진행한다.
``scripts/capture_eval_ui.py`` 도 같은 스텁을 uvicorn 으로 띄워 ①→⑥ 을 자동으로 밟으며 스크린샷을 찍는다:

    .venv/bin/python -m uvicorn backend.tests.test_eval_ui:build_stub_app --factory --port 8766

환경변수
    EVAL_STUB_TICK   GET 한 번에 진행시킬 칸 수(기본 1)

수치는 전부 합성(seed 고정)이다 — 지점 이름으로 결정되는 계절 + AR(1) 수위 계열 위에서 모델 계열을
"평년값으로 되돌아가는 비율 + 잡음"으로 흉내 내고, 평가표의 RMSE·개선율은 그 계열에서 실제로 계산한다.

페이지가 이 스텁에서 읽어 가는 가정(백엔드 계약 — screens_api.py 가 맞춰야 할 모양):
    * GET  /regions                                  → {items:[RegionSummary]}  (실서버와 같은 필드)
    * POST /regions {id,name,description,bbox_wgs84} → 201 RegionSummary
    * GET  /regions/{id}/readiness                   → {gauge:{n_stations,period,n_days}, weather:{available,station,period,source},
                                                        models:{trained,trained_at,split}, satellite:{n_scenes,n_with_masks}, next_steps:[str]}
    * GET  /wamis/stations?basin=&match=             → {basin, items:[{code,name,river,address}]}
    * POST /regions/{id}/gauge/collect (async_run)   → 202 {job_id, status, poll_url}
      GET  …/gauge/collect/jobs/{job_id}             → {status, progress:{stations_total,stations_done,current_station},
                                                        stations:[{id,name,code,n_days,coverage,status,note}], error}
    * POST /regions/{id}/weather/collect (async_run) → 202 {job_id, status, poll_url}
      GET  …/weather/collect/jobs/{job_id}           → {status, progress:{years_total,years_done,current_year}, source:"kma"|"sample",
                                                        station_id, station_name, years:[{year,n_days,status,note}], error}
    * POST …/gauge/train (async_run)                 → 202 TrainJobAccepted; GET …/gauge/train/jobs/{id} → TrainJobStatus (실서버와 동일)
    * GET  …/gauge/evaluation, …/gauge/{st}/forecast/compare, …/gauge/{st}/backtest, …/gauge/experiments,
      …/gauge/{st}/forecast                          → 실서버 응답 필드를 그대로 따른다
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.app.eval_ui import INDEX_FILE, mount_eval_ui

# --------------------------------------------------------------------------- 고정 자료
REPORT_H = [1, 3, 5, 7, 10, 14, 30]
TERMINAL = {"done", "failed"}
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# (id, 이름, 평년값 회귀율 alpha, 잡음, built_in, is_baseline, is_reference, uses_future_exog)
FORECASTERS: list[tuple[str, str, float, float, bool, bool, bool, bool]] = [
    ("analog_knn", "유사사례 k-NN", 0.06, 0.030, True, False, False, False),
    ("climatology", "계절 평년값", 1.0, 0.0, True, False, True, False),
    ("ensemble_mean", "계열 앙상블 평균", 0.10, 0.006, True, False, False, False),
    ("mean_reversion", "평균 회귀 AR(1)", 0.12, 0.010, True, False, False, False),
    ("neural_gru", "GRU 순환신경망 (GPU 학습)", 0.11, 0.024, False, False, False, False),
    ("neural_lstm", "LSTM 순환신경망 (GPU 학습)", 0.10, 0.028, False, False, False, False),
    ("neural_mlp", "MLP 신경망 (GPU 학습)", 0.13, 0.016, False, False, False, False),
    ("neural_tcn", "TCN 시간합성곱 (GPU 학습)", 0.09, 0.034, False, False, False, False),
    ("persistence", "직전값 유지 (기준선)", 0.0, 0.0, True, True, True, False),
    ("prophet_hybrid", "Prophet + 잔차 자기회귀", 0.20, 0.040, False, False, False, False),
    ("ridge_ar", "릿지 자기회귀 (AR)", 0.11, 0.014, True, False, False, False),
    ("ridge_arx", "릿지 자기회귀 + 기상 (AR-X)", 0.12, 0.012, True, False, False, False),
    ("ridge_arx_future", "릿지 AR + 강수 예보 (완전 예보 가정)", 0.12, 0.008, True, False, False, True),
    ("seasonal_anomaly_ar", "계절편차 자기회귀", 0.10, 0.018, True, False, False, False),
]
MODEL_PARAMS = {f[0]: (f[2], f[3]) for f in FORECASTERS}
MODEL_NAME = {f[0]: f[1] for f in FORECASTERS}

WAMIS: dict[int, list[tuple[str, str, str, str]]] = {
    1: [("1018680", "한강대교", "한강", "서울특별시 용산구 이촌동"), ("1007625", "여주대교", "한강", "경기도 여주시 홍문동"),
        ("1006690", "양평교", "한강", "경기도 양평군 양평읍"), ("1016615", "팔당댐", "한강", "경기도 하남시 배알미동")],
    2: [("2022680", "구포대교", "낙동강", "부산광역시 북구 구포동"), ("2022696", "낙동강하구언(내)", "낙동강", "부산광역시 사하구 하단동"),
        ("2022640", "삼랑진교", "낙동강", "경상남도 밀양시 삼랑진읍"), ("2018695", "양산교", "양산천", "경상남도 양산시 물금읍"),
        ("2022615", "호포대교", "낙동강", "경상남도 양산시 동면"), ("2004640", "구담교", "낙동강", "경상북도 안동시 풍천면")],
    3: [("3008690", "대청댐", "금강", "대전광역시 대덕구 미호동"), ("3009680", "금남교", "금강", "세종특별자치시 금남면"),
        ("3011665", "금강교", "금강", "충청남도 공주시 금성동"), ("3010670", "미호강교", "미호강", "충청북도 청주시 흥덕구 옥산면"),
        ("3009655", "세종보", "금강", "세종특별자치시 연기면"), ("3012635", "백제보", "금강", "충청남도 부여군 부여읍")],
    4: [("4008690", "섬진강댐", "섬진강", "전라북도 임실군 강진면"), ("4009670", "구례교", "섬진강", "전라남도 구례군 구례읍"),
        ("4011655", "읍내리", "섬진강", "경상남도 하동군 하동읍"), ("4004675", "회문리", "섬진강", "전라북도 순창군 동계면")],
    5: [("5001670", "담양댐", "영산강", "전라남도 담양군 금성면"), ("5002680", "극락교", "영산강", "광주광역시 서구 치평동"),
        ("5004690", "나주대교", "영산강", "전라남도 나주시 삼영동"), ("5003655", "승촌보", "영산강", "광주광역시 남구 승촌동")],
}
BASIN_NAME = {1: "한강", 2: "낙동강", 3: "금강", 4: "섬진강", 5: "영산강"}
WEATHER_STATIONS = [
    {"id": "108", "name": "서울", "aliases": ["서울특별시", "한강"]}, {"id": "159", "name": "부산", "aliases": ["부산광역시", "낙동강 하구"]},
    {"id": "143", "name": "대구", "aliases": ["대구광역시", "금호강"]}, {"id": "156", "name": "광주", "aliases": ["광주광역시", "영산강"]},
    {"id": "133", "name": "대전", "aliases": ["대전광역시", "금강"]}, {"id": "152", "name": "울산", "aliases": ["울산광역시"]},
    {"id": "184", "name": "제주", "aliases": ["제주특별자치도"]},
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _d(s: str) -> date:
    return date.fromisoformat(s)


# --------------------------------------------------------------------------- 합성 수위 계열
@dataclass
class StationSeries:
    seed: str
    start: date
    values: list[float]
    base: float
    amp: float
    phase: float
    missing: set[int] = field(default_factory=set)
    _noise: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _clim: list[float] = field(default_factory=list, repr=False)

    @classmethod
    def make(cls, seed: str, start: date, end: date) -> "StationSeries":
        rng = random.Random("series:" + seed)
        base = 1.2 + rng.random() * 2.0
        amp = 0.25 + rng.random() * 0.5
        phase = rng.random() * 2 * math.pi
        n = (end - start).days + 1
        ss = cls(seed=seed, start=start, values=[], base=base, amp=amp, phase=phase)
        ss._clim = [ss.clim_on(start + timedelta(days=i)) for i in range(n + 120)]
        x = 0.0
        vals: list[float] = []
        for i in range(n):
            x = 0.86 * x + rng.gauss(0, 0.03)
            if rng.random() < 0.03:
                x += 0.15 + rng.random() * 0.5  # 강우 상승 뒤 감수
            vals.append(round(ss._clim[i] + x, 4))
        ss.values = vals
        # 결측: seed 로 정해지는 짧은 구간 몇 개
        gaps = rng.randint(0, 3)
        for _ in range(gaps):
            g0 = rng.randrange(0, max(1, n - 40))
            ss.missing.update(range(g0, min(n, g0 + rng.randint(3, 30))))
        return ss

    @property
    def end(self) -> date:
        return self.start + timedelta(days=len(self.values) - 1)

    @property
    def n_days(self) -> int:
        return len(self.values) - len(self.missing)

    def clim_on(self, d: date) -> float:
        return self.base + self.amp * math.sin(2 * math.pi * d.timetuple().tm_yday / 365.25 + self.phase)

    def clim(self, i: int) -> float:
        if i < len(self._clim):
            return self._clim[i]
        return self.clim_on(self.start + timedelta(days=i))

    def index(self, d: date) -> int:
        return (d - self.start).days

    def noise(self, model_id: str) -> list[float]:
        arr = self._noise.get(model_id)
        if arr is None:
            rng = random.Random(f"noise:{self.seed}:{model_id}")
            arr = [rng.gauss(0, 1) for _ in range(len(self.values))]
            self._noise[model_id] = arr
        return arr

    def forecast(self, model_id: str, t: int, h: int) -> float:
        y = self.values[t]
        if model_id == "persistence":
            return y
        c = self.clim(t + h)
        if model_id == "climatology":
            return c
        alpha, nz = MODEL_PARAMS[model_id]
        w = 1 - (1 - alpha) ** h
        arr = self.noise(model_id)
        return y + w * (c - y) + nz * math.sqrt(h) * arr[(t + 7919 * h) % len(arr)]

    def metrics(self, model_id: str, h: int, lo: date, hi: date) -> dict[str, float]:
        """목표일이 [lo, hi] 안인 모든 원점에 대한 오차 통계."""
        i0 = max(h, self.index(lo))
        i1 = min(len(self.values) - 1, self.index(hi))
        errs = [self.values[i] - self.forecast(model_id, i - h, h) for i in range(i0, i1 + 1)]
        n = len(errs)
        if n == 0:
            return {"n": 0, "mse": 0.0, "rmse": 0.0, "mae": 0.0, "bias": 0.0}
        mse = sum(e * e for e in errs) / n
        return {"n": n, "mse": mse, "rmse": math.sqrt(mse), "mae": sum(abs(e) for e in errs) / n, "bias": sum(errs) / n}

    def rows(self) -> list[dict[str, Any]]:
        return [{"date": (self.start + timedelta(days=i)).isoformat(), "water_level_m": v}
                for i, v in enumerate(self.values) if i not in self.missing]


# --------------------------------------------------------------------------- 지역·잡 상태
@dataclass
class StubStation:
    id: str
    name: str
    code: str | None
    terrain: str
    series: StationSeries
    source: str = "WAMIS"

    def to_dict(self, trained: bool) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "source": self.source, "source_code": self.code, "terrain": self.terrain,
            "terrain_note": None, "unit": "m", "n_days": self.series.n_days,
            "period": [self.series.start.isoformat(), self.series.end.isoformat()], "satellite_location": None,
            "model_trained": trained, "qc": None,
        }


@dataclass
class StubRegion:
    id: str
    name: str
    description: str = ""
    bbox: list[float] | None = None
    origin: str = "user"
    stations: dict[str, StubStation] = field(default_factory=dict)
    weather: dict[str, Any] | None = None
    weather_station: dict[str, str] | None = None
    split: dict[str, str] | None = None
    evaluation: dict[str, Any] | None = None
    trained_at: str | None = None
    forecasters_used: list[str] | None = None
    horizon_max: int = 30
    n_scenes: int = 0
    n_with_masks: int = 0
    experiments: dict[str, Any] | None = None

    @property
    def trained(self) -> bool:
        return self.evaluation is not None

    def default_split(self) -> dict[str, str] | None:
        if not self.stations:
            return None
        end_year = max(s.series.end.year for s in self.stations.values())
        return {"train_end": f"{end_year - 2}-12-31", "val_start": f"{end_year - 1}-01-01", "val_end": f"{end_year - 1}-12-31",
                "test_start": f"{end_year}-01-01", "test_end": f"{end_year}-12-31"}

    def period(self) -> list[str] | None:
        if not self.stations:
            return None
        return [min(s.series.start for s in self.stations.values()).isoformat(), max(s.series.end for s in self.stations.values()).isoformat()]

    def summary(self) -> dict[str, Any]:
        split = self.split or self.default_split()
        return {
            "id": self.id, "name": self.name, "description": self.description, "origin": self.origin, "crs": "EPSG:32652",
            "weather": {"provider": "KMA_ASOS", **self.weather_station} if self.weather_station else None,
            "satellite_available": self.n_scenes > 0,
            "satellite": {"available": self.n_scenes > 0, "sensors": ["ICEYE", "PlanetScope"] if self.n_scenes else [],
                          "scenes": {"ICEYE": [f"2026-09-{d:02d}" for d in range(1, 1 + self.n_scenes)]} if self.n_scenes else {}},
            "gauge_available": bool(self.stations),
            "stations": [s.to_dict(self.trained) for s in self.stations.values()],
            "n_stations": len(self.stations), "n_stations_trained": len(self.stations) if self.trained else 0,
            "forecast": {"horizon_max": self.horizon_max, "horizons_report": REPORT_H, "split": split, "split_active": split},
            "bbox_wgs84": self.bbox,
            "weather_daily_available": self.weather is not None,
            "timeseries_eval_available": self.trained, "scenes_available": self.n_scenes > 0,
        }

    def readiness(self) -> dict[str, Any]:
        next_steps: list[str] = []
        if not self.stations:
            next_steps.append("② 유역에서 수위 관측소를 골라 자료를 수집하세요 (2개 이상 권장)")
        if self.weather is None:
            next_steps.append("③ 기상청 ASOS 지점을 골라 기상 자료를 수집하세요")
        if self.stations and not self.trained:
            next_steps.append("④ 학습·검증·시험 연도를 정하고 학습을 실행하세요")
        if self.trained:
            if self.n_scenes == 0:
                next_steps.append("(선택) 서비스 화면에서 위성 장면을 올리면 수체 면적도 볼 수 있어요")
        return {
            "region_id": self.id,
            "gauge": {"n_stations": len(self.stations), "period": self.period(), "n_days": sum(s.series.n_days for s in self.stations.values())},
            "weather": {"available": self.weather is not None, "station": (self.weather or {}).get("station_name"),
                        "period": (self.weather or {}).get("period"), "source": (self.weather or {}).get("source")},
            "models": {"trained": self.trained, "trained_at": self.trained_at, "split": self.split if self.trained else None},
            "satellite": {"n_scenes": self.n_scenes, "n_with_masks": self.n_with_masks},
            "next_steps": next_steps,
        }

    # ---- 평가
    def candidates(self, forecasters: list[str] | None) -> list[str]:
        if forecasters:
            return [f for f in forecasters if f != "persistence"]
        return [f[0] for f in FORECASTERS if not f[5] and not f[7]]

    def build_evaluation(self, split: dict[str, str], forecasters: list[str] | None, horizon_max: int) -> dict[str, Any]:
        cands = self.candidates(forecasters)
        rows: list[dict[str, Any]] = []
        val = (_d(split["val_start"]), _d(split["val_end"]))
        test = (_d(split["test_start"]), _d(split["test_end"]))
        for st in self.stations.values():
            ss = st.series
            for h in [x for x in REPORT_H if x <= horizon_max]:
                scored = [(ss.metrics(m, h, *val)["rmse"], m) for m in cands]
                scored.sort()
                val_rmse, best = scored[0]
                tm = ss.metrics(best, h, *test)
                pm = ss.metrics("persistence", h, *test)
                cm = ss.metrics("climatology", h, *test)
                rows.append({
                    "station_id": st.id, "station_name": st.name, "terrain": st.terrain, "horizon": h,
                    "forecaster_id": best, "forecaster_name": MODEL_NAME[best], "hparams": {},
                    "val_rmse": val_rmse, "test_n": tm["n"], "test_mse": tm["mse"], "test_rmse": tm["rmse"], "test_mae": tm["mae"],
                    "persistence_test_rmse": pm["rmse"], "persistence_test_mse": pm["mse"], "climatology_test_rmse": cm["rmse"],
                    "skill_vs_persistence": 1 - tm["rmse"] / pm["rmse"] if pm["rmse"] else 0.0,
                    "regime_shift_flag": False, "uses_weather": best in {"ridge_arx", "neural_mlp", "neural_gru", "neural_lstm", "neural_tcn"},
                })
        return {
            "schema": "watercast.gauge.evaluation/1", "region_id": self.id, "generated_at": _now(), "split": dict(split),
            "n_stations": len(self.stations), "horizon_max": horizon_max, "forecasters": cands, "rows": rows,
        }

    def selected_model(self, station_id: str, h: int) -> str:
        if not self.evaluation:
            return "mean_reversion"
        best: tuple[int, str] | None = None
        for r in self.evaluation["rows"]:
            if r["station_id"] != station_id:
                continue
            if r["horizon"] <= h and (best is None or r["horizon"] > best[0]):
                best = (r["horizon"], r["forecaster_id"])
        return best[1] if best else "mean_reversion"

    def report_horizon(self, h: int) -> int:
        hs = [x for x in REPORT_H if x <= max(h, 1)]
        return hs[-1] if hs else 1


@dataclass
class CollectJob:
    job_id: str
    region_id: str
    stations: list[dict[str, Any]]
    start: date
    end: date
    min_coverage: float
    status: str = "queued"
    queued_at: str = field(default_factory=_now)
    finished_at: str | None = None
    error: str | None = None

    def advance(self, region: StubRegion) -> None:
        if self.status in TERMINAL:
            return
        self.status = "running"
        for s in self.stations:
            if s["status"] == "running":
                self._finish(s, region)
                continue
            if s["status"] == "pending":
                s["status"] = "running"
                return
        if all(s["status"] != "running" for s in self.stations):
            self.status = "done"
            self.finished_at = _now()

    def _finish(self, s: dict[str, Any], region: StubRegion) -> None:
        code = str(s.get("code") or "")
        if code == "0000000":
            s.update(status="failed", note="WAMIS 응답 없음 (스텁 오류)")
            return
        total = (self.end - self.start).days + 1
        series = StationSeries.make(f"{region.id}:{s['id']}", self.start, self.end)
        if code.endswith("55"):  # 결측 많은 관측소 흉내 (세종보·승촌보)
            drop = int(total * 0.32)
            series.missing.update(range(0, drop))
        cov = series.n_days / total
        s["n_days"] = series.n_days
        s["coverage"] = round(cov, 4)
        region.stations[s["id"]] = StubStation(s["id"], s.get("name") or s["id"], code or None, s.get("terrain") or "river", series)
        region.evaluation = None  # 자료가 바뀌면 다시 학습해야 한다
        s["period"] = [series.start.isoformat(), series.end.isoformat()]
        if cov < self.min_coverage:  # 실서버처럼 병합은 하되 low_coverage 로 표시한다
            s.update(status="low_coverage", note=f"커버리지 {cov:.1%} 가 기준 {self.min_coverage:.0%} 미만 — 병합은 했으나 학습 전 자료를 확인하세요")
            return
        s.update(status="done", note="저장됨" + (f" · 결측 {len(series.missing)}일 보간 없이 제외" if series.missing else ""))

    def to_dict(self) -> dict[str, Any]:
        done = sum(1 for s in self.stations if s["status"] in {"done", "failed", "skipped"})
        cur = next((s["id"] for s in self.stations if s["status"] == "running"), None)
        return {
            "job_id": self.job_id, "region_id": self.region_id, "status": self.status,
            "progress": {"stations_total": len(self.stations), "stations_done": done, "current_station": cur},
            "stations": [dict(s) for s in self.stations], "start": self.start.isoformat(), "end": self.end.isoformat(),
            "min_coverage": self.min_coverage, "queued_at": self.queued_at, "finished_at": self.finished_at, "error": self.error,
        }


@dataclass
class WeatherJob:
    job_id: str
    region_id: str
    station_id: str
    station_name: str
    years: list[dict[str, Any]]
    start: date
    end: date
    source: str = "sample"
    status: str = "queued"
    queued_at: str = field(default_factory=_now)
    finished_at: str | None = None
    error: str | None = None

    def advance(self, region: StubRegion) -> None:
        if self.status in TERMINAL:
            return
        self.status = "running"
        for y in self.years:
            if y["status"] == "running":
                yr = int(y["year"])
                y0 = max(self.start, date(yr, 1, 1))
                y1 = min(self.end, date(yr, 12, 31))
                n = (y1 - y0).days + 1
                y.update(status="done", n_days=n, n_rows=n, note="표본 자료" if self.source == "sample" else "기상청 ASOS")
                continue
            if y["status"] == "pending":
                y["status"] = "running"
                return
        self.status = "done"
        self.finished_at = _now()
        region.weather = {"station_id": self.station_id, "station_name": self.station_name, "period": [self.start.isoformat(), self.end.isoformat()], "source": self.source}
        region.weather_station = {"station_id": self.station_id, "station_name": self.station_name}

    def to_dict(self) -> dict[str, Any]:
        done = sum(1 for y in self.years if y["status"] in {"done", "failed", "skipped"})
        cur = next((y["year"] for y in self.years if y["status"] == "running"), None)
        return {
            "job_id": self.job_id, "region_id": self.region_id, "status": self.status, "source": self.source,
            "station_id": self.station_id, "station_name": self.station_name,
            "progress": {"years_total": len(self.years), "years_done": done, "current_year": cur},
            "years": [dict(y) for y in self.years], "queued_at": self.queued_at, "finished_at": self.finished_at, "error": self.error,
            "warnings": ["기상청 API 키가 없어 표본(합성) 기상 자료를 썼어요 — 실측이 아닙니다."] if self.source == "sample" else [],
        }


@dataclass
class TrainJob:
    job_id: str
    region_id: str
    split: dict[str, str]
    custom_split: bool
    stations: list[str]
    forecasters: list[str] | None
    horizon_max: int
    status: str = "queued"
    done: int = 0
    current: str | None = None
    queued_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    evaluation: dict[str, Any] | None = None

    def advance(self, region: StubRegion) -> None:
        if self.status in TERMINAL:
            return
        if self.status == "queued":
            self.status = "running"
            self.started_at = _now()
        if self.current is not None:
            self.done += 1
            self.current = None
        if self.done < len(self.stations):
            self.current = self.stations[self.done]
            return
        self.evaluation = region.build_evaluation(self.split, self.forecasters, self.horizon_max)
        region.evaluation = self.evaluation
        region.split = dict(self.split)
        region.forecasters_used = self.forecasters
        region.horizon_max = self.horizon_max
        region.trained_at = _now()
        self.status = "done"
        self.finished_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "region_id": self.region_id, "status": self.status,
            "progress": {"stations_total": len(self.stations), "stations_done": self.done, "current_station": self.current},
            "queued_at": self.queued_at, "started_at": self.started_at, "finished_at": self.finished_at,
            "elapsed_s": 4.2 * self.done + (2.0 if self.current else 0.0), "split_used": dict(self.split), "custom_split": self.custom_split,
            "stations": list(self.stations), "forecasters": self.forecasters, "horizon_max": self.horizon_max,
            "error": self.error, "evaluation": self.evaluation if self.status == "done" else None,
        }


class StubState:
    def __init__(self, tick: int = 1) -> None:
        self.tick = max(1, int(tick))
        self.regions: dict[str, StubRegion] = {}
        self.collect_jobs: dict[str, CollectJob] = {}
        self.weather_jobs: dict[str, WeatherJob] = {}
        self.train_jobs: dict[str, TrainJob] = {}

    def seed(self) -> None:
        """학습까지 끝난 지역 하나(위성 장면·실험 포함)와 자료만 있는 지역 하나."""
        busan = StubRegion("nakdong_busan", "낙동강 하구 (부산·김해)", "부산 낙동강 하구 6개 지점 · ICEYE/PlanetScope 장면 보유", [128.85, 35.05, 129.05, 35.35], origin="builtin")
        for code, name, river, _addr in WAMIS[2][:5]:
            terrain = "estuary" if name in {"구포대교", "낙동강하구언(내)"} else "tributary" if river != "낙동강" else "mainstem"
            busan.stations[name] = StubStation(name, name, code, terrain, StationSeries.make(f"nakdong_busan:{name}", date(2015, 1, 1), date(2024, 12, 31)))
        busan.weather = {"station_id": "159", "station_name": "부산", "period": ["2015-01-01", "2024-12-31"], "source": "asos_stored"}
        busan.weather_station = {"station_id": "159", "station_name": "부산"}
        busan.split = {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}
        busan.n_scenes, busan.n_with_masks = 23, 19
        busan.experiments = _area_models_experiment()
        self.regions[busan.id] = busan

        geum = StubRegion("geum_mainstem", "금강 본류 (세종·공주·청주)", "수위 자료만 있고 아직 학습 전", [127.1, 36.3, 127.5, 36.7], origin="builtin")
        for code, name, river, _addr in WAMIS[3][1:4]:
            geum.stations[name] = StubStation(name, name, code, "tributary" if river != "금강" else "mainstem", StationSeries.make(f"geum_mainstem:{name}", date(2015, 1, 1), date(2024, 12, 31)))
        geum.weather_station = {"station_id": "133", "station_name": "대전"}
        self.regions[geum.id] = geum

    def ensure_evaluation(self, region: StubRegion) -> None:
        """시드 지역은 평가표를 처음 요청할 때 만든다 (학습 끝난 상태로 간주)."""
        if region.evaluation is None and region.origin == "builtin" and region.split and region.id == "nakdong_busan":
            region.evaluation = region.build_evaluation(region.split, None, 30)
            region.trained_at = "2026-09-08T17:01:58Z"

    def active_for(self, region_id: str) -> bool:
        return any(j.region_id == region_id and j.status not in TERMINAL for j in list(self.collect_jobs.values()) + list(self.weather_jobs.values()) + list(self.train_jobs.values()))


def _area_models_experiment() -> dict[str, Any]:
    cands = {
        "persistence": ("직전 관측 유지 (기준선)", 2.8119, 0.0, None),
        "series_mean": ("계열 평균", 2.0599, 0.4633, 0.625),
        "linear_trend": ("시간 선형추세", 3.6298, -0.6664, 0.625),
        "doy_climatology": ("연중 시기 평년값 (day-of-year)", 2.1534, 0.4135, 0.625),
        "ar1_mean_reversion": ("AR(1) 평균회귀", 2.0599, 0.4633, 0.625),
        "level_isotonic:구포대교": ("같은 날 수위 단조회귀 (PAVA)", 2.5066, 0.2053, 0.625),
    }
    def pack(n: int, scale: float) -> dict[str, Any]:
        return {cid: {"family": cid.split(":")[0], "name_ko": nm, "level_station": cid.split(":")[1] if ":" in cid else None, "n": n,
                      "rmse_km2": round(r * scale, 4), "mae_km2": round(r * scale * 0.86, 4), "skill_vs_persistence": sk,
                      "sign_test_vs_persistence": None if p is None else {"wins": 3, "losses": 1, "ties": 0, "p_two_sided": p, "min_attainable_p": 0.125}}
                for cid, (nm, r, sk, p) in cands.items()}
    return {
        "region_id": "nakdong_busan",
        "items": {
            "future_rain": None,
            "area_models": {
                "schema": "watercast.satellite.experiment.area-models/1", "region_id": "nakdong_busan", "generated_at": "2026-09-09T10:00:00Z",
                "series": {"ICEYE": {"n": 4, "points": [{}] * 4, "candidates": pack(4, 1.0)}, "PlanetScope": {"n": 7, "points": [{}] * 7, "candidates": pack(7, 0.72)}},
                "bridge": {"any_defensible": False},
                "notes_ko": ["표본이 4~10점이라 순위가 우연히 뒤집힐 수 있다."],
                "verdict_ko": "두 센서 모두 어떤 후보도 직전값 유지보다 유의하게 낫지 않다 (최소 p 0.125).",
            },
        },
        "sources": {"future_rain": None, "area_models": "builtin"},
        "note_ko": "실험 결과는 공식 평가표와 별개다.",
    }


# --------------------------------------------------------------------------- 라우터
class RegionCreate(BaseModel):
    id: str
    name: str
    description: str = ""
    bbox_wgs84: list[float] | None = None
    overwrite: bool = False


class CollectStation(BaseModel):
    id: str
    name: str | None = None
    code: str | None = None
    terrain: str | None = "river"


class CollectRequest(BaseModel):
    stations: list[CollectStation]
    start: str
    end: str
    min_coverage: float = 0.85
    async_run: bool = True


class WeatherCollectRequest(BaseModel):
    station_id: str
    start: str
    end: str
    async_run: bool = True


class TrainRequest(BaseModel):
    stations: list[str] | None = None
    forecasters: list[str] | None = None
    horizon_max: int | None = None
    split: dict[str, str] | None = None
    async_run: bool = False


def _parse_series_rows(raw: bytes) -> list[tuple[date, float]]:
    text = raw.decode("utf-8-sig", errors="replace")
    rows: list[tuple[date, float]] = []
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        data = json.loads(stripped)
        items = data if isinstance(data, list) else data.get("rows", [])
        for it in items:
            rows.append((_d(str(it.get("date") or it.get("obs_date"))), float(it.get("water_level_m", it.get("level")))))
        return rows
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        dkey = next((k for k in ("date", "obs_date", "ymd") if k in r), None)
        vkey = next((k for k in ("water_level_m", "level", "wl", "value") if k in r), None)
        if not dkey or not vkey:
            raise ValueError("열 이름은 date 와 water_level_m 이어야 해요")
        if not r[dkey] or r[vkey] in (None, ""):
            continue
        rows.append((_d(r[dkey].strip()), float(r[vkey])))
    return rows


def build_stub_router(state: StubState) -> APIRouter:
    router = APIRouter()

    def region_or_404(region_id: str) -> StubRegion:
        region = state.regions.get(region_id)
        if region is None:
            raise HTTPException(404, detail=f"지역을 찾을 수 없어요: {region_id}")
        return region

    def station_or_404(region: StubRegion, station_id: str) -> StubStation:
        st = region.stations.get(station_id)
        if st is None:
            raise HTTPException(404, detail=f"지점을 찾을 수 없어요: {station_id}")
        return st

    def split_or_422(region: StubRegion, split: dict[str, str] | None) -> tuple[dict[str, str], bool]:
        if not split:
            default = region.default_split()
            if default is None:
                raise HTTPException(422, detail="수위 자료가 있는 지점이 없어요")
            return default, False
        keys = ("train_end", "val_start", "val_end", "test_start", "test_end")
        if any(k not in split or not DATE_RE.match(str(split[k])) for k in keys):
            raise HTTPException(422, detail="split 은 train_end/val_start/val_end/test_start/test_end (YYYY-MM-DD) 전부 필요해요")
        d = [_d(split[k]) for k in keys]
        if not (d[0] < d[1] <= d[2] < d[3] <= d[4]):
            raise HTTPException(422, detail="분할 순서가 어긋났어요: train_end < val_start ≤ val_end < test_start ≤ test_end")
        period = region.period()
        if period and (d[4] > _d(period[1]) or d[0] < _d(period[0]) + timedelta(days=730)):
            raise HTTPException(422, detail=f"분할이 지역 시계열 범위({period[0]} ~ {period[1]})를 벗어나거나 학습 구간이 730일 미만이에요")
        return {k: split[k] for k in keys}, True

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "WATERCAST eval-ui stub", "version": "stub-eval-0.1"}

    # ---- 지역
    @router.get("/regions")
    async def list_regions() -> dict[str, Any]:
        return {"items": [r.summary() for r in state.regions.values()]}

    @router.post("/regions", status_code=201)
    async def create_region(body: RegionCreate) -> dict[str, Any]:
        if not ID_RE.match(body.id):
            raise HTTPException(422, detail="id 는 영문·숫자·'_-.' 1~64자여야 해요")
        if not body.name.strip():
            raise HTTPException(422, detail="name 은 비울 수 없어요")
        if body.id in state.regions and not body.overwrite:
            raise HTTPException(422, detail=f"이미 있는 지역이에요: {body.id} (overwrite=true 로 덮어쓸 수 있어요)")
        if body.bbox_wgs84 is not None and (len(body.bbox_wgs84) != 4 or not (body.bbox_wgs84[0] < body.bbox_wgs84[2] and body.bbox_wgs84[1] < body.bbox_wgs84[3])):
            raise HTTPException(422, detail="bbox_wgs84 는 [minlon, minlat, maxlon, maxlat] 네 숫자예요")
        region = StubRegion(body.id, body.name.strip(), body.description, body.bbox_wgs84)
        state.regions[body.id] = region
        return region.summary()

    @router.get("/regions/{region_id}")
    async def get_region(region_id: str) -> dict[str, Any]:
        return region_or_404(region_id).summary()

    @router.get("/regions/{region_id}/readiness")
    async def readiness(region_id: str) -> dict[str, Any]:
        region = region_or_404(region_id)
        state.ensure_evaluation(region)
        return region.readiness()

    # ---- WAMIS 검색
    @router.get("/wamis/stations")
    async def wamis_stations(basin: int = 1, match: str = "") -> dict[str, Any]:
        if basin not in WAMIS:
            raise HTTPException(422, detail="basin 은 1(한강) 2(낙동강) 3(금강) 4(섬진강) 5(영산강) 중 하나예요")
        q = match.strip()
        items = [{"code": c, "name": n, "river": r, "address": a} for c, n, r, a in WAMIS[basin] if not q or q in n or q in r or q in a]
        if q and not items and q.isdigit():
            items = [{"code": c, "name": n, "river": r, "address": a} for c, n, r, a in WAMIS[basin] if c.startswith(q)]
        return {"basin": basin, "basin_name": BASIN_NAME[basin], "items": items}

    # ---- 수위 수집
    @router.post("/regions/{region_id}/gauge/collect", status_code=202)
    async def gauge_collect(region_id: str, body: CollectRequest) -> dict[str, Any]:
        region = region_or_404(region_id)
        if not body.stations:
            raise HTTPException(422, detail="stations 가 비어 있어요")
        if not (DATE_RE.match(body.start) and DATE_RE.match(body.end)) or body.start >= body.end:
            raise HTTPException(422, detail="start < end (YYYY-MM-DD) 이어야 해요")
        if state.active_for(region_id):
            raise HTTPException(409, detail="이 지역에 진행 중인 작업이 있어요")
        job = CollectJob("collect-" + uuid.uuid4().hex[:8], region_id,
                         [{"id": s.id, "name": s.name or s.id, "code": s.code, "terrain": s.terrain or "river", "n_days": None, "coverage": None, "status": "pending", "note": None} for s in body.stations],
                         _d(body.start), _d(body.end), body.min_coverage)
        state.collect_jobs[job.job_id] = job
        if not body.async_run:
            while job.status not in TERMINAL:
                job.advance(region)
            return JSONResponse(job.to_dict(), status_code=200)
        return {"job_id": job.job_id, "status": job.status, "region_id": region_id, "poll_url": f"/api/v1/regions/{region_id}/gauge/collect/jobs/{job.job_id}"}

    @router.get("/regions/{region_id}/gauge/collect/jobs/{job_id}")
    async def gauge_collect_job(region_id: str, job_id: str) -> dict[str, Any]:
        region = region_or_404(region_id)
        job = state.collect_jobs.get(job_id)
        if job is None or job.region_id != region_id:
            raise HTTPException(404, detail="수집 작업을 찾을 수 없어요")
        for _ in range(state.tick):
            job.advance(region)
        return job.to_dict()

    @router.post("/regions/{region_id}/gauge/{station_id}/series/file")
    async def gauge_series_upload_file(
        region_id: str, station_id: str, file: UploadFile = File(...), replace: bool = Form(False),
        station_name: str | None = Form(None), terrain: str | None = Form(None), source: str | None = Form(None),
    ) -> dict[str, Any]:
        region = region_or_404(region_id)
        raw = await file.read()
        try:
            rows = _parse_series_rows(raw)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(422, detail=f"파일을 읽지 못했어요: {exc}") from exc
        if len(rows) < 2:
            raise HTTPException(422, detail="행이 2개 이상 필요해요")
        rows.sort()
        start, end = rows[0][0], rows[-1][0]
        series = StationSeries.make(f"{region_id}:{station_id}:upload", start, end)
        by_date = {d: v for d, v in rows}
        series.values = [by_date.get(start + timedelta(days=i), series.values[i]) for i in range(len(series.values))]
        series.missing = {i for i in range(len(series.values)) if (start + timedelta(days=i)) not in by_date}
        st = StubStation(station_id, station_name or station_id, None, terrain or "river", series, source=source or "user_upload")
        region.stations[station_id] = st
        region.evaluation = None
        return st.to_dict(False)

    # ---- 기상 수집
    @router.get("/weather/stations")
    async def weather_stations() -> dict[str, Any]:
        return {"items": WEATHER_STATIONS}

    @router.post("/regions/{region_id}/weather/collect", status_code=202)
    async def weather_collect(region_id: str, body: WeatherCollectRequest) -> dict[str, Any]:
        region = region_or_404(region_id)
        st = next((s for s in WEATHER_STATIONS if s["id"] == body.station_id), None)
        if st is None:
            raise HTTPException(422, detail=f"알 수 없는 ASOS 지점: {body.station_id}")
        if not (DATE_RE.match(body.start) and DATE_RE.match(body.end)) or body.start >= body.end:
            raise HTTPException(422, detail="start < end (YYYY-MM-DD) 이어야 해요")
        if state.active_for(region_id):
            raise HTTPException(409, detail="이 지역에 진행 중인 작업이 있어요")
        y0, y1 = _d(body.start), _d(body.end)
        years = [{"year": y, "n_days": None, "status": "pending", "note": None} for y in range(y0.year, y1.year + 1)]
        job = WeatherJob("wx-" + uuid.uuid4().hex[:8], region_id, st["id"], st["name"], years, y0, y1, source="kma" if os.environ.get("KMA_API_KEY") else "sample")
        state.weather_jobs[job.job_id] = job
        if not body.async_run:
            while job.status not in TERMINAL:
                job.advance(region)
            return JSONResponse(job.to_dict(), status_code=200)
        return {"job_id": job.job_id, "status": job.status, "region_id": region_id, "poll_url": f"/api/v1/regions/{region_id}/weather/collect/jobs/{job.job_id}"}

    @router.get("/regions/{region_id}/weather/collect/jobs/{job_id}")
    async def weather_collect_job(region_id: str, job_id: str) -> dict[str, Any]:
        region = region_or_404(region_id)
        job = state.weather_jobs.get(job_id)
        if job is None or job.region_id != region_id:
            raise HTTPException(404, detail="기상 수집 작업을 찾을 수 없어요")
        for _ in range(state.tick):
            job.advance(region)
        return job.to_dict()

    # ---- 예측기·학습
    @router.get("/gauge/forecasters")
    async def forecasters() -> list[dict[str, Any]]:
        return [{"id": f[0], "name": f[1], "description": f"{f[1]} — 스텁 설명", "is_baseline": f[5], "is_reference": f[6],
                 "hyperparameter_grid": [{}], "min_history_days": 365, "built_in": f[4], "uses_future_exog": f[7]} for f in FORECASTERS]

    @router.post("/regions/{region_id}/gauge/train")
    async def gauge_train(region_id: str, body: TrainRequest) -> JSONResponse:
        region = region_or_404(region_id)
        if not region.stations:
            raise HTTPException(422, detail="학습할 지점이 없어요 — 먼저 수위 자료를 수집하세요")
        stations = body.stations or list(region.stations)
        unknown = [s for s in stations if s not in region.stations]
        if unknown:
            raise HTTPException(422, detail=f"알 수 없는 지점: {', '.join(unknown)}")
        if body.forecasters is not None:
            bad = [f for f in body.forecasters if f not in MODEL_PARAMS]
            if bad:
                raise HTTPException(422, detail=f"알 수 없는 예측기: {', '.join(bad)}")
            if not [f for f in body.forecasters if f != "persistence"]:
                raise HTTPException(422, detail="persistence 는 기준선이라 후보로만 넣을 수 없어요")
        split, custom = split_or_422(region, body.split)
        if state.active_for(region_id):
            raise HTTPException(409, detail="이 지역에 진행 중인 학습이 있어요")
        job = TrainJob("train-" + uuid.uuid4().hex[:8], region_id, split, custom, stations, body.forecasters, body.horizon_max or 30)
        state.train_jobs[job.job_id] = job
        if not body.async_run:
            while job.status not in TERMINAL:
                job.advance(region)
            return JSONResponse(job.evaluation, status_code=200)
        return JSONResponse({"job_id": job.job_id, "status": "queued", "region_id": region_id, "split_used": split, "stations": stations,
                             "poll_url": f"/api/v1/regions/{region_id}/gauge/train/jobs/{job.job_id}"}, status_code=202)

    @router.get("/regions/{region_id}/gauge/train/jobs")
    async def gauge_train_jobs(region_id: str) -> dict[str, Any]:
        region_or_404(region_id)
        return {"items": [j.to_dict() | {"evaluation": None} for j in state.train_jobs.values() if j.region_id == region_id]}

    @router.get("/regions/{region_id}/gauge/train/jobs/{job_id}")
    async def gauge_train_job(region_id: str, job_id: str) -> dict[str, Any]:
        region = region_or_404(region_id)
        job = state.train_jobs.get(job_id)
        if job is None or job.region_id != region_id:
            raise HTTPException(404, detail="학습 작업을 찾을 수 없어요")
        for _ in range(state.tick):
            job.advance(region)
        return job.to_dict()

    # ---- 평가·비교·백테스트·실험·전망
    @router.get("/regions/{region_id}/gauge/evaluation")
    async def gauge_evaluation(region_id: str) -> dict[str, Any]:
        region = region_or_404(region_id)
        state.ensure_evaluation(region)
        if region.evaluation is None:
            raise HTTPException(404, detail="아직 평가표가 없어요 — 먼저 학습하세요")
        return region.evaluation

    @router.get("/regions/{region_id}/gauge/experiments")
    async def gauge_experiments(region_id: str) -> dict[str, Any]:
        region = region_or_404(region_id)
        if region.experiments:
            return region.experiments
        return {"region_id": region_id, "items": {"future_rain": None, "area_models": None}, "sources": {"future_rain": None, "area_models": None}, "note_ko": "실험 결과가 없어요"}

    def _steps(region: StubRegion, st: StubStation, as_of: date, horizon: int, override: str | None) -> list[dict[str, Any]]:
        ss = st.series
        t = ss.index(as_of)
        val = (_d(region.split["val_start"]), _d(region.split["val_end"])) if region.split else (ss.start, ss.end)
        steps = []
        for h in range(1, horizon + 1):
            model = override or region.selected_model(st.id, h)
            f = ss.forecast(model, t, h)
            sigma = max(0.02, ss.metrics(model, region.report_horizon(h), *val)["rmse"])
            ti = t + h
            obs = ss.values[ti] if 0 <= ti < len(ss.values) and ti not in ss.missing else None
            steps.append({
                "horizon": h, "target_date": (as_of + timedelta(days=h)).isoformat(), "forecast_m": f, "forecaster_id": model,
                "persistence_reference_m": ss.values[t], "sigma_m": sigma,
                "lower_1sigma_m": f - sigma, "upper_1sigma_m": f + sigma, "lower_2sigma_m": f - 2 * sigma, "upper_2sigma_m": f + 2 * sigma,
                "q05_m": f - 1.645 * sigma, "q95_m": f + 1.645 * sigma,
                "observed_m": obs, "error_m": (obs - f) if obs is not None else None,
            })
        return steps

    def _history(ss: StationSeries, as_of: date, days: int) -> list[dict[str, float | str]]:
        t = ss.index(as_of)
        return [{"date": (ss.start + timedelta(days=i)).isoformat(), "water_level_m": ss.values[i]} for i in range(max(0, t - days + 1), t + 1) if i not in ss.missing]

    def _as_of(ss: StationSeries, as_of: str | None) -> date:
        if as_of is None:
            return ss.end
        if not DATE_RE.match(as_of):
            raise HTTPException(422, detail="as_of 는 YYYY-MM-DD 형식이어야 해요")
        d = _d(as_of)
        if d < ss.start + timedelta(days=400) or d > ss.end:
            raise HTTPException(422, detail=f"as_of 는 {(ss.start + timedelta(days=400)).isoformat()} ~ {ss.end.isoformat()} 안이어야 해요")
        return d

    @router.get("/regions/{region_id}/gauge/{station_id}/forecast/compare")
    async def forecast_compare(region_id: str, station_id: str, horizon: int = 7, as_of: str | None = None) -> dict[str, Any]:
        region = region_or_404(region_id)
        st = station_or_404(region, station_id)
        state.ensure_evaluation(region)
        if region.evaluation is None:
            raise HTTPException(409, detail="먼저 학습하세요")
        ss = st.series
        origin = _as_of(ss, as_of)
        t = ss.index(origin)
        models = [f[0] for f in FORECASTERS if not f[5]]
        selected = _steps(region, st, origin, horizon, None)
        table = []
        for s in selected:
            h = s["horizon"]
            table.append({
                "horizon": h, "target_date": s["target_date"], "selected_model": s["forecaster_id"], "selected_m": s["forecast_m"],
                "persistence_m": ss.values[t], "observed_m": s["observed_m"],
                "models": {m: {"forecast_m": ss.forecast(m, t, h), "diff_vs_selected_m": ss.forecast(m, t, h) - s["forecast_m"]} for m in models},
            })
        rh = region.report_horizon(horizon)
        test = (_d(region.split["test_start"]), _d(region.split["test_end"]))
        pers = ss.metrics("persistence", rh, *test)
        sel_at_h = region.selected_model(st.id, horizon)
        per_model = {}
        for m in models:
            tm = ss.metrics(m, rh, *test)
            per_model[m] = {"verification": None, "test_at_horizon": {"horizon": rh, **tm, "skill_vs_persistence": 1 - tm["rmse"] / pers["rmse"] if pers["rmse"] else 0.0, "selected": m == sel_at_h}}
        return {
            "region_id": region_id, "station": st.to_dict(True), "as_of": origin.isoformat(), "horizon": horizon,
            "in_sample": origin <= _d(region.split["val_end"]), "params_note_ko": "전체 자료로 적합한 배포 파라미터(params_deploy)로 전망 (스텁)",
            "origin_level_m": ss.values[t], "history": _history(ss, origin, 90), "selected": selected, "models": models, "table": table,
            "per_model": per_model, "note_ko": "모든 계열은 검증 최상 하이퍼파라미터로 같은 규칙으로 적합된 파라미터를 쓴다. 선택 모델은 검증 연도별 폴드 성적이 가장 좋은 것.",
        }

    @router.get("/regions/{region_id}/gauge/{station_id}/backtest")
    async def backtest(region_id: str, station_id: str, horizon: int = 7) -> dict[str, Any]:
        region = region_or_404(region_id)
        st = station_or_404(region, station_id)
        state.ensure_evaluation(region)
        if region.evaluation is None:
            raise HTTPException(409, detail="먼저 학습하세요")
        if horizon not in REPORT_H:
            raise HTTPException(422, detail=f"보고 지평만 저장돼 있어요: {REPORT_H}")
        ss = st.series
        model = region.selected_model(st.id, horizon)
        t0, t1 = _d(region.split["test_start"]), _d(region.split["test_end"])
        i0, i1 = max(horizon, ss.index(t0)), min(len(ss.values) - 1, ss.index(t1))
        dates = [(ss.start + timedelta(days=i)).isoformat() for i in range(i0, i1 + 1)]
        truth = [ss.values[i] for i in range(i0, i1 + 1)]
        pred = [ss.forecast(model, i - horizon, horizon) for i in range(i0, i1 + 1)]
        pers = [ss.values[i - horizon] for i in range(i0, i1 + 1)]
        tm, pm, cm = ss.metrics(model, horizon, t0, t1), ss.metrics("persistence", horizon, t0, t1), ss.metrics("climatology", horizon, t0, t1)
        return {
            "region_id": region_id, "station": st.to_dict(True), "horizon": horizon, "split": dict(region.split),
            "forecaster_id": model, "forecaster_name": MODEL_NAME[model], "hparams": {},
            "test": tm, "test_baselines": {"climatology": cm, "persistence": pm},
            "skill_vs_persistence": 1 - tm["rmse"] / pm["rmse"] if pm["rmse"] else 0.0,
            "trace": {"dates": dates, "truth": truth, "pred": pred, "persistence": pers},
        }

    @router.get("/regions/{region_id}/gauge/{station_id}/forecast")
    async def forecast(region_id: str, station_id: str, as_of: str | None = None, horizon: int = 7, history_days: int = 120, forecaster: str | None = None) -> dict[str, Any]:
        region = region_or_404(region_id)
        st = station_or_404(region, station_id)
        state.ensure_evaluation(region)
        if region.evaluation is None:
            raise HTTPException(409, detail="먼저 학습하세요")
        if forecaster is not None and forecaster not in MODEL_PARAMS:
            raise HTTPException(422, detail=f"알 수 없는 예측기: {forecaster}")
        if not 1 <= horizon <= 90:
            raise HTTPException(422, detail="horizon 은 1~90")
        ss = st.series
        origin = _as_of(ss, as_of)
        steps = _steps(region, st, origin, horizon, forecaster)
        observed = [s for s in steps if s["observed_m"] is not None]
        verification = None
        if observed:
            errs = [s["error_m"] for s in observed]
            perr = [s["observed_m"] - s["persistence_reference_m"] for s in observed]
            rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
            prmse = math.sqrt(sum(e * e for e in perr) / len(perr))
            verification = {"n": len(observed), "rmse_m": rmse, "mae_m": sum(abs(e) for e in errs) / len(errs), "persistence_rmse_m": prmse,
                            "skill_vs_persistence": 1 - rmse / prmse if prmse else 0.0, "note_ko": "원점 이후 실제 관측이 있는 날짜만 비교. 원점 이전 자료로만 만든 전망이다."}
        in_sample = origin <= _d(region.split["val_end"])
        return {
            "region_id": region_id, "station": st.to_dict(True), "as_of": origin.isoformat(), "origin_level_m": ss.values[ss.index(origin)],
            "origin_observed": ss.index(origin) not in ss.missing, "horizon": horizon,
            "params_used": "params_eval" if in_sample else "params_deploy",
            "params_note_ko": "원점이 학습·검증 구간 안이라 표본 내(in-sample) 전망입니다 — 성능 근거로 쓰지 말고 시험 연도 평가표를 보세요." if in_sample else "시험 연도 원점 — 학습·검증 자료로만 적합한 파라미터로 전망",
            "band_note_ko": "σ·분위 밴드는 검증 연도 잔차에서 계산한 경험적 산포이며 모델 신뢰구간이 아니다.",
            "steps": steps, "history": _history(ss, origin, max(7, min(history_days, 365))), "verification": verification,
            "models_by_horizon": {str(s["horizon"]): s["forecaster_id"] for s in steps}, "in_sample": in_sample,
            "forecaster_override": forecaster, "forecaster_note_ko": None if forecaster else "예측일수마다 검증 연도별 폴드에서 고른 모델을 쓴다.",
            "coverage_note_ko": None,
        }

    return router


def build_stub_app(tick: int | None = None, seed: bool = True) -> FastAPI:
    """uvicorn --factory 로도 쓰는 스텁 앱. 환경변수 EVAL_STUB_TICK 을 읽는다."""
    if tick is None:
        tick = int(os.environ.get("EVAL_STUB_TICK", "1") or 1)
    state = StubState(tick=tick)
    if seed:
        state.seed()
    app = FastAPI(title="WATERCAST eval-ui stub")
    app.state.stub = state
    app.include_router(build_stub_router(state), prefix="/api/v1")
    mount_eval_ui(app)
    return app


# --------------------------------------------------------------------------- HTML 구문 검사
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _TagBalance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.problems: list[str] = []
        self.tags = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        if tag not in VOID_TAGS:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if not self.stack:
            self.problems.append(f"line {self.getpos()[0]}: </{tag}> without opener")
            return
        open_tag, line = self.stack.pop()
        if open_tag != tag:
            self.problems.append(f"line {self.getpos()[0]}: </{tag}> closes <{open_tag}> opened at line {line}")


def _check_html(html: str) -> _TagBalance:
    parser = _TagBalance()
    parser.feed(html)
    parser.close()
    for tag, line in parser.stack:
        parser.problems.append(f"<{tag}> opened at line {line} never closed")
    return parser


# --------------------------------------------------------------------------- 테스트
@pytest.fixture()
def client() -> TestClient:
    return TestClient(build_stub_app(tick=1, seed=True))


def _drive(client: TestClient, url: str, limit: int = 60) -> dict[str, Any]:
    job: dict[str, Any] = {}
    for _ in range(limit):
        job = client.get(url).json()
        if job["status"] in TERMINAL:
            break
    return job


STEP_TITLES = ("지역", "수위 자료 수집", "기상 자료 수집", "학습", "성능 평가", "시험")


def test_eval_page_is_served_with_korean_steps(client: TestClient) -> None:
    for path in ("/eval", "/eval/"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith("text/html")
        assert res.headers.get("cache-control") == "no-cache"
        html = res.text
        for title in STEP_TITLES:
            assert f'aria-hidden="true">{STEP_TITLES.index(title) + 1}</span>{title}' in html, title
        for word in ("준비 상태", "다음 할 일", "모델 평가", "도움말", "서비스", "새 지역 만들기", "수집 시작", "학습 시작", "전망 보기", "직전값 유지"):
            assert word in html, word
        assert '<html lang="ko">' in html
        assert 'href="/app"' in html and 'href="/easy"' in html and 'href="/docs"' in html


def test_eval_page_is_offline_and_small() -> None:
    html = INDEX_FILE.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html, "외부 URL 참조는 금지 (CSP self)"
    assert "<script src=" not in html and 'rel="stylesheet"' not in html and "@import" not in html and "@font-face" not in html
    assert INDEX_FILE.stat().st_size < 450 * 1024
    assert "Content-Security-Policy" in html and "default-src 'self'" in html


def test_eval_page_html_is_well_formed() -> None:
    parsed = _check_html(INDEX_FILE.read_text(encoding="utf-8"))
    assert parsed.tags > 150
    assert parsed.problems == []


def test_mount_does_not_touch_root_or_schema(client: TestClient) -> None:
    assert client.get("/").status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert "/eval" not in paths and "/eval/" not in paths


def test_mount_on_real_app_keeps_docs_redirect(tmp_path: Path) -> None:
    from backend.app.config import Settings
    from backend.app.main import create_app

    app = create_app(Settings(result_dir=tmp_path / "results"))
    mount_eval_ui(app)
    real = TestClient(app)
    root = real.get("/", follow_redirects=False)
    assert root.status_code in (301, 302, 307, 308)
    assert real.get("/eval").status_code == 200
    assert real.get("/api/v1/health").status_code == 200


def test_stub_seeded_region_is_ready_and_evaluated(client: TestClient) -> None:
    regions = client.get("/api/v1/regions").json()["items"]
    busan = next(r for r in regions if r["id"] == "nakdong_busan")
    assert busan["gauge_available"] and busan["satellite_available"] and busan["n_stations"] == 5
    assert busan["stations"][0]["period"] == ["2015-01-01", "2024-12-31"]
    ready = client.get("/api/v1/regions/nakdong_busan/readiness").json()
    assert ready["gauge"]["n_stations"] == 5 and ready["weather"]["available"] and ready["models"]["trained"]
    assert ready["satellite"]["n_scenes"] == 23
    assert all("②" not in s and "④" not in s for s in ready["next_steps"])

    ev = client.get("/api/v1/regions/nakdong_busan/gauge/evaluation").json()
    assert ev["split"]["test_start"] == "2024-01-01"
    assert len(ev["rows"]) == 5 * len(REPORT_H)
    row = ev["rows"][0]
    for key in ("station_id", "horizon", "forecaster_id", "forecaster_name", "test_mse", "test_rmse", "persistence_test_rmse", "skill_vs_persistence"):
        assert key in row, key
    assert row["forecaster_id"] != "persistence" and row["forecaster_id"] != "ridge_arx_future"
    assert any(r["skill_vs_persistence"] > 0 for r in ev["rows"]), "합성 자료에서도 몇 셀은 개선돼야 그림이 된다"

    cmp_ = client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast/compare", params={"horizon": 7}).json()
    assert cmp_["as_of"] == "2024-12-31" and len(cmp_["table"]) == 7 and len(cmp_["history"]) >= 80
    assert set(cmp_["models"]) == {f[0] for f in FORECASTERS if not f[5]}
    assert cmp_["table"][0]["observed_m"] is None
    assert sum(1 for m in cmp_["per_model"].values() if m["test_at_horizon"]["selected"]) == 1

    bt = client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/backtest", params={"horizon": 7}).json()
    assert bt["forecaster_id"] == cmp_["table"][6]["selected_model"]
    assert len(bt["trace"]["dates"]) == len(bt["trace"]["truth"]) == len(bt["trace"]["pred"]) == len(bt["trace"]["persistence"]) == 366
    assert client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/backtest", params={"horizon": 8}).status_code == 422

    ex = client.get("/api/v1/regions/nakdong_busan/gauge/experiments").json()
    assert ex["items"]["area_models"]["series"]["ICEYE"]["candidates"]["series_mean"]["sign_test_vs_persistence"]["p_two_sided"] == 0.625
    assert client.get("/api/v1/regions/geum_mainstem/gauge/experiments").json()["items"]["area_models"] is None

    fc = client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"as_of": "2024-02-18", "horizon": 7}).json()
    assert fc["in_sample"] is False and len(fc["steps"]) == 7 and fc["steps"][0]["observed_m"] is not None
    assert fc["verification"]["n"] == 7 and fc["forecaster_override"] is None
    fc2 = client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"as_of": "2020-02-18", "horizon": 3, "forecaster": "climatology"}).json()
    assert fc2["in_sample"] is True and all(s["forecaster_id"] == "climatology" for s in fc2["steps"])
    assert client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"forecaster": "nope"}).status_code == 422


def test_stub_full_onboarding_flow(client: TestClient) -> None:
    created = client.post("/api/v1/regions", json={"id": "geum_test", "name": "금강 시험", "description": "d", "bbox_wgs84": [127.2, 36.4, 127.4, 36.6]})
    assert created.status_code == 201, created.text
    assert created.json()["gauge_available"] is False
    ready = client.get("/api/v1/regions/geum_test/readiness").json()
    assert ready["gauge"]["n_stations"] == 0 and not ready["models"]["trained"] and len(ready["next_steps"]) >= 2

    found = client.get("/api/v1/wamis/stations", params={"basin": 3, "match": "금강"}).json()
    assert found["basin"] == 3 and {i["name"] for i in found["items"]} >= {"금남교", "금강교"}
    assert client.get("/api/v1/wamis/stations", params={"basin": 9}).status_code == 422

    body = {"stations": [{"id": "금남교", "name": "금남교", "code": "3009680", "terrain": "river"},
                         {"id": "세종보", "name": "세종보", "code": "3009655", "terrain": "dam"},
                         {"id": "금강교", "name": "금강교", "code": "3011665", "terrain": "river"}],
            "start": "2015-01-01", "end": "2025-12-31", "min_coverage": 0.85, "async_run": True}
    acc = client.post("/api/v1/regions/geum_test/gauge/collect", json=body)
    assert acc.status_code == 202 and acc.json()["job_id"].startswith("collect-")
    # 진행 중에는 같은 지역에 다른 잡을 못 건다
    assert client.post("/api/v1/regions/geum_test/gauge/train", json={"async_run": True}).status_code in (409, 422)
    job = _drive(client, acc.json()["poll_url"])
    assert job["status"] == "done"
    by_id = {s["id"]: s for s in job["stations"]}
    assert by_id["금남교"]["status"] == "done" and by_id["금남교"]["coverage"] > 0.85
    assert by_id["세종보"]["status"] == "low_coverage" and by_id["세종보"]["coverage"] < 0.85, "코드 끝 55 는 결측 많은 관측소 — 병합하되 표시"
    assert by_id["세종보"]["period"] == ["2015-01-01", "2025-12-31"]
    ready = client.get("/api/v1/regions/geum_test/readiness").json()
    assert ready["gauge"]["n_stations"] == 3 and ready["gauge"]["period"] == ["2015-01-01", "2025-12-31"]

    wx = client.post("/api/v1/regions/geum_test/weather/collect", json={"station_id": "133", "start": "2015-01-01", "end": "2025-12-31", "async_run": True})
    assert wx.status_code == 202
    wjob = _drive(client, wx.json()["poll_url"])
    assert wjob["status"] == "done" and wjob["source"] == "sample" and len(wjob["years"]) == 11 and wjob["warnings"]
    assert all(y["status"] == "done" and y["n_days"] in (365, 366) and y["n_rows"] == y["n_days"] for y in wjob["years"])
    assert client.get("/api/v1/regions/geum_test/readiness").json()["weather"]["available"]

    split = {"train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"}
    bad = client.post("/api/v1/regions/geum_test/gauge/train", json={"split": {**split, "val_start": "2021-01-01"}, "async_run": True})
    assert bad.status_code == 422 and "순서" in bad.json()["detail"]
    assert client.post("/api/v1/regions/geum_test/gauge/train", json={"forecasters": ["persistence"], "async_run": True}).status_code == 422
    tr = client.post("/api/v1/regions/geum_test/gauge/train", json={"split": split, "forecasters": ["mean_reversion", "ridge_ar", "climatology"], "horizon_max": 14, "async_run": True})
    assert tr.status_code == 202, tr.text
    assert tr.json()["split_used"] == split and tr.json()["stations"] == ["금남교", "세종보", "금강교"]
    first = client.get(tr.json()["poll_url"]).json()
    assert first["status"] == "running" and first["progress"]["stations_total"] == 3
    tjob = _drive(client, tr.json()["poll_url"])
    assert tjob["status"] == "done" and tjob["progress"]["stations_done"] == 3 and tjob["custom_split"] is True
    assert tjob["evaluation"]["horizon_max"] == 14
    assert client.get("/api/v1/regions/geum_test/gauge/train/jobs").json()["items"][0]["job_id"] == tr.json()["job_id"]

    ev = client.get("/api/v1/regions/geum_test/gauge/evaluation").json()
    assert {r["horizon"] for r in ev["rows"]} == {1, 3, 5, 7, 10, 14}
    assert {r["forecaster_id"] for r in ev["rows"]} <= {"mean_reversion", "ridge_ar", "climatology"}
    ready = client.get("/api/v1/regions/geum_test/readiness").json()
    assert ready["models"]["trained"] and ready["models"]["split"] == split
    assert not any(s.startswith(("②", "③", "④")) for s in ready["next_steps"])
    regions = {r["id"]: r for r in client.get("/api/v1/regions").json()["items"]}
    assert regions["geum_test"]["n_stations_trained"] == 3 and regions["geum_test"]["forecast"]["split"] == split

    fc = client.get("/api/v1/regions/geum_test/gauge/금남교/forecast", params={"as_of": "2025-01-01", "horizon": 14}).json()
    assert fc["in_sample"] is False and fc["verification"]["n"] == 14
    assert client.get("/api/v1/regions/geum_test/gauge/금강교/backtest", params={"horizon": 14}).json()["trace"]["dates"][0] == "2025-01-01"


def test_stub_rejects_bad_input(client: TestClient) -> None:
    assert client.post("/api/v1/regions", json={"id": "Bad Id!", "name": "x"}).status_code == 422
    assert client.post("/api/v1/regions", json={"id": "nakdong_busan", "name": "dup"}).status_code == 422
    assert client.post("/api/v1/regions", json={"id": "ok_id", "name": "x", "bbox_wgs84": [1, 2, 3]}).status_code == 422
    assert client.get("/api/v1/regions/nope/readiness").status_code == 404
    assert client.get("/api/v1/regions/nakdong_busan/gauge/없는지점/forecast").status_code == 404
    assert client.post("/api/v1/regions/geum_mainstem/weather/collect", json={"station_id": "999", "start": "2015-01-01", "end": "2016-01-01"}).status_code == 422
    assert client.post("/api/v1/regions/geum_mainstem/gauge/collect", json={"stations": [], "start": "2015-01-01", "end": "2016-01-01"}).status_code == 422
    assert client.get("/api/v1/regions/geum_mainstem/gauge/evaluation").status_code == 404
    assert client.get("/api/v1/regions/geum_mainstem/gauge/금남교/forecast/compare").status_code == 409


def test_stub_csv_upload_creates_station(client: TestClient) -> None:
    rows = "date,water_level_m\n" + "\n".join(f"{(date(2015, 1, 1) + timedelta(days=i)).isoformat()},{1.5 + 0.001 * i:.3f}" for i in range(0, 3650, 2))
    res = client.post("/api/v1/regions/geum_mainstem/gauge/세종보/series/file", files={"file": ("sejong.csv", rows.encode(), "text/csv")}, data={"terrain": "dam", "station_name": "세종보"})
    assert res.status_code == 200, res.text
    info = res.json()
    assert info["id"] == "세종보" and info["terrain"] == "dam" and info["source"] == "user_upload" and info["n_days"] == 1825
    assert info["period"] == ["2015-01-01", "2024-12-27"]
    bad = client.post("/api/v1/regions/geum_mainstem/gauge/x/series/file", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 422
    regions = {r["id"]: r for r in client.get("/api/v1/regions").json()["items"]}
    assert "세종보" in {s["id"] for s in regions["geum_mainstem"]["stations"]}
