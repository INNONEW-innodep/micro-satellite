"""서비스 화면(/app) 테스트.

페이지가 기대하는 API 계약 전부를 **인메모리 스텁 라우터**로 구현해 두고 그 위에 ``mount_service_ui`` 를 붙여 검사한다.
스텁은 GET /api/v1/scenes/{id} 가 호출될 때마다 가짜 파이프라인을 한 칸씩 진행시키므로 브라우저가 2초 폴링만 해도
등록→탐지→면적→수위→예측→규약 파일 순서로 끝까지 흘러간다. ``scripts/capture_service_ui.py`` 도 같은 스텁을 uvicorn 으로
띄워 스크린샷을 찍는다:

    .venv/bin/python -m uvicorn backend.tests.test_service_ui:build_stub_app --factory --port 8766

환경변수
    SERVICE_STUB_TICK   GET 한 번에 진행시킬 칸 수(기본 1)
    SERVICE_STUB_EVERY  GET 몇 번마다 진행시킬지(기본 1; 스크린샷은 2 로 느리게 돌린다)
    SERVICE_STUB_SEED   "1" 이면 지역 지도에 있는 기존 장면을 완료 상태로, 실패 장면 1개를 미리 넣는다

스텁이 구현하는 계약(설계 문서 docs/ui/서비스_화면_설계.md §4 + 기존 API 의 실제 응답 모양):
    GET  /health, /scenes/options, /scenes, /scenes/{id}, /scenes/{id}/{asset}
    POST /scenes (multipart, 202), POST /scenes/{id}/predict {model_id, horizon, as_of?}
    GET  /regions/{id}/scenarios?as_of=&horizon=  → 자료 기간(gauge/weather/satellite) · forecast_window · scenarios[] · requested
    GET  /regions/{id}/geometry   → bbox_wgs84 · stations · aois · scenes[footprint_lonlat]
    GET  /regions/{id}/satellite/area-series, /regions/{id}/satellite/levels?start=&end=
    GET  /regions/{id}/gauge/stations, /gauge/forecasters, /regions/{id}/gauge/{station}/forecast
    GET  /models

통합 측이 이 스텁에서 읽어 갈 가정:
    * 새 장면은 register 단계가 끝난 뒤부터 geometry.scenes 에 footprint_lonlat 과 함께 나타난다
    * 끝난 장면의 날짜는 area-series rows 와 levels records 에 바로 합류한다
    * forecast 의 steps[] 는 forecast_m · lower/upper_1sigma_m · observed_m(없으면 null) 을 가진다
    * predict 응답은 summary.steps[] (= steps[]) 에 target_date · water_area_km2 · risk 를 가진다
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from fastapi.testclient import TestClient

from backend.app.service_ui import INDEX_FILE, mount_service_ui

# --------------------------------------------------------------------------- 스텁 데이터
STEP_LABELS: list[tuple[str, str]] = [
    ("register", "장면 등록"),
    ("detect", "수체 탐지"),
    ("area", "면적 계산"),
    ("levels", "수위 산출"),
    ("predict", "시계열 예측"),
    ("deliver", "규약 파일·로그"),
]
STEP_ELAPSED = {"register": 1.2, "detect": 8.4, "area": 0.6, "levels": 2.1, "predict": 3.3, "deliver": 0.9}
STEP_DONE_DETAIL = {
    "register": "영상 1장을 등록했어요",
    "detect": "물 영역을 찾았어요",
    "area": "면적을 계산했어요",
    "levels": "수위계 자료와 맞춰 봤어요",
    "predict": "앞으로 7일을 예측했어요",
    "deliver": "결과 파일과 기록을 정리했어요",
}
REGIONS: list[dict[str, Any]] = [
    {"id": "nakdong_busan", "name": "낙동강 하구 (부산·김해)", "testbed": "busan", "sensors": ["ICEYE", "PlanetScope"],
     "has_gauge": True, "satellite_available": True, "detect_available": True, "n_scenes": 8},
    {"id": "geum_daecheong", "name": "금강 대청댐 (대전·청주)", "testbed": "dcd", "sensors": ["PlanetScope", "UMBRA"],
     "has_gauge": True, "satellite_available": True, "detect_available": True, "n_scenes": 6},
    {"id": "sejong_geum", "name": "세종 금강 (수위계 없음)", "testbed": None, "sensors": ["PlanetScope"],
     "has_gauge": False, "satellite_available": True, "detect_available": True, "n_scenes": 0},
]
SENSORS: list[dict[str, Any]] = [
    {"id": "ICEYE", "label": "ICEYE 레이더 영상", "type": "SAR", "needs_mask": True},
    {"id": "UMBRA", "label": "Umbra 레이더 영상", "type": "SAR", "needs_mask": True},
    {"id": "PlanetScope", "label": "PlanetScope 광학 영상", "type": "OPTIC", "needs_mask": False},
]
MODELS: list[dict[str, Any]] = [
    {"id": "gauge-level-forecast", "name": "Station water-level forecast (trained on 10-year gauge records)", "version": "1.0.0",
     "description": "수위는 지역 레지스트리의 학습 모델에서, 마스크는 면적 추세로 만든다.", "min_frames": 1, "supports_weather": True,
     "produces_water_level": True, "built_in": True, "options": {}},
    {"id": "irregular-area-trend", "name": "Irregular-date water-area trend baseline", "version": "1.0.0",
     "description": "날짜가 불규칙한 마스크들로 면적 추세를 맞춘다.", "min_frames": 2, "supports_weather": True,
     "produces_water_level": False, "built_in": True, "options": {}},
    {"id": "persistence", "name": "Persistence baseline", "version": "1.0.0",
     "description": "마지막 마스크를 그대로 반복한다.", "min_frames": 1, "supports_weather": False,
     "produces_water_level": True, "built_in": True, "options": {}},
    {"id": "weather-morphology", "name": "Weather morphology baseline", "version": "1.0.0",
     "description": "비가 오면 넓히고 가물면 줄이는 규칙 기반.", "min_frames": 1, "supports_weather": True,
     "produces_water_level": False, "built_in": True, "options": {}},
    {"id": "level-area-bridge", "name": "Level-to-area bridge (gauge forecast × stage-area slope)", "version": "1.0.0",
     "description": "면적을 수위 전망에 종속시킨다: 마지막 탐지 면적 + 기울기 × 수위 변화.", "min_frames": 1, "supports_weather": False,
     "produces_water_level": True, "built_in": True, "options": {}},
]
FORECASTERS: list[dict[str, Any]] = [
    {"id": "analog_knn", "name": "유사사례 k-NN", "description": "비슷한 과거 창의 이후 변화를 평균한다.", "built_in": True},
    {"id": "climatology", "name": "계절 평년값", "description": "연중일별 평균 수위.", "built_in": True, "is_reference": True},
    {"id": "ensemble_mean", "name": "계열 앙상블 평균", "description": "다섯 구성원의 예측 평균.", "built_in": True},
    {"id": "mean_reversion", "name": "평균 회귀 AR(1)", "description": "기준 수위로 되돌아가는 비율을 학습.", "built_in": True},
    {"id": "neural_mlp", "name": "MLP 신경망 (GPU 학습)", "description": "완전연결 신경망.", "built_in": False},
    {"id": "persistence", "name": "현재 수위 유지 (기준선)", "description": "원점 수위를 그대로 반복.", "built_in": True, "is_baseline": True},
]

# 지점 (실제 WAMIS 좌표). model_trained=False 인 지점은 전망을 거절한다(409)
STATIONS: dict[str, list[dict[str, Any]]] = {
    "nakdong_busan": [
        {"id": "낙동강하구언(내)", "name": "낙동강하구언(내)", "lat": 35.108056, "lon": 128.950556, "terrain": "estuary", "satellite_location": None, "model_trained": True},
        {"id": "구포대교", "name": "구포대교", "lat": 35.203611, "lon": 128.993333, "terrain": "estuary", "satellite_location": "gupo", "model_trained": True},
        {"id": "삼랑진교", "name": "삼랑진교", "lat": 35.382778, "lon": 128.821667, "terrain": "mainstem", "satellite_location": None, "model_trained": True},
        {"id": "양산교", "name": "양산교", "lat": 35.3575, "lon": 129.036111, "terrain": "mainstem", "satellite_location": None, "model_trained": True},
        {"id": "호포대교", "name": "호포대교", "lat": 35.289444, "lon": 129.011389, "terrain": "mainstem", "satellite_location": "hupo", "model_trained": True},
        {"id": "정천교", "name": "정천교", "lat": 35.205556, "lon": 128.833611, "terrain": "tributary", "satellite_location": "jeongcheon", "model_trained": False},
    ],
    "geum_daecheong": [
        {"id": "대청댐", "name": "대청댐", "lat": 36.47861, "lon": 127.48056, "terrain": "reservoir", "satellite_location": "daecheong", "model_trained": True},
        {"id": "대청조정지댐", "name": "대청조정지댐", "lat": 36.44972, "lon": 127.45222, "terrain": "reservoir", "satellite_location": "jojeongji", "model_trained": True},
        {"id": "대청댐방수로", "name": "대청댐방수로", "lat": 36.4745, "lon": 127.4735, "terrain": "channel", "satellite_location": "bangsuro", "model_trained": True},
    ],
    "sejong_geum": [],
}


def _rect(lon0: float, lat0: float, lon1: float, lat1: float) -> list[list[float]]:
    return [[lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]]


AOIS: dict[str, list[dict[str, Any]]] = {
    "nakdong_busan": [
        {"loc_id": "gupo", "station_id": "구포대교", "polygon_lonlat": _rect(128.9895, 35.2035, 128.9930, 35.2065), "anchor_lonlat": [128.993333, 35.203611]},
        {"loc_id": "hupo", "station_id": "호포대교", "polygon_lonlat": _rect(129.011216, 35.287666, 129.014516, 35.290372), "anchor_lonlat": [129.011389, 35.289444]},
        {"loc_id": "gimhae", "station_id": None, "polygon_lonlat": _rect(129.002077, 35.285205, 129.005376, 35.287911), "anchor_lonlat": [128.994167, 35.289444]},
        {"loc_id": "jeongcheon", "station_id": "정천교", "polygon_lonlat": _rect(128.832462, 35.206316, 128.835764, 35.209026), "anchor_lonlat": [128.833611, 35.205556]},
    ],
    "geum_daecheong": [
        {"loc_id": "daecheong", "station_id": "대청댐", "polygon_lonlat": _rect(127.485863, 36.48062, 127.487564, 36.481993), "anchor_lonlat": [127.48056, 36.47861]},
        {"loc_id": "jojeongji", "station_id": "대청조정지댐", "polygon_lonlat": _rect(127.457942, 36.453219, 127.459642, 36.454592), "anchor_lonlat": [127.45222, 36.44972]},
        {"loc_id": "bangsuro", "station_id": "대청댐방수로", "polygon_lonlat": _rect(127.475899, 36.4765, 127.4776, 36.477788), "anchor_lonlat": [127.4735, 36.4745]},
    ],
    "sejong_geum": [],
}
BBOX: dict[str, list[float]] = {
    "nakdong_busan": [128.66, 34.94, 129.20, 35.46],
    "geum_daecheong": [127.40, 36.40, 127.56, 36.52],
    "sejong_geum": [127.20, 36.45, 127.40, 36.60],
}


def _foot(cx: float, cy: float, w: float, h: float, tilt: float) -> list[list[float]]:
    """중심·폭·높이(도)·기울기(도)로 살짝 돌아간 촬영범위 사각형을 만든다 — 위성 궤도 때문에 비스듬한 느낌."""
    t = math.radians(tilt)
    pts = []
    for dx, dy in ((-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2), (-w / 2, -h / 2)):
        pts.append([round(cx + dx * math.cos(t) - dy * math.sin(t), 5), round(cy + dx * math.sin(t) + dy * math.cos(t), 5)])
    return pts


# 기존 장면(지역 지도의 회색 촬영범위 + 면적 시계열 행). (scene_id, sensor, date, footprint, area_km2_common_frame)
EXISTING_SCENES: dict[str, list[tuple[str, str, str, list[list[float]], float]]] = {
    "nakdong_busan": [
        ("20260622_PlanetScope", "PlanetScope", "2026-06-22", _foot(128.99, 35.24, 0.26, 0.24, 4), 39.52),
        ("20260705_ICEYE", "ICEYE", "2026-07-05", _foot(128.94, 35.21, 0.44, 0.50, 12), 40.21),
        ("20260718_PlanetScope", "PlanetScope", "2026-07-18", _foot(128.98, 35.23, 0.26, 0.24, 3), 35.91),
        ("20260729_ICEYE", "ICEYE", "2026-07-29", _foot(128.95, 35.20, 0.44, 0.50, 11), 36.61),
        ("20260809_PlanetScope", "PlanetScope", "2026-08-09", _foot(129.00, 35.22, 0.26, 0.24, 5), 37.18),
        ("20260820_ICEYE", "ICEYE", "2026-08-20", _foot(128.93, 35.22, 0.44, 0.50, 13), 38.07),
        ("20260902_PlanetScope", "PlanetScope", "2026-09-02", _foot(128.99, 35.23, 0.26, 0.24, 4), 36.84),
        ("20260914_ICEYE", "ICEYE", "2026-09-14", _foot(128.94, 35.21, 0.44, 0.50, 12), 37.42),
    ],
    "geum_daecheong": [
        ("20260820_PlanetScope", "PlanetScope", "2026-08-20", _foot(127.47, 36.465, 0.10, 0.09, 2), 1.31),
        ("20260905_PlanetScope", "PlanetScope", "2026-09-05", _foot(127.475, 36.462, 0.10, 0.09, 3), 1.27),
        ("20260920_PlanetScope", "PlanetScope", "2026-09-20", _foot(127.47, 36.466, 0.10, 0.09, 2), 1.35),
        ("20260925_UMBRA", "UMBRA", "2026-09-25", _foot(127.472, 36.465, 0.08, 0.08, 9), 1.32),
        ("20260928_UMBRA", "UMBRA", "2026-09-28", _foot(127.474, 36.464, 0.08, 0.08, 8), 1.18),
        ("20261003_UMBRA", "UMBRA", "2026-10-03", _foot(127.471, 36.466, 0.08, 0.08, 10), 1.47),
    ],
    "sejong_geum": [],
}
# 새 장면의 촬영범위 — 지점 일부만 덮게 해서 "지점 n개를 덮어요" 문장이 의미 있게 나오도록 한다
NEW_FOOTPRINT: dict[str, dict[str, list[list[float]]]] = {
    "nakdong_busan": {
        "ICEYE": [[128.78, 35.05], [129.08, 35.02], [129.10, 35.33], [128.80, 35.36]],          # 하구언·구포·호포·정천 (4)
        "PlanetScope": [[128.93, 35.16], [129.07, 35.15], [129.08, 35.31], [128.94, 35.32]],    # 구포·호포 (2)
    },
    "geum_daecheong": {
        "PlanetScope": _foot(127.47, 36.465, 0.11, 0.10, 2),
        "UMBRA": _foot(127.472, 36.465, 0.09, 0.09, 9),
    },
}
LEVEL_LOCS: dict[str, list[str]] = {"nakdong_busan": ["gupo", "hupo", "gimhae", "jeongcheon"], "geum_daecheong": ["daecheong", "jojeongji", "bangsuro"], "sejong_geum": []}
LEVEL_BASE = {"gupo": 2.36, "hupo": 1.78, "gimhae": 1.47, "jeongcheon": 1.58, "daecheong": 72.4, "jojeongji": 29.8, "bangsuro": 71.9}
GAUGE_END = date(2026, 10, 10)   # 스텁 수위계 관측이 있는 마지막 날 (전망 검증용)
OBS_START = date(2015, 1, 1)
TERMINAL = {"done", "failed"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _noise(*keys: Any) -> float:
    """[-1, 1) 결정적 잡음 — 같은 입력이면 늘 같은 값이라 스크린샷과 테스트가 흔들리지 않는다."""
    h = hashlib.md5("|".join(str(k) for k in keys).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") / 2**31 - 1.0


def _region(region_id: str) -> dict[str, Any] | None:
    return next((r for r in REGIONS if r["id"] == region_id), None)


def _station_level(region_id: str, station_id: str, day: date) -> float:
    """지점 '실측' 수위: 계절 사인 + 느린 변동 + 잡음. 과거(학습)와 전망 검증 모두 이 함수를 쓴다."""
    idx = [s["id"] for s in STATIONS.get(region_id, [])].index(station_id)
    base = {"nakdong_busan": 2.0, "geum_daecheong": 60.0, "sejong_geum": 1.0}[region_id] + idx * 0.35
    doy = day.timetuple().tm_yday
    seasonal = 0.45 * math.sin(2 * math.pi * (doy - 110) / 365.25)
    slow = 0.18 * math.sin(day.toordinal() / 23.0)
    return round(base + seasonal + slow + 0.04 * _noise(region_id, station_id, day.isoformat()), 4)


def _sat_level(loc_id: str, day: str, sensor: str) -> float:
    return round(LEVEL_BASE[loc_id] + 0.12 * math.sin(date.fromisoformat(day).toordinal() / 17.0) + 0.05 * _noise(loc_id, day, sensor), 4)


@dataclass
class FakeScene:
    scene_id: str
    region_id: str
    sensor: str
    image_date: str
    detect: bool = True
    predict: bool = True
    fail_at: str | None = None
    note: str | None = None
    liaison: dict[str, Any] = field(default_factory=dict)
    footprint: list[list[float]] | None = None
    seeded: bool = False
    pipeline_id: str = field(default_factory=lambda: "pipe-" + uuid.uuid4().hex[:8])
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    registered: bool = False

    def __post_init__(self) -> None:
        self.steps = [
            {"key": key, "label": label, "status": "pending", "started_at": None, "finished_at": None, "elapsed_s": None, "detail": None}
            for key, label in STEP_LABELS
        ]
        self.outputs = {k: None for k in (
            "quicklook_url", "overlay_url", "mask_url", "product_url", "product_name", "water_area_km2",
            "area_series_url", "levels_url", "prediction_id", "prediction_url", "folder",
        )}
        if self.footprint is None:
            per_sensor = NEW_FOOTPRINT.get(self.region_id, {})
            self.footprint = per_sensor.get(self.sensor) or next(iter(per_sensor.values()), None) or _rect(*BBOX.get(self.region_id, [0, 0, 1, 1]))

    # ---- 가짜 파이프라인
    def _skip_reason(self, key: str) -> str | None:
        region = _region(self.region_id)
        if key == "detect" and not self.detect:
            return "탐지 실행을 끄고 올렸어요"
        if key == "levels" and region is not None and not region["has_gauge"]:
            return "이 지역에는 수위계가 없어요"
        if key == "predict" and not self.predict:
            return "예측 실행을 끄고 올렸어요"
        return None

    def area_km2(self) -> float | None:
        if not self.detect:
            return None
        base = {"ICEYE": 38.64, "PlanetScope": 37.05, "UMBRA": 1.41}.get(self.sensor, 20.0)
        if self.region_id == "geum_daecheong":
            base = 1.41 if self.sensor == "UMBRA" else 1.29
        return round(base + 0.6 * _noise(self.scene_id) * (0.05 if base < 5 else 1.0), 2)

    def advance(self) -> None:
        """한 칸 진행: 진행 중인 단계를 끝내고 다음 단계를 시작한다. 건너뛰는 단계는 비용 없이 지나간다."""
        if self.status in TERMINAL:
            return
        now = _now()
        started_new = False
        for step in self.steps:
            if step["status"] in {"done", "skipped"}:
                continue
            if step["status"] == "running":
                if self.fail_at == step["key"]:
                    step.update(status="failed", finished_at=now, elapsed_s=STEP_ELAPSED[step["key"]],
                                detail="모델이 응답하지 않았어요 (스텁 오류)")
                    self.status = "failed"
                    self.error = f"'{step['label']}' 단계에서 멈췄어요: 모델이 응답하지 않았어요 (스텁 오류). 파일이 올바른 위성 영상인지 확인해 주세요."
                    self.finished_at = now
                    break
                detail = STEP_DONE_DETAIL[step["key"]]
                if step["key"] == "register":
                    self.registered = True
                elif step["key"] == "area" and self.area_km2() is not None:
                    detail = f"{self.area_km2():.2f} km²"
                step.update(status="done", finished_at=now, elapsed_s=STEP_ELAPSED[step["key"]], detail=detail)
                continue
            reason = self._skip_reason(step["key"])
            if reason:
                step.update(status="skipped", started_at=now, finished_at=now, elapsed_s=0.0, detail=reason)
                continue
            step.update(status="running", started_at=now, detail=None)
            self.status = "running"
            started_new = True
            break
        if self.status != "failed" and not started_new and all(s["status"] in {"done", "skipped"} for s in self.steps):
            self._complete(now)
        self._update_progress()

    def _update_progress(self) -> None:
        total = len(self.steps)
        score = 0.0
        for step in self.steps:
            if step["status"] in {"done", "skipped", "failed"}:
                score += 1
            elif step["status"] == "running":
                score += 0.5
        self.progress = round(min(1.0, score / total), 3)

    def _complete(self, now: str) -> None:
        self.status = "done"
        self.finished_at = now
        region = _region(self.region_id)
        base = f"/api/v1/scenes/{self.scene_id}"
        product_name = f"{self.region_id}_{self.sensor}_{self.image_date.replace('-', '')}_water.zip"
        prediction_id = "pred-" + uuid.uuid4().hex[:8]
        self.outputs.update({
            "quicklook_url": f"{base}/quicklook.png",
            "overlay_url": f"{base}/overlay.png" if self.detect else None,
            "mask_url": f"{base}/mask.tif" if self.detect else None,
            "product_url": f"{base}/product.zip",
            "product_name": product_name,
            "water_area_km2": self.area_km2(),
            "area_series_url": f"/api/v1/regions/{self.region_id}/satellite/area-series",
            "levels_url": (f"/api/v1/regions/{self.region_id}/satellite/levels?start={self.image_date}&end={self.image_date}"
                           if region and region["has_gauge"] else None),
            "prediction_id": prediction_id if self.predict else None,
            "prediction_url": f"/api/v1/predictions/{prediction_id}" if self.predict else None,
            "folder": f"scenes/{self.region_id}/{self.scene_id}",
        })
        self.warnings = ["표본 자료로 만든 결과예요. 실제 영상이 아니에요."]
        if self.sensor in {"ICEYE", "UMBRA"}:
            self.warnings.append("3-class 마스크가 없어 영상 유효 범위로 생성(바다 없음 가정)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "pipeline_id": self.pipeline_id,
            "region_id": self.region_id,
            "sensor": self.sensor,
            "image_date": self.image_date,
            "status": self.status,
            "progress": self.progress,
            "steps": [dict(s) for s in self.steps],
            "outputs": dict(self.outputs),
            "warnings": list(self.warnings),
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "liaison": dict(self.liaison),
            "note": self.note,
        }

    def to_item(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "region_id": self.region_id,
            "sensor": self.sensor,
            "image_date": self.image_date,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "water_area_km2": self.outputs.get("water_area_km2"),
            "product_name": self.outputs.get("product_name"),
            "pipeline_id": self.pipeline_id,
        }


# --------------------------------------------------------------------------- 가짜 그림 (PIL)
def _fake_images() -> tuple[bytes, bytes]:
    """원본 미리보기(흑백)와 물 영역 겹친 그림(파란 물)을 결정적으로 만든다."""
    from PIL import Image, ImageDraw, ImageFilter

    w, h = 800, 600
    base = Image.effect_noise((w, h), 48).filter(ImageFilter.GaussianBlur(2.2))
    base = base.point(lambda v: int(40 + v * 0.65))
    terrain = Image.merge("RGB", (base.point(lambda v: int(v * 0.72)), base.point(lambda v: int(v * 0.78)), base.point(lambda v: int(v * 0.62))))

    def lake(cx: float, cy: float, rx: float, ry: float, wobble: float = 0.18) -> list[tuple[float, float]]:
        pts = []
        for i in range(72):
            a = 2 * math.pi * i / 72
            k = 1 + wobble * math.sin(3 * a + 0.7) + 0.08 * math.cos(7 * a)
            pts.append((cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)))
        return pts

    water = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(water)
    d.polygon(lake(300, 330, 170, 120), fill=255)
    river = [(40, 520), (140, 470), (240, 430), (330, 410), (470, 390), (600, 330), (700, 250), (790, 180)]
    d.line(river, fill=255, width=46, joint="curve")
    d.polygon(lake(640, 160, 70, 50, 0.25), fill=255)
    water = water.filter(ImageFilter.GaussianBlur(1.2))

    quick = terrain.convert("L")
    dark = quick.point(lambda v: int(v * 0.45))
    quick.paste(dark, mask=water)

    overlay = terrain.copy()
    tint = Image.new("RGB", (w, h), (28, 120, 230))
    overlay = Image.composite(tint, overlay, water.point(lambda v: int(v * 0.72)))
    edge = water.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 40 else 0)
    overlay.paste(Image.new("RGB", (w, h), (120, 210, 255)), mask=edge)

    def png(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    return png(quick), png(overlay)


# --------------------------------------------------------------------------- 스텁 상태/라우터
class StubState:
    def __init__(self, tick: int = 1, every: int = 1) -> None:
        self.tick = max(1, int(tick))
        self.every = max(1, int(every))
        self.polls: dict[str, int] = {}
        self.scenes: dict[str, FakeScene] = {}
        self.predictions: dict[str, dict[str, Any]] = {}
        self._images: tuple[bytes, bytes] | None = None

    def images(self) -> tuple[bytes, bytes]:
        if self._images is None:
            self._images = _fake_images()
        return self._images

    def running(self) -> FakeScene | None:
        return next((s for s in self.scenes.values() if s.status not in TERMINAL), None)

    def seed(self) -> None:
        """지역 지도의 기존 장면을 모두 완료 상태로, 실패 장면 하나를 과거 날짜로 넣는다."""
        order = 0
        for region_id, scenes in EXISTING_SCENES.items():
            for scene_id, sensor, day, footprint, _area in scenes:
                order += 1
                sc = FakeScene(scene_id, region_id, sensor, day, footprint=footprint, seeded=True)
                sc.created_at = (datetime.fromisoformat(day + "T03:12:00+00:00") + timedelta(hours=2)).isoformat(timespec="seconds").replace("+00:00", "Z")
                while sc.status not in TERMINAL:
                    sc.advance()
                started = datetime.fromisoformat(sc.created_at.replace("Z", "+00:00"))
                sc.finished_at = (started + timedelta(seconds=19)).isoformat(timespec="seconds").replace("+00:00", "Z")
                self.scenes[scene_id] = sc
        failed = FakeScene("scene-seed-failed", "nakdong_busan", "PlanetScope", "2026-09-30", fail_at="detect", seeded=True)
        failed.created_at = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds").replace("+00:00", "Z")
        while failed.status not in TERMINAL:
            failed.advance()
        started = datetime.fromisoformat(failed.created_at.replace("Z", "+00:00"))
        failed.finished_at = (started + timedelta(seconds=11)).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.scenes[failed.scene_id] = failed

    # ---- 파생 자료 (기존 + 끝난 새 장면)
    def done_new_scenes(self, region_id: str) -> list[FakeScene]:
        return [s for s in self.scenes.values() if s.region_id == region_id and s.status == "done" and not s.seeded]

    def area_series(self, region_id: str) -> dict[str, Any]:
        sensors: dict[str, dict[str, Any]] = {}
        rows_by_sensor: dict[str, list[dict[str, Any]]] = {}
        for scene_id, sensor, day, footprint, area in EXISTING_SCENES.get(region_id, []):
            rows_by_sensor.setdefault(sensor, []).append({
                "date": day, "scene_id": scene_id, "area_km2_common_frame": area,
                "area_km2_scene_footprint": round(area * 1.004, 3), "gt_area_km2": None,
            })
        for sc in self.done_new_scenes(region_id):
            if sc.outputs.get("water_area_km2") is None:
                continue
            rows_by_sensor.setdefault(sc.sensor, []).append({
                "date": sc.image_date, "scene_id": sc.scene_id, "area_km2_common_frame": sc.outputs["water_area_km2"],
                "area_km2_scene_footprint": round(sc.outputs["water_area_km2"] * 1.004, 3), "gt_area_km2": None,
            })
        for sensor, rows in rows_by_sensor.items():
            rows.sort(key=lambda r: r["date"])
            sensors[sensor] = {"n_scenes": len(rows), "frame": {"coverage_km2": 3376.8 if sensor == "ICEYE" else 980.0}, "rows": rows}
        return {"region_id": region_id, "available": bool(sensors), "schema": "watercast.satellite.area-series/1", "sensors": sensors}

    def level_records(self, region_id: str) -> list[dict[str, Any]]:
        aois = {a["loc_id"]: a for a in AOIS.get(region_id, [])}
        stations = {s["id"]: s for s in STATIONS.get(region_id, [])}
        recs: list[dict[str, Any]] = []

        def add(day: str, sensor: str) -> None:
            for loc_id in LEVEL_LOCS.get(region_id, []):
                st_id = aois.get(loc_id, {}).get("station_id")
                gauge = None
                truth = _station_level(region_id, st_id, date.fromisoformat(day)) if st_id in stations else None
                if truth is not None and date.fromisoformat(day) <= GAUGE_END and _noise(loc_id, day, "gauge") > -0.6:
                    gauge = truth  # 같은 날 실측이 있는 날만 실측값을 붙인다
                # 위성 수위는 실측 근처에서 흔들린다 (실측이 비는 날도 같은 수위 계열을 따른다)
                lvl = round(truth + 0.09 * _noise(loc_id, day, "dev"), 4) if truth is not None else _sat_level(loc_id, day, sensor)
                recs.append({
                    "date": day, "loc_id": loc_id, "satellite": sensor, "water_level_m": lvl,
                    "water_area_km2": round(0.02 + 0.06 * abs(_noise(loc_id, day)), 6),
                    "gauge_level_m": gauge,
                    "corrected_minus_gauge_m": round(lvl - gauge, 4) if gauge is not None else None,
                    "paired": gauge is not None,
                })

        for _scene_id, sensor, day, _fp, _area in EXISTING_SCENES.get(region_id, []):
            add(day, sensor)
        for sc in self.done_new_scenes(region_id):
            if sc.detect:
                add(sc.image_date, sc.sensor)
        recs.sort(key=lambda r: (r["date"], r["loc_id"]))
        return recs

    def geometry(self, region_id: str) -> dict[str, Any]:
        region = _region(region_id)
        scenes: list[dict[str, Any]] = []
        for sc in sorted(self.scenes.values(), key=lambda s: s.image_date):
            if sc.region_id != region_id or not sc.registered or sc.status == "failed":
                continue
            scenes.append({
                "scene_id": sc.scene_id, "sensor": sc.sensor, "date": sc.image_date, "footprint_lonlat": sc.footprint,
                "quicklook_url": sc.outputs.get("quicklook_url"), "overlay_url": sc.outputs.get("overlay_url"),
                "detection_available": sc.status == "done" and sc.detect, "water_area_km2": sc.outputs.get("water_area_km2"),
            })
        if not any(s["scene_id"] == sid for sid in [e[0] for e in EXISTING_SCENES.get(region_id, [])] for s in scenes):
            # seed 를 안 한 경우에도 지도에 기존 촬영범위가 보이도록 정적 목록을 쓴다
            for scene_id, sensor, day, footprint, area in EXISTING_SCENES.get(region_id, []):
                if any(s["scene_id"] == scene_id for s in scenes):
                    continue
                scenes.append({"scene_id": scene_id, "sensor": sensor, "date": day, "footprint_lonlat": footprint,
                               "quicklook_url": None, "overlay_url": None, "detection_available": True, "water_area_km2": area})
            scenes.sort(key=lambda s: s["date"])
        return {
            "region_id": region_id,
            "name": region["name"] if region else region_id,
            "bbox_wgs84": BBOX[region_id],
            "stations": [dict(s) for s in STATIONS.get(region_id, [])],
            "aois": [dict(a) for a in AOIS.get(region_id, [])],
            "scenes": scenes,
        }


def build_stub_router(state: StubState) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": "stub-0.2", "service": "WATERCAST service-ui stub"}

    @router.get("/scenes/options")
    async def scene_options() -> dict[str, Any]:
        return {"regions": REGIONS, "sensors": SENSORS}

    @router.get("/scenes")
    async def list_scenes(region_id: str | None = Query(None), limit: int = Query(200, ge=1, le=500)) -> dict[str, Any]:
        items = sorted(state.scenes.values(), key=lambda s: s.created_at, reverse=True)
        if region_id:
            items = [s for s in items if s.region_id == region_id]
        return {"items": [s.to_item() for s in items[:limit]], "total": len(items)}

    @router.post("/scenes", status_code=202)
    async def create_scene(
        region_id: str = Form(...),
        sensor: str = Form(...),
        image_date: str = Form(...),
        pre_file: UploadFile = File(...),
        acquisition_time: str | None = Form(None),
        mask_file: UploadFile | None = File(None),
        mask3_file: UploadFile | None = File(None),
        order_id: str | None = Form(None),
        scene_id: str | None = Form(None),
        work_id: str | None = Form(None),
        algm_id: str | None = Form(None),
        product_type: str | None = Form(None),
        product_id: str | None = Form(None),
        detect: str = Form("true"),
        predict: str = Form("true"),
        note: str | None = Form(None),
    ) -> dict[str, Any]:
        region = _region(region_id)
        if region is None:
            raise HTTPException(404, detail=f"알 수 없는 지역: {region_id}")
        sensor_norm = next((s["id"] for s in SENSORS if s["id"].lower() == sensor.lower()), None)
        if sensor_norm is None:
            raise HTTPException(422, detail=f"알 수 없는 영상 종류: {sensor}")
        if not DATE_RE.match(image_date):
            raise HTTPException(422, detail="image_date 는 YYYY-MM-DD 형식이어야 해요")
        if acquisition_time and not TIME_RE.match(acquisition_time):
            raise HTTPException(422, detail="acquisition_time 은 HH:MM:SS 형식이어야 해요")
        if not (pre_file.filename or "").lower().endswith((".tif", ".tiff")):
            raise HTTPException(422, detail="pre_file 은 .tif/.tiff 파일이어야 해요")
        if state.running() is not None:
            raise HTTPException(409, detail="이미 진행 중인 장면이 있어요")
        await pre_file.read()
        for extra in (mask_file, mask3_file):
            if extra is not None:
                await extra.read()
        new_id = f"{image_date.replace('-', '')}_{sensor_norm}_{uuid.uuid4().hex[:4]}"
        scene = FakeScene(
            new_id, region_id, sensor_norm, image_date,
            detect=_truthy(detect) and mask_file is None, predict=_truthy(predict),
            fail_at="detect" if note and "fail" in note.lower() else None,
            note=note,
            liaison={"order_id": order_id, "scene_id": scene_id, "work_id": work_id, "algm_id": algm_id,
                     "product_type": product_type, "product_id": product_id},
        )
        if mask_file is not None:  # 마스크를 주면 탐지는 건너뛰되 면적은 낸다
            scene.detect = True
            scene.steps[1]["detail"] = "올려 준 물 영역 파일을 그대로 썼어요"
        state.scenes[new_id] = scene
        return {"scene_id": new_id, "pipeline_id": scene.pipeline_id, "status_url": f"/api/v1/scenes/{new_id}"}

    @router.get("/scenes/{scene_id}")
    async def get_scene(scene_id: str) -> dict[str, Any]:
        scene = state.scenes.get(scene_id)
        if scene is None:
            raise HTTPException(404, detail="장면을 찾을 수 없어요")
        state.polls[scene_id] = state.polls.get(scene_id, 0) + 1
        if state.polls[scene_id] % state.every == 1 or state.every == 1:
            for _ in range(state.tick):
                scene.advance()
        return scene.to_dict()

    @router.post("/scenes/{scene_id}/predict")
    async def predict_scene(scene_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        scene = state.scenes.get(scene_id)
        if scene is None:
            raise HTTPException(404, detail="장면을 찾을 수 없어요")
        if scene.status != "done":
            raise HTTPException(409, detail="탐지가 끝난 뒤에 예측할 수 있어요")
        model_id = str(body.get("model_id") or "")
        model = next((m for m in MODELS if m["id"] == model_id), None)
        if model is None:
            raise HTTPException(404, detail=f"알 수 없는 모델: {model_id}")
        try:
            horizon = int(body.get("horizon") or 0)
        except (TypeError, ValueError):
            horizon = 0
        if not 1 <= horizon <= 30:
            raise HTTPException(422, detail="horizon 은 1~30 사이여야 해요")
        as_of_raw = body.get("as_of")
        as_of: date | None = None
        if as_of_raw:
            if not DATE_RE.match(str(as_of_raw)):
                raise HTTPException(422, detail="as_of 는 YYYY-MM-DD 형식이어야 해요")
            as_of = date.fromisoformat(str(as_of_raw))
        same = sorted(
            [s for s in state.scenes.values() if s.region_id == scene.region_id and s.sensor == scene.sensor and s.status == "done"],
            key=lambda s: s.image_date,
        )
        used = [s for s in same if as_of is None or date.fromisoformat(s.image_date) <= as_of] or [scene]
        later = [s for s in same if as_of is not None and date.fromisoformat(s.image_date) > as_of]
        if as_of is not None and len(used) < 3:
            raise HTTPException(422, detail=f"{scene.sensor} 공통 격자 프레임 {len(used)}개 (원점 {as_of} 까지) — 시계열 예측에는 최소 3개 필요")
        origin_scene = used[-1] if as_of is not None else scene
        area0 = origin_scene.outputs.get("water_area_km2") or 1.0
        origin = date.fromisoformat(origin_scene.image_date)
        region = _region(scene.region_id)
        station = next((s for s in STATIONS.get(scene.region_id, []) if s["model_trained"]), None) if region and region["has_gauge"] else None
        steps = []
        for h in range(1, horizon + 1):
            trend = {"persistence": 0.0, "irregular-area-trend": -0.004, "weather-morphology": 0.006, "gauge-level-forecast": -0.003, "level-area-bridge": -0.002}[model_id]
            pct = trend * h * 100 + 0.8 * math.sin(h / 2.3) * (0 if model_id == "persistence" else 1)
            area = round(area0 * (1 + pct / 100), 4)
            risk = "normal" if abs(pct) < 5 else ("caution" if abs(pct) < 15 else "risk")
            level = None
            if model["produces_water_level"] and station is not None:
                level = _station_level(scene.region_id, station["id"], origin + timedelta(days=h)) if model_id != "persistence" else _station_level(scene.region_id, station["id"], origin)
            steps.append({"horizon": h, "target_date": (origin + timedelta(days=h)).isoformat(), "water_area_km2": area,
                          "area_change_pct": round(pct, 3), "water_level_m": level, "risk": risk, "rule_based": model_id != "gauge-level-forecast"})
        pid = "pred-" + uuid.uuid4().hex[:8]
        payload = {
            "prediction_id": pid, "prediction_url": f"/api/v1/predictions/{pid}", "scene_id": scene_id,
            "model_id": model_id, "model_name": model["name"], "horizon": horizon, "as_of": as_of.isoformat() if as_of else None,
            "summary": {"model_id": model_id, "horizon": horizon, "origin_date": origin.isoformat(), "origin_area_km2": area0, "steps": steps,
                        "frame_count": len(used), "source_dates": [s.image_date for s in used],
                        "later_frames": [{"date": s.image_date, "scene_id": s.scene_id, "water_area_km2": s.outputs.get("water_area_km2")} for s in later]},
            "steps": [{"key": "predict", "label": "시계열 예측", "status": "done", "detail": f"{model_id}, {horizon}일"}],
            "warnings": ["표본 자료로 만든 전망이에요."],
        }
        state.predictions[pid] = payload
        return payload

    @router.get("/scenes/{scene_id}/{asset}")
    async def get_asset(scene_id: str, asset: str) -> Response:
        scene = state.scenes.get(scene_id)
        if scene is None:
            raise HTTPException(404, detail="장면을 찾을 수 없어요")
        if scene.status != "done":
            raise HTTPException(409, detail="아직 결과가 준비되지 않았어요")
        quick, overlay = state.images()
        if asset == "quicklook.png":
            return Response(quick, media_type="image/png")
        if asset == "overlay.png":
            return Response(overlay, media_type="image/png")
        if asset == "mask.tif":
            return Response(b"II*\x00" + b"\x00" * 60, media_type="image/tiff",
                            headers={"Content-Disposition": f'attachment; filename="{scene_id}_mask.tif"'})
        if asset == "product.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("README.txt", f"WATERCAST stub product for {scene_id}\n")
                zf.writestr("mask.tif", b"II*\x00")
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{scene.outputs["product_name"]}"'})
        raise HTTPException(404, detail="없는 파일이에요")

    @router.get("/predictions/{prediction_id}")
    async def get_prediction(prediction_id: str) -> dict[str, Any]:
        return state.predictions.get(prediction_id) or {"id": prediction_id, "horizon_days": 7, "note": "stub prediction"}

    # ---- 지역 자료
    @router.get("/regions/{region_id}/geometry")
    async def region_geometry(region_id: str) -> dict[str, Any]:
        if region_id not in BBOX or region_id == "sejong_geum":  # sejong_geum: 지도 자료가 없는 지역 (문장 대체 경로)
            raise HTTPException(404, detail="이 지역의 지도 자료가 없어요")
        return state.geometry(region_id)

    @router.get("/regions/{region_id}/satellite/area-series")
    async def area_series(region_id: str) -> dict[str, Any]:
        if _region(region_id) is None:
            raise HTTPException(404, detail="알 수 없는 지역")
        return state.area_series(region_id)

    @router.get("/regions/{region_id}/satellite/levels")
    async def satellite_levels(region_id: str, start: str | None = None, end: str | None = None, loc_id: str | None = None) -> dict[str, Any]:
        if _region(region_id) is None:
            raise HTTPException(404, detail="알 수 없는 지역")
        recs = state.level_records(region_id)
        if start:
            recs = [r for r in recs if r["date"] >= start]
        if end:
            recs = [r for r in recs if r["date"] <= end]
        if loc_id:
            recs = [r for r in recs if r["loc_id"] == loc_id]
        return {"region_id": region_id, "loc_id": loc_id, "records": recs, "n": len(recs)}

    @router.get("/regions/{region_id}/gauge/stations")
    async def gauge_stations(region_id: str) -> list[dict[str, Any]]:
        if _region(region_id) is None:
            raise HTTPException(404, detail="알 수 없는 지역")
        return [dict(s, period=[OBS_START.isoformat(), GAUGE_END.isoformat()], n_days=3600, source="WAMIS") for s in STATIONS.get(region_id, [])]

    @router.get("/gauge/forecasters")
    async def forecasters() -> list[dict[str, Any]]:
        return FORECASTERS

    @router.get("/regions/{region_id}/gauge/{station_id}/forecast")
    async def forecast(
        region_id: str, station_id: str,
        horizon: int = Query(7, ge=1, le=30), forecaster: str | None = Query(None), as_of: str | None = Query(None),
    ) -> dict[str, Any]:
        station = next((s for s in STATIONS.get(region_id, []) if s["id"] == station_id), None)
        if _region(region_id) is None or station is None:
            raise HTTPException(404, detail="지점을 찾을 수 없어요")
        if not station["model_trained"]:
            raise HTTPException(409, detail="이 지점은 아직 학습된 모델이 없어요")
        if forecaster and not any(f["id"] == forecaster for f in FORECASTERS):
            raise HTTPException(404, detail=f"알 수 없는 예측기: {forecaster}")
        requested = date.fromisoformat(as_of) if as_of and DATE_RE.match(as_of) else None
        if requested is not None and requested < OBS_START:
            raise HTTPException(422, detail=f"원점 {as_of} 이전에는 {station_id} 수위 자료가 없습니다 — 자료 기간 {OBS_START} ~ {GAUGE_END} 안에서 고르세요")
        clamped = requested is not None and requested > GAUGE_END
        origin = GAUGE_END if clamped else (requested if requested is not None else GAUGE_END - timedelta(days=7))
        origin_level = _station_level(region_id, station_id, origin)
        history = [{"date": (origin - timedelta(days=i)).isoformat(),
                    "water_level_m": _station_level(region_id, station_id, origin - timedelta(days=i))} for i in range(120, -1, -1)]
        chosen = ["ensemble_mean", "neural_mlp", "neural_mlp", "neural_mlp", "mean_reversion", "mean_reversion", "mean_reversion"]
        steps = []
        for h in range(1, horizon + 1):
            target = origin + timedelta(days=h)
            truth = _station_level(region_id, station_id, target)
            if forecaster:
                fid = forecaster
            else:
                fid = chosen[h - 1] if h <= len(chosen) else "analog_knn"
            if forecaster == "persistence":
                fc = origin_level
            elif forecaster == "climatology":
                fc = round(truth - 0.3 * (truth - origin_level) + 0.05 * _noise(station_id, target.isoformat(), "clim"), 4)
            else:
                fc = round(origin_level + (truth - origin_level) * 0.75 + 0.035 * math.sqrt(h) * _noise(station_id, target.isoformat(), fid), 4)
            sigma = round(0.07 * math.sqrt(h), 4)
            observed = truth if target <= GAUGE_END else None
            steps.append({
                "horizon": h, "target_date": target.isoformat(), "forecast_m": fc, "forecaster_id": fid,
                "persistence_reference_m": origin_level, "sigma_m": sigma,
                "lower_1sigma_m": round(fc - sigma, 4), "upper_1sigma_m": round(fc + sigma, 4),
                "lower_2sigma_m": round(fc - 2 * sigma, 4), "upper_2sigma_m": round(fc + 2 * sigma, 4),
                "q05_m": round(fc - 1.5 * sigma, 4), "q95_m": round(fc + 1.5 * sigma, 4),
                "observed_m": observed, "error_m": round(fc - observed, 4) if observed is not None else None,
            })
        errs = [s["error_m"] for s in steps if s["error_m"] is not None]
        verification = None
        if errs:
            verification = {"n": len(errs), "rmse_m": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 4),
                            "mae_m": round(sum(abs(e) for e in errs) / len(errs), 4),
                            "note_ko": "원점 이후 실제 관측이 있는 날짜만 비교. 원점 이전 자료로만 만든 전망이다."}
        return {
            "region_id": region_id, "station": dict(station), "as_of": origin.isoformat(), "origin_level_m": origin_level,
            "requested_as_of": requested.isoformat() if requested is not None else None, "as_of_clamped": clamped,
            "data_period": [OBS_START.isoformat(), GAUGE_END.isoformat()],
            "observed_days_after": sum(1 for s in steps if s["observed_m"] is not None),
            "origin_note_ko": (f"요청한 출발일 {as_of} 에는 수위 자료가 없어 자료 마지막 날 {GAUGE_END} 을 출발일로 썼습니다 "
                               f"(자료 기간 {OBS_START} ~ {GAUGE_END}).") if clamped else None,
            "origin_observed": True, "horizon": horizon, "steps": steps, "history": history, "verification": verification,
            "models_by_horizon": {str(s["horizon"]): s["forecaster_id"] for s in steps},
            "in_sample": origin.year < 2024, "forecaster_override": forecaster,
            "forecaster_note_ko": "예측일수마다 검증 연도별 폴드에서 고른 모델을 쓴다." if not forecaster else f"모든 예측일수에 {forecaster} 를 썼다.",
            "band_note_ko": "σ·분위 밴드는 검증 연도 잔차에서 계산한 경험적 산포이며 모델 신뢰구간이 아니다.",
        }

    @router.get("/regions/{region_id}/scenarios")
    async def scenarios(region_id: str, as_of: str | None = Query(None), horizon: int = Query(7, ge=1, le=90)) -> dict[str, Any]:
        """④ 자료가 있는 기간 · 시나리오 — 실제 계약(backend/app/screens/scenarios.py)과 같은 모양."""
        region = _region(region_id)
        if region is None:
            raise HTTPException(404, detail="알 수 없는 지역")
        stations = STATIONS.get(region_id, [])
        has_gauge = bool(region["has_gauge"]) and bool(stations)
        gauge_period = [OBS_START.isoformat(), GAUGE_END.isoformat()] if has_gauge else None
        horizon_max = 30
        origin_min = OBS_START + timedelta(days=60)
        split = {"train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}
        done_scenes = sorted([s for s in state.scenes.values() if s.region_id == region_id and s.status == "done"], key=lambda s: (s.image_date, s.sensor))
        scene_rows = [{
            "scene_id": s.scene_id, "sensor": s.sensor, "date": s.image_date, "has_frame": True,
            "water_area_km2": s.outputs.get("water_area_km2"), "area_km2_common_frame": s.outputs.get("water_area_km2"),
            "in_gauge_period": bool(has_gauge and OBS_START <= date.fromisoformat(s.image_date) <= GAUGE_END),
        } for s in done_scenes]

        def after(origin: date, h: int) -> int:
            return max(0, min(h, (GAUGE_END - origin).days))

        items: list[dict[str, Any]] = []
        if has_gauge:
            for row in scene_rows:
                day = date.fromisoformat(row["date"])
                ok = OBS_START <= day <= GAUGE_END
                items.append({
                    "id": f"scene:{row['date'] if ok else ''}:{row['scene_id']}", "kind": "scene" if ok else "scene_no_gauge",
                    "as_of": row["date"] if ok else None,
                    "label_ko": f"영상 날짜 {row['date']} ({row['sensor']})" + ("" if ok else " — 수위 자료 없음"),
                    "note_ko": "이 영상이 찍힌 날을 출발일로" if ok else f"이 날짜에는 지상 수위계 자료가 없어요 (자료 {OBS_START} ~ {GAUGE_END})",
                    "available": ok, "observed_days_after": after(day, horizon_max) if ok else 0,
                    "in_sample": (day.isoformat() <= split["val_end"]) if ok else None,
                    "scene_id": row["scene_id"], "sensor": row["sensor"],
                })
            verify = GAUGE_END - timedelta(days=horizon_max)
            items += [
                {"id": "test_start:2024-01-01:", "kind": "test_start", "as_of": "2024-01-01", "label_ko": "시험 연도 첫날 2024-01-01",
                 "note_ko": "모델이 학습 때 보지 않은 구간이라 공정한 비교예요", "available": True, "observed_days_after": horizon_max,
                 "in_sample": False, "scene_id": None, "sensor": None},
                {"id": f"verify:{verify.isoformat()}:", "kind": "verify", "as_of": verify.isoformat(), "label_ko": f"자료 끝 {horizon_max}일 전 {verify.isoformat()}",
                 "note_ko": f"{horizon_max}일 전망 전부를 실제 수위와 비교할 수 있는 마지막 출발일이에요", "available": True,
                 "observed_days_after": horizon_max, "in_sample": False, "scene_id": None, "sensor": None},
                {"id": f"latest:{GAUGE_END.isoformat()}:", "kind": "latest", "as_of": GAUGE_END.isoformat(), "label_ko": f"자료 마지막 날 {GAUGE_END.isoformat()}",
                 "note_ko": "그 뒤 실측이 없어 순수한 미래 전망이에요 (비교 불가)", "available": True, "observed_days_after": 0,
                 "in_sample": False, "scene_id": None, "sensor": None},
            ]
        requested: dict[str, Any] | None = None
        if as_of:
            if not DATE_RE.match(as_of):
                raise HTTPException(422, detail="as_of 는 YYYY-MM-DD 형식이어야 해요")
            day = date.fromisoformat(as_of)
            h = min(horizon, horizon_max)
            if not has_gauge:
                requested = {"as_of": as_of, "available": False, "resolved_as_of": None, "clamped": False, "observed_days_after": 0,
                             "in_sample": None, "horizon": h, "target_date": None,
                             "note_ko": "이 지역에는 지상 수위계 자료가 없어 수위 전망을 만들 수 없어요"}
            else:
                clamped = day > GAUGE_END or day < OBS_START
                verify_origin = GAUGE_END - timedelta(days=horizon_max)  # 자료 뒤 날짜 → 전망 전부를 비교할 수 있는 마지막 출발일
                resolved = verify_origin if day > GAUGE_END else (origin_min if day < OBS_START else day)
                requested = {
                    "as_of": as_of, "available": not clamped, "resolved_as_of": resolved.isoformat(), "clamped": clamped,
                    "observed_days_after": after(resolved, h), "in_sample": resolved.isoformat() <= split["val_end"], "horizon": h,
                    "target_date": (resolved + timedelta(days=h)).isoformat(),
                    "note_ko": (f"{as_of} 에는 수위 자료가 없어요 (자료 {OBS_START} ~ {GAUGE_END}). 출발일을 자료 끝 {horizon_max}일 전인 {resolved} 로 맞췄어요 — "
                                "전망 전부를 실제 수위와 비교할 수 있어요. 아래에서 바꿀 수 있어요."
                                if clamped else None),
                }
        return {
            "region_id": region_id, "name": region["name"],
            "gauge": {"available": has_gauge, "period": gauge_period, "n_days": 3600 if has_gauge else 0, "n_stations": len(stations),
                      "stations": [{"id": s["id"], "name": s["name"], "terrain": s["terrain"], "period": gauge_period, "n_days": 3600,
                                    "model_trained": s["model_trained"]} for s in stations]},
            "weather": {"available": has_gauge, "station": ("159" if region_id == "nakdong_busan" else "133") if has_gauge else None,
                        "station_name": ("부산" if region_id == "nakdong_busan" else "대전") if has_gauge else None,
                        "period": ["2015-01-01", GAUGE_END.isoformat()] if has_gauge else None, "n_days": 4300 if has_gauge else 0},
            "models": {"trained": has_gauge, "trained_at": "2026-10-01T00:00:00+00:00" if has_gauge else None,
                       "split": split if has_gauge else None, "horizon_max": horizon_max,
                       "n_stations_trained": sum(1 for s in stations if s["model_trained"])},
            "satellite": {"available": bool(scene_rows), "n_scenes": len(scene_rows), "sensors": sorted({r["sensor"] for r in scene_rows}),
                          "period": [scene_rows[0]["date"], scene_rows[-1]["date"]] if scene_rows else None, "scenes": scene_rows},
            "forecast_window": ({"origin_min": origin_min.isoformat(), "origin_max": GAUGE_END.isoformat(), "horizon_max": horizon_max,
                                 "verifiable_until": GAUGE_END.isoformat(), "min_history_days": 60} if has_gauge else None),
            "scenarios": items, "requested": requested,
            "note_ko": (f"출발일은 {origin_min.isoformat()} ~ {GAUGE_END.isoformat()} 사이에서 고르세요. {GAUGE_END.isoformat()} 까지는 실제 수위가 있어 전망과 비교할 수 있어요."
                        if has_gauge else "지상 수위계 자료가 없어 수위 전망 시나리오를 만들 수 없어요 — 모델 평가 화면 ② 에서 수위 자료를 먼저 모으세요"),
            # 다지역 확장 경로 판정 — 실제 계약(scenarios.py expansion_mode)과 같은 키. 스텁 지역은 장면이 적어 전부 경로 B.
            "expansion": {"mode": "gauge_first", "label_ko": "경로 B — 지상 수위계 우선, 위성은 검증",
                          "rule_ko": "경로 A 조건: 센서당 장면 ≥ 36 · 기간 ≥ 2년 · 홍수기 포함 · (정답 라벨 또는 대조 ≥ 30건). 넘으면 경로 A 를 추가로 돌려 둘을 병기(승격)",
                          "reasons_ko": [f"센서당 장면 최대 {len(scene_rows)}장 (< 36장)" if scene_rows else "위성 장면 없음",
                                         "지상 수위계 있음 → 경로 B 가동" if has_gauge else "지상 수위계 없음 — 경로 B 도 수집 뒤에 가능"],
                          "checks": {"scenes_per_sensor": {"value": {}, "min": 36, "ok": False}, "years": {"value": 0.0, "min": 2.0, "ok": False},
                                     "flood_season": {"value": False, "months": [6, 7, 8, 9], "ok": False},
                                     "labels_or_pairs": {"labels": False, "pairs": 0, "min_pairs": 30, "ok": False},
                                     "gauge": {"value": has_gauge, "ok": has_gauge}, "dem": {"value": False, "ok": False}}},
        }

    @router.get("/models")
    async def models() -> list[dict[str, Any]]:
        return MODELS

    return router


def build_stub_app(tick: int | None = None, seed: bool | None = None, every: int | None = None) -> FastAPI:
    """uvicorn --factory 로도 쓰는 스텁 앱. 환경변수 SERVICE_STUB_TICK / SERVICE_STUB_EVERY / SERVICE_STUB_SEED 를 읽는다."""
    if tick is None:
        tick = int(os.environ.get("SERVICE_STUB_TICK", "1") or 1)
    if every is None:
        every = int(os.environ.get("SERVICE_STUB_EVERY", "1") or 1)
    if seed is None:
        seed = _truthy(os.environ.get("SERVICE_STUB_SEED"))
    state = StubState(tick=tick, every=every)
    if seed:
        state.seed()
    app = FastAPI(title="WATERCAST service-ui stub")
    app.state.stub = state
    app.include_router(build_stub_router(state), prefix="/api/v1")
    mount_service_ui(app)
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
        if tag in VOID_TAGS:
            return
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
    return TestClient(build_stub_app(tick=1, seed=False))


@pytest.fixture()
def seeded() -> TestClient:
    return TestClient(build_stub_app(tick=1, seed=True))


def _dummy_tif(name: str = "scene.tif") -> tuple[str, bytes, str]:
    return (name, b"II*\x00" + b"\x00" * 128, "image/tiff")


def _post_scene(client: TestClient, **overrides: Any) -> Any:
    data = {"region_id": "nakdong_busan", "sensor": "ICEYE", "image_date": "2026-10-03", "detect": "true", "predict": "true"}
    data.update(overrides)
    return client.post("/api/v1/scenes", data=data, files={"pre_file": _dummy_tif()})


def _drive(client: TestClient, scene_id: str, limit: int = 40) -> dict[str, Any]:
    scene: dict[str, Any] = {}
    for _ in range(limit):
        scene = client.get(f"/api/v1/scenes/{scene_id}").json()
        if scene["status"] in TERMINAL:
            break
    return scene


def test_service_page_is_served_with_korean_steps(client: TestClient) -> None:
    for path in ("/app", "/app/"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith("text/html")
        assert res.headers.get("cache-control") == "no-cache"
        html = res.text
        for title in ("자료 넣기 · 위치 확인", "탐지", "시각화", "예측", "위치 지도", "최근 결과", "시작하기", "다시 시도",
                      "최근 장면 고르기", "수위 전망", "수체 면적 전망", "모델 평가", "도움말", "WATERCAST",
                      # ④ 시나리오 (2026-10-14): 자료가 있는 기간 → 출발일·어디까지 고르기
                      "자료가 있는 기간", "시나리오", "출발일 (원점)", "며칠 뒤까지", "끝 날짜", "출발 영상 (원점)",
                      # ① 지역을 고르면 자료가 언제 있는지 (2026-10-15)
                      "이 지역 자료"):
            assert title in html, title
        assert '<html lang="ko">' in html
    body = re.sub(r"<script.*?</script>", "", client.get("/app").text, flags=re.S)
    for jargon in ("GeoTIFF", "pipeline", "endpoint", "multipart"):
        assert jargon not in body, jargon


def test_service_page_is_offline_except_openstreetmap() -> None:
    html = INDEX_FILE.read_text(encoding="utf-8")
    urls = re.findall(r"https?://[^\s\"'<>)]+", html)
    # SVG 네임스페이스 식별자(createElementNS 인자)는 네트워크로 나가지 않는다 — 외부 링크는 OpenStreetMap 하나뿐
    urls = [u for u in urls if u != "http://www.w3.org/2000/svg"]
    assert urls, "OpenStreetMap 링크는 있어야 한다"
    assert all("openstreetmap.org" in u for u in urls), [u for u in urls if "openstreetmap.org" not in u]
    assert "<script src=" not in html and 'rel="stylesheet"' not in html and "@import" not in html and "@font-face" not in html
    assert INDEX_FILE.stat().st_size < 400 * 1024
    assert "Content-Security-Policy" in html
    assert "default-src 'self'" in html and "img-src 'self' data:" in html


def test_service_page_html_is_well_formed() -> None:
    parsed = _check_html(INDEX_FILE.read_text(encoding="utf-8"))
    assert parsed.tags > 150
    assert parsed.problems == []


def test_mount_does_not_touch_root(client: TestClient) -> None:
    assert client.get("/").status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert "/app" not in paths and "/app/" not in paths


def test_mount_on_real_app_keeps_docs_redirect(tmp_path: Path) -> None:
    from backend.app.config import Settings
    from backend.app.main import create_app

    app = create_app(Settings(result_dir=tmp_path / "results"))
    mount_service_ui(app)
    real = TestClient(app)
    root = real.get("/", follow_redirects=False)
    assert root.status_code in (301, 302, 307, 308)
    assert root.headers["location"] == "/app"  # 서비스 화면이 정문 (2026-10-14)
    assert real.get("/app").status_code == 200
    assert real.get("/api/v1/health").status_code == 200


def test_stub_options_and_catalogs(client: TestClient) -> None:
    opts = client.get("/api/v1/scenes/options").json()
    assert [r["id"] for r in opts["regions"]] == ["nakdong_busan", "geum_daecheong", "sejong_geum"]
    assert [s["id"] for s in opts["sensors"]] == ["ICEYE", "UMBRA", "PlanetScope"]
    assert {m["id"] for m in client.get("/api/v1/models").json()} >= {"persistence", "irregular-area-trend", "gauge-level-forecast"}
    assert any(f["id"] == "ensemble_mean" for f in client.get("/api/v1/gauge/forecasters").json())
    stations = client.get("/api/v1/regions/nakdong_busan/gauge/stations").json()
    assert {s["id"] for s in stations} >= {"구포대교", "호포대교"}
    assert all({"lat", "lon", "model_trained"} <= set(s) for s in stations)
    assert client.get("/api/v1/regions/nope/gauge/stations").status_code == 404


def test_stub_geometry_contract(seeded: TestClient) -> None:
    geo = seeded.get("/api/v1/regions/nakdong_busan/geometry").json()
    assert set(geo) == {"region_id", "name", "bbox_wgs84", "stations", "aois", "scenes"}
    assert len(geo["bbox_wgs84"]) == 4
    assert all({"id", "name", "lat", "lon"} <= set(s) for s in geo["stations"])
    assert all({"loc_id", "station_id", "polygon_lonlat", "anchor_lonlat"} <= set(a) for a in geo["aois"])
    assert len(geo["scenes"]) == 8 and all(len(s["footprint_lonlat"]) == 4 for s in geo["scenes"])
    assert all({"scene_id", "sensor", "date", "footprint_lonlat", "detection_available", "water_area_km2"} <= set(s) for s in geo["scenes"])
    assert seeded.get("/api/v1/regions/sejong_geum/geometry").status_code == 404


def test_stub_pipeline_runs_to_done_and_feeds_map_and_series(seeded: TestClient) -> None:
    created = _post_scene(seeded, order_id="K-20261003", algm_id="AG000003", product_type="PT000003")
    assert created.status_code == 202, created.text
    body = created.json()
    assert set(body) == {"scene_id", "pipeline_id", "status_url"}
    scene_id = body["scene_id"]

    first = seeded.get(f"/api/v1/scenes/{scene_id}").json()
    assert first["status"] == "running" and first["steps"][0]["status"] == "running"
    assert first["liaison"]["order_id"] == "K-20261003"
    # 등록 전에는 지도에 새 장면이 없다
    assert not any(s["scene_id"] == scene_id for s in seeded.get("/api/v1/regions/nakdong_busan/geometry").json()["scenes"])
    second = seeded.get(f"/api/v1/scenes/{scene_id}").json()
    assert second["steps"][0]["status"] == "done" and second["steps"][1]["status"] == "running"
    new = [s for s in seeded.get("/api/v1/regions/nakdong_busan/geometry").json()["scenes"] if s["scene_id"] == scene_id]
    assert len(new) == 1 and new[0]["footprint_lonlat"] == NEW_FOOTPRINT["nakdong_busan"]["ICEYE"]

    scene = _drive(seeded, scene_id)
    assert scene["status"] == "done" and all(s["status"] == "done" for s in scene["steps"])
    outputs = scene["outputs"]
    assert outputs["water_area_km2"] and outputs["levels_url"].endswith("start=2026-10-03&end=2026-10-03")
    for key in ("quicklook_url", "overlay_url", "mask_url", "product_url", "levels_url", "area_series_url"):
        assert seeded.get(outputs[key]).status_code == 200, key
    levels = seeded.get(outputs["levels_url"]).json()["records"]
    assert {r["loc_id"] for r in levels} == {"gupo", "hupo", "gimhae", "jeongcheon"}
    assert all(r["date"] == "2026-10-03" and isinstance(r["water_level_m"], float) for r in levels)
    series = seeded.get(outputs["area_series_url"]).json()["sensors"]
    assert series["ICEYE"]["rows"][-1]["scene_id"] == scene_id and series["ICEYE"]["rows"][-1]["date"] == "2026-10-03"
    assert len(series["PlanetScope"]["rows"]) == 4
    items = seeded.get("/api/v1/scenes?region_id=nakdong_busan").json()["items"]
    assert items[0]["scene_id"] == scene_id and items[0]["status"] == "done"

    pred = seeded.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "irregular-area-trend", "horizon": 5})
    assert pred.status_code == 200, pred.text
    p = pred.json()
    assert {"prediction_id", "prediction_url", "summary", "steps"} <= set(p)
    assert len(p["summary"]["steps"]) == 5
    assert all({"target_date", "water_area_km2", "risk"} <= set(s) for s in p["summary"]["steps"])
    assert seeded.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "nope", "horizon": 5}).status_code == 404
    assert seeded.post(f"/api/v1/scenes/{scene_id}/predict", json={"model_id": "persistence", "horizon": 99}).status_code == 422


def test_stub_forecast_contract(client: TestClient) -> None:
    res = client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"as_of": "2026-10-03", "horizon": 7})
    assert res.status_code == 200
    f = res.json()
    assert f["as_of"] == "2026-10-03" and len(f["steps"]) == 7 and len(f["history"]) == 121
    assert f["history"][-1]["date"] == "2026-10-03"
    step = f["steps"][0]
    assert {"horizon", "target_date", "forecast_m", "lower_1sigma_m", "upper_1sigma_m", "observed_m", "forecaster_id"} <= set(step)
    assert f["verification"]["n"] == 7  # 관측이 10-10 까지 있다
    long = client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"as_of": "2026-10-03", "horizon": 14, "forecaster": "persistence"}).json()
    assert long["verification"]["n"] == 7 and long["steps"][-1]["observed_m"] is None
    assert all(s["forecaster_id"] == "persistence" for s in long["steps"]) and long["forecaster_override"] == "persistence"
    assert client.get("/api/v1/regions/nakdong_busan/gauge/정천교/forecast").status_code == 409
    assert client.get("/api/v1/regions/nakdong_busan/gauge/없는지점/forecast").status_code == 404
    assert client.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"forecaster": "nope"}).status_code == 404


def test_stub_skips_levels_without_gauge_and_predict_when_off(client: TestClient) -> None:
    created = _post_scene(client, region_id="sejong_geum", sensor="PlanetScope", predict="false")
    assert created.status_code == 202
    scene = _drive(client, created.json()["scene_id"])
    assert scene["status"] == "done"
    by_key = {s["key"]: s for s in scene["steps"]}
    assert by_key["levels"]["status"] == "skipped" and "수위계" in by_key["levels"]["detail"]
    assert by_key["predict"]["status"] == "skipped"
    assert scene["outputs"]["levels_url"] is None


def test_stub_rejects_bad_input_and_concurrent_runs(client: TestClient) -> None:
    assert _post_scene(client, image_date="2026/10/03").status_code == 422
    assert _post_scene(client, sensor="Sentinel").status_code == 422
    assert _post_scene(client, region_id="nope").status_code == 404
    first = _post_scene(client)
    assert first.status_code == 202
    assert _post_scene(client).status_code == 409
    assert client.get("/api/v1/scenes/nope").status_code == 404
    assert client.post("/api/v1/scenes/nope/predict", json={"model_id": "persistence", "horizon": 3}).status_code == 404


def test_stub_failure_path(client: TestClient) -> None:
    created = _post_scene(client, note="please fail")
    scene = _drive(client, created.json()["scene_id"])
    assert scene["status"] == "failed" and scene["error"]
    statuses = [s["status"] for s in scene["steps"]]
    assert statuses[:2] == ["done", "failed"]


def test_stub_seed_fills_recent_results(seeded: TestClient) -> None:
    items = seeded.get("/api/v1/scenes").json()["items"]
    assert {i["status"] for i in items} == {"done", "failed"}
    assert len(items) == 8 + 6 + 1
    assert items[0]["scene_id"] == "scene-seed-failed"  # 가장 최근 접수
    levels = seeded.get("/api/v1/regions/nakdong_busan/satellite/levels").json()["records"]
    assert len(levels) == 8 * 4 and any(r["gauge_level_m"] is not None for r in levels)


# --------------------------------------------------------------------------- ④ 시나리오 계약 (2026-10-14)
def test_stub_scenarios_contract(seeded: TestClient) -> None:
    sc = seeded.get("/api/v1/regions/geum_daecheong/scenarios", params={"as_of": "2026-09-20", "horizon": 7}).json()
    assert sc["gauge"]["period"] == ["2015-01-01", "2026-10-10"] and sc["forecast_window"]["horizon_max"] == 30
    n_seeded = len(EXISTING_SCENES["geum_daecheong"])
    assert sc["satellite"]["n_scenes"] == n_seeded and sc["requested"]["available"] is True and sc["requested"]["clamped"] is False
    assert sc["requested"]["target_date"] == "2026-09-27" and sc["requested"]["observed_days_after"] == 7
    kinds = [s["kind"] for s in sc["scenarios"]]
    assert kinds.count("scene") == n_seeded and kinds[-3:] == ["test_start", "verify", "latest"]
    assert {"id", "kind", "as_of", "label_ko", "note_ko", "available", "observed_days_after", "in_sample", "scene_id", "sensor"} <= set(sc["scenarios"][0])
    assert sc["expansion"]["mode"] in ("gauge_first", "satellite_series") and set(sc["expansion"]["checks"]) == {"scenes_per_sensor", "years", "flood_season", "labels_or_pairs", "gauge", "dem"}
    beyond = seeded.get("/api/v1/regions/geum_daecheong/scenarios", params={"as_of": "2026-10-12"}).json()["requested"]
    assert beyond["clamped"] is True and beyond["resolved_as_of"] == "2026-09-10" and "수위 자료가 없어요" in beyond["note_ko"]
    assert beyond["observed_days_after"] == 7 and "자료 끝 30일 전" in beyond["note_ko"]  # 자료 뒤 날짜 → 비교 가능한 마지막 출발일로
    none = seeded.get("/api/v1/regions/sejong_geum/scenarios", params={"as_of": "2026-09-20"}).json()
    assert none["gauge"]["available"] is False and none["scenarios"] == [] and none["requested"]["available"] is False
    assert seeded.get("/api/v1/regions/nope/scenarios").status_code == 404
    assert seeded.get("/api/v1/regions/geum_daecheong/scenarios", params={"as_of": "nope"}).status_code == 422


def test_stub_forecast_reports_clamped_origin_and_predict_accepts_as_of(seeded: TestClient) -> None:
    beyond = seeded.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"as_of": "2026-10-12", "horizon": 7}).json()
    assert beyond["as_of"] == "2026-10-10" and beyond["requested_as_of"] == "2026-10-12" and beyond["as_of_clamped"] is True
    assert beyond["observed_days_after"] == 0 and "자료 마지막 날" in beyond["origin_note_ko"]
    assert beyond["data_period"] == ["2015-01-01", "2026-10-10"]
    inside = seeded.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"as_of": "2026-09-20", "horizon": 7}).json()
    assert inside["as_of_clamped"] is False and inside["observed_days_after"] == 7 and inside["origin_note_ko"] is None
    assert seeded.get("/api/v1/regions/nakdong_busan/gauge/구포대교/forecast", params={"as_of": "2010-01-01"}).status_code == 422

    pred = seeded.post("/api/v1/scenes/20260902_PlanetScope/predict", json={"model_id": "irregular-area-trend", "horizon": 3, "as_of": "2026-08-09"}).json()
    assert pred["as_of"] == "2026-08-09" and pred["summary"]["source_dates"] == ["2026-06-22", "2026-07-18", "2026-08-09"]
    assert pred["summary"]["frame_count"] == 3 and pred["summary"]["steps"][0]["target_date"] == "2026-08-10"
    assert [f["date"] for f in pred["summary"]["later_frames"]] == ["2026-09-02"] and pred["summary"]["later_frames"][0]["water_area_km2"] > 0
    assert seeded.post("/api/v1/scenes/20260902_PlanetScope/predict", json={"model_id": "persistence", "horizon": 3, "as_of": "2026-07-01"}).status_code == 422
    assert seeded.post("/api/v1/scenes/20260902_PlanetScope/predict", json={"model_id": "persistence", "horizon": 3, "as_of": "nope"}).status_code == 422
    whole = seeded.post("/api/v1/scenes/20260902_PlanetScope/predict", json={"model_id": "persistence", "horizon": 3}).json()
    assert whole["as_of"] is None and whole["summary"]["later_frames"] == [] and whole["summary"]["origin_date"] == "2026-09-02"
