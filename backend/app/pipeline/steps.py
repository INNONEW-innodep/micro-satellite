"""장면 파이프라인 단계 구현 — 기존 서비스를 순서대로 엮는다.

한 위성 장면에 대해 ``register → detect → area → levels → predict → deliver`` 를 돈다.
새 알고리즘은 없다. 등록(``gauge_service.register_scene``)·탐지(WBMS 잡 어댑터)·면적
(``scripts/build_common_frame_area.py``)·수위(③ calc_wlwa 컨테이너)·예측(``PredictionService``)·
연계(``wbms.liaison``)를 **호출**만 한다.

단계 실패 규칙
--------------
* ``register``·``detect``·``deliver`` 가 실패하면 파이프라인이 실패한다(뒤 단계는 skipped).
* ``area``·``levels``·``predict`` 는 부가 산출이라 실패해도 파이프라인은 계속 간다 — 그 단계만
  ``failed`` 로 남고 ``warnings`` 에 사유가 붙는다(마스크·규약 파일은 이미 나와 있기 때문).
* 입력이 없어 돌릴 수 없는 단계는 ``skipped`` + 정확한 사유.

상태는 모든 변경 직후 ``save`` 콜백(= 장면 폴더의 status.json)으로 남긴다.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Collection, Callable
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..segmentation.base import SceneRequest
from ..segmentation.wbms_plugin import WbmsDockerSegmenter
from ..services.inputs import UploadedBytes, load_mask_sequence
from ..wbms.liaison import (
    LIAISON_KEYS,
    PRODUCT_TYPE_WATER_MASK,
    derive_product_id,
    export_product,
)
from ..wbms.runner import CONTAINER_MODULE_DIR, docker_prefix, summarize_failure

REPO_ROOT = Path(__file__).resolve().parents[3]

STEP_ORDER: tuple[str, ...] = ("register", "detect", "area", "levels", "predict", "deliver")
STEP_LABELS: dict[str, str] = {
    "register": "장면 등록",
    "detect": "수체 탐지",
    "area": "면적 계산",
    "levels": "수위 산출",
    "predict": "시계열 예측",
    "deliver": "규약 파일·로그",
}
# 이 단계가 실패하면 파이프라인 전체가 실패다. 나머지는 부가 산출(경고로 남기고 계속).
FATAL_STEPS = frozenset({"register", "detect", "deliver"})
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed"})

PREDICT_MODEL_ID = "irregular-area-trend"
PREDICT_FALLBACK_MODEL_ID = "persistence"
PREDICT_HORIZON_STEPS = 3
PREDICT_THRESHOLD = 0.5
MIN_PREDICT_FRAMES = 3
LEVELS_SCHEMA = "watercast.satellite.levels/1"
LEVEL_FORECAST_SCHEMA = "watercast.level_forecast/1"  # 2026-10-20 장면 폴더 level_forecast.json
LEVELS_NOTE_KO = "WBMS ③ 위성 상대 수위와 ④ 융합 LSTM 당일 보정 수위. 보정치가 수위계 기준면과 같다."

# ③ calc_wlwa 의 DEM — 인수 패키지 06_aux 기준 상대경로. SAR(ICEYE)는 날짜별 격자 정렬본 디렉터리
# (<date>_DEM_32652.tif), 광학은 NGII 3 m 단일 파일(scripts/wbms_chain_runner.py SENSORS 와 같다).
DEM_BY_TESTBED: dict[str, dict[str, str]] = {
    "busan": {
        "SAR": "DEM_busan_ngii3m/icegrid",
        "OPTIC": "DEM_busan_ngii3m/ngii_busan_32652_3m_topofilled.tif",
    },
    "dcd": {
        # Umbra(SAR, 3 m EPSG:32652)용 날짜별 정렬 DEM 은 없다 — 광학과 같은 NGII 3 m 단일 파일을 준다.
        # calc_wlwa 는 파일이면 그대로 쓰되 WB 와 격자(원점·화소)가 다르면 모듈 안에서 실패하므로, Umbra 전처리
        # (scripts/umbra_sicd_to_pre.py)가 이 DEM 격자에 맞춰 내야 ③ 이 돈다. 실패해도 파이프라인은 계속(경고).
        "SAR": "DEM_daecheong_ngii3m/daecheong_ngii_32652_3m_topofilled.tif",
        "OPTIC": "DEM_daecheong_ngii3m/daecheong_ngii_32652_3m_topofilled.tif",
    },
}
# 컨테이너 3-class 마스크 값(infer_core.apply_ocean_mask): 0=유효 SAR, 1=제외(nodata), 2=바다
MASK3_VALID = 0
MASK3_EXCLUDE = 1
MASK3_GENERATED_WARNING = "3-class 마스크가 없어 영상 유효 범위로 생성(바다 없음 가정)"
_WB_STAMP_RE = re.compile(r"^WB_[^_]+_[^_]+_(\d{8}T\d{6}|\d{8})\.tif$")
_DATE8_RE = re.compile(r"(\d{8})")


class StepSkipped(Exception):
    """단계를 건너뛴다 — 메시지가 사유(status.json 의 detail)다."""



def _runner_writable(path: Path) -> Path:
    """wbms 컨테이너(RUN_AS uid:gid, 보통 1000:1000)가 써야 하는 디렉터리.

    API 컨테이너는 root 로 돌아 만든 디렉터리가 root 소유가 되므로, 다른 uid 로 도는 wbms
    컨테이너가 `/out/_logs` 를 못 만들어 AUX_NOT_FOUND 로 죽는다(서버 실측 2026-10-03).
    만들면서 전원 쓰기 권한을 준다 — 과제용 공유 작업 디렉터리라 허용.
    """

    path.mkdir(parents=True, exist_ok=True)
    try:
        for p in (path, *[a for a in path.parents if a.name in ("wlwa", "wlwa_input") or a == path.parent]):
            if p.is_dir():
                p.chmod(0o777)
    except OSError:
        pass
    return path

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso_date(digits: str) -> str:
    text = str(digits)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) >= 8 and text[:8].isdigit() else text[:10]


def link_or_copy(source: Path, target: Path) -> Path:
    """같은 파일시스템이면 하드링크, 아니면 복사. 이미 같은 파일이면 그대로 둔다."""

    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            if os.path.samefile(source, target):
                return target
        except OSError:
            pass
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def generate_mask3_from_pre(pre_tif: Path, target: Path) -> Path:
    """① 전처리 영상의 유효 범위로 3-class 마스크를 만든다 — 같은 격자(transform·CRS·크기)의 uint8 GeoTIFF.

    0 = 유효 화소(nodata 가 아니고 유한값), 1 = 그 외(촬영범위 밖·nodata·NaN). 바다(2)는 넣지 않는다 — 대청댐처럼
    지형 마스크가 인수되지 않은 내륙 테스트베드(Umbra)에서만 쓰는 대체재다. 컨테이너(detect_water)는 마스크가
    CRS 를 가지면 영상과 격자가 같은지 검사하므로 프로파일을 그대로 복사한다. nodata 태그가 없으면 ICEYE 규약
    -9999 로 본다(scene_assets.read_quicklook 과 같은 기본값).
    """

    import numpy as np
    import rasterio  # 지연 import — 이미 의존성이지만 모듈 로드 시점에는 필요 없다

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(pre_tif) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = {
            "driver": "GTiff", "dtype": "uint8", "count": 1, "width": src.width, "height": src.height,
            "crs": src.crs, "transform": src.transform, "compress": "deflate",
        }
        with rasterio.open(target, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                band = src.read(1, window=window)
                valid = np.isfinite(band)
                if np.isfinite(nodata):
                    valid &= band != nodata
                dst.write(np.where(valid, MASK3_VALID, MASK3_EXCLUDE).astype(np.uint8), 1, window=window)
    return target


_AREA_BUILDER: Any = None
_AREA_BUILDER_LOCK = threading.Lock()


def load_area_builder() -> Any:
    """``scripts/build_common_frame_area.py`` 를 모듈로 올린다(한 번만). scripts/ 는 패키지가 아니라 경로로 읽는다."""

    global _AREA_BUILDER
    with _AREA_BUILDER_LOCK:
        if _AREA_BUILDER is None:
            path = REPO_ROOT / "scripts" / "build_common_frame_area.py"
            spec = importlib.util.spec_from_file_location("watercast_build_common_frame_area", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _AREA_BUILDER = module
    return _AREA_BUILDER


@dataclass(slots=True)
class PipelineServices:
    """파이프라인이 쓰는 기존 서비스 묶음 — ``app.state`` 의 객체를 그대로 참조한다(복제 금지)."""

    regions: Any
    gauge_service: Any
    segmenters: Any
    wbms: Any  # WbmsSegmentationJobAdapter
    prediction_service: Any
    store: Any
    settings: Any
    wbms_settings: Any

    @classmethod
    def from_app(cls, app: Any) -> PipelineServices:
        state = app.state
        wbms = state.wbms_segmentation
        return cls(
            regions=state.regions,
            gauge_service=state.gauge_service,
            segmenters=state.segmenters,
            wbms=wbms,
            prediction_service=state.prediction_service,
            store=state.store,
            settings=state.settings,
            wbms_settings=getattr(state, "wbms_settings", None) or wbms.settings,
        )

    @property
    def api_prefix(self) -> str:
        return str(getattr(self.settings, "api_prefix", "/api/v1")).rstrip("/")

    @property
    def scenes_root(self) -> Path:
        """장면 산출 폴더 루트 — compose 에서 쓰기 가능한 backend/data 아래(``regions_user_dir`` 의 형제)."""

        return Path(self.settings.regions_user_dir).parent / "scenes"


def predict_area_series(
    services: PipelineServices,
    *,
    region_id: str,
    sensor: str,
    scene_id: str,
    model_id: str = PREDICT_MODEL_ID,
    horizon_steps: int = PREDICT_HORIZON_STEPS,
    pipeline_id: str | None = None,
    fallback_model_id: str | None = PREDICT_FALLBACK_MODEL_ID,
    as_of: Date | None = None,
    exclude_scene_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """센서별 공통 격자 프레임 전부로 수체 면적 시계열 예측을 한 번 돌린다 — 파이프라인 predict 단계와
    ``POST /scenes/{scene_id}/predict``(④ 화면의 모델·예측일수 재선택)가 같은 함수를 쓴다.

    * ``as_of`` 를 주면 그 날짜(포함)까지의 프레임만 입력으로 쓴다(시나리오 검증용). 그 뒤 프레임은 ``later_frames``
      (날짜·장면·공통 격자 면적)로 돌려줘 화면이 '이후 실제 면적'과 나란히 그릴 수 있게 한다.
    * ``exclude_scene_ids`` 로 계열에서 뺄 장면(품질이 나쁜 장면 등)을 고를 수 있다 — 출발 장면(``scene_id``)은 빼지 않는다.
      뺀 장면은 요약 ``excluded_scene_ids`` 에 남는다.
    * 프레임이 ``MIN_PREDICT_FRAMES`` 미만이면 ``StepSkipped``(사유에 개수·최소 개수).
    * ``model_id`` 가 등록돼 있지 않으면 ``fallback_model_id`` 로 대체하고 ``notes`` 에 남긴다(파이프라인 기본 동작).
      ``fallback_model_id=None`` 이면 대체하지 않고 ``KeyError``(API 는 422).
    * 예측 결과가 completed 가 아니면 ``RuntimeError``.

    돌려주는 요약(summary)은 status.json 의 ``internal.prediction`` / 장면 폴더 ``prediction.json`` 본문이다.
    """

    region = services.regions.get(region_id)
    payload = region.scenes() or {}
    scenes = [
        s for s in payload.get("scenes") or []
        if s.get("sensor") == sensor and (s.get("frame") or {}).get("path") and s.get("date")
    ]
    scenes.sort(key=lambda s: str(s["date"]))
    excluded = sorted({str(x) for x in (exclude_scene_ids or []) if str(x) != scene_id and any(str(s.get("scene_id")) == str(x) for s in scenes)})
    if excluded:
        scenes = [s for s in scenes if str(s.get("scene_id")) not in excluded]
    later_frames: list[dict[str, Any]] = []
    if as_of is not None:
        cutoff = as_of.isoformat()
        later = [s for s in scenes if str(s["date"])[:10] > cutoff]
        scenes = [s for s in scenes if str(s["date"])[:10] <= cutoff]
        area_rows: dict[str, Any] = {}
        sensor_block = ((region.area_series() or {}).get("sensors") or {}).get(sensor) or {}
        for row in sensor_block.get("rows") or []:
            if row.get("scene_id"):
                area_rows[str(row["scene_id"])] = row.get("area_km2_common_frame")
        later_frames = [
            {
                "date": str(s["date"])[:10], "scene_id": s["scene_id"],
                "water_area_km2": area_rows.get(str(s["scene_id"]), s.get("water_area_km2")),
            }
            for s in later
        ]
    if len(scenes) < MIN_PREDICT_FRAMES:
        where = (f" (원점 {as_of.isoformat()} 까지)" if as_of is not None else "") + (f" (제외 {len(excluded)}개)" if excluded else "")
        raise StepSkipped(f"{sensor} 공통 격자 프레임 {len(scenes)}개{where} — 시계열 예측에는 최소 {MIN_PREDICT_FRAMES}개 필요")
    uploads: list[UploadedBytes] = []
    dates: list[Date] = []
    for scene in scenes:
        path = region.satellite_dir / str(scene["frame"]["path"])
        if not path.is_file():
            raise FileNotFoundError(f"프레임 PNG 없음: {path}")
        uploads.append(UploadedBytes(name=f"{scene['scene_id']}.png", content=path.read_bytes()))
        dates.append(Date.fromisoformat(str(scene["date"])[:10]))
    pixel_area = float((scenes[-1].get("frame") or {}).get("pixel_area_m2") or 9.0)
    service = services.prediction_service
    notes: list[str] = []
    try:
        service.registry.get(model_id)
    except KeyError:
        if fallback_model_id is None or fallback_model_id == model_id:
            raise
        notes.append(f"예측 모델 {model_id!r} 가 등록되어 있지 않아 {fallback_model_id!r} 로 대체")
        model_id = fallback_model_id
        service.registry.get(model_id)  # 대체 모델도 없으면 KeyError 그대로
    loaded = load_mask_sequence(uploads, PREDICT_THRESHOLD)
    result = service.run(
        loaded=loaded,
        model_id=model_id,
        horizon_steps=int(horizon_steps),
        threshold=PREDICT_THRESHOLD,
        source_dates=dates,
        target_dates=[],
        input_metadata={
            "region_id": region_id, "sensor": sensor, "scene_id": scene_id,
            "pipeline_id": pipeline_id, "scene_ids": [s["scene_id"] for s in scenes], "excluded_scene_ids": excluded,
            "source": f"regions/{region_id}/satellite/masks/{sensor} (센서별 공통 격자 이진 마스크)",
            "frame_pixel_area_m2": pixel_area,
        },
        weather=[],
        historical_water_levels_m=[],
        reference_water_levels_m=[],
        evaluation_kind=None,
        evaluation_truth_provenance=None,
        pixel_area_m2=pixel_area,
        water_level_config=None,
        caution_pct=5.0,
        risk_pct=15.0,
        # 수위 연동 어댑터(gauge-level-forecast · level-area-bridge)가 이 장면의 지역·원점을 쓰도록 명시한다 —
        # 비워 두면 어댑터가 '기본 지역'을 골라 다른 지역 지점으로 전망할 수 있다(서버에서 금남교가 뽑힌 사례).
        model_options={"region_id": region_id, **({"as_of": as_of.isoformat()} if as_of is not None else {})},
    )
    if result.status != "completed":
        raise RuntimeError(str(result.error or "예측 실패"))
    prefix = services.api_prefix
    steps = [
        {
            "horizon": step.horizon,
            "target_date": step.target_date.isoformat() if step.target_date else None,
            "water_area_km2": step.water_area_km2,
            "area_change_pct": step.area_change_pct,
            "risk": str(getattr(step.risk, "value", step.risk)),
        }
        for step in result.steps
    ]
    return {
        "prediction_id": result.id, "model_id": model_id, "model_version": result.model_version,
        "status": result.status, "frame_count": len(uploads), "source_dates": [d.isoformat() for d in dates],
        "scene_ids": [s["scene_id"] for s in scenes], "excluded_scene_ids": excluded,
        "horizon_steps": int(horizon_steps), "pixel_area_m2": pixel_area, "steps": steps,
        "as_of": as_of.isoformat() if as_of is not None else None, "later_frames": later_frames,
        # 어댑터가 남긴 설명(수위-면적 다리의 기울기·원점 수위 등) — 화면 ④ 결과 표에 보인다
        "adapter_metadata": dict(getattr(result, "adapter_metadata", None) or {}),
        "warnings": list(result.warnings), "notes": notes, "result_url": f"{prefix}/predictions/{result.id}",
        "created_at": utc_now_iso(),
    }


def prediction_history_entry(summary: dict[str, Any]) -> dict[str, Any]:
    """status.json ``predictions[]`` 이력 한 줄 — 어떤 모델·예측일수(·원점)로 어떤 예측이 나왔는지."""

    entry = {
        "model_id": summary.get("model_id"), "horizon": summary.get("horizon_steps"),
        "prediction_id": summary.get("prediction_id"), "prediction_url": summary.get("result_url"),
        "created_at": summary.get("created_at") or utc_now_iso(),
    }
    if summary.get("as_of"):
        entry["as_of"] = summary["as_of"]  # 시나리오(원점 지정) 예측만 — 기존 이력 모양은 그대로
    return entry


def prediction_detail(summary: dict[str, Any]) -> str:
    """predict 단계의 짧은 결과 설명(status.json steps[].detail)."""

    steps = summary.get("steps") or []
    detail = f"{summary.get('model_id')}, {summary.get('frame_count')}프레임 → {summary.get('horizon_steps')}단계"
    if summary.get("as_of"):
        detail += f", 원점 {summary['as_of']}"
    if steps:
        detail += f" (1단계 {float(steps[0]['water_area_km2']):.2f} km², {float(steps[0]['area_change_pct']):+.1f}%)"
    return detail


class ScenePipelineRunner:
    """레코드(dict, status.json 내용) 하나를 끝까지 돌린다. 변경은 ``lock`` 안에서, 직후 ``save``."""

    def __init__(
        self,
        services: PipelineServices,
        *,
        lock: threading.RLock,
        save: Callable[[dict[str, Any]], None],
        poll_s: float = 0.5,
    ) -> None:
        self.services = services
        self.lock = lock
        self.save = save
        self.poll_s = poll_s

    # ── 공통 보조 ───────────────────────────────────────────────────────────
    def _step(self, record: dict[str, Any], key: str) -> dict[str, Any]:
        for step in record["steps"]:
            if step["key"] == key:
                return step
        raise KeyError(key)

    def _start_step(self, record: dict[str, Any], key: str) -> float:
        with self.lock:
            step = self._step(record, key)
            step["status"] = "running"
            step["started_at"] = utc_now_iso()
            step["detail"] = None
            self.save(record)
        return time.monotonic()

    def _finish_step(self, record: dict[str, Any], key: str, status: str, detail: str | None, started: float | None) -> None:
        with self.lock:
            step = self._step(record, key)
            step["status"] = status
            step["finished_at"] = utc_now_iso()
            if started is not None:
                step["elapsed_s"] = round(time.monotonic() - started, 3)
                if step.get("started_at") is None:
                    step["started_at"] = step["finished_at"]
            step["detail"] = detail
            self.save(record)

    def _warn(self, record: dict[str, Any], message: str) -> None:
        with self.lock:
            if message not in record["warnings"]:
                record["warnings"].append(message)
            self.save(record)

    def _set_output(self, record: dict[str, Any], **values: Any) -> None:
        with self.lock:
            record["outputs"].update(values)
            self.save(record)

    def _liaison(self, record: dict[str, Any]) -> dict[str, str]:
        return {k: str(v) for k, v in (record.get("liaison") or {}).items() if k in LIAISON_KEYS and v not in (None, "")}

    def _absorb_scene(self, record: dict[str, Any], scene: dict[str, Any]) -> None:
        """등록 결과(scenes.json 항목)를 레코드에 반영 — 퀵룩·오버레이 URL, 면적, 마스크 URL."""

        prefix = self.services.api_prefix
        region_id = record["region_id"]
        images = scene.get("images") or {}
        values: dict[str, Any] = {}
        if images.get("input"):
            values["quicklook_url"] = f"{prefix}/regions/{region_id}/satellite/files/{images['input']}"
        if images.get("overlay"):
            values["overlay_url"] = f"{prefix}/regions/{region_id}/satellite/files/{images['overlay']}"
        if scene.get("water_area_km2") is not None:
            values["water_area_km2"] = scene.get("water_area_km2")
        if scene.get("detection_available") and scene.get("source_mask"):
            values["mask_url"] = f"{prefix}/regions/{region_id}/satellite/scenes/{record['scene_id']}/mask"
        with self.lock:
            record["internal"]["scene"] = dict(scene)
            record["outputs"].update(values)
            self.save(record)

    # ── 실행 ────────────────────────────────────────────────────────────────
    def run(self, record: dict[str, Any]) -> None:
        with self.lock:
            record["status"] = "running"
            record["started_at"] = utc_now_iso()
            record["internal"]["run_started_mono"] = time.monotonic()
            self.save(record)
        failed = False
        for key in STEP_ORDER:
            label = STEP_LABELS[key]
            if failed:
                self._finish_step(record, key, "skipped", "앞 단계 실패로 건너뜀", None)
                continue
            started = self._start_step(record, key)
            handler = getattr(self, f"step_{key}")
            try:
                detail = handler(record)
            except StepSkipped as skip:
                self._finish_step(record, key, "skipped", str(skip), started)
            except Exception as exc:  # noqa: BLE001 - 단계 실패는 상태로 기록
                text = str(exc).strip() or exc.__class__.__name__
                self._finish_step(record, key, "failed", text, started)
                if key in FATAL_STEPS:
                    failed = True
                    with self.lock:
                        record["error"] = f"{label} 실패: {text}"
                        self.save(record)
                else:
                    self._warn(record, f"{label} 실패(파이프라인은 계속): {text}")
            else:
                self._finish_step(record, key, "done", detail, started)
        with self.lock:
            record["status"] = "failed" if failed else "done"
            record["finished_at"] = utc_now_iso()
            self.save(record)

    # ── 1. register ─────────────────────────────────────────────────────────
    def step_register(self, record: dict[str, Any]) -> str:
        internal = record["internal"]
        work_dir = Path(internal["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        stem = str(internal["pre_stem"])
        pre_source = Path(internal["pre_source"])
        if not pre_source.is_file():
            raise FileNotFoundError(f"전처리 영상이 없음: {pre_source}")
        if internal.get("pre_is_upload"):
            pre_tif = pre_source
        else:
            # 서버 경로 등록: 인수 패키지처럼 사이드카가 이미 옆에 있으면 제자리에서 쓰고, 없으면 작업 폴더로
            # 하드링크(또는 복사)해 규약 이름을 붙인다 — detect_water 는 입력 옆의 _meta.json 을 요구한다(X-12).
            existing_sidecar = pre_source.with_name(pre_source.stem + "_meta.json")
            pre_tif = pre_source if existing_sidecar.is_file() else link_or_copy(pre_source, work_dir / f"{stem}.tif")
        sidecar = pre_tif.with_name(pre_tif.stem + "_meta.json")
        if not sidecar.is_file():
            stamp = str(internal["acquisition_stamp"])  # YYYYMMDDTHHMMSS
            acquired = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z"
            meta = {
                "Document": "PreprocessedImage",
                "TestbedName": internal.get("testbed_code"),
                "SatelliteName": record["sensor"],
                "ImageDate": internal["date_digits"],
                "Entry": {"date": internal["date_digits"], "AcquisitionDateTime": acquired},
                "Provenance": {
                    "stage": "upload",
                    "input_file": internal.get("pre_original_name") or (record.get("liaison") or {}).get("product_id") or pre_tif.name,
                    "pipeline_id": record["pipeline_id"],
                },
            }
            write_json(sidecar, meta)
        mask = Path(internal["mask_source"]) if internal.get("mask_source") else None
        if mask is not None and not mask.is_file():
            raise FileNotFoundError(f"마스크 파일이 없음: {mask}")
        gt = Path(internal["gt_source"]) if internal.get("gt_source") else None
        result = self.services.gauge_service.register_scene(
            record["region_id"],
            sensor=record["sensor"],
            sensor_type=record["sensor_type"],
            image_date=str(internal["date_digits"]),
            pre_tif=pre_tif,
            mask_tif=mask,
            gt_tif=gt,
            source_note=record.get("note"),
        )
        with self.lock:
            internal["pre_tif"] = str(pre_tif)
            internal["pre_meta"] = str(sidecar)
            if mask is not None:
                internal["mask_tif"] = str(mask)
                internal["mask_origin"] = "upload"
            self.save(record)
        self._absorb_scene(record, result["scene"])
        detail = pre_tif.name + (f", 마스크 {mask.name}" if mask is not None else ", 사진만")
        if result.get("frame_grid_reused"):
            detail += " (기존 공통 격자 재사용)"
        return detail

    # ── 2. detect ───────────────────────────────────────────────────────────
    def step_detect(self, record: dict[str, Any]) -> str:
        internal = record["internal"]
        if internal.get("mask_source"):
            raise StepSkipped("업로드한 마스크를 사용 — 탐지 생략")
        if not (record.get("options") or {}).get("detect", True):
            raise StepSkipped("detect=false — 탐지 생략")
        testbed = internal.get("testbed")
        if not testbed:
            raise RuntimeError("지역 정의에 satellite.wbms_testbed 가 없어 탐지 어댑터를 고를 수 없음")
        sensor_key = str(record["sensor_key"])
        date_digits = str(internal["date_digits"])
        adapter = self.services.segmenters.resolve(sensor_key, testbed)
        if adapter is None:
            raise RuntimeError(f"'{record['sensor']}' × '{testbed}' 를 받는 탐지 어댑터가 없음")
        liaison = self._liaison(record)
        wbms = self.services.wbms
        if isinstance(adapter, WbmsDockerSegmenter):
            # 내장 WBMS 어댑터: 3-class 마스크 대체(mask_tif)까지 실어야 하므로 잡 어댑터에 직접 제출한다
            body: dict[str, Any] = {
                "sensor": sensor_key, "testbed": testbed, "image_date": date_digits,
                "input_tif": str(internal["pre_tif"]),
                # 새 영상을 올린 것이므로 같은 날짜의 완료 마커가 있어도 다시 돈다(모듈 --force)
                "force": True, **liaison,
            }
            profile = wbms.catalog.sensor(sensor_key)
            if profile.get("needs_mask"):
                mask3 = self._pick_mask3(record, wbms, testbed, date_digits)
                if mask3 is not None:
                    body["mask_tif"] = mask3
            job = wbms.submit(body)
            fetch: Callable[[], dict[str, Any]] = lambda: wbms.get(job_id)
        else:
            submission = adapter.submit(
                SceneRequest(sensor=sensor_key, testbed=testbed, image_date=date_digits,
                             input_path=Path(internal["pre_tif"]), options=liaison)
            )
            job = dict(submission.raw) if submission.raw else {
                "job_id": submission.job_id, "status": submission.status, "cached": submission.cached,
                "result": submission.result, "warnings": list(submission.warnings),
            }
            fetch = lambda: adapter.get_job(job_id)
        job_id = str(job["job_id"])
        with self.lock:
            internal["detection_job_id"] = job_id
            self._step(record, "detect")["detail"] = f"탐지 잡 {job_id} 실행 중"
            self.save(record)
        timeout_s = float(getattr(self.services.wbms_settings, "segmentation_timeout_s", 3600.0)) + 60.0
        deadline = time.monotonic() + timeout_s
        while str(job.get("status")) not in TERMINAL_JOB_STATUSES:
            if time.monotonic() > deadline:
                raise TimeoutError(f"탐지 잡 {job_id} 이 {timeout_s:.0f} s 안에 끝나지 않음")
            time.sleep(self.poll_s)
            job = fetch()
        if job.get("status") != "succeeded":
            raise RuntimeError(str(job.get("error") or "탐지 잡 실패"))
        result = dict(job.get("result") or {})
        mask_tif = Path(str(result.get("mask_tif") or ""))
        if not mask_tif.is_file():
            raise RuntimeError(f"탐지 잡은 성공했지만 마스크 파일이 없음: {mask_tif}")
        for warning in job.get("warnings") or []:
            text = str(warning)
            if "mounted read-only at /scene" in text or "mounted read-only at /mask3" in text:
                continue  # 업로드 입력·작업 폴더의 마스크는 늘 인수 패키지 밖에 있다 — 파이프라인에서는 정보가 아니다
            self._warn(record, text)
        with self.lock:
            internal["mask_tif"] = str(mask_tif)
            internal["mask_origin"] = "detect"
            internal["product_tif"] = result.get("product_tif")
            internal["product_id"] = result.get("product_id")
            internal["detection"] = {
                "job_id": job_id, "cached": bool(job.get("cached")), "source": result.get("source"),
                "elapsed_s": result.get("elapsed_s"), "output_dir": result.get("output_dir"),
                "qc_json": result.get("qc_json"), "meta_json": result.get("meta_json"),
            }
            self.save(record)
        # 마스크가 생겼으니 같은 scene_id 로 다시 등록 → 오버레이·공통 격자 프레임·면적
        registered = self.services.gauge_service.register_scene(
            record["region_id"], sensor=record["sensor"], sensor_type=record["sensor_type"],
            image_date=date_digits, pre_tif=Path(internal["pre_tif"]), mask_tif=mask_tif,
            gt_tif=Path(internal["gt_source"]) if internal.get("gt_source") else None,
            source_note=record.get("note"),
        )
        self._absorb_scene(record, registered["scene"])
        area = registered["scene"].get("water_area_km2")
        detail = mask_tif.name + (f", {float(area):.2f} km²" if area is not None else "")
        if job.get("cached"):
            detail += " (캐시)"
        elapsed = result.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            detail += f", 컨테이너 {float(elapsed):.0f} s"
        return detail

    def _pick_mask3(self, record: dict[str, Any], wbms: Any, testbed: str, date_digits: str) -> str | None:
        """SAR 3-class 마스크 고르기 — 업로드(mask3_file) → 날짜별 인수 마스크 → 인접 날짜 → 영상 유효 범위로 생성.

        None 을 돌려주면 어댑터가 카탈로그 템플릿의 날짜별 마스크(/aux)를 그대로 쓴다. 그 밖은 호스트 경로를
        돌려주고 ``internal.mask3_tif``·``mask3_origin``(upload|delivered|nearest|generated) 에 남긴다.
        """

        internal = record["internal"]
        uploaded = internal.get("mask3_source")
        if uploaded:
            path = Path(uploaded)
            if not path.is_file():
                raise FileNotFoundError(f"업로드한 3-class 마스크가 없음: {path}")
            self._note_mask3(record, path, "upload")
            return str(path)
        expected = wbms.mask3_path(testbed, date_digits)
        if expected.is_file():
            self._note_mask3(record, expected, "delivered")
            return None
        nearest = wbms.nearest_mask3(testbed, date_digits)
        if nearest is not None:
            found = _DATE8_RE.search(nearest.name)
            near_date = found.group(1) if found else nearest.name
            self._warn(
                record,
                f"{date_digits} 의 3-class 마스크가 없어 {near_date} 것을 대신 사용 "
                "(바다·내륙수·육지 지형 분류라 날짜 간 변화가 거의 없다)",
            )
            self._note_mask3(record, nearest, "nearest")
            return str(nearest)
        # 인수된 지형 마스크가 하나도 없는 테스트베드(대청댐·Umbra): 영상의 유효 범위를 그대로 마스크로 쓴다.
        generated = generate_mask3_from_pre(
            Path(internal["pre_tif"]), Path(internal["work_dir"]) / f"{testbed}_{date_digits}_mask3_utm.tif"
        )
        self._warn(record, MASK3_GENERATED_WARNING)
        self._note_mask3(record, generated, "generated")
        return str(generated)

    def _note_mask3(self, record: dict[str, Any], path: Path, origin: str) -> None:
        with self.lock:
            record["internal"]["mask3_tif"] = str(path)
            record["internal"]["mask3_origin"] = origin
            self.save(record)

    def _module_sensor(self, record: dict[str, Any]) -> tuple[str, str]:
        """(컨테이너 ``--sensor`` 키, 컨테이너가 산출물에 적는 표기). 빌린 모델로 도는 센서(umbra)는 (iceye, ICEYE).

        카탈로그에 없는 센서(외부 플러그인 어댑터로 탐지한 장면)는 기록값을 그대로 쓴다.
        """

        key = str(record.get("sensor_key") or "").lower()
        catalog = getattr(self.services.wbms, "catalog", None)
        try:
            return catalog.module_sensor(key), catalog.module_name(key)
        except (AttributeError, KeyError):
            return key or str(record["sensor"]).lower(), str(record["sensor"])

    # ── 3. area ─────────────────────────────────────────────────────────────
    def step_area(self, record: dict[str, Any]) -> str:
        internal = record["internal"]
        scene = internal.get("scene") or {}
        if not scene.get("frame"):
            raise StepSkipped("마스크가 없어 면적을 셀 수 없음")
        water = scene.get("water_area_km2")
        region_id = record["region_id"]
        prefix = self.services.api_prefix
        common: float | None = None
        rows: list[dict[str, Any]] = []
        try:
            builder = load_area_builder()
            region = self.services.regions.get(region_id)
            out = builder.build_region(region_id, verbose=False, region_dir=str(region.path))
            block = ((out or {}).get("sensors") or {}).get(record["sensor"]) or {}
            rows = [dict(r) for r in block.get("rows") or []]
            common = next((r.get("area_km2_common_frame") for r in rows if r.get("scene_id") == record["scene_id"]), None)
        except Exception as exc:  # noqa: BLE001 - 계열 갱신 실패는 경고(장면 면적 자체는 이미 안다)
            self._warn(record, f"공통 격자 면적 계열 갱신 실패: {type(exc).__name__}: {exc}")
        with self.lock:
            internal["area"] = {
                "water_area_km2": water, "valid_area_km2": scene.get("valid_area_km2"),
                "area_common_frame_km2": common, "series_rows": rows,
            }
            if common is not None:
                internal["scene"]["area_common_frame_km2"] = common
            record["outputs"]["area_series_url"] = f"{prefix}/regions/{region_id}/satellite/area-series"
            if water is not None:
                record["outputs"]["water_area_km2"] = water
            self.save(record)
        detail = f"{float(water):.2f} km²" if water is not None else "면적 미상"
        if common is not None:
            detail += f" (공통 격자 {float(common):.2f} km², {record['sensor']} 계열 {len(rows)}장면)"
        return detail

    # ── 4. levels (③ calc_wlwa) ─────────────────────────────────────────────
    def step_levels(self, record: dict[str, Any]) -> str:
        internal = record["internal"]
        mask = internal.get("mask_tif")
        if not mask:
            raise StepSkipped("마스크가 없어 수위를 산출할 수 없음")
        wbms = self.services.wbms
        wbms_settings = self.services.wbms_settings
        testbed = str(internal.get("testbed") or "")
        if not testbed or testbed not in wbms.catalog.testbeds:
            raise StepSkipped(f"지역 '{record['region_id']}' 에 WBMS 테스트베드가 없어 ③ calc_wlwa 를 돌릴 수 없음")
        sensor_type = str(record["sensor_type"]).upper()
        dem_rel = (DEM_BY_TESTBED.get(testbed) or {}).get(sensor_type)
        if not dem_rel:
            raise StepSkipped(f"테스트베드 '{testbed}' × {sensor_type} 용 DEM 경로가 정의되지 않음 (pipeline.steps.DEM_BY_TESTBED)")
        dem_host = Path(wbms_settings.aux_dir) / dem_rel
        if not dem_host.exists():
            raise StepSkipped(f"③ DEM 없음: {dem_host}")
        date_digits = str(internal["date_digits"])
        if dem_host.is_dir():
            dated = dem_host / f"{date_digits}_DEM_32652.tif"
            if not dated.is_file():
                raise StepSkipped(
                    f"{date_digits} 용 날짜별 정렬 DEM 없음: {dated} — SAR 는 장면 격자에 맞춘 DEM 이 있어야 ③ 이 돈다"
                )
        stations = [str(s) for s in (wbms.catalog.testbed(testbed).get("stations") or [])]
        if not stations:
            raise StepSkipped(f"테스트베드 '{testbed}' 에 AOI 지점 목록이 없음")
        wb_input = self._stage_wb_input(record, Path(mask))
        tb_code = str(wbms.catalog.testbed(testbed).get("code") or testbed.capitalize())
        sensor_key = str(record["sensor_key"])
        # 컨테이너 --sensor 와 산출물 표기: 빌린 모델로 도는 센서(umbra)는 iceye/ICEYE 로 돈다(카탈로그 module_sensor).
        # 호스트 출력 디렉터리(wlwa/<sensor_key>)와 levels 기록의 satellite 는 카탈로그 키·표시명을 그대로 쓴다.
        module_sensor, module_name = self._module_sensor(record)
        dem_container = f"/aux/{dem_rel}"
        fusion_root = Path(wbms_settings.fusion_work_dir)
        records: list[dict[str, Any]] = []
        failures: list[str] = []
        notes: list[str] = []
        timeout_s = float(getattr(wbms_settings, "fusion_timeout_s", 600.0))
        for loc in stations:
            out_dir = fusion_root / "wlwa" / sensor_key / loc
            _runner_writable(out_dir)
            mounts: list[tuple[Path, str, bool]] = [
                (Path(wbms_settings.aux_dir), "/aux", True),
                (wb_input, "/wb", True),
                (out_dir, "/out", False),
            ]
            argv = docker_prefix(wbms_settings.docker_bin, wbms_settings.docker_image, mounts)
            argv += [
                "python3", f"{CONTAINER_MODULE_DIR}/calc_wlwa.py",
                "--wb_result_dir", "/wb",
                "--sensor", module_sensor, "--testbed", testbed, "--loc_id", loc,
                "--start_date", date_digits, "--end_date", date_digits,
                "--dem", dem_container, "--output_dir", "/out",
            ]
            result = wbms.runner.run(argv, timeout_s=timeout_s)
            if result.returncode != 0:
                failures.append(f"{loc}: {summarize_failure(result)}")
                continue
            stem = f"WLWA_{tb_code}_{module_name}_{loc}_{date_digits}_{date_digits}"
            qc_path = out_dir / f"{stem}_qc.json"
            doc_path = out_dir / f"{stem}.json"
            qc = self._read_json(qc_path)
            if qc.get("status") != "COMPLETE" or not doc_path.is_file():
                notes.append(f"{loc}: 산출 없음(AOI 커버리지 미달 또는 기간 내 WB 없음)")
                continue
            doc = self._read_json(doc_path)
            for raw in doc.get("records") or []:
                if str(raw.get("date") or "")[:8] != date_digits:
                    continue
                records.append(self._level_record(record, raw, loc, doc_path))
        if not records:
            if failures:
                raise RuntimeError("; ".join(failures))
            raise StepSkipped("; ".join(notes) or "③ calc_wlwa 산출 없음")
        for failure in failures:
            self._warn(record, f"③ calc_wlwa 실패: {failure}")
        self._merge_levels(record, records)
        iso = iso_date(date_digits)
        prefix = self.services.api_prefix
        with self.lock:
            internal["levels"] = records
            record["outputs"]["levels_url"] = (
                f"{prefix}/regions/{record['region_id']}/satellite/levels?start={iso}&end={iso}"
            )
            self.save(record)
        parts = [
            f"{r['loc_id']} {float(r['water_level_m']):.3f} m" for r in records if r.get("water_level_m") is not None
        ]
        return f"{len(records)}지점 — " + ", ".join(parts) if parts else f"{len(records)}지점(수위 없음)"

    def _stage_wb_input(self, record: dict[str, Any], mask: Path) -> Path:
        """③ 입력 디렉터리를 장면 하나짜리로 꾸민다 — calc_wlwa 는 ``WB_*_qc.json``(COMPLETE)·``.tif``·``_meta.json`` 을 본다.

        탐지 산출이면 옆의 메타·QC 를 그대로 잇고, 업로드 마스크면 메타(촬영 시각·격자 원점)를 합성한다.
        """

        internal = record["internal"]
        wbms = self.services.wbms
        testbed = str(internal["testbed"])
        tb_code = str(wbms.catalog.testbed(testbed).get("code") or testbed.capitalize())
        # calc_wlwa 는 WB_<TB>_<KEY2NAME[--sensor]>_* 만 모은다 — 빌린 모델로 도는 센서(umbra)는 ICEYE 표기로 꾸민다
        _module_key, sensor_name = self._module_sensor(record)
        found = _WB_STAMP_RE.match(mask.name)
        stamp = found.group(1) if found else str(internal["acquisition_stamp"])
        base = f"WB_{tb_code}_{sensor_name}_{stamp}"
        wb_input = Path(self.services.wbms_settings.fusion_work_dir) / "wlwa_input" / record["scene_id"]
        if wb_input.exists():
            shutil.rmtree(wb_input)
        _runner_writable(wb_input)
        link_or_copy(mask, wb_input / f"{base}.tif")
        source_meta = mask.with_name(mask.stem + "_meta.json")
        if source_meta.is_file():
            link_or_copy(source_meta, wb_input / f"{base}_meta.json")
        else:
            write_json(wb_input / f"{base}_meta.json", self._synth_wb_meta(record, mask, stamp))
        write_json(wb_input / f"{base}_qc.json", {
            "base": base, "status": "COMPLETE",
            "products": [f"{base}.tif", f"{base}_meta.json"],
            "params": {"sensor_name": sensor_name, "testbed": testbed, "image_date": str(internal["date_digits"]),
                       "sensor_catalog_key": record.get("sensor_key")},
            "note_ko": "장면 파이프라인이 ③ calc_wlwa 입력용으로 꾸민 완료 마커(원 마스크 하드링크/복사)",
            "source_mask": str(mask), "pipeline_id": record["pipeline_id"],
        })
        with self.lock:
            internal["wlwa_input_dir"] = str(wb_input)
            self.save(record)
        return wb_input

    def _synth_wb_meta(self, record: dict[str, Any], mask: Path, stamp: str) -> dict[str, Any]:
        internal = record["internal"]
        acquired = None
        if len(stamp) == 15 and stamp[8] == "T":
            acquired = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z"
        entry: dict[str, Any] = {"date": str(internal["date_digits"]), "AcquisitionDateTime": acquired}
        try:
            import rasterio  # 지연 import

            with rasterio.open(mask) as ds:
                entry["Scene_UL"] = {"x": str(ds.transform.c), "y": str(ds.transform.f)}
                entry["Scene_Posting"] = {"x": str(abs(ds.transform.a)), "y": str(abs(ds.transform.e))}
                entry["CRS"] = str(ds.crs) if ds.crs else None
                entry["PixelSize_m"] = float(abs(ds.transform.a))
        except Exception as exc:  # noqa: BLE001 - 메타 없이도 ③ 은 돈다(read_product_meta 는 관대하다)
            entry["meta_note"] = f"격자 메타를 읽지 못함: {type(exc).__name__}: {exc}"
        return {
            "Document": "WaterBodyMap",
            "TestbedName": internal.get("testbed_code"),
            "SatelliteName": record["sensor"],
            "ImageDate": str(internal["date_digits"]),
            "Entry": entry,
            "Provenance": {"stage": "upload", "input_file": mask.name, "pipeline_id": record["pipeline_id"]},
        }

    def _level_record(self, record: dict[str, Any], raw: dict[str, Any], loc: str, doc_path: Path) -> dict[str, Any]:
        corrected = raw.get("corrected_water_level_m")
        try:
            source = str(doc_path.relative_to(Path(self.services.wbms_settings.fusion_work_dir).parent))
        except ValueError:
            source = str(doc_path)
        return {
            "date": iso_date(str(raw.get("date") or record["internal"]["date_digits"])),
            "loc_id": loc,
            "satellite": record["sensor"],
            # 카탈로그 키와 ③ 이 산출물에 적은 표기 — 빌린 모델로 돈 센서(umbra)는 module_satellite 가 ICEYE 다
            "sensor_catalog_key": record.get("sensor_key"),
            "module_satellite": raw.get("satellite"),
            "water_level_m": raw.get("water_level_m"),
            "water_area_km2": raw.get("water_area_km2"),
            "corrected_water_level_m": corrected,
            "offset_m": raw.get("offset_m"),
            "correction_mode": raw.get("correction_mode") or "none",
            "quality": raw.get("quality"),
            "acq_datetime_utc": raw.get("acq_datetime_utc"),
            "uncertainty_m": raw.get("uncertainty_m"),
            "aoi_coverage": raw.get("aoi_coverage"),
            "source_wb": raw.get("source_wb"),
            "source": source,
            "pipeline_id": record["pipeline_id"],
            "correction_note": (
                "③ calc_wlwa 원 수위(AOI 물가선 DEM 표고 중앙값) — 장면 파이프라인 산출, ④ 융합 보정 미적용"
                if corrected is None else "③ calc_wlwa 산출 + 모듈 내 보정(run.yaml 보정표)"
            ),
        }

    def _merge_levels(self, record: dict[str, Any], new_records: list[dict[str, Any]]) -> Path:
        """지역 사용자 사본의 satellite/levels.json 에 병합 — 같은 (date, loc_id, satellite) 는 교체, 나머지는 유지."""

        region_id = record["region_id"]
        sat_dir = self.services.regions.satellite_write_dir(region_id)
        path = sat_dir / "levels.json"
        payload = self._read_json(path) if path.is_file() else {}
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            payload = {"schema": LEVELS_SCHEMA, "region_id": region_id, "note_ko": LEVELS_NOTE_KO, "records": []}
        payload.setdefault("schema", LEVELS_SCHEMA)
        payload.setdefault("region_id", region_id)

        def key(item: dict[str, Any]) -> tuple[str, str, str]:
            return (iso_date(str(item.get("date") or "")), str(item.get("loc_id") or "").lower(), str(item.get("satellite") or "").lower())

        replaced = {key(item) for item in new_records}
        kept = [item for item in payload["records"] if isinstance(item, dict) and key(item) not in replaced]
        merged = kept + [dict(item) for item in new_records]
        merged.sort(key=lambda item: (iso_date(str(item.get("date") or "")), str(item.get("loc_id") or ""), str(item.get("satellite") or "")))
        payload["records"] = merged
        payload["updated_at"] = utc_now_iso()
        write_json(path, payload)
        with self.lock:
            record["internal"]["levels_file"] = str(path)
            self.save(record)
        return path

    # ── 5. predict ──────────────────────────────────────────────────────────
    def step_predict(self, record: dict[str, Any]) -> str:
        internal = record["internal"]
        if not (record.get("options") or {}).get("predict", True):
            raise StepSkipped("predict=false — 예측 생략")
        if not internal.get("mask_tif"):
            raise StepSkipped("이 장면의 마스크가 없어 공통 격자 프레임을 만들 수 없음")
        # 예측 자체는 단독 함수(④ 화면의 모델 재예측과 공용). 기본 모델·지평은 파이프라인 기본값 그대로.
        summary = predict_area_series(
            self.services,
            region_id=str(record["region_id"]), sensor=str(record["sensor"]), scene_id=str(record["scene_id"]),
            pipeline_id=record.get("pipeline_id"),
        )
        for note in summary.get("notes") or []:
            self._warn(record, note)
        prefix = self.services.api_prefix
        with self.lock:
            internal["prediction"] = summary
            record["outputs"]["prediction_id"] = summary["prediction_id"]
            record["outputs"]["prediction_url"] = f"{prefix}/predictions/{summary['prediction_id']}"
            record.setdefault("predictions", []).append(prediction_history_entry(summary))
            self.save(record)
        return prediction_detail(summary)

    # ── 6. deliver ──────────────────────────────────────────────────────────
    def _level_forecast_payload(self, region_id: str, scene_id: str, iso: str) -> dict[str, Any]:
        """장면 폴더의 ``level_forecast.json`` — 촬영일을 원점으로, 위성 구역과 연결된 지상 수위계 지점마다 일 단위 수위 전망
        (``GET /regions/{id}/gauge/{station}/forecast`` 와 같은 본문, history 제외). 촬영일이 자료 밖이면 서비스 규칙대로
        자료 끝에서 출발하고 ``as_of_clamped``·``origin_note_ko`` 로 알린다. 지상 자료·지점 연결이 없으면 stations=[] 와 사유."""

        out: dict[str, Any] = {
            "schema": LEVEL_FORECAST_SCHEMA, "region_id": region_id, "scene_id": scene_id, "as_of_requested": iso,
            "stations": [], "n": 0, "n_ok": 0,
            "note_ko": "촬영일을 원점으로 한 지상 수위계 일 단위 전망(검증에서 고른 모델). 같은 값은 GET /regions/{id}/gauge/{station}/forecast?as_of=촬영일 로도 받는다. "
                       "시 단위 전망은 GET …/forecast-hourly.",
        }
        try:
            region = self.services.regions.get(region_id)
        except Exception as exc:  # noqa: BLE001
            out["reason_ko"] = f"지역을 읽지 못했습니다: {exc}"
            return out
        manifest = getattr(region, "manifest", None) or {}
        gauge_cfg = manifest.get("gauge") or {}
        links = dict(gauge_cfg.get("satellite_location_links") or {})
        if not gauge_cfg.get("available", True) or not links:
            out["reason_ko"] = "이 지역은 위성 구역과 연결된 지상 수위계 지점이 없습니다(region.json gauge.satellite_location_links)."
            return out
        service = getattr(self.services, "gauge_service", None)
        if service is None or not hasattr(service, "forecast"):
            out["reason_ko"] = "수위계 서비스가 없습니다."
            return out
        try:
            as_of = Date.fromisoformat(str(iso)[:10])
        except ValueError:
            as_of = None
        for station_id, loc_id in links.items():
            entry: dict[str, Any] = {"station_id": station_id, "loc_id": loc_id}
            try:
                payload = service.forecast(region_id, station_id, as_of=as_of, horizon=None, history_days=1)
            except (KeyError, FileNotFoundError) as exc:  # 학습 산출물(models/gauge/<지점>)이 없다
                entry.update(status="failed", error=f"지점 {station_id} 의 학습 산출물이 없습니다({exc}) — POST /regions/{region_id}/gauge/train 로 먼저 학습하세요")
                out["stations"].append(entry)
                continue
            except Exception as exc:  # noqa: BLE001 — 한 지점 실패가 납품을 막지 않는다
                entry.update(status="failed", error=str(exc) or exc.__class__.__name__)
                out["stations"].append(entry)
                continue
            entry["status"] = "ok"
            entry.update({k: v for k, v in dict(payload).items() if k != "history"})
            out["stations"].append(entry)
        out["n"] = len(out["stations"])
        out["n_ok"] = sum(1 for e in out["stations"] if e.get("status") == "ok")
        return out

    def step_deliver(self, record: dict[str, Any]) -> str:
        internal = record["internal"]
        folder = Path(internal["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        prefix = self.services.api_prefix
        region_id = record["region_id"]
        scene_id = record["scene_id"]
        files: list[str] = []
        product_name: str | None = None
        mask = internal.get("mask_tif")
        if mask:
            mask_path = Path(mask)
            product_tif = internal.get("product_tif")
            if not product_tif or not Path(str(product_tif)).is_file():
                # 업로드 마스크(또는 탐지 결과에 규약 사본이 붙지 않은 경우): 직접 규약 파일을 만든다
                liaison = self._liaison(record)
                product_type = liaison.get("product_type") or PRODUCT_TYPE_WATER_MASK
                product_id = liaison.get("product_id") or derive_product_id(internal.get("pre_tif"))
                if product_id:
                    target = export_product(
                        mask_path, Path(self.services.wbms_settings.product_dir),
                        product_type=product_type, product_id=str(product_id),
                    )
                    product_tif = str(target)
                    with self.lock:
                        internal["product_tif"] = product_tif
                        internal["product_id"] = str(product_id)
                        self.save(record)
                    if internal.get("mask_origin") == "upload":
                        # 탐지 잡은 어댑터가 이미 완료 로그를 보냈다 — 업로드 마스크일 때만 여기서 1건
                        self._log_completion(record, target)
                else:
                    self._warn(record, "product_id 를 파생하지 못해 규약 파일을 만들지 않음 (product_id 를 지정하세요)")
            if product_tif and Path(str(product_tif)).is_file():
                copied = link_or_copy(Path(str(product_tif)), folder / Path(str(product_tif)).name)
                product_name = copied.name
                files.append(copied.name)
            mask_copy = link_or_copy(mask_path, folder / mask_path.name)
            files.append(mask_copy.name)
        scene = dict(internal.get("scene") or {})
        try:
            sat_dir = self.services.regions.get(region_id).satellite_dir
        except Exception:  # noqa: BLE001
            sat_dir = None
        images = scene.get("images") or {}
        if sat_dir is not None:
            for kind, name in (("input", "quicklook.png"), ("overlay", "overlay.png")):
                rel = images.get(kind)
                if rel and (sat_dir / rel).is_file():
                    shutil.copyfile(sat_dir / rel, folder / name)
                    files.append(name)
        iso = record["image_date"]
        area = internal.get("area") or {"water_area_km2": scene.get("water_area_km2"), "series_rows": []}
        write_json(folder / "area.json", {
            "scene_id": scene_id, "region_id": region_id, "sensor": record["sensor"], "date": iso,
            "water_area_km2": area.get("water_area_km2", scene.get("water_area_km2")),
            "valid_area_km2": area.get("valid_area_km2", scene.get("valid_area_km2")),
            "area_common_frame_km2": area.get("area_common_frame_km2", scene.get("area_common_frame_km2")),
            "series": area.get("series_rows") or [],
            "note_ko": "water_area_km2 는 장면 촬영범위 안 면적, area_common_frame_km2 는 센서별 공통 격자 면적(날짜 간 비교용).",
        })
        files.append("area.json")
        levels = internal.get("levels") or []
        write_json(folder / "levels.json", {
            "schema": LEVELS_SCHEMA, "region_id": region_id, "scene_id": scene_id, "date": iso,
            "records": levels, "n": len(levels),
            "note_ko": "이 장면(촬영일)의 ③ 위성 수위 기록. 지역 전체 기록은 levels_url.",
        })
        files.append("levels.json")
        prediction = internal.get("prediction")
        write_json(folder / "prediction.json", prediction or {
            "prediction_id": None, "status": "skipped", "reason": self._step(record, "predict").get("detail"),
        })
        files.append("prediction.json")
        # 2026-10-20: 연계 모듈이 장면 폴더만 보고도 수위 전망을 가져가도록 ④ 지상 수위계 전망을 파일로 같이 둔다
        # (prediction.json 은 면적 전망만 담는다 — 다른 모듈이 수위 전망을 못 가져가던 원인).
        write_json(folder / "level_forecast.json", self._level_forecast_payload(region_id, scene_id, iso))
        files.append("level_forecast.json")
        write_json(folder / "scene.json", scene)
        files.append("scene.json")
        try:
            folder_rel = str(folder.relative_to(self.services.scenes_root.parent))
        except ValueError:
            folder_rel = str(folder)
        values: dict[str, Any] = {"folder": folder_rel, "product_name": product_name}
        values["product_url"] = f"{prefix}/scenes/{scene_id}/files/{product_name}" if product_name else None
        if mask and scene.get("detection_available"):
            values["mask_url"] = f"{prefix}/regions/{region_id}/satellite/scenes/{scene_id}/mask"
        self._set_output(record, **values)
        with self.lock:
            internal["files"] = sorted(set(files)) + ["status.json"]
            self.save(record)
        return (product_name or "규약 파일 없음(마스크 없음)") + f" · {len(files) + 1}개 파일 → {folder_rel}"

    def _log_completion(self, record: dict[str, Any], product: Path) -> None:
        """통합 플랫폼 완료 로그 1건 — 전송 실패는 경고로만."""

        logger = getattr(self.services.wbms, "liaison", None)
        if logger is None:
            return
        internal = record["internal"]
        started = internal.get("run_started_mono")
        elapsed = round(time.monotonic() - float(started), 1) if isinstance(started, (int, float)) else None
        job_like = {
            "job_id": record["pipeline_id"],
            "request": self._liaison(record),
            "started_at": record.get("started_at") or record.get("created_at"),
            "finished_at": utc_now_iso(),
            "result": {"elapsed_s": elapsed},
        }
        warning = logger.log_job(job_like, status="succeeded", msg=f"{product.name} 산출물이 생성되었습니다.")
        with self.lock:
            internal["liaison_log_sent"] = bool(getattr(logger, "enabled", False)) and warning is None
            self.save(record)
        if warning:
            self._warn(record, str(warning))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}


__all__ = [
    "DEM_BY_TESTBED",
    "FATAL_STEPS",
    "MASK3_GENERATED_WARNING",
    "MIN_PREDICT_FRAMES",
    "PREDICT_HORIZON_STEPS",
    "PREDICT_MODEL_ID",
    "STEP_LABELS",
    "STEP_ORDER",
    "PipelineServices",
    "ScenePipelineRunner",
    "StepSkipped",
    "generate_mask3_from_pre",
    "iso_date",
    "link_or_copy",
    "load_area_builder",
    "predict_area_series",
    "prediction_detail",
    "prediction_history_entry",
    "utc_now_iso",
    "write_json",
]
