"""장면 파이프라인 잡 저장소 — 장면당 백그라운드 스레드 1개, 상태는 장면 폴더의 status.json.

``backend/app/gauge/jobs.py`` 의 ``GaugeTrainJobs`` 와 같은 꼴(ThreadPoolExecutor + 잠금)이되,
상태를 메모리에만 두지 않고 ``<backend/data>/scenes/<region_id>/<scene_id>/status.json`` 에
변경 때마다 적는다 — GET 이 언제든 답하고 서버를 재시작해도 기록이 남는다. 재시작 시
아직 ``queued``/``running`` 이던 기록은 ``failed`` ("서버 재시작으로 중단") 로 바꾼다.

같은 scene_id(``<yyyymmdd>_<센서>``)의 파이프라인은 한 번에 하나만 돈다(409).
"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..wbms.liaison import LIAISON_KEYS
from .steps import (
    STEP_LABELS,
    STEP_ORDER,
    PipelineServices,
    ScenePipelineRunner,
    iso_date,
    prediction_detail,
    prediction_history_entry,
    utc_now_iso,
    write_json,
)

ACTIVE_STATUSES = frozenset({"queued", "running"})
RESTART_ERROR = "서버 재시작으로 중단"
OUTPUT_KEYS: tuple[str, ...] = (
    "quicklook_url", "overlay_url", "mask_url", "product_url", "product_name", "water_area_km2",
    "area_series_url", "levels_url", "prediction_id", "prediction_url", "folder",
)


class SceneConflictError(RuntimeError):
    """같은 장면의 파이프라인이 아직 돌고 있다."""


def new_pipeline_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"pl_{stamp}_{uuid.uuid4().hex[:8]}"


def _created_now_iso() -> str:
    """접수 시각 — 같은 초 안에 여러 장면이 들어와도 최신순이 흔들리지 않게 밀리초까지 적는다."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _newest_first(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """created_at 내림차순. 같은 시각이면 나중에 들어온 것이 앞 — 안정 정렬을 오름차순으로 한 뒤 뒤집는다."""

    return sorted(items, key=lambda item: str(item.get("created_at") or ""))[::-1]


def _empty_steps() -> list[dict[str, Any]]:
    return [
        {"key": key, "label": STEP_LABELS[key], "status": "pending", "started_at": None,
         "finished_at": None, "elapsed_s": None, "detail": None}
        for key in STEP_ORDER
    ]


def _progress(record: dict[str, Any]) -> float:
    if record.get("status") in ("done", "failed"):
        return 1.0
    steps = record.get("steps") or []
    if not steps:
        return 0.0
    terminal = sum(1 for s in steps if s.get("status") in ("done", "skipped", "failed"))
    return round(terminal / len(steps), 3)


def public_view(record: dict[str, Any]) -> dict[str, Any]:
    """API 응답 형태 — 내부 경로(internal)는 빼고 progress 를 계산해 붙인다."""

    out = {k: copy.deepcopy(v) for k, v in record.items() if k != "internal"}
    out["progress"] = _progress(record)
    outputs = {key: None for key in OUTPUT_KEYS}
    outputs.update(out.get("outputs") or {})
    out["outputs"] = outputs
    out["liaison"] = {key: (out.get("liaison") or {}).get(key) for key in LIAISON_KEYS}
    return out


def summary_view(record: dict[str, Any]) -> dict[str, Any]:
    outputs = record.get("outputs") or {}
    return {
        "scene_id": record["scene_id"],
        "region_id": record["region_id"],
        "sensor": record["sensor"],
        "image_date": record["image_date"],
        "status": record["status"],
        "progress": _progress(record),
        "created_at": record.get("created_at"),
        "finished_at": record.get("finished_at"),
        "water_area_km2": outputs.get("water_area_km2"),
        "product_name": outputs.get("product_name"),
        "pipeline_id": record.get("pipeline_id"),
    }


class ScenePipelineJobs:
    """장면 파이프라인 접수·조회·복구."""

    def __init__(self, services: PipelineServices, *, max_workers: int = 1, poll_s: float = 0.5) -> None:
        self.services = services
        self.scenes_root = services.scenes_root
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scene-pipeline")
        self.runner = ScenePipelineRunner(services, lock=self._lock, save=self._save, poll_s=poll_s)
        self.recover()

    # ── 영속화 ────────────────────────────────────────────────────────────
    def _save(self, record: dict[str, Any]) -> None:
        folder = Path(record["internal"]["folder"])
        write_json(folder / "status.json", record)

    def recover(self) -> list[str]:
        """디스크의 status.json 을 모두 읽어 메모리 색인을 만든다. 돌던 기록은 실패로 마감한다."""

        recovered: list[str] = []
        if not self.scenes_root.is_dir():
            return recovered
        for path in sorted(self.scenes_root.glob("*/*/status.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict) or not record.get("scene_id"):
                continue
            record.setdefault("internal", {})
            record["internal"]["folder"] = str(path.parent)
            if record.get("status") in ACTIVE_STATUSES:
                record["status"] = "failed"
                record["error"] = RESTART_ERROR
                record["finished_at"] = record.get("finished_at") or utc_now_iso()
                for step in record.get("steps") or []:
                    if step.get("status") == "running":
                        step["status"] = "failed"
                        step["detail"] = RESTART_ERROR
                        step["finished_at"] = step.get("finished_at") or record["finished_at"]
                    elif step.get("status") == "pending":
                        step["status"] = "skipped"
                        step["detail"] = RESTART_ERROR
                record.setdefault("warnings", []).append(RESTART_ERROR)
                try:
                    self._save(record)
                except OSError:
                    pass
                recovered.append(str(record["scene_id"]))
            with self._lock:
                existing = self._records.get(str(record["scene_id"]))
                # 같은 scene_id 가 두 지역에 있으면(드묾) 최신 기록이 이긴다
                if existing is None or str(existing.get("created_at") or "") <= str(record.get("created_at") or ""):
                    self._records[str(record["scene_id"])] = record
        return recovered

    # ── 접수 ──────────────────────────────────────────────────────────────
    def active(self, scene_id: str) -> bool:
        with self._lock:
            record = self._records.get(scene_id)
            return bool(record and record.get("status") in ACTIVE_STATUSES)

    def submit(self, spec: dict[str, Any]) -> dict[str, Any]:
        """검증이 끝난 접수 명세(spec)로 레코드를 만들고 작업자 스레드에 넘긴다. 돌고 있으면 409 용 예외."""

        scene_id = str(spec["scene_id"])
        region_id = str(spec["region_id"])
        folder = self.scenes_root / region_id / scene_id
        with self._lock:
            existing = self._records.get(scene_id)
            if existing is not None and existing.get("status") in ACTIVE_STATUSES:
                raise SceneConflictError(
                    f"장면 {scene_id} 의 파이프라인이 아직 진행 중입니다 "
                    f"(pipeline_id={existing.get('pipeline_id')}, status={existing.get('status')}) — 끝난 뒤 다시 제출하세요"
                )
            liaison = dict(spec.get("liaison") or {})
            record: dict[str, Any] = {
                "scene_id": scene_id,
                "region_id": region_id,
                "sensor": str(spec["sensor"]),
                "sensor_key": str(spec["sensor_key"]),
                "sensor_type": str(spec["sensor_type"]),
                "image_date": iso_date(str(spec["date_digits"])),
                "testbed": spec.get("testbed"),
                "pipeline_id": new_pipeline_id(),
                "status": "queued",
                "progress": 0.0,
                "steps": _empty_steps(),
                "outputs": {key: None for key in OUTPUT_KEYS},
                "warnings": [],
                "error": None,
                "created_at": _created_now_iso(),
                "started_at": None,
                "finished_at": None,
                "liaison": {key: liaison.get(key) for key in LIAISON_KEYS},
                "note": spec.get("note"),
                "options": {"detect": bool(spec.get("detect", True)), "predict": bool(spec.get("predict", True))},
                "internal": {
                    "folder": str(folder),
                    "work_dir": str(spec["work_dir"]),
                    "pre_source": str(spec["pre_source"]),
                    "pre_is_upload": bool(spec.get("pre_is_upload")),
                    "pre_stem": str(spec["pre_stem"]),
                    "pre_original_name": spec.get("pre_original_name"),
                    "mask_source": str(spec["mask_source"]) if spec.get("mask_source") else None,
                    "mask3_source": str(spec["mask3_source"]) if spec.get("mask3_source") else None,
                    "gt_source": str(spec["gt_source"]) if spec.get("gt_source") else None,
                    "date_digits": str(spec["date_digits"]),
                    "acquisition_stamp": str(spec["acquisition_stamp"]),
                    "testbed": spec.get("testbed"),
                    "testbed_code": spec.get("testbed_code"),
                },
            }
            self._records[scene_id] = record
            self._save(record)
            snapshot = public_view(record)
        self._executor.submit(self._work, record)
        return snapshot

    def _work(self, record: dict[str, Any]) -> None:
        try:
            self.runner.run(record)
        except Exception as exc:  # noqa: BLE001 - 어떤 경우에도 종료 상태를 남긴다
            with self._lock:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["finished_at"] = utc_now_iso()
                for step in record.get("steps") or []:
                    if step.get("status") == "running":
                        step["status"] = "failed"
                        step["detail"] = record["error"]
                    elif step.get("status") == "pending":
                        step["status"] = "skipped"
                try:
                    self._save(record)
                except OSError:
                    pass

    # ── 조회 ──────────────────────────────────────────────────────────────
    def get(self, scene_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(scene_id)
            return public_view(record) if record is not None else None

    def list(self, region_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            records = [r for r in self._records.values() if region_id is None or r.get("region_id") == region_id]
            return [summary_view(r) for r in _newest_first(records)]

    def folder_for(self, scene_id: str) -> Path | None:
        with self._lock:
            record = self._records.get(scene_id)
            if record is None:
                return None
            return Path(record["internal"]["folder"])

    def known_scene_ids(self) -> set[str]:
        with self._lock:
            return set(self._records)

    # ── 파이프라인 기록이 없는 기존 장면(내장 지역 자산 등) ───────────────────
    def _synthesize(self, region: Any, scene: dict[str, Any]) -> dict[str, Any]:
        prefix = self.services.api_prefix
        region_id = region.id
        scene_id = str(scene.get("scene_id"))
        images = scene.get("images") or {}
        date_text = iso_date(str(scene.get("date") or ""))
        has_mask = bool(scene.get("detection_available") and scene.get("source_mask"))
        levels = [
            r for r in region.satellite_levels()
            if iso_date(str(r.get("date") or "")) == date_text and str(r.get("satellite") or "").lower() == str(scene.get("sensor") or "").lower()
        ]
        registered = scene.get("registered_at")

        def step(key: str, status: str, detail: str) -> dict[str, Any]:
            return {"key": key, "label": STEP_LABELS[key], "status": status, "started_at": registered,
                    "finished_at": registered, "elapsed_s": None, "detail": detail}

        steps = [
            step("register", "done", "지역 장면 기록(scenes.json)에서 합성"),
            step("detect", "done" if has_mask else "skipped", "탐지 마스크 있음" if has_mask else "마스크 없음"),
            step("area", "done" if scene.get("frame") else "skipped",
                 f"{float(scene['water_area_km2']):.2f} km²" if scene.get("water_area_km2") is not None else "면적 없음"),
            step("levels", "done" if levels else "skipped", f"{len(levels)}지점 기록" if levels else "이 날짜의 위성 수위 기록 없음"),
            step("predict", "skipped", "파이프라인 기록 없음(기존 장면)"),
            step("deliver", "skipped", "파이프라인 기록 없음(기존 장면)"),
        ]
        outputs = {key: None for key in OUTPUT_KEYS}
        if images.get("input"):
            outputs["quicklook_url"] = f"{prefix}/regions/{region_id}/satellite/files/{images['input']}"
        if images.get("overlay"):
            outputs["overlay_url"] = f"{prefix}/regions/{region_id}/satellite/files/{images['overlay']}"
        if has_mask:
            outputs["mask_url"] = f"{prefix}/regions/{region_id}/satellite/scenes/{scene_id}/mask"
        outputs["water_area_km2"] = scene.get("water_area_km2")
        if scene.get("frame"):
            outputs["area_series_url"] = f"{prefix}/regions/{region_id}/satellite/area-series"
        if levels:
            outputs["levels_url"] = f"{prefix}/regions/{region_id}/satellite/levels?start={date_text}&end={date_text}"
        return {
            "scene_id": scene_id, "region_id": region_id, "sensor": str(scene.get("sensor") or ""),
            "sensor_type": scene.get("sensor_type"), "image_date": date_text, "pipeline_id": None,
            "status": "done", "progress": 1.0, "steps": steps, "outputs": outputs,
            "warnings": ["이 장면은 파이프라인 기록이 없어 지역 장면 기록(scenes.json)으로 합성한 상태입니다"],
            "error": None, "created_at": registered, "finished_at": registered,
            "liaison": {key: None for key in LIAISON_KEYS}, "synthesized": True,
        }

    def synthesize(self, scene_id: str) -> dict[str, Any] | None:
        for region in self.services.regions.list():
            payload = region.scenes() or {}
            for scene in payload.get("scenes") or []:
                if str(scene.get("scene_id")) == scene_id:
                    return self._synthesize(region, scene)
        return None

    def synthesize_all(self, region_id: str | None = None) -> list[dict[str, Any]]:
        known = self.known_scene_ids()
        out: list[dict[str, Any]] = []
        for region in self.services.regions.list():
            if region_id is not None and region.id != region_id:
                continue
            payload = region.scenes() or {}
            for scene in payload.get("scenes") or []:
                if str(scene.get("scene_id")) in known:
                    continue
                out.append(summary_view(self._synthesize(region, scene)))
        return out

    # ── ④ 화면의 모델 재예측 결과 반영 ─────────────────────────────────────────
    def record_prediction(self, scene_id: str, summary: dict[str, Any]) -> dict[str, Any]:
        """``POST /scenes/{scene_id}/predict`` 결과를 장면 기록에 남긴다 — ``outputs.prediction_id/url`` 갱신,
        ``predictions[]`` 이력 추가, predict 단계 detail 갱신, status.json 저장(장면 폴더에 prediction.json 이 있으면 함께).

        파이프라인 기록이 없는 기존 장면(내장 지역 자산 등)은 지역 장면 기록으로 합성한 레코드를 만들어
        ``<scenes_root>/<region>/<scene_id>/status.json`` 에 처음 저장한다(``pipeline_id=null`` 유지). 모르는 장면은 KeyError.
        """

        prefix = self.services.api_prefix
        with self._lock:
            record = self._records.get(scene_id)
            if record is None:
                synthesized = self.synthesize(scene_id)
                if synthesized is None:
                    raise KeyError(scene_id)
                record = synthesized
                record["internal"] = {"folder": str(self.scenes_root / str(record["region_id"]) / scene_id)}
                record["warnings"] = [
                    w for w in record.get("warnings") or [] if "합성한 상태" not in str(w)
                ] + ["파이프라인 기록이 없던 기존 장면 — 모델 재예측 요청으로 장면 기록을 처음 만들었다(등록·탐지 단계는 지역 장면 기록에서 합성)"]
                self._records[scene_id] = record
            record.setdefault("internal", {})["prediction"] = summary
            record.setdefault("outputs", {})["prediction_id"] = summary["prediction_id"]
            record["outputs"]["prediction_url"] = f"{prefix}/predictions/{summary['prediction_id']}"
            record.setdefault("predictions", []).append(prediction_history_entry(summary))
            now = utc_now_iso()
            for step in record.get("steps") or []:
                if step.get("key") == "predict":
                    step["status"] = "done"
                    step["detail"] = prediction_detail(summary) + " (재예측)"
                    step["finished_at"] = now
                    if not step.get("started_at"):
                        step["started_at"] = now
            self._save(record)
            folder = Path(record["internal"]["folder"])
            if (folder / "prediction.json").is_file():
                write_json(folder / "prediction.json", summary)
            return public_view(record)

    # ── 접수 화면용 선택지 ──────────────────────────────────────────────────
    def options(self) -> dict[str, Any]:
        catalog = self.services.wbms.catalog

        def sensor_id(name: str) -> str:
            # regions[].sensors 와 sensors[].id 는 같은 표기(파일 라벨)여야 화면이 지역별 선택지를 가릴 수 있다.
            # 지역 정의·장면 등록은 표시명(Umbra)을 적으므로 카탈로그로 라벨(UMBRA)에 맞춘다. 모르는 센서는 그대로.
            key = catalog.resolve_key(name)
            return catalog.file_label(key) if key else str(name)

        regions: list[dict[str, Any]] = []
        for record in self.services.regions.list():
            satellite = record.manifest.get("satellite") or {}
            testbed = (str(satellite.get("wbms_testbed") or "").lower()) or None
            # 지역 정의의 센서 목록 + 카탈로그가 이 테스트베드에 받는 센서(인수 장면이 아직 없어도 올릴 수 있다)
            sensors: list[str] = []
            raw_names = [str(s) for s in satellite.get("sensors") or []]
            if testbed:
                raw_names += [str(p["name"]) for p in catalog.sensors.values() if testbed in (p.get("pre_dirs") or {})]
            for raw in raw_names:
                label = sensor_id(raw)
                if label not in sensors:
                    sensors.append(label)
            scenes = (record.scenes() or {}).get("scenes") or []
            regions.append({
                "id": record.id, "name": record.name, "testbed": testbed, "sensors": sensors,
                "has_gauge": bool(record.gauge_available), "satellite_available": bool(record.satellite_available),
                "detect_available": bool(testbed and testbed in catalog.testbeds), "n_scenes": len(scenes),
            })
        sensors = [
            {
                "id": catalog.file_label(key), "key": key, "name": str(profile["name"]),
                "label": f"{profile['name']} {'레이더' if str(profile['type']).upper() == 'SAR' else '광학'} 영상",
                "type": str(profile["type"]).upper(), "needs_mask": bool(profile.get("needs_mask")),
                "testbeds": sorted(profile.get("pre_dirs") or {}), "description_ko": profile.get("description_ko", ""),
                "module_sensor": catalog.module_sensor(key),
            }
            for key, profile in catalog.sensors.items()
        ]
        return {
            "regions": regions,
            "sensors": sensors,
            "note_ko": (
                "needs_mask 인 센서(SAR)는 날짜별 3-class 마스크가 없으면 가장 가까운 날짜 것을, 그것도 없으면 영상 유효 범위로 "
                "생성한 마스크(바다 없음 가정)를 쓴다(경고로 알린다). module_sensor 가 key 와 다른 센서(umbra→iceye)는 그 모델로 탐지한다."
            ),
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
