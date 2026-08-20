from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ..wbms.config import WbmsSettings
from ..wbms.jobs import FileJobStore, utc_now
from ..wbms.runner import (
    CONTAINER_MODULE_DIR,
    DockerRunner,
    docker_prefix,
    resolve_gpu_device,
    summarize_failure,
)

TB_CODES = {"busan": "Busan", "dcd": "DCD"}

SENSORS: dict[str, dict[str, Any]] = {
    "iceye": {
        "name": "ICEYE",
        "type": "SAR",
        "weights": "WBMS_SAR_ICEYE.h5",
        "pre_dirs": {"busan": "iceye_pre"},
        # SAR cannot separate sea from inland water: the 3-class mask is mandatory.
        "needs_mask": True,
    },
    "planet": {
        "name": "PlanetScope",
        "type": "OPTIC",
        "weights": "WBMS_Optic_PlanetScope.h5",
        "pre_dirs": {"busan": "planet_pre", "dcd": "dcd_pre"},
        "needs_mask": False,
    },
}


class WbmsInputError(ValueError):
    """Request cannot be executed as given (maps to HTTP 422)."""


class WbmsExecutionError(RuntimeError):
    """The container run itself failed (maps to HTTP 502)."""


class WbmsSegmentationJobAdapter:
    """② detect_water as an asynchronous host-runner job.

    A CPU scene takes ~610 s (measured), so the API is job-based:
    ``submit`` returns a job id immediately, state lives in
    ``data/wbms_jobs/<id>.json`` and results land in ``data/wbms_runs/wb``.
    Scenes already segmented under the output root are returned instantly
    through the cache fast path without touching docker.
    """

    model_id = "wbms-detect-water"

    def __init__(
        self,
        settings: WbmsSettings,
        runner: DockerRunner,
        job_store: FileJobStore | None = None,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.jobs = job_store or FileJobStore(settings.jobs_dir)

    # ── public API ─────────────────────────────────────────────────────────
    def submit(self, request: dict[str, Any], *, start_thread: bool = True) -> dict[str, Any]:
        sensor = request["sensor"]
        testbed = request["testbed"]
        image_date = request["image_date"]

        # 사용자 지정 input_tif는 캐시와 다른 입력일 수 있으므로 캐시를 타지 않는다
        if not bool(request.get("force")) and not request.get("input_tif"):
            cached = self.find_cached(sensor, testbed, image_date)
            if cached is not None:
                job = {
                    "job_id": FileJobStore.new_job_id("seg"),
                    "model_id": self.model_id,
                    "module": "detect_water",
                    "status": "succeeded",
                    "cached": True,
                    "request": dict(request),
                    "command": [],
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                    "result": cached,
                    "error": None,
                    "warnings": [
                        "returned from the completed-mask cache; pass force=true to rerun"
                    ],
                }
                return self.jobs.create(job)

        scene = self.resolve_scene(sensor, testbed, image_date, request.get("input_tif"))
        command = self.build_command(scene)
        job = {
            "job_id": FileJobStore.new_job_id("seg"),
            "model_id": self.model_id,
            "module": "detect_water",
            "status": "queued",
            "cached": False,
            "request": dict(request),
            "command": command,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "warnings": scene.get("warnings", []),
        }
        self.jobs.create(job)
        if start_thread:
            thread = threading.Thread(
                target=self._execute, args=(job["job_id"],), daemon=True,
                name=f"wbms-{job['job_id']}",
            )
            thread.start()
        return self.jobs.get(job["job_id"])

    def get(self, job_id: str) -> dict[str, Any]:
        return self.jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        return self.jobs.list()

    # ── cache fast path ────────────────────────────────────────────────────
    def find_cached(
        self,
        sensor: str,
        testbed: str,
        image_date: str,
        *,
        min_mtime: float | None = None,
    ) -> dict[str, Any] | None:
        """min_mtime이 주어지면 그 시각 이후 기록된 qc만 인정하고 최신 것을 고른다
        — 실행 직후 성공 판정이 이전 실행의 잔존 산출물을 집어오지 못하게 한다."""
        profile = SENSORS[sensor]
        tb = TB_CODES[testbed]
        scene_dir = self.settings.wb_output_dir / image_date[:4] / image_date
        if not scene_dir.is_dir():
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for qc_path in sorted(scene_dir.glob(f"WB_{tb}_{profile['name']}_*_qc.json")):
            try:
                qc_mtime = qc_path.stat().st_mtime
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if min_mtime is not None and qc_mtime < min_mtime:
                continue
            if qc.get("status") != "COMPLETE":
                continue
            base = str(qc.get("base") or qc_path.name[: -len("_qc.json")])
            products = [str(name) for name in qc.get("products", [])]
            mask_tif = scene_dir / f"{base}.tif"
            if not mask_tif.is_file() or any(
                not (scene_dir / name).is_file() for name in products
            ):
                continue
            meta_json = scene_dir / f"{base}_meta.json"
            elapsed = ((qc.get("runtime") or {}).get("elapsed_s"))
            entry = {
                "source": "cache",
                "output_dir": str(scene_dir),
                "base": base,
                "mask_tif": str(mask_tif),
                "meta_json": str(meta_json) if meta_json.is_file() else None,
                "qc_json": str(qc_path),
                "products": products,
                "elapsed_s": float(elapsed) if isinstance(elapsed, (int, float)) else None,
            }
            if min_mtime is None:
                return entry
            if best is None or qc_mtime > best[0]:
                best = (qc_mtime, entry)
        return best[1] if best is not None else None

    def list_cached(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        root = self.settings.wb_output_dir
        if not root.is_dir():
            return found
        for qc_path in sorted(root.glob("*/*/WB_*_qc.json")):
            try:
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if qc.get("status") != "COMPLETE":
                continue
            params = qc.get("params") or {}
            found.append(
                {
                    "base": qc.get("base"),
                    "image_date": params.get("image_date") or qc_path.parent.name,
                    "sensor_name": params.get("sensor_name"),
                    "testbed": params.get("testbed"),
                    "qc_json": str(qc_path),
                }
            )
        return found

    # ── command construction ───────────────────────────────────────────────
    def resolve_scene(
        self,
        sensor: str,
        testbed: str,
        image_date: str,
        input_tif: str | None = None,
    ) -> dict[str, Any]:
        profile = SENSORS[sensor]
        pre_dir_name = profile["pre_dirs"].get(testbed)
        if pre_dir_name is None:
            raise WbmsInputError(
                f"sensor {sensor!r} has no delivered scenes for testbed {testbed!r}"
            )
        warnings: list[str] = []

        if input_tif:
            input_host = Path(input_tif).resolve()
            if not input_host.is_file():
                raise WbmsInputError(f"input_tif not found: {input_host}")
        else:
            scene_dir = (
                self.settings.l1_pre_dir / pre_dir_name / image_date[:4] / image_date
            )
            candidates = sorted(
                list(scene_dir.glob(f"Pre_*_{profile['name']}_{image_date}T*.tif"))
                + list(scene_dir.glob(f"Pre_*_{profile['name']}_{image_date}.tif"))
            )
            candidates = [c for c in candidates if not c.name.endswith("_lsmap.tif")]
            if not candidates:
                raise WbmsInputError(
                    f"no ① Pre_*.tif found for {profile['name']} {image_date} under {scene_dir}"
                )
            input_host = candidates[0]

        meta_sidecar = input_host.with_name(input_host.stem + "_meta.json")
        if not meta_sidecar.is_file():
            raise WbmsInputError(
                f"① sidecar meta missing: {meta_sidecar.name} — detect_water inherits the "
                "acquisition stamp from the upstream meta and fails without it (X-12)"
            )

        mounts: list[tuple[Path, str, bool]] = [
            (self.settings.handover_dir, "/data", True),
            (self.settings.model_dir, "/model", True),
            (self.settings.aux_dir, "/aux", True),
            (self.settings.wb_output_dir, "/out", False),
        ]
        try:
            relative = input_host.relative_to(self.settings.handover_dir)
            input_container = f"/data/{relative.as_posix()}"
        except ValueError:
            mounts.append((input_host.parent, "/scene", True))
            input_container = f"/scene/{input_host.name}"
            warnings.append(
                "input_tif lies outside the handover package; its directory is mounted "
                "read-only at /scene"
            )

        weights_host = self.settings.model_dir / profile["weights"]
        if not weights_host.is_file():
            raise WbmsInputError(f"model weights missing: {weights_host}")

        mask_container: str | None = None
        if profile["needs_mask"]:
            mask_host = (
                self.settings.aux_dir
                / "Masks"
                / "Img_DEM_3class_utm"
                / f"{testbed}_{image_date}_mask3_utm.tif"
            )
            if not mask_host.is_file():
                raise WbmsInputError(
                    f"3-class mask required for SAR but missing: {mask_host} — "
                    "detect_water fails legitimately without it (sea vs water)"
                )
            mask_container = (
                f"/aux/Masks/Img_DEM_3class_utm/{testbed}_{image_date}_mask3_utm.tif"
            )

        return {
            "sensor": sensor,
            "testbed": testbed,
            "image_date": image_date,
            "input_host": str(input_host),
            "input_container": input_container,
            "weights_container": f"/model/{profile['weights']}",
            "mask_container": mask_container,
            "mounts": mounts,
            "warnings": warnings,
        }

    def build_command(self, scene: dict[str, Any]) -> list[str]:
        self.settings.wb_output_dir.mkdir(parents=True, exist_ok=True)
        device, gpu_warning = resolve_gpu_device(
            self.settings.gpu_mode, self.settings.gpu_min_free_mib
        )
        if gpu_warning:
            scene.setdefault("warnings", []).append(gpu_warning)
        argv = docker_prefix(
            self.settings.docker_bin, self.settings.docker_image, scene["mounts"],
            gpu=(device == "gpu"),
        )
        argv += [
            "python3",
            f"{CONTAINER_MODULE_DIR}/detect_water.py",
            "--input", scene["input_container"],
            "--sensor", scene["sensor"],
            "--testbed", scene["testbed"],
            "--image_date", scene["image_date"],
            "--weights", scene["weights_container"],
        ]
        if scene["mask_container"]:
            argv += ["--mask", scene["mask_container"]]
        argv += ["--gpu", device, "--output_dir", "/out"]
        return argv

    # ── worker ─────────────────────────────────────────────────────────────
    def _execute(self, job_id: str) -> None:
        job = self.jobs.update(job_id, status="running", started_at=utc_now())
        run_started = time.time()
        try:
            result = self.runner.run(
                list(job["command"]), timeout_s=self.settings.segmentation_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - job must always reach a terminal state
            self.jobs.update(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error=f"runner crashed: {type(exc).__name__}: {exc}",
            )
            return
        request = job.get("request", {})
        if result.returncode == 0:
            # 이번 실행이 실제로 만든(=실행 시작 이후 기록된) qc만 성공 근거로 인정
            produced = self.find_cached(
                request.get("sensor", "iceye"),
                request.get("testbed", "busan"),
                request.get("image_date", ""),
                min_mtime=run_started - 5.0,
            )
            if produced is None:
                self.jobs.update(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    error=(
                        "container exited 0 but no COMPLETE WB_*_qc.json was found under "
                        f"{self.settings.wb_output_dir} — refusing to report success"
                    ),
                )
                return
            produced = {**produced, "source": "run", "elapsed_s": round(result.elapsed_s, 1)}
            self.jobs.update(
                job_id, status="succeeded", finished_at=utc_now(), result=produced
            )
        else:
            self.jobs.update(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error=summarize_failure(result),
            )
