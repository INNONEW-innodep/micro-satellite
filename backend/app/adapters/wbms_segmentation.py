from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from ..wbms.catalog import WbmsCatalog, default_catalog
from ..wbms.config import WbmsSettings
from ..wbms.liaison import LiaisonError, derive_product_id, export_product, liaison_params
from ..wbms.liaison_log import LiaisonLogger
from ..wbms.jobs import FileJobStore, utc_now
from ..wbms.runner import (
    CONTAINER_MODULE_DIR,
    DockerRunner,
    docker_prefix,
    resolve_gpu_device,
    summarize_failure,
)

# 센서·테스트베드 표는 wbms/catalog.json 이 단일 근거다. 모듈 상수는 하위 호환용
# 별칭이며 같은 객체를 가리킨다.
_CATALOG = default_catalog()
TB_CODES: dict[str, str] = _CATALOG.tb_codes
SENSORS: dict[str, dict[str, Any]] = _CATALOG.sensors


def cpu_fallback_command(command: list[str]) -> list[str]:
    """GPU 용 docker 명령을 CPU 용으로 바꾼다 — ``--gpus device=0`` 과 TF GPU 환경변수 두 쌍을 빼고 ``--gpu gpu`` → ``--gpu cpu``."""

    out: list[str] = []
    skip = 0
    for i, tok in enumerate(command):
        if skip:
            skip -= 1
            continue
        if tok == "--gpus":
            skip = 1
            continue
        if tok == "-e" and i + 1 < len(command) and command[i + 1] in ("TF_FORCE_GPU_ALLOW_GROWTH=true", "TF_GPU_ALLOCATOR=cuda_malloc_async"):
            skip = 1
            continue
        out.append(tok)
    for i in range(len(out) - 1):
        if out[i] == "--gpu" and out[i + 1] == "gpu":
            out[i + 1] = "cpu"
    return out


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
        catalog: WbmsCatalog | None = None,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.jobs = job_store or FileJobStore(settings.jobs_dir)
        self.catalog = catalog or default_catalog()
        # 연계 로그 발신기 — URL 이 비면 no-op. 테스트는 인스턴스를 바꿔 끼운다.
        self.liaison = LiaisonLogger(
            settings.liaison_log_url,
            system_code=settings.liaison_system_code,
            algm_id=settings.liaison_algm_id,
            timeout_s=settings.liaison_timeout_s,
        )

    def validate_request(self, sensor: str, testbed: str) -> tuple[str, str]:
        """카탈로그에 없는 센서·테스트베드는 422. Literal 열거 대신 데이터로 검증한다."""

        sensor_key = str(sensor or "").lower()
        testbed_key = str(testbed or "").lower()
        if sensor_key not in self.catalog.sensors:
            raise WbmsInputError(
                f"unknown sensor {sensor!r}; catalog knows {sorted(self.catalog.sensors)} "
                f"({self.catalog.source})"
            )
        if testbed_key not in self.catalog.testbeds:
            raise WbmsInputError(
                f"unknown testbed {testbed!r}; catalog knows {sorted(self.catalog.testbeds)} "
                f"({self.catalog.source})"
            )
        return sensor_key, testbed_key

    # ── public API ─────────────────────────────────────────────────────────
    def submit(self, request: dict[str, Any], *, start_thread: bool = True) -> dict[str, Any]:
        sensor, testbed = self.validate_request(request.get("sensor", "iceye"), request.get("testbed", "busan"))
        request = {**request, "sensor": sensor, "testbed": testbed}
        image_date = request["image_date"]
        try:
            liaison = liaison_params(request)
        except LiaisonError as exc:
            raise WbmsInputError(str(exc)) from exc
        request = {k: v for k, v in request.items() if k not in liaison or v is not None}
        request.update(liaison)

        # 사용자 지정 input_tif는 캐시와 다른 입력일 수 있으므로 캐시를 타지 않는다
        if not bool(request.get("force")) and not request.get("input_tif"):
            cached = self.find_cached(sensor, testbed, image_date)
            if cached is not None:
                cached = self._attach_product(cached, request)
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
                created = self.jobs.create(job)
                # 캐시 히트는 대기·진행을 거치지 않으므로 완료 로그 1건만 보낸다
                self._log(created, "succeeded", f"{image_date} {sensor} 장면의 기존 수체탐지 산출물을 반환하였습니다.")
                return self.jobs.get(created["job_id"])

        scene = self.resolve_scene(
            sensor, testbed, image_date, request.get("input_tif"), mask_tif=request.get("mask_tif"),
        )
        # force=true 는 백엔드 캐시뿐 아니라 모듈의 완료 마커(SKIP)도 무시해야 실제로 다시 돈다
        scene["force"] = bool(request.get("force"))
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
        created = self.jobs.create(job)
        self._log(created, "queued", f"{image_date} {sensor} 장면 수체탐지 작업이 접수되었습니다.")
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
        — 실행 직후 성공 판정이 이전 실행의 잔존 산출물을 집어오지 못하게 한다.

        센서 표기 허용 범위: 컨테이너는 산출물 이름과 qc ``params.sensor_name`` 을 ``KEY2NAME[--sensor]`` 로
        적는다. 다른 센서의 모델을 빌려 쓰는 센서(umbra → iceye)는 그래서 ``WB_<TB>_ICEYE_*`` 로 나오므로,
        카탈로그 ``cache_sensor_names``(Umbra·UMBRA·ICEYE)를 모두 받되 자기 라벨(UMBRA)이 든 산출물이
        있으면 그것을 우선한다. 빌린 표기는 그 센서가 인수된 테스트베드(umbra: dcd)에서만 받는다.
        알려진 모호성: 같은 날짜·테스트베드에 ICEYE 장면과 Umbra 장면이 둘 다 있으면 서로의
        ``WB_<TB>_ICEYE_*`` 를 구분하지 못한다(ICEYE 는 busan, Umbra 는 dcd 에만 인수돼 지금은 겹치지
        않는다 — 겹치면 force=true 로 다시 돌리거나 input_tif 를 지정하라).
        """
        profile = self.catalog.sensor(sensor)
        tb = self.catalog.testbed(testbed)["code"]
        scene_dir = self.settings.wb_output_dir / image_date[:4] / image_date
        if not scene_dir.is_dir():
            return None
        accepted = self.catalog.cache_sensor_names(sensor, testbed)
        accepted_lower = {name.lower() for name in accepted}
        own_label = str(profile["file_label"]).upper()
        candidates = sorted({path for name in accepted for path in scene_dir.glob(f"WB_{tb}_{name}_*_qc.json")})
        best: tuple[tuple[int, float], dict[str, Any]] | None = None
        for qc_path in candidates:
            try:
                qc_mtime = qc_path.stat().st_mtime
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if min_mtime is not None and qc_mtime < min_mtime:
                continue
            if qc.get("status") != "COMPLETE":
                continue
            params_name = str((qc.get("params") or {}).get("sensor_name") or "")
            if params_name and params_name.lower() not in accepted_lower:
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
            # 자기 라벨이 든 산출물 우선, 그다음(min_mtime 모드에서만) 최신 qc. 동률이면 정렬 순서상 먼저 본 것.
            rank = (1 if own_label in base.upper() else 0, qc_mtime if min_mtime is not None else 0.0)
            if best is None or rank > best[0]:
                best = (rank, entry)
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

    # ── 3-class 마스크 보조 ─────────────────────────────────────────────────
    def mask3_path(self, testbed: str, image_date: str) -> Path:
        """카탈로그 템플릿이 가리키는 그 날짜의 3-class 마스크 호스트 경로(존재 여부와 무관)."""

        mask_rel = str(self.catalog.testbed(testbed)["mask_template"]).format(
            testbed=str(testbed).lower(), image_date=image_date
        )
        return self.settings.aux_dir / mask_rel

    def nearest_mask3(self, testbed: str, image_date: str) -> Path | None:
        """``image_date`` 에 날짜가 가장 가까운, 실제로 있는 3-class 마스크(aux_dir 아래). 없으면 None.

        3-class 마스크(바다·내륙수·육지)는 지형 분류라 날짜 간 변화가 거의 없다. 새 촬영일의
        전용 마스크가 아직 없을 때 파이프라인이 인접 날짜 것을 대신 쓰기 위한 탐색이다 — 같은 날짜
        마스크가 있으면 그것(거리 0)을 돌려준다. 거리가 같으면 앞선 날짜를 고른다.
        """

        try:
            template = str(self.catalog.testbed(testbed)["mask_template"])
        except KeyError:
            return None
        pattern = template.format(testbed=str(testbed).lower(), image_date="*")
        try:
            target = datetime.strptime(str(image_date)[:8], "%Y%m%d").date()  # noqa: DTZ007 - 날짜 비교만
        except ValueError:
            return None
        prefix, _, suffix = pattern.partition("*")
        best: tuple[int, str, Path] | None = None
        for candidate in sorted(self.settings.aux_dir.glob(pattern)):
            rel = candidate.relative_to(self.settings.aux_dir).as_posix()
            if not (rel.startswith(prefix) and rel.endswith(suffix)):
                continue
            digits = rel[len(prefix): len(rel) - len(suffix)] if suffix else rel[len(prefix):]
            try:
                found = datetime.strptime(digits, "%Y%m%d").date()  # noqa: DTZ007 - 날짜 비교만
            except ValueError:
                continue
            distance = abs((found - target).days)
            key = (distance, digits, candidate)
            if best is None or key[:2] < best[:2]:
                best = key
        return best[2] if best is not None else None

    # ── command construction ───────────────────────────────────────────────
    def resolve_scene(
        self,
        sensor: str,
        testbed: str,
        image_date: str,
        input_tif: str | None = None,
        mask_tif: str | None = None,
    ) -> dict[str, Any]:
        profile = self.catalog.sensor(sensor)
        pre_dir_name = profile["pre_dirs"].get(testbed)
        if pre_dir_name is None:
            raise WbmsInputError(
                f"sensor {sensor!r} has no delivered scenes for testbed {testbed!r}"
            )
        # ① 파일명의 센서 라벨(ICEYE·PlanetScope·UMBRA) — 표시명과 다를 수 있다(Umbra)
        label = str(profile.get("file_label") or profile["name"])
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
                list(scene_dir.glob(f"Pre_*_{label}_{image_date}T*.tif"))
                + list(scene_dir.glob(f"Pre_*_{label}_{image_date}.tif"))
            )
            candidates = [c for c in candidates if not c.name.endswith("_lsmap.tif")]
            if not candidates:
                raise WbmsInputError(
                    f"no ① Pre_*.tif found for {label} {image_date} under {scene_dir}"
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
        if profile["needs_mask"] and mask_tif:
            # 호출자가 고른 3-class 마스크(예: 날짜별 마스크가 없어 인접 날짜 것을 대신 쓰는 파이프라인).
            # aux 안에 있으면 /aux 경로 그대로, 밖이면 그 디렉터리를 읽기전용으로 따로 붙인다.
            mask_host = Path(mask_tif).expanduser().resolve()
            if not mask_host.is_file():
                raise WbmsInputError(f"mask_tif not found: {mask_host}")
            try:
                mask_rel_path = mask_host.relative_to(self.settings.aux_dir.resolve())
                mask_container = f"/aux/{mask_rel_path.as_posix()}"
            except ValueError:
                mounts.append((mask_host.parent, "/mask3", True))
                mask_container = f"/mask3/{mask_host.name}"
                warnings.append(
                    "mask_tif lies outside the handover aux directory; its directory is mounted "
                    "read-only at /mask3"
                )
        elif profile["needs_mask"]:
            mask_rel = str(self.catalog.testbed(testbed)["mask_template"]).format(
                testbed=testbed, image_date=image_date
            )
            mask_host = self.settings.aux_dir / mask_rel
            if not mask_host.is_file():
                raise WbmsInputError(
                    f"3-class mask required for SAR but missing: {mask_host} — "
                    "detect_water fails legitimately without it (sea vs water)"
                )
            mask_container = f"/aux/{mask_rel}"

        return {
            "sensor": sensor,
            # 컨테이너 --sensor: 전용 모델이 없는 센서(umbra)는 빌린 센서 키(iceye)로 돈다. 잡 요청 에코·캐시
            # 조회·호스트 경로는 카탈로그 키(sensor)를 그대로 쓴다.
            "module_sensor": self.catalog.module_sensor(sensor),
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
            "--sensor", scene.get("module_sensor") or scene["sensor"],
            "--testbed", scene["testbed"],
            "--image_date", scene["image_date"],
            "--weights", scene["weights_container"],
        ]
        if scene["mask_container"]:
            argv += ["--mask", scene["mask_container"]]
        argv += ["--gpu", device, "--output_dir", "/out"]
        if scene.get("force"):
            argv.append("--force")
        return argv

    # ── worker ─────────────────────────────────────────────────────────────
    def _execute(self, job_id: str) -> None:
        job = self.jobs.update(job_id, status="running", started_at=utc_now())
        request = job.get("request", {})
        scene_label = f"{request.get('image_date', '')} {request.get('sensor', '')}".strip()
        self._log(job, "running", f"{scene_label} 장면에 대해 수체탐지 알고리즘을 구동하였습니다.")
        run_started = time.time()
        try:
            result = self.runner.run(
                list(job["command"]), timeout_s=self.settings.segmentation_timeout_s
            )
            # 2026-10-20: GPU 로 돌렸는데 런타임에서 죽으면(드라이버·CUDA·cuDNN 불일치 — 새 세대 GPU 에 옛 CUDA 11.8 이미지)
            # auto 모드에서는 같은 명령을 CPU 로 한 번 더 돌린다. VRAM 프리플라이트는 이런 실패를 미리 알 수 없다.
            if result.returncode != 0 and not result.timed_out and self.settings.gpu_mode == "auto" and "--gpus" in job["command"]:
                reason = summarize_failure(result)
                cpu_command = cpu_fallback_command(list(job["command"]))
                job = self.jobs.update(
                    job_id, command=cpu_command,
                    gpu_fallback={"reason": reason[:400], "note_ko": "GPU 실행이 실패해 CPU 로 다시 돌렸습니다(auto 모드)"},
                )
                self._log(job, "running", f"{scene_label} GPU 실행 실패 → CPU 로 재시도합니다.")
                run_started = time.time()
                result = self.runner.run(cpu_command, timeout_s=self.settings.segmentation_timeout_s)
        except Exception as exc:  # noqa: BLE001 - job must always reach a terminal state
            failed = self.jobs.update(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error=f"runner crashed: {type(exc).__name__}: {exc}",
            )
            self._log(failed, "failed", f"{scene_label} 수체탐지 처리 중 에러가 발생하였습니다. {failed.get('error')}")
            return
        if result.returncode == 0:
            # 이번 실행이 실제로 만든(=실행 시작 이후 기록된) qc만 성공 근거로 인정
            produced = self.find_cached(
                request.get("sensor", "iceye"),
                request.get("testbed", "busan"),
                request.get("image_date", ""),
                min_mtime=run_started - 5.0,
            )
            if produced is None:
                failed = self.jobs.update(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    error=(
                        "container exited 0 but no COMPLETE WB_*_qc.json was found under "
                        f"{self.settings.wb_output_dir} — refusing to report success"
                    ),
                )
                self._log(failed, "failed", f"{scene_label} 수체탐지 처리 중 에러가 발생하였습니다. 산출물 없음")
                return
            produced = {**produced, "source": "run", "elapsed_s": round(result.elapsed_s, 1)}
            if job.get("gpu_fallback"):
                produced["gpu_fallback"] = dict(job["gpu_fallback"])
            produced = self._attach_product(produced, request)
            done = self.jobs.update(
                job_id, status="succeeded", finished_at=utc_now(), result=produced
            )
            product_name = Path(produced["product_tif"]).name if produced.get("product_tif") else Path(produced["mask_tif"]).name
            self._log(done, "succeeded", f"{product_name} 산출물이 생성되었습니다.")
        else:
            failed = self.jobs.update(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error=summarize_failure(result),
            )
            self._log(failed, "failed", f"{scene_label} 수체탐지 처리 중 에러가 발생하였습니다. {failed.get('error')}")

    # ── 연계 보조 ───────────────────────────────────────────────────────────
    def _attach_product(self, produced: dict[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
        """규약 파일명 ``{product_type}_{product_id}.tif`` 복사본을 만들어 결과에 경로를 붙인다.

        product_id 가 요청에 없으면 잡의 입력 전처리 사이드카(또는 qc 의 input.path)에서 파생한다.
        실패해도 탐지 결과는 그대로 반환한다 — 규약 파일은 부가 산출물이다.
        """

        product_type = str(request.get("product_type") or "PT000003")
        product_id = request.get("product_id")
        if not product_id:
            # 호스트에서 열 수 있는 전처리 경로 후보를 순서대로: 요청 input_tif → 카탈로그 해석(호스트) →
            # qc.input.path(컨테이너 경로 /data/… 는 인수자료 루트로 되짚는다)
            candidates: list[str] = []
            if request.get("input_tif"):
                candidates.append(str(request["input_tif"]))
            try:
                scene = self.resolve_scene(
                    request.get("sensor", "iceye"), request.get("testbed", "busan"),
                    request.get("image_date", ""), None,
                )
                if scene.get("input_host"):
                    candidates.append(str(scene["input_host"]))
            except Exception:  # noqa: BLE001 - 파생 실패는 부가 기능
                pass
            qc = self._read_json(Path(produced.get("qc_json", "")))
            qc_path = ((qc.get("input") or {}).get("path")) if isinstance(qc, Mapping) else None
            if qc_path:
                text = str(qc_path)
                if text.startswith("/data/"):
                    text = str(self.settings.handover_dir / text[len("/data/"):])
                candidates.append(text)
            for candidate in candidates:
                product_id = derive_product_id(candidate)
                if product_id:
                    break
        if not product_id:
            return {**produced, "product_tif": None, "product_id": None}
        try:
            target = export_product(
                produced["mask_tif"], self.settings.product_dir,
                product_type=product_type, product_id=str(product_id),
            )
        except (OSError, LiaisonError):
            return {**produced, "product_tif": None, "product_id": str(product_id)}
        return {**produced, "product_tif": str(target), "product_id": str(product_id)}

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, Any]:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, Mapping) else {}

    def _log(self, job: Mapping[str, Any], status: str, msg: str) -> None:
        """연계 로그 1건. 전송 실패는 잡 warnings 에만 남긴다."""

        warning = self.liaison.log_job(job, status=status, msg=msg)
        if warning:
            try:
                current = self.jobs.get(str(job["job_id"]))
                self.jobs.update(str(job["job_id"]), warnings=[*current.get("warnings", []), warning])
            except Exception:  # noqa: BLE001
                pass
