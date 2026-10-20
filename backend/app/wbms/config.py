from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_IMAGE = "wbms:0.13"
DEFAULT_DOCKER_BIN = "docker"
DEFAULT_HANDOVER_DIR = _REPO_ROOT / "data" / "incoming" / "handover"
DEFAULT_JOBS_DIR = _REPO_ROOT / "data" / "wbms_jobs"
DEFAULT_WB_OUTPUT_DIR = _REPO_ROOT / "data" / "wbms_runs" / "wb"
DEFAULT_FUSION_WORK_DIR = _REPO_ROOT / "data" / "wbms_runs" / "fusion"
DEFAULT_SEGMENTATION_TIMEOUT_S = 3600.0  # CPU scene is ~610 s; leave generous headroom
DEFAULT_FUSION_TIMEOUT_S = 600.0  # LSTM inference is ~3 s + container/TF startup
# 이미지 안에서 저장소가 놓이는 경로(Dockerfile WORKDIR). BACKEND_CONTAINER_ROOT 로 바꿀 수 있다.
DEFAULT_CONTAINER_ROOT = Path("/app")


def _bundle_dir_for(handover: Path) -> Path:
    return handover / "02_package" / "deploy_package" / "config" / "WBMS_Fusion_LSTM_Busan"


def _aws_csv_for(handover: Path) -> Path:
    return handover / "06_aux" / "busan_aws_2019_202004.csv"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw).resolve() if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class WbmsSettings:
    """Paths and limits for driving the delivered wbms container image.

    Every path defaults to the repository handover layout and can be overridden
    with ``BACKEND_WBMS_*`` environment variables (tests point them at tmp dirs).
    """

    docker_image: str = DEFAULT_IMAGE
    docker_bin: str = DEFAULT_DOCKER_BIN
    handover_dir: Path = DEFAULT_HANDOVER_DIR
    jobs_dir: Path = DEFAULT_JOBS_DIR
    wb_output_dir: Path = DEFAULT_WB_OUTPUT_DIR
    fusion_work_dir: Path = DEFAULT_FUSION_WORK_DIR
    bundle_dir: Path = _bundle_dir_for(DEFAULT_HANDOVER_DIR)
    aws_csv: Path = _aws_csv_for(DEFAULT_HANDOVER_DIR)
    segmentation_timeout_s: float = DEFAULT_SEGMENTATION_TIMEOUT_S
    fusion_timeout_s: float = DEFAULT_FUSION_TIMEOUT_S
    # 추론 장치: cpu | gpu(여유 VRAM 부족 시 422) | auto(부족 시 CPU 폴백+경고).
    # GPU 는 vucatcher 상주 서비스와 공유라 프리플라이트가 필수다.
    gpu_mode: str = "cpu"
    gpu_min_free_mib: int = 3000
    # 연계 — 규약 파일명 산출물 디렉터리와 알고리즘 로그 API (비면 전송 안 함)
    product_dir: Path = DEFAULT_WB_OUTPUT_DIR.parent / "products"
    liaison_log_url: str | None = None
    liaison_system_code: str = "SC000000"
    liaison_algm_id: str = "AG000000"
    liaison_timeout_s: float = 5.0
    # 컨테이너 안에서 호스트 docker 데몬(docker.sock)으로 wbms 를 띄울 때의 경로 변환:
    # 호스트의 host_root(저장소 디렉터리)가 컨테이너의 container_root(/app)에 마운트돼 있다는
    # 뜻이며, docker run -v 의 호스트 경로를 그 기준으로 바꾼다(runner.translate_host_path).
    # host_root 가 None 이면(BACKEND_HOST_ROOT 미설정 = 호스트에서 직접 실행) 변환하지 않는다.
    host_root: Path | None = None
    container_root: Path = DEFAULT_CONTAINER_ROOT

    @property
    def model_dir(self) -> Path:
        return self.handover_dir / "03_model"

    @property
    def aux_dir(self) -> Path:
        return self.handover_dir / "06_aux"

    @property
    def l1_pre_dir(self) -> Path:
        return self.handover_dir / "05_l1_pre"

    @classmethod
    def from_env(cls) -> WbmsSettings:
        handover = _env_path("BACKEND_WBMS_HANDOVER_DIR", DEFAULT_HANDOVER_DIR)
        # 호스트 경로는 컨테이너 안에서 resolve 하면 안 된다(그 경로가 여기엔 없다) — 문자열 그대로.
        raw_host_root = os.getenv("BACKEND_HOST_ROOT", "").strip()
        raw_container_root = os.getenv("BACKEND_CONTAINER_ROOT", "").strip()
        settings = cls(
            docker_image=os.getenv("BACKEND_WBMS_IMAGE", DEFAULT_IMAGE),
            docker_bin=os.getenv("BACKEND_WBMS_DOCKER_BIN", DEFAULT_DOCKER_BIN),
            handover_dir=handover,
            jobs_dir=_env_path("BACKEND_WBMS_JOBS_DIR", DEFAULT_JOBS_DIR),
            wb_output_dir=_env_path("BACKEND_WBMS_WB_DIR", DEFAULT_WB_OUTPUT_DIR),
            fusion_work_dir=_env_path("BACKEND_WBMS_FUSION_DIR", DEFAULT_FUSION_WORK_DIR),
            bundle_dir=_env_path("BACKEND_WBMS_BUNDLE_DIR", _bundle_dir_for(handover)),
            aws_csv=_env_path("BACKEND_WBMS_AWS_CSV", _aws_csv_for(handover)),
            segmentation_timeout_s=_env_float(
                "BACKEND_WBMS_SEGMENTATION_TIMEOUT_S", DEFAULT_SEGMENTATION_TIMEOUT_S
            ),
            fusion_timeout_s=_env_float(
                "BACKEND_WBMS_FUSION_TIMEOUT_S", DEFAULT_FUSION_TIMEOUT_S
            ),
            gpu_mode=os.getenv("BACKEND_WBMS_GPU", "cpu").strip().lower() or "cpu",
            gpu_min_free_mib=int(_env_float("BACKEND_WBMS_GPU_MIN_FREE_MIB", 3000.0)),
            product_dir=_env_path("BACKEND_WBMS_PRODUCT_DIR", _env_path("BACKEND_WBMS_WB_DIR", DEFAULT_WB_OUTPUT_DIR).parent / "products"),
            liaison_log_url=(os.getenv("BACKEND_LIAISON_LOG_URL") or "").strip() or None,
            liaison_system_code=os.getenv("BACKEND_LIAISON_SYSTEM_CODE", "SC000000").strip() or "SC000000",
            liaison_algm_id=os.getenv("BACKEND_LIAISON_ALGM_ID", "AG000000").strip() or "AG000000",
            liaison_timeout_s=_env_float("BACKEND_LIAISON_TIMEOUT_S", 5.0),
            host_root=Path(raw_host_root) if raw_host_root else None,
            container_root=Path(raw_container_root) if raw_container_root else DEFAULT_CONTAINER_ROOT,
        )
        # 어댑터는 설정 객체 없이 runner.docker_prefix()/resolve_gpu_device() 를 부르므로
        # 경로 변환표와 GPU 프리플라이트용 docker/이미지는 runner 모듈 전역에 넣어 둔다
        # (BACKEND_HOST_ROOT 가 비면 변환표를 비워 호스트 직접 실행과 같아진다).
        from . import runner  # 지연 import — runner 가 이 모듈의 상수를 import 한다

        runner.set_path_map(settings.container_root, settings.host_root)
        runner.set_gpu_probe(settings.docker_bin, settings.docker_image)
        return settings
