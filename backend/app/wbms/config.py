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
        return cls(
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
        )
