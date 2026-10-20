from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings kept intentionally small and environment-friendly."""

    result_dir: Path
    predictor_plugins: str = ""
    api_prefix: str = "/api/v1"
    title: str = "Water Time-series Prediction API"
    version: str = "0.2.0"
    # 지역 레지스트리: 내장(저장소 regions/)과 사용자(운영 중 추가, 바인드 마운트) 디렉터리.
    regions_dir: Path = Path(__file__).resolve().parents[2] / "regions"
    regions_user_dir: Path = Path(__file__).resolve().parents[1] / "data" / "regions"
    gauge_forecaster_plugins: str = ""
    segmenter_plugins: str = ""
    # 위성 장면 업로드(POST /regions/{id}/satellite/scenes) 파일 하나의 최대 바이트 — 무제한 업로드로 디스크를 채우지 못하게
    scene_upload_max_bytes: int = 2 * 1024 * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        default_result_dir = Path(__file__).resolve().parents[1] / "data" / "results"
        default_regions_dir = Path(__file__).resolve().parents[2] / "regions"
        default_user_dir = Path(__file__).resolve().parents[1] / "data" / "regions"
        return cls(
            result_dir=Path(
                os.getenv("BACKEND_RESULT_DIR", str(default_result_dir))
            ).resolve(),
            predictor_plugins=os.getenv("PREDICTOR_PLUGINS", ""),
            regions_dir=Path(os.getenv("REGIONS_DIR", str(default_regions_dir))).resolve(),
            regions_user_dir=Path(os.getenv("REGIONS_USER_DIR", str(default_user_dir))).resolve(),
            gauge_forecaster_plugins=os.getenv("GAUGE_FORECASTER_PLUGINS", ""),
            segmenter_plugins=os.getenv("SEGMENTER_PLUGINS", ""),
            scene_upload_max_bytes=int(os.getenv("SCENE_UPLOAD_MAX_BYTES", str(2 * 1024 * 1024 * 1024))),
        )
