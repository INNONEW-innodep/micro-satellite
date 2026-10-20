"""지역 레지스트리(`regions/<id>/satellite/`)에서 수체 면적 시계열 예측 샘플을 만든다.

`scripts/make_region_scene_assets.py`가 만든 센서별 공통 격자 마스크 PNG를 그대로
프레임으로 쓴다. 코드에 지역·날짜를 적지 않으므로 새 지역은 자산만 추가하면 목록에
나타난다. 마스크는 WBMS 실모델 탐지 산출이며(부산은 GT도 있음), 면적은 리샘플 격자의
픽셀 면적 × 화소 수로 계산한 근사값이다.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType

try:  # package imports in tests; direct imports when Streamlit executes app.py
    from .samples import SampleDataset, SampleFrame
except ImportError:  # pragma: no cover - Streamlit entry point
    from samples import SampleDataset, SampleFrame

REGIONS_DIR = Path(__file__).resolve().parent.parent / "regions"
SENSOR_KO = {"ICEYE": "ICEYE SAR", "PlanetScope": "PlanetScope 광학"}


def region_sample_id(region_id: str, sensor: str) -> str:
    return f"region:{region_id}:{sensor}"


def _load_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def user_regions_dir() -> Path | None:
    """백엔드와 같은 규약의 사용자 지역 디렉터리(REGIONS_USER_DIR). API 로 등록한 장면은 여기에 쌓인다."""

    raw = os.getenv("REGIONS_USER_DIR")
    if raw:
        return Path(raw).expanduser()
    default = Path(__file__).resolve().parent.parent / "backend" / "data" / "regions"
    return default if default.is_dir() else None


def region_mask_datasets(regions_dir: Path = REGIONS_DIR, user_dir: Path | None = None) -> tuple[SampleDataset, ...]:
    """지역 × 센서마다 프레임 2장 이상인 샘플을 만든다. 같은 지역 id 가 사용자 디렉터리에도 있으면 그쪽(최신 등록)을 쓴다."""

    out: list[SampleDataset] = []
    user_dir = user_dir if user_dir is not None else user_regions_dir()
    candidates: dict[str, Path] = {}
    for base in (regions_dir, user_dir):
        if base is None or not base.is_dir():
            continue
        for region_dir in sorted(base.iterdir()):
            if (region_dir / "region.json").is_file():
                candidates[region_dir.name] = region_dir  # 사용자 디렉터리가 나중이라 같은 id 를 덮어쓴다
    for _, region_dir in sorted(candidates.items()):
        manifest = _load_json(region_dir / "region.json")
        scenes_payload = _load_json(region_dir / "satellite" / "scenes.json")
        if not manifest or not scenes_payload:
            continue
        region_id = str(manifest.get("id") or region_dir.name)
        region_name = str(manifest.get("name") or region_id)
        weather = manifest.get("weather") or {}
        by_sensor: dict[str, list[dict]] = {}
        for scene in scenes_payload.get("scenes") or []:
            if scene.get("frame"):
                by_sensor.setdefault(str(scene["sensor"]), []).append(scene)
        for sensor, scenes in by_sensor.items():
            scenes = sorted(scenes, key=lambda s: str(s["date"]))
            frames: list[SampleFrame] = []
            pixel_area = None
            for scene in scenes:
                frame_meta = scene["frame"]
                mask_path = region_dir / "satellite" / frame_meta["path"]
                overlay_rel = (scene.get("images") or {}).get("overlay")
                overlay_path = region_dir / "satellite" / overlay_rel if overlay_rel else None
                try:
                    mask_bytes = mask_path.read_bytes()
                except OSError:
                    continue
                context = overlay_path.read_bytes() if overlay_path and overlay_path.is_file() else None
                frames.append(
                    SampleFrame(
                        name=mask_path.name,
                        observed_on=date.fromisoformat(str(scene["date"])),
                        png_bytes=mask_bytes,
                        water_level_m=None,
                        context_png_bytes=context,
                    )
                )
                pixel_area = float(frame_meta.get("pixel_area_m2") or pixel_area or 9.0)
            if len(frames) < 2:
                continue
            last = frames[-1].observed_on
            gt = any(s.get("gt_available") for s in scenes)
            out.append(
                SampleDataset(
                    sample_id=region_sample_id(region_id, sensor),
                    display_name=f"{region_name} · {SENSOR_KO.get(sensor, sensor)} {len(frames)}시점",
                    region_name=region_name,
                    scenario_name="WBMS 실모델 탐지 마스크 시계열",
                    summary_ko=(
                        f"{region_name}의 {SENSOR_KO.get(sensor, sensor)} {len(frames)}개 장면을 인도 WBMS 모델로 탐지한 "
                        f"수체 마스크를 센서 공통 격자({frames[0].png_bytes and ''}약 {pixel_area ** 0.5:.0f} m 화소)로 맞춘 시계열입니다."
                    ),
                    frames=tuple(frames),
                    target_dates=tuple(last + timedelta(days=day) for day in range(1, 4)),
                    weather_rows=(),
                    water_level_config=None,
                    model_options=MappingProxyType(
                        {
                            "max_daily_area_change_pct": 1.0,
                            "max_total_area_change_pct": 25.0,
                            "rainfall_response_pct_per_20mm": 0.2,
                            "rainfall_memory_decay": 0.65,
                        }
                    ),
                    recommended_model_id="irregular-area-trend",
                    pixel_area_m2=pixel_area,
                    threshold=0.5,
                    synthetic_masks=False,
                    derived_demo=False,
                    measured_data=False,
                    trained_model=True,
                    recommended=True,
                    raw_source_available=True,
                    raw_data_provenance_known=True,
                    georeferenced=False,
                    pixel_area_known=True,
                    area_validation_allowed=gt,
                    water_level_validation_allowed=False,
                    provenance=f"regions/{region_id}/satellite/scenes.json; sensor={sensor}; frames={len(frames)}",
                    weather_provenance="runtime_kma_asos_historical_replay",
                    disclaimer_ko=(
                        "WBMS 실모델이 탐지한 마스크를 공통 격자로 리샘플링한 시계열입니다. 면적은 리샘플 격자 "
                        "픽셀 면적으로 계산한 근사값이며, 미래 정답이 없어 예측 성능 검증에는 쓰지 않습니다."
                        if gt
                        else "WBMS 실모델이 탐지한 마스크(GT 라벨 없음)를 공통 격자로 리샘플링한 시계열입니다. 탐지 정확도는 "
                        "검증되지 않았고, 면적은 근사값이며 미래 정답이 없습니다."
                    ),
                    data_classification="region_registry_wbms_detection_masks",
                    data_classification_ko="지역 레지스트리 · WBMS 실모델 탐지 마스크",
                    source_description_ko=(
                        f"인수 패키지 05_l1_pre 전처리 장면을 WBMS 컨테이너(wbms:0.13)로 탐지한 결과입니다. "
                        f"원본은 {region_name} {SENSOR_KO.get(sensor, sensor)} 장면 {len(frames)}개."
                    ),
                    mask_provenance_ko=(
                        "전체 해상도 WB 마스크(0=육지, 1=수체)를 센서별 공통 범위로 잘라 긴 변 640px로 최근접 리샘플링한 PNG입니다."
                    ),
                    weather_provenance_ko=(
                        f"빠른 실행 시 {weather.get('station_name') or '지역'} ASOS {weather.get('station_id') or ''} 지점의 과거 일자료를 "
                        "조회해 목표 기간 forcing으로 씁니다. 실관측이며 예보가 아닙니다."
                    ),
                    water_level_provenance_ko="면적 시계열 샘플에는 수위가 없습니다. 지점 수위는 '수위 예측' 화면에서 수위계 기반으로 따로 예측합니다.",
                    region_label_note_ko=f"지역 정의: regions/{region_id}/region.json",
                    intended_use_ko="위성 탐지 마스크의 날짜 정렬, 면적 변화, 기준선 모델 교체 흐름 시연",
                    not_suitable_for_ko="경계 정확도 평가, 절대 수위 산출, 미래 정답 기반 RMSE 근거",
                    source_assets=tuple(str(s.get("source_mask") or "") for s in scenes),
                    actual_event_data=False,
                    region_verified=True,
                    observation_dates_verified=True,
                    classification_badge_ko="실모델 탐지 마스크" + (" · GT 있음" if gt else " · GT 없음"),
                    sensor_name_ko=SENSOR_KO.get(sensor, sensor),
                    source_manifest_path=f"regions/{region_id}/satellite/scenes.json",
                    kma_station_id=str(weather.get("station_id")) if weather.get("station_id") else None,
                    kma_station_name=str(weather.get("station_name")) if weather.get("station_name") else None,
                    region_id=region_id,
                )
            )
    return tuple(out)
