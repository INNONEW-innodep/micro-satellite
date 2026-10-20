"""위성 영상 입력 컴포넌트 — 어느 지역이든 화면에서 GeoTIFF 를 올려 장면을 등록한다.

위성 장면이 없는 지역(수위계만 있는 지역)도 전처리 GeoTIFF 하나만 올리면
`POST /api/v1/regions/{id}/satellite/scenes` 로 등록되어 퀵룩·(마스크가 있으면) 오버레이·면적이
만들어지고, 탐지·면적·위성 수위 대조 단계가 열린다. 쉬운 화면(슬라이드 3)과 위성 파이프라인 화면이
같은 컴포넌트를 쓴다.

순수 도우미(`validate_submission`, `detector_status_from`, `next_steps_ko`, `quicklook_url`)는
Streamlit 없이 검사할 수 있게 분리해 두었다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date as Date
from datetime import datetime
from typing import Any

import streamlit as st
from api_client import APIClient, APIError

SENSOR_SUGGESTIONS: tuple[str, ...] = ("ICEYE", "PlanetScope", "Sentinel-1")
SENSOR_TYPE_DEFAULT: dict[str, str] = {"ICEYE": "SAR", "Sentinel-1": "SAR", "PlanetScope": "OPTIC"}
TIF_TYPES: tuple[str, ...] = ("tif", "tiff")
_SENSOR_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_DETECTOR_STATE_KEY = "satellite_detector_status"
_SCENE_CACHE_PREFIXES = ("scenes:", "regions", "region_detail:", "satlevels:", "img:", "tseval:", "detection:")


# --- 순수 도우미 --------------------------------------------------------------------------------


def normalize_sensor(name: str | None) -> str:
    return (name or "").strip()


def validate_submission(
    *,
    sensor: str | None,
    image_date: Date | str | None,
    pre_file: tuple[str, bytes] | None,
    pre_path: str | None,
    mask_file: tuple[str, bytes] | None = None,
    mask_path: str | None = None,
) -> list[str]:
    """제출 전 화면에서 잡을 수 있는 오류를 한국어 문장 목록으로 돌려준다(비어 있으면 통과)."""

    errors: list[str] = []
    sensor_name = normalize_sensor(sensor)
    if not sensor_name:
        errors.append("센서 이름을 입력하세요 (예: ICEYE, PlanetScope, Sentinel-1).")
    elif not _SENSOR_RE.match(sensor_name):
        errors.append("센서 이름은 영문·숫자·'_'·'-' 1~32자여야 합니다.")
    if not image_date:
        errors.append("촬영일을 고르세요.")
    else:
        digits = re.sub(r"[^0-9]", "", str(image_date))[:8]
        if len(digits) != 8:
            errors.append("촬영일은 YYYYMMDD 형식이어야 합니다.")
        else:
            try:
                datetime.strptime(digits, "%Y%m%d")
            except ValueError:
                errors.append(f"촬영일이 달력에 없는 날짜입니다: {digits}")
    if pre_file is None and not (pre_path or "").strip():
        errors.append("전처리 GeoTIFF 를 올리거나(.tif/.tiff) 고급에서 서버 경로를 적으세요.")
    if pre_file is not None and not _looks_like_tif(pre_file[0]):
        errors.append("전처리 파일은 GeoTIFF(.tif/.tiff)여야 합니다.")
    if pre_file is not None and len(pre_file[1]) == 0:
        errors.append("전처리 파일이 비어 있습니다.")
    if mask_file is not None and not _looks_like_tif(mask_file[0]):
        errors.append("마스크 파일은 GeoTIFF(.tif/.tiff)여야 합니다.")
    if mask_file is not None and (mask_path or "").strip():
        errors.append("마스크는 업로드와 서버 경로 중 하나만 주세요.")
    return errors


def _looks_like_tif(name: str) -> bool:
    return str(name or "").lower().rsplit(".", 1)[-1] in TIF_TYPES


def detector_status_from(segmenters: Mapping[str, Any] | None, wbms_status: Mapping[str, Any] | None) -> dict[str, Any]:
    """`/segmenters` + `/wbms/status` 응답으로 '탐지 실행' 체크박스를 켤 수 있는지 판단한다.

    반환: ``{"available": bool, "reason_ko": str, "adapter_id": str | None, "adapter_name": str | None,
    "execution": str | None, "testbeds": [...], "sensors": [...]}``
    """

    items = list((segmenters or {}).get("items") or []) if isinstance(segmenters, Mapping) else []
    if not items:
        return {"available": False, "reason_ko": "등록된 수체 탐지 어댑터가 없습니다(SEGMENTER_PLUGINS 로 등록). 마스크를 직접 올려 주세요.",
                "adapter_id": None, "adapter_name": None, "execution": None, "testbeds": [], "sensors": []}
    adapter = items[0]
    caps = adapter.get("capabilities") or {}
    testbeds = sorted((caps.get("testbeds") or {}).keys())
    sensors = sorted((caps.get("sensors") or {}).keys())
    execution = str(adapter.get("execution") or "")
    base = {"adapter_id": adapter.get("id"), "adapter_name": adapter.get("name"), "execution": execution,
            "testbeds": testbeds, "sensors": sensors}
    if execution == "docker":
        status = wbms_status if isinstance(wbms_status, Mapping) else {}
        if not status:
            return base | {"available": False, "reason_ko": "WBMS 상태(/wbms/status)를 확인하지 못해 탐지 실행을 켤 수 없습니다. 마스크를 직접 올려 주세요."}
        if not status.get("image_available"):
            image = status.get("docker_image") or "wbms"
            return base | {"available": False, "reason_ko": f"이 서버에는 도커 이미지 {image} 가 없어 탐지를 돌릴 수 없습니다(호스트 docker 필요). 마스크를 직접 올려 주세요."}
        weights = status.get("weights_available") or caps.get("weights_available") or {}
        if isinstance(weights, Mapping) and weights and not any(weights.values()):
            return base | {"available": False, "reason_ko": "탐지 모델 가중치(.h5)가 없어 탐지를 돌릴 수 없습니다. 마스크를 직접 올려 주세요."}
    return base | {"available": True, "reason_ko": ""}


def region_testbed(region: Mapping[str, Any]) -> str:
    return str(((region.get("satellite") or {}).get("wbms_testbed")) or "").strip().lower()


def quicklook_url(result: Mapping[str, Any], *, api_prefix: str = "/api/v1") -> str | None:
    """등록 응답의 scene.images.input 을 `/satellite/files/…` 상대 URL 로 만든다."""

    scene = result.get("scene") or {}
    images = scene.get("images") or {}
    rel = images.get("overlay") or images.get("input")
    region_id = result.get("region_id")
    if not rel or not region_id:
        return None
    return f"{api_prefix}/regions/{region_id}/satellite/files/{rel}"


def next_steps_ko(result: Mapping[str, Any]) -> list[str]:
    """등록 결과에 따라 사용자가 다음에 볼 것을 문장 목록으로 만든다."""

    scene = result.get("scene") or {}
    steps: list[str] = []
    if scene.get("detection_available"):
        area = scene.get("water_area_km2")
        area_txt = f" 탐지 수체 면적 {area:.2f} km²." if isinstance(area, (int, float)) else ""
        steps.append(f"탐지 완료 — 퀵룩·오버레이(청록 = 물)와 공통 격자 마스크가 만들어졌습니다.{area_txt}")
        steps.append("같은 센서 장면이 2개 이상이면 면적 시계열 그래프가, 지점 연결(satellite_location)이 있으면 위성 수위 vs 수위계 대조가 열립니다.")
        steps.append("면적 예측 샘플(④ 시계열 예측 › ② 수체 면적)은 저장소 regions/<id>/satellite/scenes.json 을 UI 시작 때 읽으므로, 사용자 등록 장면은 UI 재시작 뒤 반영됩니다.")
    else:
        job = result.get("detection_job")
        if job:
            steps.append(f"탐지 대기 — WBMS 탐지 잡 {job.get('job_id')} ({job.get('status')}) 를 제출했습니다. 마스크가 나오면 같은 센서·촬영일로 다시 등록하면 갱신됩니다.")
        else:
            steps.append("사진만 등록되었습니다(탐지 대기). 탐지 마스크 GeoTIFF(0=육지, 1=수체)가 생기면 같은 센서·촬영일로 다시 등록하면 오버레이·면적이 붙습니다.")
        if result.get("detection_note_ko"):
            steps.append(str(result["detection_note_ko"]))
    if scene.get("gt_available"):
        steps.append("GT 라벨이 함께 등록되어 정답 오버레이(노랑)를 볼 수 있습니다.")
    return steps


def invalidate_scene_caches(region_id: str | None = None) -> None:
    """장면·지역 캐시(gauge_cache)를 비워 새 장면이 바로 보이게 한다."""

    try:
        cache = st.session_state.get("gauge_cache")
    except Exception:  # noqa: BLE001 - 세션 밖(순수 테스트)
        return
    if not isinstance(cache, dict):
        return
    for key in list(cache):
        if key.startswith(_SCENE_CACHE_PREFIXES):
            cache.pop(key, None)
    for state_key in ("pl_area_done", "pl_level_done"):
        st.session_state.pop(state_key, None)
    # 면적 예측 샘플(마스크 시계열)도 다시 읽는다 — 새 마스크가 면적 예측 탭에 바로 붙게
    try:
        try:
            from .samples import refresh_region_samples
        except ImportError:  # pragma: no cover - Streamlit entry point
            from samples import refresh_region_samples
        refresh_region_samples()
    except Exception:  # noqa: BLE001 - 샘플 재구성 실패가 등록 성공을 가리면 안 된다
        pass


# --- 화면 ------------------------------------------------------------------------------------


def _detector_status(client: APIClient) -> dict[str, Any]:
    cached = st.session_state.get(_DETECTOR_STATE_KEY)
    if isinstance(cached, dict):
        return cached
    segmenters: Mapping[str, Any] | None
    wbms: Mapping[str, Any] | None
    try:
        segmenters = client.segmenters()
    except APIError:
        segmenters = None
    try:
        wbms = client.wbms_status()
    except APIError:
        wbms = None
    status = detector_status_from(segmenters, wbms)
    st.session_state[_DETECTOR_STATE_KEY] = status
    return status


def _uploaded(file: Any) -> tuple[str, bytes] | None:
    if file is None:
        return None
    return (str(getattr(file, "name", "upload.tif")), bytes(file.getvalue()))


def render_satellite_input(client: APIClient, region: Mapping[str, Any], *, key: str) -> Mapping[str, Any] | None:
    """위성 영상 입력 폼. 등록에 성공하면 응답(dict)을 돌려주고 장면 캐시를 비운다."""

    region_id = str(region.get("id") or "")
    region_name = str(region.get("name") or region_id)
    detector = _detector_status(client)
    testbed_default = region_testbed(region)
    known_sensors = list((region.get("satellite") or {}).get("sensors") or [])

    with st.form(f"{key}_form", clear_on_submit=False):
        st.markdown(f"**{region_name}** 에 새 위성 장면을 등록합니다. 전처리(정사·방사보정)된 GeoTIFF 한 장이면 됩니다.")
        pre_upload = st.file_uploader("전처리 GeoTIFF (필수)", type=list(TIF_TYPES), key=f"{key}_pre",
                                      help="SAR 는 γ⁰ dB 1밴드, 광학은 R·G·B(+NIR) 반사도. 좌표계가 있는 GeoTIFF 여야 퀵룩과 면적을 만들 수 있습니다.")
        c1, c2 = st.columns(2)
        mask_upload = c1.file_uploader("탐지 마스크 GeoTIFF (선택, 0=육지 1=수체)", type=list(TIF_TYPES), key=f"{key}_mask",
                                       help="이미 탐지된 마스크가 있으면 올리세요. 없으면 사진만 등록되고 '탐지 대기' 상태가 됩니다.")
        gt_upload = c2.file_uploader("정답(GT) 라벨 GeoTIFF (선택)", type=list(TIF_TYPES), key=f"{key}_gt",
                                     help="사람이 그린 정답 수체 라벨(1=수체). 있으면 노란 오버레이로 함께 보여 줍니다.")
        s1, s2, s3 = st.columns([2, 1, 1])
        sensor_options = list(dict.fromkeys([*known_sensors, *SENSOR_SUGGESTIONS]))
        sensor_pick = s1.selectbox("센서", [*sensor_options, "직접 입력"], key=f"{key}_sensor_pick",
                                   help="ICEYE(SAR)·PlanetScope(광학)·Sentinel-1(SAR) 이 기본 제안입니다. 다른 센서는 '직접 입력'.")
        sensor_custom = s1.text_input("센서 이름 (직접 입력 시)", value="", key=f"{key}_sensor_custom", placeholder="예: KOMPSAT-5")
        type_index = 0 if SENSOR_TYPE_DEFAULT.get(sensor_pick, "OPTIC") == "SAR" else 1
        sensor_type = s2.radio("영상 종류", ["SAR", "OPTIC"], index=type_index, key=f"{key}_type", horizontal=True,
                               help="SAR = 레이더(구름 무관), OPTIC = 광학(가시광·근적외). ICEYE·Sentinel-1 은 SAR, PlanetScope 는 OPTIC 으로 자동 지정되며 이 선택은 '직접 입력' 센서에만 적용됩니다.")
        image_date = s3.date_input("촬영일", value=Date.today(), min_value=Date(2000, 1, 1), max_value=Date.today(), key=f"{key}_date")
        note = st.text_input("메모 (선택)", value="", key=f"{key}_note", placeholder="예: 2026-09 신규 촬영, 궤도 12")
        det_help = detector["reason_ko"] if not detector["available"] else (
            f"어댑터 {detector.get('adapter_name') or detector.get('adapter_id')} 로 탐지 잡을 제출합니다"
            f"(받는 센서: {', '.join(detector.get('sensors') or []) or '-'} · 테스트베드: {', '.join(detector.get('testbeds') or []) or '-'})."
        )
        run_detection = st.checkbox("마스크가 없으면 탐지 실행(WBMS)", value=False, disabled=not detector["available"],
                                    key=f"{key}_run_det", help=det_help)
        if not detector["available"]:
            st.caption(f"탐지 실행을 켤 수 없습니다 — {detector['reason_ko']}")
        elif testbed_default and testbed_default not in (detector.get("testbeds") or []):
            st.caption(f"이 지역의 테스트베드 '{testbed_default}' 는 탐지 카탈로그에 없어 잡이 제출되지 않을 수 있습니다.")
        elif not testbed_default:
            st.caption("이 지역은 탐지 카탈로그(testbed)에 없습니다. 탐지를 돌리려면 아래 고급에서 테스트베드 키를 적어 주세요.")
        with st.expander("고급: 서버 경로로 등록", expanded=False):
            st.caption("파일이 이미 API 서버(호스트)에 있으면 업로드 대신 경로를 적습니다. 업로드와 경로를 함께 주면 업로드가 우선합니다.")
            pre_path = st.text_input("전처리 GeoTIFF 서버 경로", value="", key=f"{key}_pre_path",
                                     placeholder="/home/.../data/incoming/handover/05_l1_pre/…/Pre_*.tif")
            mask_path = st.text_input("탐지 마스크 GeoTIFF 서버 경로 (선택)", value="", key=f"{key}_mask_path")
            gt_path = st.text_input("GT 라벨 GeoTIFF 서버 경로 (선택)", value="", key=f"{key}_gt_path")
            testbed = st.text_input("탐지 테스트베드 키 (선택)", value=testbed_default, key=f"{key}_testbed",
                                    help="탐지 잡용 카탈로그 키. 비우면 지역 정의의 wbms_testbed 를 씁니다.")
        submitted = st.form_submit_button("장면 등록 ▶", type="primary", width="stretch")

    if not submitted:
        return None

    sensor = normalize_sensor(sensor_custom) if sensor_pick == "직접 입력" or sensor_custom.strip() else sensor_pick
    # st.form 안에서는 센서를 바꿔도 라디오 기본값이 다시 계산되지 않는다(예: PlanetScope 를 골라도 SAR 로 남음)
    # → 종류가 물리적으로 정해진 센서는 제출 시점에 확정한다
    sensor_type = SENSOR_TYPE_DEFAULT.get(sensor, sensor_type)
    pre_file = _uploaded(pre_upload)
    mask_file = _uploaded(mask_upload)
    gt_file = _uploaded(gt_upload)
    errors = validate_submission(sensor=sensor, image_date=image_date, pre_file=pre_file, pre_path=pre_path,
                                 mask_file=mask_file, mask_path=mask_path)
    if errors:
        for message in errors:
            st.error(message)
        return None

    with st.spinner("장면을 등록하고 퀵룩·오버레이를 만드는 중…"):
        try:
            result = client.register_scene(
                region_id, sensor=sensor, sensor_type=sensor_type, image_date=image_date.strftime("%Y%m%d"),
                pre_file=pre_file, mask_file=mask_file, gt_file=gt_file,
                pre_path=(pre_path or "").strip() or None, mask_path=(mask_path or "").strip() or None,
                gt_path=(gt_path or "").strip() or None, run_detection=bool(run_detection),
                testbed=(testbed or "").strip() or None, note=(note or "").strip() or None,
            )
        except APIError as exc:
            st.error(_error_ko(exc))
            return None

    scene = result.get("scene") or {}
    st.success(f"등록됨: {scene.get('scene_id', '?')} · {scene.get('sensor', sensor)} {scene.get('sensor_type', sensor_type)} · {scene.get('date', '')}")
    for line in next_steps_ko(result):
        st.info(line)
    url = quicklook_url(result)
    if url:
        try:
            st.image(client.fetch_bytes(url), caption=f"퀵룩 · {scene.get('scene_id', '')}", width=560)
        except APIError as exc:
            st.warning(f"퀵룩을 불러오지 못했습니다: {exc}")
    with st.expander("등록된 장면 기록 (scenes.json 항목)", expanded=False):
        st.json({k: v for k, v in scene.items() if k not in ("images",)} | {"images": scene.get("images")})
    invalidate_scene_caches(region_id)
    st.session_state[f"{key}_last_result"] = {"region_id": region_id, "scene_id": scene.get("scene_id")}
    return result


def _error_ko(exc: APIError) -> str:
    code = exc.status_code
    text = str(exc)
    if code == 413:
        return f"파일이 너무 큽니다 — {text}"
    if code == 422:
        return f"등록할 수 없습니다 — {text}"
    if code == 404:
        return f"지역을 찾지 못했습니다 — {text}"
    if code in (502, 503):
        return f"탐지 서비스가 응답하지 않습니다 — {text}"
    return f"등록 실패 — {text}"


__all__ = [
    "SENSOR_SUGGESTIONS",
    "TIF_TYPES",
    "detector_status_from",
    "invalidate_scene_caches",
    "next_steps_ko",
    "normalize_sensor",
    "quicklook_url",
    "region_testbed",
    "render_satellite_input",
    "validate_submission",
]
