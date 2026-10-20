"""위성 영상 입력 컴포넌트 — 순수 도우미, APIClient multipart 계약, AppTest 배선."""

from __future__ import annotations

import sys
import textwrap
from datetime import date as Date
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:  # satellite_input_ui 는 slides.py 처럼 ui_next/ 를 sys.path 에 둔 채 평면 import 한다
    sys.path.insert(0, str(UI_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

from satellite_input_ui import (  # noqa: E402
    SENSOR_SUGGESTIONS,
    detector_status_from,
    next_steps_ko,
    quicklook_url,
    region_testbed,
    validate_submission,
)
from ui_next.api_client import APIClient, APIError  # noqa: E402
from ui_next.tests.test_api_client import FakeResponse, FakeSession  # noqa: E402

# --- 순수 도우미 -----------------------------------------------------------------------------


def test_validate_submission_requires_sensor_date_and_a_geotiff() -> None:
    errors = validate_submission(sensor="", image_date=None, pre_file=None, pre_path="")
    assert any("센서" in e for e in errors) and any("촬영일" in e for e in errors) and any("GeoTIFF" in e for e in errors)
    assert validate_submission(sensor="ICEYE", image_date=Date(2024, 5, 1), pre_file=("a.tif", b"x"), pre_path="") == []
    assert validate_submission(sensor="ICEYE", image_date="20240501", pre_file=None, pre_path="/srv/pre.tif") == []
    # 서버가 거부할 센서 이름·확장자·빈 파일은 화면에서 먼저 잡는다
    assert any("영문" in e for e in validate_submission(sensor="../x", image_date="20240501", pre_file=("a.tif", b"x"), pre_path=""))
    assert any(".tif" in e for e in validate_submission(sensor="ICEYE", image_date="20240501", pre_file=("a.png", b"x"), pre_path=""))
    assert any("비어" in e for e in validate_submission(sensor="ICEYE", image_date="20240501", pre_file=("a.tif", b""), pre_path=""))
    both = validate_submission(sensor="ICEYE", image_date="20240501", pre_file=("a.tif", b"x"), pre_path="", mask_file=("m.tif", b"x"), mask_path="/srv/m.tif")
    assert any("하나만" in e for e in both)


def test_detector_status_explains_when_detection_cannot_run() -> None:
    assert detector_status_from(None, None)["available"] is False
    assert "어댑터" in detector_status_from({"items": []}, {})["reason_ko"]
    docker = {"items": [{"id": "wbms-detect-water", "name": "WBMS", "execution": "docker",
                         "capabilities": {"sensors": {"iceye": {}, "planet": {}}, "testbeds": {"busan": {}, "dcd": {}}}}]}
    no_image = detector_status_from(docker, {"docker_image": "wbms:0.13", "image_available": False})
    assert no_image["available"] is False and "wbms:0.13" in no_image["reason_ko"]
    unknown = detector_status_from(docker, None)
    assert unknown["available"] is False and "wbms/status" in unknown["reason_ko"]
    ok = detector_status_from(docker, {"image_available": True, "weights_available": {"iceye": True, "planet": True}})
    assert ok["available"] is True and ok["testbeds"] == ["busan", "dcd"] and ok["sensors"] == ["iceye", "planet"]
    in_process = detector_status_from({"items": [{"id": "x", "execution": "in-process", "capabilities": {}}]}, None)
    assert in_process["available"] is True


def test_next_steps_and_quicklook_url_follow_the_registration_result() -> None:
    waiting = {"region_id": "han_river", "scene": {"scene_id": "20240501_ICEYE", "detection_available": False, "images": {"input": "quicklooks/20240501_ICEYE_input.png"}},
               "detection_job": None, "detection_note_ko": "'ICEYE' × '?' 를 받는 탐지 어댑터가 없어 잡을 제출하지 않았습니다."}
    steps = next_steps_ko(waiting)
    assert steps[0].startswith("사진만 등록") and any("탐지 어댑터가 없어" in s for s in steps)
    assert quicklook_url(waiting) == "/api/v1/regions/han_river/satellite/files/quicklooks/20240501_ICEYE_input.png"
    job = next_steps_ko(waiting | {"detection_job": {"job_id": "j1", "status": "queued"}, "detection_note_ko": None})
    assert "탐지 대기" in job[0] and "j1" in job[0]
    done = {"region_id": "r", "scene": {"scene_id": "s", "detection_available": True, "water_area_km2": 1.2345, "gt_available": True,
                                       "images": {"input": "quicklooks/s_input.png", "overlay": "quicklooks/s_overlay.png"}}}
    steps = next_steps_ko(done)
    assert steps[0].startswith("탐지 완료") and "1.23 km²" in steps[0] and any("GT" in s for s in steps)
    assert quicklook_url(done).endswith("quicklooks/s_overlay.png")
    assert quicklook_url({"region_id": "r", "scene": {}}) is None
    assert region_testbed({"satellite": {"wbms_testbed": "Busan"}}) == "busan" and region_testbed({}) == ""
    assert SENSOR_SUGGESTIONS == ("ICEYE", "PlanetScope", "Sentinel-1")


# --- APIClient multipart 계약 ------------------------------------------------------------------


def test_register_scene_sends_multipart_files_and_form() -> None:
    session = FakeSession([FakeResponse({"region_id": "han_river", "scene": {"scene_id": "20240501_ICEYE"}})])
    client = APIClient("http://backend:8000/", session=session)
    result = client.register_scene(
        "han_river", sensor=" ICEYE ", sensor_type="sar", image_date="2024-05-01",
        pre_file=("pre.tif", b"PRE"), mask_file=("mask.tif", b"MASK"), run_detection=True, testbed="busan", note="메모",
    )
    assert result["scene"]["scene_id"] == "20240501_ICEYE"
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "http://backend:8000/api/v1/regions/han_river/satellite/scenes")
    assert kwargs["data"] == {"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501", "run_detection": "true", "testbed": "busan", "note": "메모"}
    assert [(field, name, content, mime) for field, (name, content, mime) in kwargs["files"]] == [
        ("pre_file", "pre.tif", b"PRE", "image/tiff"), ("mask_file", "mask.tif", b"MASK", "image/tiff"),
    ]
    assert kwargs["timeout"] == 600.0
    # 서버 경로만 주면 files 없이 form 만
    session2 = FakeSession([FakeResponse({"region_id": "r", "scene": {}})])
    client2 = APIClient("http://backend:8000/", session=session2)
    client2.register_scene("r", sensor="PlanetScope", sensor_type="OPTIC", image_date="20240501", pre_path="/srv/pre.tif", gt_path="/srv/gt.tif")
    _m, _u, kwargs2 = session2.calls[0]
    assert kwargs2["files"] is None and kwargs2["data"]["pre_path"] == "/srv/pre.tif" and kwargs2["data"]["gt_path"] == "/srv/gt.tif"
    assert kwargs2["data"]["run_detection"] == "false"
    with pytest.raises(APIError):
        client2.register_scene("r", sensor="ICEYE", sensor_type="SAR", image_date="20240501")


def test_client_has_segmenters_and_wbms_status() -> None:
    session = FakeSession([FakeResponse({"items": [], "total": 0}), FakeResponse({"image_available": False})])
    client = APIClient("http://backend:8000/", session=session)
    assert client.segmenters()["total"] == 0
    assert client.wbms_status()["image_available"] is False
    assert [c[1] for c in session.calls] == ["http://backend:8000/api/v1/segmenters", "http://backend:8000/api/v1/wbms/status"]


# --- AppTest: 파일 업로드 → register_scene 호출 기록 -------------------------------------------------

_SCRIPT = textwrap.dedent(
    '''
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import streamlit as st
    from api_client import APIError
    from satellite_input_ui import render_satellite_input

    class FakeClient:
        def __init__(self):
            self.calls = []
        def segmenters(self):
            return {"items": []}
        def wbms_status(self):
            raise APIError("no wbms")
        def register_scene(self, region_id, **kwargs):
            st.session_state["_calls"] = st.session_state.get("_calls", []) + [{"region_id": region_id, **{k: (v[0], len(v[1])) if isinstance(v, tuple) else v for k, v in kwargs.items()}}]
            if kwargs.get("sensor") == "BAD":
                raise APIError("장면 자산 생성 실패: RasterioIOError", status_code=422)
            return {"region_id": region_id, "scene": {"scene_id": "20240501_" + kwargs["sensor"], "sensor": kwargs["sensor"], "sensor_type": kwargs["sensor_type"],
                                                       "date": "2024-05-01", "detection_available": False, "images": {"input": "quicklooks/x_input.png"}},
                    "detection_job": None, "detection_note_ko": None}
        def fetch_bytes(self, url):
            st.session_state["_fetched"] = url
            return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63f8cfc0f01f0005000101d1a2b34a0000000049454e44ae426082")

    st.session_state["gauge_cache"] = {"scenes:han_river": {"available": False}, "regions": [], "forecasters": ["keep"]}
    region = {"id": "han_river", "name": "한강", "satellite": {"available": False}}
    result = render_satellite_input(FakeClient(), region, key="t")
    st.session_state["_result"] = result
    '''
)


def _script_path(tmp_path) -> str:
    # 스크립트는 tmp 에 두고 ui_next/ 경로를 직접 넣는다(평면 import)
    path = tmp_path / "satin_app.py"
    path.write_text(_SCRIPT.replace("str(Path(__file__).resolve().parent)", repr(str(UI_DIR))), encoding="utf-8")
    return str(path)


def test_component_uploads_geotiff_and_records_multipart_call(tmp_path) -> None:
    app = AppTest.from_file(_script_path(tmp_path), default_timeout=20)
    app.run()
    assert not app.exception
    labels = [u.label for u in app.file_uploader]
    assert labels == ["전처리 GeoTIFF (필수)", "탐지 마스크 GeoTIFF (선택, 0=육지 1=수체)", "정답(GT) 라벨 GeoTIFF (선택)"]
    assert app.file_uploader[0].allowed_type == [".tif", ".tiff"]
    # 탐지 어댑터가 없으면 체크박스는 잠기고 이유가 보인다
    det = next(c for c in app.checkbox if c.label.startswith("마스크가 없으면 탐지 실행"))
    assert det.disabled is True
    assert any("탐지 실행을 켤 수 없습니다" in str(c.value) for c in app.caption)
    assert any(e.label == "고급: 서버 경로로 등록" for e in app.expander)

    # 아무것도 올리지 않고 제출 → 화면 검증 오류, 호출 없음
    app.button[0].click().run()
    assert any("GeoTIFF" in str(e.value) for e in app.error)
    assert "_calls" not in app.session_state

    # 파일을 올리고 제출 → register_scene 호출·성공 메시지·퀵룩·캐시 비움
    app.file_uploader[0].set_value(("scene.tif", b"II*\x00fake", "image/tiff"))
    app.date_input[0].set_value(Date(2024, 5, 1))
    app.button[0].click().run()
    assert not app.exception
    calls = app.session_state["_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["region_id"] == "han_river" and call["sensor"] == "ICEYE" and call["sensor_type"] == "SAR"
    assert call["image_date"] == "20240501" and call["pre_file"] == ("scene.tif", 8) and call["mask_file"] is None
    assert call["run_detection"] is False and call["pre_path"] is None
    assert any("등록됨: 20240501_ICEYE" in str(s.value) for s in app.success)
    assert any("사진만 등록" in str(i.value) for i in app.info)
    assert app.session_state["_fetched"] == "/api/v1/regions/han_river/satellite/files/quicklooks/x_input.png"
    assert app.session_state["_result"]["scene"]["scene_id"] == "20240501_ICEYE"
    cache = app.session_state["gauge_cache"]
    assert "scenes:han_river" not in cache and "regions" not in cache and cache["forecasters"] == ["keep"]
    assert app.session_state["t_last_result"] == {"region_id": "han_river", "scene_id": "20240501_ICEYE"}


def test_component_shows_readable_error_from_api(tmp_path) -> None:
    app = AppTest.from_file(_script_path(tmp_path), default_timeout=20)
    app.run()
    pick = next(s for s in app.selectbox if s.label == "센서")
    pick.set_value("직접 입력")
    custom = next(t for t in app.text_input if t.label.startswith("센서 이름"))
    custom.set_value("BAD")
    app.file_uploader[0].set_value(("scene.tif", b"II*\x00fake", "image/tiff"))
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["_calls"][0]["sensor"] == "BAD"
    assert any("등록할 수 없습니다" in str(e.value) and "RasterioIOError" in str(e.value) for e in app.error)
    assert app.session_state["_result"] is None


def test_slides_and_pipeline_use_the_component() -> None:
    slides = (UI_DIR / "slides.py").read_text(encoding="utf-8")
    pipeline = (UI_DIR / "pipeline_view.py").read_text(encoding="utf-8")
    assert "from satellite_input_ui import render_satellite_input" in slides
    assert "이 지역은 아직 위성 장면이 없습니다" in slides and 'st.expander("새 위성 장면 등록 (GeoTIFF 업로드)"' in slides
    assert "sl_area_goto_upload" in slides  # 면적 예측 탭: 장면 없는 지역 → 업로드 폼 안내
    assert "from satellite_input_ui import render_satellite_input" in pipeline
    assert "client._request(" not in pipeline.split("새 위성 데이터가 왔을 때")[1]
