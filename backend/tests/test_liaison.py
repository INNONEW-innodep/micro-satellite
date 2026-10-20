"""과제 통합 플랫폼 연계 — 파라미터 6종 수신, 규약 파일명 산출물, 알고리즘 로그 4종.

규격: 알고리즘_산출물_명명규칙(2026-09-18), 알고리즘 적용 로그 포맷 v1.0.0.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from backend.app.wbms.liaison import (
    LiaisonError,
    derive_product_id,
    export_product,
    liaison_params,
    product_filename,
)
from backend.app.wbms.liaison_log import LOG_LEVEL, WORK_STATUS, LiaisonLogger, build_log, format_log_time
from backend.tests.test_wbms import SCENE_DATE, SCENE_STAMP, make_client, wait_terminal


# ── 규약 파일명 ───────────────────────────────────────────────────────────


def test_product_filename_follows_spec() -> None:
    assert product_filename("PT000003", "S1A_IW_GRDH_20260905") == "PT000003_S1A_IW_GRDH_20260905.tif"
    assert product_filename("PT000003", "ICEYE_X5_SLC_SM_23467_20200302T183857") == (
        "PT000003_ICEYE_X5_SLC_SM_23467_20200302T183857.tif"
    )
    with pytest.raises(LiaisonError):
        product_filename("PT3", "x")
    with pytest.raises(LiaisonError):
        product_filename("PT000003", "../etc/passwd")


def test_derive_product_id_prefers_provenance_then_stem(tmp_path: Path) -> None:
    pre = tmp_path / "Pre_Busan_ICEYE_20200302T183857.tif"
    pre.write_bytes(b"x")
    assert derive_product_id(pre) == "Pre_Busan_ICEYE_20200302T183857"
    (tmp_path / "Pre_Busan_ICEYE_20200302T183857_meta.json").write_text(
        json.dumps({"Provenance": {"input_file": "ICEYE_X5_SLC_SM_23467_20200302T183857.h5"}}), encoding="utf-8"
    )
    assert derive_product_id(pre) == "ICEYE_X5_SLC_SM_23467_20200302T183857"
    assert derive_product_id(None) is None


def test_export_product_copies_once_and_is_idempotent(tmp_path: Path) -> None:
    mask = tmp_path / "WB_Busan_ICEYE_20200302T183857.tif"
    mask.write_bytes(b"GEOTIFF")
    out = export_product(mask, tmp_path / "products", product_id="ICEYE_X5_SLC_SM_23467_20200302T183857")
    assert out.name == "PT000003_ICEYE_X5_SLC_SM_23467_20200302T183857.tif"
    assert out.read_bytes() == b"GEOTIFF"
    again = export_product(mask, tmp_path / "products", product_id="ICEYE_X5_SLC_SM_23467_20200302T183857")
    assert again == out and mask.is_file(), "원본은 그대로, 두 번째 호출은 멱등"


def test_liaison_params_validates_and_normalises() -> None:
    assert liaison_params({"order_id": 1, "scene_id": " 7 ", "work_id": None, "force": True}) == {
        "order_id": "1", "scene_id": "7",
    }
    with pytest.raises(LiaisonError):
        liaison_params({"product_type": "PT3"})
    with pytest.raises(LiaisonError):
        liaison_params({"product_id": "a/b"})


# ── 로그 페이로드 ────────────────────────────────────────────────────────


def test_build_log_matches_spec_shape() -> None:
    params = {"order_id": "1", "scene_id": "1", "work_id": "1", "algm_id": "AG000002", "product_type": "PT000002"}
    done = build_log(status="succeeded", msg="0000 산출물이 생성 되었습니다.", params=params, system_code="SC000001",
                     comp_time="2025-11-18T14:12:10Z", proc_time_s=60.0)
    assert set(done) == {"log_time", "log_level", "system_code", "contents"}
    assert len(done["log_time"]) == 14 and done["log_time"].isdigit()
    assert done["log_level"] == LOG_LEVEL["info"] and done["system_code"] == "SC000001"
    assert done["contents"] == {
        "order_id": "1", "scene_id": "1", "work_id": "1", "algm_id": "AG000002", "product_type": "PT000002",
        "msg": "0000 산출물이 생성 되었습니다.", "work_status": "WS000002",
        "algm_comp_time": "20251118231210", "algm_proc_time": "60",
    }, "규격 샘플과 같은 키·코드·KST 시각"
    running = build_log(status="running", msg="m", params=params, system_code="SC000001", exec_time="2025-11-18T14:19:10Z")
    assert running["contents"]["work_status"] == "WS000001" and running["contents"]["algm_exec_time"] == "20251118231910"
    assert "algm_comp_time" not in running["contents"]
    error = build_log(status="failed", msg="m", params=params, system_code="SC000001")
    assert error["log_level"] == LOG_LEVEL["error"] and error["contents"]["work_status"] == "WS000003"
    with pytest.raises(ValueError):
        build_log(status="done", msg="m", params=params, system_code="SC000001")


def test_format_log_time_is_kst_14_digits() -> None:
    from datetime import datetime, timezone

    assert format_log_time(datetime(2025, 7, 21, 1, 23, 19, tzinfo=timezone.utc)) == "20250721102319"


# ── 발신기: 가짜 수신 서버로 4종 순서 검증 ─────────────────────────────────


class _Collector(BaseHTTPRequestHandler):
    received: list[dict] = []
    reject = False

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).received.append(body)
        self.send_response(500 if type(self).reject else 200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args) -> None:  # 조용히
        return


@pytest.fixture
def log_server():
    _Collector.received = []
    _Collector.reject = False
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/log", _Collector
    finally:
        server.shutdown()


def test_logger_noop_without_url() -> None:
    logger = LiaisonLogger(None)
    assert not logger.enabled
    assert logger.emit({"contents": {"work_status": "WS000000"}}) is None


def test_async_job_emits_queued_running_succeeded_and_exports_product(tmp_path: Path, log_server) -> None:
    url, collector = log_server
    client, runner, settings = make_client(tmp_path)
    adapter = client.app.state.wbms_segmentation
    adapter.liaison = LiaisonLogger(url, system_code="SC000009", algm_id="AG000009")

    response = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE,
              "order_id": "42", "scene_id": "7", "work_id": "W-1"},
    )
    assert response.status_code == 202, response.text
    job = wait_terminal(client, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert job["request"]["order_id"] == "42" and job["request"]["work_id"] == "W-1"

    # 규약 파일명 산출물 — 전처리 사이드카 Provenance.input_file 의 원본 ID 로 파생
    assert job["result"]["product_id"] == f"ICEYE_X5_SLC_SM_23467_{SCENE_STAMP}"
    product = Path(job["result"]["product_tif"])
    assert product.name == f"PT000003_ICEYE_X5_SLC_SM_23467_{SCENE_STAMP}.tif"
    assert product.is_file() and product.parent == settings.product_dir
    assert product.read_bytes() == Path(job["result"]["mask_tif"]).read_bytes()

    statuses = [r["contents"]["work_status"] for r in collector.received]
    assert statuses == [WORK_STATUS["queued"], WORK_STATUS["running"], WORK_STATUS["succeeded"]]
    for rec in collector.received:
        assert rec["system_code"] == "SC000009"
        assert rec["contents"]["order_id"] == "42" and rec["contents"]["scene_id"] == "7"
        assert rec["contents"]["work_id"] == "W-1", "외부 work_id 가 있으면 그대로"
        assert rec["contents"]["algm_id"] == "AG000009" and rec["contents"]["product_type"] == "PT000003"
    done = collector.received[-1]["contents"]
    assert "algm_comp_time" in done and done["algm_proc_time"].isdigit()
    assert product.name in done["msg"]
    assert "algm_exec_time" in collector.received[1]["contents"]
    assert not any("liaison log" in w for w in job.get("warnings", []))


def test_failed_job_emits_error_event_and_uses_job_id_as_work_id(tmp_path: Path, log_server) -> None:
    url, collector = log_server
    client, runner, _ = make_client(tmp_path)
    client.app.state.wbms_segmentation.liaison = LiaisonLogger(url)
    runner.mode = "fail"
    created = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE, "order_id": "9"},
    ).json()
    job = wait_terminal(client, created["job_id"])
    assert job["status"] == "failed"
    statuses = [r["contents"]["work_status"] for r in collector.received]
    assert statuses == [WORK_STATUS["queued"], WORK_STATUS["running"], WORK_STATUS["failed"]]
    last = collector.received[-1]
    assert last["log_level"] == LOG_LEVEL["error"]
    assert last["contents"]["work_id"] == job["job_id"], "외부 work_id 가 없으면 우리 job_id"
    assert "에러" in last["contents"]["msg"]


def test_cache_hit_emits_only_completion_and_still_exports_product(tmp_path: Path, log_server) -> None:
    url, collector = log_server
    client, runner, settings = make_client(tmp_path)
    client.app.state.wbms_segmentation.liaison = LiaisonLogger(url)
    first = client.post("/api/v1/wbms/segmentation/jobs",
                        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE}).json()
    wait_terminal(client, first["job_id"])
    collector.received.clear()

    second = client.post("/api/v1/wbms/segmentation/jobs",
                         json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE,
                               "product_id": "ICEYE_X5_SLC_SM_23467_20200302T183857"}).json()
    assert second["cached"] is True and second["status"] == "succeeded"
    assert Path(second["result"]["product_tif"]).name == "PT000003_ICEYE_X5_SLC_SM_23467_20200302T183857.tif"
    assert [r["contents"]["work_status"] for r in collector.received] == [WORK_STATUS["succeeded"]]


def test_log_delivery_failure_is_a_warning_not_a_job_failure(tmp_path: Path, log_server) -> None:
    url, collector = log_server
    collector.reject = True
    client, runner, _ = make_client(tmp_path)
    client.app.state.wbms_segmentation.liaison = LiaisonLogger(url)
    created = client.post("/api/v1/wbms/segmentation/jobs",
                          json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE}).json()
    job = wait_terminal(client, created["job_id"])
    assert job["status"] == "succeeded", "로그 서버가 500 이어도 탐지는 성공"
    # 완료 로그는 상태가 succeeded 로 바뀐 직후 보내므로 경고 3개가 다 붙을 때까지 잠깐 기다린다
    import time as _time
    for _ in range(50):
        job = client.get(f"/api/v1/wbms/segmentation/jobs/{created['job_id']}").json()
        if sum("liaison log rejected" in w for w in job["warnings"]) >= 3:
            break
        _time.sleep(0.05)
    assert sum("liaison log rejected" in w for w in job["warnings"]) == 3


def test_unreachable_log_url_does_not_block(tmp_path: Path) -> None:
    client, runner, _ = make_client(tmp_path)
    client.app.state.wbms_segmentation.liaison = LiaisonLogger("http://127.0.0.1:9/nowhere", timeout_s=0.5)
    created = client.post("/api/v1/wbms/segmentation/jobs",
                          json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE}).json()
    job = wait_terminal(client, created["job_id"])
    assert job["status"] == "succeeded"
    assert any("not delivered" in w for w in job["warnings"])


def test_invalid_liaison_params_are_422() -> None:
    from backend.tests.test_wbms import make_client as _mc  # noqa: F401 - 같은 픽스처
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        client, _, _ = make_client(Path(tmp))
        bad_type = client.post("/api/v1/wbms/segmentation/jobs",
                               json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE, "product_type": "X"})
        assert bad_type.status_code == 422
        bad_id = client.post("/api/v1/wbms/segmentation/jobs",
                             json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE, "product_id": "../x"})
        assert bad_id.status_code == 422


# ── 장면 등록 화면 경로: Form 의 연계 파라미터가 탐지 잡 요청까지 흘러가는가 ─────────


def test_scene_registration_form_forwards_liaison_params_to_detection_job(tmp_path: Path, monkeypatch) -> None:
    from backend.app.segmentation.base import SceneRequest, SegmentationSubmission
    from backend.tests.test_gauge import _client, _geotiff_bytes

    client = _client(tmp_path)
    captured: list[SceneRequest] = []

    class _FakeSegmenter:
        def submit(self, request: SceneRequest) -> SegmentationSubmission:
            captured.append(request)
            return SegmentationSubmission(job_id="seg_test", status="queued", cached=False, result=None, warnings=[], raw={})

    monkeypatch.setattr(client.app.state.segmenters, "resolve", lambda sensor, testbed: _FakeSegmenter())
    response = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "2024-05-01", "run_detection": "true",
              "testbed": "busan", "order_id": "42", "scene_id": "7", "work_id": "W-9",
              "algm_id": "AG000003", "product_type": "PT000003", "product_id": "ICEYE_X5_SLC_SM_23467_20240501T000000"},
        files={"pre_file": ("pre.tif", _geotiff_bytes(tmp_path, "pre.tif"), "image/tiff")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["detection_job"]["job_id"] == "seg_test"
    assert captured and dict(captured[0].options) == {
        "order_id": "42", "scene_id": "7", "work_id": "W-9", "algm_id": "AG000003",
        "product_type": "PT000003", "product_id": "ICEYE_X5_SLC_SM_23467_20240501T000000",
    }

    bad = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "2024-05-02", "product_type": "nope"},
        files={"pre_file": ("pre.tif", _geotiff_bytes(tmp_path, "pre2.tif"), "image/tiff")},
    )
    assert bad.status_code == 422


def test_plugin_merges_options_into_job_body() -> None:
    from backend.app.segmentation.base import SceneRequest
    from backend.app.segmentation.wbms_plugin import WbmsDockerSegmenter

    bodies: list[dict] = []

    class _Jobs:
        def submit(self, body: dict) -> dict:
            bodies.append(body)
            return {"job_id": "j", "status": "queued", "cached": False, "result": None, "warnings": []}

    segmenter = WbmsDockerSegmenter.__new__(WbmsDockerSegmenter)
    segmenter._jobs = _Jobs()
    segmenter.submit(SceneRequest(sensor="iceye", testbed="busan", image_date="20200302",
                                  options={"order_id": "1", "product_id": "", "work_id": None}))
    assert bodies[0]["order_id"] == "1"
    assert "product_id" not in bodies[0] and "work_id" not in bodies[0], "빈 값은 싣지 않는다"


# ── 배치 러너: ② 성공 태스크 뒤 규약 파일 내보내기 + 완료 로그 ─────────────


def test_chain_runner_exports_product_and_logs_completion(tmp_path: Path, log_server) -> None:
    import argparse
    import importlib.util
    import sys

    url, collector = log_server
    spec = importlib.util.spec_from_file_location("wbms_chain_runner", Path("scripts/wbms_chain_runner.py"))
    runner = importlib.util.module_from_spec(spec)
    sys.modules["wbms_chain_runner"] = runner
    spec.loader.exec_module(runner)

    pre = tmp_path / "pre" / "Pre_Busan_ICEYE_20200302T183857.tif"
    pre.parent.mkdir()
    pre.write_bytes(b"x")
    (pre.parent / "Pre_Busan_ICEYE_20200302T183857_meta.json").write_text(
        json.dumps({"Provenance": {"input_file": "ICEYE_X5_SLC_SM_23467_20200302T183857.h5"}}), encoding="utf-8"
    )
    host_out = tmp_path / "wb"
    scene_dir = host_out / "2020" / "20200302"
    scene_dir.mkdir(parents=True)
    (scene_dir / "WB_Busan_ICEYE_20200302T183857.tif").write_bytes(b"GEOTIFF")
    man_path = tmp_path / "chain_manifest.json"
    man: dict = {}
    args = argparse.Namespace(
        product_id=None, product_type="PT000003", product_dir=None, out_root=str(tmp_path), scenes="20200302",
        liaison_log_url=url, order_id="5", scene_id="2", work_id=None, algm_id="AG000003",
    )
    rec = {"status": "success", "ts": "2026-09-29T04:00:00Z", "elapsed_s": 31.6}
    runner.export_liaison_product(args, {"input_host": str(pre)}, str(host_out), "wb_iceye_20200302", rec, man, str(man_path))

    product = tmp_path / "products" / "PT000003_ICEYE_X5_SLC_SM_23467_20200302T183857.tif"
    assert product.is_file() and product.read_bytes() == b"GEOTIFF"
    assert json.loads(man_path.read_text(encoding="utf-8"))["wb_iceye_20200302"]["product_tif"] == str(product)
    assert len(collector.received) == 1
    sent = collector.received[0]["contents"]
    assert sent["work_status"] == "WS000002" and sent["work_id"] == "wb_iceye_20200302"
    assert sent["order_id"] == "5" and sent["algm_id"] == "AG000003" and sent["algm_proc_time"] == "32"
