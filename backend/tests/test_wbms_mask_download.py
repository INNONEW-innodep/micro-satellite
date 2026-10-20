"""마스크 GeoTIFF 다운로드 API (3세부 연계용).

- GET /api/v1/wbms/segmentation/jobs/{job_id}/mask
- GET /api/v1/wbms/segmentation/jobs/{job_id}/files/{name}
- GET /api/v1/regions/{region_id}/satellite/scenes/{scene_id}/mask
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.tests.test_gauge import _client as make_gauge_client
from backend.tests.test_gauge import _geotiff_bytes
from backend.tests.test_wbms import SCENE_DATE, make_client, wait_terminal

SEG_URL = "/api/v1/wbms/segmentation/jobs"
SUBMIT = {"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE}


def _succeeded_job(client) -> dict:
    created = client.post(SEG_URL, json=SUBMIT)
    assert created.status_code == 202, created.text
    job = wait_terminal(client, created.json()["job_id"])
    assert job["status"] == "succeeded"
    return job


# ── /jobs/{job_id}/mask ──────────────────────────────────────────────────────


def _preferred_mask(result: dict) -> Path:
    """엔드포인트가 골라야 하는 파일: 규약명 사본(product_tif)이 있으면 그것, 없으면 mask_tif."""

    return Path(result.get("product_tif") or result["mask_tif"])


def test_mask_download_streams_geotiff_bytes(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    job = _succeeded_job(client)
    mask_path = Path(job["result"]["mask_tif"])
    expected = _preferred_mask(job["result"])
    assert expected.read_bytes() == mask_path.read_bytes()  # 사본은 원본과 같은 바이트

    response = client.get(f"{SEG_URL}/{job['job_id']}/mask")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/tiff"
    assert response.headers["content-disposition"].startswith("attachment")
    assert expected.name in response.headers["content-disposition"]
    assert response.content == mask_path.read_bytes()


def test_mask_download_prefers_product_tif_when_present(tmp_path: Path) -> None:
    client, _runner, settings = make_client(tmp_path)
    job = _succeeded_job(client)
    # 다른 담당자가 넣을 규약 파일명 사본(PT000003_{product_id}.tif)을 흉내 — 잡 파일을 직접 고친다
    product = Path(job["result"]["output_dir"]) / "PT000003_0001.tif"
    product.write_bytes(b"product-copy")
    job_file = settings.jobs_dir / f"{job['job_id']}.json"
    doc = json.loads(job_file.read_text(encoding="utf-8"))
    doc["result"]["product_tif"] = str(product)
    job_file.write_text(json.dumps(doc), encoding="utf-8")

    response = client.get(f"{SEG_URL}/{job['job_id']}/mask")
    assert response.status_code == 200
    assert response.content == b"product-copy"
    assert "PT000003_0001.tif" in response.headers["content-disposition"]
    # 규약명 사본도 files/{name} 으로 받을 수 있다
    by_name = client.get(f"{SEG_URL}/{job['job_id']}/files/PT000003_0001.tif")
    assert by_name.status_code == 200 and by_name.content == b"product-copy"


def test_mask_download_unknown_job_is_404(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    assert client.get(f"{SEG_URL}/does-not-exist/mask").status_code == 404
    assert client.get(f"{SEG_URL}/does-not-exist/files/x.tif").status_code == 404


def test_mask_download_before_success_is_409(tmp_path: Path) -> None:
    client, runner, _settings = make_client(tmp_path)
    # 스레드를 띄우지 않고 제출 → queued 로 머문다
    adapter = client.app.state.wbms_segmentation
    queued = adapter.submit(dict(SUBMIT), start_thread=False)
    assert queued["status"] == "queued"
    response = client.get(f"{SEG_URL}/{queued['job_id']}/mask")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]
    assert client.get(f"{SEG_URL}/{queued['job_id']}/files/x.tif").status_code == 409

    # 실패한 잡도 409
    runner.mode = "fail"
    created = client.post(SEG_URL, json=SUBMIT).json()
    failed = wait_terminal(client, created["job_id"])
    assert failed["status"] == "failed"
    assert client.get(f"{SEG_URL}/{failed['job_id']}/mask").status_code == 409


def test_mask_download_missing_file_on_disk_is_404(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    job = _succeeded_job(client)
    _preferred_mask(job["result"]).unlink()
    Path(job["result"]["mask_tif"]).unlink(missing_ok=True)
    response = client.get(f"{SEG_URL}/{job['job_id']}/mask")
    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


# ── /jobs/{job_id}/files/{name} ──────────────────────────────────────────────


def test_files_endpoint_serves_products_with_media_types(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    job = _succeeded_job(client)
    result = job["result"]
    base = result["base"]

    qc = client.get(f"{SEG_URL}/{job['job_id']}/files/{base}_qc.json")
    assert qc.status_code == 200, qc.text
    assert qc.headers["content-type"].startswith("application/json")
    assert qc.json()["status"] == "COMPLETE"

    meta = client.get(f"{SEG_URL}/{job['job_id']}/files/{base}_meta.json")
    assert meta.status_code == 200
    assert meta.headers["content-type"].startswith("application/json")

    tif = client.get(f"{SEG_URL}/{job['job_id']}/files/{base}.tif")
    assert tif.status_code == 200
    assert tif.headers["content-type"] == "image/tiff"
    assert tif.content == Path(result["mask_tif"]).read_bytes()


def test_files_endpoint_refuses_traversal_and_unlisted_names(tmp_path: Path) -> None:
    client, _runner, settings = make_client(tmp_path)
    job = _succeeded_job(client)
    out_dir = Path(job["result"]["output_dir"])
    # 산출물 디렉터리 옆에 진짜 파일을 두어도 목록에 없으면 못 연다
    (out_dir.parent / "secret.txt").write_text("nope", encoding="utf-8")
    (out_dir / "unlisted.txt").write_text("nope", encoding="utf-8")
    (settings.jobs_dir / "x").mkdir(exist_ok=True)

    # ".." 단독은 httpx 가 URL 을 정규화해 잡 상세로 가므로 여기서는 서버에 닿는 형태만 본다
    for name in ("../x", "..%2Fx", "../secret.txt", "%2e%2e/secret.txt", "unlisted.txt", "secret.txt"):
        response = client.get(f"{SEG_URL}/{job['job_id']}/files/{name}")
        assert response.status_code in {400, 404}, (name, response.status_code, response.text)
        assert response.headers.get("content-type", "").startswith("application/json"), name
        assert b"nope" not in response.content, name


# ── /regions/{region_id}/satellite/scenes/{scene_id}/mask ────────────────────


def test_scene_mask_download_serves_registered_source_mask(tmp_path: Path) -> None:
    client = make_gauge_client(tmp_path)
    # 마스크 없는 장면 → 404
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    created = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "2024-05-01"},
        files={"pre_file": ("pre.tif", pre, "image/tiff")},
    )
    assert created.status_code == 201, created.text
    scene_id = created.json()["scene"]["scene_id"]
    no_mask = client.get(f"/api/v1/regions/toy/satellite/scenes/{scene_id}/mask")
    assert no_mask.status_code == 404
    assert "no water mask" in no_mask.json()["detail"]

    # 마스크를 붙여 다시 등록 → 원본 GeoTIFF 바이트 그대로 내려온다
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    updated = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
    )
    assert updated.status_code == 201, updated.text
    scene = updated.json()["scene"]
    assert scene["detection_available"] is True
    source_mask = Path(scene["source_mask"])
    assert source_mask.is_file()

    response = client.get(f"/api/v1/regions/toy/satellite/scenes/{scene_id}/mask")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/tiff"
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.content == source_mask.read_bytes()
    assert response.content[:4] in (b"II*\x00", b"MM\x00*")  # 진짜 TIFF 헤더

    # 지역·장면이 없으면 404, 파일이 사라져도 404
    assert client.get("/api/v1/regions/nowhere/satellite/scenes/x/mask").status_code == 404
    assert client.get("/api/v1/regions/toy/satellite/scenes/20990101_ICEYE/mask").status_code == 404
    source_mask.unlink()
    assert client.get(f"/api/v1/regions/toy/satellite/scenes/{scene_id}/mask").status_code == 404


def test_mask_download_falls_back_to_mask_tif_when_product_copy_is_gone(tmp_path: Path) -> None:
    """규약명 사본(product_tif)이 기록돼 있어도 디스크에서 사라졌으면 원본 WB_…tif 를 보낸다."""

    client, _runner, settings = make_client(tmp_path)
    job = _succeeded_job(client)
    mask_path = Path(job["result"]["mask_tif"])
    job_file = settings.jobs_dir / f"{job['job_id']}.json"
    doc = json.loads(job_file.read_text(encoding="utf-8"))
    doc["result"]["product_tif"] = str(mask_path.parent / "PT000003_gone.tif")  # 존재하지 않는 사본
    job_file.write_text(json.dumps(doc), encoding="utf-8")

    response = client.get(f"{SEG_URL}/{job['job_id']}/mask")
    assert response.status_code == 200, response.text
    assert response.content == mask_path.read_bytes()
    assert mask_path.name in response.headers["content-disposition"]
