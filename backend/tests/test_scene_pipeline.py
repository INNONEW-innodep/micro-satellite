"""통합 장면 파이프라인 — ``POST /scenes`` 한 번으로 등록→탐지→면적→수위→예측→연계가 이어지는지.

test_wbms 의 FakeRunner(도커 경계 대역)·인수 패키지 픽스처와 test_gauge 의 토이 지역·합성 GeoTIFF 를 재사용한다.
토이 지역에 ``satellite.wbms_testbed = busan`` 을 붙여 카탈로그의 탐지 어댑터·③ 지점 목록이 걸리게 한다.
대청댐 토이 지역(``wbms_testbed = dcd``, 3-class 마스크 없음)은 Umbra — ICEYE 모델을 빌려 iceye 로 도는 센서 — 를 시험한다.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.pipeline import install_pipeline
from backend.app.pipeline.steps import MASK3_GENERATED_WARNING, STEP_LABELS, STEP_ORDER, generate_mask3_from_pre
from backend.app.wbms.liaison_log import WORK_STATUS, LiaisonLogger
from backend.app.wbms.runner import CommandResult
from backend.tests.test_gauge import _geotiff_bytes, write_region
from backend.tests.test_liaison import (
    log_server,  # noqa: F401 - 픽스처 재사용(가짜 로그 수신 서버)
)
from backend.tests.test_wbms import FakeRunner, _arg, _mounts, make_wbms_settings

REGION_ID = "busan_toy"      # 카탈로그 테스트베드 busan 을 가진 토이 지역
DCD_REGION_ID = "dcd_toy"    # 테스트베드 dcd(대청댐) — 인수된 3-class 마스크가 없다 → Umbra 는 마스크를 생성해야 한다
PLAIN_REGION_ID = "plain_toy"  # 테스트베드가 없는 지역 — 자동 탐지가 불가능해야 한다
SENSOR_NAMES = {"iceye": "ICEYE", "planet": "PlanetScope"}  # 컨테이너 KEY2NAME — umbra 는 iceye 로 들어오므로 여기 없다
TB_CODES = {"busan": "Busan", "dcd": "DCD"}                   # 컨테이너 tb_code
MASK_AREA_KM2 = 64 * 32 * 9 / 1e6  # _geotiff_bytes(mask=True): 64×64 화소(3 m) 중 왼쪽 절반이 수체


def _write_regions(builtin: Path) -> None:
    region = write_region(builtin, region_id=REGION_ID)
    manifest_path = region / "region.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["satellite"].update({"wbms_testbed": "busan", "sensors": ["ICEYE", "PlanetScope"]})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    dcd = write_region(builtin, region_id=DCD_REGION_ID)
    dcd_manifest_path = dcd / "region.json"
    dcd_manifest = json.loads(dcd_manifest_path.read_text(encoding="utf-8"))
    dcd_manifest["satellite"].update({"wbms_testbed": "dcd", "sensors": ["PlanetScope"]})  # 실제 geum_daecheong 처럼 광학만 적혀 있다
    dcd_manifest_path.write_text(json.dumps(dcd_manifest, ensure_ascii=False), encoding="utf-8")
    write_region(builtin, region_id=PLAIN_REGION_ID)


class GeoFakeRunner(FakeRunner):
    """test_wbms.FakeRunner 에 더해 (1) 진짜 GeoTIFF 마스크를 내고 (2) ③ calc_wlwa 산출을 흉내 낸다."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self.tmp_path = tmp_path
        self.wlwa_level = 0.42
        self.block: threading.Event | None = None  # 탐지를 붙잡아 두는 스위치(409 시험)
        self.wlwa_inputs: list[list[str]] = []

    def run(self, argv: list[str], timeout_s: float) -> CommandResult:
        if "/WBMS/wbms_modules/calc_wlwa.py" in argv:
            self.calls.append(list(argv))
            self._fake_wlwa(argv)
            return CommandResult(returncode=0, stdout='{"type":"done","ts":"t"}\n', stderr="", elapsed_s=0.1)
        if self.block is not None and "/WBMS/wbms_modules/detect_water.py" in argv:
            self.block.wait(10.0)
        return super().run(argv, timeout_s)

    def _fake_detect_water(self, argv: list[str]) -> None:
        out_host = _mounts(argv)[_arg(argv, "--output_dir")]
        image_date = _arg(argv, "--image_date")
        # 컨테이너처럼 산출물 표기는 --sensor(KEY2NAME)·--testbed(tb_code)에서만 유도한다 — umbra 로 올려도 ICEYE 로 적힌다
        sensor_name = SENSOR_NAMES[_arg(argv, "--sensor")]
        tb = TB_CODES[_arg(argv, "--testbed")]
        scene_dir = out_host / image_date[:4] / image_date
        scene_dir.mkdir(parents=True, exist_ok=True)
        base = f"WB_{tb}_{sensor_name}_{image_date}T000000"
        (scene_dir / f"{base}.tif").write_bytes(_geotiff_bytes(self.tmp_path, f"fake_{base}.tif", mask=True))
        (scene_dir / f"{base}_meta.json").write_text(json.dumps({"Entry": {"date": image_date}}), encoding="utf-8")
        qc = {
            "base": base, "status": "COMPLETE", "products": [f"{base}.tif", f"{base}_meta.json"],
            "params": {"sensor_name": sensor_name, "testbed": _arg(argv, "--testbed"), "image_date": image_date},
            "runtime": {"elapsed_s": 12.5},
        }
        (scene_dir / f"{base}_qc.json").write_text(json.dumps(qc), encoding="utf-8")

    def _fake_wlwa(self, argv: list[str]) -> None:
        mounts = _mounts(argv)
        testbed = _arg(argv, "--testbed")
        tb = TB_CODES[testbed]
        sensor_name = SENSOR_NAMES[_arg(argv, "--sensor")]
        # 실제 calc_wlwa 처럼 WB_<TB>_<KEY2NAME[--sensor]>_* 만 모은다 — 다른 표기로 꾸민 입력은 보이지 않는다
        wb_qc = [p for p in sorted(mounts["/wb"].glob("**/WB_*_qc.json")) if p.name.startswith(f"WB_{tb}_{sensor_name}_")]
        self.wlwa_inputs.append([p.name for p in wb_qc])
        if not wb_qc:
            return
        out = mounts["/out"]
        loc = _arg(argv, "--loc_id")
        start, end = _arg(argv, "--start_date"), _arg(argv, "--end_date")
        base = json.loads(wb_qc[0].read_text(encoding="utf-8"))["base"]
        stem = f"WLWA_{tb}_{sensor_name}_{loc}_{start}_{end}"
        doc = {
            "testbed": testbed, "satellite": sensor_name, "loc_id": loc, "start_date": start, "end_date": end,
            "records": [{
                "testbed": testbed, "satellite": sensor_name, "date": start, "water_level_m": self.wlwa_level,
                "water_area_km2": 0.05, "quality": "normal", "source_wb": f"{base}.tif",
                "corrected_water_level_m": None, "offset_m": None, "correction_mode": "none",
            }],
        }
        (out / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
        (out / f"{stem}_qc.json").write_text(
            json.dumps({"base": stem, "status": "COMPLETE", "products": [f"{stem}.json"]}), encoding="utf-8"
        )


def _app(tmp_path: Path, *, runner: GeoFakeRunner | None = None):
    builtin = tmp_path / "builtin"
    _write_regions(builtin)
    wbms_settings = dataclasses.replace(make_wbms_settings(tmp_path), product_dir=tmp_path / "products")
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user")
    app = create_app(settings, wbms_settings)
    runner = runner or GeoFakeRunner(tmp_path)
    app.state.wbms_segmentation.runner = runner
    app.state.wbms_fusion.runner = runner
    install_pipeline(app, poll_s=0.02)
    return TestClient(app), runner, settings, wbms_settings


def _seed_iceye_dem(wbms_settings, image_date: str) -> None:
    icegrid = wbms_settings.aux_dir / "DEM_busan_ngii3m" / "icegrid"
    icegrid.mkdir(parents=True, exist_ok=True)
    (icegrid / f"{image_date}_DEM_32652.tif").write_bytes(b"dem")


def _seed_dcd_dem(wbms_settings) -> Path:
    """대청댐 NGII 3 m 단일 DEM — 광학·SAR(Umbra) 공용(pipeline.steps.DEM_BY_TESTBED)."""
    dem = wbms_settings.aux_dir / "DEM_daecheong_ngii3m" / "daecheong_ngii_32652_3m_topofilled.tif"
    dem.parent.mkdir(parents=True, exist_ok=True)
    dem.write_bytes(b"dem")
    return dem


NODATA_COLS = 12  # _sar_pre_with_nodata: 오른쪽 12열이 촬영범위 밖(-9999) + NaN 1화소


def _sar_pre_with_nodata(tmp_path: Path, name: str, *, size: int = 64) -> bytes:
    """Umbra 변환 산출을 흉내 낸 1밴드 float32 γ⁰ dB GeoTIFF(EPSG:32652, 3 m, nodata -9999) — 오른쪽 띠는 nodata, NaN 1화소."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / name
    yy, xx = np.mgrid[0:size, 0:size]
    data = (-18.0 + 12.0 * (xx / size) + 1.5 * np.cos(yy / 7.0)).astype(np.float32)
    data[:, size - NODATA_COLS:] = -9999.0
    data[3, 3] = np.nan
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, crs="EPSG:32652",
                       transform=from_origin(364_000.0, 4_039_000.0, 3.0, 3.0), dtype="float32", nodata=-9999.0) as ds:
        ds.write(data, 1)
    return path.read_bytes()


def _wait(client: TestClient, scene_id: str, deadline_s: float = 60.0) -> dict:
    end = time.monotonic() + deadline_s
    payload: dict = {}
    while time.monotonic() < end:
        payload = client.get(f"/api/v1/scenes/{scene_id}").json()
        if payload.get("status") in ("done", "failed"):
            return payload
        time.sleep(0.03)
    pytest.fail(f"scene {scene_id} did not finish within {deadline_s}s: {payload}")


def _steps(payload: dict) -> dict[str, dict]:
    return {s["key"]: s for s in payload["steps"]}


# ── (a) 업로드 → 탐지(FakeRunner) → 면적 → 수위 → 예측(건너뜀) → 규약 파일 ─────────────────


def test_upload_runs_full_chain_with_detection(tmp_path: Path) -> None:
    client, runner, _settings, wbms_settings = _app(tmp_path)
    _seed_iceye_dem(wbms_settings, "20200416")
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": REGION_ID, "sensor": "iceye", "image_date": "2020-04-16", "acquisition_time": "18:36:30",
              "order_id": "42", "work_id": "W-1"},
        files={"pre_file": ("ICEYE_X5_SLC_SM_23467_20200416T183630.tif", pre, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["scene_id"] == "20200416_ICEYE"
    assert accepted["pipeline_id"].startswith("pl_") and accepted["status_url"] == "/api/v1/scenes/20200416_ICEYE"

    payload = _wait(client, "20200416_ICEYE")
    assert payload["status"] == "done", payload
    assert [s["key"] for s in payload["steps"]] == list(STEP_ORDER)
    assert [s["label"] for s in payload["steps"]] == [STEP_LABELS[k] for k in STEP_ORDER]
    assert [s["label"] for s in payload["steps"]] == ["장면 등록", "수체 탐지", "면적 계산", "수위 산출", "시계열 예측", "규약 파일·로그"]
    steps = _steps(payload)
    for key in ("register", "detect", "area", "levels", "deliver"):
        assert steps[key]["status"] == "done", (key, steps[key])
        assert steps[key]["elapsed_s"] is not None and steps[key]["started_at"] and steps[key]["finished_at"] and steps[key]["detail"]
    assert steps["predict"]["status"] == "skipped" and "3" in steps["predict"]["detail"]
    assert payload["progress"] == 1.0 and payload["error"] is None and payload["pipeline_id"] == accepted["pipeline_id"]
    assert payload["liaison"]["order_id"] == "42" and payload["liaison"]["work_id"] == "W-1" and payload["liaison"]["product_id"] is None
    assert payload["created_at"] and payload["finished_at"]
    assert "WB_Busan_ICEYE_20200416T000000.tif" in steps["detect"]["detail"] and "km²" in steps["detect"]["detail"]

    outputs = payload["outputs"]
    assert set(outputs) >= {"quicklook_url", "overlay_url", "mask_url", "product_url", "product_name", "water_area_km2",
                            "area_series_url", "levels_url", "prediction_id", "prediction_url", "folder"}
    # 규약 파일명: 업로드 원본 파일명(① 사이드카 Provenance.input_file)에서 파생한 product_id
    assert outputs["product_name"] == "PT000003_ICEYE_X5_SLC_SM_23467_20200416T183630.tif"
    assert outputs["product_url"] == "/api/v1/scenes/20200416_ICEYE/files/PT000003_ICEYE_X5_SLC_SM_23467_20200416T183630.tif"
    assert outputs["water_area_km2"] == pytest.approx(MASK_AREA_KM2, abs=1e-4)
    assert outputs["prediction_id"] is None and outputs["prediction_url"] is None
    mask_tif = wbms_settings.wb_output_dir / "2020" / "20200416" / "WB_Busan_ICEYE_20200416T000000.tif"
    product = client.get(outputs["product_url"])
    assert product.status_code == 200 and product.headers["content-type"].startswith("image/tiff")
    assert product.content == mask_tif.read_bytes(), "규약 파일 바이트 = 탐지 마스크 바이트"
    assert client.get(outputs["mask_url"]).content == mask_tif.read_bytes()
    assert client.get(outputs["quicklook_url"]).content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(outputs["overlay_url"]).status_code == 200

    # 장면 산출 폴더(backend/data 기준 상대경로 = regions_user_dir 의 형제 scenes/)
    assert outputs["folder"] == f"scenes/{REGION_ID}/20200416_ICEYE"
    folder = tmp_path / outputs["folder"]
    names = {p.name for p in folder.iterdir()}
    assert {"status.json", outputs["product_name"], "WB_Busan_ICEYE_20200416T000000.tif", "quicklook.png", "overlay.png",
            "area.json", "levels.json", "prediction.json", "level_forecast.json"} <= names
    assert json.loads((folder / "status.json").read_text(encoding="utf-8"))["status"] == "done"
    assert client.get("/api/v1/scenes/20200416_ICEYE/files/status.json").json()["scene_id"] == "20200416_ICEYE"
    assert client.get("/api/v1/scenes/20200416_ICEYE/files/nope.tif").status_code == 404
    assert client.get("/api/v1/scenes/20200416_ICEYE/files/.hidden").status_code == 404
    area_doc = json.loads((folder / "area.json").read_text(encoding="utf-8"))
    assert area_doc["water_area_km2"] == pytest.approx(MASK_AREA_KM2, abs=1e-4) and len(area_doc["series"]) == 1
    assert area_doc["area_common_frame_km2"] is not None

    # 탐지 커맨드: 작업 폴더의 규약 이름 Pre tif(인수 패키지 밖이라 /scene 마운트) + ① 사이드카 + 날짜 3-class 마스크
    detect = next(c for c in runner.calls if "/WBMS/wbms_modules/detect_water.py" in c)
    assert _arg(detect, "--input") == "/scene/Pre_Busan_ICEYE_20200416T183630.tif"
    assert _arg(detect, "--mask") == "/aux/Masks/Img_DEM_3class_utm/busan_20200416_mask3_utm.tif"
    work = tmp_path / "user" / REGION_ID / "uploads" / "20200416_ICEYE"
    assert (work / "Pre_Busan_ICEYE_20200416T183630.tif").read_bytes() == pre
    meta = json.loads((work / "Pre_Busan_ICEYE_20200416T183630_meta.json").read_text(encoding="utf-8"))
    assert meta["Entry"]["AcquisitionDateTime"] == "2020-04-16T18:36:30Z"
    assert meta["Provenance"] == {"stage": "upload", "input_file": "ICEYE_X5_SLC_SM_23467_20200416T183630.tif",
                                  "pipeline_id": accepted["pipeline_id"]}
    assert not any("/scene" in w for w in payload["warnings"]), "업로드 입력의 /scene 마운트는 경고가 아니다"

    # ③ calc_wlwa: 지점 4곳 × 날짜 하나, 장면 하나짜리 WB 입력, SAR 는 날짜별 DEM 디렉터리, --gpu 없음
    wlwa = [c for c in runner.calls if "/WBMS/wbms_modules/calc_wlwa.py" in c]
    assert sorted(_arg(c, "--loc_id") for c in wlwa) == ["gimhae", "gupo", "hupo", "jeongcheon"]
    for argv in wlwa:
        assert _arg(argv, "--start_date") == _arg(argv, "--end_date") == "20200416"
        assert _arg(argv, "--dem") == "/aux/DEM_busan_ngii3m/icegrid" and _arg(argv, "--sensor") == "iceye"
        assert "--gpu" not in argv and _arg(argv, "--wb_result_dir") == "/wb" and _arg(argv, "--output_dir") == "/out"
        mounts = _mounts(argv)
        assert mounts["/aux"] == wbms_settings.aux_dir
        assert mounts["/out"] == wbms_settings.fusion_work_dir / "wlwa" / "iceye" / _arg(argv, "--loc_id")
        assert mounts["/wb"] == wbms_settings.fusion_work_dir / "wlwa_input" / "20200416_ICEYE"
    assert all(names == ["WB_Busan_ICEYE_20200416T000000_qc.json"] for names in runner.wlwa_inputs)
    staged_qc = json.loads((wbms_settings.fusion_work_dir / "wlwa_input" / "20200416_ICEYE" / "WB_Busan_ICEYE_20200416T000000_qc.json").read_text(encoding="utf-8"))
    assert staged_qc["status"] == "COMPLETE" and staged_qc["base"] == "WB_Busan_ICEYE_20200416T000000"
    assert "4지점" in steps["levels"]["detail"] and "gupo 0.420 m" in steps["levels"]["detail"]
    levels = json.loads((tmp_path / "user" / REGION_ID / "satellite" / "levels.json").read_text(encoding="utf-8"))["records"]
    fresh = [r for r in levels if r["date"] == "2020-04-16" and r["satellite"] == "ICEYE"]
    assert sorted(r["loc_id"] for r in fresh) == ["gimhae", "gupo", "hupo", "jeongcheon"]
    assert all(r["water_level_m"] == pytest.approx(0.42) and r["corrected_water_level_m"] is None for r in fresh)
    assert any(r["date"] == "2023-06-01" and r["loc_id"] == "gupo" for r in levels), "기존 기록은 그대로"
    assert outputs["levels_url"] == f"/api/v1/regions/{REGION_ID}/satellite/levels?start=2020-04-16&end=2020-04-16"
    via_api = client.get(outputs["levels_url"]).json()
    assert via_api["n"] == 4 and all(r["date"] == "2020-04-16" for r in via_api["records"])
    scene_levels = json.loads((folder / "levels.json").read_text(encoding="utf-8"))
    assert scene_levels["n"] == 4 and scene_levels["date"] == "2020-04-16"
    # 2026-10-20: 연계 모듈용 수위 전망 파일 — 위성 구역과 연결된 지점(A→gupo)의 촬영일 원점 일 단위 전망(GET …/gauge/A/forecast 와 같은 본문)
    lf = json.loads((folder / "level_forecast.json").read_text(encoding="utf-8"))
    assert lf["schema"] == "watercast.level_forecast/1" and lf["scene_id"] == "20200416_ICEYE" and lf["as_of_requested"] == "2020-04-16"
    assert lf["n"] == 1 and lf["stations"][0]["station_id"] == "A" and lf["stations"][0]["loc_id"] == "gupo", lf["stations"]
    st0 = lf["stations"][0]
    # 이 픽스처의 지역은 수위 모델을 학습하지 않았다 → 지점은 나열하되 사유를 남긴다(납품은 막지 않는다)
    assert st0["status"] == "failed" and "학습 산출물이 없습니다" in st0["error"] and lf["n_ok"] == 0
    assert client.get("/api/v1/scenes/20200416_ICEYE/files/level_forecast.json").json()["n"] == 1

    # 공통 격자 면적 계열은 지역 사용자 사본에 생긴다(build_common_frame_area.build_region(region_dir=...))
    assert outputs["area_series_url"] == f"/api/v1/regions/{REGION_ID}/satellite/area-series"
    series = client.get(outputs["area_series_url"]).json()
    assert series["available"] is True and series["sensors"]["ICEYE"]["n_scenes"] == 1
    assert (tmp_path / "user" / REGION_ID / "satellite" / "area_series.json").is_file()
    assert "공통 격자" in steps["area"]["detail"]

    # 목록: 최신순, 요약 키
    listed = client.get("/api/v1/scenes").json()
    assert listed["total"] >= 1 and listed["items"][0]["scene_id"] == "20200416_ICEYE"
    first = listed["items"][0]
    assert first["status"] == "done" and first["product_name"] == outputs["product_name"] and first["region_id"] == REGION_ID
    assert set(first) >= {"scene_id", "region_id", "sensor", "image_date", "status", "progress", "created_at", "finished_at",
                          "water_area_km2", "product_name"}
    assert client.get("/api/v1/scenes", params={"region_id": PLAIN_REGION_ID}).json()["items"] == []


# ── (b) 마스크 업로드 → 탐지 생략, 규약 파일 내보내기, 완료 로그 1건 ─────────────────────────


def test_uploaded_mask_skips_detection_exports_product_and_logs(tmp_path: Path, log_server) -> None:  # noqa: F811 - 픽스처
    url, collector = log_server
    client, runner, _settings, wbms_settings = _app(tmp_path)
    client.app.state.wbms_segmentation.liaison = LiaisonLogger(url, system_code="SC000009", algm_id="AG000009")
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": REGION_ID, "sensor": "PlanetScope", "image_date": "20200414", "order_id": "42", "scene_id": "7",
              "work_id": "W-1", "algm_id": "AG000003", "product_type": "PT000003", "product_id": "PSScene_20200414_x", "note": "마스크 업로드"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    assert response.json()["scene_id"] == "20200414_PlanetScope"
    payload = _wait(client, "20200414_PlanetScope")
    assert payload["status"] == "done", payload
    steps = _steps(payload)
    assert steps["detect"]["status"] == "skipped" and "마스크" in steps["detect"]["detail"]
    assert not any("/WBMS/wbms_modules/detect_water.py" in c for c in runner.calls)
    assert steps["register"]["status"] == "done" and steps["area"]["status"] == "done" and steps["deliver"]["status"] == "done"
    # ③: 광학 DEM 파일이 픽스처에 없다 → 사유를 적고 건너뛴다(파이프라인은 done)
    assert steps["levels"]["status"] == "skipped" and "DEM" in steps["levels"]["detail"]
    assert payload["outputs"]["levels_url"] is None

    product = wbms_settings.product_dir / "PT000003_PSScene_20200414_x.tif"
    assert product.is_file() and product.read_bytes() == mask
    outputs = payload["outputs"]
    assert outputs["product_name"] == product.name and outputs["water_area_km2"] == pytest.approx(MASK_AREA_KM2, abs=1e-4)
    assert client.get(outputs["product_url"]).content == mask
    assert client.get(outputs["mask_url"]).content == mask
    folder = tmp_path / outputs["folder"]
    assert (folder / product.name).read_bytes() == mask and (folder / "WB_Busan_PlanetScope_20200414T000000.tif").read_bytes() == mask
    assert payload["liaison"] == {"order_id": "42", "scene_id": "7", "work_id": "W-1", "algm_id": "AG000003",
                                  "product_type": "PT000003", "product_id": "PSScene_20200414_x"}

    assert len(collector.received) == 1, collector.received
    sent = collector.received[0]
    assert sent["system_code"] == "SC000009" and sent["contents"]["work_status"] == WORK_STATUS["succeeded"]
    assert sent["contents"]["work_id"] == "W-1" and sent["contents"]["order_id"] == "42" and sent["contents"]["scene_id"] == "7"
    assert sent["contents"]["algm_id"] == "AG000003" and sent["contents"]["product_type"] == "PT000003"
    assert product.name in sent["contents"]["msg"] and "algm_comp_time" in sent["contents"]
    assert not any("liaison log" in w for w in payload["warnings"])


def test_register_only_when_detect_false_without_mask(tmp_path: Path) -> None:
    client, runner, *_ = _app(tmp_path)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": REGION_ID, "sensor": "ICEYE", "image_date": "20200416", "detect": "false"},
        files={"pre_file": ("pre.tif", pre, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    payload = _wait(client, "20200416_ICEYE")
    assert payload["status"] == "done", payload
    steps = _steps(payload)
    assert steps["register"]["status"] == "done" and steps["deliver"]["status"] == "done"
    assert steps["detect"]["status"] == "skipped" and "detect=false" in steps["detect"]["detail"]
    assert {steps[k]["status"] for k in ("area", "levels", "predict")} == {"skipped"}
    assert payload["outputs"]["product_name"] is None and payload["outputs"]["mask_url"] is None
    assert payload["outputs"]["quicklook_url"] and payload["outputs"]["water_area_km2"] is None
    assert runner.calls == []


# ── (c) 같은 장면 동시 제출 409, 잘못된 입력 422 ─────────────────────────────────────────


def test_duplicate_running_is_409_and_bad_input_is_422(tmp_path: Path) -> None:
    client, runner, *_ = _app(tmp_path)
    runner.block = threading.Event()
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    form = {"region_id": REGION_ID, "sensor": "ICEYE", "image_date": "20200416"}
    files = {"pre_file": ("pre.tif", pre, "image/tiff")}
    first = client.post("/api/v1/scenes", data=form, files=files)
    assert first.status_code == 202, first.text
    duplicate = client.post("/api/v1/scenes", data=form, files=files)
    assert duplicate.status_code == 409 and "진행 중" in duplicate.json()["detail"]
    mid = client.get("/api/v1/scenes/20200416_ICEYE").json()
    assert mid["status"] in ("queued", "running") and mid["progress"] < 1.0
    runner.block.set()
    assert _wait(client, "20200416_ICEYE")["status"] == "done"
    # 끝난 뒤의 재제출은 받는다(다시 돈다)
    again = client.post("/api/v1/scenes", data=form, files=files)
    assert again.status_code == 202, again.text
    assert again.json()["pipeline_id"] != first.json()["pipeline_id"]
    assert _wait(client, "20200416_ICEYE")["status"] == "done"

    def post(**overrides):
        data = {**form, **overrides}
        return client.post("/api/v1/scenes", data=data, files={"pre_file": ("pre.tif", pre, "image/tiff")})

    assert post(image_date="2020-13-01").status_code == 422
    assert "image_date" in post(image_date="20200230").json()["detail"]
    assert post(product_type="X").status_code == 422
    assert post(product_id="../x").status_code == 422
    assert post(sensor="Sentinel-9").status_code == 422
    assert post(acquisition_time="25:00:00").status_code == 422
    assert post(note="x" * 501).status_code == 422
    assert post(region_id="nowhere").status_code == 404
    # 테스트베드가 없는 지역은 마스크 없이 자동 탐지를 받을 수 없다
    no_adapter = post(region_id=PLAIN_REGION_ID, image_date="20200417")
    assert no_adapter.status_code == 422 and "탐지 어댑터" in no_adapter.json()["detail"]
    assert client.post("/api/v1/scenes", data=form).status_code == 422  # pre_file/pre_path 없음
    empty = client.post("/api/v1/scenes", data={**form, "image_date": "20200418"}, files={"pre_file": ("pre.tif", b"", "image/tiff")})
    assert empty.status_code == 422
    assert not (tmp_path / "user" / REGION_ID / "uploads" / "20200418_ICEYE" / "Pre_Busan_ICEYE_20200418T000000.tif").exists()
    outside = client.post("/api/v1/scenes", data={**form, "image_date": "20200418", "pre_path": "/etc/hostname"})
    assert outside.status_code == 422 and "허용 루트" in outside.json()["detail"]


def test_non_geotiff_upload_fails_register_and_marks_pipeline_failed(tmp_path: Path) -> None:
    client, runner, *_ = _app(tmp_path)
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": REGION_ID, "sensor": "ICEYE", "image_date": "20200416"},
        files={"pre_file": ("pre.tif", b"not a tiff at all", "image/tiff")},
    )
    assert response.status_code == 202, response.text
    payload = _wait(client, "20200416_ICEYE")
    assert payload["status"] == "failed" and payload["error"] and payload["error"].startswith("장면 등록 실패")
    steps = _steps(payload)
    assert steps["register"]["status"] == "failed" and steps["register"]["detail"]
    assert {steps[k]["status"] for k in ("detect", "area", "levels", "predict", "deliver")} == {"skipped"}
    assert payload["progress"] == 1.0 and runner.calls == []
    assert client.get("/api/v1/scenes").json()["items"][0]["status"] == "failed"


# ── (d) 접수 화면 선택지 ───────────────────────────────────────────────────────────────


def test_options_lists_region_and_sensors(tmp_path: Path) -> None:
    client, *_ = _app(tmp_path)
    payload = client.get("/api/v1/scenes/options").json()
    region = next(r for r in payload["regions"] if r["id"] == REGION_ID)
    assert region["testbed"] == "busan" and region["sensors"] == ["ICEYE", "PlanetScope"] and region["has_gauge"] is True
    assert region["name"] and region["detect_available"] is True
    plain = next(r for r in payload["regions"] if r["id"] == PLAIN_REGION_ID)
    assert plain["testbed"] is None and plain["detect_available"] is False
    assert [s["id"] for s in payload["sensors"]] == ["ICEYE", "UMBRA", "PlanetScope"]
    iceye, umbra, planet = payload["sensors"]
    assert {k: iceye[k] for k in ("id", "label", "type", "needs_mask")} == {"id": "ICEYE", "label": "ICEYE 레이더 영상", "type": "SAR", "needs_mask": True}
    assert {k: umbra[k] for k in ("id", "label", "type", "needs_mask")} == {"id": "UMBRA", "label": "Umbra 레이더 영상", "type": "SAR", "needs_mask": True}
    assert umbra["key"] == "umbra" and umbra["name"] == "Umbra" and umbra["module_sensor"] == "iceye" and umbra["testbeds"] == ["dcd"]
    assert iceye["module_sensor"] == "iceye" and planet["module_sensor"] == "planet"
    assert {k: planet[k] for k in ("id", "label", "type", "needs_mask")} == {"id": "PlanetScope", "label": "PlanetScope 광학 영상", "type": "OPTIC", "needs_mask": False}
    # 지역 센서 목록 = 지역 정의 + 카탈로그가 그 테스트베드에 받는 센서, 표기는 sensors[].id 와 같은 파일 라벨
    dcd = next(r for r in payload["regions"] if r["id"] == DCD_REGION_ID)
    assert dcd["testbed"] == "dcd" and dcd["sensors"] == ["PlanetScope", "UMBRA"] and dcd["detect_available"] is True


def test_builtin_geum_daecheong_offers_umbra(tmp_path: Path) -> None:
    """저장소의 실제 지역 정의(regions/geum_daecheong: 광학만 기재)도 카탈로그 umbra pre_dirs(dcd) 덕에 UMBRA 를 받는다."""
    wbms_settings = make_wbms_settings(tmp_path)
    app = create_app(Settings(result_dir=tmp_path / "results", regions_user_dir=tmp_path / "user"), wbms_settings)
    install_pipeline(app, poll_s=0.02)
    client = TestClient(app)
    payload = client.get("/api/v1/scenes/options").json()
    daecheong = next(r for r in payload["regions"] if r["id"] == "geum_daecheong")
    assert daecheong["testbed"] == "dcd" and daecheong["sensors"] == ["PlanetScope", "UMBRA"]
    busan = next(r for r in payload["regions"] if r["id"] == "nakdong_busan")
    assert "UMBRA" not in busan["sensors"] and {"ICEYE", "PlanetScope"} <= set(busan["sensors"])


# ── (d2) Umbra: 대청댐에 3-class 마스크가 없다 → 영상 유효 범위로 생성, ICEYE 모델로 iceye 로 돈다 ─────────────


def test_umbra_upload_generates_mask3_and_runs_as_iceye(tmp_path: Path) -> None:
    client, runner, _settings, wbms_settings = _app(tmp_path)
    _seed_dcd_dem(wbms_settings)
    assert not list((wbms_settings.aux_dir / "Masks" / "Img_DEM_3class_utm").glob("dcd_*")), "픽스처에 dcd 마스크가 없어야 한다"
    pre = _sar_pre_with_nodata(tmp_path, "umbra_pre.tif")
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": DCD_REGION_ID, "sensor": "UMBRA", "image_date": "2026-05-01", "acquisition_time": "093000", "order_id": "77"},
        files={"pre_file": ("2026-05-01-09-30-00_UMBRA-10_SICD.nitf.tif", pre, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    assert response.json()["scene_id"] == "20260501_Umbra"
    payload = _wait(client, "20260501_Umbra")
    assert payload["status"] == "done", payload
    steps = _steps(payload)
    assert steps["register"]["status"] == "done" and steps["detect"]["status"] == "done" and steps["levels"]["status"] == "done"
    assert steps["area"]["status"] == "done" and steps["deliver"]["status"] == "done"
    assert payload["sensor"] == "Umbra" and payload["image_date"] == "2026-05-01"
    assert MASK3_GENERATED_WARNING in payload["warnings"], payload["warnings"]
    assert not any("/mask3" in w or "/scene" in w for w in payload["warnings"]), "마운트 안내는 파이프라인 경고가 아니다"

    # ① 스테이징: 파일 라벨 UMBRA, 사이드카 SatelliteName 은 표시명 Umbra, 테스트베드 코드 DCD
    work = tmp_path / "user" / DCD_REGION_ID / "uploads" / "20260501_Umbra"
    staged = work / "Pre_DCD_UMBRA_20260501T093000.tif"
    assert staged.read_bytes() == pre
    meta = json.loads((work / "Pre_DCD_UMBRA_20260501T093000_meta.json").read_text(encoding="utf-8"))
    assert meta["SatelliteName"] == "Umbra" and meta["TestbedName"] == "DCD" and meta["ImageDate"] == "20260501"
    assert meta["Entry"]["AcquisitionDateTime"] == "2026-05-01T09:30:00Z"
    assert meta["Provenance"]["input_file"] == "2026-05-01-09-30-00_UMBRA-10_SICD.nitf.tif"

    # ② 탐지 argv: 컨테이너에는 iceye 로, 가중치는 ICEYE SAR, 마스크는 작업 폴더에 생성한 파일(/mask3 마운트)
    detect = next(c for c in runner.calls if "/WBMS/wbms_modules/detect_water.py" in c)
    assert _arg(detect, "--sensor") == "iceye" and _arg(detect, "--testbed") == "dcd" and _arg(detect, "--image_date") == "20260501"
    assert _arg(detect, "--weights") == "/model/WBMS_SAR_ICEYE.h5"
    assert _arg(detect, "--input") == "/scene/Pre_DCD_UMBRA_20260501T093000.tif"
    assert _arg(detect, "--mask") == "/mask3/dcd_20260501_mask3_utm.tif" and detect[-1] == "--force"
    generated = work / "dcd_20260501_mask3_utm.tif"
    assert _mounts(detect)["/mask3"] == work and generated.is_file()
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(generated) as m, rasterio.open(staged) as p:
        assert m.count == 1 and m.dtypes[0] == "uint8" and m.shape == p.shape and m.transform == p.transform and m.crs == p.crs
        assert m.profile.get("compress", "").lower() == "deflate" and m.nodata is None
        arr, band = m.read(1), p.read(1)
    expected = np.where(np.isfinite(band) & (band != -9999.0), 0, 1).astype(np.uint8)
    assert np.array_equal(arr, expected) and set(np.unique(arr).tolist()) == {0, 1}, "0=유효, 1=nodata/NaN, 바다(2) 없음"
    assert int((arr == 1).sum()) == 64 * NODATA_COLS + 1

    # 컨테이너(iceye)는 WB_DCD_ICEYE_* 를 쓴다 — umbra 조회가 그 산출물을 받아들이고, 잡 요청 에코는 umbra 그대로
    assert "WB_DCD_ICEYE_20260501T000000.tif" in steps["detect"]["detail"]
    status = json.loads((tmp_path / payload["outputs"]["folder"] / "status.json").read_text(encoding="utf-8"))
    assert status["sensor_key"] == "umbra" and status["sensor_type"] == "SAR" and status["internal"]["mask3_origin"] == "generated"
    assert status["internal"]["mask3_tif"] == str(generated)
    job = client.get(f"/api/v1/wbms/segmentation/jobs/{status['internal']['detection_job_id']}").json()
    assert job["request"]["sensor"] == "umbra" and job["request"]["testbed"] == "dcd" and job["result"]["base"] == "WB_DCD_ICEYE_20260501T000000"
    assert client.app.state.wbms_segmentation.find_cached("umbra", "dcd", "20260501")["base"] == "WB_DCD_ICEYE_20260501T000000"

    # ③ calc_wlwa: --sensor iceye, 단일 NGII DEM 파일, 입력 WB 는 모듈 표기(ICEYE)로 꾸며야 모듈이 집는다
    wlwa = [c for c in runner.calls if "/WBMS/wbms_modules/calc_wlwa.py" in c]
    assert sorted(_arg(c, "--loc_id") for c in wlwa) == ["bangsuro", "daecheong", "jojeongji"]
    for argv in wlwa:
        assert _arg(argv, "--sensor") == "iceye" and _arg(argv, "--testbed") == "dcd"
        assert _arg(argv, "--dem") == "/aux/DEM_daecheong_ngii3m/daecheong_ngii_32652_3m_topofilled.tif"
        assert _mounts(argv)["/out"] == wbms_settings.fusion_work_dir / "wlwa" / "umbra" / _arg(argv, "--loc_id")
        assert _mounts(argv)["/wb"] == wbms_settings.fusion_work_dir / "wlwa_input" / "20260501_Umbra"
    assert all(names == ["WB_DCD_ICEYE_20260501T000000_qc.json"] for names in runner.wlwa_inputs), runner.wlwa_inputs
    staged_qc = json.loads((wbms_settings.fusion_work_dir / "wlwa_input" / "20260501_Umbra" / "WB_DCD_ICEYE_20260501T000000_qc.json").read_text(encoding="utf-8"))
    assert staged_qc["params"]["sensor_name"] == "ICEYE" and staged_qc["params"]["sensor_catalog_key"] == "umbra"
    assert "3지점" in steps["levels"]["detail"] and "daecheong 0.420 m" in steps["levels"]["detail"]
    levels = json.loads((tmp_path / "user" / DCD_REGION_ID / "satellite" / "levels.json").read_text(encoding="utf-8"))["records"]
    fresh = [r for r in levels if r["date"] == "2026-05-01"]
    assert sorted(r["loc_id"] for r in fresh) == ["bangsuro", "daecheong", "jojeongji"]
    assert all(r["satellite"] == "Umbra" and r["sensor_catalog_key"] == "umbra" and r["module_satellite"] == "ICEYE" for r in fresh)
    assert all(r["water_level_m"] == pytest.approx(0.42) for r in fresh)
    scene_levels = json.loads((tmp_path / payload["outputs"]["folder"] / "levels.json").read_text(encoding="utf-8"))
    assert scene_levels["n"] == 3 and all(r["sensor_catalog_key"] == "umbra" for r in scene_levels["records"])

    # 지역 장면 기록·면적 계열·규약 파일은 표시명 Umbra / 업로드 원본 이름에서 파생한 product_id
    outputs = payload["outputs"]
    assert outputs["product_name"] == "PT000003_2026-05-01-09-30-00_UMBRA-10_SICD.tif"
    assert outputs["water_area_km2"] == pytest.approx(MASK_AREA_KM2, abs=1e-4)
    assert client.get(outputs["area_series_url"]).json()["sensors"]["Umbra"]["n_scenes"] == 1
    assert client.get(outputs["quicklook_url"]).content[:8] == b"\x89PNG\r\n\x1a\n", "1밴드 dB 퀵룩"
    folder = tmp_path / outputs["folder"]
    assert {"WB_DCD_ICEYE_20260501T000000.tif", outputs["product_name"], "quicklook.png", "overlay.png"} <= {p.name for p in folder.iterdir()}
    # 등록으로 지역 정의에 표시명 Umbra 가 적혀도 선택지는 UMBRA 하나로 정규화된다
    options = client.get("/api/v1/scenes/options").json()
    dcd = next(r for r in options["regions"] if r["id"] == DCD_REGION_ID)
    assert dcd["sensors"] == ["PlanetScope", "UMBRA"] and dcd["n_scenes"] == 1
    # 같은 장면을 소문자 umbra 로 다시 올려도 같은 scene_id(409 아님 — 끝났으므로 다시 돈다)
    again = client.post("/api/v1/scenes", data={"region_id": DCD_REGION_ID, "sensor": "umbra", "image_date": "20260501", "detect": "false"},
                        files={"pre_file": ("pre.tif", pre, "image/tiff")})
    assert again.status_code == 202 and again.json()["scene_id"] == "20260501_Umbra"
    assert _wait(client, "20260501_Umbra")["status"] == "done"


def test_umbra_uploaded_mask3_is_used_instead_of_generated(tmp_path: Path) -> None:
    client, runner, *_ = _app(tmp_path)
    pre = _sar_pre_with_nodata(tmp_path, "umbra_pre2.tif")
    mask3 = _geotiff_bytes(tmp_path, "my_mask3.tif", mask=True)  # 내용은 가짜 러너가 보지 않는다 — 전달 경로만 검증
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": DCD_REGION_ID, "sensor": "Umbra", "image_date": "20260502"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask3_file": ("whatever_mask3.tif", mask3, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    payload = _wait(client, "20260502_Umbra")
    assert payload["status"] == "done", payload
    steps = _steps(payload)
    assert steps["detect"]["status"] == "done" and MASK3_GENERATED_WARNING not in payload["warnings"]
    assert steps["levels"]["status"] == "skipped" and "DEM" in steps["levels"]["detail"]  # dcd DEM 을 심지 않았다
    work = tmp_path / "user" / DCD_REGION_ID / "uploads" / "20260502_Umbra"
    assert (work / "dcd_20260502_mask3_utm.tif").read_bytes() == mask3
    detect = next(c for c in runner.calls if "/WBMS/wbms_modules/detect_water.py" in c)
    assert _arg(detect, "--mask") == "/mask3/dcd_20260502_mask3_utm.tif" and _arg(detect, "--sensor") == "iceye"
    status = json.loads((tmp_path / payload["outputs"]["folder"] / "status.json").read_text(encoding="utf-8"))
    assert status["internal"]["mask3_origin"] == "upload" and status["internal"]["mask3_tif"] == str(work / "dcd_20260502_mask3_utm.tif")
    # 빈 mask3 업로드는 422
    bad = client.post("/api/v1/scenes", data={"region_id": DCD_REGION_ID, "sensor": "Umbra", "image_date": "20260503"},
                      files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask3_file": ("m.tif", b"", "image/tiff")})
    assert bad.status_code == 422 and "mask3_file" in bad.json()["detail"]


def test_generate_mask3_from_pre_unit(tmp_path: Path) -> None:
    """nodata 태그가 없으면 -9999 를 nodata 로 본다(ICEYE 규약); NaN 도 제외."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    src = tmp_path / "pre_untagged.tif"
    data = np.full((8, 10), -5.0, dtype=np.float32)
    data[0, :] = -9999.0
    data[7, 9] = np.inf
    with rasterio.open(src, "w", driver="GTiff", height=8, width=10, count=1, crs="EPSG:32652",
                       transform=from_origin(0.0, 24.0, 3.0, 3.0), dtype="float32") as ds:
        ds.write(data, 1)
    out = generate_mask3_from_pre(src, tmp_path / "nested" / "mask3.tif")
    with rasterio.open(out) as m:
        arr = m.read(1)
        assert m.dtypes[0] == "uint8" and m.transform == from_origin(0.0, 24.0, 3.0, 3.0) and str(m.crs) == "EPSG:32652"
    assert arr[0].tolist() == [1] * 10 and arr[7, 9] == 1 and int(arr.sum()) == 11


# ── (e) 재시작 복구: 돌던 기록은 실패로 마감 ─────────────────────────────────────────────


def test_restart_recovery_marks_running_as_failed(tmp_path: Path) -> None:
    folder = tmp_path / "scenes" / REGION_ID / "20200301_ICEYE"
    folder.mkdir(parents=True)
    statuses = {"register": "done", "detect": "running", "area": "pending", "levels": "pending", "predict": "pending", "deliver": "pending"}
    record = {
        "scene_id": "20200301_ICEYE", "region_id": REGION_ID, "sensor": "ICEYE", "sensor_key": "iceye", "sensor_type": "SAR",
        "image_date": "2020-03-01", "pipeline_id": "pl_20260101T000000Z_deadbeef", "status": "running", "progress": 0.2,
        "steps": [{"key": k, "label": STEP_LABELS[k], "status": statuses[k], "started_at": None, "finished_at": None,
                   "elapsed_s": None, "detail": None} for k in STEP_ORDER],
        "outputs": {}, "warnings": [], "error": None, "created_at": "2026-01-01T00:00:00+00:00", "finished_at": None,
        "liaison": {}, "internal": {"folder": str(folder)},
    }
    (folder / "status.json").write_text(json.dumps(record), encoding="utf-8")

    client, *_ = _app(tmp_path)  # install_pipeline → recover()
    payload = client.get("/api/v1/scenes/20200301_ICEYE").json()
    assert payload["status"] == "failed" and payload["error"] == "서버 재시작으로 중단" and payload["progress"] == 1.0
    steps = _steps(payload)
    assert steps["register"]["status"] == "done"
    assert steps["detect"]["status"] == "failed" and steps["detect"]["detail"] == "서버 재시작으로 중단"
    assert {steps[k]["status"] for k in ("area", "levels", "predict", "deliver")} == {"skipped"}
    assert payload["finished_at"] and payload["pipeline_id"] == "pl_20260101T000000Z_deadbeef"
    assert json.loads((folder / "status.json").read_text(encoding="utf-8"))["status"] == "failed", "디스크에도 반영"
    listed = client.get("/api/v1/scenes").json()
    assert any(i["scene_id"] == "20200301_ICEYE" and i["status"] == "failed" for i in listed["items"])
    # 실패로 마감됐으니 같은 장면을 다시 제출할 수 있다(409 아님)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    retry = client.post("/api/v1/scenes", data={"region_id": REGION_ID, "sensor": "ICEYE", "image_date": "20200301", "detect": "false"},
                        files={"pre_file": ("pre.tif", pre, "image/tiff")})
    assert retry.status_code == 202, retry.text
    assert _wait(client, "20200301_ICEYE")["status"] == "done"


# ── (f) ICEYE 날짜별 3-class 마스크가 없으면 가장 가까운 날짜 것으로 대체 ────────────────────


def test_iceye_missing_mask3_falls_back_to_nearest(tmp_path: Path) -> None:
    client, runner, _settings, wbms_settings = _app(tmp_path)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": REGION_ID, "sensor": "ICEYE", "image_date": "20200418"},
        files={"pre_file": ("pre.tif", pre, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    payload = _wait(client, "20200418_ICEYE")
    assert payload["status"] == "done", payload
    assert any(w.startswith("20200418 의 3-class 마스크가 없어 20200416 것을 대신 사용") for w in payload["warnings"]), payload["warnings"]
    detect = next(c for c in runner.calls if "/WBMS/wbms_modules/detect_water.py" in c)
    assert _arg(detect, "--mask") == "/aux/Masks/Img_DEM_3class_utm/busan_20200416_mask3_utm.tif"
    assert _steps(payload)["detect"]["status"] == "done"

    # 어댑터 단위: 가장 가까운 날짜(동률이면 앞선 날짜), 없는 테스트베드는 None, aux 밖 마스크는 /mask3 로 마운트
    adapter = client.app.state.wbms_segmentation
    mask_dir = wbms_settings.aux_dir / "Masks" / "Img_DEM_3class_utm"
    (mask_dir / "busan_20200419_mask3_utm.tif").write_bytes(b"m")
    assert adapter.nearest_mask3("busan", "20200418").name == "busan_20200419_mask3_utm.tif"
    (mask_dir / "busan_20200417_mask3_utm.tif").write_bytes(b"m")
    assert adapter.nearest_mask3("busan", "20200418").name == "busan_20200417_mask3_utm.tif"
    assert adapter.nearest_mask3("busan", "20200416").name == "busan_20200416_mask3_utm.tif"
    assert adapter.nearest_mask3("dcd", "20200416") is None
    outside = tmp_path / "elsewhere" / "custom_mask3.tif"
    outside.parent.mkdir()
    outside.write_bytes(b"m")
    pre_host = wbms_settings.l1_pre_dir / "iceye_pre" / "2020" / "20200416" / "Pre_Busan_ICEYE_20200416T183630.tif"
    scene = adapter.resolve_scene("iceye", "busan", "20200416", str(pre_host), mask_tif=str(outside))
    assert scene["mask_container"] == "/mask3/custom_mask3.tif" and (outside.parent, "/mask3", True) in scene["mounts"]
    inside = adapter.resolve_scene("iceye", "busan", "20200416", str(pre_host), mask_tif=str(mask_dir / "busan_20200417_mask3_utm.tif"))
    assert inside["mask_container"] == "/aux/Masks/Img_DEM_3class_utm/busan_20200417_mask3_utm.tif"
    assert not any(c == "/mask3" for _h, c, _ro in inside["mounts"])


# ── (g) 공통 격자 프레임이 3개가 되면 면적 시계열 예측이 돈다 ─────────────────────────────


def test_predict_runs_once_three_frames_exist(tmp_path: Path) -> None:
    client, *_ = _app(tmp_path)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    payloads = []
    for day in ("20200301", "20200302", "20200303"):
        response = client.post(
            "/api/v1/scenes",
            data={"region_id": REGION_ID, "sensor": "planet", "image_date": day},
            files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
        )
        assert response.status_code == 202, response.text
        assert response.json()["scene_id"] == f"{day}_PlanetScope"
        payload = _wait(client, f"{day}_PlanetScope")
        assert payload["status"] == "done", payload
        payloads.append(payload)
    assert _steps(payloads[0])["predict"]["status"] == "skipped" and "1개" in _steps(payloads[0])["predict"]["detail"]
    assert _steps(payloads[1])["predict"]["status"] == "skipped" and "2개" in _steps(payloads[1])["predict"]["detail"]
    last = payloads[2]
    steps = _steps(last)
    assert steps["predict"]["status"] == "done" and "irregular-area-trend" in steps["predict"]["detail"]
    prediction_id = last["outputs"]["prediction_id"]
    assert prediction_id and last["outputs"]["prediction_url"] == f"/api/v1/predictions/{prediction_id}"
    prediction = client.get(last["outputs"]["prediction_url"]).json()
    assert prediction["status"] == "completed" and prediction["model_id"] == "irregular-area-trend"
    assert prediction["input"]["frame_count"] == 3 and len(prediction["steps"]) == 3
    assert prediction["input"]["source_dates"] == ["2020-03-01", "2020-03-02", "2020-03-03"]
    assert prediction["input"]["source_metadata"]["scene_id"] == "20200303_PlanetScope"
    # 화소 넓이는 원본(3 m → 9 m²)이 아니라 공통 격자 프레임의 것 — 64×64 장면을 긴 변 640 으로 올렸으니 0.3 m → 0.09 m²
    scenes = json.loads((tmp_path / "user" / REGION_ID / "satellite" / "scenes.json").read_text(encoding="utf-8"))["scenes"]
    frame = next(s for s in scenes if s["scene_id"] == "20200303_PlanetScope")["frame"]
    assert prediction["input"]["pixel_area_m2"] == pytest.approx(frame["pixel_area_m2"]) and frame["pixel_area_m2"] != 9.0
    folder = tmp_path / last["outputs"]["folder"]
    summary = json.loads((folder / "prediction.json").read_text(encoding="utf-8"))
    assert summary["prediction_id"] == prediction_id and len(summary["steps"]) == 3 and summary["frame_count"] == 3
    series = client.get(last["outputs"]["area_series_url"]).json()
    assert series["sensors"]["PlanetScope"]["n_scenes"] == 3
    listed = client.get("/api/v1/scenes").json()
    assert [i["scene_id"] for i in listed["items"][:3]] == ["20200303_PlanetScope", "20200302_PlanetScope", "20200301_PlanetScope"]


# ── (h) 파이프라인 기록이 없는 기존 장면은 지역 장면 기록으로 합성 ───────────────────────────


def test_scene_without_pipeline_record_is_synthesized_from_region(tmp_path: Path) -> None:
    client, *_ = _app(tmp_path)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    created = client.post(
        f"/api/v1/regions/{REGION_ID}/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "2024-05-01"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
    )
    assert created.status_code == 201, created.text
    payload = client.get("/api/v1/scenes/20240501_ICEYE").json()
    assert payload["status"] == "done" and payload["pipeline_id"] is None and payload["progress"] == 1.0
    assert [s["key"] for s in payload["steps"]] == list(STEP_ORDER)
    steps = _steps(payload)
    assert steps["register"]["status"] == "done" and steps["detect"]["status"] == "done" and steps["predict"]["status"] == "skipped"
    outputs = payload["outputs"]
    assert outputs["water_area_km2"] == pytest.approx(MASK_AREA_KM2, abs=1e-4)
    assert outputs["mask_url"] == f"/api/v1/regions/{REGION_ID}/satellite/scenes/20240501_ICEYE/mask"
    assert outputs["product_name"] is None and outputs["folder"] is None
    assert client.get(outputs["quicklook_url"]).status_code == 200
    assert payload["liaison"] == {k: None for k in ("order_id", "scene_id", "work_id", "algm_id", "product_type", "product_id")}
    assert client.get("/api/v1/scenes/nope").status_code == 404
    assert client.get("/api/v1/scenes/20240501_ICEYE/files/status.json").status_code == 404
    listed = client.get("/api/v1/scenes").json()
    assert any(i["scene_id"] == "20240501_ICEYE" and i["pipeline_id"] is None and i["status"] == "done" for i in listed["items"])


def test_install_pipeline_is_idempotent(tmp_path: Path) -> None:
    client, *_ = _app(tmp_path)
    app = client.app
    first = app.state.scene_pipeline
    assert install_pipeline(app) is first
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/api/v1/scenes", "/api/v1/scenes/options", "/api/v1/scenes/{scene_id}", "/api/v1/scenes/{scene_id}/files/{name}"} <= paths
    assert len([p for p in paths if p == "/api/v1/scenes"]) == 1


def test_pipeline_forces_redetection_even_when_marker_exists(tmp_path: Path) -> None:
    """같은 날짜의 완료 마커가 캐시에 있어도 새로 올린 영상은 다시 탐지한다 — 잡 argv 에 --force."""
    client, runner, _settings, wbms_settings = _app(tmp_path)
    _seed_iceye_dem(wbms_settings, "20200416")
    pre = _geotiff_bytes(tmp_path, "pre_force.tif")
    response = client.post(
        "/api/v1/scenes",
        data={"region_id": REGION_ID, "sensor": "iceye", "image_date": "2020-04-16", "acquisition_time": "18:36:30"},
        files={"pre_file": ("ICEYE_X5_SLC_SM_23467_20200416T183630.tif", pre, "image/tiff")},
    )
    assert response.status_code == 202, response.text
    payload = _wait(client, response.json()["scene_id"])
    assert payload["status"] == "done", payload
    detect_calls = [c for c in runner.calls if "/WBMS/wbms_modules/detect_water.py" in c]
    assert detect_calls and detect_calls[-1][-1] == "--force"


def test_runner_output_dirs_are_world_writable(tmp_path: Path) -> None:
    """③ calc_wlwa 출력·입력 디렉터리는 다른 uid 로 도는 wbms 컨테이너가 쓸 수 있어야 한다 (0o777)."""
    import stat
    from backend.app.pipeline.steps import _runner_writable

    out = _runner_writable(tmp_path / "fusion" / "wlwa" / "iceye" / "gupo")
    assert out.is_dir() and stat.S_IMODE(out.stat().st_mode) == 0o777
    assert stat.S_IMODE((tmp_path / "fusion" / "wlwa").stat().st_mode) == 0o777
