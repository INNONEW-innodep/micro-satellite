from __future__ import annotations

import json
import time
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.adapters.wbms_fusion import WbmsFusionAdapter
from backend.app.adapters.wbms_segmentation import WbmsSegmentationJobAdapter
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.wbms.config import WbmsSettings
from backend.app.wbms.jobs import FileJobStore
from backend.app.wbms.runner import CommandResult

BUNDLE_META = {
    "schema": "wbms.fusion.lstm/1",
    "head": "absolute",
    "days": 60,
    "loc_index": {"jeongcheon": 0, "hupo": 1, "gimhae": 2, "gupo": 3},
    "sat_type": {"planet": 0, "iceye": 1},
    "weather_channels": ["T", "H", "PP"],
    "weather_min": [0.3, 23.6, 0.0],
    "weather_max": [17.0, 95.8, 61.0],
    "wl_min": 0.0896,
    "wl_max": 2.415,
    "metrics": {
        "lodo_rmse_m": 0.0453,
        "baseline_rmse_m": 0.0199,
        "discriminative": False,
    },
    "model_sha256": "feedbeef",
}

SCENE_DATE = "20200416"
SCENE_STAMP = f"{SCENE_DATE}T183630"


def make_wbms_settings(tmp_path: Path) -> WbmsSettings:
    handover = tmp_path / "handover"
    scene = handover / "05_l1_pre" / "iceye_pre" / SCENE_DATE[:4] / SCENE_DATE
    scene.mkdir(parents=True)
    (scene / f"Pre_Busan_ICEYE_{SCENE_STAMP}.tif").write_bytes(b"not-a-real-tif")
    (scene / f"Pre_Busan_ICEYE_{SCENE_STAMP}_meta.json").write_text(
        json.dumps({"Entry": {"AcquisitionDateTime": "2020-04-16T18:36:30Z"}}),
        encoding="utf-8",
    )

    model_dir = handover / "03_model"
    model_dir.mkdir(parents=True)
    (model_dir / "WBMS_SAR_ICEYE.h5").write_bytes(b"weights")
    (model_dir / "WBMS_Optic_PlanetScope.h5").write_bytes(b"weights")

    mask_dir = handover / "06_aux" / "Masks" / "Img_DEM_3class_utm"
    mask_dir.mkdir(parents=True)
    (mask_dir / f"busan_{SCENE_DATE}_mask3_utm.tif").write_bytes(b"mask")

    # CP949-encoded daily weather CSV, positional columns like the delivered
    # busan_aws_2019_202004.csv: 지점,일시,평균기온,일강수량,습도,일사.
    # Winter temperatures stay at 2.0 C; April jumps to 20.0 C so a window
    # ending mid-April exceeds the bundle's 17.0 C training maximum.
    csv_path = handover / "06_aux" / "busan_aws.csv"
    lines = ["지점,일시,평균기온(°C),일강수량(mm),평균 상대습도(%),합계 일사량(MJ/m2)"]
    day = Date(2019, 12, 1)
    while day <= Date(2020, 4, 30):
        temp = 20.0 if day >= Date(2020, 4, 1) else 2.0
        lines.append(f"257,{day.month}/{day.day}/{day.year},{temp},0.0,60.0,8.0")
        day += timedelta(days=1)
    csv_path.write_text("\n".join(lines) + "\n", encoding="cp949")

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "meta.json").write_text(json.dumps(BUNDLE_META), encoding="utf-8")
    (bundle / "model.keras").write_bytes(b"keras")

    return WbmsSettings(
        docker_image="wbms:0.13",
        docker_bin="docker",
        handover_dir=handover,
        jobs_dir=tmp_path / "jobs",
        wb_output_dir=tmp_path / "wb",
        fusion_work_dir=tmp_path / "fusion_work",
        bundle_dir=bundle,
        aws_csv=csv_path,
        segmentation_timeout_s=10.0,
        fusion_timeout_s=10.0,
    )


def seed_cached_mask(settings: WbmsSettings, status: str = "COMPLETE") -> Path:
    scene_dir = settings.wb_output_dir / SCENE_DATE[:4] / SCENE_DATE
    scene_dir.mkdir(parents=True, exist_ok=True)
    base = f"WB_Busan_ICEYE_{SCENE_STAMP}"
    (scene_dir / f"{base}.tif").write_bytes(b"mask")
    (scene_dir / f"{base}_meta.json").write_text("{}", encoding="utf-8")
    qc = {
        "base": base,
        "status": status,
        "products": [f"{base}.tif", f"{base}_meta.json"],
        "params": {
            "sensor_name": "ICEYE",
            "testbed": "busan",
            "image_date": SCENE_DATE,
        },
        "runtime": {"elapsed_s": 610.6},
    }
    qc_path = scene_dir / f"{base}_qc.json"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    return qc_path


def _mounts(argv: list[str]) -> dict[str, Path]:
    """container path -> host path from -v host:container[:ro] flags."""

    mapping: dict[str, Path] = {}
    for flag, value in zip(argv, argv[1:]):
        if flag != "-v":
            continue
        host, container, *_ = value.split(":")
        mapping[container] = Path(host)
    return mapping


def _arg(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


class FakeRunner:
    """Stands in for DockerRunner — the subprocess boundary is never crossed."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.mode = "ok"
        self.fusion_offset_m = 0.1

    def image_available(self, image: str) -> bool:
        return True

    def run(self, argv: list[str], timeout_s: float) -> CommandResult:
        self.calls.append(list(argv))
        if self.mode == "fail":
            stdout = (
                '{"type":"error","ts":"t","code":"MODEL_LOAD_FAIL",'
                '"param":"weights","message":"가중치 없음"}\n'
            )
            return CommandResult(returncode=2, stdout=stdout, stderr="boom", elapsed_s=0.1)
        if "/WBMS/wbms_modules/detect_water.py" in argv:
            self._fake_detect_water(argv)
        elif "/WBMS/wbms_modules/Correct.py" in argv:
            self._fake_correct(argv)
        return CommandResult(returncode=0, stdout='{"type":"done","ts":"t"}\n', stderr="", elapsed_s=0.2)

    def _fake_detect_water(self, argv: list[str]) -> None:
        out_host = _mounts(argv)[_arg(argv, "--output_dir")]
        image_date = _arg(argv, "--image_date")
        scene_dir = out_host / image_date[:4] / image_date
        scene_dir.mkdir(parents=True, exist_ok=True)
        base = f"WB_Busan_ICEYE_{image_date}T000000"
        (scene_dir / f"{base}.tif").write_bytes(b"mask")
        (scene_dir / f"{base}_meta.json").write_text("{}", encoding="utf-8")
        qc = {
            "base": base,
            "status": "COMPLETE",
            "products": [f"{base}.tif", f"{base}_meta.json"],
            "params": {
                "sensor_name": "ICEYE",
                "testbed": _arg(argv, "--testbed"),
                "image_date": image_date,
            },
        }
        (scene_dir / f"{base}_qc.json").write_text(json.dumps(qc), encoding="utf-8")

    def _fake_correct(self, argv: list[str]) -> None:
        work_host = _mounts(argv)["/work"]
        loc = _arg(argv, "--loc_id")
        records = []
        for wlwa in sorted(work_host.glob("wlwa/*/WLWA_*.json")):
            if wlwa.name.endswith("_qc.json"):
                continue
            doc = json.loads(wlwa.read_text(encoding="utf-8"))
            for rec in doc["records"]:
                raw = float(rec["water_level_m"])
                corrected = raw + self.fusion_offset_m
                records.append(
                    {
                        "testbed": doc["testbed"],
                        "loc_id": loc,
                        "satellite": rec["satellite"],
                        "date": rec["date"],
                        "water_level_m": raw,
                        "water_area_km2": rec.get("water_area_km2"),
                        "corrected_water_level_m": round(corrected, 6),
                        "correction_mode": "absolute",
                        "offset_m": self.fusion_offset_m,
                        "bias_m": None,
                        "residual_m": None,
                        "wl_out_of_train_range": bool(
                            raw < BUNDLE_META["wl_min"] or raw > BUNDLE_META["wl_max"]
                        ),
                        "pair_gap_days": None,
                        "paired": False,
                        "source_wb": rec.get("source_wb"),
                    }
                )
        doc = {"testbed": "busan", "method": "lstm", "records": records}
        fused_dir = work_host / "fusion"
        fused_dir.mkdir(parents=True, exist_ok=True)
        (fused_dir / f"FUSED_Busan_{loc}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )


def make_client(tmp_path: Path) -> tuple[TestClient, FakeRunner, WbmsSettings]:
    wbms_settings = make_wbms_settings(tmp_path)
    app = create_app(Settings(result_dir=tmp_path / "results"), wbms_settings)
    runner = FakeRunner()
    app.state.wbms_segmentation.runner = runner
    app.state.wbms_fusion.runner = runner
    return TestClient(app), runner, wbms_settings


def wait_terminal(client: TestClient, job_id: str, deadline_s: float = 5.0) -> dict:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        job = client.get(f"/api/v1/wbms/segmentation/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish within {deadline_s}s: {job}")


# ── status ──────────────────────────────────────────────────────────────────


def test_wbms_status_reports_readiness(tmp_path: Path) -> None:
    client, _runner, settings = make_client(tmp_path)
    seed_cached_mask(settings)
    payload = client.get("/api/v1/wbms/status").json()
    assert payload["docker_image"] == "wbms:0.13"
    assert payload["image_available"] is True
    assert payload["bundle_available"] is True
    assert payload["aws_csv_available"] is True
    assert payload["handover_available"] is True
    assert payload["cached_scenes"] and payload["cached_scenes"][0]["image_date"] == SCENE_DATE


# ── ② segmentation job adapter ──────────────────────────────────────────────


def test_segmentation_cache_hit_completes_without_docker(tmp_path: Path) -> None:
    client, runner, settings = make_client(tmp_path)
    seed_cached_mask(settings)
    response = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["status"] == "succeeded"
    assert job["cached"] is True
    assert job["result"]["source"] == "cache"
    assert job["result"]["base"] == f"WB_Busan_ICEYE_{SCENE_STAMP}"
    assert Path(job["result"]["mask_tif"]).is_file()
    assert job["result"]["elapsed_s"] == pytest.approx(610.6)
    assert runner.calls == []  # the cache path never launches a container
    # the job is persisted as a plain JSON file
    assert (settings.jobs_dir / f"{job['job_id']}.json").is_file()


def test_incomplete_marker_is_not_a_cache_hit(tmp_path: Path) -> None:
    _client, runner, settings = make_client(tmp_path)
    seed_cached_mask(settings, status="RUNNING")
    adapter = WbmsSegmentationJobAdapter(settings, runner)
    assert adapter.find_cached("iceye", "busan", SCENE_DATE) is None
    # a COMPLETE marker whose product file is missing is also rejected
    qc_path = seed_cached_mask(settings)
    (qc_path.parent / f"WB_Busan_ICEYE_{SCENE_STAMP}.tif").unlink()
    assert adapter.find_cached("iceye", "busan", SCENE_DATE) is None


def test_segmentation_async_flow_builds_verified_command(tmp_path: Path) -> None:
    client, runner, settings = make_client(tmp_path)
    response = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE},
    )
    assert response.status_code == 202, response.text
    created = response.json()
    assert created["status"] in {"queued", "running", "succeeded"}
    assert created["cached"] is False

    job = wait_terminal(client, created["job_id"])
    assert job["status"] == "succeeded"
    assert job["result"]["source"] == "run"
    assert Path(job["result"]["mask_tif"]).is_file()
    assert job["started_at"] and job["finished_at"]

    argv = runner.calls[0]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "wbms:0.13" in argv
    assert "/WBMS/wbms_modules/detect_water.py" in argv
    assert _arg(argv, "--input") == (
        f"/data/05_l1_pre/iceye_pre/2020/{SCENE_DATE}/Pre_Busan_ICEYE_{SCENE_STAMP}.tif"
    )
    assert _arg(argv, "--sensor") == "iceye"
    assert _arg(argv, "--image_date") == SCENE_DATE
    assert _arg(argv, "--weights") == "/model/WBMS_SAR_ICEYE.h5"
    assert _arg(argv, "--mask") == (
        f"/aux/Masks/Img_DEM_3class_utm/busan_{SCENE_DATE}_mask3_utm.tif"
    )
    assert _arg(argv, "--gpu") == "cpu"
    assert _arg(argv, "--output_dir") == "/out"
    mounts = _mounts(argv)
    assert mounts["/data"] == settings.handover_dir
    assert mounts["/out"] == settings.wb_output_dir
    # read-only where it must be
    volume_args = [value for flag, value in zip(argv, argv[1:]) if flag == "-v"]
    assert any(value.endswith(":/data:ro") for value in volume_args)
    assert any(value.endswith(":/model:ro") for value in volume_args)
    assert all(not value.endswith(":/out:ro") for value in volume_args)


def test_segmentation_failure_surfaces_container_error(tmp_path: Path) -> None:
    client, runner, _settings = make_client(tmp_path)
    runner.mode = "fail"
    created = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE},
    ).json()
    job = wait_terminal(client, created["job_id"])
    assert job["status"] == "failed"
    assert "MODEL_LOAD_FAIL" in job["error"]
    assert "exit=2" in job["error"]


def test_segmentation_unknown_scene_and_missing_mask_are_422(tmp_path: Path) -> None:
    client, _runner, settings = make_client(tmp_path)
    response = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": "20200101"},
    )
    assert response.status_code == 422
    assert "Pre_*.tif" in response.json()["detail"]

    mask = (
        settings.aux_dir / "Masks" / "Img_DEM_3class_utm" / f"busan_{SCENE_DATE}_mask3_utm.tif"
    )
    mask.unlink()
    response = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE},
    )
    assert response.status_code == 422
    assert "mask" in response.json()["detail"]


def test_unknown_job_is_404(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    assert client.get("/api/v1/wbms/segmentation/jobs/does-not-exist").status_code == 404


def test_job_store_transitions_and_stale_owner(tmp_path: Path) -> None:
    store = FileJobStore(tmp_path / "jobs")
    job = store.create({"job_id": "seg_x_1", "status": "queued"})
    assert job["created_at"]
    store.update("seg_x_1", status="running", started_at="t1")
    assert store.get("seg_x_1")["status"] == "running"
    store.update("seg_x_1", status="succeeded", finished_at="t2")
    assert store.get("seg_x_1")["status"] == "succeeded"
    assert [item["job_id"] for item in store.list()] == ["seg_x_1"]

    # a queued/running job owned by a dead process is reported failed, not stuck
    store.create({"job_id": "seg_x_2", "status": "running", "runner_pid": 2_147_000_111})
    fixed = store.get("seg_x_2")
    assert fixed["status"] == "failed"
    assert "no longer running" in fixed["error"]
    # terminal states are never rewritten
    assert store.get("seg_x_1")["status"] == "succeeded"


# ── ④ fusion LSTM adapter ───────────────────────────────────────────────────


def test_fusion_correction_schema_and_command(tmp_path: Path) -> None:
    client, runner, settings = make_client(tmp_path)
    response = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "GUPO",
            "observations": [
                {"date": "2020-04-16", "satellite": "ICEYE", "water_level_m": 0.6}
            ],
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_id"] == "wbms-fusion-lstm"
    assert payload["semantics"] == "same_day_correction"
    assert payload["loc_id"] == "gupo"

    record = payload["records"][0]
    assert record["date"] == "20200416"
    assert record["satellite"] == "ICEYE"
    assert record["satellite_water_level_m"] == pytest.approx(0.6)
    assert record["corrected_water_level_m"] == pytest.approx(0.7)
    assert record["offset_m"] == pytest.approx(0.1)
    assert record["correction_mode"] == "absolute"
    assert record["wl_out_of_train_range"] is False

    window = payload["weather_windows"][0]
    assert window["window_days"] == 60
    assert window["window_end"] == "2020-04-16"
    assert window["temperature_out_of_train_range"] is True  # April 20 C > train max 17 C
    assert any("training range" in warning for warning in payload["warnings"])
    assert any("winter" in warning for warning in payload["warnings"])
    # bundle honesty note travels with every response
    assert payload["bundle"]["wl_valid_range_m"] == [0.0896, 2.415]
    assert payload["bundle"]["discriminative"] is False
    assert any("baseline" in warning for warning in payload["warnings"])

    argv = runner.calls[0]
    assert "/WBMS/wbms_modules/Correct.py" in argv
    assert _arg(argv, "--loc_id") == "gupo"
    assert _arg(argv, "--gpu") == "cpu"
    assert _arg(argv, "--model") == "/bundle"
    assert _arg(argv, "--sar_result_dir") == "/work/wlwa/ICEYE"
    mounts = _mounts(argv)
    assert mounts["/bundle"] == settings.bundle_dir
    workspace = Path(payload["runtime"]["workspace"])
    assert mounts["/work"] == workspace
    # the synthesized ③-style input carries a COMPLETE marker, as Correct.py requires
    qc_files = list(workspace.glob("wlwa/ICEYE/WLWA_*_qc.json"))
    assert qc_files and json.loads(qc_files[0].read_text())["status"] == "COMPLETE"


def test_fusion_winter_window_has_no_temperature_warning(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    payload = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "gupo",
            "observations": [
                {"date": "20200215", "satellite": "iceye", "water_level_m": 0.6}
            ],
        },
    ).json()
    assert payload["weather_windows"][0]["temperature_out_of_train_range"] is False
    assert not any("training range [0.3, 17.0]" in warning for warning in payload["warnings"])


def test_fusion_unknown_loc_is_422_with_choices(tmp_path: Path) -> None:
    client, runner, _settings = make_client(tmp_path)
    response = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "nowhere",
            "observations": [
                {"date": "20200416", "satellite": "iceye", "water_level_m": 0.6}
            ],
        },
    )
    assert response.status_code == 422
    assert "gupo" in response.json()["detail"]
    assert runner.calls == []


def test_fusion_short_weather_window_is_422_no_zero_padding(tmp_path: Path) -> None:
    client, runner, _settings = make_client(tmp_path)
    response = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "gupo",
            "observations": [
                {"date": "20191205", "satellite": "iceye", "water_level_m": 0.6}
            ],
        },
    )
    assert response.status_code == 422
    assert "zero-padding" in response.json()["detail"].lower()
    assert runner.calls == []

    # a date entirely outside the CSV is refused too
    response = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "gupo",
            "observations": [
                {"date": "20210101", "satellite": "iceye", "water_level_m": 0.6}
            ],
        },
    )
    assert response.status_code == 422


def test_fusion_out_of_range_level_warns_and_flags(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    payload = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "gupo",
            "observations": [
                {"date": "20200416", "satellite": "iceye", "water_level_m": 5.0}
            ],
        },
    ).json()
    assert payload["records"][0]["wl_out_of_train_range"] is True
    assert any("[0.0896, 2.415]" in warning for warning in payload["warnings"])


def test_fusion_persistence_extension_is_labeled(tmp_path: Path) -> None:
    client, _runner, _settings = make_client(tmp_path)
    payload = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "gupo",
            "observations": [
                {"date": "20200416", "satellite": "iceye", "water_level_m": 0.6}
            ],
            "persistence_horizon_days": 3,
        },
    ).json()
    steps = payload["persistence_forecast"]
    assert [step["date"] for step in steps] == ["20200417", "20200418", "20200419"]
    assert all(step["method"] == "persistence" for step in steps)
    assert all(
        step["corrected_water_level_m"]
        == payload["records"][0]["corrected_water_level_m"]
        for step in steps
    )
    assert any("does not forecast" in warning for warning in payload["warnings"])


def test_fusion_container_failure_maps_to_502(tmp_path: Path) -> None:
    client, runner, _settings = make_client(tmp_path)
    runner.mode = "fail"
    response = client.post(
        "/api/v1/wbms/fusion/corrections",
        json={
            "loc_id": "gupo",
            "observations": [
                {"date": "20200416", "satellite": "iceye", "water_level_m": 0.6}
            ],
        },
    )
    assert response.status_code == 502
    assert "MODEL_LOAD_FAIL" in response.json()["detail"]


def test_fusion_adapter_duplicate_observation_rejected(tmp_path: Path) -> None:
    settings = make_wbms_settings(tmp_path)
    adapter = WbmsFusionAdapter(settings, FakeRunner())
    from backend.app.adapters.wbms_segmentation import WbmsInputError

    with pytest.raises(WbmsInputError, match="duplicate"):
        adapter.correct(
            {
                "loc_id": "gupo",
                "testbed": "busan",
                "observations": [
                    {"date": "20200416", "satellite": "iceye", "water_level_m": 0.5},
                    {"date": "20200416", "satellite": "iceye", "water_level_m": 0.7},
                ],
            }
        )
