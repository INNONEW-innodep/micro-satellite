from __future__ import annotations

import json
import time
from datetime import date as Date
from datetime import timedelta
from itertools import pairwise
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
        json.dumps({"Entry": {"AcquisitionDateTime": "2020-04-16T18:36:30Z"},
                    "Provenance": {"stage": "preprocess", "input_file": f"ICEYE_X5_SLC_SM_23467_{SCENE_STAMP}.h5"}}),
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


# ── GPU 모드 (BACKEND_WBMS_GPU) ───────────────────────────────────────────────


def test_resolve_gpu_device_cpu_mode_never_probes(monkeypatch):
    from backend.app.wbms import runner as r

    monkeypatch.setattr(r, "gpu_free_mib", lambda: (_ for _ in ()).throw(AssertionError))
    assert r.resolve_gpu_device("cpu", 3000) == ("cpu", None)


def test_resolve_gpu_device_auto_falls_back_with_warning(monkeypatch):
    from backend.app.wbms import runner as r

    monkeypatch.setattr(r, "gpu_free_mib", lambda: 512)
    device, warning = r.resolve_gpu_device("auto", 3000)
    assert device == "cpu"
    assert warning and "512" in warning


def test_resolve_gpu_device_strict_gpu_raises_when_starved(monkeypatch):
    import pytest

    from backend.app.wbms import runner as r

    monkeypatch.setattr(r, "gpu_free_mib", lambda: 100)
    with pytest.raises(ValueError):
        r.resolve_gpu_device("gpu", 3000)


def test_resolve_gpu_device_grants_gpu_when_free(monkeypatch):
    from backend.app.wbms import runner as r

    monkeypatch.setattr(r, "gpu_free_mib", lambda: 20000)
    assert r.resolve_gpu_device("auto", 3000) == ("gpu", None)


def test_docker_prefix_gpu_flag_adds_gpus_and_growth(tmp_path):
    from backend.app.wbms.runner import docker_prefix

    argv = docker_prefix("docker", "wbms:0.13", [(tmp_path, "/x", True)], gpu=True)
    assert "--gpus" in argv and "device=0" in argv
    assert "TF_FORCE_GPU_ALLOW_GROWTH=true" in argv
    cpu_argv = docker_prefix("docker", "wbms:0.13", [(tmp_path, "/x", True)])
    assert "--gpus" not in cpu_argv


# ── 컨테이너 안에서 호스트 docker 로 실행 (경로 변환·BACKEND_RUN_AS·docker 경유 GPU 프리플라이트) ──
# docker-compose.yml 이 /var/run/docker.sock 을 API 컨테이너에 넣으면 argv 는 컨테이너 안에서
# 조립되고 실행은 호스트 데몬이 한다. 그래서 -v 호스트 경로는 /app → HOST_ROOT 로 바꿔야 하고,
# --user 는 호스트 사용자여야 하며, nvidia-smi 는 docker 경유로 묻는다.


@pytest.fixture
def clean_runner(monkeypatch):
    """runner 모듈 전역(경로 변환표·GPU 프로브)과 관련 환경변수를 테스트 뒤 원상복구한다."""
    from backend.app.wbms import runner as r

    monkeypatch.setattr(r, "_PATH_MAP", None)
    monkeypatch.setattr(r, "_GPU_PROBE", ("docker", "wbms:0.13"))
    for key in ("BACKEND_RUN_AS", "BACKEND_HOST_ROOT", "BACKEND_CONTAINER_ROOT",
                "BACKEND_WBMS_DOCKER_BIN", "BACKEND_WBMS_IMAGE"):
        monkeypatch.delenv(key, raising=False)
    return r


def _volumes(argv: list[str]) -> list[str]:
    return [value for flag, value in pairwise(argv) if flag == "-v"]


def test_docker_prefix_translates_mounts_under_container_root(tmp_path, clean_runner):
    r = clean_runner
    container_root = tmp_path / "app"
    (container_root / "data" / "incoming").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    r.set_path_map(container_root, Path("/srv/watercast"))
    argv = r.docker_prefix("docker", "wbms:0.13", [
        (container_root / "data" / "incoming", "/data", True),
        (container_root / "data" / "wbms_runs" / "wb", "/out", False),  # 아직 없는 경로도 변환
        (outside, "/scene", True),  # container_root 밖 → 그대로
    ])
    vols = _volumes(argv)
    assert "/srv/watercast/data/incoming:/data:ro" in vols
    assert "/srv/watercast/data/wbms_runs/wb:/out" in vols
    assert f"{outside}:/scene:ro" in vols
    # 변환표가 없으면(호스트 직접 실행) 예전과 같다
    r.set_path_map(None, None)
    assert f"{container_root / 'data' / 'incoming'}:/data:ro" in _volumes(
        r.docker_prefix("docker", "wbms:0.13", [(container_root / "data" / "incoming", "/data", True)])
    )


def test_translate_host_path_resolves_symlinks_first(tmp_path, clean_runner):
    r = clean_runner
    container_root = tmp_path / "app"
    real = container_root / "data" / "real"
    real.mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    shortcut = tmp_path / "shortcut"  # container_root 밖의 링크 → 안을 가리킨다
    shortcut.symlink_to(real)
    escape = container_root / "escape"  # container_root 안의 링크 → 밖을 가리킨다
    escape.symlink_to(tmp_path / "outside")
    explicit = (container_root, Path("/srv/wc"))

    assert r.translate_host_path(shortcut, explicit) == Path("/srv/wc/data/real")
    assert r.translate_host_path(escape, explicit) == escape  # 실제 위치가 밖이면 손대지 않는다
    # 인자로 준 변환표가 모듈 기본값보다 우선하고, 기본값이 없으면 그대로
    r.set_path_map(container_root, Path("/module"))
    assert r.translate_host_path(real) == Path("/module/data/real")
    assert r.translate_host_path(real, explicit) == Path("/srv/wc/data/real")
    r.set_path_map(None, None)
    assert r.translate_host_path(real) == real


def test_docker_prefix_user_from_backend_run_as(tmp_path, monkeypatch, clean_runner):
    import os

    r = clean_runner
    mounts = [(tmp_path, "/x", True)]
    assert _arg(r.docker_prefix("docker", "wbms:0.13", mounts), "--user") == f"{os.getuid()}:{os.getgid()}"
    monkeypatch.setenv("BACKEND_RUN_AS", "1000:1000")
    assert _arg(r.docker_prefix("docker", "wbms:0.13", mounts), "--user") == "1000:1000"
    monkeypatch.setenv("BACKEND_RUN_AS", "nobody")  # 형식 오류 → 프로세스 uid:gid 로 폴백
    assert _arg(r.docker_prefix("docker", "wbms:0.13", mounts), "--user") == f"{os.getuid()}:{os.getgid()}"
    assert _arg(r.docker_prefix("docker", "wbms:0.13", mounts, run_as="42:42"), "--user") == "42:42"


def test_gpu_free_mib_falls_back_to_docker_when_nvidia_smi_missing(monkeypatch, clean_runner):
    import subprocess

    r = clean_runner
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")  # API 컨테이너 안
        assert kwargs.get("timeout") == 60.0
        return subprocess.CompletedProcess(argv, 0, stdout="20480\n", stderr="")

    monkeypatch.setattr(r.subprocess, "run", fake_run)
    r.set_gpu_probe("/usr/local/bin/docker", "wbms:9.9")
    assert r.gpu_free_mib() == 20480
    assert calls[0][0] == "nvidia-smi"
    docker_argv = calls[1]
    assert docker_argv[:5] == ["/usr/local/bin/docker", "run", "--rm", "--gpus", "device=0"]
    assert _arg(docker_argv, "--entrypoint") == "nvidia-smi"
    assert "wbms:9.9" in docker_argv and "--query-gpu=memory.free" in docker_argv
    # 인자가 모듈 기본값보다 우선한다
    assert r.gpu_free_mib("docker", "other:1") == 20480 and calls[-1][0] == "docker" and "other:1" in calls[-1]
    # auto 모드는 그 값을 그대로 쓴다
    assert r.resolve_gpu_device("auto", 3000) == ("gpu", None)


def test_gpu_free_mib_none_when_docker_probe_fails_too(monkeypatch, clean_runner):
    import subprocess

    r = clean_runner

    def no_toolkit(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return subprocess.CompletedProcess(argv, 125, stdout="", stderr="could not select device driver")

    monkeypatch.setattr(r.subprocess, "run", no_toolkit)
    assert r.gpu_free_mib() is None
    device, warning = r.resolve_gpu_device("auto", 3000)
    assert device == "cpu" and warning and "auto 모드라 CPU 로 폴백" in warning
    with pytest.raises(ValueError):
        r.resolve_gpu_device("gpu", 3000)

    def no_socket(argv, **kwargs):  # docker 자체가 터지는 경우(소켓 미마운트·타임아웃)
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        raise subprocess.TimeoutExpired(argv, 60)

    monkeypatch.setattr(r.subprocess, "run", no_socket)
    assert r.gpu_free_mib() is None


def test_gpu_free_mib_prefers_host_nvidia_smi(monkeypatch, clean_runner):
    import subprocess

    r = clean_runner
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="512\n", stderr="")

    monkeypatch.setattr(r.subprocess, "run", fake_run)
    assert r.gpu_free_mib() == 512
    assert len(calls) == 1 and calls[0][0] == "nvidia-smi"  # 호스트에 있으면 docker 를 부르지 않는다


def test_settings_from_env_sets_path_map_and_gpu_probe(monkeypatch, clean_runner):
    r = clean_runner
    monkeypatch.setenv("BACKEND_HOST_ROOT", "/srv/watercast")
    monkeypatch.setenv("BACKEND_CONTAINER_ROOT", "/app")
    monkeypatch.setenv("BACKEND_WBMS_DOCKER_BIN", "/usr/local/bin/docker")
    monkeypatch.setenv("BACKEND_WBMS_IMAGE", "wbms:0.13")
    monkeypatch.setenv("BACKEND_RUN_AS", "1000:1000")
    settings = WbmsSettings.from_env()
    assert settings.host_root == Path("/srv/watercast")
    assert settings.container_root == Path("/app")
    assert r.get_path_map() == (Path("/app"), Path("/srv/watercast"))
    assert r._GPU_PROBE == ("/usr/local/bin/docker", "wbms:0.13")
    # 어댑터가 부르는 형태(설정 객체 없이)로도 변환과 --user 가 적용된다
    argv = r.docker_prefix(settings.docker_bin, settings.docker_image,
                           [(Path("/app/data/wbms_runs/wb"), "/out", False)])
    assert "/srv/watercast/data/wbms_runs/wb:/out" in _volumes(argv)
    assert _arg(argv, "--user") == "1000:1000"
    # BACKEND_HOST_ROOT 가 비면 변환표도 비고(호스트 직접 실행), container_root 기본은 /app
    monkeypatch.delenv("BACKEND_HOST_ROOT")
    monkeypatch.delenv("BACKEND_CONTAINER_ROOT")
    settings = WbmsSettings.from_env()
    assert settings.host_root is None and settings.container_root == Path("/app")
    assert r.get_path_map() is None


def test_segmentation_job_command_uses_host_paths_when_path_map_set(tmp_path, clean_runner):
    """어댑터를 고치지 않고도 잡 command 의 -v 가 호스트 경로로 나가는지 — 끝단 통합 확인.

    호스트 쪽 경로를 tmp_path 를 가리키는 링크로 만들어, FakeRunner 가 '호스트 경로'에 쓴
    산출물이 실제로는 같은 디렉터리에 떨어지게 한다(변환이 안 됐다면 링크 경로가 안 보인다).
    """
    r = clean_runner
    host_alias = tmp_path.parent / f"{tmp_path.name}-host"
    host_alias.symlink_to(tmp_path, target_is_directory=True)
    r.set_path_map(tmp_path, host_alias)
    client, _runner, _settings = make_client(tmp_path)
    created = client.post(
        "/api/v1/wbms/segmentation/jobs",
        json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE, "force": True},
    ).json()
    job = wait_terminal(client, created["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["source"] == "run"
    vols = _volumes(job["command"])
    assert vols and all(value.startswith(str(host_alias)) for value in vols), vols
    assert f"{host_alias}/handover:/data:ro" in vols
    assert f"{host_alias}/handover/03_model:/model:ro" in vols
    assert any(value.endswith(":/out") for value in vols)
    assert Path(job["result"]["mask_tif"]).is_file()


def test_force_is_forwarded_to_detect_water(tmp_path: Path) -> None:
    """force=true 는 백엔드 캐시 우회만이 아니라 모듈의 완료 마커도 무시해야 한다 — argv 에 --force."""
    client, runner, _ = make_client(tmp_path)
    first = client.post("/api/v1/wbms/segmentation/jobs", json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE}).json()
    wait_terminal(client, first["job_id"])
    assert "--force" not in runner.calls[0]
    forced = client.post("/api/v1/wbms/segmentation/jobs",
                         json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE, "force": True}).json()
    job = wait_terminal(client, forced["job_id"])
    assert job["cached"] is False and job["status"] == "succeeded"
    assert runner.calls[-1][-1] == "--force" and "--output_dir" in runner.calls[-1]


# ── Umbra: 전용 모델이 없어 ICEYE 모델을 빌려 쓰는 센서 — 카탈로그 module_sensor·캐시 표기 허용 ──────


def test_catalog_module_sensor_and_labels() -> None:
    from backend.app.wbms.catalog import default_catalog

    catalog = default_catalog()
    assert list(catalog.sensors) == ["iceye", "umbra", "planet"], "선택지 순서: ICEYE 바로 뒤에 UMBRA"
    assert catalog.module_sensor("iceye") == "iceye" and catalog.module_sensor("planet") == "planet"
    assert catalog.module_sensor("umbra") == "iceye" and catalog.module_name("umbra") == "ICEYE"
    umbra = catalog.sensor("umbra")
    assert umbra["name"] == "Umbra" and umbra["type"] == "SAR" and umbra["weights"] == "WBMS_SAR_ICEYE.h5" and umbra["needs_mask"] is True
    assert umbra["pre_dirs"] == {"dcd": "umbra_pre"} and catalog.sensor("iceye")["pre_dirs"] == {"busan": "iceye_pre"}
    assert "ICEYE" in umbra["description_ko"] and "iceye" in umbra["description_ko"]
    assert catalog.supports("umbra", "dcd") and not catalog.supports("umbra", "busan") and not catalog.supports("iceye", "dcd")
    assert catalog.file_label("umbra") == "UMBRA" and catalog.file_label("iceye") == "ICEYE" and catalog.file_label("planet") == "PlanetScope"
    assert catalog.cache_sensor_names("umbra") == ["Umbra", "UMBRA", "ICEYE"] and catalog.cache_sensor_names("iceye") == ["ICEYE"]
    # 빌린 표기(ICEYE)는 umbra 가 인수된 테스트베드(dcd)에서만 — busan 조회가 ICEYE 부산 산출물을 집지 않게
    assert catalog.cache_sensor_names("umbra", "dcd") == ["Umbra", "UMBRA", "ICEYE"] and catalog.cache_sensor_names("umbra", "busan") == ["Umbra", "UMBRA"]
    assert catalog.cache_sensor_names("iceye", "dcd") == ["ICEYE"]
    assert catalog.resolve_key("UMBRA") == catalog.resolve_key("Umbra") == catalog.resolve_key(" umbra ") == "umbra"
    assert catalog.resolve_key("PlanetScope") == catalog.resolve_key("planet") == "planet" and catalog.resolve_key("kompsat5") is None
    caps = catalog.capabilities()["sensors"]["umbra"]
    assert caps["module_sensor"] == "iceye" and caps["file_label"] == "UMBRA" and caps["testbeds"] == ["dcd"]
    assert catalog.capabilities()["sensors"]["iceye"]["module_sensor"] == "iceye"


def test_catalog_module_sensor_defaults_and_rejects_unknown(tmp_path: Path) -> None:
    from backend.app.wbms.catalog import DEFAULT_CATALOG_PATH, WbmsCatalog

    doc = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    # module_sensor·file_label 을 적지 않은 항목은 자기 키·이름으로 채워진다
    doc["sensors"]["kompsat5"] = {"name": "KOMPSAT-5", "type": "SAR", "weights": "k5.h5", "pre_dirs": {"busan": "k5_pre"}}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = WbmsCatalog.load(path)
    assert loaded.module_sensor("kompsat5") == "kompsat5" and loaded.module_name("kompsat5") == "KOMPSAT-5"
    assert loaded.file_label("kompsat5") == "KOMPSAT-5" and loaded.cache_sensor_names("kompsat5") == ["KOMPSAT-5"]
    # 카탈로그에 없는 모듈 센서를 빌리면 로드 단계에서 거부
    doc["sensors"]["umbra"]["module_sensor"] = "capella"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="module_sensor"):
        WbmsCatalog.load(path)


def _seed_cached_wb(settings: WbmsSettings, *, tb: str, sensor_name: str, image_date: str, stamp: str,
                    params_sensor: str | None, status: str = "COMPLETE") -> Path:
    scene_dir = settings.wb_output_dir / image_date[:4] / image_date
    scene_dir.mkdir(parents=True, exist_ok=True)
    base = f"WB_{tb}_{sensor_name}_{stamp}"
    (scene_dir / f"{base}.tif").write_bytes(b"mask")
    (scene_dir / f"{base}_meta.json").write_text("{}", encoding="utf-8")
    params = {"testbed": tb.lower(), "image_date": image_date}
    if params_sensor is not None:
        params["sensor_name"] = params_sensor
    qc_path = scene_dir / f"{base}_qc.json"
    qc_path.write_text(json.dumps({"base": base, "status": status, "products": [f"{base}.tif", f"{base}_meta.json"], "params": params}), encoding="utf-8")
    return qc_path


def test_find_cached_accepts_module_sensor_name_for_umbra(tmp_path: Path) -> None:
    _client, runner, settings = make_client(tmp_path)
    adapter = WbmsSegmentationJobAdapter(settings, runner)
    date, stamp = "20260501", "20260501T093000"
    # 컨테이너가 iceye 로 돌아 WB_DCD_ICEYE_* 를 썼다(qc.params.sensor_name 도 ICEYE) — umbra 조회가 이를 받아들인다
    _seed_cached_wb(settings, tb="DCD", sensor_name="ICEYE", image_date=date, stamp=stamp, params_sensor="ICEYE")
    found = adapter.find_cached("umbra", "dcd", date)
    assert found is not None and found["base"] == f"WB_DCD_ICEYE_{stamp}" and found["source"] == "cache"
    # (문서화된 모호성) 같은 날짜·테스트베드의 ICEYE 조회도 같은 파일을 본다
    assert adapter.find_cached("iceye", "dcd", date)["base"] == found["base"]
    # 자기 라벨(UMBRA)이 든 산출물이 함께 있으면 그것을 우선한다 — ICEYE 조회는 UMBRA 표기를 받지 않는다
    _seed_cached_wb(settings, tb="DCD", sensor_name="UMBRA", image_date=date, stamp=stamp, params_sensor="Umbra")
    assert adapter.find_cached("umbra", "dcd", date)["base"] == f"WB_DCD_UMBRA_{stamp}"
    assert adapter.find_cached("iceye", "dcd", date)["base"] == f"WB_DCD_ICEYE_{stamp}"
    # 실행 직후 판정(min_mtime): 실행 시작 이후 기록된 것만 — 미래 시각이면 없음
    assert adapter.find_cached("umbra", "dcd", date, min_mtime=time.time() + 3600) is None
    assert adapter.find_cached("umbra", "dcd", date, min_mtime=time.time() - 3600)["base"] == f"WB_DCD_UMBRA_{stamp}"
    # qc.params.sensor_name 이 허용 표기 밖이면 파일명이 맞아도 거부; sensor_name 이 없으면 파일명만으로 인정
    _seed_cached_wb(settings, tb="DCD", sensor_name="UMBRA", image_date="20260502", stamp="20260502T093000", params_sensor="PlanetScope")
    assert adapter.find_cached("umbra", "dcd", "20260502") is None
    _seed_cached_wb(settings, tb="DCD", sensor_name="ICEYE", image_date="20260503", stamp="20260503T093000", params_sensor=None)
    assert adapter.find_cached("umbra", "dcd", "20260503")["base"] == "WB_DCD_ICEYE_20260503T093000"
    # umbra 는 busan 에 인수되지 않았다 — ICEYE 부산 산출물을 umbra 것으로 집지 않는다; 기존 ICEYE/busan 캐시는 그대로
    seed_cached_mask(settings)
    assert adapter.find_cached("umbra", "busan", SCENE_DATE) is None
    assert adapter.find_cached("iceye", "busan", SCENE_DATE)["base"] == f"WB_Busan_ICEYE_{SCENE_STAMP}"
    assert {c["sensor_name"] for c in adapter.list_cached()} == {"ICEYE", "Umbra", "PlanetScope", None}


def test_umbra_scene_runs_container_as_iceye(tmp_path: Path) -> None:
    from backend.app.adapters.wbms_segmentation import WbmsInputError

    _client, runner, settings = make_client(tmp_path)
    adapter = WbmsSegmentationJobAdapter(settings, runner)
    scene_dir = settings.l1_pre_dir / "umbra_pre" / "2026" / "20260501"
    scene_dir.mkdir(parents=True)
    (scene_dir / "Pre_DCD_UMBRA_20260501T093000.tif").write_bytes(b"tif")
    (scene_dir / "Pre_DCD_UMBRA_20260501T093000_meta.json").write_text(
        json.dumps({"SatelliteName": "Umbra", "Entry": {"AcquisitionDateTime": "2026-05-01T09:30:00Z"},
                    "Provenance": {"input_file": "2026-05-01-09-30-00_UMBRA-10_SICD.nitf"}}), encoding="utf-8",
    )
    # SAR 인데 인수 3-class 마스크가 없으면 422 (파이프라인은 영상 유효 범위로 생성해 mask_tif 로 넘긴다)
    with pytest.raises(WbmsInputError, match="3-class mask"):
        adapter.resolve_scene("umbra", "dcd", "20260501")
    mask3 = tmp_path / "gen" / "dcd_20260501_mask3_utm.tif"
    mask3.parent.mkdir()
    mask3.write_bytes(b"m")
    scene = adapter.resolve_scene("umbra", "dcd", "20260501", mask_tif=str(mask3))
    assert scene["sensor"] == "umbra" and scene["module_sensor"] == "iceye" and scene["testbed"] == "dcd"
    assert scene["input_container"] == "/data/05_l1_pre/umbra_pre/2026/20260501/Pre_DCD_UMBRA_20260501T093000.tif"
    assert scene["weights_container"] == "/model/WBMS_SAR_ICEYE.h5" and scene["mask_container"] == "/mask3/dcd_20260501_mask3_utm.tif"
    argv = adapter.build_command(scene)
    assert _arg(argv, "--sensor") == "iceye" and _arg(argv, "--testbed") == "dcd" and _arg(argv, "--image_date") == "20260501"
    assert _arg(argv, "--weights") == "/model/WBMS_SAR_ICEYE.h5" and _arg(argv, "--mask") == "/mask3/dcd_20260501_mask3_utm.tif"
    # 잡 요청 에코는 카탈로그 키 umbra(대소문자 무시), 컨테이너 인자만 iceye; iceye 는 여전히 busan 전용
    job = adapter.submit({"sensor": "UMBRA", "testbed": "dcd", "image_date": "20260501", "mask_tif": str(mask3)}, start_thread=False)
    assert job["request"]["sensor"] == "umbra" and job["request"]["testbed"] == "dcd" and _arg(job["command"], "--sensor") == "iceye"
    assert adapter.resolve_scene("iceye", "busan", SCENE_DATE)["module_sensor"] == "iceye"
    with pytest.raises(WbmsInputError, match="no delivered scenes"):
        adapter.resolve_scene("umbra", "busan", "20260501")


# ── GPU 실행 실패 → CPU 재시도 (2026-10-20, 현장 Blackwell GPU 대비) ─────────────────────────


def test_cpu_fallback_command_strips_gpu_flags_and_switches_device():
    from backend.app.adapters.wbms_segmentation import cpu_fallback_command

    gpu_argv = ["docker", "run", "--rm", "--gpus", "device=0", "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true", "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async",
                "--user", "1000:1000", "-e", "HOME=/tmp", "-v", "/a:/aux:ro", "wbms:0.13", "python3", "/WBMS/wbms_modules/detect_water.py", "--gpu", "gpu", "--output_dir", "/out"]
    cpu_argv = cpu_fallback_command(gpu_argv)
    assert "--gpus" not in cpu_argv and "device=0" not in cpu_argv and "TF_FORCE_GPU_ALLOW_GROWTH=true" not in cpu_argv and "TF_GPU_ALLOCATOR=cuda_malloc_async" not in cpu_argv
    assert cpu_argv[cpu_argv.index("--gpu") + 1] == "cpu" and "--user" in cpu_argv and "HOME=/tmp" in cpu_argv and "-v" in cpu_argv
    assert cpu_fallback_command(cpu_argv) == cpu_argv  # CPU 명령은 그대로


def test_segmentation_retries_on_cpu_when_gpu_run_fails_in_auto_mode(tmp_path: Path, monkeypatch) -> None:
    """현장 GPU(새 세대)에서 옛 CUDA 이미지가 런타임에 죽는 경우 — auto 모드면 CPU 로 한 번 더 돌려 결과를 낸다."""
    import dataclasses

    from backend.app.adapters import wbms_segmentation as seg
    from backend.app.wbms.runner import CommandResult

    client, runner, _settings = make_client(tmp_path)
    adapter = client.app.state.wbms_segmentation
    adapter.settings = dataclasses.replace(adapter.settings, gpu_mode="auto")
    monkeypatch.setattr(seg, "resolve_gpu_device", lambda mode, min_free: ("gpu", None))
    original = runner.run

    def run(argv: list[str], timeout_s: float) -> CommandResult:
        if "--gpus" in argv:
            runner.calls.append(list(argv))
            return CommandResult(returncode=1, stdout='{"type":"error","ts":"t","code":"CUDA_ERROR","param":"gpu","message":"no kernel image is available for execution on the device"}\n', stderr="cuDNN", elapsed_s=0.1)
        return original(argv, timeout_s)

    runner.run = run
    created = client.post("/api/v1/wbms/segmentation/jobs", json={"sensor": "iceye", "testbed": "busan", "image_date": SCENE_DATE}).json()
    job = wait_terminal(client, created["job_id"])
    assert job["status"] == "succeeded", job
    assert "--gpus" in runner.calls[0] and "--gpus" not in runner.calls[1] and _arg(runner.calls[1], "--gpu") == "cpu"
    assert job["result"]["gpu_fallback"]["note_ko"].startswith("GPU 실행이 실패") and "CUDA_ERROR" in job["result"]["gpu_fallback"]["reason"]
    assert job["command"][-1] == "/out" and "--gpus" not in job["command"]
