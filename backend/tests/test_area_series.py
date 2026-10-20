"""공통 격자 수체 면적 계열 — 씬별 촬영범위 면적과 섞이지 않는지, API 가 계약대로 주는지.

이 계열이 필요한 이유는 scenes.json 의 `water_area_km2` 가 **씬마다 자기 촬영범위에서** 센
값이기 때문이다. 촬영범위가 옮겨가면 물이 늘지 않아도 면적이 커진다(부산 광학 2020-03-25).
여기서는 (1) 파일 규약, (2) PNG 화소를 직접 세서 기록값과 맞는지, (3) API 노출·404 안전,
(4) 03-25 가 홍수가 아니라 촬영범위 이동임을 데이터로 재확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.gauge.regions import RegionRegistry
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[2]
REGIONS_DIR = ROOT / "regions"
SCHEMA = "watercast.satellite.area-series/1"
WITH_SCENES = ["nakdong_busan", "geum_daecheong"]

needs_regions = pytest.mark.skipif(
    not (REGIONS_DIR / "nakdong_busan" / "satellite" / "area_series.json").is_file(),
    reason="공통 격자 면적 계열 자산이 없다 — scripts/build_common_frame_area.py 를 먼저 돌릴 것",
)


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(result_dir=tmp_path / "results", regions_dir=REGIONS_DIR, regions_user_dir=tmp_path / "user")
    return TestClient(create_app(settings))


def _area_series(region_id: str) -> dict:
    return json.loads((REGIONS_DIR / region_id / "satellite" / "area_series.json").read_text(encoding="utf-8"))


@needs_regions
@pytest.mark.parametrize("region_id", WITH_SCENES)
def test_area_series_file_follows_schema(region_id: str) -> None:
    payload = _area_series(region_id)
    assert payload["schema"] == SCHEMA and payload["region_id"] == region_id
    assert payload["notes_ko"] and all(isinstance(n, str) for n in payload["notes_ko"])
    assert payload["sensors"], "센서가 하나도 없다"
    for sensor, block in payload["sensors"].items():
        frame = block["frame"]
        minx, miny, maxx, maxy = frame["bounds_epsg32652"]
        assert maxx > minx and maxy > miny
        assert frame["width"] > 0 and frame["height"] > 0 and frame["pixel_area_m2"] > 0
        # coverage_km2 는 격자 넓이와 맞아야 한다(화소 넓이 × 화소 수)
        assert frame["coverage_km2"] == pytest.approx(
            frame["width"] * frame["height"] * frame["pixel_area_m2"] / 1e6, abs=0.01)
        rows = block["rows"]
        assert len(rows) == block["n_scenes"] >= 1
        assert [r["date"] for r in rows] == sorted(r["date"] for r in rows), f"{sensor} 날짜 정렬"
        for row in rows:
            assert row["area_km2_common_frame"] > 0
            assert row["area_km2_scene_footprint"] > 0
            # 원계열은 기존 값 그대로여야 한다(덮어쓰기 금지)
            assert row["scene_footprint_km2"] > 0 and len(row["scene_footprint_bounds_epsg32652"]) == 4
            assert isinstance(row["gauge_levels"], dict)
            assert row["area_km2_common_frame"] <= frame["coverage_km2"]


@needs_regions
@pytest.mark.parametrize("region_id", WITH_SCENES)
def test_common_frame_area_recomputes_from_the_mask_png(region_id: str) -> None:
    """기록값은 공통 격자 마스크 PNG 의 물 화소 수 × 화소 넓이와 정확히 같아야 한다."""

    sat_dir = REGIONS_DIR / region_id / "satellite"
    scenes = {s["scene_id"]: s for s in json.loads((sat_dir / "scenes.json").read_text(encoding="utf-8"))["scenes"]}
    for block in _area_series(region_id)["sensors"].values():
        pixel_area = block["frame"]["pixel_area_m2"]
        for row in block["rows"]:
            png = sat_dir / scenes[row["scene_id"]]["frame"]["path"]
            water = np.asarray(Image.open(png).convert("L")) > 127
            assert float(water.sum()) * pixel_area / 1e6 == pytest.approx(row["area_km2_common_frame"], abs=5e-4)


@needs_regions
@pytest.mark.parametrize("region_id", WITH_SCENES)
def test_scenes_json_carries_the_comparable_area_without_losing_the_old_one(region_id: str) -> None:
    sat_dir = REGIONS_DIR / region_id / "satellite"
    scenes = json.loads((sat_dir / "scenes.json").read_text(encoding="utf-8"))["scenes"]
    by_id = {r["scene_id"]: r for block in _area_series(region_id)["sensors"].values() for r in block["rows"]}
    for scene in scenes:
        assert scene["water_area_km2"] > 0, "기존 씬별 면적을 지우면 안 된다"
        assert scene["area_common_frame_km2"] == pytest.approx(by_id[scene["scene_id"]]["area_km2_common_frame"])


@needs_regions
def test_busan_20200325_spike_is_a_footprint_shift_not_a_flood() -> None:
    """03-25 의 '면적 급증'은 촬영범위가 남쪽으로 옮겨간 결과다 — 공통 격자에서는 거의 안 변한다."""

    rows = {r["date"]: r for r in _area_series("nakdong_busan")["sensors"]["PlanetScope"]["rows"]}
    before, spike = rows["2020-03-12"], rows["2020-03-25"]
    # 원계열은 두 배 가까이 뛴다 (씬마다 다른 창으로 센 탓)
    assert spike["area_km2_scene_footprint"] / before["area_km2_scene_footprint"] > 1.9
    # 같은 창에서는 5 % 안쪽 변화다 — 홍수가 아니다
    change = spike["area_km2_common_frame"] / before["area_km2_common_frame"] - 1
    assert 0 < change < 0.05, change
    # 촬영범위가 실제로 남쪽으로 8 km 넘게 내려갔다
    southward_m = before["scene_footprint_bounds_epsg32652"][1] - spike["scene_footprint_bounds_epsg32652"][1]
    assert southward_m > 8000, southward_m
    # 같은 창에서 GT 라벨도 탐지와 일치한다 → 탐지가 아니라 계열이 문제였다
    assert spike["gt_area_km2"] == pytest.approx(spike["area_km2_common_frame"], rel=0.02)


@needs_regions
def test_daecheong_negative_level_correlation_is_a_coverage_artefact() -> None:
    """대청 '수위↑인데 면적↓'은 공통 격자 안 유효자료 비율 차이다 — 고정 창에서는 사라진다."""

    rows = sorted(_area_series("geum_daecheong")["sensors"]["PlanetScope"]["rows"], key=lambda r: r["date"])
    batch_2019 = [r for r in rows if r["date"] < "2021"]
    batch_2022 = [r for r in rows if r["date"] >= "2021"]
    assert batch_2019 and batch_2022
    # 2019 배치는 공통 격자를 부분적으로만 덮는다 — 그래서 면적이 구조적으로 작게 나온다
    assert max(r["frame_valid_fraction"] for r in batch_2019) < min(r["frame_valid_fraction"] for r in batch_2022)
    # 그 편향은 '모든 씬 유효영역 교집합'이라는 고정 창에서 사라진다
    fixed_2019 = [r["area_km2_common_valid_footprint"] for r in batch_2019]
    fixed_2022 = [r["area_km2_common_valid_footprint"] for r in batch_2022]
    assert min(fixed_2019) > 0 and min(fixed_2022) > 0
    overlap = min(max(fixed_2019), max(fixed_2022)) - max(min(fixed_2019), min(fixed_2022))
    assert overlap > 0, "고정 창에서는 두 배치의 면적 범위가 겹쳐야 한다(배치 편향이 사라진 증거)"
    # 대청댐 수위는 매 행에 붙어 있어야 한다(면적만 보고 수문 해석하지 못하게)
    assert all(r["gauge_levels"].get("대청댐") for r in rows)


@needs_regions
def test_api_serves_the_area_series_and_the_scene_field(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = client.get("/api/v1/regions/nakdong_busan/satellite/area-series").json()
    assert payload["available"] is True and payload["region_id"] == "nakdong_busan"
    assert payload["schema"] == SCHEMA
    assert set(payload["sensors"]) == {"ICEYE", "PlanetScope"}
    assert payload["notes_ko"]

    scenes = client.get("/api/v1/regions/nakdong_busan/satellite/scenes").json()
    assert scenes["area_series_available"] is True
    assert "공통" in scenes["area_note_ko"] and "satellite/area-series" in scenes["area_note_ko"]
    by_id = {r["scene_id"]: r for block in payload["sensors"].values() for r in block["rows"]}
    for scene in scenes["scenes"]:
        assert scene["area_common_frame_km2"] == pytest.approx(by_id[scene["scene_id"]]["area_km2_common_frame"])

    region = client.get("/api/v1/regions/nakdong_busan").json()
    assert region["area_series_available"] is True


def test_area_series_is_404_safe_and_null_when_absent(tmp_path: Path) -> None:
    """계열 파일이 없는 지역은 available=false·sensors=null 로 답하고, 없는 지역은 404."""

    region = tmp_path / "builtin" / "toy"
    (region / "gauge").mkdir(parents=True)
    (region / "region.json").write_text(json.dumps({
        "schema": "watercast.region/1", "id": "toy", "name": "토이", "crs": "EPSG:32652",
        "bbox_wgs84": [128.0, 35.0, 129.0, 36.0],
        "weather": {"provider": "KMA_ASOS", "station_id": "159"},
        "satellite": {"available": False},
        "gauge": {"available": True, "series_file": "gauge/series.json"},
        "forecast": {"horizon_max_days": 7},
    }, ensure_ascii=False), encoding="utf-8")
    (region / "gauge" / "series.json").write_text(json.dumps({
        "schema": "watercast.gauge.series/1", "region_id": "toy",
        "stations": [{"id": "A", "name": "A"}],
        "series": {"A": [{"date": "2023-01-01", "water_level_m": 1.0}]},
    }, ensure_ascii=False), encoding="utf-8")

    settings = Settings(result_dir=tmp_path / "results", regions_dir=tmp_path / "builtin",
                        regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    payload = client.get("/api/v1/regions/toy/satellite/area-series").json()
    assert payload == {"region_id": "toy", "available": False, "sensors": None, "period": {"start": None, "end": None},
                       "note_ko": payload["note_ko"]}
    assert "build_common_frame_area" in payload["note_ko"]
    assert client.get("/api/v1/regions/nope/satellite/area-series").status_code == 404

    record = RegionRegistry(tmp_path / "builtin", tmp_path / "user").get("toy")
    assert record.area_series() is None


# ── 생성 스크립트 (합성 지역으로 로직만 검사 — 원본 GeoTIFF 없이도 돌아야 한다) ──────────
def _load_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_common_frame_area", ROOT / "scripts" / "build_common_frame_area.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_region(root: Path, region_id: str = "toy") -> Path:
    """공통 격자 4×4, 두 날짜. 물 화소 수만 다르고 촬영범위는 크게 다르다."""

    sat = root / region_id / "satellite"
    (sat / "masks" / "S1").mkdir(parents=True)
    (root / region_id / "gauge").mkdir(parents=True)
    frame = {"path": "", "height": 4, "width": 4, "pixel_size_m": 1000.0,
             "pixel_area_m2": 1_000_000.0, "bounds_epsg32652": [0.0, 0.0, 4000.0, 4000.0]}
    scenes = []
    for date, water, footprint, raw in (
        ("2020-01-01", 4, [0.0, 0.0, 4000.0, 4000.0], 4.0),
        ("2020-02-01", 6, [0.0, -6000.0, 8000.0, 4000.0], 40.0),
    ):
        scene_id = f"{date.replace('-', '')}_S1"
        array = np.zeros((4, 4), np.uint8)
        array.flat[:water] = 255
        Image.fromarray(array).save(sat / "masks" / "S1" / f"{scene_id}.png")
        scenes.append({
            "scene_id": scene_id, "date": date, "sensor": "S1", "sensor_type": "SAR",
            "detection_available": True, "water_area_km2": raw, "valid_area_km2": 12.0,
            "frame": {**frame, "path": f"masks/S1/{scene_id}.png"},
            "quicklook": {"crs": "EPSG:32652", "bounds_epsg32652": footprint, "width": 8, "height": 8},
        })
    (sat / "scenes.json").write_text(json.dumps(
        {"schema": "watercast.satellite.scenes/1", "region_id": region_id, "scenes": scenes},
        ensure_ascii=False), encoding="utf-8")
    (root / region_id / "gauge" / "series.json").write_text(json.dumps({
        "schema": "watercast.gauge.series/1", "region_id": region_id,
        "stations": [{"id": "A", "name": "A"}],
        "series": {"A": [{"date": "2020-01-01", "water_level_m": 1.0},
                         {"date": "2020-02-01", "water_level_m": 1.5}]},
    }, ensure_ascii=False), encoding="utf-8")
    return sat


def test_builder_counts_png_pixels_and_keeps_the_old_number(tmp_path: Path, monkeypatch) -> None:
    mod = _load_builder()
    sat = _synthetic_region(tmp_path)
    monkeypatch.setattr(mod, "REGIONS", str(tmp_path))

    payload = mod.build_region("toy", verbose=False)
    block = payload["sensors"]["S1"]
    assert block["frame"]["coverage_km2"] == pytest.approx(16.0)
    rows = {r["date"]: r for r in block["rows"]}
    # 물 화소 4개·6개 × 1 km² = 4 / 6 km² (원계열 4 / 40 과 전혀 다르다)
    assert rows["2020-01-01"]["area_km2_common_frame"] == pytest.approx(4.0)
    assert rows["2020-02-01"]["area_km2_common_frame"] == pytest.approx(6.0)
    assert rows["2020-02-01"]["area_km2_scene_footprint"] == pytest.approx(40.0)
    assert rows["2020-02-01"]["scene_footprint_km2"] == pytest.approx(80.0)
    assert rows["2020-01-01"]["gauge_levels"] == {"A": 1.0}
    # 원본 GeoTIFF 가 없으면 유효자료·GT 항목은 조용히 null (면적은 그래도 나온다)
    assert rows["2020-01-01"]["frame_valid_fraction"] is None
    assert rows["2020-01-01"]["area_km2_common_valid_footprint"] is None
    assert rows["2020-01-01"]["gt_area_km2"] is None
    # 촬영범위와 함께 움직이는 원계열이 상관으로 드러난다
    assert block["correlations"]["raw_area_vs_footprint_km2"] is None  # n=2 는 상관을 내지 않는다

    scenes = json.loads((sat / "scenes.json").read_text(encoding="utf-8"))["scenes"]
    for scene in scenes:
        assert scene["water_area_km2"] in (4.0, 40.0), "기존 값을 덮어쓰면 안 된다"
        assert scene["area_common_frame_km2"] in (4.0, 6.0)


def test_builder_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    mod = _load_builder()
    sat = _synthetic_region(tmp_path)
    monkeypatch.setattr(mod, "REGIONS", str(tmp_path))

    mod.build_region("toy", verbose=False)
    first = (sat / "area_series.json").read_text(encoding="utf-8")
    first_scenes = (sat / "scenes.json").read_text(encoding="utf-8")
    mod.build_region("toy", verbose=False)
    second = (sat / "area_series.json").read_text(encoding="utf-8")

    def without_timestamp(text: str) -> dict:
        payload = json.loads(text)
        payload.pop("generated_at", None)
        return payload

    assert without_timestamp(first) == without_timestamp(second)
    assert (sat / "scenes.json").read_text(encoding="utf-8") == first_scenes


def test_builder_skips_a_region_without_scenes(tmp_path: Path, monkeypatch) -> None:
    mod = _load_builder()
    (tmp_path / "empty" / "satellite").mkdir(parents=True)
    monkeypatch.setattr(mod, "REGIONS", str(tmp_path))
    assert mod.build_region("empty", verbose=False) is None
    assert not (tmp_path / "empty" / "satellite" / "area_series.json").exists()


def test_pearson_needs_three_points_and_variation() -> None:
    mod = _load_builder()
    assert mod.pearson([1.0, 2.0], [1.0, 2.0]) is None
    assert mod.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert mod.pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert mod.pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)


# ── 정정 서사가 코드·자산에 실제로 반영됐는지 ────────────────────────────────
@needs_regions
def test_region_timeseries_eval_keeps_the_published_number_and_adds_the_correction() -> None:
    payload = json.loads(
        (REGIONS_DIR / "nakdong_busan" / "satellite" / "timeseries_eval.json").read_text(encoding="utf-8"))
    g = payload["series"]["G_iitp"]
    assert g["composite_norm_mse"] == 0.13526, "공표값을 조용히 덮어쓰면 안 된다"
    fix = g["correction"]
    assert fix["level_norm_mse"] == g["level_norm_mse"], "수위 항은 면적 계열과 무관하므로 그대로여야 한다"
    assert 0 < fix["composite_norm_mse"] < g["composite_norm_mse"] < fix["target"]
    caveat = g["caveat_ko"]
    assert "촬영범위" in caveat and "홍수가 아니라" in caveat
    assert "이벤트 예측이 ConvLSTM+기상 융합의 존재 이유" not in caveat


@pytest.mark.skipif(not (ROOT / "data" / "eval" / "timeseries_metrics_common_frame.json").is_file(),
                    reason="공통 격자 재계산 산출물이 없다")
def test_common_frame_metrics_reproduce_the_published_composite() -> None:
    payload = json.loads(
        (ROOT / "data" / "eval" / "timeseries_metrics_common_frame.json").read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["일치"] is True, "원본 계열 재현이 안 되면 두 변종을 비교할 근거가 없다"
    assert comparison["종합_정규화MSE"]["scene_footprint"] == payload["published_scene_footprint"]["종합_정규화MSE"]
    assert comparison["종합_정규화MSE"]["common_frame"] < comparison["종합_정규화MSE"]["scene_footprint"]
    assert comparison["수위변화_정규화MSE"]["scene_footprint"] == comparison["수위변화_정규화MSE"]["common_frame"]
    spike = payload["footprint_effect"]["spike_20200325"]
    assert spike["explained"]["share_from_footprint_pct"] > 90
    daecheong = payload["daecheong_negative_correlation"]
    assert abs(daecheong["correlations"]["level_vs_common_valid_area"]) < 0.5
    assert daecheong["correlations"]["common_frame_area_vs_frame_valid_fraction"] > 0.8


@pytest.mark.skipif(not (ROOT / "data" / "eval" / "timeseries_metrics_common_frame.json").is_file(),
                    reason="공통 격자 재계산 산출물이 없다")
def test_common_frame_variant_does_not_carry_the_scene_footprint_label() -> None:
    """'정정_참고' 는 씬 footprint 계열을 가리키는 문구다.

    eval_indicator 가 그 블록을 들고 있으므로, 같은 함수를 공통 격자 계열에 재사용하면
    정정 변종에 '이 값은 씬 footprint 계열이다' 라는 거짓 주석이 그대로 따라붙는다.
    """
    payload = json.loads(
        (ROOT / "data" / "eval" / "timeseries_metrics_common_frame.json").read_text(encoding="utf-8"))
    variants = payload["variants"]
    assert "정정_참고" not in variants["common_frame"]["지표"], "정정 변종에 씬 footprint 라벨이 붙어 있다"
    # 원본 변종에서는 그 문구가 맞는 말이므로 남아 있어야 한다
    assert variants["scene_footprint"]["지표"]["정정_참고"]["공통격자_종합_정규화MSE"] == (
        payload["comparison"]["종합_정규화MSE"]["common_frame"])


@pytest.mark.skipif(not (ROOT / "regions" / "nakdong_busan" / "models" / "gauge" / "experiments"
                         / "area_models.json").is_file(), reason="면적 모델 실험 산출물이 없다")
@pytest.mark.parametrize("region_id", ["nakdong_busan", "geum_daecheong"])
def test_area_models_notes_disclose_that_loo_is_not_a_forecast_evaluation(region_id: str) -> None:
    """LOO 후보는 뺀 날짜보다 나중 관측까지 써서 적합한다 — 예보 성적으로 읽히면 안 된다."""
    path = (ROOT / "regions" / region_id / "models" / "gauge" / "experiments" / "area_models.json")
    notes = " ".join(json.loads(path.read_text(encoding="utf-8"))["notes_ko"])
    assert "내삽" in notes and "예보" in notes, "LOO 가 내삽 평가라는 고지가 없다"


@pytest.mark.skipif(not (ROOT / "data" / "eval" / "IITP_성과지표_요약_PPT용.md").is_file(),
                    reason="IITP 요약 문서가 없다")
def test_iitp_summary_flags_the_contaminated_change_tracking_skill() -> None:
    """계열 F 의 '면적 변화 추적 skill 0.998' 도 같은 촬영범위 이동에 오염돼 있다.

    분모(무변화 MSE)가 449 (km²)² 로 부풀어 나온 값이라, 정정 없이 인용하면 안 된다.
    """
    text = (ROOT / "data" / "eval" / "IITP_성과지표_요약_PPT용.md").read_text(encoding="utf-8")
    head, _, tail = text.partition("0.86 / 0.998")
    assert tail, "면적 변화 추적 skill 줄이 사라졌다"
    assert "449" in tail[:1500] and "0.707" in tail[:1500], "0.998 옆에 정정 고지가 없다"


def test_hardcoded_common_frame_composite_matches_generated_file() -> None:
    """eval_timeseries_metrics.py 의 '정정_참고' 안 숫자가 실제 공통격자 산출물과 어긋나지 않게 지킨다."""
    import json
    import re
    from pathlib import Path

    generated = Path("data/eval/timeseries_metrics_common_frame.json")
    if not generated.is_file():
        import pytest as _pytest

        _pytest.skip("공통격자 산출물이 없다 — scripts/eval_timeseries_common_frame.py 미실행")
    payload = json.loads(generated.read_text(encoding="utf-8"))
    composite = payload["variants"]["common_frame"]["지표"]["종합_정규화MSE"]
    published = payload["variants"]["scene_footprint"]["지표"]["종합_정규화MSE"]
    assert abs(published - 0.13526) < 5e-6, "공표값 0.13526 이 더 이상 재현되지 않는다"
    text = Path("scripts/eval_timeseries_metrics.py").read_text(encoding="utf-8")
    hardcoded = [float(m) for m in re.findall(r"0\.0\d{4}", text)]
    assert any(abs(h - composite) < 5e-5 for h in hardcoded), (
        f"스크립트에 적힌 공통격자 종합값이 산출물 {composite} 과 다르다 (스크립트 내 후보 {hardcoded})"
    )


def test_user_copy_keeps_builtin_satellite_assets_visible(tmp_path: Path) -> None:
    """TTA 절차 7(시험구간 변경 재학습) 뒤에도 위성 장면·마스크가 계속 보여야 한다.

    재학습은 지역을 사용자 디렉터리로 복사해 내장 지역을 가린다. 사본에는 satellite/ 가
    복사되지 않으므로, 폴백이 없으면 절차 15 의 지표 산출이 '위성 장면이 없습니다' 로 멈춘다.
    """
    from backend.app.gauge.regions import RegionRegistry

    builtin = Path("regions")
    if not (builtin / "nakdong_busan" / "satellite" / "scenes.json").is_file():
        import pytest as _pytest

        _pytest.skip("내장 지역 자산이 없는 환경")
    registry = RegionRegistry(builtin, tmp_path / "user")
    before = registry.get("nakdong_busan")
    n_scenes = len((before.scenes() or {}).get("scenes") or [])
    assert n_scenes > 0 and before.origin == "builtin"

    registry._materialize_user_copy(before)  # 절차 7 과 동일한 그림자화
    after = registry.get("nakdong_busan")
    assert after.origin == "user", "사용자 사본이 생겨야 하는 전제"
    assert len((after.scenes() or {}).get("scenes") or []) == n_scenes, "장면이 사라지면 절차 15 가 중단된다"
    assert after.satellite_levels(), "위성 산출 수위도 계속 보여야 한다"
    assert after.area_series() is not None, "공통 격자 면적 계열도 계속 보여야 한다"
    assert after.detection_eval() is not None
    assert (builtin / "nakdong_busan" / "satellite" / "scenes.json").is_file(), "내장 원본은 건드리지 않는다"
