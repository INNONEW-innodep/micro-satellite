"""지상 수위계 시계열 예측 계층 — 예측기·학습 규율·지역 레지스트리·API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.gauge.forecasters import (
    DailySeries,
    build_forecaster_registry,
)
from backend.app.gauge.regions import RegionError, RegionRegistry, parse_series_rows
from backend.app.gauge.service import GaugeForecastService
from backend.app.gauge.training import Split, TrainingError, train_region, train_station
from backend.app.main import create_app


# 합성 시계열: 계절 순환 + AR(1) 잡음 + 가끔 급상승 → 예측기가 이길 여지가 있는 신호.
def make_series(days: int = 365 * 6, seed: int = 7, missing: int = 12) -> list[dict]:
    rng = np.random.default_rng(seed)
    start = np.datetime64("2018-01-01", "D")
    dates = start + np.arange(days).astype("timedelta64[D]")
    doy = (dates - dates.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    season = 0.4 * np.sin(2 * np.pi * doy / 365.25)
    noise = np.zeros(days)
    for t in range(1, days):
        noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.05)
    spikes = np.zeros(days)
    for t in rng.choice(days, size=days // 120, replace=False):
        spikes[t : t + 10] += np.linspace(0.8, 0.0, 10)[: max(0, min(10, days - t))]
    y = 2.0 + season + noise + spikes
    rows = [{"date": str(d), "water_level_m": round(float(v), 4)} for d, v in zip(dates, y)]
    for index in rng.choice(days, size=missing, replace=False):
        rows[index] = {"date": rows[index]["date"], "water_level_m": None}
    return [r for r in rows if r["water_level_m"] is not None]


SPLIT = {
    "train_end": "2020-12-31",
    "val_start": "2021-01-01",
    "val_end": "2022-12-31",
    "test_start": "2023-01-01",
    "test_end": "2023-12-31",
}


# 토이 지역의 위성 산출 위치(AOI) — 실제 구포대교 AOI 와 같은 구조·좌표계(EPSG:32652)
TOY_LOCATION = {
    "name_ko": "구포대교 부근 본류 하도(64 m)", "gauge_station": "A", "obscd": "2022680", "river": "낙동강", "crs": "EPSG:32652",
    "bbox_epsg32652": [499186.885, 3895504.01, 499486.885, 3895804.01],
    "bbox_lonlat": [[128.991067, 35.205242], [128.994363, 35.205242], [128.994363, 35.202537], [128.991067, 35.202537]],
    "center_lonlat": [128.992715, 35.203889], "anchor_lonlat": [128.993333, 35.203611], "snap_distance_m": 63.8,
    "note_ko": "OK", "source": "aoi_busan.geojson",
}


def write_region(root: Path, region_id: str = "toy", stations: dict[str, list[dict]] | None = None) -> Path:
    stations = stations or {"A": make_series(), "B": make_series(seed=11)}
    region = root / region_id
    (region / "gauge").mkdir(parents=True)
    (region / "region.json").write_text(
        json.dumps(
            {
                "schema": "watercast.region/1",
                "id": region_id,
                "name": "장난감 지역",
                "satellite": {"available": False, "locations": {"gupo": TOY_LOCATION}},
                "gauge": {"available": True, "series_file": "gauge/series.json", "satellite_location_links": {"A": "gupo"}},
                "forecast": {"artifacts_dir": "models/gauge", "horizons_report": [1, 7], "horizon_max": 7, "split": SPLIT},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (region / "satellite").mkdir(exist_ok=True)
    (region / "satellite" / "levels.json").write_text(
        json.dumps({"schema": "watercast.satellite.levels/1", "records": [
            {"date": "2023-06-01", "loc_id": "gupo", "satellite": "ICEYE", "water_level_m": 0.1, "corrected_water_level_m": 2.30},
            {"date": "2023-06-02", "loc_id": "other", "satellite": "ICEYE", "water_level_m": 0.2, "corrected_water_level_m": 9.0},
        ]}),
        encoding="utf-8",
    )
    (region / "gauge" / "series.json").write_text(
        json.dumps(
            {
                "schema": "watercast.gauge.series/1",
                "unit": "m",
                "stations": [
                    {"id": k, "name": k, "terrain": "toy", "lat": 35.2036 if k == "A" else None, "lon": 128.9933 if k == "A" else None,
                     "address": "부산 북구" if k == "A" else None, "river": "낙동강" if k == "A" else None}
                    for k in stations
                ],
                "series": stations,
            }
        ),
        encoding="utf-8",
    )
    return region


# --- DailySeries ---------------------------------------------------------------------


def test_daily_series_fills_gaps_but_keeps_truth_nan() -> None:
    rows = [{"date": "2020-01-01", "water_level_m": 1.0}, {"date": "2020-01-04", "water_level_m": 4.0}]
    series = DailySeries.from_rows(rows)
    assert series.n == 4
    assert np.isnan(series.y[1]) and np.isnan(series.y[2])
    assert series.filled.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert series.valid.sum() == 2


def test_daily_series_truncated_uses_only_past() -> None:
    series = DailySeries.from_rows(make_series(days=100))
    cut = series.truncated(np.datetime64("2018-02-01", "D"))
    assert str(cut.dates[-1]) == "2018-02-01"
    assert cut.n == 32


# --- 예측기 ---------------------------------------------------------------------------


@pytest.mark.parametrize("forecaster_id", ["ridge_ar", "seasonal_anomaly_ar", "mean_reversion", "analog_knn", "ensemble_mean", "climatology"])
def test_each_forecaster_fits_and_predicts_finite(forecaster_id: str) -> None:
    registry = build_forecaster_registry()
    forecaster = registry.get(forecaster_id)
    series = DailySeries.from_rows(make_series(days=365 * 4))
    origins = np.arange(400, series.n - 7)
    params = forecaster.fit(series, origins, 7, dict(forecaster.info.hyperparameter_grid[0]))
    # JSON 직렬화 가능해야 산출물로 저장된다
    json.dumps(params)
    pred = forecaster.predict(series, np.array([series.n - 8, series.n - 20]), 7, params)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_persistence_is_baseline_and_excluded_from_candidates() -> None:
    registry = build_forecaster_registry()
    ids = {f.info.id for f in registry.candidates()}
    assert "persistence" not in ids
    assert {"ridge_ar", "seasonal_anomaly_ar", "mean_reversion", "analog_knn", "ensemble_mean", "climatology"} <= ids
    assert {f.info.id for f in registry.references()} == {"persistence", "climatology"}


def test_plugin_loading_accepts_custom_forecaster(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = tmp_path / "my_forecaster.py"
    module.write_text(
        '''
import numpy as np
from backend.app.gauge.forecasters import GaugeForecaster, ForecasterInfo

class Constant(GaugeForecaster):
    info = ForecasterInfo(id="constant_zero", name="상수", description="테스트")
    def fit(self, series, origins, horizon, hparams):
        return {"c": float(np.nanmean(series.y[origins]))}
    def predict(self, series, origins, horizon, params):
        return np.full(len(origins), params["c"])
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = build_forecaster_registry("my_forecaster:Constant")
    assert registry.load_errors == []
    assert registry.get("constant_zero").info.name == "상수"
    bad = build_forecaster_registry("nonexistent.module:Nope")
    assert bad.load_errors and "nonexistent.module" in bad.load_errors[0]


# --- 학습 규율 --------------------------------------------------------------------------


def test_train_station_selects_non_persistence_and_reports_test_split() -> None:
    registry = build_forecaster_registry()
    series = DailySeries.from_rows(make_series())
    split = Split.from_mapping(SPLIT)
    artifact = train_station(series, split, registry, horizon_max=3, report_horizons=[1, 3])
    for h in ("1", "2", "3"):
        block = artifact["horizons"][h]
        assert block["forecaster_id"] != "persistence"
        assert block["test"]["n"] > 300  # 2023년 전체
        assert "persistence" in block["test_baselines"]
        assert block["n_val_folds"] == 2  # 2021, 2022
        assert block["skill_vs_persistence"] is not None
        selected = [c for c in block["candidates"] if c.get("selected")]
        assert len(selected) == 1 and selected[0]["forecaster_id"] == block["forecaster_id"]
        assert "params_eval" in block and "params_deploy" in block
    assert set(artifact["backtest"]) == {"1", "3"}
    assert len(artifact["backtest"]["1"]["dates"]) == artifact["horizons"]["1"]["test"]["n"]
    assert artifact["diagnostics"]["missing_days"] >= 0
    json.dumps(artifact)  # NaN 없이 직렬화


def test_selection_uses_only_validation_never_test() -> None:
    """시험 연도를 바꿔도(다른 자료로 교체) 선택 결과(후보 순위)는 같아야 한다."""

    registry = build_forecaster_registry()
    rows = make_series()
    series = DailySeries.from_rows(rows)
    split = Split.from_mapping(SPLIT)
    art_a = train_station(series, split, registry, horizon_max=1, report_horizons=[1], allowed_forecasters=["ridge_ar", "mean_reversion"])
    # 시험 연도 값을 뒤집어도 검증 기반 선택은 동일
    flipped = [dict(r, water_level_m=(6.0 - r["water_level_m"]) if r["date"] >= "2023-01-01" else r["water_level_m"]) for r in rows]
    art_b = train_station(DailySeries.from_rows(flipped), split, registry, horizon_max=1, report_horizons=[1], allowed_forecasters=["ridge_ar", "mean_reversion"])
    assert art_a["horizons"]["1"]["forecaster_id"] == art_b["horizons"]["1"]["forecaster_id"]
    assert art_a["horizons"]["1"]["hparams"] == art_b["horizons"]["1"]["hparams"]
    assert art_a["horizons"]["1"]["val"]["mse"] == pytest.approx(art_b["horizons"]["1"]["val"]["mse"])
    assert art_a["horizons"]["1"]["test"]["mse"] != pytest.approx(art_b["horizons"]["1"]["test"]["mse"])


def test_persistence_only_training_is_refused() -> None:
    registry = build_forecaster_registry()
    series = DailySeries.from_rows(make_series())
    with pytest.raises(TrainingError):
        train_station(series, Split.from_mapping(SPLIT), registry, horizon_max=1, allowed_forecasters=["persistence"])


# --- 지역 레지스트리 ----------------------------------------------------------------------


def test_parse_series_rows_accepts_csv_and_aliases() -> None:
    rows = parse_series_rows("date,wl\n20200101,1.5\n2020-01-02,\n2020-01-03,1.7\n")
    assert rows == [{"date": "2020-01-01", "water_level_m": 1.5}, {"date": "2020-01-03", "water_level_m": 1.7}]
    with pytest.raises(RegionError):
        parse_series_rows("date,wl\nnot-a-date,1\n")


def test_region_registry_user_overlay_and_upload(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    write_region(builtin)
    registry = RegionRegistry(builtin, user)
    assert [r.id for r in registry.list()] == ["toy"]
    assert registry.get("toy").origin == "builtin"
    loaded = registry.load_series("toy")
    assert {s.id for s in loaded.stations} == {"A", "B"}
    assert loaded.station("A").satellite_location == "gupo"

    # 내장 지역에 지점을 올리면 사용자 디렉터리에 사본이 생기고 그쪽이 우선한다
    station = registry.upsert_station_series("toy", "C", make_series(days=800, seed=3), station_meta={"terrain": "mainstem"})
    assert station.n_days > 700
    assert registry.get("toy").origin == "user"
    assert (user / "toy" / "gauge" / "series.json").is_file()
    assert (builtin / "toy" / "gauge" / "series.json").read_text(encoding="utf-8").count('"C"') == 0
    assert {s.id for s in registry.load_series("toy").stations} == {"A", "B", "C"}

    created = registry.create_region({"id": "new_river", "name": "새 하천"})
    assert created.origin == "user" and not created.gauge_available
    with pytest.raises(RegionError):
        registry.create_region({"id": "../evil", "name": "x"})


def test_train_region_writes_artifacts_and_evaluation(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    write_region(builtin)
    registry = RegionRegistry(builtin, tmp_path / "user")
    forecasters = build_forecaster_registry()
    evaluation = train_region(registry, forecasters, "toy", stations=["A"], allowed_forecasters=["ridge_ar", "mean_reversion"])
    assert evaluation["n_stations"] == 1
    assert {row["horizon"] for row in evaluation["rows"]} == {1, 7}
    assert all(row["forecaster_id"] != "persistence" for row in evaluation["rows"])
    write_dir = tmp_path / "user" / "toy" / "models" / "gauge"
    assert (write_dir / "A.json").is_file() and (write_dir / "evaluation.json").is_file()

    service = GaugeForecastService(registry, forecasters)
    forecast = service.forecast("toy", "A", as_of=None, horizon=7)
    assert len(forecast["steps"]) == 7
    assert forecast["params_used"] == "params_deploy"
    assert all(step["forecaster_id"] != "persistence" for step in forecast["steps"])
    # 시험 연도 안의 원점 → 시험을 보지 않은 파라미터 + 실측 대조
    past = service.forecast("toy", "A", as_of=__import__("datetime").date(2023, 6, 1), horizon=7)
    assert past["params_used"] == "params_eval"
    assert past["verification"]["n"] == 7
    assert past["steps"][0]["observed_m"] is not None
    backtest = service.backtest("toy", "A", 7)
    assert len(backtest["trace"]["dates"]) == backtest["test"]["n"]


# --- API ---------------------------------------------------------------------------------


PLUGIN_SPECS = (
    "backend.app.gauge.plugins.neural:NeuralMLP,backend.app.gauge.plugins.neural:NeuralLSTM,"
    "backend.app.gauge.plugins.neural:NeuralGRU,backend.app.gauge.plugins.neural:NeuralTCN,"
    "backend.app.gauge.plugins.prophet_hybrid:ProphetHybrid"
)


def _client(tmp_path: Path) -> TestClient:
    builtin = tmp_path / "builtin"
    write_region(builtin)
    settings = Settings(
        result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user",
        gauge_forecaster_plugins=PLUGIN_SPECS,  # 서버 compose 와 같은 등록 상태
    )
    return TestClient(create_app(settings))


def test_gauge_api_end_to_end(tmp_path: Path) -> None:
    client = _client(tmp_path)
    forecasters = client.get("/api/v1/gauge/forecasters").json()
    assert any(f["id"] == "persistence" and f["is_baseline"] for f in forecasters)
    regions = client.get("/api/v1/regions").json()
    assert regions["total"] == 1 and regions["items"][0]["gauge_available"] is True
    assert regions["items"][0]["n_stations_trained"] == 0

    assert client.get("/api/v1/regions/toy/gauge/evaluation").status_code == 404
    refused = client.post("/api/v1/regions/toy/gauge/train", json={"forecasters": ["persistence"]})
    assert refused.status_code == 422

    trained = client.post(
        "/api/v1/regions/toy/gauge/train",
        json={"stations": ["A"], "forecasters": ["mean_reversion", "climatology"], "horizon_max": 3},
    )
    assert trained.status_code == 200, trained.text
    payload = trained.json()
    assert payload["trained_now"] == ["A"]
    assert all(row["forecaster_id"] in {"mean_reversion", "climatology"} for row in payload["rows"])

    evaluation = client.get("/api/v1/regions/toy/gauge/evaluation").json()
    assert evaluation["n_stations"] == 1
    series = client.get("/api/v1/regions/toy/gauge/A/series", params={"start": "2023-12-01"}).json()
    assert series["n"] <= 31 and series["rows"][0]["date"] >= "2023-12-01"
    forecast = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": "2023-03-01", "horizon": 3}).json()
    assert forecast["params_used"] == "params_eval" and len(forecast["steps"]) == 3
    assert forecast["verification"]["n"] == 3
    backtest = client.get("/api/v1/regions/toy/gauge/A/backtest", params={"horizon": 1}).json()
    assert backtest["forecaster_id"] != "persistence"
    assert client.get("/api/v1/regions/toy/gauge/A/backtest", params={"horizon": 2}).status_code == 404
    model = client.get("/api/v1/regions/toy/gauge/A/model").json()
    assert "params_eval" not in model["horizons"]["1"]
    assert client.get("/api/v1/regions/toy/gauge/B/forecast").status_code == 404
    assert client.get("/api/v1/regions/nope").status_code == 404

    # 데이터 입력 구멍: JSON 업로드 → 지점 추가 → 학습
    upload = client.post(
        "/api/v1/regions/toy/gauge/C/series",
        json={"rows": [{"date": r["date"], "level": r["water_level_m"]} for r in make_series(days=1500, seed=5)], "station": {"terrain": "tributary"}},
    )
    assert upload.status_code == 200, upload.text
    assert upload.json()["terrain"] == "tributary"
    created = client.post("/api/v1/regions", json={"id": "geum", "name": "금강"})
    assert created.status_code == 201 and created.json()["origin"] == "user"
    assert client.post("/api/v1/regions", json={"id": "geum", "name": "금강"}).status_code == 422
    csv_upload = client.post(
        "/api/v1/regions/geum/gauge/공주/series/file",
        files={"file": ("levels.csv", b"date,water_level_m\n2020-01-01,1.0\n2020-01-02,1.1\n", "text/csv")},
        data={"terrain": "mainstem"},
    )
    assert csv_upload.status_code == 200, csv_upload.text
    assert csv_upload.json()["n_days"] == 2


def test_satellite_levels_join_gauge_same_day(tmp_path: Path) -> None:
    client = _client(tmp_path)
    region = client.get("/api/v1/regions/toy").json()
    assert region["satellite_levels_available"] is True
    payload = client.get("/api/v1/regions/toy/satellite/levels", params={"station_id": "A"}).json()
    assert payload["loc_id"] == "gupo" and payload["n"] == 1
    assert payload["location"]["name_ko"] == TOY_LOCATION["name_ko"] and payload["location"]["bbox_lonlat"] == TOY_LOCATION["bbox_lonlat"]
    assert payload["location"]["gauge_station"] == "A"
    record = payload["records"][0]
    assert record["gauge_level_m"] is not None
    assert record["corrected_minus_gauge_m"] == pytest.approx(2.30 - record["gauge_level_m"])
    assert payload["comparison"]["n"] == 1
    unlinked = client.get("/api/v1/regions/toy/satellite/levels", params={"station_id": "B"}).json()
    assert unlinked["n"] == 0 and unlinked["loc_id"] is None and unlinked["location"] is None
    # 지역 요약도 같은 위치 사전과 지점 좌표를 준다(지도·퀵룩 위에 그리는 입력)
    assert region["satellite_locations"] == {"gupo": TOY_LOCATION}
    by_id = {s["id"]: s for s in region["stations"]}
    assert by_id["A"]["lat"] == pytest.approx(35.2036) and by_id["A"]["lon"] == pytest.approx(128.9933)
    assert by_id["A"]["address"] == "부산 북구" and by_id["A"]["river"] == "낙동강"
    assert by_id["B"]["lat"] is None and by_id["B"]["lon"] is None and by_id["B"]["address"] is None
    everything = client.get("/api/v1/regions/toy/satellite/levels").json()
    assert everything["n"] == 2


REAL_REGIONS_DIR = Path(__file__).resolve().parents[2] / "regions"


@pytest.mark.skipif(not (REAL_REGIONS_DIR / "nakdong_busan" / "region.json").is_file(), reason="저장소 내장 지역이 없다")
def test_builtin_busan_region_exposes_geo_for_map_overlay(tmp_path: Path) -> None:
    """내장 낙동강 하구 지역(읽기 전용): AOI 위치·지점 좌표·퀵룩 지리참조가 모두 한국 안에 있고 서로 맞물린다."""

    settings = Settings(result_dir=tmp_path / "results", regions_dir=REAL_REGIONS_DIR, regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    region = client.get("/api/v1/regions/nakdong_busan").json()
    locations = region["satellite_locations"]
    assert set(locations) == {"gupo", "gimhae", "jeongcheon", "hupo"}
    for loc_id, item in locations.items():
        assert item["crs"] == "EPSG:32652" and len(item["bbox_epsg32652"]) == 4 and len(item["bbox_lonlat"]) == 4
        assert all(124 <= lon <= 132 and 33 <= lat <= 39 for lon, lat in item["bbox_lonlat"]), loc_id
        assert 128.8 <= item["center_lonlat"][0] <= 129.1 and 35.0 <= item["center_lonlat"][1] <= 35.4, loc_id
        assert item["name_ko"] and item["source"] == "aoi_busan.geojson"
    assert locations["gupo"]["gauge_station"] == "구포대교" and locations["jeongcheon"]["gauge_station"] == "정천교"
    assert locations["hupo"]["gauge_station"] == "호포대교"  # WAMIS 2022670 = AOI obscd — 위성 보정 수위와 1 cm 안에서 일치
    assert locations["gimhae"]["gauge_station"] is None  # 공개 일 수위계가 없는 위치는 지점 없이 남는다
    stations = {s["id"]: s for s in region["stations"]}
    gupo = stations["구포대교"]
    assert gupo["lat"] == pytest.approx(35.203611, abs=1e-5) and gupo["lon"] == pytest.approx(128.993333, abs=1e-5)
    assert gupo["river"] == "낙동강" and "구포" in gupo["address"]
    assert all(124 <= s["lon"] <= 132 and 33 <= s["lat"] <= 39 for s in stations.values())
    # AOI 는 그 지점에서 옮겨 온 것 — 앵커가 지점 좌표와 같고 상자 중심이 앵커에서 수백 m 안
    assert locations["gupo"]["anchor_lonlat"] == pytest.approx([gupo["lon"], gupo["lat"]], abs=1e-5)
    levels = client.get("/api/v1/regions/nakdong_busan/satellite/levels", params={"station_id": "구포대교"}).json()
    assert levels["loc_id"] == "gupo" and levels["location"] == locations["gupo"]
    scenes = client.get("/api/v1/regions/nakdong_busan/satellite/scenes").json()
    assert scenes["n_scenes"] == 8
    for scene in scenes["scenes"]:
        quicklook = scene["quicklook"]
        assert quicklook["crs"] == "EPSG:32652" and quicklook["width"] > 0 and quicklook["height"] > 0
        assert max(quicklook["width"], quicklook["height"]) == 1000
        minx, miny, maxx, maxy = quicklook["bounds_epsg32652"]
        gx0, gy0, gx1, gy1 = locations["gupo"]["bbox_epsg32652"]
        assert minx <= gx0 and gx1 <= maxx and miny <= gy0 and gy1 <= maxy, scene["scene_id"]  # 구포 AOI 가 장면 안에 든다
        assert all(124 <= lon <= 132 and 33 <= lat <= 39 for lon, lat in quicklook["footprint_lonlat"])
    assert not (tmp_path / "user").exists() or not any((tmp_path / "user").iterdir())  # 읽기만 했다


def test_segmenter_registry_lists_wbms_and_catalog(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = client.get("/api/v1/segmenters").json()
    assert payload["total"] == 1
    wbms = payload["items"][0]
    assert wbms["id"] == "wbms-detect-water" and wbms["execution"] == "docker"
    assert set(wbms["capabilities"]["sensors"]) == {"iceye", "planet", "umbra"}
    umbra = wbms["capabilities"]["sensors"]["umbra"]
    assert umbra["module_sensor"] == "iceye" and umbra["file_label"] == "UMBRA" and umbra["weights"] == "WBMS_SAR_ICEYE.h5"
    assert "busan" in wbms["capabilities"]["testbeds"]
    detail = client.get("/api/v1/segmenters/wbms-detect-water").json()
    assert "cached_scenes" in detail
    assert client.get("/api/v1/segmenters/nope").status_code == 404
    # 카탈로그에 없는 센서는 422 (Literal 열거 대신 데이터 검증)
    bad = client.post("/api/v1/wbms/segmentation/jobs", json={"sensor": "kompsat5", "testbed": "busan", "image_date": "20200302"})
    assert bad.status_code == 422 and "catalog" in bad.json()["detail"]


def test_custom_segmenter_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = tmp_path / "my_segmenter.py"
    module.write_text(
        '''
from backend.app.segmentation.base import SegmentationAdapter, SegmenterInfo, SegmentationSubmission

class Dummy(SegmentationAdapter):
    info = SegmenterInfo(id="dummy-seg", name="더미", description="테스트", execution="in-process")
    def capabilities(self):
        return {"sensors": {"sentinel1": {"testbeds": ["anywhere"]}}, "testbeds": {"anywhere": {}}}
    def supports(self, sensor, testbed):
        return sensor == "sentinel1"
    def submit(self, request):
        return SegmentationSubmission(job_id="j1", status="succeeded", result={"mask": "x"})
    def get_job(self, job_id):
        return {"job_id": job_id}
    def list_cached(self):
        return []
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    builtin = tmp_path / "builtin"
    write_region(builtin)
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user", segmenter_plugins="my_segmenter:Dummy")
    client = TestClient(create_app(settings))
    payload = client.get("/api/v1/segmenters").json()
    assert {item["id"] for item in payload["items"]} == {"dummy-seg", "wbms-detect-water"}
    assert payload["plugin_errors"] == []
    registry = client.app.state.segmenters
    assert registry.resolve("sentinel1", "anywhere").info.id == "dummy-seg"
    assert registry.resolve("iceye", "busan").info.id == "wbms-detect-water"
    assert registry.resolve("iceye", "mars") is None


def test_legacy_prediction_api_can_return_varying_levels(tmp_path: Path) -> None:
    """외부 모듈이 쓰는 POST /predictions 에서 persistence 대신 학습 수위 모델을 고를 수 있어야 한다."""

    from io import BytesIO

    from PIL import Image

    client = _client(tmp_path)
    client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["A"], "forecasters": ["mean_reversion"], "horizon_max": 5})
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    stream = BytesIO()
    Image.fromarray(mask * 255).save(stream, format="PNG")
    png = stream.getvalue()
    form = {
        "model_id": "gauge-level-forecast",
        "horizon_steps": "5",
        "threshold": "0.5",
        "pixel_area_m2": "9",
        "source_dates_json": '["2023-06-01","2023-06-02"]',
        "historical_water_levels_json": "[null,null]",
        "model_options_json": '{"region_id":"toy","station_id":"A"}',
        "weather_json": "[]",
    }
    response = client.post(
        "/api/v1/predictions",
        files=[("files", ("m1.png", png, "image/png")), ("files", ("m2.png", png, "image/png"))],
        data=form,
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    levels = [step["water_level_m"] for step in payload["steps"]]
    assert all(isinstance(v, float) for v in levels)
    assert len(set(round(v, 6) for v in levels)) > 1, "날짜마다 다른 수위여야 한다 (persistence 가 아니다)"
    assert payload["steps"][0]["water_level_source"] == "adapter_output"
    assert payload["adapter_metadata"]["gauge_station_id"] == "A"
    assert payload["adapter_metadata"]["gauge_as_of"] == "2023-06-02"

    # 같은 입력을 persistence 로 보내면 여전히 상수 — 계약은 그대로다
    baseline = client.post(
        "/api/v1/predictions",
        files=[("files", ("m1.png", png, "image/png")), ("files", ("m2.png", png, "image/png"))],
        data={**form, "model_id": "persistence", "historical_water_levels_json": "[1.5,1.6]", "model_options_json": "{}"},
    )
    assert baseline.status_code == 201
    assert len({step["water_level_m"] for step in baseline.json()["steps"]}) == 1


def test_model_guide_and_timeseries_endpoints(tmp_path: Path) -> None:
    client = _client(tmp_path)
    guide = client.get("/api/v1/gauge/model-guide").json()
    assert "skill" in guide and "why_no_lstm_ko" in guide and len(guide["why_no_lstm_ko"]) >= 3
    ids = {f["id"] for f in guide["forecasters"]}
    assert ids <= set(guide["registered_ids"]) | {"persistence"}
    assert any(f.get("uses_weather") for f in guide["forecasters"])
    ts = client.get("/api/v1/regions/toy/satellite/timeseries").json()
    assert ts["available"] is False


def test_forecast_marks_in_sample_origins(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["A"], "forecasters": ["mean_reversion"], "horizon_max": 3})
    inside = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": "2021-06-01", "horizon": 3}).json()
    assert inside["in_sample"] is True and "in-sample" in inside["params_note_ko"]
    test_year = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": "2023-06-01", "horizon": 3}).json()
    assert test_year["in_sample"] is False and test_year["params_used"] == "params_eval"


def test_weather_exog_forecaster_available_when_region_has_weather(tmp_path: Path) -> None:
    from backend.app.gauge.forecasters import DailySeries, build_forecaster_registry

    rows = make_series(days=365 * 4)
    series = DailySeries.from_rows(rows)
    weather = [{"date": r["date"], "precipitation_mm": float(i % 9 == 0) * 12.0, "avg_temperature_c": 10.0} for i, r in enumerate(rows)]
    with_wx = series.with_exog(weather, {"precipitation_mm": "rain_mm", "avg_temperature_c": "temp_c"})
    assert with_wx.has_exog("rain_mm", "temp_c")
    arx = build_forecaster_registry().get("ridge_arx")
    origins = np.arange(400, with_wx.n - 3)
    params = arx.fit(with_wx, origins, 3, {"p": 7, "lam": 10.0, "mode": "delta"})
    pred = arx.predict(with_wx, np.array([with_wx.n - 5]), 3, params)
    assert np.isfinite(pred).all()
    with pytest.raises(ValueError):
        arx.fit(series, origins, 3, {"p": 7, "lam": 10.0, "mode": "delta"})  # 기상 없음 → 후보 제외
    cut = with_wx.truncated(np.datetime64("2019-06-01", "D"))
    assert cut.has_exog("rain_mm") and cut.exog["rain_mm"].shape[0] == cut.n


def test_deep_and_prophet_plugins_load_and_infer_without_torch(tmp_path: Path) -> None:
    """서버(torch·prophet 없음)에서도 플러그인이 등록되고, 공유 파라미터가 없으면 NaN 을 돌려 조용히 제외된다."""

    from backend.app.gauge.forecasters import build_forecaster_registry
    from backend.app.gauge.plugins.neural import NeuralForecaster

    specs = ",".join(
        f"backend.app.gauge.plugins.neural:{cls}" for cls in ("NeuralMLP", "NeuralLSTM", "NeuralGRU", "NeuralTCN")
    ) + ",backend.app.gauge.plugins.prophet_hybrid:ProphetHybrid"
    registry = build_forecaster_registry(specs)
    assert registry.load_errors == []
    ids = {f.info.id for f in registry.candidates()}
    assert {"neural_mlp", "neural_lstm", "neural_gru", "neural_tcn", "prophet_hybrid"} <= ids
    series = DailySeries.from_rows(make_series(days=800))
    mlp = registry.get("neural_mlp")
    pred = mlp.predict(series, np.array([700]), 3, {"shared_ref": "mlp:missing", "horizon": 3})
    assert np.isnan(pred).all()
    # 가짜 공유 가중치로 numpy 추론 경로를 통과시킨다 (형상만 맞춘 0 가중치 → 예측 = 현재 수위)
    window, hidden, seq_dim, static_dim = 30, 8, 1, 6
    zeros = {
        "fc1.weight": np.zeros((hidden, window * seq_dim + static_dim), np.float32), "fc1.bias": np.zeros(hidden, np.float32),
        "fc2.weight": np.zeros((hidden, hidden), np.float32), "fc2.bias": np.zeros(hidden, np.float32),
        "head.weight": np.zeros((30, hidden), np.float32), "head.bias": np.zeros(30, np.float32),
    }
    blob = {"mlp:test": {"arch": "mlp", "window": window, "hidden": hidden, "scale": 1.0, "seq_channels": seq_dim, "static_dim": static_dim,
                         "dilations": [], "weights": {k: NeuralForecaster._encode(v) for k, v in zeros.items()}}}
    mlp.import_shared(json.loads(json.dumps(blob)))
    pred = mlp.predict(series, np.array([700]), 3, {"shared_ref": "mlp:test", "horizon": 3})
    assert np.isfinite(pred).all() and pred[0] == pytest.approx(series.filled[700])


def test_forecast_compare_returns_every_family_and_differences(tmp_path: Path) -> None:
    """모델별 비교: 학습된 모든 계열의 전망·선택 모델 대비 차이·모델별 시험 지표를 한 응답으로."""
    client = _client(tmp_path)
    trained = client.post(
        "/api/v1/regions/toy/gauge/train",
        json={"stations": ["A"], "forecasters": ["mean_reversion", "ridge_ar", "climatology"], "horizon_max": 3},
    )
    assert trained.status_code == 200
    payload = client.get("/api/v1/regions/toy/gauge/A/forecast/compare", params={"as_of": "2023-06-01", "horizon": 3}).json()
    assert set(payload["models"]) >= {"mean_reversion", "ridge_ar", "climatology"}
    assert len(payload["table"]) == 3 and payload["table"][0]["horizon"] == 1
    row = payload["table"][-1]
    assert row["selected_model"] in payload["models"]
    for fid, cell in row["models"].items():
        assert fid in payload["models"]
        if cell["forecast_m"] is not None:
            assert abs(cell["diff_vs_selected_m"] - (cell["forecast_m"] - row["selected_m"])) < 1e-9
    # 기준선은 상수(직전값 유지) — 학습 모델과 달라야 한다
    persist = [r["persistence_m"] for r in payload["table"]]
    assert len(set(persist)) == 1
    learned = [r["models"]["mean_reversion"]["forecast_m"] for r in payload["table"]]
    assert len(set(learned)) > 1
    per_model = payload["per_model"]
    assert "mean_reversion" in per_model and "test_at_horizon" in per_model["mean_reversion"]
    test_metrics = per_model["mean_reversion"]["test_at_horizon"]
    assert test_metrics and {"rmse", "mse", "mae"} <= set(test_metrics)
    # 원점 이후 실측이 있으므로 검증 지표도 채워진다
    assert per_model["mean_reversion"]["verification"] is not None
    evaluation = client.get("/api/v1/regions/toy/gauge/evaluation").json()
    assert "model_summary" in evaluation and "mean_reversion" in evaluation["model_summary"]
    assert {"mean_rmse", "mean_mse", "mean_mae", "selected_count"} <= set(evaluation["model_summary"]["mean_reversion"]["1"])


def test_user_copy_of_builtin_region_keeps_weather_and_station_meta(tmp_path: Path) -> None:
    """내장 지역에 무엇이든 쓰면 사용자 사본이 생기는데, 기상 파일이 빠지면 기상 모델 예측이 422 로 막혔다(감사 발견)."""
    builtin = tmp_path / "builtin"
    write_region(builtin)
    region_dir = builtin / "toy"
    (region_dir / "weather").mkdir(parents=True, exist_ok=True)
    rows = make_series(days=365 * 4)
    weather = {"schema": "watercast.weather.daily/1", "rows": [{"date": r["date"], "precipitation_mm": 1.0, "avg_temperature_c": 10.0} for r in rows]}
    (region_dir / "weather" / "asos_daily.json").write_text(json.dumps(weather), encoding="utf-8")
    registry = RegionRegistry(builtin, tmp_path / "user")
    assert registry.load_weather_rows("toy")
    # 지점 이어 붙이기(메타 없이) → 사용자 사본 생성
    before = registry.load_series("toy").station("A")
    registry.upsert_station_series("toy", "A", [{"date": "2024-12-31", "water_level_m": 1.0}], replace=False)
    record = registry.get("toy")
    assert record.origin == "user"
    assert registry.load_weather_rows("toy"), "사용자 사본에서도 기상 자료가 보여야 한다"
    assert (record.path / "weather" / "asos_daily.json").is_file()
    after = registry.load_series("toy").station("A")
    assert after.name == before.name == "A" and after.terrain == before.terrain == "toy"
    assert after.source == "user_upload"  # 원래 비어 있던 출처만 기본값으로 채워진다


def test_quality_control_drops_flat_unit_error_and_isolated_spike_but_keeps_floods() -> None:
    from backend.app.gauge.regions import quality_control_rows

    rows = make_series(days=365 * 6, missing=0)
    base = {r["date"]: r["water_level_m"] for r in rows}
    dates = [r["date"] for r in rows]
    # 2020년 300일을 1/100 로 (단위 오류 흉내), 하루짜리 13 m 스파이크, 사흘 홍수 봉우리
    flat_dates = [d for d in dates if d.startswith("2020")][:300]
    for d in flat_dates:
        base[d] = base[d] / 100.0
    spike_date = dates[500]
    base[spike_date] = 13.79
    flood = dates[1500:1503]  # 2022년 — 정체 구간(2020) 밖
    base[flood[0]], base[flood[1]], base[flood[2]] = 3.5, 6.1, 4.0
    clean, flags = quality_control_rows([{"date": d, "water_level_m": v} for d, v in base.items()])
    kept = {r["date"] for r in clean}
    assert flags["removed_days"] == len(flat_dates) + 1
    assert not (set(flat_dates) & kept) and spike_date not in kept
    assert all(d in kept for d in flood), "홍수 봉우리는 남아야 한다"
    assert flags["flat_runs"] and flags["spikes"][0]["date"] == spike_date


# --- 위성 장면 업로드 (화면의 파일 업로드 → POST /regions/{id}/satellite/scenes) ------------------------


def _geotiff_bytes(tmp_path: Path, name: str, *, mask: bool = False, size: int = 64) -> bytes:
    """64×64 합성 GeoTIFF(EPSG:32652, 3 m 화소). mask=True 면 uint8 0/1 (왼쪽 절반이 수체)."""

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / name
    transform = from_origin(330_000.0, 3_910_000.0, 3.0, 3.0)
    if mask:
        data = np.zeros((size, size), dtype=np.uint8)
        data[:, : size // 2] = 1
        profile = {"dtype": "uint8", "nodata": 255}
    else:
        yy, xx = np.mgrid[0:size, 0:size]
        data = (-20.0 + 15.0 * (xx / size) + 2.0 * np.sin(yy / 5.0)).astype(np.float32)
        profile = {"dtype": "float32", "nodata": -9999.0}
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, crs="EPSG:32652", transform=transform, **profile) as ds:
        ds.write(data, 1)
    return path.read_bytes()


def test_satellite_scene_upload_creates_quicklook_and_lists_scene(tmp_path: Path) -> None:
    client = _client(tmp_path)
    before = client.get("/api/v1/regions/toy/satellite/scenes").json()
    assert before["available"] is False and before["scenes"] == []

    pre = _geotiff_bytes(tmp_path, "pre.tif")
    # 1) 사진만 (탐지 대기) — 파일명은 서버가 pre.tif 로 고정해 저장하므로 이상한 클라이언트 이름도 안전하다
    created = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "2024-05-01", "note": "업로드 시험", "run_detection": "true"},
        files={"pre_file": ("../../evil name.tif", pre, "image/tiff")},
    )
    assert created.status_code == 201, created.text
    payload = created.json()
    scene = payload["scene"]
    assert scene["scene_id"] == "20240501_ICEYE" and scene["detection_available"] is False
    assert scene["images"]["input"] == "quicklooks/20240501_ICEYE_input.png"
    # 퀵룩 지리참조: 원본 GeoTIFF 범위(330000, 3910000 좌상, 64×3 m) + 퀵룩 화소 크기 + 경위도 발자국
    quicklook_meta = scene["quicklook"]
    assert quicklook_meta["crs"] == "EPSG:32652" and quicklook_meta["source_crs"] == "EPSG:32652"
    assert quicklook_meta["bounds_epsg32652"] == pytest.approx([330_000.0, 3_910_000.0 - 64 * 3, 330_000.0 + 64 * 3, 3_910_000.0])
    assert quicklook_meta["width"] == 1000 and quicklook_meta["height"] == 1000  # 64×64 원본을 긴 변 1000 으로
    assert scene["crs"] == "EPSG:32652" and scene["size_px"] == [64, 64] and scene["pixel_size_m"] == [3.0, 3.0]
    footprint = quicklook_meta["footprint_lonlat"]
    assert len(footprint) == 4 and all(124 <= lon <= 132 and 33 <= lat <= 39 for lon, lat in footprint)
    assert footprint[0][1] > footprint[3][1] and footprint[1][0] > footprint[0][0]  # 좌상 → 우상 → 우하 → 좌하
    assert payload["detection_job"] is None and "탐지 어댑터" in payload["detection_note_ko"]  # 토이 지역은 카탈로그에 없다 → 우아한 실패
    upload_dir = tmp_path / "user" / "toy" / "uploads" / "20240501_ICEYE"
    assert sorted(p.name for p in upload_dir.iterdir()) == ["pre.tif"]
    assert not any("evil" in p.name for p in (tmp_path / "user").rglob("*"))

    listed = client.get("/api/v1/regions/toy/satellite/scenes").json()
    assert listed["available"] is True and listed["n_scenes"] == 1 and listed["n_with_masks"] == 0
    item = listed["scenes"][0]
    assert item["image_urls"]["input"].endswith("/satellite/files/quicklooks/20240501_ICEYE_input.png")
    assert item["quicklook"] == quicklook_meta  # 목록 API 도 같은 지리참조를 그대로 준다
    quicklook = client.get(item["image_urls"]["input"])
    assert quicklook.status_code == 200 and quicklook.content[:8] == b"\x89PNG\r\n\x1a\n"
    region = client.get("/api/v1/regions/toy").json()
    assert region["satellite_available"] is True and region["scenes_available"] is True and region["satellite"]["sensors"] == ["ICEYE"]

    # 2) 같은 센서·촬영일로 마스크를 붙여 다시 등록 → 갱신(장면 수 그대로), 면적·프레임 생성
    mask = _geotiff_bytes(tmp_path, "mask.tif", mask=True)
    updated = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", mask, "image/tiff")},
    )
    assert updated.status_code == 201, updated.text
    scene2 = updated.json()["scene"]
    assert scene2["detection_available"] is True
    assert scene2["water_area_km2"] == pytest.approx(64 * 32 * 9 / 1e6, abs=1e-4)  # scenes.json 은 소수 4자리
    assert scene2["frame"]["path"] == "masks/ICEYE/20240501_ICEYE.png"
    listed2 = client.get("/api/v1/regions/toy/satellite/scenes").json()
    assert listed2["n_scenes"] == 1 and listed2["n_with_masks"] == 1
    assert client.get(listed2["scenes"][0]["image_urls"]["overlay"]).status_code == 200


def test_satellite_scene_upload_rejects_bad_input_and_cleans_up(tmp_path: Path) -> None:
    client = _client(tmp_path)
    # GeoTIFF 가 아닌 업로드 → 422, 업로드 파일은 남지 않는다
    bad = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", b"not a tiff at all", "image/tiff")},
    )
    assert bad.status_code == 422 and "장면 자산 생성 실패" in bad.json()["detail"]
    assert not (tmp_path / "user" / "toy" / "uploads" / "20240501_ICEYE").exists()
    assert client.get("/api/v1/regions/toy/satellite/scenes").json()["available"] is False
    # 빈 파일 → 422
    empty = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", b"", "image/tiff")},
    )
    assert empty.status_code == 422
    # 센서 이름에 경로 문자가 섞이면 422 (uploads/<date>_<sensor> 디렉터리 이름이 되므로)
    traversal = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "../x", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", b"x", "image/tiff")},
    )
    assert traversal.status_code == 422 and "sensor" in traversal.json()["detail"]
    # 경로 없음 → 422 ; 서버 경로가 없으면 422
    assert client.post("/api/v1/regions/toy/satellite/scenes", data={"sensor": "ICEYE", "image_date": "20240501"}).status_code == 422
    missing = client.post("/api/v1/regions/toy/satellite/scenes", data={"sensor": "ICEYE", "image_date": "20240501", "pre_path": str(tmp_path / "nope.tif")})
    assert missing.status_code == 422


def test_satellite_scene_upload_rejects_calendar_invalid_date(tmp_path: Path) -> None:
    client = _client(tmp_path)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    for bad_date in ("20241301", "20240230", "2024-02-30"):
        res = client.post(
            "/api/v1/regions/toy/satellite/scenes",
            data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": bad_date},
            files={"pre_file": ("pre.tif", pre, "image/tiff")},
        )
        assert res.status_code == 422 and "image_date" in res.json()["detail"], (bad_date, res.text)
    assert client.get("/api/v1/regions/toy/satellite/scenes").json()["available"] is False
    assert not (tmp_path / "user" / "toy" / "uploads").exists()


def test_satellite_scene_reupload_failure_keeps_previous_source(tmp_path: Path) -> None:
    """같은 센서·촬영일로 다시 올리다 실패하면(마스크가 GeoTIFF 가 아님) 먼저 등록된 장면의 원본 pre.tif 가 남아 있어야 한다."""
    client = _client(tmp_path)
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    first = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", pre, "image/tiff")},
    )
    assert first.status_code == 201, first.text
    scene = first.json()["scene"]
    assert scene["source_pre"] == "pre.tif" and "/" not in scene["source_pre"]  # 호스트 경로를 노출하지 않는다
    source_pre = tmp_path / "user" / "toy" / "uploads" / f"{scene['scene_id']}" / "pre.tif"
    assert source_pre.is_file(), source_pre
    retry = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", b"garbage" * 50, "image/tiff")},
    )
    assert retry.status_code == 422
    assert source_pre.is_file(), "재등록 실패가 기존 장면의 원본을 지웠다"
    assert sorted(p.name for p in source_pre.parent.iterdir()) == ["pre.tif"]
    listed = client.get("/api/v1/regions/toy/satellite/scenes").json()
    assert listed["n_scenes"] == 1 and listed["scenes"][0]["scene_id"] == "20240501_ICEYE"


def test_satellite_scene_upload_size_limit(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    write_region(builtin)
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user", scene_upload_max_bytes=1024)
    client = TestClient(create_app(settings))
    pre = _geotiff_bytes(tmp_path, "pre.tif")
    assert len(pre) > 1024
    too_big = client.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", pre, "image/tiff")},
    )
    assert too_big.status_code == 413
    assert not list((tmp_path / "user").rglob("*.tif"))
    # pre 는 한도 안이고 mask 가 한도를 넘으면 — 이미 저장된 pre 업로드도 남기지 않는다
    small_settings = Settings(result_dir=tmp_path / "results2", regions_dir=builtin, regions_user_dir=tmp_path / "user2", scene_upload_max_bytes=len(pre) + 16)
    client2 = TestClient(create_app(small_settings))
    mask_rejected = client2.post(
        "/api/v1/regions/toy/satellite/scenes",
        data={"sensor": "ICEYE", "sensor_type": "SAR", "image_date": "20240501"},
        files={"pre_file": ("pre.tif", pre, "image/tiff"), "mask_file": ("mask.tif", pre + b"\0" * 64, "image/tiff")},
    )
    assert mask_rejected.status_code == 413 and "mask" in mask_rejected.json()["detail"]
    assert not list((tmp_path / "user2").rglob("*.tif"))
    assert not (tmp_path / "user2" / "toy" / "uploads" / "20240501_ICEYE").exists()


def test_scene_server_path_registration_is_limited_to_data_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """서버 경로 등록은 허용 루트(data/·regions/·사용자 지역·SCENE_PATH_ROOTS) 안만 — 임의 경로 존재 여부가 새지 않는다."""
    client = _client(tmp_path)
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"not really a tiff")
    refused = client.post("/api/v1/regions/toy/satellite/scenes", data={"sensor": "ICEYE", "image_date": "20240501", "pre_path": str(outside)})
    assert refused.status_code == 422 and "허용 루트" in refused.json()["detail"]
    # 허용 루트에 추가하면 경로 검사는 통과하고(파일 내용 오류로) 다음 단계에서 422 가 난다
    monkeypatch.setenv("SCENE_PATH_ROOTS", str(tmp_path))
    allowed = client.post("/api/v1/regions/toy/satellite/scenes", data={"sensor": "ICEYE", "image_date": "20240501", "pre_path": str(outside)})
    assert allowed.status_code == 422 and "허용 루트" not in allowed.json()["detail"]


def test_quality_control_keeps_record_floods_and_regime_shifts_but_drops_absurd_values() -> None:
    """나주대교(2020-08 기록 홍수, 2018 기준면 +2 m)·안동대교(4167 m)·양산교(-0.01 m 정체) 사례를 흉내 낸다."""
    from backend.app.gauge.regions import quality_control_rows

    rows = make_series(days=365 * 6, missing=0)
    base = {r["date"]: r["water_level_m"] for r in rows}
    dates = [r["date"] for r in rows]
    for d in dates[365 * 3:]:  # 기준면 변경(+2 m) — 남아야 한다
        base[d] += 2.0
    peak = dates[365 * 4 + 200]
    base[dates[365 * 4 + 199]] = base[peak] + 2.5   # 전날도 높음(홍수 상승)
    base[peak] = base[peak] + 8.0                    # 기록적 봉우리
    base[dates[365 * 4 + 201]] = base[peak] - 4.0    # 다음날 감쇠 중
    absurd = dates[365 * 2 + 100]
    base[absurd] = 4167.63                            # 전송 오류 — 이웃이 저수위여도 빠져야 한다
    for d in dates[365 * 1 + 300: 365 * 1 + 360]:     # 두 달 -0.01 m 정체(계기 고장)
        base[d] = -0.01
    clean, flags = quality_control_rows([{"date": d, "water_level_m": v} for d, v in base.items()])
    kept = {r["date"] for r in clean}
    assert peak in kept and dates[365 * 4 + 199] in kept, "기록적 홍수 봉우리는 남아야 한다"
    assert absurd not in kept and any(s["date"] == absurd for s in flags["spikes"])
    assert not (set(dates[365 * 1 + 300: 365 * 1 + 360]) & kept) and flags["flat_runs"]
    assert all(d in kept for d in dates[365 * 3: 365 * 3 + 30]), "기준면 변경 직후 값은 남아야 한다"


# --- 사용자 지정 분할 · 비동기 학습 잡 · 내장 산출물로 되돌리기 -----------------------------------------


CUSTOM_SPLIT = {
    "train_end": "2020-12-31",
    "val_start": "2021-01-01",
    "val_end": "2021-12-31",
    "test_start": "2022-01-01",
    "test_end": "2022-12-31",
}
SMALL_TRAIN = {"stations": ["A"], "forecasters": ["mean_reversion"], "horizon_max": 1}


def _poll_job(client: TestClient, region_id: str, job_id: str, timeout_s: float = 90.0) -> dict:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/regions/{region_id}/gauge/train/jobs/{job_id}")
        assert status.status_code == 200, status.text
        job = status.json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


def test_custom_split_is_recorded_in_artifacts_evaluation_and_region_summary(tmp_path: Path) -> None:
    """(a) 요청 split 이 받아들여지고 지점 산출물·evaluation.json·지역 요약(split_active)에 그대로 기록된다."""
    client = _client(tmp_path)
    before = client.get("/api/v1/regions/toy").json()["forecast"]
    assert before["split_default"] == SPLIT and before["split_active"] == SPLIT
    assert before["custom_split"] is False and before["trained_at"] is None and before["reset_available"] is False
    assert before["series_range"][0] == "2018-01-01"

    trained = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT})
    assert trained.status_code == 200, trained.text
    payload = trained.json()
    assert payload["split"] == CUSTOM_SPLIT and payload["custom_split"] is True
    assert payload["trained_at"] and payload["artifacts_origin"] == "user"
    assert payload["trained_now"] == ["A"] and payload["failures"] == {}

    artifact = json.loads((tmp_path / "user" / "toy" / "models" / "gauge" / "A.json").read_text(encoding="utf-8"))
    assert artifact["split"] == CUSTOM_SPLIT
    on_disk = json.loads((tmp_path / "user" / "toy" / "models" / "gauge" / "evaluation.json").read_text(encoding="utf-8"))
    assert on_disk["split"] == CUSTOM_SPLIT and on_disk["custom_split"] is True and on_disk["split_source"] == "request"

    evaluation = client.get("/api/v1/regions/toy/gauge/evaluation").json()
    assert evaluation["split"] == CUSTOM_SPLIT and evaluation["custom_split"] is True
    assert evaluation["trained_at"] == on_disk["generated_at"]
    forecast = client.get("/api/v1/regions/toy").json()["forecast"]
    assert forecast["split_active"] == CUSTOM_SPLIT and forecast["split_default"] == SPLIT
    assert forecast["custom_split"] is True and forecast["trained_at"] == on_disk["generated_at"]
    assert forecast["artifacts_origin"] == "user"

    # 분할 없이 다시 학습하면 지역 기본 분할로 돌아가고 custom_split 도 내려간다(동기 경로는 예전과 같다)
    again = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN)
    assert again.status_code == 200, again.text
    assert again.json()["split"] == SPLIT and again.json()["custom_split"] is False
    assert client.get("/api/v1/regions/toy").json()["forecast"]["custom_split"] is False


@pytest.mark.parametrize(
    ("patch", "needle"),
    [
        ({"train_end": "2021-06-30"}, "분할 순서"),  # train_end ≥ val_start
        ({"test_end": "2021-12-31"}, "분할 순서"),  # test_end < test_start
        ({"test_end": "2030-12-31"}, "시계열 범위"),  # 범위 밖
        ({"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"}, "학습 구간이 너무 짧습니다"),
        ({"val_end": "2021-02-28"}, "검증 구간이 너무 짧습니다"),
        ({"test_end": "2022-01-20"}, "시험 구간이 너무 짧습니다"),
    ],
)
def test_custom_split_rules_reject_with_korean_message(tmp_path: Path, patch: dict, needle: str) -> None:
    """(b) 규칙마다 422 + 한국어 사유, 사유에 지역 시계열 범위가 들어 있어 화면이 그대로 보여 줄 수 있다."""
    client = _client(tmp_path)
    bad = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT | patch})
    assert bad.status_code == 422, bad.text
    detail = bad.json()["detail"]
    assert needle in detail, detail
    assert "2018-01-01" in detail, detail  # 지역 시계열 범위(시작)
    # 비동기 요청도 학습 전에 같은 검증을 받는다 — 잡이 생기지 않는다
    bad_async = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT | patch, "async_run": True})
    assert bad_async.status_code == 422 and needle in bad_async.json()["detail"]
    assert client.get("/api/v1/regions/toy/gauge/train/jobs").json()["items"] == []
    # 산출물이 생기지 않았다
    assert not (tmp_path / "user" / "toy" / "models" / "gauge").exists()


def test_custom_split_missing_field_is_422_from_schema(tmp_path: Path) -> None:
    client = _client(tmp_path)
    bad = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": {"train_end": "2020-12-31"}})
    assert bad.status_code == 422


def test_async_training_job_accepts_polls_and_refuses_concurrent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """(c) async_run=true → 202 → 폴링으로 done → evaluation 포함. 같은 지역에 잡이 돌면 409."""
    import threading
    import time

    from backend.app.gauge import router as gauge_router

    release = threading.Event()
    original = gauge_router.train_region

    def slow_train_region(*args, **kwargs):  # 첫 잡을 붙들어 두어 동시 요청을 확실히 겹치게 한다
        release.wait(timeout=30)
        return original(*args, **kwargs)

    monkeypatch.setattr(gauge_router, "train_region", slow_train_region)
    client = _client(tmp_path)
    accepted = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT, "async_run": True})
    assert accepted.status_code == 202, accepted.text
    body = accepted.json()
    assert body["status"] == "queued" and body["region_id"] == "toy"
    assert body["split_used"] == CUSTOM_SPLIT and body["stations"] == ["A"]
    job_id = body["job_id"]
    assert body["poll_url"].endswith(f"/regions/toy/gauge/train/jobs/{job_id}")

    # 진행 중: 같은 지역의 비동기·동기 학습 모두 409, 되돌리기도 409
    time.sleep(0.2)
    running = client.get(f"/api/v1/regions/toy/gauge/train/jobs/{job_id}").json()
    assert running["status"] in {"queued", "running"} and running["evaluation"] is None
    conflict = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"async_run": True})
    assert conflict.status_code == 409 and "진행 중" in conflict.json()["detail"]
    assert client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN).status_code == 409
    assert client.delete("/api/v1/regions/toy/gauge/models").status_code == 409
    # 다른 지역은 막히지 않는다(지점이 없어 422 — 409 가 아님)
    client.post("/api/v1/regions", json={"id": "other", "name": "다른 지역"})
    assert client.post("/api/v1/regions/other/gauge/train", json={"async_run": True}).status_code == 422

    release.set()
    job = _poll_job(client, "toy", job_id)
    assert job["status"] == "done", job.get("error")
    assert job["started_at"] and job["finished_at"] and job["elapsed_s"] is not None and job["elapsed_s"] >= 0
    assert job["progress"] == {"stations_total": 1, "stations_done": 1, "current_station": None}
    assert job["split_used"] == CUSTOM_SPLIT and job["custom_split"] is True and job["error"] is None
    evaluation = job["evaluation"]
    assert evaluation["split"] == CUSTOM_SPLIT and evaluation["custom_split"] is True and evaluation["trained_now"] == ["A"]
    served = client.get("/api/v1/regions/toy/gauge/evaluation").json()
    assert served["generated_at"] == evaluation["generated_at"] and served["custom_split"] is True

    listed = client.get("/api/v1/regions/toy/gauge/train/jobs").json()["items"]
    assert [j["job_id"] for j in listed] == [job_id] and listed[0]["evaluation"] is None
    assert client.get(f"/api/v1/regions/toy/gauge/train/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/v1/regions/other/gauge/train/jobs/{job_id}").status_code == 404
    assert client.get("/api/v1/regions/toy/gauge/train/jobs/nope").status_code == 404

    # 끝난 뒤에는 새 잡을 받는다 — 목록은 최신순
    second = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"async_run": True})
    assert second.status_code == 202
    assert second.json()["split_used"] == SPLIT
    done = _poll_job(client, "toy", second.json()["job_id"])
    assert done["status"] == "done" and done["custom_split"] is False and done["evaluation"]["split"] == SPLIT
    assert [j["job_id"] for j in client.get("/api/v1/regions/toy/gauge/train/jobs").json()["items"]] == [done["job_id"], job_id]


def test_async_job_failure_is_recorded_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.gauge import router as gauge_router

    client = _client(tmp_path)
    # 존재하지 않는 지점은 접수 전에 422 — 잡이 생기지 않는다
    assert client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["ZZ"], "async_run": True}).status_code == 422
    assert client.get("/api/v1/regions/toy/gauge/train/jobs").json()["items"] == []

    def boom(*args, **kwargs):
        raise TrainingError("학습 중 예외 — 테스트")

    monkeypatch.setattr(gauge_router, "train_region", boom)
    accepted = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"async_run": True})
    assert accepted.status_code == 202, accepted.text
    job = _poll_job(client, "toy", accepted.json()["job_id"])
    assert job["status"] == "failed" and "테스트" in job["error"] and job["evaluation"] is None
    assert job["finished_at"] and job["elapsed_s"] is not None
    # 실패한 잡은 지역을 잠그지 않는다
    monkeypatch.setattr(gauge_router, "train_region", train_region)
    assert client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"async_run": True}).status_code == 202


def _train_builtin_artifacts(builtin_root: Path, region_id: str = "toy") -> Path:
    registry = RegionRegistry(builtin_root, builtin_root.parent / "unused_user")
    forecasters = build_forecaster_registry()
    write_dir = builtin_root / region_id / "models" / "gauge"
    train_region(registry, forecasters, region_id, allowed_forecasters=["mean_reversion"], horizon_max=1, write_dir=write_dir)
    return write_dir


def test_models_reset_restores_builtin_artifacts(tmp_path: Path) -> None:
    """(d) 내장 지역에 사용자 분할로 학습 → custom_split true / DELETE → 내장 산출물 복귀, custom_split false, 내장 파일은 그대로."""
    builtin = tmp_path / "builtin"
    write_region(builtin)
    builtin_models = _train_builtin_artifacts(builtin)
    builtin_snapshot = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in builtin_models.glob("*.json")}
    assert {"A.json", "B.json", "evaluation.json"} <= set(builtin_snapshot)

    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user",
                        gauge_forecaster_plugins=PLUGIN_SPECS)
    client = TestClient(create_app(settings))
    summary = client.get("/api/v1/regions/toy").json()
    assert summary["origin"] == "builtin" and summary["n_stations_trained"] == 2
    assert summary["forecast"]["artifacts_origin"] == "builtin" and summary["forecast"]["custom_split"] is False
    assert summary["forecast"]["reset_available"] is False and summary["forecast"]["split_active"] == SPLIT
    base_eval = client.get("/api/v1/regions/toy/gauge/evaluation").json()
    assert base_eval["custom_split"] is False and base_eval["artifacts_origin"] == "builtin" and base_eval["trained_at"]

    trained = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT})
    assert trained.status_code == 200, trained.text
    assert trained.json()["custom_split"] is True and trained.json()["artifacts_origin"] == "user"
    # 학습하지 않은 지점 B 는 내장 산출물이 평가표에 그대로 섞이고, 다른 분할로 표시된다
    assert trained.json()["n_stations"] == 2 and trained.json()["stations_with_other_split"] == ["B"]
    user_models = tmp_path / "user" / "toy" / "models" / "gauge"
    assert (user_models / "A.json").is_file() and (user_models / "evaluation.json").is_file()
    (user_models / "keep.txt").write_text("not json", encoding="utf-8")
    (user_models / "sub").mkdir()
    (user_models / "sub" / "nested.json").write_text("{}", encoding="utf-8")
    mid = client.get("/api/v1/regions/toy").json()["forecast"]
    assert mid["custom_split"] is True and mid["split_active"] == CUSTOM_SPLIT and mid["reset_available"] is True
    assert mid["artifacts_origin"] == "user"
    assert client.get("/api/v1/regions/toy/gauge/evaluation").json()["custom_split"] is True

    reset = client.delete("/api/v1/regions/toy/gauge/models")
    assert reset.status_code == 200, reset.text
    body = reset.json()
    assert body["active"] == "builtin" and body["region_id"] == "toy"
    assert set(body["removed_files"]) >= {"A.json", "evaluation.json"} and "nested.json" not in body["removed_files"]
    assert Path(body["artifacts_dir"]).resolve() == builtin_models.resolve()
    assert not list(user_models.glob("*.json")) and (user_models / "keep.txt").is_file() and (user_models / "sub" / "nested.json").is_file()
    # 내장 파일은 바이트·수정 시각 모두 그대로
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in builtin_models.glob("*.json")} == builtin_snapshot

    after = client.get("/api/v1/regions/toy").json()
    assert after["forecast"]["custom_split"] is False and after["forecast"]["split_active"] == SPLIT
    assert after["forecast"]["artifacts_origin"] == "builtin" and after["forecast"]["reset_available"] is False
    assert after["forecast"]["trained_at"] == base_eval["generated_at"] and after["n_stations_trained"] == 2
    restored = client.get("/api/v1/regions/toy/gauge/evaluation").json()
    assert restored["custom_split"] is False and restored["split"] == SPLIT and restored["generated_at"] == base_eval["generated_at"]
    assert client.get("/api/v1/regions/toy/gauge/A/forecast", params={"horizon": 1}).status_code == 200
    # 두 번째 되돌리기는 지울 것이 없어도 성공(멱등)
    assert client.delete("/api/v1/regions/toy/gauge/models").json()["removed_files"] == []
    # 되돌린 뒤 다시 학습하면 사용자 산출물이 다시 활성
    again = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN)
    assert again.status_code == 200 and again.json()["artifacts_origin"] == "user" and again.json()["n_stations"] == 2


def test_models_reset_refuses_region_without_builtin_artifacts(tmp_path: Path) -> None:
    client = _client(tmp_path)  # 내장 지역이지만 내장 산출물이 없다
    trained = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN)
    assert trained.status_code == 200
    refused = client.delete("/api/v1/regions/toy/gauge/models")
    assert refused.status_code == 409 and "내장" in refused.json()["detail"]
    assert (tmp_path / "user" / "toy" / "models" / "gauge" / "A.json").is_file()  # 아무것도 지우지 않았다
    # 사용자 생성 지역도 409
    client.post("/api/v1/regions", json={"id": "mine", "name": "내 지역"})
    assert client.delete("/api/v1/regions/mine/gauge/models").status_code == 409
    assert client.delete("/api/v1/regions/nope/gauge/models").status_code == 404


def test_plugin_whose_fit_raises_is_skipped_in_custom_split_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """(e) 서버 이미지처럼 torch/prophet 이 없어 fit 이 ValueError 를 내는 플러그인은 후보 error 로 기록되고 학습은 성공한다."""
    module = tmp_path / "broken_forecaster.py"
    module.write_text(
        '''
import numpy as np
from backend.app.gauge.forecasters import GaugeForecaster, ForecasterInfo

class NoTorch(GaugeForecaster):
    info = ForecasterInfo(id="needs_torch", name="torch 필요", description="테스트")
    def fit(self, series, origins, horizon, hparams):
        raise ValueError("torch 가 없어 신경망을 학습할 수 없습니다 (추론은 numpy 로 가능)")
    def predict(self, series, origins, horizon, params):
        return np.full(len(origins), np.nan)
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    builtin = tmp_path / "builtin"
    write_region(builtin)
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user",
                        gauge_forecaster_plugins=PLUGIN_SPECS + ",broken_forecaster:NoTorch")
    client = TestClient(create_app(settings))
    assert any(f["id"] == "needs_torch" for f in client.get("/api/v1/gauge/forecasters").json())

    for async_run in (False, True):
        response = client.post(
            "/api/v1/regions/toy/gauge/train",
            json={"stations": ["A"], "forecasters": ["mean_reversion", "needs_torch", "neural_mlp"], "horizon_max": 1,
                  "split": CUSTOM_SPLIT, "async_run": async_run},
        )
        if async_run:
            assert response.status_code == 202, response.text
            job = _poll_job(client, "toy", response.json()["job_id"])
            assert job["status"] == "done", job.get("error")
            evaluation = job["evaluation"]
        else:
            assert response.status_code == 200, response.text
            evaluation = response.json()
        assert evaluation["failures"] == {} and evaluation["trained_now"] == ["A"] and evaluation["custom_split"] is True
        assert all(row["forecaster_id"] == "mean_reversion" for row in evaluation["rows"])
        model = client.get("/api/v1/regions/toy/gauge/A/model").json()
        candidates = model["horizons"]["1"]["candidates"]
        by_id = {c["forecaster_id"]: c for c in candidates}
        assert "torch" in by_id["needs_torch"]["error"]
        assert "torch" in by_id["neural_mlp"]["error"]  # .venv 에도 torch 가 없다 — 서버와 같은 동작
        assert any(c.get("selected") for c in candidates if c["forecaster_id"] == "mean_reversion")


def test_partial_retrain_after_reset_keeps_untrained_stations_servable(tmp_path: Path) -> None:
    """되돌리기(reset) 뒤 지점 하나만 사용자 분할로 다시 학습해도 나머지 지점의 내장 산출물이 계속 보여야 한다.

    (적대적 검증에서 발견) 되돌리기는 빈 models/gauge 디렉터리를 남기고, 그 뒤 지점 A 만 학습하면 사용자
    디렉터리가 활성이 되면서 B 의 /model·/forecast 가 404 였다 — 평가표는 B 를 포함해 n_stations=2 라고 하면서.
    """
    builtin = tmp_path / "builtin"
    write_region(builtin)
    builtin_models = _train_builtin_artifacts(builtin)
    snapshot = {p.name: p.read_bytes() for p in builtin_models.glob("*.json")}
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    assert client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN).status_code == 200
    assert client.delete("/api/v1/regions/toy/gauge/models").status_code == 200
    user_models = tmp_path / "user" / "toy" / "models" / "gauge"
    # 사용자가 더한 것이 없는 사본은 되돌리기 때 통째로 걷힌다(내장 갱신을 가리지 않게) — 있어도 비어 있어야 한다
    assert not user_models.is_dir() or not list(user_models.glob("*.json"))

    accepted = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT, "async_run": True})
    assert accepted.status_code == 202, accepted.text
    job = _poll_job(client, "toy", accepted.json()["job_id"])
    assert job["status"] == "done" and job["evaluation"]["n_stations"] == 2
    assert job["evaluation"]["stations_with_other_split"] == ["B"]
    assert client.get("/api/v1/regions/toy/gauge/B/model").status_code == 200
    forecast_b = client.get("/api/v1/regions/toy/gauge/B/forecast", params={"as_of": "2023-06-01", "horizon": 1})
    assert forecast_b.status_code == 200 and forecast_b.json()["params_used"] == "params_eval"  # 내장 분할의 시험 연도
    forecast_a = client.get("/api/v1/regions/toy/gauge/A/forecast", params={"as_of": "2022-06-01", "horizon": 1})
    assert forecast_a.json()["params_used"] == "params_eval"  # 새 분할의 시험 연도(2022)
    assert client.get("/api/v1/regions/toy").json()["n_stations_trained"] == 2
    assert {p.name: p.read_bytes() for p in builtin_models.glob("*.json")} == snapshot
    removed = client.delete("/api/v1/regions/toy/gauge/models").json()["removed_files"]
    assert {"A.json", "B.json", "evaluation.json"} <= set(removed)


def test_custom_split_where_no_station_trains_is_an_error_not_a_silent_evaluation(tmp_path: Path) -> None:
    """(적대적 검증에서 발견) 학습 730일·검증 1년 분할은 규칙은 통과하지만 폴드를 하나도 못 만든다 — 예전에는
    n_stations=0 인 평가표를 사용자 분할로 기록해 화면이 잘못된 구간을 보여 줬다. 지금은 422 / 잡 실패."""
    client = _client(tmp_path)
    split = {"train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"}
    sync = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": split})
    assert sync.status_code == 422 and "학습된 지점이 없습니다" in sync.json()["detail"], sync.text
    assert client.get("/api/v1/regions/toy/gauge/evaluation").status_code == 404
    accepted = client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": split, "async_run": True})
    assert accepted.status_code == 202
    job = _poll_job(client, "toy", accepted.json()["job_id"])
    assert job["status"] == "failed" and "학습된 지점이 없습니다" in job["error"] and job["evaluation"] is None
    assert client.get("/api/v1/regions/toy").json()["forecast"]["custom_split"] is False
    # 실패한 잡은 지역을 잠그지 않는다
    assert client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN).status_code == 200


def test_sync_training_holds_region_lock_for_async_and_reset() -> None:
    """동기 학습이 도는 동안 같은 지역의 비동기 접수·초기화는 409 — 잠금은 끝나면 풀린다."""
    from backend.app.gauge.jobs import GaugeTrainJobs, JobConflictError

    jobs = GaugeTrainJobs()
    with jobs.hold("toy"):
        assert jobs.active_for("toy")["status"] == "running"
        with pytest.raises(JobConflictError):
            with jobs.hold("toy"):
                pass
        with pytest.raises(JobConflictError):
            jobs.submit("toy", lambda on_progress: {}, split_used={}, custom_split=False, stations=["A"], forecasters=None, horizon_max=None)
    assert jobs.active_for("toy") is None
    assert jobs.active_for("other") is None


def test_reset_removes_pristine_user_copy_but_keeps_user_additions(tmp_path: Path) -> None:
    """되돌리기: 사본이 내장 복사본뿐이면 사본 전체 제거 → 내장 지역이 다시 보인다. 장면·지점 추가가 있으면 사본은 남긴다."""
    builtin = tmp_path / "builtin"
    write_region(builtin)
    _train_builtin_artifacts(builtin)
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user", gauge_forecaster_plugins=PLUGIN_SPECS)
    client = TestClient(create_app(settings))
    assert client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT}).status_code == 200
    user_region = tmp_path / "user" / "toy"
    assert user_region.is_dir() and client.get("/api/v1/regions/toy").json()["origin"] == "user"
    assert client.delete("/api/v1/regions/toy/gauge/models").status_code == 200
    assert not user_region.exists(), "복사본뿐인 사본은 통째로 사라진다"
    assert client.get("/api/v1/regions/toy").json()["origin"] == "builtin"
    # 지점을 새로 올린 사본은 남는다
    client.post("/api/v1/regions/toy/gauge/Z/series", json={"rows": [{"date": "2019-01-01", "water_level_m": 1.0}, {"date": "2019-01-02", "water_level_m": 1.1}]})
    assert user_region.is_dir()
    client.post("/api/v1/regions/toy/gauge/train", json=SMALL_TRAIN | {"split": CUSTOM_SPLIT})
    assert client.delete("/api/v1/regions/toy/gauge/models").status_code == 200
    assert user_region.is_dir() and (user_region / "gauge" / "series.json").is_file()


# --- 미래 강수(완전 예보 가정) 실험 ---------------------------------------------------------


def _toy_weather(rows: list[dict], seed: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {"date": r["date"], "precipitation_mm": float(rng.gamma(0.4, 20.0)) if rng.random() < 0.3 else 0.0, "avg_temperature_c": 12.0}
        for r in rows
    ]


def test_future_rain_forecaster_uses_future_window_and_is_excluded_by_default() -> None:
    from backend.app.gauge.forecasters import RidgeARXFutureForecaster, future_rain_features

    rows = make_series(days=365 * 4)
    series = DailySeries.from_rows(rows).with_exog(_toy_weather(rows), {"precipitation_mm": "rain_mm", "avg_temperature_c": "temp_c"})
    registry = build_forecaster_registry()
    future = registry.get("ridge_arx_future")
    assert isinstance(future, RidgeARXFutureForecaster) and future.info.uses_future_exog
    # 기본 후보 풀에서 제외, 명시하면 포함
    assert all(not f.info.uses_future_exog for f in registry.candidates())
    assert "ridge_arx_future" not in {f.info.id for f in registry.candidates()}
    assert {f.info.id for f in registry.candidates(["ridge_arx", "ridge_arx_future"])} == {"ridge_arx", "ridge_arx_future"}
    assert not registry.get("ridge_arx").info.uses_future_exog

    origins = np.arange(400, series.n - 40)
    for future_days in (3, "all"):
        params = future.fit(series, origins, 7, {"p": 7, "lam": 10.0, "mode": "delta", "future_days": future_days})
        assert params["future_days"] == future_days
        pred = future.predict(series, np.array([series.n - 20]), 7, params)
        assert np.isfinite(pred).all() and future.last_note_ko is None
        # 창이 관측 끝을 넘으면 0 으로 두고 사유를 남긴다
        pred_edge = future.predict(series, np.array([series.n - 1]), 7, params)
        assert np.isfinite(pred_edge).all() and future.last_note_ko and "기상청 예보" in future.last_note_ko
    # 미래 창 특징: K=3 이면 h=7 이어도 3일치만, 'all' 이면 7일치
    rain = np.nan_to_num(series.exog["rain_mm"], nan=0.0)
    feats3, trunc3 = future_rain_features(series, np.array([500]), 7, 3)
    feats_all, trunc_all = future_rain_features(series, np.array([500]), 7, "all")
    assert trunc3 == 0 and trunc_all == 0
    assert feats3[0, 0] == pytest.approx(np.log1p(rain[501:504].sum()))
    assert feats_all[0, 0] == pytest.approx(np.log1p(rain[501:508].sum()))
    assert feats_all[0, 1] == pytest.approx(np.log1p(rain[501:508].max()))
    # 기상 없는 시계열은 후보에서 빠진다
    with pytest.raises(ValueError):
        future.fit(DailySeries.from_rows(rows), origins, 3, {"p": 7, "lam": 10.0, "mode": "delta", "future_days": 3})


def test_forecasters_endpoint_flags_future_exog_and_train_never_picks_it(tmp_path: Path) -> None:
    client = _client(tmp_path)
    listed = {f["id"]: f for f in client.get("/api/v1/gauge/forecasters").json()}
    assert listed["ridge_arx_future"]["uses_future_exog"] is True
    assert listed["ridge_arx"]["uses_future_exog"] is False
    guide = client.get("/api/v1/gauge/model-guide").json()
    entry = next(f for f in guide["forecasters"] if f["id"] == "ridge_arx_future")
    assert entry["uses_weather"] and entry["future_exog"]
    assert "예보" in guide["weather_ko"]
    # 기본 학습(후보 미지정)은 미래 강수 예측기를 후보로 올리지 않는다 (토이 지역엔 기상이 없어 어차피 못 쓰지만, 후보 목록에도 없어야 한다)
    trained = client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["A"], "forecasters": ["mean_reversion", "ridge_ar"], "horizon_max": 1}).json()
    model = client.get("/api/v1/regions/toy/gauge/A/model").json()
    ids = {c["forecaster_id"] for c in model["horizons"]["1"]["candidates"]}
    assert "ridge_arx_future" not in ids and trained["rows"]


def _load_experiment_script():
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("eval_future_rain_experiment", root / "scripts" / "eval_future_rain_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_future_rain_experiment_schema_and_endpoint(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    rows_a, rows_b = make_series(), make_series(seed=11)
    write_region(builtin, stations={"A": rows_a, "B": rows_b})
    (builtin / "toy" / "weather").mkdir()
    (builtin / "toy" / "weather" / "asos_daily.json").write_text(json.dumps({"rows": _toy_weather(rows_a)}), encoding="utf-8")
    registry = RegionRegistry(builtin, tmp_path / "user")
    forecasters = build_forecaster_registry()
    script = _load_experiment_script()

    payload = script.run_region(registry, forecasters, "toy", scratch_dir=tmp_path / "scratch", horizon_max=3, report_horizons=(1, 3))
    assert payload["schema"] == script.EXPERIMENT_SCHEMA
    assert payload["split"] == SPLIT and payload["n_stations"] == 2 and payload["report_horizons"] == [1, 3]
    assert set(payload["stations"]) == {"A", "B"} and payload["notes_ko"]
    for station in payload["stations"].values():
        for horizon in ("1", "3"):
            models = station["horizons"][horizon]["models"]
            assert set(models) == {"persistence", "ridge_arx", "future_3d", "future_all"}
            for name in ("persistence", "ridge_arx", "future_3d", "future_all"):
                assert all(isinstance(models[name][k], float) for k in ("rmse", "mse", "mae"))
                assert models[name]["skill_vs_persistence"] is not None
            assert isinstance(models["future_all"]["delta_rmse_cm"], float)
            # D+1·D+3 에서는 두 변형의 입력이 같다
            assert models["future_3d"]["rmse"] == pytest.approx(models["future_all"]["rmse"])
        assert len(station["events"]) == 5
        event = station["events"][0]
        assert event["date"].startswith("2023") and event["rise_m"] >= station["events"][-1]["rise_m"]
        assert {"persistence", "ridge_arx", "future_all", "future_rain_mm"} <= set(event["errors"]["D+1"])
    summary = payload["summary"]
    assert set(summary["by_horizon"]) == {"1", "3"} and summary["cells_total"] == 4
    assert set(summary["by_horizon"]["1"]["mean_skill"]) == {"persistence", "ridge_arx", "future_3d", "future_all"}
    assert set(summary["cells_future_beats_arx"]) == {"future_3d", "future_all"}
    # 스크래치에만 학습 산출물, 지역 산출물 디렉터리에는 experiments/ 만
    assert (tmp_path / "scratch" / "toy" / "models" / "gauge" / "A.json").is_file()
    experiment_path = builtin / "toy" / "models" / "gauge" / "experiments" / "future_rain.json"
    assert experiment_path.is_file() and not (builtin / "toy" / "models" / "gauge" / "A.json").exists()
    docs = tmp_path / "docs.md"
    script.write_docs({"toy": payload}, docs)
    assert "## D+1" in docs.read_text(encoding="utf-8") and "## D+3" in docs.read_text(encoding="utf-8")

    # API: 내장 실험 파일을 읽고, 사용자 산출물이 활성이어도 내장으로 떨어진다
    settings = Settings(result_dir=tmp_path / "results", regions_dir=builtin, regions_user_dir=tmp_path / "user")
    client = TestClient(create_app(settings))
    assert client.get("/api/v1/regions/nope/gauge/experiments").status_code == 404
    got = client.get("/api/v1/regions/toy/gauge/experiments").json()
    assert got["items"]["future_rain"]["schema"] == script.EXPERIMENT_SCHEMA and got["sources"]["future_rain"] == "builtin"
    client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["A"], "forecasters": ["mean_reversion"], "horizon_max": 1})
    assert client.get("/api/v1/regions/toy/gauge/evaluation").json()["artifacts_origin"] == "user"
    again = client.get("/api/v1/regions/toy/gauge/experiments").json()
    assert again["items"]["future_rain"]["n_stations"] == 2 and again["sources"]["future_rain"] == "builtin"
    # 실험 파일이 없는 지역은 null (사용자 사본이 만들어졌으면 그것도 지운다)
    experiment_path.unlink()
    user_copy = tmp_path / "user" / "toy" / "models" / "gauge" / "experiments" / "future_rain.json"
    assert user_copy.is_file()  # 내장 지역을 고치면 experiments/ 도 사본에 따라온다
    user_copy.unlink()
    empty = client.get("/api/v1/regions/toy/gauge/experiments").json()
    assert empty["items"]["future_rain"] is None and empty["sources"]["future_rain"] is None
    # 실험 항목은 EXPERIMENT_FILES 가 정의하는 만큼 늘 존재한다(없으면 값이 null)
    assert set(empty["items"]) == set(GaugeForecastService.EXPERIMENT_FILES) >= {"future_rain", "area_models"}
