"""외생 입력 피드(2026-10-20): 기상청 단기예보 → DailySeries.exog_tail, 댐 운영 일자료 → dam_* 외생, 계절 분리 예측기.

인터넷이 필요한 부분(기상청·WAMIS)은 바꿔 끼운다. 자료는 장난감 지역의 합성 시계열이며 성능 수치가 아니라 배선을 검사한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.gauge import kma_forecast
from backend.app.gauge.forecasters import (
    DailySeries,
    RidgeARXSeasonForecaster,
    RidgeARXSeasonFutureForecaster,
    build_forecaster_registry,
    dam_features,
    future_rain_features,
)
from backend.app.gauge.regions import RegionRegistry
from backend.app.screens import exog_feeds as feeds
from backend.app.screens_api import install_screens_api
from backend.tests.test_gauge import _client, make_series, write_region


def _weather_rows(rows: list[dict]) -> list[dict]:
    return [{"date": r["date"], "precipitation_mm": float(i % 9 == 0) * 12.0, "avg_temperature_c": 10.0 + (i % 30) / 3} for i, r in enumerate(rows)]


def _dam_rows(rows: list[dict]) -> list[dict]:
    return [{"date": r["date"], "dam_level_m": 70.0 + (i % 50) / 10, "inflow_cms": 20.0 + (i % 7) * 3, "outflow_cms": 25.0 + (i % 5) * 4} for i, r in enumerate(rows)]


# ── 시계열 꼬리(예보) ────────────────────────────────────────────────────────────────


def test_with_exog_puts_rows_after_calendar_end_into_tail_and_truncation_drops_it() -> None:
    rows = make_series(days=400)
    series = DailySeries.from_rows(rows)
    last = str(series.dates[-1])
    extra = [{"date": str(series.dates[-1] + np.timedelta64(k, "D")), "precipitation_mm": 10.0 * k} for k in (1, 2, 3)]
    with_fc = series.with_exog(_weather_rows(rows) + extra, {"precipitation_mm": "rain_mm", "avg_temperature_c": "temp_c"})
    assert with_fc.exog["rain_mm"].shape[0] == series.n and with_fc.exog_tail["rain_mm"].tolist() == [10.0, 20.0, 30.0]
    assert np.isnan(with_fc.exog_tail["temp_c"]).all()  # 예보 행에 기온이 없으면 NaN
    feats, truncated = future_rain_features(with_fc, np.array([series.n - 1]), 3, 3)
    assert truncated == 0 and feats[0, 0] == pytest.approx(np.log1p(60.0)) and feats[0, 1] == pytest.approx(np.log1p(30.0))
    cut = with_fc.truncated(np.datetime64(last) - np.timedelta64(10, "D"))
    assert cut.exog_tail == {} and cut.exog["rain_mm"].shape[0] == series.n - 10


def test_dam_features_need_outflow_and_add_optional_columns() -> None:
    rows = make_series(days=200)
    series = DailySeries.from_rows(rows)
    assert dam_features(series, np.array([100])) is None
    with_dam = series.with_exog(_dam_rows(rows), {"outflow_cms": "dam_out", "inflow_cms": "dam_in", "dam_level_m": "dam_level"})
    feats = dam_features(with_dam, np.array([100, 150]))
    assert feats.shape == (2, 6) and np.isfinite(feats).all()
    only_out = series.with_exog(_dam_rows(rows), {"outflow_cms": "dam_out"})
    assert dam_features(only_out, np.array([100])).shape == (1, 3)


# ── 계절 분리 예측기 ───────────────────────────────────────────────────────────────


def test_season_forecaster_fits_two_groups_and_predicts_with_and_without_dam() -> None:
    rows = make_series(days=365 * 4)
    base = DailySeries.from_rows(rows).with_exog(_weather_rows(rows), {"precipitation_mm": "rain_mm", "avg_temperature_c": "temp_c"})
    origins = np.arange(400, base.n - 10)
    fc = RidgeARXSeasonForecaster()
    params = fc.fit(base, origins, 3, {"p": 7, "lam": 10.0, "mode": "delta"})
    assert params["dam_used"] is False and params["flood"] is not None and params["other"] is not None and params["n_flood"] > 100
    pred = fc.predict(base, np.array([base.n - 5, base.n - 1]), 3, params)
    assert np.isfinite(pred).all()
    with_dam = base.with_exog(_dam_rows(rows), {"outflow_cms": "dam_out", "inflow_cms": "dam_in", "dam_level_m": "dam_level"})
    params_dam = fc.fit(with_dam, origins, 3, {"p": 7, "lam": 10.0, "mode": "delta"})
    assert params_dam["dam_used"] is True and len(params_dam["pooled"]["w"] if isinstance(params_dam["pooled"], dict) and "w" in params_dam["pooled"] else [0] * 1) >= 1
    assert np.isfinite(fc.predict(with_dam, np.array([with_dam.n - 1]), 3, params_dam)).all()
    with pytest.raises(ValueError):  # 학습 때 댐을 썼는데 예측 시 없으면 분명한 오류
        fc.predict(base, np.array([base.n - 1]), 3, params_dam)
    json.dumps(params_dam)  # 산출물로 저장 가능해야 한다

    future = RidgeARXSeasonFutureForecaster()
    pf = future.fit(base, origins, 3, {"p": 7, "lam": 10.0, "mode": "delta"})
    out = future.predict(base, np.array([base.n - 1]), 3, pf)
    assert np.isfinite(out).all() and future.last_note_ko and "0(비 없음)" in future.last_note_ko


def test_registry_lists_season_models_and_keeps_future_one_out_of_default_pool() -> None:
    registry = build_forecaster_registry()
    ids = {f.info.id for f in registry.list()}
    assert {"ridge_arx_season", "ridge_arx_season_future"} <= ids
    default = {f.info.id for f in registry.candidates()}
    assert "ridge_arx_season" in default and "ridge_arx_season_future" not in default
    assert "ridge_arx_season_future" in {f.info.id for f in registry.candidates(["ridge_arx_season_future"])}


# ── 기상청 단기예보 모듈 ──────────────────────────────────────────────────────────────


def test_kma_grid_parse_and_daily_aggregation() -> None:
    assert kma_forecast.dfs_grid(37.5665, 126.978) == (60, 127)
    assert [kma_forecast.parse_pcp(x) for x in ("강수없음", "1mm 미만", "30.0~50.0mm", "50.0mm 이상", "3.2mm", "")] == [0.0, 0.5, 40.0, 50.0, 3.2, 0.0]
    items = [
        {"fcstDate": "20261020", "fcstTime": "0600", "category": "PCP", "fcstValue": "1.0mm"},
        {"fcstDate": "20261020", "fcstTime": "0700", "category": "PCP", "fcstValue": "강수없음"},
        {"fcstDate": "20261020", "fcstTime": "0600", "category": "TMP", "fcstValue": "12"},
        {"fcstDate": "20261020", "fcstTime": "0700", "category": "TMP", "fcstValue": "14"},
        {"fcstDate": "20261020", "fcstTime": "0600", "category": "POP", "fcstValue": "60"},
        {"fcstDate": "20261021", "fcstTime": "0000", "category": "PCP", "fcstValue": "30.0~50.0mm"},
    ]
    rows = kma_forecast.daily_rows(items)
    assert rows[0] == {"date": "2026-10-20", "precipitation_mm": 1.0, "pop_max_pct": 60, "n_hours": 2, "avg_temperature_c": 13.0}
    assert rows[1]["precipitation_mm"] == 40.0 and rows[1]["avg_temperature_c"] is None


def test_kma_fetch_items_checks_result_code() -> None:
    ok = json.dumps({"response": {"header": {"resultCode": "00"}, "body": {"items": {"item": [{"category": "PCP"}]}}}})
    assert kma_forecast.fetch_items("k", 60, 127, "20261020", "0500", fetch=lambda url: ok) == [{"category": "PCP"}]
    bad = json.dumps({"response": {"header": {"resultCode": "30", "resultMsg": "SERVICE_KEY_IS_NOT_REGISTERED_ERROR"}}})
    with pytest.raises(RuntimeError):
        kma_forecast.fetch_items("k", 60, 127, "20261020", "0500", fetch=lambda url: bad)
    doc = kma_forecast.collect("k", 36.47, 127.48, fetch=lambda url: ok)
    assert doc["provider"] == "KMA_VILAGE_FCST" and doc["nx"] == 69 and doc["issued_at"].endswith("+09:00")


# ── API ────────────────────────────────────────────────────────────────────────────


def _screens(tmp_path: Path) -> TestClient:
    client = _client(tmp_path)
    install_screens_api(client.app)
    return client


def test_forecast_collect_requires_key_then_stores_and_feeds_future_forecaster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _screens(tmp_path)
    builtin = tmp_path / "builtin" / "toy"
    rows = make_series()
    (builtin / "weather").mkdir(exist_ok=True)
    (builtin / "weather" / "asos_daily.json").write_text(json.dumps({"schema": "watercast.weather.daily/1", "rows": _weather_rows(rows)}), encoding="utf-8")
    monkeypatch.delenv("KMA_API_KEY", raising=False)
    assert client.post("/api/v1/regions/toy/weather/forecast-collect", json={}).status_code == 422
    assert client.get("/api/v1/regions/toy/weather/forecast").json()["available"] is False

    monkeypatch.setenv("KMA_API_KEY", "test-key")
    last = rows[-1]["date"]
    future_dates = [str(np.datetime64(last, "D") + np.timedelta64(k, "D")) for k in (0, 1, 2, 3)]
    fake_doc = {"schema": kma_forecast.SCHEMA, "provider": kma_forecast.PROVIDER, "issued_at": "2026-10-20T05:00+09:00", "nx": 1, "ny": 1,
                "rows": [{"date": d, "precipitation_mm": 25.0, "avg_temperature_c": 15.0, "pop_max_pct": 80, "n_hours": 24} for d in future_dates]}
    seen: dict = {}
    monkeypatch.setattr(feeds, "kma_collect", lambda key, lat, lon: seen.update(key=key, lat=lat, lon=lon) or dict(fake_doc))
    r = client.post("/api/v1/regions/toy/weather/forecast-collect", json={"station_id": "A"})
    assert r.status_code == 200, r.text
    assert seen["key"] == "test-key" and seen["lat"] == pytest.approx(35.2036) and r.json()["n_days"] == 4 and r.json()["point_source"] == "A"
    assert (tmp_path / "user" / "toy" / "weather" / "kma_forecast_daily.json").is_file()
    assert client.get("/api/v1/regions/toy/weather/forecast").json()["issued_at"] == "2026-10-20T05:00+09:00"
    assert client.post("/api/v1/regions/toy/weather/forecast-collect", json={"station_id": "zzz"}).status_code == 422

    registry: RegionRegistry = client.app.state.regions
    series_set = registry.load_series("toy")
    assert series_set.meta["weather_forecast_available"] is True and series_set.meta["weather_forecast_issued_at"] == "2026-10-20T05:00+09:00"
    tail = series_set.series["A"].exog_tail["rain_mm"]
    assert tail.tolist() == [25.0, 25.0, 25.0]  # 관측이 있는 마지막 날은 관측 우선 → 그 뒤 3일만 꼬리

    # 학습 뒤 ridge_arx_future 전망은 자료 끝 원점에서 예보를 쓴다(future_exog_feed), 과거 원점에서는 0 으로 둔다
    trained = client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["A"], "forecasters": ["ridge_ar"], "horizon_max": 3})
    assert trained.status_code == 200, trained.text
    f = client.get("/api/v1/regions/toy/gauge/A/forecast?horizon=3&forecaster=ridge_arx_future")
    assert f.status_code == 200, f.text
    feed = f.json()["future_exog_feed"]
    assert feed["n_days"] == 3 and feed["issued_at"] == "2026-10-20T05:00+09:00" and "단기예보" in feed["note_ko"]
    past = client.get("/api/v1/regions/toy/gauge/A/forecast?horizon=3&forecaster=ridge_arx_future&as_of=2023-06-01").json()
    assert past["future_exog_feed"]["n_days"] == 0 and "과거 원점" in past["future_exog_feed"]["note_ko"]
    plain = client.get("/api/v1/regions/toy/gauge/A/forecast?horizon=3").json()
    assert plain["future_exog_feed"] is None


def test_dam_collect_stores_rows_updates_manifest_and_feeds_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _screens(tmp_path)
    rows = make_series(days=365 * 3)
    monkeypatch.setattr(feeds, "fetch_dam_rows", lambda code, start, end: ([{"date": r["date"], "dam_level_m": 70.0, "inflow_cms": 20.0, "outflow_cms": 30.0, "rain_mm": None} for r in rows], ["2018-02: 0 rows"]))
    bad = client.post("/api/v1/regions/toy/dam/collect", json={"dam_code": "12", "start": "2018-01-01"})
    assert bad.status_code == 422
    r = client.post("/api/v1/regions/toy/dam/collect", json={"dam_code": "3008110", "dam_name": "대청댐", "start": "2018-01-01", "end": "2020-12-31", "async_run": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "done" and body["n_total"] == len(rows) and body["kind"] == "dam_collect" and "1개 달" in body["note"]
    status = client.get("/api/v1/regions/toy/dam").json()
    assert status["available"] and status["dam_code"] == "3008110" and status["n"] == len(rows) and len(status["last_rows"]) == 7
    manifest = json.loads((tmp_path / "user" / "toy" / "region.json").read_text(encoding="utf-8"))
    assert manifest["dam"]["code"] == "3008110" and manifest["dam"]["daily_file"] == "weather/dam_daily.json"
    registry: RegionRegistry = client.app.state.regions
    series_set = registry.load_series("toy")
    assert series_set.meta["dam_available"] is True and series_set.meta["dam_code"] == "3008110"
    assert series_set.series["A"].has_exog("dam_out", "dam_in", "dam_level") and np.nanmax(series_set.series["A"].exog["dam_out"]) == 30.0
    assert client.get("/api/v1/regions/toy/dam/collect/jobs/nope").status_code == 404
    assert client.get("/api/v1/regions/nope/dam").status_code == 404


def test_parse_dam_payload_recovers_truncated_json() -> None:
    text = '{"result":{"code":"success"},"count":2,"list":[{"obsymd":"20150101","rwl":"73.31","iqty":"28.909","tdqty":"42.273","edqty":"31.029","spdqty":"0","otltdqty":"0","itqty":"11.244","rf":""},{"obsymd":"20150102","rwl":"73.3'
    rows = feeds.parse_dam_payload(text)
    assert len(rows) == 1 and rows[0]["date"] == "2015-01-01" and rows[0]["outflow_cms"] == 42.273 and rows[0]["rain_mm"] is None
