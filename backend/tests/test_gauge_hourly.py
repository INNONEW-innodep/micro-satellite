"""시 단위 수위(2026-10-20) — ``backend/app/gauge/hourly.py`` 와 화면 계약 ``screens/gauge_hourly.py``.

WAMIS 는 인터넷이 필요하므로 ``fetch_hourly_rows`` 를 바꿔 끼운다. 자료는 합성 시 수위(하루 주기 + 느린 추세)다 —
성능 수치가 아니라 배선(수집→저장→전망→검증 키)을 검사한다.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.gauge import hourly as H
from backend.app.screens import gauge_hourly as screen
from backend.app.screens_api import install_screens_api
from backend.tests.test_gauge import _client


def _rows(start: datetime, n_hours: int, *, base: float = 3.0, gap_at: tuple[int, int] | None = None) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    for i in range(n_hours):
        if gap_at and gap_at[0] <= i < gap_at[1]:
            continue
        t = start + timedelta(hours=i)
        # WAMIS 표기: HH 1..24, 24 는 다음날 0시 → t 가 0시면 전날 24시
        if t.hour == 0:
            key = (t - timedelta(days=1)).strftime("%Y%m%d") + "24"
        else:
            key = t.strftime("%Y%m%d%H")
        level = base + 0.3 * math.sin(2 * math.pi * i / 24) + 0.002 * i % 1.5 + 0.02 * math.sin(i * 0.37)
        out.append((key, round(level, 3)))
    return out


def _client_with_screens(tmp_path: Path) -> TestClient:
    client = _client(tmp_path)
    install_screens_api(client.app)
    return client


# ── 모듈 ───────────────────────────────────────────────────────────────────────────


def test_parse_hourly_payload_handles_full_and_truncated_json() -> None:
    full = json.dumps({"list": [{"ymdh": "2026101801", "wl": "72.01"}, {"ymdh": "2026101802", "wl": ""}]})
    assert H.parse_hourly_payload(full) == [("2026101801", 72.01), ("2026101802", None)]
    cut = '{"list":[{"ymdh":"2026101801","wl":"72.01"},{"ymdh":"2026101802","wl":"72.0'
    assert H.parse_hourly_payload(cut) == [("2026101801", 72.01)]


def test_parse_ymdh_maps_hour_24_to_next_midnight() -> None:
    assert H.parse_ymdh("2024123124") == datetime(2025, 1, 1, 0, 0)
    assert H.parse_ymdh("2024010101") == datetime(2024, 1, 1, 1, 0)


def test_to_grid_fills_short_gaps_only() -> None:
    rows = [[k, v] for k, v in _rows(datetime(2024, 1, 1, 1), 48, gap_at=(10, 12))]  # 2시간 결측
    rows += [[k, v] for k, v in _rows(datetime(2024, 1, 3, 1), 24)]                  # 앞 구간 끝과 이어짐
    rows_long = [[k, v] for k, v in _rows(datetime(2024, 1, 1, 1), 48, gap_at=(10, 20))]  # 10시간 결측
    times, y = H.to_grid(rows)
    assert len(times) == 72 and np.isfinite(y).all()
    times2, y2 = H.to_grid(rows_long)
    assert len(times2) == 48 and np.isnan(y2[10:20]).all() and np.isfinite(y2[:10]).all()


def test_merge_hourly_replaces_same_hour_and_keeps_existing_on_none() -> None:
    first = H.merge_hourly(None, [("2024010101", 1.0), ("2024010102", 1.1)], station={"id": "A", "name": "A", "code": "1234567"})
    assert first["n"] == 2 and first["period"] == ["2024010101", "2024010102"] and first["n_new"] == 2
    second = H.merge_hourly(first, [("2024010102", 1.5), ("2024010103", None), ("2024010104", 1.7)], station={"id": "A", "name": "A", "code": "1234567"})
    assert dict((k, v) for k, v in second["rows"]) == {"2024010101": 1.0, "2024010102": 1.5, "2024010104": 1.7}
    assert second["n_new"] == 1 and second["schema"] == H.HOURLY_SCHEMA


def test_forecaster_falls_back_to_persistence_when_short_and_fits_ridge_when_long() -> None:
    short = H.merge_hourly(None, _rows(datetime(2024, 1, 1, 1), 24 * 10), station={"id": "A"})
    fc = H.HourlyForecaster.from_doc(short)
    out = fc.forecast(as_of=None, horizon_hours=6)
    assert out["model"]["id"] == "persistence" and all(s["method"] == "persistence" for s in out["steps"])
    assert out["steps"][-1]["forecast_m"] == out["origin_level_m"]

    long = H.merge_hourly(None, _rows(datetime(2024, 1, 1, 1), 24 * 60), station={"id": "A"})
    fc = H.HourlyForecaster.from_doc(long)
    origin = datetime(2024, 2, 20, 12)
    out = fc.forecast(as_of=origin, horizon_hours=24, history_hours=48)
    assert out["as_of"] == "2024-02-20T12:00" and out["horizon_hours"] == 24 and len(out["steps"]) == 24
    assert out["model"]["id"] == "ridge_ar_hourly" and set(out["model"]["val_skill_by_h"]) == {str(h) for h in range(1, 25)}
    assert all(s["method"] == "ridge_ar_hourly" and s["lower_1sigma_m"] <= s["forecast_m"] <= s["upper_1sigma_m"] for s in out["steps"])
    assert out["verification"]["n"] == 24 and out["verification"]["rmse_m"] >= 0
    assert out["steps"][0]["target_time"] == "2024-02-20T13:00" and len(out["history"]) == 48
    # 하루 주기가 뚜렷한 합성 자료라 직전값 유지보다 나아야 한다
    assert out["verification"]["rmse_m"] < out["verification"]["persistence_rmse_m"]


def test_forecaster_origin_rules() -> None:
    doc = H.merge_hourly(None, _rows(datetime(2024, 1, 1, 1), 24 * 5), station={"id": "A"})
    fc = H.HourlyForecaster.from_doc(doc)
    with pytest.raises(ValueError):
        fc.origin_index(datetime(2023, 12, 1))
    assert fc.times[fc.origin_index(datetime(2024, 1, 3))] == datetime(2024, 1, 3, 0, 0)
    assert fc.times[fc.origin_index(datetime(2030, 1, 1))] == fc.times[-1]


def test_holdout_eval_reports_by_horizon() -> None:
    doc = H.merge_hourly(None, _rows(datetime(2024, 1, 1, 1), 24 * 70), station={"id": "A"})
    ev = H.holdout_eval(doc)
    assert ev["status"] == "ok" and set(ev["by_horizon"]) == {"1", "3", "6", "12", "24"}
    assert all(v["n_test"] > 24 and v["rmse_m"] >= 0 and v["persistence_rmse_m"] >= 0 for v in ev["by_horizon"].values())
    assert H.holdout_eval(H.merge_hourly(None, _rows(datetime(2024, 1, 1, 1), 24 * 20), station={"id": "A"}))["status"] == "insufficient"


def test_wamis_fetch_hourly_chunks_by_month_and_records_failures() -> None:
    calls: list[str] = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        if "startdt=20240201" in url:
            return "__error__ timeout"
        return json.dumps({"list": [{"ymdh": "2024010101", "wl": "1.0"}]})

    rows, failures = H.wamis_fetch_hourly("1234567", datetime(2024, 1, 15).date(), datetime(2024, 3, 10).date(), fetch=fake_fetch)
    assert len(calls) == 3 and "startdt=20240115&enddt=20240131" in calls[0] and "startdt=20240301&enddt=20240310" in calls[2]
    assert len(rows) == 2 and failures == ["2024-02: __error__ timeout"]


# ── API ────────────────────────────────────────────────────────────────────────────


def test_collect_hourly_sync_stores_file_then_status_and_forecast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_screens(tmp_path)
    monkeypatch.setattr(screen, "fetch_hourly_rows", lambda code, start, end: (_rows(datetime(2024, 1, 1, 1), 24 * 60), ["2024-03: 0 rows"]))

    r = client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [{"id": "A"}], "start": "2024-01-01", "end": "2024-03-01", "async_run": False})
    assert r.status_code == 422, r.text  # 장난감 지점에는 source_code 가 없다 → code 필요
    r = client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [{"id": "A", "code": "2022680"}], "start": "2024-01-01", "end": "2024-03-01", "async_run": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "done" and body["kind"] == "gauge_hourly_collect" and body["n_done"] == 1
    st = body["stations"][0]
    assert st["status"] == "done" and st["n_total"] == 24 * 60 and st["code"] == "2022680" and "1개 달" in st["note"]
    path = tmp_path / "user_regions" / "toy" / "gauge" / "hourly" / "A.json"
    assert path.is_file() or any(p.name == "A.json" for p in tmp_path.rglob("hourly/A.json"))

    r = client.get("/api/v1/regions/toy/gauge/hourly")
    assert r.status_code == 200
    items = {i["station_id"]: i for i in r.json()["stations"]}
    assert items["A"]["available"] and items["A"]["n_hours"] == 24 * 60 and items["A"]["enough_to_forecast"] and not items["B"]["available"]

    r = client.get("/api/v1/regions/toy/gauge/A/hourly?with_eval=true")
    assert r.status_code == 200 and r.json()["holdout"]["status"] == "ok"

    r = client.get("/api/v1/regions/toy/gauge/A/forecast-hourly?as_of=2024-02-20T12:00&horizon_hours=6")
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["unit"] == "hour" and f["as_of"] == "2024-02-20T12:00" and len(f["steps"]) == 6 and f["verification"]["n"] == 6
    assert f["station_id"] == "A" and f["model"]["id"] == "ridge_ar_hourly" and f["as_of_clamped"] is False

    r = client.get("/api/v1/regions/toy/gauge/A/forecast-hourly?as_of=2031-01-01&horizon_hours=3")
    assert r.status_code == 200 and r.json()["as_of_clamped"] is True and "자료 끝" in r.json()["origin_note_ko"]
    r = client.get("/api/v1/regions/toy/gauge/A/forecast-hourly?as_of=2023-01-01")
    assert r.status_code == 422
    r = client.get("/api/v1/regions/toy/gauge/A/forecast-hourly?as_of=not-a-time")
    assert r.status_code == 422
    r = client.get("/api/v1/regions/toy/gauge/B/forecast-hourly")
    assert r.status_code == 404
    r = client.get("/api/v1/regions/nope/gauge/hourly")
    assert r.status_code == 404


def test_collect_hourly_async_polls_and_refuses_concurrent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_screens(tmp_path)
    gate = threading.Event()

    def blocking(code, start, end):
        gate.wait(5)
        return _rows(datetime(2024, 1, 1, 1), 24 * 3), []

    monkeypatch.setattr(screen, "fetch_hourly_rows", blocking)
    r = client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [{"id": "A", "code": "2022680"}], "start": "2024-01-01", "end": "2024-01-03"})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status_url"].endswith(f"/regions/toy/gauge/collect-hourly/jobs/{job['job_id']}")
    r = client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [{"id": "A", "code": "2022680"}], "start": "2024-01-01", "end": "2024-01-03"})
    assert r.status_code == 409
    gate.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        payload = client.get(job["status_url"]).json()
        if payload["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert payload["status"] == "done" and payload["stations"][0]["n_total"] == 72
    assert client.get("/api/v1/regions/toy/gauge/collect-hourly/jobs").json()["total"] == 1
    assert client.get("/api/v1/regions/toy/gauge/collect-hourly/jobs/zzz").status_code == 404


def test_collect_hourly_validation(tmp_path: Path) -> None:
    client = _client_with_screens(tmp_path)
    assert client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [], "start": "2024-01-01"}).status_code == 422
    assert client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [{"id": "A", "code": "12"}], "start": "2024-01-01"}).status_code == 422
    assert client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [{"id": "A", "code": "2022680"}], "start": "2024-02-01", "end": "2024-01-01"}).status_code == 422
    assert client.post("/api/v1/regions/toy/gauge/collect-hourly", json={"stations": [{"id": "A", "code": "2022680"}], "start": "2015-01-01", "end": "2024-01-01"}).status_code == 422
