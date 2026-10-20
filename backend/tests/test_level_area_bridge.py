"""``level-area-bridge`` 어댑터 — 면적을 수위 전망에 종속시키는 다리 모델 (2026-10-15).

토이 지역(test_gauge.write_region)에서 지점 A 를 빠르게 학습한 뒤, 기존 예측 API(``POST /predictions``)로
마스크 2장을 넣고 (1) 기울기 옵션이 주어지면 날짜별 면적이 수위 전망 변화에 비례해 달라지는지, (2) 기울기 0 이면
면적은 직전값과 같은지, (3) 메타데이터·경고가 계약대로 나오는지, (4) 수위도 함께 나오는지 본다.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.adapters.level_area_bridge import fit_slope
from backend.tests.test_gauge import _client


def _png(size: int = 24, water: int = 8) -> bytes:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[4 : 4 + water, 4 : 4 + water] = 1
    stream = BytesIO()
    Image.fromarray(mask * 255).save(stream, format="PNG")
    return stream.getvalue()


def _predict(client, model_options: dict, horizon: int = 5, pixel_area: float = 2500.0):
    png = _png()
    form = {
        "model_id": "level-area-bridge", "horizon_steps": str(horizon), "threshold": "0.5", "pixel_area_m2": str(pixel_area),
        "source_dates_json": '["2023-06-01","2023-06-02"]', "historical_water_levels_json": "[null,null]",
        "model_options_json": json.dumps(model_options), "weather_json": "[]",
    }
    return client.post("/api/v1/predictions", files=[("files", ("m1.png", png, "image/png")), ("files", ("m2.png", png, "image/png"))], data=form)


def test_fit_slope_guards() -> None:
    assert fit_slope([1.0, 2.0], [1.0, 2.0]) == (None, None, 2)          # 쌍 3개 미만
    assert fit_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == (None, None, 3)  # 수위 변동 없음
    slope, corr, n = fit_slope([70.0, 71.0, 72.0, 73.0], [10.0, 10.5, 11.0, 11.5])
    assert n == 4 and abs(slope - 0.5) < 1e-9 and abs(corr - 1.0) < 1e-9


def test_level_area_bridge_scales_area_with_forecast_level(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert any(m["id"] == "level-area-bridge" for m in client.get("/api/v1/models").json())
    trained = client.post("/api/v1/regions/toy/gauge/train", json={"stations": ["A"], "forecasters": ["mean_reversion"], "horizon_max": 5})
    assert trained.status_code == 200, trained.text

    # 기울기 2 km²/m, 화소 2500 m² → 1 m 수위 변화 = 800 화소
    response = _predict(client, {"region_id": "toy", "station_id": "A", "slope_km2_per_m": 2.0, "as_of": "2023-06-02"})
    assert response.status_code == 201, response.text
    payload = response.json()
    meta = payload["adapter_metadata"]
    assert meta["model"] == "level-area-bridge" and meta["slope_km2_per_m"] == 2.0 and meta["slope_source"] == "option"
    assert meta["gauge_station_id"] == "A" and meta["gauge_as_of"] == "2023-06-02" and meta["anchor"] == "last"
    origin = meta["origin_level_m"]
    seed_km2 = 64 * 2500 / 1e6  # 입력 마스크 8×8 화소
    assert abs(meta["anchor_area_km2"] - seed_km2) < 1e-6
    levels = [s["water_level_m"] for s in payload["steps"]]
    assert all(isinstance(v, float) for v in levels) and payload["steps"][0]["water_level_source"] == "adapter_output"
    for step, detail in zip(payload["steps"], meta["steps"]):
        expected = max(0.0, seed_km2 + 2.0 * (step["water_level_m"] - origin))
        assert abs(detail["target_area_km2"] - expected) < 1e-3
        # 마스크 면적은 목표 면적에 화소 단위로 맞춘다(마스크가 전부 마르거나 캔버스를 넘기지 않는 한)
        assert abs(step["water_area_km2"] - expected) < 2500 / 1e6 + 1e-6
    assert len({round(s["water_area_km2"], 6) for s in payload["steps"]}) > 1, "수위가 변하면 면적도 변해야 한다"

    # 기울기 0 → 면적은 출발값 그대로, 수위는 여전히 전망값
    flat = _predict(client, {"region_id": "toy", "station_id": "A", "slope_km2_per_m": 0, "as_of": "2023-06-02"}).json()
    assert {round(s["water_area_km2"], 6) for s in flat["steps"]} == {round(seed_km2, 6)}
    assert len({round(s["water_level_m"], 6) for s in flat["steps"]}) > 1

    # 평균회귀 출발점: 두 프레임 면적이 같으므로 출발 면적은 그대로지만 anchor 가 기록된다
    mr = _predict(client, {"region_id": "toy", "station_id": "A", "slope_km2_per_m": 1.0, "anchor": "mean_reversion"}).json()
    assert mr["adapter_metadata"]["anchor"] == "mean_reversion" and abs(mr["adapter_metadata"]["anchor_area_km2"] - seed_km2) < 1e-6

    # 기울기 자료가 없는 토이 지역(면적 계열 없음)은 0 으로 두고 경고를 남긴다
    none = _predict(client, {"region_id": "toy", "station_id": "A"}).json()
    assert none["adapter_metadata"]["slope_source"] == "none" and any("기울기" in w for w in none["warnings"])

    # 모르는 지점: 예측 API 는 다른 어댑터와 같이 실패 기록(status=failed, error)으로 돌려준다
    bad = _predict(client, {"region_id": "toy", "station_id": "없는지점", "slope_km2_per_m": 1.0})
    assert bad.status_code in (201, 404, 422), bad.text
    if bad.status_code == 201:
        body = bad.json()
        assert body["status"] == "failed" and "없는지점" in (body.get("error") or "") and body["steps"] == []
