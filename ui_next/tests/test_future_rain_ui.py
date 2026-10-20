"""미래 강수(완전 예보 가정) 상한 실험 화면(ui_next/future_rain_ui.py) — 순수 표 도우미와 AppTest.

실험 JSON 은 고정 자료(fixtures/future_rain_experiment.json)로 대신하고, 서버는 가짜 클라이언트로 대신한다
(계약: GET /regions/{id}/gauge/experiments → {items: {future_rain: <json | null>}}). 규율: 완전 예보 가정은
상한이며 공식 성적표에 들어가지 않는다 — 화면 문구가 그 사실을 말해야 한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:  # slides.py 처럼 ui_next/ 를 sys.path 에 둔 채 평면 import 한다
    sys.path.insert(0, str(UI_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

import future_rain_ui as fr  # noqa: E402
from api_client import APIClient, Endpoints  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "future_rain_experiment.json"
PAYLOAD = json.loads(FIXTURE.read_text(encoding="utf-8"))


# --- 순수 도우미 --------------------------------------------------------------------------------


def test_classify_variant_names() -> None:
    assert fr.classify_variant("persistence") == "persistence"
    assert fr.classify_variant("ridge_arx") == "ridge_arx"
    assert fr.classify_variant("models/ridge_arx_future/3") == "future_3"
    assert fr.classify_variant("ridge_arx_future_3d") == "future_3"
    assert fr.classify_variant("ridge_arx_future/all") == "future_all"
    assert fr.classify_variant("future_rain_all") == "future_all"
    assert fr.classify_variant("abs_error/1/ridge_arx_future_all") == "future_all"
    assert fr.classify_variant("climatology") is None
    assert fr.classify_variant("ridge_arx_future") == "future_all"  # 변형 표기 없는 future = 이벤트 블록의 future(all)


def test_experiment_rows_from_nested_stations_fixture() -> None:
    rows = fr.experiment_rows(PAYLOAD)
    assert len(rows) == 6 and fr.horizons_available(rows) == [1, 3, 7]
    row = next(r for r in rows if r["station_id"] == "s1" and r["horizon"] == 7)
    assert row["station_name"] == "가교" and row["terrain"] == "mainstem" and row["n"] == 360
    assert row["metrics"]["persistence"]["rmse"] == pytest.approx(0.30)
    assert row["metrics"]["ridge_arx"]["skill"] == pytest.approx(0.19) and row["metrics"]["future_all"]["skill_gain_pp"] == 56.0
    assert row["metrics"]["future_3"]["rmse"] == pytest.approx(0.24)
    assert row["metrics"]["future_all"]["rmse"] == pytest.approx(0.15) and row["metrics"]["future_all"]["mae"] == pytest.approx(0.075)
    assert row["delta_rmse_cm"] == {"future_3": 3.0, "future_all": 12.0}
    assert fr.experiment_rows(None) == [] and fr.experiment_rows({}) == []


def test_experiment_rows_accepts_flat_rows_and_alias_keys() -> None:
    """백엔드가 평면 rows 배열·test_rmse/skill_vs_persistence 별칭을 써도 같은 행이 나와야 한다."""

    flat = {
        "rows": [
            {"station_id": "x", "station_name": "X교", "horizon": "D+7", "test_n": 10,
             "persistence": {"test_rmse": 0.2, "test_mse": 0.04, "test_mae": 0.1},
             "ridge_arx": {"test": {"rmse": 0.18, "mse": 0.0324, "mae": 0.09}, "skill_vs_persistence": 0.19},
             "ridge_arx_future_3": {"test_rmse": 0.15, "test_mse": 0.0225, "skill_vs_persistence": 0.44},
             "ridge_arx_future_all": {"test_rmse": 0.10, "test_mse": 0.01, "skill_vs_persistence": 0.75}},
            {"station_id": "y", "station_name": "Y교", "horizon": 1,
             "persistence_rmse": 0.1, "ridge_arx_rmse": 0.09, "ridge_arx_skill": 0.19, "future_3_rmse": 0.08, "future_3_skill": 0.36, "future_all_rmse": 0.08, "future_all_skill": 0.36},
        ]
    }
    rows = fr.experiment_rows(flat)
    assert [(r["station_id"], r["horizon"]) for r in rows] == [("y", 1), ("x", 7)]
    x = rows[1]["metrics"]
    assert x["persistence"]["rmse"] == 0.2 and x["ridge_arx"]["rmse"] == 0.18 and x["ridge_arx"]["skill"] == 0.19
    assert x["future_3"]["skill"] == 0.44 and x["future_all"]["rmse"] == 0.10
    y = rows[0]["metrics"]
    assert y["persistence"] == {"rmse": 0.1} and y["future_all"] == {"rmse": 0.08, "skill": 0.36}
    # 지점 사전(dict) + 지평 목록 + 변형을 ridge_arx_future: {3, all} 로 묶은 형태 + 셀 수준 delta_rmse_cm
    as_dict = {"stations": {"z": {"station_name": "Z", "horizons": [
        {"horizon": 3, "test_n": 5, "persistence": {"rmse": 1}, "ridge_arx": {"rmse": 0.5, "skill": 0.75},
         "ridge_arx_future": {"3": {"rmse": 0.4, "skill": 0.84}, "all": {"rmse": 0.3, "skill": 0.91}}, "delta_rmse_cm": {"future_3": 10.0, "future_all": 20.0}}]}}}
    z = fr.experiment_rows(as_dict)
    assert z[0]["station_id"] == "z" and z[0]["station_name"] == "Z" and z[0]["horizon"] == 3 and z[0]["n"] == 5
    assert z[0]["metrics"]["ridge_arx"]["skill"] == 0.75 and z[0]["metrics"]["future_3"]["rmse"] == 0.4 and z[0]["metrics"]["future_all"]["skill"] == 0.91
    assert z[0]["delta_rmse_cm"] == {"future_3": 10.0, "future_all": 20.0}


def test_station_table_rows_format_and_gain() -> None:
    rows = fr.experiment_rows(PAYLOAD)
    table = fr.station_table_rows(rows, 7)
    assert [t["지점"] for t in table] == ["가교", "나교"]
    ga = table[0]
    assert list(ga) == ["지점", "직전값 유지 (기준선) RMSE", "과거 강수 AR-X", "+3일 예보 (완전 예보)", "+전 구간 예보 (완전 예보)", "전 구간 예보 이득"]
    assert ga["직전값 유지 (기준선) RMSE"] == "0.300 m (30.0 cm)"
    assert ga["과거 강수 AR-X"] == "0.270 m (27.0 cm) · +19.0 %"
    assert ga["+3일 예보 (완전 예보)"] == "0.240 m (24.0 cm) · +36.0 %"
    assert ga["+전 구간 예보 (완전 예보)"] == "0.150 m (15.0 cm) · +75.0 %"
    assert ga["전 구간 예보 이득"] == "+12.0 cm"
    assert table[1]["전 구간 예보 이득"] == "+0.2 cm"
    # delta 블록이 없으면 RMSE 차이로 계산한다
    stripped = [dict(r, delta_rmse_cm={}) for r in rows]
    assert fr.station_table_rows(stripped, 7)[0]["전 구간 예보 이득"] == "+12.0 cm"
    assert fr.station_table_rows(rows, 30) == []


def test_summary_by_horizon_means_and_beats() -> None:
    rows = fr.experiment_rows(PAYLOAD)
    summary = fr.summary_by_horizon(rows)
    assert sorted(summary) == [1, 3, 7]
    h7 = summary[7]
    assert h7["n_stations"] == 2
    assert h7["mean_skill"]["ridge_arx"] == pytest.approx((0.19 + 0.10) / 2)
    assert h7["mean_skill"]["future_all"] == pytest.approx((0.75 + 0.14) / 2)
    assert h7["mean_rmse"]["persistence"] == pytest.approx((0.30 + 0.095) / 2)
    assert h7["beats_arx"] == {"future_3": 1, "future_all": 2}  # 나교 D+7 은 3일 예보로는 그대로, 전 구간 예보로는 +4 %p
    assert summary[1]["beats_arx"] == {"future_3": 1, "future_all": 1}  # 나교 D+1 은 +0.2 %p → 문턱(0.5 %p) 미만
    assert summary[3]["beats_arx"]["future_all"] == 1  # 나교 D+3 은 오히려 나빠짐


def test_event_rows_join_errors_by_horizon_and_variant() -> None:
    events = fr.event_rows(PAYLOAD)
    assert [(e["지점"], e["날짜"]) for e in events] == [("가교", "2024-07-09"), ("가교", "2024-08-21"), ("나교", "2024-07-10")]
    first = events[0]
    assert first["1일 상승"] == "+1.85 m"
    assert first["D+1 |오차| 직전값"] == "185.0 cm" and first["D+1 |오차| AR-X"] == "160.0 cm" and first["D+1 |오차| +전 구간 예보"] == "40.0 cm"
    assert first["D+3 |오차| +전 구간 예보"] == "55.0 cm"
    assert first["~D+1 강수"] == "0.0 mm" and first["~D+3 강수"] == "41.5 mm"  # future_rain_mm 은 오차가 아니라 강수 열로
    assert list(first) == ["지점", "날짜", "1일 상승", "D+1 |오차| 직전값", "D+1 |오차| AR-X", "D+1 |오차| +전 구간 예보", "~D+1 강수",
                           "D+3 |오차| 직전값", "D+3 |오차| AR-X", "D+3 |오차| +전 구간 예보", "~D+3 강수"]
    assert fr.event_rows(PAYLOAD, station_id="s2") == [events[2]]
    # 다른 표기: 최상위 events 목록 + d1/d3 접두 키 + 부호 있는 오차 + 강수 없음
    alt = {"events": [{"station_id": "q", "station_name": "Q", "top_rises": [{"date": "2024-06-01", "rise": 0.5, "errors": {"d1_persistence": -0.5, "d1_ridge_arx": 0.4, "d1_future": 0.1, "d3_persistence": 0.6, "d3_arx": 0.5, "d3_future_all": 0.2}}]}]}
    alt_rows = fr.event_rows(alt)
    assert alt_rows[0]["D+1 |오차| 직전값"] == "50.0 cm" and alt_rows[0]["D+3 |오차| AR-X"] == "50.0 cm" and alt_rows[0]["D+3 |오차| +전 구간 예보"] == "20.0 cm"
    assert "~D+1 강수" not in alt_rows[0]
    # 최상위 events 사전(지점 id → 목록) 형태
    top = {"events": {"s9": [{"date": "2024-01-02", "rise_m": 0.2, "abs_error_m": {"1": {"persistence": 0.2, "ridge_arx_future_all": 0.1}}}]}}
    assert fr.event_rows(top)[0]["D+1 |오차| +전 구간 예보"] == "10.0 cm" and fr.event_rows(top)[0]["지점"] == "s9"
    assert fr.event_rows({}) == [] and fr.event_rows(None) == []


def test_region_summary_rows_for_deck_handle_missing_regions() -> None:
    rows = fr.region_summary_rows({"toy": PAYLOAD, "none": None, "empty": {}}, {"toy": "토이 유역", "none": "없음"}, horizons=(1, 3, 7, 14, 30))
    assert len(rows) == 2  # 토이 + 전체
    assert rows[0][:2] == ["토이 유역", "2"]
    assert rows[0][2] == "+21 / +44 / +44 %"  # D+1: AR-X / +3일 / +전 구간 (지점 평균)
    assert rows[0][4] == "+15 / +23 / +44 %"  # D+7 (44.5 → 은행가 반올림)
    assert rows[0][5] == "-" and rows[0][6] == "-"  # D+14·30 은 실험에 없음
    assert rows[1][0] == "전체 (지점 가중)" and rows[1][1] == "2" and rows[1][4] == rows[0][4]
    assert fr.region_summary_rows({"a": None}, {}) == []
    assert fr.notes_ko(PAYLOAD)[0].startswith("완전 예보 가정") and fr.notes_ko(None) == []


def test_api_client_experiments_endpoint() -> None:
    calls: list[tuple] = []

    class Fake(APIClient):
        def _request(self, method, path, **kwargs):  # noqa: D401
            calls.append((method, path))
            return {"items": {"future_rain": None}}

    client = Fake("http://x")
    assert client.gauge_experiments("nakdong_busan") == {"items": {"future_rain": None}}
    assert calls == [("GET", "/api/v1/regions/nakdong_busan/gauge/experiments")]
    assert Endpoints.GAUGE_EXPERIMENTS == "/api/v1/regions/{region_id}/gauge/experiments"


def test_component_is_wired_into_slides_gauge_page_capture_and_deck() -> None:
    slides = (UI_DIR / "slides.py").read_text(encoding="utf-8")
    gauge = (UI_DIR / "gauge_forecast_ui.py").read_text(encoding="utf-8")
    assert 'render_future_rain_experiment(client, region, int(horizon), key="sl_ev_fr")' in slides
    assert slides.index("model_summary = evaluation.get") < slides.index("render_future_rain_experiment(") < slides.index("winners = winners_table(evaluation)")
    assert 'render_future_rain_experiment(client, region, int(fr_h), key="gf_fr")' in gauge
    capture = (UI_DIR.parent / "scripts" / "capture_ui_screens.py").read_text(encoding="utf-8")
    assert "19g_slide6_future_rain.png" in capture and "height=900" in capture.split("19g_slide6_future_rain.png")[1][:80]
    assert capture.index("미래 강수(예보)를 넣으면") < capture.index("19g_slide6_future_rain.png")  # 펼친 뒤 찍는다
    deck = (UI_DIR.parent / "scripts" / "make_results_deck.py").read_text(encoding="utf-8")
    assert "미래 강수(예보)를 넣으면 — 완전 예보 가정 상한 실험" in deck and "region_summary_rows" in deck and "19g_slide6_future_rain.png" in deck
    assert deck.index("딥러닝(GPU)·Prophet") < deck.index("미래 강수(예보)를 넣으면 — 완전 예보 가정 상한 실험") < deck.index("한계와 다음 단계")
    assert "완전 예보" in deck.split('d.slide("한계와 다음 단계")')[1]


# --- AppTest ----------------------------------------------------------------------------------


def _experiment_app(ui_dir: str, fixture: str, mode: str, horizon: int) -> None:  # AppTest.from_function 이 이 본문을 스크립트로 실행한다
    import json
    import sys

    import streamlit as st

    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    from api_client import APIError
    from future_rain_ui import render_future_rain_experiment

    payload = json.loads(open(fixture, encoding="utf-8").read())
    calls = st.session_state.setdefault("fake_calls", [])

    class FakeClient:
        def gauge_experiments(self, rid):
            calls.append(rid)
            if mode == "missing":
                return {"items": {"future_rain": None}}
            if mode == "error":
                raise APIError("404 Not Found", status_code=404)
            return {"items": {"future_rain": payload}}

    render_future_rain_experiment(FakeClient(), {"id": "toy", "name": "토이 유역"}, horizon, key="t_fr")


def _run(mode: str, horizon: int = 7) -> AppTest:
    app = AppTest.from_function(_experiment_app, kwargs={"ui_dir": str(UI_DIR), "fixture": str(FIXTURE), "mode": mode, "horizon": horizon}, default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_apptest_renders_tables_and_honesty_text() -> None:
    app = _run("ok", horizon=7)
    assert app.expander[0].label == fr.EXPANDER_TITLE
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "완전 예보 가정" in markdown and "상한" in markdown and "포함하지 않습니다" in markdown and "기상청" in markdown
    assert "D+7 지점별 비교" in markdown and "급상승일 오차 비교" in markdown and "시험 2024년" in markdown
    assert "예보 오차만큼 이득이 줄어든다" in markdown  # notes_ko 가 화면에 보인다
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["D+7 과거 강수 AR-X"] == "+14.5 %" and metrics["D+7 +전 구간 예보"] == "+44.5 %" and metrics["예보가 AR-X 를 앞선 지점"] == "2 / 2"
    tables = app.table
    assert len(tables) == 2
    station = tables[0].value
    assert list(station["지점"]) == ["가교", "나교"] and "0.150 m (15.0 cm) · +75.0 %" in list(station["+전 구간 예보 (완전 예보)"])
    events = tables[1].value
    assert len(events) == 3 and list(events["D+1 |오차| +전 구간 예보"])[0] == "40.0 cm"
    assert "<b>실측</b> 미래 강수" in markdown and "**실측**" not in markdown  # 메모의 **굵게** 는 HTML 로 바뀐다
    # 급상승일 표 지점 필터
    app.selectbox(key="t_fr_ev_station").set_value("나교").run()
    assert not app.exception and len(app.table[1].value) == 1 and list(app.table[1].value["지점"]) == ["나교"]
    assert app.session_state["fake_calls"] == ["toy"]
    # 두 번째 실행은 캐시를 써서 API 를 다시 부르지 않는다
    app.run()
    assert not app.exception and app.session_state["fake_calls"] == ["toy"]
    assert "experiments:future_rain:toy" in app.session_state["gauge_cache"]


def test_apptest_falls_back_to_nearest_horizon() -> None:
    app = _run("ok", horizon=30)
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "D+7 지점별 비교" in markdown and "실험은 D+7 까지만 보고합니다" in markdown


def test_apptest_missing_experiment_and_api_error_are_graceful() -> None:
    app = _run("missing")
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "아직 미래 강수 실험 결과가 없습니다" in markdown and "eval_future_rain_experiment.py" in markdown
    assert not app.table and not app.metric
    assert "완전 예보 가정" in markdown  # 설명 상자는 결과가 없어도 보인다
    app = _run("error")
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "불러오지 못했습니다" in markdown and "404" in markdown and not app.table and not app.error


REAL = UI_DIR.parent / "regions" / "nakdong_busan" / "models" / "gauge" / "experiments" / "future_rain.json"


@pytest.mark.skipif(not REAL.is_file(), reason="실험 산출물(future_rain.json)이 없다")
def test_real_experiment_json_parses_and_matches_its_own_summary() -> None:
    """백엔드 스크립트가 실제로 남긴 JSON 을 같은 도우미로 읽어 요약 블록과 수치가 맞는지 본다(계약 회귀)."""

    payload = json.loads(REAL.read_text(encoding="utf-8"))
    rows = fr.experiment_rows(payload)
    horizons = [int(h) for h in payload.get("report_horizons") or fr.REPORT_HORIZONS]
    assert fr.horizons_available(rows) == horizons
    assert len(rows) == int(payload["n_stations"]) * len(horizons)
    summary = fr.summary_by_horizon(rows)
    by_h = (payload.get("summary") or {}).get("by_horizon") or {}
    for h in horizons:
        theirs = by_h[str(h)]["mean_skill"]
        assert summary[h]["mean_skill"]["ridge_arx"] == pytest.approx(theirs["ridge_arx"], abs=1e-9)
        assert summary[h]["mean_skill"]["future_3"] == pytest.approx(theirs["future_3d"], abs=1e-9)
        assert summary[h]["mean_skill"]["future_all"] == pytest.approx(theirs["future_all"], abs=1e-9)
    beats = (payload.get("summary") or {}).get("cells_future_beats_arx") or {}
    if beats:
        assert sum(summary[h]["beats_arx"]["future_all"] for h in horizons) == beats["future_all"]
        assert sum(summary[h]["beats_arx"]["future_3"] for h in horizons) == beats["future_3d"]
    events = fr.event_rows(payload)
    assert events and all(e["날짜"].startswith(str(payload["split"]["test_start"])[:4]) for e in events)
    assert fr.station_table_rows(rows, 7) and fr.notes_ko(payload)


def _gauge_tab_app(ui_dir: str, fixture: str) -> None:  # 고급 화면 '모델 평가' 탭에 같은 요소가 붙어 있는지
    import json
    import sys

    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    from gauge_forecast_ui import _render_evaluation_tab

    payload = json.loads(open(fixture, encoding="utf-8").read())
    evaluation = {
        "split": {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
        "rows": [{"station_id": "s1", "station_name": "가교", "terrain": "mainstem", "horizon": h, "forecaster_id": "ridge_ar", "val_rmse": 0.1, "test_n": 360,
                  "test_mse": 0.01, "test_rmse": 0.1, "test_mae": 0.05, "persistence_test_rmse": 0.12, "skill_vs_persistence": 0.3} for h in (1, 3, 7)],
        "cells_total": 3, "cells_beating_persistence": 3, "winners_by_terrain": {"mainstem": {"ridge_ar": 3}},
    }

    class FakeClient:
        def gauge_evaluation(self, rid):
            return evaluation

        def gauge_experiments(self, rid):
            return {"items": {"future_rain": payload}}

    _render_evaluation_tab(FakeClient(), {"id": "toy", "name": "토이"}, {"id": "s1", "name": "가교"})


def test_apptest_gauge_page_evaluation_tab_has_the_expander_with_its_own_horizon() -> None:
    app = AppTest.from_function(_gauge_tab_app, kwargs={"ui_dir": str(UI_DIR), "fixture": str(FIXTURE)}, default_timeout=30)
    app.run()
    assert not app.exception
    assert any(e.label == fr.EXPANDER_TITLE for e in app.expander)
    assert app.select_slider(key="gf_fr_horizon").value == 7
    app.select_slider(key="gf_fr_horizon").set_value(3).run()
    assert not app.exception
    assert "D+3 지점별 비교" in " ".join(str(m.value) for m in app.markdown)
    assert {m.label for m in app.metric} >= {"D+3 과거 강수 AR-X", "D+3 +전 구간 예보"}
