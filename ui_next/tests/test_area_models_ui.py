"""수체 면적 시계열 모델 비교 실험 화면(ui_next/area_models_ui.py) — 순수 표 도우미와 AppTest.

실험 JSON 은 고정 자료(fixtures/area_models_experiment.json)로, 서버는 가짜 클라이언트로 대신한다
(계약: GET /regions/{id}/gauge/experiments → {items: {area_models: <json | null>}}).

규율(화면이 반드시 말해야 하는 것):
- 면적은 **공통 격자**에서 다시 잰 값이다 — 장면별 면적은 계열이 아니다.
- 관측이 4~10개뿐이라 **모델 우열을 가릴 수 없다**.
- 검량곡선이 문턱을 넘지 못하면 파생 면적 모델을 **공개하지 않는다**.
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

import area_models_ui as am  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "area_models_experiment.json"
PAYLOAD = json.loads(FIXTURE.read_text(encoding="utf-8"))


# --- 순수 도우미 --------------------------------------------------------------------------------


def test_sensors_and_blocks_tolerate_missing_payload() -> None:
    assert am.sensors_available(PAYLOAD) == ["ICEYE"]
    assert am.sensors_available(None) == [] and am.sensors_available({}) == []
    assert am.series_blocks({"sensors": {"S": {"n": 1}}}) == {"S": {"n": 1}}
    assert am.series_blocks({"series": "nope"}) == {}


def test_candidate_label_uses_name_and_station() -> None:
    assert am.candidate_label("persistence") == "직전 관측 유지 (기준선)"
    assert am.candidate_label("level_linear:가교") == "같은 날 수위 선형회귀 — 가교"
    assert am.candidate_label("level_isotonic:나교", {"name_ko": "같은 날 수위 단조회귀 (PAVA)", "level_station": "나교"}) == "같은 날 수위 단조회귀 (PAVA) — 나교"
    assert am.candidate_label("unknown_thing") == "unknown_thing"


def test_sensor_rows_summarise_series() -> None:
    rows = am.sensor_rows(PAYLOAD)
    assert len(rows) == 1
    row = rows[0]
    assert row["센서"] == "ICEYE" and row["관측 수"] == 4
    assert row["기간"] == "2020-03-02 ~ 2020-04-16"
    assert row["면적 평균"] == "38.08 km²" and row["면적 변동폭"] == "3.50 km²"
    assert row["직전값 유지 LOO RMSE"] == "2.800 km²"
    assert row["최저 LOO 후보"] == "계열 평균" and row["그 RMSE"] == "2.060 km²"
    assert row["관측 잡음 하한"] == "2.40 km²"  # 1일 재방문 면적 차 — 계열 변동폭 3.5 의 대부분
    assert am.sensor_rows(None) == []


def test_candidate_table_rows_ordered_and_signed() -> None:
    rows = am.candidate_table_rows(PAYLOAD, "ICEYE")
    assert [r["후보"] for r in rows] == ["계열 평균", "직전 관측 유지 (기준선)", "같은 날 수위 선형회귀 — 가교"]
    assert rows[0]["LOO RMSE"] == "2.060 km²" and rows[0]["직전값 유지 대비"] == "+45.9 %"
    assert rows[0]["이긴 점 / 진 점"] == "3 / 1" and rows[0]["부호검정 p"] == "0.625"
    assert rows[1]["부호검정 p"] == "-"  # 기준선 자신은 검정하지 않는다
    assert rows[2]["직전값 유지 대비"] == "-47.4 %"
    assert am.candidate_table_rows(PAYLOAD, "없는센서") == []
    # ranking 이 없으면 RMSE 순으로 스스로 정렬한다
    stripped = {"series": {"ICEYE": {k: v for k, v in PAYLOAD["series"]["ICEYE"].items() if k != "ranking_by_loo_rmse"}}}
    assert [r["후보"] for r in am.candidate_table_rows(stripped, "ICEYE")][0] == "계열 평균"


def test_points_table_shows_common_frame_next_to_scene_footprint() -> None:
    rows = am.points_table_rows(PAYLOAD, "ICEYE")
    assert len(rows) == 4 and list(rows[0]) == ["날짜", "공통 격자 면적", "장면 자기 범위 면적", "가교 수위"]
    assert rows[0]["공통 격자 면적"] == "40.000 km²" and rows[0]["장면 자기 범위 면적"] == "40.100 km²"
    assert rows[0]["가교 수위"] == "2.000 m"


def test_bridge_table_and_defensibility() -> None:
    rows = am.bridge_table_rows(PAYLOAD)
    assert len(rows) == 1
    row = rows[0]
    assert row["판정"] == "탈락" and "기울기 > 0" in row["탈락 사유"]
    assert row["상관 r"] == "-0.200" and row["기울기"] == "-3.0 km²/m"
    assert am.bridge_defensible(PAYLOAD) is False
    assert am.bridge_defensible({"bridge": {"any_defensible": True}}) is True
    assert am.bridge_defensible(None) is False and am.bridge_table_rows(None) == []


def test_bridge_demo_rows_pair_area_skill_with_level_skill() -> None:
    rows = am.bridge_demo_rows(PAYLOAD)
    assert [r["지평"] for r in rows] == ["D+1", "D+7"]
    assert rows[0]["개선율"] == rows[0]["같은 분할 수위 모델 개선율"] == "+31.7 %"  # 아핀 변환이라 skill 이 같다
    meta = am.bridge_demo_meta(PAYLOAD)
    assert meta["published"] is False and meta["level_station"] == "가교"
    assert meta["rating_loo_rmse_km2"] == 3.1 and meta["warnings_ko"]
    assert am.bridge_demo_rows({"bridge": {}}) == [] and am.bridge_demo_meta(None) == {}


def test_end_to_end_rows_decompose_error() -> None:
    rows = am.end_to_end_rows(PAYLOAD)
    assert len(rows) == 2
    first = rows[0]
    assert first["검량 오차"] == "-2.00 km²" and first["수위 예보 오차"] == "+0.30 km²" and first["총오차"] == "-1.70 km²"
    assert first["분할"] == "val"  # 위성 관측일은 검증 구간이라 학습 구간 파라미터로만 예보했다
    assert am.end_to_end_rows(None) == []


def test_notes_and_verdict_passthrough() -> None:
    assert am.notes_ko(PAYLOAD)[0].startswith("면적 계열은")
    assert "검증되지 않는다" in am.verdict_ko(PAYLOAD)[-1]
    assert am.notes_ko(None) == [] and am.verdict_ko(None) == []
    assert am.notes_ko({"notes_ko": "한 줄"}) == ["한 줄"]


def test_region_summary_rows_for_deck() -> None:
    rows = am.region_summary_rows({"toy": PAYLOAD, "none": None, "empty": {}}, {"toy": "토이 유역"})
    assert rows == [["토이 유역", "ICEYE", "4", "2.80", "계열 평균", "2.06", "+45.9 %", "탈락 (모델 미공개)"]]
    assert am.region_summary_rows({"a": None}, {}) == []


def test_component_is_wired_into_slides_capture_and_deck() -> None:
    slides = (UI_DIR / "slides.py").read_text(encoding="utf-8")
    assert 'render_area_models_experiment(client, region, key="sl_ev_am")' in slides
    assert slides.index("render_future_rain_experiment(") < slides.index("render_area_models_experiment(") < slides.index("winners = winners_table(evaluation)")
    capture = (UI_DIR.parent / "scripts" / "capture_ui_screens.py").read_text(encoding="utf-8")
    assert "19h_slide6_area_models.png" in capture
    assert capture.index("19g_slide6_future_rain.png") < capture.index("19h_slide6_area_models.png")
    assert capture.index("수체 면적 시계열 모델") < capture.index("19h_slide6_area_models.png")  # 펼친 뒤 찍는다
    deck = (UI_DIR.parent / "scripts" / "make_results_deck.py").read_text(encoding="utf-8")
    assert "수체 면적 시계열은 persistence 말고 다른 모델로 검증되나" in deck and "19h_slide6_area_models.png" in deck
    assert deck.index("미래 강수(예보)를 넣으면 — 완전 예보 가정 상한 실험") < deck.index("수체 면적 시계열은 persistence 말고") < deck.index("한계와 다음 단계")
    assert "area_models.json" in deck
    service = (UI_DIR.parent / "backend" / "app" / "gauge" / "service.py").read_text(encoding="utf-8")
    assert '"area_models": "area_models.json"' in service
    docs = (UI_DIR.parent / "backend" / "app" / "api_docs.py").read_text(encoding="utf-8")
    assert "watercast.satellite.experiment.area-models/1" in docs


# --- AppTest ----------------------------------------------------------------------------------


def _experiment_app(ui_dir: str, fixture: str, mode: str) -> None:  # AppTest.from_function 이 이 본문을 스크립트로 실행한다
    import json
    import sys

    import streamlit as st

    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    from api_client import APIError
    from area_models_ui import render_area_models_experiment

    payload = json.loads(open(fixture, encoding="utf-8").read())
    calls = st.session_state.setdefault("fake_calls", [])

    class FakeClient:
        def gauge_experiments(self, rid):
            calls.append(rid)
            if mode == "missing":
                return {"items": {"area_models": None}}
            if mode == "error":
                raise APIError("404 Not Found", status_code=404)
            return {"items": {"area_models": payload}}

    render_area_models_experiment(FakeClient(), {"id": "toy", "name": "토이 유역"}, key="t_am")


def _run(mode: str) -> AppTest:
    app = AppTest.from_function(_experiment_app, kwargs={"ui_dir": str(UI_DIR), "fixture": str(FIXTURE), "mode": mode}, default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_apptest_renders_tables_and_honesty_text() -> None:
    app = _run("ok")
    assert app.expander[0].label == am.EXPANDER_TITLE
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "공통 격자" in markdown and "LOO" in markdown
    assert "모델 우열을 가릴 수 없습니다" in markdown and "0.125" in markdown
    assert "포함하지 않습니다" in markdown  # 공식 성적표와 별개
    assert "공개하지 않습니다" in markdown  # 다리가 문턱을 못 넘었다
    assert "센서별 요약" in markdown and "후보별 LOO 성적" in markdown and "검량곡선 판정" in markdown
    assert "시연(공개 아님)" in markdown and "끝단 대조" in markdown
    assert "검증되지 않는다" in markdown and "<b>검증되지 않는다</b>" in markdown  # 메모의 **굵게** 는 HTML 로
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["관측 수 (LOO 시험점)"] == "4" and metrics["최저 LOO RMSE"] == "2.060 km²"
    assert metrics["유의한 후보"] == "0 개"
    tables = [t.value for t in app.table]
    assert len(tables) == 6  # 센서 요약 · 관측 · 후보 · 다리 · 시연 · 끝단
    assert list(tables[2]["후보"])[0] == "계열 평균"
    assert list(tables[5]["총오차"])[0] == "-1.70 km²"
    assert app.session_state["fake_calls"] == ["toy"]
    app.run()  # 두 번째 실행은 캐시를 써서 API 를 다시 부르지 않는다
    assert not app.exception and app.session_state["fake_calls"] == ["toy"]
    assert "experiments:area_models:toy" in app.session_state["gauge_cache"]


def test_apptest_missing_experiment_and_api_error_are_graceful() -> None:
    app = _run("missing")
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "아직 면적 모델 실험 결과가 없습니다" in markdown and "eval_area_models_experiment.py" in markdown
    assert "위성 공통 격자 마스크가 있는 지역에만" in markdown
    assert not app.table and not app.metric
    assert "공통 격자" in markdown  # 설명 상자는 결과가 없어도 보인다
    app = _run("error")
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "불러오지 못했습니다" in markdown and "404" in markdown and not app.table and not app.error


REAL_BUSAN = UI_DIR.parent / "regions" / "nakdong_busan" / "models" / "gauge" / "experiments" / "area_models.json"


@pytest.mark.skipif(not REAL_BUSAN.is_file(), reason="실험 산출물(area_models.json)이 없다")
def test_real_experiment_json_matches_scenes_and_its_own_claims() -> None:
    """스크립트가 실제로 남긴 JSON 을 같은 도우미로 읽고, scenes.json 과 대조한다(계약 회귀)."""

    payload = json.loads(REAL_BUSAN.read_text(encoding="utf-8"))
    assert payload["schema"] == "watercast.satellite.experiment.area-models/1"
    scenes = json.loads((UI_DIR.parent / "regions" / "nakdong_busan" / "satellite" / "scenes.json").read_text(encoding="utf-8"))["scenes"]
    by_scene = {(s["sensor"], s["date"]): s for s in scenes}
    assert am.sensors_available(payload) == ["ICEYE", "PlanetScope"]
    for sensor, block in am.series_blocks(payload).items():
        assert block["n"] == len(block["points"]) == sum(1 for s in scenes if s["sensor"] == sensor)
        for point in block["points"]:
            scene = by_scene[(sensor, point["date"])]
            assert point["scene_own_footprint_area_km2"] == pytest.approx(scene["water_area_km2"])
            assert 0 < point["area_km2"] <= block["frame"]["frame_area_km2"]
        # 공통 격자 면적은 장면 자기 범위 면적과 다르다 — 그 차이가 이 실험의 출발점이다
        assert any(abs(p["area_km2"] - p["scene_own_footprint_area_km2"]) > 1.0 for p in block["points"]) or sensor == "ICEYE"
        assert am.candidate_table_rows(payload, sensor)
        assert block["candidates"]["persistence"]["skill_vs_persistence"] == 0.0
    # 2020-03-25 광학 장면: 자기 범위로는 65 km² 지만 공통 격자로는 30 km² 대 — 급증은 계열 문제였다
    planet = am.series_blocks(payload)["PlanetScope"]
    spike = next(p for p in planet["points"] if p["date"] == "2020-03-25")
    assert spike["scene_own_footprint_area_km2"] > 60 and 28 < spike["area_km2"] < 32
    # 표본이 작아 유의한 후보가 없어야 하고, 다리는 공개되지 않아야 한다
    assert all(not b["any_candidate_significant_vs_persistence"] for b in am.series_blocks(payload).values())
    assert am.bridge_defensible(payload) is False
    demo = am.bridge_demo_meta(payload)
    assert demo and demo["published"] is False and demo["warnings_ko"]
    for row in am.bridge_demo_rows(payload):  # 파생 면적 skill 과 수위 skill 이 같다
        assert row["개선율"] == row["같은 분할 수위 모델 개선율"]
