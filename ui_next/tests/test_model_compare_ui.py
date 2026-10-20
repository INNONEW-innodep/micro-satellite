"""모델별 비교 구성 요소(ui_next/model_compare_ui.py) — 순수 도우미와 AppTest 스모크.

payload 는 `GET /regions/{id}/gauge/{station}/forecast/compare` 응답 모양을 그대로 흉내 낸 합성 fixture 다
(모델 12종, 검증 있음/없음 두 가지). 로컬 API(127.0.0.1:18765)가 살아 있으면 실제 응답으로도 한 번 더 확인한다.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:  # slides.py 등은 ui_next/ 를 sys.path 에 둔 채 평면 import 한다
    sys.path.insert(0, str(UI_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

import model_compare_ui as mc  # noqa: E402

MODELS = ["analog_knn", "climatology", "ensemble_mean", "mean_reversion", "neural_gru", "neural_lstm", "neural_mlp", "neural_tcn",
          "prophet_hybrid", "ridge_ar", "ridge_arx", "seasonal_anomaly_ar"]
LOCAL_API = "http://127.0.0.1:18765"


def make_payload(*, horizon: int = 7, verified: bool = True, models: list[str] | None = None) -> dict[str, Any]:
    models = list(models or MODELS)
    origin = 2.0
    history = [{"date": f"2024-06-{d:02d}", "water_level_m": origin - 0.01 * (30 - d)} for d in range(1, 31)]
    table = []
    for h in range(1, horizon + 1):
        selected_model = ("ensemble_mean" if h <= 3 else "neural_mlp") if "neural_mlp" in models else models[0]
        offsets = {fid: 0.01 * (i - 4) * h for i, fid in enumerate(models)}  # 모델별로 다른 편차 → 순위가 정해진다
        selected_m = origin + offsets[selected_model]
        observed = origin + 0.004 * h if verified else None
        table.append({
            "horizon": h, "target_date": f"2024-07-{h:02d}", "selected_model": selected_model, "selected_m": selected_m,
            "persistence_m": origin, "observed_m": observed,
            "models": {fid: {"forecast_m": origin + offsets[fid], "diff_vs_selected_m": offsets[fid] - offsets[selected_model]} for fid in models},
        })
    per_model = {}
    for i, fid in enumerate(models):
        rmse = 0.02 + 0.01 * i
        per_model[fid] = {
            "verification": {"n": horizon, "rmse_m": rmse * 3, "mae_m": rmse * 2, "persistence_rmse_m": 0.2, "skill_vs_persistence": 1 - rmse * 3 / 0.2} if verified else None,
            "test_at_horizon": {"horizon": horizon, "n": 366, "mse": rmse ** 2, "rmse": rmse, "mae": rmse * 0.7, "bias": 0.0,
                                "skill_vs_persistence": 1 - rmse / 0.1, "selected": fid == "ensemble_mean"},
        }
    # 한 모델은 시험 성적이 없는 경우(산출물 결손)도 다룬다
    per_model[models[-1]]["test_at_horizon"] = {"horizon": horizon, "selected": False}
    return {
        "region_id": "toy", "station": {"id": "s1", "name": "토이 지점"}, "as_of": "2024-06-30", "horizon": horizon, "in_sample": False,
        "params_note_ko": "합성", "origin_level_m": origin, "history": history, "selected": [], "models": models, "table": table,
        "per_model": per_model, "note_ko": "합성 fixture",
    }


# --- 순위 --------------------------------------------------------------------------------------


def test_metric_options_depend_on_verification() -> None:
    assert mc.has_verification(make_payload(verified=True))
    assert not mc.has_verification(make_payload(verified=False))
    with_ver = mc.metric_options(make_payload(verified=True))
    without = mc.metric_options(make_payload(verified=False))
    assert with_ver[0] == "verify_rmse" and "test_rmse" in with_ver
    assert without[0] == "test_rmse" and not any(m.startswith("verify_") for m in without)
    assert mc.metric_label("test_rmse", 7) == "시험 연도 RMSE · D+7"
    assert mc.metric_label("verify_rmse", 7) == "이번 원점 검증 RMSE"


def test_rank_models_orders_best_first_and_flags_worst() -> None:
    payload = make_payload()
    ranking = mc.rank_models(payload, "test_rmse")
    ids = [e["id"] for e in ranking]
    assert ids[0] == "analog_knn" and ranking[0]["rank"] == 1 and ranking[0]["best"]
    # 시험 성적이 없는 마지막 모델은 순위 없이 맨 뒤
    assert ranking[-1]["id"] == "seasonal_anomaly_ar" and ranking[-1]["rank"] is None and not ranking[-1]["worst"]
    worst = [e for e in ranking if e["worst"]]
    assert len(worst) == 1 and worst[0]["id"] == "ridge_arx" and worst[0]["rank"] == 11
    assert mc.rank_suffix(ranking[0]) == " · 1위 🥇"
    assert mc.rank_suffix(worst[0]) == " · 11위 ⚠"
    assert mc.rank_suffix(ranking[-1]) == ""
    # 개선율은 클수록 좋다 → 같은 fixture 에서 같은 1위
    by_skill = mc.rank_models(payload, "test_skill")
    assert by_skill[0]["id"] == "analog_knn" and by_skill[0]["value"] > by_skill[1]["value"]
    # 이번 원점 검증 기준: MSE 는 RMSE² 로 채운다
    by_ver = mc.rank_models(payload, "verify_rmse")
    assert by_ver[0]["id"] == "analog_knn"
    assert by_ver[0]["verify"]["mse"] == pytest.approx(by_ver[0]["verify"]["rmse"] ** 2)


def test_ranking_rows_show_units_medals_and_selection() -> None:
    payload = make_payload()
    rows = mc.ranking_rows(mc.rank_models(payload, "test_rmse"), "test_rmse")
    assert rows[0]["순위"].startswith("🥇 1위")
    assert rows[1]["순위"].startswith("🥈 2위") and rows[2]["순위"].startswith("🥉 3위")
    assert any("⚠ 최하위" in r["순위"] for r in rows)
    assert rows[0]["시험 RMSE"] == "0.020 m (2.0 cm)"
    assert rows[0]["시험 MSE"] == "0.00040 m² (4.0 cm²)"
    assert rows[0]["시험 MAE"].endswith("cm)")
    selected = next(r for r in rows if "ensemble_mean" in r["모델"] or "앙상블" in r["모델"])
    assert "★" in selected["모델"] and selected["선택"].startswith("★")
    # 값 없는 모델은 대시
    assert rows[-1]["순위"] == "–" and rows[-1]["시험 RMSE"] == "-"


def test_ranking_without_verification_uses_test_metrics_only() -> None:
    payload = make_payload(verified=False)
    ranking = mc.rank_models(payload, "verify_rmse")  # 검증 없음 → 전부 순위 없음, 예외 없음
    assert all(e["rank"] is None for e in ranking)
    rows = mc.ranking_rows(mc.rank_models(payload, "test_mae"), "test_mae")
    assert rows[0]["시험 MAE"] != "-" and "이번 원점 RMSE" not in rows[0]


# --- 보이기 필터 ---------------------------------------------------------------------------------


def test_presets_filter_models() -> None:
    payload = make_payload()
    ranking = mc.rank_models(payload, "test_rmse")
    assert mc.apply_preset(payload, ranking, "전체") == MODELS
    assert mc.apply_preset(payload, ranking, "상위 3") == [e["id"] for e in ranking[:3]]
    assert mc.apply_preset(payload, ranking, "선택 모델만") == ["ensemble_mean", "neural_mlp"]
    assert mc.apply_preset(payload, ranking, "딥러닝만") == ["neural_gru", "neural_lstm", "neural_mlp", "neural_tcn"]
    hidden = mc.apply_preset(payload, ranking, "기준선 숨기기")
    assert "climatology" not in hidden and len(hidden) == len(MODELS) - 1
    # 결과가 비면 전체로 되돌린다
    assert mc.apply_preset(make_payload(models=["ridge_ar", "climatology"]), [], "딥러닝만") == ["ridge_ar", "climatology"]
    assert mc.sanitize_visible(payload, ["neural_mlp", "unknown", "ridge_ar"]) == ["neural_mlp", "ridge_ar"]
    assert mc.sanitize_visible(payload, None) == MODELS


# --- 차이·파고들기 --------------------------------------------------------------------------------


def test_difference_series_in_cm_with_and_without_observed() -> None:
    payload = make_payload(horizon=3)
    series = mc.difference_series(payload, ["ridge_ar", "ensemble_mean"])
    assert series["has_observed"] and series["dates"] == ["2024-07-01", "2024-07-02", "2024-07-03"]
    assert set(series["vs_selected"]) == {"ridge_ar", "ensemble_mean"}
    assert series["vs_selected"]["ensemble_mean"] == [0.0, 0.0, 0.0]  # D+1~3 선택 모델 = 자기 자신
    ridge_index = MODELS.index("ridge_ar")
    assert series["vs_selected"]["ridge_ar"][0] == pytest.approx((ridge_index - MODELS.index("ensemble_mean")) * 1.0)  # 0.01 m·h → cm
    assert series["vs_observed"]["ensemble_mean"][0] == pytest.approx(-2.4)  # 1.98 − 2.004 m
    assert series["observed_vs_selected"][0] == pytest.approx(2.4)
    empty = mc.difference_series(make_payload(horizon=3, verified=False), None)
    assert not empty["has_observed"] and empty["vs_observed"] == {} and empty["observed_vs_selected"] is None
    assert len(empty["vs_selected"]) == len(MODELS)


def test_drilldown_rows_and_summary_both_units() -> None:
    payload = make_payload(horizon=3)
    rows = mc.drilldown_rows(payload, "ridge_ar")
    assert [r["D+"] for r in rows] == [1, 2, 3]
    assert rows[0]["예측"].endswith(" cm)") and " m (" in rows[0]["예측"]
    assert rows[0]["오차 (예측−실제)"].startswith("+") or rows[0]["오차 (예측−실제)"].startswith("-")
    assert "cm (" in rows[0]["선택 대비 차이"] and rows[0]["선택 대비 차이"].endswith(" m)")
    summary = mc.drilldown_summary(payload, "ridge_ar")
    assert summary["n_observed"] == 3 and summary["max_abs_err_cm"] >= summary["mean_abs_err_cm"] > 0
    assert summary["worst_horizon"] == 3
    no_obs = mc.drilldown_summary(make_payload(horizon=3, verified=False), "ridge_ar")
    assert no_obs["n_observed"] == 0 and no_obs["mean_abs_err_cm"] is None and no_obs["mean_abs_diff_cm"] is not None
    assert mc.drilldown_rows(make_payload(horizon=2, verified=False), "ridge_ar")[0]["실제"] == "-"


def test_compare_difference_rows_units() -> None:
    payload = make_payload(horizon=7)
    both = mc.compare_difference_rows(payload)
    assert [r["D+"] for r in both] == [1, 3, 5, 7]
    cell = both[0]["릿지 자기회귀 (AR)"]
    assert " m (" in cell and cell.endswith(" cm)")
    cm = mc.compare_difference_rows(payload, unit="cm")[0]
    assert cm["릿지 자기회귀 (AR)"].endswith(" cm)") and " cm (" in cm["릿지 자기회귀 (AR)"]
    assert cm["선택 예측"].endswith(" cm") and cm["실제"].endswith(" cm")
    m = mc.compare_difference_rows(payload, unit="m")[0]
    assert m["릿지 자기회귀 (AR)"].endswith(" m)") and m["직전값"] == "2.000 m"
    assert mc.compare_difference_rows(make_payload(horizon=2, verified=False))[0]["실제"] == "-"


def test_compare_metric_rows_keep_legacy_columns() -> None:
    rows = mc.compare_metric_rows(make_payload(horizon=7))
    assert "시험 RMSE D+7" in rows[0] and "이번 원점 RMSE" in rows[0]
    assert rows[0]["시험 RMSE D+7"] == "0.020 m (2.0 cm)"  # 개선율 내림차순 → 가장 좋은 모델이 먼저
    # slides.py 에서 기존 이름으로 그대로 import 된다
    import slides

    assert slides.compare_metric_rows is mc.compare_metric_rows
    assert slides._compare_figure is mc._compare_figure
    assert slides.compare_difference_rows is mc.compare_difference_rows


# --- 그래프 -----------------------------------------------------------------------------------------


def test_compare_figure_filters_traces_marks_rank_and_units() -> None:
    payload = make_payload(horizon=7)
    ranking = mc.rank_models(payload, "test_rmse")
    fig = mc._compare_figure(payload, "t", visible=["analog_knn", "ridge_arx"], ranking=ranking, history_days=10)
    names = [t.name for t in fig.data]
    assert "관측 (원점까지)" in names and "선택 모델 (검증 최적)" in names and "직전값 유지 (기준선)" in names and "실제 관측" in names
    model_traces = [t for t in fig.data if t.name.startswith("유사사례") or t.name.startswith("릿지 자기회귀 + 기상")]
    assert len(model_traces) == 2
    best = next(t for t in fig.data if "🥇" in t.name)
    worst = next(t for t in fig.data if "⚠" in t.name)
    assert best.line.width > worst.line.width and worst.line.dash == "dashdot"
    assert len(fig.data[0].x) == 10  # 과거 10일
    assert "cm" in best.hovertemplate and "m (" in best.hovertemplate
    # 기본 호출(예전 서명)도 그대로 동작
    legacy = mc._compare_figure(payload, "legacy")
    assert len([t for t in legacy.data if "위" in t.name]) == 0 and len(legacy.data) == len(MODELS) + 4
    # 예측 구간만 + cm 축
    zoom = mc._compare_figure(payload, "z", visible=[], ranking=ranking, forecast_only=True, unit="cm", show_persistence=False)
    assert "관측 (원점까지)" not in [t.name for t in zoom.data] and "직전값 유지 (기준선)" not in [t.name for t in zoom.data]
    assert zoom.layout.xaxis.range[1] == "2024-07-07" and zoom.layout.yaxis.title.text == "수위 (cm)"
    assert zoom.layout.yaxis.range[0] > 100  # cm 눈금


def test_difference_and_drilldown_figures_without_observed() -> None:
    payload = make_payload(horizon=5, verified=False)
    ranking = mc.rank_models(payload, "test_rmse")
    diff = mc._difference_figure(payload, "d", visible=None, ranking=ranking, kind="selected")
    assert "직전값 유지 − 선택 모델" in [t.name for t in diff.data] and "실제 − 선택 모델" not in [t.name for t in diff.data]
    assert diff.layout.yaxis.title.text == "차이 (cm)"
    err = mc._difference_figure(payload, "e", visible=None, ranking=ranking, kind="observed")
    assert len(err.data) == 0  # 실제 없음 → 선 없음, 예외 없음
    bar = mc._drilldown_figure(payload, "ridge_ar", "b")
    assert bar.layout.yaxis.title.text == "차이 (cm)" and list(bar.data[0].x) == [f"D+{h}" for h in range(1, 6)]
    bar_obs = mc._drilldown_figure(make_payload(horizon=5), "ridge_ar", "b")
    assert bar_obs.layout.yaxis.title.text == "|오차| (cm)"
    # 실제 관측·기준선을 끄면 차이 그래프의 y 축이 모델 간 차이에만 맞춰진다
    verified = make_payload(horizon=5)
    full = mc._difference_figure(verified, "d", visible=["ridge_ar"], ranking=None, kind="selected")
    trimmed = mc._difference_figure(verified, "d", visible=["ridge_ar"], ranking=None, kind="selected", show_observed=False, show_persistence=False)
    assert [t.name for t in full.data] == ["릿지 자기회귀 (AR)", "실제 − 선택 모델", "직전값 유지 − 선택 모델"]
    assert [t.name for t in trimmed.data] == ["릿지 자기회귀 (AR)"]
    assert (trimmed.layout.yaxis.range[1] - trimmed.layout.yaxis.range[0]) < (full.layout.yaxis.range[1] - full.layout.yaxis.range[0])


# --- AppTest 스모크 ---------------------------------------------------------------------------------


def _compare_app(payload: dict, ui_dir: str) -> None:  # AppTest.from_function 이 이 함수 본문을 스크립트로 실행한다
    import sys
    from datetime import date

    import streamlit as st

    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    from model_compare_ui import render_model_compare

    class FakeClient:
        calls = 0

        def gauge_forecast_compare(self, region_id, station_id, *, as_of=None, horizon=None):
            FakeClient.calls += 1
            return payload

    if st.toggle("모든 모델로 한 번에 비교하기", value=False, key="t_compare"):
        render_model_compare(FakeClient(), {"id": "toy"}, {"id": "s1", "name": "토이 지점"}, as_of=date(2024, 6, 30), horizon=int(payload["horizon"]), key="t_lv")


def _run_compare_app(payload: dict) -> AppTest:
    app = AppTest.from_function(_compare_app, kwargs={"payload": payload, "ui_dir": str(UI_DIR)}, default_timeout=30)
    app.run()
    assert not app.exception
    app.toggle(key="t_compare").set_value(True).run()
    assert not app.exception
    return app


@pytest.mark.parametrize("verified", [True, False])
def test_apptest_compare_controls_do_not_raise(verified: bool) -> None:
    app = _run_compare_app(make_payload(horizon=7, verified=verified))
    assert app.session_state["t_lv_compare_payload"]["horizon"] == 7
    radios = {r.key: r for r in app.radio}
    metric = radios["t_lv_cmp_metric"]
    assert (metric.value == "verify_rmse") == verified
    assert ("오차 (실제 대비)" in radios["t_lv_cmp_view"].options) == verified
    assert len(app.get("plotly_chart")) >= 2  # 비교 그래프 + 파고들기 막대
    table_text = " ".join(str(t.value) for t in app.table)
    assert "🥇 1위" in table_text and "⚠ 최하위" in table_text
    if not verified:
        assert "이번 원점 검증" in " ".join(str(m.value) for m in app.markdown)

    # 빠른 버튼 → 보일 모델 상태
    app.button(key="t_lv_cmp_preset_딥러닝만").click().run()
    assert not app.exception
    assert app.multiselect(key="t_lv_cmp_visible").value == ["neural_gru", "neural_lstm", "neural_mlp", "neural_tcn"]
    app.button(key="t_lv_cmp_preset_상위 3").click().run()
    assert not app.exception and len(app.multiselect(key="t_lv_cmp_visible").value) == 3
    app.button(key="t_lv_cmp_preset_기준선 숨기기").click().run()
    assert "climatology" not in app.multiselect(key="t_lv_cmp_visible").value

    # 보기 조절: 예측 구간만, 과거 N일, 축 단위, 차이 보기, 지표 변경, 파고들기 모델 변경
    app.toggle(key="t_lv_cmp_forecast_only").set_value(True).run()
    assert not app.exception
    app.toggle(key="t_lv_cmp_forecast_only").set_value(False).run()
    app.slider(key="t_lv_cmp_history_days").set_value(7).run()
    assert not app.exception
    app.radio(key="t_lv_cmp_unit").set_value("cm").run()
    assert not app.exception
    app.radio(key="t_lv_cmp_view").set_value("차이 (선택 모델 대비)").run()
    assert not app.exception
    if verified:
        app.radio(key="t_lv_cmp_view").set_value("오차 (실제 대비)").run()
        assert not app.exception
        app.radio(key="t_lv_cmp_metric").set_value("verify_skill").run()
        assert not app.exception
    app.radio(key="t_lv_cmp_metric").set_value("test_mse").run()
    assert not app.exception
    drill = app.selectbox(key="t_lv_cmp_drill")
    app.selectbox(key="t_lv_cmp_drill").set_value(drill.options[-1]).run()
    assert not app.exception
    app.radio(key="t_lv_cmp_table_unit").set_value("cm").run()
    assert not app.exception
    app.toggle(key="t_compare").set_value(False).run()
    assert not app.exception


def test_apptest_short_history_does_not_crash_slider() -> None:
    """원점이 계열 시작 직후라 history 가 7일 이하인 payload — 슬라이더 min==max 예외가 나면 안 된다."""
    payload = make_payload(horizon=7, verified=True)
    payload["history"] = payload["history"][-3:]
    app = _run_compare_app(payload)
    assert not app.exception
    assert not [s for s in app.slider if s.key == "t_lv_cmp_history_days"]
    assert len(app.get("plotly_chart")) >= 2


def test_nan_metric_is_treated_as_missing() -> None:
    payload = make_payload(horizon=7, verified=True)
    payload["per_model"]["ridge_ar"]["verification"]["rmse_m"] = float("nan")
    ranking = mc.rank_models(payload, "verify_rmse")
    entry = next(e for e in ranking if e["id"] == "ridge_ar")
    assert entry["rank"] is None and entry["value"] is None
    assert mc.fmt_m_cm(float("nan")) == "-"


def _local_compare_payload() -> dict | None:
    try:
        with urllib.request.urlopen(f"{LOCAL_API}/api/v1/regions/nakdong_busan/gauge/%EA%B5%AC%ED%8F%AC%EB%8C%80%EA%B5%90/forecast/compare?as_of=2024-07-01&horizon=14", timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — API 가 없으면 건너뛴다
        return None


def test_apptest_with_local_api_payload_if_reachable() -> None:
    payload = _local_compare_payload()
    if not payload or not payload.get("models"):
        pytest.skip("local API 127.0.0.1:18765 not reachable")
    app = _run_compare_app(payload)
    assert len(app.table) >= 2
    ranking = mc.rank_models(payload, "verify_rmse" if mc.has_verification(payload) else "test_rmse")
    assert ranking[0]["rank"] == 1
