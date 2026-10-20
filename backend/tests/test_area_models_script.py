"""면적 모델 실험 스크립트(scripts/eval_area_models_experiment.py) — 순수 수치 도우미와 산출 계약.

여기서 지키는 규율:

* LOO 는 **뺀 점을 쓰지 않는다** (persistence 는 앞선 관측만, 회귀는 나머지 n−1 점으로만 적합).
* 부호검정은 정확 이항이라 n=4 면 최소 달성 가능 p 가 0.125 → 유의 판정이 나올 수 없다.
* 검량곡선 문턱(gate)은 기울기 음수·표본 부족·수위 변동폭 부족을 각각 잡아낸다.
* 실제 산출 JSON 은 scenes.json·area_series.json 과 어긋나지 않아야 한다.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_area_models_experiment", ROOT / "scripts" / "eval_area_models_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


script = _load_script()


def test_pava_is_monotone_and_averages_violators() -> None:
    xs, fit = script.pava_fit([1.0, 2.0, 3.0], [1.0, 5.0, 3.0])
    assert list(xs) == [1.0, 2.0, 3.0]
    assert fit[0] == pytest.approx(1.0) and fit[1] == fit[2] == pytest.approx(4.0)  # 5,3 을 평균 4 로 합친다
    assert all(fit[i] <= fit[i + 1] + 1e-12 for i in range(len(fit) - 1))
    # 예측은 선형 보간, 구간 밖은 양 끝 값으로 고정
    assert script.pava_predict(xs, fit, 1.5) == pytest.approx(2.5)
    assert script.pava_predict(xs, fit, 0.0) == pytest.approx(1.0)
    assert script.pava_predict(xs, fit, 9.0) == pytest.approx(4.0)
    assert math.isnan(script.pava_predict(np.array([]), np.array([]), 1.0))


def test_sign_test_cannot_be_significant_with_four_points() -> None:
    perfect = script.sign_test(np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert perfect["wins"] == 4 and perfect["losses"] == 0
    assert perfect["p_two_sided"] == pytest.approx(0.125) == perfect["min_attainable_p"]
    assert perfect["significant_at_0_05"] is False  # n=4 에서는 전승해도 유의하지 않다
    # n=10 전승이면 유의하다
    ten = script.sign_test(np.zeros(10), np.ones(10))
    assert ten["p_two_sided"] == pytest.approx(0.002) and ten["significant_at_0_05"] is True  # 2/1024, 소수 4자리 반올림
    tie = script.sign_test(np.ones(3), np.ones(3))
    assert tie["ties"] == 3 and tie["p_two_sided"] is None and tie["significant_at_0_05"] is False


def test_loo_uses_only_the_other_points() -> None:
    dates = ["2020-01-01", "2020-01-11", "2020-01-21", "2020-01-31"]
    areas = [10.0, 12.0, 14.0, 16.0]
    levels = {"가교": [1.0, 2.0, 3.0, 4.0]}
    preds = script.loo_predictions(dates, areas, levels)
    # persistence: 앞선 관측이 있으면 그 값, 첫 점은 예외로 다음 관측
    assert list(preds["persistence"]) == [12.0, 10.0, 12.0, 14.0]
    # 계열 평균: 자기 자신을 빼고 평균
    assert preds["series_mean"][0] == pytest.approx((12 + 14 + 16) / 3)
    # 완전한 직선이면 선형추세·수위 선형회귀가 뺀 점을 정확히 복원한다
    assert preds["linear_trend"] == pytest.approx(np.array(areas))
    assert preds["level_linear:가교"] == pytest.approx(np.array(areas))
    # 단조회귀는 구간 밖 외삽을 하지 않고 양 끝 값으로 고정하므로, 첫 점(수위가 가장 낮음)은 이웃 값 12 로 고정돼 2 km² 틀린다
    assert preds["level_isotonic:가교"][1] == pytest.approx(12.0)
    assert preds["level_isotonic:가교"][0] == pytest.approx(12.0) and preds["level_isotonic:가교"][3] == pytest.approx(14.0)
    scored = script.score_loo(preds, areas, dates)
    assert scored["linear_trend"]["rmse_km2"] == 0.0 and scored["linear_trend"]["skill_vs_persistence"] == 1.0
    assert scored["persistence"]["skill_vs_persistence"] == 0.0 and scored["persistence"]["sign_test_vs_persistence"] is None
    assert [p["date"] for p in scored["persistence"]["per_point"]] == dates
    assert scored["level_linear:가교"]["level_station"] == "가교" and scored["series_mean"]["level_station"] is None


def test_loo_skips_level_models_when_the_level_never_moves() -> None:
    preds = script.loo_predictions(["2020-01-01", "2020-01-02", "2020-01-03"], [1.0, 2.0, 3.0], {"고정": [5.0, 5.0, 5.0]})
    assert not any(k.startswith("level_") for k in preds)


def test_rating_curve_reports_slope_and_loo_uncertainty() -> None:
    curve = script.rating_curve([1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0])
    assert curve["slope_km2_per_m"] == pytest.approx(2.0) and curve["pearson_r"] == pytest.approx(1.0)
    assert curve["level_range_m"] == pytest.approx(3.0) and curve["loo"]["rmse_km2"] == pytest.approx(0.0)
    noisy = script.rating_curve([1.0, 2.0, 3.0, 4.0], [10.0, 16.0, 11.0, 15.0])
    assert noisy["loo"]["rmse_km2"] > 1.0 and noisy["loo"]["n"] == 4


def test_gate_rejects_negative_slope_small_sample_and_flat_level() -> None:
    frame = {"frame_area_km2": 200.0}
    backwards = script.evaluate_gate(script.rating_curve([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [16, 15, 14, 13, 12, 11]), frame, 3.0)
    assert backwards["defensible"] is False and backwards["passes"]["require_positive_slope"] is False
    assert backwards["passes"]["min_pairs"] is True
    small = script.evaluate_gate(script.rating_curve([1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]), frame, 3.0)
    assert small["defensible"] is False and small["passes"]["min_pairs"] is False
    flat = script.rating_curve([2.000, 2.005, 2.010, 2.015, 2.020, 2.025], [30.0, 28.0, 30.0, 31.0, 30.5, 29.5])
    verdict = script.evaluate_gate(flat, frame, 3.0)
    assert verdict["passes"]["min_level_range_m"] is False and verdict["defensible"] is False
    assert verdict["implied_area_range_km2"] is not None and verdict["gauge_level_range_10y_m"] == 3.0
    good = script.evaluate_gate(script.rating_curve([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], [10.0, 11.0, 12.1, 12.9, 14.0, 15.1]), frame, 3.0)
    assert good["defensible"] is True and all(good["passes"].values())


def test_area_series_file_loader_accepts_the_canonical_and_flat_shapes(tmp_path: Path) -> None:
    assert script._load_area_series_file(tmp_path) is None
    (tmp_path / "area_series.json").write_text(json.dumps({
        "schema": "watercast.satellite.area-series/1",
        "sensors": {"ICEYE": {"rows": [{"date": "2020-03-02", "area_km2_common_frame": 40.207, "area_km2_scene_footprint": 40.1}]}},
    }), encoding="utf-8")
    assert script._load_area_series_file(tmp_path) == {"ICEYE": {"2020-03-02": 40.207}}
    (tmp_path / "area_series.json").write_text(json.dumps(
        {"rows": [{"sensor": "P", "date": "2020-01-01", "area_km2": 3.5}]}), encoding="utf-8")
    assert script._load_area_series_file(tmp_path) == {"P": {"2020-01-01": 3.5}}
    (tmp_path / "area_series.json").write_text("not json", encoding="utf-8")
    assert script._load_area_series_file(tmp_path) is None


def test_repeat_pairs_only_keeps_close_revisits() -> None:
    points = [{"date": "2020-04-15", "area_km2": 36.7}, {"date": "2020-04-16", "area_km2": 39.1}, {"date": "2020-05-30", "area_km2": 30.0}]
    pairs = script.repeat_pairs(points, {"가교": [1.0, 1.1, 2.0]})
    assert len(pairs) == 1 and pairs[0]["gap_days"] == 1
    assert pairs[0]["delta_area_km2"] == pytest.approx(2.4) and pairs[0]["delta_level_m"]["가교"] == pytest.approx(0.1)


ARTIFACTS = {rid: ROOT / "regions" / rid / "models" / "gauge" / "experiments" / "area_models.json"
             for rid in ("nakdong_busan", "geum_daecheong")}


@pytest.mark.skipif(not all(p.is_file() for p in ARTIFACTS.values()), reason="실험 산출물(area_models.json)이 없다")
def test_written_artifacts_agree_with_scenes_and_stay_honest() -> None:
    for region_id, path in ARTIFACTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == script.EXPERIMENT_SCHEMA and payload["region_id"] == region_id
        scenes = json.loads((ROOT / "regions" / region_id / "satellite" / "scenes.json").read_text(encoding="utf-8"))["scenes"]
        for sensor, block in payload["series"].items():
            expected = sorted(str(s["date"])[:10] for s in scenes if s["sensor"] == sensor)
            assert [p["date"] for p in block["points"]] == expected
            # 계열 출처가 area_series.json 이어도 마스크 재계산과 어긋나면 안 된다
            assert block["max_abs_diff_vs_recompute_km2"] < 0.01
            assert block["candidates"]["persistence"]["skill_vs_persistence"] == 0.0
            assert block["best_by_loo"] in block["candidates"]
            assert block["ranking_by_loo_rmse"][0] == block["best_by_loo"]
            for candidate in block["candidates"].values():
                assert candidate["n"] == block["n"] and len(candidate["per_point"]) == block["n"]
            if block["n"] <= 6:  # 작은 표본에서는 유의 판정이 나올 수 없다
                assert block["any_candidate_significant_vs_persistence"] is False
        bridge = payload["bridge"]
        assert bridge["any_defensible"] is False  # 이 자료에서는 검량곡선이 서지 않는다
        demo = bridge.get("demonstration")
        if demo is not None:
            assert demo["published"] is False and demo["demonstration_only"] is True
            assert demo["summary"]["warnings_ko"]
            # 파생 면적 예측 skill 은 같은 분할 수위 모델 skill 과 사실상 같다 (아핀 변환)
            assert demo["summary"]["skill_max_abs_diff_vs_level"] < 1e-6
            for row in demo["end_to_end"]["rows"]:
                if row.get("total_error_km2") is not None:
                    assert row["total_error_km2"] == pytest.approx(row["rating_error_km2"] + row["level_forecast_error_km2"], abs=2e-3)
        assert any("공통 격자" in note for note in payload["notes_ko"])
        assert any("검증되지 않는다" in line for line in payload["verdict_ko"])
