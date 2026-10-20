"""학습·검증·시험 구간 컨트롤(ui_next/training_window_ui.py) — 순수 도우미와 AppTest.

서버는 가짜 클라이언트로 대신한다(계약: POST train {split, async_run} → {job_id}, GET jobs/{id} → 상태,
DELETE models → 되돌리기). 진행 폴링의 time.sleep 은 monkeypatch 로 없앤다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:  # slides.py 처럼 ui_next/ 를 sys.path 에 둔 채 평면 import 한다
    sys.path.insert(0, str(UI_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

import training_window_ui as tw  # noqa: E402
from api_client import APIClient, Endpoints  # noqa: E402

DEFAULT_SPLIT = {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}
CUSTOM_SPLIT = {"train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2024-12-31"}
RANGE = ("2015-01-01", "2024-12-31")


# --- 순수 도우미 --------------------------------------------------------------------------------


def test_years_to_split_builds_year_boundaries() -> None:
    assert tw.years_to_split(2018, (2019, 2023), (2024, 2024)) == DEFAULT_SPLIT
    assert tw.years_to_split(2020, (2021, 2022), (2023, 2024)) == CUSTOM_SPLIT


def test_validate_split_rules() -> None:
    assert tw.validate_split(DEFAULT_SPLIT, RANGE) == []
    # 순서
    bad = dict(DEFAULT_SPLIT, val_start="2018-06-01")
    assert any("순서" in p for p in tw.validate_split(bad, RANGE))
    # 학습 730일
    short_train = tw.years_to_split(2015, (2016, 2023), (2024, 2024))
    assert any("학습 구간" in p and "730" in p for p in tw.validate_split(short_train, RANGE))
    assert tw.validate_split(tw.years_to_split(2016, (2017, 2023), (2024, 2024)), RANGE) == []  # 731일 → 통과
    # 검증 90일 · 시험 30일
    assert any("검증 구간" in p for p in tw.validate_split(dict(DEFAULT_SPLIT, val_start="2023-11-01"), RANGE))
    assert any("시험 구간" in p for p in tw.validate_split(dict(DEFAULT_SPLIT, test_end="2024-01-20"), RANGE))
    # 범위 밖 — 오류문에 기록 범위가 들어간다
    outside = tw.validate_split(tw.years_to_split(2018, (2019, 2023), (2024, 2025)), RANGE)
    assert any("2015-01-01 ∼ 2024-12-31" in p and "test_end" in p for p in outside)
    # 범위를 모르면 순서·최소 일수만 본다
    assert tw.validate_split(DEFAULT_SPLIT, None) == []
    assert tw.validate_split({"train_end": "2018-12-31"}, RANGE) == ["구간 날짜 다섯 개(train_end, val_start, val_end, test_start, test_end)가 모두 있어야 합니다."]
    assert tw.split_days(DEFAULT_SPLIT, RANGE) == {"train": 1461, "val": 1826, "test": 366}


def test_describe_split_uses_tilde_operator_not_markdown_strikethrough() -> None:
    text = tw.describe_split(DEFAULT_SPLIT)
    assert text == "학습 ≤2018 · 검증 2019∼2023 · 시험 2024"
    assert "~" not in text
    assert tw.describe_split(CUSTOM_SPLIT) == "학습 ≤2020 · 검증 2021∼2022 · 시험 2023∼2024"
    # 연도 경계가 아닌 분할(Split.default_for)은 날짜를 그대로 보여 준다
    odd = {"train_end": "2022-12-30", "val_start": "2022-12-31", "val_end": "2023-12-30", "test_start": "2023-12-31", "test_end": "2024-12-31"}
    assert tw.describe_split(odd) == "학습 ≤2022-12-30 · 검증 2022-12-31∼2023-12-30 · 시험 2023-12-31∼2024-12-31"
    assert tw.describe_split({}) == "구간 정보 없음"


def test_split_from_region_reads_active_default_custom() -> None:
    detail = {"forecast": {"split_default": DEFAULT_SPLIT, "split_active": CUSTOM_SPLIT, "custom_split": True, "trained_at": "2026-09-09T10:20:30"}}
    info = tw.split_from_region(detail)
    assert info["active"] == CUSTOM_SPLIT and info["default"] == DEFAULT_SPLIT and info["custom"] is True
    assert tw.split_badge_text(detail) == "학습 ≤2020 · 검증 2021∼2022 · 시험 2023∼2024 · 사용자 지정 · 학습 2026-09-09 10:20"
    # 예전 서버(split 만): 기본으로 취급
    legacy = tw.split_from_region({"forecast": {"split": DEFAULT_SPLIT}})
    assert legacy["active"] == DEFAULT_SPLIT and legacy["default"] == DEFAULT_SPLIT and legacy["custom"] is False
    assert tw.split_badge_text({"forecast": {"split": DEFAULT_SPLIT}}) == "학습 ≤2018 · 검증 2019∼2023 · 시험 2024 · 기본"
    assert tw.split_from_region(None)["active"] == {}


def test_series_range_and_years_from_station_periods() -> None:
    region = {"stations": [{"period": ["2016-03-01", "2024-12-31"]}, {"period": ["2015-01-01", "2023-06-30"]}, {"period": None}]}
    assert tw.series_range_from_region(region) == ("2015-01-01", "2024-12-31")
    assert tw.series_years(region) == (2015, 2024)
    assert tw.series_range_from_region({"stations": []}) is None
    assert tw.series_years({}, DEFAULT_SPLIT) == (2014, 2024)
    # 서버가 forecast.series_range 를 주면 그것이 우선
    assert tw.series_range_from_region({"forecast": {"series_range": ["2010-05-01", "2024-12-31"]}, "stations": region["stations"]}) == ("2010-05-01", "2024-12-31")
    assert tw.trained_at_text("2026-09-09T00:34:49+00:00") == "2026-09-09 09:34"  # UTC → KST
    assert tw.trained_at_text("2026-09-09T10:00:00") == "2026-09-09 10:00"
    assert tw.trained_at_text(None) == ""
    assert tw.split_years(DEFAULT_SPLIT) == (2018, (2019, 2023), (2024, 2024))
    # 연도 경계가 아닌 구간(API 로 만든 사용자 지정)은 연도가 겹치지 않게 밀어 기본값부터 순서 경고가 뜨지 않는다
    unaligned = {"train_end": "2021-06-30", "val_start": "2021-07-01", "val_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2024-06-30"}
    assert tw.split_years(unaligned) == (2021, (2022, 2023), (2024, 2024))
    assert tw.validate_split(tw.years_to_split(*tw.split_years(unaligned)), ("2019-01-01", "2024-12-31")) == []
    assert tw.progress_text({"status": "running", "progress": {"stations_total": 5, "stations_done": 2, "current_station": "구포대교"}, "elapsed_s": 75}) == "학습 중 · 2/5 지점 · 현재 구포대교 · 1분 15초 경과"


def test_api_client_sends_split_and_async_and_has_job_endpoints() -> None:
    calls: list[tuple] = []

    class Fake(APIClient):
        def _request(self, method, path, **kwargs):  # noqa: D401
            calls.append((method, path, kwargs.get("json")))
            if path.endswith("/jobs"):
                return {"items": [{"job_id": "j1", "status": "running"}]}
            return {"ok": True, "job_id": "j1"}

    client = Fake("http://x")
    client.gauge_train("r1", split=DEFAULT_SPLIT, async_run=True)
    assert calls[-1][0] == "POST" and calls[-1][1] == "/api/v1/regions/r1/gauge/train"
    assert calls[-1][2] == {"split": DEFAULT_SPLIT, "async_run": True}
    client.gauge_train("r1", stations=["a"])  # 예전 서명 — split/async_run 을 보내지 않는다
    assert calls[-1][2] == {"stations": ["a"]}
    client.gauge_train_job("r1", "j1")
    assert calls[-1][:2] == ("GET", "/api/v1/regions/r1/gauge/train/jobs/j1")
    assert client.gauge_train_jobs("r1") == [{"job_id": "j1", "status": "running"}]
    client.gauge_reset_models("r1")
    assert calls[-1][:2] == ("DELETE", "/api/v1/regions/r1/gauge/models")
    assert Endpoints.GAUGE_TRAIN_JOB == "/api/v1/regions/{region_id}/gauge/train/jobs/{job_id}"
    assert Endpoints.GAUGE_MODELS == "/api/v1/regions/{region_id}/gauge/models"


# --- AppTest ----------------------------------------------------------------------------------


def _window_app(ui_dir: str, custom: bool) -> None:  # AppTest.from_function 이 이 본문을 스크립트로 실행한다
    import sys

    import streamlit as st

    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    from training_window_ui import render_training_window

    DEFAULT = {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}
    CUSTOM = {"train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2024-12-31"}

    # 가짜 서버 상태는 세션에 둔다(재실행을 넘어 살아남아야 '학습 뒤 사용자 지정' 이 보인다)
    server = st.session_state.setdefault("fake_server", {"custom": custom, "polls": 0, "calls": []})

    class FakeClient:
        def get_region(self, rid):
            server["calls"].append(("get_region", rid))
            active = CUSTOM if server["custom"] else DEFAULT
            return {"id": rid, "name": "토이", "stations": [{"id": "s1", "period": ["2015-01-01", "2024-12-31"]}],
                    "forecast": {"split": DEFAULT, "split_default": DEFAULT, "split_active": active, "custom_split": server["custom"],
                                 "trained_at": "2026-09-09T10:00:00" if server["custom"] else None}}

        def list_gauge_forecasters(self):
            return [{"id": "persistence", "is_baseline": True}, {"id": "ridge_ar"}, {"id": "analog_knn"}, {"id": "neural_mlp"}]

        def gauge_train(self, rid, *, split=None, async_run=False, **kw):
            server["calls"].append(("train", rid, dict(split or {}), async_run))
            return {"job_id": "job-1", "status": "queued", "region_id": rid, "split_used": split, "stations": ["s1"]}

        def gauge_train_job(self, rid, job_id):
            server["polls"] += 1
            server["calls"].append(("job", job_id, server["polls"]))
            if server["polls"] < 3:
                return {"job_id": job_id, "status": "running", "progress": {"stations_total": 1, "stations_done": 0, "current_station": "s1"}, "elapsed_s": 2 * server["polls"], "split_used": CUSTOM}
            server["custom"] = True
            return {"job_id": job_id, "status": "done", "progress": {"stations_total": 1, "stations_done": 1, "current_station": None}, "elapsed_s": 6,
                    "split_used": CUSTOM, "evaluation": {"n_stations": 1, "cells_total": 7, "cells_beating_persistence": 5, "split": CUSTOM}}

        def gauge_train_jobs(self, rid):
            return []

        def gauge_reset_models(self, rid):
            server["calls"].append(("reset", rid))
            server["custom"] = False
            return {"region_id": rid, "removed_files": ["s1.json", "evaluation.json"], "active": "builtin"}

    if "seeded" not in st.session_state:  # 첫 실행에서만 캐시·예측 결과를 심는다(재실행 뒤 비워졌는지 봐야 한다)
        st.session_state.seeded = True
        st.session_state.gauge_cache = {
            "evaluation:toy": 1, "model:toy:s1": 1, "forecast:toy:s1": 1, "backtest:toy:s1:7": 1,
            "compare:toy": 1, "satlevels:toy:s1": 1, "regions": [], "weather_status": {"keep": True},
        }
        st.session_state.sl_level_forecast = {"region_id": "toy"}
        st.session_state.sl_lv_compare_payload = {"horizon": 7}
    render_training_window(FakeClient(), {"id": "toy", "name": "토이", "stations": [{"id": "s1", "period": ["2015-01-01", "2024-12-31"]}]}, key="t_tw")


def _run_window(custom: bool) -> AppTest:
    app = AppTest.from_function(_window_app, kwargs={"ui_dir": str(UI_DIR), "custom": custom}, default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_apptest_retrain_polls_until_done_and_clears_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    real_sleep = time.sleep

    def fake_sleep(seconds: float) -> None:  # 폴링 대기(2초)만 없앤다 — AppTest 러너 자체의 짧은 sleep 은 그대로
        if seconds == tw.POLL_INTERVAL_S:
            sleeps.append(seconds)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    app = _run_window(custom=False)
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "학습 ≤2018 · 검증 2019∼2023 · 시험 2024" in markdown and "`기본`" in markdown
    captions = " ".join(str(c.value) for c in app.caption)
    assert "후보 예측기 3개" in captions and "직접 비교할 수 없습니다" in captions
    assert app.select_slider(key="t_tw:toy:train_end").value == 2018
    assert tuple(app.select_slider(key="t_tw:toy:val_years").value) == (2019, 2023)
    assert tuple(app.select_slider(key="t_tw:toy:test_years").value) == (2024, 2024)
    assert not [b for b in app.button if b.key == "t_tw:toy:reset_btn"]  # 기본 구간이면 되돌리기 없음

    # 구간을 바꾸면 날짜 다섯 개가 따라 바뀌고, 잘못된 구간이면 경고 + 버튼 비활성
    app.select_slider(key="t_tw:toy:train_end").set_value(2020).run()
    app.select_slider(key="t_tw:toy:val_years").set_value((2021, 2022)).run()
    app.select_slider(key="t_tw:toy:test_years").set_value((2023, 2024)).run()
    assert not app.exception
    assert "2020-12-31" in " ".join(str(m.value) for m in app.markdown) and not app.warning
    app.select_slider(key="t_tw:toy:val_years").set_value((2019, 2022)).run()
    assert any("순서" in str(w.value) for w in app.warning)
    assert app.button(key="t_tw:toy:train_btn").disabled
    app.select_slider(key="t_tw:toy:val_years").set_value((2021, 2022)).run()
    assert not app.warning and not app.button(key="t_tw:toy:train_btn").disabled

    app.button(key="t_tw:toy:train_btn").click().run()
    assert not app.exception
    server = app.session_state["fake_server"]
    train_calls = [c for c in server["calls"] if c[0] == "train"]
    assert train_calls == [("train", "toy", CUSTOM_SPLIT, True)]
    assert server["polls"] == 3 and sleeps == [tw.POLL_INTERVAL_S, tw.POLL_INTERVAL_S]  # 두 번 대기 뒤 완료
    # 완료 → 캐시 비움 + 재실행: 작업 키가 사라지고 배지가 사용자 지정으로 바뀐다
    assert "t_tw:toy:job" not in app.session_state
    cache = app.session_state["gauge_cache"]
    assert not any(k.startswith(("evaluation:", "model:", "forecast:", "backtest:", "compare:", "satlevels:")) or k == "regions" for k in cache)
    assert cache.get("weather_status") == {"keep": True}  # 무관한 캐시는 남는다
    assert "sl_level_forecast" not in app.session_state and "sl_lv_compare_payload" not in app.session_state
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "학습 ≤2020 · 검증 2021∼2022 · 시험 2023∼2024" in markdown and "`사용자 지정`" in markdown
    success = " ".join(str(s.value) for s in app.success)
    assert "학습 완료" in success and "5/7셀" in success
    assert [b for b in app.button if b.key == "t_tw:toy:reset_btn"]


def test_apptest_reset_button_restores_builtin_split() -> None:
    app = _run_window(custom=True)
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "`사용자 지정`" in markdown and "학습 2026-09-09 10:00" in markdown
    assert app.select_slider(key="t_tw:toy:train_end").value == 2020
    app.button(key="t_tw:toy:reset_btn").click().run()
    assert not app.exception
    server = app.session_state["fake_server"]
    assert ("reset", "toy") in server["calls"] and not [c for c in server["calls"] if c[0] == "train"]
    cache = app.session_state["gauge_cache"]
    assert "evaluation:toy" not in cache and "sl_level_forecast" not in app.session_state
    markdown = " ".join(str(m.value) for m in app.markdown)
    assert "`기본`" in markdown and "학습 ≤2018 · 검증 2019∼2023 · 시험 2024" in markdown
    assert any("기본 구간 산출물로 되돌렸습니다" in str(i.value) and "2개 삭제" in str(i.value) for i in app.info)
    assert not [b for b in app.button if b.key == "t_tw:toy:reset_btn"]


def test_expander_is_wired_into_slides_and_gauge_page() -> None:
    slides = (UI_DIR / "slides.py").read_text(encoding="utf-8")
    gauge = (UI_DIR / "gauge_forecast_ui.py").read_text(encoding="utf-8")
    assert "render_training_window(client, region, key=\"sl_tw\")" in slides
    assert slides.count("render_split_badge(client, region)") >= 2
    assert "render_training_window(" in gauge
    capture = (UI_DIR.parent / "scripts" / "capture_ui_screens.py").read_text(encoding="utf-8")
    assert "15a_slide4_training_window.png" in capture and "이 구간으로 다시 학습" not in capture.split("15a_slide4_training_window.png")[0][-400:]


@pytest.mark.parametrize("prefix", ["region_detail:", "evaluation:", "model:", "forecast:", "backtest:", "compare:", "satlevels:"])
def test_cache_prefixes_cover_contract(prefix: str) -> None:
    assert prefix in tw.CACHE_PREFIXES
