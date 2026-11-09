"""TTA TC2 시험 스크립트(scripts/eval_iitp_indicator.py) — 산정식·성립 조건·한 번에 도는 흐름.

여기서 지키는 규율:

* 시험 연도는 위성 촬영 연도에서 정하고, 그 전 해가 검증, 그 이전이 학습이다. 연도가 둘 이상이면 멈춘다.
* 원점이 학습·검증 구간 안(표본 내)이면 값을 내지 않고 **중단**한다 — 조용히 '적합'을 찍는 사고를 막는다.
* 활성 산출물의 구간이 시험 설정과 다르면 중단한다(--no-train 경로).
* 수체면적은 '그 시점까지 관측된 값'만 쓰고, 제출값(공통 격자·평균회귀 0.02663)을 그대로 재현한다.
* 한 번 실행에 학습 → 수위 D+1·D+7 → 수체면적 → 지표가 전부 나오고 비교표가 출력된다.
* 학습은 지역 지점 전부를 하나씩 순서대로 돌리고, 지점 하나가 끝날 때마다 쉰다(기본 60초) — 대기는 대기라고 찍는다.
* SSH 로 들어와 실행했으면 결과 파일을 클라이언트 PC 로 가져가는 scp 명령을 끝에 찍는다.
"""

from __future__ import annotations

import getpass
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_iitp_indicator", ROOT / "scripts" / "eval_iitp_indicator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


script = _load_script()

API = "http://test"
REGION = "r"
# 제출 문안의 공통 격자 면적 계열(2020, 낙동강 하구) — 평균회귀 정규화 MSE 0.02663 의 근거
ICEYE = [("2020-03-02", 40.207), ("2020-03-30", 36.607), ("2020-04-15", 36.742), ("2020-04-16", 39.127)]
PLANET = [("2020-02-18", 30.152), ("2020-03-12", 28.870), ("2020-03-25", 30.148), ("2020-04-14", 30.974)]
SPLIT_2020 = {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-12-31",
              "test_start": "2020-01-01", "test_end": "2020-12-31"}


def _series(days: dict[str, float]) -> dict[str, Any]:
    return {"rows": [{"date": d, "water_level_m": v} for d, v in days.items()]}


class FakeApi:
    """시험대상 API 흉내 — 경로별 응답을 사전으로 들고, 학습 잡은 running → done 으로 흘러간다."""

    def __init__(self, *, in_sample: bool = False, clamp: bool = False, split: dict[str, str] | None = None) -> None:
        self.in_sample = in_sample
        self.clamp = clamp
        self.split = dict(split or SPLIT_2020)
        self.calls: list[tuple[str, str]] = []
        self.polls = 0
        self.trained_with: dict[str, Any] | None = None
        self.train_bodies: list[dict[str, Any]] = []
        self.other_split: list[str] = []
        self.levels = {
            "A": {"2020-02-18": 2.0, "2020-02-19": 2.1, "2020-02-25": 2.3, "2020-03-02": 2.2, "2020-03-03": 2.25,
                  "2020-03-09": 2.0, "2020-06-01": 3.0, "2020-12-31": 1.0},
            "B": {"2020-02-18": 1.0, "2020-02-19": 1.0, "2020-02-25": 1.1, "2020-03-02": 1.2, "2020-03-03": 1.2,
                  "2020-03-09": 1.3, "2020-07-01": 1.5},
        }

    # 수위 예측 : 원점값 + 0.05×h 로 정해 두면 오차를 손으로 검산할 수 있다
    def _forecast(self, station: str, as_of: str, horizon: int) -> dict[str, Any]:
        origin = self.levels[station][as_of]
        steps = []
        for h in range(1, horizon + 1):
            day = int(as_of[-2:]) + h
            target = f"{as_of[:8]}{day:02d}" if day <= 28 else f"2020-03-{day - 29:02d}"
            steps.append({"horizon": h, "target_date": target, "forecast_m": origin + 0.05 * h, "forecaster_id": "fake"})
        return {"in_sample": self.in_sample, "as_of_clamped": self.clamp, "as_of": as_of, "origin_level_m": origin, "steps": steps}

    def get(self, api: str, path: str, **params: Any) -> Any:
        self.calls.append(("GET", path))
        if path.endswith("/satellite/scenes"):
            return {"scenes": [{"date": d, "sensor": "ICEYE"} for d, _ in ICEYE] + [{"date": d, "sensor": "PlanetScope"} for d, _ in PLANET]}
        if path.endswith("/satellite/area-series"):
            return {"available": True, "sensors": {
                "ICEYE": {"rows": [{"date": d, "area_km2_common_frame": a} for d, a in ICEYE]},
                "PlanetScope": {"rows": [{"date": d, "area_km2_common_frame": a} for d, a in PLANET]},
            }}
        if path == f"/api/v1/regions/{REGION}":
            return {"id": REGION, "name": "가짜 지역", "stations": [
                {"id": "A", "satellite_location": "a"}, {"id": "B", "satellite_location": "b"}, {"id": "C", "satellite_location": None}]}
        if path.endswith("/gauge/evaluation"):
            return {"split": self.split, "trained_at": "2026-11-08T00:00:00+00:00", "artifacts_origin": "user",
                    "stations_with_other_split": list(self.other_split)}
        if "/gauge/train/jobs/" in path:
            self.polls += 1
            if self.polls < 2:
                return {"status": "running", "progress": {"stations_done": 1, "stations_total": 2, "current_station": "A"}}
            trained = (self.trained_with or {}).get("stations") or ["A", "B"]
            return {"status": "done", "elapsed_s": 12.5, "stations": list(trained), "split_used": self.split,
                    "progress": {"stations_done": 2, "stations_total": 2}, "evaluation": {"trained_at": "2026-11-08T00:00:00+00:00"}}
        for station in self.levels:
            if path.endswith(f"/gauge/{station}/series"):
                return _series(self.levels[station])
            if path.endswith(f"/gauge/{station}/forecast"):
                return self._forecast(station, str(params["as_of"]), int(params["horizon"]))
        raise AssertionError(f"뜻밖의 GET {path}")

    def post(self, api: str, path: str, body: dict[str, Any]) -> Any:
        self.calls.append(("POST", path))
        assert path.endswith("/gauge/train")
        self.trained_with = body
        self.train_bodies.append(body)
        self.split = dict(body["split"])  # 학습하면 활성 구간이 요청 구간으로 바뀐다
        return {"job_id": "job1", "status": "queued"}

    def delete(self, api: str, path: str) -> Any:
        self.calls.append(("DELETE", path))
        return {"removed_files": ["a.json"], "active": "builtin"}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    api = FakeApi()
    monkeypatch.setattr(script, "get", api.get)
    monkeypatch.setattr(script, "post", api.post)
    monkeypatch.setattr(script, "delete", api.delete)
    monkeypatch.setattr(script.time, "sleep", lambda s: None)
    return api


# --- 구간 · 연도 ------------------------------------------------------------------------------


def test_split_follows_satellite_year() -> None:
    assert script.split_for_year(2020) == SPLIT_2020
    assert script.test_year_from_scenes(["2020-02-18", "2020-04-16"], None) == 2020
    assert script.test_year_from_scenes(["2020-02-18", "2022-04-16"], 2022) == 2022
    with pytest.raises(RuntimeError, match="--test-year"):
        script.test_year_from_scenes(["2020-02-18", "2022-04-16"], None)


def test_assert_split_names_the_mismatch() -> None:
    script.assert_split(SPLIT_2020, dict(SPLIT_2020))
    other = dict(SPLIT_2020, test_start="2024-01-01", test_end="2024-12-31")
    with pytest.raises(RuntimeError, match="test_start: 시험 2020-01-01 / 현재 2024-01-01"):
        script.assert_split(SPLIT_2020, other)


# --- [3] 수체면적 : 제출값 재현 ---------------------------------------------------------------------


def test_area_term_reproduces_submitted_value(fake: FakeApi) -> None:
    area = script.area_term(API, REGION)
    assert area["n_pairs"] == 6 and area["n_observations"] == 8
    assert area["range_km2"] == pytest.approx(11.337, abs=1e-3)
    assert area["norm_mse"] == pytest.approx(0.02663, abs=5e-5)
    # 두 번째 ICEYE 쌍 : 지금까지 평균 38.407, 직전 36.607 → 36.607 + 0.5 × 1.8 = 37.507 (미래 관측 미사용)
    second = area["pairs"][1]
    assert (second["from"], second["to"]) == ("2020-03-30", "2020-04-15")
    assert second["pred_km2"] == pytest.approx(37.507, abs=1e-3)
    persistence = script.area_term(API, REGION, model="persistence")
    assert persistence["norm_mse"] == pytest.approx(0.02934, abs=5e-5)
    assert persistence["pairs"][1]["pred_km2"] == pytest.approx(36.607)


# --- [2] 수위 : 산정 · 성립 조건 --------------------------------------------------------------------


def test_level_term_normalises_by_station_test_range(fake: FakeApi) -> None:
    level = script.level_term(API, REGION, ("A", "B"), ["2020-02-18", "2020-03-02"], SPLIT_2020, (1, 7))
    # A 의 2020 관측범위 3.0 − 1.0 = 2.0, B 는 1.5 − 1.0 = 0.5
    assert level["ranges_m"] == {"A": 2.0, "B": 0.5}
    d1 = level["by_horizon"][1]
    assert d1["n_origins"] == 4 and d1["n_failed"] == 0
    # 오차 : A 02-18 → 2.05−2.1 = −0.05, A 03-02 → 2.25−2.25 = 0, B 02-18 → 1.05−1.0 = 0.05, B 03-02 → 1.25−1.2 = 0.05
    expected = ((0.05 / 2.0) ** 2 + 0 + (0.05 / 0.5) ** 2 + (0.05 / 0.5) ** 2) / 4
    assert d1["norm_mse"] == pytest.approx(expected)
    d7 = level["by_horizon"][7]
    assert d7["n_origins"] == 4
    row = next(r for r in level["rows"] if r["station"] == "A" and r["origin"] == "2020-02-18")
    assert row["h7"]["target_date"] == "2020-02-25" and row["h7"]["err_m"] == pytest.approx(2.35 - 2.3)


def test_level_term_refuses_in_sample_forecasts(fake: FakeApi) -> None:
    fake.in_sample = True
    with pytest.raises(RuntimeError, match="표본 내"):
        script.level_term(API, REGION, ("A",), ["2020-02-18"], SPLIT_2020)


def test_level_term_excludes_clamped_origin_instead_of_scoring_it(fake: FakeApi) -> None:
    fake.clamp = True
    with pytest.raises(RuntimeError, match="성립한 예측 원점이 없습니다"):
        script.level_term(API, REGION, ("A",), ["2020-02-18"], SPLIT_2020)


def test_level_term_skips_origin_without_observation(fake: FakeApi) -> None:
    level = script.level_term(API, REGION, ("A",), ["2020-02-18", "2020-04-14"], SPLIT_2020, (1,))
    assert level["by_horizon"][1]["n_origins"] == 1 and level["by_horizon"][1]["n_failed"] == 1
    assert [r.get("error") for r in level["rows"]] == [None, "원점일 실측 없음"]


# --- [1] 학습 ----------------------------------------------------------------------------------


def test_train_polls_until_done_and_reports_progress(fake: FakeApi) -> None:
    seen: list[str] = []
    info = script.train(API, REGION, SPLIT_2020, progress=seen.append, sleep=lambda s: None)
    assert fake.trained_with == {"split": SPLIT_2020, "async_run": True}
    assert info["elapsed_s"] == 12.5 and info["n_stations"] == 2 and info["job_id"] == "job1"
    assert seen[0].startswith("1/2 지점") and seen[-1].startswith("2/2 지점")


def test_train_all_runs_stations_one_by_one_and_pauses_after_each(fake: FakeApi) -> None:
    slept: list[float] = []
    seen: list[str] = []
    info = script.train_all(API, REGION, SPLIT_2020, ("A", "B", "C"), pause_s=60, progress=seen.append, sleep=slept.append)
    assert [b["stations"] for b in fake.train_bodies] == [["A"], ["B"], ["C"]]
    assert slept.count(60) == 3  # 지점 하나가 끝날 때마다 한 번씩, 마지막 지점 뒤에도
    assert info["n_stations"] == 3 and info["elapsed_s"] == 37.5 and info["pause_total_s"] == 180.0
    assert info["stations"] == ["A", "B", "C"] and info["split"] == SPLIT_2020
    assert seen[0] == "1/3 지점 · A 학습 중 …" and seen[1] == "1/3 지점 · A 완료 (12.5초)"
    assert seen[2].startswith("60초 대기 …") and seen[-1].startswith("60초 대기 …")
    assert not any(m.startswith("0/1") or m.startswith("1/1") for m in seen)  # 잡 폴링 진행률은 따로 찍지 않는다


def test_train_all_without_pause_never_sleeps_for_the_gap(fake: FakeApi) -> None:
    slept: list[float] = []
    seen: list[str] = []
    info = script.train_all(API, REGION, SPLIT_2020, ("A", "B"), pause_s=0, progress=seen.append, sleep=slept.append)
    assert info["pause_total_s"] == 0.0 and all(t == 2.0 for t in slept)  # 남는 건 폴링 간격만
    assert not any("대기" in m for m in seen)


def _fail_station_b(fake: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    original = fake.get

    def failing(api: str, path: str, **params: Any) -> Any:
        if "/gauge/train/jobs/" in path and (fake.trained_with or {}).get("stations") == ["B"]:
            return {"status": "failed", "error": "자료 부족"}
        return original(api, path, **params)

    monkeypatch.setattr(script, "get", failing)


def test_train_all_aborts_on_failed_station_without_pausing(fake: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_station_b(fake, monkeypatch)
    slept: list[float] = []
    with pytest.raises(RuntimeError, match="학습 실패: 자료 부족"):
        script.train_all(API, REGION, SPLIT_2020, ("A", "B", "C"), pause_s=60, sleep=slept.append)
    assert [b["stations"] for b in fake.train_bodies] == [["A"], ["B"]]  # C 는 접수하지 않는다
    assert slept.count(60) == 1  # A 뒤에만 쉬고, 실패 뒤에는 쉬지 않는다


def test_main_exits_2_when_a_station_job_fails(fake: FakeApi, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _fail_station_b(fake, monkeypatch)
    assert script.main(["--api", API, "--region", REGION]) == 2
    assert "[오류] 학습 실패: 자료 부족" in capsys.readouterr().err
    assert [b["stations"] for b in fake.train_bodies] == [["A"], ["B"]]
    assert all(not p.endswith("/forecast") for _, p in fake.calls)  # 실패 뒤 예측 단계로 넘어가지 않는다


def test_train_raises_on_failed_job(fake: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    original = fake.get

    def failing(api: str, path: str, **params: Any) -> Any:
        if "/gauge/train/jobs/" in path:
            return {"status": "failed", "error": "자료 부족"}
        return original(api, path, **params)

    monkeypatch.setattr(script, "get", failing)
    with pytest.raises(RuntimeError, match="학습 실패: 자료 부족"):
        script.train(API, REGION, SPLIT_2020, sleep=lambda s: None)


# --- 한 번에 도는 흐름 ------------------------------------------------------------------------


def _run(argv: list[str]) -> tuple[int, str]:
    lines: list[str] = []
    args = script.build_parser().parse_args(argv)
    code = script.run(args, out=lines.append)
    return code, "\n".join(lines)


def test_run_trains_then_scores_both_terms_in_one_go(fake: FakeApi, tmp_path: Path) -> None:
    out_file = tmp_path / "r.json"
    code, text = _run(["--api", API, "--region", REGION, "--json-out", str(out_file)])
    assert code == 0
    # 순서 : 학습 → 수위 → 수체면적 → 지표
    assert text.index("[1/4] 수위 예측 모델 학습") < text.index("[2/4] 수위 예측") < text.index("[3/4] 수체면적 예측") < text.index("[4/4] 지표 산출")
    assert "시험 연도 : 2020 (위성 촬영 연도)" in text
    assert "예측 원점 16건" in text  # 촬영일 8 × 연계 지점 2 (satellite_location 없는 C 는 제외)
    assert "2.100" in text and "ICEYE" in text and "PlanetScope" in text  # 비교표에 실측·탐지값이 찍힌다
    assert text.count("적합") >= 2 and "부적합" not in text
    assert ("POST", f"/api/v1/regions/{REGION}/gauge/train") in fake.calls
    # 학습은 지역 지점 전부(A·B·C)를 하나씩 순서대로 — 비교(연계 2개소)와 별개로 C 도 학습한다
    assert [b["stations"] for b in fake.train_bodies] == [["A"], ["B"], ["C"]]
    assert all(b["split"] == SPLIT_2020 and b["async_run"] is True for b in fake.train_bodies)
    assert "3지점 순차 학습, 지점마다 60초 대기" in text and text.count("60초 대기 …") == 3
    assert "완료 — 3지점, 학습 37.5초 + 대기 180초" in text
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["split"] == SPLIT_2020 and set(payload["result"]) == {"D+1", "D+7"}
    assert payload["result"]["D+1"]["area_norm_mse"] == pytest.approx(0.02663, abs=5e-5)
    assert payload["train"]["elapsed_s"] == 37.5 and payload["train"]["n_stations"] == 3
    assert payload["train"]["job_ids"] == ["job1", "job1", "job1"] and payload["train"]["pause_total_s"] == 180.0


def test_run_station_pause_zero_skips_the_waits(fake: FakeApi) -> None:
    code, text = _run(["--api", API, "--region", REGION, "--station-pause", "0"])
    assert code == 0 and "대기" not in text and "3지점 순차 학습)" in text
    assert len(fake.train_bodies) == 3


def test_run_labels_fractional_pause_with_the_seconds_actually_slept(fake: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(script.time, "sleep", slept.append)
    code, text = _run(["--api", API, "--region", REGION, "--station-pause", "0.5"])
    assert code == 0 and slept.count(0.5) == 3
    assert "지점마다 0.5초 대기" in text and text.count("0.5초 대기 …") == 3 and "+ 대기 1.5초" in text


def test_run_rejects_station_pause_out_of_range_before_touching_the_api(fake: FakeApi) -> None:
    with pytest.raises(RuntimeError, match="--station-pause"):
        _run(["--api", API, "--region", REGION, "--station-pause", "-5"])
    assert fake.calls == []


def test_run_stops_when_a_compared_station_is_left_on_another_split(fake: FakeApi) -> None:
    fake.other_split = ["B"]  # 지점별 학습이 B 만 반영되지 않은 상태를 흉내낸다
    with pytest.raises(RuntimeError, match=r"다른 구간의 산출물로 남아 있습니다: \['B'\]"):
        _run(["--api", API, "--region", REGION, "--station-pause", "0"])
    fake.other_split = ["C"]  # 비교하지 않는 지점이면 상관없다
    code, _ = _run(["--api", API, "--region", REGION, "--station-pause", "0"])
    assert code == 0


def test_run_prints_scp_command_only_when_run_over_ssh(fake: FakeApi, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_json, out_txt = tmp_path / "r.json", tmp_path / "r.txt"
    monkeypatch.setenv("SSH_CONNECTION", "172.18.10.100 51000 172.18.10.113 22")
    code, text = _run(["--api", API, "--region", REGION, "--no-train", "--json-out", str(out_json), "--text-out", str(out_txt)])
    assert code == 0
    user = getpass.getuser()
    assert f"scp {user}@172.18.10.113:{out_json.resolve()} {user}@172.18.10.113:{out_txt.resolve()} ." in text
    assert "결과 파일 가져가기" not in out_txt.read_text(encoding="utf-8")  # 저장된 출력물에는 안내가 들어가지 않는다
    monkeypatch.delenv("SSH_CONNECTION")
    code, text = _run(["--api", API, "--region", REGION, "--no-train", "--json-out", str(out_json)])
    assert code == 0 and "결과 파일 가져가기" not in text  # 분석 PC 에서 직접 실행했으면 안내가 없다


def test_fetch_hint_carries_a_non_default_ssh_port_and_escapes_spaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = getpass.getuser()
    plain = tmp_path / "r.json"
    spaced = tmp_path / "결과 폴더" / "r 1.txt"
    monkeypatch.setenv("SSH_CONNECTION", "172.18.10.100 51000 172.18.10.113 10322")
    hint = script.fetch_hint([str(plain), str(spaced)])
    escaped = str(spaced.resolve()).replace(" ", "\\ ")
    assert hint == f'scp -P 10322 {user}@172.18.10.113:{plain.resolve()} "{user}@172.18.10.113:{escaped}" .'
    monkeypatch.setenv("SSH_CONNECTION", "fe80::1 51000 fe80::2 22")
    assert script.fetch_hint([str(plain)]) == f"scp {user}@[fe80::2]:{plain.resolve()} ."
    assert script.fetch_hint([]) is None


def test_run_no_train_stops_when_active_split_differs(fake: FakeApi) -> None:
    fake.split = dict(SPLIT_2020, test_start="2024-01-01", test_end="2024-12-31")
    with pytest.raises(RuntimeError, match="--no-train 없이"):
        _run(["--api", API, "--region", REGION, "--no-train"])
    assert all(m != "POST" for m, _ in fake.calls)


def test_run_no_train_uses_existing_artifacts(fake: FakeApi) -> None:
    code, text = _run(["--api", API, "--region", REGION, "--no-train"])
    assert code == 0 and "건너뜀" in text
    assert all(m != "POST" for m, _ in fake.calls)


def test_run_restore_only_deletes_user_artifacts(fake: FakeApi) -> None:
    code, text = _run(["--api", API, "--region", REGION, "--restore"])
    assert code == 0 and "되돌렸습니다" in text
    assert fake.calls[-1] == ("DELETE", f"/api/v1/regions/{REGION}/gauge/models")
    assert all(not p.endswith("/forecast") for _, p in fake.calls)


def test_main_turns_runtime_error_into_exit_2(fake: FakeApi, capsys: pytest.CaptureFixture[str]) -> None:
    fake.in_sample = True
    assert script.main(["--api", API, "--region", REGION]) == 2
    assert "[오류]" in capsys.readouterr().err


def test_table_aligns_wide_korean_cells() -> None:
    text = script.table(["지점", "값"], [["구포대교", "1.0"], ["A", "22.0"]], "<>")
    lines = text.splitlines()
    assert len({script._w(line) for line in lines}) == 1  # 한글 2칸 폭을 감안해 모든 줄 폭이 같다


# --- 검토 반영(2026-11-08) : 결과 파일 사전 확인 · 진행 중 잡 이어받기 · 예측일수 검증 · 예외 → 2 ---------------


def test_parse_horizons_rejects_garbage() -> None:
    assert script.parse_horizons("7,1") == (1, 7)
    for bad in ("1,a", "", "0", "91"):
        with pytest.raises(RuntimeError, match="--horizons"):
            script.parse_horizons(bad)


def test_run_checks_result_paths_before_training(fake: FakeApi, tmp_path: Path) -> None:
    (tmp_path / "막힘").write_text("file", encoding="utf-8")  # 디렉터리 자리에 파일이 있어 만들 수 없다
    with pytest.raises(RuntimeError, match="결과 파일을 저장할 수 없습니다"):
        _run(["--api", API, "--region", REGION, "--json-out", str(tmp_path / "막힘" / "x.json")])
    assert all(m != "POST" for m, _ in fake.calls)  # 학습을 시작하기 전에 멈춘다


def test_run_text_out_keeps_the_printed_tables(fake: FakeApi, tmp_path: Path) -> None:
    txt = tmp_path / "r.txt"
    code, text = _run(["--api", API, "--region", REGION, "--no-train", "--text-out", str(txt)])
    assert code == 0
    saved = txt.read_text(encoding="utf-8")
    assert "■ 지표" in saved and "적합" in saved and "ICEYE" in saved and "PlanetScope" in saved
    assert "화면 출력 저장" in text


def test_train_waits_for_running_job_on_409_then_submits_again(fake: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[dict[str, Any]] = []

    def conflict_once(api: str, path: str, body: dict[str, Any]) -> Any:
        fake.calls.append(("POST", path))
        posts.append(body)
        if len(posts) == 1:
            raise RuntimeError(f"POST {path} → HTTP 409: 진행 중인 학습 잡이 있습니다")
        return {"job_id": "job2", "status": "queued"}

    original_get = fake.get
    polled: list[str] = []

    def with_listing(api: str, path: str, **params: Any) -> Any:
        if path.endswith("/gauge/train/jobs"):
            return {"items": [{"job_id": "job1", "status": "running", "region_id": REGION}]}
        if "/gauge/train/jobs/" in path:
            polled.append(path.rsplit("/", 1)[-1])
            return {"status": "done", "elapsed_s": 3.0, "stations": ["A"], "split_used": SPLIT_2020,
                    "progress": {"stations_done": 1, "stations_total": 1}, "evaluation": {"trained_at": "t"}}
        return original_get(api, path, **params)

    monkeypatch.setattr(script, "post", conflict_once)
    monkeypatch.setattr(script, "get", with_listing)
    seen: list[str] = []
    info = script.train(API, REGION, SPLIT_2020, stations=["A"], progress=seen.append, sleep=lambda s: None)
    assert info["job_id"] == "job2" and any("끝나기를 기다림" in s for s in seen)
    assert polled[0] == "job1" and polled[-1] == "job2"  # 먼저 돌던 잡을 기다린 뒤 새 잡을 폴링
    assert len(posts) == 2 and posts[1] == {"split": SPLIT_2020, "async_run": True, "stations": ["A"]}


def test_train_waits_for_sync_training_lock_on_409(fake: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    """동기 학습(async_run=false)의 지역 잠금은 잡 목록에 나오지 않는다 — 409 가 풀릴 때까지 같은 요청을 다시 낸다."""
    posts = 0

    def conflict_twice(api: str, path: str, body: dict[str, Any]) -> Any:
        nonlocal posts
        posts += 1
        if posts <= 2:
            raise RuntimeError(f"POST {path} → HTTP 409: 같은 지역에 진행 중인 학습이 있습니다 (job_id=None, running)")
        return fake.post(api, path, body)

    original_get = fake.get

    def with_listing(api: str, path: str, **params: Any) -> Any:
        if path.endswith("/gauge/train/jobs"):
            return {"items": []}  # 실제 서버 모양 — 동기 잠금은 목록에 없다
        return original_get(api, path, **params)

    monkeypatch.setattr(script, "post", conflict_twice)
    monkeypatch.setattr(script, "get", with_listing)
    slept: list[float] = []
    seen: list[str] = []
    info = script.train(API, REGION, SPLIT_2020, progress=seen.append, sleep=slept.append)
    assert info["job_id"] == "job1" and posts == 3 and slept.count(2.0) >= 2
    assert any("동기 학습이 끝나기를 기다림" in m for m in seen)


def test_train_gives_up_when_409_persists_past_timeout(fake: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    def conflict(api: str, path: str, body: dict[str, Any]) -> Any:
        raise RuntimeError(f"POST {path} → HTTP 409: 잠금")

    original_get = fake.get

    def with_listing(api: str, path: str, **params: Any) -> Any:
        if path.endswith("/gauge/train/jobs"):
            return {"items": []}
        return original_get(api, path, **params)

    monkeypatch.setattr(script, "post", conflict)
    monkeypatch.setattr(script, "get", with_listing)
    with pytest.raises(RuntimeError, match="4초 안에 끝나지 않았습니다.*HTTP 409"):
        script.train(API, REGION, SPLIT_2020, poll_s=2.0, timeout_s=4.0, sleep=lambda s: None)


def test_main_maps_unexpected_exceptions_to_exit_2(fake: FakeApi, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def timed_out(api: str, path: str, **params: Any) -> Any:
        raise TimeoutError("timed out")

    monkeypatch.setattr(script, "get", timed_out)
    assert script.main(["--api", API, "--region", REGION, "--no-train"]) == 2
    assert "TimeoutError" in capsys.readouterr().err


def test_error_cells_are_truncated_so_the_table_stays_readable() -> None:
    level = {"rows": [{"station": "A", "origin": "2020-02-18", "error": "x" * 200}], "ranges_m": {}}
    text = script.render_level_table(level, (1, 7))
    assert max(script._w(line) for line in text.splitlines()) < 140
