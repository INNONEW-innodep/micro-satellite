"""scripts/fetch_wamis_levels.py 의 네트워크 없는 부분 — 잘린 응답 파싱·후보 파싱·커버리지·series.json 본문."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("fetch_wamis_levels", ROOT / "scripts" / "fetch_wamis_levels.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("fetch_wamis_levels", module)
    spec.loader.exec_module(module)
    return module


def test_parse_rows_recovers_truncated_json() -> None:
    mod = _load()
    payload = '{"result":{"code":"success"},"count":3,"list":[{"ymd":"20240101","wl":"4.10"},{"ymd":"20240102","wl":""},{"ymd":"20240103","wl":"4.2'
    assert mod.parse_rows(payload) == [("20240101", 4.10), ("20240102", None)]
    full = '{"result":{"code":"success"},"count":1,"list":[{"ymd":"20240101","wl":"4.10"}]}'
    assert mod.parse_rows(full) == [("20240101", 4.10)]


def test_station_list_regex_recovers_partial_list() -> None:
    mod = _load()
    partial = '{"result":{"code":"success"},"count":2,"list":[{"bbsnnm":"영산강","obscd":"5001610","obsnm":"담양군(담양댐)","clsyn":"운영","obsknd":"T/M","sbsncd":"500103","mngorg":"기후에너지환경부"},{"bbsnnm":"영산강","obscd":"5001615","obsnm":"담양군(금'
    found = mod._STATION_ROW.findall(partial)
    assert len(found) == 1 and found[0][1] == "5001610" and found[0][2] == "담양군(담양댐)"


def test_parse_station_arg_and_validation() -> None:
    mod = _load()
    assert mod.parse_station_arg("나주대교=5004650, 담양댐=5001610") == {"나주대교": "5004650", "담양댐": "5001610"}
    with pytest.raises(SystemExit):
        mod.parse_station_arg("나주대교=abc")


def test_coverage_and_series_payload() -> None:
    mod = _load()
    rows = [{"date": f"2024{m:02d}01", "water_level_m": 1.0 + m, "obscd": "0000001"} for m in range(1, 13)]
    report = mod.coverage_report({"a": rows}, "20240101", "20241231")
    assert report["a"]["n_days"] == 12 and report["a"]["expected_days"] == 366
    assert report["a"]["coverage"] == pytest.approx(12 / 366, abs=1e-4)
    spec = [{"id": "a", "name": "A", "code": "0000001", "terrain": "mainstem", "terrain_note": "본류"}, {"id": "b", "name": "B", "code": "0000002"}]
    payload = mod.build_series_payload("toy", spec, {"a": rows}, "20240101", "20241231")
    assert payload["schema"] == "watercast.gauge.series/1" and payload["fetched_range"] == ["2024-01-01", "2024-12-31"]
    assert [s["id"] for s in payload["stations"]] == ["a"]  # 자료 없는 b 는 빠진다
    assert payload["stations"][0]["source_code"] == "0000001" and payload["stations"][0]["terrain"] == "mainstem"
    assert payload["series"]["a"][0] == {"date": "2024-01-01", "water_level_m": 2.0}
