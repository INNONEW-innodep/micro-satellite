"""오류 응답 통일(2026-10-20) — 422/404/405/HTTPException/500 모두 같은 봉투(detail 유지 + error 블록), X-Request-ID, 최근 오류 조회."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.testclient import TestClient

from backend.app.errors import translate_validation_error
from backend.app.screens_api import install_screens_api
from backend.tests.test_gauge import _client


def _app(tmp_path: Path) -> TestClient:
    client = _client(tmp_path)
    install_screens_api(client.app)
    return client


def test_validation_error_has_korean_fields_and_keeps_detail(tmp_path: Path) -> None:
    client = _app(tmp_path)
    r = client.post("/api/v1/regions/toy/dam/collect", json={"dam_code": "12", "start": "2024-13-01"})
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], list) and body["detail"][0]["loc"] == ["body", "dam_code"]  # 기존 호출자 호환
    err = body["error"]
    assert err["status"] == 422 and err["code"] == "validation_error" and err["method"] == "POST" and err["path"].endswith("/dam/collect")
    assert "입력 2곳이 잘못됐습니다" in err["message_ko"] and "body.dam_code" in err["message_ko"]
    locs = {f["loc"]: f for f in err["fields"]}
    assert "패턴" in locs["body.dam_code"]["msg_ko"] and locs["body.dam_code"]["input"] == "12" and "본문" in locs["body.dam_code"]["hint_ko"]
    assert "YYYY-MM-DD" in locs["body.start"]["msg_ko"]
    assert r.headers["x-request-id"] == err["request_id"] and len(err["request_id"]) >= 8
    assert err["recent_errors_url"].endswith("/api/v1/errors/recent")


def test_single_field_message_and_query_hint(tmp_path: Path) -> None:
    client = _app(tmp_path)
    r = client.get("/api/v1/regions/toy/gauge/A/forecast?as_of=hello")
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["message_ko"].startswith("입력이 잘못됐습니다: query.as_of") and "주소의" in err["fields"][0]["hint_ko"]


def test_bad_json_body_is_explained(tmp_path: Path) -> None:
    client = _app(tmp_path)
    r = client.post("/api/v1/regions/toy/dam/collect", content=b"{bad", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "JSON" in r.json()["error"]["fields"][0]["msg_ko"]


def test_model_validator_korean_reason_passes_through(tmp_path: Path) -> None:
    client = _app(tmp_path)
    r = client.post("/api/v1/regions/toy/dam/collect", json={"dam_code": "3008110", "start": "2024-02-01", "end": "2024-01-01"})
    assert r.status_code == 422
    assert "end 가 start 보다 앞입니다" in r.json()["error"]["fields"][0]["msg_ko"]


def test_http_exceptions_404_405_and_custom_detail_get_envelope(tmp_path: Path) -> None:
    client = _app(tmp_path)
    r = client.get("/api/v1/regions/nope/gauge/hourly")
    assert r.status_code == 404 and r.json()["detail"] == "unknown region 'nope'" and r.json()["error"]["message_ko"] == "unknown region 'nope'"
    assert r.json()["error"]["code"] == "not_found" and r.headers["x-request-id"]
    r = client.get("/api/v1/nothing")
    assert r.status_code == 404 and "없는 경로입니다" in r.json()["error"]["message_ko"] and r.json()["detail"] == "Not Found"
    r = client.delete("/api/v1/health")
    assert r.status_code == 405 and "허용되지 않습니다" in r.json()["error"]["message_ko"]
    r = client.post("/api/v1/regions/toy/weather/forecast-collect", json={})
    assert r.status_code == 422 and "KMA_API_KEY" in r.json()["error"]["message_ko"]  # HTTPException(422, 한국어 사유)


def test_unhandled_exception_returns_500_envelope_with_request_id_and_is_logged(tmp_path: Path, caplog) -> None:
    client = _client(tmp_path)
    router = APIRouter()

    @router.get("/api/v1/_boom")
    async def boom() -> dict:
        raise RuntimeError("kaboom")

    client.app.include_router(router)
    client = TestClient(client.app, raise_server_exceptions=False)
    with caplog.at_level("ERROR", logger="watercast.api"):
        r = client.get("/api/v1/_boom", headers={"X-Request-ID": "trace-123"})
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["request_id"] == "trace-123" and err["exception"].startswith("RuntimeError: kaboom") and "trace-123" in err["message_ko"]
    assert r.headers["x-request-id"] == "trace-123" and r.json()["detail"].startswith("internal error: RuntimeError")
    assert any("trace-123" in rec.message and "kaboom" in rec.message for rec in caplog.records)
    recent = client.get("/api/v1/errors/recent?min_status=500").json()
    assert recent["n"] >= 1 and recent["items"][0]["request_id"] == "trace-123" and "kaboom" in recent["items"][0]["traceback_tail"]


def test_recent_errors_lists_newest_first_with_counts(tmp_path: Path) -> None:
    client = _app(tmp_path)
    client.get("/api/v1/regions/nope/gauge/hourly")
    client.post("/api/v1/regions/toy/dam/collect", json={"dam_code": "12", "start": "2024-01-01"})
    body = client.get("/api/v1/errors/recent?limit=5").json()
    assert body["n"] >= 2 and body["items"][0]["status"] == 422 and body["items"][1]["status"] == 404
    assert body["items"][0]["fields"][0]["loc"] == "body.dam_code" and body["counts"]["422"] >= 1 and body["counts"]["404"] >= 1
    assert client.get("/api/v1/errors/recent?min_status=500").json()["n"] == 0


def test_translate_validation_error_covers_common_types() -> None:
    assert translate_validation_error({"type": "missing", "loc": ("body", "x"), "msg": "Field required"})["msg_ko"].startswith("필수 항목")
    t = translate_validation_error({"type": "greater_than_equal", "loc": ("query", "h"), "msg": "", "ctx": {"ge": 1}, "input": 0})
    assert t["msg_ko"] == "1 이상이어야 합니다" and t["input"] == 0
    assert "최대 3개" in translate_validation_error({"type": "too_long", "loc": ("body", "l"), "msg": "", "ctx": {"max_length": 3}})["msg_ko"]
    assert translate_validation_error({"type": "weird_new_type", "loc": ("body",), "msg": "raw msg"})["msg_ko"] == "raw msg"
    assert translate_validation_error({"type": "value_error", "loc": ("body",), "msg": "Value error, 사유"})["msg_ko"] == "사유"
