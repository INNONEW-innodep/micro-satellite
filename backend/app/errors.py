"""API 오류 응답 통일 + 오류 로그 (2026-10-20).

모든 오류(422 검증·404/405·HTTPException·500)를 같은 봉투로 돌려준다. 기존 호출자가 읽던 ``detail`` 은 **그대로** 두고
``error`` 블록을 덧붙인다 — 한국어 요약(``message_ko``), 필드별 사유(``fields[]``: 위치·문제·받은 값·고치는 법), 요청 id,
경로·메서드·시각, 문서 링크. 응답 헤더 ``X-Request-ID`` 로 서버 로그와 맞춰 볼 수 있다.

서버 쪽: 4xx 는 WARNING, 5xx 는 ERROR(+traceback) 로 ``watercast.api`` 로거에 남기고, 최근 오류 200건을 메모리에 보관해
``GET /api/v1/errors/recent`` 로 보여 준다(운영자 점검용, 프로세스가 살아 있는 동안만).

    {"detail": ..., "error": {"status": 422, "code": "validation_error", "message_ko": "입력 2곳이 잘못됐습니다: body.dam_code, body.start",
                              "fields": [{"loc": "body.dam_code", "msg_ko": "형식이 맞지 않습니다 (패턴 ^\\d{7}$)", "msg": "...", "input": "12", "hint_ko": "..."}],
                              "request_id": "a1b2c3d4e5f6", "method": "POST", "path": "/api/v1/regions/x/dam/collect", "timestamp": "...", "docs_url": "/docs"}}
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("watercast.api")

RECENT_LIMIT = 200
_HINTS: dict[str, str] = {
    "missing": "필수 항목입니다 — 값을 넣어 주세요",
    "string_pattern_mismatch": "형식이 맞지 않습니다",
    "string_too_short": "너무 짧습니다",
    "string_too_long": "너무 깁니다",
    "string_type": "문자열이어야 합니다",
    "int_parsing": "정수여야 합니다",
    "int_type": "정수여야 합니다",
    "float_parsing": "숫자여야 합니다",
    "float_type": "숫자여야 합니다",
    "bool_parsing": "true/false 여야 합니다",
    "bool_type": "true/false 여야 합니다",
    "date_from_datetime_parsing": "날짜는 YYYY-MM-DD 꼴이어야 합니다",
    "date_parsing": "날짜는 YYYY-MM-DD 꼴이어야 합니다",
    "date_type": "날짜는 YYYY-MM-DD 꼴이어야 합니다",
    "datetime_parsing": "시각은 YYYY-MM-DDTHH:MM 꼴이어야 합니다",
    "greater_than_equal": "허용 범위보다 작습니다",
    "greater_than": "허용 범위보다 작습니다",
    "less_than_equal": "허용 범위보다 큽니다",
    "less_than": "허용 범위보다 큽니다",
    "too_short": "항목 수가 모자랍니다",
    "too_long": "항목 수가 너무 많습니다",
    "enum": "허용된 값 중 하나여야 합니다",
    "literal_error": "허용된 값 중 하나여야 합니다",
    "json_invalid": "본문이 올바른 JSON 이 아닙니다 (따옴표·쉼표 확인)",
    "json_type": "본문은 JSON 객체여야 합니다",
    "model_attributes_type": "본문은 JSON 객체여야 합니다",
    "dict_type": "객체(JSON {})여야 합니다",
    "list_type": "목록(JSON [])이어야 합니다",
    "extra_forbidden": "허용되지 않는 항목입니다",
    "value_error": "값이 조건에 맞지 않습니다",
    "assertion_error": "값이 조건에 맞지 않습니다",
    "uuid_parsing": "id 형식이 맞지 않습니다",
    "url_parsing": "URL 형식이 맞지 않습니다",
}
_CODES: dict[int, str] = {
    400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    409: "conflict", 413: "payload_too_large", 415: "unsupported_media_type", 422: "validation_error",
    429: "too_many_requests", 500: "internal_error", 502: "upstream_error", 503: "unavailable", 504: "upstream_timeout",
}
_STATUS_KO: dict[int, str] = {
    400: "잘못된 요청입니다", 404: "찾을 수 없습니다", 405: "이 경로에서 허용되지 않는 메서드입니다", 409: "진행 중인 작업과 겹칩니다",
    413: "본문이 너무 큽니다", 415: "지원하지 않는 본문 형식입니다", 422: "입력이 잘못됐습니다", 500: "서버 안에서 오류가 났습니다",
    502: "외부 서비스가 답하지 않았습니다", 503: "지금은 처리할 수 없습니다", 504: "외부 서비스 응답이 늦습니다",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if not rid:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
    return str(rid)[:64]


def _loc_text(loc: Iterable[Any]) -> str:
    parts = [str(p) for p in loc]
    return ".".join(parts) if parts else "body"


def _trim(value: Any, limit: int = 120) -> Any:
    if isinstance(value, (dict, list)):
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "…"
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def translate_validation_error(err: Mapping[str, Any]) -> dict[str, Any]:
    """pydantic/FastAPI 검증 오류 한 건 → {loc, msg, msg_ko, input, hint_ko}."""

    etype = str(err.get("type") or "")
    msg = str(err.get("msg") or "")
    ctx = err.get("ctx") or {}
    raw_loc = tuple(err.get("loc") or ())
    if etype in ("json_invalid", "json_type", "model_attributes_type", "dict_type") and raw_loc and raw_loc[0] == "body":
        raw_loc = ("body",)  # 본문 전체가 문제인데 ('body', 1) 처럼 문자 위치가 붙는 것은 사용자에게 뜻이 없다
    loc = _loc_text(raw_loc)
    base = _HINTS.get(etype)
    if etype == "value_error" and msg:
        # 모델 validator 가 던진 한국어 사유(Value error, ... ) 는 그대로 쓴다
        detail = msg[len("Value error, "):] if msg.startswith("Value error, ") else msg
        msg_ko = detail
    elif etype == "string_pattern_mismatch":
        msg_ko = f"형식이 맞지 않습니다 (패턴 {ctx.get('pattern')})"
    elif etype in ("greater_than_equal", "greater_than", "less_than_equal", "less_than"):
        bound = ctx.get("ge", ctx.get("gt", ctx.get("le", ctx.get("lt"))))
        word = {"greater_than_equal": "이상", "greater_than": "초과", "less_than_equal": "이하", "less_than": "미만"}[etype]
        msg_ko = f"{bound} {word}이어야 합니다"
    elif etype in ("too_short", "string_too_short"):
        msg_ko = f"최소 {ctx.get('min_length')}개(자) 이상이어야 합니다" if ctx.get("min_length") is not None else base or msg
    elif etype in ("too_long", "string_too_long"):
        msg_ko = f"최대 {ctx.get('max_length')}개(자) 이하여야 합니다" if ctx.get("max_length") is not None else base or msg
    elif etype in ("enum", "literal_error"):
        msg_ko = f"허용된 값 중 하나여야 합니다: {ctx.get('expected')}" if ctx.get("expected") else base or msg
    elif etype == "json_invalid":
        msg_ko = f"{base} — {ctx.get('error')}" if ctx.get("error") else base
    elif etype in ("date_from_datetime_parsing", "date_parsing", "datetime_parsing"):
        msg_ko = f"{base} — {ctx.get('error')}" if ctx.get("error") else base
    else:
        msg_ko = base or msg
    where = loc.split(".")[0]
    hint = {
        "body": "요청 본문(JSON)의 해당 항목을 고쳐 다시 보내세요",
        "query": "주소의 ?항목=값 을 고쳐 다시 보내세요",
        "path": "주소 경로의 값을 확인하세요",
        "header": "요청 헤더를 확인하세요",
    }.get(where, "입력 값을 확인하세요")
    return {"loc": loc, "msg_ko": msg_ko, "msg": msg, "type": etype, "input": _trim(err.get("input")), "hint_ko": hint}


class RecentErrors:
    """최근 오류 봉투를 메모리에 보관(최대 RECENT_LIMIT) — ``GET /errors/recent`` 가 보여 준다."""

    def __init__(self, limit: int = RECENT_LIMIT) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=limit)
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def add(self, item: Mapping[str, Any]) -> None:
        with self._lock:
            self._items.append(dict(item))
            key = str(item.get("status"))
            self.counts[key] = self.counts.get(key, 0) + 1

    def list(self, *, limit: int = 50, min_status: int = 400) -> list[dict[str, Any]]:
        with self._lock:
            items = [i for i in self._items if int(i.get("status") or 0) >= min_status]
        return list(reversed(items))[: max(0, limit)]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.counts.clear()


def build_envelope(request: Request, status: int, *, code: str | None = None, message_ko: str | None = None, detail: Any = None,
                   fields: list[dict[str, Any]] | None = None, exception: str | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prefix = str(getattr(getattr(request.app.state, "settings", None), "api_prefix", "/api/v1"))
    error: dict[str, Any] = {
        "status": int(status),
        "code": code or _CODES.get(int(status), "error"),
        "message_ko": message_ko or _STATUS_KO.get(int(status), "요청을 처리하지 못했습니다"),
        "request_id": _request_id(request),
        "method": request.method,
        "path": request.url.path,
        "query": str(request.url.query) if request.url.query else None,
        "timestamp": _now(),
        "docs_url": "/docs",
        "recent_errors_url": f"{prefix}/errors/recent",
    }
    if fields:
        error["fields"] = fields
    if exception:
        error["exception"] = exception
    if extra:
        error.update({k: v for k, v in extra.items() if k not in error})
    return {"detail": detail if detail is not None else error["message_ko"], "error": error}


def _record(request: Request, envelope: Mapping[str, Any], *, tb: str | None = None) -> None:
    error = dict(envelope.get("error") or {})
    status = int(error.get("status") or 0)
    store: RecentErrors | None = getattr(request.app.state, "recent_errors", None)
    item = {k: error.get(k) for k in ("status", "code", "message_ko", "request_id", "method", "path", "query", "timestamp", "exception")}
    item["detail"] = _trim(envelope.get("detail"), 400)
    if error.get("fields"):
        item["fields"] = [{"loc": f.get("loc"), "msg_ko": f.get("msg_ko")} for f in error["fields"]][:10]
    item["client"] = request.client.host if request.client else None
    if tb:
        item["traceback_tail"] = tb[-1500:]
    if store is not None:
        store.add(item)
    line = f"{error.get('request_id')} {request.method} {request.url.path} -> {status} {error.get('code')}: {_trim(envelope.get('detail'), 300)}"
    if status >= 500:
        log.error("%s%s", line, ("\n" + tb) if tb else "")
    else:
        log.warning("%s", line)


def install_error_handlers(app: FastAPI) -> None:
    """예외 처리기·요청 id 미들웨어·최근 오류 조회 엔드포인트를 설치한다(멱등)."""

    if getattr(app.state, "error_handlers_installed", False):
        return
    app.state.error_handlers_installed = True
    if getattr(app.state, "recent_errors", None) is None:
        app.state.recent_errors = RecentErrors()
    prefix = str(getattr(getattr(app.state, "settings", None), "api_prefix", "/api/v1"))

    @app.middleware("http")
    async def request_id_header(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = _request_id(request)
        try:
            response = await call_next(request)
        except Exception:  # 처리기까지 못 간 예외 — 아래 generic 처리기가 받지 못하는 경우 대비
            raise
        response.headers.setdefault("X-Request-ID", rid)
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        raw = exc.errors()
        fields = [translate_validation_error(e) for e in raw]
        locs = [f["loc"] for f in fields]
        if len(fields) == 1:
            message = f"입력이 잘못됐습니다: {locs[0]} — {fields[0]['msg_ko']}"
        else:
            message = f"입력 {len(fields)}곳이 잘못됐습니다: {', '.join(locs[:6])}" + (" …" if len(locs) > 6 else "")
        # detail 은 FastAPI 기본 모양(loc 목록·msg·type·input·ctx)을 유지하되 JSON 으로 못 싣는 값(예외 객체·bytes)만 문자열로 줄인다
        safe_detail = [
            {**{k: v for k, v in e.items() if k not in ("url", "loc", "input", "ctx")}, "loc": list(e.get("loc") or ()),
             "input": _trim(e.get("input"), 300), **({"ctx": {k: _trim(v, 200) for k, v in e["ctx"].items()}} if isinstance(e.get("ctx"), dict) else {})}
            for e in raw
        ]
        envelope = build_envelope(request, 422, code="validation_error", message_ko=message, detail=safe_detail, fields=fields)
        _record(request, envelope)
        return JSONResponse(status_code=422, content=envelope, headers={"X-Request-ID": envelope["error"]["request_id"]})

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        status = int(exc.status_code)
        if isinstance(detail, str):
            message = detail
        elif isinstance(detail, Mapping) and detail.get("message_ko"):
            message = str(detail["message_ko"])
        else:
            message = _STATUS_KO.get(status, "요청을 처리하지 못했습니다")
        if status == 404 and detail == "Not Found":
            message = f"없는 경로입니다: {request.method} {request.url.path} — 전체 목록은 /docs"
        elif status == 405:
            message = f"{request.method} 는 {request.url.path} 에서 허용되지 않습니다 — /docs 에서 메서드를 확인하세요"
        envelope = build_envelope(request, status, message_ko=message, detail=detail)
        _record(request, envelope)
        headers = dict(exc.headers or {})
        headers["X-Request-ID"] = envelope["error"]["request_id"]
        return JSONResponse(status_code=status, content=envelope, headers=headers)

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        name = f"{type(exc).__name__}: {str(exc)[:300]}"
        rid = _request_id(request)
        envelope = build_envelope(
            request, 500, message_ko=f"서버 안에서 오류가 났습니다 ({type(exc).__name__}). 요청 id {rid} 로 서버 로그를 찾으세요.",
            detail=f"internal error: {name}", exception=name,
        )
        _record(request, envelope, tb=tb)
        return JSONResponse(status_code=500, content=envelope, headers={"X-Request-ID": rid})

    @app.get(f"{prefix}/errors/recent", tags=["system"], summary="Recent API errors kept in memory (newest first) — operator check",
             description="최근 오류 봉투(최대 200건, 프로세스 수명 동안)를 최신순으로 돌려준다. `limit`(기본 50), `min_status`(기본 400, 500 이면 서버 오류만). "
                         "각 항목 {status, code, message_ko, request_id, method, path, query, timestamp, detail, fields?, exception?, traceback_tail?, client}. "
                         "`counts` 는 상태코드별 누적 수.")
    async def recent_errors(request: Request, limit: int = 50, min_status: int = 400) -> dict[str, Any]:
        store: RecentErrors = request.app.state.recent_errors
        items = store.list(limit=max(1, min(int(limit), RECENT_LIMIT)), min_status=int(min_status))
        return {"items": items, "n": len(items), "counts": dict(store.counts), "limit": RECENT_LIMIT,
                "note_ko": "프로세스 메모리에만 남는 최근 오류 — 영구 기록은 컨테이너 로그(docker compose logs watercast)."}


__all__ = ["RecentErrors", "build_envelope", "install_error_handlers", "translate_validation_error"]
