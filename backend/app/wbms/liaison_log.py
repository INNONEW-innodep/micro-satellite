"""알고리즘 적용 로그 — 통합 플랫폼 로그 API 로 JSON 전송 (규격 v1.0.0, 2026-09-18).

이벤트 4종을 보낸다. 시작(대기)·구동중·완료·에러. 페이로드::

    {"log_time": "yyyymmddHHMMSS", "log_level": "LL00000x", "system_code": "SC00000x",
     "contents": {"order_id", "scene_id", "work_id", "algm_id", "product_type", "msg",
                  "work_status", ["algm_exec_time"], ["algm_comp_time"], ["algm_proc_time"]}}

표준 라이브러리 ``urllib`` 만 쓴다(의존성 추가 없음). URL 이 비어 있으면 아무것도 보내지 않는다.
전송 실패는 잡 상태에 영향을 주지 않고 경고로만 남긴다 — 로그가 탐지를 막으면 안 된다.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

# 규격 코드표
LOG_LEVEL = {"trace": "LL000000", "debug": "LL000001", "info": "LL000002", "warn": "LL000003", "error": "LL000004"}
WORK_STATUS = {"queued": "WS000000", "running": "WS000001", "succeeded": "WS000002", "failed": "WS000003"}
# 배정 전 자리표시 — 통합 측이 수체탐지 코드를 주면 환경변수로 덮는다.
DEFAULT_SYSTEM_CODE = "SC000000"
DEFAULT_ALGM_ID = "AG000000"
DEFAULT_PRODUCT_TYPE = "PT000003"


def format_log_time(when: datetime | None = None, tz: str = "Asia/Seoul") -> str:
    """규격 ``yyyymmddHHMMSS``. 기본 KST. ISO 문자열이 오면 파싱해서 변환한다."""

    moment = when or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(ZoneInfo(tz)).strftime("%Y%m%d%H%M%S")


def _iso_to_log_time(value: str | None, tz: str) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return format_log_time(parsed, tz)


def build_log(
    *,
    status: str,
    msg: str,
    params: Mapping[str, Any],
    system_code: str,
    algm_id: str | None = None,
    product_type: str | None = None,
    exec_time: str | None = None,
    comp_time: str | None = None,
    proc_time_s: float | None = None,
    tz: str = "Asia/Seoul",
    now: datetime | None = None,
) -> dict[str, Any]:
    """규격 페이로드 하나를 만든다. ``status`` 는 우리 잡 상태(queued/running/succeeded/failed)."""

    if status not in WORK_STATUS:
        raise ValueError(f"unknown job status {status!r}")
    contents: dict[str, Any] = {
        "order_id": str(params.get("order_id") or ""),
        "scene_id": str(params.get("scene_id") or ""),
        "work_id": str(params.get("work_id") or ""),
        "algm_id": str(params.get("algm_id") or algm_id or DEFAULT_ALGM_ID),
        "product_type": str(params.get("product_type") or product_type or DEFAULT_PRODUCT_TYPE),
        "msg": msg,
        "work_status": WORK_STATUS[status],
    }
    if status == "running" and exec_time:
        contents["algm_exec_time"] = _iso_to_log_time(exec_time, tz) or exec_time
    if status == "succeeded":
        if comp_time:
            contents["algm_comp_time"] = _iso_to_log_time(comp_time, tz) or comp_time
        if proc_time_s is not None:
            contents["algm_proc_time"] = str(int(round(float(proc_time_s))))
    return {
        "log_time": format_log_time(now, tz),
        "log_level": LOG_LEVEL["error"] if status == "failed" else LOG_LEVEL["info"],
        "system_code": system_code,
        "contents": contents,
    }


class LiaisonLogger:
    """로그 API 발신기. ``url`` 이 비면 no-op. 스레드 안전."""

    def __init__(
        self,
        url: str | None,
        *,
        system_code: str = DEFAULT_SYSTEM_CODE,
        algm_id: str = DEFAULT_ALGM_ID,
        product_type: str = DEFAULT_PRODUCT_TYPE,
        timeout_s: float = 5.0,
        tz: str = "Asia/Seoul",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.url = (url or "").strip() or None
        self.system_code = system_code
        self.algm_id = algm_id
        self.product_type = product_type
        self.timeout_s = timeout_s
        self.tz = tz
        self.headers = dict(headers or {})
        self.sent: list[dict[str, Any]] = []  # 마지막 몇 건 — 테스트·진단용
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def emit(self, payload: Mapping[str, Any]) -> str | None:
        """전송. 성공이면 None, 실패면 경고 문자열(잡 warnings 에 넣는다). 비활성이면 None."""

        if not self.enabled:
            return None
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8", **self.headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                code = int(getattr(response, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            return f"liaison log rejected: HTTP {exc.code} ({payload['contents']['work_status']})"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return f"liaison log not delivered: {type(exc).__name__}: {exc}"
        with self._lock:
            self.sent.append(dict(payload))
            del self.sent[:-50]
        return None if code < 400 else f"liaison log rejected: HTTP {code}"

    def log_job(self, job: Mapping[str, Any], *, status: str, msg: str) -> str | None:
        """잡 문서 하나로 규격 이벤트를 만들어 보낸다. work_id 가 없으면 우리 job_id 를 쓴다."""

        if not self.enabled:
            return None
        request = dict(job.get("request") or {})
        params = {k: request.get(k) for k in ("order_id", "scene_id", "work_id", "algm_id", "product_type")}
        if not params.get("work_id"):
            params["work_id"] = job.get("job_id")
        result = job.get("result") or {}
        payload = build_log(
            status=status, msg=msg, params=params, system_code=self.system_code,
            algm_id=self.algm_id, product_type=self.product_type,
            exec_time=job.get("started_at"), comp_time=job.get("finished_at"),
            proc_time_s=result.get("elapsed_s") if isinstance(result, Mapping) else None,
            tz=self.tz,
        )
        return self.emit(payload)
