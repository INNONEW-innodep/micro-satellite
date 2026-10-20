"""화면용 수집 잡 저장소 — 수위(WAMIS)·기상(ASOS) 수집이 같은 꼴로 쓴다.

``backend/app/gauge/jobs.py`` 의 ``GaugeTrainJobs`` 와 같은 구조(프로세스 메모리 + ``ThreadPoolExecutor``
작업자 1개 + 지역당 동시 1개)이되, 잡의 본문(``data``)을 자유 형식 dict 로 두어 수집 종류마다
다른 진행 정보(지점별 커버리지, 연도별 행 수)를 담는다. 잡은 프로세스가 살아 있는 동안만 남는다 —
결과(지역 시계열·기상 파일)는 디스크에 남는다.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ACTIVE_STATES = frozenset({"queued", "running"})

# 잡 본문이 진행 상황을 갱신할 때 쓰는 콜백: update(progress=0.3, stations=[...]) — None 인 progress 는 그대로 둔다
ProgressUpdate = Callable[..., None]


class JobConflictError(RuntimeError):
    """같은 지역에 아직 끝나지 않은 같은 종류의 잡이 있다."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ScreenJob:
    job_id: str
    region_id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    queued_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    _started_mono: float | None = None
    _finished_mono: float | None = None
    _seq: int = 0

    def elapsed_s(self) -> float | None:
        if self._started_mono is None:
            return None
        end = self._finished_mono if self._finished_mono is not None else time.monotonic()
        return round(end - self._started_mono, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "region_id": self.region_id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(float(self.progress), 3),
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": self.elapsed_s(),
            "error": self.error,
            **{k: v for k, v in self.data.items() if k not in {"job_id", "region_id", "kind", "status", "progress", "error"}},
        }


class ScreenJobs:
    """한 종류(kind)의 수집 잡 저장소 + 단일 작업자 스레드풀."""

    def __init__(self, kind: str, max_workers: int = 1) -> None:
        self.kind = kind
        self._lock = threading.Lock()
        self._jobs: dict[str, ScreenJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"screen-{kind}")
        self._seq = 0

    # --- 조회 ----------------------------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list(self, region_id: str | None = None) -> list[dict[str, Any]]:
        """최신순."""

        with self._lock:
            jobs = [j for j in self._jobs.values() if region_id is None or j.region_id == region_id]
            jobs.sort(key=lambda j: j._seq, reverse=True)
            return [j.to_dict() for j in jobs]

    def active_for(self, region_id: str) -> dict[str, Any] | None:
        with self._lock:
            for job in self._jobs.values():
                if job.region_id == region_id and job.status in ACTIVE_STATES:
                    return job.to_dict()
            return None

    # --- 제출 ----------------------------------------------------------------------

    def _register(self, region_id: str, data: Mapping[str, Any]) -> ScreenJob:
        with self._lock:
            for job in self._jobs.values():
                if job.region_id == region_id and job.status in ACTIVE_STATES:
                    raise JobConflictError(
                        f"'{region_id}' 에 진행 중인 {self.kind} 잡이 있습니다 (job_id={job.job_id}, {job.status}) — 끝난 뒤 다시 시도하세요"
                    )
            self._seq += 1
            job = ScreenJob(job_id=uuid.uuid4().hex[:12], region_id=region_id, kind=self.kind, data=dict(data))
            job._seq = self._seq
            self._jobs[job.job_id] = job
            return job

    @contextmanager
    def _running(self, job: ScreenJob) -> Iterator[ProgressUpdate]:
        def update(progress: float | None = None, **fields: Any) -> None:
            with self._lock:
                if progress is not None:
                    job.progress = max(0.0, min(1.0, float(progress)))
                job.data.update(fields)

        with self._lock:
            job.status = "running"
            job.started_at = _now_iso()
            job._started_mono = time.monotonic()
        try:
            yield update
        except Exception as exc:  # noqa: BLE001 - 잡 실패는 상태로 기록(예외를 삼킨다)
            with self._lock:
                job.status = "failed"
                job.error = str(exc) or exc.__class__.__name__
                job.finished_at = _now_iso()
                job._finished_mono = time.monotonic()
            return
        with self._lock:
            job.status = "done"
            job.progress = 1.0
            job.finished_at = _now_iso()
            job._finished_mono = time.monotonic()

    def submit(
        self, region_id: str, run: Callable[[ProgressUpdate], Mapping[str, Any] | None], *, data: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """``run(update)`` 를 작업자 스레드에서 실행할 잡을 등록한다. 돌려주는 값(dict)은 잡 본문에 병합된다.
        같은 지역에 진행 중인 잡이 있으면 ``JobConflictError``."""

        job = self._register(region_id, data or {})
        snapshot = job.to_dict()

        def work() -> None:
            with self._running(job) as update:
                result = run(update)
                if result:
                    update(**dict(result))

        self._executor.submit(work)
        return snapshot

    def run_sync(
        self, region_id: str, run: Callable[[ProgressUpdate], Mapping[str, Any] | None], *, data: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """같은 잡을 호출자 스레드에서 끝까지 돌리고(``async_run=false``) 완료 상태를 돌려준다. 목록·조회에도 남는다."""

        job = self._register(region_id, data or {})
        with self._running(job) as update:
            result = run(update)
            if result:
                update(**dict(result))
        return job.to_dict()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


__all__ = ["ACTIVE_STATES", "JobConflictError", "ProgressUpdate", "ScreenJob", "ScreenJobs"]
