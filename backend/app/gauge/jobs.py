"""백그라운드 학습 잡 — 프로세스 메모리 안의 작은 저장소.

``POST .../gauge/train`` 에 ``async_run=true`` 가 오면 요청 검증(지점·예측기·분할)은 라우터에서
동기로 끝내고, 학습 자체는 ``ThreadPoolExecutor(max_workers=1)`` 에서 한 번에 하나씩 돈다.
지역당 동시에 하나만 허용한다(같은 산출물 디렉터리를 두 학습이 동시에 쓰면 평가표가 깨진다).
잡은 프로세스가 살아 있는 동안만 남는다 — 재시작하면 목록이 비고, 산출물은 디스크에 남는다.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ACTIVE_STATES = frozenset({"queued", "running"})


class JobConflictError(RuntimeError):
    """같은 지역에 아직 끝나지 않은 잡이 있다."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TrainJob:
    job_id: str
    region_id: str
    split_used: dict[str, str]
    custom_split: bool
    stations: list[str]
    forecasters: list[str] | None
    horizon_max: int | None
    status: str = "queued"
    queued_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    _started_mono: float | None = None
    _finished_mono: float | None = None
    stations_total: int = 0
    stations_done: int = 0
    current_station: str | None = None
    error: str | None = None
    evaluation: dict[str, Any] | None = None
    _seq: int = 0

    def elapsed_s(self) -> float | None:
        if self._started_mono is None:
            return None
        end = self._finished_mono if self._finished_mono is not None else time.monotonic()
        return round(end - self._started_mono, 3)

    def to_dict(self, *, include_evaluation: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": self.job_id,
            "region_id": self.region_id,
            "status": self.status,
            "progress": {
                "stations_total": self.stations_total,
                "stations_done": self.stations_done,
                "current_station": self.current_station,
            },
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": self.elapsed_s(),
            "split_used": dict(self.split_used),
            "custom_split": self.custom_split,
            "stations": list(self.stations),
            "forecasters": list(self.forecasters) if self.forecasters else None,
            "horizon_max": self.horizon_max,
            "error": self.error,
            "evaluation": (self.evaluation if include_evaluation and self.status == "done" else None),
        }
        return out


class GaugeTrainJobs:
    """지역별 학습 잡 저장소 + 단일 작업자 스레드풀."""

    def __init__(self, max_workers: int = 1) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, TrainJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gauge-train")
        self._seq = 0
        self._sync_holds: dict[str, str] = {}  # 동기 학습이 점유한 지역 → 시작 시각

    def _active_locked(self, region_id: str) -> str | None:
        """(락 안에서) 진행 중인 잡·동기 학습 설명. 없으면 None."""
        for job in self._jobs.values():
            if job.region_id == region_id and job.status in ACTIVE_STATES:
                return f"job_id={job.job_id}, {job.status}"
        if region_id in self._sync_holds:
            return f"동기 학습 실행 중 (시작 {self._sync_holds[region_id]})"
        return None

    @contextmanager
    def hold(self, region_id: str) -> Iterator[None]:
        """동기 학습도 같은 지역 잠금을 잡는다 — 그 사이 비동기 잡 접수·산출물 초기화가 409 로 막힌다."""
        with self._lock:
            active = self._active_locked(region_id)
            if active:
                raise JobConflictError(f"'{region_id}' 에 진행 중인 학습이 있습니다 ({active}) — 끝난 뒤 다시 시도하세요")
            self._sync_holds[region_id] = _now_iso()
        try:
            yield
        finally:
            with self._lock:
                self._sync_holds.pop(region_id, None)

    # --- 조회 ----------------------------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list(self, region_id: str | None = None) -> list[dict[str, Any]]:
        """최신순. 목록에서는 평가표 본문을 빼 응답을 가볍게 한다."""

        with self._lock:
            jobs = [j for j in self._jobs.values() if region_id is None or j.region_id == region_id]
            jobs.sort(key=lambda j: j._seq, reverse=True)
            return [j.to_dict(include_evaluation=False) for j in jobs]

    def active_for(self, region_id: str) -> dict[str, Any] | None:
        with self._lock:
            for job in self._jobs.values():
                if job.region_id == region_id and job.status in ACTIVE_STATES:
                    return job.to_dict(include_evaluation=False)
            if region_id in self._sync_holds:
                return {"job_id": None, "region_id": region_id, "status": "running", "kind": "sync", "started_at": self._sync_holds[region_id]}
            return None

    # --- 제출 ----------------------------------------------------------------------

    def submit(
        self,
        region_id: str,
        run: Callable[[Callable[[str | None, int, int], None]], dict[str, Any]],
        *,
        split_used: dict[str, str],
        custom_split: bool,
        stations: Iterable[str],
        forecasters: Iterable[str] | None,
        horizon_max: int | None,
        finalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """``run(on_progress)`` 를 작업자 스레드에서 실행할 잡을 등록한다. 같은 지역에 진행 중인 잡이 있으면
        ``JobConflictError``. ``finalize`` 는 완료된 평가표를 응답 형태로 다듬는 훅(예: custom_split·trained_at 추가)."""

        station_ids = [str(s) for s in stations]
        with self._lock:
            active = self._active_locked(region_id)
            if active:
                raise JobConflictError(f"'{region_id}' 에 진행 중인 학습이 있습니다 ({active}) — 끝난 뒤 다시 시도하세요")
            self._seq += 1
            job = TrainJob(
                job_id=uuid.uuid4().hex[:12],
                region_id=region_id,
                split_used=dict(split_used),
                custom_split=custom_split,
                stations=station_ids,
                forecasters=[str(f) for f in forecasters] if forecasters else None,
                horizon_max=horizon_max,
                stations_total=len(station_ids),
            )
            job._seq = self._seq
            self._jobs[job.job_id] = job
            snapshot = job.to_dict()

        def on_progress(current: str | None, done: int, total: int) -> None:
            with self._lock:
                job.current_station = current
                job.stations_done = int(done)
                job.stations_total = int(total)

        def work() -> None:
            with self._lock:
                job.status = "running"
                job.started_at = _now_iso()
                job._started_mono = time.monotonic()
            try:
                evaluation = run(on_progress)
                if finalize is not None:
                    evaluation = finalize(evaluation)
            except Exception as exc:  # noqa: BLE001 - 잡 실패는 상태로 기록
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc) or exc.__class__.__name__
                    job.finished_at = _now_iso()
                    job._finished_mono = time.monotonic()
                    job.current_station = None
                return
            with self._lock:
                job.evaluation = evaluation
                job.status = "done"
                job.finished_at = _now_iso()
                job._finished_mono = time.monotonic()
                job.current_station = None
                job.stations_done = job.stations_total

        self._executor.submit(work)
        return snapshot

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
