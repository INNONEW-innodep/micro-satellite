from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"succeeded", "failed"}


class JobNotFoundError(KeyError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FileJobStore:
    """Deliberately simple file-backed job state: one JSON per job.

    ``data/wbms_jobs/<job_id>.json`` is the single source of truth, so job
    status survives backend restarts and stays inspectable with plain tools.
    Only the submitting process mutates a job; readers get a liveness fixup
    for jobs orphaned by a dead backend process.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def new_job_id(prefix: str = "seg") -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"

    def create(self, job: dict[str, Any]) -> dict[str, Any]:
        job.setdefault("created_at", utc_now())
        job.setdefault("runner_pid", os.getpid())
        self._write(job)
        return job

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        job = self._read(job_id)
        job.update(fields)
        self._write(job)
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        return self._fixup_stale(self._read(job_id))

    def list(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return jobs
        for path in self.root.glob("*.json"):
            try:
                jobs.append(self._fixup_stale(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        jobs.sort(key=lambda job: str(job.get("created_at", "")), reverse=True)
        return jobs

    def _path(self, job_id: str) -> Path:
        safe = "".join(ch for ch in job_id if ch.isalnum() or ch in "_-")
        if not safe or safe != job_id:
            raise JobNotFoundError(job_id)
        return self.root / f"{safe}.json"

    def _read(self, job_id: str) -> dict[str, Any]:
        path = self._path(job_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise JobNotFoundError(job_id) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise JobNotFoundError(job_id) from exc

    def _write(self, job: dict[str, Any]) -> None:
        path = self._path(str(job["job_id"]))
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        tmp.replace(path)

    def _fixup_stale(self, job: dict[str, Any]) -> dict[str, Any]:
        """A queued/running job whose owning process died can never finish."""

        if job.get("status") in TERMINAL_STATUSES:
            return job
        pid = job.get("runner_pid")
        if not isinstance(pid, int) or pid == os.getpid() or _pid_alive(pid):
            return job
        job.update(
            status="failed",
            finished_at=utc_now(),
            error=(
                "backend process that owned this job is no longer running; "
                "the container run outcome is unknown — resubmit the job"
            ),
        )
        self._write(job)
        return job


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
