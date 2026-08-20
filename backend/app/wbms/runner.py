from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTAINER_MODULE_DIR = "/WBMS/wbms_modules"


@dataclass(slots=True)
class CommandResult:
    """Outcome of one host-runner subprocess invocation."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


class DockerRunner:
    """The single subprocess boundary between the backend and ``docker``.

    Security decision (fixed): the backend shells out to the host ``docker``
    CLI with ``docker run --rm``. It does not mount docker.sock into any
    container and does not talk to the Docker API directly. Tests mock this
    class, so no test ever launches a container.
    """

    def __init__(self, docker_bin: str = "docker") -> None:
        self.docker_bin = docker_bin

    def run(self, argv: list[str], timeout_s: float) -> CommandResult:
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                returncode=-1,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr),
                elapsed_s=time.monotonic() - started,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"docker executable not found: {exc}",
                elapsed_s=time.monotonic() - started,
            )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            elapsed_s=time.monotonic() - started,
        )

    def image_available(self, image: str) -> bool:
        result = self.run([self.docker_bin, "image", "inspect", image], timeout_s=30.0)
        return result.returncode == 0


def gpu_free_mib() -> int | None:
    """nvidia-smi 로 여유 VRAM(MiB)을 읽는다. GPU 부재/오류는 None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - 드라이버 부재 포함 전부 "사용 불가"
        return None


def resolve_gpu_device(mode: str, min_free_mib: int) -> tuple[str, str | None]:
    """(device, warning) 반환. mode='gpu' 인데 여유 VRAM 부족이면 ValueError
    (라우터가 422 로 변환). 이 서버 GPU 는 vucatcher 상주 서비스와 공유라
    프리플라이트가 필수다."""
    if mode not in ("cpu", "gpu", "auto"):
        raise ValueError(f"BACKEND_WBMS_GPU 값이 잘못됨: {mode!r} (cpu|gpu|auto)")
    if mode == "cpu":
        return "cpu", None
    free = gpu_free_mib()
    if free is not None and free >= min_free_mib:
        return "gpu", None
    msg = (f"GPU 여유 VRAM 부족(free={free} MiB < {min_free_mib} MiB) — "
           "상주 서비스(triton/DeepStream) 조정 후 재시도 필요")
    if mode == "gpu":
        raise ValueError(msg)
    return "cpu", msg + " · auto 모드라 CPU 로 폴백"


def docker_prefix(
    docker_bin: str,
    image: str,
    mounts: list[tuple[Path, str, bool]],
    *,
    gpu: bool = False,
) -> list[str]:
    """``docker run --rm`` prefix shared by both adapters.

    ``mounts`` is a list of ``(host_path, container_path, read_only)``.
    The container runs as the backend's uid:gid so produced artifacts stay
    manageable by the backend user (the delivered image has no entrypoint and
    would otherwise write root-owned files, as the first manual smoke did).
    ``gpu=True`` 는 --gpus device=0 + TF memory growth(동거 서비스 보호)를 붙인다.
    """

    argv = [docker_bin, "run", "--rm"]
    if gpu:
        argv += ["--gpus", "device=0",
                 "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
                 "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async"]
    if hasattr(os, "getuid"):
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
    argv += ["-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp"]
    for host, container, read_only in mounts:
        suffix = ":ro" if read_only else ""
        argv += ["-v", f"{host}:{container}{suffix}"]
    argv.append(image)
    return argv


def parse_ipc_events(stdout: str) -> list[dict[str, Any]]:
    """WBMS modules emit machine events as JSONL on stdout (ipc.py contract)."""

    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("type"):
            events.append(item)
    return events


def summarize_failure(result: CommandResult) -> str:
    """Human-readable failure line built from IPC error events and stderr."""

    if result.timed_out:
        return f"container run timed out after {result.elapsed_s:.0f}s"
    errors = [event for event in parse_ipc_events(result.stdout) if event.get("type") == "error"]
    parts: list[str] = []
    if errors:
        last = errors[-1]
        code = last.get("code") or last.get("error") or "ERROR"
        message = last.get("message") or last.get("detail") or ""
        where = last.get("param") or last.get("where") or ""
        parts.append(f"[{code}] {where}: {message}".strip())
    tail = [line for line in result.stderr.splitlines() if line.strip()][-3:]
    if tail:
        parts.append("stderr: " + " | ".join(tail))
    if not parts:
        parts.append("container exited with a non-zero status and no diagnostics")
    return f"exit={result.returncode} " + " — ".join(parts)


def _decode(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw
