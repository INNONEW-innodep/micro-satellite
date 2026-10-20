from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_DOCKER_BIN, DEFAULT_IMAGE

CONTAINER_MODULE_DIR = "/WBMS/wbms_modules"

log = logging.getLogger(__name__)

# ── 컨테이너 안에서 호스트 docker 를 부를 때의 모듈 전역 기본값 ─────────────────────
# 백엔드가 API 컨테이너(/app) 안에서 돌면서 호스트 /var/run/docker.sock 으로 wbms 컨테이너를
# 띄우는 구성(docker-compose.yml)에서는 두 가지가 "호스트 기준"이어야 한다.
#   1) ``docker run -v 호스트경로:컨테이너경로`` 의 호스트 경로 — argv 는 컨테이너 안에서
#      조립되지만 실제 실행은 호스트 데몬이 하므로 /app/... 을 호스트 경로로 바꿔야 한다
#      (_PATH_MAP, translate_host_path()).
#   2) 여유 VRAM 프리플라이트 — API 컨테이너에는 nvidia-smi 가 없어 docker 경유로 묻는다
#      (_GPU_PROBE 의 docker 바이너리·이미지, gpu_free_mib()).
# 어댑터(adapters/wbms_*.py)는 docker_prefix()/resolve_gpu_device() 를 설정 객체 없이 부르므로
# 값을 모듈 전역에 둔다. WbmsSettings.from_env() 가 환경변수(BACKEND_HOST_ROOT·
# BACKEND_CONTAINER_ROOT, BACKEND_WBMS_DOCKER_BIN·BACKEND_WBMS_IMAGE)로 채우고, 테스트는
# set_path_map()/set_gpu_probe() 로 직접 바꾼다. 변환표가 비어 있으면(호스트에서 직접 실행,
# BACKEND_HOST_ROOT 미설정) 예전과 똑같이 경로를 손대지 않는다.
_PATH_MAP: tuple[Path, Path] | None = None  # (container_root, host_root)
_GPU_PROBE: tuple[str, str] = (DEFAULT_DOCKER_BIN, DEFAULT_IMAGE)  # (docker_bin, image)

_RUN_AS_RE = re.compile(r"^\d+:\d+$")
_NVIDIA_SMI_QUERY = ["--query-gpu=memory.free", "--format=csv,noheader,nounits"]


def set_path_map(container_root: Path | str | None, host_root: Path | str | None) -> None:
    """docker_prefix() 가 기본으로 쓸 경로 변환표 (container_root → host_root).

    host_root 가 None 이면 변환을 끈다(호스트에서 직접 실행). WbmsSettings.from_env() 가
    호출하며, 테스트는 monkeypatch 로 ``runner._PATH_MAP`` 을 되돌린다.
    """
    global _PATH_MAP
    if container_root is None or host_root is None:
        _PATH_MAP = None
        return
    _PATH_MAP = (Path(container_root), Path(host_root))


def get_path_map() -> tuple[Path, Path] | None:
    return _PATH_MAP


def set_gpu_probe(docker_bin: str, image: str) -> None:
    """nvidia-smi 가 없을 때 VRAM 을 묻는 데 쓸 docker 바이너리와 이미지(gpu_free_mib 참고)."""
    global _GPU_PROBE
    _GPU_PROBE = (docker_bin or DEFAULT_DOCKER_BIN, image or DEFAULT_IMAGE)


def translate_host_path(path: Path | str, path_map: tuple[Path, Path] | None = None) -> Path:
    """컨테이너 안 경로를 호스트 docker 데몬이 이해하는 호스트 경로로 바꾼다.

    심볼릭 링크를 먼저 푼 뒤(resolve) 그 결과가 container_root 아래면 같은 상대 경로를
    host_root 에 붙여 돌려준다. container_root 밖 경로는 손대지 않고 원래 값을 그대로 돌려준다
    (예: 사용자가 input_tif 로 준 handover 밖 절대경로). ``path_map`` 을 주지 않으면
    set_path_map() 으로 설정된 모듈 기본값을 쓰고, 그것도 없으면 변환하지 않는다.
    """
    mapping = _PATH_MAP if path_map is None else path_map
    original = Path(path)
    if mapping is None:
        return original
    container_root, host_root = mapping
    resolved = original.resolve()
    try:
        relative = resolved.relative_to(Path(container_root).resolve())
    except ValueError:
        return original
    return Path(host_root) / relative


def run_as_user(override: str | None = None) -> str | None:
    """``docker run --user`` 값.

    BACKEND_RUN_AS(예: ``1000:1000``)가 있으면 그것을, 없으면 현재 프로세스의 uid:gid 를 쓴다.
    API 컨테이너 안에서는 프로세스가 root(0:0)라 그대로 두면 wbms 컨테이너 산출물이 호스트에서
    root 소유가 되므로 compose 가 호스트 사용자 id 를 넘긴다. 형식이 틀리면 경고만 남기고
    프로세스 uid:gid 로 돈다. uid 개념이 없는 플랫폼(Windows)에서 환경변수도 없으면 None.
    """
    raw = (os.getenv("BACKEND_RUN_AS", "") if override is None else override).strip()
    if raw:
        if _RUN_AS_RE.match(raw):
            return raw
        log.warning("BACKEND_RUN_AS 형식이 잘못됨(uid:gid 여야 함): %r — 프로세스 uid:gid 사용", raw)
    if hasattr(os, "getuid"):
        return f"{os.getuid()}:{os.getgid()}"
    return None


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

    The backend shells out to the ``docker`` CLI with ``docker run --rm`` and never talks
    to the Docker API directly. Two deployments share this one path:

    - 호스트에서 직접 실행: 호스트 docker CLI → 호스트 데몬. 경로 변환 없음.
    - API 컨테이너 안(docker-compose.yml): 이미지에 든 docker CLI → 마운트된
      /var/run/docker.sock → 호스트 데몬. 소켓은 컨테이너에 호스트 docker 권한을 주므로
      이 과제 범위(단일 서버, 내부망)에서만 허용한다(DEPLOY.md). 이때 ``-v`` 경로는
      docker_prefix() 가 호스트 경로로 바꿔 넘긴다.

    Tests mock this class, so no test ever launches a container.
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


def _parse_free_mib(stdout: str) -> int | None:
    try:
        return int(stdout.strip().splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def gpu_free_mib(docker_bin: str | None = None, image: str | None = None) -> int | None:
    """여유 VRAM(MiB). GPU 부재/오류는 None.

    1) 호스트 ``nvidia-smi`` 를 먼저 묻는다(백엔드를 호스트에서 직접 돌리는 경우).
    2) 그 바이너리가 없으면(API 컨테이너 안) ``docker run --gpus device=0`` 으로 wbms 이미지에
       nvidia-smi 를 띄워 묻는다 — nvidia-container-toolkit 이 ``--gpus`` 컨테이너에 nvidia-smi
       를 넣어 주므로 이미지 자체에 없어도 된다. 60초 제한. docker/이미지는 인자 → 모듈
       기본값(set_gpu_probe, 즉 BACKEND_WBMS_DOCKER_BIN·BACKEND_WBMS_IMAGE) 순.
    3) 둘 다 실패하면 None → auto 모드는 CPU 폴백(경고), gpu 모드는 ValueError.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", *_NVIDIA_SMI_QUERY],
            capture_output=True, text=True, timeout=10, check=False)
    except FileNotFoundError:
        pass  # API 컨테이너 안 — 아래 docker 경유로 재시도
    except Exception:  # noqa: BLE001 - 드라이버 부재 포함 전부 "사용 불가"
        return None
    else:
        return _parse_free_mib(out.stdout) if out.returncode == 0 else None
    return gpu_free_mib_via_docker(docker_bin or _GPU_PROBE[0], image or _GPU_PROBE[1])


def gpu_free_mib_via_docker(docker_bin: str, image: str, timeout_s: float = 60.0) -> int | None:
    """``docker run --rm --gpus device=0 --entrypoint nvidia-smi <image> ...`` 로 여유 VRAM 을 읽는다.

    wbms:0.13 은 ENTRYPOINT 가 없고 CMD 가 ``sleep infinity`` 라 --entrypoint 로 고정한다.
    docker 부재·소켓 미마운트·toolkit 부재·타임아웃은 전부 None(= GPU 사용 불가).
    """
    argv = [docker_bin, "run", "--rm", "--gpus", "device=0",
            "--entrypoint", "nvidia-smi", image, *_NVIDIA_SMI_QUERY]
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        log.info("docker 경유 GPU 프리플라이트 실패(exit=%s): %s",
                 out.returncode, (out.stderr or "").strip()[-200:])
        return None
    return _parse_free_mib(out.stdout)


def resolve_gpu_device(mode: str, min_free_mib: int) -> tuple[str, str | None]:
    """(device, warning) 반환. mode='gpu' 인데 여유 VRAM 부족이면 ValueError
    (라우터가 422 로 변환). 이 서버 GPU 는 vucatcher 상주 서비스와 공유라
    프리플라이트가 필수다. 프리플라이트는 gpu_free_mib() — 호스트 nvidia-smi 또는
    (API 컨테이너 안이면) docker 경유."""
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
    path_map: tuple[Path, Path] | None = None,
    run_as: str | None = None,
) -> list[str]:
    """``docker run --rm`` prefix shared by both adapters.

    ``mounts`` is a list of ``(host_path, container_path, read_only)``.
    The container runs as the backend's uid:gid so produced artifacts stay
    manageable by the backend user (the delivered image has no entrypoint and
    would otherwise write root-owned files, as the first manual smoke did).
    ``gpu=True`` 는 --gpus device=0 + TF memory growth(동거 서비스 보호)를 붙인다.

    ``path_map=(container_root, host_root)`` — 백엔드가 API 컨테이너 안에서 돌고 docker.sock
    으로 호스트 데몬에 명령할 때, 각 마운트의 호스트 경로(컨테이너 안에서 본 /app/...)를
    translate_host_path() 로 호스트 경로로 바꾼다. 생략하면 set_path_map() 값(즉
    BACKEND_HOST_ROOT/BACKEND_CONTAINER_ROOT), 그것도 없으면 변환 없음.
    ``run_as`` 는 --user 값(생략 시 BACKEND_RUN_AS → 프로세스 uid:gid, run_as_user() 참고).
    """

    argv = [docker_bin, "run", "--rm"]
    if gpu:
        argv += ["--gpus", "device=0",
                 "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
                 "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async"]
    user = run_as_user(run_as)
    if user:
        argv += ["--user", user]
    argv += ["-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp"]
    for host, container, read_only in mounts:
        suffix = ":ro" if read_only else ""
        argv += ["-v", f"{translate_host_path(host, path_map)}:{container}{suffix}"]
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
