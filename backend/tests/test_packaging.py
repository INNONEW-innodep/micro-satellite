"""설치 패키지(packaging/, scripts/package/build_release.sh) 정합성 — 빌드 없이 확인할 수 있는 것만.

실제 설치는 scripts/package/build_release.sh 로 묶은 뒤 빈 디렉터리에서 ./install.sh 로 검증한다.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "packaging"
SCRIPTS = sorted(PKG.glob("*.sh")) + [ROOT / "scripts" / "package" / "build_release.sh"]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert os.access(script, os.X_OK), f"{script.name} 에 실행 권한이 없다"


def test_release_compose_uses_image_not_build_and_same_env_as_dev() -> None:
    release = (PKG / "docker-compose.yml").read_text(encoding="utf-8")
    dev = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "build:" not in release and "image: ${WATERCAST_IMAGE" in release
    # 개발 compose 가 백엔드에 주는 환경변수는 설치 compose 에도 전부 있어야 한다
    dev_keys = set(re.findall(r"^\s*- ([A-Z_]+)=", dev, re.M))
    rel_keys = set(re.findall(r"^\s*- ([A-Z_]+)=", release, re.M))
    assert dev_keys <= rel_keys, f"설치 compose 에 빠진 환경변수: {sorted(dev_keys - rel_keys)}"
    assert "./backend/data:/app/backend/data" in release and "./data:/app/data:ro" in release


def test_env_example_covers_every_variable_the_compose_reads() -> None:
    release = (PKG / "docker-compose.yml").read_text(encoding="utf-8")
    example = (PKG / ".env.example").read_text(encoding="utf-8")
    used = set(re.findall(r"\$\{([A-Z_]+)(?::-[^}]*)?\}", release))
    declared = set(re.findall(r"^([A-Z_]+)=", example, re.M))
    assert used <= declared, f".env.example 에 없는 변수: {sorted(used - declared)}"


def test_image_version_tag_matches_backend_version() -> None:
    version = re.search(r'version: str = "([^"]+)"', (ROOT / "backend/app/config.py").read_text(encoding="utf-8")).group(1)
    assert f"watercast-runtime:{version}" in (PKG / ".env.example").read_text(encoding="utf-8")
    assert f"watercast-runtime:{version}" in (PKG / "docker-compose.yml").read_text(encoding="utf-8")
    assert f"watercast-runtime:0.2.0" in (PKG / "common.sh").read_text(encoding="utf-8")  # build_release.sh 가 sed 로 바꾸는 기준 문자열


def test_release_layout_runtime_image_plus_app_bundle() -> None:
    """2026-10-21: 런타임 이미지(의존성만) + 앱 바이너리(app/current 마운트) 분리 — 코드만 바뀌면 묶음만 교체."""
    release = (PKG / "docker-compose.yml").read_text(encoding="utf-8")
    assert release.count("./app/current:/app") == 2 and "working_dir: /app" in release  # watercast · pipeline-ui 둘 다
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS runtime" in dockerfile and "FROM runtime AS full" in dockerfile and dockerfile.index("AS runtime") < dockerfile.index("COPY . .")
    build = (ROOT / "scripts/package/build_release.sh").read_text(encoding="utf-8")
    assert "--target runtime" in build and "watercast-runtime-${VER}.tar.gz" in build and "--app-only" in build and "VERSION.json" in build
    assert "packaging/패키지_안내.md" in build and "packaging/배포_실행순서.md" in build  # 문서 두 장이 묶음에 들어간다
    common = (PKG / "common.sh").read_text(encoding="utf-8")
    assert "install_app_bundle()" in common and "start_log()" in common and "on_error()" in common
    for name in ("update_app.sh", "postcheck.sh", "배포_실행순서.md", "패키지_안내.md"):
        assert (PKG / name).exists(), name
    assert "--rollback" in (PKG / "update_app.sh").read_text(encoding="utf-8")
    assert "logs/postcheck_" in (PKG / "postcheck.sh").read_text(encoding="utf-8") or "postcheck_" in (PKG / "postcheck.sh").read_text(encoding="utf-8")
    runbook = (PKG / "배포_실행순서.md").read_text(encoding="utf-8")
    for key in ("install.sh", "postcheck.sh", "update_app.sh", "KMA_API_KEY", "RUN_AS", "인증이 없다", "05_l1_pre"):
        assert key in runbook, key


def test_readme_mentions_every_script_and_every_env_variable() -> None:
    readme = (PKG / "README_설치.md").read_text(encoding="utf-8")
    for script in PKG.glob("*.sh"):
        if script.name == "common.sh":
            continue
        assert script.name in readme, f"README 에 {script.name} 설명이 없다"
    for key in re.findall(r"^([A-Z_]+)=", (PKG / ".env.example").read_text(encoding="utf-8"), re.M):
        if key in ("WATERCAST_IMAGE", "WATERCAST_NAME"):
            continue
        assert key.split("_PORT")[0] in readme or key in readme, f"README 에 {key} 설명이 없다"


def test_dockerignore_keeps_dist_and_secrets_out_of_the_image() -> None:
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for needed in ("dist", ".env", "backend/data", "/data", "*.h5", ".git"):
        assert needed in ignore, f".dockerignore 에 {needed} 가 없다"
