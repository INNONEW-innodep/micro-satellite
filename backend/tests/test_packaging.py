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
    assert f"watercast:{version}" in (PKG / ".env.example").read_text(encoding="utf-8")
    assert f"watercast:{version}" in (PKG / "docker-compose.yml").read_text(encoding="utf-8")


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
