#!/usr/bin/env python3
"""쉬운 화면(/easy) 스크린샷 — 스텁 서버를 띄우고 폼 입력→파일 올리기→진행→결과까지 자동으로 넘기며 찍는다.

    .venv/bin/python scripts/capture_easy_ui.py
    (옵션) --out docs/presentation/screens --keep-server

산출(1440x900 뷰포트, 전체 페이지):
    docs/presentation/screens/easy_01_form.png      1·2단계 입력이 끝난 화면
    docs/presentation/screens/easy_02_progress.png  3단계 진행 중(완료·진행·대기 섞임)
    docs/presentation/screens/easy_03_result.png    결과 카드 + 최근 결과
    docs/presentation/screens/easy_04_failed.png    실패 카드 (덤 — 오류 안내 확인용)

구동 환경: FastAPI/uvicorn 은 저장소 .venv 에, Playwright/Chromium 은 pipeline_ui/.venv 에 있다.
서버는 항상 .venv 로 띄우고, 현재 인터프리터에 playwright 가 없으면 pipeline_ui/.venv 로 스스로 다시 실행한다.
브라우저 콘솔의 error / 페이지 예외가 하나라도 있으면 실패(exit 1)한다.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_PY = ROOT / ".venv" / "bin" / "python"
PLAYWRIGHT_PY = ROOT / "pipeline_ui" / ".venv" / "bin" / "python"
DEFAULT_OUT = ROOT / "docs" / "presentation" / "screens"
VIEWPORT = {"width": 1440, "height": 900}


def _ensure_playwright() -> None:
    try:
        import playwright  # noqa: F401
    except ImportError:
        # 두 venv 모두 같은 시스템 python 을 가리키므로 실행 파일이 아니라 venv 디렉터리(sys.prefix)로 구분한다
        in_pw_venv = Path(sys.prefix).resolve() == PLAYWRIGHT_PY.parent.parent.resolve()
        if PLAYWRIGHT_PY.exists() and not in_pw_venv:
            os.execv(str(PLAYWRIGHT_PY), [str(PLAYWRIGHT_PY), *sys.argv])
        sys.exit("playwright 를 찾을 수 없어요. pipeline_ui/.venv/bin/python 으로 실행해 주세요.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_stub(port: int) -> subprocess.Popen:
    python = SERVER_PY if SERVER_PY.exists() else Path(sys.executable)
    env = dict(os.environ, EASY_STUB_TICK="1", EASY_STUB_SEED="1", PYTHONPATH=str(ROOT))
    cmd = [str(python), "-m", "uvicorn", "backend.tests.test_easy_ui:build_stub_app", "--factory",
           "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    deadline = time.time() + 30
    url = f"http://127.0.0.1:{port}/api/v1/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"스텁 서버가 바로 종료됐어요 (exit {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=1) as res:  # noqa: S310
                if res.status == 200:
                    return proc
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("스텁 서버가 30초 안에 뜨지 않았어요")


def wait_until(page, js: str, timeout_s: float = 60, every_s: float = 0.25) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if page.evaluate(js):
            return
        page.wait_for_timeout(int(every_s * 1000))
    raise TimeoutError(f"조건이 {timeout_s}s 안에 참이 되지 않았어요: {js}")


def snap(page, path: Path) -> None:
    """전체 페이지 캡처 — 고정 상단바가 중간에 찍히지 않도록 맨 위로 올리고, 입력 포커스를 푼다."""
    page.evaluate("() => { if (document.activeElement) document.activeElement.blur(); window.scrollTo({top: 0, behavior: 'instant'}); }")
    page.wait_for_timeout(350)
    page.screenshot(path=str(path), full_page=True)


def main() -> int:
    _ensure_playwright()
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-server", action="store_true", help="끝나도 스텁 서버를 끄지 않는다 (주소를 출력)")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    port = free_port()
    server = start_stub(port)
    base = f"http://127.0.0.1:{port}"
    console_errors: list[str] = []
    shots: list[Path] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--headless=new", "--lang=ko-KR"])
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1, locale="ko-KR")
            page.on("console", lambda m: console_errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

            page.goto(f"{base}/easy", wait_until="load")
            page.wait_for_selector("#region option[value='busan_nakdong']", state="attached", timeout=15000)
            page.wait_for_selector("#health.is-ok", timeout=15000)

            # 1단계 입력
            page.select_option("#region", "busan_nakdong")
            page.click(".choice[data-sensor='ICEYE']")  # 라디오는 숨겨져 있고 카드(label)가 조작 대상이다
            assert page.is_checked("input[name='sensor'][value='ICEYE']")
            page.fill("#imageDate", "2026-10-03")
            page.fill("#acqTime", "06:12:00")
            # 2단계 파일
            page.set_input_files("#preFile", {"name": "ICEYE_busan_20261003.tif", "mimeType": "image/tiff",
                                              "buffer": b"II*\x00" + bytes(range(256)) * 64})
            page.wait_for_selector("#startBtn:not([disabled])", timeout=5000)
            page.wait_for_timeout(500)
            shot = out / "easy_01_form.png"
            snap(page, shot)
            shots.append(shot)

            # 시작 → 진행 중(완료 2개 이상 + 진행 1개) 포착
            page.click("#startBtn")
            page.wait_for_selector("#step3:not([hidden])", timeout=10000)
            wait_until(page, "document.querySelectorAll('#checklist .step.is-done').length >= 2 && !!document.querySelector('#checklist .step.is-running')", 40)
            page.wait_for_timeout(300)
            shot = out / "easy_02_progress.png"
            snap(page, shot)
            shots.append(shot)

            # 결과
            page.wait_for_selector("#result:not([hidden])", timeout=90000)
            # wait_for_function 은 페이지 안에서 eval 을 써서 CSP(script-src 에 unsafe-eval 없음)에 막힌다 → evaluate 폴링
            wait_until(page, "document.querySelector('#overlayImg').complete && document.querySelector('#overlayImg').naturalWidth > 0", 15)
            wait_until(page, "document.querySelectorAll('#recentBody tr').length >= 3", 10)
            page.wait_for_timeout(800)
            shot = out / "easy_03_result.png"
            snap(page, shot)
            shots.append(shot)
            area_text = page.inner_text("#areaValue")
            assert "30.15" in area_text, area_text
            assert page.get_attribute("#btnExpert", "href").endswith(":8501")

            # 최근 결과 줄 클릭 → 같은 결과 렌더(실패 장면 열기)
            page.click("#recentBody tr.scene-row:has(.badge--failed)")
            page.wait_for_selector("#failed:not([hidden])", timeout=10000)
            assert page.inner_text("#failedText").strip()

            # 실패 경로 (덤): 메모에 fail 을 적고 다시 올리면 스텁이 탐지 단계에서 실패시킨다
            page.click("#retryBtn")
            page.wait_for_selector("#failed[hidden]", state="attached", timeout=5000)
            page.evaluate("document.querySelector('#advanced').open = true")
            page.fill("#f_note", "fail 테스트")
            page.set_input_files("#preFile", {"name": "ICEYE_busan_20261003_b.tif", "mimeType": "image/tiff", "buffer": b"II*\x00" * 64})
            page.wait_for_selector("#startBtn:not([disabled])", timeout=5000)
            page.click("#startBtn")
            page.wait_for_selector("#failed:not([hidden])", timeout=60000)
            page.wait_for_timeout(600)
            shot = out / "easy_04_failed.png"
            snap(page, shot)
            shots.append(shot)

            browser.close()
    finally:
        if args.keep_server:
            print(f"[keep] 스텁 서버 유지: {base}/easy (pid {server.pid})")
        else:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

    for s in shots:
        print(f"saved {s.relative_to(ROOT) if s.is_relative_to(ROOT) else s} ({s.stat().st_size // 1024} KB)")
    if console_errors:
        print("브라우저 콘솔 오류:", file=sys.stderr)
        for line in console_errors:
            print("  " + line, file=sys.stderr)
        return 1
    print("콘솔 오류 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
