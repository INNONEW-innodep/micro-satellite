#!/usr/bin/env python3
"""서비스 화면(/app) 스크린샷 — 스텁 서버를 띄우고 입력→위치 지도→탐지 진행→시각화→예측까지 자동으로 넘기며 찍는다.

    .venv/bin/python scripts/capture_service_ui.py
    (옵션) --out docs/presentation/screens --keep-server

산출(1440x900 뷰포트, 전체 페이지):
    docs/presentation/screens/app_01_form.png      ① 입력이 끝난 화면 (지역·영상·날짜·파일)
    docs/presentation/screens/app_02_map.png       ① 위치 지도 — 새 영상 촬영범위(파란색)·덮는 지점 문장
    docs/presentation/screens/app_03_progress.png  ② 탐지 진행 중 (완료·진행·대기 섞임)
    docs/presentation/screens/app_04_viz.png       ③ 오버레이 + 면적·수위 시계열 차트
    docs/presentation/screens/app_05_predict.png   ④ 수위 전망 차트 + 면적 전망
    docs/presentation/screens/app_06_failed.png    ② 실패 카드 (덤 — 오류 안내 확인용)

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
    env = dict(os.environ, SERVICE_STUB_TICK="1", SERVICE_STUB_EVERY="2", SERVICE_STUB_SEED="1", PYTHONPATH=str(ROOT))
    cmd = [str(python), "-m", "uvicorn", "backend.tests.test_service_ui:build_stub_app", "--factory",
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
    """wait_for_function 은 페이지 안에서 eval 을 써서 CSP(script-src 에 unsafe-eval 없음)에 막힌다 → evaluate 폴링."""
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

            page.goto(f"{base}/app", wait_until="load")
            page.wait_for_selector("#region option[value='nakdong_busan']", state="attached", timeout=15000)
            page.wait_for_selector("#health.is-ok", timeout=15000)
            page.wait_for_selector("#apModel option[value='irregular-area-trend']", state="attached", timeout=15000)

            # ① 입력
            page.select_option("#region", "nakdong_busan")
            page.click(".choice[data-sensor='ICEYE']")  # 라디오는 숨겨져 있고 카드(label)가 조작 대상이다
            assert page.is_checked("input[name='sensor'][value='ICEYE']")
            page.fill("#imageDate", "2026-10-03")
            page.fill("#acqTime", "06:12:00")
            page.set_input_files("#preFile", {"name": "ICEYE_busan_20261003.tif", "mimeType": "image/tiff",
                                              "buffer": b"II*\x00" + bytes(range(256)) * 64})
            page.wait_for_selector("#startBtn:not([disabled])", timeout=5000)
            page.wait_for_timeout(400)
            shot = out / "app_01_form.png"
            snap(page, shot)
            shots.append(shot)

            # 시작 → 위치 지도 (새 촬영범위가 파란색으로 그려지고 "지점 n개" 문장이 나온 순간)
            page.click("#startBtn")
            page.wait_for_selector("#mapPanel:not([hidden])", timeout=10000)
            wait_until(page, "!!document.querySelector('#mapSentence .n')", 30)
            assert page.evaluate("document.querySelectorAll('#mapSvg polygon').length") >= 8
            sentence = page.inner_text("#mapSentence")
            assert "4개" in sentence and "구포대교" in sentence, sentence
            assert page.get_attribute("#osmLink", "href").startswith("https://www.openstreetmap.org/#map=12/")
            page.wait_for_timeout(300)
            shot = out / "app_02_map.png"
            snap(page, shot)
            shots.append(shot)

            # ② 진행 중 (①은 한 줄 요약으로 접히고 ②가 펼쳐진 상태, 완료 2개 이상 + 진행 1개)
            page.wait_for_selector("#step2.is-active.is-expanded", timeout=30000)
            assert not page.evaluate("document.querySelector('#step1').classList.contains('is-expanded')")
            assert "지점 4개" in page.inner_text("#step1Summary")
            wait_until(page, "document.querySelectorAll('#checklist .step.is-done').length >= 2 && !!document.querySelector('#checklist .step.is-running')", 40)
            page.wait_for_timeout(300)
            shot = out / "app_03_progress.png"
            snap(page, shot)
            shots.append(shot)

            # ③ 시각화 (오버레이 + 차트 두 개)
            page.wait_for_selector("#step3.is-active.is-expanded", timeout=90000)
            wait_until(page, "document.querySelector('#overlayImg').complete && document.querySelector('#overlayImg').naturalWidth > 0", 15)
            wait_until(page, "document.querySelectorAll('#areaChart svg .series path').length >= 2 && document.querySelectorAll('#levelChart svg .series path').length >= 3", 15)
            wait_until(page, "document.querySelectorAll('#levelList li').length >= 4", 10)
            wait_until(page, "document.querySelectorAll('#recentBody tr').length >= 10", 10)
            page.wait_for_timeout(600)
            shot = out / "app_04_viz.png"
            snap(page, shot)
            shots.append(shot)
            assert "km²" in page.inner_text("#areaValue")
            assert page.evaluate("document.querySelectorAll('#areaChart .marker line').length") == 1
            assert page.evaluate("document.querySelectorAll('#areaChart .chart-legend li').length") == 2

            # ④ 예측 — 수위 전망 + 면적 전망
            page.click("#toPredictBtn")
            page.wait_for_selector("#step4.is-active.is-expanded", timeout=10000)
            page.wait_for_selector("#fcStation option[value='구포대교']", state="attached", timeout=10000)
            assert page.evaluate("document.querySelector('#fcStation').value") in ("낙동강하구언(내)", "구포대교")
            page.select_option("#fcStation", "구포대교")
            page.select_option("#fcHorizon", "7")
            page.click("#fcBtn")
            page.wait_for_selector("#fcResult:not([hidden])", timeout=15000)
            wait_until(page, "document.querySelectorAll('#fcChart svg .series path').length >= 2 && document.querySelectorAll('#fcChart svg .band').length === 1 && document.querySelectorAll('#fcBody tr').length === 7", 10)
            page.select_option("#apHorizon", "7")
            page.click("#apBtn")
            page.wait_for_selector("#apResult:not([hidden])", timeout=15000)
            wait_until(page, "document.querySelectorAll('#apBody tr').length === 7 && document.querySelectorAll('#apChart svg .series path').length === 1", 10)
            page.wait_for_timeout(500)
            shot = out / "app_05_predict.png"
            snap(page, shot)
            shots.append(shot)

            # 최근 결과 줄 클릭 → 실패 장면 열기 → 실패 카드
            page.click("#recentBody tr.scene-row:has(.badge--failed)")
            page.wait_for_selector("#failedBox:not([hidden])", timeout=10000)
            assert page.inner_text("#failedText").strip()
            page.wait_for_timeout(400)
            shot = out / "app_06_failed.png"
            snap(page, shot)
            shots.append(shot)

            # 다시 시도 → ① 폼이 다시 보인다
            page.click("#retryBtn")
            page.wait_for_selector("#formArea:not([hidden])", timeout=5000)
            assert page.evaluate("document.querySelector('#step1').classList.contains('is-active')")

            browser.close()
    finally:
        if args.keep_server:
            print(f"[keep] 스텁 서버 유지: {base}/app (pid {server.pid})")
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
