#!/usr/bin/env python3
"""모델 평가 화면(/eval) 스크린샷 — 스텁 서버를 띄우고 새 지역 만들기 → 수위 수집 → 기상 수집 → 학습 → 성능 평가 → 시험까지
자동으로 밟으며 찍는다.

    .venv/bin/python scripts/capture_eval_ui.py
    (옵션) --out docs/presentation/screens --keep-server

산출(1440x900 뷰포트, 전체 페이지):
    docs/presentation/screens/eval_01_region.png    ① 새 지역을 만든 직후 — 준비 상태 패널에 "다음 할 일" 셋
    docs/presentation/screens/eval_02_gauge.png     ② WAMIS 검색·선택·수집 완료 표 (자료 부족 관측소 포함)
    docs/presentation/screens/eval_03_weather.png   ③ ASOS 수집 완료 연도표 + 표본 자료 경고
    docs/presentation/screens/eval_04_train.png     ④ 구간 타임라인·후보 예측기·학습 완료
    docs/presentation/screens/eval_05_eval.png      ⑤ 지점×예측일수 표 · 모델 비교 차트(강조 2개) · 백테스트
    docs/presentation/screens/eval_06_test.png      ⑥ 시험 연도 원점 전망 vs 실측

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
        in_pw_venv = Path(sys.prefix).resolve() == PLAYWRIGHT_PY.parent.parent.resolve()
        if PLAYWRIGHT_PY.exists() and not in_pw_venv:
            os.execv(str(PLAYWRIGHT_PY), [str(PLAYWRIGHT_PY), *sys.argv])
        sys.exit("playwright 를 찾을 수 없어요. pipeline_ui/.venv/bin/python 으로 실행해 주세요.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_stub(port: int, tick: int) -> subprocess.Popen:
    python = SERVER_PY if SERVER_PY.exists() else Path(sys.executable)
    env = dict(os.environ, EVAL_STUB_TICK=str(tick), PYTHONPATH=str(ROOT))
    cmd = [str(python), "-m", "uvicorn", "backend.tests.test_eval_ui:build_stub_app", "--factory",
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
    """CSP 에 unsafe-eval 이 없어 page.wait_for_function 이 막히므로 evaluate 폴링으로 기다린다."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if page.evaluate(js):
            return
        page.wait_for_timeout(int(every_s * 1000))
    raise TimeoutError(f"조건이 {timeout_s}s 안에 참이 되지 않았어요: {js}")


def snap(page, path: Path, scroll_top: bool = True) -> None:
    """전체 페이지 캡처 — 입력 포커스를 풀고 찍는다. 호버 툴팁을 살려야 하는 장면은 scroll_top=False
    (스크롤하면 합성 mousemove 로 툴팁이 사라진다). 상단바·패널은 캡처 동안 static 이라 중간에 찍히지 않는다."""
    if scroll_top:
        page.evaluate("() => { if (document.activeElement) document.activeElement.blur(); window.scrollTo({top: 0, behavior: 'instant'}); }")
    else:
        page.evaluate("() => { if (document.activeElement) document.activeElement.blur(); }")
    page.wait_for_timeout(400)
    page.screenshot(path=str(path), full_page=True)


def hover_chart(page, selector: str, fx: float, fy: float) -> None:
    """차트를 뷰포트 안으로 끌어온 뒤 (fx, fy) 비율 지점에 마우스를 올려 크로스헤어·툴팁을 띄운다."""
    loc = page.locator(selector)
    loc.scroll_into_view_if_needed()
    page.wait_for_timeout(150)
    box = loc.bounding_box()
    page.mouse.move(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
    page.wait_for_timeout(400)


def main() -> int:
    _ensure_playwright()
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tick", type=int, default=2, help="스텁이 GET 한 번에 진행시킬 칸 수 (기본 2)")
    parser.add_argument("--keep-server", action="store_true", help="끝나도 스텁 서버를 끄지 않는다 (주소를 출력)")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    port = free_port()
    server = start_stub(port, args.tick)
    base = f"http://127.0.0.1:{port}"
    console_errors: list[str] = []
    shots: list[Path] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--headless=new", "--lang=ko-KR"])
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1, locale="ko-KR")
            page.on("console", lambda m: console_errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

            page.goto(f"{base}/eval", wait_until="load")
            page.wait_for_selector("#regionSelect option[value='nakdong_busan']", state="attached", timeout=15000)
            page.wait_for_selector("#health.is-ok", timeout=15000)
            page.wait_for_selector("#forecasterList input[type=checkbox]", state="attached", timeout=15000)
            # 전체 페이지 캡처에서 sticky 상단바·준비 상태 패널이 스크롤 위치에 따라 중간에 찍히지 않도록 캡처 동안만 static
            page.add_style_tag(content=".topbar, .panel { position: static !important; }")

            # ① 새 지역 만들기
            page.click(".segment button[data-mode='new']")
            page.fill("#nrId", "geum_sejong")
            page.fill("#nrName", "금강 세종 (신규)")
            page.fill("#nrDesc", "세종보·금남교 일대 — 모델 평가 화면 시연용")
            page.fill("#nrW", "127.20"); page.fill("#nrS", "36.40"); page.fill("#nrE", "127.40"); page.fill("#nrN", "36.60")
            page.click("#nrSubmit")
            wait_until(page, "document.querySelector('#readyRegion').textContent.indexOf('금강 세종') >= 0 && document.querySelectorAll('#readyNextList li').length >= 2", 15)
            assert page.evaluate("localStorage.getItem('watercast.eval.v1')").find("geum_sejong") >= 0
            page.wait_for_timeout(400)
            shot = out / "eval_01_region.png"
            snap(page, shot)
            shots.append(shot)

            # ② 수위 자료 수집 — 금강 유역에서 '금' 검색, 3곳 선택(하나는 결측 많아 '자료 부족' 표시)
            page.select_option("#basin", "3")
            page.fill("#wamisQuery", "금")
            page.click("#wamisSearch")
            page.wait_for_selector("#wamisResults input[type=checkbox]", state="attached", timeout=10000)
            for code in ("3009680", "3009655", "3011665"):
                page.check(f"#wamisResults input[value='{code}']")
            page.wait_for_selector("#pickedBody tr", timeout=5000)
            page.select_option("#pickedBody tr:nth-child(2) select", "dam")
            page.click("#collectBtn")
            page.wait_for_selector("#collectDone:not([hidden])", timeout=90000)
            wait_until(page, "document.querySelectorAll('#collectBody .badge--warn').length === 1 && document.querySelectorAll('#collectBody .badge--done').length === 2", 10)
            page.wait_for_timeout(500)
            shot = out / "eval_02_gauge.png"
            snap(page, shot)
            shots.append(shot)

            # ③ 기상 자료 수집 — 대전(133)
            page.select_option("#asosStation", "133")
            page.wait_for_selector("#wxBtn:not([disabled])", timeout=5000)
            page.click("#wxBtn")
            page.wait_for_selector("#wxDone:not([hidden])", timeout=120000)
            wait_until(page, "!document.querySelector('#wxSourceNote').hidden && document.querySelectorAll('#wxBody tr').length >= 10", 10)
            page.wait_for_timeout(400)
            shot = out / "eval_03_weather.png"
            snap(page, shot)
            shots.append(shot)

            # ④ 학습 — 구간은 자료 범위(2015~2025)의 기본값, 신경망 후보 둘을 끄고 14일까지
            wait_until(page, "!document.querySelector('#trainBtn').disabled", 15)
            page.uncheck("#forecasterList input[value='neural_tcn']")
            page.uncheck("#forecasterList input[value='prophet_hybrid']")
            page.select_option("#yTrainEnd", "2022"); page.select_option("#yValStart", "2023"); page.select_option("#yValEnd", "2024")
            page.select_option("#yTestStart", "2025"); page.select_option("#yTestEnd", "2025")
            page.select_option("#horizonMax", "14")
            page.click("#trainBtn")
            page.wait_for_selector("#trainDone:not([hidden])", timeout=120000)
            page.wait_for_timeout(500)
            shot = out / "eval_04_train.png"
            snap(page, shot)
            shots.append(shot)

            # ⑤ 성능 평가 — 평가표·모델 비교·백테스트가 다 그려진 뒤, 범례에서 두 모델을 강조
            page.wait_for_selector("#evalTableWrap:not([hidden])", timeout=30000)
            wait_until(page, "!!document.querySelector('#compareChart svg') && !!document.querySelector('#backtestChart svg') && document.querySelectorAll('#compareBody tr').length >= 10", 30)
            page.click("#compareLegend button:has-text('릿지 자기회귀 (AR)')")
            page.click("#compareLegend button:has-text('계절 평년값')")
            wait_until(page, "document.querySelectorAll('#compareLegend .is-on').length === 2", 5)
            assert page.evaluate("document.querySelectorAll('#evalTable tbody tr').length") == 3
            assert page.evaluate("document.querySelectorAll('#evalTable .skill--pos').length") >= 1
            # 백테스트 차트 위에 마우스를 올려 크로스헤어·툴팁이 함께 찍히게 한다
            hover_chart(page, "#backtestChart svg", 0.62, 0.45)
            wait_until(page, "!document.querySelector('#backtestChart .tip').hidden", 5)
            shot = out / "eval_05_eval.png"
            snap(page, shot, scroll_top=False)
            shots.append(shot)

            # ⑥ 시험 — 시험 연도 원점(2025-01-01 기본) · 14일 · 기본 모델
            wait_until(page, "!document.querySelector('#fcBtn').disabled && document.querySelector('#fcDate').value === '2025-01-01'", 10)
            page.select_option("#fcHorizon", "14")
            page.click("#fcBtn")
            page.wait_for_selector("#fcResult:not([hidden])", timeout=30000)
            wait_until(page, "document.querySelectorAll('#fcBody tr').length === 14 && !!document.querySelector('#fcChart svg')", 10)
            hover_chart(page, "#fcChart svg", 0.86, 0.4)
            wait_until(page, "!document.querySelector('#fcChart .tip').hidden", 5)
            shot = out / "eval_06_test.png"
            snap(page, shot, scroll_top=False)
            shots.append(shot)

            # 덤: 도움말 모달이 열리고 닫히는지
            page.click("#helpMenu summary")
            page.click("#helpBtn")
            page.wait_for_selector("#helpDialog[open]", timeout=3000)
            page.click("#helpClose")
            assert page.get_attribute("#expertLink", "href").endswith(":8501")

            browser.close()
    finally:
        if args.keep_server:
            print(f"[keep] 스텁 서버 유지: {base}/eval (pid {server.pid})")
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
