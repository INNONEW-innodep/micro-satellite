"""
Streamlit 앱의 각 단계별 스크린샷을 캡처합니다.
"""
import asyncio
import time
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8501"
OUT = Path(__file__).parent / "screens"
OUT.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        # 큰 화면(노트북 풀스크린)
        ctx = await browser.new_context(
            viewport={"width": 1600, "height": 1000},
            device_scale_factor=2,  # 고해상도
        )
        page = await ctx.new_page()

        print("→ 페이지 접속")
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_selector("[data-testid='stSidebar']", timeout=15000)
        await asyncio.sleep(3)

        # ---- 01. 초기 (STEP 1 입력 화면) ----
        print("→ 01. 초기 화면")
        await page.screenshot(path=str(OUT / "01_step1_input.png"),
                              full_page=False)
        # 전체 페이지 (사이드바 포함)
        await page.screenshot(path=str(OUT / "01_step1_full.png"),
                              full_page=True)

        # ---- 02. 사이드바 강조 ----
        print("→ 02. 사이드바")
        sidebar = await page.query_selector("[data-testid='stSidebar']")
        if sidebar:
            await sidebar.screenshot(path=str(OUT / "02_sidebar.png"))

        # ---- 03. 1차 처리 버튼 클릭 (수체 탐지) ----
        print("→ 03. 1차 처리 실행")
        # Find detect button (text contains "수체 탐지 실행")
        await page.get_by_role("button", name="1차 처리: 수체 탐지 실행", exact=False).click()
        # 진행 → 결과 렌더 대기
        await asyncio.sleep(5)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)
        await page.screenshot(path=str(OUT / "03_step2_detection.png"),
                              full_page=True)

        # STEP 2 영역만 캡처
        # 페이지를 STEP 2 (🌊 수체 탐지 결과)로 스크롤
        # 텍스트로 찾기
        try:
            heading = await page.query_selector("text=STEP 2")
            if heading:
                await heading.scroll_into_view_if_needed()
                await asyncio.sleep(1)
                await page.screenshot(path=str(OUT / "03_step2_view.png"),
                                      full_page=False)
        except Exception as e:
            print("  step2 scroll err:", e)

        # ---- 04. 기상 데이터 ----
        print("→ 04. 기상 데이터")
        try:
            heading = await page.query_selector("text=STEP 3")
            if heading:
                await heading.scroll_into_view_if_needed()
                await asyncio.sleep(2)
                await page.screenshot(path=str(OUT / "04_step3_weather.png"),
                                      full_page=False)
        except Exception as e:
            print("  step3 scroll err:", e)

        # ---- 05. 2차 처리 버튼 클릭 (시계열 분석) ----
        print("→ 05. 2차 처리 실행")
        # 페이지 하단으로 스크롤해서 버튼 보이게
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
        try:
            # 정확한 텍스트로 찾기
            btn = page.locator("button", has_text="시계열 분석 실행")
            await btn.first.click(timeout=10000)
            print("  버튼 클릭 성공, 결과 대기 중...")
            # 진행률 바 표시 시간(약 1.5초) + rerun + 렌더링
            await asyncio.sleep(10)
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(3)
        except Exception as e:
            print("  2차 처리 클릭 실패:", e)

        # STEP 4 결과 캡처 - 결과 안의 고유 요소를 anchor 로 사용
        try:
            # 결과 영역 안에만 존재하는 텍스트로 wait
            await page.locator("text=시계열 수체 면적 변화").first.wait_for(timeout=15000)
            print("  STEP 4 결과 렌더링 확인됨")
        except Exception as e:
            print("  결과 대기 실패:", e)

        try:
            # 1) STEP 4 상단 (마스크 시퀀스)
            step4_title = page.locator("text=STEP 4 · 시계열 예측 결과").first
            await step4_title.scroll_into_view_if_needed()
            await asyncio.sleep(2)
            await page.screenshot(path=str(OUT / "05_step4_top.png"),
                                  full_page=False)
            print("  → step4_top 저장")

            # 2) 차트 영역
            chart = page.locator("text=시계열 수체 면적 변화").first
            await chart.scroll_into_view_if_needed()
            await asyncio.sleep(2)
            await page.screenshot(path=str(OUT / "05_step4_charts.png"),
                                  full_page=False)
            print("  → step4_charts 저장")

            # 3) 메트릭 + 위험평가
            risk = page.locator("text=위험 평가").first
            if await risk.count() > 0:
                await risk.scroll_into_view_if_needed()
                await asyncio.sleep(2)
                await page.screenshot(path=str(OUT / "05_step4_metrics.png"),
                                      full_page=False)
                print("  → step4_metrics 저장")

            # 4) 3세부 연계 태그 부분
            link3 = page.locator("text=3세부 시뮬레이션 연계").first
            if await link3.count() > 0:
                await link3.scroll_into_view_if_needed()
                await asyncio.sleep(2)
                await page.screenshot(path=str(OUT / "05_step4_link.png"),
                                      full_page=False)
        except Exception as e:
            print("  step4 capture err:", e)

        # 전체 페이지 (모든 STEP 완료 상태)
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(2)
        await page.screenshot(path=str(OUT / "06_complete_full.png"),
                              full_page=True)

        # 상단 헤더 + 진행 인디케이터
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)
        # 헤더 + 파이프라인 인디케이터까지만 잘라서
        await page.set_viewport_size({"width": 1600, "height": 350})
        await asyncio.sleep(1)
        await page.screenshot(path=str(OUT / "07_header_top.png"),
                              full_page=False)
        await page.set_viewport_size({"width": 1600, "height": 1000})

        await browser.close()
        print(f"\n✓ 저장 위치: {OUT}")
        for p in sorted(OUT.glob("*.png")):
            print(f"  - {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
