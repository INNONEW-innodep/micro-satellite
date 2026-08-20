"""추가 캡처: 헤더 상단, STEP 2 깨끗한 결과 영역."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8501"
OUT = Path(__file__).parent / "screens"


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1600, "height": 900},
            device_scale_factor=2,
        )
        page = await ctx.new_page()

        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_selector("[data-testid='stSidebar']", timeout=15000)
        await asyncio.sleep(3)

        # 1) 헤더 + 진행 인디케이터만 (페이지 최상단)
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)
        # viewport 작게 줄여서 헤더만 보이게
        await page.set_viewport_size({"width": 1600, "height": 380})
        await asyncio.sleep(1)
        await page.screenshot(path=str(OUT / "00_header.png"), full_page=False)
        print("→ 00_header 저장")

        # viewport 복원
        await page.set_viewport_size({"width": 1600, "height": 900})
        await asyncio.sleep(1)

        # 2) 수체 탐지 실행 → 결과 영역만 깨끗하게
        print("→ 1차 처리 실행")
        await page.locator("button", has_text="수체 탐지 실행").first.click()
        await page.wait_for_selector("text=수체 탐지 결과", timeout=15000)
        await asyncio.sleep(3)
        # STEP 2 안의 메트릭 행으로 스크롤 (3 패널 + 메트릭이 한 화면에 다 잡히게)
        step2_title = page.locator("text=STEP 2 · 수체 탐지 결과").first
        await step2_title.scroll_into_view_if_needed()
        await asyncio.sleep(2)
        # 약간 위로 (스텝 2 카드 전체가 잡히도록)
        await page.evaluate("window.scrollBy(0, -40)")
        await asyncio.sleep(1)
        await page.screenshot(path=str(OUT / "08_step2_clean.png"), full_page=False)
        print("→ 08_step2_clean 저장")

        # 3) STEP 3 깨끗한 차트만
        step3_title = page.locator("text=STEP 3 · 기상청 데이터").first
        await step3_title.scroll_into_view_if_needed()
        await asyncio.sleep(2)
        await page.screenshot(path=str(OUT / "09_step3_clean.png"), full_page=False)
        print("→ 09_step3_clean 저장")

        # 4) 2차 처리 → 깨끗한 STEP 4 (마스크 시퀀스 위주)
        print("→ 2차 처리 실행")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
        await page.locator("button", has_text="시계열 분석 실행").first.click()
        await page.locator("text=시계열 수체 면적 변화").first.wait_for(timeout=20000)
        await asyncio.sleep(3)

        # 마스크 시퀀스가 보이는 곳으로 스크롤
        step4_title = page.locator("text=STEP 4 · 시계열 예측 결과").first
        await step4_title.scroll_into_view_if_needed()
        await asyncio.sleep(2)
        # 살짝 위로 (제목+lead가 들어가도록)
        await page.evaluate("window.scrollBy(0, -50)")
        await asyncio.sleep(1)
        await page.screenshot(path=str(OUT / "10_step4_clean.png"), full_page=False)
        print("→ 10_step4_clean 저장")

        await browser.close()
        for p in sorted(OUT.glob("*.png")):
            print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
