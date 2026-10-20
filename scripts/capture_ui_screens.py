#!/usr/bin/env python3
"""업무 UI(ui_next)를 실제로 띄워 화면을 넘기며 스크린샷을 찍는다 (발표자료용).

    pipeline_ui/.venv/bin/python scripts/capture_ui_screens.py --api http://127.0.0.1:18765 --ui http://127.0.0.1:18766

백엔드·Streamlit 이 이미 떠 있어야 한다. 산출: docs/presentation/screens/*.png
Streamlit 은 본문이 내부 스크롤 컨테이너라 full_page 캡처가 잘리므로, 캡처 전에 뷰포트 높이를 본문 높이로 늘린다.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "presentation" / "screens"
WIDTH = 1600
BASE_HEIGHT = 1000


async def settle(page, ms: int = 2500) -> None:
    await page.wait_for_timeout(ms)
    for _ in range(60):  # Streamlit 실행 표시가 사라질 때까지
        if await page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        await page.wait_for_timeout(500)


async def expand_viewport(page) -> int:
    """본문 스크롤 높이만큼 뷰포트를 키워 한 장에 다 보이게 한다."""
    main = page.locator('[data-testid="stMain"]').first
    # 본문은 min-height:100vh 라 늘어난 뷰포트를 기준으로 재면 계속 커진다 → 기본 높이에서 잰다
    await page.set_viewport_size({"width": WIDTH, "height": BASE_HEIGHT})
    await page.wait_for_timeout(400)
    height = int(await main.evaluate("e => e.scrollHeight"))
    height = min(max(height + 40, BASE_HEIGHT), 9000)
    await page.set_viewport_size({"width": WIDTH, "height": height})
    await page.wait_for_timeout(700)
    return height


async def click_button(page, text: str, *, exact: bool = False, timeout: int = 15000, nth: int = 0) -> bool:
    locator = page.get_by_role("button", name=text, exact=exact)
    try:
        await locator.nth(nth).wait_for(timeout=timeout)
        await locator.nth(nth).scroll_into_view_if_needed()
        await locator.nth(nth).click()
        await settle(page)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] button '{text}' not clicked: {exc}", file=sys.stderr)
        return False


async def shot(page, name: str) -> None:
    await settle(page, 800)
    await expand_viewport(page)
    path = OUT / name
    await page.screenshot(path=str(path), full_page=False)
    print("wrote", path.name)


async def crop(page, name: str, anchor: str, *, height: int = 900, above: int = 12, exact: bool = False) -> None:
    """앵커 텍스트가 있는 곳부터 height 픽셀만 잘라 찍는다(발표용 확대 컷).

    본문이 뷰포트 상한(9000px)보다 길면 전체를 펼칠 수 없으므로, 앵커를 화면 위로 스크롤한 뒤
    현재 보이는 영역에서 잘라낸다.
    """
    await settle(page, 300)
    total = await expand_viewport(page)
    try:
        loc = page.get_by_text(anchor, exact=exact).first
        await loc.wait_for(timeout=8000)
        box = await loc.bounding_box()
        if box is None:
            raise RuntimeError("no bounding box")
        viewport = page.viewport_size or {"height": BASE_HEIGHT}
        top = max(0, int(box["y"]) - above)
        if top + height > viewport["height"]:
            # 앵커를 최상단으로 스크롤해 보이는 영역 안에서 자른다
            await loc.scroll_into_view_if_needed()
            await page.evaluate("el => el.scrollIntoView({block: 'start'})", await loc.element_handle())
            await page.wait_for_timeout(700)
            box = await loc.bounding_box()
            if box is None:
                raise RuntimeError("no bounding box after scroll")
            top = max(0, int(box["y"]) - above)
        available = max(200, min(height, viewport["height"] - top, total - top))
        path = OUT / name
        await page.screenshot(path=str(path), clip={"x": 0, "y": top, "width": WIDTH, "height": available})
        print("wrote", path.name)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] crop '{name}' ({anchor}): {exc}", file=sys.stderr)


async def open_sidebar(page) -> None:
    sidebar = page.locator('section[data-testid="stSidebar"]').first
    expanded = await sidebar.get_attribute("aria-expanded")
    if expanded != "true":
        btn = page.locator('[data-testid="stExpandSidebarButton"]').first
        if await btn.count():
            await btn.click()
            await page.wait_for_timeout(800)


async def close_sidebar(page) -> None:
    for sel in ('[data-testid="stSidebarCollapseButton"]', '[data-testid="stSidebarCollapseButton"] button', 'section[data-testid="stSidebar"] button[aria-label*="sidebar" i]'):
        btn = page.locator(sel).first
        if await btn.count():
            try:
                await btn.click(timeout=3000)
                await page.wait_for_timeout(600)
                return
            except Exception:  # noqa: BLE001
                continue


async def set_ui_mode(page, label: str) -> None:
    await open_sidebar(page)
    radio = page.locator('section[data-testid="stSidebar"]').get_by_text(label, exact=True).first
    await radio.scroll_into_view_if_needed()
    await radio.click()
    await settle(page, 3000)
    await close_sidebar(page)


async def set_api_url(page, api: str) -> None:
    try:
        await open_sidebar(page)
        await page.locator('section[data-testid="stSidebar"]').get_by_text("백엔드 연결 (고급)").first.click()
        await settle(page, 800)
        box = page.get_by_role("textbox", name="FastAPI 주소")
        await box.fill(api)
        await box.press("Enter")
        await settle(page, 2500)
        await close_sidebar(page)
    except Exception as exc:  # noqa: BLE001
        print("  [warn] api url:", exc, file=sys.stderr)


async def run(api: str, ui: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--headless=new"])
        page = await browser.new_page(viewport={"width": WIDTH, "height": BASE_HEIGHT}, device_scale_factor=1.25)
        await page.goto(ui, wait_until="networkidle")
        await settle(page, 4000)
        if api:
            await set_api_url(page, api)

        # ---- 쉬운 화면 6장 (부산)
        await shot(page, "10_slide1_region_select.png")
        await click_button(page, "이 지역 보기")  # 첫 카드 = 낙동강(위성 있는 지역 우선)
        await shot(page, "11_slide2_data_overview.png")
        await crop(page, "11a_slide2_gauge_history.png", "수위계 기록 살펴보기", height=1000)
        await crop(page, "11b_slide2_station_map.png", "지상 수위계", height=800)
        await click_button(page, "다음: 위성 수체 탐지 ▶")
        await shot(page, "12_slide3_detection_busan_iceye.png")
        await crop(page, "12a_slide3_detection_overlay.png", "위성 영상", height=1300, exact=False)
        await crop(page, "12b_slide3_satellite_vs_gauge.png", "위성이 잰 수위 vs 지상 수위계", height=900)
        await crop(page, "12c_slide3_location_overlay.png", "어디를 잰 것인가", height=1100)
        try:
            await page.get_by_text("PlanetScope (광학)", exact=True).first.click()
            await settle(page)
            await shot(page, "13_slide3_detection_busan_planet.png")
            await crop(page, "13a_slide3_detection_planet_overlay.png", "위성 영상", height=1300)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] sensor switch:", exc, file=sys.stderr)
        await click_button(page, "다음: 시계열 예측 ▶")
        await shot(page, "14_slide4_forecast_setup.png")
        await click_button(page, "수위 예측 실행 ▶")
        await shot(page, "15_slide4_forecast_run.png")
        # 학습·검증·시험 구간 컨트롤(펼치기만 한다 — 재학습 버튼은 누르지 않는다: 서버 산출물이 바뀐다)
        try:
            await page.get_by_text("학습·검증·시험 구간 직접 정하기", exact=False).first.click()
            await settle(page, 1500)
            await crop(page, "15a_slide4_training_window.png", "학습·검증·시험 구간 직접 정하기", height=900)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] training window expander:", exc, file=sys.stderr)
        try:
            await page.get_by_text("모든 모델로 한 번에 비교하기", exact=False).first.click()
            await settle(page, 8000)
            await shot(page, "16_slide4_model_compare.png")
            await crop(page, "16a_model_compare_controls.png", "순위 기준 지표", height=520)
            await crop(page, "16b_model_compare_chart.png", "전망 비교", height=760)
            await crop(page, "16c_model_ranking.png", "모델 순위", height=700)
            # 예측 구간만 + cm 축
            try:
                await page.get_by_text("예측 구간만 보기", exact=True).first.click()
                await settle(page, 2500)
                await crop(page, "16d_model_compare_forecast_only.png", "전망 비교", height=760)
            except Exception as exc:  # noqa: BLE001
                print("  [warn] forecast-only toggle:", exc, file=sys.stderr)
            # 차이 그래프 (선택 모델 대비)
            try:
                await page.get_by_text("차이 (선택 모델 대비)", exact=True).first.click()
                await settle(page, 2500)
                await crop(page, "16e_model_compare_difference.png", "전망 비교", height=760)
            except Exception as exc:  # noqa: BLE001
                print("  [warn] difference view:", exc, file=sys.stderr)
            # 오차 그래프 (실제 대비) + 상위 3
            try:
                await page.get_by_text("오차 (실제 대비)", exact=True).first.click()
                await settle(page, 2500)
                await click_button(page, "상위 3", exact=True)
                await crop(page, "16f_model_compare_error_top3.png", "전망 비교", height=760)
                await click_button(page, "전체", exact=True)
            except Exception as exc:  # noqa: BLE001
                print("  [warn] error view:", exc, file=sys.stderr)
            await crop(page, "16g_model_drilldown.png", "자세히 볼 모델", height=1000)
            try:
                await page.get_by_text("지평별 모델 예측과 선택 모델 대비 차이", exact=False).first.click()
                await settle(page, 1500)
                await crop(page, "16h_model_compare_diff_table.png", "지평별 모델 예측과 선택 모델 대비 차이", height=700)
            except Exception as exc:  # noqa: BLE001
                print("  [warn] diff expander:", exc, file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] compare toggle:", exc, file=sys.stderr)
        try:
            await page.get_by_role("tab", name="② 수체 면적 예측").first.click()
            await settle(page)
            await click_button(page, "수체 면적 예측 실행 ▶")
            await shot(page, "17_slide4_area_forecast.png")
        except Exception as exc:  # noqa: BLE001
            print("  [warn] area tab:", exc, file=sys.stderr)
        await click_button(page, "다음: 결과 그래프 ▶")
        await shot(page, "18_slide5_results.png")
        await crop(page, "18a_slide5_level_result.png", "수위 예측", height=900)
        await click_button(page, "다음: 성능 평가 ▶")
        await shot(page, "19_slide6_evaluation_busan.png")
        await crop(page, "19a_slide6_horizon_table.png", "예측일수별 한눈에 보기", height=700)
        await crop(page, "19b_slide6_model_metrics.png", "모델별 성능지표 · D+", height=760)
        try:
            await page.get_by_text("지점 × 모델 시험 RMSE 행렬", exact=False).first.click()
            await settle(page, 1500)
            await crop(page, "19c_slide6_station_model_matrix.png", "지점 × 모델 시험 RMSE 행렬", height=600)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] matrix expander:", exc, file=sys.stderr)
        # 미래 강수(완전 예보 가정) 상한 실험 — 펼침 상자를 연 뒤 찍는다(실험 산출물이 없으면 안내 문구만 찍힌다)
        try:
            await page.get_by_text("미래 강수(예보)를 넣으면", exact=False).first.click()
            await settle(page, 1500)
            await crop(page, "19g_slide6_future_rain.png", "미래 강수(예보)", height=900)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] future-rain expander:", exc, file=sys.stderr)
        # 수체 면적 시계열 모델 직접 비교(LOO) — 펼침 상자를 연 뒤 찍는다(실험 산출물이 없으면 안내 문구만 찍힌다)
        try:
            await page.get_by_text("수체 면적 시계열 모델", exact=False).first.click()
            await settle(page, 1500)
            await crop(page, "19h_slide6_area_models.png", "수체 면적 시계열 모델", height=1500)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] area-models expander:", exc, file=sys.stderr)
        await crop(page, "19d_slide6_backtest.png", "시험 연도 전체를 예측해 본 결과", height=800)
        await crop(page, "19e_slide6_satellite_metrics.png", "위성 데이터로 잰 수위·수체 시계열 지표", height=900)
        await crop(page, "19f_slide6_detection_accuracy.png", "위성 수체 탐지 정확도", height=700)

        # ---- 대청
        await click_button(page, "처음으로 ↺")
        try:
            card = page.locator('[data-testid="stColumn"]', has_text="대청").first
            await card.get_by_role("button", name="이 지역 보기").first.click()
            await settle(page)
            await click_button(page, "다음: 위성 수체 탐지 ▶")
            await shot(page, "20_slide3_detection_daecheong.png")
            await crop(page, "20a_slide3_detection_daecheong_overlay.png", "위성 영상", height=1300)
            await click_button(page, "다음: 시계열 예측 ▶")
            await click_button(page, "수위 예측 실행 ▶")
            await shot(page, "22_slide4_forecast_daecheong.png")
            await click_button(page, "다음: 결과 그래프 ▶")
            await click_button(page, "다음: 성능 평가 ▶")
            await shot(page, "21_slide6_evaluation_daecheong.png")
            await crop(page, "21a_slide6_daecheong_model_metrics.png", "모델별 성능지표 · D+", height=760)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] daecheong:", exc, file=sys.stderr)

        # ---- 위성 없는 새 지역(영산강): 업로드 화면 + 예측·평가
        await click_button(page, "처음으로 ↺")
        await shot(page, "10b_slide1_region_select_all.png")
        try:
            card = page.locator('[data-testid="stColumn"]', has_text="영산강").first
            await card.get_by_role("button", name="이 지역 보기").first.click()
            await settle(page)
            await shot(page, "23_slide2_data_yeongsan.png")
            await click_button(page, "다음: 위성 수체 탐지 ▶")
            await shot(page, "24_slide3_upload_form_yeongsan.png")
            await crop(page, "24a_slide3_upload_form.png", "아직 위성 장면이 없습니다", height=1100)
            await click_button(page, "다음: 시계열 예측 ▶")
            await click_button(page, "수위 예측 실행 ▶")
            await shot(page, "25_slide4_forecast_yeongsan.png")
            await click_button(page, "다음: 결과 그래프 ▶")
            await crop(page, "26_slide5_result_yeongsan.png", "수위 예측", height=900)
            await click_button(page, "다음: 성능 평가 ▶")
            await crop(page, "27_slide6_eval_yeongsan.png", "예측일수별 한눈에 보기", height=700)
            await crop(page, "27a_slide6_eval_yeongsan_models.png", "모델별 성능지표 · D+", height=760)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] yeongsan:", exc, file=sys.stderr)

        # ---- 위성 파이프라인 화면
        try:
            await set_ui_mode(page, "위성 파이프라인 (입력→예측)")
            await shot(page, "30_pipeline_busan.png")
            await click_button(page, "수위 예측 실행 ▶")
            await click_button(page, "면적 예측 실행 ▶")
            await shot(page, "31_pipeline_busan_forecasts.png")
            try:
                await page.get_by_text("모델은 어떤 것들이고", exact=False).first.click()
                await settle(page, 1500)
                await crop(page, "32_pipeline_model_guide.png", "모델은 어떤 것들이고", height=1400)
            except Exception as exc:  # noqa: BLE001
                print("  [warn] guide:", exc, file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print("  [warn] pipeline mode:", exc, file=sys.stderr)

        # ---- 고급 화면 · 지역 발표
        try:
            await set_ui_mode(page, "고급 화면 (상세 탭)")
            await click_button(page, "지역 발표")
            await shot(page, "40_presentation_page_busan.png")
        except Exception as exc:  # noqa: BLE001
            print("  [warn] presentation page:", exc, file=sys.stderr)
        await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="", help="비우면 UI 의 기본(WATER_API_URL)을 그대로 쓴다")
    parser.add_argument("--ui", default="http://127.0.0.1:18766")
    args = parser.parse_args()
    asyncio.run(run(args.api, args.ui))
    return 0


if __name__ == "__main__":
    sys.exit(main())
