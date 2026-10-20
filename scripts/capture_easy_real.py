#!/usr/bin/env python3
"""실제 API 에 붙은 쉬운 화면 캡처 — 최근 결과의 첫 행을 눌러 결과 카드를 찍는다.

    pipeline_ui/.venv/bin/python scripts/capture_easy_real.py http://127.0.0.1:18798 docs/presentation/screens
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

base = sys.argv[1].rstrip("/")
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--headless=new"])
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"{base}/easy", wait_until="networkidle")
    page.wait_for_timeout(1500)
    page.screenshot(path=str(out / "easy_real_01_form.png"), full_page=True)
    rows = page.locator("table tbody tr")
    rows.first.wait_for(timeout=15000)
    rows.first.click()
    page.wait_for_timeout(2500)
    page.screenshot(path=str(out / "easy_real_02_result.png"), full_page=True)
    browser.close()
print("saved", sorted(p.name for p in out.glob("easy_real_*")), "| console errors:", errors or "none")
