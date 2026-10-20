#!/usr/bin/env python3
"""실제 API 에 붙은 서비스 화면(/app) 캡처 — 최근 결과 첫 행을 눌러 4단계까지 연다."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
base=sys.argv[1].rstrip("/"); out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); errors=[]
with sync_playwright() as p:
    b=p.chromium.launch(args=["--headless=new"]); page=b.new_page(viewport={"width":1440,"height":900})
    page.on("console", lambda m: errors.append(m.text) if m.type=="error" else None)
    page.goto(f"{base}/app", wait_until="networkidle"); page.wait_for_timeout(1500)
    page.screenshot(path=str(out/"app_real_01.png"), full_page=True)
    rows=page.locator("table tbody tr"); rows.first.wait_for(timeout=15000); rows.first.click(); page.wait_for_timeout(6000)
    page.screenshot(path=str(out/"app_real_02.png"), full_page=True)
    # 예측 버튼이 보이면 눌러 본다
    for label in ("수위 전망 보기","면적 전망 보기"):
        btn=page.get_by_role("button", name=label)
        if btn.count(): btn.first.click(); page.wait_for_timeout(5000)
    page.screenshot(path=str(out/"app_real_03.png"), full_page=True); b.close()
print("saved", sorted(x.name for x in out.glob("app_real_*")), "| console errors:", errors or "none")
