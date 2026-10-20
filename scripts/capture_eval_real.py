#!/usr/bin/env python3
"""실제 API 에 붙은 평가 화면(/eval) 캡처 — 기존 지역을 고르고 ⑤ 성능 평가·⑥ 시험을 연다."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
base=sys.argv[1].rstrip("/"); out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); errors=[]
with sync_playwright() as p:
    b=p.chromium.launch(args=["--headless=new"]); page=b.new_page(viewport={"width":1440,"height":900})
    page.on("console", lambda m: errors.append(m.text) if m.type=="error" else None)
    page.goto(f"{base}/eval", wait_until="networkidle"); page.wait_for_timeout(1500)
    sel=page.locator("select").first
    try:
        sel.select_option("nakdong_busan"); page.wait_for_timeout(2500)
    except Exception as e: errors.append(f"select: {e}")
    page.screenshot(path=str(out/"eval_real_01.png"), full_page=True)
    for label in ("성능 평가 보기","평가 보기","성능 보기","모델 비교","전망 보기","시험하기","시험 실행"):
        btn=page.get_by_role("button", name=label)
        if btn.count(): btn.first.click(); page.wait_for_timeout(4000)
    page.screenshot(path=str(out/"eval_real_02.png"), full_page=True); b.close()
print("saved", sorted(x.name for x in out.glob("eval_real_*")), "| console errors:", errors or "none")
