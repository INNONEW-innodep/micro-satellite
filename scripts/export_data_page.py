#!/usr/bin/env python3
"""시험 자료 현황 화면(/data)을 서버 없이 열 수 있는 정지 사본(HTML)으로 떠내고, 원하면 PDF 로도 찍는다.

서버가 돌고 있는 분석 PC 에서 실행한다. 화면이 부르는 API 응답(지역·수위관측소 일 수위·위성 장면·수체 면적 계열·
기상)을 전부 받아 HTML 안에 넣고, 화면의 fetch 를 그 안에 든 응답으로 돌려 준다. 퀵룩 이미지는 작게 줄여 같이 넣는다.
결과 HTML 은 브라우저로 그냥 열면 되고(지점 전환·구간 탭·CSV 내려받기까지 동작), 값은 떠낸 시각의 서버 응답 그대로다.
'원본 JSON'·'서비스 화면' 링크는 서버 주소로 절대 경로를 적어 두므로 분석 PC 에 닿는 환경에서만 열린다.

    python3 scripts/export_data_page.py                                  # ~/tta/WATERCAST_시험자료현황_nakdong_busan_<날짜>.html
    python3 scripts/export_data_page.py --pdf                            # 같은 이름의 .pdf 도 (헤드리스 크롬 필요)
    python3 scripts/export_data_page.py --api http://localhost:18000 --public http://172.18.10.113:18000 --region nakdong_busan --out-dir ~/tta

PDF 는 HTML 을 '#print' 로 열어 찍는다 — 그 모드에서는 ① 수위 절이 관측소마다 한 번씩 펼쳐져(기본 구간 = 시험 연도)
모든 관측소의 표가 들어간다. 표준 라이브러리만 쓰고, Pillow 가 있으면 퀵룩을 JPEG 로 줄인다(없으면 PNG 원본을 넣는다).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

DEFAULT_API = "http://localhost:18000"
DEFAULT_REGION = "nakdong_busan"
THUMB_MAX_SIDE = 640
CHROME_CANDIDATES = (
    "~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
    "~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
)


# ----------------------------------------------------------------------------- 받기
def http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/json, image/png;q=0.9, */*;q=0.5"})
    with urllib.request.urlopen(req, timeout=timeout) as res:  # noqa: S310 — 분석 PC 로컬 API
        return res.read()


def shrink_image(data: bytes, max_side: int = THUMB_MAX_SIDE) -> tuple[bytes, str]:
    """퀵룩을 긴 변 max_side 로 줄인 JPEG. Pillow 가 없거나 실패하면 원본 PNG 그대로."""
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover — Pillow 없는 환경
        return data, "image/png"
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:  # pragma: no cover — 깨진 이미지
        return data, "image/png"


def data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def collect(api: str, region: str, fetch: Callable[[str], bytes] = http_get, *, images: bool = True) -> dict[str, Any]:
    """화면이 부르는 응답 전부를 {API 경로: 응답} 으로. 경로 키는 퍼센트 인코딩을 푼 형태(화면 쪽에서 decodeURIComponent 로 맞춘다)."""
    api = api.rstrip("/")
    prefix = "/api/v1"

    def get_json(path: str) -> Any:
        return json.loads(fetch(api + prefix + path).decode("utf-8"))

    out: dict[str, Any] = {}
    out[f"{prefix}/health"] = get_json("/health")
    regions = get_json("/regions")
    items = [r for r in regions.get("items", []) if r.get("id") == region]
    if not items:
        raise SystemExit(f"지역 '{region}' 이 서버 지역 목록에 없습니다: {[r.get('id') for r in regions.get('items', [])]}")
    out[f"{prefix}/regions"] = {**regions, "items": items, "n": len(items)}
    detail = get_json(f"/regions/{urllib.parse.quote(region, safe='')}")
    out[f"{prefix}/regions/{region}"] = detail
    try:
        scenes = get_json(f"/regions/{urllib.parse.quote(region, safe='')}/satellite/scenes")
    except Exception:
        scenes = {"scenes": [], "n_scenes": 0}
    n_images = 0
    for sc in scenes.get("scenes", []):
        urls = sc.get("image_urls") or {}
        kept: dict[str, str] = {}
        href = urls.get("overlay") or urls.get("mask") or urls.get("input")
        if href and images:
            try:
                raw = fetch(api + href)
                small, mime = shrink_image(raw)
                kept["overlay"] = data_uri(small, mime)
                n_images += 1
            except Exception:
                kept = {}
        sc["image_urls"] = kept
        sc.pop("frame_url", None)
    out[f"{prefix}/regions/{region}/satellite/scenes"] = scenes
    for suffix, fallback in (("/satellite/area-series", {"available": False}), ("/weather", None)):
        try:
            out[f"{prefix}/regions/{region}{suffix}"] = get_json(f"/regions/{urllib.parse.quote(region, safe='')}{suffix}")
        except Exception:
            if fallback is not None:
                out[f"{prefix}/regions/{region}{suffix}"] = fallback
    for st in detail.get("stations", []):
        sid = str(st.get("id") or "")
        if not sid:
            continue
        out[f"{prefix}/regions/{region}/gauge/{sid}/series"] = get_json(f"/regions/{urllib.parse.quote(region, safe='')}/gauge/{urllib.parse.quote(sid, safe='')}/series")
    out["__meta__"] = {"n_images": n_images, "n_stations": len(detail.get("stations", [])), "n_scenes": len(scenes.get("scenes", []))}
    return out


# ----------------------------------------------------------------------------- 만들기
def js_literal(obj: Any) -> str:
    """스크립트 안에 넣어도 안전한 JSON — '</' 와 '<!--' 를 끊는다."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")


PRINT_CSS = """
  /* ---- 정지 사본: 인쇄·PDF */
  @page { size: A4; margin: 11mm 10mm; }
  .snap { margin:10px 0 0; padding:10px 14px; border-radius:12px; background:#fff3d6; color:#5c3d00; font-size:.86rem; line-height:1.5; border:1.5px solid #f0d9a0; }
  .snap b { font-weight:800; }
  @media print {
    html { font-size:11.5px; } body { background:#fff; }
    .wrap { width:100%; max-width:none; padding:0; }
    header nav, .actions, #regionSel, .tabs, .toolbar { display:none !important; }
    .card { break-inside:auto; border:1px solid #cfd6df; box-shadow:none; margin-top:10px; }
    h2 { break-after:avoid; }
    .tile, .sensor h3, tr, .thumb, .src, .legend, .chart, .chips { break-inside:avoid; }
    .tscroll { max-height:none !important; overflow:visible !important; }  /* 화면의 520px 스크롤 상자를 풀어 표가 끝까지 인쇄되게 */
    table { font-size:9.5px; } th, td { padding:2px 5px !important; }
    .thumbs img { max-height:170px; }
    a { color:inherit; text-decoration:none; }
  }
"""


def shim_script(snapshot: dict[str, Any]) -> str:
    return (
        "<script>\n"
        "// 정지 사본: 화면의 fetch 를 아래에 담긴 서버 응답으로 돌려 준다(네트워크 없음)\n"
        "window.__SNAPSHOT__ = " + js_literal(snapshot) + ";\n"
        "(function () {\n"
        "  var SNAP = window.__SNAPSHOT__;\n"
        "  SNAP.printMode = location.hash === '#print';  // 페이지 스크립트가 replaceState 로 '#print' 를 지우기 전에 잡아 둔다\n"
        "  function key(u) { u = String(u); var i = u.indexOf('/api/v1'); u = i >= 0 ? u.slice(i) : u; u = u.split('?')[0]; try { u = decodeURIComponent(u); } catch (e) { /* 그대로 */ } return u; }\n"
        "  function reply(status, body) { return Promise.resolve({ ok: status < 400, status: status, text: function () { return Promise.resolve(JSON.stringify(body)); } }); }\n"
        "  window.fetch = function (u) { var k = key(u); return Object.prototype.hasOwnProperty.call(SNAP.responses, k) ? reply(200, SNAP.responses[k]) : reply(404, { detail: '정지 사본에 없는 자료: ' + k }); };\n"
        "})();\n"
        "</script>\n"
    )


PRINT_SCRIPT = """<script>
// 정지 사본: 연결 표시를 '정지 사본' 으로 바꾸고, '#print' 로 열면 ① 수위 절을 관측소마다 펼친다(PDF 용)
(function () {
  function $(s) { return document.querySelector(s); }
  function $$(s) { return Array.prototype.slice.call(document.querySelectorAll(s)); }
  var meta = window.__SNAPSHOT__ && window.__SNAPSHOT__.meta || {};
  (function badge() { var h = $('#healthText'); if (!h) return; if (h.textContent === '연결 확인 중…') return setTimeout(badge, 50); h.textContent = '정지 사본 · ' + (meta.generated_at_ko || ''); })();
  if (!(window.__SNAPSHOT__ && window.__SNAPSHOT__.printMode)) return;  // '#print' 는 shim 이 먼저 봐 둔다(페이지가 URL 을 ?region= 으로 바꾼다)
  document.documentElement.setAttribute('data-print', '1');
  function ready() { return $('#regionTiles .tile') && $$('#stationChips .chip').length && ($$('#truthTable tbody tr').length > 0 || !$('#truthEmpty').hidden) && ($('#wxStats .tile') || !$('#wxEmpty').hidden) && $('#levelTableSub').textContent; }
  var polls = 0;  // 가상 시간(헤드리스 크롬 --virtual-time-budget)에서는 Date.now() 가 건너뛰므로 횟수로 기다린다
  (function wait() { if (ready() || ++polls > 2000) return go(); setTimeout(wait, 50); })();
  function go() {
    var card = $('#levelCard'), chips = $$('#stationChips .chip'), copies = [], i = 0;
    if (!card || !chips.length) return document.documentElement.setAttribute('data-print-ready', '1');
    function next() {
      if (i >= chips.length) {
        copies.forEach(function (c) { card.parentNode.insertBefore(c, card); });
        card.remove();
        document.documentElement.setAttribute('data-print-ready', '1');
        return;
      }
      var chip = chips[i++], name = chip.getAttribute('data-id') || '', tries = 0;
      chip.click();
      (function w() {
        var sub = $('#levelTableSub').textContent;
        if ((sub.indexOf(name) >= 0 && (sub.indexOf('행') >= 0 || sub.indexOf('없음') >= 0)) || ++tries > 400) {
          var c = card.cloneNode(true); c.removeAttribute('id');
          var sub2 = c.querySelector('h2 .sub'); if (sub2) sub2.textContent = '지상 수위관측소 일 수위 — ' + name;
          Array.prototype.slice.call(c.querySelectorAll('[id]')).forEach(function (e) { e.removeAttribute('id'); });  // 사본 안의 id 만 지운다(원본 화면의 id 는 다음 지점 전환에 필요)
          copies.push(c); return next();
        }
        setTimeout(w, 30);
      })();
    }
    next();
  }
})();
</script>
"""


def build_html(page: str, *, public_base: str, region: str, responses: dict[str, Any], generated_at: dt.datetime, source_url: str) -> str:
    """서버가 준 /data 페이지 원문을 정지 사본으로 바꾼다."""
    base = public_base.rstrip("/")
    s = page
    needle = "var API = '/api/v1';"
    if s.count(needle) != 1:
        raise RuntimeError("/data 페이지에서 API 기본 경로를 찾지 못했습니다 — 화면이 바뀌었으면 이 스크립트도 손봐야 합니다")
    s = s.replace(needle, f"var API = '{base}/api/v1';")
    s = s.replace('href="/app"', f'href="{base}/app"').replace('href="/eval"', f'href="{base}/eval"')
    ko = generated_at.strftime("%Y-%m-%d %H:%M")
    s = s.replace("<title>WATERCAST 시험 자료 현황</title>", f"<title>WATERCAST 시험 자료 현황 — 정지 사본 {html.escape(ko)}</title>", 1)
    if "</style>" not in s:
        raise RuntimeError("스타일 블록을 찾지 못했습니다")
    s = s.replace("</style>", PRINT_CSS + "</style>", 1)
    banner = (
        '<div class="snap"><b>정지 사본</b> — ' + html.escape(ko) + ' 에 <b>' + html.escape(source_url) + '</b> 화면을 그 시각의 서버 응답째 떠낸 파일이다. '
        '표·그래프·CSV 내려받기는 그대로 쓸 수 있고 값은 떠낸 시각 그대로다(이 화면의 값은 학습 상태와 무관). '
        "'원본 JSON'·'서비스 화면' 링크는 서버 주소로 적어 두어 분석 PC 에 닿는 환경에서만 열린다. 퀵룩 이미지는 줄인 것이다.</div>"
    )
    err_marker = '<div class="banner" id="err" hidden></div>'
    if s.count(err_marker) != 1:
        raise RuntimeError("오류 배너 자리를 찾지 못했습니다")
    s = s.replace(err_marker, err_marker + "\n  " + banner, 1)
    meta = {"region": region, "generated_at": generated_at.isoformat(timespec="seconds"), "generated_at_ko": ko, "source_url": source_url, **{k: v for k, v in responses.get("__meta__", {}).items()}}
    snapshot = {"meta": meta, "responses": {k: v for k, v in responses.items() if k != "__meta__"}}
    main_open = "<script>\n(function () {"
    if s.count(main_open) != 1:
        raise RuntimeError("페이지 스크립트 시작을 찾지 못했습니다")
    s = s.replace(main_open, shim_script(snapshot) + main_open, 1)
    if s.count("</body>") != 1:
        raise RuntimeError("body 끝을 찾지 못했습니다")
    s = s.replace("</body>", PRINT_SCRIPT + "</body>", 1)
    return s


# ----------------------------------------------------------------------------- PDF
def find_chrome(explicit: str | None = None) -> str | None:
    import glob

    if explicit:
        return explicit if Path(explicit).exists() else None
    for pat in CHROME_CANDIDATES:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        for h in reversed(hits):
            if os.access(h, os.X_OK):
                return h
    return shutil.which("chromium") or shutil.which("google-chrome")


def render_pdf(html_path: Path, pdf_path: Path, chrome: str, *, budget_ms: int = 40000) -> None:
    url = html_path.resolve().as_uri() + "#print"
    cmd = [chrome, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--no-pdf-header-footer",
           f"--virtual-time-budget={budget_ms}", "--run-all-compositor-stages-before-draw", f"--print-to-pdf={pdf_path}", url]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0 or not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise RuntimeError("PDF 변환 실패: " + (res.stderr.strip()[-800:] or f"exit {res.returncode}"))


# ----------------------------------------------------------------------------- 실행
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default=DEFAULT_API, help="응답을 받아 올 서버 (기본: 분석 PC 로컬)")
    p.add_argument("--public", default=None, help="사본 안의 링크에 적을 서버 주소 (기본: --api 와 같음. 예: http://172.18.10.113:18000)")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--out-dir", default="~/tta", help="결과를 둘 폴더 (기본 ~/tta — 저장소 밖)")
    p.add_argument("--name", default=None, help="파일 이름(확장자 제외). 기본 WATERCAST_시험자료현황_<지역>_<날짜>")
    p.add_argument("--pdf", action="store_true", help="HTML 과 함께 PDF 도 만든다 (헤드리스 크롬 필요)")
    p.add_argument("--chrome", default=None, help="크롬/크로미움 실행 파일 경로 (기본: Playwright 캐시·시스템에서 찾음)")
    p.add_argument("--no-images", action="store_true", help="퀵룩 이미지를 넣지 않는다 (파일을 작게)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api = args.api.rstrip("/")
    public = (args.public or api).rstrip("/")
    now = dt.datetime.now().astimezone()
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"WATERCAST_시험자료현황_{args.region}_{now:%Y-%m-%d}"
    html_path = out_dir / f"{name}.html"

    print(f" 서버 {api} 에서 /data 와 지역 '{args.region}' 응답을 받는 중 …")
    page = http_get(api + "/data").decode("utf-8")
    responses = collect(api, args.region, images=not args.no_images)
    meta = responses["__meta__"]
    print(f"   관측소 {meta['n_stations']}곳 · 위성 장면 {meta['n_scenes']}장 · 퀵룩 {meta['n_images']}장")
    doc = build_html(page, public_base=public, region=args.region, responses=responses, generated_at=now,
                     source_url=f"{public}/data?region={urllib.parse.quote(args.region, safe='')}")
    html_path.write_text(doc, encoding="utf-8")
    print(f" HTML 저장 : {html_path}  ({html_path.stat().st_size / 1e6:.1f} MB)")

    if args.pdf:
        chrome = find_chrome(args.chrome)
        if not chrome:
            print(" [오류] 헤드리스 크롬을 찾지 못해 PDF 는 만들지 못했습니다 (--chrome 로 경로 지정)", file=sys.stderr)
            return 2
        pdf_path = out_dir / f"{name}.pdf"
        print(f" PDF 변환 중 ({Path(chrome).name}) …")
        render_pdf(html_path, pdf_path, chrome)
        print(f" PDF 저장  : {pdf_path}  ({pdf_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 — 사용자에게 한 줄로
        print(f" [오류] {exc}", file=sys.stderr)
        sys.exit(2)
