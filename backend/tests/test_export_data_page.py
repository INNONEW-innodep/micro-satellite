"""scripts/export_data_page.py — /data 화면을 서버 없이 열리는 정지 사본(HTML)으로 떠내는 도구.

* collect : 화면이 부르는 응답 전부를 경로 키(퍼센트 인코딩 푼 형태)로 모으고, 지역 목록은 그 지역 하나로 줄이며,
            퀵룩은 data URI 하나(overlay)만 남긴다.
* build_html : API 기본 경로·화면 링크를 서버 절대 주소로, fetch 를 담긴 응답으로 돌리는 shim, 인쇄용 CSS·스크립트, 안내 배너.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "backend" / "app" / "static" / "data" / "index.html"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("export_data_page", ROOT / "scripts" / "export_data_page.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["export_data_page"] = mod
    spec.loader.exec_module(mod)
    return mod


PNG_1PX = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cfc0000003010100c9fe92ef0000000049454e44ae426082")


def _fake_server(region: str = "nakdong_busan"):
    calls: list[str] = []
    body = {
        "/api/v1/health": {"status": "ok", "version": "0.2.0"},
        "/api/v1/regions": {"items": [{"id": "han_river", "name": "한강"}, {"id": region, "name": "낙동강 하구"}], "n": 2},
        f"/api/v1/regions/{region}": {"id": region, "stations": [{"id": "구포대교"}, {"id": "낙동강하구언(내)"}]},
        f"/api/v1/regions/{region}/satellite/scenes": {"scenes": [{"scene_id": "s1", "date": "2020-03-02", "sensor": "ICEYE",
                                                                   "image_urls": {"overlay": f"/api/v1/regions/{region}/satellite/files/quicklooks/s1_overlay.png", "input": "/x/input.png"},
                                                                   "frame_url": "/x/frame.png"}], "n_scenes": 1},
        f"/api/v1/regions/{region}/satellite/area-series": {"available": True, "sensors": {}},
        f"/api/v1/regions/{region}/weather": {"provider": "KMA_ASOS", "rows": [{"date": "2020-03-02", "precipitation_mm": 0.0}]},
        f"/api/v1/regions/{region}/gauge/%EA%B5%AC%ED%8F%AC%EB%8C%80%EA%B5%90/series": {"station": {"id": "구포대교"}, "rows": [{"date": "2020-03-02", "water_level_m": 2.3863}]},
        f"/api/v1/regions/{region}/gauge/%EB%82%99%EB%8F%99%EA%B0%95%ED%95%98%EA%B5%AC%EC%96%B8%28%EB%82%B4%29/series": {"station": {"id": "낙동강하구언(내)"}, "rows": []},
    }

    def fetch(url: str) -> bytes:
        calls.append(url)
        path = url[len("http://api.test"):]
        if path.endswith("s1_overlay.png"):
            return PNG_1PX
        if path not in body:
            raise RuntimeError("404 " + path)
        return json.dumps(body[path], ensure_ascii=False).encode("utf-8")

    return fetch, calls


def test_collect_gathers_every_response_under_decoded_keys_and_embeds_quicklooks(script) -> None:
    fetch, calls = _fake_server()
    out = script.collect("http://api.test", "nakdong_busan", fetch)
    assert out["/api/v1/regions"]["items"] == [{"id": "nakdong_busan", "name": "낙동강 하구"}]  # 지역 목록은 그 지역 하나
    assert set(out) >= {"/api/v1/health", "/api/v1/regions/nakdong_busan", "/api/v1/regions/nakdong_busan/weather",
                        "/api/v1/regions/nakdong_busan/satellite/area-series", "/api/v1/regions/nakdong_busan/satellite/scenes",
                        "/api/v1/regions/nakdong_busan/gauge/구포대교/series", "/api/v1/regions/nakdong_busan/gauge/낙동강하구언(내)/series"}
    # 괄호가 든 지점 이름도 서버에는 퍼센트 인코딩으로 묻고, 키는 푼 형태 — 화면의 encodeURIComponent 가 괄호를 안 바꿔도 맞도록 shim 이 decode 한다
    assert any("%28%EB%82%B4%29/series" in c for c in calls)
    sc = out["/api/v1/regions/nakdong_busan/satellite/scenes"]["scenes"][0]
    assert set(sc["image_urls"]) == {"overlay"} and sc["image_urls"]["overlay"].startswith("data:image/") and "frame_url" not in sc
    assert out["__meta__"] == {"n_images": 1, "n_stations": 2, "n_scenes": 1}


def test_collect_refuses_unknown_region(script) -> None:
    fetch, _ = _fake_server()
    with pytest.raises(SystemExit, match="없습니다"):
        script.collect("http://api.test", "nowhere", fetch)


def test_build_html_turns_the_real_page_into_an_offline_copy(script) -> None:
    fetch, _ = _fake_server()
    responses = script.collect("http://api.test", "nakdong_busan", fetch)
    when = dt.datetime(2026, 11, 10, 15, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
    doc = script.build_html(PAGE.read_text(encoding="utf-8"), public_base="http://172.18.10.113:18000/", region="nakdong_busan",
                            responses=responses, generated_at=when, source_url="http://172.18.10.113:18000/data?region=nakdong_busan")
    assert "var API = 'http://172.18.10.113:18000/api/v1';" in doc and "var API = '/api/v1';" not in doc
    assert 'href="http://172.18.10.113:18000/app"' in doc and 'href="http://172.18.10.113:18000/eval"' in doc and 'href="/app"' not in doc
    assert "<title>WATERCAST 시험 자료 현황 — 정지 사본 2026-11-10 15:00</title>" in doc
    assert 'class="snap"' in doc and "정지 사본" in doc and "2026-11-10 15:00" in doc
    # shim 이 페이지 스크립트보다 먼저, 인쇄 스크립트는 body 끝 — 응답은 '</' 가 끊긴 JSON 으로 들어간다
    # '#print' 는 페이지가 URL 을 ?region= 으로 바꾸기 전에 shim 이 잡아 둔다(실제 크롬에서 replaceState 가 fragment 를 지우던 결함)
    assert doc.index("window.__SNAPSHOT__ =") < doc.index("SNAP.printMode = location.hash === '#print'") < doc.index("<script>\n(function () {") < doc.index("__SNAPSHOT__.printMode") < doc.index("</body>")
    assert '"__meta__"' not in doc and '"generated_at_ko":"2026-11-10 15:00"' in doc
    assert "@page { size: A4;" in doc and "@media print" in doc
    assert doc.count("Content-Security-Policy") == 1  # CSP 는 그대로(data: 이미지·인라인 스크립트는 원래 허용)


def test_js_literal_cannot_close_the_script_tag(script) -> None:
    lit = script.js_literal({"detail": "</script><!-- x"})
    assert "</script>" not in lit and "<!--" not in lit
    assert json.loads(lit.replace("<\\/", "</").replace("<\\!--", "<!--")) == {"detail": "</script><!-- x"}


def test_shrink_image_returns_a_small_jpeg_or_the_original(script) -> None:
    data, mime = script.shrink_image(PNG_1PX)
    assert mime in ("image/jpeg", "image/png") and len(data) > 0
    assert script.data_uri(b"\x00\x01", "image/png") == "data:image/png;base64,AAE="
