"""쉬운 화면(/easy) 테스트.

페이지가 기대하는 장면 API 계약(/api/v1/scenes*)을 **인메모리 스텁 라우터**로 구현해 두고
그 위에 ``mount_easy_ui`` 를 붙여 검사한다. 스텁은 GET /api/v1/scenes/{id} 가 호출될 때마다
가짜 파이프라인을 한 칸씩 진행시키므로, 브라우저가 2초 폴링만 해도 등록→탐지→면적→수위→예측→규약 파일
순서로 끝까지 흘러간다. ``scripts/capture_easy_ui.py`` 도 같은 스텁을 uvicorn 으로 띄워 스크린샷을 찍는다:

    .venv/bin/python -m uvicorn backend.tests.test_easy_ui:build_stub_app --factory --port 8765

환경변수
    EASY_STUB_TICK   GET 한 번에 진행시킬 칸 수(기본 1)
    EASY_STUB_SEED   "1" 이면 최근 결과 표가 비지 않게 완료/실패 장면 2개를 미리 넣는다

통합 측이 이 스텁에서 읽어 갈 가정:
    * 단계 6개 키/라벨: register/detect/area/levels/predict/deliver
    * 수위계가 없는 지역(has_gauge=false)은 levels 가 skipped, predict=false 면 predict 가 skipped
    * 결과 outputs 의 *_url 은 같은 출처의 상대 경로이고 이미지(overlay/quicklook)는 PNG
    * 진행 중인 장면이 하나라도 있으면 POST 는 409
"""

from __future__ import annotations

import io
import math
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.testclient import TestClient

from backend.app.easy_ui import INDEX_FILE, mount_easy_ui

# --------------------------------------------------------------------------- 스텁 데이터
STEP_LABELS: list[tuple[str, str]] = [
    ("register", "장면 등록"),
    ("detect", "수체 탐지"),
    ("area", "면적 계산"),
    ("levels", "수위 산출"),
    ("predict", "시계열 예측"),
    ("deliver", "규약 파일·로그"),
]
STEP_ELAPSED = {"register": 1.2, "detect": 8.4, "area": 0.6, "levels": 2.1, "predict": 3.3, "deliver": 0.9}
STEP_DONE_DETAIL = {
    "register": "영상 1장을 등록했어요",
    "detect": "물 영역을 찾았어요",
    "area": "면적을 계산했어요",
    "levels": "수위계 자료와 맞춰 봤어요",
    "predict": "앞으로 7일을 예측했어요",
    "deliver": "결과 파일과 기록을 정리했어요",
}
REGIONS: list[dict[str, Any]] = [
    {"id": "busan_nakdong", "name": "부산 낙동강 하구", "testbed": "부산", "sensors": ["ICEYE", "PlanetScope"], "has_gauge": True},
    {"id": "sejong_geum", "name": "세종 금강", "testbed": "세종", "sensors": ["PlanetScope"], "has_gauge": False},
]
SENSORS: list[dict[str, Any]] = [
    {"id": "ICEYE", "label": "ICEYE 레이더 영상", "type": "SAR", "needs_mask": True},
    {"id": "PlanetScope", "label": "PlanetScope 광학 영상", "type": "OPTIC", "needs_mask": False},
]
TERMINAL = {"done", "failed"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class FakeScene:
    scene_id: str
    region_id: str
    sensor: str
    image_date: str
    detect: bool = True
    predict: bool = True
    fail_at: str | None = None
    note: str | None = None
    pipeline_id: str = field(default_factory=lambda: "pipe-" + uuid.uuid4().hex[:8])
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.steps = [
            {"key": key, "label": label, "status": "pending", "started_at": None, "finished_at": None, "elapsed_s": None, "detail": None}
            for key, label in STEP_LABELS
        ]
        self.outputs = {k: None for k in (
            "quicklook_url", "overlay_url", "mask_url", "product_url", "product_name", "water_area_km2",
            "area_series_url", "levels_url", "prediction_id", "prediction_url", "folder",
        )}

    # ---- 가짜 파이프라인
    def _skip_reason(self, key: str) -> str | None:
        region = next((r for r in REGIONS if r["id"] == self.region_id), None)
        if key == "detect" and not self.detect:
            return "탐지 실행을 끄고 올렸어요"
        if key == "levels" and region is not None and not region["has_gauge"]:
            return "이 지역에는 수위계가 없어요"
        if key == "predict" and not self.predict:
            return "예측 실행을 끄고 올렸어요"
        return None

    def advance(self) -> None:
        """한 칸 진행: 진행 중인 단계를 끝내고 다음 단계를 시작한다. 건너뛰는 단계는 비용 없이 지나간다."""
        if self.status in TERMINAL:
            return
        now = _now()
        started_new = False
        for step in self.steps:
            if step["status"] in {"done", "skipped"}:
                continue
            if step["status"] == "running":
                if self.fail_at == step["key"]:
                    step.update(status="failed", finished_at=now, elapsed_s=STEP_ELAPSED[step["key"]],
                                detail="모델이 응답하지 않았어요 (스텁 오류)")
                    self.status = "failed"
                    self.error = f"'{step['label']}' 단계에서 멈췄어요: 모델이 응답하지 않았어요 (스텁 오류). 파일이 올바른 위성 영상인지 확인해 주세요."
                    self.finished_at = now
                    break
                step.update(status="done", finished_at=now, elapsed_s=STEP_ELAPSED[step["key"]],
                            detail=STEP_DONE_DETAIL[step["key"]])
                continue
            reason = self._skip_reason(step["key"])
            if reason:
                step.update(status="skipped", started_at=now, finished_at=now, elapsed_s=0.0, detail=reason)
                continue
            step.update(status="running", started_at=now, detail=None)
            self.status = "running"
            started_new = True
            break
        if self.status != "failed" and not started_new and all(s["status"] in {"done", "skipped"} for s in self.steps):
            self._complete(now)
        self._update_progress()

    def _update_progress(self) -> None:
        total = len(self.steps)
        score = 0.0
        for step in self.steps:
            if step["status"] in {"done", "skipped", "failed"}:
                score += 1
            elif step["status"] == "running":
                score += 0.5
        self.progress = round(min(1.0, score / total), 3)

    def _complete(self, now: str) -> None:
        self.status = "done"
        self.finished_at = now
        region = next((r for r in REGIONS if r["id"] == self.region_id), None)
        base = f"/api/v1/scenes/{self.scene_id}"
        area = 30.15 if self.sensor == "ICEYE" else 28.72
        if not self.detect:
            area = None
        product_name = f"{self.region_id}_{self.sensor}_{self.image_date.replace('-', '')}_water.zip"
        prediction_id = "pred-" + uuid.uuid4().hex[:8]
        self.outputs.update({
            "quicklook_url": f"{base}/quicklook.png",
            "overlay_url": f"{base}/overlay.png" if self.detect else None,
            "mask_url": f"{base}/mask.tif" if self.detect else None,
            "product_url": f"{base}/product.zip",
            "product_name": product_name,
            "water_area_km2": area,
            "area_series_url": f"{base}/area_series.csv",
            "levels_url": f"{base}/levels.csv" if region and region["has_gauge"] else None,
            "prediction_id": prediction_id if self.predict else None,
            "prediction_url": f"/api/v1/predictions/{prediction_id}" if self.predict else None,
            "folder": f"/data/results/{self.scene_id}",
        })
        self.warnings = ["표본 자료로 만든 결과예요. 실제 영상이 아니에요."]
        if self.sensor == "ICEYE":
            self.warnings.append("레이더 영상은 바람이 강한 날 물 경계가 흐릿하게 나올 수 있어요.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "pipeline_id": self.pipeline_id,
            "region_id": self.region_id,
            "sensor": self.sensor,
            "image_date": self.image_date,
            "status": self.status,
            "progress": self.progress,
            "steps": [dict(s) for s in self.steps],
            "outputs": dict(self.outputs),
            "warnings": list(self.warnings),
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    def to_item(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "region_id": self.region_id,
            "sensor": self.sensor,
            "image_date": self.image_date,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "water_area_km2": self.outputs.get("water_area_km2"),
            "product_name": self.outputs.get("product_name"),
        }


# --------------------------------------------------------------------------- 가짜 그림 (PIL)
def _fake_images() -> tuple[bytes, bytes]:
    """원본 미리보기(흑백)와 물 영역 겹친 그림(파란 물)을 결정적으로 만든다."""
    from PIL import Image, ImageDraw, ImageFilter

    w, h = 800, 600
    base = Image.effect_noise((w, h), 48).filter(ImageFilter.GaussianBlur(2.2))
    base = base.point(lambda v: int(40 + v * 0.65))
    terrain = Image.merge("RGB", (base.point(lambda v: int(v * 0.72)), base.point(lambda v: int(v * 0.78)), base.point(lambda v: int(v * 0.62))))

    def lake(cx: float, cy: float, rx: float, ry: float, wobble: float = 0.18) -> list[tuple[float, float]]:
        pts = []
        for i in range(72):
            a = 2 * math.pi * i / 72
            k = 1 + wobble * math.sin(3 * a + 0.7) + 0.08 * math.cos(7 * a)
            pts.append((cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)))
        return pts

    water = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(water)
    d.polygon(lake(300, 330, 170, 120), fill=255)
    river = [(40, 520), (140, 470), (240, 430), (330, 410), (470, 390), (600, 330), (700, 250), (790, 180)]
    d.line(river, fill=255, width=46, joint="curve")
    d.polygon(lake(640, 160, 70, 50, 0.25), fill=255)
    water = water.filter(ImageFilter.GaussianBlur(1.2))

    quick = terrain.convert("L")
    dark = quick.point(lambda v: int(v * 0.45))
    quick.paste(dark, mask=water)  # 물은 어둡게 보인다

    overlay = terrain.copy()
    tint = Image.new("RGB", (w, h), (28, 120, 230))
    overlay = Image.composite(tint, overlay, water.point(lambda v: int(v * 0.72)))
    edge = water.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 40 else 0)
    overlay.paste(Image.new("RGB", (w, h), (120, 210, 255)), mask=edge)

    def png(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    return png(quick), png(overlay)


# --------------------------------------------------------------------------- 스텁 상태/라우터
class StubState:
    def __init__(self, tick: int = 1) -> None:
        self.tick = max(1, int(tick))
        self.scenes: dict[str, FakeScene] = {}
        self._images: tuple[bytes, bytes] | None = None

    def images(self) -> tuple[bytes, bytes]:
        if self._images is None:
            self._images = _fake_images()
        return self._images

    def running(self) -> FakeScene | None:
        return next((s for s in self.scenes.values() if s.status not in TERMINAL), None)

    def seed(self) -> None:
        """최근 결과 표용 — 끝난 장면 둘(완료 1·실패 1)을 과거 날짜로 넣는다."""
        done = FakeScene("scene-seed-done", "sejong_geum", "PlanetScope", "2026-09-28")
        done.created_at = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(timespec="seconds").replace("+00:00", "Z")
        while done.status not in TERMINAL:
            done.advance()
        failed = FakeScene("scene-seed-failed", "busan_nakdong", "ICEYE", "2026-09-30", fail_at="detect")
        failed.created_at = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds").replace("+00:00", "Z")
        while failed.status not in TERMINAL:
            failed.advance()
        for scene in (done, failed):  # 끝난 시각을 시작 시각 기준으로 맞춰 '걸린 시간'이 며칠로 보이지 않게 한다
            started = datetime.fromisoformat(scene.created_at.replace("Z", "+00:00"))
            scene.finished_at = (started + timedelta(seconds=16)).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.scenes[done.scene_id] = done
        self.scenes[failed.scene_id] = failed


def build_stub_router(state: StubState) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": "stub-0.1"}

    @router.get("/scenes/options")
    async def scene_options() -> dict[str, Any]:
        return {"regions": REGIONS, "sensors": SENSORS}

    @router.get("/scenes")
    async def list_scenes() -> dict[str, Any]:
        items = sorted(state.scenes.values(), key=lambda s: s.created_at, reverse=True)
        return {"items": [s.to_item() for s in items], "total": len(items)}

    @router.post("/scenes", status_code=202)
    async def create_scene(
        region_id: str = Form(...),
        sensor: str = Form(...),
        image_date: str = Form(...),
        pre_file: UploadFile = File(...),
        acquisition_time: str | None = Form(None),
        mask_file: UploadFile | None = File(None),
        order_id: str | None = Form(None),
        scene_id: str | None = Form(None),
        work_id: str | None = Form(None),
        algm_id: str | None = Form(None),
        product_type: str | None = Form(None),
        product_id: str | None = Form(None),
        detect: str = Form("true"),
        predict: str = Form("true"),
        note: str | None = Form(None),
    ) -> dict[str, Any]:
        if not any(r["id"] == region_id for r in REGIONS):
            raise HTTPException(422, detail=f"알 수 없는 지역: {region_id}")
        if not any(s["id"] == sensor for s in SENSORS):
            raise HTTPException(422, detail=f"알 수 없는 영상 종류: {sensor}")
        if not DATE_RE.match(image_date):
            raise HTTPException(422, detail="image_date 는 YYYY-MM-DD 형식이어야 해요")
        if acquisition_time and not TIME_RE.match(acquisition_time):
            raise HTTPException(422, detail="acquisition_time 은 HH:MM:SS 형식이어야 해요")
        if not (pre_file.filename or "").lower().endswith((".tif", ".tiff")):
            raise HTTPException(422, detail="pre_file 은 .tif/.tiff 파일이어야 해요")
        if state.running() is not None:
            raise HTTPException(409, detail="이미 진행 중인 장면이 있어요")
        await pre_file.read()
        if mask_file is not None:
            await mask_file.read()
        new_id = "scene-" + uuid.uuid4().hex[:10]
        scene = FakeScene(
            new_id, region_id, sensor, image_date,
            detect=_truthy(detect), predict=_truthy(predict),
            fail_at="detect" if note and "fail" in note.lower() else None,
            note=note,
        )
        state.scenes[new_id] = scene
        return {"scene_id": new_id, "pipeline_id": scene.pipeline_id, "status_url": f"/api/v1/scenes/{new_id}"}

    @router.get("/scenes/{scene_id}")
    async def get_scene(scene_id: str) -> dict[str, Any]:
        scene = state.scenes.get(scene_id)
        if scene is None:
            raise HTTPException(404, detail="장면을 찾을 수 없어요")
        for _ in range(state.tick):
            scene.advance()
        return scene.to_dict()

    @router.get("/scenes/{scene_id}/{asset}")
    async def get_asset(scene_id: str, asset: str) -> Response:
        scene = state.scenes.get(scene_id)
        if scene is None:
            raise HTTPException(404, detail="장면을 찾을 수 없어요")
        if scene.status != "done":
            raise HTTPException(409, detail="아직 결과가 준비되지 않았어요")
        quick, overlay = state.images()
        if asset == "quicklook.png":
            return Response(quick, media_type="image/png")
        if asset == "overlay.png":
            return Response(overlay, media_type="image/png")
        if asset == "mask.tif":
            return Response(b"II*\x00" + b"\x00" * 60, media_type="image/tiff",
                            headers={"Content-Disposition": f'attachment; filename="{scene_id}_mask.tif"'})
        if asset == "area_series.csv":
            return Response("date,water_area_km2\n2026-09-01,29.80\n2026-09-15,30.02\n", media_type="text/csv")
        if asset == "levels.csv":
            return Response("date,level_m\n2026-09-01,1.21\n2026-09-15,1.34\n", media_type="text/csv")
        if asset == "product.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("README.txt", f"WATERCAST stub product for {scene_id}\n")
                zf.writestr("mask.tif", b"II*\x00")
                zf.writestr("area.csv", "date,water_area_km2\n")
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{scene.outputs["product_name"]}"'})
        raise HTTPException(404, detail="없는 파일이에요")

    @router.get("/predictions/{prediction_id}")
    async def get_prediction(prediction_id: str) -> dict[str, Any]:
        return {"id": prediction_id, "horizon_days": 7, "note": "stub prediction"}

    return router


def build_stub_app(tick: int | None = None, seed: bool | None = None) -> FastAPI:
    """uvicorn --factory 로도 쓰는 스텁 앱. 환경변수 EASY_STUB_TICK / EASY_STUB_SEED 를 읽는다."""
    if tick is None:
        tick = int(os.environ.get("EASY_STUB_TICK", "1") or 1)
    if seed is None:
        seed = _truthy(os.environ.get("EASY_STUB_SEED"))
    state = StubState(tick=tick)
    if seed:
        state.seed()
    app = FastAPI(title="WATERCAST easy-ui stub")
    app.state.stub = state
    app.include_router(build_stub_router(state), prefix="/api/v1")
    mount_easy_ui(app)
    return app


# --------------------------------------------------------------------------- HTML 구문 검사
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _TagBalance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.problems: list[str] = []
        self.tags = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        if tag in VOID_TAGS:
            return
        self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if not self.stack:
            self.problems.append(f"line {self.getpos()[0]}: </{tag}> without opener")
            return
        open_tag, line = self.stack.pop()
        if open_tag != tag:
            self.problems.append(f"line {self.getpos()[0]}: </{tag}> closes <{open_tag}> opened at line {line}")


def _check_html(html: str) -> _TagBalance:
    parser = _TagBalance()
    parser.feed(html)
    parser.close()
    for tag, line in parser.stack:
        parser.problems.append(f"<{tag}> opened at line {line} never closed")
    return parser


# --------------------------------------------------------------------------- 테스트
@pytest.fixture()
def client() -> TestClient:
    return TestClient(build_stub_app(tick=1, seed=False))


def _dummy_tif(name: str = "scene.tif") -> tuple[str, bytes, str]:
    return (name, b"II*\x00" + b"\x00" * 128, "image/tiff")


def _post_scene(client: TestClient, **overrides: Any) -> Any:
    data = {"region_id": "busan_nakdong", "sensor": "ICEYE", "image_date": "2026-10-03", "detect": "true", "predict": "true"}
    data.update(overrides)
    return client.post("/api/v1/scenes", data=data, files={"pre_file": _dummy_tif()})


def _drive(client: TestClient, scene_id: str, limit: int = 40) -> dict[str, Any]:
    scene: dict[str, Any] = {}
    for _ in range(limit):
        scene = client.get(f"/api/v1/scenes/{scene_id}").json()
        if scene["status"] in TERMINAL:
            break
    return scene


def test_easy_page_is_served_with_korean_steps(client: TestClient) -> None:
    for path in ("/easy", "/easy/"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith("text/html")
        assert res.headers.get("cache-control") == "no-cache"
        html = res.text
        for title in ("어디, 무엇, 언제", "파일 올리기", "진행 상황", "결과", "최근 결과", "시작하기", "다시 시도", "전문가 화면 열기"):
            assert title in html, title
        assert '<html lang="ko">' in html
    # 화면에 전문 용어가 보이지 않아야 한다 (주석/코드가 아닌 본문 기준으로 느슨하게)
    body = client.get("/easy").text
    for jargon in ("GeoTIFF", "API", "pipeline", "파이프라인"):
        assert jargon not in re.sub(r"<script.*?</script>", "", body, flags=re.S), jargon


def test_easy_page_is_offline_and_small() -> None:
    html = INDEX_FILE.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html, "외부 URL 참조는 금지 (오프라인 요구)"
    assert "<script src=" not in html and 'rel="stylesheet"' not in html and "@import" not in html
    assert "@font-face" not in html
    assert INDEX_FILE.stat().st_size < 300 * 1024
    assert "Content-Security-Policy" in html


def test_easy_page_html_is_well_formed() -> None:
    parsed = _check_html(INDEX_FILE.read_text(encoding="utf-8"))
    assert parsed.tags > 100
    assert parsed.problems == []


def test_mount_does_not_touch_root(client: TestClient) -> None:
    # 스텁 앱에는 루트 라우트가 없다 → mount_easy_ui 가 루트를 추가하지 않았어야 한다
    assert client.get("/").status_code == 404
    # /easy 는 OpenAPI 문서에 노출하지 않는다
    paths = client.get("/openapi.json").json()["paths"]
    assert "/easy" not in paths and "/easy/" not in paths


def test_mount_on_real_app_keeps_docs_redirect(tmp_path: Path) -> None:
    from backend.app.config import Settings
    from backend.app.main import create_app

    app = create_app(Settings(result_dir=tmp_path / "results"))
    mount_easy_ui(app)
    real = TestClient(app)
    root = real.get("/", follow_redirects=False)
    assert root.status_code in (301, 302, 307, 308)
    assert root.headers["location"] == "/app"  # 서비스 화면이 정문 (2026-10-14)
    assert real.get("/easy").status_code == 200
    assert real.get("/api/v1/health").status_code == 200


def test_stub_pipeline_runs_to_done(client: TestClient) -> None:
    created = _post_scene(client)
    assert created.status_code == 202, created.text
    body = created.json()
    assert set(body) == {"scene_id", "pipeline_id", "status_url"}
    scene_id = body["scene_id"]

    first = client.get(f"/api/v1/scenes/{scene_id}").json()
    assert first["status"] == "running"
    assert [s["key"] for s in first["steps"]] == [k for k, _ in STEP_LABELS]
    assert first["steps"][0]["status"] == "running"
    assert 0 < first["progress"] < 1

    scene = _drive(client, scene_id)
    assert scene["status"] == "done"
    assert scene["progress"] == 1
    assert all(s["status"] == "done" for s in scene["steps"])
    outputs = scene["outputs"]
    assert outputs["water_area_km2"] == 30.15
    assert outputs["product_name"].endswith(".zip")
    for key in ("quicklook_url", "overlay_url", "mask_url", "product_url", "levels_url", "prediction_url"):
        res = client.get(outputs[key])
        assert res.status_code == 200, key
    assert client.get(outputs["overlay_url"]).headers["content-type"] == "image/png"
    assert scene["warnings"]
    assert scene["error"] is None
    assert scene["finished_at"]

    listing = client.get("/api/v1/scenes").json()
    assert listing["total"] == 1
    assert listing["items"][0]["scene_id"] == scene_id
    assert listing["items"][0]["water_area_km2"] == 30.15


def test_stub_skips_levels_without_gauge_and_predict_when_off(client: TestClient) -> None:
    created = _post_scene(client, region_id="sejong_geum", sensor="PlanetScope", predict="false")
    assert created.status_code == 202
    scene = _drive(client, created.json()["scene_id"])
    assert scene["status"] == "done"
    by_key = {s["key"]: s for s in scene["steps"]}
    assert by_key["levels"]["status"] == "skipped" and "수위계" in by_key["levels"]["detail"]
    assert by_key["predict"]["status"] == "skipped"
    assert scene["outputs"]["levels_url"] is None
    assert scene["outputs"]["prediction_url"] is None


def test_stub_rejects_bad_input_and_concurrent_runs(client: TestClient) -> None:
    assert _post_scene(client, image_date="2026/10/03").status_code == 422
    assert _post_scene(client, sensor="Sentinel").status_code == 422
    missing = client.post("/api/v1/scenes", data={"region_id": "busan_nakdong"})
    assert missing.status_code == 422
    assert "detail" in missing.json()

    first = _post_scene(client)
    assert first.status_code == 202
    second = _post_scene(client)
    assert second.status_code == 409
    assert "detail" in second.json()
    assert client.get("/api/v1/scenes/nope").status_code == 404


def test_stub_failure_path(client: TestClient) -> None:
    created = _post_scene(client, note="please fail")
    scene = _drive(client, created.json()["scene_id"])
    assert scene["status"] == "failed"
    assert scene["error"]
    statuses = [s["status"] for s in scene["steps"]]
    assert statuses[:2] == ["done", "failed"]
    assert all(s == "pending" for s in statuses[2:])
    assert client.get(f"/api/v1/scenes/{created.json()['scene_id']}/overlay.png").status_code == 409


def test_stub_seed_fills_recent_results() -> None:
    seeded = TestClient(build_stub_app(tick=1, seed=True))
    items = seeded.get("/api/v1/scenes").json()["items"]
    assert {i["status"] for i in items} == {"done", "failed"}
    assert seeded.get("/api/v1/scenes/options").json()["sensors"][0]["id"] == "ICEYE"
