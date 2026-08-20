"""
초소형 위성영상 기반 수체 시계열 분석 파이프라인 (발표용 데모 UI)

흐름:
  1) SAR 영상 입력  →  2) 수체 탐지(1차 처리)  →
  3) 기상청 데이터 자동 연동  →  4) ConvLSTM 시계열 분석(2차 처리)
"""

import io
import json
import time
import zlib
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS_TS = ROOT / "2_time_series_prediction" / "docs" / "examples"
WEATHER_CSV = ROOT / "2_time_series_prediction" / "data" / "weather_ex.csv"
ASSETS_UI = Path(__file__).resolve().parent / "assets"
ICEYE_PNG = ASSETS_UI / "iceye_20200302_busan.png"
ICEYE_META_JSON = ASSETS_UI / "iceye_meta.json"
# 실모델(WBMS U-Net) 산출 자산 — scripts/make_pipeline_ui_wbms_assets.py 생성물
WBMS_PRE_PNG = ASSETS_UI / "wbms_20200302_pre_crop.png"
WBMS_OVERLAY_PNG = ASSETS_UI / "wbms_20200302_overlay_crop.png"
WBMS_FULL_PNG = ASSETS_UI / "wbms_20200302_overlay_full.png"
WBMS_META_JSON = ASSETS_UI / "wbms_20200302_meta.json"

# ====================== Page Config ======================
st.set_page_config(
    page_title="초소형 위성 수체 시계열 분석 파이프라인",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== Custom CSS ======================
st.markdown(
    """
<style>
:root {
    --primary: #0E7C7B;
    --secondary: #17BEBB;
    --accent: #FFC107;
    --bg-dark: #0B1A2E;
    --bg-card: #14253D;
}

/* Main container */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Hide hamburger / footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Header banner */
.header-banner {
    background: linear-gradient(90deg, #0B1A2E 0%, #0E7C7B 60%, #17BEBB 100%);
    padding: 26px 34px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    margin-bottom: 18px;
}
.header-banner h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.header-banner p {
    margin: 6px 0 0 0;
    opacity: 0.92;
    font-size: 14px;
}
.header-banner .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    margin-right: 8px;
    font-weight: 600;
}

/* Step pipeline indicator */
.pipe-wrap {
    display: flex;
    align-items: center;
    background: rgba(14, 124, 123, 0.06);
    border: 1px solid rgba(23, 190, 187, 0.18);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 18px;
}
.pipe-step {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #9aa6b2;
}
.pipe-step.active { color: #17BEBB; font-weight: 700; }
.pipe-step.done { color: #FFC107; font-weight: 700; }
.pipe-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px; height: 30px;
    border-radius: 50%;
    background: rgba(154,166,178,0.18);
    font-weight: 700;
    font-size: 13px;
}
.pipe-step.active .pipe-num { background: #17BEBB; color: white; }
.pipe-step.done .pipe-num { background: #FFC107; color: #0B1A2E; }
.pipe-arrow {
    color: rgba(154,166,178,0.4);
    font-size: 18px;
    margin: 0 8px;
}

/* Streamlit 기본 툴바(Deploy 버튼)·Plotly 모드바 숨김 — 시연 화면 정리 */
[data-testid="stToolbar"] { visibility: hidden; }
.modebar { display: none !important; }

/* Step card */
.step-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 18px rgba(11,26,46,0.06);
}
.step-card h3 {
    color: #0E7C7B;
    margin: 0 0 14px 0;
    font-size: 18px;
    font-weight: 800;
    display: flex; align-items: center; gap: 10px;
}
.lead {
    color: #5a6976;
    font-size: 15px;
    margin-bottom: 14px;
}

/* Metric */
.metric-row { display: flex; gap: 14px; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 150px;
    background: linear-gradient(135deg, #f0fbfa, #e0f5f4);
    border-left: 4px solid #17BEBB;
    padding: 14px 16px;
    border-radius: 10px;
}
.metric-label {
    font-size: 11px;
    color: #5a6976;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
}
.metric-value {
    font-size: 22px;
    color: #0E7C7B;
    font-weight: 800;
    margin-top: 4px;
}
.metric-sub { font-size: 12px; color: #7a8a96; }

/* Big primary button */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0E7C7B, #17BEBB) !important;
    color: white !important;
    border: none !important;
    padding: 14px 26px !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    width: 100%;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 22px rgba(23,190,187,0.35) !important;
    transform: translateY(-1px);
}

/* ============== Sidebar — 강조 스타일 ============== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1A2E 0%, #102540 100%);
    border-right: 2px solid #17BEBB;
    box-shadow: 4px 0 18px rgba(0,0,0,0.15);
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 6px;
}

/* 사이드바 내 모든 텍스트 색상 기본 */
section[data-testid="stSidebar"] * {
    color: #E8F0F7 !important;
}

/* 사이드바 섹션 헤더 ('### 분석 설정' 등) */
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
    background: linear-gradient(90deg, #17BEBB 0%, rgba(23,190,187,0.0) 100%);
    padding: 8px 12px !important;
    border-left: 4px solid #FFC107 !important;
    border-radius: 6px !important;
    margin: 18px 0 10px 0 !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px;
}

/* 라벨 (selectbox, slider, input label) */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #B8D4E8 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}

/* Selectbox 박스 강조 */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.08) !important;
    border: 1.5px solid rgba(23,190,187,0.45) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
    border-color: #17BEBB !important;
    background-color: rgba(23,190,187,0.12) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* TextInput / DateInput */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.08) !important;
    border: 1.5px solid rgba(23,190,187,0.45) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] input::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* Date input 컨테이너 */
section[data-testid="stSidebar"] div[data-baseweb="datepicker"] input,
section[data-testid="stSidebar"] div[data-testid="stDateInput"] input,
section[data-testid="stSidebar"] div[data-testid="stDateInput"] div[data-baseweb="input"],
section[data-testid="stSidebar"] div[data-testid="stDateInput"] div[data-baseweb="base-input"] {
    background-color: #123047 !important;
    color: #FFFFFF !important;
}

/* Slider 트랙 강조 */
section[data-testid="stSidebar"] div[data-baseweb="slider"] {
    padding-top: 8px;
}
section[data-testid="stSidebar"] div[data-baseweb="slider"] > div {
    background: rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"] {
    background-color: #FFC107 !important;
    border: 3px solid #FFFFFF !important;
    box-shadow: 0 0 10px rgba(255,193,7,0.6) !important;
    height: 20px !important;
    width: 20px !important;
}
/* Slider 활성 트랙 */
section[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, #17BEBB, #FFC107) !important;
    height: 6px !important;
}
/* Slider 값 표시 */
section[data-testid="stSidebar"] div[data-baseweb="slider"] + div,
section[data-testid="stSidebar"] [data-testid="stTickBar"] {
    color: #FFC107 !important;
    font-weight: 800 !important;
}

/* Checkbox 강조 */
section[data-testid="stSidebar"] [data-baseweb="checkbox"] {
    background: rgba(255,255,255,0.05);
    padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid rgba(23,190,187,0.25);
    margin: 4px 0;
}
section[data-testid="stSidebar"] [data-baseweb="checkbox"]:hover {
    background: rgba(23,190,187,0.12);
}

/* 사이드바 내부 버튼 */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,193,7,0.15) !important;
    color: #FFC107 !important;
    border: 1.5px solid #FFC107 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #FFC107 !important;
    color: #0B1A2E !important;
}

/* 사이드바 구분선 */
section[data-testid="stSidebar"] hr {
    border-color: rgba(23,190,187,0.25) !important;
    margin: 16px 0 !important;
}

/* 사이드바 caption */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: rgba(184,212,232,0.6) !important;
    font-size: 11px !important;
    text-align: center;
    padding-top: 12px;
    border-top: 1px dashed rgba(23,190,187,0.2);
    margin-top: 12px;
}

/* Help/tooltip 아이콘 */
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
    fill: #FFC107 !important;
}

/* Info tag */
.tag {
    display: inline-block;
    background: rgba(23,190,187,0.12);
    color: #0E7C7B;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 6px;
}
.tag.warn { background: rgba(255,193,7,0.18); color: #B07700; }
.tag.danger { background: rgba(220,53,69,0.15); color: #b02a37; }
</style>
""",
    unsafe_allow_html=True,
)

# ====================== Session State ======================
def init_state():
    defaults = dict(
        step=1,
        sar_image=None,
        sar_meta=None,
        detection_mask=None,
        detection_meta=None,
        weather_df=None,
        prediction_frames=None,
        prediction_meta=None,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

# ====================== Helpers: synthetic SAR + processing ======================
def _stamp_disks(mask: np.ndarray, xs, ys, widths, rng, size: int):
    """경로점마다 원반을 찍어 수로 형태 생성 (점 주변 윈도우만 계산)."""
    for xi, yi, wi in zip(xs, ys, widths):
        wi = wi * (1 + rng.normal(0, 0.08))
        r = int(np.ceil(wi)) + 1
        x0, x1 = max(int(xi) - r, 0), min(int(xi) + r + 1, size)
        y0, y1 = max(int(yi) - r, 0), min(int(yi) + r + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue
        wy, wx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= ((wx - xi) ** 2 + (wy - yi) ** 2) < wi * wi


def gen_water_mask(rng, size: int, n_water: int, water_scale: float) -> np.ndarray:
    """사행(meander) 주 수로 + 지류 + 하구 확장의 강/하구형 수체 마스크."""
    gt = np.zeros((size, size), dtype=bool)
    # 주 수로: 한쪽 끝에서 반대쪽으로 흐르는 사행 곡선, 하류로 갈수록 폭 증가
    n_pts = 160
    t = np.linspace(0, 1, n_pts)
    x_main = t * (size + 80) - 40
    y0 = rng.uniform(0.35, 0.65) * size
    drift = rng.uniform(-0.25, 0.25) * size
    a1, a2 = rng.uniform(30, 75), rng.uniform(8, 25)
    f1, f2 = rng.uniform(0.8, 1.5), rng.uniform(2.5, 4.0)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    y_main = y0 + drift * t + a1 * np.sin(2 * np.pi * f1 * t + p1) + a2 * np.sin(2 * np.pi * f2 * t + p2)
    w_main = (rng.uniform(7, 11) + t * rng.uniform(8, 20)) * water_scale
    # 하구(estuary): 끝 12% 구간 폭 확장
    est = t > 0.88
    w_main = np.where(est, w_main * (1 + (t - 0.88) * rng.uniform(8, 14)), w_main)
    _stamp_disks(gt, x_main, y_main, w_main, rng, size)
    # 지류: 주 수로 중간 지점에서 분기해 상류로 갈수록 가늘게
    for _ in range(max(0, n_water - 1)):
        j = int(rng.integers(int(n_pts * 0.15), int(n_pts * 0.75)))
        bt = np.linspace(0, 1, 70)
        ang = rng.uniform(0.35, 1.2) * rng.choice([-1, 1])
        blen = rng.uniform(0.3, 0.55) * size
        bx = x_main[j] - bt * blen * np.cos(ang) * rng.choice([-1, 1])
        by = y_main[j] - bt * blen * np.sin(ang)
        by = by + rng.uniform(8, 20) * np.sin(2 * np.pi * rng.uniform(1, 2.5) * bt + rng.uniform(0, 2 * np.pi))
        bw = (rng.uniform(4, 7) + (1 - bt) * rng.uniform(2, 5)) * water_scale
        _stamp_disks(gt, bx, by, bw, rng, size)
    # 방향 다양화: 시드에 따라 전치/반전
    if rng.random() < 0.5:
        gt = gt.T
    if rng.random() < 0.5:
        gt = gt[::-1]
    return np.ascontiguousarray(gt.astype(np.uint8))


def gen_sar_scene(seed: int, size: int = 384, n_water: int = 4, water_scale: float = 1.0):
    """SAR backscatter 영상을 모방한 합성 이미지 + GT 마스크 생성."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    # 지형 텍스처 (저주파)
    base = np.zeros((size, size), dtype=np.float32)
    for _ in range(8):
        cx, cy = rng.integers(0, size, 2)
        sigma = rng.uniform(60, 160)
        amp = rng.uniform(0.05, 0.18)
        base += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    base = (base - base.min()) / (base.max() - base.min() + 1e-6)
    base = 0.30 + 0.45 * base  # land albedo
    # Rayleigh speckle
    speckle = rng.rayleigh(0.10, (size, size))
    img = base + (speckle - speckle.mean()) * 0.6
    # 수체 영역(어둡게) — 사행 하천 + 지류 + 하구
    gt = gen_water_mask(rng, size, n_water, water_scale)
    # 수체는 어둡게 표현
    img = np.where(gt == 1, img * 0.18 + rng.normal(0, 0.03, img.shape), img)
    img = np.clip(img, 0, 1).astype(np.float32)
    return img, gt


def morph_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    out = mask.copy().astype(np.uint8)
    for _ in range(iters):
        nxt = out.copy()
        nxt[1:] |= out[:-1]
        nxt[:-1] |= out[1:]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def morph_erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    out = mask.copy().astype(np.uint8)
    for _ in range(iters):
        nxt = out.copy()
        nxt[1:] &= out[:-1]
        nxt[:-1] &= out[1:]
        nxt[:, 1:] &= out[:, :-1]
        nxt[:, :-1] &= out[:, 1:]
        out = nxt
    return out


def detect_water(sar: np.ndarray, threshold: float = 0.18) -> np.ndarray:
    """문턱값 기반 가짜 수체 탐지 + 형태학적 정제. (데모 시각화용)"""
    m = (sar < threshold).astype(np.uint8)
    m = morph_erode(m, 1)
    m = morph_dilate(m, 3)
    m = morph_erode(m, 2)
    return m


def predict_future(mask: np.ndarray, weather_df: pd.DataFrame, n_future: int = 3,
                   precip_bias: float = 0.0) -> list:
    """기상(강수량) 기반 수체 확장/축소 시뮬레이션.
    경계 픽셀을 면적 대비 성장률만큼만 추가/제거해 형상 둘레와 무관하게 변화율 제어."""
    futures = []
    cur = mask.copy()
    if weather_df is None or weather_df.empty:
        precip = [0.0] * n_future
    else:
        # 마지막 N일 평균 강수량으로 변화 정도 결정
        precip_col = "precipitation (mm)"
        avg = weather_df[precip_col].astype(float).rolling(3, min_periods=1).mean().values
        precip = list(avg[-n_future:]) if len(avg) >= n_future else list(avg) + [avg.mean()] * (n_future - len(avg))
    rng = np.random.default_rng(4242)
    for p in precip:
        # 프레임당 상대 성장률: 강수 3일 평균(+시나리오 보정)에 비례
        g = float(np.clip(0.006 * (p + precip_bias - 1.5), -0.08, 0.09))
        area = int(cur.sum())
        if g > 0.001:
            need = max(1, int(round(g * area)))
            cur = cur.copy()
            while need > 0:
                ring = np.flatnonzero(((morph_dilate(cur, 1) - cur) == 1).ravel())
                if len(ring) == 0:
                    break
                if len(ring) <= need:
                    cur.ravel()[ring] = 1
                    need -= len(ring)
                else:
                    cur.ravel()[rng.choice(ring, size=need, replace=False)] = 1
                    need = 0
        elif g < -0.001:
            edge = np.flatnonzero(((cur - morph_erode(cur, 1)) == 1).ravel())
            n_pick = min(len(edge), max(1, int(round(-g * area))))
            cur = cur.copy()
            cur.ravel()[rng.choice(edge, size=n_pick, replace=False)] = 0
        futures.append(cur.copy())
    return futures


def overlay_mask(sar: np.ndarray, mask: np.ndarray, color=(56, 189, 248)) -> np.ndarray:
    """SAR 이미지(그레이) 위에 수체 마스크 오버레이."""
    rgb = (np.stack([sar] * 3, axis=-1) * 255).astype(np.uint8)
    overlay = rgb.copy()
    overlay[mask == 1] = color
    blended = (rgb.astype(np.float32) * 0.55 + overlay.astype(np.float32) * 0.45).astype(np.uint8)
    return blended


def mask_to_rgb(mask: np.ndarray, color=(56, 189, 248), bg=(15, 27, 46)) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = bg
    rgb[mask == 1] = color
    return rgb


def area_km2(mask: np.ndarray, pixel_area_m2: float) -> float:
    """수체 픽셀 수 × 픽셀당 지상 면적 → km²."""
    px = int(mask.sum())
    return px * pixel_area_m2 / 1e6


# ====================== Helpers: ICEYE 실데이터 ======================
def _mtime_ns(path) -> int:
    # mtime을 캐시 키에 포함해, 파일이 나중에 생성/재생성되면 캐시가 무효화되게 함
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data
def load_iceye_meta(mtime_ns: int) -> dict:
    try:
        return json.loads(ICEYE_META_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@st.cache_data
def iceye_display_pixel_m(mtime_ns: int, meta_mtime_ns: int, size: int = 384):
    """화면에 뿌린 퀵룩 1픽셀이 지상에서 덮는 azimuth·range 거리(m).

    퀵룩은 전체 장면을 stride 데시메이션한 것이고, 화면에는 중앙 정사각형을
    crop해 size×size로 리샘플한 결과가 올라간다. 그래서 원본 SLC의 픽셀 간격이
    아니라 이 변환을 되짚은 값이 실제 축척이다. 여기를 원본 간격으로 잡으면
    면적이 세 자릿수 배로 틀어진다.
    """
    if not ICEYE_PNG.exists():
        return None
    scene = (load_iceye_meta(meta_mtime_ns) or {}).get("scene_size_km") or {}
    azimuth_km, range_km = scene.get("azimuth"), scene.get("range")
    if not azimuth_km or not range_km:
        return None
    try:
        with Image.open(ICEYE_PNG) as img:
            width, height = img.size
    except OSError:
        return None
    if not width or not height:
        return None
    scale = min(width, height) / size
    return (
        float(azimuth_km) * 1000.0 / height * scale,
        float(range_km) * 1000.0 / width * scale,
    )


@st.cache_data
def load_iceye_quicklook(mtime_ns: int, size: int = 384):
    """세로로 긴 slant-range 퀵룩 → 중앙 정사각형 crop → size×size float32 [0,1]."""
    if not ICEYE_PNG.exists():
        return None
    try:
        img = Image.open(ICEYE_PNG).convert("L")
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.BILINEAR)
        return np.asarray(img).astype(np.float32) / 255.0
    except OSError:  # 잘린 PNG·비이미지 파일(UnidentifiedImageError 포함) → 합성 폴백
        return None


# ====================== Helpers: WBMS U-Net 실모델 산출 자산 ======================
@st.cache_data
def load_wbms_png(path_str: str, mtime_ns: int):
    """실모델(WBMS U-Net) 산출 PNG 로드. mtime 캐시 키 → 자산 재생성 시 무효화."""
    if mtime_ns == 0:  # _mtime_ns()가 0이면 파일 부재
        return None
    try:
        with Image.open(path_str) as img:
            return np.asarray(img.convert("RGB"))
    except OSError:  # 잘린 PNG·비이미지 파일 → 섹션 미표시
        return None


@st.cache_data
def load_wbms_meta(mtime_ns: int) -> dict:
    try:
        return json.loads(WBMS_META_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ====================== Helpers: weather ======================
@st.cache_data
def load_weather_template() -> pd.DataFrame:
    if WEATHER_CSV.exists():
        df = pd.read_csv(WEATHER_CSV)
    else:
        df = pd.DataFrame()
    return df


def fetch_weather(region: str, start: dt.date, end: dt.date, api_key: str = "") -> pd.DataFrame:
    """기상청 API 또는 샘플 CSV 기반 기상 데이터.
    실제 KMA API 호출은 API 키가 있을 때만 시도, 실패하면 샘플 패턴으로 폴백."""
    days = (end - start).days + 1
    days = max(days, 3)
    base = load_weather_template()
    # base 가 짧으면 반복
    if base.empty:
        # 완전 합성
        rng = np.random.default_rng(int(start.strftime("%Y%m%d")))
        dates = [start + dt.timedelta(days=i) for i in range(days)]
        df = pd.DataFrame({
            "일자(date)": [d.strftime("%Y. %-m. %-d") for d in dates],
            "humidity (%)": rng.integers(45, 85, days),
            "precipitation (mm)": np.round(rng.gamma(1.5, 2.5, days), 1),
            "tmp_min (°C)": rng.integers(13, 22, days),
            "tmp_max (°C)": rng.integers(23, 32, days),
            "pressure (hPa)": rng.integers(1005, 1024, days),
            "wind_speed (m/s)": np.round(rng.uniform(1, 7, days), 1),
        })
        return df
    # 샘플 base를 잘라 길이 맞추기
    rep = int(np.ceil(days / len(base)))
    df = pd.concat([base] * rep, ignore_index=True).iloc[:days].copy()
    dates = [start + dt.timedelta(days=i) for i in range(days)]
    df["일자(date)"] = [d.strftime("%Y. %-m. %-d") for d in dates]
    # 지역별 약간의 변동
    rng = np.random.default_rng(zlib.crc32(region.encode("utf-8")))
    df["precipitation (mm)"] = (df["precipitation (mm)"].astype(float) +
                                rng.normal(0, 1.2, days)).clip(lower=0).round(1)
    df["humidity (%)"] = (df["humidity (%)"].astype(float) + rng.normal(0, 3, days)).clip(20, 100).round(0)
    df["tmp_max (°C)"] = (df["tmp_max (°C)"].astype(float) + rng.normal(0, 1.5, days)).round(1)
    df["tmp_min (°C)"] = (df["tmp_min (°C)"].astype(float) + rng.normal(0, 1.0, days)).round(1)
    return df


# ====================== Plot helpers ======================
def fig_image(arr: np.ndarray, title: str = "", height: int = 380, colorscale="gray", showscale=False):
    if arr.ndim == 2:
        fig = go.Figure(data=go.Heatmap(z=arr[::-1], colorscale=colorscale, showscale=showscale))
    else:
        # RGB
        img = Image.fromarray(arr)
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        b64 = "data:image/png;base64," + __import__("base64").b64encode(buf.read()).decode()
        fig = go.Figure()
        fig.add_layout_image(dict(source=b64, x=0, y=1, xref="x", yref="y",
                                  sizex=1, sizey=1, xanchor="left", yanchor="top", layer="below"))
        fig.update_xaxes(visible=False, range=[0, 1])
        fig.update_yaxes(visible=False, range=[0, 1], scaleanchor="x")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#0E7C7B"), x=0.02),
        margin=dict(l=10, r=10, t=36, b=10),
        height=height,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


def fig_weather(df: pd.DataFrame):
    dates = df["일자(date)"].tolist()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=dates, y=df["precipitation (mm)"], name="강수량 (mm)",
               marker_color="#17BEBB", opacity=0.85),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=df["tmp_max (°C)"], name="최고기온 (°C)",
                   mode="lines+markers", line=dict(color="#FF6B6B", width=3),
                   marker=dict(size=7)),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=df["tmp_min (°C)"], name="최저기온 (°C)",
                   mode="lines+markers", line=dict(color="#4D96FF", width=2, dash="dot"),
                   marker=dict(size=6)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="강수량 (mm)", secondary_y=False, gridcolor="#eef2f6")
    fig.update_yaxes(title_text="기온 (°C)", secondary_y=True, gridcolor="#eef2f6")
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=20, t=20, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2, x=0.0),
    )
    return fig


def fig_area_timeseries(history_areas, predicted_areas, history_labels, predicted_labels):
    fig = go.Figure()
    # 과거(관측)
    fig.add_trace(go.Scatter(
        x=history_labels, y=history_areas,
        mode="lines+markers", name="관측 수체 면적",
        line=dict(color="#0E7C7B", width=3),
        marker=dict(size=10, symbol="circle"),
    ))
    # 미래(예측) - 점선
    fig.add_trace(go.Scatter(
        x=predicted_labels, y=predicted_areas,
        mode="lines+markers", name="ConvLSTM 예측",
        line=dict(color="#FFC107", width=3, dash="dot"),
        marker=dict(size=11, symbol="diamond"),
    ))
    # 경계선
    if history_labels and predicted_labels:
        boundary_x = history_labels[-1]
        fig.add_vline(x=boundary_x, line=dict(color="#bbb", dash="dash"))
        fig.add_annotation(x=boundary_x, y=max(history_areas + predicted_areas),
                           text="현재", showarrow=False, yanchor="bottom",
                           font=dict(color="#666", size=11))
    fig.update_layout(
        title=dict(text="시계열 수체 면적 변화 (관측 → 예측)", font=dict(size=14, color="#0E7C7B"), x=0.02),
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="시점", gridcolor="#eef2f6", type="category"),
        yaxis=dict(title="수체 면적 (km²)", gridcolor="#eef2f6"),
        legend=dict(orientation="h", y=-0.18, x=0.0),
    )
    return fig


# ====================== Header ======================
st.markdown(
    """
<div class="header-banner">
  <div>
    <span class="badge">IITP 과제</span>
    <span class="badge">2세부 · 이노뎁</span>
    <span class="badge">3차년도 데모</span>
  </div>
  <h1>🛰️ 초소형 위성영상 기반 수체 시계열 분석 파이프라인</h1>
  <p>SAR 영상 입력 → 수체 탐지(U-Net) → 기상 데이터 융합 → ConvLSTM 시계열 예측</p>
</div>
""",
    unsafe_allow_html=True,
)


def render_pipeline_indicator(current_step: int):
    steps = [
        ("01", "SAR 영상 입력"),
        ("02", "수체 탐지 (1차)"),
        ("03", "기상 데이터 연동"),
        ("04", "시계열 예측 (2차)"),
    ]
    html = '<div class="pipe-wrap">'
    for i, (num, label) in enumerate(steps):
        if i + 1 < current_step:
            cls = "done"; icon = "✓"
        elif i + 1 == current_step:
            cls = "active"; icon = num
        else:
            cls = ""; icon = num
        html += f'<div class="pipe-step {cls}"><span class="pipe-num">{icon}</span><span>{label}</span></div>'
        if i < len(steps) - 1:
            html += '<span class="pipe-arrow">▶</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


render_pipeline_indicator(st.session_state.step)

# ====================== Sidebar ======================
with st.sidebar:
    # 사이드바 상단 배너 — 시각적 강조
    st.markdown("""
<div style="
    background: linear-gradient(135deg, #FFC107 0%, #FF6B35 100%);
    padding: 14px 16px;
    border-radius: 12px;
    margin: -8px -8px 16px -8px;
    color: #0B1A2E;
    text-align: center;
    box-shadow: 0 4px 14px rgba(255,193,7,0.35);
">
    <div style="font-size:11px; font-weight:800; letter-spacing:1px; opacity:0.85">⚙️ CONTROL PANEL</div>
    <div style="font-size:18px; font-weight:900; margin-top:2px">분석 설정</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 📍 관측 대상")
    region = st.selectbox("관측 지역", ["부산 (낙동강 하구)", "서울 (한강)", "대구 (금호강)", "광주 (영산강)"])
    obs_date = st.date_input("관측 일자", value=dt.date(2025, 8, 7))
    n_history = st.slider("입력 시퀀스 길이 (N프레임)", 3, 6, 4)
    n_future = st.slider("예측 시퀀스 길이 (M프레임)", 1, 5, 3)
    st.markdown("---")
    st.markdown("### 🌧️ 기상 데이터")
    api_key = st.text_input("기상청 API 키 (선택)", type="password",
                            help="비워두면 샘플 데이터로 동작합니다.")
    use_weather = st.checkbox("기상 데이터 융합 사용", value=True)
    st.markdown("---")
    st.markdown("### 🔬 모델 설정")
    model_name = st.selectbox("탐지 모델", ["U-Net (WBMS_SAR_ICEYE.h5)", "DeepLabv3+", "SegFormer"])
    pred_model = st.selectbox("시계열 모델", ["ConvLSTM (Encoder-Decoder)", "PredRNN", "SimVP"])
    threshold = st.slider("이진화 임계값", 0.05, 0.5, 0.18, step=0.01)
    st.markdown("---")
    if st.button("🔄 파이프라인 초기화", use_container_width=True):
        for k in ["sar_image", "sar_meta", "detection_mask", "detection_meta",
                  "weather_df", "prediction_frames", "prediction_meta"]:
            st.session_state[k] = None
        st.session_state.step = 1
        st.rerun()
    st.caption(f"v0.1 · {dt.date.today().isoformat()}")

# ====================== STEP 1: SAR 영상 입력 ======================
st.markdown("### 📡 STEP 1 · 위성 SAR 영상 입력")
st.markdown('<p class="lead">시립대(1세부)로부터 수신한 ICEYE-X5 SAR 영상을 입력합니다. '
            '본 데모에서는 ICEYE 실데이터·합성 샘플·사용자 업로드 영상을 지원합니다.</p>',
            unsafe_allow_html=True)

REAL_CHOICE = "🛰️ 실데이터: ICEYE-X5 SLC (부산, 2020-03-02)"

col_in1, col_in2 = st.columns([1.2, 1])
with col_in1:
    sample_choice = st.radio(
        "입력 소스 선택",
        [REAL_CHOICE,
         "📂 샘플 영상 (부산 낙동강 하구)", "📂 샘플 영상 (한강 - 가뭄 시나리오)",
         "📂 샘플 영상 (광주 - 홍수 시나리오)", "⬆️ 직접 업로드"],
        horizontal=False,
    )
    uploaded = None
    if sample_choice.startswith("⬆️"):
        uploaded = st.file_uploader("SAR 영상 업로드 (.tif / .png / .jpg)", type=["tif", "tiff", "png", "jpg", "jpeg"])

use_real = sample_choice == REAL_CHOICE
iceye_meta = load_iceye_meta(_mtime_ns(ICEYE_META_JSON))
iceye_sar = load_iceye_quicklook(_mtime_ns(ICEYE_PNG)) if use_real else None
real_loaded = use_real and iceye_sar is not None

# 합성 샘플은 지리참조가 없어 3 m를 명목값으로 쓰지만, 실데이터 퀵룩은
# 실제 장면 크기에서 축척을 계산해야 면적이 맞는다.
SYNTHETIC_PIXEL_M = 3.0
_display_pixel_m = (
    iceye_display_pixel_m(_mtime_ns(ICEYE_PNG), _mtime_ns(ICEYE_META_JSON))
    if real_loaded
    else None
)
if _display_pixel_m is not None:
    pixel_area_m2 = _display_pixel_m[0] * _display_pixel_m[1]
    pixel_scale_note = (
        f"{_display_pixel_m[0]:,.0f}m(azimuth) × {_display_pixel_m[1]:,.0f}m(range) 픽셀 · "
        "퀵룩 축척"
    )
else:
    pixel_area_m2 = SYNTHETIC_PIXEL_M * SYNTHETIC_PIXEL_M
    pixel_scale_note = f"{SYNTHETIC_PIXEL_M:g}m × {SYNTHETIC_PIXEL_M:g}m 픽셀 기준 (합성 샘플)"

with col_in2:
    if real_loaded:
        _proc = iceye_meta.get("processing", {})
        _looks = _proc.get("looks", [2, 2])
        obs_display = f"{iceye_meta.get('acquisition_date', '2020-03-02')} (촬영일 기준)"
        sat_value = f"{iceye_meta.get('satellite', 'ICEYE-X5')} (X-band SAR)"
        sat_sub = (f"SLC → Sigma0 · 멀티룩 {_looks[0]}×{_looks[1]} · "
                   f"{_proc.get('geometry', 'slant-range').split(' (')[0]} · "
                   f"{iceye_meta.get('polarization', 'VV')} · {iceye_meta.get('orbit_direction', 'DESCENDING')}")
    else:
        obs_display = obs_date.strftime('%Y-%m-%d')
        sat_value = "ICEYE-X5 (X-band SAR)"
        sat_sub = "SLC → Sigma0 정사보정 후 · 3m 해상도"
    st.markdown(f"""
<div class='metric-card'>
    <div class='metric-label'>관측 지역</div>
    <div class='metric-value' style='font-size:18px'>{region}</div>
    <div class='metric-sub'>관측일: {obs_display}</div>
</div>
<div class='metric-card' style='margin-top:10px'>
    <div class='metric-label'>위성 / 센서</div>
    <div class='metric-value' style='font-size:18px'>{sat_value}</div>
    <div class='metric-sub'>{sat_sub}</div>
</div>
""", unsafe_allow_html=True)

# Load SAR image
sar = None
if uploaded is not None:
    img = Image.open(uploaded).convert("L").resize((384, 384))
    sar = np.array(img).astype(np.float32) / 255.0
    sar_seed_meta = "사용자 업로드"
elif use_real:
    if real_loaded:
        sar = iceye_sar
        sar_seed_meta = "ICEYE-X5 실데이터 (부산)"
    else:
        st.info("실데이터 퀵룩(assets/iceye_20200302_busan.png)이 아직 생성되지 않았습니다. "
                "scripts/iceye_slc_ingest.py 완료 후 새로고침하세요. 합성 샘플(부산)로 대체 표시합니다.")
        sar, _ = gen_sar_scene(seed=7, size=384, n_water=4, water_scale=1.0)
        sar_seed_meta = "합성 대체 (부산)"
else:
    seed_map = {
        "📂 샘플 영상 (부산 낙동강 하구)": (7, 4, 1.0),
        "📂 샘플 영상 (한강 - 가뭄 시나리오)": (31, 3, 0.7),
        "📂 샘플 영상 (광주 - 홍수 시나리오)": (89, 5, 1.3),
    }
    seed, n_water, ws = seed_map.get(sample_choice, (7, 4, 1.0))
    sar, _ = gen_sar_scene(seed=seed, size=384, n_water=n_water, water_scale=ws)
    sar_seed_meta = sample_choice.replace("📂 샘플 영상 (", "").replace(")", "")

# 표시
col_disp1, col_disp2 = st.columns([1, 1])
with col_disp1:
    st.plotly_chart(fig_image(sar, title="입력 SAR Sigma0 영상", height=380),
                    use_container_width=True, key="sar_input_fig")
with col_disp2:
    if real_loaded:
        coord_val, coord_sub = "Slant-range", f"비정사보정 · {iceye_meta.get('orbit_direction', 'DESCENDING')}"
    else:
        coord_val, coord_sub = "EPSG:32652", "UTM Zone 52N"
    st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'><div class='metric-label'>이미지 크기</div>
       <div class='metric-value'>384 × 384</div><div class='metric-sub'>픽셀</div></div>
  <div class='metric-card'><div class='metric-label'>데이터 타입</div>
       <div class='metric-value'>Float32</div><div class='metric-sub'>정규화 [0, 1]</div></div>
</div>
<div class='metric-row' style='margin-top:10px'>
  <div class='metric-card'><div class='metric-label'>좌표계</div>
       <div class='metric-value' style='font-size:16px'>{coord_val}</div><div class='metric-sub'>{coord_sub}</div></div>
  <div class='metric-card'><div class='metric-label'>입력 소스</div>
       <div class='metric-value' style='font-size:16px'>{sar_seed_meta}</div><div class='metric-sub'>SAR Backscatter</div></div>
</div>
""", unsafe_allow_html=True)
    st.markdown("##### 📋 처리 준비 항목")
    st.markdown("""
- ✅ Sigma0 정사보정 영상 검증
- ✅ Min-Max 정규화 (ASC/DSC 궤도 보정)
- ✅ 결측치 보간 (KNN, n_neighbors=5)
- ⏳ 패치 분할 (512×512, 25% overlap)
""")

# 입력 소스가 바뀌면 이전 입력으로 만든 마스크·예측·기상 상태를 무효화
# (안 하면 STEP2가 새 영상 위에 이전 마스크를 오버레이하는 모순 화면이 됨)
prev_source = (st.session_state.sar_meta or {}).get("source")
if prev_source is not None and prev_source != sar_seed_meta:
    for k in ("detection_mask", "detection_meta", "prediction_frames",
              "prediction_meta", "weather_df"):
        st.session_state[k] = None
    st.session_state.step = 1

st.session_state.sar_image = sar
sar_date = iceye_meta.get("acquisition_date", "2020-03-02") if real_loaded else str(obs_date)
st.session_state.sar_meta = dict(region=region, date=sar_date, source=sar_seed_meta)

# ====================== Action: 1차 처리 ======================
col_btn1, col_btn2, col_btn3 = st.columns([1, 1.2, 1])
with col_btn2:
    run_detect = st.button("▶  1차 처리: 수체 탐지 실행", type="primary", key="btn_detect",
                           use_container_width=True)

if run_detect:
    progress = st.progress(0, text="U-Net 모델 로딩...")
    for pct, msg in [(15, "패치 분할 (512×512)..."),
                     (35, "Min-Max 정규화..."),
                     (55, "U-Net 추론 중..."),
                     (78, "이진화 (threshold=%.2f)..." % threshold),
                     (92, "패치 병합 및 후처리...")]:
        time.sleep(0.18)
        progress.progress(pct, text=msg)
    mask = detect_water(sar, threshold=threshold)
    progress.progress(100, text="완료")
    time.sleep(0.2); progress.empty()
    st.session_state.detection_mask = mask
    st.session_state.detection_meta = dict(model=model_name, threshold=threshold)
    st.session_state.step = max(st.session_state.step, 3)
    st.rerun()

# ====================== STEP 2: 수체 탐지 결과 ======================
if st.session_state.detection_mask is not None:
    mask = st.session_state.detection_mask
    sar = st.session_state.sar_image
    area = area_km2(mask, pixel_area_m2)
    water_ratio = float(mask.sum()) / mask.size * 100

    st.markdown("### 🌊 STEP 2 · 수체 탐지 결과 (1차 처리)")
    st.markdown('<p class="lead">U-Net 모델이 입력 SAR 영상에서 수체 영역(픽셀 값 1) 분류 결과를 산출했습니다. '
                '하단의 메트릭과 마스크 오버레이로 검증할 수 있습니다.</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.plotly_chart(fig_image(sar, title="입력 SAR", height=320),
                        use_container_width=True, key="det_sar")
    with c2:
        st.plotly_chart(fig_image(mask_to_rgb(mask), title="수체 마스크 (예측)", height=320),
                        use_container_width=True, key="det_mask")
    with c3:
        st.plotly_chart(fig_image(overlay_mask(sar, mask), title="오버레이", height=320),
                        use_container_width=True, key="det_overlay")

    st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'><div class='metric-label'>수체 면적</div>
    <div class='metric-value'>{area:,.2f} km²</div>
    <div class='metric-sub'>{pixel_scale_note}</div></div>
  <div class='metric-card'><div class='metric-label'>수체 비율</div>
    <div class='metric-value'>{water_ratio:.2f} %</div>
    <div class='metric-sub'>전체 영역 대비</div></div>
  <div class='metric-card'><div class='metric-label'>탐지 모델</div>
    <div class='metric-value' style='font-size:16px'>{model_name.split(' (')[0]}</div>
    <div class='metric-sub'>{model_name.split(' (')[1].rstrip(')') if ' (' in model_name else '학습 가중치'}</div></div>
  <div class='metric-card'><div class='metric-label'>이진화 임계값</div>
    <div class='metric-value'>{threshold:.2f}</div>
    <div class='metric-sub'>Sigma0 dB scale</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-top:14px"><span class="tag">✓ 시립대 추론 결과 호환</span>'
        '<span class="tag">GeoTIFF 0/1 마스크</span>'
        '<span class="tag">→ 시계열 분석 입력</span></div>',
        unsafe_allow_html=True
    )

    # ---- 실모델(WBMS U-Net) 산출 — 정사(EPSG:32652) 기준 별도 패널 ----
    # STEP1 실데이터 퀵룩은 slant-range(비정사) 기하라 정사 격자의 실모델 마스크와
    # 픽셀 단위 오버레이가 불가 → 정사 기준 입력(Pre γ0)과 마스크 쌍을 병기한다.
    # 자산(scripts/make_pipeline_ui_wbms_assets.py 산출)이 없으면 아무것도 추가하지 않음.
    wbms_pre_img = load_wbms_png(str(WBMS_PRE_PNG), _mtime_ns(WBMS_PRE_PNG)) if use_real else None
    wbms_ovl_img = (load_wbms_png(str(WBMS_OVERLAY_PNG), _mtime_ns(WBMS_OVERLAY_PNG))
                    if use_real else None)
    if wbms_pre_img is not None and wbms_ovl_img is not None:
        st.markdown("#### 🧠 실모델(WBMS U-Net) 산출 — 정사 기준")
        st.markdown('<p class="lead">위 패널은 시연용 임계값 데모이고, 아래는 배포 U-Net'
                    '(WBMS_SAR_ICEYE.h5)이 정사보정 입력(EPSG:32652 · 3 m 라벨 격자)에서 '
                    '산출한 실제 수체 마스크입니다. STEP 1 퀵룩은 slant-range(비정사) 기하라 '
                    '정사 격자 마스크와 픽셀 단위로 겹칠 수 없어 별도 패널로 표시합니다.</p>',
                    unsafe_allow_html=True)
        wbms_meta = load_wbms_meta(_mtime_ns(WBMS_META_JSON))
        _crop = (wbms_meta or {}).get("crop", {})
        _roi_km = _crop.get("size_km")
        _roi_txt = (f" · 관심영역 {_roi_km:g}×{_roi_km:g} km (낙동강 하구)"
                    if isinstance(_roi_km, (int, float)) else "")
        wc1, wc2 = st.columns(2)
        with wc1:
            st.plotly_chart(
                fig_image(wbms_pre_img, title="정사 기준 입력 (Pre γ⁰ dB · 관심영역)", height=380),
                use_container_width=True, key="wbms_pre_fig")
        with wc2:
            st.plotly_chart(
                fig_image(wbms_ovl_img, title="실모델 수체 마스크 오버레이", height=380),
                use_container_width=True, key="wbms_overlay_fig")
        wbms_full_img = load_wbms_png(str(WBMS_FULL_PNG), _mtime_ns(WBMS_FULL_PNG))
        if wbms_full_img is not None:
            with st.expander("🗺️ 전체 씬 오버레이 보기 (축소)"):
                st.image(wbms_full_img, use_container_width=True,
                         caption="전체 씬 축소 오버레이 · EPSG:32652 · 3 m 라벨 격자")
        st.caption("모델: WBMS U-Net (WBMS_SAR_ICEYE.h5) · "
                   "IoU 0.8786 (배포 프로토콜, 20200416 시험씬 기준) · CPU 산출 · "
                   f"2020-03-02 부산 씬 실모델 산출{_roi_txt}")

# ====================== STEP 3: 기상청 데이터 연동 ======================
if st.session_state.detection_mask is not None and use_weather:
    st.markdown("### 🌧️ STEP 3 · 기상청 데이터 자동 연동")
    st.markdown('<p class="lead">관측 일자 기준 최근 N일 간의 강수량·기온·습도 등 '
                '기상 데이터를 자동 수집합니다. 시계열 예측 모델의 보조 입력으로 활용됩니다.</p>',
                unsafe_allow_html=True)

    if st.session_state.weather_df is None:
        with st.spinner("기상청 일자료 조회서비스 호출 중..."):
            time.sleep(0.6)
            start_date = obs_date - dt.timedelta(days=n_history + 2)
            wdf = fetch_weather(region, start_date, obs_date, api_key=api_key)
            st.session_state.weather_df = wdf

    wdf = st.session_state.weather_df
    if api_key:
        st.markdown('<span class="tag">✓ API 연결</span>'
                    '<span class="tag">기상청 일자료 조회서비스</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="tag warn">샘플 데이터 (API 키 없음)</span>'
                    '<span class="tag">data/weather_ex.csv 기반</span>', unsafe_allow_html=True)

    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        st.plotly_chart(fig_weather(wdf), use_container_width=True, key="weather_fig")
    with col_w2:
        total_rain = float(wdf["precipitation (mm)"].astype(float).sum())
        avg_hum = float(wdf["humidity (%)"].astype(float).mean())
        max_t = float(wdf["tmp_max (°C)"].astype(float).max())
        min_t = float(wdf["tmp_min (°C)"].astype(float).min())
        st.markdown(f"""
<div class='metric-card'><div class='metric-label'>누적 강수량</div>
  <div class='metric-value'>{total_rain:.1f} mm</div>
  <div class='metric-sub'>최근 {len(wdf)}일</div></div>
<div class='metric-card' style='margin-top:10px'><div class='metric-label'>평균 습도</div>
  <div class='metric-value'>{avg_hum:.0f} %</div></div>
<div class='metric-card' style='margin-top:10px'><div class='metric-label'>기온 범위</div>
  <div class='metric-value' style='font-size:18px'>{min_t:.0f} ~ {max_t:.0f} °C</div></div>
""", unsafe_allow_html=True)

    with st.expander("📋 수집된 기상 데이터 보기"):
        st.dataframe(wdf, use_container_width=True, hide_index=True)


# ====================== Action: 2차 처리 ======================
if st.session_state.detection_mask is not None:
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.2, 1])
    with col_btn2:
        run_predict = st.button("▶  2차 처리: 추가 시계열 분석 실행", type="primary",
                                key="btn_predict", use_container_width=True)
    if run_predict:
        progress = st.progress(0, text="ConvLSTM 모델 로딩...")
        for pct, msg in [(12, "지리적 정렬 (--align_geo)..."),
                         (28, "과거 N프레임 시퀀스 구성..."),
                         (45, "기상 데이터 임베딩..."),
                         (62, "ConvLSTM 인코더-디코더 추론 중..."),
                         (80, "자기회귀(Autoregressive) 미래 프레임 생성..."),
                         (93, "패치 병합 및 시각화...")]:
            time.sleep(0.22)
            progress.progress(pct, text=msg)
        mask = st.session_state.detection_mask
        wdf = st.session_state.weather_df if use_weather else None
        src = (st.session_state.sar_meta or {}).get("source", "")
        precip_bias = 10.0 if "홍수" in src else -15.0 if "가뭄" in src else 0.0
        future_masks = predict_future(mask, wdf, n_future=n_future, precip_bias=precip_bias)
        progress.progress(100, text="완료")
        time.sleep(0.2); progress.empty()
        st.session_state.prediction_frames = future_masks
        st.session_state.prediction_meta = dict(model=pred_model, n_future=n_future)
        st.session_state.step = 4
        st.rerun()

# ====================== STEP 4: 시계열 예측 결과 ======================
if st.session_state.prediction_frames is not None:
    futures = st.session_state.prediction_frames
    base_mask = st.session_state.detection_mask
    sar = st.session_state.sar_image

    st.markdown("### 📈 STEP 4 · 시계열 예측 결과 (2차 처리)")
    st.markdown('<p class="lead">ConvLSTM 인코더-디코더가 과거 수체 마스크 시퀀스와 기상 데이터를 '
                '결합하여 미래 시점의 수체 분포를 예측한 결과입니다.</p>', unsafe_allow_html=True)

    # 시점별 마스크 시각화
    n_show = len(futures) + 1
    cols = st.columns(n_show)
    try:
        base_date = dt.date.fromisoformat((st.session_state.sar_meta or {}).get("date", str(obs_date)))
    except (ValueError, TypeError):
        base_date = obs_date
    with cols[0]:
        st.plotly_chart(
            fig_image(mask_to_rgb(base_mask, color=(14, 124, 123)),
                      title=f"T₀ · {base_date} (관측)", height=260),
            use_container_width=True, key="ts_t0"
        )
    _rc = iceye_meta.get("orbit_repeat_cycle_days")
    revisit = _rc if isinstance(_rc, int) and 0 < _rc <= 60 else 18  # ICEYE-X5 궤도 반복 주기
    for i, fm in enumerate(futures):
        target_date = base_date + dt.timedelta(days=revisit * (i + 1))
        with cols[i + 1]:
            st.plotly_chart(
                fig_image(mask_to_rgb(fm, color=(255, 193, 7)),
                          title=f"T₊{i + 1} · {target_date.strftime('%Y-%m-%d')} (예측)",
                          height=260),
                use_container_width=True, key=f"ts_t{i+1}"
            )

    # 면적 변화 차트
    base_area = area_km2(base_mask, pixel_area_m2)
    pred_areas = [area_km2(fm, pixel_area_m2) for fm in futures]
    history_areas = [base_area * v for v in [0.94, 0.98, 0.97, 1.0]][-n_history:]
    history_dates = [
        (base_date - dt.timedelta(days=revisit * (len(history_areas) - 1 - i))).strftime("%m-%d")
        for i in range(len(history_areas))
    ]
    pred_dates = [(base_date + dt.timedelta(days=revisit * (i + 1))).strftime("%m-%d") for i in range(len(futures))]
    st.plotly_chart(
        fig_area_timeseries(history_areas, pred_areas, history_dates, pred_dates),
        use_container_width=True, key="ts_area_chart"
    )

    # 변화량 분석
    delta_pct = (pred_areas[-1] - base_area) / max(base_area, 1e-3) * 100
    risk_label = "정상"
    risk_class = ""
    if delta_pct > 15:
        risk_label = "홍수 위험"
        risk_class = "danger"
    elif delta_pct < -15:
        risk_label = "가뭄 위험"
        risk_class = "warn"
    elif abs(delta_pct) > 5:
        risk_label = "변화 주의"
        risk_class = "warn"

    st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'><div class='metric-label'>현재 (T₀) 면적</div>
    <div class='metric-value'>{base_area:,.2f} km²</div></div>
  <div class='metric-card'><div class='metric-label'>예측 (T₊{n_future}) 면적</div>
    <div class='metric-value'>{pred_areas[-1]:,.2f} km²</div></div>
  <div class='metric-card'><div class='metric-label'>변화율</div>
    <div class='metric-value' style='color:{"#d63031" if delta_pct>0 else "#0E7C7B"}'>
      {delta_pct:+.1f} %</div>
    <div class='metric-sub'>T₀ → T₊{n_future}</div></div>
  <div class='metric-card'><div class='metric-label'>위험 평가</div>
    <div class='metric-value' style='font-size:18px'>{risk_label}</div>
    <div class='metric-sub'>임계값: ±5% 주의 · ±15% 위험</div></div>
</div>
""", unsafe_allow_html=True)

    # 평가 메트릭 (모형)
    with st.expander("🎯 모델 성능 지표 (검증 셋 기준)"):
        st.markdown(f"""
| 지표 | 값 | 비고 |
|---|---|---|
| **IoU** | 0.7848 ± 0.34 | Intersection over Union |
| **Dice Score** | 0.8160 ± 0.33 | F1과 유사 |
| **Pixel Accuracy** | 0.9948 ± 0.01 | 전체 픽셀 정확도 |
| **Precision** | 0.5612 ± 0.49 | 예측 수체 정확도 |
| **Recall** | 0.4555 ± 0.45 | 실제 수체 검출률 |
| **모델** | {pred_model} | |
| **입력 시퀀스** | {n_history} 프레임 | |
| **예측 시퀀스** | {n_future} 프레임 | |
""")

    # 3세부 연계
    st.markdown(
        '<div style="margin-top:18px">'
        f'<span class="tag {risk_class if risk_class else ""}">'
        f'{"⚠️" if risk_class else "✅"} 3세부 시뮬레이션 연계: '
        f'{risk_label if risk_label != "정상" else "정상 관측"}</span>'
        '<span class="tag">GeoTIFF 출력</span>'
        '<span class="tag">REST API 전송 대기</span></div>',
        unsafe_allow_html=True
    )


# ====================== Footer ======================
st.markdown("---")
st.caption("IITP 초소형 위성영상 기반 주요 지역 분석 및 실감화 지능 기술 개발 · 2세부 (이노뎁) · "
           "본 화면은 발표 시연용 데모이며, 실제 추론 결과는 학습된 모델 가중치를 사용합니다.")
