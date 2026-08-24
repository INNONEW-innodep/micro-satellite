"""클릭 한 번씩 따라가는 전체 흐름 안내 화면.

샘플 선택 → 수체 탐지 → 수위 산출 → 시계열 예측을 한 화면에서 순서대로 밟는다.
기존 화면들은 각 단계를 깊이 다루지만 처음 보는 사람은 어디서 시작해 어디로
가는지 알기 어렵다. 이 화면은 그 경로만 보여주고, 자세한 검토가 필요하면 해당
전문 화면으로 보낸다.

각 단계는 **무엇을 하는지 · 무엇이 진짜 측정값인지 · 무엇이 아닌지**를 함께
적는다. 특히 2단계 수체 탐지는 이 앱이 수행하는 것이 아니라 앞단에서 받은
실모델 산출을 보여주는 것이고, 3단계 보정 수위는 당일 보정치이지 예측이
아니라는 점을 화면에서 숨기지 않는다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import plotly.graph_objects as go
import streamlit as st

try:  # package imports used by pytest and other Python callers
    from .eda import axis_range_for
    from .forecast_bands import (
        BAND_DISCLAIMER_KO,
        SIGMA_LEVELS,
        band_surface,
        build_bands,
        forecast_with_bands,
        measure_variability,
        summarize_spread,
    )
    from .samples import SampleDataset, get_sample, list_samples
except ImportError:  # direct Streamlit execution adds ui_next/ to sys.path
    from eda import axis_range_for
    from forecast_bands import (
        BAND_DISCLAIMER_KO,
        SIGMA_LEVELS,
        band_surface,
        build_bands,
        forecast_with_bands,
        measure_variability,
        summarize_spread,
    )
    from samples import SampleDataset, get_sample, list_samples

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = _REPO_ROOT / "data"
WB_DIR: Final[Path] = DATA_DIR / "wbms_runs" / "wb" / "2020"
FUSED_DIR: Final[Path] = DATA_DIR / "wbms_runs" / "fused"
BATCH_EVAL_DIR: Final[Path] = DATA_DIR / "eval"

DEFAULT_SAMPLE_ID: Final[str] = "busan-nas-water-labels"

# 각 기준선이 무엇을 하는지. 결과 모양이 왜 그런지 미리 알려 주면 평평한 예측선을
# 고장으로 오해하지 않는다.
MODEL_NOTES: Final[Mapping[str, str]] = {
    "persistence": (
        "**직전 관측을 그대로 반복합니다.** 연동이 제대로 되는지 확인하는 기준선이라 "
        "예측 구간의 면적이 **일부러 변하지 않습니다**. 변화가 없다고 고장이 아닙니다."
    ),
    "irregular-area-trend": (
        "관측일 간격이 불규칙한 점을 감안해 **면적 추세를 로그선형으로 적합**하고, "
        "강수 반응을 더해 날짜별 면적을 늘리거나 줄입니다. 값이 날짜마다 달라집니다."
    ),
    "weather-morphology": (
        "강수량에 따라 **수체 경계를 팽창·침식**시켜 마스크를 바꿉니다. 기상 입력이 "
        "없으면 변화가 거의 없습니다."
    ),
}

STEP_KEYS: Final[tuple[str, ...]] = ("sample", "detect", "level", "forecast")
STEP_TITLES: Final[Mapping[str, str]] = {
    "sample": "1단계 · 샘플 데이터 넣기",
    "detect": "2단계 · 수체 탐지 결과 보기",
    "level": "3단계 · 수위 산출 보기",
    "forecast": "4단계 · 시계열 예측 실행",
}

WALKTHROUGH_CSS = """
<style>
.wt-step { margin:10px 0 4px; padding:16px 18px; border:1px solid #334155;
  border-left:4px solid #22d3ee; border-radius:10px; background:#0b1324; }
.wt-step--done { border-left-color:#34d399; background:rgba(6,78,59,.18); }
.wt-step--todo { border-left-color:#475569; opacity:.82; }
.wt-step h4 { margin:0 0 6px!important; color:#e0f2fe!important; font-size:1.02rem!important; }
.wt-step p { margin:0; color:#94a3b8; line-height:1.6; }
.wt-why { margin:8px 0 12px; padding:11px 14px; border-left:3px solid #38bdf8;
  background:rgba(8,47,73,.30); color:#bae6fd; line-height:1.6; border-radius:0 6px 6px 0; }
.wt-not { margin:8px 0 12px; padding:11px 14px; border-left:3px solid #f59e0b;
  background:rgba(120,53,15,.16); color:#fde68a; line-height:1.6; border-radius:0 6px 6px 0; }
.wt-rail { display:flex; gap:8px; margin:6px 0 16px; flex-wrap:wrap; }
.wt-pill { display:inline-flex; align-items:center; gap:6px; border-radius:999px;
  padding:5px 13px; font-size:.76rem; font-weight:800; border:1px solid #334155;
  color:#7c8ea3; background:#0b1324; }
.wt-pill--done { border-color:rgba(52,211,153,.5); color:#6ee7b7; background:rgba(6,78,59,.22); }
.wt-pill--now { border-color:rgba(34,211,238,.55); color:#67e8f9; background:rgba(8,145,178,.18); }
.wt-locked { display:flex; align-items:center; gap:10px; margin:6px 0; padding:9px 14px;
  border:1px dashed #334155; border-radius:8px; background:rgba(15,23,42,.5); color:#64748b; }
.wt-locked b { color:#94a3b8; font-weight:700; }
.wt-locked span { margin-left:auto; font-size:.78rem; }
</style>
"""


def _state() -> dict[str, Any]:
    if "walkthrough" not in st.session_state:
        st.session_state.walkthrough = {key: False for key in STEP_KEYS}
    return st.session_state.walkthrough


def reset_walkthrough() -> None:
    st.session_state.walkthrough = {key: False for key in STEP_KEYS}


def _render_rail(done: Mapping[str, bool]) -> None:
    current = next((key for key in STEP_KEYS if not done.get(key)), None)
    chips = []
    for index, key in enumerate(STEP_KEYS, start=1):
        if done.get(key):
            variant, mark = "wt-pill--done", "✓"
        elif key == current:
            variant, mark = "wt-pill--now", "▶"
        else:
            variant, mark = "", str(index)
        label = STEP_TITLES[key].split("·", 1)[1].strip()
        chips.append(f'<span class="wt-pill {variant}">{mark} {label}</span>')
    st.markdown(f'<div class="wt-rail">{"".join(chips)}</div>', unsafe_allow_html=True)


def _step_header(key: str, blurb: str, done: bool, enabled: bool) -> None:
    """단계 머리말. 아직 못 여는 단계는 한 줄로 접는다.

    잠긴 단계까지 큰 카드로 그리면 첫 화면이 '먼저 완료하세요'만 적힌 빈 상자
    셋으로 가득 찬다. 실제로 화면이 비어 보인다는 지적이 있었다.
    """

    if not enabled:
        need = STEP_TITLES[_previous_step(key)].split("·", 1)[1].strip()
        st.markdown(
            f'<div class="wt-locked">🔒 <b>{STEP_TITLES[key]}</b>'
            f'<span>{need} 완료 후 열립니다</span></div>',
            unsafe_allow_html=True,
        )
        return
    variant = "wt-step--done" if done else ""
    st.markdown(
        f'<div class="wt-step {variant}"><h4>{STEP_TITLES[key]}</h4><p>{blurb}</p></div>',
        unsafe_allow_html=True,
    )


def _previous_step(key: str) -> str:
    index = STEP_KEYS.index(key)
    return STEP_KEYS[max(0, index - 1)]


def _why(text: str) -> None:
    st.markdown(f'<div class="wt-why">{text}</div>', unsafe_allow_html=True)


def _not(text: str) -> None:
    st.markdown(f'<div class="wt-not">{text}</div>', unsafe_allow_html=True)


# --- 2단계: 전달받은 실모델 수체 탐지 산출 ---------------------------------


def load_detection_results() -> tuple[Mapping[str, Any], ...]:
    """체인 러너가 만든 날짜별 수체 마스크와 그 정확도를 모은다.

    마스크 자체(수 MB GeoTIFF)는 화면에 띄우지 않고, 같은 날짜의 배치 평가
    JSON에 있는 실측 지표만 읽는다. 정확도가 없는 날짜도 산출물이 있으면
    그대로 보여준다 — 조용히 빼면 몇 장을 돌렸는지 알 수 없게 된다.
    """

    if not WB_DIR.is_dir():
        return ()
    metrics: dict[str, Mapping[str, Any]] = {}
    if BATCH_EVAL_DIR.is_dir():
        for path in sorted(BATCH_EVAL_DIR.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            name = str(payload.get("wb") or "")
            if name.startswith("WB_") and isinstance(payload.get("metrics"), Mapping):
                metrics[name] = payload["metrics"]

    rows: list[Mapping[str, Any]] = []
    for date_dir in sorted(WB_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        for tif in sorted(date_dir.glob("WB_*.tif")):
            metric = metrics.get(tif.name, {})
            rows.append(
                {
                    "관측일": f"{date_dir.name[:4]}-{date_dir.name[4:6]}-{date_dir.name[6:]}",
                    "산출 마스크": tif.name,
                    "IoU": metric.get("water_iou"),
                    "F1": metric.get("f1"),
                    "Precision": metric.get("precision"),
                    "Recall": metric.get("recall"),
                }
            )
    return tuple(rows)


# --- 3단계: 지점별 수위 -------------------------------------------------------


def load_level_results() -> tuple[Mapping[str, Any], ...]:
    """FUSED 산출에서 지점·날짜별 위성 수위와 당일 보정 수위를 읽는다."""

    if not FUSED_DIR.is_dir():
        return ()
    rows: list[Mapping[str, Any]] = []
    for path in sorted(FUSED_DIR.glob("*/FUSED_*.json")):
        if path.name.endswith("_qc.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload["records"]
        except (OSError, ValueError, KeyError):
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            raw = record.get("date")
            date = str(raw) if raw is not None else ""
            if len(date) == 8 and date.isdigit():
                date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            rows.append(
                {
                    "관측일": date,
                    "지점": str(record.get("loc_id") or ""),
                    "위성 수위 (m)": record.get("water_level_m"),
                    "수체 면적 (km²)": record.get("water_area_km2"),
                    "당일 보정 수위 (m)": record.get("corrected_water_level_m"),
                    "정렬량 offset (m)": record.get("offset_m"),
                    "교차센서 쌍": "있음" if record.get("paired") else "없음(단독 센서)",
                }
            )
    rows.sort(key=lambda item: (item["관측일"], item["지점"]))
    return tuple(rows)


def render_walkthrough_phase(
    *,
    activate_sample,
    execute_quick_sample,
    go_to_phase,
    format_api_error,
    api_error_type,
) -> None:
    """네 단계를 순서대로 밟는 안내 화면.

    실행 함수들은 app.py에서 주입받는다. 이 모듈이 app.py를 import하면
    순환 참조가 되기 때문이다. ``go_to_phase``는 단계 **제목**을 받는다 —
    번호를 쓰면 단계가 추가될 때 조용히 어긋난다.
    """

    st.markdown(WALKTHROUGH_CSS, unsafe_allow_html=True)
    done = _state()

    st.markdown("### 처음이신가요? 여기서 순서대로 눌러보세요")
    st.write(
        "위성영상이 어떻게 수체 마스크가 되고, 거기서 면적과 수위가 나오고, 다시 "
        "미래 시점을 예측하는지 네 번의 클릭으로 확인합니다. 각 단계마다 무엇을 "
        "보고 있는지, 그 숫자가 어디서 나온 값인지 함께 설명합니다."
    )
    _render_rail(done)

    if st.button("처음부터 다시", icon=":material/restart_alt:"):
        reset_walkthrough()
        st.rerun()

    st.divider()
    _render_step_sample(done, activate_sample, go_to_phase)
    st.divider()
    _render_step_detect(done)
    st.divider()
    _render_step_level(done, go_to_phase)
    st.divider()
    _render_step_forecast(
        done, execute_quick_sample, go_to_phase, format_api_error, api_error_type
    )


def _render_step_sample(done: dict[str, bool], activate_sample, go_to_phase) -> None:
    _step_header(
        "sample",
        "분석할 관측 자료를 고릅니다. 부산 낙동강 하구를 서로 다른 날짜에 촬영한 "
        "위성 수체 마스크 묶음을 사용합니다.",
        done["sample"],
        True,
    )
    _why(
        "<b>왜 여러 장인가요?</b> 한 장으로는 '지금 물이 얼마나 있는지'만 알 수 있습니다. "
        "미래를 예측하려면 물이 <b>어느 방향으로 얼마나 빠르게</b> 변해왔는지가 필요하고, "
        "그건 여러 날짜를 이어 붙여야 보입니다."
    )

    samples = list_samples()
    if not samples:
        st.error("사용할 수 있는 샘플이 없습니다.")
        return
    labels = {item.sample_id: f"{item.region_name} · {item.scenario_name}" for item in samples}
    # 부분 문자열로 고르면 'busan-doc-recovered'(문서 그림 복원 데모)가 먼저 걸린다.
    # 그 샘플의 권장 모델은 persistence라 예측선이 평평하게 나와, 처음 보는 사람이
    # 곧바로 "예측이 안 되는 앱"으로 오해한다. 실자료 세트를 명시적으로 고른다.
    available = {item.sample_id for item in samples}
    default_id = next(
        (
            candidate
            for candidate in (DEFAULT_SAMPLE_ID, "iceye-nas-water-labels")
            if candidate in available
        ),
        samples[0].sample_id,
    )
    chosen = st.selectbox(
        "샘플 선택",
        [item.sample_id for item in samples],
        index=[item.sample_id for item in samples].index(default_id),
        format_func=lambda value: labels.get(value, value),
        key="wt_sample_id",
    )
    sample = get_sample(chosen)
    st.caption(
        f"관측 {len(sample.frames)}장 · {sample.source_date_strings[0]} ~ "
        f"{sample.source_date_strings[-1]} · 권장 모델 `{sample.recommended_model_id}`"
    )
    note = MODEL_NOTES.get(sample.recommended_model_id)
    if note:
        st.caption(f"이 샘플은 4단계에서 {note}")

    if st.button(
        "① 이 샘플 불러오기",
        type="primary",
        icon=":material/download:",
        key="wt_btn_sample",
    ):
        activate_sample(sample, 30)
        # σ 밴드는 입력 관측이 실제로 얼마나 출렁였는지에서 나온다. 여기서 재두지
        # 않으면 4단계에서 근거 없이 변동 폭을 추정하게 된다.
        st.session_state.walkthrough_observed_areas = _frame_areas(sample)
        done["sample"] = True
        st.success(
            f"불러왔습니다. {len(sample.frames)}장의 수체 마스크와 기상 시나리오가 "
            "준비됐습니다. 아래 2단계로 내려가세요."
        )

    if done["sample"]:
        preview = st.columns(2)
        preview[0].image(
            sample.frames[0].png_bytes,
            caption=f"첫 관측 · {sample.source_date_strings[0]}",
            width="stretch",
        )
        preview[1].image(
            sample.frames[-1].png_bytes,
            caption=f"마지막 관측 · {sample.source_date_strings[-1]}",
            width="stretch",
        )
        st.caption(
            "흰색이 물로 판정된 픽셀입니다. 두 장을 비교하면 그 사이에 물이 얼마나 "
            "늘거나 줄었는지 눈으로 확인할 수 있습니다."
        )
        _render_loaded_weather()
        if st.button("데이터 화면에서 자세히 보기", icon=":material/database:", key="wt_go_data"):
            go_to_phase("데이터")


def _render_step_detect(done: dict[str, bool]) -> None:
    enabled = done["sample"]
    _step_header(
        "detect",
        "위성 원본 영상에서 '여기가 물이다'를 찾아낸 결과입니다. 딥러닝 수체 탐지 "
        "모델(U-Net)이 만든 날짜별 마스크와 그 정확도를 봅니다.",
        done["detect"],
        enabled,
    )
    if not enabled:
        return

    _why(
        "<b>수체 탐지란?</b> SAR 위성은 물 표면에서 신호가 잘 되돌아오지 않아 어둡게 "
        "찍힙니다. 모델은 이 밝기 패턴과 주변 지형을 함께 보고 픽셀마다 물/뭍을 "
        "판정합니다. 결과는 0(뭍)과 1(물)만 있는 마스크입니다."
    )
    _not(
        "<b>이 화면이 지금 탐지를 돌리는 것은 아닙니다.</b> 탐지는 앞단에서 수행하고, "
        "여기서는 전달받은 실모델(WBMS U-Net)이 이미 산출해 둔 결과를 읽어 보여줍니다. "
        "아래 정확도는 배포된 정답 라벨과 대조해 실제로 계산한 값입니다."
    )

    rows = load_detection_results()
    if not rows:
        st.warning(
            "이 서버에는 아직 탐지 산출물이 없습니다. 체인 러너를 실행하면 "
            "`data/wbms_runs/wb/` 아래에 날짜별 마스크가 생기고 이 표가 채워집니다."
        )
        return

    if st.button("② 탐지 결과 보기", type="primary", icon=":material/visibility:", key="wt_btn_detect"):
        done["detect"] = True

    if done["detect"]:
        st.table(rows)
        scored = [row["IoU"] for row in rows if isinstance(row["IoU"], (int, float))]
        if scored:
            st.metric("평균 IoU", f"{sum(scored) / len(scored):.4f}", help="정답 라벨과 겹치는 정도")
            st.caption(
                "IoU는 모델이 물이라고 한 영역과 정답이 물인 영역이 **얼마나 겹치는지**를 "
                "0~1로 나타냅니다. 1이면 완전히 같습니다. 여기 값은 배포된 정답 라벨과 "
                "픽셀 단위로 비교해 산출했습니다."
            )
        st.success("탐지 결과를 확인했습니다. 이제 이 마스크에서 수위를 뽑습니다.")


def _render_step_level(done: dict[str, bool], go_to_phase) -> None:
    enabled = done["detect"]
    _step_header(
        "level",
        "수체 마스크와 지형 고도(DEM)를 겹쳐 지점별 수위를 계산하고, 실측 게이지 "
        "체계에 맞추는 보정까지 적용한 결과입니다.",
        done["level"],
        enabled,
    )
    if not enabled:
        return

    _why(
        "<b>마스크에서 어떻게 수위가 나오나요?</b> 물가장자리 픽셀이 놓인 지면의 고도를 "
        "DEM에서 읽으면, 그 높이까지 물이 차 있다는 뜻이 됩니다. 면적은 물 픽셀 수에 "
        "픽셀 면적을 곱해 구합니다."
    )
    _not(
        "<b>'당일 보정 수위'는 예측이 아닙니다.</b> 관측한 날의 위성 수위를 게이지의 "
        "절대 수위 체계로 옮겨 놓은 값입니다. 옆의 <b>정렬량 offset</b>은 두 체계의 "
        "기준면 차이일 뿐 모델 오차가 아닙니다."
    )

    rows = load_level_results()
    if not rows:
        st.warning(
            "이 서버에는 아직 수위 산출물이 없습니다. 체인 러너의 `wlwa,fused` 단계를 "
            "실행하면 `data/wbms_runs/fused/` 아래에 생깁니다."
        )
        return

    if st.button("③ 수위 결과 보기", type="primary", icon=":material/water:", key="wt_btn_level"):
        done["level"] = True

    if done["level"]:
        st.table(rows)
        solo = sum(1 for row in rows if row["교차센서 쌍"].startswith("없음"))
        if solo:
            st.warning(
                f"{solo}건은 같은 시기 다른 위성 관측이 없어 단독 센서로 보정했습니다. "
                "숨기지 않고 표시합니다."
            )
        _render_level_forecast(rows)
        st.caption(
            "이 수치들의 정확도와 계열 구분(배포 프로토콜·연구·스모크·배치·당일 보정)은 "
            "**정량 평가** 화면에서 자세히 볼 수 있습니다."
        )
        if st.button("정량 평가 화면으로", icon=":material/fact_check:", key="wt_go_eval"):
            go_to_phase("정량 평가")
        st.success("수위까지 확인했습니다. 마지막으로 미래 시점을 예측합니다.")


def _render_step_forecast(
    done: dict[str, bool],
    execute_quick_sample,
    go_to_phase,
    format_api_error,
    api_error_type,
) -> None:
    enabled = done["level"]
    _step_header(
        "forecast",
        "지금까지의 관측 흐름을 백엔드 예측 API에 보내 미래 시점의 마스크와 면적을 "
        "만듭니다. 실제 REST 호출이며 결과는 파일로 내려받을 수 있습니다.",
        done["forecast"],
        enabled,
    )
    if not enabled:
        return

    _why(
        "<b>무엇을 예측하나요?</b> 과거 관측들의 면적 변화 추세와 강수 입력을 함께 보고, "
        "지정한 날짜만큼 뒤의 수체 마스크와 면적을 만듭니다. 결과는 JSON·CSV·마스크 "
        "파일로 모두 내보낼 수 있습니다."
    )
    _not(
        "<b>내장 모델은 학습된 정확도 모델이 아니라 설명 가능한 기준선(baseline)입니다.</b> "
        "연동 경로와 산출 형식을 확인하는 용도이며, 이 결과로 예측 성능을 주장하면 "
        "안 됩니다. 실제 모델은 어댑터를 등록해 화면 변경 없이 교체합니다."
    )

    horizon = st.slider(
        "며칠 뒤까지 예측할까요?", min_value=7, max_value=60, value=30, step=1, key="wt_horizon"
    )
    sample = get_sample(st.session_state.get("wt_sample_id") or DEFAULT_SAMPLE_ID)

    model_id = sample.recommended_model_id
    st.markdown(f"**사용할 모델** · `{model_id}`")
    note = MODEL_NOTES.get(model_id)
    if note:
        st.info(note)
    if model_id == "persistence":
        st.warning(
            "이 샘플의 권장 모델은 직전 관측을 반복하는 기준선이라 **예측 면적이 "
            "모든 날짜에서 같게 나옵니다.** 날짜에 따라 값이 변하는 결과를 보시려면 "
            "1단계에서 `부산 NAS 수체 라벨 2020`처럼 실측 라벨 세트를 고르세요."
        )

    if st.button(
        "④ 예측 실행",
        type="primary",
        icon=":material/play_arrow:",
        key="wt_btn_forecast",
    ):
        try:
            with st.spinner("백엔드에 전송하고 결과 산출물을 만드는 중입니다..."):
                # 이 화면에 남아 완료 안내를 보여주려면 자동 이동을 꺼야 한다.
                execute_quick_sample(sample, horizon, navigate=False)
            done["forecast"] = True
        except api_error_type as exc:
            st.error(format_api_error(exc))

    if done["forecast"]:
        st.success("예측이 끝났습니다.")
        result = st.session_state.get("prediction_result") or {}
        steps = result.get("steps") or []
        if steps:
            first, last = steps[0], steps[-1]
            summary = st.columns(4)
            summary[0].metric("예측 프레임", f"{len(steps)}개")
            summary[1].metric(
                "첫 예측일", str(first.get("target_date") or "-"),
                help="가장 가까운 미래 시점",
            )
            summary[2].metric(
                "마지막 예측 면적",
                f"{last.get('water_area_km2', 0):,.4f} km²"
                if isinstance(last.get("water_area_km2"), (int, float))
                else "-",
            )
            summary[3].metric(
                "면적 변화",
                f"{last.get('area_change_pct', 0):+.2f} %"
                if isinstance(last.get("area_change_pct"), (int, float))
                else "-",
                help="마지막 입력 관측 대비",
            )
            st.caption(
                "'면적 변화'는 마지막 실제 관측 대비 변화율입니다. 위험도 등급은 이 변화율에 "
                "규칙을 적용한 값이며 모델이 판단한 것이 아닙니다."
            )
        _render_area_and_level(result)
        _render_sigma_bands(result)
        st.write(
            "프레임별 마스크·면적·위험도와 JSON·CSV·NPY·PNG·TIFF 내려받기는 결과 화면에 "
            "있습니다. 방금 만든 산출물이 그대로 들어 있습니다."
        )
        if st.button("결과 화면으로", type="primary", icon=":material/monitoring:", key="wt_go_result"):
            go_to_phase("결과")


def _observed_areas(result: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """예측 요청에 쓰인 입력 관측의 면적과 경과일을 되짚는다.

    응답에 입력 프레임의 면적이 그대로 들어오지 않으므로, 세션에 남은 입력
    번들에서 마스크를 다시 세는 대신 결과의 첫 단계 기준선을 쓴다. 기준선이
    없으면 빈 값을 돌려 밴드를 그리지 않는다 — 없는 변동성을 지어내지 않는다.
    """

    import datetime as _dt

    source_dates = ((result.get("input") or {}).get("source_dates")) or []
    areas = st.session_state.get("walkthrough_observed_areas") or []
    if not areas or len(areas) != len(source_dates):
        return [], []
    days: list[float] = []
    try:
        base = _dt.date.fromisoformat(str(source_dates[0]))
        for item in source_dates:
            days.append((_dt.date.fromisoformat(str(item)) - base).days)
    except (TypeError, ValueError):
        days = [float(i) for i in range(len(source_dates))]
    return [float(a) for a in areas], days


def _render_sigma_bands(result: Mapping[str, Any]) -> None:
    """예측 궤적에 관측 변동성 기반 σ 밴드를 씌우고 3D로 보여준다."""

    steps = result.get("steps") or []
    areas, days = _observed_areas(result)
    if len(steps) < 2 or len(areas) < 3:
        return

    profile = measure_variability(areas, days)
    if not profile.usable:
        st.info("입력 관측의 변동 폭을 잴 수 없어 산포 범위를 생략합니다.")
        return

    rows = build_bands(steps, profile)
    if not rows:
        return
    summary = summarize_spread(rows, profile)

    st.markdown("##### 예측 산포 범위 · 관측 변동성 기반")
    st.warning(BAND_DISCLAIMER_KO)

    metrics = st.columns(4)
    metrics[0].metric(
        "관측 변동 σ",
        f"{summary['sigma_per_interval_pct']:.2f} %",
        help=f"관측 간격 1회(평균 {summary['mean_interval_days']:.0f}일)당 면적 상대 변화율의 표준편차",
    )
    metrics[1].metric("하루당 σ", f"{summary['sigma_per_day_pct']:.2f} %")
    metrics[2].metric(
        f"D+{summary['horizon']} 3σ 폭",
        f"{summary['spread_pct']:.1f} %",
        help="±3σ 사이 폭을 예측값 대비로 나타낸 값",
    )
    metrics[3].metric("사용한 관측", f"{summary['observation_count']}시점")

    dates, levels, grid = band_surface(rows)
    figure = go.Figure(
        data=[
            go.Surface(
                x=list(range(len(dates))),
                y=levels,
                z=grid,
                colorscale="Tealrose",
                opacity=0.92,
                colorbar={"title": "수체 픽셀"},
                hovertemplate=(
                    "%{customdata}<br>σ 단계 %{y:+d}<br>수체 %{z:,.0f} px<extra></extra>"
                ),
                customdata=[[date for date in dates] for _ in levels],
            )
        ]
    )
    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 34, "b": 0},
        title="예측 궤적 × σ 단계 × 수체 픽셀",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        scene={
            "xaxis": {
                "title": "예측 시점",
                "tickmode": "array",
                "tickvals": list(range(0, len(dates), max(1, len(dates) // 6))),
                "ticktext": [dates[i] for i in range(0, len(dates), max(1, len(dates) // 6))],
            },
            "yaxis": {"title": "σ 단계", "tickvals": list(levels)},
            # 밴드 전체 범위에 딱 맞춘다. 0부터 그리면 σ 부채꼴이 납작해진다.
            "zaxis": {"title": "수체 픽셀", "range": _grid_range(grid)},
            "camera": {"eye": {"x": 1.7, "y": -1.5, "z": 0.8}},
        },
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(
        "가운데 0σ 능선이 모델 예측이고, 위아래로 갈수록 관측이 보여 준 변동 폭만큼 "
        "벌어집니다. 시점이 멀수록 넓어지는 것은 무작위 보행 가정(√시간)에 따른 것입니다."
    )

    with st.expander("σ 단계별 값 표로 보기", expanded=False):
        table = [
            {
                "시점": row.get("target_date"),
                **{
                    key: f"{row[key]:,.0f}"
                    for key in (f"{level:+d}σ" if level else "0σ" for level in SIGMA_LEVELS)
                },
            }
            for row in rows
        ]
        st.dataframe(table, width="stretch", hide_index=True)


def _frame_areas(sample: SampleDataset) -> list[float]:
    """샘플 프레임의 수체 픽셀 수. 미리보기 PNG를 그대로 센다."""

    import io

    import numpy as np
    from PIL import Image

    areas: list[float] = []
    for frame in sample.frames:
        try:
            with Image.open(io.BytesIO(frame.png_bytes)) as image:
                array = np.asarray(image.convert("L"))
            areas.append(float((array > 127).sum()))
        except (OSError, ValueError):
            return []
    return areas


def _level_series(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    """지점별 (수위 값, 경과일) 시계열. 면적과 무관하게 수위 관측만 쓴다."""

    import datetime as _dt

    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        level = row.get("위성 수위 (m)")
        date = str(row.get("관측일") or "")
        if not isinstance(level, (int, float)) or isinstance(level, bool):
            continue
        if float(level) <= 0.0 or len(date) != 10:
            continue
        grouped.setdefault(str(row.get("지점") or ""), []).append((date, float(level)))

    series: dict[str, tuple[list[float], list[float]]] = {}
    for station, items in grouped.items():
        items.sort()
        try:
            base = _dt.date.fromisoformat(items[0][0])
            days = [float((_dt.date.fromisoformat(d) - base).days) for d, _ in items]
        except ValueError:
            continue
        values = [value for _, value in items]
        if len(values) >= 3:
            series[station] = (values, days)
    return series


def _render_level_forecast(level_rows: Sequence[Mapping[str, Any]]) -> None:
    """수위를 면적과 **따로** 전망한다.

    면적에서 수위를 환산하면 검증되지 않은 면적-수위 관계를 만들어 내는 셈이다.
    수위 관측이 지점마다 따로 존재하므로 그 시계열 자체의 추세와 변동으로
    전망하고, σ 도 수위 자신의 것을 쓴다.
    """

    series = _level_series(level_rows)
    if not series:
        return

    st.markdown("##### 지점별 수위 전망 · 수위 관측 자체의 추세")
    _not(
        "<b>면적에서 환산한 값이 아닙니다.</b> 지점별 위성 수위 관측 시계열에 추세를 "
        "적합해 따로 전망하고, 산포도 수위 자신의 변동(σ)에서 계산합니다. 검증된 "
        "수문 예측 모델이 아니라 <b>관측 추세 기반 참고 전망</b>입니다."
    )

    horizons = [7.0, 14.0, 30.0]
    station = st.selectbox(
        "지점", sorted(series), key="wt_level_station",
        help="지점마다 수위 관측 시계열이 따로 있습니다.",
    )
    values, days = series[station]
    rows, profile = forecast_with_bands(
        values, days, horizons, labels=[f"D+{int(h)}" for h in horizons]
    )
    if not rows or not profile.usable:
        st.info(f"{station}의 수위 관측이 부족해 전망을 생략합니다.")
        return

    metrics = st.columns(4)
    metrics[0].metric("최근 관측 수위", f"{values[-1]:.3f} m")
    metrics[1].metric("수위 변동 σ", f"{profile.sigma_per_interval * 100:.2f} %")
    metrics[2].metric("D+30 전망", f"{rows[-1]['예측값']:.3f} m")
    metrics[3].metric(
        "D+30 3σ 범위",
        f"{rows[-1]['-3σ']:.2f}–{rows[-1]['+3σ']:.2f} m",
    )

    table = [
        {
            "시점": row["target_date"],
            **{
                key: f"{row[key]:.3f}"
                for key in (f"{level:+d}σ" if level else "0σ" for level in SIGMA_LEVELS)
            },
        }
        for row in rows
    ]
    st.table(table)
    st.caption(
        f"{station} 관측 {profile.observation_count}시점 · 평균 간격 "
        f"{profile.mean_interval_days:.0f}일 · 하루당 σ {profile.sigma_per_day * 100:.2f}%. "
        "면적 전망은 4단계에 따로 있습니다."
    )


def _grid_range(grid: Sequence[Sequence[float]], pad_ratio: float = 0.06) -> list[float]:
    """3D z축 범위. 밴드 최소~최대에 약간의 여백만 준다."""

    flat = [float(value) for line in grid for value in line]
    if not flat:
        return [0.0, 1.0]
    low, high = min(flat), max(flat)
    span = high - low
    if span <= 0:
        pad = abs(high) * 0.1 or 1.0
        return [high - pad, high + pad]
    pad = span * pad_ratio
    return [low - pad, high + pad]


def _render_loaded_weather() -> None:
    """1단계에서 자동으로 불러온 기상을 보여 준다.

    기상 조회는 샘플을 불러올 때 이미 자동으로 일어난다. 그런데 화면에 아무
    표시가 없어서 '기상이 안 불러와진다'는 오해가 생겼다. 무엇을 언제 어디서
    받았는지 그 자리에서 밝힌다.
    """

    rows = st.session_state.get("weather_data") or []
    notice = st.session_state.get("weather_query_notice") or {}
    if not rows:
        st.warning("기상 데이터를 불러오지 못했습니다. 기상 화면에서 직접 조회해 보세요.")
        return

    observed = [r for r in rows if str(r.get("kind")) == "observed"]
    scenario = [r for r in rows if str(r.get("kind")) == "scenario"]
    rain = [
        float(r["precipitation_mm"])
        for r in rows
        if isinstance(r.get("precipitation_mm"), (int, float))
    ]
    st.markdown("**기상 데이터도 함께 자동으로 불러왔습니다**")
    columns = st.columns(4)
    columns[0].metric("전체 행", f"{len(rows)}일")
    columns[1].metric("관측 구간", f"{len(observed)}일", help="ASOS 과거 관측")
    columns[2].metric("미래 시나리오", f"{len(scenario)}일", help="예측 구간 입력")
    columns[3].metric(
        "누적 강수", f"{sum(rain):,.1f} mm" if rain else "—",
        help="불러온 전체 구간 합계",
    )
    source = str(notice.get("source") or "")
    note = str(notice.get("note") or "")
    st.caption(
        (f"출처 · {source} · " if source else "")
        + (note or "기상 화면에서 값을 확인·편집할 수 있습니다.")
    )


def _render_area_and_level(result: Mapping[str, Any]) -> None:
    """수체 면적과 수위를 한 그래프에 겹쳐 보여 준다.

    둘은 별개의 양이지만 같은 예측에서 나온 결과라 함께 봐야 읽힌다. 축을
    좌우로 나누고 각각 자기 데이터 범위에 맞춰 조인다 — 0부터 그리면 변화가
    묻힌다.
    """

    from plotly.subplots import make_subplots

    steps = result.get("steps") or []
    if len(steps) < 2:
        return
    dates = [str(s.get("target_date") or "") for s in steps]
    areas = [s.get("water_area_km2") for s in steps]
    levels = [s.get("water_level_m") for s in steps]
    has_level = any(v is not None for v in levels)

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=dates, y=areas, name="수체 면적 (km²)", mode="lines+markers",
            line={"color": "#38bdf8", "width": 3}, marker={"size": 5},
            hovertemplate="%{x}<br>면적 %{y:,.4f} km²<extra></extra>",
        ),
        secondary_y=False,
    )
    if has_level:
        figure.add_trace(
            go.Scatter(
                x=dates, y=levels, name="수위 (m)", mode="lines+markers",
                line={"color": "#fbbf24", "width": 2, "dash": "dot"}, marker={"size": 5},
                hovertemplate="%{x}<br>수위 %{y:.4f} m<extra></extra>",
            ),
            secondary_y=True,
        )
    figure.update_layout(
        height=380, hovermode="x unified",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title={"text": "예측 구간 · 수체 면적과 수위", "x": 0.01},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        legend={"orientation": "h", "y": 1.14},
    )
    figure.update_xaxes(title_text="예측 날짜", gridcolor="#1e293b")
    figure.update_yaxes(
        title_text="수체 면적 (km²)", gridcolor="#1e293b",
        range=axis_range_for(figure, secondary_y=False), secondary_y=False,
    )
    if has_level:
        figure.update_yaxes(
            title_text="수위 (m)", showgrid=False,
            range=axis_range_for(figure, secondary_y=True), secondary_y=True,
        )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    if has_level:
        st.caption(
            "수위는 김해 게이지 관측 8쌍에서 잰 면적–수위 민감도(r=+0.919)로 산출한 "
            "값입니다. 검증된 수위계 환산식이 아니라 **관측 기반 근사**입니다."
        )
    else:
        st.caption("이 샘플에는 면적–수위 계수가 없어 수위를 산출하지 않았습니다.")
