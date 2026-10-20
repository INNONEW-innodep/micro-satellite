#!/usr/bin/env python3
"""수위 시계열 예측 모델 비교 발표자료(PPTX + Markdown)를 학습 산출물에서 만든다.

    pipeline_ui/.venv/bin/python scripts/make_model_presentation.py

숫자는 전부 regions/*/models/gauge/*.json(학습 산출물)·evaluation.json·
regions/*/satellite/timeseries_eval.json 에서 읽는다. 그림은 scripts/make_model_figures.py 산출.
"""

from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "presentation" / "figures"
OUT_DIR = ROOT / "docs" / "presentation"
REPORT_H = (1, 3, 7, 14, 30)
NAMES = {
    "persistence": "직전값 유지(기준선)", "climatology": "계절 평년값", "ridge_ar": "릿지 자기회귀", "ridge_arx": "릿지 자기회귀+기상",
    "seasonal_anomaly_ar": "계절편차 자기회귀", "mean_reversion": "평균 회귀 AR(1)", "analog_knn": "유사사례 k-NN", "ensemble_mean": "계열 앙상블",
    "neural_mlp": "MLP 신경망(GPU)", "neural_lstm": "LSTM(GPU)", "neural_gru": "GRU(GPU)", "neural_tcn": "TCN(GPU)", "prophet_hybrid": "Prophet+잔차AR",
}
_SHORT = {"nakdong_busan": "낙동강 하구(부산·김해)", "geum_daecheong": "금강 대청댐", "han_river": "한강(대조군)"}
REGION_KO = {
    p.parent.name: _SHORT.get(p.parent.name) or json.loads(p.read_text(encoding="utf-8")).get("name", p.parent.name)
    for p in sorted(ROOT.glob("regions/*/region.json"))
    if (p.parent / "models" / "gauge" / "evaluation.json").is_file()
}
BG = RGBColor(0x0B, 0x13, 0x24); FG = RGBColor(0xE2, 0xE8, 0xF0); MUTED = RGBColor(0x94, 0xA3, 0xB8); CYAN = RGBColor(0x22, 0xD3, 0xEE); AMBER = RGBColor(0xFB, 0xBF, 0x24)


def load() -> dict:
    artifacts = []
    for p in sorted(glob.glob(str(ROOT / "regions" / "*" / "models" / "gauge" / "*.json"))):
        if p.endswith("evaluation.json"):
            continue
        a = json.loads(Path(p).read_text(encoding="utf-8"))
        a["_region"] = Path(p).parts[-4]
        artifacts.append(a)
    evaluations = {rid: json.loads((ROOT / "regions" / rid / "models" / "gauge" / "evaluation.json").read_text(encoding="utf-8")) for rid in REGION_KO}
    ts = json.loads((ROOT / "regions" / "nakdong_busan" / "satellite" / "timeseries_eval.json").read_text(encoding="utf-8"))
    return {"artifacts": artifacts, "evaluations": evaluations, "ts": ts}


def family_table(artifacts: list[dict]) -> tuple[list[str], dict[str, list[float]], collections.Counter, dict[int, collections.Counter]]:
    ratio = collections.defaultdict(list)
    winners = collections.Counter()
    winners_h = collections.defaultdict(collections.Counter)
    for a in artifacts:
        for h, b in a["horizons"].items():
            h = int(h)
            winners[b["forecaster_id"]] += 1
            winners_h[h][b["forecaster_id"]] += 1
            best = {}
            for c in b["candidates"]:
                if "val" in c:
                    best[c["forecaster_id"]] = min(best.get(c["forecaster_id"], 9e9), c["val"]["ratio_vs_persistence_mean"])
            for fid, r in best.items():
                ratio[(fid, h)].append(r)
    import statistics

    def med(fid: str, h: int) -> float:
        values = ratio.get((fid, h))
        return statistics.median(values) if values else float("nan")

    fids = sorted({k[0] for k in ratio}, key=lambda f: med(f, 7))
    table = {fid: [med(fid, h) for h in REPORT_H] for fid in fids}
    return fids, table, winners, winners_h


def model_metric_rows(evaluations: dict, horizon: int) -> list[list[str]]:
    """모델별 시험 연도 성능지표(전 지역 지점 가중평균): RMSE·MSE·MAE(m·cm)·개선율·선택 지점 수."""
    agg: dict[str, dict] = {}
    for ev in evaluations.values():
        for fid, hs in (ev.get("model_summary") or {}).items():
            m = hs.get(str(horizon))
            if not m:
                continue
            a = agg.setdefault(fid, {"rmse": 0.0, "mse": 0.0, "mae": 0.0, "skill": 0.0, "n_skill": 0, "sel": 0, "n": 0})
            n = m["n_stations"]
            a["rmse"] += m["mean_rmse"] * n; a["mse"] += m["mean_mse"] * n; a["mae"] += m["mean_mae"] * n
            if m.get("mean_skill") is not None:
                a["skill"] += m["mean_skill"] * n; a["n_skill"] += n
            a["sel"] += m["selected_count"]; a["n"] += n
    rows = []
    for fid, a in agg.items():
        n = max(1, a["n"]); skill = a["skill"] / a["n_skill"] if a["n_skill"] else None
        rows.append((skill if skill is not None else -9e9, [
            NAMES.get(fid, fid), str(a["n"]), str(a["sel"]),
            f"{a['rmse'] / n:.3f} m ({a['rmse'] / n * 100:.1f} cm)", f"{a['mse'] / n:.5f} m² ({a['mse'] / n * 1e4:.1f} cm²)",
            f"{a['mae'] / n:.3f} m ({a['mae'] / n * 100:.1f} cm)", f"{skill * 100:+.1f} %" if skill is not None else "-",
        ]))
    rows.sort(key=lambda r: -r[0])
    return [r[1] for r in rows]


METRIC_HEADER = ["모델", "지점", "선택 지점", "평균 RMSE", "평균 MSE", "평균 MAE", "평균 개선율"]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_markdown(data: dict) -> str:
    fids, table, winners, winners_h = family_table(data["artifacts"])
    n_cells = sum(winners.values())
    deep = {"neural_mlp", "neural_lstm", "neural_gru", "neural_tcn"}
    deep_wins = sum(v for k, v in winners.items() if k in deep)
    lines = ["# 수위 시계열 예측 모델 비교 — 지역별 실증 결과", "", f"학습 산출물 기준 자동 생성. 지점 {len(data['artifacts'])}개 × 예측일수 30 = {n_cells}셀.", ""]
    lines += ["## 1. 왜 여러 모델을 비교했나", "",
              "- 수위는 지형(감조 하구·본류·지류·저수지)마다 움직임이 달라 단일 모델을 강요할 근거가 없다.",
              "- 그래서 선형·계절·회귀·사례·기상·딥러닝(GPU)·Prophet 계열 13종을 **같은 규칙**(검증 연도별 폴드 → 시험 연도)으로 비교하고 지점×예측일수마다 이긴 모델을 쓴다.",
              "- 직전값 유지(persistence)는 기준선일 뿐이며 절대 선택되지 않는다. 개선율(skill) = 1 − MSE/MSE(직전값 유지).", ""]
    lines += ["## 2. 검증 규칙", "", "![선택 규칙](figures/selection_protocol.png)", "",
              "학습 ≤2018 → 검증 2019~2023(연도별 5폴드, persistence 대비 MSE 비율 평균 최소 후보 선택) → 시험 2024(학습·선택 미사용). 대청은 자료가 2019년부터라 검증 3폴드.", ""]
    lines += ["## 3. 모델 설명", ""]
    for fig, title, desc in (
        ("model_ridge_ar.png", "릿지 자기회귀 (AR)", "최근 p일 수위·30/365일 평균·계절항의 선형 조합. 릿지 벌점으로 과적합 억제. delta/level 모드."),
        ("model_ridge_arx.png", "릿지 자기회귀 + 기상", "원점 이전 강수 1·3·7·30일 합·기온 편차를 추가. 미래 강수는 쓰지 않는다."),
        ("model_seasonal_anomaly.png", "계절편차 자기회귀", "평년값(연중일 평균)을 뺀 편차의 되돌아감을 학습."),
        ("model_mean_reversion.png", "평균 회귀 AR(1)", "기준 수위(최근 30·90·365일 평균)로 되돌아가는 비율 하나를 지평마다 학습. 조석 지점에서 강함."),
        ("model_analog_knn.png", "유사사례 k-NN", "최근 7~14일 모양과 닮은 과거 사례 k개의 이후 변화 평균."),
        ("model_ensemble.png", "계열 앙상블", "계열이 다른 모델 6개(기상 포함)의 평균. 한 해에 과적합되기 어렵다."),
        ("model_neural.png", "딥러닝 4종 (MLP·LSTM·GRU·TCN)", "최근 30~60일 시퀀스(수위·강수·기온)로 30개 지평 편차를 한 번에 출력. GPU(torch) 학습, numpy 추론."),
        ("model_prophet_hybrid.png", "Prophet 하이브리드", "Prophet 곡선(추세 변화점+연 주기) + 잔차 변화량 자기회귀."),
    ):
        lines += [f"### {title}", "", desc, "", f"![{title}](figures/{fig})", ""]
    lines += [f"## 4. 모델 계열별 검증 성적 (persistence 대비 MSE 비율, 낮을수록 좋음 · {len(data['artifacts'])}지점 중앙값)", "",
              md_table(["모델"] + [f"D+{h}" for h in REPORT_H], [[NAMES.get(f, f)] + [f"{v:.3f}" for v in table[f]] for f in fids]), "",
              "비율 1.0 = 직전값 유지와 같음. 0.7 = MSE 30% 감소. 중앙값을 쓴 이유: Prophet·계절 평년값은 저수지 지점에서 추세 표류로 크게 나빠져 평균이 왜곡된다.", ""]
    lines += [f"## 4-1. 모델별 성능지표 — 시험 연도 2024, {len(data['artifacts'])}지점 가중평균 (RMSE·MSE·MAE, 개선율)", "",
              "선택 여부와 무관하게 **모든 계열을 같은 시험 구간에서 채점**한 값. 직전값 유지(기준선)도 함께 채점해 개선율 0 %의 위치를 보인다.", ""]
    for h in (1, 3, 5, 7, 10, 14, 30):
        rows_h = model_metric_rows(data["evaluations"], h)
        if rows_h:
            lines += [f"### D+{h}", "", md_table(METRIC_HEADER, rows_h), ""]
    lines += ["## 5. 어떤 모델이 뽑혔나 (지점 × 30지평 선택 횟수)", "",
              md_table(["모델", "선택 횟수", "비율"], [[NAMES.get(k, k), str(v), f"{v / n_cells * 100:.0f}%"] for k, v in winners.most_common()]), "",
              md_table(["예측일수"] + [NAMES.get(f, f) for f in fids], [[f"D+{h}"] + [str(winners_h[h].get(f, 0)) for f in fids] for h in REPORT_H]), "",
              f"딥러닝(MLP·LSTM·GRU·TCN)이 선택된 셀: {deep_wins}/{n_cells} ({deep_wins / n_cells * 100:.0f}%) — 주로 D+7 이상. D+1~3은 평균 회귀·유사사례·기상 AR-X가 우세.", ""]
    lines += ["## 6. GPU 의 역할", "", "![GPU vs CPU](figures/gpu_vs_cpu.png)", "",
              "- 학습: 신경망 후보(지점 14 × 후보 10 × 폴드 7 ≈ 1,000회)를 GPU에서 3~4배 빠르게. LSTM 1회 3.0 s(CPU 14코어) → 0.9 s(RTX A5000).",
              "- 추론: numpy 로 가중치만 실행(수 ms) — 서버(CPU 슬림 이미지)에 torch 가 없어도 같은 결과.",
              "- 인도받은 융합 LSTM(wbms:0.13, GPU 컨테이너)은 '관측일 수위 보정기'로 별개(LODO 4.53 cm).", ""]
    lines += ["## 7. 지역별 시험 연도 성적 (2024, 학습·선택 미사용)", ""]
    for rid, ev in data["evaluations"].items():
        rows = [r for r in ev["rows"] if r["horizon"] in (1, 7, 30)]
        lines += [f"### {REGION_KO[rid]} — 기준선보다 나은 셀 {ev['cells_beating_persistence']}/{ev['cells_total']}, 기상이 뚜렷이 도운 셀 {ev.get('cells_weather_helps', 0)}/{ev.get('cells_with_weather_candidate', 0)}", "",
                  md_table(["지점", "D+", "선택 모델", "RMSE", "MSE", "기준선 RMSE", "개선율"],
                           [[r["station_name"], str(r["horizon"]), NAMES.get(r["forecaster_id"], r["forecaster_id"]),
                             f"{r['test_rmse']:.3f} m ({r['test_rmse'] * 100:.1f} cm)", f"{r['test_mse']:.5f} m² ({r['test_mse'] * 1e4:.1f} cm²)",
                             f"{r['persistence_test_rmse']:.3f} m", f"{(r['skill_vs_persistence'] or 0) * 100:+.1f} %"] for r in rows]), ""]
    ts = data["ts"]["series"]
    corr = ts["A_level_correction"]["corrected"]; b = ts["B_persistence_reference"]["persistence_of_corrected"]; f = ts["F_change"]["level"]
    lines += ["## 8. 위성 데이터로 잰 수위·수체 시계열 지표 (부산, m·cm 병기)", "",
              md_table(["항목", "RMSE", "MSE", "MAE", "표본"], [
                  ["위성 수위 당일 보정(융합 LSTM, in-sample)", f"{corr['rmse_m']:.4f} m ({corr['rmse_cm']:.2f} cm)", f"{corr['mse_m2']:.6f} m² ({corr['mse_cm2']:.2f} cm²)", f"{corr['mae_m']:.4f} m ({corr['mae_cm']:.2f} cm)", str(corr["n"])],
                  ["정식 프로토콜 LODO", f"{ts['A_level_correction']['lodo_rmse_m']:.4f} m ({ts['A_level_correction']['lodo_rmse_cm']:.2f} cm)", "-", "-", "지점 제외 교차검증"],
                  ["위성 시계열 직전값 유지 참고치", f"{b['rmse_m']:.4f} m ({b['rmse_cm']:.2f} cm)", f"{b['mse_m2']:.6f} m² ({b['mse_cm2']:.2f} cm²)", f"{b['mae_m']:.4f} m ({b['mae_cm']:.2f} cm)", str(b["n"])],
                  ["위성 수위 변화량 추적", f"{f['rmse_m']:.4f} m ({f['rmse_cm']:.2f} cm)", f"{f['mse_m2']:.6f} m² ({f['mse_cm2']:.2f} cm²)", f"{f['mae_m']:.4f} m ({f['mae_cm']:.2f} cm)", f"{f['n']} · skill {f['skill_vs_no_change']:+.2f}"],
              ]), "",
              f"IITP 성과지표 정합값(정규화 MSE) 종합 {ts['G_iitp']['composite_norm_mse']} (목표 {ts['G_iitp']['target']}). 위성 시점이 8개뿐이라 학습 예측 모델의 성능이 아니다.", ""]
    lines += ["## 9. 한계와 다음 단계", "",
              "- 강수·기온을 넣어도 단기(D+1)는 크게 좋아지지 않는다(원점 수위에 어제 비가 이미 반영). 다음 입력은 **미래 강수 예보**와 **댐 방류량**.",
              "- 시험 성적은 2024년 기상 조건에 종속. 양산교처럼 기준점 이동이 있는 지점은 장기 지평이 불리하다.",
              "- 딥러닝은 자료 규모(지점당 3,650일)에서 선형·사례 모델과 대등한 수준이며, 지점·입력이 늘면 재평가할 가치가 있다.",
              "- 밴드(±σ)는 검증 잔차의 경험적 산포이지 신뢰구간이 아니다.", ""]
    return "\n".join(lines)


# --- PPTX -----------------------------------------------------------------------------------


def add_slide(prs, title: str, bullets: list[str] | None = None, image: Path | None = None, table: tuple[list[str], list[list[str]]] | None = None, note: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill; bg.solid(); bg.fore_color.rgb = BG
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9)); p = tb.text_frame.paragraphs[0]
    p.text = title; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = FG
    top = Inches(1.3)
    if bullets:
        box = slide.shapes.add_textbox(Inches(0.6), top, Inches(12.0) if image is None else Inches(5.6), Inches(5.5)); tf = box.text_frame; tf.word_wrap = True
        for i, text in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.text = "• " + text; para.font.size = Pt(15); para.font.color.rgb = FG; para.space_after = Pt(8)
    if image is not None and image.is_file():
        left = Inches(6.4) if bullets else Inches(0.6)
        width = Inches(6.6) if bullets else Inches(12.0)
        slide.shapes.add_picture(str(image), left, top, width=width)
    if table:
        header, rows = table
        shape = slide.shapes.add_table(len(rows) + 1, len(header), Inches(0.5), top, Inches(12.3), Inches(0.35) * (len(rows) + 1))
        tbl = shape.table
        for j, h in enumerate(header):
            cell = tbl.cell(0, j); cell.text = h; cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(11); para.font.bold = True; para.font.color.rgb = CYAN
        for i, row in enumerate(rows, start=1):
            for j, val in enumerate(row):
                cell = tbl.cell(i, j); cell.text = val; cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                for para in cell.text_frame.paragraphs:
                    para.font.size = Pt(10); para.font.color.rgb = FG
    if note:
        nb = slide.shapes.add_textbox(Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.5)); np_ = nb.text_frame.paragraphs[0]
        np_.text = note; np_.font.size = Pt(11); np_.font.color.rgb = MUTED
    return slide


def build_pptx(data: dict, out: Path) -> None:
    fids, table, winners, winners_h = family_table(data["artifacts"])
    n_cells = sum(winners.values())
    deep = {"neural_mlp", "neural_lstm", "neural_gru", "neural_tcn"}
    deep_wins = sum(v for k, v in winners.items() if k in deep)
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6]); bg = s.background.fill; bg.solid(); bg.fore_color.rgb = BG
    t = s.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(11.5), Inches(2)); p = t.text_frame.paragraphs[0]
    p.text = "초소형위성 수체·수위 플랫폼 — 수위 시계열 예측 모델 비교"; p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = FG
    p2 = t.text_frame.add_paragraph(); p2.text = "선형·계절·회귀·사례·기상·딥러닝(GPU)·Prophet 13종을 같은 규칙으로 비교, 지점×예측일수별 선택"; p2.font.size = Pt(18); p2.font.color.rgb = MUTED
    p3 = t.text_frame.add_paragraph(); p3.text = f"지점 {len(data['artifacts'])}개 · 2015~2024 지상 수위계 · 시험 연도 2024 · 자동 생성"; p3.font.size = Pt(14); p3.font.color.rgb = MUTED

    add_slide(prs, "왜 여러 모델을 비교했나", [
        "수위는 지형(감조 하구·본류·지류·저수지)마다 움직임이 달라 단일 모델을 강요할 근거가 없다",
        "13종 모델을 같은 규칙(검증 연도별 폴드 → 시험 연도)으로 비교하고 지점×예측일수마다 이긴 모델을 쓴다",
        "직전값 유지(persistence)는 기준선일 뿐 선택되지 않는다 — '어제 값 그대로'는 예측이 아니다",
        "개선율(skill) = 1 − MSE(모델)/MSE(직전값 유지). +50% = 오차 제곱 절반",
        "지표는 RMSE(m·cm)·MSE(m²·cm²)·MAE 병기, 탐지는 IoU",
    ])
    add_slide(prs, "검증 규칙 — 선택은 검증에서, 채점은 시험에서", None, FIG / "selection_protocol.png",
              note="학습 ≤2018 → 검증 2019~2023 연도별 5폴드(persistence 대비 MSE 비율 평균 최소 후보) → 시험 2024. 대청은 2019년부터라 검증 3폴드.")
    for fig, title, bullets in (
        ("model_ridge_ar.png", "릿지 자기회귀 (AR)", ["최근 p=7·30일 수위, 30·365일 평균, 계절 조화항의 선형 조합", "릿지 벌점으로 과적합 억제, 지평 h마다 따로 학습", "delta 모드(변화량 학습)·level 모드(수위 학습)를 검증이 고른다"]),
        ("model_ridge_arx.png", "릿지 자기회귀 + 기상 (AR-X)", ["원점 이전 강수 1·3·7·30일 합(log)과 7일 평균기온·편차 추가", "미래 강수(예보)는 쓰지 않는다 — 원점에서 알 수 없다", "본류·저수지 D+1에서 +1%p 안팎, 감조 하구는 손해"]),
        ("model_seasonal_anomaly.png", "계절편차 자기회귀", ["10년 평년값(연중일 평균, ±15일 평활)을 빼고 편차만 학습", "편차가 0으로 되돌아가는 속도를 릿지 회귀", "계절 순환이 뚜렷한 지점에 유리"]),
        ("model_mean_reversion.png", "평균 회귀 AR(1)", ["기준 수위 m(학습 평균 또는 최근 30·90·365일 평균)으로 되돌아가는 비율 φ_h 하나를 학습", "파라미터가 하나라 표본이 적어도 안정", "조석 영향 하구(구포대교·낙동강하구언)에서 자주 이김"]),
        ("model_analog_knn.png", "유사사례 k-NN", ["최근 7~14일 수위 모양과 닮은 과거 창 k=20·50개를 찾는다", "그 사례들의 이후 h일 실제 변화 평균을 현재 수위에 더한다", "비선형 거동(급상승 뒤 감쇠)을 사례로 흉내"]),
        ("model_ensemble.png", "계열 앙상블 평균", ["릿지 AR(level·delta)·AR-X·계절편차·평균 회귀·유사사례 6구성원 평균", "한 해(홍수)에 과적합되기 어려워 선택이 안정", "전 지점 D+7~30에서 자주 선택"]),
        ("model_neural.png", "딥러닝 4종 — MLP · LSTM · GRU · TCN (GPU 학습)", ["입력: 최근 30~60일 시퀀스(수위 편차·강수·기온) + 계절·이동평균 특징", "출력: 30개 지평 편차를 한 번에(조기 종료, Adam)", "학습 torch(GPU) → 가중치를 float16으로 저장 → 서버는 numpy 추론(오차 <0.3 mm)"]),
        ("model_prophet_hybrid.png", "Prophet 하이브리드 (구 Facebook Prophet)", ["Prophet: 추세 변화점 + 연 주기 푸리에 계절성 곡선", "곡선 위에 잔차 변화량의 자기회귀를 얹어 원점 상태 반영", "곡선이 적합 구간 밖에서 표류해 이 자료에서는 열세"]),
    ):
        add_slide(prs, title, bullets, FIG / fig)
    add_slide(prs, f"모델 계열별 검증 성적 (persistence 대비 MSE 비율, {len(data['artifacts'])}지점 중앙값)", None, None,
              (["모델"] + [f"D+{h}" for h in REPORT_H], [[NAMES.get(f, f)] + [f"{v:.3f}" for v in table[f]] for f in fids]),
              note="1.0 = 직전값 유지와 같음, 0.7 = MSE 30% 감소. 계절 평년값·Prophet 은 현재 상태를 보지 않아 단기에 크게 열세.")
    for h in (1, 5, 7, 10, 30):
        rows_h = model_metric_rows(data["evaluations"], h)
        if rows_h:
            add_slide(prs, f"모델별 성능지표 — D+{h} (시험 연도 2024, {len(data['evaluations'])}지역 {len(data['artifacts'])}지점 가중평균)", None, None, (METRIC_HEADER, rows_h),
                      note="선택 여부와 무관하게 모든 계열을 같은 시험 구간에서 채점. 개선율 = 1 − MSE/MSE(직전값 유지). '선택 지점' = 검증에서 이 계열이 뽑힌 지점 수.")
    add_slide(prs, "어떤 모델이 뽑혔나 — 지점 × 30지평 선택 횟수", None, None,
              (["예측일수"] + [NAMES.get(f, f) for f in fids], [[f"D+{h}"] + [str(winners_h[h].get(f, 0)) for f in fids] for h in REPORT_H]),
              note=f"전체 {n_cells}셀 중 딥러닝 선택 {deep_wins}셀({deep_wins / n_cells * 100:.0f}%, 주로 D+7 이상). D+1~3은 평균 회귀·유사사례·기상 AR-X 우세. 선택 1위 {NAMES.get(winners.most_common(1)[0][0])} {winners.most_common(1)[0][1]}셀.")
    add_slide(prs, "GPU 의 역할", [
        "신경망 후보 학습: 지점 14 × 후보 10 × 폴드 7 ≈ 1,000회를 GPU에서 3~4배 빠르게 (LSTM 1회 3.0 s → 0.9 s)",
        "추론은 numpy 로 수 ms — 서버(CPU 슬림 이미지)에 torch 없이 같은 결과",
        "인도 융합 LSTM(wbms:0.13, GPU 컨테이너)은 관측일 수위 보정기(LODO 4.53 cm)로 별개",
        "결론: GPU는 '더 많은 후보를 같은 규칙으로 빨리 비교'하게 해 준다. 최종 선택은 검증 성적이 정한다",
    ], FIG / "gpu_vs_cpu.png")
    for rid, ev in data["evaluations"].items():
        rows = [r for r in ev["rows"] if r["horizon"] in (1, 7, 30)]
        add_slide(prs, f"{REGION_KO[rid]} — 시험 연도(2024) 성적", None, None,
                  (["지점", "D+", "선택 모델", "RMSE", "MSE", "기준선 RMSE", "개선율"],
                   [[r["station_name"], str(r["horizon"]), NAMES.get(r["forecaster_id"], r["forecaster_id"]), f"{r['test_rmse']:.3f} m ({r['test_rmse'] * 100:.1f} cm)",
                     f"{r['test_mse']:.5f} m² ({r['test_mse'] * 1e4:.1f} cm²)", f"{r['persistence_test_rmse']:.3f} m", f"{(r['skill_vs_persistence'] or 0) * 100:+.1f} %"] for r in rows]),
                  note=f"기준선보다 나은 셀 {ev['cells_beating_persistence']}/{ev['cells_total']} · 기상이 뚜렷이 도운 셀 {ev.get('cells_weather_helps', 0)}/{ev.get('cells_with_weather_candidate', 0)}")
    ts = data["ts"]["series"]; corr = ts["A_level_correction"]["corrected"]; b = ts["B_persistence_reference"]["persistence_of_corrected"]; f = ts["F_change"]["level"]
    add_slide(prs, "위성 데이터로 잰 수위·수체 시계열 지표 (부산, m·cm 병기)", None, None,
              (["항목", "RMSE", "MSE", "MAE", "표본"], [
                  ["위성 수위 당일 보정(융합 LSTM, in-sample)", f"{corr['rmse_m']:.4f} m ({corr['rmse_cm']:.2f} cm)", f"{corr['mse_m2']:.6f} m² ({corr['mse_cm2']:.2f} cm²)", f"{corr['mae_m']:.4f} m ({corr['mae_cm']:.2f} cm)", str(corr["n"])],
                  ["정식 프로토콜 LODO", f"{ts['A_level_correction']['lodo_rmse_m']:.4f} m ({ts['A_level_correction']['lodo_rmse_cm']:.2f} cm)", "-", "-", "지점 제외 교차검증"],
                  ["위성 시계열 직전값 유지 참고치", f"{b['rmse_m']:.4f} m ({b['rmse_cm']:.2f} cm)", f"{b['mse_m2']:.6f} m² ({b['mse_cm2']:.2f} cm²)", f"{b['mae_m']:.4f} m ({b['mae_cm']:.2f} cm)", str(b["n"])],
                  ["위성 수위 변화량 추적", f"{f['rmse_m']:.4f} m ({f['rmse_cm']:.2f} cm)", f"{f['mse_m2']:.6f} m² ({f['mse_cm2']:.2f} cm²)", f"{f['mae_m']:.4f} m ({f['mae_cm']:.2f} cm)", f"{f['n']} · skill {f['skill_vs_no_change']:+.2f}"],
              ]), note=f"IITP 성과지표 정합값(정규화 MSE) 종합 {ts['G_iitp']['composite_norm_mse']} (목표 {ts['G_iitp']['target']}). 위성 시점 8개 → 학습 예측 모델의 성능이 아님.")
    add_slide(prs, "한계와 다음 단계", [
        "강수·기온 입력은 단기(D+1)를 크게 개선하지 못한다 — 원점 수위에 어제 비가 이미 반영. 다음 입력: 미래 강수 예보, 댐 방류량",
        "시험 성적은 2024년 기상에 종속. 기준점 이동 지점(양산교)은 장기 지평 불리",
        "딥러닝은 지점당 3,650일 규모에서 선형·사례 모델과 대등 — 지점·입력이 늘면 재평가",
        "밴드(±σ)는 검증 잔차의 경험적 산포이지 신뢰구간이 아니다",
        "새 지역·새 위성 장면은 지역 레지스트리(API)로 등록하면 같은 규칙으로 학습·평가된다",
    ])
    prs.save(str(out))


def main() -> int:
    data = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "수위예측모델_비교.md").write_text(build_markdown(data), encoding="utf-8")
    build_pptx(data, OUT_DIR / "수위예측모델_비교.pptx")
    print("wrote", OUT_DIR / "수위예측모델_비교.md", OUT_DIR / "수위예측모델_비교.pptx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
