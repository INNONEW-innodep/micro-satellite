#!/usr/bin/env python3
"""수위 예측 모델 설명용 그림을 실제 자료(구포대교 등)로 그린다.

산출: ui_next/assets/models/*.png (화면 안내 상자) + docs/presentation/figures/*.png (발표 자료).
pipeline_ui/.venv 의 matplotlib 를 쓴다:  pipeline_ui/.venv/bin/python scripts/make_model_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.app.gauge.forecasters import climatology, rolling_mean  # noqa: E402
from backend.app.gauge.regions import RegionRegistry  # noqa: E402

_fonts = {f.name for f in font_manager.fontManager.ttflist}
for candidate in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"):
    if candidate in _fonts:
        mpl.rcParams["font.family"] = candidate
        break
mpl.rcParams["axes.unicode_minus"] = False
DARK = {"bg": "#0b1324", "panel": "#0f172a", "grid": "#1e293b", "fg": "#cbd5e1", "cyan": "#22d3ee", "amber": "#fbbf24", "pink": "#f472b6", "green": "#34d399", "gray": "#64748b", "violet": "#a78bfa"}
OUT_UI = ROOT / "ui_next" / "assets" / "models"
OUT_DOC = ROOT / "docs" / "presentation" / "figures"


def style(ax, title: str, ylabel: str = "수위 (m)"):
    ax.set_facecolor(DARK["panel"])
    ax.grid(color=DARK["grid"], linewidth=0.6)
    ax.tick_params(colors=DARK["fg"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(DARK["grid"])
    ax.set_title(title, color="#e0f2fe", fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel, color=DARK["fg"], fontsize=9)


def save(fig, name: str):
    for out in (OUT_UI, OUT_DOC):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, dpi=150, facecolor=DARK["bg"], bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main() -> int:
    registry = RegionRegistry(ROOT / "regions", ROOT / "backend" / "data" / "regions")
    series_set = registry.load_series("nakdong_busan")
    s = series_set.series["구포대교"]
    dates = s.dates.astype("datetime64[D]").astype(object)
    y = s.filled
    n = s.n

    # 1) ridge_ar: lags + rolling means + seasonal
    origin = np.where(s.dates == np.datetime64("2024-07-10", "D"))[0][0]
    lo, hi = origin - 60, origin + 31
    fig, ax = plt.subplots(figsize=(9, 3.6))
    style(ax, "릿지 자기회귀 (AR): 최근 p일 수위 + 30·365일 평균 + 계절항 → h일 뒤 수위")
    ax.plot(dates[lo:origin + 1], y[lo:origin + 1], color=DARK["cyan"], lw=2, label="관측(원점까지)")
    ax.plot(dates[origin + 1:hi], y[origin + 1:hi], color=DARK["gray"], lw=1.2, ls="--", label="원점 이후 실제")
    ax.axvspan(dates[origin - 6], dates[origin], color=DARK["amber"], alpha=0.18, label="최근 p=7일 (입력 특징)")
    r30 = rolling_mean(y, 30, np.arange(lo, hi))
    ax.plot(dates[lo:hi], r30, color=DARK["violet"], lw=1.2, label="30일 이동평균 (특징)")
    ax.axvline(dates[origin], color=DARK["pink"], lw=1, ls=":")
    ax.annotate("원점 t", (dates[origin], y[origin]), color=DARK["pink"], fontsize=9, xytext=(5, 12), textcoords="offset points")
    ax.annotate("목표 y[t+7]", (dates[origin + 7], y[origin + 7]), color=DARK["amber"], fontsize=9, xytext=(5, -16), textcoords="offset points", arrowprops={"arrowstyle": "->", "color": DARK["amber"]})
    ax.legend(loc="upper left", fontsize=8, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    fig.text(0.01, -0.06, "y_hat[t+h] = w·[y[t..t−p+1], 평균30, 평균365, sin/cos(연중일)] + b   (릿지 벌점으로 과적합 억제, 지평 h마다 따로 학습)", color=DARK["fg"], fontsize=9)
    save(fig, "model_ridge_ar.png")

    # 2) seasonal anomaly: climatology + anomaly
    mask = s.dates <= np.datetime64("2018-12-31", "D")
    clim = climatology(s, mask)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    style(axes[0], "계절편차 자기회귀: 10년 평년값(연중일)을 빼고 남는 편차를 학습")
    a, b = np.where(s.dates == np.datetime64("2022-01-01", "D"))[0][0], np.where(s.dates == np.datetime64("2023-12-31", "D"))[0][0]
    axes[0].plot(dates[a:b], y[a:b], color=DARK["cyan"], lw=1, label="관측")
    axes[0].plot(dates[a:b], clim[s.doy[a:b] - 1], color=DARK["amber"], lw=2, label="평년값 (연중일 평균, ±15일 평활)")
    axes[0].legend(loc="upper left", fontsize=8, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    style(axes[1], "편차 = 관측 − 평년값  (이 편차가 0으로 되돌아가는 속도를 릿지 회귀로 학습)", "편차 (m)")
    axes[1].plot(dates[a:b], y[a:b] - clim[s.doy[a:b] - 1], color=DARK["green"], lw=1)
    axes[1].axhline(0, color=DARK["gray"], lw=0.8)
    save(fig, "model_seasonal_anomaly.png")

    # 3) mean reversion
    fig, ax = plt.subplots(figsize=(9, 3.6))
    style(ax, "평균 회귀 AR(1): y[t+h] − m = φ_h·(y[t] − m),  m = 최근 365일 평균")
    lo, hi = origin - 120, origin + 31
    m365 = rolling_mean(y, 365, np.arange(lo, hi))
    ax.plot(dates[lo:origin + 1], y[lo:origin + 1], color=DARK["cyan"], lw=2, label="관측")
    ax.plot(dates[lo:hi], m365, color=DARK["amber"], lw=1.5, label="기준 수위 m (365일 평균)")
    phi = [0.9, 0.75, 0.55, 0.35, 0.2]
    hs = [1, 3, 7, 14, 30]
    pts = [(dates[origin + h], m365[origin - lo + h] + p * (y[origin] - m365[origin - lo])) for h, p in zip(hs, phi)]
    ax.plot([dates[origin]] + [p[0] for p in pts], [y[origin]] + [p[1] for p in pts], color=DARK["pink"], lw=2, marker="o", label="예측: 편차가 φ_h 배로 줄며 m 으로 회귀 (그림용 φ)")
    ax.axvline(dates[origin], color=DARK["pink"], lw=1, ls=":")
    ax.legend(loc="upper left", fontsize=8, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    save(fig, "model_mean_reversion.png")

    # 4) analog kNN
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1.2]})
    style(axes[0], "유사사례 k-NN: 최근 7일 '모양'과 닮은 과거 창 k개를 찾는다")
    q = y[origin - 6:origin + 1] - y[origin]
    lib = np.arange(400, origin - 40)
    windows = np.stack([y[t - 6:t + 1] - y[t] for t in lib])
    d = ((windows - q) ** 2).sum(axis=1)
    nn_idx = lib[np.argsort(d)[:20]]
    for t in nn_idx[:8]:
        axes[0].plot(range(-6, 1), y[t - 6:t + 1] - y[t], color=DARK["gray"], lw=1, alpha=0.7)
    axes[0].plot(range(-6, 1), q, color=DARK["cyan"], lw=2.5, label="현재 최근 7일 (원점 대비 편차)")
    axes[0].plot([], [], color=DARK["gray"], label="닮은 과거 사례 (8개 표시)")
    axes[0].set_xlabel("원점 기준 일수", color=DARK["fg"], fontsize=9)
    axes[0].legend(loc="upper left", fontsize=8, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    style(axes[1], "그 사례들이 이후 어떻게 됐나 → 평균 변화를 더한다", "이후 변화 (m)")
    fut = np.stack([y[t + 1:t + 31] - y[t] for t in nn_idx])
    for row in fut[:8]:
        axes[1].plot(range(1, 31), row, color=DARK["gray"], lw=0.8, alpha=0.6)
    axes[1].plot(range(1, 31), fut.mean(axis=0), color=DARK["pink"], lw=2.5, label="k=20 평균 = 예측 변화")
    axes[1].axhline(0, color=DARK["grid"], lw=0.8)
    axes[1].set_xlabel("h (일)", color=DARK["fg"], fontsize=9)
    axes[1].legend(loc="upper left", fontsize=8, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    save(fig, "model_analog_knn.png")

    # 5) ARX with rain
    wx = registry.load_weather_rows("nakdong_busan")
    rain = {r["date"]: r.get("precipitation_mm") or 0 for r in wx}
    lo, hi = np.where(s.dates == np.datetime64("2023-07-01", "D"))[0][0], np.where(s.dates == np.datetime64("2023-08-15", "D"))[0][0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    style(ax, "릿지 자기회귀 + 기상 (AR-X): 원점 이전 강수 1·3·7·30일 합, 기온을 특징에 추가")
    ax2 = ax.twinx()
    rr = [rain.get(str(dd), 0) for dd in s.dates[lo:hi]]
    ax2.bar(dates[lo:hi], rr, color=DARK["violet"], alpha=0.45, width=1.0, label="일 강수 (mm)")
    ax2.set_ylabel("강수 (mm)", color=DARK["fg"], fontsize=9)
    ax2.tick_params(colors=DARK["fg"], labelsize=9)
    ax.plot(dates[lo:hi], y[lo:hi], color=DARK["cyan"], lw=2, label="구포대교 수위")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    fig.text(0.01, -0.06, "원점의 수위에는 어제까지의 비가 이미 반영돼 있어 단기 이득이 작다(본류 +1%p, 감조 하구는 손해). 미래 강수 예보가 다음 입력.", color=DARK["fg"], fontsize=9)
    save(fig, "model_ridge_arx.png")

    # 6) ensemble
    fig, ax = plt.subplots(figsize=(9, 3.4))
    style(ax, "계열 앙상블: 서로 다른 계열(선형·계절·회귀·사례·기상·신경망) 예측의 평균")
    hs = np.arange(1, 31)
    rng = np.random.default_rng(3)
    base = y[origin]
    members = {"릿지 AR": 0.06, "계절편차 AR": 0.02, "평균 회귀": 0.09, "유사사례": 0.04, "AR-X(기상)": 0.05}
    curves = []
    for i, (name, amp) in enumerate(members.items()):
        c = base + amp * (1 - np.exp(-hs / 10)) + rng.normal(0, 0.01, 30).cumsum() * 0.3
        curves.append(c)
        ax.plot(hs, c, lw=1, alpha=0.75, label=name)
    ax.plot(hs, np.mean(curves, axis=0), color=DARK["pink"], lw=3, label="앙상블 평균 (그림용)")
    ax.set_xlabel("h (일)", color=DARK["fg"], fontsize=9)
    ax.legend(loc="upper left", fontsize=8, ncol=2, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    save(fig, "model_ensemble.png")

    # 7) neural: LSTM sequence → 30 outputs (schematic)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_facecolor(DARK["bg"]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.set_title("딥러닝(LSTM·GRU·TCN·MLP): 최근 30~60일 시퀀스(수위·강수·기온) → 30개 지평 편차를 한 번에 출력", color="#e0f2fe", fontsize=12, fontweight="bold", loc="left")
    for i in range(6):
        ax.add_patch(plt.Rectangle((0.3 + i * 0.6, 1.2), 0.5, 0.6, color=DARK["cyan"], alpha=0.35 + 0.1 * i))
        ax.text(0.55 + i * 0.6, 0.85, f"t−{5 - i}", color=DARK["fg"], ha="center", fontsize=8)
    ax.text(2.1, 2.05, "입력 시퀀스 (window 일 × [수위 편차, 강수, 기온])", color=DARK["fg"], fontsize=9, ha="center")
    ax.annotate("", (4.6, 1.5), (4.0, 1.5), arrowprops={"arrowstyle": "->", "color": DARK["fg"], "lw": 1.5})
    ax.add_patch(plt.Rectangle((4.7, 0.8), 1.8, 1.4, color=DARK["violet"], alpha=0.45))
    ax.text(5.6, 1.5, "LSTM / GRU\nTCN / MLP\n(GPU 학습)", color="#f5f3ff", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.annotate("", (7.1, 1.5), (6.6, 1.5), arrowprops={"arrowstyle": "->", "color": DARK["fg"], "lw": 1.5})
    for j in range(5):
        ax.add_patch(plt.Rectangle((7.3 + j * 0.5, 1.2), 0.42, 0.6, color=DARK["amber"], alpha=0.6))
        ax.text(7.51 + j * 0.5, 0.85, ["D+1", "D+2", "…", "D+29", "D+30"][j], color=DARK["fg"], ha="center", fontsize=8)
    ax.text(8.5, 2.05, "출력: Δy[t+1..t+30] (한 번에)", color=DARK["fg"], fontsize=9, ha="center")
    ax.text(0.3, 0.25, "학습: torch(GPU, 조기 종료) · 추론: numpy 로 가중치만 실행(서버에 torch 불필요) · 같은 검증 규칙으로 선형 모델과 공정 비교", color=DARK["gray"], fontsize=9)
    save(fig, "model_neural.png")

    # 8) prophet hybrid
    fig, ax = plt.subplots(figsize=(9, 3.6))
    style(ax, "Prophet 하이브리드: Prophet 곡선(추세 변화점 + 연 주기) + 잔차 자기회귀")
    a, b = np.where(s.dates == np.datetime64("2021-01-01", "D"))[0][0], np.where(s.dates == np.datetime64("2023-12-31", "D"))[0][0]
    ax.plot(dates[a:b], y[a:b], color=DARK["cyan"], lw=0.9, label="관측")
    trend = rolling_mean(y, 365, np.arange(a, b))
    season = clim[s.doy[a:b] - 1] - clim.mean()
    ax.plot(dates[a:b], trend + season, color=DARK["amber"], lw=2, label="곡선 (추세 + 계절, 그림용 근사)")
    ax.legend(loc="upper left", fontsize=8, facecolor=DARK["panel"], labelcolor=DARK["fg"], edgecolor=DARK["grid"])
    fig.text(0.01, -0.06, "y_hat[t+h] = prophet(t+h) + AR_h(잔차 최근 p일). Prophet 은 폴드마다 1회 적합, 곡선을 저장해 numpy 로 추론.", color=DARK["fg"], fontsize=9)
    save(fig, "model_prophet_hybrid.png")

    # 9) selection protocol
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_facecolor(DARK["bg"]); ax.axis("off"); ax.set_xlim(2015, 2025); ax.set_ylim(0, 6)
    ax.set_title("모델 선택 규칙: 학습 ≤2018 → 검증 2019~2023 연도별 폴드(직전값 유지 대비 MSE 비율 평균 최소) → 시험 2024 (채점)", color="#e0f2fe", fontsize=11, fontweight="bold", loc="left")
    for i, yr in enumerate(range(2019, 2024)):
        ax.add_patch(plt.Rectangle((2015, 5 - i), yr - 2015, 0.8, color=DARK["gray"], alpha=0.35))
        ax.add_patch(plt.Rectangle((yr, 5 - i), 1, 0.8, color=DARK["cyan"], alpha=0.6))
        ax.text(2015.1, 5.4 - i, f"폴드 {i + 1}: ≤{yr - 1} 학습 → {yr} 검증", color=DARK["fg"], fontsize=8, va="center")
    ax.add_patch(plt.Rectangle((2015, 0), 9, 0.8, color=DARK["gray"], alpha=0.35))
    ax.add_patch(plt.Rectangle((2024, 0), 1, 0.8, color=DARK["amber"], alpha=0.7))
    ax.text(2015.1, 0.4, "최종: ≤2023 재적합 → 2024 시험 (학습·선택에 미사용)", color=DARK["fg"], fontsize=8, va="center")
    for yr in range(2015, 2026):
        ax.text(yr, -0.4, str(yr), color=DARK["gray"], fontsize=7, ha="center")
    save(fig, "selection_protocol.png")

    # 10) GPU vs CPU (measured)
    fig, ax = plt.subplots(figsize=(6, 3))
    style(ax, "GPU 가 하는 일: 신경망 후보 학습 시간 (LSTM 64, window 60, 1지점 1폴드)", "초")
    ax.bar(["CPU (14코어)", "GPU (RTX A5000)"], [3.0, 0.9], color=[DARK["gray"], DARK["cyan"]])
    for i, v in enumerate([3.0, 0.9]):
        ax.text(i, v + 0.08, f"{v:.1f} s", color=DARK["fg"], ha="center", fontsize=10)
    fig.text(0.01, -0.08, "지점 14 × 후보 10 × 폴드 7 ≈ 1,000회 학습: CPU ≈ 50분 → GPU ≈ 15분. 추론은 numpy(CPU)로 충분(수 ms).", color=DARK["fg"], fontsize=9)
    save(fig, "gpu_vs_cpu.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
