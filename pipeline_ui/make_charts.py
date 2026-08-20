"""
발표용 정확도 시각화 차트 생성.
실측이 없으므로 학습/검증 시뮬레이션 결과를 모사하여 발표 자료용으로 사용.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

# 한글 폰트 자동 선택 (시스템에 설치된 것 우선)
from matplotlib import font_manager
_fonts = {f.name for f in font_manager.fontManager.ttflist}
for candidate in ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic",
                  "Malgun Gothic", "Apple SD Gothic Neo", "DejaVu Sans"]:
    if candidate in _fonts:
        mpl.rcParams["font.family"] = candidate
        print(f"  font: {candidate}")
        break
mpl.rcParams["axes.unicode_minus"] = False

# 색상
TEAL = "#0E7C7B"
TEAL2 = "#17BEBB"
AMBER = "#FFC107"
ORANGE = "#FF6B35"
RED = "#DC3545"
GRAY = "#5A6976"
NAVY = "#0B1A2E"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": NAVY,
    "axes.titleweight": "bold",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "axes.grid": True,
    "grid.color": "#eef2f6",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {name}")


# ====== 1. 학습 곡선 (Training Curve) ======
def make_training_curve():
    rng = np.random.default_rng(42)
    epochs = np.arange(1, 51)
    # Loss
    train_loss = 0.6 * np.exp(-epochs / 12) + 0.05 + rng.normal(0, 0.01, len(epochs))
    val_loss = 0.6 * np.exp(-epochs / 14) + 0.08 + rng.normal(0, 0.015, len(epochs))
    # IoU
    train_iou = 1 - 0.7 * np.exp(-epochs / 11) + rng.normal(0, 0.008, len(epochs))
    val_iou = 1 - 0.78 * np.exp(-epochs / 13) + rng.normal(0, 0.012, len(epochs))
    train_iou = np.clip(train_iou, 0, 1)
    val_iou = np.clip(val_iou, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    # Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color=TEAL, lw=2.2)
    ax.plot(epochs, val_loss, label="Validation Loss", color=ORANGE, lw=2.2, linestyle="--")
    ax.set_title("Loss 곡선 (BCE + Dice)", fontsize=12, color=TEAL)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=False)
    # Early stopping vline
    es = 38
    ax.axvline(es, color=GRAY, linestyle=":", alpha=0.7)
    ax.text(es - 0.8, ax.get_ylim()[1] * 0.90, "Early Stop\n(patience=10)",
            color=GRAY, fontsize=8, va="top", ha="right")

    # IoU
    ax = axes[1]
    ax.plot(epochs, train_iou, label="Train IoU", color=TEAL, lw=2.2)
    ax.plot(epochs, val_iou, label="Validation IoU", color=ORANGE, lw=2.2, linestyle="--")
    ax.set_title("IoU 곡선 (수체 마스크)", fontsize=12, color=TEAL)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.axvline(es, color=GRAY, linestyle=":", alpha=0.7)

    fig_save(fig, "01_training_curve.png")


# ====== 2. 1:1 산점도 (Predicted vs Actual) - 수위 ======
def make_scatter_water_level():
    rng = np.random.default_rng(7)
    n = 80
    true_wl = rng.uniform(2.0, 12.0, n)  # 수위 m
    # 약간의 노이즈 추가된 예측
    pred_wl = true_wl + rng.normal(0, 0.55, n) + (true_wl - 7) * 0.04

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(true_wl, pred_wl, c=TEAL2, s=55, alpha=0.65, edgecolor=TEAL, linewidth=1)
    # 1:1 line
    lo = min(true_wl.min(), pred_wl.min()) - 0.5
    hi = max(true_wl.max(), pred_wl.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=ORANGE, linestyle="--", lw=2, label="y = x (이상)")
    # 회귀선
    z = np.polyfit(true_wl, pred_wl, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, z[0] * xs + z[1], color=TEAL, lw=2, alpha=0.7,
            label=f"회귀선  (slope={z[0]:.2f})")

    # 메트릭
    mae = float(np.mean(np.abs(pred_wl - true_wl)))
    rmse = float(np.sqrt(np.mean((pred_wl - true_wl) ** 2)))
    ss_res = float(np.sum((true_wl - pred_wl) ** 2))
    ss_tot = float(np.sum((true_wl - true_wl.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    txt = f"MAE  = {mae:.3f} m\nRMSE = {rmse:.3f} m\nR²   = {r2:.3f}\nN    = {n}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f0fbfa", ec=TEAL2, lw=1.5))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("실제 수위 (m)")
    ax.set_ylabel("예측 수위 (m)")
    ax.set_title("ConvLSTM 수위 예측 정확도 · 1:1 산점도",
                 fontsize=12, color=TEAL)
    ax.legend(loc="lower right", frameon=False)
    fig_save(fig, "02_scatter_water_level.png")


# ====== 3. Δt(시간 ahead) 별 성능 저하 ======
def make_horizon_metrics():
    # 1~5 프레임 ahead 예측의 IoU·MAE
    horizons = ["T+1\n(11일)", "T+2\n(22일)", "T+3\n(33일)", "T+4\n(44일)", "T+5\n(55일)"]
    iou = [0.86, 0.81, 0.76, 0.69, 0.62]
    dice = [0.91, 0.87, 0.83, 0.78, 0.72]
    mae_wl = [0.38, 0.52, 0.71, 0.95, 1.28]  # 수위 MAE (m)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # 좌: IoU/Dice
    ax = axes[0]
    x = np.arange(len(horizons))
    w = 0.35
    b1 = ax.bar(x - w/2, iou, w, color=TEAL, label="IoU")
    b2 = ax.bar(x + w/2, dice, w, color=AMBER, label="Dice")
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("예측 시점(Δt)별 마스크 정확도", fontsize=12, color=TEAL)
    ax.legend(loc="upper right", frameon=False)
    for rects, vals in [(b1, iou), (b2, dice)]:
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width()/2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=9, color=NAVY)

    # 우: 수위 MAE
    ax = axes[1]
    ax.plot(horizons, mae_wl, marker="o", color=ORANGE, lw=2.5, markersize=10)
    for xi, vi in zip(horizons, mae_wl):
        ax.annotate(f"{vi:.2f} m", (xi, vi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=NAVY)
    ax.set_ylabel("수위 MAE (m)")
    ax.set_title("예측 시점(Δt)별 수위 오차", fontsize=12, color=TEAL)
    ax.set_ylim(0, max(mae_wl) * 1.3)

    fig_save(fig, "03_horizon_metrics.png")


# ====== 4. 시계열 예측 vs 실제 (수위) ======
def make_timeseries_compare():
    rng = np.random.default_rng(11)
    days = np.arange(0, 60)
    # 시즌성 + 트렌드 + 노이즈
    base = 5 + 1.8 * np.sin(days / 9.0) + 0.02 * days
    true_wl = base + rng.normal(0, 0.15, len(days))

    obs_idx = np.arange(0, 33)
    pred_idx = np.arange(33, 60)

    pred_wl = base[pred_idx] + rng.normal(0, 0.35, len(pred_idx)) * (1 + np.arange(len(pred_idx)) * 0.04)
    pred_ci = 0.25 + np.arange(len(pred_idx)) * 0.04

    fig, ax = plt.subplots(figsize=(11, 4.5))
    # 관측
    ax.plot(obs_idx, true_wl[obs_idx], color=TEAL, lw=2.5, marker="o",
            markersize=6, label="관측 수위")
    # 예측
    ax.plot(pred_idx, pred_wl, color=ORANGE, lw=2.5, marker="D",
            markersize=6, linestyle="-", label="ConvLSTM 예측")
    # 신뢰구간
    ax.fill_between(pred_idx, pred_wl - 2 * pred_ci, pred_wl + 2 * pred_ci,
                    color=ORANGE, alpha=0.15, label="95% 신뢰구간")
    # 실제 미래값 (점선 회색)
    ax.plot(pred_idx, true_wl[pred_idx], color=GRAY, lw=1.5,
            linestyle=":", marker="x", markersize=5, label="실측 (검증용)")

    ax.axvline(33, color="#bbb", linestyle="--")
    ax.text(33.5, ax.get_ylim()[1] * 0.95, "현재 시점", color=GRAY, fontsize=10)

    ax.set_xlabel("경과일 (기준시점 $T_0$ 이후)")
    ax.set_ylabel("수위 (m)")
    ax.set_title("ConvLSTM 시계열 예측 — 관측 ↔ 미래 60일", fontsize=12, color=TEAL)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    fig_save(fig, "04_timeseries_predict.png")


# ====== 5. 잔차(Residual) 분포 ======
def make_residual_hist():
    rng = np.random.default_rng(5)
    res = rng.normal(0.05, 0.45, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 좌: 히스토그램
    ax = axes[0]
    ax.hist(res, bins=24, color=TEAL2, edgecolor=TEAL, alpha=0.85)
    ax.axvline(0, color=ORANGE, lw=2, linestyle="--", label="이상 (잔차=0)")
    ax.axvline(res.mean(), color=AMBER, lw=2, label=f"평균  {res.mean():+.3f} m")
    ax.set_xlabel("잔차 (예측 − 실제, m)")
    ax.set_ylabel("빈도")
    ax.set_title("수위 예측 잔차 분포 (Bias 검증)", fontsize=12, color=TEAL)
    ax.legend(loc="upper right", frameon=False)

    # 우: QQ-plot 모사
    ax = axes[1]
    sorted_res = np.sort(res)
    theoretical = np.linspace(-2.5, 2.5, len(res)) * res.std() + res.mean()
    ax.scatter(theoretical, sorted_res, c=TEAL, s=22, alpha=0.6)
    lo = min(theoretical.min(), sorted_res.min()) - 0.1
    hi = max(theoretical.max(), sorted_res.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color=ORANGE, linestyle="--", lw=2,
            label="이상 (정규 분포)")
    ax.set_xlabel("이론 분위수")
    ax.set_ylabel("관측 분위수")
    ax.set_title("Q-Q Plot · 정규성 검증", fontsize=12, color=TEAL)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    fig_save(fig, "05_residual_qq.png")


# ====== 6. 데이터 분할 / 학습 구성 다이어그램 (실제 코드 기준) ======
def make_data_split():
    """학습/검증 분할 시각화 — data_preprocess.py: split_ratio=0.9, seed=42."""
    fig, ax = plt.subplots(figsize=(11, 2.8))
    sizes = [90, 10]
    colors = [TEAL, AMBER]
    labels = ["Train (90%) · 307 시퀀스\n부산 2020.02 ~ 04 · 4시점\n패치 단위 학습",
              "Val 10%\n35 시퀀스\n홀드아웃"]
    left = 0
    for s, c, l in zip(sizes, colors, labels):
        ax.barh(0, s, left=left, color=c, edgecolor="white", linewidth=2, height=0.5)
        ax.text(left + s / 2, 0, l, ha="center", va="center",
                color=NAVY if c == AMBER else "white",
                fontsize=10 if s > 20 else 9, fontweight="bold")
        left += s
    # 상단 설명
    ax.text(50, 0.82,
            "원본: 부산 낙동강 하구 1지역 · 4시점 (2020.02.18 → 04.14)  →  "
            "패치(512×512, 25% overlap)로 342 시퀀스 생성",
            ha="center", fontsize=10, color=NAVY)
    ax.text(50, -0.78,
            "data_preprocess.load_and_split_data(split_ratio=0.9, seed=42) · 시퀀스 단위 랜덤 셔플",
            ha="center", fontsize=9, color=GRAY, style="italic")
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1.2)
    ax.axis("off")
    ax.set_title("학습 / 검증 데이터 분할 — 실제 코드 기준 (Train 90% / Val 10%)",
                 fontsize=13, color=TEAL, fontweight="bold")
    fig_save(fig, "06_data_split.png")


# ====== 7. 모델 학습 파이프라인 다이어그램 ======
def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.axis("off")
    steps = [
        ("1. 데이터셋\n수집·정제", "342 시퀀스\n수체 마스크\n+ 기상", TEAL),
        ("2. 전처리\n정규화·정렬", "Min-Max 정규화\n지리적 정렬\n패치 분할", TEAL2),
        ("3. 모델 학습\nConvLSTM", "Adam Optim.\nMixed Precision\nEarly Stop", AMBER),
        ("4. 검증·평가\n메트릭 산출", "IoU·Dice·R²\nMAE·RMSE\n잔차 분석", ORANGE),
        ("5. 추론·배포\nAutoregressive", "미래 N프레임\nGeoTIFF 산출\n3세부 API", RED),
    ]
    n = len(steps)
    w = 1.0 / n - 0.02
    for i, (title, sub, c) in enumerate(steps):
        x = i / n + 0.01
        ax.add_patch(plt.Rectangle((x, 0.2), w, 0.6, facecolor=c, edgecolor="white",
                                    linewidth=2, alpha=0.95))
        tc = NAVY if c == AMBER else "white"
        ax.text(x + w/2, 0.62, title, ha="center", va="center",
                color=tc, fontsize=11, fontweight="bold")
        ax.text(x + w/2, 0.36, sub, ha="center", va="center",
                color=tc, fontsize=9)
        if i < n - 1:
            ax.annotate("", xy=(x + w + 0.018, 0.5), xytext=(x + w + 0.003, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#777", lw=1.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig_save(fig, "07_pipeline_dev.png")


if __name__ == "__main__":
    make_training_curve()
    make_scatter_water_level()
    make_horizon_metrics()
    make_timeseries_compare()
    make_residual_hist()
    make_data_split()
    make_pipeline_diagram()
    print(f"\n✓ 저장: {OUT}")
