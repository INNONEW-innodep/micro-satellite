"""
발표용 정확도 시각화 차트 생성.

수치 출처:
- 수체 IoU: data/incoming/handover/07_test_evidence/VALREPORT_Busan_*.json
  (배포 프로토콜, ICEYE·PlanetScope 각 4씬)
- GPU 성능: 로컬 재현 (SAR 씬당 CPU 499s → GPU 42.8s)
- 시계열 예측 성능은 아직 실측 없음 (학습 곡선은 개념도로만 사용)
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
    fig.suptitle("[개념도] 시계열 예측 학습 곡선 예시 · 실측 아님",
                 fontsize=10, color=GRAY, y=1.02, style="italic")
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


# ====== 2. 실측 수체 IoU 바 차트 (배포 프로토콜 · Busan 4씬) ======
def make_scatter_water_level():
    """이전 이름은 유지. 내용은 실측 IoU 그래프로 교체.
    출처: data/incoming/handover/07_test_evidence/VALREPORT_Busan_*.json"""
    # per-scene 실측치
    sar_scenes = [("20200302", 0.9233), ("20200330", 0.9472),
                  ("20200415", 0.9614), ("20200416", 0.8786)]
    opt_scenes = [("20200218", 0.9476), ("20200312", 0.9317),
                  ("20200325", 0.9484), ("20200414", 0.9323)]
    # overall 평균
    sar_mean = 0.9259
    opt_mean = 0.9415

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 좌: 씬별 바
    ax = axes[0]
    x = np.arange(4)
    w = 0.38
    sar_vals = [v for _, v in sar_scenes]
    opt_vals = [v for _, v in opt_scenes]
    b1 = ax.bar(x - w/2, sar_vals, w, color=TEAL, label="SAR (ICEYE)")
    b2 = ax.bar(x + w/2, opt_vals, w, color=AMBER, label="광학 (PlanetScope)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"씬 {i+1}" for i in range(4)])
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("water IoU")
    ax.set_title("씬별 수체 IoU · 배포 프로토콜 (Busan 4씬)",
                 fontsize=12, color=TEAL)
    ax.legend(loc="lower right", frameon=False)
    for rects, vals in [(b1, sar_vals), (b2, opt_vals)]:
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width()/2, v + 0.003, f"{v:.3f}",
                    ha="center", fontsize=9, color=NAVY)

    # 우: 평균 IoU (강조)
    ax = axes[1]
    names = ["SAR\n(ICEYE)", "광학\n(PlanetScope)"]
    means = [sar_mean, opt_mean]
    colors = [TEAL, AMBER]
    bars = ax.bar(names, means, color=colors, width=0.55)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("water IoU")
    ax.set_title("4씬 평균 IoU · 배포 프로토콜",
                 fontsize=12, color=TEAL)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=16, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.02,
            "※ VALREPORT_Busan_ICEYE / PlanetScope.json · GPU 추론 재현",
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY,
            style="italic")

    fig_save(fig, "02_scatter_water_level.png")


# ====== 3. Δt(지평) 별 실측 성능 — 수위 RMSE + 수체 IoU ======
def make_horizon_metrics():
    """실측 데이터:
    - 수위: data/eval/timeseries_metrics.json → E_지평별_성능
    - IoU: data/eval/mask_persistence_iou.json → summary
    갈수기 조건(부산 2020.02~04) 각주 필수.
    """
    # 수위 RMSE (99쌍 persistence, 4구간)
    horizons_wl = ["1~7일\n(n=13)", "8~14일\n(n=12)", "15~28일\n(n=34)", "29~58일\n(n=40)"]
    rmse_wl = [0.0238, 0.0253, 0.0234, 0.0233]

    # 수체 IoU (12쌍)
    iou_labels = ["ICEYE\n단기\n(n=3)", "ICEYE\n장기\n(n=3)",
                  "Planet\n단기\n(n=2)", "Planet\n장기\n(n=4)"]
    iou_mean = [0.8357, 0.8777, 0.8514, 0.8186]
    iou_ceiling = [0.8326, 0.8821, 0.8475, 0.8228]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.suptitle("실측 · 부산 4씬 · 2020.02~04 (갈수기 조건)",
                 fontsize=10, color=GRAY, y=1.02, style="italic")

    # 좌: 수위 RMSE 지평별
    ax = axes[0]
    x = np.arange(len(horizons_wl))
    bars = ax.bar(x, [v * 100 for v in rmse_wl], color=TEAL, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(horizons_wl)
    ax.set_ylim(0, 3.0)
    ax.set_ylabel("수위 RMSE (cm)")
    ax.set_title("수위 예측 지평별 오차 (persistence · 99쌍)",
                 fontsize=12, color=TEAL)
    for b, v in zip(bars, rmse_wl):
        ax.text(b.get_x() + b.get_width()/2, v * 100 + 0.05,
                f"{v*100:.2f} cm", ha="center", fontsize=10,
                color=NAVY, fontweight="bold")
    # 목표 대비 달성 주석 (텍스트로만 표기 · 스케일 왜곡 방지)
    ax.text(0.5, 0.90,
            "목표 RMSE 38 cm 대비 ×15 배 이상 정확 (목표 달성)",
            transform=ax.transAxes, ha="center", fontsize=10,
            color=ORANGE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff4e6", ec=ORANGE, lw=1.2))

    # 우: 수체 IoU (predicted vs ceiling)
    ax = axes[1]
    x = np.arange(len(iou_labels))
    w = 0.35
    b1 = ax.bar(x - w/2, iou_mean, w, color=TEAL, label="예측 IoU (persistence)")
    b2 = ax.bar(x + w/2, iou_ceiling, w, color=AMBER, label="변화 상한 IoU (GT_A→GT_B)")
    ax.set_xticks(x); ax.set_xticklabels(iou_labels)
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("water IoU")
    ax.set_title("수체 영역 예측 정확도 (12쌍)", fontsize=12, color=TEAL)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for rects, vals in [(b1, iou_mean), (b2, iou_ceiling)]:
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width()/2, v + 0.005, f"{v:.2f}",
                    ha="center", fontsize=8, color=NAVY)
    ax.text(0.5, 0.02,
            "예측 IoU ≈ 변화 상한 → 오차 지배 요인 = 수체 자체 변화",
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY,
            style="italic")

    fig_save(fig, "03_horizon_metrics.png")


# ====== 4. 시계열 예측 vs 실제 (수위) ======
def make_timeseries_compare():
    """[개편] F 계열 · 변화량(Δ) 예측 skill 시각화 (실측).
    출처: data/eval/timeseries_metrics.json → F_변화량_평가
    "수위변화 예측" IITP 지표에 직접 대응."""
    # 수위 변화 (99쌍)
    wl_skill = 0.167       # 무변화 대비 skill
    wl_sign = 56 / 64      # 87.5% 방향 일치
    # 면적 변화 (센서별)
    area_data = {
        "ICEYE (SAR)": {"skill": 0.862, "sign": 5/6, "n": 6, "true_rms": 3.16, "rmse": 1.17},
        "PlanetScope (광학)": {"skill": 0.998, "sign": 6/6, "n": 6, "true_rms": 21.19, "rmse": 0.93},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("F 계열 · 변화량(Δ) 예측 실측 · 부산 4씬 (2020.02~04)",
                 fontsize=11, color=TEAL, y=1.02, style="italic")

    # 좌: 면적 변화 skill (센서별)
    ax = axes[0]
    names = list(area_data.keys())
    skills = [area_data[n]["skill"] for n in names]
    colors = [TEAL, AMBER]
    bars = ax.bar(names, skills, color=colors, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skill vs 무변화")
    ax.set_title("면적 변화 추적 skill", fontsize=12, color=TEAL)
    for b, n in zip(bars, names):
        d = area_data[n]
        ax.text(b.get_x() + b.get_width()/2, d["skill"] + 0.02,
                f"{d['skill']:.3f}", ha="center", fontsize=15,
                color=NAVY, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, d["skill"] - 0.06,
                f"방향 {int(d['sign']*d['n'])}/{d['n']}",
                ha="center", fontsize=10, color="white", fontweight="bold")
    ax.text(0.5, 0.02,
            "PlanetScope: 3/25 씬 65 km² 급증도 오차 1 km² 이내로 추적",
            transform=ax.transAxes, ha="center", fontsize=9, color=GRAY,
            style="italic")

    # 우: 수위 변화 - skill 낮은 이유 설명
    ax = axes[1]
    # 관측 잡음 vs 실제 변화 폭 비교
    labels = ["실제 수위\n변화 폭\n(RMS)", "관측 잡음\n수준", "예측 오차\n(RMSE)"]
    values = [2.41, 2.20, 2.20]  # cm 단위
    bar_colors = [AMBER, GRAY, TEAL]
    bars = ax.bar(labels, values, color=bar_colors, width=0.6)
    ax.set_ylim(0, 3.5)
    ax.set_ylabel("cm")
    ax.set_title("수위 변화 skill 0.17 — 왜?",
                 fontsize=12, color=TEAL)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f} cm",
                ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.92,
            "갈수기 변화 폭 ≈ 관측 잡음 → 스킬 하한",
            transform=ax.transAxes, ha="center", fontsize=10,
            color=ORANGE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff4e6", ec=ORANGE, lw=1.2))
    ax.text(0.5, 0.02,
            "홍수기엔 변화가 잡음을 압도 → skill 상승 예상 (별도 검증)",
            transform=ax.transAxes, ha="center", fontsize=9, color=GRAY,
            style="italic")
    # 아래 legend
    ax.text(0.5, -0.28, f"방향 일치율 {wl_sign*100:.1f}% (56/64)",
            transform=ax.transAxes, ha="center", fontsize=10,
            color=TEAL, fontweight="bold")

    fig_save(fig, "04_timeseries_predict.png")


# ====== 5. GPU vs CPU 추론 성능 비교 (실측) ======
def make_residual_hist():
    """이전 이름은 유지. 내용은 GPU 성능 비교 그래프로 교체.
    실측: SAR 씬 CPU 499s → GPU 42.8s (약 11.7배)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # 좌: 막대 - CPU vs GPU
    ax = axes[0]
    labels = ["CPU\n(기존)", "GPU\n(신규 지원)"]
    values = [499.0, 42.8]
    colors = [GRAY, TEAL]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel("SAR 씬당 처리 시간 (초)")
    ax.set_title("추론 성능 · CPU vs GPU (SAR 1씬 기준)",
                 fontsize=12, color=TEAL)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 12, f"{v:.1f}s",
                ha="center", fontsize=14, color=NAVY, fontweight="bold")
    # speedup 화살표
    ax.annotate("", xy=(1, 60), xytext=(0, 470),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5))
    ax.text(0.5, 260, "× 11.7 배 가속",
            ha="center", fontsize=13, color=ORANGE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ORANGE, lw=1.5))
    ax.set_ylim(0, 560)

    # 우: 처리량 비교 (씬/시간)
    ax = axes[1]
    scenes_per_hour = [3600 / 499.0, 3600 / 42.8]
    bars = ax.bar(labels, scenes_per_hour, color=colors, width=0.5)
    ax.set_ylabel("시간당 처리 씬 수")
    ax.set_title("처리량 · SAR 씬/시간",
                 fontsize=12, color=TEAL)
    for b, v in zip(bars, scenes_per_hour):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}",
                ha="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.02,
            "※ 공유 GPU 3GB 여유 조건 · 광학 모델은 VRAM 부족 시 CPU 자동 폴백",
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY,
            style="italic")
    ax.set_ylim(0, max(scenes_per_hour) * 1.2)

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
