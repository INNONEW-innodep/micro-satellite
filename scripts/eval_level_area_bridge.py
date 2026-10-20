#!/usr/bin/env python3
"""수위→면적 다리 실증 — "면적 시계열을 수위 시계열 예측에 종속시키면 어떻게 되나"를 현재 자료로 끝까지 돌린다.

두 지역(금강 대청댐 PlanetScope 10장면, 낙동강 하구 ICEYE 4 + PlanetScope 4장면)의 공통 격자 면적 계열과
같은 날 지상 수위, 그리고 학습된 지점 수위 예측기(in-process)로 다음을 계산한다.

  A. 같은 날 다리(검량): 면적 = a + b·수위 — 피어슨 r, 기울기, LOO RMSE (계열 평균·직전값과 비교)
  B. 인접 장면쌍 전망: 원점 장면 i → 다음 장면 j (같은 배치, Δ ≤ 30일)
       직전값 유지            A_i
       평균회귀(공식 지표)     A_i + 0.5·(평균_{≤i} − A_i)
       수위 다리              A_i + b·(ĥ_j − h_i)          ĥ_j = 원점 i 에서 Δ일 수위 전망
       평균회귀 + 수위 다리    MR + b·(ĥ_j − h_i)
       (각각 '완전 수위 예보' 변형: ĥ_j 대신 실측 h_j — 다리 오차와 수위 예측 오차를 분리)
     b 는 ① DEM 상부 곡선(대청 0.47 km²/m) / 제방 하도 0(부산) = 매니페스트 값, ② 위성 적합(LOO) 두 가지.
     지표: RMSE(km²), 정규화 MSE(= MSE / 관측범위², 공식 지표 정의), 직전값 대비 skill.

출력: docs/presentation/data/level_area_bridge.json, docs/presentation/수위면적다리_실증_2026-10-15.md,
      docs/presentation/figures/level_area_*.png
실행: .venv/bin/python scripts/eval_level_area_bridge.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from datetime import date as Date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.main import create_app  # noqa: E402
from backend.app.adapters.level_area_bridge import fit_slope  # noqa: E402

OUT_JSON = ROOT / "docs/presentation/data/level_area_bridge.json"
OUT_MD = ROOT / "docs/presentation/수위면적다리_실증_2026-10-15.md"
FIG_DIR = ROOT / "docs/presentation/figures"
PHI = 0.5
MAX_H = 30

# 한글 글꼴 (있으면)
for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK SC", "Malgun Gothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False


def rmse(errs):
    return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan")


def build_service():
    from backend.tests.test_gauge import PLUGIN_SPECS  # 서버 compose 와 같은 예측기 등록

    tmp = Path(tempfile.mkdtemp(prefix="bridge_eval_"))
    app = create_app(Settings(result_dir=tmp / "r", regions_dir=ROOT / "regions", regions_user_dir=tmp / "u", gauge_forecaster_plugins=PLUGIN_SPECS))
    return app.state.gauge_service


TARGETS = [
    # region, station, sensors, batch key(앞 4글자 = 연도), 계열 변형
    ("geum_daecheong", "대청댐", ["PlanetScope"]),
    ("nakdong_busan", "구포대교", ["ICEYE", "PlanetScope"]),
]
VARIANTS = [("area_km2_common_frame", "공통 격자(공식)"), ("area_km2_common_valid_footprint", "공통 유효범위(교집합)")]


def main() -> int:
    service = build_service()
    results: dict = {"generated_for": "2026-10-15", "regions": {}}
    md: list[str] = ["# 수위→면적 다리 실증 — 현재 자료로 돌린 결과 (2026-10-15)", "",
                     "생성: `scripts/eval_level_area_bridge.py` · 면적 `regions/<id>/satellite/area_series.json` · 수위 전망 = 지역 레지스트리 학습 모델(in-process) · "
                     "수치는 전부 계산값(합성 없음).", ""]
    for region_id, station, sensors in TARGETS:
        record = service.regions.get(region_id)
        manifest_slope = float(((record.manifest.get("satellite") or {}).get("level_area") or {}).get("slope_km2_per_m") or 0.0)
        area_payload = record.area_series() or {}
        region_out: dict = {"station": station, "manifest_slope_km2_per_m": manifest_slope, "variants": {}}
        md.append(f"## {record.name} (`{region_id}`) — 지점 {station}, 매니페스트 기울기 {manifest_slope:.2f} km²/m")
        md.append("")
        for key, label in VARIANTS:
            rows_all = []
            for sensor in sensors:
                block = (area_payload.get("sensors") or {}).get(sensor) or {}
                for r in sorted(block.get("rows") or [], key=lambda r: r["date"]):
                    if r.get(key) is None or (r.get("gauge_levels") or {}).get(station) is None:
                        continue
                    rows_all.append({"sensor": sensor, "date": r["date"], "area": float(r[key]), "level": float(r["gauge_levels"][station])})
            if len(rows_all) < 3:
                continue
            # ── A. 같은 날 다리 (전체 / 배치별)
            same_day = {}
            groups = {"전체": rows_all}
            years = sorted({r["date"][:4] for r in rows_all})
            if len(years) > 1:
                for y in years:
                    groups[f"{y}만"] = [r for r in rows_all if r["date"].startswith(y)]
            for gname, grp in groups.items():
                if len(grp) < 3:
                    continue
                lv = [r["level"] for r in grp]; ar = [r["area"] for r in grp]
                slope, corr, n = fit_slope(lv, ar)
                loo = []
                for i in range(len(grp)):
                    xs = [v for j, v in enumerate(lv) if j != i]; ys = [v for j, v in enumerate(ar) if j != i]
                    b, _c, _n = fit_slope(xs, ys)
                    if b is None:
                        continue
                    a0 = float(np.mean(ys)) - b * float(np.mean(xs))
                    loo.append(a0 + b * lv[i] - ar[i])
                loo_mean = [float(np.mean([v for j, v in enumerate(ar) if j != i])) - ar[i] for i in range(len(grp))]
                same_day[gname] = {
                    "n": n, "level_range_m": round(max(lv) - min(lv), 3), "area_range_km2": round(max(ar) - min(ar), 3),
                    "r": None if corr is None else round(corr, 3), "slope_km2_per_m": None if slope is None else round(slope, 3),
                    "loo_rmse_linear": round(rmse(loo), 3) if loo else None, "loo_rmse_mean": round(rmse(loo_mean), 3),
                }
            # ── B. 인접 장면쌍 전망 (같은 센서, 같은 배치, Δ ≤ 30일)
            pairs = []
            by_sensor: dict[str, list] = {}
            for r in rows_all:
                by_sensor.setdefault(r["sensor"], []).append(r)
            obs_all = [r["area"] for r in rows_all]
            span = max(obs_all) - min(obs_all)
            for sensor, rs in by_sensor.items():
                for i in range(len(rs) - 1):
                    o, t = rs[i], rs[i + 1]
                    d0, d1 = Date.fromisoformat(o["date"]), Date.fromisoformat(t["date"])
                    delta = (d1 - d0).days
                    if delta < 1 or delta > MAX_H or o["date"][:4] != t["date"][:4]:
                        continue
                    hist = [x["area"] for x in rs[: i + 1]]
                    mr = o["area"] + PHI * (float(np.mean(hist)) - o["area"])
                    fc = service.forecast(region_id, station, as_of=d0, horizon=delta, history_days=7)
                    step = next((s for s in fc["steps"] if int(s["horizon"]) == delta), None)
                    h_hat = step["forecast_m"] if step else None
                    # LOO 위성 적합 기울기: 목표 장면을 뺀 같은 날 쌍으로 (전체 / 같은 배치만)
                    others = [x for x in rows_all if x["date"] != t["date"]]
                    b_fit, _c, _n = fit_slope([x["level"] for x in others], [x["area"] for x in others])
                    b_fit = max(0.0, b_fit) if b_fit is not None else 0.0
                    same_batch = [x for x in others if x["date"][:4] == t["date"][:4]]
                    b_batch, _c2, _n2 = fit_slope([x["level"] for x in same_batch], [x["area"] for x in same_batch])
                    b_batch = max(0.0, b_batch) if b_batch is not None else 0.0
                    row = {
                        "sensor": sensor, "from": o["date"], "to": t["date"], "delta_days": delta,
                        "level_from": o["level"], "level_to": t["level"], "level_forecast": h_hat,
                        "area_from": o["area"], "area_to": t["area"],
                        "pred": {
                            "persistence": o["area"], "mean_reversion": mr,
                        },
                        "b_fit_loo": round(b_fit, 3), "b_batch_loo": round(b_batch, 3), "b_manifest": manifest_slope,
                    }
                    for bname, b in (("manifest", manifest_slope), ("fit", b_fit), ("batch", b_batch)):
                        if h_hat is not None:
                            row["pred"][f"bridge_{bname}"] = o["area"] + b * (h_hat - o["level"])
                            row["pred"][f"mr_bridge_{bname}"] = mr + b * (h_hat - o["level"])
                        row["pred"][f"bridge_{bname}_perfect"] = o["area"] + b * (t["level"] - o["level"])
                        row["pred"][f"mr_bridge_{bname}_perfect"] = mr + b * (t["level"] - o["level"])
                    pairs.append(row)
            metrics = {}
            model_ids = sorted({m for p in pairs for m in p["pred"]})
            for m in model_ids:
                errs = [p["pred"][m] - p["area_to"] for p in pairs if m in p["pred"]]
                mse = sum(e * e for e in errs) / len(errs)
                mse_p = sum((p["pred"]["persistence"] - p["area_to"]) ** 2 for p in pairs if m in p["pred"]) / len(errs)
                metrics[m] = {"n": len(errs), "rmse_km2": round(math.sqrt(mse), 3), "norm_mse": round(mse / span ** 2, 5) if span > 0 else None,
                              "skill_vs_persistence": round(1 - mse / mse_p, 3) if mse_p > 0 else None}
            region_out["variants"][key] = {"label": label, "n_scenes": len(rows_all), "span_km2": round(span, 3), "same_day": same_day, "pairs": pairs, "metrics": metrics}
            # ── 마크다운
            md.append(f"### 계열: {label} — 장면 {len(rows_all)}개, 관측 범위 {span:.2f} km²")
            md.append("")
            md.append("**A. 같은 날 다리(검량)** — 면적 = a + b·수위")
            md.append("")
            md.append("| 표본 | n | 수위폭 (m) | 면적폭 (km²) | r | 기울기 (km²/m) | LOO RMSE 선형 | LOO RMSE 평균 |")
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for g, v in same_day.items():
                md.append(f"| {g} | {v['n']} | {v['level_range_m']:.3f} | {v['area_range_km2']:.2f} | {v['r'] if v['r'] is not None else '—'} | {v['slope_km2_per_m'] if v['slope_km2_per_m'] is not None else '—'} | {v['loo_rmse_linear'] if v['loo_rmse_linear'] is not None else '—'} | {v['loo_rmse_mean']} |")
            md.append("")
            md.append(f"**B. 인접 장면쌍 전망** — {len(pairs)}쌍 (같은 센서·같은 배치, Δ ≤ {MAX_H}일), 수위 전망은 원점 날짜에서 Δ일 학습 모델 전망")
            md.append("")
            md.append("| 모델 | n | RMSE (km²) | 정규화 MSE | 직전값 대비 skill |")
            md.append("|---|---:|---:|---:|---:|")
            order = ["persistence", "mean_reversion", "bridge_manifest", "mr_bridge_manifest", "bridge_batch", "mr_bridge_batch", "bridge_fit", "mr_bridge_fit",
                     "bridge_manifest_perfect", "mr_bridge_manifest_perfect", "bridge_batch_perfect", "mr_bridge_batch_perfect", "bridge_fit_perfect", "mr_bridge_fit_perfect"]
            names = {"persistence": "직전값 유지", "mean_reversion": "평균회귀 (공식 면적 모델)",
                     "bridge_manifest": f"수위 다리 (기울기 {manifest_slope:.2f}, 매니페스트)", "mr_bridge_manifest": f"평균회귀 + 수위 다리 (기울기 {manifest_slope:.2f})",
                     "bridge_batch": "수위 다리 (기울기 같은 배치 위성 LOO 적합)", "mr_bridge_batch": "평균회귀 + 수위 다리 (같은 배치 LOO 적합)",
                     "bridge_fit": "수위 다리 (기울기 전체 위성 LOO 적합)", "mr_bridge_fit": "평균회귀 + 수위 다리 (전체 LOO 적합)",
                     "bridge_manifest_perfect": "수위 다리 (매니페스트) · 완전 수위 예보", "mr_bridge_manifest_perfect": "평균회귀 + 다리 (매니페스트) · 완전 수위 예보",
                     "bridge_batch_perfect": "수위 다리 (같은 배치 적합) · 완전 수위 예보", "mr_bridge_batch_perfect": "평균회귀 + 다리 (같은 배치 적합) · 완전 수위 예보",
                     "bridge_fit_perfect": "수위 다리 (전체 적합) · 완전 수위 예보", "mr_bridge_fit_perfect": "평균회귀 + 다리 (전체 적합) · 완전 수위 예보"}
            for m in order:
                if m in metrics:
                    v = metrics[m]
                    md.append(f"| {names[m]} | {v['n']} | {v['rmse_km2']:.3f} | {v['norm_mse']} | {v['skill_vs_persistence']:+.3f} |")
            md.append("")
            md.append("쌍별 (실측 면적 → 다음 장면 면적, 수위 원점 → 실측/전망):")
            md.append("")
            md.append("| 센서 | 원점 → 목표 | Δ일 | 수위 원점 → 목표(실측 / 전망) | 면적 원점 → 목표 | 직전값 | 평균회귀 | 다리(매니페스트) | 평균회귀+다리 |")
            md.append("|---|---|---:|---|---|---:|---:|---:|---:|")
            for p in pairs:
                hf = f"{p['level_forecast']:.2f}" if p["level_forecast"] is not None else "—"
                md.append(f"| {p['sensor']} | {p['from']} → {p['to']} | {p['delta_days']} | {p['level_from']:.2f} → {p['level_to']:.2f} / {hf} | {p['area_from']:.2f} → {p['area_to']:.2f} | {p['pred']['persistence']:.2f} | {p['pred']['mean_reversion']:.2f} | {p['pred'].get('bridge_manifest', float('nan')):.2f} | {p['pred'].get('mr_bridge_manifest', float('nan')):.2f} |")
            md.append("")
        results["regions"][region_id] = region_out

    # ── 그림 1: 대청 같은 날 수위 vs 면적 (공통 격자 · 유효범위), 배치별 색, 기울기 선
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dcd = results["regions"]["geum_daecheong"]["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, v) in zip(axes, dcd.items()):
        pts = [(r["level"], r["area"], r["date"][:4]) for r in _rows_for(service, "geum_daecheong", "대청댐", ["PlanetScope"], key)]
        for yr, color in (("2019", "#E07A2F"), ("2022", "#1F7A8C")):
            xs = [p[0] for p in pts if p[2] == yr]; ys = [p[1] for p in pts if p[2] == yr]
            ax.scatter(xs, ys, s=46, color=color, label=f"{yr} 장면 ({len(xs)})", zorder=3)
        sd = v["same_day"]
        if "2022만" in sd and sd["2022만"]["slope_km2_per_m"] is not None:
            x22 = [p[0] for p in pts if p[2] == "2022"]; y22 = [p[1] for p in pts if p[2] == "2022"]
            b = sd["2022만"]["slope_km2_per_m"]; a0 = float(np.mean(y22)) - b * float(np.mean(x22))
            xx = np.linspace(min(x22) - 0.2, max(x22) + 0.2, 10)
            ax.plot(xx, a0 + b * xx, color="#1F7A8C", lw=1.4, ls="--", label=f"2022 적합 {b:+.2f} km²/m (r={sd['2022만']['r']:+.2f})")
        m = results["regions"]["geum_daecheong"]["manifest_slope_km2_per_m"]
        x22 = [p[0] for p in pts if p[2] == "2022"]; y22 = [p[1] for p in pts if p[2] == "2022"]
        if x22:
            xx = np.linspace(70.0, 75.0, 10)
            ax.plot(xx, float(np.mean(y22)) + m * (xx - float(np.mean(x22))), color="#4E9F6E", lw=1.4, ls=":", label=f"DEM 상부 기울기 {m:.2f} km²/m")
        ax.set_title(v["label"], fontsize=11); ax.set_xlabel("대청댐 실측 수위 (EL.m)"); ax.set_ylabel("탐지 면적 (km²)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("금강 대청댐 — 같은 날 수위 vs 위성 탐지 면적 (PlanetScope 10장면)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "level_area_bridge_daecheong_scatter.png", dpi=160); plt.close(fig)

    # ── 그림 2: 인접 쌍 예측 vs 실측 (대청 공통 격자, 부산 공통 격자)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (region_id, title) in zip(axes, (("geum_daecheong", "금강 대청댐 (PlanetScope, 같은 배치 쌍)"), ("nakdong_busan", "낙동강 하구 (ICEYE·PlanetScope)"))):
        v = results["regions"][region_id]["variants"]["area_km2_common_frame"]
        pairs = v["pairs"]; n = len(pairs); x = np.arange(n); w = 0.2
        ax.bar(x - 1.5 * w, [p["area_to"] for p in pairs], w, color="#16202C", label="실측(다음 장면)")
        ax.bar(x - 0.5 * w, [p["pred"]["persistence"] for p in pairs], w, color="#E07A2F", label="직전값 유지")
        ax.bar(x + 0.5 * w, [p["pred"]["mean_reversion"] for p in pairs], w, color="#C9A227", label="평균회귀(공식)")
        ax.bar(x + 1.5 * w, [p["pred"].get("mr_bridge_manifest", np.nan) for p in pairs], w, color="#1F7A8C", label="평균회귀+수위 다리")
        ax.set_xticks(x); ax.set_xticklabels([f"{p['sensor'][:3]} {p['from'][5:]}→{p['to'][5:]}" for p in pairs], rotation=35, ha="right", fontsize=8)
        lo = min(min(p["area_to"] for p in pairs), min(p["pred"]["persistence"] for p in pairs)); ax.set_ylim(max(0, lo - 3), None)
        ax.set_title(title, fontsize=11); ax.set_ylabel("km²"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("인접 장면쌍 면적 예측 vs 실측 — 공통 격자 계열", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "level_area_bridge_pairs.png", dpi=160); plt.close(fig)

    # ── 그림 3: 2024년 대청댐 D+7 수위 백테스트 → 파생 면적 시계열 (수위 시계열에 종속된 면적 시계열의 모양)
    deck = ROOT / "docs/presentation/data/timeseries_deck_data.json"
    if deck.is_file():
        bt = json.loads(deck.read_text(encoding="utf-8")).get("backtest_dcd_7")
        if bt:
            m = results["regions"]["geum_daecheong"]["manifest_slope_km2_per_m"]
            rows22 = [r for r in _rows_for(service, "geum_daecheong", "대청댐", ["PlanetScope"], "area_km2_common_frame") if r["date"].startswith("2022")]
            h_ref = float(np.mean([r["level"] for r in rows22])); a_ref = float(np.mean([r["area"] for r in rows22]))
            dates = bt["dates"]; truth = np.array([np.nan if v is None else v for v in bt["truth"]], dtype=float); pred = np.array([np.nan if v is None else v for v in bt["pred"]], dtype=float)
            a_truth = a_ref + m * (truth - h_ref); a_pred = a_ref + m * (pred - h_ref)
            fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
            x = np.arange(len(dates))
            axes[0].plot(x, truth, color="#16202C", lw=1.4, label="실측 수위"); axes[0].plot(x, pred, color="#1F7A8C", lw=1.2, label="D+7 수위 전망 (GRU)")
            axes[0].set_ylabel("대청댐 수위 (EL.m)"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=9)
            axes[1].plot(x, a_truth, color="#16202C", lw=1.4, label="실측 수위로 환산한 면적")
            axes[1].plot(x, a_pred, color="#1F7A8C", lw=1.2, label="D+7 수위 전망으로 환산한 면적")
            for r in rows22:
                pass
            axes[1].axhline(a_ref, color="#E07A2F", lw=1, ls=":", label=f"2022-05 위성 탐지 평균 {a_ref:.1f} km² (기준점, 수위 {h_ref:.2f} m)")
            axes[1].set_ylabel("파생 면적 (km², 공통 격자)"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=9)
            ticks = [i for i, d in enumerate(dates) if d.endswith("-01")]
            axes[1].set_xticks(ticks); axes[1].set_xticklabels([dates[i][5:] for i in ticks])
            err = a_pred - a_truth; err = err[~np.isnan(err)]
            fig.suptitle(f"대청댐 2024 — 수위 시계열에 종속된 면적 시계열 (기울기 {m:.2f} km²/m, DEM 상부 곡선) · D+7 파생 면적 RMSE {math.sqrt(float(np.mean(err ** 2))):.2f} km²", fontsize=11)
            fig.tight_layout(); fig.savefig(FIG_DIR / "level_area_bridge_daecheong_2024_derived.png", dpi=160); plt.close(fig)
            results["derived_2024"] = {"slope_km2_per_m": m, "h_ref": h_ref, "a_ref_km2": a_ref, "d7_area_rmse_km2": round(math.sqrt(float(np.mean(err ** 2))), 3),
                                      "area_range_truth_km2": [round(float(np.nanmin(a_truth)), 2), round(float(np.nanmax(a_truth)), 2)]}
            md.append(f"## 파생 면적 시계열 (대청댐 2024, 그림 `figures/level_area_bridge_daecheong_2024_derived.png`)")
            md.append("")
            md.append(f"면적(t) = {a_ref:.2f} + {m:.2f} × (수위(t) − {h_ref:.2f}). 실측 수위로 환산한 2024 면적 범위 {float(np.nanmin(a_truth)):.1f}~{float(np.nanmax(a_truth)):.1f} km², "
                      f"D+7 수위 전망으로 환산한 면적의 RMSE {math.sqrt(float(np.mean(err ** 2))):.2f} km² (= 기울기 × 수위 D+7 RMSE 1.19 m). 위성 검증 없음 — 수위 시계열의 모양이 면적으로 옮겨지는 것을 보이는 그림.")
            md.append("")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    md.append("그림: `figures/level_area_bridge_daecheong_scatter.png`, `figures/level_area_bridge_pairs.png`, `figures/level_area_bridge_daecheong_2024_derived.png`")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("saved", OUT_JSON, OUT_MD)
    return 0


def _rows_for(service, region_id, station, sensors, key):
    record = service.regions.get(region_id)
    payload = record.area_series() or {}
    out = []
    for sensor in sensors:
        for r in sorted(((payload.get("sensors") or {}).get(sensor) or {}).get("rows") or [], key=lambda r: r["date"]):
            if r.get(key) is not None and (r.get("gauge_levels") or {}).get(station) is not None:
                out.append({"sensor": sensor, "date": r["date"], "area": float(r[key]), "level": float(r["gauge_levels"][station])})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
