#!/usr/bin/env python3
"""댐 방류 입력 실험 — 대청댐 본댐·조정지댐·방수로 수위 D+1(·3·7) 전망에 댐 운영 자료를 넣으면 얼마나 좋아지나.

가정(사용자 요청): 총방류·여수로 방류는 '하루 전 예고'로 다음 날 값을 안다(D+1). D+3·D+7 은 예고된 방류가 유지된다고 본다.
유입량은 미리 알 수 없으므로 원점 값만 쓴다(참고로 '완전 유입 예보' 상한도 병기).

자료: regions/geum_daecheong/gauge/series.json(WAMIS 일 수위), weather/asos_daily.json(대전 ASOS),
      data/wamis/dam_3008110_daily_2019_2024.json(WAMIS 댐 운영 일자료, 무인증 API). 합성값 없음.
분할: 지역 학습 규칙과 같음 — 학습 ≤2020, 검증 2021~2023(릿지 λ 선택), 시험 2024.
모델: 모두 릿지 회귀(표준화 특징, 목표 = 원점 대비 변화량). 공식 모델과 다른 구현이라 '같은 표본의 직전값 유지 대비 개선율'로 비교한다.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAUGE = ROOT / "regions" / "geum_daecheong" / "gauge" / "series.json"
WEATHER = ROOT / "regions" / "geum_daecheong" / "weather" / "asos_daily.json"
DAM = ROOT / "data" / "wamis" / "dam_3008110_daily_2019_2024.json"
OUT_JSON = ROOT / "docs" / "presentation" / "data" / "dam_discharge_d1_experiment.json"
STATIONS = ["대청댐", "대청조정지댐", "대청댐방수로"]
SPLIT = {"train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}
HORIZONS = (1, 3, 7)
P_LAGS = 7
LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def load() -> tuple[list[Date], dict[str, np.ndarray], dict[str, np.ndarray]]:
    g = json.loads(GAUGE.read_text(encoding="utf-8"))["series"]
    dates = [Date.fromisoformat(r["date"]) for r in g["대청댐"]]
    idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    lv = {}
    for st in STATIONS:
        arr = np.full(n, np.nan)
        for r in g[st]:
            d = Date.fromisoformat(r["date"])
            if d in idx and r.get("water_level_m") is not None:
                arr[idx[d]] = float(r["water_level_m"])
        lv[st] = arr
    rows = json.loads(WEATHER.read_text(encoding="utf-8"))["rows"]
    rain = np.full(n, np.nan)
    for r in rows:
        d = Date.fromisoformat(r["date"])
        if d in idx and r.get("precipitation_mm") is not None:
            rain[idx[d]] = float(r["precipitation_mm"])
    dam = json.loads(DAM.read_text(encoding="utf-8"))["rows"]
    ex = {k: np.full(n, np.nan) for k in ("tdqty", "spdqty", "edqty", "iqty", "rwl", "rf")}
    for r in dam:
        s = r["obsymd"]; d = Date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        if d not in idx:
            continue
        for k in ex:
            v = r.get(k)
            try:
                ex[k][idx[d]] = float(v) if v not in (None, "", " ") else np.nan
            except ValueError:
                pass
    ex["rain"] = rain
    return dates, lv, ex


def roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    c = np.nancumsum(np.nan_to_num(x))
    for i in range(len(x)):
        lo = i - w + 1
        if lo < 0:
            continue
        out[i] = c[i] - (c[lo - 1] if lo > 0 else 0.0)
    return out


def features(y: np.ndarray, ex: dict[str, np.ndarray], t: np.ndarray, h: int, variant: str, dam_level: np.ndarray) -> np.ndarray:
    """원점 t 에서의 특징 행렬. variant: ar | arx | dam_obs | dam_d1 | dam_d1_inflow."""
    cols = []
    for i in range(P_LAGS):  # 최근 수위 변화 p 개
        cols.append(y[t - i] - y[t - i - 1])
    cols.append(y[t] - np.nanmean(y[np.maximum(t - 30, 0)]))  # 30일 전 대비
    doy = np.array([d.timetuple().tm_yday for d in t_dates(t)])
    cols.append(np.sin(2 * np.pi * doy / 365.25)); cols.append(np.cos(2 * np.pi * doy / 365.25))
    if variant in ("arx", "dam_obs", "dam_d1", "dam_d1_inflow"):
        r1, r3, r7, r30 = (roll_sum(ex["rain"], w) for w in (1, 3, 7, 30))
        cols += [r1[t], r3[t], r7[t], r30[t]]
    if variant in ("dam_obs", "dam_d1", "dam_d1_inflow"):
        cols += [ex["tdqty"][t], ex["spdqty"][t], ex["iqty"][t], ex["tdqty"][t] - ex["tdqty"][t - 1], ex["iqty"][t] - ex["tdqty"][t], dam_level[t]]
    if variant in ("dam_d1", "dam_d1_inflow"):
        # '하루 전 예고'로 아는 다음 날 방류 — D+h 에서는 예고된 값이 유지된다고 본다(같은 값)
        cols += [ex["tdqty"][t + 1], ex["spdqty"][t + 1], ex["tdqty"][t + 1] - ex["tdqty"][t]]
    if variant == "dam_d1_inflow":  # 참고 상한: 다음 날 유입량까지 안다면
        cols += [ex["iqty"][t + 1], ex["iqty"][t + 1] - ex["tdqty"][t + 1]]
    return np.column_stack(cols)


_DATES: list[Date] = []


def t_dates(t: np.ndarray) -> list[Date]:
    return [_DATES[i] for i in t]


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    ym = y.mean()
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return w, mu, sd, ym


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    w, mu, sd, ym = model
    return ((X - mu) / sd) @ w + ym


def main() -> int:
    global _DATES
    dates, lv, ex = load()
    _DATES = dates
    n = len(dates)
    d_arr = np.array(dates)
    results: dict = {"split": SPLIT, "horizons": list(HORIZONS), "variants": {
        "persistence": "직전값 유지(원점 수위 그대로)",
        "ar": "릿지 AR: 최근 7일 수위 변화·30일 대비·계절",
        "arx": "AR + 과거 강수(1·3·7·30일 합)",
        "dam_obs": "ARX + 원점까지의 댐 운영 실측(총방류·여수로·유입·방류 변화·유입−방류·본댐 저수위)",
        "dam_d1": "dam_obs + 다음 날 총방류·여수로 방류(하루 전 예고 가정; D+3·7 은 예고값 유지)",
        "dam_d1_inflow": "dam_d1 + 다음 날 유입량(알 수 없는 값 — 참고 상한)",
    }, "stations": {}}
    for st in STATIONS:
        y = lv[st]
        results["stations"][st] = {}
        for h in HORIZONS:
            t_all = np.arange(P_LAGS + 31, n - h - 1)
            target = y[t_all + h] - y[t_all]
            per_station = {}
            for variant in ("ar", "arx", "dam_obs", "dam_d1", "dam_d1_inflow"):
                X = features(y, ex, t_all, h, variant, lv["대청댐"])
                ok = np.isfinite(X).all(axis=1) & np.isfinite(target) & np.isfinite(y[t_all])
                tt, XX, yy = t_all[ok], X[ok], target[ok]
                dd = d_arr[tt]
                tr = dd <= Date.fromisoformat(SPLIT["train_end"])
                va = (dd >= Date.fromisoformat(SPLIT["val_start"])) & (dd <= Date.fromisoformat(SPLIT["val_end"]))
                te = dd >= Date.fromisoformat(SPLIT["test_start"])
                best = None
                for lam in LAMBDAS:
                    m = ridge_fit(XX[tr], yy[tr], lam)
                    mse = float(np.mean((ridge_predict(m, XX[va]) - yy[va]) ** 2))
                    if best is None or mse < best[0]:
                        best = (mse, lam)
                lam = best[1]
                m = ridge_fit(XX[tr | va], yy[tr | va], lam)  # 검증까지 포함해 재적합(지역 규칙과 동일), 시험 2024 는 미사용
                pred = ridge_predict(m, XX[te])
                err = pred - yy[te]
                pers = -yy[te]  # 직전값 유지의 오차 = 0 − 변화량
                rm, rp = math.sqrt(float(np.mean(err ** 2))), math.sqrt(float(np.mean(pers ** 2)))
                dte = dd[te]
                summer = np.array([Date(2024, 7, 1) <= d <= Date(2024, 9, 30) for d in dte])
                rm_s, rp_s = math.sqrt(float(np.mean(err[summer] ** 2))), math.sqrt(float(np.mean(pers[summer] ** 2)))
                per_station[variant] = {"n_test": int(te.sum()), "lambda": lam, "rmse_cm": round(rm * 100, 1), "mae_cm": round(float(np.mean(np.abs(err))) * 100, 1),
                                        "persistence_rmse_cm": round(rp * 100, 1), "skill_vs_persistence": round(1 - (rm / rp) ** 2, 3),
                                        "summer_jul_sep": {"n": int(summer.sum()), "rmse_cm": round(rm_s * 100, 1), "persistence_rmse_cm": round(rp_s * 100, 1), "skill_vs_persistence": round(1 - (rm_s / rp_s) ** 2, 3)},
                                        "test_dates": [dte[0].isoformat(), dte[-1].isoformat()],
                                        "daily": {d.isoformat(): {"origin_level_m": round(float(y[tt[te][i]]), 3), "true_change_m": round(float(yy[te][i]), 3), "pred_change_m": round(float(pred[i]), 3)}
                                                  for i, d in enumerate(dte) if Date(2024, 7, 6) <= d <= Date(2024, 7, 20)}}
            results["stations"][st][f"D+{h}"] = per_station
            row = " | ".join(f"{v} {per_station[v]['rmse_cm']} ({per_station[v]['skill_vs_persistence']*100:+.0f}%)" for v in per_station)
            print(f"{st} D+{h}: 직전값 {per_station['ar']['persistence_rmse_cm']} cm | {row}  [n={per_station['ar']['n_test']}]")
            srow = " | ".join(f"{v} {per_station[v]['summer_jul_sep']['rmse_cm']}" for v in per_station)
            print(f"   7~9월(n={per_station['ar']['summer_jul_sep']['n']}): 직전값 {per_station['ar']['summer_jul_sep']['persistence_rmse_cm']} cm | {srow}")
    # 2024 홍수기만 따로 (7~9월) — dam_d1 효과가 어디서 나오나
    results["note_ko"] = ("릿지 구현은 공식 예측기와 다르므로 절대값은 공식 성적표(evaluation.json)와 직접 비교하지 말고, 같은 표본 안에서 "
                          "변형 간 차이(특히 dam_obs → dam_d1)를 본다. 댐 자료 결측일은 모든 변형에서 같은 날을 뺐다.")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
