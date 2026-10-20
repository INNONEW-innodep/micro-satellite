#!/usr/bin/env python3
"""세 아이디어 실증(2026-10-16): ① 시 단위 수위 예측(실시간형) ② 기상 예보 입력이 도움 되는 구간(월·예측일수) ③ 오차 큰 달 전용 모델.

자료: regions/<id>/gauge/series.json(WAMIS 일 수위) · weather/asos_daily.json(ASOS 일 강수·기온) ·
      data/wamis/hourly/*_hourly_2022_2024.json(WAMIS 시 수위, 무인증 API). 합성값 없음.
모델: 전부 릿지 회귀(표준화 특징, 목표 = 원점 대비 변화량). 공식 예측기와 구현이 달라 변형 간 차이로 읽는다.
예보: 과거 기상청 예보 원문은 공개 API 로 받을 수 없어(단기예보는 최근분만) '실제 내린 비 = 완벽한 예보'로 둔 상한과
      '+3일만 아는 예보' 두 가지로 본다. 실제 예보는 이보다 못하므로 이득의 상한이다.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs" / "presentation" / "data" / "realtime_forecast_experiment_2026-10-16.json"
DAILY = [("geum_daecheong", "대청댐", {"train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2023-12-31"}),
         ("nakdong_busan", "삼랑진교", {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31"}),
         ("nakdong_busan", "호포대교", {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31"}),
         ("nakdong_busan", "구포대교", {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2023-12-31"})]
HOURLY = [("3008690_대청댐_hourly_2022_2024.json", "대청댐"), ("2022680_구포대교_hourly_2022_2024.json", "구포대교")]
H_DAILY = (1, 3, 7)
H_HOURLY = (1, 3, 6, 12, 24)
LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
FLOOD_MONTHS = {6, 7, 8, 9}


def ridge_fit(X, y, lam):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd; ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return w, mu, sd, ym


def ridge_predict(m, X):
    w, mu, sd, ym = m
    return ((X - mu) / sd) @ w + ym


def fit_select(X, y, tr, va):
    best = None
    for lam in LAMBDAS:
        m = ridge_fit(X[tr], y[tr], lam)
        mse = float(np.mean((ridge_predict(m, X[va]) - y[va]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam)
    return ridge_fit(X[tr | va], y[tr | va], best[1]), best[1]


def roll_sum(x, w):
    out = np.full_like(x, np.nan); c = np.nancumsum(np.nan_to_num(x))
    for i in range(len(x)):
        lo = i - w + 1
        if lo >= 0: out[i] = c[i] - (c[lo - 1] if lo > 0 else 0.0)
    return out


def rmse(e):
    e = np.asarray(e, dtype=float); e = e[np.isfinite(e)]
    return float(math.sqrt(np.mean(e ** 2))) if e.size else float("nan")


# ───────────────────────── ② ③ 일 단위
def load_daily(region, station):
    g = json.loads((ROOT / "regions" / region / "gauge" / "series.json").read_text(encoding="utf-8"))["series"][station]
    dates = [Date.fromisoformat(r["date"]) for r in g]; y = np.array([r.get("water_level_m") if r.get("water_level_m") is not None else np.nan for r in g], dtype=float)
    idx = {d: i for i, d in enumerate(dates)}
    w = json.loads((ROOT / "regions" / region / "weather" / "asos_daily.json").read_text(encoding="utf-8"))["rows"]
    rain = np.full(len(dates), np.nan)
    for r in w:
        d = Date.fromisoformat(r["date"])
        if d in idx and r.get("precipitation_mm") is not None: rain[idx[d]] = float(r["precipitation_mm"])
    return dates, y, rain


def daily_features(y, rain, dates, t, h, variant):
    cols = [y[t - i] - y[t - i - 1] for i in range(7)]
    cols.append(y[t] - np.nanmean(y[np.maximum(t - 30, 0)]))
    doy = np.array([dates[i].timetuple().tm_yday for i in t]); cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    if variant != "ar":
        for w in (1, 3, 7, 30): cols.append(np.log1p(np.maximum(roll_sum(rain, w)[t], 0)))
    if variant.startswith("future"):
        K = h if variant == "future_all" else min(3, h)
        fut = np.zeros(len(t)); pk = np.zeros(len(t))
        for k in range(1, K + 1):
            v = np.nan_to_num(rain[t + k]); fut += v; pk = np.maximum(pk, v)
        cols += [np.log1p(fut), np.log1p(pk)]
    return np.column_stack(cols)


def run_daily():
    out = {}
    for region, station, split in DAILY:
        dates, y, rain = load_daily(region, station); n = len(dates); d_arr = np.array(dates)
        res = {}
        for h in H_DAILY:
            t_all = np.arange(40, n - h - 1); target = y[t_all + h] - y[t_all]
            month = np.array([dates[i + h].month for i in t_all])
            per = {}
            for variant in ("ar", "arx", "future_3d", "future_all"):
                X = daily_features(y, rain, dates, t_all, h, variant)
                ok = np.isfinite(X).all(axis=1) & np.isfinite(target)
                tt, XX, yy, mm = t_all[ok], X[ok], target[ok], month[ok]; dd = d_arr[tt]
                tr = dd <= Date.fromisoformat(split["train_end"]); va = (dd >= Date.fromisoformat(split["val_start"])) & (dd <= Date.fromisoformat(split["val_end"])); te = dd >= Date(2024, 1, 1)
                m, lam = fit_select(XX, yy, tr, va); pred = ridge_predict(m, XX[te]); err = pred - yy[te]; pers = -yy[te]
                # ③ 홍수기 전용 분리 모델: 홍수기(6~9월) 원점과 비홍수기 원점을 따로 적합(같은 특징)
                flood = np.array([Date.fromtimestamp(0).replace(year=1) and (x.month in FLOOD_MONTHS) for x in dd])
                pred_split = np.full(te.sum(), np.nan)
                for grp in (True, False):
                    g_tr = tr & (flood == grp); g_va = va & (flood == grp); g_te = te & (flood == grp)
                    if g_tr.sum() > 50 and g_va.sum() > 20 and g_te.sum():
                        ms, _ = fit_select(XX, yy, g_tr, g_va); pred_split[(flood[te] == grp)] = ridge_predict(ms, XX[g_te])
                err_split = pred_split - yy[te]
                rows = {"all": {"rmse_cm": round(rmse(err) * 100, 1), "persistence_rmse_cm": round(rmse(pers) * 100, 1), "split_rmse_cm": round(rmse(err_split) * 100, 1)}, "by_month": {}}
                for mo in range(1, 13):
                    sel = mm[te] == mo
                    rows["by_month"][mo] = {"n": int(sel.sum()), "rmse_cm": round(rmse(err[sel]) * 100, 1), "persistence_rmse_cm": round(rmse(pers[sel]) * 100, 1), "split_rmse_cm": round(rmse(err_split[sel]) * 100, 1)}
                per[variant] = rows
            res[f"D+{h}"] = per
            a, b, c = per["arx"]["all"], per["future_all"]["all"], per["future_3d"]["all"]
            print(f"{station} D+{h}: 직전값 {a['persistence_rmse_cm']} | ar {per['ar']['all']['rmse_cm']} | arx {a['rmse_cm']} (홍수기분리 {a['split_rmse_cm']}) | +3일예보 {c['rmse_cm']} | +전구간예보 {b['rmse_cm']} (분리 {b['split_rmse_cm']}) cm", flush=True)
        out[station] = res
    return out


# ───────────────────────── ① 시 단위
def load_hourly(fname):
    doc = json.loads((ROOT / "data" / "wamis" / "hourly" / fname).read_text(encoding="utf-8"))
    rows = doc["rows"]
    # ymdh: 'YYYYMMDDHH' (HH 1..24 → 24 는 다음날 0시)
    def parse(k):
        y, mo, d, hh = int(k[:4]), int(k[4:6]), int(k[6:8]), int(k[8:10])
        base = datetime(y, mo, d) + timedelta(hours=hh)
        return base
    ts = [parse(k) for k, _ in rows]; vals = [v for _, v in rows]
    t0, t1 = min(ts), max(ts); n = int((t1 - t0).total_seconds() // 3600) + 1
    y = np.full(n, np.nan)
    for t, v in zip(ts, vals): y[int((t - t0).total_seconds() // 3600)] = v
    # 3시간 이하 결측은 선형 보간
    isn = np.isnan(y); idx = np.arange(n)
    if isn.any():
        yi = np.interp(idx, idx[~isn], y[~isn]); gap = np.zeros(n, dtype=int)
        run = 0
        for i in range(n):
            run = run + 1 if isn[i] else 0; gap[i] = run
        for i in range(n):
            if isn[i] and gap[i] <= 3: y[i] = yi[i]
    times = [t0 + timedelta(hours=int(i)) for i in range(n)]
    return times, y


def hourly_features(y, times, t):
    cols = [y[t - i] - y[t - i - 1] for i in range(24)]
    def rmean(w):
        out = np.full(len(t), np.nan)
        for j, i in enumerate(t): out[j] = np.nanmean(y[i - w + 1:i + 1])
        return out
    cols.append(y[t] - rmean(24)); cols.append(y[t] - rmean(24 * 7))
    hod = np.array([times[i].hour for i in t]); doy = np.array([times[i].timetuple().tm_yday for i in t])
    cols += [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    return np.column_stack(cols)


def run_hourly():
    out = {}
    for fname, station in HOURLY:
        times, y = load_hourly(fname); n = len(y); t_arr = np.array(times)
        res = {"n_hours": n, "coverage": round(float(np.mean(np.isfinite(y))), 4), "period": [times[0].isoformat(), times[-1].isoformat()]}
        for h in H_HOURLY:
            t_all = np.arange(24 * 7 + 1, n - h); target = y[t_all + h] - y[t_all]
            X = hourly_features(y, times, t_all); ok = np.isfinite(X).all(axis=1) & np.isfinite(target)
            tt, XX, yy = t_all[ok], X[ok], target[ok]; dd = t_arr[tt]
            tr = dd < datetime(2023, 1, 1); va = (dd >= datetime(2023, 1, 1)) & (dd < datetime(2024, 1, 1)); te = dd >= datetime(2024, 1, 1)
            m, lam = fit_select(XX, yy, tr, va); pred = ridge_predict(m, XX[te]); err = pred - yy[te]; pers = -yy[te]
            mm = np.array([x.month for x in dd[te]])
            res[f"h+{h}"] = {"n_test": int(te.sum()), "rmse_cm": round(rmse(err) * 100, 1), "persistence_rmse_cm": round(rmse(pers) * 100, 1), "mae_cm": round(float(np.mean(np.abs(err))) * 100, 1),
                             "skill": round(1 - (rmse(err) / rmse(pers)) ** 2, 3), "lambda": lam,
                             "flood_months_rmse_cm": round(rmse(err[np.isin(mm, list(FLOOD_MONTHS))]) * 100, 1), "flood_months_persistence_cm": round(rmse(pers[np.isin(mm, list(FLOOD_MONTHS))]) * 100, 1)}
            r = res[f"h+{h}"]; print(f"{station} 시간 h+{h}: 릿지 {r['rmse_cm']} cm vs 직전값 {r['persistence_rmse_cm']} cm (개선 {r['skill']*100:+.0f} %) · 홍수기 {r['flood_months_rmse_cm']} vs {r['flood_months_persistence_cm']} [n={r['n_test']}]", flush=True)
        out[station] = res
    return out


def main() -> int:
    daily = run_daily(); hourly = run_hourly()
    OUT_JSON.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "daily": daily, "hourly": hourly, "note_ko": __doc__}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", OUT_JSON); return 0


if __name__ == "__main__":
    sys.exit(main())
