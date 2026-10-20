#!/usr/bin/env python3
"""위성 수위 대조 지점(구포대교·호포대교·정천교 + 시립대 모델 2 지점 양산교)의 시계열 그래프 자료를
발표자료 데이터(docs/presentation/data/timeseries_deck_data.json)에 삼랑진교와 같은 형식으로 추가한다.
수치는 전부 배포 API(공개 주소)와 저장소 산출물에서 읽는다 — 합성값 없음.

추가 키
  nakdong_busan:<지점>            2024-07-01 원점 30일 전망(선택 모델) — 기존 구포대교·삼랑진교와 같은 모양(+verification, models_by_horizon)
  backtest_<slug>_7               2024 D+7 백테스트 궤적 {dates, truth, pred, persistence}
  horizon_rmse:nakdong_busan      지점별 7지평 선택 모델·직전값 RMSE(cm) — evaluation.json rows
  monthly_d7:nakdong_busan        지점별 월별 D+7 RMSE 12개 + 급변 57일(7월 · 9/15~10/10) 제곱오차 기여율·제외 시 RMSE
  scene_origin_eval:nakdong_busan 위성 관측일 8일을 원점으로 한 D+1·D+7 예측 vs 실측(지점별) + 완전 예보 가정(ridge_arx_future) 병기
  future_rain_example_2024-07-01  장마철 원점 30일 전망을 선택 모델 / ridge_arx / ridge_arx_future(완전 예보 가정)로 각각 낸 검증 RMSE
  future_rain_experiment:nakdong_busan  experiments/future_rain.json 의 지점별 지평별 값(과거 기상 AR-X vs +3일 예보 vs 전 구간 예보)

실행: python3 scripts/build_satellite_station_deck_data.py [--api https://wc-api.semyeongsoft.com]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import date as Date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "presentation" / "data" / "timeseries_deck_data.json"
REGION = "nakdong_busan"
STATIONS = ["구포대교", "호포대교", "정천교", "양산교"]
SLUG = {"구포대교": "gupo", "호포대교": "hopo", "정천교": "jeongcheon", "양산교": "yangsan"}
OFFICIAL = ("구포대교", "호포대교", "정천교")  # 공식 지표 24건 = 3지점 × 위성 관측일 8일
SCENE_DATES = ["2020-02-18", "2020-03-02", "2020-03-12", "2020-03-25", "2020-03-30", "2020-04-14", "2020-04-15", "2020-04-16"]
SCENE_SENSOR = {"2020-02-18": "PlanetScope", "2020-03-02": "ICEYE", "2020-03-12": "PlanetScope", "2020-03-25": "PlanetScope",
                "2020-03-30": "ICEYE", "2020-04-14": "PlanetScope", "2020-04-15": "ICEYE", "2020-04-16": "ICEYE"}
EXAMPLE_ORIGIN = "2024-07-01"
SURGE_WINDOWS = [("2024-07-01", "2024-07-31"), ("2024-09-15", "2024-10-10")]  # 57일


def get(api: str, path: str, **params: object) -> dict:
    url = f"{api}/api/v1{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.load(r)


def rmse(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None


def cm(x: float | None) -> float | None:
    return None if x is None else round(x * 100, 1)


def in_surge(d: str) -> bool:
    return any(a <= d <= b for a, b in SURGE_WINDOWS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default="https://wc-api.semyeongsoft.com")
    args = ap.parse_args()
    api = args.api.rstrip("/")
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    quote = urllib.parse.quote

    # 1) 예측일수별 RMSE — evaluation.json rows
    ev = json.loads((ROOT / "regions" / REGION / "models" / "gauge" / "evaluation.json").read_text(encoding="utf-8"))
    horizon_rmse: dict[str, list[dict]] = {}
    for st in STATIONS:
        rows = sorted((r for r in ev["rows"] if r["station_id"] == st), key=lambda r: int(r["horizon"]))
        horizon_rmse[st] = [{
            "horizon": int(r["horizon"]), "forecaster_id": r["forecaster_id"], "forecaster_name": r["forecaster_name"],
            "test_rmse_cm": cm(r["test_rmse"]), "persistence_rmse_cm": cm(r["persistence_test_rmse"]),
            "skill_vs_persistence": round(float(r["skill_vs_persistence"]), 4), "uses_weather": bool(r.get("uses_weather")),
        } for r in rows]
    doc[f"horizon_rmse:{REGION}"] = {"source": "regions/nakdong_busan/models/gauge/evaluation.json rows (2024 시험 연도)", "stations": horizon_rmse}

    # 2) 2024-07-01 원점 30일 전망 — 선택 모델 / ridge_arx / ridge_arx_future
    example: dict[str, dict] = {}
    for st in STATIONS + ["대청댐"]:
        region = REGION if st != "대청댐" else "geum_daecheong"
        sel = get(api, f"/regions/{region}/gauge/{quote(st)}/forecast", as_of=EXAMPLE_ORIGIN, horizon=30)
        if st != "대청댐":
            doc[f"{REGION}:{st}"] = {
                "as_of": sel["as_of"], "origin_level_m": sel.get("origin_level_m"), "history": sel.get("history"), "steps": sel.get("steps"),
                "verification": sel.get("verification"), "models_by_horizon": sel.get("models_by_horizon"),
                "source": f"API /regions/{region}/gauge/{st}/forecast?as_of={EXAMPLE_ORIGIN}&horizon=30 (선택 모델)",
            }
        variants = {"selected": sel}
        for fid in ("ridge_arx", "ridge_arx_future"):
            variants[fid] = get(api, f"/regions/{region}/gauge/{quote(st)}/forecast", as_of=EXAMPLE_ORIGIN, horizon=30, forecaster=fid)
        example[st] = {
            k: {"rmse_cm": cm(v["verification"]["rmse_m"]), "mae_cm": cm(v["verification"]["mae_m"]),
                "persistence_rmse_cm": cm(v["verification"]["persistence_rmse_m"]), "skill_vs_persistence": round(v["verification"]["skill_vs_persistence"], 3),
                "n": v["verification"]["n"], "models_by_horizon": (v.get("models_by_horizon") if k == "selected" else None),
                "steps_forecast_m": [s["forecast_m"] for s in v["steps"]]}
            for k, v in variants.items()
        }
        example[st]["note_ko"] = ("ridge_arx_future 는 원점 이후 실측 강수를 완전 예보로 간주한 상한(실험용, 배포 후보 제외). "
                                  "실시간에는 기상청 단기예보 연결이 필요하며 예보 오차만큼 이득이 준다.")
        print(f"[예보 포함 2024-07-01] {st}: 선택 {example[st]['selected']['rmse_cm']} / ridge_arx {example[st]['ridge_arx']['rmse_cm']} / "
              f"완전예보 {example[st]['ridge_arx_future']['rmse_cm']} cm (직전값 {example[st]['selected']['persistence_rmse_cm']})", flush=True)
    doc["future_rain_example_2024-07-01"] = example

    # 3) D+7 백테스트 궤적 + 4) 월별·급변 구간
    monthly: dict[str, dict] = {}
    for st in STATIONS:
        bt = get(api, f"/regions/{REGION}/gauge/{quote(st)}/backtest", horizon=7)
        tr = bt["trace"]
        doc[f"backtest_{SLUG[st]}_7"] = {"dates": tr["dates"], "truth": tr["truth"], "pred": tr["pred"], "persistence": tr["persistence"],
                                        "forecaster_id": bt.get("forecaster_id"), "forecaster_name": bt.get("forecaster_name"),
                                        "source": f"API /regions/{REGION}/gauge/{st}/backtest?horizon=7 trace"}
        errs = [(d, (p - t), (q - t)) for d, t, p, q in zip(tr["dates"], tr["truth"], tr["pred"], tr["persistence"]) if t is not None and p is not None]
        total_sq = sum(e * e for _, e, _ in errs)
        months = []
        for m in range(1, 13):
            sel = [(e, q) for d, e, q in errs if int(d[5:7]) == m]
            months.append({"month": m, "n": len(sel), "rmse_cm": cm(rmse([e for e, _ in sel])), "persistence_rmse_cm": cm(rmse([q for _, q in sel])),
                           "sq_err_share": round(sum(e * e for e, _ in sel) / total_sq, 4) if total_sq else None})
        surge = [(e, q) for d, e, q in errs if in_surge(d)]
        rest = [(e, q) for d, e, q in errs if not in_surge(d)]
        monthly[st] = {
            "forecaster_id": bt.get("forecaster_id"), "months": months,
            "surge_window": {"definition_ko": "급변 57일 = 7월 전체(31일) + 9/15~10/10(26일)", "n_days": len(surge),
                             "sq_err_share": round(sum(e * e for e, _ in surge) / total_sq, 4) if total_sq else None,
                             "rmse_cm_all": cm(rmse([e for _d, e, _q in errs])), "rmse_cm_surge": cm(rmse([e for e, _ in surge])),
                             "rmse_cm_excluding_surge": cm(rmse([e for e, _ in rest])),
                             "persistence_rmse_cm_all": cm(rmse([q for _d, _e, q in errs])), "persistence_rmse_cm_excluding_surge": cm(rmse([q for _, q in rest]))},
        }
        sw = monthly[st]["surge_window"]
        print(f"[월별 D+7] {st}: 전체 {sw['rmse_cm_all']} cm, 급변 57일 기여 {sw['sq_err_share']*100:.0f} %, 제외 시 {sw['rmse_cm_excluding_surge']} cm", flush=True)
    doc[f"monthly_d7:{REGION}"] = {"source": "backtest trace(D+7, 2024) 에서 계산", "stations": monthly}

    # 5) 위성 관측일 원점 D+1·D+7 예측 vs 실측 (+ 완전 예보 가정)
    rows = []
    for st in STATIONS:
        for d in SCENE_DATES:
            sel = get(api, f"/regions/{REGION}/gauge/{quote(st)}/forecast", as_of=d, horizon=7)
            fut = get(api, f"/regions/{REGION}/gauge/{quote(st)}/forecast", as_of=d, horizon=7, forecaster="ridge_arx_future")
            s1, s7 = sel["steps"][0], sel["steps"][6]
            f1, f7 = fut["steps"][0], fut["steps"][6]
            rows.append({
                "station": st, "origin": d, "sensor": SCENE_SENSOR[d], "origin_level_m": sel.get("origin_level_m"),
                "h1": {"target_date": s1["target_date"], "forecast_m": s1["forecast_m"], "observed_m": s1.get("observed_m"), "persistence_m": s1["persistence_reference_m"], "forecaster_id": s1["forecaster_id"], "future_rain_forecast_m": f1["forecast_m"]},
                "h7": {"target_date": s7["target_date"], "forecast_m": s7["forecast_m"], "observed_m": s7.get("observed_m"), "persistence_m": s7["persistence_reference_m"], "forecaster_id": s7["forecaster_id"], "future_rain_forecast_m": f7["forecast_m"]},
            })
            print(f"[관측일 원점] {st} {d}: D+1 {s1['forecast_m']:.3f}/{s1.get('observed_m')} D+7 {s7['forecast_m']:.3f}/{s7.get('observed_m')}", flush=True)

    def summarize(rs: list[dict]) -> dict:
        out = {"n": len(rs)}
        for h in ("h1", "h7"):
            obs = [r[h]["observed_m"] for r in rs]
            out[h] = {
                "rmse_cm": cm(rmse([r[h]["forecast_m"] - r[h]["observed_m"] for r in rs if r[h]["observed_m"] is not None])),
                "mae_cm": cm((sum(abs(r[h]["forecast_m"] - r[h]["observed_m"]) for r in rs if r[h]["observed_m"] is not None) / max(1, sum(1 for o in obs if o is not None)))),
                "persistence_rmse_cm": cm(rmse([r[h]["persistence_m"] - r[h]["observed_m"] for r in rs if r[h]["observed_m"] is not None])),
                "future_rain_rmse_cm": cm(rmse([r[h]["future_rain_forecast_m"] - r[h]["observed_m"] for r in rs if r[h]["observed_m"] is not None])),
                "n_observed": sum(1 for o in obs if o is not None),
            }
            m, p = out[h]["rmse_cm"], out[h]["persistence_rmse_cm"]
            out[h]["skill_vs_persistence"] = round(1 - (m / p) ** 2, 3) if m is not None and p else None
        return out

    summary = {st: summarize([r for r in rows if r["station"] == st]) for st in STATIONS}
    summary["official_24"] = summarize([r for r in rows if r["station"] in OFFICIAL])
    doc[f"scene_origin_eval:{REGION}"] = {
        "definition_ko": "위성 촬영일을 원점으로 D+1·D+7 수위 전망(선택 모델) vs 같은 날 수위계 실측 — 공식 지표 수위 항의 24건은 구포대교·호포대교·정천교 × 관측일 8일. "
                         "future_rain_* 는 ridge_arx_future(원점 이후 실측 강수를 완전 예보로 간주, 실험용)로 같은 원점에서 낸 값.",
        "stations": STATIONS, "official_stations": list(OFFICIAL), "rows": rows, "summary": summary,
    }
    for st, s in summary.items():
        print(f"[관측일 원점 요약] {st}: n={s['n']} D+1 {s['h1']['rmse_cm']} (직전값 {s['h1']['persistence_rmse_cm']}, 완전예보 {s['h1']['future_rain_rmse_cm']}) · "
              f"D+7 {s['h7']['rmse_cm']} (직전값 {s['h7']['persistence_rmse_cm']}, 완전예보 {s['h7']['future_rain_rmse_cm']}) cm", flush=True)

    # 6) 미래 강수 실험(저장소 산출물)
    exp = json.loads((ROOT / "regions" / REGION / "models" / "gauge" / "experiments" / "future_rain.json").read_text(encoding="utf-8"))
    fr: dict[str, dict] = {}
    for st in STATIONS:
        blk = exp["stations"][st]["horizons"]
        fr[st] = {h: {k: {"rmse_cm": cm(v["rmse"]), "skill_vs_persistence": round(v["skill_vs_persistence"], 3), "skill_gain_pp": (round(v["skill_gain_pp"], 1) if "skill_gain_pp" in v else None)}
                      for k, v in blk[h]["models"].items()} for h in ("1", "3", "7", "14", "30") if h in blk}
    doc[f"future_rain_experiment:{REGION}"] = {"generated_at": exp.get("generated_at"), "variants": exp.get("variants"),
                                               "note_ko": "과거 기상 AR-X 대비 +3일 예보 / 전 구간 예보(실측 미래 강수 = 완전 예보 가정)의 2024 시험 RMSE·개선율 이득(%p)",
                                               "stations": fr}
    DATA.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", DATA, "keys", len(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
