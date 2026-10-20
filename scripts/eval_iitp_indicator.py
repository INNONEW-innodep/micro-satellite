#!/usr/bin/env python3
"""TTA 시험용 — 성과지표 '수체면적 및 수위변화 시계열 예측 정확도(MSE)' 산출.

시험 절차서의 절차 15에서 실행한다. 배포된 시스템의 공개 API 만 호출하므로
시험 현장에서 클라이언트#2(터미널)로 그대로 돌릴 수 있다.

    python3 scripts/eval_iitp_indicator.py --origin satellite --horizon 1
    python3 scripts/eval_iitp_indicator.py --origin satellite --horizon 7

산출식 (시험 절차서와 동일)
  ① 변수별 [0,1] 정규화 — 분모는 시험 구간 실측 관측범위(max-min)
  ② 수위 정규화 MSE : 위성 촬영일을 원점으로 한 D+h 예측 vs 동일 일자 수위관측소 실측
  ③ 수체면적 정규화 MSE : 공통 격자 면적을 평균회귀 모델로 차기 관측시점 예측 vs 실측
  ④ 종합 = (② + ③) / 2,  목표 0.38 이하

수체면적을 '공통 격자'에서 재는 이유
  위성은 날짜마다 조금씩 다른 범위를 촬영한다. 장면마다 자기 촬영범위에서 센 면적을
  나열하면 촬영 범위 변화가 면적 변화로 둔갑한다(부산 2020-03-25 는 촬영범위가 남쪽으로
  8.6 km 이동해 하구·바다가 들어왔다). 전 날짜가 공통으로 덮는 격자에서 재야 시계열
  비교가 성립한다. 탐지 정확도(IoU)와는 별개의 축이며 탐지 성능과 무관하다.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_API = "https://wc-api.semyeongsoft.com"
DEFAULT_REGION = "nakdong_busan"
# 위성 산출 수위와 연계된 수위관측소 (시험 절차서 사전조건)
DEFAULT_STATIONS = ("구포대교", "호포대교", "정천교")
TARGET = 0.38


def get(api: str, path: str, **params: Any) -> Any:
    url = api.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:  # 404 등은 호출부에서 판단
        raise RuntimeError(f"{url} → HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} 에 연결할 수 없습니다: {exc}") from exc


def quote(value: str) -> str:
    return urllib.parse.quote(str(value), safe="")


# --- ② 수위 -------------------------------------------------------------------------


def level_term(api: str, region: str, stations: tuple[str, ...], horizon: int) -> dict[str, Any]:
    """위성 촬영일을 원점으로 한 D+h 예측과 동일 일자 실측의 정규화 MSE."""

    scenes = get(api, f"/api/v1/regions/{quote(region)}/satellite/scenes")
    scene_dates = sorted({str(s["date"]) for s in scenes.get("scenes") or []})
    if not scene_dates:
        raise RuntimeError(f"{region} 에 위성 장면이 없습니다")

    rows: list[dict[str, Any]] = []
    for station in stations:
        series = get(api, f"/api/v1/regions/{quote(region)}/gauge/{quote(station)}/series")
        actual = {str(r["date"]): float(r["water_level_m"]) for r in series.get("rows") or [] if r.get("water_level_m") is not None}
        if not actual:
            raise RuntimeError(f"{station} 수위 시계열을 받지 못했습니다")
        # 분모: 시험 구간 실측 관측범위
        evaluation = get(api, f"/api/v1/regions/{quote(region)}/gauge/evaluation")
        split = evaluation.get("split") or {}
        lo, hi = str(split.get("test_start", ""))[:10], str(split.get("test_end", ""))[:10]
        test_values = [v for d, v in actual.items() if (not lo or d >= lo) and (not hi or d <= hi)]
        if len(test_values) < 2:
            raise RuntimeError(f"{station}: 시험 구간({lo}~{hi}) 실측이 부족합니다 — 절차 7의 재학습을 먼저 수행하세요")
        span = max(test_values) - min(test_values)
        if span <= 0:
            raise RuntimeError(f"{station}: 시험 구간 수위 관측범위가 0 입니다")

        for origin in scene_dates:
            if origin not in actual:
                continue  # 그 지점에 그 날짜 위성 산출이 없으면 제외 (절차서: 정천교 6일)
            try:
                payload = get(
                    api,
                    f"/api/v1/regions/{quote(region)}/gauge/{quote(station)}/forecast",
                    as_of=origin, horizon=horizon, history_days=7,
                )
            except RuntimeError as exc:
                rows.append({"station": station, "origin": origin, "error": str(exc)[:120]})
                continue
            step = next((s for s in payload.get("steps") or [] if int(s.get("horizon", 0)) == horizon), None)
            if not step or step.get("forecast_m") is None:
                rows.append({"station": station, "origin": origin, "error": "예측값 없음"})
                continue
            target_date = str(step.get("target_date"))
            truth = actual.get(target_date)
            if truth is None:
                rows.append({"station": station, "origin": origin, "error": f"{target_date} 실측 없음"})
                continue
            err = float(step["forecast_m"]) - truth
            rows.append({
                "station": station, "origin": origin, "target_date": target_date,
                "forecast_m": round(float(step["forecast_m"]), 4), "truth_m": round(truth, 4),
                "err_m": round(err, 4), "range_m": round(span, 4),
                "norm_sq": (err / span) ** 2,
            })

    usable = [r for r in rows if "norm_sq" in r]
    if not usable:
        raise RuntimeError("수위 예측 원점이 하나도 성립하지 않았습니다")
    return {
        "n_origins": len(usable),
        "n_failed": len(rows) - len(usable),
        "norm_mse": sum(r["norm_sq"] for r in usable) / len(usable),
        "rmse_m": (sum(r["err_m"] ** 2 for r in usable) / len(usable)) ** 0.5,
        "rows": rows,
    }


# --- ③ 수체면적 ---------------------------------------------------------------------


def area_term(api: str, region: str, model: str = "mean_reversion", phi: float = 0.5) -> dict[str, Any]:
    """공통 격자 면적을 평균회귀로 차기 관측시점 예측했을 때의 정규화 MSE.

    각 예측은 '그 시점까지 관측된 값'만 쓴다(미래 미사용). 평균회귀는
    예측 = 직전값 + phi × (지금까지 평균 − 직전값) 이며 phi=0 이면 직전값 유지와 같다.
    """

    payload = get(api, f"/api/v1/regions/{quote(region)}/satellite/area-series")
    if not payload.get("available"):
        raise RuntimeError(
            "공통 격자 면적 계열이 없습니다 — scripts/build_common_frame_area.py 를 먼저 실행하세요"
        )
    sensors: dict[str, Any] = payload.get("sensors") or {}
    observations: list[float] = []
    pairs: list[dict[str, Any]] = []
    for sensor, block in sorted(sensors.items()):
        rows = sorted(block.get("rows") or [], key=lambda r: str(r["date"]))
        areas = [float(r["area_km2_common_frame"]) for r in rows if r.get("area_km2_common_frame") is not None]
        dates = [str(r["date"]) for r in rows if r.get("area_km2_common_frame") is not None]
        observations += areas
        for i in range(len(areas) - 1):
            history = areas[: i + 1]
            previous = history[-1]
            if model == "persistence":
                prediction = previous
            else:
                mean = sum(history) / len(history)
                prediction = previous + phi * (mean - previous)
            pairs.append({
                "sensor": sensor, "from": dates[i], "to": dates[i + 1],
                "pred_km2": round(prediction, 4), "truth_km2": round(areas[i + 1], 4),
                "err_km2": round(prediction - areas[i + 1], 4),
            })
    if len(pairs) < 2:
        raise RuntimeError("인접 관측쌍이 부족합니다")
    span = max(observations) - min(observations)
    if span <= 0:
        raise RuntimeError("수체면적 관측범위가 0 입니다")
    mse = sum(p["err_km2"] ** 2 for p in pairs) / len(pairs)
    return {
        "model": model, "phi": phi if model != "persistence" else 0.0,
        "n_pairs": len(pairs), "n_observations": len(observations),
        "range_km2": round(span, 4), "mse_km2sq": round(mse, 4),
        "rmse_km2": round(mse ** 0.5, 4), "norm_mse": mse / span ** 2,
        "pairs": pairs,
    }


# --- 실행 ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=DEFAULT_API, help="시험대상 API 주소")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--origin", choices=("satellite",), default="satellite", help="예측 원점(위성 촬영일)")
    parser.add_argument("--horizon", type=int, default=1, choices=(1, 3, 5, 7, 10, 14, 30))
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    parser.add_argument("--area-model", default="mean_reversion", choices=("mean_reversion", "persistence"))
    parser.add_argument("--json", action="store_true", help="계산 상세를 JSON 으로 출력")
    args = parser.parse_args()

    stations = tuple(s.strip() for s in args.stations.split(",") if s.strip())
    try:
        level = level_term(args.api, args.region, stations, args.horizon)
        area = area_term(args.api, args.region, args.area_model)
    except RuntimeError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2

    composite = (level["norm_mse"] + area["norm_mse"]) / 2
    verdict = "적합" if composite <= TARGET else "부적합"

    print("=" * 74)
    print(f" 성과지표 — 수체면적 및 수위변화 시계열 예측 정확도 (MSE)   [D+{args.horizon}]")
    print("=" * 74)
    print(f" 시험대상 : {args.api}   지역 : {args.region}")
    print()
    print(f" ② 수위      원점 {level['n_origins']}건" + (f" (제외 {level['n_failed']}건)" if level["n_failed"] else "")
          + f"   RMSE {level['rmse_m'] * 100:.1f} cm   정규화 MSE {level['norm_mse']:.5f}")
    print(f" ③ 수체면적  인접쌍 {area['n_pairs']}개 (관측 {area['n_observations']}시점, 공통 격자)")
    print(f"             모델 {'평균회귀' if area['model'] == 'mean_reversion' else '직전값 유지'}"
          + (f" (phi={area['phi']})" if area["model"] != "persistence" else "")
          + f"   RMSE {area['rmse_km2']:.3f} km²   관측범위 {area['range_km2']:.3f} km²")
    print(f"             정규화 MSE {area['norm_mse']:.5f}")
    print()
    print(f" ④ 종합 MSE = ({level['norm_mse']:.5f} + {area['norm_mse']:.5f}) / 2 = {composite:.5f}")
    print(f"    목표 {TARGET} 이하  →  판정 : {verdict}")
    print("=" * 74)

    if args.json:
        print(json.dumps({"horizon": args.horizon, "level": level, "area": area,
                          "composite_norm_mse": composite, "target": TARGET, "verdict": verdict},
                         ensure_ascii=False, indent=2))
    return 0 if verdict == "적합" else 1


if __name__ == "__main__":
    sys.exit(main())
