#!/usr/bin/env python3
"""지역 레지스트리의 수위계 시계열로 지점별 예측 모델을 학습·평가해 저장한다.

백엔드 ``POST /api/v1/regions/{id}/gauge/train`` 과 같은 코드를 쓰되, 산출물을
저장소의 ``regions/<id>/models/gauge/`` (내장 지역) 에 쓴다. 그래서 학습 결과가
커밋되어 배포 이미지에 함께 들어간다 — 서버에서 다시 학습하지 않아도 화면이 뜬다.

    python3 scripts/train_gauge_models.py                       # 내장 지역 전부
    python3 scripts/train_gauge_models.py --region nakdong_busan --stations 구포대교
    python3 scripts/train_gauge_models.py --forecasters analog_knn   # 모델 교체 실험

규율은 backend/app/gauge/training.py 머리말 참조: 검증 연도에서 고르고 시험 연도에서
보고하며, persistence 는 기준선일 뿐 후보가 아니다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.gauge.forecasters import build_forecaster_registry  # noqa: E402
from backend.app.gauge.regions import RegionRegistry  # noqa: E402
from backend.app.gauge.training import TrainingError, train_region  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions-dir", default=str(ROOT / "regions"))
    parser.add_argument("--user-dir", default=str(ROOT / "backend" / "data" / "regions"))
    parser.add_argument("--region", action="append", help="지역 id (반복 가능). 비우면 전부")
    parser.add_argument("--stations", default=None, help="쉼표 구분 지점 id")
    parser.add_argument("--forecasters", default=None, help="쉼표 구분 예측기 id (후보 한정)")
    parser.add_argument("--horizon-max", type=int, default=None)
    parser.add_argument("--plugins", default="", help="GAUGE_FORECASTER_PLUGINS 형식")
    parser.add_argument("--to-user-dir", action="store_true", help="저장소 대신 사용자 디렉터리에 저장")
    args = parser.parse_args()

    registry = RegionRegistry(Path(args.regions_dir), Path(args.user_dir))
    forecasters = build_forecaster_registry(args.plugins)
    if forecasters.load_errors:
        print("[plugin] 로드 실패:", forecasters.load_errors, file=sys.stderr)

    region_ids = args.region or [r.id for r in registry.list() if r.gauge_available]
    stations = [s.strip() for s in args.stations.split(",")] if args.stations else None
    allowed = [s.strip() for s in args.forecasters.split(",")] if args.forecasters else None

    exit_code = 0
    for region_id in region_ids:
        record = registry.get(region_id)
        write_dir = registry.artifacts_write_dir(region_id, prefer_builtin=not args.to_user_dir)
        started = time.time()
        print(f"== {region_id} ({record.name}) → {write_dir}")
        try:
            evaluation = train_region(
                registry,
                forecasters,
                region_id,
                stations=stations,
                allowed_forecasters=allowed,
                horizon_max=args.horizon_max,
                write_dir=write_dir,
            )
        except TrainingError as exc:
            print(f"   실패: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        rows = evaluation["rows"]
        print(f"   지점 {evaluation['n_stations']}개 · 셀 {evaluation['cells_total']}개 · "
              f"persistence 를 이기는 셀 {evaluation['cells_beating_persistence']}개 · {time.time() - started:.1f}s")
        header = f"   {'지점':14} {'지형':12} {'D+':>3} {'모델':22} {'val RMSE':>9} {'test RMSE':>9} {'pers RMSE':>9} {'skill':>7}"
        print(header)
        for row in rows:
            skill = row.get("skill_vs_persistence")
            print(
                f"   {row['station_name']:14} {row['terrain']:12} {row['horizon']:>3} {row['forecaster_id']:22} "
                f"{(row.get('val_rmse') or float('nan')):>9.4f} {(row.get('test_rmse') or float('nan')):>9.4f} "
                f"{(row.get('persistence_test_rmse') or float('nan')):>9.4f} "
                f"{(skill * 100 if skill is not None else float('nan')):>+6.1f}%"
            )
        if evaluation.get("failures"):
            print("   실패 지점:", json.dumps(evaluation["failures"], ensure_ascii=False))
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
