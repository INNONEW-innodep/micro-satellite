#!/usr/bin/env python3
"""K-water 2026-10 공동실증 자료(대청댐 직하류, Umbra SAR 3 + PlanetScope 3) 결과 정리.

서버 파이프라인(`/api/v1/scenes`) 결과를 모아 표로 만들고, 같은 날짜 근처의 SAR·광학 쌍은
SAR 촬영범위 안에서 물 면적을 맞대어 본다.

    .venv/bin/python scripts/kwater_202610_report.py --api https://wc-api.semyeongsoft.com \
        --out data/incoming/kwater_202610/RESULTS.md
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REGION = "geum_daecheong"
SCENES = [  # (scene_id, 센서, 날짜, 비고)
    ("20260920_PlanetScope", "PlanetScope", "2026-09-20", "TOA 반사도(대기보정 전) — 참고용"),
    ("20260925_Umbra", "Umbra", "2026-09-25", "우측 관측, 입사각 48°"),
    ("20260927_PlanetScope", "PlanetScope", "2026-09-27", "지표반사도"),
    ("20260928_Umbra", "Umbra", "2026-09-28", "우측 관측, 입사각 36°"),
    ("20261002_PlanetScope", "PlanetScope", "2026-10-02", "지표반사도"),
    ("20261003_Umbra", "Umbra", "2026-10-03", "좌측 관측, 입사각 36°"),
]
PAIRS = [("20260925_Umbra", "20260927_PlanetScope"), ("20260928_Umbra", "20260927_PlanetScope"), ("20261003_Umbra", "20261002_PlanetScope")]


def get(api: str, path: str, raw: bool = False):
    with urllib.request.urlopen(api.rstrip("/") + path, timeout=300) as r:
        return r.read() if raw else json.load(r)


def fmt(v, nd=2):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://wc-api.semyeongsoft.com")
    ap.add_argument("--out", default="data/incoming/kwater_202610/RESULTS.md")
    ap.add_argument("--mask_dir", default="data/incoming/kwater_202610/results_masks", help="마스크 GeoTIFF 내려받을 곳")
    a = ap.parse_args()
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    Path(a.mask_dir).mkdir(parents=True, exist_ok=True)
    rows, records = [], {}
    for sid, sensor, date, note in SCENES:
        try:
            d = get(a.api, f"/api/v1/scenes/{sid}")
        except Exception as exc:  # noqa: BLE001
            rows.append((sid, sensor, date, "없음", "—", "—", "—", "—", f"조회 실패: {exc}")); continue
        records[sid] = d
        steps = {s["key"]: s for s in d["steps"]}
        o = d.get("outputs") or {}
        lv = get(a.api, f"/api/v1/regions/{REGION}/satellite/levels?start={date}&end={date}") if o.get("levels_url") else {"records": []}
        levels = "; ".join(f"{r.get('loc_id')} {fmt(r.get('water_level_m'), 3)} m" for r in lv.get("records", [])) or "—"
        pred = "—"
        if o.get("prediction_id"):
            p = get(a.api, f"/api/v1/predictions/{o['prediction_id']}")
            st = p.get("steps") or []
            if st:
                pred = f"{st[0].get('water_area_km2') or 0:.2f} km² ({st[0].get('risk') or p.get('risk') or 'n/a'})"
        det = steps.get("detect", {})
        rows.append((sid, sensor, date, d["status"], fmt(o.get("water_area_km2")), levels, pred,
                     f"{det.get('status')} {fmt(det.get('elapsed_s'), 0)}s", note + ("; " + "; ".join(w[:60] for w in d.get("warnings", [])[:2]) if d.get("warnings") else "")))
        # 마스크 내려받기(쌍 비교용)
        if o.get("mask_url"):
            dest = Path(a.mask_dir) / f"{sid}.tif"
            if not dest.is_file():
                dest.write_bytes(get(a.api, o["mask_url"], raw=True))

    out = io.StringIO()
    out.write("# K-water 공동실증 자료 처리 결과 — 대청댐 직하류 (2026-09-20 ~ 10-03)\n\n")
    out.write(f"서버 `{a.api}` 의 장면 파이프라인(`POST /api/v1/scenes`) 결과. 장면마다 등록 → 탐지 → 면적 → 수위 → 예측 → 규약 파일.\n\n")
    out.write("| 장면 | 센서 | 날짜 | 상태 | 수체 면적 km² | 위성 수위 (지점) | 예측 1단계 | 탐지 | 비고 |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        out.write("| " + " | ".join(str(x) for x in r) + " |\n")

    # SAR·광학 쌍 비교 — SAR 촬영범위(유효 화소) 안에서 두 마스크의 물 면적·일치율
    out.write("\n## SAR·광학 같은 시기 쌍 비교 (SAR 촬영범위 안)\n\n")
    out.write("| SAR | 광학 | SAR 물 km² | 광학 물 km² | 교집합 km² | IoU | 비고 |\n|---|---|---|---|---|---|---|\n")
    for sar_id, opt_id in PAIRS:
        sp, op = Path(a.mask_dir) / f"{sar_id}.tif", Path(a.mask_dir) / f"{opt_id}.tif"
        if not (sp.is_file() and op.is_file()):
            out.write(f"| {sar_id} | {opt_id} | — | — | — | — | 마스크 없음 |\n"); continue
        with rasterio.open(sp) as ds, rasterio.open(op) as do:
            sar = ds.read(1); b = ds.bounds
            win = from_bounds(b.left, b.bottom, b.right, b.top, do.transform)
            opt = do.read(1, window=win, out_shape=sar.shape, resampling=Resampling.nearest, boundless=True, fill_value=255)
            px = abs(ds.res[0] * ds.res[1]) / 1e6
        valid = (sar != 255) & (opt != 255)
        ws, wo = (sar == 1) & valid, (opt == 1) & valid
        inter, union = (ws & wo).sum(), (ws | wo).sum()
        out.write(f"| {sar_id} | {opt_id} | {ws.sum() * px:.3f} | {wo.sum() * px:.3f} | {inter * px:.3f} | {inter / union if union else 0:.3f} | 유효 겹침 {valid.mean() * 100:.0f}% |\n")

    out.write("\n## 읽는 법\n\n")
    out.write("- 수체 면적은 각 장면 자기 촬영범위 기준이라 센서 간 직접 비교는 위 쌍 비교(SAR 범위 안)로 본다. 광학은 저수지 전체를 담고 SAR 은 댐 서쪽 하류 6×7 km 만 담는다.\n")
    out.write("- 위성 수위는 지점 AOI 가 촬영범위에 들어온 경우만 나온다. SAR 은 조정지댐(jojeongji)만 들어온다.\n")
    out.write("- Umbra 는 인수 ICEYE SAR 모델을 그대로 쓴 것이라(동일 X밴드, 학습 자료는 ICEYE 부산) 결과는 참고용이며, 광학과의 IoU 가 그 신뢰도의 근거다.\n")
    out.write("- 9-20 광학은 지표반사도 자료가 없어 TOA 반사도로 넣었다. 모델 학습 조건과 달라 참고용이다.\n")
    Path(a.out).write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
