"""대청댐 본댐 — DEM(NGII 3 m) 수위-면적 곡선 vs 위성 공통 격자 탐지 면적.

사용자 아이디어("수위↔면적 관계를 알면 면적 시계열을 수위 예측에 의존시킬 수 있다") 검증용.
위성 장면이 아니라 DEM 으로 곡선을 만들고, 2022년 5월 장면(마스크가 안정적인 배치)에서 맞는지 본다.
"""
import json
import sys

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import from_bounds
from scipy import ndimage

ROOT = "/home/innonew98/micro-satellite"
DEM = f"{ROOT}/data/incoming/handover/06_aux/DEM_daecheong_ngii3m/daecheong_ngii_32652_3m_topofilled.tif"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dcd_hypsometry.json"

a = json.load(open(f"{ROOT}/regions/geum_daecheong/satellite/area_series.json"))
blk = a["sensors"]["PlanetScope"]
fb = blk["frame"]["bounds_epsg32652"]
rows = {r["date"]: r for r in blk["rows"]}
with rasterio.open(DEM) as ds:
    win = from_bounds(*fb, transform=ds.transform)
    dem = ds.read(1, window=win, masked=True).filled(np.nan).astype(np.float32)
H, W = dem.shape
cell = 9.0  # m² (3 m 화소)
print("DEM window", dem.shape, "cells", H * W)


def load_mask(date: str) -> np.ndarray:
    im = np.array(Image.open(f"{ROOT}/regions/geum_daecheong/satellite/masks/PlanetScope/{date.replace('-', '')}_PlanetScope.png")) > 127
    ys = (np.arange(H) * im.shape[0] / H).astype(int)
    xs = (np.arange(W) * im.shape[1] / W).astype(int)
    return im[ys][:, xs]


masks = {d: load_mask(d) for d in rows}
dates22 = sorted(d for d in rows if d.startswith("2022"))
core = np.logical_and.reduce([masks[d] for d in dates22])
union = np.logical_or.reduce([masks[d] for d in dates22])
print(f"2022 5장 항상 물 {core.sum() * cell / 1e6:.2f} km² / 한 번이라도 물 {union.sum() * cell / 1e6:.2f} km²")
cv = dem[core]
cv = cv[~np.isnan(cv)]
print("항상-물 구역 DEM 백분위 1/5/25/50/75/95/99:", np.round(np.percentile(cv, [1, 5, 25, 50, 75, 95, 99]), 2))
main_core = core & (dem > 60)   # 본댐 호수(EL 70 m대) — 조정지·하류(EL 30 m대)와 분리
low_core = core & (dem <= 60)
mc = dem[main_core]
h0 = float(np.nanmedian(mc))
print(f"본댐 핵심 구역 {main_core.sum() * cell / 1e6:.2f} km², DEM 중앙값(=DEM 이 채운 호수면) {h0:.2f} m, p5/p95 {np.percentile(mc, 5):.2f}/{np.percentile(mc, 95):.2f}")
print(f"조정지·하류 핵심 구역 {low_core.sum() * cell / 1e6:.2f} km², DEM 중앙값 {np.nanmedian(dem[low_core]) if low_core.any() else float('nan'):.2f} m")


def flooded(h: float) -> np.ndarray:
    """수위 h 에서 본댐 핵심 구역과 연결된 (DEM ≤ h) 영역 — 댐체(마루 EL 83 m)로 하류와는 끊긴다."""
    wet = (dem <= h) & (dem > 55)
    lab, _n = ndimage.label(wet)
    ids = np.unique(lab[main_core])
    ids = ids[ids > 0]
    return np.isin(lab, ids)


levels = np.arange(68.0, 80.01, 0.5)
curve = []
for h in levels:
    f = flooded(float(h))
    curve.append((float(h), round(f.sum() * cell / 1e6, 3)))
print("=== DEM 수위-면적 곡선 (본댐 호수, 공통 격자 안)")
for h, A in curve:
    print(f"  h={h:5.1f} m  A={A:7.2f} km²")
dom = flooded(80.0)
print("=== 장면별: 실측 수위 → DEM 곡선 면적 vs 위성 탐지 면적 (본댐 영역 안 / 격자 전체)")
cmp = []
for d in sorted(rows):
    lvl = rows[d]["gauge_levels"]["대청댐"]
    A_dem = flooded(lvl).sum() * cell / 1e6
    det_dom = (masks[d] & dom).sum() * cell / 1e6
    det_all = masks[d].sum() * cell / 1e6
    cmp.append({"date": d, "level": lvl, "A_dem": round(A_dem, 3), "det_main": round(det_dom, 3), "det_all": round(det_all, 3), "official": rows[d]["area_km2_common_frame"]})
    print(f"  {d} 수위 {lvl:6.2f}  DEM곡선 {A_dem:6.2f}  탐지(본댐영역) {det_dom:6.2f}  탐지(격자전체) {det_all:6.2f}  공식계열 {rows[d]['area_km2_common_frame']:6.2f}")
json.dump({"h0_dem_surface": h0, "curve": curve, "scenes": cmp, "frame": fb}, open(OUT, "w"), ensure_ascii=False, indent=1)
print("saved", OUT)
