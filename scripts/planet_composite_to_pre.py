#!/usr/bin/env python3
"""PlanetScope SuperDove 8밴드 모자이크(composite.tif) → WBMS ① Pre 규격(4밴드 B,G,R,NIR 반사도).

K-water 가 보낸 대청댐 자료는 PSScene 8밴드 composite 다(Coastal Blue, Blue, Green I, Green, Yellow, Red,
Red Edge, NIR). 인수 모델은 4밴드(B,G,R,NIR) float32 반사도·3 m·EPSG:32652·무효 6.5535 를 받으므로
2·4·6·8 밴드를 뽑아 규격으로 맞춘다.

  - `*_sr_udm2` 묶음: 지표반사도 ×10000 (uint16) → ×1e-4.
  - `*_udm2`(SR 아님) 묶음: 방사휘도 → 각 장면 메타 XML 의 ps:reflectanceCoefficient(밴드별) 평균을 곱해
    대기상단(TOA) 반사도로 만든다. 모델은 지표반사도로 학습됐으므로 **참고용** — 사이드카에 표시한다.
  - UDM2 가 있으면 clear≠1 이면서 cloud/shadow/haze_heavy 인 화소를 무효(6.5535) 처리한다(선택, 기본 켬).

    .venv/bin/python scripts/planet_composite_to_pre.py --bundle <…_psscene_analytic_8b_sr_udm2 폴더> \
        --testbed dcd --output_dir data/incoming/kwater_202610/planet_pre
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INVALID = 6.5535
BAND_MAP_8B = {"B": 2, "G": 4, "R": 6, "NIR": 8}  # SuperDove 8밴드 → 4밴드


def log(msg: str) -> None:
    print(f"[planet→pre] {msg}", flush=True)


def read_scene_meta(bundle: Path) -> dict:
    """장면 XML 들에서 촬영시각·궤도방향·반사도 계수(8밴드)를 모은다."""

    xmls = sorted(bundle.glob("*_metadata_clip.xml")) or sorted(bundle.glob("*metadata*.xml"))
    acq, orbit, coeffs, sats = [], set(), [], set()
    for x in xmls:
        t = x.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"acquisitionDateTime>([^<]+)<", t)
        if m:
            acq.append(m.group(1).strip())
        m = re.search(r"orbitDirection>([^<]+)<", t)
        if m:
            orbit.add(m.group(1).strip())
        m = re.search(r"serialIdentifier>([^<]+)<", t)
        if m:
            sats.add(m.group(1).strip())
        bands = re.findall(r"bandNumber>(\d+)<.*?reflectanceCoefficient>([0-9.eE+-]+)<", t, re.S)
        if bands:
            coeffs.append({int(b): float(c) for b, c in bands})
    return {"acq": sorted(acq), "orbit": sorted(orbit), "coeffs": coeffs, "satellites": sorted(sats), "n_scenes": len(xmls)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="composite.tif 가 있는 psscene 폴더")
    ap.add_argument("--testbed", default="dcd")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--no_udm", action="store_true", help="UDM2 구름 마스킹 끄기")
    a = ap.parse_args()

    import rasterio

    bundle = Path(a.bundle)
    comp = bundle / "composite.tif"
    udm = bundle / "composite_udm2.tif"
    if not comp.is_file():
        log(f"composite.tif 없음: {bundle}"); return 2
    meta = read_scene_meta(bundle)
    is_sr = "_sr_" in bundle.name or bundle.name.endswith("_sr")
    if not meta["acq"]:
        log("촬영시각을 XML 에서 못 읽음 — 중단"); return 2
    first = meta["acq"][0]
    dt = datetime.fromisoformat(first.replace("Z", "+00:00")).astimezone(timezone.utc)
    stamp = dt.strftime("%Y%m%dT%H%M%S"); date8 = stamp[:8]
    tb_code = {"dcd": "DCD", "busan": "Busan"}.get(a.testbed, a.testbed.upper())
    stem = f"Pre_{tb_code}_PlanetScope_{stamp}"
    out_dir = Path(a.output_dir) / date8[:4] / date8
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(comp) as ds:
        if ds.count < 8:
            log(f"8밴드가 아님({ds.count}) — 4밴드 입력이면 그대로 1..4 를 쓴다")
            idx = [1, 2, 3, 4]
        else:
            idx = [BAND_MAP_8B[k] for k in ("B", "G", "R", "NIR")]
        raw = ds.read(idx).astype(np.float32)  # (4, H, W)
        nodata = ds.nodata if ds.nodata is not None else 0
        profile = ds.profile
        crs, transform, res = ds.crs, ds.transform, ds.res
        log(f"{bundle.name}: {ds.count}밴드 {ds.width}x{ds.height} {crs} res={res} nodata={nodata} | SR={is_sr} | 장면 {meta['n_scenes']}개 {meta['satellites']}")
    invalid = np.all(raw == nodata, axis=0)
    if is_sr:
        refl = raw * 1e-4
        value_type, note = "reflectance", "surface reflectance (SR) ×1e-4"
    else:
        # 방사휘도 → TOA 반사도: 장면별 계수 평균 (모자이크라 장면 경계에서 수 % 차이 가능)
        if not meta["coeffs"]:
            log("reflectanceCoefficient 없음 — TOA 변환 불가, 중단"); return 2
        coef = np.array([np.mean([c[b] for c in meta["coeffs"] if b in c]) for b in idx], dtype=np.float32)
        refl = raw * coef[:, None, None]
        value_type, note = "toa_reflectance", f"radiance × mean reflectanceCoefficient {coef.round(6).tolist()} (TOA, 대기보정 안 됨 — 참고용)"
    refl = np.clip(refl, 0, 2.0).astype(np.float32)
    udm_note = "UDM2 없음"
    if udm.is_file() and not a.no_udm:
        with rasterio.open(udm) as u:
            if u.count >= 6:
                clear = u.read(1); shadow = u.read(3); haze_heavy = u.read(5); cloud = u.read(6)
                bad = (cloud == 1) | (shadow == 1) | (haze_heavy == 1)
                frac = float(bad[~invalid].mean()) if (~invalid).any() else 0.0
                invalid |= bad
                udm_note = f"UDM2 구름·그림자·짙은연무 {frac * 100:.1f}% 무효 처리"
    refl[:, invalid] = INVALID
    valid_frac = float((~invalid).mean())

    pre_path = out_dir / f"{stem}.tif"
    profile.update(count=4, dtype="float32", nodata=INVALID, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    profile.pop("photometric", None)
    with rasterio.open(pre_path, "w", **profile) as dst:
        dst.write(refl)
        dst.update_tags(AREA_OR_POINT="Area", BAND_ORDER="B,G,R,NIR", SOURCE=f"PlanetScope 8b composite {bundle.name}")
    v = refl[:, ~invalid]
    stats = {k: [round(float(np.percentile(v[i], q)), 4) for q in (5, 50, 95)] for i, k in enumerate(("B", "G", "R", "NIR"))}
    meta_out = {
        "Document": "Processed", "TestbedName": tb_code, "SatelliteName": "PlanetScope",
        "OrbitNumber": ",".join(meta["satellites"]) or "", "OrbitDirection": (meta["orbit"] or ["DESCENDING"])[0].upper(),
        "ImageDate": date8,
        "Entry": {"date": date8, "AcquisitionDateTime": dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "PixelSize_m": float(res[0]),
                  "Bands": 4, "BandOrder": "B,G,R,NIR", "ValueType": value_type, "Normalized": "false", "CRS": str(crs),
                  "Scene_UL": {"x": str(transform.c), "y": str(transform.f)}, "Scene_Posting": {"x": str(res[0]), "y": str(res[1])}},
        "Provenance": {"stage": "planet_composite_to_pre", "input_file": f"{bundle.name}/composite.tif",
                       "meta_source": ";".join(p.name for p in sorted(bundle.glob("*_metadata_clip.xml"))[:1]),
                       "source_bands_8b": idx, "scaling": note, "udm2": udm_note, "scene_acquisitions": meta["acq"]},
        "Quality": {"valid_fraction": round(valid_frac, 4), "reflectance_p5_p50_p95": stats},
    }
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from PIL import Image
        rgb = np.stack([refl[2], refl[1], refl[0]], axis=-1)
        lo, hi = np.percentile(rgb[~invalid], 2), np.percentile(rgb[~invalid], 98)
        q = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1); q[invalid] = 0
        s = max(1, int(np.ceil(max(q.shape[:2]) / 1200)))
        Image.fromarray((q[::s, ::s] * 255).astype(np.uint8)).save(out_dir / f"{stem}_quicklook.png")
    except Exception as exc:  # noqa: BLE001
        log(f"퀵룩 생략: {exc}")
    log(f"완료 → {pre_path} | 유효 {valid_frac * 100:.1f}% | {note} | {udm_note} | NIR p50 {stats['NIR'][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
