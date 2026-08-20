#!/usr/bin/env python3
"""과제 B: 핸드오버 05_l1_pre 8씬(ICEYE 4 + PlanetScope 4) → Module0 형식 학습 페어 v2.

형식 (scripts/iceye_geocode.py make_patch_pairs 관례):
  {"<SENSOR>": {"train": {"image": [...], "label": [...]},
                "test":  {"image": [...], "label": [...]}}}
  image = float32 (512,512,C), 결측 -9999 (Module0 KNN 보간 규약) / label = uint8 (512,512)

스플릿 (ATBD 준용): SAR test=20200416, 광학 test=20200325, 나머지 train.
패치: 512, 25% overlap(step 384), 씬당 상한 300, 수체≥1%, 유효≥80%, 시드 42.

유효성 규약 (핸드오버 코드 실측 — validate.py / detect_water.py / gamma_core.py):
  · lsmap 사이드카: 값 1 = 정상 가시 → **1만 유지**, 그 외(layover/shadow 계열,
    0=미매핑) 제외.  (validate.py:229 `valid &= (lsv == 1)` 이 정본)
  · 3-class 마스크(busan_<date>_mask3_utm.tif): 0=유효 SAR, 1=제외, 2=해양
    (detect_water.py:38). GT 는 mask!=0 에서 0 으로 강제 (validate.py:232).
  · GT 라벨: 수체=1, nodata(ICEYE 15 / Planet 3) 는 무효 화소.
  · 광학은 4밴드 전부 nodata(6.5535) 아닌 화소만 유효.

대형 tif 는 512행 윈도우 단위 2-pass(지표 산정 → 선정 패치만 재독)로 읽는다.

산출: data/ingest/WBMS_ai_data_v2_{iceye,planet}.pkl + WBMS_ai_data_v2_stats.json
"""
from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
H = REPO / "data/incoming/handover"
OUT_DIR = REPO / "data/ingest"
NODATA = -9999.0

PATCH = 512
STEP = PATCH - int(PATCH * 0.25)   # 384 — make_patch_pairs 와 동일
CAP = 300
WATER_MIN = 0.01
VALID_MIN = 0.8
SEED = 42

SENSORS = {
    "ICEYE": {
        "pkl": OUT_DIR / "WBMS_ai_data_v2_iceye.pkl",
        "test_dates": {"20200416"},
        "scenes": {
            d: dict(
                # ⚠️ 반드시 시각(숫자)으로 끝나는 파일만 — `T*.tif` 는 *_lsmap.tif 도
                #    매칭해 glob 순서에 따라 사이드카를 영상으로 오인한다 (실측 사고)
                img=next((H / f"05_l1_pre/iceye_pre/2020/{d}").glob(
                    f"Pre_Busan_ICEYE_{d}T*[0-9].tif")),
                label=H / f"06_aux/Labels_GT/{d}_Label.tif",
                mask=H / f"06_aux/Masks/Img_DEM_3class_utm/busan_{d}_mask3_utm.tif",
            ) for d in ("20200302", "20200330", "20200415", "20200416")
        },
    },
    "PlanetScope": {
        "pkl": OUT_DIR / "WBMS_ai_data_v2_planet.pkl",
        "test_dates": {"20200325"},
        "scenes": {
            d: dict(
                img=next((H / f"05_l1_pre/planet_pre/2020/{d}").glob(
                    f"Pre_Busan_PlanetScope_{d}T*[0-9].tif")),
                label=next((H / "06_aux/Labels_GT_planet").glob(f"{d}*label.tif")),
                mask=None,
            ) for d in ("20200218", "20200312", "20200325", "20200414")
        },
    },
}
# lsmap 사이드카는 SAR 만: Pre_*.tif 와 같은 stem + _lsmap.tif
for d, s in SENSORS["ICEYE"]["scenes"].items():
    ls = s["img"].with_name(s["img"].stem + "_lsmap.tif")
    s["lsmap"] = ls if ls.exists() else None
for s in SENSORS["PlanetScope"]["scenes"].values():
    s["lsmap"] = None


def same_grid(a, b, tol=1e-4):
    return ((a.width, a.height) == (b.width, b.height)
            and str(a.crs) == str(b.crs)
            and all(abs(x - y) <= tol for x, y in zip(a.transform[:6],
                                                      b.transform[:6])))


def scene_masks(win_arrays, sensor, lbl_nodata, img_nodata):
    """윈도우 배열 묶음 → (img_observed, valid, water). 규약은 모듈 docstring 참조."""
    wi, wl, wls, wm = win_arrays
    if sensor == "ICEYE":
        img_ok = wi[0] > NODATA + 1
        obs = img_ok if wls is None else (img_ok & (wls == 1))
    else:
        # ⚠️ Pre 화소의 센티널은 float32(65535*1e-4)=6.5534997 인데 메타 nodata 는
        #    float32(6.5535)=6.5535002 — 1 ULP 어긋나 등호 비교가 전부 놓친다(실측).
        #    허용오차 비교가 정본.
        img_ok = np.isfinite(wi).all(axis=0)
        for bnd in wi:
            img_ok &= np.abs(bnd - np.float32(img_nodata)) > 1e-3
        obs = img_ok
    lbl_ok = wl != lbl_nodata
    valid = obs & lbl_ok
    water = wl == 1
    if wm is not None:                       # 3-class: 0 만 유효 SAR, gt 는 !=0 → 0
        valid &= wm == 0
        water &= wm == 0
    return obs, valid, water & valid


def process_scene(sensor, d, paths, split):
    import rasterio
    srcs = {}
    srcs["img"] = rasterio.open(paths["img"])
    srcs["label"] = rasterio.open(paths["label"])
    srcs["lsmap"] = rasterio.open(paths["lsmap"]) if paths["lsmap"] else None
    srcs["mask"] = rasterio.open(paths["mask"]) if paths["mask"] else None
    img = srcs["img"]
    if img.dtypes[0] != "float32" or paths["img"].stem.endswith("_lsmap"):
        raise SystemExit(f"{sensor} {d}: 영상 파일 오인({paths['img'].name}, "
                         f"{img.dtypes[0]}) — 중단")
    for k, ds in srcs.items():
        if ds is not None and not same_grid(img, ds):
            raise SystemExit(f"{sensor} {d}: {k} 격자 불일치 — 중단")
    lbl_nodata = int(srcs["label"].nodata) if srcs["label"].nodata is not None else 255
    img_nodata = float(img.nodata) if img.nodata is not None else NODATA
    W, Hh, C = img.width, img.height, img.count

    def read_band(ds, r0, r1, all_bands=False):
        win = ((r0, r1), (0, W))
        return ds.read(window=win) if all_bands else ds.read(1, window=win)

    # ── pass 1: 후보 지표 (행 윈도우 순차) ─────────────────────────────
    r0s = list(range(0, Hh - PATCH + 1, STEP))
    c0s = list(range(0, W - PATCH + 1, STEP))
    metrics = {}
    dec_hist = np.zeros(256, dtype=np.int64)     # 라벨 값 분포 (씬 전수)
    obs_n = valid_n = tot_n = 0
    for r0 in r0s:
        wi = read_band(srcs["img"], r0, r0 + PATCH, all_bands=True)
        wl = read_band(srcs["label"], r0, r0 + PATCH)
        wls = read_band(srcs["lsmap"], r0, r0 + PATCH) if srcs["lsmap"] else None
        wm = read_band(srcs["mask"], r0, r0 + PATCH) if srcs["mask"] else None
        obs, valid, water = scene_masks((wi, wl, wls, wm), sensor, lbl_nodata, img_nodata)
        core = slice(0, STEP) if r0 != r0s[-1] else slice(0, PATCH)   # 중복행 제외 집계
        dec_hist += np.bincount(wl[core].ravel(), minlength=256)
        obs_n += int(obs[core].sum()); valid_n += int(valid[core].sum())
        tot_n += obs[core].size
        for c0 in c0s:
            sl = (slice(None), slice(c0, c0 + PATCH))
            metrics[(r0, c0)] = (float(valid[sl].mean()), float(water[sl].mean()))

    # ── 선정: make_patch_pairs 와 동일한 셔플·순차 수락 ────────────────
    cands = [(r0, c0) for r0 in r0s for c0 in c0s]
    rng = np.random.default_rng(SEED)
    rng.shuffle(cands)
    selected = []
    for rc in cands:
        if len(selected) >= CAP:
            break
        vf, wf = metrics[rc]
        if vf >= VALID_MIN and wf >= WATER_MIN:
            selected.append(rc)
    sel_set = {}
    for i, rc in enumerate(selected):
        sel_set.setdefault(rc[0], []).append((i, rc[1]))

    # ── pass 2: 선정 패치 페이로드 (행 윈도우 순차 재독) ───────────────
    images = [None] * len(selected)
    labels = [None] * len(selected)
    wfs = [0.0] * len(selected)
    vfs = [0.0] * len(selected)
    for r0 in sorted(sel_set):
        wi = read_band(srcs["img"], r0, r0 + PATCH, all_bands=True)
        wl = read_band(srcs["label"], r0, r0 + PATCH)
        wls = read_band(srcs["lsmap"], r0, r0 + PATCH) if srcs["lsmap"] else None
        wm = read_band(srcs["mask"], r0, r0 + PATCH) if srcs["mask"] else None
        obs, valid, water = scene_masks((wi, wl, wls, wm), sensor, lbl_nodata, img_nodata)
        for i, c0 in sel_set[r0]:
            p = np.transpose(wi[:, :, c0:c0 + PATCH],
                             (1, 2, 0)).astype(np.float32).copy()
            p[~obs[:, c0:c0 + PATCH]] = NODATA      # Module0 결측 규약
            images[i] = p
            labels[i] = water[:, c0:c0 + PATCH].astype(np.uint8)
            vfs[i], wfs[i] = metrics[(r0, c0)]
    for ds in srcs.values():
        if ds is not None:
            ds.close()

    stat = dict(
        split=split, file=str(paths["img"].relative_to(REPO)),
        grid=f"{W}x{Hh}", bands=C,
        n_candidates=len(cands), n_accepted=len(selected),
        water_frac_mean=float(np.mean(wfs)) if selected else 0.0,
        water_frac_median=float(np.median(wfs)) if selected else 0.0,
        valid_frac_mean=float(np.mean(vfs)) if selected else 0.0,
        scene_label_hist={str(v): int(n) for v, n in enumerate(dec_hist) if n},
        scene_observed_frac=obs_n / tot_n,
        scene_valid_frac=valid_n / tot_n,
        lsmap=bool(paths["lsmap"]), mask3=bool(paths["mask"]),
    )
    print(f"  {sensor} {d} [{split}] accepted {len(selected)}/{len(cands)} "
          f"water_mean={stat['water_frac_mean']:.3f} valid_mean={stat['valid_frac_mean']:.3f}")
    return images, labels, stat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {
        "generated": date.today().isoformat(),
        "script": "scripts/make_wbms_training_pairs.py",
        "spec": dict(patch=PATCH, step=STEP, overlap="25%", cap_per_scene=CAP,
                     water_min=WATER_MIN, valid_min=VALID_MIN, seed=SEED,
                     missing_value=NODATA,
                     split="ATBD: SAR test=20200416, optical test=20200325"),
        "conventions": {
            "lsmap": "1=정상 가시(유지), 그 외 제외 — validate.py:229 `valid &= (lsv==1)` "
                     "실측 (지시문 'lsmap==1 제외' 는 부호 반대 — 코드 정본을 따름). "
                     "제외 화소는 image=-9999 처리",
            "mask3": "0=유효 SAR, 1=제외, 2=해양 (detect_water.py:38); "
                     "mask!=0 은 무효 + GT 0 강제 (validate.py:232)",
            "label": "수체=1 / ICEYE nodata=15 / Planet nodata=3 — nodata 는 무효 화소",
            "optical": "4밴드 전부 nodata(6.5535) 아닌 화소만 유효",
        },
        "sensors": {},
    }
    for sensor, cfg in SENSORS.items():
        ai = {sensor: {"train": {"image": [], "label": []},
                       "test": {"image": [], "label": []}}}
        sstats = {}
        for d in sorted(cfg["scenes"]):
            split = "test" if d in cfg["test_dates"] else "train"
            imgs, lbls, st = process_scene(sensor, d, cfg["scenes"][d], split)
            ai[sensor][split]["image"].extend(imgs)
            ai[sensor][split]["label"].extend(lbls)
            sstats[d] = st
        with open(cfg["pkl"], "wb") as fh:
            pickle.dump(ai, fh)
        all_stats["sensors"][sensor] = dict(
            pkl=str(cfg["pkl"].relative_to(REPO)),
            pkl_bytes=cfg["pkl"].stat().st_size,
            train_patches=len(ai[sensor]["train"]["image"]),
            test_patches=len(ai[sensor]["test"]["image"]),
            scenes=sstats)
        print(f"{sensor}: train {all_stats['sensors'][sensor]['train_patches']} / "
              f"test {all_stats['sensors'][sensor]['test_patches']} → {cfg['pkl'].name} "
              f"({cfg['pkl'].stat().st_size/1e6:.0f} MB)")
        del ai
    stats_path = OUT_DIR / "WBMS_ai_data_v2_stats.json"
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print(f"done: {stats_path}")


if __name__ == "__main__":
    main()
