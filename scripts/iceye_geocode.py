#!/usr/bin/env python3
"""ICEYE SLC sigma0(dB) → EPSG:32652 3m 지오코딩 + 라벨 패치 페어링.

1_water_body_detection/README.md 절차의 2(LUT)·7(정사보정) 단계 실구현.
scripts/iceye_slc_ingest.py 산출물(멀티룩 sigma0 dB slant-range GeoTIFF)을
h5 내장 지오레퍼런싱 스플라인(픽셀→위경도)으로 라벨 고유 3m 격자에 리샘플한다.

파이프라인
  1. h5의 lat/lon_spline pickle에서 포인트 배열만 추출(스텁 언피클러 —
     2020년산 scipy 객체를 현재 scipy로 직접 쓰지 않고 재구축해 버전 비호환 회피)
  2. 5개 코너/센터 GCP로 축 순서·좌표 규약 자가검증 (오차 한도 초과 시 중단)
  3. 역방향 LUT: (E,N)→(az,rg)를 coarse 격자에 평가 후 bilinear 업샘플
  4. 청크 단위 리샘플 → Processed_<date>_ICEYE.tif (float32, nodata -9999)
  5. NAS 20200302_Input.tif와 창 단위 상관 교차검증 (있을 때)
  6. 라벨과 512×512(25% overlap) 패치 페어링 → Module0 형식 pkl (test 스플릿)
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_H5 = REPO / "ICEYE_X5_SLC_SM_23467_20200302T183857-008.h5"
NODATA = -9999.0

# 라벨 고유 3m 격자 (scripts/materialize_nas_iceye.py AUDITED_INPUTS와 동일)
TARGET = {
    "20200302": dict(width=19571, height=24857, crs="EPSG:32652",
                     transform=[3.0, 0.0, 473010.36353185785,
                                0.0, -3.0, 3921935.664009152]),
}


# ---------------------------------------------------------------- unpickle
class _Stub:
    def __init__(self, *a, **k):
        self._args = a

    def __call__(self, *a, **k):
        return _Stub(*a)

    def __setstate__(self, state):
        self._state = state


def _stub_to_array(stub: "_Stub") -> np.ndarray | None:
    """dill이 _reconstruct를 스텁으로 감싼 경우: __setstate__로 들어온
    (version, shape, dtype, is_fortran, data_bytes)에서 배열을 직접 복원."""
    st = getattr(stub, "_state", None)
    if (isinstance(st, tuple) and len(st) == 5 and isinstance(st[4], bytes)
            and isinstance(st[2], np.dtype) and isinstance(st[1], tuple)):
        arr = np.frombuffer(st[4], dtype=st[2])
        if arr.size != int(np.prod(st[1] or (arr.size,))):
            return None
        return np.array(arr.reshape(st[1], order="F" if st[3] else "C"))
    return None


def _extract_arrays(blob: bytes) -> list[np.ndarray]:
    """pickle 그래프에서 ndarray를 수집. scipy/dill 클래스는 전부 스텁 처리."""

    class Extractor(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy"):
                mod = __import__(module, fromlist=[name])
                return getattr(mod, name)
            return _Stub

        def persistent_load(self, pid):
            return _Stub()

    root = Extractor(io.BytesIO(blob)).load()
    arrays: list[np.ndarray] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            arrays.append(o)
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, _Stub):
            arr = _stub_to_array(o)
            if arr is not None:
                arrays.append(arr)
            else:
                stack.append(getattr(o, "_args", ()))
                stack.append(getattr(o, "_state", None))
    return arrays


def load_forward_spline(f: h5py.File):
    """h5 스플라인에서 (native_points[N,2], lat[N], lon[N]) 복원."""
    packs = {}
    for key in ("lat_spline", "lon_spline"):
        arrays = _extract_arrays(bytes(f[key]["data_0"][0]))
        pts = [a for a in arrays if a.ndim == 2 and a.shape[1] == 2
               and a.shape[0] > 1000 and a.dtype == np.float64]
        vals = [a.ravel() for a in arrays if a.dtype == np.float64
                and a.size > 1000 and (a.ndim == 1 or
                                       (a.ndim == 2 and a.shape[1] == 1))]
        two = [a.astype(np.float64) for a in arrays
               if a.ndim == 1 and a.shape == (2,)]
        if not pts or not vals or len(two) < 2:
            raise RuntimeError(f"{key}: 배열 추출 실패 "
                               f"(shapes={[a.shape for a in arrays]})")
        packs[key] = (pts[0], vals[0], two)
    pts_norm, lat, two = packs["lat_spline"]
    _, lon, _ = packs["lon_spline"]
    if pts_norm.shape[0] != lat.shape[0] or lat.shape[0] != lon.shape[0]:
        raise RuntimeError("lat/lon 스플라인 포인트 수 불일치")
    return pts_norm, lat, lon, two


def resolve_native_points(pts_norm, lat, lon, two, gcps):
    """offset/scale 배정·축 순서를 GCP 실측으로 결정. (points, order) 반환."""
    from scipy.interpolate import LinearNDInterpolator
    cands = []
    uniq = {tuple(a.round(4)): a for a in two}
    two = list(uniq.values())
    for i, s in enumerate(two):
        for j, o in enumerate(two):
            if i == j or np.any(np.abs(s) < 1e-6):
                continue
            cands.append(pts_norm * s + o)
    best = None
    for native in cands:
        tree = LinearNDInterpolator(native, np.c_[lat, lon])
        for order in ("az_rg", "rg_az"):
            err = 0.0
            ok = True
            for g in gcps:
                q = (g["az0"], g["rg0"]) if order == "az_rg" else (g["rg0"], g["az0"])
                ll = np.asarray(tree(np.array([q]))).reshape(-1)
                if ll.size != 2 or np.any(np.isnan(ll)):
                    ok = False
                    break
                err = max(err, float(np.hypot((ll[0] - g["lat"]) * 111_000,
                                              (ll[1] - g["lon"]) * 91_000)))
            if ok and (best is None or err < best[0]):
                best = (err, native, order)
    if best is None:
        raise RuntimeError("GCP와 정합하는 offset/scale/축순서 조합 없음")
    err, native, order = best
    print(f"  축순서={order}, GCP 최대오차 {err:.1f} m")
    if err > 60.0:
        raise RuntimeError(f"GCP 오차 {err:.1f} m > 60 m — 지오코딩 중단")
    return native, order


def read_gcps(f: h5py.File):
    gcps = []
    for k in ("coord_first_near", "coord_first_far", "coord_last_near",
              "coord_last_far", "coord_center"):
        c = f[k][:]  # [range_px(col), az_line(row), lat, lon], 1-기반
        gcps.append(dict(rg0=float(c[0]) - 1.0, az0=float(c[1]) - 1.0,
                         lat=float(c[2]), lon=float(c[3])))
    return gcps


# ---------------------------------------------------------------- geocode
def build_inverse_lut(native, order, lat, lon, tgt, stride):
    """coarse 타깃 격자에 (az,rg) LUT 평가."""
    from rasterio.warp import transform as warp_transform
    from scipy.interpolate import LinearNDInterpolator

    e, n = warp_transform("EPSG:4326", tgt["crs"], lon.tolist(), lat.tolist())
    azrg = native if order == "az_rg" else native[:, ::-1]
    inv = LinearNDInterpolator(np.c_[e, n], azrg)

    t = tgt["transform"]
    cols = np.unique(np.r_[np.arange(0, tgt["width"], stride), tgt["width"] - 1])
    rows = np.unique(np.r_[np.arange(0, tgt["height"], stride), tgt["height"] - 1])
    E = t[2] + (cols + 0.5) * t[0]
    N = t[5] + (rows + 0.5) * t[4]
    EE, NN = np.meshgrid(E, N)
    lut = inv(np.c_[EE.ravel(), NN.ravel()]).reshape(len(rows), len(cols), 2)
    print(f"  LUT {lut.shape[0]}x{lut.shape[1]} (stride {stride}), "
          f"유효 {100 * np.isfinite(lut[..., 0]).mean():.1f}%")
    return rows.astype(np.float64), cols.astype(np.float64), lut


def geocode(db_ml, looks, rows, cols, lut, tgt, out_path, chunk=1024):
    import rasterio
    from rasterio.transform import Affine
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import map_coordinates

    rg_az = RegularGridInterpolator((rows, cols), lut[..., 0],
                                    bounds_error=False, fill_value=np.nan)
    rg_rg = RegularGridInterpolator((rows, cols), lut[..., 1],
                                    bounds_error=False, fill_value=np.nan)
    # bilinear가 NODATA(-9999)와 유효값을 혼합하면 가드(< NODATA+1)를 통과하는
    # 쓰레기 dB가 생긴다 — 유효 마스크를 같이 리샘플해 오염 픽셀을 걸러낸다.
    # (20200302처럼 nodata 없는 장면은 오버헤드 0)
    has_nodata = bool((db_ml <= NODATA + 1).any())
    vmask_src = (db_ml > NODATA).astype(np.float32) if has_nodata else None
    t = tgt["transform"]
    profile = dict(driver="GTiff", width=tgt["width"], height=tgt["height"],
                   count=1, dtype="float32", crs=tgt["crs"],
                   transform=Affine(*t), nodata=NODATA,
                   compress="deflate", tiled=True,
                   blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    cgrid = np.arange(tgt["width"], dtype=np.float64)
    valid_total = 0
    with rasterio.open(out_path, "w", **profile) as dst:
        for r0 in range(0, tgt["height"], chunk):
            r1 = min(r0 + chunk, tgt["height"])
            rr, cc = np.meshgrid(np.arange(r0, r1, dtype=np.float64),
                                 cgrid, indexing="ij")
            pts = np.c_[rr.ravel(), cc.ravel()]
            az = rg_az(pts) / looks[0]
            rg = rg_rg(pts) / looks[1]
            good = (np.isfinite(az) & np.isfinite(rg)
                    & (az >= 0) & (az <= db_ml.shape[0] - 1)
                    & (rg >= 0) & (rg <= db_ml.shape[1] - 1))
            out = np.full(pts.shape[0], NODATA, dtype=np.float32)
            if good.any():
                samp = map_coordinates(db_ml, [az[good], rg[good]],
                                       order=1, mode="nearest")
                if vmask_src is not None:
                    vw = map_coordinates(vmask_src, [az[good], rg[good]],
                                         order=1, mode="nearest")
                    samp[vw < 0.999] = NODATA
                samp[samp < NODATA + 1] = NODATA
                out[good] = samp
            valid_total += int((out > NODATA).sum())
            dst.write(out.reshape(r1 - r0, tgt["width"]), 1,
                      window=((r0, r1), (0, tgt["width"])))
            print(f"  geocode {r1}/{tgt['height']} rows", flush=True)
    print(f"  유효 픽셀 {valid_total:,} "
          f"({100 * valid_total / (tgt['width'] * tgt['height']):.1f}%)")


def cross_validate(out_path, ref_path, n_win=6, win=512):
    """NAS Input.tif와 창 단위 Pearson 상관."""
    import rasterio
    results = []
    with rasterio.open(out_path) as a, rasterio.open(ref_path) as b:
        rng = np.random.default_rng(42)
        tries = 0
        while len(results) < n_win and tries < 200:
            tries += 1
            r = int(rng.integers(0, a.height - win))
            c = int(rng.integers(0, a.width - win))
            wa = a.read(1, window=((r, r + win), (c, c + win)))
            wb = b.read(1, window=((r, r + win), (c, c + win)))
            m = (wa > NODATA) & (wb != 0)
            if m.mean() < 0.7:
                continue
            va, vb = wa[m], wb[m]
            r_p = float(np.corrcoef(va, vb)[0, 1])
            results.append(dict(row=r, col=c, pearson_r=r_p,
                                bias_db=float((va - vb).mean()),
                                mad_db=float(np.abs(va - vb).mean())))
    return results


# ---------------------------------------------------------------- pairing
def make_patch_pairs(out_path, label_path, pkl_path, patch=512,
                     water_min=0.01, valid_min=0.8, cap=400):
    import rasterio
    step = patch - int(patch * 0.25)  # Module0과 동일: 25% overlap
    cands = []
    with rasterio.open(out_path) as img, rasterio.open(label_path) as lbl:
        assert (img.width, img.height) == (lbl.width, lbl.height)
        for r0 in range(0, img.height - patch + 1, step):
            for c0 in range(0, img.width - patch + 1, step):
                cands.append((r0, c0))
        rng = np.random.default_rng(42)
        rng.shuffle(cands)
        images, labels, water_fracs = [], [], []
        for r0, c0 in cands:
            if len(images) >= cap:
                break
            wl = lbl.read(1, window=((r0, r0 + patch), (c0, c0 + patch)))
            wi = img.read(1, window=((r0, r0 + patch), (c0, c0 + patch)))
            valid = wi > NODATA
            wf = float((wl == 1).mean())
            if valid.mean() < valid_min or wf < water_min:
                continue
            wi = wi.copy()
            wi[~valid] = -9999.0  # Module0 결측 규약 (KNN 보간 대상)
            images.append(wi[..., None].astype(np.float32))
            labels.append((wl == 1).astype(np.uint8))
            water_fracs.append(wf)
    ai_data = {"ICEYE": {"train": {"image": [], "label": []},
                         "test": {"image": images, "label": labels}}}
    with open(pkl_path, "wb") as fh:
        pickle.dump(ai_data, fh)
    return dict(n_patches=len(images),
                water_frac_mean=float(np.mean(water_fracs)) if water_fracs else 0.0,
                pkl_bytes=Path(pkl_path).stat().st_size)


def save_preview(out_path, label_path, png_path, max_dim=1100):
    import rasterio
    from PIL import Image
    with rasterio.open(out_path) as img, rasterio.open(label_path) as lbl:
        stride = max(1, int(np.ceil(max(img.height, img.width) / max_dim)))
        a = img.read(1)[::stride, ::stride]
        l = lbl.read(1)[::stride, ::stride]
    valid = a > NODATA
    lo, hi = np.percentile(a[valid], [2, 98])
    g = np.zeros(a.shape, dtype=np.uint8)
    g[valid] = np.clip((a[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[l == 1] = (rgb[l == 1] * 0.4 + np.array([30, 144, 255]) * 0.6).astype(np.uint8)
    Image.fromarray(rgb).save(png_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument("--slant-tif", type=Path,
                    default=REPO / "data/ingest/20200302_ICEYE_sigma0db_slant.tif")
    ap.add_argument("--label", type=Path,
                    default=REPO / "data/nas_staging/20200302_Label.tif")
    ap.add_argument("--ref-input", type=Path,
                    default=REPO / "data/nas_staging/20200302_Input.tif")
    ap.add_argument("--looks", type=int, nargs=2, default=(2, 2))
    ap.add_argument("--lut-stride", type=int, default=16)
    ap.add_argument("--date", default="20200302")
    ap.add_argument("--patch-cap", type=int, default=400)
    args = ap.parse_args()

    tgt = TARGET.get(args.date)
    if tgt is None:
        raise SystemExit(f"{args.date}: TARGET에 라벨 격자가 등록돼 있지 않습니다 — "
                         f"materialize_nas_iceye.py의 감사 격자를 먼저 추가하세요")
    out_dir = REPO / "data" / "ingest"
    out_path = out_dir / f"Processed_{args.date}_ICEYE.tif"
    report = {"date": args.date, "target_grid": tgt}

    print("[1/6] 스플라인 배열 추출 + GCP 자가검증")
    with h5py.File(args.h5, "r") as f:
        # 다른 날짜 h5를 기본 --date/--slant-tif와 섞어 돌리면 기존 산출물이
        # 오정합 데이터로 조용히 덮어써진다 — 취득일 일치를 강제한다.
        acq = f["acquisition_start_utc"][()]
        h5_date = (acq.decode() if isinstance(acq, bytes) else str(acq))[:10].replace("-", "")
        if h5_date != args.date:
            raise SystemExit(f"--date {args.date} ≠ h5 취득일 {h5_date} — 중단")
        gcps = read_gcps(f)
        pts_norm, lat, lon, two = load_forward_spline(f)
    native, order = resolve_native_points(pts_norm, lat, lon, two, gcps)
    report["spline_points"] = int(native.shape[0])
    report["axis_order"] = order

    print("[2/6] 역방향 LUT 구축")
    # LUT 간격 48m(16px)는 스플라인 원 포인트 간격(~350m)보다 촘촘 —
    # bilinear 업샘플 오차가 원 지오로케이션 정밀도에 묻힌다
    rows, cols, lut = build_inverse_lut(native, order, lat, lon,
                                        tgt, args.lut_stride)

    print("[3/6] slant sigma0 로드")
    import rasterio
    with rasterio.open(args.slant_tif) as ds:
        prod = ds.tags().get("PRODUCT", "")
        if args.date not in prod and not args.slant_tif.name.startswith(args.date):
            raise SystemExit(f"slant tif({args.slant_tif.name})가 --date {args.date}와 "
                             f"불일치 (PRODUCT={prod!r}) — 중단")
        db_ml = ds.read(1)

    print("[4/6] 지오코딩 리샘플")
    geocode(db_ml, args.looks, rows, cols, lut, tgt, out_path)
    del db_ml

    if args.ref_input.exists():
        print("[5/6] NAS Input.tif 교차검증")
        cv = cross_validate(out_path, args.ref_input)
        report["cross_validation_vs_nas_input"] = cv
        for w in cv:
            print(f"  window({w['row']},{w['col']}): r={w['pearson_r']:.3f} "
                  f"bias={w['bias_db']:+.2f}dB mad={w['mad_db']:.2f}dB")
    else:
        print("[5/6] NAS Input.tif 없음 — 교차검증 생략")

    if args.label.exists():
        print("[6/6] 라벨 패치 페어링 + 프리뷰")
        pkl_path = out_dir / f"ICEYE_ai_data_{args.date}.pkl"
        report["patch_pairs"] = make_patch_pairs(
            out_path, args.label, pkl_path, cap=args.patch_cap)
        save_preview(out_path, args.label,
                     out_dir / f"preview_geocode_{args.date}.png")
        print(f"  {report['patch_pairs']}")
    else:
        print("[6/6] 라벨 없음 — 패치 페어링 생략")

    (out_dir / f"geocode_report_{args.date}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(f"done: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
