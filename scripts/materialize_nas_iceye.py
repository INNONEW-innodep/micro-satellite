#!/usr/bin/env python3
"""Materialize a small, provenance-rich ICEYE water-body demo bundle.

The script expects a read-only local mount or staging copy of the NAS
``ICEYE_WB`` directory.  It never writes below the source directories.  Four
binary label rasters are intersected in map coordinates and nearest-sampled
onto one aspect-preserving preview grid.  Raw ICEYE quicklooks are exported as
browse-only context; they are deliberately not presented as pixel-aligned
imagery.

Example::

    python scripts/materialize_nas_iceye.py \
      --iceye-wb-root /read-only/ICEYE_WB \
      --raw-quicklook-root /read-only/ICEYE \
      --output ui_next/assets/nas/iceye_2020

The output contains four mask PNGs, four label-validity PNGs, up to four
browse previews, and ``manifest.json``.  Full-resolution source TIFFs are not
copied to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window

DATES = ("20200302", "20200330", "20200415", "20200416")
DATASET_LOGICAL_ROOT = "신규데이터_위성영상_라벨링/ICEYE_WB"
RAW_LOGICAL_ROOT = "신규데이터_위성영상_원본/ICEYE"
WATER_VALUE = 1
CHUNK_ROWS = 512

# Read-only TIFF IFD audit captured from the NAS on 2026-08-10.  This lets the
# small label-only staging workflow preserve an honest pair-grid check without
# copying 5.4 GB of input pixels.  When Images/*.tif is present the script
# ignores this snapshot and verifies the live file instead.
AUDITED_INPUTS: dict[str, dict[str, Any]] = {
    "20200302": {
        "width": 19_571,
        "height": 24_857,
        "bands": 1,
        "dtypes": ["float32"],
        "crs": "EPSG:32652",
        "transform": [3.0, 0.0, 473010.36353185785, 0.0, -3.0, 3921935.664009152],
        "bounds": [473010.36353185785, 3847364.664009152, 531723.3635318578, 3921935.664009152],
        "resolution": [3.0, 3.0],
        "nodata": 0,
        "block_shapes": [[128, 128]],
        "compression": "LZW",
        "pixel_count": 486_476_347,
        "bytes": 1_463_390_651,
        "bigtiff": False,
    },
    "20200330": {
        "width": 17_356,
        "height": 24_488,
        "bands": 1,
        "dtypes": ["float32"],
        "crs": "EPSG:32652",
        "transform": [3.0, 0.0, 476715.6203433549, 0.0, -3.0, 3916319.282163941],
        "bounds": [476715.6203433549, 3842855.282163941, 528783.6203433549, 3916319.282163941],
        "resolution": [3.0, 3.0],
        "nodata": 0,
        "block_shapes": [[128, 128]],
        "compression": "LZW",
        "pixel_count": 425_013_728,
        "bytes": 1_125_649_675,
        "bigtiff": False,
    },
    "20200415": {
        "width": 16_703,
        "height": 23_853,
        "bands": 1,
        "dtypes": ["float32"],
        "crs": "EPSG:32652",
        "transform": [3.0, 0.0, 479038.67391907226, 0.0, -3.0, 3915251.9979939666],
        "bounds": [479038.67391907226, 3843692.9979939666, 529147.6739190723, 3915251.9979939666],
        "resolution": [3.0, 3.0],
        "nodata": 0,
        "block_shapes": [[128, 128]],
        "compression": "LZW",
        "pixel_count": 398_416_659,
        "bytes": 1_207_518_571,
        "bigtiff": False,
    },
    "20200416": {
        "width": 20_440,
        "height": 24_684,
        "bands": 1,
        "dtypes": ["float32"],
        "crs": "EPSG:32652",
        "transform": [3.0000000000000027, 0.0, 472996.88501411973, 0.0, -3.0, 3916822.0100970687],
        "bounds": [472996.88501411973, 3842770.0100970687, 534316.8850141198, 3916822.0100970687],
        "resolution": [3.0000000000000027, 3.0],
        "nodata": 0,
        "block_shapes": [[128, 128]],
        "compression": "LZW",
        "pixel_count": 504_540_960,
        "bytes": 1_622_083_501,
        "bigtiff": True,
    },
}


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _json_number(value: Any) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _raster_metadata(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    return {
        "width": dataset.width,
        "height": dataset.height,
        "bands": dataset.count,
        "dtypes": list(dataset.dtypes),
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "transform": [float(value) for value in tuple(dataset.transform)[:6]],
        "bounds": [
            float(dataset.bounds.left),
            float(dataset.bounds.bottom),
            float(dataset.bounds.right),
            float(dataset.bounds.top),
        ],
        "resolution": [float(abs(dataset.res[0])), float(abs(dataset.res[1]))],
        "nodata": _json_number(dataset.nodata),
        "block_shapes": [list(shape) for shape in dataset.block_shapes],
        "compression": dataset.compression.name if dataset.compression else "NONE",
        "pixel_count": dataset.width * dataset.height,
    }


def _same_crs(left: rasterio.io.DatasetReader, right: rasterio.io.DatasetReader) -> bool:
    return left.crs == right.crs


def _same_transform(
    left: rasterio.io.DatasetReader, right: rasterio.io.DatasetReader
) -> bool:
    return bool(
        np.allclose(
            np.asarray(tuple(left.transform)[:6]),
            np.asarray(tuple(right.transform)[:6]),
            rtol=0.0,
            atol=1e-8,
        )
    )


def _pair_alignment(
    image: rasterio.io.DatasetReader, label: rasterio.io.DatasetReader
) -> dict[str, Any]:
    shape_equal = (image.height, image.width) == (label.height, label.width)
    crs_equal = _same_crs(image, label)
    transform_equal = _same_transform(image, label)
    bounds_equal = bool(
        np.allclose(
            np.asarray(tuple(image.bounds)),
            np.asarray(tuple(label.bounds)),
            rtol=0.0,
            atol=1e-6,
        )
    )
    exact = shape_equal and crs_equal and transform_equal and bounds_equal
    return {
        "verified": True,
        "shape_equal": shape_equal,
        "crs_equal": crs_equal,
        "transform_equal": transform_equal,
        "bounds_equal": bounds_equal,
        "same_pixel_grid": exact,
        "conclusion": (
            "input and label are co-registered for this date"
            if exact
            else "input and label require co-registration before pixel-wise use"
        ),
    }


def _pair_alignment_from_metadata(
    image: dict[str, Any], label: dict[str, Any]
) -> dict[str, Any]:
    shape_equal = (image["height"], image["width"]) == (
        label["height"],
        label["width"],
    )
    crs_equal = image["crs"] == label["crs"]
    transform_equal = bool(
        np.allclose(
            np.asarray(image["transform"]),
            np.asarray(label["transform"]),
            rtol=0.0,
            atol=1e-8,
        )
    )
    bounds_equal = bool(
        np.allclose(
            np.asarray(image["bounds"]),
            np.asarray(label["bounds"]),
            rtol=0.0,
            atol=1e-6,
        )
    )
    exact = shape_equal and crs_equal and transform_equal and bounds_equal
    return {
        "verified": True,
        "verification_method": "read_only_nas_tiff_ifd_header_audit_2026-08-10",
        "input_pixels_materialized": False,
        "shape_equal": shape_equal,
        "crs_equal": crs_equal,
        "transform_equal": transform_equal,
        "bounds_equal": bounds_equal,
        "same_pixel_grid": exact,
        "conclusion": (
            "audited input header and live label are co-registered for this date"
            if exact
            else "audited input header and live label require co-registration"
        ),
    }


def _center_window(
    dataset: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    """Return pixels whose centers fall inside a north-up map-space bbox."""

    left, bottom, right, top = bounds
    transform = dataset.transform
    if transform.b != 0.0 or transform.d != 0.0 or transform.a <= 0.0 or transform.e >= 0.0:
        raise ValueError("only north-up rasters are supported")
    pixel_x = transform.a
    pixel_y = abs(transform.e)
    column_start = max(0, math.ceil((left - transform.c) / pixel_x - 0.5))
    column_stop = min(
        dataset.width,
        math.floor((right - transform.c) / pixel_x - 0.5) + 1,
    )
    row_start = max(0, math.ceil((transform.f - top) / pixel_y - 0.5))
    row_stop = min(
        dataset.height,
        math.floor((transform.f - bottom) / pixel_y - 0.5) + 1,
    )
    if column_start >= column_stop or row_start >= row_stop:
        raise ValueError("common bounds do not intersect the source raster")
    return row_start, row_stop, column_start, column_stop


def _accumulate_counts(counts: np.ndarray, values: np.ndarray) -> None:
    if values.dtype.kind not in {"u", "i"}:
        raise ValueError(f"label dtype must be integral, got {values.dtype}")
    if values.size == 0:
        return
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum < 0 or maximum >= counts.size:
        raise ValueError(f"label values outside supported range 0..{counts.size - 1}")
    counts += np.bincount(values.reshape(-1), minlength=counts.size)


def _count_window(
    dataset: rasterio.io.DatasetReader,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
) -> np.ndarray:
    counts = np.zeros(65_536, dtype=np.int64)
    for first_row in range(row_start, row_stop, CHUNK_ROWS):
        height = min(CHUNK_ROWS, row_stop - first_row)
        values = dataset.read(
            1,
            window=Window(column_start, first_row, column_stop - column_start, height),
        )
        _accumulate_counts(counts, values)
    return counts


def _counts_dict(counts: np.ndarray) -> dict[str, int]:
    return {str(index): int(count) for index, count in enumerate(counts) if count}


def _count_summary(
    counts: np.ndarray,
    pixel_area_m2: float,
    declared_nodata: float | None,
) -> dict[str, Any]:
    total = int(counts.sum())
    nodata_index = None
    if declared_nodata is not None and float(declared_nodata).is_integer():
        candidate = int(declared_nodata)
        if 0 <= candidate < counts.size:
            nodata_index = candidate
    nodata_count = int(counts[nodata_index]) if nodata_index is not None else 0
    valid_count = total - nodata_count
    water_count = int(counts[WATER_VALUE])
    return {
        "pixel_count": total,
        "value_counts": _counts_dict(counts),
        "valid_pixel_count": valid_count,
        "declared_nodata_pixel_count": nodata_count,
        "water_pixel_count": water_count,
        "water_ratio_of_all": water_count / total if total else None,
        "water_ratio_of_valid": water_count / valid_count if valid_count else None,
        "water_area_km2": water_count * pixel_area_m2 / 1_000_000.0,
    }


def _find_quicklook(raw_root: Path, observed_on: str) -> Path | None:
    matches = sorted(raw_root.rglob(f"*QUICKLOOK*{observed_on}*.png"))
    if not matches:
        matches = sorted(raw_root.rglob(f"*{observed_on}*.png"))
    if len(matches) > 1:
        raise ValueError(f"multiple quicklooks found for {observed_on}: {matches}")
    return matches[0] if matches else None


def _find_grd_xml(raw_root: Path, observed_on: str) -> Path | None:
    matches = sorted(raw_root.rglob(f"*_GRD_*{observed_on}*.xml"))
    if len(matches) > 1:
        raise ValueError(f"multiple GRD XML files found for {observed_on}: {matches}")
    return matches[0] if matches else None


def _xml_leaf_values(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    wanted = {
        "product_name",
        "product_type",
        "product_level",
        "satellite_name",
        "acquisition_mode",
        "acquisition_start_utc",
        "acquisition_end_utc",
        "orbit_direction",
        "polarization",
        "look_side",
        "satellite_look_angle",
        "incidence_center",
        "range_spacing",
        "azimuth_spacing",
        "calibration_factor",
    }
    result: dict[str, Any] = {}
    for element in ET.parse(path).getroot().iter():
        tag = element.tag.split("}")[-1]
        if tag in wanted and element.text:
            text = element.text.strip()
            try:
                result[tag] = float(text)
            except ValueError:
                result[tag] = text
    return result


def _logical_raw_path(path: Path | None, raw_root: Path) -> str | None:
    if path is None:
        return None
    try:
        suffix = path.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        suffix = path.name
    # Selective staging often flattens the original SM_<scene>_<date>
    # directory.  Restore that logical NAS path from the ICEYE product name.
    if "/" not in suffix:
        match = re.search(r"_SM_(\d+)_(\d{8})", path.name)
        if match:
            suffix = f"SM_{match.group(1)}_{match.group(2)}/{path.name}"
    return f"{RAW_LOGICAL_ROOT}/{suffix}"


def _save_browse(source: Path, destination: Path, maximum: int) -> None:
    with Image.open(source) as image:
        output = image.convert("RGB")
        output.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
        output.save(destination, quality=88, optimize=True, progressive=True)


def _source_file(
    path: Path, logical_path: str, *, hash_file: bool = True
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "logical_path": logical_path,
        "bytes": path.stat().st_size,
    }
    if hash_file:
        result["sha256"] = _sha256(path)
    else:
        result["sha256"] = None
        result["sha256_note"] = "skipped by --skip-input-sha256"
    return result


def _intersection_bounds(metadata: Iterable[dict[str, Any]]) -> tuple[float, float, float, float]:
    items = tuple(metadata)
    left = max(item["bounds"][0] for item in items)
    bottom = max(item["bounds"][1] for item in items)
    right = min(item["bounds"][2] for item in items)
    top = min(item["bounds"][3] for item in items)
    if left >= right or bottom >= top:
        raise ValueError("the four labels have no common map-space intersection")
    return float(left), float(bottom), float(right), float(top)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.iceye_wb_root.resolve()
    raw_root = args.raw_quicklook_root.resolve() if args.raw_quicklook_root else None
    output_root = args.output.resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("output must not be inside the read-only source tree")
    output_root.mkdir(parents=True, exist_ok=True)

    label_metadata: dict[str, dict[str, Any]] = {}
    for observed_on in DATES:
        label_path = source_root / "Labels" / f"{observed_on}_Label.tif"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        with rasterio.open(label_path, "r") as label:
            label_metadata[observed_on] = _raster_metadata(label)

    crs_values = {item["crs"] for item in label_metadata.values()}
    if len(crs_values) != 1 or None in crs_values:
        raise ValueError(f"labels do not share one projected CRS: {crs_values}")
    common_bounds = _intersection_bounds(label_metadata.values())
    common_width_m = common_bounds[2] - common_bounds[0]
    common_height_m = common_bounds[3] - common_bounds[1]
    preview_width = args.preview_width
    preview_height = max(1, round(preview_width * common_height_m / common_width_m))
    target_transform = from_bounds(*common_bounds, preview_width, preview_height)
    target_pixel_area_m2 = abs(target_transform.a * target_transform.e)

    items: list[dict[str, Any]] = []
    for observed_on in DATES:
        image_path = source_root / "Images" / f"{observed_on}_Input.tif"
        label_path = source_root / "Labels" / f"{observed_on}_Label.tif"
        quicklook = _find_quicklook(raw_root, observed_on) if raw_root else None
        metadata_xml = _find_grd_xml(raw_root, observed_on) if raw_root else None

        with rasterio.open(label_path, "r") as label:
            label_native = _raster_metadata(label)
            if image_path.is_file():
                with rasterio.open(image_path, "r") as image:
                    image_native = _raster_metadata(image)
                    alignment = _pair_alignment(image, label)
                    alignment["verification_method"] = "live_staged_input_and_label_headers"
                    alignment["input_pixels_materialized"] = True
                input_source = _source_file(
                    image_path,
                    f"{DATASET_LOGICAL_ROOT}/Images/{image_path.name}",
                    hash_file=not args.skip_input_sha256,
                )
            else:
                image_native = dict(AUDITED_INPUTS[observed_on])
                alignment = _pair_alignment_from_metadata(image_native, label_native)
                input_source = {
                    "logical_path": f"{DATASET_LOGICAL_ROOT}/Images/{image_path.name}",
                    "bytes": image_native["bytes"],
                    "sha256": None,
                    "sha256_note": (
                        "input pixels not materialized; size and grid metadata come from a "
                        "read-only NAS TIFF IFD header audit"
                    ),
                }
            if not alignment["same_pixel_grid"]:
                raise ValueError(f"{observed_on}: input and label are not co-registered")

            pixel_area_m2 = abs(label.transform.a * label.transform.e)
            full_counts = _count_window(label, 0, label.height, 0, label.width)
            row_start, row_stop, column_start, column_stop = _center_window(
                label, common_bounds
            )
            common_counts = _count_window(
                label, row_start, row_stop, column_start, column_stop
            )

            destination_nodata = (
                int(label.nodata)
                if label.nodata is not None and float(label.nodata).is_integer()
                else 255
            )
            preview = np.full(
                (preview_height, preview_width), destination_nodata, dtype=np.uint8
            )
            reproject(
                source=rasterio.band(label, 1),
                destination=preview,
                src_transform=label.transform,
                src_crs=label.crs,
                src_nodata=label.nodata,
                dst_transform=target_transform,
                dst_crs=label.crs,
                dst_nodata=destination_nodata,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
            )

        preview_counts = np.zeros(65_536, dtype=np.int64)
        _accumulate_counts(preview_counts, preview)
        mask = np.where(preview == WATER_VALUE, 255, 0).astype(np.uint8)
        validity = np.where(preview == destination_nodata, 0, 255).astype(np.uint8)
        mask_path = output_root / f"{observed_on}_mask.png"
        validity_path = output_root / f"{observed_on}_validity.png"
        Image.fromarray(mask, mode="L").save(mask_path, optimize=True)
        Image.fromarray(validity, mode="L").save(validity_path, optimize=True)

        browse_output: Path | None = None
        if quicklook is not None:
            browse_output = output_root / f"{observed_on}_browse.jpg"
            _save_browse(quicklook, browse_output, args.browse_max_size)

        actual_values = [int(index) for index in np.flatnonzero(full_counts)]
        declared_nodata = label_native["nodata"]
        nodata_present = bool(
            declared_nodata is not None
            and float(declared_nodata).is_integer()
            and 0 <= int(declared_nodata) < full_counts.size
            and full_counts[int(declared_nodata)] > 0
        )
        item: dict[str, Any] = {
            "date": f"{observed_on[:4]}-{observed_on[4:6]}-{observed_on[6:]}",
            "source": {
                "input": input_source,
                "label": _source_file(
                    label_path,
                    f"{DATASET_LOGICAL_ROOT}/Labels/{label_path.name}",
                ),
                "raw_quicklook": (
                    _source_file(
                        quicklook,
                        _logical_raw_path(quicklook, raw_root) or quicklook.name,
                    )
                    if quicklook is not None and raw_root is not None
                    else None
                ),
                "raw_grd_metadata": (
                    {
                        "logical_path": _logical_raw_path(metadata_xml, raw_root),
                        "sha256": _sha256(metadata_xml),
                        "fields": _xml_leaf_values(metadata_xml),
                    }
                    if metadata_xml is not None and raw_root is not None
                    else None
                ),
            },
            "native": {
                "input": image_native,
                "label": label_native,
                "pair_alignment": alignment,
            },
            "label_qa": {
                "declared_nodata_tag": declared_nodata,
                "actual_unique_values": actual_values,
                "declared_nodata_occurs_in_pixels": nodata_present,
                "class_semantics": {"0": "background", "1": "water_body"},
                "warning": (
                    "label nodata is declared but absent; use input != 0 as the SAR footprint "
                    "validity mask for model training"
                ),
            },
            "counts": {
                "full_native_extent": _count_summary(
                    full_counts, pixel_area_m2, declared_nodata
                ),
                "common_bounds_native_centers": {
                    **_count_summary(common_counts, pixel_area_m2, declared_nodata),
                    "window": {
                        "row_start": row_start,
                        "row_stop": row_stop,
                        "column_start": column_start,
                        "column_stop": column_stop,
                    },
                },
                "preview_nearest": _count_summary(
                    preview_counts, target_pixel_area_m2, destination_nodata
                ),
            },
            "assets": {
                "mask": _asset(mask_path),
                "label_validity": _asset(validity_path),
                "browse": _asset(browse_output) if browse_output else None,
            },
            "browse_semantics": {
                "role": "display_only_raw_quicklook_context",
                "aligned_with_materialized_mask": False,
                "pixelwise_analysis_allowed": False,
                "note": (
                    "The raw quicklook has its own browse geometry and is not asserted to align "
                    "with the processed 3 m label or common preview grid."
                ),
            },
        }
        items.append(item)

    label_shapes = {
        (item["native"]["label"]["height"], item["native"]["label"]["width"])
        for item in items
    }
    label_transforms = {
        tuple(item["native"]["label"]["transform"]) for item in items
    }
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "nas-iceye-water-body-2020",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "real_iceye_sar_with_binary_water_body_labels",
        "source_access": {
            "mode": "read_only_local_mount_or_staging_copy",
            "nas_modified": False,
            "logical_roots_only": True,
        },
        "temporal_coverage": {
            "dates": [item["date"] for item in items],
            "span_days": 45,
            "interval_days": [28, 16, 1],
            "forecasting_limit": (
                "Four irregular observations support segmentation and change EDA, not a "
                "validated 7/14/30-day learned water-level forecast."
            ),
        },
        "common_grid": {
            "crs": next(iter(crs_values)),
            "bounds": list(common_bounds),
            "width_m": common_width_m,
            "height_m": common_height_m,
            "bbox_area_km2": common_width_m * common_height_m / 1_000_000.0,
            "preview_width": preview_width,
            "preview_height": preview_height,
            "transform": [float(value) for value in tuple(target_transform)[:6]],
            "sampling": "nearest",
            "aspect_preserved": True,
        },
        "cross_date_alignment": {
            "same_crs": True,
            "same_shape": len(label_shapes) == 1,
            "same_transform": len(label_transforms) == 1,
            "direct_array_stack_allowed": False,
            "warp_to_common_grid_required": True,
            "reason": (
                "Dates have different footprints and sub-pixel-shifted 3 m grid origins. "
                "Use map-space reprojection before temporal comparison."
            ),
        },
        "output_semantics": {
            "mask_png": "0=background, 255=water sampled from label value 1",
            "validity_png": (
                "255=label value is not declared label nodata; this is not the SAR input "
                "footprint validity mask"
            ),
            "browse_preview": "non-aligned display context only",
        },
        "items": items,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iceye-wb-root",
        type=Path,
        required=True,
        help="read-only ICEYE_WB directory containing Images/ and Labels/",
    )
    parser.add_argument(
        "--raw-quicklook-root",
        type=Path,
        help="optional read-only raw ICEYE root containing quicklook PNG and GRD XML files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-width", type=int, default=512)
    parser.add_argument("--browse-max-size", type=int, default=1_200)
    parser.add_argument(
        "--skip-input-sha256",
        action="store_true",
        help="skip hashing the large input TIFFs; label/output hashes are always computed",
    )
    args = parser.parse_args()
    if args.preview_width <= 0 or args.browse_max_size <= 0:
        parser.error("preview dimensions must be positive")
    return args


def main() -> None:
    manifest = materialize(_arguments())
    grid = manifest["common_grid"]
    print(
        f"materialized {len(manifest['items'])} ICEYE dates at "
        f"{grid['preview_width']}x{grid['preview_height']}"
    )


if __name__ == "__main__":
    main()
