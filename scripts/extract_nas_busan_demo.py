# ruff: noqa: UP007, UP045
"""Stream a small, co-registered demo bundle from the NAS Busan GeoTIFFs.

The NAS host only needs Python and NumPy.  Source GeoTIFFs are opened read-only;
PNG previews, binary masks, and a provenance manifest are written to stdout as
a tar archive.  This lets an operator run the script over SSH without creating
temporary files on the NAS::

    ssh USER@HOST "python3 - /path/to/dataset" \
      < scripts/extract_nas_busan_demo.py > busan_demo.tar

The source rasters use slightly different footprints.  The script intersects
their UTM bounds first, then samples every date on one common output grid.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import zlib
from datetime import date, datetime, timezone
from typing import Optional, Union

import numpy as np

DATES = ("20200218", "20200312", "20200325", "20200414")
OUTPUT_WIDTH = 512
OUTPUT_HEIGHT = 512
TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
TIFF_TYPE_FORMATS = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d"}


def read_tiff_tags(path: str) -> dict[int, Union[tuple[object, ...], bytes]]:
    """Read the small subset of classic TIFF tags needed by this dataset."""

    tags: dict[int, Union[tuple[object, ...], bytes]] = {}
    with open(path, "rb") as handle:
        byte_order = handle.read(2)
        endian = "<" if byte_order == b"II" else ">"
        magic, ifd_offset = struct.unpack(endian + "HI", handle.read(6))
        if magic != 42:
            raise ValueError(f"{path}: expected classic TIFF magic 42, got {magic}")
        handle.seek(ifd_offset)
        entry_count = struct.unpack(endian + "H", handle.read(2))[0]
        for _ in range(entry_count):
            raw = handle.read(12)
            tag, value_type, count, value_offset = struct.unpack(endian + "HHII", raw)
            element_size = TIFF_TYPE_SIZES.get(value_type)
            if element_size is None:
                continue
            byte_count = element_size * count
            return_position = handle.tell()
            if byte_count <= 4:
                data = raw[8 : 8 + byte_count]
            else:
                handle.seek(value_offset)
                data = handle.read(byte_count)
                handle.seek(return_position)
            value_format = TIFF_TYPE_FORMATS.get(value_type)
            if value_format is None:
                tags[tag] = data
            else:
                tags[tag] = struct.unpack(endian + value_format * count, data)
    return tags


def first(tags: dict[int, Union[tuple[object, ...], bytes]], tag: int) -> float:
    value = tags[tag]
    if isinstance(value, bytes):
        raise TypeError(f"tag {tag} was not numeric")
    return float(value[0])


def raster_metadata(
    path: str, reference: Optional[dict[str, object]] = None
) -> dict[str, object]:
    tags = read_tiff_tags(path)
    width = int(first(tags, 256))
    height = int(first(tags, 257))
    samples = int(first(tags, 277))
    bits = tuple(int(value) for value in tags[258])  # type: ignore[arg-type]
    sample_format = tuple(int(value) for value in tags.get(339, (1,) * samples))  # type: ignore[arg-type]
    if 33550 in tags and 33922 in tags:
        scale = tuple(float(value) for value in tags[33550])  # type: ignore[arg-type]
        tie = tuple(float(value) for value in tags[33922])  # type: ignore[arg-type]
        pixel_x, pixel_y = scale[:2]
        origin_x, origin_y = tie[3], tie[4]
    elif reference is not None:
        pixel_x = float(reference["pixel_x"])
        pixel_y = float(reference["pixel_y"])
        origin_x = float(reference["origin_x"])
        origin_y = float(reference["origin_y"])
    else:
        raise ValueError(f"{path}: georeferencing tags are missing")
    strip_offsets = tuple(int(value) for value in tags[273])  # type: ignore[arg-type]
    compression = int(first(tags, 259))
    planar = int(first(tags, 284))
    if compression != 1 or planar != 1:
        raise ValueError(f"{path}: expected uncompressed contiguous TIFF")
    if len(strip_offsets) == 0:
        raise ValueError(f"{path}: missing strip offsets")
    return {
        "path": path,
        "width": width,
        "height": height,
        "samples": samples,
        "bits": bits,
        "sample_format": sample_format,
        "offset": strip_offsets[0],
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "xmin": origin_x,
        "xmax": origin_x + width * pixel_x,
        "ymin": origin_y - height * pixel_y,
        "ymax": origin_y,
        "size_bytes": os.path.getsize(path),
        "mtime_utc": datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat(),
    }


def memmap_raster(metadata: dict[str, object]) -> np.memmap:
    samples = int(metadata["samples"])
    bits = tuple(metadata["bits"])
    sample_format = tuple(metadata["sample_format"])
    if bits == (8,) and sample_format == (1,):
        dtype = np.dtype("u1")
    elif bits == (32, 32, 32, 32) and sample_format == (3, 3, 3, 3):
        dtype = np.dtype("<f4")
    else:
        raise ValueError(f"unsupported TIFF sample layout: bits={bits}, format={sample_format}")
    shape = (int(metadata["height"]), int(metadata["width"]))
    if samples > 1:
        shape = shape + (samples,)
    return np.memmap(
        str(metadata["path"]),
        dtype=dtype,
        mode="r",
        offset=int(metadata["offset"]),
        shape=shape,
    )


def grid_indices(metadata: dict[str, object], bounds: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    xs = bounds["xmin"] + (np.arange(OUTPUT_WIDTH) + 0.5) * (
        (bounds["xmax"] - bounds["xmin"]) / OUTPUT_WIDTH
    )
    ys = bounds["ymax"] - (np.arange(OUTPUT_HEIGHT) + 0.5) * (
        (bounds["ymax"] - bounds["ymin"]) / OUTPUT_HEIGHT
    )
    columns = np.floor((xs - float(metadata["origin_x"])) / float(metadata["pixel_x"])).astype(int)
    rows = np.floor((float(metadata["origin_y"]) - ys) / float(metadata["pixel_y"])).astype(int)
    if columns.min() < 0 or columns.max() >= int(metadata["width"]):
        raise ValueError("common grid columns fall outside a source raster")
    if rows.min() < 0 or rows.max() >= int(metadata["height"]):
        raise ValueError("common grid rows fall outside a source raster")
    return rows, columns


def png_bytes(array: np.ndarray) -> bytes:
    """Encode one uint8 grayscale or RGB array without Pillow."""

    values = np.ascontiguousarray(array, dtype=np.uint8)
    if values.ndim == 2:
        height, width = values.shape
        color_type = 0
    elif values.ndim == 3 and values.shape[2] == 3:
        height, width, _ = values.shape
        color_type = 2
    else:
        raise ValueError(f"unsupported PNG shape {values.shape}")

    def chunk(name: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(name)
        checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", checksum)

    scanlines = b"".join(b"\x00" + values[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(scanlines, level=9)
    ) + chunk(b"IEND", b"")


def rgb_preview(values: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    # PlanetScope AnalyticMS SR order is blue, green, red, NIR.
    rgb = np.asarray(values[..., (2, 1, 0)], dtype=np.float32)
    quantiles: list[list[float]] = []
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        source = rgb[..., channel]
        valid = np.isfinite(source) & (source >= 0.0) & (source <= 1.5)
        if not np.any(valid):
            low, high = 0.0, 1.0
        else:
            low, high = (float(value) for value in np.percentile(source[valid], (2, 98)))
            if high <= low:
                high = low + 1.0
        scaled = np.clip((source - low) / (high - low), 0.0, 1.0)
        scaled[~valid] = 0.0
        output[..., channel] = np.rint(scaled * 255.0).astype(np.uint8)
        quantiles.append([low, high])
    return output, quantiles


def count_source_water(mask: np.memmap, metadata: dict[str, object], bounds: dict[str, float]) -> int:
    column_start = round((bounds["xmin"] - float(metadata["origin_x"])) / float(metadata["pixel_x"]))
    column_stop = round((bounds["xmax"] - float(metadata["origin_x"])) / float(metadata["pixel_x"]))
    row_start = round((float(metadata["origin_y"]) - bounds["ymax"]) / float(metadata["pixel_y"]))
    row_stop = round((float(metadata["origin_y"]) - bounds["ymin"]) / float(metadata["pixel_y"]))
    total = 0
    for start in range(row_start, row_stop, 512):
        block = np.asarray(mask[start : min(start + 512, row_stop), column_start:column_stop])
        total += int(np.count_nonzero(block == 1))
    return total


def add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def build_bundle(dataset_root: str) -> None:
    inputs: dict[str, dict[str, object]] = {}
    labels: dict[str, dict[str, object]] = {}
    for observed_on in DATES:
        labels[observed_on] = raster_metadata(
            os.path.join(dataset_root, "label_busan", f"{observed_on}_SR_label.tif")
        )
        inputs[observed_on] = raster_metadata(
            os.path.join(dataset_root, "input_busan", f"{observed_on}_SR_input.tif"),
            reference=labels[observed_on],
        )

    bounds = {
        "xmin": max(float(item["xmin"]) for item in labels.values()),
        "xmax": min(float(item["xmax"]) for item in labels.values()),
        "ymin": max(float(item["ymin"]) for item in labels.values()),
        "ymax": min(float(item["ymax"]) for item in labels.values()),
    }
    if bounds["xmin"] >= bounds["xmax"] or bounds["ymin"] >= bounds["ymax"]:
        raise ValueError("source rasters do not share a common spatial intersection")

    preview_pixel_area_m2 = (
        (bounds["xmax"] - bounds["xmin"]) / OUTPUT_WIDTH
    ) * ((bounds["ymax"] - bounds["ymin"]) / OUTPUT_HEIGHT)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": "nas-busan-planetscope-water-labels-2020",
        "classification": "real_satellite_imagery_with_derived_binary_water_labels",
        "sensor": "PlanetScope AnalyticMS Surface Reflectance (inferred from source filenames)",
        "label_semantics": {"0": "non-water", "1": "water"},
        "crs": "EPSG:32652",
        "source_pixel_size_m": [3.0, 3.0],
        "source_pixel_area_m2": 9.0,
        "common_bounds_utm": bounds,
        "preview_shape": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
        "preview_pixel_area_m2": preview_pixel_area_m2,
        "dates": list(DATES),
        "source_root": dataset_root,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "processing": [
            "intersect four source footprints",
            "nearest-neighbour sample on one common 512x512 grid",
            "convert label values 0/1 to PNG values 0/255",
            "render RGB from PlanetScope bands red/green/blue with per-date 2-98% stretch",
        ],
        "limitations": [
            "preview masks are spatially resampled demo assets, not training rasters",
            "no measured gauge water level is bundled",
            "no future water-label truth exists after 2020-04-14",
            "the sensor identity is inferred from matching PlanetScope dates and filenames",
        ],
        "frames": [],
    }

    payloads: list[tuple[str, bytes]] = []
    for observed_on in DATES:
        input_meta = inputs[observed_on]
        label_meta = labels[observed_on]
        input_grid = memmap_raster(input_meta)
        label_grid = memmap_raster(label_meta)
        rows, columns = grid_indices(label_meta, bounds)
        input_rows, input_columns = grid_indices(input_meta, bounds)
        sampled_input = np.asarray(input_grid[input_rows[:, None], input_columns[None, :], :])
        sampled_label = np.asarray(label_grid[rows[:, None], columns[None, :]])
        binary_label = np.where(sampled_label == 1, 255, 0).astype(np.uint8)
        preview, stretch = rgb_preview(sampled_input)
        preview_name = f"previews/{observed_on}_rgb.png"
        mask_name = f"masks/{observed_on}_water_mask.png"
        preview_payload = png_bytes(preview)
        mask_payload = png_bytes(binary_label)
        payloads.extend(((preview_name, preview_payload), (mask_name, mask_payload)))
        source_water_pixels = count_source_water(label_grid, label_meta, bounds)
        preview_water_pixels = int(np.count_nonzero(binary_label))
        manifest["frames"].append(  # type: ignore[union-attr]
            {
                "date": date(
                    int(observed_on[:4]),
                    int(observed_on[4:6]),
                    int(observed_on[6:8]),
                ).isoformat(),
                "preview": preview_name,
                "mask": mask_name,
                "preview_sha256": hashlib.sha256(preview_payload).hexdigest(),
                "mask_sha256": hashlib.sha256(mask_payload).hexdigest(),
                "preview_water_pixels": preview_water_pixels,
                "preview_water_area_m2": preview_water_pixels * preview_pixel_area_m2,
                "source_common_grid_water_pixels": source_water_pixels,
                "source_common_grid_water_area_m2": source_water_pixels * 9.0,
                "rgb_stretch_2_98": stretch,
                "input_source": {
                    "relative_path": os.path.relpath(str(input_meta["path"]), dataset_root),
                    "size_bytes": input_meta["size_bytes"],
                    "width": input_meta["width"],
                    "height": input_meta["height"],
                    "bounds_utm": {key: input_meta[key] for key in ("xmin", "xmax", "ymin", "ymax")},
                },
                "label_source": {
                    "relative_path": os.path.relpath(str(label_meta["path"]), dataset_root),
                    "size_bytes": label_meta["size_bytes"],
                    "width": label_meta["width"],
                    "height": label_meta["height"],
                    "bounds_utm": {key: label_meta[key] for key in ("xmin", "xmax", "ymin", "ymax")},
                },
            }
        )
        del input_grid, label_grid, sampled_input, sampled_label

    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        add_tar_bytes(archive, "manifest.json", manifest_payload)
        for name, payload in payloads:
            add_tar_bytes(archive, name, payload)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: extract_nas_busan_demo.py DATASET_ROOT")
    build_bundle(sys.argv[1])


if __name__ == "__main__":
    main()
