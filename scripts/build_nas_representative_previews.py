"""Build lightweight catalog thumbnails from selectively copied NAS assets.

The input directory is a temporary extraction of representative source files;
the output directory contains only browser-friendly previews.  This script does
not connect to the NAS and never changes source data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling


def _find_one(root: Path, pattern: str) -> Path:
    matches = tuple(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r} below {root}, found {len(matches)}")
    return matches[0]


def _fit(image: Image.Image, width: int = 768) -> Image.Image:
    if image.width <= width:
        return image.copy()
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _save_image(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        output = _fit(image.convert("RGB"))
        output.save(destination, quality=88, optimize=True)


def _udm2_preview(source: Path, destination: Path) -> None:
    with rasterio.open(source) as dataset:
        width = 768
        height = max(1, round(dataset.height * width / dataset.width))
        bands = dataset.read(
            out_shape=(dataset.count, height, width),
            resampling=Resampling.nearest,
        )
    # Planet UDM2 band semantics: clear, snow, shadow, light haze, heavy haze,
    # cloud, confidence, legacy UDM1.  Colors are categorical UI colors.
    output = np.zeros((height, width, 3), dtype=np.uint8)
    confidence = np.clip(bands[6].astype(np.float32) / 100.0, 0.0, 1.0)
    output[:] = (10, 18, 30)
    clear = bands[0] > 0
    output[clear] = np.stack(
        (
            25 + 20 * confidence[clear],
            75 + 90 * confidence[clear],
            95 + 105 * confidence[clear],
        ),
        axis=1,
    ).astype(np.uint8)
    output[bands[2] > 0] = (99, 65, 145)  # shadow
    output[bands[3] > 0] = (229, 187, 86)  # light haze
    output[bands[4] > 0] = (241, 146, 54)  # heavy haze
    output[bands[1] > 0] = (157, 226, 255)  # snow
    output[bands[5] > 0] = (244, 91, 105)  # cloud
    Image.fromarray(output).save(destination, optimize=True)


def _water_body_preview(source: Path, destination: Path) -> None:
    with rasterio.open(source) as dataset:
        width = 768
        height = max(1, round(dataset.height * width / dataset.width))
        mask = dataset.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.average,
        )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:] = (5, 15, 28)
    output[mask > 0.01] = (34, 211, 238)
    Image.fromarray(output).save(destination, optimize=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_nas_representative_previews.py INPUT_ROOT OUTPUT_ROOT")
    source_root = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sources = {
        "iceye_sar": _find_one(source_root, "ICEYE_X5_QUICKLOOK*.png"),
        "planetscope_udm2": _find_one(source_root, "*_udm2_clip.tif"),
        "university_water_body": _find_one(source_root, "Pred_WB_*.tif"),
        "giheung_optical": _find_one(source_root, "23JUL30*-M2AS-*-BROWSE.JPG"),
        "hoedong_optical": _find_one(source_root, "24AUG23*-M2AS-*-BROWSE.JPG"),
    }
    outputs = {
        "iceye_sar": output_root / "iceye_sar_quicklook_20200302.jpg",
        "planetscope_udm2": output_root / "planetscope_udm2_20200218.png",
        "university_water_body": output_root / "university_pred_wb_20210411.png",
        "giheung_optical": output_root / "giheung_browse_20230730.jpg",
        "hoedong_optical": output_root / "hoedong_browse_20240823.jpg",
    }
    _save_image(sources["iceye_sar"], outputs["iceye_sar"])
    _udm2_preview(sources["planetscope_udm2"], outputs["planetscope_udm2"])
    _water_body_preview(
        sources["university_water_body"], outputs["university_water_body"]
    )
    _save_image(sources["giheung_optical"], outputs["giheung_optical"])
    _save_image(sources["hoedong_optical"], outputs["hoedong_optical"])

    metadata = {
        "classification": "catalog_preview_only",
        "items": {
            key: {
                "asset": outputs[key].name,
                "source_logical_path": str(sources[key].relative_to(source_root)),
                "scientific_input_allowed": False,
            }
            for key in sources
        },
        "notes": [
            "ICEYE quicklook and commercial browse images are display-only.",
            "UDM2 colors show quality categories and do not represent water labels.",
            "The university Pred_WB thumbnail is a one-date derived water-body output.",
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
