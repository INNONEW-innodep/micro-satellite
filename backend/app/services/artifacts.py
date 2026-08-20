from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .inputs import GeoReference

try:
    import rasterio
    from affine import Affine
except ImportError:  # pragma: no cover - Pillow writes a plain TIFF fallback
    rasterio = None  # type: ignore[assignment]
    Affine = None  # type: ignore[assignment]


def write_mask_artifacts(
    directory: Path,
    mask: np.ndarray,
    horizon: int,
    geo_reference: GeoReference | None,
) -> tuple[Path, Path, Path]:
    stem = f"mask_{horizon:03d}"
    npy_path = directory / f"{stem}.npy"
    png_path = directory / f"{stem}.png"
    tif_path = directory / f"{stem}.tif"
    binary = (mask > 0).astype(np.uint8)
    np.save(npy_path, binary, allow_pickle=False)
    image_values = binary * np.uint8(255)
    Image.fromarray(image_values, mode="L").save(png_path, format="PNG")
    _write_tiff(tif_path, binary, geo_reference)
    return npy_path, png_path, tif_path


def _write_tiff(path: Path, values: np.ndarray, reference: GeoReference | None) -> None:
    if reference is None or rasterio is None or Affine is None:
        Image.fromarray(values, mode="L").save(path, format="TIFF")
        return
    transform = Affine(*reference.transform)
    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=reference.height,
        width=reference.width,
        count=1,
        dtype="uint8",
        crs=reference.crs,
        transform=transform,
        compress="deflate",
    ) as dataset:
        dataset.write(values, 1)
