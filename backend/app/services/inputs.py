from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence, UnidentifiedImageError

try:  # rasterio is preferred for GeoTIFF metadata but Pillow remains a fallback
    from rasterio.io import MemoryFile
except ImportError:  # pragma: no cover - exercised in minimal deployments
    MemoryFile = None  # type: ignore[assignment]


SUPPORTED_SUFFIXES = {".npy", ".png", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class UploadedBytes:
    name: str
    content: bytes


@dataclass(frozen=True, slots=True)
class GeoReference:
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int


@dataclass(slots=True)
class LoadedFrames:
    frames: np.ndarray
    source_files: list[str]
    frame_sources: list[str]
    geo_reference: GeoReference | None = None
    warnings: list[str] = field(default_factory=list)


def load_mask_sequence(
    files: Sequence[UploadedBytes], threshold: float
) -> LoadedFrames:
    if not files:
        raise ValueError("at least one .npy, .png, .tif, or .tiff file is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    decoded: list[np.ndarray] = []
    frame_sources: list[str] = []
    references: list[GeoReference | None] = []
    warnings: list[str] = []
    for upload in files:
        suffix = Path(upload.name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise ValueError(
                f"unsupported file {upload.name!r}; expected one of {supported}"
            )
        if not upload.content:
            raise ValueError(f"uploaded file {upload.name!r} is empty")

        if suffix == ".npy":
            arrays = _read_npy(upload)
            reference = None
        elif suffix in {".tif", ".tiff"}:
            arrays, reference = _read_tiff(upload)
        else:
            arrays = _read_pillow(upload)
            reference = None

        for index, array in enumerate(arrays):
            normalized = _normalize_mask(array, upload.name)
            decoded.append((normalized >= threshold).astype(np.uint8))
            frame_sources.append(
                upload.name if len(arrays) == 1 else f"{upload.name}#{index + 1}"
            )
            references.append(reference)

    if not decoded:
        raise ValueError("uploaded files did not contain any mask frames")
    expected_shape = decoded[0].shape
    for source, frame in zip(frame_sources, decoded, strict=True):
        if frame.shape != expected_shape:
            raise ValueError(
                f"all frames must share one shape; {source!r} is {frame.shape}, expected {expected_shape}"
            )

    geo_reference = _common_reference(references)
    if any(reference is not None for reference in references) and geo_reference is None:
        warnings.append(
            "GeoTIFF metadata was not propagated because inputs were mixed or used different grids."
        )
    return LoadedFrames(
        frames=np.stack(decoded, axis=0),
        source_files=[upload.name for upload in files],
        frame_sources=frame_sources,
        geo_reference=geo_reference,
        warnings=warnings,
    )


def _read_npy(upload: UploadedBytes) -> list[np.ndarray]:
    try:
        array = np.load(BytesIO(upload.content), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not read NumPy file {upload.name!r}: {exc}") from exc
    if array.ndim == 2:
        return [array]
    if array.ndim == 3 and array.shape[-1] == 1:
        return [array[..., 0]]
    if array.ndim == 3:
        return [array[index] for index in range(array.shape[0])]
    if array.ndim == 4 and array.shape[-1] == 1:
        return [array[index, ..., 0] for index in range(array.shape[0])]
    raise ValueError(
        f"NumPy file {upload.name!r} must have shape [H,W], [T,H,W], or a singleton channel"
    )


def _read_pillow(upload: UploadedBytes) -> list[np.ndarray]:
    try:
        with Image.open(BytesIO(upload.content)) as image:
            return [
                np.asarray(frame.convert("L"))
                for frame in ImageSequence.Iterator(image)
            ]
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"could not read image {upload.name!r}: {exc}") from exc


def _read_tiff(upload: UploadedBytes) -> tuple[list[np.ndarray], GeoReference | None]:
    if MemoryFile is None:
        return _read_pillow(upload), None
    try:
        with MemoryFile(upload.content) as memory_file, memory_file.open() as dataset:
            values = dataset.read(masked=True)
            arrays = [
                np.asarray(values[index].filled(0)) for index in range(values.shape[0])
            ]
            reference = None
            if dataset.crs is not None:
                transform = dataset.transform
                reference = GeoReference(
                    crs=dataset.crs.to_string(),
                    transform=(
                        float(transform.a),
                        float(transform.b),
                        float(transform.c),
                        float(transform.d),
                        float(transform.e),
                        float(transform.f),
                    ),
                    width=dataset.width,
                    height=dataset.height,
                )
            return arrays, reference
    except Exception:  # noqa: BLE001 - raster readers vary; Pillow is the format fallback
        # Non-geospatial TIFFs are valid inputs too.
        return _read_pillow(upload), None


def _normalize_mask(array: np.ndarray, source: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"frame from {source!r} is not two-dimensional")
    if array.dtype == np.bool_:
        return array.astype(np.float32)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"frame from {source!r} must contain numeric values")

    values = array.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"frame from {source!r} contains NaN or infinite values")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0:
        raise ValueError(f"frame from {source!r} contains negative mask values")
    if maximum <= 1:
        return values
    if np.issubdtype(array.dtype, np.integer):
        if maximum <= 255:
            return values / 255.0
        type_maximum = float(np.iinfo(array.dtype).max)
        return values / type_maximum
    if maximum <= 255:
        return values / 255.0
    raise ValueError(
        f"floating-point frame from {source!r} must use a 0..1 or 0..255 value range"
    )


def _common_reference(references: Sequence[GeoReference | None]) -> GeoReference | None:
    if not references or any(reference is None for reference in references):
        return None
    first = references[0]
    if first is None:
        return None
    return first if all(reference == first for reference in references[1:]) else None
