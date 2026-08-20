from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend.app.services.artifacts import write_mask_artifacts
from backend.app.services.inputs import UploadedBytes, load_mask_sequence


def _image_bytes(format_name: str) -> bytes:
    array = np.zeros((4, 6), dtype=np.uint8)
    array[1:3, 2:5] = 255
    stream = BytesIO()
    Image.fromarray(array).save(stream, format=format_name)
    return stream.getvalue()


def test_load_npy_sequence() -> None:
    values = np.zeros((3, 4, 5), dtype=np.float32)
    values[:, 1, 1] = 0.8
    stream = BytesIO()
    np.save(stream, values, allow_pickle=False)
    loaded = load_mask_sequence(
        [UploadedBytes("series.npy", stream.getvalue())], threshold=0.7
    )
    assert loaded.frames.shape == (3, 4, 5)
    assert loaded.frame_sources == ["series.npy#1", "series.npy#2", "series.npy#3"]
    assert int(loaded.frames.sum()) == 3


def test_uint16_mask_encoded_as_0_255_is_not_lost() -> None:
    values = np.zeros((4, 5), dtype=np.uint16)
    values[1, 2] = 255
    stream = BytesIO()
    np.save(stream, values, allow_pickle=False)
    loaded = load_mask_sequence(
        [UploadedBytes("uint16-mask.npy", stream.getvalue())], threshold=0.5
    )
    assert int(loaded.frames.sum()) == 1


def test_reject_empty_numpy_sequence() -> None:
    stream = BytesIO()
    np.save(stream, np.zeros((0, 4, 5), dtype=np.uint8), allow_pickle=False)
    try:
        load_mask_sequence(
            [UploadedBytes("empty.npy", stream.getvalue())], threshold=0.5
        )
    except ValueError as exc:
        assert "did not contain any" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("zero-frame array should fail")


def test_load_png_and_tiff() -> None:
    loaded = load_mask_sequence(
        [
            UploadedBytes("one.png", _image_bytes("PNG")),
            UploadedBytes("two.tif", _image_bytes("TIFF")),
        ],
        threshold=0.5,
    )
    assert loaded.frames.shape == (2, 4, 6)
    assert int(loaded.frames[0].sum()) == 6
    assert int(loaded.frames[1].sum()) == 6


def test_geotiff_grid_is_preserved_in_output(tmp_path: Path) -> None:
    values = np.zeros((4, 5), dtype=np.uint8)
    values[1:3, 2:4] = 1
    transform = from_origin(1000, 2000, 3, 3)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=4,
            width=5,
            count=1,
            dtype="uint8",
            crs="EPSG:5186",
            transform=transform,
        ) as dataset:
            dataset.write(values, 1)
        payload = memory.read()

    loaded = load_mask_sequence([UploadedBytes("mask.tif", payload)], threshold=0.5)
    assert loaded.geo_reference is not None
    _, _, output_path = write_mask_artifacts(
        tmp_path, loaded.frames[0], 1, loaded.geo_reference
    )
    with MemoryFile(output_path.read_bytes()) as memory, memory.open() as dataset:
        assert dataset.crs.to_string() == "EPSG:5186"
        assert dataset.transform == transform
        assert np.array_equal(dataset.read(1), values)


def test_reject_shape_mismatch() -> None:
    first = BytesIO()
    second = BytesIO()
    np.save(first, np.zeros((3, 3)), allow_pickle=False)
    np.save(second, np.zeros((4, 3)), allow_pickle=False)
    try:
        load_mask_sequence(
            [
                UploadedBytes("one.npy", first.getvalue()),
                UploadedBytes("two.npy", second.getvalue()),
            ],
            threshold=0.5,
        )
    except ValueError as exc:
        assert "share one shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shape mismatch should fail")
