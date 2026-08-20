from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from ui_next.crop_detection import (
    ESA_DESERT_FIELDS_PATH,
    ESA_LICENSE_URL,
    ESA_SHA256,
    ESA_SOURCE_URL,
    crop_candidate_score,
    decode_crop_image,
    get_crop_sample,
    list_crop_samples,
    load_crop_sample,
    run_crop_detector,
)


def _read_png(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        return np.asarray(image.convert("RGB"))


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_esa_asset_and_catalog_have_verifiable_provenance() -> None:
    assert ESA_DESERT_FIELDS_PATH.is_file()
    assert hashlib.sha256(ESA_DESERT_FIELDS_PATH.read_bytes()).hexdigest() == ESA_SHA256

    samples = list_crop_samples()
    assert [sample.sample_id for sample in samples] == [
        "esa_saudi_full",
        "esa_saudi_west",
        "esa_saudi_east",
    ]
    for sample in samples:
        assert sample.source_url == ESA_SOURCE_URL
        assert sample.license_url == ESA_LICENSE_URL
        assert sample.credit == "Copernicus Sentinel data (2015)/ESA"
        assert "정확도" in sample.limitation_ko


def test_sample_regions_are_deterministic_and_correctly_sized() -> None:
    assert load_crop_sample("esa_saudi_full").size == (1500, 639)
    assert load_crop_sample("esa_saudi_west").size == (750, 639)
    assert load_crop_sample("esa_saudi_east").size == (750, 639)
    assert get_crop_sample("esa_saudi_full").detector_mode == "false_color_red"

    with pytest.raises(KeyError, match="unknown crop sample"):
        get_crop_sample("not-a-sample")


def test_colour_index_modes_react_to_the_declared_target_channel() -> None:
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[:, :40] = [225, 35, 45]
    pixels[:, 40:] = [35, 215, 55]
    image = Image.fromarray(pixels, mode="RGB")

    red_score = crop_candidate_score(image, "false_color_red")
    green_score = crop_candidate_score(image, "natural_rgb_green")

    assert red_score[:, :40].mean() > red_score[:, 40:].mean() + 0.5
    assert green_score[:, 40:].mean() > green_score[:, :40].mean() + 0.5


def test_detector_outputs_are_deterministic_and_summary_is_honest() -> None:
    image = load_crop_sample("esa_saudi_west")
    first = run_crop_detector(
        image,
        detector_mode="false_color_red",
        threshold=0.38,
        overlay_alpha=0.48,
    )
    second = run_crop_detector(
        image,
        detector_mode="false_color_red",
        threshold=0.38,
        overlay_alpha=0.48,
    )

    assert first == second
    assert (first.width, first.height) == image.size
    assert first.valid_pixels == first.width * first.height
    assert first.candidate_pixels > 0
    assert first.candidate_ratio_pct == pytest.approx(
        first.candidate_pixels / first.valid_pixels * 100.0
    )
    assert _read_png(first.source_png).shape == (first.height, first.width, 3)
    assert _read_png(first.score_png).shape == (first.height, first.width, 3)
    assert _read_png(first.mask_png).shape == (first.height, first.width, 3)
    assert _read_png(first.overlay_png).shape == (first.height, first.width, 3)

    mask = _read_png(first.mask_png)
    candidate = np.all(mask == [74, 222, 128], axis=-1)
    source = _read_png(first.source_png)
    overlay = _read_png(first.overlay_png)
    assert np.array_equal(source[~candidate], overlay[~candidate])

    summary = first.summary()
    assert summary["detector_type"] == "rule_based_colour_index"
    assert summary["trained_model"] is False
    assert summary["physical_area_available"] is False
    assert "not a validated model" in summary["warning"]


def test_decode_crop_image_validates_content_and_size() -> None:
    valid = Image.new("RGB", (64, 48), (20, 140, 60))
    assert decode_crop_image(_png_bytes(valid)).size == (64, 48)

    with pytest.raises(ValueError, match="비어"):
        decode_crop_image(b"")
    with pytest.raises(ValueError, match="읽지 못했습니다"):
        decode_crop_image(b"not-an-image")
    with pytest.raises(ValueError, match="32px"):
        decode_crop_image(_png_bytes(Image.new("RGB", (31, 64))))
    with pytest.raises(ValueError, match="너무 큽니다"):
        decode_crop_image(_png_bytes(valid), max_pixels=100)


def test_detector_rejects_invalid_settings_and_mode() -> None:
    image = Image.new("RGB", (64, 64), (40, 190, 70))
    with pytest.raises(ValueError, match="threshold"):
        run_crop_detector(image, detector_mode="natural_rgb_green", threshold=1.0)
    with pytest.raises(ValueError, match="overlay_alpha"):
        run_crop_detector(
            image,
            detector_mode="natural_rgb_green",
            overlay_alpha=1.1,
        )
    with pytest.raises(ValueError, match="unsupported crop detector mode"):
        crop_candidate_score(image, "unknown")  # type: ignore[arg-type]
