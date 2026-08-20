"""Deterministic crop-candidate detection for the standalone UI POC.

The repository contains a roadmap item for crop segmentation, but no trained
crop model or labelled crop dataset.  This module therefore implements an
explicitly labelled colour-index baseline.  It is useful for the UI contract
and for visualising candidate agricultural vegetation; it must not be reported
as crop-type classification, model accuracy, or an operational crop map.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

MODULE_DIR = Path(__file__).resolve().parent
CROP_ASSET_DIR = MODULE_DIR / "assets" / "crop"
ESA_DESERT_FIELDS_PATH = CROP_ASSET_DIR / "esa_sentinel2_desert_fields.jpg"
ESA_SOURCE_URL = "https://www.esa.int/ESA_Multimedia/Images/2015/07/Desert_fields"
ESA_DOWNLOAD_URL = (
    "https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2015/07/"
    "desert_fields/15536865-1-eng-GB/Desert_fields.jpg"
)
ESA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/igo/"
ESA_SHA256 = "dbd6eb75a1b3ae197423a3f419a31b1b97131556cd57f2d7b25faaf5b9c112aa"

DetectorMode = Literal["false_color_red", "natural_rgb_green"]


@dataclass(frozen=True, slots=True)
class CropSample:
    sample_id: str
    display_name: str
    summary_ko: str
    crop_box: tuple[int, int, int, int] | None
    detector_mode: DetectorMode
    source_title: str = "ESA · Desert fields"
    source_url: str = ESA_SOURCE_URL
    download_url: str = ESA_DOWNLOAD_URL
    credit: str = "Copernicus Sentinel data (2015)/ESA"
    license_name: str = "CC BY-SA 3.0 IGO"
    license_url: str = ESA_LICENSE_URL
    imagery_description_ko: str = (
        "2015년 Sentinel-2A가 촬영한 사우디아라비아 중앙 피벗 관개 농업 지역의 "
        "false-color JPEG입니다. 붉은색 계열은 활발한 식생 신호를 강조합니다."
    )
    intended_use_ko: str = (
        "작물 재배지로 보이는 활발한 식생 후보를 이진 마스크와 오버레이로 표현하는 "
        "UI/알고리즘 POC에 사용합니다."
    )
    limitation_ko: str = (
        "원본 다중분광 밴드와 지리참조가 없는 게시용 JPEG입니다. 작물 종류, 필지 경계, "
        "정확도, 실제 면적을 검증할 수 없고 붉은 비작물 물체도 오탐할 수 있습니다."
    )


@dataclass(frozen=True, slots=True)
class CropDetectionResult:
    width: int
    height: int
    candidate_pixels: int
    valid_pixels: int
    candidate_ratio_pct: float
    threshold: float
    detector_mode: DetectorMode
    source_png: bytes
    score_png: bytes
    mask_png: bytes
    overlay_png: bytes
    source_sha256: str
    detector_id: str = "rgb-colour-index-v1"
    trained_model: bool = False
    physical_area_available: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "detector_type": "rule_based_colour_index",
            "trained_model": self.trained_model,
            "detector_mode": self.detector_mode,
            "threshold": self.threshold,
            "image_width": self.width,
            "image_height": self.height,
            "valid_pixels": self.valid_pixels,
            "crop_candidate_pixels": self.candidate_pixels,
            "crop_candidate_ratio_pct": self.candidate_ratio_pct,
            "physical_area_available": self.physical_area_available,
            "source_sha256": self.source_sha256,
            "warning": (
                "Colour-index crop/vegetation candidate POC; not crop-type "
                "classification and not a validated model result."
            ),
        }


_SAMPLES = (
    CropSample(
        sample_id="esa_saudi_full",
        display_name="ESA 사우디 농업지대 · 전체",
        summary_ko="중앙 피벗 관개 원형 농지가 넓게 분포한 전체 Sentinel-2 false-color 장면입니다.",
        crop_box=None,
        detector_mode="false_color_red",
    ),
    CropSample(
        sample_id="esa_saudi_west",
        display_name="ESA 사우디 농업지대 · 서부 밀집",
        summary_ko="원형 농지가 조밀한 서쪽 절반을 잘라 탐지 결과를 크게 확인하는 파생 샘플입니다.",
        crop_box=(0, 0, 750, 639),
        detector_mode="false_color_red",
    ),
    CropSample(
        sample_id="esa_saudi_east",
        display_name="ESA 사우디 농업지대 · 동부 산개",
        summary_ko="사막 배경에 원형 농지가 산개한 동쪽 절반을 잘라 오탐·누락을 보기 쉬운 파생 샘플입니다.",
        crop_box=(750, 0, 1500, 639),
        detector_mode="false_color_red",
    ),
)
_CATALOG: Mapping[str, CropSample] = MappingProxyType(
    {sample.sample_id: sample for sample in _SAMPLES}
)


def list_crop_samples() -> tuple[CropSample, ...]:
    return _SAMPLES


def get_crop_sample(sample_id: str) -> CropSample:
    try:
        return _CATALOG[sample_id]
    except KeyError as exc:
        available = ", ".join(_CATALOG)
        raise KeyError(
            f"unknown crop sample {sample_id!r}; available: {available}"
        ) from exc


def load_crop_sample(sample_id: str) -> Image.Image:
    sample = get_crop_sample(sample_id)
    if not ESA_DESERT_FIELDS_PATH.is_file():
        raise FileNotFoundError(
            f"crop sample asset is missing: {ESA_DESERT_FIELDS_PATH}"
        )
    with Image.open(ESA_DESERT_FIELDS_PATH) as opened:
        image = opened.convert("RGB")
    if sample.crop_box is not None:
        image = image.crop(sample.crop_box)
    return image


def decode_crop_image(content: bytes, *, max_pixels: int = 24_000_000) -> Image.Image:
    if not content:
        raise ValueError("이미지 파일이 비어 있습니다.")
    try:
        with Image.open(BytesIO(content)) as opened:
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("PNG 또는 JPEG 이미지를 읽지 못했습니다.") from exc
    width, height = image.size
    if width < 32 or height < 32:
        raise ValueError("작물 탐지 이미지는 가로·세로가 각각 32px 이상이어야 합니다.")
    if width * height > max_pixels:
        raise ValueError(
            f"이미지가 너무 큽니다. 최대 {max_pixels:,}픽셀까지 지원합니다."
        )
    return image


def resize_for_detection(image: Image.Image, max_dimension: int = 1200) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image.convert("RGB")
    scale = max_dimension / longest
    resized = (
        max(32, round(width * scale)),
        max(32, round(height * scale)),
    )
    return image.convert("RGB").resize(resized, Image.Resampling.LANCZOS)


def crop_candidate_score(image: Image.Image, detector_mode: DetectorMode) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = np.moveaxis(rgb, -1, 0)
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = (maximum - minimum) / (maximum + 0.05)

    if detector_mode == "false_color_red":
        target = red
        competitor = np.maximum(green, blue)
    elif detector_mode == "natural_rgb_green":
        target = green
        competitor = np.maximum(red, blue)
    else:
        raise ValueError(f"unsupported crop detector mode: {detector_mode!r}")

    dominance = np.clip((target - competitor + 0.15) / 0.75, 0.0, 1.0)
    score = 0.74 * dominance + 0.26 * saturation * target
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def run_crop_detector(
    image: Image.Image,
    *,
    detector_mode: DetectorMode,
    threshold: float = 0.38,
    overlay_alpha: float = 0.48,
) -> CropDetectionResult:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between 0 and 1")

    prepared = resize_for_detection(image)
    score = crop_candidate_score(prepared, detector_mode)
    mask = _clean_binary_mask(score >= threshold)
    source = np.asarray(prepared, dtype=np.uint8)
    boundary = mask & ~_erode(mask, 1)

    overlay = source.astype(np.float32)
    candidate_colour = np.array([34.0, 197.0, 94.0], dtype=np.float32)
    overlay[mask] = (
        (1.0 - overlay_alpha) * overlay[mask]
        + overlay_alpha * candidate_colour
    )
    overlay[boundary] = np.array([250.0, 204.0, 21.0], dtype=np.float32)
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), "RGB")

    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[:] = np.array([2, 6, 23], dtype=np.uint8)
    mask_rgb[mask] = np.array([74, 222, 128], dtype=np.uint8)
    mask_image = Image.fromarray(mask_rgb, "RGB")
    score_image = _score_to_image(score)

    source_png = _png_bytes(prepared)
    candidate_pixels = int(mask.sum())
    valid_pixels = int(mask.size)
    return CropDetectionResult(
        width=prepared.width,
        height=prepared.height,
        candidate_pixels=candidate_pixels,
        valid_pixels=valid_pixels,
        candidate_ratio_pct=(candidate_pixels / valid_pixels * 100.0),
        threshold=float(threshold),
        detector_mode=detector_mode,
        source_png=source_png,
        score_png=_png_bytes(score_image),
        mask_png=_png_bytes(mask_image),
        overlay_png=_png_bytes(overlay_image),
        source_sha256=hashlib.sha256(source_png).hexdigest(),
    )


def _clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    candidate = np.asarray(mask, dtype=bool)
    neighbour_count = _neighbour_sum(candidate)
    candidate &= neighbour_count >= 3
    candidate = _erode(_dilate(candidate, 1), 1)
    return candidate


def _neighbour_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    neighbours = [
        padded[row : row + mask.shape[0], col : col + mask.shape[1]]
        for row in range(3)
        for col in range(3)
    ]
    return np.add.reduce(neighbours)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbours = [
            padded[row : row + result.shape[0], col : col + result.shape[1]]
            for row in range(3)
            for col in range(3)
        ]
        result = np.logical_or.reduce(neighbours)
    return result


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbours = [
            padded[row : row + result.shape[0], col : col + result.shape[1]]
            for row in range(3)
            for col in range(3)
        ]
        result = np.logical_and.reduce(neighbours)
    return result


def _score_to_image(score: np.ndarray) -> Image.Image:
    clipped = np.clip(score, 0.0, 1.0)[..., None]
    low = np.array([2.0, 6.0, 23.0], dtype=np.float32)
    high = np.array([250.0, 204.0, 21.0], dtype=np.float32)
    colours = low + clipped * (high - low)
    return Image.fromarray(colours.astype(np.uint8), "RGB")


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


__all__ = [
    "CROP_ASSET_DIR",
    "ESA_DESERT_FIELDS_PATH",
    "ESA_DOWNLOAD_URL",
    "ESA_LICENSE_URL",
    "ESA_SHA256",
    "ESA_SOURCE_URL",
    "CropDetectionResult",
    "CropSample",
    "DetectorMode",
    "crop_candidate_score",
    "decode_crop_image",
    "get_crop_sample",
    "list_crop_samples",
    "load_crop_sample",
    "resize_for_detection",
    "run_crop_detector",
]
