"""과제 통합 플랫폼 연계 — 알고리즘 구동 파라미터와 산출물 명명규칙.

통합 측 규격(알고리즘_산출물_명명규칙 2026-09-18):

* 구동 파라미터 6종: ``order_id``, ``scene_id``, ``work_id``, ``algm_id``, ``product_type``, ``product_id``
* 산출물 파일명: ``{product_type}_{product_id}.tif`` (GeoTIFF). 수체탐지는 ``PT000003``.
* ``product_id`` 는 원본 영상의 제품 ID. 입력이 여러 개면 메인 입력 영상의 ID.

우리 탐지 마스크는 컨테이너 규약 ``WB_<TB>_<센서>_<시각>.tif`` 로 나오고 이미 EPSG:32652
GeoTIFF 이므로, 규약 이름의 **복사본**을 하나 더 만든다(원본 이름은 캐시 탐색 규약이라 유지).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PRODUCT_TYPE_WATER_MASK = "PT000003"
LIAISON_KEYS: tuple[str, ...] = ("order_id", "scene_id", "work_id", "algm_id", "product_type", "product_id")
_PRODUCT_TYPE_RE = re.compile(r"^PT\d{6}$")
_PRODUCT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,199}$")


class LiaisonError(ValueError):
    """연계 파라미터·파일명 규약 위반."""


def liaison_params(request: Mapping[str, Any]) -> dict[str, str]:
    """요청 본문에서 연계 파라미터만 골라 문자열로 정규화한다. 비어 있는 키는 뺀다."""

    out: dict[str, str] = {}
    for key in LIAISON_KEYS:
        value = request.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out[key] = text
    if "product_type" in out and not _PRODUCT_TYPE_RE.match(out["product_type"]):
        raise LiaisonError(f"product_type must look like PT000003, got {out['product_type']!r}")
    if "product_id" in out and not _PRODUCT_ID_RE.match(out["product_id"]):
        raise LiaisonError("product_id may contain letters, digits, '_', '-', '.' only")
    return out


def product_filename(product_type: str, product_id: str, suffix: str = ".tif") -> str:
    """규약 파일명 ``{product_type}_{product_id}{suffix}``."""

    if not _PRODUCT_TYPE_RE.match(str(product_type)):
        raise LiaisonError(f"product_type must look like PT000003, got {product_type!r}")
    if not _PRODUCT_ID_RE.match(str(product_id)):
        raise LiaisonError("product_id may contain letters, digits, '_', '-', '.' only")
    return f"{product_type}_{product_id}{suffix}"


def derive_product_id(pre_tif: Path | str | None, *, meta_json: Path | str | None = None) -> str | None:
    """원본 영상 ID 를 전처리 사이드카에서 파생한다.

    우선순위: ``<pre>_meta.json`` 의 ``Provenance.input_file`` 줄기 → 전처리 파일 이름 줄기.
    ICEYE 예: ``ICEYE_X5_SLC_SM_23467_20200302T183857``. 아무 근거도 없으면 None.
    """

    candidates: list[Path] = []
    if meta_json:
        candidates.append(Path(meta_json))
    if pre_tif:
        pre = Path(pre_tif)
        candidates.append(pre.with_name(pre.stem + "_meta.json"))
    for sidecar in candidates:
        if not sidecar.is_file():
            continue
        try:
            doc = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prov = doc.get("Provenance") if isinstance(doc, Mapping) else None
        source = (prov or {}).get("input_file") if isinstance(prov, Mapping) else None
        if source:
            stem = Path(str(source)).name
            # .h5 / .tif / .tif.gz 등 확장자를 모두 뗀다
            while "." in stem and not stem.endswith("."):
                head, _, tail = stem.rpartition(".")
                if not tail or len(tail) > 5 or not tail.isalnum():
                    break
                stem = head
            if _PRODUCT_ID_RE.match(stem):
                return stem
    if pre_tif:
        stem = Path(pre_tif).stem
        if _PRODUCT_ID_RE.match(stem):
            return stem
    return None


def export_product(
    mask_tif: Path | str,
    dest_dir: Path | str,
    *,
    product_type: str = PRODUCT_TYPE_WATER_MASK,
    product_id: str,
) -> Path:
    """마스크 GeoTIFF 를 규약 이름으로 ``dest_dir`` 에 둔다. 같은 파일시스템이면 하드링크, 아니면 복사.

    이미 같은 내용의 파일이 있으면 그대로 둔다(멱등). 원본은 건드리지 않는다.
    """

    source = Path(mask_tif)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    dest_root = Path(dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / product_filename(product_type, product_id)
    if target.exists():
        try:
            if os.path.samefile(source, target) or target.stat().st_size == source.stat().st_size:
                return target
        except OSError:
            pass
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target
