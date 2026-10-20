from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..api_docs import API_DOCS
from .registry import SegmenterRegistry

router = APIRouter(prefix="/segmenters", tags=["wbms"])


def _registry(request: Request) -> SegmenterRegistry:
    return request.app.state.segmenters


@router.get(
    "",
    summary="List swappable water-body segmentation adapters and what inputs they accept",
    description=API_DOCS["segmenters_list"],
)
async def list_segmenters(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    items = []
    for adapter in registry.list():
        info = adapter.info
        try:
            capabilities = await run_in_threadpool(adapter.capabilities)
        except Exception as exc:  # noqa: BLE001 - 한 어댑터의 상태 조회 실패를 격리
            capabilities = {"error": str(exc)}
        items.append(
            {
                "id": info.id,
                "name": info.name,
                "description": info.description,
                "version": info.version,
                "execution": info.execution,
                "input_kind": info.input_kind,
                "built_in": info.built_in,
                "capabilities": capabilities,
            }
        )
    return {
        "items": items,
        "total": len(items),
        "plugin_errors": registry.load_errors,
        "plugin_env": "SEGMENTER_PLUGINS",
        "note_ko": (
            "탐지 모델은 SegmentationAdapter 를 구현해 SEGMENTER_PLUGINS=패키지.모듈:클래스 로 등록한다. "
            "센서·테스트베드 표는 backend/app/wbms/catalog.json(BACKEND_WBMS_CATALOG_JSON) 이 단일 근거다."
        ),
    }


@router.get(
    "/{segmenter_id}",
    summary="Describe one segmentation adapter",
    description=API_DOCS["segmenters_detail"],
)
async def segmenter_detail(segmenter_id: str, request: Request) -> dict[str, Any]:
    try:
        adapter = _registry(request).get(segmenter_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    info = adapter.info
    capabilities = await run_in_threadpool(adapter.capabilities)
    return {
        "id": info.id,
        "name": info.name,
        "description": info.description,
        "version": info.version,
        "execution": info.execution,
        "input_kind": info.input_kind,
        "built_in": info.built_in,
        "capabilities": capabilities,
        "cached_scenes": await run_in_threadpool(adapter.list_cached),
    }
