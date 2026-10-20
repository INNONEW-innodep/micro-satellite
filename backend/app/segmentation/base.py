from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SegmenterInfo:
    id: str
    name: str
    description: str
    version: str = "0.0.0"
    execution: str = "in-process"  # in-process | docker | remote
    input_kind: str = "geotiff"  # 어댑터가 받는 장면 형식
    built_in: bool = True


@dataclass(slots=True)
class SceneRequest:
    """센서·지역·날짜로 장면을 특정한다. ``input_path`` 가 있으면 그 파일을 직접 쓴다."""

    sensor: str
    testbed: str
    image_date: str
    input_path: Path | None = None
    force: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationSubmission:
    """비동기 제출 결과. 즉시 끝나면 ``result`` 가 채워진다."""

    job_id: str
    status: str
    cached: bool = False
    result: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class SegmentationAdapter(ABC):
    """장면→마스크 어댑터 인터페이스.

    구현체는 (1) 어떤 센서×테스트베드를 받을 수 있는지 ``capabilities`` 로 밝히고,
    (2) ``submit`` 으로 장면을 받아 잡을 만들며, (3) ``list_cached`` 로 이미 만든 마스크를
    보고한다. 실행 방식(도커·프로세스 내·원격)은 어댑터가 정한다.
    """

    @property
    @abstractmethod
    def info(self) -> SegmenterInfo:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """``{"sensors": {key: {...,"testbeds": [...]}}, "testbeds": {...}, "ready": bool, ...}``"""

        raise NotImplementedError

    @abstractmethod
    def supports(self, sensor: str, testbed: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def submit(self, request: SceneRequest) -> SegmentationSubmission:
        raise NotImplementedError

    @abstractmethod
    def get_job(self, job_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_cached(self) -> list[dict[str, Any]]:
        raise NotImplementedError
