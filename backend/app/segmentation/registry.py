from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable

from .base import SegmentationAdapter, SegmenterInfo


class SegmenterRegistry:
    def __init__(self) -> None:
        self._items: dict[str, SegmentationAdapter] = {}
        self.load_errors: list[str] = []

    def register(self, adapter: SegmentationAdapter) -> None:
        info = adapter.info
        if not isinstance(info, SegmenterInfo):
            raise TypeError("adapter.info 는 SegmenterInfo 여야 합니다")
        if not callable(getattr(adapter, "submit", None)):
            raise TypeError("adapter.submit 이 호출 가능해야 합니다")
        if info.id in self._items:
            raise ValueError(f"segmenter id {info.id!r} 가 이미 등록되어 있습니다")
        self._items[info.id] = adapter

    def get(self, segmenter_id: str) -> SegmentationAdapter:
        try:
            return self._items[segmenter_id]
        except KeyError as exc:
            raise KeyError(f"unknown segmenter {segmenter_id!r}; available: {', '.join(sorted(self._items))}") from exc

    def list(self) -> list[SegmentationAdapter]:
        return [self._items[key] for key in sorted(self._items)]

    def resolve(self, sensor: str, testbed: str) -> SegmentationAdapter | None:
        """센서×테스트베드를 받을 수 있는 첫 어댑터."""

        for adapter in self.list():
            try:
                if adapter.supports(sensor, testbed):
                    return adapter
            except Exception:  # noqa: BLE001 - 한 어댑터의 오류가 탐색을 막지 않게
                continue
        return None

    def load_plugins(self, specs: str | Iterable[str]) -> None:
        if isinstance(specs, str):
            plugin_specs = [item.strip() for item in specs.split(",") if item.strip()]
        else:
            plugin_specs = [item.strip() for item in specs if item.strip()]
        for spec in plugin_specs:
            try:
                self.register(_load_plugin(spec))
            except Exception as exc:  # noqa: BLE001
                self.load_errors.append(f"{spec}: {exc}")

    def __len__(self) -> int:
        return len(self._items)


def _load_plugin(spec: str) -> SegmentationAdapter:
    if ":" not in spec:
        raise ValueError("plugin must use 'package.module:ClassOrFactory' syntax")
    module_name, attribute_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute_name)
    if inspect.isclass(candidate) or callable(candidate):
        candidate = candidate()
    if not isinstance(candidate, SegmentationAdapter):
        raise TypeError("plugin must produce a SegmentationAdapter instance")
    return candidate
