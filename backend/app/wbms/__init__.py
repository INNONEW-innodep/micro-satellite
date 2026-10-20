"""WBMS real-model integration (wbms:0.13 container via host-runner subprocess).

Two adapters are exposed:

- ``WbmsSegmentationJobAdapter`` — ② detect_water as an asynchronous job
  (CPU inference is ~610 s per scene, so synchronous HTTP is not viable).
- ``WbmsFusionAdapter`` — ④ fusion LSTM correction as a synchronous call
  (~3 s once TensorFlow is warm inside the container).

Execution boundary is a host-side ``docker run --rm`` subprocess. The backend
never imports TensorFlow itself. docker.sock 은 compose 가 마운트하면 쓰고(컨테이너 안 탐지 실행, 2026-10-03), 없으면 호스트 CLI 로 돈다.
"""

from .config import WbmsSettings

__all__ = ["WbmsSettings"]
