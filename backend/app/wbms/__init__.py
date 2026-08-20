"""WBMS real-model integration (wbms:0.13 container via host-runner subprocess).

Two adapters are exposed:

- ``WbmsSegmentationJobAdapter`` — ② detect_water as an asynchronous job
  (CPU inference is ~610 s per scene, so synchronous HTTP is not viable).
- ``WbmsFusionAdapter`` — ④ fusion LSTM correction as a synchronous call
  (~3 s once TensorFlow is warm inside the container).

Execution boundary is a host-side ``docker run --rm`` subprocess. The backend
never mounts docker.sock and never imports TensorFlow itself.
"""

from .config import WbmsSettings

__all__ = ["WbmsSettings"]
