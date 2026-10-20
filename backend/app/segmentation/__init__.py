"""수체 탐지(세그멘테이션) 모델 교체 계층.

예측 어댑터(`adapters/registry.py`)가 마스크→미래 마스크를 다루듯, 이 패키지는
**장면(GeoTIFF)→수체 마스크** 를 다루는 어댑터를 등록·조회한다. 인도받은 WBMS
도커 모델이 첫 내장 어댑터이고, 다른 모델은 :class:`SegmentationAdapter` 를
구현해 ``SEGMENTER_PLUGINS`` 로 등록한다.
"""
