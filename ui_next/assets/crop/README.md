# 작물 탐지 POC 샘플 자산

## `esa_sentinel2_desert_fields.jpg`

- 표시명: **Desert fields**
- 원천: ESA Multimedia
- 원천 페이지: <https://www.esa.int/ESA_Multimedia/Images/2015/07/Desert_fields>
- 직접 다운로드: <https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2015/07/desert_fields/15536865-1-eng-GB/Desert_fields.jpg>
- 설명: 2015년 Sentinel-2A가 촬영한 사우디아라비아 중앙 피벗 관개 농업 지역의 false-color 게시용 JPEG
- 크레디트: `Copernicus Sentinel data (2015)/ESA`
- 라이선스: [CC BY-SA 3.0 IGO](https://creativecommons.org/licenses/by-sa/3.0/igo/) 또는 ESA Standard Licence
- 다운로드 파일 SHA-256: `dbd6eb75a1b3ae197423a3f419a31b1b97131556cd57f2d7b25faaf5b9c112aa`
- 로컬 파일 크기/해상도: 약 780KB, 1500×639 RGB JPEG

ESA 설명에 따르면 원형 영역은 중앙 피벗 관개 방식으로 만들어진 농지입니다. 이 UI는 붉게
강조된 활발한 식생 신호를 색상지수로 찾습니다. 원본 다중분광 밴드, 정답 segmentation
라벨, GeoTIFF 지리참조와 픽셀 면적은 이 게시용 JPEG에 없으므로 다음 용도로 쓰면 안 됩니다.

- 학습 모델의 정확도 평가
- 작물 품종 분류
- 필지 경계 확정
- 실제 면적 산출
- 운영 의사결정

`전체`, `서부 밀집`, `동부 산개` UI 샘플은 모두 이 한 장에서 파생되며, 잘라낸 범위 외에
픽셀 값을 합성하거나 모델 정답으로 꾸미지 않습니다.
