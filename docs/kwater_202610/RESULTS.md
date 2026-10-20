# K-water 공동실증 자료 처리 결과 — 대청댐 직하류 (2026-09-20 ~ 10-03)

서버 `https://wc-api.semyeongsoft.com` 의 장면 파이프라인(`POST /api/v1/scenes`) 결과. 장면마다 등록 → 탐지 → 면적 → 수위 → 예측 → 규약 파일.

| 장면 | 센서 | 날짜 | 상태 | 수체 면적 km² | 위성 수위 (지점) | 예측 1단계 | 탐지 | 비고 |
|---|---|---|---|---|---|---|---|---|
| 20260920_PlanetScope | PlanetScope | 2026-09-20 | done | 36.33 | bangsuro — m; daecheong — m; jojeongji 29.765 m | 17.27 km² (normal) | done 14s | TOA 반사도(대기보정 전) — 참고용 |
| 20260925_Umbra | Umbra | 2026-09-25 | done | 1.32 | jojeongji 29.835 m | 2.38 km² (normal) | done 7s | 우측 관측, 입사각 48°; 3-class 마스크가 없어 영상 유효 범위로 생성(바다 없음 가정) |
| 20260927_PlanetScope | PlanetScope | 2026-09-27 | done | 51.00 | bangsuro 30.236 m; daecheong 71.664 m; jojeongji 29.784 m | 17.27 km² (normal) | done 14s | 지표반사도 |
| 20260928_Umbra | Umbra | 2026-09-28 | done | 1.18 | jojeongji 29.411 m | 2.38 km² (normal) | done 6s | 우측 관측, 입사각 36°; 3-class 마스크가 없어 영상 유효 범위로 생성(바다 없음 가정) |
| 20261002_PlanetScope | PlanetScope | 2026-10-02 | done | 53.01 | bangsuro 29.893 m; daecheong 71.675 m; jojeongji 29.815 m | 17.27 km² (normal) | done 14s | 지표반사도 |
| 20261003_Umbra | Umbra | 2026-10-03 | done | 1.47 | jojeongji 29.793 m | 1.39 km² (normal) | done 6s | 좌측 관측, 입사각 36°; 3-class 마스크가 없어 영상 유효 범위로 생성(바다 없음 가정) |

## SAR·광학 같은 시기 쌍 비교 (SAR 촬영범위 안)

| SAR | 광학 | SAR 물 km² | 광학 물 km² | 교집합 km² | IoU | 비고 |
|---|---|---|---|---|---|---|
| 20260925_Umbra | 20260927_PlanetScope | 0.925 | 0.816 | 0.721 | 0.707 | 유효 겹침 51% |
| 20260928_Umbra | 20260927_PlanetScope | 0.755 | 0.829 | 0.649 | 0.695 | 유효 겹침 41% |
| 20261003_Umbra | 20261002_PlanetScope | 1.176 | 0.953 | 0.861 | 0.678 | 유효 겹침 35% |

## 읽는 법

- 수체 면적은 각 장면 자기 촬영범위 기준이라 센서 간 직접 비교는 위 쌍 비교(SAR 범위 안)로 본다. 광학은 저수지 전체를 담고 SAR 은 댐 서쪽 하류 6×7 km 만 담는다.
- 위성 수위는 지점 AOI 가 촬영범위에 들어온 경우만 나온다. SAR 은 조정지댐(jojeongji)만 들어온다.
- Umbra 는 인수 ICEYE SAR 모델을 그대로 쓴 것이라(동일 X밴드, 학습 자료는 ICEYE 부산) 결과는 참고용이며, 광학과의 IoU 가 그 신뢰도의 근거다.
- 9-20 광학은 지표반사도 자료가 없어 TOA 반사도로 넣었다. 모델 학습 조건과 달라 참고용이다.
