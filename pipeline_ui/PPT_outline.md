# 발표용 PPT 아웃라인

> Gamma · Tome · Beautiful.ai · ChatGPT PPT 생성기 등 **웹 기반 슬라이드 생성 도구**에 그대로 붙여넣으면 슬라이드로 변환됩니다.
> 한국어 폰트가 깨질 수 있는 도구에서는 `발표자료_파이프라인.pptx` (직접 생성본)를 권장합니다.

---

## Slide 1 · 표지

# 초소형 위성영상 기반 수체 시계열 분석 파이프라인

**IITP · 2세부 (이노뎁) · 3차년도 데모 v0.1**

SAR → U-Net → ConvLSTM → Disaster Sim
관측–예측–대응을 잇는 End-to-End 파이프라인

- **01** 위성 SAR 영상 수신
- **02** U-Net 수체 탐지
- **03** 기상청 데이터 자동 융합
- **04** ConvLSTM 시계열 예측

발표일자: 2026.06.  ·  이노뎁

---

## Slide 2 · 과제 개요 — 2세부 이노뎁의 역할

**컨소시엄 구성**

| 세부 | 기관 | 역할 |
|---|---|---|
| 1세부 | 수자원공사 / 인하대 | 위성 데이터 수집 · 초해상화 · 시각화 · 플랫폼 |
| 2세부 | **이노뎁 / 서울시립대** | **수체·수위 탐지 · 시계열 분석 · 특수목적 적용** |
| 3세부 | 세명소프트 / 스페이스디자인 | 3D 시각화 · 재난재해 4종 시뮬레이션 |

**이노뎁 잔여 업무 (3차년도)**

1. **시계열 분석/예측 기술 개발** — ConvLSTM 모델 초도구현 완료 / 검증·연계·TTA 평가 예정
2. **재난재해 시뮬레이션 연동** — 강우·홍수·가뭄·침수 4종 · API 개발 · 3세부 송신
3. **작물 분류 기술 개발** — 데이터셋 확보 · Segmentation 모델 · POC (TTA 없음)

---

## Slide 3 · 전체 파이프라인 흐름

> 1세부 → **2세부 (이노뎁)** → 3세부 연계

```
[01] SAR 영상 입력    →    [02] 수체 탐지        →    [03] 기상 융합       →    [04] 시계열 예측
TerraSAR-X Sigma0        U-Net Segmentation      기상청 API              ConvLSTM 인코더-디코더
정사보정                  0/1 마스크              강수·기온·습도          미래 N프레임
(1세부 수신)              (1차 처리)              (보조 입력)             (2차 처리)
```

**데이터 인터페이스**

| 위치 | 포맷 |
|---|---|
| 입력 | GeoTIFF (SAR Sigma0) · Float32 · EPSG:32652 |
| 1차 출력 | GeoTIFF 수체 마스크 (0=배경, 1=수체) |
| 보조 입력 | 기상 CSV (강수/기온/습도/풍속) |
| 2차 출력 | 미래 수체 마스크 + 변화율 → 3세부 REST API |

---

## Slide 4 · STEP 1 · 위성 SAR 영상 입력

**입력 데이터 사양**

| 항목 | 값 |
|---|---|
| 위성/센서 | TerraSAR-X (X-band SAR) |
| 해상도 | 3 m / pixel |
| 재방문주기 | 11일 |
| 처리 단계 | SLC → intensity → 정합 → Sigma0 → 정사보정 |
| 좌표계 | EPSG:32652 (UTM Zone 52N) |
| 데이터형식 | GeoTIFF · Float32 [0, 1] 정규화 |
| 패치 크기 | 512 × 512 (overlap 25%) |

**처리 준비 항목**

- Min-Max 정규화 (ASC/DSC 궤도별)
- 결측치 보간 (KNN, n_neighbors=5)
- 지리적 정렬 (`--align_geo`, 공통영역 클리핑)
- 수체 비율 1% 이상 패치만 필터링

---

## Slide 5 · STEP 2 · 수체 탐지 (1차 처리)

**U-Net Semantic Segmentation**

- 출력: 2-class (배경 / 수체)
- Loss: Weighted Sparse Categorical CE
- Class weights: [0.15, 0.85] (불균형 보정)
- Optimizer: Adam, lr=1e-4, batch=10
- Early stopping + ReduceLROnPlateau
- 가중치: `TSX_Unet_Commit.h5`

**검증 성능 (water class)**

| 지표 | 값 |
|---|---|
| IoU | **0.84** |
| Precision | 0.86 |
| Recall | 0.97 |
| F1 | 0.91 |

**산출물**: GeoTIFF (수체 마스크) — `*_SR_label.tif` (지리좌표 포함, 시계열 분석 입력으로 그대로 사용)

---

## Slide 6 · STEP 3 · 기상청 데이터 자동 융합

**데이터 소스**

- **기상청 일자료 조회서비스** (data.go.kr OpenAPI #15059093)
- 관측소 코드: 서울(108) · 부산(159) · 대구(143) ···
- JSON → CSV 자동 변환 파이프라인 내장

**수집 변수**

| JSON 필드 | CSV 컬럼 | 한국어 |
|---|---|---|
| avgRhm | humidity (%) | 평균 습도 |
| sumRn | precipitation (mm) | 일강수량 |
| minTa / maxTa | tmp_min/max (°C) | 최저/최고 기온 |
| avgPa | pressure (hPa) | 평균 기압 |
| avgWs | wind_speed (m/s) | 평균 풍속 |

**모델 융합 방법**

- 기상 벡터를 공간 차원으로 확장
- ConvLSTM 인코더 출력과 concatenation
- 강수량 ↑ → 수체 확장 경향 학습 / 건조기 → 수체 축소 경향 학습

---

## Slide 7 · STEP 4 · ConvLSTM 시계열 예측 (2차 처리)

**모델 아키텍처**

```
[Image Input 512×512] → ConvLSTM Encoder (→ 256 → 128)
                                   ↓
                  Feature Fusion (+ Weather, 공간 확장 결합)
                                   ↓
                  Conv3D Decoder (128 → 256 → 512)
                                   ↓
                  [Output Mask · 미래 N프레임]
```

**학습 설정**

- Batch Size: 1 × GPU 수 (mixed precision)
- Epochs: 100, Early stopping (patience=10)
- Optimizer: Adam + ReduceLROnPlateau
- Input length: 3 → Predict: 2 (자기회귀로 연장)
- Loss: BCE + Dice (hybrid)
- 데이터: 부산 낙동강 (2020.02 ~ 2020.04)

**평가 결과 (validation set)**

| IoU | Dice | Pixel Acc. | MSE |
|---|---|---|---|
| 0.78 ± 0.34 | 0.82 ± 0.33 | 0.99 ± 0.01 | 0.002 ± 0.001 |

**위험 평가 자동 분류** — 정상 (±5%) / 주의 (±5~15%) / 위험 (>15%)

---

## Slide 8 · 데모 UI (Streamlit Web)

> 발표 시연용 인터랙티브 대시보드 · `http://localhost:8501`

**화면 구성**

- **STEP 1** — 위성 SAR 영상 입력 / 업로드
- **▶ 1차 처리 버튼** — 수체 탐지 실행
- **STEP 2** — 수체 탐지 결과 (면적·비율 메트릭)
- **STEP 3** — 기상청 데이터 자동 연동 (차트)
- **▶ 2차 처리 버튼** — 추가 시계열 분석 실행
- **STEP 4** — ConvLSTM 예측 + 위험 평가

**사이드바 (CONTROL PANEL)**

| 섹션 | 항목 |
|---|---|
| 📍 관측 대상 | 지역 · 일자 · 시퀀스 길이 |
| 🌧️ 기상 데이터 | API 키 · 융합 사용 여부 |
| 🔬 모델 설정 | 탐지/시계열 모델 · 임계값 |

**접속 정보**: 내부망 `http://172.18.10.113:8501` · 로컬 `http://localhost:8501`

---

## Slide 9 · 3세부 연계 — 재난재해 4종 시뮬레이션

| 재해 | 정의 | 필요 데이터 |
|---|---|---|
| **강우/폭우** | 기상특보 활용, 위험지역 집중 관찰 요구 | 강수량·기상특보 (기상청 API) |
| **홍수** | 수체 면적 임계값 이상 확대 | 과거+미래 수체 추론 (2세부) |
| **가뭄** | 수체 면적 임계값 이상 축소 (DEM 보간) | 과거+미래 수체 추론 (2세부) |
| **침수** | 수원 없는 지역 내 수체 생성 확률 ↑ | 기상·지역별 침수 발생 통계 |

**데이터 흐름 (REST API)**
1세부 (영상 수신) → 2세부 이노뎁 (수체 탐지 + 시계열 예측) → 3세부 (재난재해 표출)

---

## Slide 10 · 향후 계획 · 3차년도 잔여 일정

| # | 항목 | 내용 |
|---|---|---|
| **01** | **TTA 평가 준비** | 재현성 검증 · 테스트셋 확보 · 성능표 · 평가 시나리오 |
| **02** | **3세부 연계 API** | REST API 스펙 협의 · 데이터 포맷(GeoTIFF/JSON) · 시나리오 송신 |
| **03** | **재난재해 4종** | 강우(API 래핑) · 홍수/가뭄(예측 재활용) · 침수(DEM+통계 신규) |
| **04** | **작물 분류 POC** | 데이터셋 확보 · Segmentation 모델 · 자체 평가 |

**3차년도**: 2026.01.01 ~ 2026.12.31    ·    **전체 과제**: 2024.04 ~ 2026.12

---

## Slide 11 · Thank you / Q & A

# 감사합니다

## Q & A

IITP 초소형 위성영상 기반 주요 지역 분석 및 실감화 지능 기술 개발
2세부 · 이노뎁
minkyu_choi@innodep.com
