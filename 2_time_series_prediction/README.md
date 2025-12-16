# 시계열 예측 프로젝트 (Time Series Prediction)

위성 이미지 시계열 데이터와 기상 데이터를 활용한 ConvLSTM 기반 예측 모델

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [환경 설정](#환경-설정)
3. [데이터 준비](#데이터-준비)
4. [기상 데이터 수집](#기상-데이터-수집)
5. [모델 학습](#모델-학습)
6. [모델 평가](#모델-평가)
7. [추론 및 시각화](#추론-및-시각화)
8. [유틸리티 스크립트](#유틸리티-스크립트)
9. [파일 구조](#파일-구조)
10. [문제 해결](#문제-해결)

---

## 프로젝트 개요

이 프로젝트는 위성 래스터 이미지의 시계열 변화를 예측하기 위한 딥러닝 파이프라인입니다. ConvLSTM(Convolutional Long Short-Term Memory) 아키텍처를 사용하며, 기상 데이터를 보조 입력으로 활용하여 예측 정확도를 향상시킵니다.

### 주요 기능
- 래스터(.tif) → NumPy 배열 변환 및 패치 분할
- **지리적 정렬(Geo-alignment)**: 서로 다른 범위의 위성 이미지를 공통 영역으로 정렬
- 기상청 API를 통한 기상 데이터 자동 수집 및 JSON→CSV 변환
- ConvLSTM 인코더-디코더 모델 학습
- 다양한 메트릭(IoU, Dice, F1 등)을 통한 정량적 평가
- 자기회귀(Autoregressive) 방식의 연속 프레임 예측
- 패치 병합을 통한 전체 이미지 복원 및 시각화

---

## 환경 설정

### 방법 1: Docker 사용 (권장)

Docker 이미지를 빌드하고 컨테이너를 실행합니다:

```bash
# 프로젝트 루트에서 Docker 이미지 빌드
cd infra/docker
docker build -t conv-lstm:gpu .

# 개발 컨테이너 실행 (GPU 지원)
cd ../script
./run_dev_container.sh
```

컨테이너 내에서 Jupyter Lab 실행:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 방법 2: 로컬 환경

필요 패키지 설치:

```bash
pip install numpy pandas matplotlib tensorflow keras rasterio geopandas shapely imageio requests
```

**요구사항:**
- Python 3.11+
- TensorFlow 2.17+ (GPU 지원 권장)
- Keras 3.0+

---

## 데이터 준비

### 1단계: 래스터 데이터 배치

시계열 예측할 래스터 파일(.tif)을 준비합니다. 파일명은 날짜순으로 정렬되도록 명명하세요.

**예시: `data/busan/raster/` 디렉토리 구조**
```
data/busan/raster/
├── 20200218_SR_label.tif   # 1번째 시점
├── 20200312_SR_label.tif   # 2번째 시점
├── 20200325_SR_label.tif   # 3번째 시점
└── 20200414_SR_label.tif   # 4번째 시점
```

**변환된 이미지 예시:**

![변환된 이미지](data/busan/img/20200218_SR_label.png)
*시점 1: 2020년 2월 18일*

![변환된 이미지](data/busan/img/20200312_SR_label.png)
*시점 2: 2020년 3월 12일*

![변환된 이미지](data/busan/img/20200325_SR_label.png)
*시점 3: 2020년 3월 25일*

![변환된 이미지](data/busan/img/20200414_SR_label.png)
*시점 4: 2020년 4월 14일*

> ⚠️ **주의사항:**
> - 모든 래스터 파일은 **동일한 CRS(좌표계)**와 **동일한 해상도**를 가져야 합니다.
> - 파일명이 알파벳/숫자 순으로 정렬되어 시간순이 되도록 명명하세요.
> - 최소 4개 이상의 시점 데이터가 필요합니다.

### 2단계: 지리 정보 확인 (선택)

서로 다른 시점의 위성 이미지는 촬영 범위가 다를 수 있습니다. 변환 전에 지리 정보를 확인하세요:

```bash
python check_geo_info.py --raster_dir data/busan/raster
```

**출력 예시:**
```
공통 영역 (모든 이미지가 겹치는 부분):
  West (left):   480933.000000
  South (bottom): 3888207.000000
  East (right):  509310.000000
  North (top):   3914898.000000
  공통 영역 픽셀 크기 (예상): 9459 x 8897
```

### 3단계: 래스터 → NumPy 변환 (지리적 정렬 포함)

`convert_data.py` 스크립트를 사용하여 래스터 데이터를 학습용 NumPy 배열로 변환합니다.

#### 기본 변환 (정렬 없음)
```bash
python convert_data.py \
    --raster_dir ./data/busan/raster \
    --npy_dir ./data/busan/npy \
    --img_dir ./data/busan/img \
    --save_data_path ./data/busan/data.npy
```

#### 지리적 정렬 포함 (권장) ⭐
```bash
python convert_data.py \
    --raster_dir ./data/busan/raster \
    --npy_dir ./data/busan/npy \
    --img_dir ./data/busan/img \
    --save_data_path ./data/busan/data.npy \
    --align_geo \
    --save_geo_info ./data/busan/geo_info.json
```

**인자 설명:**
| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--raster_dir` | 입력 래스터 파일 디렉토리 | `./data/raster` |
| `--npy_dir` | 개별 npy 파일 저장 디렉토리 | `./data/npy` |
| `--img_dir` | 시각화 이미지 저장 디렉토리 (선택) | None |
| `--save_data_path` | 최종 전처리 데이터 저장 경로 | `./data/data.npy` |
| `--align_geo` | **지리적 정렬 활성화** (공통 영역으로 클리핑) | False |
| `--save_geo_info` | 공통 영역 정보를 JSON으로 저장 | None |

**`--align_geo` 옵션 사용 시:**
1. 모든 래스터 파일의 공통 영역(intersection) 자동 계산
2. 각 이미지를 공통 영역으로 클리핑
3. 모든 이미지가 동일한 지리적 위치에 정렬됨
4. 시계열 분석 시 정확한 위치 비교 가능

**출력 예시:**
```
============================================================
지리적 정렬(Geo-alignment) 활성화
============================================================

공통 영역 정보:
  CRS: EPSG:32652
  공통 영역 크기: 9459 x 8897 픽셀

Converted (aligned) 20200218_SR_label.tif → 20200218_SR_label.npy, shape: (8897, 9459)
Converted (aligned) 20200312_SR_label.tif → 20200312_SR_label.npy, shape: (8897, 9459)
...
✓ 모든 이미지가 동일한 패치 수를 가집니다: 342

Preprocessing completed. Dataset shape: (342, 4, 512, 512, 1)
```

**패치 분할 결과 시각화:**

원본 이미지를 패치로 분할한 결과를 시각화하려면:

```bash
python visualize_patches.py \
    --input_path data/busan/npy/20200218_SR_label.npy \
    --output_path data/busan/img/20200218_patches.png \
    --patch_size 512
```

이 명령어는 원본 이미지와 패치 그리드를 나란히 보여주는 시각화를 생성합니다.

![패치 분할 결과](data/busan/img/20200218_patches.png)
*원본 이미지(왼쪽)와 패치 그리드(오른쪽) 비교*

### 4단계: 정렬 상태 확인

변환 후 정렬 상태를 확인합니다:

```bash
python verify_alignment.py \
    --data_path data/busan/data.npy \
    --output_dir data/busan \
    --raster_dir data/busan/raster
```

**정상 출력 예시:**
```
✓ 정렬 상태 양호
  시간에 따른 물 영역 변화가 10% 미만입니다.
  이는 정상적인 시계열 변화 범위입니다.
```

**정렬 상태 확인 시각화:**

![정렬 상태 확인](data/busan/alignment_verification.png)
*프레임 간 차이 분석 - 각 프레임과 첫 번째 프레임의 차이, 연속 프레임 간 차이를 시각화*

![정렬 비교](data/busan/alignment_comparison.png)
*정렬 전/후 비교 - 전체 이미지와 공통 영역 클리핑 결과 비교*

---

## 기상 데이터 수집

기상 데이터는 모델의 보조 입력으로 사용되어 예측 정확도를 향상시킵니다.

### 기상청 API를 통한 데이터 수집

`examples/req_weather_data.py` 스크립트를 수정하여 기상 데이터를 수집합니다:

```python
# examples/req_weather_data.py 수정
start_date = "20200218"  # 첫 번째 래스터 날짜
end_date = "20200414"    # 마지막 래스터 날짜
```

```bash
python examples/req_weather_data.py
```

**API 정보:**
- 출처: [기상청 일자료 조회서비스](https://www.data.go.kr/data/15059093/openapi.do)
- 관측소 코드(stnIds): 서울(108), 부산(159), 대구(143) 등

### JSON → CSV 변환

수집된 JSON 파일을 CSV로 변환합니다:

```bash
python examples/json_to_csv.py \
    --json_path data/weather_20200218_20200414.json \
    --csv_path data/busan/weather.csv
```

**변환되는 필드:**
| JSON 필드 | CSV 컬럼 | 설명 |
|-----------|----------|------|
| `tm` | 일자(date) | 날짜 |
| `avgRhm` | humidity (%) | 평균 습도 |
| `sumRn` | precipitation (mm) | 일강수량 |
| `minTa` | tmp_min (°C) | 최저 기온 |
| `maxTa` | tmp_max (°C) | 최고 기온 |
| `avgPa` | pressure (hPa) | 평균 기압 |
| `avgWs` | wind_speed (m/s) | 평균 풍속 |

### 기상 데이터 형식

CSV 파일 형식:

```csv
일자(date),간격 (Δt),humidity (%),precipitation (mm),tmp_min (°C),tmp_max (°C),pressure (hPa),wind_speed (m/s)
2020. 2. 18,,55,0.3,17,27,1020,3
2020. 2. 19,,58,2.4,18,28,1020,3
...
```

**기상 데이터 CSV 예시:**

| 일자(date) | 간격 (Δt) | humidity (%) | precipitation (mm) | tmp_min (°C) | tmp_max (°C) | pressure (hPa) | wind_speed (m/s) |
|------------|-----------|--------------|---------------------|--------------|--------------|----------------|------------------|
| 2020. 2. 18 | | 55 | 0.3 | 17 | 27 | 1020 | 3 |
| 2020. 2. 19 | | 58 | 2.4 | 18 | 28 | 1020 | 3 |
| 2020. 2. 20 | | 60 | 0.0 | 19 | 29 | 1018 | 4 |
| 2020. 2. 21 | | 62 | 5.2 | 16 | 25 | 1015 | 5 |

---

## 모델 학습

### 기본 학습 실행

```bash
python train_conv_lstm.py
```

### 주요 설정 (train_conv_lstm.py 내부 수정)

```python
# 데이터 경로 설정
SEQ_DATA_PATH = "./data/busan/data.npy"           # 전처리된 데이터
WEATHER_DATA_PATH = "./data/weather_20200218_20200414.csv"  # 기상 데이터
SAVE_WEIGHT_DIR = "./weights"
SAVE_WEIGHT_PATH = "busan_model.weights.h5"

# 기상 데이터 간격 설정 (래스터 날짜 간격에 맞춤)
DURATIONS = [1, 2, 2, 1]  # 기상 데이터 일수에 맞게 조정
```

### 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| Batch Size | 1 × GPU 수 | GPU 메모리에 따라 조정 |
| Epochs | 100 | 최대 학습 반복 횟수 |
| Early Stopping | patience=10 | 검증 손실 개선 없으면 조기 종료 |
| Learning Rate | Adam 기본값 | ReduceLROnPlateau로 자동 조절 |
| Mixed Precision | True | 메모리 절감을 위한 float16 사용 |

### 모델 아키텍처

```
[Image Input] ──┐
                ├──> ConvLSTM Encoder ──> Feature Fusion ──> Conv3D Decoder ──> [Output]
[Weather Input]─┘

인코더: 512×512 → 256×256 → 128×128 (다운샘플링)
결합: 기상 데이터를 공간 차원으로 확장하여 결합
디코더: 128×128 → 256×256 → 512×512 (업샘플링)
```

**학습 과정 예시:**

학습이 진행되면 다음과 같은 출력을 확인할 수 있습니다:
```
Epoch 1/100
  - loss: 0.5234 - val_loss: 0.4891
  - lr: 0.0010
  
Epoch 2/100
  - loss: 0.4123 - val_loss: 0.4012
  - lr: 0.0010
  
...
  
Epoch 20/100
  - loss: 0.1234 - val_loss: 0.1156
  - Early stopping triggered (patience=10)
```

### GPU 메모리 부족 시

```python
# train_conv_lstm.py에서 필터 수 감소
f = 8  # 기본값 16에서 감소

# 또는 mixed precision 활성화 확인
USE_MIXED_PRECISION = True
```

---

## 모델 평가

학습된 모델의 성능을 정량적으로 평가합니다.

### 기본 평가 실행

```bash
python evaluate.py \
    --data_path ./data/busan/data.npy \
    --weather_path ./data/weather_20200218_20200414.csv \
    --weight_path ./weights/busan_model.weights.h5 \
    --output_dir ./eval_results \
    --visualize
```

### 평가 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--eval_mode` | 평가 모드 (`standard`, `autoregressive`, `both`) | `both` |
| `--threshold` | 이진화 임계값 | 0.5 |
| `--input_len` | 자기회귀 입력 프레임 수 | 3 |
| `--num_to_predict` | 자기회귀 예측 프레임 수 | 2 |
| `--visualize` | 시각화 저장 | False |
| `--n_worst` | 최하위 샘플 시각화 수 | 5 |
| `--n_best` | 최상위 샘플 시각화 수 | 5 |

### 평가 메트릭

| 메트릭 | 설명 |
|--------|------|
| **IoU** | Intersection over Union - 예측과 GT 영역 중첩 비율 |
| **Dice Score** | F1과 유사한 세그멘테이션 메트릭 |
| **Pixel Accuracy** | 올바르게 분류된 픽셀 비율 |
| **Precision** | 예측된 양성 중 실제 양성 비율 |
| **Recall** | 실제 양성 중 예측된 양성 비율 |
| **F1 Score** | Precision과 Recall의 조화 평균 |
| **MSE/MAE** | 연속값 비교용 오차 메트릭 |

### 결과 출력 예시

```
============================================================
 표준 평가 결과
============================================================
전체 샘플 수: 51
유효 샘플 수: 45 (빈 샘플 제외)
임계값: 0.5

[메트릭 요약]
----------------------------------------
  iou               : 0.7848 ± 0.3452
  dice              : 0.8160 ± 0.3309
  pixel_accuracy    : 0.9948 ± 0.0142
  precision         : 0.5612 ± 0.4890
  recall            : 0.4555 ± 0.4492
  f1                : 0.4827 ± 0.4572
  mse               : 0.0021 ± 0.0015
  mae               : 0.0456 ± 0.0123
============================================================
```

**평가 결과 시각화:**

평가 실행 시 `--visualize` 옵션을 사용하면 다음과 같은 시각화 결과를 확인할 수 있습니다:

- **메트릭 요약**: 각 메트릭의 분포 및 통계
- **유효/빈 샘플 분석**: 유효한 샘플과 빈 샘플의 비율 및 분포
- **최상위/최하위 샘플**: 성능이 가장 좋은/나쁜 샘플들의 시각화

예시 결과는 `eval_results/[timestamp]/visualizations/` 디렉토리에 저장됩니다.

**평가 결과 예시:**

![평가 메트릭 요약](eval_results/20251215_120642/visualizations/metrics_summary.png)
*평가 메트릭 분포 및 통계*

![유효/빈 샘플 분석](eval_results/20251215_120642/visualizations/valid_vs_empty_analysis.png)
*유효 샘플과 빈 샘플 분석*

---

## 추론 및 시각화

새로운 데이터에 대해 예측을 수행하고 결과를 시각화합니다.

### 기본 추론 실행

```bash
python inference.py \
    --data_path ./data/busan/data.npy \
    --weather_path ./data/weather_20200218_20200414.csv \
    --weight_path ./weights/busan_model.weights.h5 \
    --output_dir ./inference_results \
    --input_len 3 \
    --num_to_predict 1
```

### 전체 이미지 병합 (패치 → 원본 크기)

패치들을 병합하여 원본 크기의 전체 이미지로 시각화:

```bash
python inference.py \
    --data_path ./data/busan/data.npy \
    --weather_path ./data/weather_20200218_20200414.csv \
    --weight_path ./weights/busan_model.weights.h5 \
    --output_dir ./inference_results \
    --merge_patches \
    --original_height 8897 \
    --original_width 9459 \
    --patch_size 512
```

### 추론 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input_len` | 입력 프레임 수 | 3 |
| `--num_to_predict` | 예측할 프레임 수 | 2 |
| `--threshold` | 이진화 임계값 (None=원본) | None |
| `--sample_indices` | 추론할 샘플 인덱스 (예: "0,1,2") | 자동 선택 |
| `--save_gif` | GIF 애니메이션 저장 | False |
| `--merge_patches` | 패치 병합 시각화 | False |
| `--no_visualize` | 시각화 비활성화 | False |

### 결과 디렉토리 구조

```
inference_results/
└── 20251215_120000/
    ├── predictions.npy          # 예측 결과 (NumPy)
    ├── images/                   # 개별 프레임 이미지
    │   ├── pred_b0_f0.png
    │   └── pred_b0_f1.png
    ├── visualizations/           # 비교 시각화
    │   └── sample_0.png
    ├── full_images/              # 전체 이미지 (--merge_patches 사용 시)
    │   ├── original_frame_0.png
    │   ├── prediction_frame_3.png
    │   └── full_comparison.png
    └── sample_0.gif              # 애니메이션 (--save_gif 사용 시)
```

**추론 결과 예시:**

**개별 패치 시각화:**
- 각 패치별로 입력 프레임, Ground Truth, 예측 결과를 비교하여 시각화
- `visualizations/sample_*.png` 파일로 저장

![개별 패치 추론 결과](inference_results/20251215_121608/visualizations/sample_0.png)
*개별 패치의 입력 프레임, Ground Truth, 예측 결과 비교*

**전체 이미지 비교:**
`--merge_patches` 옵션 사용 시 전체 이미지로 병합된 결과를 확인할 수 있습니다:

![전체 이미지 비교](inference_results/20251215_121608/full_images/full_comparison.png)
*입력 프레임 → Ground Truth vs 예측 결과 비교 (전체 이미지)*

**예측 결과 상세:**

![예측 프레임](inference_results/20251215_121608/full_images/prediction_frame_3.png)
*예측된 시점 3 (전체 이미지, 패치 병합 후)*

![원본 프레임](inference_results/20251215_121608/full_images/original_frame_3.png)
*실제 시점 3 (Ground Truth, 전체 이미지)*

---

## 유틸리티 스크립트

### 지리 정보 확인

래스터 파일들의 CRS, 범위, 해상도, 공통 영역을 확인합니다:

```bash
python check_geo_info.py --raster_dir data/busan/raster
```

### 회전/변환 분석

래스터 파일들의 Transform 행렬을 분석하여 회전 여부를 확인합니다:

```bash
python check_rotation.py --raster_dir data/busan/raster
```

### 정렬 상태 확인

변환된 데이터의 정렬 상태를 시각적으로 확인합니다:

```bash
python verify_alignment.py \
    --data_path data/busan/data.npy \
    --output_dir data/busan \
    --raster_dir data/busan/raster
```

### 데이터 값 확인

데이터의 값 범위, 프레임 간 변화 등을 상세히 분석합니다:

```bash
python check_data_values.py \
    --data_path data/busan/data.npy \
    --npy_dir data/busan/npy
```

### JSON → CSV 변환

기상청 API JSON 응답을 CSV로 변환합니다:

```bash
python examples/json_to_csv.py \
    --json_path data/weather_20200218_20200414.json \
    --csv_path data/busan/weather.csv
```

### 패치 분할 시각화

원본 이미지가 어떻게 패치로 분할되는지 시각화합니다:

```bash
python visualize_patches.py \
    --input_path data/busan/npy/20200218_SR_label.npy \
    --output_path data/busan/img/20200218_patches.png \
    --patch_size 512
```

이 스크립트는 원본 이미지와 패치 그리드를 나란히 보여주는 시각화를 생성합니다.

---

## 파일 구조

```
2_time_series_prediction/
├── README.md                     # 본 문서
│
├── convert_data.py               # 래스터 → NumPy 변환 (지리적 정렬 포함)
├── data_preprocess.py            # 데이터 전처리 유틸리티
├── train_conv_lstm.py            # 모델 학습 스크립트
├── evaluate.py                   # 모델 평가 스크립트
├── inference.py                  # 추론 스크립트
│
├── check_geo_info.py             # 지리 정보 확인 유틸리티
├── check_rotation.py             # 회전/변환 분석 유틸리티
├── verify_alignment.py           # 정렬 상태 확인 유틸리티
├── check_data_values.py          # 데이터 값 확인 유틸리티
├── visualize_patches.py          # 패치 분할 시각화 유틸리티
│
├── data/
│   ├── busan/                    # 부산 데이터셋
│   │   ├── raster/               # 래스터 파일 (.tif)
│   │   ├── npy/                  # 개별 NumPy 파일
│   │   ├── img/                  # 시각화 이미지
│   │   ├── data.npy              # 전처리된 학습 데이터
│   │   └── geo_info.json         # 지리 정보 (공통 영역)
│   ├── weather_ex.csv            # 기상 데이터 예시
│   └── weather_*.json            # 기상청 API 응답
│
├── weights/                      # 학습된 모델 가중치
│   └── busan_model.weights.h5
│
├── eval_results/                 # 평가 결과
├── inference_results/            # 추론 결과
│
├── examples/                     # 예제 및 유틸리티
│   ├── req_weather_data.py       # 기상 데이터 수집 예제
│   ├── json_to_csv.py            # JSON → CSV 변환
│   ├── raster_to_np_ex.ipynb     # 래스터 변환 노트북
│   └── train_conv_lstm_ex.ipynb  # 학습 예제 노트북
│
└── infra/                        # 인프라 설정
    ├── docker/
    │   └── Dockerfile            # Docker 이미지 정의
    └── script/
        └── run_dev_container.sh  # 컨테이너 실행 스크립트
```

---

## 전체 워크플로우 요약

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 래스터 데이터 준비                                             │
│    data/busan/raster/*.tif                                       │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 지리 정보 확인 (선택)                                          │
│    python check_geo_info.py --raster_dir data/busan/raster      │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 래스터 → NumPy 변환 (지리적 정렬 포함)                          │
│    python convert_data.py --raster_dir data/busan/raster \      │
│        --align_geo --save_geo_info data/busan/geo_info.json     │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 정렬 상태 확인                                                 │
│    python verify_alignment.py --data_path data/busan/data.npy   │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. 기상 데이터 수집 및 변환                                        │
│    python examples/req_weather_data.py                           │
│    python examples/json_to_csv.py --json_path data/weather.json │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. 모델 학습                                                     │
│    python train_conv_lstm.py                                     │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. 모델 평가                                                     │
│    python evaluate.py --data_path data/busan/data.npy \         │
│        --weight_path weights/busan_model.weights.h5 --visualize │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. 추론 및 시각화                                                │
│    python inference.py --data_path data/busan/data.npy \        │
│        --weight_path weights/busan_model.weights.h5             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 문제 해결

### Q1: "ResourceExhaustedError (OOM)" 발생
- **해결**: 배치 크기 감소 또는 필터 수 감소 (`f = 8`)
- Mixed Precision 활성화 확인

### Q2: "durations 합 != 기상 데이터 일수" 오류
- **해결**: `DURATIONS` 리스트의 합이 기상 CSV의 행 수와 일치하도록 수정

### Q3: 래스터 파일 순서가 잘못됨
- **해결**: 파일명이 알파벳 순으로 시간순이 되도록 이름 변경
- 예: `20200218.tif`, `20200312.tif`, `20200325.tif`, `20200414.tif`

### Q4: 이미지 범위(bounds)가 다름
- **해결**: `--align_geo` 옵션을 사용하여 공통 영역으로 정렬
```bash
python convert_data.py --align_geo --raster_dir data/busan/raster ...
```

### Q5: CRS가 다름
- **해결**: QGIS, GDAL 등으로 동일한 CRS로 재투영 후 사용
```bash
gdalwarp -t_srs EPSG:32652 input.tif output.tif
```

### Q6: "Dimension mismatch" 오류
- **원인**: 이미지 시퀀스와 기상 데이터 시퀀스 길이 불일치
- **해결**: 기상 데이터가 이미지 시점 수에 맞게 준비되었는지 확인

### Q7: 정렬 상태가 불량으로 나옴
- `check_data_values.py`로 실제 값 확인
- 이진 마스크(0/1) 데이터에서 1% 내외의 변화는 정상 (실제 수위 변화)

---

## 라이선스

이 프로젝트는 내부 연구 목적으로 개발되었습니다.

## 연락처

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.
