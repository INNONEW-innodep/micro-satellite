# WATERCAST · 수체 시계열 예측 UI

`pipeline_ui/` 발표용 합성 데모와 분리된 실제 FastAPI 클라이언트입니다. UI 안에서 모델을 흉내 내지 않고 모든 기상 조회·예측·산출물 요청을 백엔드로 보냅니다.

## 실행

백엔드를 먼저 `http://localhost:8000`에 실행한 다음:

```bash
cd ui_next
bash run.sh
```

다른 주소는 환경 변수 또는 사이드바에서 지정합니다.

```bash
WATER_API_URL=http://backend:8000 bash ui_next/run.sh
```

## 작물 탐지 POC

화면 상단 **분석 업무 전환**에서 `🌾 작물 탐지`를 선택합니다. 수체 시계열 예측의 입력·기상·
예측 결과 상태는 그대로 보존되고, 작물 화면의 `crop_*` 상태와 섞이지 않습니다.

1. 내장 ESA 샘플의 전체/서부/동부 보기를 고르거나 PNG/JPEG RGB 이미지를 올립니다.
2. ESA false-color는 붉은 식생 모드, 일반 자연색 RGB는 녹색 식생 모드를 사용합니다.
3. 임계값과 오버레이 불투명도를 조절하면 결과가 즉시 다시 계산됩니다.
4. 원본·오버레이, 0~1 색상 점수, 이진 마스크와 후보 픽셀 비율을 확인합니다.
5. 오버레이 PNG, 마스크 PNG, 계보·설정·한계를 포함한 JSON을 내려받을 수 있습니다.

내장 이미지는 ESA의 [Desert fields](https://www.esa.int/ESA_Multimedia/Images/2015/07/Desert_fields)
공개 Sentinel-2A false-color JPEG입니다. 크레디트는
`Copernicus Sentinel data (2015)/ESA`, 라이선스는
[CC BY-SA 3.0 IGO](https://creativecommons.org/licenses/by-sa/3.0/igo/)입니다. 전체와 두 ROI는
모두 같은 원본 한 장에서 파생되며, 생성형 샘플이나 정답 라벨이 아닙니다.

저장소의 기존 “작물 탐지” 항목에는 구현 코드·학습 데이터·가중치가 없었습니다. 따라서 현재
결과는 작물 종류를 분류하는 AI 결과가 아니라 붉은색/녹색 식생 신호를 찾는 결정론적 기준선입니다.
숲·잔디·붉은 물체를 오탐할 수 있고, 게시용 JPEG에는 지리참조가 없어 실제 면적도 계산하지
않습니다. 검증된 segmentation 모델이 준비되면 `crop_detection.py`의 탐지 부분만 어댑터로
교체하고 원본/마스크/오버레이/내보내기 UI 계약은 유지할 수 있습니다. 자세한 자산 계보는
[assets/crop/README.md](assets/crop/README.md)에 있습니다.

## 가장 빠른 확인

1. 첫 화면에서 **빠른 테스트**를 선택합니다.
2. 부산·ICEYE NAS 실제 라벨, 부산 문서 복원 또는 낙동강·한강·광주 합성 예시 중 하나를 고릅니다.
3. **선택 샘플로 바로 예측**을 누릅니다.
4. 같은 multipart API가 실행되고 결과 추이·위험도·마스크 미리보기 화면으로 바로 이동합니다.

부산 문서 복원 샘플은 저장소 문서 그림의 4개 패널을 이진 마스크로 복원한 `derived_demo`입니다.
원본 래스터·지리참조·픽셀 면적·실측 수위가 없으므로 물리 면적과 수위 정확도 검증에 쓰면
안 됩니다. 낙동강·한강·광주 3개도 고정 합성 시나리오이며 학습 모델의 성능 자료가 아닙니다. 화면에
분류와 제한 사항을 계속 표시합니다.

`NAS 전달자료` 입력 모드에는 실제 NAS 감사 결과가 있습니다. 300개 파일을 8개 그룹으로
분리하고 전체 용량·타임라인·준비도와 PlanetScope UDM2, ICEYE 관측기하, 시립대 결과,
기흥·회동 WorldView 메타데이터 EDA를 제공합니다. 부산 512×512와 ICEYE 512×699 공통격자
라벨은 원본 면적·리샘플 오차를 확인한 뒤 `irregular-area-trend` 기준선으로 바로 실행할 수
있습니다. 이 기준선은 모든 과거 날짜를 사용하지만 학습된 수문 모델은 아닙니다.

광주·한강·낙동강 시나리오 이름과 합성 변화 개념은 기존 `pipeline_ui`에도 있었지만, 현재
선택되는 픽셀 데이터는 `ui_next/samples.py`가 런타임에 새로 그립니다. 해당 지역 실측 파일,
실제 사건 날짜, 실측 수위가 아닙니다. 데이터 화면의 **설명 보기 · 이 샘플은 어디서
왔나요?**에 마스크·기상·수위 생성법과 허용/금지 용도를 나눠 표시합니다.

## 화면 흐름

1. 저장소 샘플로 즉시 테스트하거나 동일 격자의 수체 마스크 여러 개와 날짜, 선택 관측 수위를 입력합니다.
2. 기상청 ASOS 과거 관측(D-1까지) 또는 서버 샘플을 조회합니다. 표를 편집하거나 사용자 CSV로 교체할 수 있으며, 미래 값은 반드시 `scenario`로 표시합니다.
3. `GET /api/v1/models`에서 모델을 선택하거나 모델 ID를 직접 넣어 예측합니다.
4. 프레임별 마스크·면적·선택 수위와 산출물을 시계열 차트에서 확인하고 JSON/CSV/파일로 내려받습니다.
5. **이해 가이드**의 첫 탭에서 1세부 전처리, 프레임, 수체 감지·수위 차이, U-Net·ConvLSTM과 모델 교체를 보고, 둘째 탭에서 모든 메뉴·버튼·입력값·차트, FAQ와 발표 스크립트를 확인합니다.
6. **API 가이드**에서 서버의 `/openapi.json`을 읽어 API 표를 동적으로 만들고 Swagger/ReDoc, curl/Python 예제와 데이터 흐름 그림을 봅니다.

각 업무 화면 위의 `화면 사용법` 펼침 패널도 같은 내용을 현재 단계에 맞춰 짧게 보여줍니다.

### EDA에서 확인하는 값

- 데이터: 프레임·고유 날짜 수, 시작/종료와 경과 일수, 최소/중앙/최대 관측 간격, 해상도, 수체 픽셀 변화
- NAS: 8개 그룹 용량·날짜 lane·기능별 준비도, 원본 센서/밴드/품질/NoData/정합, 부산·ICEYE 공통영역 면적과 축소 오차
- 기상: 전체 행과 observed/scenario 수, 분석 기간, 누적·평균·최대 강수, 강수/핵심 특성 결측률
- 결과: horizon과 목표 기간, 수체 면적 또는 픽셀 변화, 수위 산출 개수·변화, 위험도 분포

픽셀 면적을 확인하지 못한 샘플은 물리 면적처럼 보이지 않도록 km²를 숨기고 `area_pixels`만
사용합니다. EDA 해석 문장은 입력 변화와 예측 정확도를 구분하고, scenario가 예보가 아니라는
한계도 함께 표시합니다.

### 기상청 인증키가 없을 때

기상 화면은 `GET /api/v1/weather/status`로 서버 설정 여부를 먼저 확인합니다. 서버에
`KMA_API_KEY`가 없으면 다음 중 하나를 선택합니다.

1. 공공데이터포털에서 발급받은 키를 화면의 비밀번호 입력 칸에 붙여 넣어 이번 조회에만 사용
2. **키 없이 같은 기간 샘플 조회**로 합성 자료를 불러와 전체 화면/API 흐름을 즉시 시험

샘플과 ASOS 실관측은 표의 `source`와 안내 문구로 구분됩니다.

입력, 기상, 모델, 백엔드 설정의 fingerprint가 바뀌면 해당 설정에 종속된 캐시 결과만 자동 무효화합니다. 순수 상태 함수는 Streamlit 없이 테스트할 수 있습니다.

## 기대 API

| Method | Path | 역할 |
|---|---|---|
| GET | `/api/v1/health` | 서비스 상태 |
| GET | `/api/v1/models` | 등록 모델 목록 |
| GET | `/api/v1/weather/status` | ASOS 인증키 설정과 fallback 기능 상태 |
| GET | `/api/v1/weather/stations` | ASOS 관측소 목록 |
| POST | `/api/v1/weather/observations` | 과거 관측/샘플 조회 |
| POST | `/api/v1/predictions` | multipart 예측 생성 |
| GET | `/api/v1/predictions` | 저장된 예측 목록 |
| GET | `/api/v1/predictions/{id}` | 예측 상태·결과 |
| GET | `/api/v1/predictions/{id}/files` | 산출물 목록 |
| GET | `/api/v1/predictions/{id}/files/{name}` | 산출물 파일 |
| GET | `/api/v1/predictions/{id}/bundle` | 결과 전체 ZIP |

경로는 `api_client.py`의 `Endpoints`에 모아 두었습니다. 모델 교체는 백엔드 모델 레지스트리에 새 ID를 추가하면 되고, UI 코드를 수정할 필요가 없습니다.

백엔드 API는 단일 3D `[T,H,W]` NPY도 디코딩할 수 있지만, 이 경우 `source_dates_json`과 `historical_water_levels_json` 길이는 파일 수가 아니라 디코딩된 T 프레임 수와 일치해야 합니다. 이 고급 입력은 UI 대신 직접 API 호출을 사용합니다.

## 테스트

저장소 루트에서 실행합니다.

```bash
python3 -m unittest discover -s ui_next/tests -v
```

샘플·HTTP 계약·상태 전이·테마 테스트를 함께 실행합니다.

```bash
python3 -m pytest -q ui_next/tests
```

UI 의존성을 설치한 환경에서는 다음으로 화면 스모크 테스트도 할 수 있습니다.

```bash
python3 -m py_compile ui_next/app.py ui_next/api_client.py ui_next/samples.py ui_next/state.py ui_next/theme.py
```
