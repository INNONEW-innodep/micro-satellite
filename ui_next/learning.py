"""Beginner learning material for the WATERCAST mask forecasting workflow.

This module deliberately separates three things that are easy to confuse:

* upstream satellite-image preprocessing and water-body detection,
* the mask time-series forecasting contract implemented by this application,
* optional water-level estimation or calibration.

The copy reflects the repository as checked in.  In particular, the only
runtime-registered models are the three built-in baselines.  The supplied U-Net
and ConvLSTM material is source/notebook material; no corresponding checkpoint
or API adapter is present in the current workspace.  Keeping that distinction
in immutable content also makes over-claiming regressions straightforward to
test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html import escape
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class LearningItem:
    label: str
    explanation: str


@dataclass(frozen=True, slots=True)
class InputFormatSpec:
    extension: str
    accepted_shape: str
    recommended_use: str
    caution: str


@dataclass(frozen=True, slots=True)
class PreprocessStep:
    number: int
    title: str
    input_value: str
    action: str
    output_value: str
    quality_check: str


@dataclass(frozen=True, slots=True)
class ModelExplanation:
    model_id: str
    name: str
    status: str
    role: str
    input_value: str
    behavior: str
    output_value: str
    limitation: str


@dataclass(frozen=True, slots=True)
class LearningTopic:
    topic_id: str
    title: str
    short_description: str


FRAME_CONCEPTS = (
    LearningItem(
        "프레임(frame)",
        "특정 관측 날짜의 수체 상태를 담은 2차원 마스크 한 장입니다. 동영상의 초당 프레임이 아니라, 한 번의 위성 관측을 한 장으로 세는 시계열 단위입니다.",
    ),
    LearningItem(
        "입력 프레임",
        "이미 관측된 과거 마스크입니다. 예를 들어 8월 1일·5일·10일 마스크는 입력 3프레임입니다.",
    ),
    LearningItem(
        "예측 프레임",
        "모델이 미래 목표 날짜별로 생성한 마스크입니다. horizon=3이면 미래 마스크 3장을 만듭니다. 목표 날짜가 하루 간격일 때만 horizon=7·14·30이 각각 1주·2주·한 달의 일별 전망을 뜻합니다.",
    ),
    LearningItem(
        "프레임 간격",
        "관측 또는 예측 날짜 사이의 시간 차이입니다. 현재 API는 목표 날짜를 직접 받거나, 없으면 마지막 관측 다음 날부터 하루 간격으로 정합니다. 위성 재방문 간격의 과거 프레임과 매일 생성하는 미래 프레임은 간격이 다를 수 있으므로 파일 개수보다 실제 날짜를 먼저 확인합니다.",
    ),
    LearningItem(
        "파일과 프레임의 차이",
        "UI 직접 업로드는 파일 하나를 2D 프레임 하나로 취급합니다. API의 NPY 한 파일은 [T,H,W]로 여러 프레임을 담을 수 있고, TIFF의 여러 밴드도 각각 프레임으로 해석될 수 있습니다.",
    ),
    LearningItem(
        "장면(scene)과 프레임",
        "장면은 한 날짜에 촬영한 전체 지역 래스터를 가리키는 경우가 많고, 프레임은 모델에 실제로 넣는 한 날짜의 2D 배열입니다. 전체 장면을 그대로 쓰면 둘이 같지만 패치로 자르면 한 장면에서 여러 공간 위치 프레임이 생깁니다.",
    ),
    LearningItem(
        "패치(patch)",
        "큰 프레임을 모델이 처리할 수 있게 자른 작은 공간 조각입니다. 제공 연구 코드는 주로 512×512 패치를 사용합니다. 같은 위치의 패치를 날짜별로 정확히 묶어야 하며, 패치 개수만 맞춘다고 공간 정렬이 보장되지는 않습니다.",
    ),
    LearningItem(
        "시퀀스(sequence)",
        "같은 공간 위치의 프레임을 날짜순으로 쌓은 묶음입니다. 입력 N프레임은 과거 N장, horizon M프레임은 모델이 만들 미래 M장이며 N과 M은 서로 다를 수 있습니다.",
    ),
)


INPUT_FORMATS = (
    InputFormatSpec(
        ".png",
        "파일당 2D 회색조 마스크 1장",
        "눈으로 빠르게 확인하는 데모와 단순 연동. 0=검정(비수체), 255=흰색(수체)를 권장합니다.",
        "CRS·픽셀 크기 같은 지리정보를 보존하지 않으므로, PNG만으로 km²를 알 수 없습니다.",
    ),
    InputFormatSpec(
        ".tif / .tiff / GeoTIFF",
        "2D 단일 밴드 권장; API에서는 밴드별 프레임 해석 가능",
        "공간정보가 필요한 운영 연계. 모든 입력의 CRS·transform·가로·세로가 같으면 출력 GeoTIFF에도 지리정보를 이어갑니다.",
        "확장자가 TIFF여도 원본 위성영상은 곧바로 수체 마스크가 아닙니다. 다중 밴드는 날짜 대응이 모호하므로 1밴드 1날짜를 권장합니다.",
    ),
    InputFormatSpec(
        ".npy",
        "[H,W], [T,H,W], [H,W,1], [T,H,W,1]",
        "NumPy 기반 모델·배치 파이프라인 연동. bool 또는 유한한 비음수 숫자를 사용하고 0/1 값을 권장합니다.",
        "API에서 T개를 디코딩하면 날짜와 선택 수위도 파일 수가 아닌 T개 프레임에 맞춰야 합니다. 객체 배열/pickle은 읽지 않습니다.",
    ),
)


INPUT_FIELDS = (
    LearningItem(
        "마스크(mask)",
        "각 픽셀이 물인지 아닌지를 나타내는 2D 마스크(2차원 숫자 배열)입니다. 현재 로더는 값을 0~1로 정규화한 뒤 threshold 이상을 1(수체), 미만을 0(비수체)으로 만듭니다.",
    ),
    LearningItem(
        "관측 날짜(source date)",
        "그 입력 프레임을 촬영·생성한 날짜입니다. 디코딩된 프레임 수와 1:1이어야 하고 업로드 순서에서 엄격히 증가해야 합니다. 권장 표기는 YYYY-MM-DD입니다.",
    ),
    LearningItem(
        "목표 날짜(target date)",
        "각 예측 프레임이 의미하는 미래 날짜입니다. horizon 수와 1:1이어야 하며 마지막 입력 날짜보다 뒤여야 합니다.",
    ),
    LearningItem(
        "관측 수위(water_level_m)",
        "해당 입력 날짜에 별도 수위계·검증된 산출 절차로 얻은 미터 단위 값입니다. 마스크의 흰색 픽셀 수와는 다른 물리량이며, 값이 없으면 null로 둡니다.",
    ),
    LearningItem(
        "픽셀 면적(pixel_area_m2)",
        "한 픽셀이 지상에서 차지하는 면적입니다. 정확한 CRS와 해상도에서 계산한 값만 넣어야 수체 면적 km²가 의미를 가집니다.",
    ),
    LearningItem(
        "기상(weather)",
        "날짜별 강수(mm), 기온(°C), 습도(%), 풍속(m/s) 같은 보조 입력입니다. observed는 지난 관측이고 scenario는 호출자가 넣은 미래 가정이며 자동 예보가 아닙니다. 전달자료의 '직전 60일 AWS'는 현재 수위를 보정하는 과거 관측 창이고, 미래 예측일의 기상 입력과는 역할이 다릅니다.",
    ),
)


# The delivery deck describes this four-stage upstream contract.  These names
# are kept as data/interface documentation rather than presented as runnable
# modules: the corresponding scripts and their trained weights are not present
# in this workspace.  WATERCAST starts after these products have been checked
# and mapped to its mask/date/water-level contract.
PROJECT_PIPELINE_STEPS = (
    PreprocessStep(
        1,
        "전처리 · preprocess.py",
        "1세부 ICEYE(SAR)·PlanetScope(광학) 원천영상과 날짜·센서·CRS·해상도·NoData",
        "SAR와 광학에 각각 맞는 노이즈·방사/대기·정사 보정, 정규화와 좌표 정렬을 수행합니다.",
        "Processed_*.tif · 수체를 판별하기 전의 정렬·보정 영상",
        "센서별 처리법, CRS·transform·값 단위·NoData와 파일-날짜 대응이 기록됐는가?",
    ),
    PreprocessStep(
        2,
        "수체 탐지 · detect_water.py",
        "Processed_*.tif와 해당 센서용으로 검증된 탐지 모델·threshold",
        "픽셀별 물/비물을 분류하고 원래 지도 격자에 맞춰 수체 마스크를 만듭니다.",
        "WB_*.tif · 0=비수체, 1=수체, 255=NoData(관측/분석 제외)",
        "255가 GeoTIFF NoData로 선언됐는가? 이를 흰색 수체값 255와 혼동하지 않고 라벨 표본을 검수했는가?",
    ),
    PreprocessStep(
        3,
        "수위·면적 산출 · calc_wlwa.py",
        "날짜순 WB_*.tif, 분석영역(AOI), DEM, 픽셀 면적",
        "수체 경계선(water line·수계선)을 찾고 그 위치의 DEM 고도 대표값으로 간접 수위를 추정하며 수체 픽셀로 면적을 계산합니다.",
        "WLWA_*.csv · 날짜별 water_level_m과 water_area_km2 계열",
        "DEM 기준면·수직 단위, AOI·격자, 경계 추출법과 현장값 비교 근거가 있는가?",
    ),
    PreprocessStep(
        4,
        "이종 센서 퓨전 보정 · Correct.py",
        "ICEYE·PlanetScope WLWA 계열, 직전 60일 AWS(기온·강수·습도·일사), 선택 현장 실측",
        "두 센서의 관측 차이와 기상 영향을 결합해 현재까지의 수위 계열을 보정하고 보정 근거를 모드로 남깁니다.",
        "CWLWA_*.csv · corrected_water_level_m과 correction_mode",
        "absolute·relative·none의 의미, AWS 관측 기간·결측 처리, 실측 기준면과 데이터 누출 여부를 기록했는가?",
    ),
)


PROJECT_PIPELINE_NOTES = (
    LearningItem(
        "이 파일명은 무엇인가요?",
        "Processed → WB → WLWA → CWLWA는 전달자료가 정의한 데이터 성숙도 순서입니다. 이름만 바뀌는 것이 아니라 보정 영상 → 수체 마스크 → 수위·면적 표 → 보정 수위 표로 의미가 달라집니다.",
    ),
    LearningItem(
        "0 / 1 / 255를 어떻게 읽나요?",
        "WB GeoTIFF 규약에서는 0이 비수체, 1이 수체, 255가 NoData입니다. 반면 일반 PNG에서는 255를 흰색 수체로 쓰기도 합니다. 따라서 값만 보고 추측하지 말고 metadata의 NoData 선언을 확인해야 합니다. NoData 선언이 빠진 WB 파일은 현재 로더에 넣기 전에 유효영역 마스크를 적용해야 합니다.",
    ),
    LearningItem(
        "보정 모드는 무엇인가요?",
        "absolute는 같은 기준면의 현장 실측으로 절대 수위를 맞춘 결과, relative는 실측 없이 변화 방향·상대량을 맞춘 결과, none은 보정을 적용하지 않은 원 산출값을 뜻하는 전달 규약입니다. relative를 현장 절대 수위처럼 발표하면 안 됩니다.",
    ),
    LearningItem(
        "현재 관측과 미래 예측의 경계",
        "CWLWA와 직전 60일 AWS는 마지막 관측일까지의 입력 이력을 만드는 단계입니다. 그 뒤 WATERCAST가 날짜별 WB 마스크·수위와 미래 날짜에 맞는 기상 예보 또는 명시적 시나리오를 받아 7·14·30일 같은 전망을 생성합니다.",
    ),
    LearningItem(
        "현재 작업공간에서의 상태",
        "위 네 스크립트명과 산출물명은 제공 발표자료의 인계 계약을 설명합니다. 현재 작업공간에서는 동일 이름의 실행 스크립트·학습 가중치를 확인하지 못했으므로, 화면에서 실행 완료된 기능이나 검증 성능으로 주장하지 않습니다.",
    ),
)


PREPROCESS_STEPS = (
    PreprocessStep(
        1,
        "인수·목록화",
        "1세부 원천영상, 촬영시각, 센서, CRS, transform, 해상도, NoData",
        "파일-날짜 대응표를 만들고 누락·중복·손상 파일을 찾습니다.",
        "원천자료 인벤토리와 제외 사유",
        "영상 수와 날짜 수가 맞고 동일 날짜 중복이 없는가?",
    ),
    PreprocessStep(
        2,
        "센서별 보정",
        "SAR 또는 광학 원천영상",
        "센서에 맞는 방사·대기·노이즈·정사 보정과 NoData 처리를 수행합니다. 제공 TerraSAR-X 문서는 intensity→Sigma0, median filter, dB 변환, 정사보정 흐름을 설명합니다.",
        "비교 가능한 보정 영상",
        "단위·값 범위·NoData가 문서화됐는가? 센서가 달라지면 같은 전처리를 기계적으로 재사용하지 않았는가?",
    ),
    PreprocessStep(
        3,
        "동일 격자 정렬",
        "날짜별 보정 영상",
        "하나의 CRS·해상도·공간 범위·가로·세로·transform에 맞춰 재투영·리샘플링·클리핑합니다.",
        "같은 위치의 픽셀이 매 날짜 같은 지점을 뜻하는 영상 묶음",
        "겹쳐 봤을 때 고정 지형이 움직이지 않는가? 모든 H×W가 같은가?",
    ),
    PreprocessStep(
        4,
        "수체 감지",
        "동일 격자 보정 영상",
        "제공 U-Net 또는 검증된 탐지기를 적용하고 확률 출력을 검증된 threshold로 이진화합니다.",
        "날짜별 2D 이진 수체 마스크",
        "라벨과 IoU·정밀도·재현율을 확인했는가? 구름·그림자·SAR speckle 오탐을 표본 검사했는가?",
    ),
    PreprocessStep(
        5,
        "마스크 정제·검수",
        "0/1 마스크와 품질정보",
        "잘못된 값, 구멍, 작은 잡음, NoData를 정책에 따라 처리하되 원본과 처리 이력을 함께 보존합니다.",
        "동일 격자의 PNG/TIFF/NPY 마스크",
        "2D·유한·비음수인가? 값은 0/1 또는 0/255인가? 임의 정제가 실제 변화를 지우지 않았는가?",
    ),
    PreprocessStep(
        6,
        "시계열 묶기",
        "마스크, 날짜, 선택 수위, 픽셀 면적, 기상",
        "날짜순으로 정렬하고 프레임별 메타데이터를 1:1로 연결합니다. 미래 기상은 observed와 scenario를 분리합니다.",
        "WATERCAST 업로드/API 입력 묶음",
        "날짜 간격과 결측을 확인했는가? 학습 때 사용한 전처리·채널·값 범위가 추론에도 같은가?",
    ),
    PreprocessStep(
        7,
        "예측·결과 검수",
        "정리된 시계열과 선택 모델",
        "모델 adapter를 실행하고 threshold를 적용해 미래 마스크와 선택 수위를 공통 응답으로 만듭니다.",
        "프레임별 마스크, 픽셀 수, 면적, 선택 수위, 규칙 기반 risk, artifact",
        "모델 버전·입력 기간·경고를 기록했는가? 미래 정답이 확보되면 별도 평가했는가?",
    ),
)


DETECTION_VS_LEVEL = (
    LearningItem(
        "수체 감지",
        "영상의 각 픽셀을 물/비물로 분류하는 공간 문제입니다. 대표 출력은 2D 이진 마스크이며, '어디까지 물인가'를 답합니다.",
    ),
    LearningItem(
        "수체 면적",
        "마스크에서 물 픽셀을 세고 정확한 픽셀 면적을 곱해 계산합니다. 지리정보가 없으면 픽셀 수까지만 확실합니다.",
    ),
    LearningItem(
        "수위 추정",
        "물 표면의 높이를 m 단위로 구하는 별도 문제입니다. 마스크만으로 자동 확정되지 않으며 수위계, DEM과 수체 경계, 또는 현장 검증된 면적-수위 보정관계가 필요합니다.",
    ),
    LearningItem(
        "시계열 예측",
        "과거 여러 프레임과 선택적인 기상을 이용해 미래 수체 마스크 또는 수위를 추정하는 문제입니다. 감지는 현재 한 장, 예측은 날짜 순서와 미래를 다룹니다.",
    ),
)


RUNTIME_MODELS = (
    ModelExplanation(
        "persistence",
        "Persistence baseline · 마지막 상태 유지",
        "현재 백엔드에 등록된 내장 기준선",
        "API·화면·파일 저장이 끝까지 연결되는지 확인하는 가장 단순한 비교 기준",
        "최소 1개 수체 마스크, 선택 관측 수위; 기상은 사용하지 않음",
        "마지막 입력 마스크를 horizon만큼 그대로 반복합니다. 유효한 마지막 관측 수위가 있으면 그것도 반복합니다.",
        "동일한 미래 마스크들과 선택적인 반복 수위",
        "학습도 수문 계산도 하지 않으므로 변화·홍수·가뭄을 예측하는 모델이 아닙니다. 새 모델은 최소한 이 기준선보다 나은지 비교해야 합니다.",
    ),
    ModelExplanation(
        "irregular-area-trend",
        "Irregular area trend · 불규칙 관측일 다중시점 추세",
        "현재 백엔드에 등록된 내장 기준선",
        "NAS처럼 간격이 일정하지 않은 여러 수체 마스크가 실제로 예측 계산에 쓰이는지 확인",
        "최소 2개 동일 격자 수체 마스크와 각 관측일, 선택적으로 목표일별 강수",
        "모든 관측 수체 픽셀에 관측일 간격을 반영한 로그 선형 추세를 맞춥니다. 목표일 강수는 감쇠 메모리 규칙으로 면적을 소폭 보정하고, 마지막 수체 경계를 목표 픽셀 수까지 확장·축소합니다.",
        "일별 미래 마스크, 사용 관측 수·일 변화율·강수 보정·제한 전후 목표면적 메타데이터",
        "4시점 같은 희소 자료의 설명 가능한 외삽 기준선일 뿐 학습된 수문·유체 모델이 아닙니다. 경계 모양도 물리 흐름 예측이 아니며 미래 정답 홀드아웃으로 별도 검증해야 합니다.",
    ),
    ModelExplanation(
        "weather-morphology",
        "Weather morphology baseline · 기상 규칙 형태 변화",
        "현재 백엔드에 등록된 내장 기준선",
        "기상 날짜 정렬과 모델 옵션 전달, 변화하는 산출물 연동을 설명 가능하게 시험",
        "최소 1개 수체 마스크와 목표 날짜별 기상 행; 수위를 직접 출력하지 않음",
        "기본값에서 일 강수 20mm 단위로 마스크를 최대 3회 팽창시키고, 0.1mm 이하 건조가 3일 누적될 때 1회 침식합니다. 옵션으로 규칙을 바꿀 수 있습니다.",
        "팽창·침식된 미래 마스크와 단계별 operation 메타데이터",
        "지형·유역·배수·토양·유량을 학습하거나 계산하지 않는 데모 규칙입니다. 결과 risk도 면적 변화율 후처리이며 재난 확률이 아닙니다.",
    ),
)


SUPPLIED_MODEL_MATERIAL = (
    ModelExplanation(
        "tsx-unet-material",
        "제공 U-Net 수체 감지 자료",
        "코드·노트북·문서 제공; 현재 API 미등록, 참조된 .h5 체크포인트는 작업공간에서 확인되지 않음",
        "TerraSAR-X 영상 한 시점에서 수체/비수체 픽셀을 분할해 예측용 마스크의 앞단 후보를 생성",
        "문서상 min-max 정규화한 512×512 패치와 0/1 라벨; ASC/DSC별 범위, -9999 처리, 25% 중첩 설명 포함",
        "필터 64→128→256→512의 U-Net이 공간 특징을 압축한 뒤 skip connection으로 경계를 복원합니다. 두 클래스 확률을 내고, class weight [0.15,0.85], Adam 학습률 1e-4, batch 10, weighted sparse categorical cross-entropy를 쓰도록 기록돼 있습니다.",
        "한 날짜의 512×512×2 클래스 확률과 argmax 0/1 수체 마스크; 수위나 미래 시계열을 직접 출력하지 않음",
        "현재 TSX_Unet_Commit.h5와 학습 PKL이 없어 화면에서 재현할 수 없습니다. 노트북에는 Accuracy 0.9777, water IoU 0.8386, F1 0.9122 등의 실행 기록이 있지만 학습 중 사용한 test set을 다시 평가한 정황이 있어 독립 운영 성능이 아닙니다. 새 데이터에서 재현·교차검증하기 전 운영 성능으로 인용하면 안 됩니다.",
    ),
    ModelExplanation(
        "convlstm-material",
        "제공 ConvLSTM 시계열 예측 자료",
        "학습·추론 스크립트와 노트북 제공; 현재 API 미등록, 구성된 .weights.h5 파일은 작업공간에서 확인되지 않음",
        "공간 패턴과 시간 변화를 함께 학습해 다음 수체 마스크 프레임을 만드는 외부 학습 모델 후보",
        "코드상 이미지 [B,T,512,512,1]과 기상 [B,T,7]. 기상 6개 특징에 관측 간격(duration) 특징을 더하는 구성이며 같은 위치의 시간순 패치를 요구합니다.",
        "두 ConvLSTM2D 층이 공간을 512→256→128로 줄이고, Dense로 128×128에 펼친 기상을 결합한 뒤 Conv3DTranspose로 512 크기를 복원해 sigmoid 마스크를 냅니다. T1…Tn-1→T2…Tn shifted-frame 학습과, 직전 예측을 다시 넣는 자기회귀 추론 코드가 있습니다.",
        "미래 마스크 확률 시퀀스; 제공 구현은 수위를 직접 반환하지 않음",
        "현재 가중치·학습 NPY·데이터 계보·재현 평가가 없고 PredictionAdapter에도 연결되지 않았습니다. 날짜 간격과 기상 scaler가 코드에 견고하게 버전 고정되지 않았고, 랜덤 패치 분할은 독립 지역·기간 검증이 아닙니다. 자기회귀 장기 예측은 이전 오차가 누적될 수 있습니다.",
    ),
    ModelExplanation(
        "external-adapter",
        "교체 가능한 외부 학습 모델",
        "PredictionAdapter 구현과 plugin 등록 후 API에서 선택 가능",
        "검증된 ConvLSTM, Transformer, 수문 모델 등 실제 과제 모델을 UI/API 변경 없이 연결",
        "공통 frames [T,H,W], JSON 기상 행, horizon·threshold·날짜·옵션을 받되 모델 고유 전처리는 adapter 내부에서 명시",
        "adapter가 모델 입력 변환과 추론을 수행하고 공통 PredictionOutput으로 되돌립니다.",
        "정확히 [horizon,H,W] 마스크, 선택적인 horizon 길이 water_levels_m, metadata와 warnings",
        "플러그인 등록만으로 정확도가 생기는 것은 아닙니다. 동일 공간·시간 분할의 독립 검증, 체크포인트·전처리 버전 고정, 실패 처리와 성능 한계 문서가 필요합니다.",
    ),
)


MODEL_SWAP_CHECKLIST = (
    LearningItem("1 · 목적 고정", "미래 마스크, 수위, 또는 둘 다 중 무엇을 출력할지와 horizon·목표 날짜 단위를 먼저 고정합니다. 일별 7·14·30일 전망이면 출력 프레임 수도 7·14·30이고 target date도 하루 간격이어야 합니다."),
    LearningItem("2 · 입력 계약 기록", "필요 프레임 수, H×W, 채널, 값 범위, threshold 전/후, 기상 열과 단위, 결측 처리, 날짜 간격을 적습니다. WB_*.tif의 255 NoData 처리와 CWLWA_*.csv의 corrected_water_level_m·correction_mode 매핑도 여기서 확정합니다."),
    LearningItem("3 · 학습 전처리 재사용", "학습 때의 CRS·격자 정렬·정규화·패치 분할·기상 스케일러와 사용한 AWS 시간창을 체크포인트와 함께 버전 관리하고 추론에서 그대로 적용합니다."),
    LearningItem("4 · Adapter 구현", "PredictionAdapter의 info와 predict를 구현합니다. masks는 [horizon,H,W], 수위가 있으면 water_levels_m도 horizon 길이로 반환합니다."),
    LearningItem("5 · Plugin 등록", "`PREDICTOR_PLUGINS='package.module:ClassOrFactory'`로 서버에 등록하고 /api/v1/health의 plugin_errors와 /api/v1/models를 확인합니다."),
    LearningItem("6 · 계약 테스트", "최소 프레임, shape, 날짜, 255 NoData, 기상 누락, NaN, 잘못된 체크포인트, horizon 1/7/14/30을 테스트하고 기존 API 응답 필드가 유지되는지 확인합니다."),
    LearningItem("7 · 성능 검증", "시간·지역이 겹치지 않는 미래 정답 자료에서 persistence와 비교해 IoU/Dice 등 마스크 지표와 MAE/RMSE 등 수위 지표를 따로 평가합니다. 정답이 없는 실행은 성능 검증이 아니라 예측 시연으로 표시합니다."),
    LearningItem("8 · 운영 표시", "모델 ID·버전·체크포인트 해시·전처리 버전·학습 범위·검증 범위·한계·경고를 결과와 발표 자료에 남깁니다."),
    LearningItem("9 · 되돌리기", "새 모델 오류 시 persistence 등 안전한 기준선으로 선택을 되돌릴 수 있게 이전 plugin 설정과 체크포인트를 보존합니다."),
)


HANDOFF_CHECKLIST = (
    LearningItem("원천 계보", "1세부 ICEYE(SAR)·PlanetScope(광학) 원천 파일명, 촬영시각, 센서/제품 단계, 처리 책임자와 버전을 함께 받습니다."),
    LearningItem("필수 마스크", "날짜별 2D 이진 마스크 최소 모델 요구 개수. UI는 파일당 한 프레임, API는 NPY [T,H,W]도 가능."),
    LearningItem("필수 날짜", "각 디코딩 프레임의 촬영/산출 날짜 YYYY-MM-DD. 엄격한 오름차순."),
    LearningItem("필수 격자", "모든 프레임의 같은 H×W와 같은 지상 위치. GeoTIFF면 CRS·transform·해상도·범위를 함께 전달."),
    LearningItem("권장 품질정보", "NoData 정책, 구름/노이즈율, 탐지 모델 ID·체크포인트·threshold, 검수 여부와 제외 사유."),
    LearningItem("선택 수위", "프레임별 m 단위 관측값 또는 null, 관측소·기준면·산출 방식. 마스크 면적에서 임의로 만든 값은 실측으로 표시하지 않음."),
    LearningItem("선택 기상", "날짜별 단위가 명시된 observed/scenario 행과 관측소·출처. 목표 날짜와 맞는 미래값이 없으면 기상 모델 결과가 유지될 수 있음."),
    LearningItem("선택 면적정보", "정확한 pixel_area_m2와 계산 근거. 모르면 UI에서 픽셀 수로만 해석."),
    LearningItem("선택 WLWA/CWLWA", "날짜별 water_level_m·water_area_km2와, 보정값이면 corrected_water_level_m·correction_mode·기준면·단위를 받습니다. UI/API에는 사용할 수위 열을 골라 프레임 날짜와 1:1로 맞춥니다."),
)


GLOSSARY = (
    LearningItem("원천영상", "센서가 촬영하고 아직 수체 마스크로 분류되지 않은 SAR/광학 래스터."),
    LearningItem("ICEYE", "SAR 방식으로 관측하는 초소형 위성 계열. 구름·주야 영향이 비교적 적지만 레이더 노이즈와 관측기하에 맞는 전처리가 필요합니다."),
    LearningItem("PlanetScope", "가시광·근적외선 기반 광학 소형위성 영상 계열. 직관적인 표면 정보를 주지만 구름·그림자와 대기 영향을 품질검수해야 합니다."),
    LearningItem("SAR", "레이더를 쏘아 되돌아오는 신호를 영상화하는 센서 방식. 날씨·주야 영향이 비교적 적지만 speckle 등 센서별 처리가 필요."),
    LearningItem("SLC", "SAR의 진폭과 위상 정보를 복소수로 담은 원천 제품 단계. 현재 예측 API의 0/1 수체 마스크와 전혀 다른 입력."),
    LearningItem("intensity", "복소수 SAR 신호의 세기를 나타낸 값. 제공 문서는 SLC를 intensity로 바꾼 뒤 보정을 진행한다고 설명."),
    LearningItem("Sigma0", "레이더가 지표에서 되돌아온 세기를 관측 기하에 맞춰 보정한 후방산란 계수. 그 자체가 수체 마스크는 아님."),
    LearningItem("dB", "값의 큰 범위를 로그 척도로 표현하는 단위. SAR 후방산란을 보기·처리 쉽게 바꿀 때 흔히 사용."),
    LearningItem("speckle", "SAR 영상에 보이는 소금·후추 같은 간섭성 잡음. 필터링하되 실제 작은 수체 경계를 지우지 않는지 확인해야 함."),
    LearningItem("DEM", "Digital Elevation Model, 지표 높이를 담은 격자. 정사보정이나 검증된 경계고도 기반 수위 추정에 쓸 수 있지만 마스크만으로 DEM 수위가 자동 생성되지는 않음."),
    LearningItem("water line / 수계선", "수체 마스크에서 물과 비물이 맞닿는 경계선. calc_wlwa.py 전달 규약에서는 이 경계 위치의 DEM 고도 대표값으로 수위를 간접 추정하지만, DEM 오차·완만한 지형·식생에 따라 불확실할 수 있습니다."),
    LearningItem("AWS", "Automatic Weather Station, 자동기상관측장비의 관측 자료. 전달자료의 60일 AWS 창은 현재 수위 보정 입력이며 미래 기상 예보와 같은 뜻이 아닙니다."),
    LearningItem("absolute / relative / none", "CWLWA 보정 모드. absolute는 현장 실측 기준면에 맞춘 절대 보정, relative는 실측 없이 상대 변화 보정, none은 미보정을 뜻합니다."),
    LearningItem("ASC / DSC", "위성이 대체로 북쪽으로 지나는 ascending 궤도와 남쪽으로 지나는 descending 궤도. 관측각과 값 분포가 달라 제공 U-Net 전처리는 따로 정규화."),
    LearningItem("광학영상", "가시광·적외선 반사 특성을 기록한 영상. 구름·그림자 영향을 확인해야 함."),
    LearningItem("CRS", "좌표 숫자가 지구상의 어디를 뜻하는지 정한 좌표참조체계."),
    LearningItem("transform", "행·열 픽셀을 실제 지도 좌표로 바꾸는 위치·회전·픽셀 크기 정보."),
    LearningItem("해상도", "픽셀 하나가 나타내는 지상 크기. 같은 H×W라도 해상도와 범위가 다르면 같은 격자가 아님."),
    LearningItem("NoData", "센서 누락이나 분석 제외처럼 유효한 관측값이 없는 픽셀. 물 0과 구분해야 함."),
    LearningItem("정사보정", "센서 자세와 지형 때문에 생긴 위치 왜곡을 줄여 지도 좌표에 맞추는 처리."),
    LearningItem("공동정합(co-registration)", "서로 다른 날짜 영상에서 같은 물체가 같은 픽셀에 오도록 맞추는 처리."),
    LearningItem("리샘플링", "격자를 바꿀 때 새 픽셀 값을 계산하는 처리. 마스크는 클래스가 섞이지 않도록 보통 nearest 방식부터 검토."),
    LearningItem("장면(scene) / 래스터", "한 날짜에 대상 지역을 덮는 전체 격자 영상. 모델 메모리에 맞게 여러 patch로 자를 수 있음."),
    LearningItem("패치(patch)", "큰 장면을 잘라 만든 작은 공간 조각. 제공 코드는 주로 512×512를 사용."),
    LearningItem("overlap / stride", "이웃 패치가 겹치는 비율과 다음 패치 시작점까지의 간격. 512 패치의 25% overlap이면 stride는 384."),
    LearningItem("이진 마스크", "0=비수체, 1=수체처럼 두 상태만 담은 2차원 배열."),
    LearningItem("threshold", "확률·연속값을 물/비물로 가르는 경계. 0.5는 기본값일 뿐 검증으로 정해야 함."),
    LearningItem("시계열", "날짜 순서와 간격을 가진 여러 프레임의 묶음."),
    LearningItem("shifted frame", "과거 T1…Tn-1을 입력, 한 칸 뒤 T2…Tn을 정답으로 만들어 다음 프레임을 학습하는 구성."),
    LearningItem("cadence / Δt", "프레임 사이 실제 시간 간격. 4프레임이 4일을 뜻하지 않으며 날짜와 함께 해석해야 함."),
    LearningItem("horizon", "한 번에 만들 미래 프레임 수. 시간 길이는 목표 날짜 또는 프레임 간격과 함께 봐야 함."),
    LearningItem("외생변수", "수체 마스크 밖에서 변화를 설명하도록 넣는 강수·기온 같은 보조 변수."),
    LearningItem("adapter", "모델 고유 입출력을 WATERCAST 공통 API 모양으로 바꾸는 연결 계층."),
    LearningItem("checkpoint", "학습된 모델 가중치 파일. 구조 코드만 있고 체크포인트가 없으면 학습 결과를 그대로 추론할 수 없음."),
    LearningItem("자기회귀", "방금 예측한 프레임을 다음 예측 입력에 다시 쓰는 방식. 멀리 갈수록 오차가 누적될 수 있음."),
    LearningItem("IoU", "예측 수체와 정답 수체의 교집합을 합집합으로 나눈 영역 겹침 지표."),
    LearningItem("Dice / F1", "예측과 정답의 겹침을 두 배의 교집합으로 요약하는 지표. 평가 threshold와 검증 자료를 함께 밝혀야 함."),
    LearningItem("scaler / 정규화", "모델 입력 범위를 학습 때와 맞추는 변환과 그 파라미터. 체크포인트와 함께 같은 버전을 보존해야 함."),
    LearningItem("MAE", "예측 수위와 실제 수위 차이의 절댓값 평균. 마스크 정확도 지표와 별개."),
    LearningItem("artifact", "예측에서 생성된 PNG·TIFF·NPY·JSON·CSV·ZIP 파일."),
    LearningItem("baseline", "복잡한 모델이 실제로 개선됐는지 비교하는 단순 기준 모델."),
)


PRESENTATION_NARRATION = (
    LearningItem("1 · 입력부터 예측까지", "1세부 ICEYE·PlanetScope 원천자료는 preprocess.py의 Processed_*.tif, detect_water.py의 WB_*.tif, calc_wlwa.py의 WLWA_*.csv, Correct.py의 CWLWA_*.csv 순서로 의미가 바뀝니다. 이 이름들은 전달자료의 계약이며, 현재 예측기는 검수된 날짜별 마스크와 선택 수위부터 받습니다."),
    LearningItem("2 · 프레임", "프레임은 동영상 속도가 아니라 특정 날짜의 수체 상태 한 장입니다. 같은 격자의 여러 프레임을 날짜순으로 쌓아 입력하며, 매일 7·14·30장을 만들 때 각각 일주일·이주일·한 달 전망이 됩니다."),
    LearningItem("3 · 수체와 수위", "수체 감지는 물의 위치, 면적은 물 픽셀 수, 수위는 물 표면 높이입니다. WLWA는 수체 경계선과 DEM으로 수위를 간접 산출하고, CWLWA는 두 센서·직전 60일 AWS·선택 실측으로 그 관측 계열을 보정한다는 전달 규약입니다."),
    LearningItem("4 · 현재 모델", "현재 API의 persistence, irregular-area-trend, weather-morphology는 연동 확인용 기준선입니다. irregular-area-trend만 여러 관측일의 면적 추세를 직접 사용하지만 학습 모델은 아닙니다. 제공된 U-Net과 ConvLSTM 코드는 가중치와 adapter가 없어 아직 실행 모델로 연결됐다고 말하지 않습니다."),
    LearningItem("5 · 교체 구조", "검증된 모델은 PredictionAdapter로 감싸 plugin으로 등록합니다. 그러면 UI와 외부 API는 그대로 두고 모델 ID와 버전만 바꿔 사용할 수 있습니다."),
    LearningItem("6 · 결과 해석", "결과는 미래 마스크·픽셀 수·정확한 경우의 면적·선택 수위·규칙 기반 위험과 다운로드 파일을 제공합니다. MAE·RMSE 같은 수치도 같은 날짜의 미래 정답이 있을 때만 성능이며, 합성 예시나 정답 없는 실행은 시연으로 구분합니다."),
)


LEARNING_TOPICS: Mapping[str, LearningTopic] = MappingProxyType(
    {
        "frame": LearningTopic("frame", "프레임이란?", "한 날짜의 수체 상태 한 장과 입력·예측 프레임의 차이"),
        "input": LearningTopic("input", "무엇을 입력해야 하나요?", "PNG/TIFF/NPY 형식과 날짜·수위·기상 필드"),
        "preprocess": LearningTopic("preprocess", "1세부 자료 전처리 전체 과정", "Processed → WB → WLWA → CWLWA 전달 흐름과 예측 입력 전 품질검수"),
        "detection": LearningTopic("detection", "수체 감지·면적·수위·예측의 차이", "비슷해 보이지만 답하는 질문이 다른 네 개념"),
        "models": LearningTopic("models", "제공 모델과 현재 실행 모델", "기준선 세 개, U-Net·ConvLSTM 자료와 실제 연결 상태"),
        "swap": LearningTopic("swap", "모델을 바꿀 때", "전처리 계약부터 adapter·검증·되돌리기까지"),
        "terms": LearningTopic("terms", "생소한 용어", "처음 보는 사람을 위한 짧은 용어사전"),
        "presentation": LearningTopic("presentation", "발표용 내레이션", "화면과 함께 읽는 여섯 문장"),
    }
)


LEARNING_COMIC_PATH = Path(__file__).resolve().parent / "assets" / "water_pipeline_comic.png"


LEARNING_CSS = r"""
<style>
.wc-learning-hero { margin:4px 0 14px; padding:18px 20px; border-color:rgba(34,211,238,.34)!important; background:linear-gradient(120deg,rgba(8,47,73,.34),rgba(15,23,42,.86))!important; }
.wc-learning-hero h2 { margin:4px 0 7px; color:var(--wc-text); }
.wc-learning-hero p { margin:0; max-width:900px; color:var(--wc-muted); line-height:1.65; }
.wc-learning-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:10px; margin:9px 0 15px; }
.wc-learning-card { min-height:100%; }
.wc-learning-card h4 { margin:5px 0 6px; color:var(--wc-text); font-size:.91rem; }
.wc-learning-card p { margin:0; color:var(--wc-muted); font-size:.79rem; line-height:1.62; white-space:pre-line; }
.wc-learning-kicker { color:var(--wc-cyan); font-size:.63rem; font-weight:800; letter-spacing:.13em; }
.wc-learning-table { width:100%; border-collapse:separate; border-spacing:0; margin:8px 0 14px; overflow:hidden; border:1px solid var(--wc-border); border-radius:var(--wc-radius); background:rgba(15,23,42,.72); }
.wc-learning-table th,.wc-learning-table td { padding:10px 11px; border-bottom:1px solid var(--wc-border); text-align:left; vertical-align:top; font-size:.76rem; line-height:1.55; }
.wc-learning-table th { color:#bae6fd; background:#0b1425; font-weight:750; white-space:nowrap; }
.wc-learning-table td { color:var(--wc-muted); }
.wc-learning-table tr:last-child td { border-bottom:0; }
.wc-learning-status { display:inline-block; margin:0 0 6px; padding:3px 7px; border:1px solid rgba(56,189,248,.32); border-radius:999px; color:#7dd3fc; font-size:.64rem; font-weight:750; }
.wc-learning-callout { margin:9px 0 13px; padding:12px 14px; border-left:3px solid var(--wc-yellow); background:rgba(120,53,15,.15); color:#fde68a; line-height:1.6; }
.wc-learning-visual { margin:9px 0 15px; padding:12px; overflow-x:auto; }
.wc-learning-visual svg { min-width:880px; width:100%; height:auto; display:block; }
.wc-learning-visual figcaption { margin:8px 5px 0; color:var(--wc-dim); font-size:.7rem; }
.wc-learning-flow__packet { animation:wc-learning-pulse 2.2s ease-in-out infinite; transform-box:fill-box; transform-origin:center; }
@keyframes wc-learning-pulse { 0%,100%{opacity:.25;transform:scale(.8)} 50%{opacity:1;transform:scale(1.15)} }
@media(prefers-reduced-motion:reduce){.wc-learning-flow__packet{animation:none;opacity:1}}
@media(max-width:720px){.wc-learning-table{display:block;overflow-x:auto}.wc-learning-visual svg{min-width:760px}}
</style>
"""


def _safe(value: object) -> str:
    return escape(str(value), quote=True)


def learning_card_html(
    title: object,
    body: object,
    *,
    kicker: object = "EXPLAIN",
    status: object | None = None,
) -> str:
    """Return one escaped card using the shared WATERCAST theme classes."""

    status_html = (
        f'<span class="wc-learning-status">{_safe(status)}</span>' if status else ""
    )
    safe_body = _safe(body).replace("\n", "<br/>")
    return (
        '<article class="wc-panel wc-learning-card">'
        f'<div class="wc-learning-kicker">{_safe(kicker)}</div>'
        f"{status_html}<h4>{_safe(title)}</h4><p>{safe_body}</p></article>"
    )


def item_grid_html(
    items: Iterable[LearningItem], *, kicker: object = "KEY POINT"
) -> str:
    """Render escaped learning items as a responsive card grid."""

    cards = "".join(
        learning_card_html(item.label, item.explanation, kicker=kicker)
        for item in items
    )
    return f'<div class="wc-learning-grid">{cards}</div>'


def input_formats_table_html() -> str:
    """Render the runtime input formats exactly as the current loader accepts them."""

    rows = "".join(
        "<tr>"
        f"<td><b>{_safe(spec.extension)}</b></td>"
        f"<td>{_safe(spec.accepted_shape)}</td>"
        f"<td>{_safe(spec.recommended_use)}</td>"
        f"<td>{_safe(spec.caution)}</td>"
        "</tr>"
        for spec in INPUT_FORMATS
    )
    return (
        '<table class="wc-learning-table"><thead><tr>'
        "<th>형식</th><th>현재 허용 shape</th><th>권장 용도</th><th>꼭 확인</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def preprocess_table_html() -> str:
    """Render the seven-stage upstream-to-result preprocessing contract."""

    rows = "".join(
        "<tr>"
        f"<td><b>{step.number}. {_safe(step.title)}</b></td>"
        f"<td>{_safe(step.input_value)}</td>"
        f"<td>{_safe(step.action)}</td>"
        f"<td>{_safe(step.output_value)}</td>"
        f"<td>{_safe(step.quality_check)}</td>"
        "</tr>"
        for step in PREPROCESS_STEPS
    )
    return (
        '<table class="wc-learning-table"><thead><tr>'
        "<th>단계</th><th>들어오는 것</th><th>처리</th><th>나오는 것</th><th>통과 조건</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def project_pipeline_table_html() -> str:
    """Render the named four-stage contract from the project delivery deck."""

    rows = "".join(
        "<tr>"
        f"<td><b>STEP {step.number}</b><br/>{_safe(step.title)}</td>"
        f"<td>{_safe(step.input_value)}</td>"
        f"<td>{_safe(step.action)}</td>"
        f"<td><b>{_safe(step.output_value)}</b></td>"
        f"<td>{_safe(step.quality_check)}</td>"
        "</tr>"
        for step in PROJECT_PIPELINE_STEPS
    )
    return (
        '<table class="wc-learning-table"><thead><tr>'
        "<th>전달자료 단계</th><th>들어오는 것</th><th>쉽게 말하면</th>"
        "<th>다음 단계에 넘기는 것</th><th>발표 전 확인</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def model_cards_html(models: Iterable[ModelExplanation]) -> str:
    """Render escaped, status-labelled model explanations."""

    cards = []
    for model in models:
        body = (
            f"역할: {model.role}\n"
            f"입력: {model.input_value}\n"
            f"동작: {model.behavior}\n"
            f"출력: {model.output_value}\n"
            f"한계: {model.limitation}"
        )
        cards.append(
            learning_card_html(
                f"{model.name} ({model.model_id})",
                body,
                kicker="MODEL FACT",
                status=model.status,
            )
        )
    return '<div class="wc-learning-grid">' + "".join(cards) + "</div>"


def process_storyboard_svg() -> str:
    """Return an original, accessible comic-like workflow illustration.

    It is a conceptual aid, not a screenshot or a claim about one specific
    satellite product.  No external asset or script is referenced.
    """

    panels = (
        (18, "1세부 전달", "원천영상 + 날짜", "SAR / 광학", "#38bdf8"),
        (237, "전처리·정렬", "CRS·격자 통일", "같은 픽셀 = 같은 곳", "#22d3ee"),
        (456, "수체 감지", "영상 → 0/1 마스크", "어디가 물인가?", "#a855f7"),
        (675, "시계열 예측", "과거 N장 → 미래 M장", "언제 어떻게 변할까?", "#f97316"),
    )
    groups: list[str] = []
    for index, (x, title, line1, line2, color) in enumerate(panels, 1):
        # Simple landscape/mask/timeline glyphs make each panel readable even
        # without external images.
        if index == 1:
            glyph = (
                f'<rect x="{x + 18}" y="55" width="166" height="73" rx="7" fill="#07101f" stroke="#334155"/>'
                f'<path d="M{x + 21} 111 Q{x + 61} 70 {x + 93} 106 T{x + 181} 87 V125 H{x + 21}Z" fill="#164e63"/>'
                f'<path d="M{x + 26} 116 Q{x + 75} 98 {x + 111} 116 T{x + 179} 105" fill="none" stroke="{color}" stroke-width="5"/>'
            )
        elif index == 2:
            glyph = (
                '<g stroke="#334155" fill="none">'
                + "".join(
                    f'<path d="M{x + 27 + n * 25} 56 V128 M{x + 18} {64 + n * 16} H{x + 184}"/>'
                    for n in range(6)
                )
                + f'</g><circle cx="{x + 95}" cy="92" r="24" fill="#083344" stroke="{color}" stroke-width="3"/>'
                f'<path d="M{x + 71} 92 H{x + 119} M{x + 95} 68 V116" stroke="{color}" stroke-width="2"/>'
            )
        elif index == 3:
            glyph = (
                f'<rect x="{x + 20}" y="55" width="164" height="73" rx="7" fill="#050b17" stroke="#334155"/>'
                f'<path d="M{x + 43} 119 C{x + 29} 91 {x + 61} 75 {x + 82} 85 C{x + 103} 94 {x + 97} 56 {x + 133} 68 C{x + 165} 78 {x + 160} 112 {x + 177} 119Z" fill="{color}" opacity=".78"/>'
                f'<text x="{x + 102}" y="96" text-anchor="middle" fill="#f1f5f9" font-size="12" font-weight="800">0 / 1</text>'
            )
        else:
            glyph = "".join(
                f'<g transform="translate({x + 19 + n * 42},58)"><rect width="34" height="58" rx="5" fill="#07101f" stroke="{color}" opacity="{.45 + n * .16}"/>'
                f'<path d="M5 50 Q12 {40 - n * 3} 19 45 T29 {30 + n * 2}" fill="none" stroke="{color}" stroke-width="4"/></g>'
                for n in range(4)
            )
        groups.append(
            f'<g data-panel="{index}"><rect x="{x}" y="17" width="202" height="184" rx="12" fill="#0f172a" stroke="{color}" stroke-width="1.4"/>'
            f'{glyph}<text x="{x + 15}" y="151" fill="#f1f5f9" font-size="13" font-weight="800">{_safe(title)}</text>'
            f'<text x="{x + 15}" y="171" fill="#94a3b8" font-size="11">{_safe(line1)}</text>'
            f'<text x="{x + 15}" y="188" fill="#64748b" font-size="10">{_safe(line2)}</text></g>'
        )
    arrows = "".join(
        f'<path d="M{220 + index * 219} 108 H{231 + index * 219}" stroke="#64748b" stroke-width="2" marker-end="url(#wc-learn-arrow)"/>'
        f'<circle class="wc-learning-flow__packet" cx="{225 + index * 219}" cy="108" r="3" fill="#67e8f9"/>'
        for index in range(3)
    )
    return (
        '<figure class="wc-panel wc-learning-visual">'
        '<svg viewBox="0 0 895 220" role="img" aria-labelledby="wc-learning-story-title wc-learning-story-desc">'
        '<title id="wc-learning-story-title">원천영상에서 미래 수체 마스크까지 네 컷 흐름</title>'
        '<desc id="wc-learning-story-desc">1세부 원천영상이 전처리와 동일 격자 정렬, 수체 감지를 거쳐 날짜별 프레임이 되고 시계열 모델이 미래 프레임을 만드는 개념 그림</desc>'
        '<defs><marker id="wc-learn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#64748b"/></marker></defs>'
        + arrows
        + "".join(groups)
        + '</svg><figcaption>개념 그림 · 실제 원천영상·탐지 결과가 아니라 역할과 데이터 경계를 설명하기 위한 자체 제작 도식입니다.</figcaption></figure>'
    )


def frame_timeline_svg() -> str:
    """Return an accessible input/forecast frame timeline."""

    observed = (
        (70, "입력 1", "8/01"),
        (215, "입력 2", "8/05"),
        (360, "입력 3", "8/10"),
    )
    forecast = (
        (570, "예측 1", "8/15"),
        (715, "예측 2", "8/20"),
        (860, "예측 3", "8/25"),
    )
    blocks = []
    for x, label, day in (*observed, *forecast):
        predicted = label.startswith("예측")
        color = "#f97316" if predicted else "#22d3ee"
        blocks.append(
            f'<g><rect x="{x - 48}" y="38" width="96" height="76" rx="9" fill="#0f172a" stroke="{color}" stroke-width="1.5"/>'
            f'<path d="M{x - 35} 100 Q{x - 18} 68 {x} 89 T{x + 35} 70" fill="none" stroke="{color}" stroke-width="5" opacity=".8"/>'
            f'<text x="{x}" y="136" text-anchor="middle" fill="#f1f5f9" font-size="12" font-weight="750">{label}</text>'
            f'<text x="{x}" y="154" text-anchor="middle" fill="#94a3b8" font-size="11">{day}</text></g>'
        )
    return (
        '<figure class="wc-panel wc-learning-visual">'
        '<svg viewBox="0 0 930 185" role="img" aria-labelledby="wc-frame-title wc-frame-desc">'
        '<title id="wc-frame-title">입력 세 프레임과 예측 세 프레임의 날짜 흐름</title>'
        '<desc id="wc-frame-desc">관측일이 서로 다른 과거 수체 마스크 세 장을 입력하고 미래 목표 날짜의 마스크 세 장을 예측하는 예시</desc>'
        '<path d="M58 90 H870" stroke="#334155" stroke-width="2" marker-end="url(#wc-frame-arrow)"/>'
        '<path d="M437 27 V165" stroke="#64748b" stroke-dasharray="5 5"/>'
        '<text x="218" y="22" text-anchor="middle" fill="#67e8f9" font-size="12" font-weight="750">이미 관측한 시계열</text>'
        '<text x="715" y="22" text-anchor="middle" fill="#fdba74" font-size="12" font-weight="750">horizon = 3</text>'
        '<text x="437" y="178" text-anchor="middle" fill="#94a3b8" font-size="10">마지막 관측 이후</text>'
        '<defs><marker id="wc-frame-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#64748b"/></marker></defs>'
        + "".join(blocks)
        + '</svg><figcaption>날짜는 설명용 예시입니다. 프레임 수만 보지 말고 각 프레임의 실제 날짜와 간격을 함께 봐야 합니다.</figcaption></figure>'
    )


def _render_process_visual(st: object) -> None:
    """Prefer the optional original comic and fall back to inline SVG.

    The generated asset is optional so a source-only deployment remains fully
    usable.  Both variants are explicitly labelled as conceptual teaching
    material rather than observed or predicted project data.
    """

    if LEARNING_COMIC_PATH.is_file():
        st.image(  # type: ignore[attr-defined]
            str(LEARNING_COMIC_PATH),
            caption=(
                "개념 설명용 자체 제작 4컷 · 1세부 원천자료가 전처리·수체 감지·시계열 "
                "예측으로 이어지는 흐름이며 실제 관측·예측 결과가 아닙니다."
            ),
            width="stretch",
        )
    else:
        st.markdown(process_storyboard_svg(), unsafe_allow_html=True)  # type: ignore[attr-defined]


def _render_topic(topic_id: str, st: object) -> None:
    if topic_id == "frame":
        st.markdown(frame_timeline_svg(), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown(item_grid_html(FRAME_CONCEPTS, kicker="FRAME"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.info("예: 8월 1일·5일·10일은 3프레임이지만 일정한 4일 간격은 아닙니다. 모델이 시간 간격을 명시적으로 쓰는지도 함께 확인해야 합니다.")  # type: ignore[attr-defined]
        return

    if topic_id == "input":
        st.markdown("##### 현재 백엔드가 받는 마스크 파일")  # type: ignore[attr-defined]
        st.markdown(input_formats_table_html(), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown("##### 파일과 함께 필요한 값")  # type: ignore[attr-defined]
        st.markdown(item_grid_html(INPUT_FIELDS, kicker="INPUT FIELD"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.warning("원본 SAR/광학 TIFF를 단지 확장자가 맞다는 이유로 업로드하면 안 됩니다. 이 예측 API가 기대하는 것은 전처리와 수체 감지를 끝낸 2D 마스크입니다.")  # type: ignore[attr-defined]
        st.info("전달자료 기준으로는 WB_*.tif가 마스크 후보이고 WLWA_*.csv 또는 CWLWA_*.csv의 날짜별 수위 열이 선택 메타데이터입니다. Processed_*.tif는 아직 보정 영상이므로 예측 마스크 입력이 아닙니다. WB의 255 NoData가 올바르게 선언·제외됐는지 먼저 확인하세요.")  # type: ignore[attr-defined]
        st.code(  # type: ignore[attr-defined]
            "delivery_reference/             # 원본 보존·계보 확인용\n"
            "  Processed_20260801.tif        # 보정 영상, 직접 예측 입력 아님\n"
            "  WB_20260801.tif               # 0=비수체, 1=수체, 255=NoData\n"
            "  WLWA_site.csv                 # 관측 수위·면적\n"
            "  CWLWA_site.csv                # 선택 보정 수위·보정 모드\n"
            "watercast_handoff/              # 현재 UI/API용 묶음\n"
            "  masks/2026-08-01_water_mask.tif\n"
            "  masks/2026-08-11_water_mask.tif\n"
            "  frames.csv  # date, filename, water_level_m(optional)\n"
            "  weather.csv # date, kind, precipitation_mm, ...(optional)\n"
            "  metadata.json # CRS/resolution/NoData/model/threshold/provenance",
            language="text",
        )
        return

    if topic_id == "preprocess":
        st.markdown(process_storyboard_svg(), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown("##### 전달자료에 적힌 4단계 파일 흐름")  # type: ignore[attr-defined]
        st.caption("파일명은 곧 데이터의 처리 단계를 뜻합니다. 왼쪽에서 오른쪽으로 읽고, CWLWA까지는 '관측 이력 준비', 그 다음이 미래 예측입니다.")  # type: ignore[attr-defined]
        st.markdown(project_pipeline_table_html(), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown(item_grid_html(PROJECT_PIPELINE_NOTES, kicker="READ THE OUTPUT"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.warning("중요: WB_*.tif의 255는 전달 규약상 NoData입니다. 일반 흑백 PNG의 255=수체 표기와 반대일 수 있으므로, GeoTIFF NoData 메타데이터가 없으면 유효영역을 먼저 분리한 뒤 0/1 마스크로 전달하세요.")  # type: ignore[attr-defined]
        st.markdown("##### WATERCAST에 넣기 전 공통 품질검수")  # type: ignore[attr-defined]
        st.markdown(preprocess_table_html(), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown("##### 1세부에 요청할 전달 묶음")  # type: ignore[attr-defined]
        st.markdown(item_grid_html(HANDOFF_CHECKLIST, kicker="HANDOFF"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.info("저장소의 convert_data.py는 같은 CRS·해상도의 TIFF들을 공통 겹침 영역으로 자르는 기능을 제공합니다. 서로 다른 CRS·해상도 자료는 그 전에 명시적으로 재투영·리샘플링해야 합니다.")  # type: ignore[attr-defined]
        return

    if topic_id == "detection":
        st.markdown(item_grid_html(DETECTION_VS_LEVEL, kicker="DO NOT CONFUSE"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown(  # type: ignore[attr-defined]
            '<div class="wc-learning-callout"><b>핵심:</b> 흰색 픽셀이 많아졌다는 사실만으로 수위가 몇 m인지 자동 결정할 수 없습니다. '
            "현재 백엔드는 모델이 수위를 직접 반환하거나 사용자가 검증된 면적-수위 보정식을 넣은 경우에만 수위를 만들고, 아니면 null로 둡니다.</div>",
            unsafe_allow_html=True,
        )
        return

    if topic_id == "models":
        st.markdown("##### 지금 UI/API에서 선택 가능한 모델")  # type: ignore[attr-defined]
        st.markdown(model_cards_html(RUNTIME_MODELS), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown("##### 저장소에 제공됐지만 아직 실행 API에 연결되지 않은 자료")  # type: ignore[attr-defined]
        st.markdown(model_cards_html(SUPPLIED_MODEL_MATERIAL[:2]), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.markdown("##### 검증된 새 모델을 연결할 때의 공통 자리")  # type: ignore[attr-defined]
        st.markdown(model_cards_html(SUPPLIED_MODEL_MATERIAL[2:]), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.warning("U-Net 문서에 적힌 예시 검증값이나 별도 발표 폴더의 그림을 현재 운영 성능으로 사용하지 마세요. 현재 작업공간에는 그 결과를 재현할 체크포인트와 데이터 계보가 없습니다.")  # type: ignore[attr-defined]
        st.info("예전 발표 UI에는 DeepLabv3+·SegFormer·PredRNN·SimVP 이름도 선택지로 보였지만, 이 저장소에는 해당 구현·가중치·입력 계약이 없고 선택값이 실제 계산에 사용되지 않았습니다. 제공된 실행 모델로 세면 안 됩니다.")  # type: ignore[attr-defined]
        return

    if topic_id == "swap":
        st.markdown(item_grid_html(MODEL_SWAP_CHECKLIST, kicker="SWAP CHECK"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        st.warning("현재 PredictionAdapter는 과거 수체 마스크로 미래 마스크를 만드는 ‘시계열 예측 모델’용입니다. U-Net처럼 원천영상 한 장을 마스크로 바꾸는 ‘수체 감지 모델’은 1세부/앞단에서 유지하거나 별도 DetectionAdapter와 API를 설계해야 하며, 그대로 PredictionAdapter에 꽂는 모델이 아닙니다.")  # type: ignore[attr-defined]
        st.code(  # type: ignore[attr-defined]
            "# 등록 형식 예시\n"
            "export PREDICTOR_PLUGINS='my_models.convlstm:ConvLSTMAdapter'\n"
            "# 확인\n"
            "curl http://localhost:8000/api/v1/health\n"
            "curl http://localhost:8000/api/v1/models",
            language="bash",
        )
        st.info("화면에 보인다는 것은 adapter가 로드됐다는 뜻일 뿐, 정확도가 검증됐다는 뜻은 아닙니다. 모델 카드에 검증 기간·지역·지표·체크포인트 버전을 별도로 기록하세요.")  # type: ignore[attr-defined]
        return

    if topic_id == "terms":
        st.markdown(item_grid_html(GLOSSARY, kicker="GLOSSARY"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        return

    if topic_id == "presentation":
        st.caption("아래 여섯 카드를 순서대로 읽으면 약 1~2분 분량의 기술 설명이 됩니다.")  # type: ignore[attr-defined]
        st.markdown(item_grid_html(PRESENTATION_NARRATION, kicker="TALK TRACK"), unsafe_allow_html=True)  # type: ignore[attr-defined]
        return

    raise KeyError(f"unknown learning topic {topic_id!r}")


def render_learning_help(topic_id: str, expanded: bool = False) -> None:
    """Render one page-local ``설명 보기`` expander.

    ``topic_id`` must be one of :data:`LEARNING_TOPICS`.  It intentionally
    fails loudly for unknown identifiers so a navigation bug cannot present the
    wrong technical explanation.
    """

    try:
        topic = LEARNING_TOPICS[topic_id]
    except KeyError as exc:
        available = ", ".join(LEARNING_TOPICS)
        raise KeyError(
            f"unknown learning topic_id {topic_id!r}; available: {available}"
        ) from exc

    import streamlit as st

    st.markdown(LEARNING_CSS, unsafe_allow_html=True)
    with st.expander(f"설명 보기 · {topic.title}", expanded=expanded):
        st.caption(topic.short_description)
        _render_topic(topic_id, st)


def render_learning_page() -> None:
    """Render the full beginner and presentation learning center."""

    import streamlit as st

    st.markdown(LEARNING_CSS, unsafe_allow_html=True)
    st.markdown(
        '<section class="wc-panel wc-learning-hero">'
        '<div class="wc-learning-kicker">FROM RAW DATA TO FORECAST</div>'
        '<h2>수체 감지부터 시계열 예측까지, 처음부터 이해하기</h2>'
        '<p>1세부 원천자료를 무엇으로 바꿔 받아야 하는지, 프레임과 마스크가 무엇인지, '
        '수체·면적·수위가 왜 다른지, 현재 모델과 교체 방법을 순서대로 펼쳐 보세요.</p></section>',
        unsafe_allow_html=True,
    )
    _render_process_visual(st)
    st.info(
        "현재 WATERCAST의 직접 입력 경계는 ‘원천 위성영상’이 아니라 ‘날짜별 2D 수체 마스크’입니다. "
        "원천자료를 받으면 센서별 보정→동일 격자 정렬→수체 감지→품질검수를 먼저 수행해야 합니다."
    )
    for topic_id in LEARNING_TOPICS:
        render_learning_help(topic_id, expanded=topic_id == "frame")


__all__ = [
    "DETECTION_VS_LEVEL",
    "FRAME_CONCEPTS",
    "GLOSSARY",
    "HANDOFF_CHECKLIST",
    "INPUT_FIELDS",
    "INPUT_FORMATS",
    "LEARNING_COMIC_PATH",
    "LEARNING_TOPICS",
    "MODEL_SWAP_CHECKLIST",
    "PREPROCESS_STEPS",
    "PRESENTATION_NARRATION",
    "PROJECT_PIPELINE_NOTES",
    "PROJECT_PIPELINE_STEPS",
    "RUNTIME_MODELS",
    "SUPPLIED_MODEL_MATERIAL",
    "InputFormatSpec",
    "LearningItem",
    "LearningTopic",
    "ModelExplanation",
    "PreprocessStep",
    "frame_timeline_svg",
    "input_formats_table_html",
    "item_grid_html",
    "learning_card_html",
    "model_cards_html",
    "preprocess_table_html",
    "process_storyboard_svg",
    "project_pipeline_table_html",
    "render_learning_help",
    "render_learning_page",
]
