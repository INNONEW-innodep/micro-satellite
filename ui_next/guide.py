"""Beginner-friendly and presentation-ready WATERCAST usage guide.

The renderer is intentionally independent from ``app.py``.  All explanatory
copy lives in immutable constants so tests can detect missing menu coverage or
accidental claims that a built-in baseline/sample is a validated forecast.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html import escape
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class GuideStep:
    number: int
    menu: str
    title: str
    action: str
    check: str


@dataclass(frozen=True, slots=True)
class GuideItem:
    label: str
    explanation: str


@dataclass(frozen=True, slots=True)
class MenuGuide:
    menu: str
    purpose: str
    when_to_use: str
    controls: tuple[GuideItem, ...]
    reading: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    term: str
    definition: str


@dataclass(frozen=True, slots=True)
class FAQItem:
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class PageHelp:
    page_id: str
    menu: str
    use_order: tuple[str, ...]
    presentation_line: str


WORKFLOW_STEPS = (
    GuideStep(
        1,
        "데이터",
        "시간순 수체 마스크 준비",
        "저장소 샘플을 선택하거나 같은 격자의 PNG/TIFF/NPY 마스크를 날짜와 함께 등록합니다.",
        "파일 수와 날짜 수가 같고 날짜가 중복되지 않는지 확인합니다.",
    ),
    GuideStep(
        2,
        "기상",
        "과거 관측과 미래 시나리오 구분",
        "기상청 ASOS 과거 관측을 조회하거나 미래 목표 날짜의 scenario 행을 입력합니다.",
        "observed와 scenario가 섞이지 않았고 목표 날짜에 필요한 행이 있는지 확인합니다.",
    ),
    GuideStep(
        3,
        "예측 실행",
        "모델과 추론 조건 선택",
        "model adapter, horizon, threshold, 픽셀 면적과 규칙 기반 위험 임계값을 정합니다.",
        "내장 기준선인지 학습 모델인지, 모델이 기상·수위를 지원하는지 확인합니다.",
    ),
    GuideStep(
        4,
        "결과",
        "프레임별 결과 해석",
        "수체 픽셀·면적·변화율·선택 수위·위험도와 예측 마스크를 함께 봅니다.",
        "픽셀 면적과 수위 보정 근거가 없으면 km²와 수위를 성능 지표로 해석하지 않습니다.",
    ),
    GuideStep(
        5,
        "API 가이드",
        "외부 시스템 연계",
        "OpenAPI, Swagger, curl/Python 예제로 모델 조회·예측 생성·결과 다운로드를 확인합니다.",
        "UI와 외부 시스템이 동일한 prediction ID와 artifact URL을 사용하는지 확인합니다.",
    ),
)


MENU_GUIDES = (
    MenuGuide(
        "데이터",
        "예측에 들어갈 수체 마스크 시계열과 날짜, 선택 수위를 구성하는 시작 화면입니다.",
        "처음 체험할 때는 빠른 테스트를, 실제 앞단 산출물을 확인할 때는 직접 업로드를 사용합니다.",
        (
            GuideItem("입력 방식", "빠른 테스트와 직접 업로드 중 하나를 선택합니다."),
            GuideItem("테스트 데이터셋", "대표 문서 복원 또는 합성 시나리오를 고릅니다. 샘플은 실측·성능 검증 자료가 아닙니다."),
            GuideItem("데이터만 불러오기", "입력과 저장소 기상 시나리오를 채운 뒤 기상 화면에서 내용을 검토합니다."),
            GuideItem("선택 샘플로 바로 예측", "권장 내장 기준선으로 API 전체 흐름을 한 번에 실행하고 결과로 이동합니다."),
            GuideItem("마스크 파일", "직접 업로드에서는 날짜가 다른 2D PNG/TIFF/NPY를 최소 2개 선택합니다."),
            GuideItem("관측 수위 입력", "게이지 실측값이 있을 때만 켭니다. 없으면 수위 결과가 null일 수 있습니다."),
            GuideItem("입력 확정", "파일·날짜·수위 순서를 고정하고 기상 단계로 이동합니다."),
        ),
        (
            "첫 관측/최근 관측 미리보기에서 흰색은 수체, 검은색은 비수체입니다.",
            "두 프레임의 차이는 입력 변화이지 모델 정확도나 미래 예측 성능이 아닙니다.",
            "문서 복원 샘플은 지리참조와 픽셀 면적을 알 수 없어 물리 면적 검증에 쓰면 안 됩니다.",
        ),
    ),
    MenuGuide(
        "기상",
        "모델이 사용할 과거 기상 관측과 호출자가 가정한 미래 기상 시나리오를 구성합니다.",
        "기상 입력을 지원하는 adapter를 실행하거나 외생변수 영향을 비교할 때 사용합니다.",
        (
            GuideItem("기상 입력 없이 모델 단계로", "persistence처럼 기상을 쓰지 않는 모델이면 빈 배열로 계속합니다."),
            GuideItem("조회 원천", "기상청 ASOS 또는 명시적인 합성 샘플을 선택합니다."),
            GuideItem("인증키 상태", "서버 KMA_API_KEY 설정 여부를 확인합니다. 키가 없으면 화면에 발급받은 키를 이번 요청용으로 붙여 넣을 수 있습니다."),
            GuideItem("관측소 새로고침/선택", "백엔드에서 ASOS 관측소 목록을 받아 대상 지점을 고릅니다."),
            GuideItem("조회 시작·종료", "ASOS가 제공할 수 있는 과거 기간을 지정합니다. 서비스 화면은 오늘 기준 D-1까지만 조회합니다."),
            GuideItem("과거 관측 조회", "KMA_API_KEY 등 유효한 공공데이터 인증키가 있어야 실제 ASOS 요청이 성공합니다."),
            GuideItem("키 없이 같은 기간 샘플 조회", "인증키가 없어도 같은 기간의 합성 자료로 UI와 예측 파이프라인을 즉시 시험합니다."),
            GuideItem("CSV 적용/양식 받기", "자체 기상자료나 미래 scenario를 표 형식으로 교체·추가합니다."),
            GuideItem("편집표", "강수·기온·습도·풍속·기압과 kind/source를 검토합니다."),
            GuideItem("기상 데이터 확정", "현재 표를 예측 요청에 포함하고 모델 선택 화면으로 이동합니다."),
        ),
        (
            "막대는 강수량, 선은 기온과 습도이며 좌우 y축 단위가 다릅니다.",
            "observed는 과거 관측, scenario는 호출자가 넣은 미래 가정입니다. scenario는 예보가 아닙니다.",
            "내장 샘플과 저장소 weather_ex.csv 변환값은 실측 ASOS가 아니며 모델 성능 근거가 아닙니다.",
        ),
    ),
    MenuGuide(
        "예측 실행",
        "등록된 model adapter와 예측 길이·이진화·위험 규칙을 선택해 백엔드 예측을 시작합니다.",
        "입력과 기상이 확정된 뒤 기준선 연동을 시험하거나 새 학습 모델 plugin을 비교할 때 사용합니다.",
        (
            GuideItem("모델 목록 새로고침", "백엔드 registry에서 현재 사용할 수 있는 adapter를 다시 읽습니다."),
            GuideItem("백엔드 등록 모델", "persistence, irregular-area-trend, weather-morphology 또는 설치한 학습 모델 adapter를 고릅니다."),
            GuideItem("model_options_json", "adapter별 세부 옵션입니다. 의미를 모르면 모델이 제공한 기본값을 유지합니다."),
            GuideItem("주의/위험 면적 변화율", "모델 출력 뒤에 적용되는 규칙 기반 risk 경계입니다. 학습된 경보 확률이 아닙니다."),
            GuideItem("1주·2주·1개월", "7·14·30일을 하루 1프레임으로 즉시 설정합니다. 직접 설정에서는 프레임 수와 간격을 따로 정합니다."),
            GuideItem("마스크 이진화 threshold", "확률 또는 연속 출력을 물/비물로 나누는 0~1 경계입니다."),
            GuideItem("픽셀 면적", "한 픽셀의 실제 면적(m²)입니다. 좌표계·해상도에서 확인한 값만 사용합니다."),
            GuideItem("예측 프레임 간격", "마지막 입력 날짜에서 각 목표 날짜까지의 간격입니다."),
            GuideItem("면적 변화율로 수위 보정", "현장에서 검증한 면적-수위 계수가 있을 때만 사용합니다."),
            GuideItem("미래 기준 수위 CSV", "같은 목표 날짜의 정답 수위를 추론 후 비교해 MAE·RMSE·MAPE·R²·Bias를 계산합니다. 모델 입력에는 쓰지 않습니다."),
            GuideItem("예측 실행", "multipart 입력과 모든 설정을 FastAPI로 보내고 prediction ID를 받습니다."),
        ),
        (
            "목표 날짜 목록과 scenario 누락 경고를 먼저 확인합니다.",
            "persistence는 최근 마스크를 반복하고, irregular-area-trend는 모든 관측일의 면적 추세와 선택 강수 보정을 쓰며, weather-morphology는 최근 마스크를 강수 규칙으로 팽창·침식합니다.",
            "세 내장 모델은 학습된 수문 모델이 아니며 정확도 주장의 근거가 될 수 없습니다.",
        ),
    ),
    MenuGuide(
        "결과",
        "예측 프레임별 수체 변화와 선택 수위, rule-based risk 및 다운로드 산출물을 확인합니다.",
        "예측 직후 결과를 검토하거나 최근 실행을 다시 열어 JSON·CSV·PNG·TIFF·NPY·ZIP을 받을 때 사용합니다.",
        (
            GuideItem("상태 새로고침", "비동기 실행 상태와 저장된 결과를 prediction ID로 다시 조회합니다."),
            GuideItem("KPI 카드", "실행 상태, 모델, 최종 면적/픽셀, 변화율, 수위, 위험도를 요약합니다."),
            GuideItem("7·14·30일 수위 전망", "관측 수위와 일별 예측, 선택 기준 수위, 강수 시나리오를 한 그래프에서 이어 봅니다."),
            GuideItem("오차·정확도", "정답 수위가 있을 때만 MAE·RMSE·MAPE·R²·Bias와 실제–예측·잔차 그래프를 표시합니다."),
            GuideItem("프레임 표", "area_pixels, area_km2, change, level, risk를 행별로 비교합니다."),
            GuideItem("산출물 선택", "예측 마스크 미리보기와 PNG/TIFF/NPY 등 개별 파일을 선택합니다."),
            GuideItem("전체 ZIP/JSON/CSV", "외부 분석이나 보고서 작성에 필요한 전체 응답과 파일을 내려받습니다."),
        ),
        (
            "청록은 관측 수위, 주황은 예측 수위, 회색 점선은 별도로 제공한 평가 기준 수위입니다.",
            "RMSE 등은 같은 날짜의 정답이 있을 때만 계산합니다. 합성 샘플 값은 화면 시연이며 모델 검증 수치가 아닙니다.",
            "위험 색은 면적 변화율 규칙 결과이며 실제 재난 경보나 발생 확률이 아닙니다.",
            "level이 null이면 오류를 감춘 것이 아니라 모델·보정이 수위를 만들지 않았다는 뜻입니다.",
            "픽셀 면적이 임시값이면 area_km2 대신 area_pixels만 해석합니다.",
        ),
    ),
    MenuGuide(
        "API 가이드",
        "UI를 거치지 않는 외부 시스템이 같은 예측·조회·다운로드 기능을 호출하도록 계약을 설명합니다.",
        "이노뎁 앞단, 관제 화면, 배치 작업 또는 다른 서비스와 WATERCAST를 연결할 때 사용합니다.",
        (
            GuideItem("엔드포인트 표", "실행 중인 /openapi.json을 우선 사용하고 연결 실패 시 fallback 요약을 보여줍니다."),
            GuideItem("Swagger UI", "브라우저에서 요청 필드와 응답 schema를 열어 직접 시험합니다."),
            GuideItem("ReDoc/OpenAPI JSON", "사람용 문서 또는 코드 생성용 원본 명세를 확인합니다."),
            GuideItem("curl/Python 예제", "모델 조회 → 예측 POST → prediction ID 조회 → artifact 다운로드 순서를 복사합니다."),
            GuideItem("전체 번들", "prediction별 manifest와 마스크 파일을 ZIP으로 회수합니다."),
        ),
        (
            "GET은 상태·모델·저장 결과 조회, POST는 기상 조회 조건 전달과 예측 생성을 뜻합니다.",
            "201 응답의 prediction ID를 보관하면 상세 JSON, 파일 목록, 개별 artifact, ZIP을 다시 가져올 수 있습니다.",
            "정확한 필드와 오류 코드는 항상 실행 중인 서버 OpenAPI를 기준으로 합니다.",
        ),
    ),
)


INPUT_CONCEPTS = (
    GuideItem("mask", "각 픽셀이 수체인지 나타내는 2D 배열입니다. 모든 시점은 같은 크기·격자·좌표계를 가져야 합니다."),
    GuideItem("date", "해당 마스크가 관측된 날짜입니다. 백엔드는 이 순서로 시계열을 정렬합니다."),
    GuideItem("pixel_area_m2", "한 픽셀의 실제 지상 면적입니다. area_km2 계산에 직접 곱해지므로 임의값은 물리 면적을 왜곡합니다."),
    GuideItem("water_level_m", "마스크 날짜와 정렬된 선택 관측 수위(m)입니다. 없으면 null을 허용합니다."),
)


MODEL_CONCEPTS = (
    GuideItem("model adapter", "서로 다른 모델을 동일한 입력·출력 API에 꽂는 얇은 변환 계층입니다."),
    GuideItem("persistence", "마지막 입력 마스크와 마지막 유효 수위를 반복하는 연동 기준선입니다."),
    GuideItem("irregular-area-trend", "모든 입력 마스크의 수체면적과 실제 관측 간격에 로그 선형 추세를 맞추고, 선택 강수를 규칙으로 보정하는 다중시점 기준선입니다."),
    GuideItem("weather-morphology", "강수와 건조일 규칙으로 마스크를 팽창·침식하는 설명 가능한 기준선입니다."),
    GuideItem("학습 모델", "실제 학습 가중치와 전처리·후처리를 가진 외부 plugin입니다. 별도 검증 전에는 성능을 주장할 수 없습니다."),
    GuideItem("horizon", "생성할 미래 프레임 수이며, 실제 날짜는 프레임 간격과 함께 결정됩니다."),
    GuideItem("threshold", "모델 출력이 이 값 이상인 픽셀을 물로 분류하는 이진화 경계입니다."),
    GuideItem("risk rule", "최근 입력 대비 면적 변화율을 주의·홍수 위험·가뭄 위험으로 나누는 후처리 규칙입니다."),
)


RESULT_CONCEPTS = (
    GuideItem("area_pixels", "예측 마스크에서 물로 분류된 픽셀 개수입니다."),
    GuideItem("area_km2", "area_pixels × pixel_area_m2 ÷ 1,000,000입니다. 픽셀 면적이 정확할 때만 물리량입니다."),
    GuideItem("change / area_change_pct", "마지막 입력 수체 면적을 기준으로 한 예측 면적 증감률입니다."),
    GuideItem("level / water_level_m", "adapter 직접 출력 또는 사용자가 제공한 면적-수위 보정 결과입니다."),
    GuideItem("reference_water_level_m", "목표 날짜와 정렬된 평가 기준 수위입니다. 추론 후 오차 계산에만 쓰고 predictor에는 전달하지 않습니다."),
    GuideItem("MAE / RMSE", "정답 수위가 있을 때 계산하는 오차 지표입니다. 둘 다 0에 가까울수록 좋고 RMSE는 큰 오차를 더 크게 반영합니다."),
    GuideItem("MAPE / R² / Bias", "비율 오차, 변동 설명력, 평균적인 과대·과소 방향을 서로 다른 관점에서 봅니다."),
    GuideItem("null", "측정값이나 산출 방법이 없음을 정직하게 나타냅니다. 0m와 다릅니다."),
    GuideItem("risk", "설정한 면적 변화율 경계로 만든 rule-based 상태이며 발생 확률이 아닙니다."),
    GuideItem("artifacts", "프레임별 PNG 미리보기, NPY 배열, TIFF/GeoTIFF와 manifest·ZIP 등 다운로드 파일입니다."),
)


API_ENDPOINTS = (
    GuideItem("GET /api/v1/health", "백엔드와 model registry 상태를 확인합니다."),
    GuideItem("GET /api/v1/models", "선택 가능한 adapter와 옵션을 조회합니다."),
    GuideItem("GET /api/v1/weather/status", "서버 인증키 설정과 요청별 키·샘플 지원 여부를 비밀값 노출 없이 확인합니다."),
    GuideItem("POST /api/v1/weather/observations", "ASOS 과거 관측 또는 명시적 샘플을 조회합니다."),
    GuideItem("POST /api/v1/predictions", "마스크·날짜·기상·모델 설정을 multipart로 전송합니다."),
    GuideItem("GET /api/v1/predictions/{id}", "저장된 전체 예측 JSON을 조회합니다."),
    GuideItem("GET /api/v1/predictions/{id}/files/{name}", "개별 artifact를 다운로드합니다."),
    GuideItem("GET /api/v1/predictions/{id}/bundle", "예측 결과 전체 ZIP을 다운로드합니다."),
)


PRESENTATION_SCRIPT = (
    GuideItem(
        "1 · 문제와 범위",
        "WATERCAST는 앞단이 만든 수체 마스크 시계열을 받아 미래 마스크와 파생 결과를 제공하는 교체형 예측 콘솔입니다. 원본 영상 탐지나 검증된 재난 경보 자체를 주장하지 않습니다.",
    ),
    GuideItem(
        "2 · 입력과 기상",
        "데이터 화면에서 마스크와 날짜를 맞추고, 기상 화면에서 과거 observed와 미래 scenario를 분리합니다. ASOS는 D-1까지의 관측이며 scenario는 사용자가 제공한 가정입니다.",
    ),
    GuideItem(
        "3 · 모델 교체 구조",
        "현재 persistence, irregular-area-trend, weather-morphology는 API 연결을 확인하는 기준선입니다. 향후 검증된 학습 모델을 adapter로 등록해도 UI와 외부 API 계약은 유지됩니다.",
    ),
    GuideItem(
        "4 · 결과 해석",
        "결과에서는 픽셀 수, 물리 면적, 변화율, 선택 수위, 위험 규칙과 마스크 파일을 함께 봅니다. 픽셀 면적이나 보정 근거가 없으면 면적·수위 정확도를 해석하지 않습니다.",
    ),
    GuideItem(
        "5 · 외부 연계",
        "외부 제품은 prediction ID로 전체 JSON과 artifact, ZIP을 가져갈 수 있습니다. 따라서 모델 구현이 바뀌어도 소비 시스템은 같은 API를 사용할 수 있습니다.",
    ),
)


PAGE_HELP: Mapping[str, PageHelp] = MappingProxyType(
    {
        "data": PageHelp(
            "data",
            "데이터",
            (
                "빠른 테스트 또는 직접 업로드를 선택합니다.",
                "마스크와 날짜, 선택 수위가 1:1로 정렬됐는지 확인합니다.",
                "데이터만 검토하거나 바로 기준선 예측을 실행합니다.",
            ),
            "앞단 수체 마스크와 날짜를 공통 입력 계약으로 만드는 단계입니다.",
        ),
        "weather": PageHelp(
            "weather",
            "기상",
            (
                "기상 미사용 또는 ASOS/샘플/CSV 입력 방식을 선택합니다.",
                "과거 observed와 미래 scenario의 날짜와 출처를 검토합니다.",
                "막대·선 차트를 확인한 뒤 기상 데이터를 확정합니다.",
            ),
            "ASOS 과거 관측과 사용자가 가정한 미래 scenario를 명확히 분리합니다.",
        ),
        "prediction": PageHelp(
            "prediction",
            "예측 실행",
            (
                "adapter가 기준선인지 학습 모델인지 확인합니다.",
                "7·14·30일 horizon, threshold, 픽셀 면적, risk 경계를 설정합니다.",
                "목표 날짜와 scenario 누락 경고를 검토하고 예측을 실행합니다.",
            ),
            "검증된 모델을 adapter로 교체해도 UI와 API 계약은 유지됩니다.",
        ),
        "result": PageHelp(
            "result",
            "결과",
            (
                "KPI에서 실행 모델과 최종 픽셀·면적·수위·위험을 확인합니다.",
                "관측–예측 일별 차트와 D+7·14·30 카드에서 변화 방향을 확인합니다.",
                "정답 수위가 있으면 RMSE 등 동적 오차와 잔차 그래프를 검토합니다.",
                "필요한 artifact 또는 JSON·CSV·ZIP을 내려받습니다.",
            ),
            "근거가 있는 값만 해석하고 null과 기준선 한계를 그대로 공개합니다.",
        ),
        "api": PageHelp(
            "api",
            "API 가이드",
            (
                "OpenAPI/Swagger에서 현재 서버 계약을 확인합니다.",
                "모델 조회, 예측 POST, prediction ID 조회 순서를 시험합니다.",
                "상세 JSON, 개별 artifact와 전체 ZIP을 외부 시스템에서 회수합니다.",
            ),
            "외부 제품은 prediction ID를 기준으로 모든 결과를 동일한 REST API에서 가져갑니다.",
        ),
    }
)


FAQ_ITEMS = (
    FAQItem("가장 빨리 확인하려면?", "데이터 메뉴에서 대표 샘플을 고른 뒤 ‘선택 샘플로 바로 예측’을 누릅니다. 백엔드가 먼저 실행 중이어야 합니다."),
    FAQItem("광주 홍수·한강 가뭄 자료가 원래 있었나요?", "시나리오 이름과 합성 변화 개념은 예전 발표 데모에 있었습니다. 그러나 사용자가 제공한 광주·한강 실측 파일은 없었고, 현재 샘플 픽셀은 ui_next/samples.py가 192×192로 새로 그립니다. 날짜·미래 강수·수위도 연동 시험용 예시입니다."),
    FAQItem("샘플 결과가 모델 성능인가요?", "아닙니다. 문서 복원·합성 샘플과 내장 기준선은 UI/API 연동 확인용이며 실측 검증이나 학습 모델 평가가 아닙니다."),
    FAQItem("RMSE가 왜 산출 불가인가요?", "RMSE에는 예측 날짜와 같은 날짜의 정답 수위가 필요합니다. 미래 기준 수위 CSV나 홀드아웃 실측을 제공하면 백엔드가 동적으로 계산하며, 예측값만으로 수치를 만들지 않습니다."),
    FAQItem("프레임은 영상의 FPS인가요?", "아닙니다. 여기서는 특정 관측 날짜의 수체 상태를 담은 2D 마스크 한 장입니다. 4프레임은 4일이 아니며 각 날짜 사이 간격을 함께 봐야 합니다."),
    FAQItem("원본 SAR TIFF를 바로 올리면 되나요?", "안 됩니다. 현재 예측 입력은 센서 보정·동일 격자 정렬·수체 감지·품질검수를 끝낸 2D 수체 마스크입니다. TIFF라는 확장자만 같아도 내용 단계가 다르면 입력이 아닙니다."),
    FAQItem("제공 U-Net·ConvLSTM이 지금 실행되나요?", "아닙니다. 연구 코드·노트북·문서는 있지만 현재 작업공간에 필요한 가중치와 데이터가 없고 백엔드 adapter에도 등록되지 않았습니다. 현재 실행 모델은 persistence, irregular-area-trend, weather-morphology 기준선 세 개입니다."),
    FAQItem("ASOS로 미래 날씨를 받을 수 있나요?", "아닙니다. 이 화면의 ASOS 연동은 오늘 기준 D-1까지의 과거 관측입니다. 미래값은 scenario 행이나 CSV로 제공해야 합니다."),
    FAQItem("ASOS 조회가 실패하는 이유는?", "백엔드 환경에 유효한 KMA_API_KEY가 없거나 공공데이터 서비스·네트워크에 연결되지 않았을 수 있습니다."),
    FAQItem("수위가 null인 이유는?", "선택 adapter가 수위를 출력하지 않고 검증된 면적-수위 보정도 제공하지 않았기 때문입니다. null은 정상적인 미산출 표시입니다."),
    FAQItem("pixel_area_m2를 모르면?", "임의로 물리 면적을 해석하지 말고 area_pixels만 비교합니다. 원본 래스터의 좌표계와 해상도에서 값을 확인해야 합니다."),
    FAQItem("threshold를 높이면?", "일반적으로 물로 분류되는 픽셀이 줄지만 영향은 모델 출력 분포에 따라 다릅니다. 검증 데이터로 선택해야 합니다."),
    FAQItem("위험도가 실제 경보인가요?", "아닙니다. 현재 risk는 면적 변화율에 적용한 단순 규칙입니다. 운영 경보에는 현장 기준과 별도 검증이 필요합니다."),
    FAQItem("새 학습 모델은 어떻게 연결하나요?", "백엔드 PredictionAdapter 계약을 구현해 plugin으로 등록합니다. 모델 목록 API에 나타나면 UI 코드 변경 없이 선택할 수 있습니다."),
    FAQItem("외부 시스템이 모든 결과를 받을 수 있나요?", "예. prediction 상세 JSON, 파일 목록, 개별 artifact와 전체 ZIP endpoint를 사용할 수 있습니다."),
)


GLOSSARY = (
    GlossaryTerm("수체 마스크(mask)", "물/비물을 픽셀 값으로 표현한 2D 배열."),
    GlossaryTerm("프레임(frame)", "특정 관측 날짜의 수체 마스크 한 장. 동영상의 FPS와 다른 시계열 단위."),
    GlossaryTerm("패치(patch)", "큰 프레임을 모델 입력 크기로 자른 작은 공간 조각."),
    GlossaryTerm("시퀀스(sequence)", "같은 공간 위치의 프레임을 날짜순으로 쌓은 묶음."),
    GlossaryTerm("시계열", "날짜 순서로 정렬된 여러 관측 프레임."),
    GlossaryTerm("observed", "이미 지나간 날짜의 관측값. ASOS 자료는 이 범주."),
    GlossaryTerm("scenario", "미래 추론을 위해 호출자가 넣은 가정값. 예보와 동일하지 않음."),
    GlossaryTerm("ASOS", "기상청 종관기상관측 자료. 이 서비스에서는 D-1까지의 과거 관측으로 사용."),
    GlossaryTerm("adapter", "모델별 입출력을 공통 WATERCAST API 계약으로 맞추는 계층."),
    GlossaryTerm("persistence", "마지막 관측을 미래에도 반복하는 기준선."),
    GlossaryTerm("irregular-area-trend", "불규칙한 실제 날짜 간격과 모든 관측 면적을 이용해 미래 면적을 외삽하는 비학습 기준선."),
    GlossaryTerm("weather-morphology", "기상 규칙으로 마스크 모양을 바꾸는 비학습 기준선."),
    GlossaryTerm("학습 모델", "데이터로 가중치를 학습한 외부 모델. 등록과 검증이 별도로 필요."),
    GlossaryTerm("horizon", "한 번의 요청으로 생성할 미래 프레임 개수."),
    GlossaryTerm("D+7 / D+14 / D+30", "마지막 관측 다음 날을 D+1로 보았을 때 각각 7·14·30번째 일별 예측."),
    GlossaryTerm("threshold", "연속 모델 출력을 이진 마스크로 나누는 경계값."),
    GlossaryTerm("pixel_area_m2", "한 픽셀이 나타내는 실제 지상 면적(m²)."),
    GlossaryTerm("water_level_m", "미터 단위 수위. 입력 관측 또는 명시된 산출 방법이 필요."),
    GlossaryTerm("reference water level", "오차 평가용 정답 수위. 모델 입력과 분리되어야 함."),
    GlossaryTerm("RMSE", "예측–정답 오차를 제곱 평균한 뒤 제곱근을 취한 지표. 큰 오차에 민감."),
    GlossaryTerm("MAE", "예측–정답 절대 오차의 평균."),
    GlossaryTerm("area_pixels", "수체로 판정된 픽셀의 개수."),
    GlossaryTerm("area_km2", "정확한 픽셀 면적을 적용한 수체 면적(km²)."),
    GlossaryTerm("area_change_pct", "마지막 입력 면적 대비 예측 면적 변화율."),
    GlossaryTerm("null", "값이 0인 것이 아니라 산출·측정되지 않았음을 뜻하는 JSON 값."),
    GlossaryTerm("risk", "면적 변화율 경계로 정한 규칙 기반 상태."),
    GlossaryTerm("artifact", "예측 과정에서 저장된 PNG, NPY, TIFF, manifest 등의 파일."),
    GlossaryTerm("prediction ID", "저장된 한 번의 예측과 모든 결과 파일을 식별하는 값."),
    GlossaryTerm("OpenAPI", "API 경로·요청·응답 schema를 기계가 읽을 수 있게 기술한 명세."),
)


GUIDE_LIMITATIONS = (
    "내장 persistence, irregular-area-trend, weather-morphology는 학습 모델이 아니라 연동 기준선입니다.",
    "내장 샘플은 실측·운영·성능 검증 데이터가 아닙니다.",
    "ASOS 연동은 D-1까지의 과거 관측이며 미래 예보를 제공하지 않습니다.",
    "픽셀 면적, 지리참조, 수위 보정 근거가 없는 자료는 면적·수위 정확도 검증에 사용할 수 없습니다.",
    "risk는 면적 변화율 후처리 규칙이며 재난 발생 확률이나 공식 경보가 아닙니다.",
)


GUIDE_CSS = """
<style>
.wc-guide-hero { margin: 4px 0 16px; padding: 18px 20px; }
.wc-guide-hero h2 { margin: 3px 0 7px; color: var(--wc-text); }
.wc-guide-hero p { margin: 0; color: var(--wc-muted); }
.wc-guide-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin:10px 0 18px; }
.wc-guide-card { min-height:100%; }
.wc-guide-card h4 { margin:6px 0; color:var(--wc-text); font-size:.9rem; }
.wc-guide-card p { margin:0; color:var(--wc-muted); font-size:.79rem; line-height:1.58; }
.wc-guide-card__kicker { color:var(--wc-sky); font-size:.65rem; font-weight:800; letter-spacing:.12em; }
.wc-guide-note { margin:10px 0; padding:12px 14px; border-left:3px solid var(--wc-yellow); background:rgba(120,53,15,.16); color:#fde68a; }
.wc-guide-flow { margin:10px 0 18px; padding:14px; overflow-x:auto; }
.wc-guide-flow svg { min-width:920px; width:100%; height:auto; }
.wc-guide-purpose { margin:5px 0 14px; }
.wc-guide-purpose b { color:var(--wc-sky); }
.wc-guide-script { border-color:rgba(56,189,248,.34)!important; background:linear-gradient(120deg,rgba(8,47,73,.36),rgba(15,23,42,.82))!important; }
.wc-guide-term { display:grid; grid-template-columns:minmax(130px,.35fr) 1fr; gap:12px; padding:9px 0; border-bottom:1px solid var(--wc-border); }
.wc-guide-term:last-child { border-bottom:0; }
.wc-guide-term b { color:#bae6fd; }
.wc-guide-term span { color:var(--wc-muted); }
@media(max-width:720px){.wc-guide-term{grid-template-columns:1fr;gap:3px}.wc-guide-flow svg{min-width:760px}}
</style>
"""


def _safe(value: object) -> str:
    return escape(str(value), quote=True)


def guide_card_html(title: object, body: object, *, kicker: object = "GUIDE", extra_class: str = "") -> str:
    """Build one escaped card using the WATERCAST theme classes."""

    safe_extra = " ".join(
        token for token in str(extra_class).split() if token.replace("-", "").isalnum()
    )
    class_name = "wc-panel wc-guide-card" + (f" {safe_extra}" if safe_extra else "")
    return (
        f'<div class="{class_name}">'
        f'<div class="wc-guide-card__kicker">{_safe(kicker)}</div>'
        f"<h4>{_safe(title)}</h4><p>{_safe(body)}</p></div>"
    )


def item_list_html(items: Iterable[GuideItem], *, ordered: bool = False) -> str:
    """Render escaped definitions without allowing content to inject markup."""

    tag = "ol" if ordered else "ul"
    rows = "".join(
        f"<li><b>{_safe(item.label)}</b> — {_safe(item.explanation)}</li>"
        for item in items
    )
    return f'<{tag} class="wc-guide-list">{rows}</{tag}>'


def flow_diagram_html() -> str:
    """Return an original, self-contained five-step SVG workflow diagram."""

    colors = ("#38bdf8", "#22d3ee", "#a855f7", "#f97316", "#22c55e")
    boxes: list[str] = []
    arrows: list[str] = []
    for index, step in enumerate(WORKFLOW_STEPS):
        x = 20 + index * 190
        boxes.append(
            f'<g data-step="{step.number}">'
            f'<rect x="{x}" y="32" width="158" height="94" rx="10" fill="#0f172a" '
            f'stroke="{colors[index]}" stroke-width="1.6"/>'
            f'<circle cx="{x + 22}" cy="54" r="12" fill="{colors[index]}"/>'
            f'<text x="{x + 22}" y="58" text-anchor="middle" fill="#020617" '
            f'font-size="11" font-weight="800">{step.number}</text>'
            f'<text x="{x + 43}" y="59" fill="#f1f5f9" font-size="13" '
            f'font-weight="700">{_safe(step.menu)}</text>'
            f'<text x="{x + 14}" y="86" fill="#94a3b8" font-size="11">'
            f'{_safe(step.title[:13])}</text>'
            f'<text x="{x + 14}" y="105" fill="#64748b" font-size="10">'
            f'{_safe(step.title[13:26])}</text></g>'
        )
        if index < len(WORKFLOW_STEPS) - 1:
            arrows.append(
                f'<path d="M{x + 160} 79 H{x + 184}" stroke="#475569" stroke-width="2" '
                'marker-end="url(#wc-guide-arrow)"/>'
            )
    return (
        '<div class="wc-panel wc-guide-flow">'
        '<svg viewBox="0 0 960 155" role="img" '
        'aria-label="WATERCAST 데이터부터 API 연계까지의 5단계 흐름">'
        '<defs><marker id="wc-guide-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10 Z" fill="#475569"/></marker></defs>'
        + "".join(arrows)
        + "".join(boxes)
        + '</svg></div>'
    )


def _grid_html(items: Iterable[GuideItem], *, kicker: str, extra_class: str = "") -> str:
    cards = "".join(
        guide_card_html(item.label, item.explanation, kicker=kicker, extra_class=extra_class)
        for item in items
    )
    return f'<div class="wc-guide-grid">{cards}</div>'


def _terms_html(items: Iterable[GuideItem | GlossaryTerm]) -> str:
    rows = []
    for item in items:
        label = item.label if isinstance(item, GuideItem) else item.term
        body = item.explanation if isinstance(item, GuideItem) else item.definition
        rows.append(
            f'<div class="wc-guide-term"><b>{_safe(label)}</b><span>{_safe(body)}</span></div>'
        )
    return '<div class="wc-panel">' + "".join(rows) + "</div>"


def _menu_tab(menu: MenuGuide, st: object) -> None:
    # ``st`` is injected by the public renderer to keep pure-content imports light.
    st.markdown(  # type: ignore[attr-defined]
        '<div class="wc-panel wc-guide-purpose">'
        f'<b>목적</b> {_safe(menu.purpose)}<br/>'
        f'<b>언제</b> {_safe(menu.when_to_use)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### 버튼과 선택값")  # type: ignore[attr-defined]
    st.markdown(item_list_html(menu.controls, ordered=True), unsafe_allow_html=True)  # type: ignore[attr-defined]
    st.markdown("#### 차트·표 읽는 법")  # type: ignore[attr-defined]
    st.markdown(  # type: ignore[attr-defined]
        "\n".join(f"- {sentence}" for sentence in menu.reading)
    )


def render_page_help(page_id: str, expanded: bool = False) -> None:
    """Render a compact, page-local guide for one top-level workflow screen.

    ``page_id`` must be one of ``data``, ``weather``, ``prediction``, ``result``
    or ``api``.  Unknown values fail loudly so app navigation cannot silently
    show instructions for the wrong screen.
    """

    try:
        page = PAGE_HELP[page_id]
    except KeyError as exc:
        available = ", ".join(PAGE_HELP)
        raise KeyError(f"unknown guide page_id {page_id!r}; available: {available}") from exc
    menu = next(item for item in MENU_GUIDES if item.menu == page.menu)

    import streamlit as st

    with st.expander(f"{page.menu} 화면 사용법", expanded=expanded):
        st.markdown(
            '<div class="wc-panel wc-guide-purpose">'
            f'<b>목적</b> {_safe(menu.purpose)}<br/>'
            f'<b>언제</b> {_safe(menu.when_to_use)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("##### 사용 순서")
        st.markdown("\n".join(f"{index}. {text}" for index, text in enumerate(page.use_order, 1)))
        st.markdown("##### 버튼·선택값")
        st.markdown(item_list_html(menu.controls, ordered=False), unsafe_allow_html=True)
        st.markdown("##### 그래프·표 읽는 법")
        st.markdown("\n".join(f"- {sentence}" for sentence in menu.reading))
        st.markdown(
            guide_card_html(
                "발표 한 줄",
                page.presentation_line,
                kicker="PRESENTATION",
                extra_class="wc-guide-script",
            ),
            unsafe_allow_html=True,
        )


def render_guide_page() -> None:
    """Render the complete WATERCAST beginner/presentation guide in Streamlit."""

    import streamlit as st

    st.markdown(GUIDE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<section class="wc-panel wc-guide-hero">'
        '<div class="wc-kicker">WATERCAST USER GUIDE</div>'
        '<h2>처음 실행부터 외부 API 연계까지</h2>'
        '<p>화면 조작법, 값의 의미, 결과 해석과 발표 문구를 한곳에 정리했습니다. '
        '내장 샘플과 기준선의 한계도 함께 확인하세요.</p></section>',
        unsafe_allow_html=True,
    )
    st.info(
        "상단의 ‘이해 가이드’는 설명 전용 메뉴입니다. 실제 작업 순서는 데이터 → 기상 → "
        "예측 실행 → 결과이며, 외부 연계는 API 가이드에서 확인합니다."
    )

    st.markdown("## 전체 흐름 · 5단계")
    st.markdown(flow_diagram_html(), unsafe_allow_html=True)
    st.markdown(
        '<div class="wc-guide-grid">'
        + "".join(
            guide_card_html(
                f"{step.number}. {step.menu} · {step.title}",
                f"할 일: {step.action} 확인: {step.check}",
                kicker=f"STEP {step.number}",
            )
            for step in WORKFLOW_STEPS
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## 상단 메뉴별 사용법")
    tabs = st.tabs([menu.menu for menu in MENU_GUIDES])
    for tab, menu in zip(tabs, MENU_GUIDES, strict=True):
        with tab:
            _menu_tab(menu, st)

    st.markdown("## 입력값을 먼저 이해하기")
    st.markdown(_terms_html(INPUT_CONCEPTS), unsafe_allow_html=True)
    st.info(
        "mask와 date는 반드시 1:1로 정렬해야 합니다. pixel_area_m2를 모르는 자료는 "
        "area_pixels로만 비교하고, water_level_m이 없으면 null을 허용하세요."
    )

    weather_column, model_column = st.columns(2)
    with weather_column:
        st.markdown("## 기상값의 진실")
        st.markdown(
            guide_card_html(
                "observed vs scenario",
                "observed는 과거 관측입니다. scenario는 호출자가 제공한 미래 가정이며 예보가 아닙니다.",
                kicker="WEATHER",
            ),
            unsafe_allow_html=True,
        )
        st.warning(
            "ASOS는 화면 기준 D-1까지의 과거 관측입니다. 실제 조회에는 유효한 "
            "KMA_API_KEY가 필요합니다. 내장 샘플은 실측 자료가 아닙니다."
        )
    with model_column:
        st.markdown("## 모델과 예측 조건")
        st.markdown(_terms_html(MODEL_CONCEPTS), unsafe_allow_html=True)

    st.markdown("## 결과 필드 읽는 법")
    st.markdown(_terms_html(RESULT_CONCEPTS), unsafe_allow_html=True)

    st.markdown("## 외부 연계 · API")
    st.markdown(
        "외부 시스템은 UI를 열지 않고도 아래 REST API로 모델을 선택하고 prediction ID를 "
        "받아 전체 결과와 산출물을 회수할 수 있습니다. 상세 schema는 실행 중인 OpenAPI가 기준입니다."
    )
    st.markdown(_terms_html(API_ENDPOINTS), unsafe_allow_html=True)

    st.markdown("## 발표할 때 이렇게 설명")
    st.caption("아래 다섯 카드를 순서대로 읽으면 약 1~2분 분량의 데모 설명이 됩니다.")
    st.markdown(
        _grid_html(PRESENTATION_SCRIPT, kicker="TALK TRACK", extra_class="wc-guide-script"),
        unsafe_allow_html=True,
    )

    st.markdown("## 반드시 함께 말할 한계")
    st.markdown(
        '<div class="wc-guide-note">'
        + "<br/>".join(f"• {_safe(item)}" for item in GUIDE_LIMITATIONS)
        + "</div>",
        unsafe_allow_html=True,
    )

    faq_column, glossary_column = st.columns([1, 1.35])
    with faq_column:
        st.markdown("## 초보 FAQ")
        for item in FAQ_ITEMS:
            with st.expander(item.question):
                st.write(item.answer)
    with glossary_column:
        st.markdown("## 용어사전")
        st.markdown(_terms_html(GLOSSARY), unsafe_allow_html=True)


__all__ = [
    "API_ENDPOINTS",
    "FAQ_ITEMS",
    "GLOSSARY",
    "GUIDE_LIMITATIONS",
    "INPUT_CONCEPTS",
    "MENU_GUIDES",
    "MODEL_CONCEPTS",
    "PAGE_HELP",
    "PRESENTATION_SCRIPT",
    "RESULT_CONCEPTS",
    "WORKFLOW_STEPS",
    "GlossaryTerm",
    "GuideItem",
    "GuideStep",
    "MenuGuide",
    "PageHelp",
    "flow_diagram_html",
    "guide_card_html",
    "item_list_html",
    "render_guide_page",
    "render_page_help",
]
