"""Immutable inventory of the delivered micro-satellite NAS datasets.

The catalog contains logical paths only.  It deliberately does not contain a
host name, account, credential, mount point, or other connection detail.  File
counts and byte sizes are a point-in-time inventory of the supplied tree; they
describe the source files, not files copied into the application.

``direct_prediction`` means that a group already contains a dated water-mask
sequence with enough frames for an API demonstration.  It does not mean that
the data is sufficient to train or scientifically validate a forecasting
model.  Raw imagery, UDM2 quality masks, DEMs, previews, and archives are never
treated as direct forecasting inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

Readiness = Literal[
    "demo_ready",
    "preprocess_required",
    "single_date_reference",
    "support_only",
]

TOTAL_FILE_COUNT = 300
TOTAL_BYTES = 57_270_553_908

MATERIALIZED_SAMPLE_ID = "busan-nas-water-labels"
ICEYE_MATERIALIZED_SAMPLE_ID = "iceye-nas-water-labels"


@dataclass(frozen=True, slots=True)
class NasDatasetGroup:
    """One source group with explicit semantics and forecast readiness."""

    group_id: str
    display_name: str
    relative_paths: tuple[str, ...]
    sensor_name: str
    sensor_modality: str
    sensor_confidence: str
    roles: tuple[str, ...]
    observation_dates: tuple[str, ...]
    interval_days: tuple[int, ...]
    acquisition_count: int
    matched_pair_count: int
    file_count: int
    total_bytes: int
    readiness: Readiness
    direct_prediction: bool
    direct_prediction_reason_ko: str
    cautions_ko: tuple[str, ...]
    related_group_ids: tuple[str, ...] = ()
    materialized_sample_id: str | None = None

    @property
    def unique_date_count(self) -> int:
        """Number of unique temporal observations, excluding same-day tiles."""

        return len(self.observation_dates)

    @property
    def materialized(self) -> bool:
        """Whether a backend-ready local sample is attached to this record."""

        return self.materialized_sample_id is not None


@dataclass(frozen=True, slots=True)
class NasCatalogSummary:
    """Immutable aggregate numbers for the source inventory."""

    file_count: int
    total_bytes: int
    group_count: int
    direct_prediction_group_count: int
    materialized_sample_count: int


NAS_DATASET_GROUPS = (
    NasDatasetGroup(
        group_id="busan-water-labels",
        display_name="부산 · 광학 수체 라벨 4시점",
        relative_paths=("신규데이터_위성영상_라벨링/0_Busan",),
        sensor_name="PlanetScope 계열 추정",
        sensor_modality="optical",
        sensor_confidence="inferred_high",
        roles=(
            "segmentation_input",
            "water_mask_label",
            "water_mask_timeseries",
        ),
        observation_dates=(
            "2020-02-18",
            "2020-03-12",
            "2020-03-25",
            "2020-04-14",
        ),
        interval_days=(23, 13, 20),
        acquisition_count=4,
        matched_pair_count=4,
        file_count=8,
        total_bytes=8_638_858_810,
        readiness="demo_ready",
        direct_prediction=True,
        direct_prediction_reason_ko=(
            "날짜가 일치하는 수체 라벨 4장이 있어 마스크 시계열 API 시연에 "
            "사용할 수 있습니다."
        ),
        cautions_ko=(
            "네 시점의 간격이 23일, 13일, 20일로 불규칙합니다.",
            "모델 학습이나 일별 7·14·30일 성능 검증에는 시간 표본이 부족합니다.",
            "날짜별 입력–라벨은 EPSG:32652·3 m 격자에서 일치하지만 원본 footprint가 달라 네 시점 공통 교집합으로 정렬했습니다.",
            "원본 라벨 0/1과 공통격자 면적은 확인했지만 512 시연본은 최근접 리샘플 파생 자산입니다.",
            "같은 날짜의 게이지 실측 수위는 인수 패키지 deploy_train_wamis_v3_finalwb.csv 30표본으로 표시할 수 있지만, 융합 LSTM 학습표본과 동일한 in-sample 값이라 일반화 수위 MAE·RMSE 검증에는 쓸 수 없습니다.",
        ),
        related_group_ids=("planetscope-raw",),
        materialized_sample_id=MATERIALIZED_SAMPLE_ID,
    ),
    NasDatasetGroup(
        group_id="iceye-water-labels",
        display_name="ICEYE · SAR 수체 라벨 4시점",
        relative_paths=("신규데이터_위성영상_라벨링/ICEYE_WB",),
        sensor_name="ICEYE",
        sensor_modality="sar",
        sensor_confidence="explicit",
        roles=(
            "segmentation_input",
            "water_mask_label",
            "water_mask_timeseries",
        ),
        observation_dates=(
            "2020-03-02",
            "2020-03-30",
            "2020-04-15",
            "2020-04-16",
        ),
        interval_days=(28, 16, 1),
        acquisition_count=4,
        matched_pair_count=4,
        file_count=8,
        total_bytes=7_236_410_760,
        readiness="demo_ready",
        direct_prediction=True,
        direct_prediction_reason_ko=(
            "날짜가 일치하는 SAR 수체 라벨 4장이 있어 품질검수와 로컬 변환 후 "
            "마스크 시계열 시연에 사용할 수 있습니다."
        ),
        cautions_ko=(
            "관측 간격이 28일, 16일, 1일로 매우 불규칙합니다.",
            "날짜별 input–label pair는 EPSG:32652·3 m로 정확히 일치하지만 날짜 간 origin·footprint가 달라 공통격자 warp가 필요합니다.",
            "라벨 NoData 태그는 15이나 실제 전체 픽셀 값은 0/1뿐이며, input=0 바깥영역을 유효 분석에서 제외해야 합니다.",
            "입사각과 look side가 달라 원시 SAR 밝기 차이를 곧바로 수체 변화로 해석하면 안 됩니다.",
            "네 시점만으로 학습 모델의 일반화 성능을 주장할 수 없습니다.",
            "NAS 전달분에는 수위 정답 계열이 없으며, 인수 패키지의 같은 날짜 게이지 실측 수위 30표본은 융합 LSTM 학습표본과 동일한 in-sample 값입니다.",
        ),
        related_group_ids=("iceye-raw",),
        materialized_sample_id=ICEYE_MATERIALIZED_SAMPLE_ID,
    ),
    NasDatasetGroup(
        group_id="iceye-raw",
        display_name="ICEYE · SAR 원천영상",
        relative_paths=("신규데이터_위성영상_원본/ICEYE",),
        sensor_name="ICEYE",
        sensor_modality="sar",
        sensor_confidence="explicit",
        roles=("sar_grd", "sar_slc", "quicklook", "metadata"),
        observation_dates=(
            "2020-03-02",
            "2020-03-30",
            "2020-04-15",
            "2020-04-16",
            "2020-04-17",
        ),
        interval_days=(28, 16, 1, 1),
        acquisition_count=5,
        matched_pair_count=0,
        file_count=30,
        total_bytes=23_527_774_070,
        readiness="preprocess_required",
        direct_prediction=False,
        direct_prediction_reason_ko=(
            "GRD·SLC는 수체 마스크가 아니므로 SAR 전처리와 수체 탐지를 먼저 "
            "수행해야 합니다."
        ),
        cautions_ko=(
            "한 날짜의 GRD, SLC, quicklook을 서로 다른 시계열 프레임으로 세면 안 됩니다.",
            "quicklook PNG는 화면 확인용이며 과학적 분석 입력이 아닙니다.",
            "2020-04-17 원천영상에는 현재 대응 수체 라벨이 없습니다.",
        ),
        related_group_ids=("iceye-water-labels",),
    ),
    NasDatasetGroup(
        group_id="planetscope-raw",
        display_name="PlanetScope · 광학 SR 및 UDM2",
        relative_paths=("신규데이터_위성영상_원본/Planetscope",),
        sensor_name="PlanetScope",
        sensor_modality="optical",
        sensor_confidence="explicit",
        roles=(
            "surface_reflectance_imagery",
            "quality_mask_udm2",
            "scene_metadata",
        ),
        observation_dates=(
            "2020-02-18",
            "2020-03-12",
            "2020-03-25",
            "2020-04-14",
        ),
        interval_days=(23, 13, 20),
        acquisition_count=8,
        matched_pair_count=0,
        file_count=52,
        total_bytes=2_257_219_305,
        readiness="preprocess_required",
        direct_prediction=False,
        direct_prediction_reason_ko=(
            "Surface Reflectance 영상과 UDM2 품질 마스크는 물/비물 수체 마스크가 "
            "아니므로 모자이크·품질처리·수체 탐지가 필요합니다."
        ),
        cautions_ko=(
            "UDM2는 구름과 품질 정보를 나타내며 수체 정답 라벨이 아닙니다.",
            "날짜마다 두 scene/tile이 있어 공간관계를 확인하고 한 격자로 만들어야 합니다.",
            "광학영상은 구름·그림자·대기 영향을 품질검수해야 합니다.",
        ),
        related_group_ids=("busan-water-labels",),
    ),
    NasDatasetGroup(
        group_id="university-single-date",
        display_name="시립대 · 수체면적·수위 단일시점 예시",
        relative_paths=("샘플데이터_시립대",),
        sensor_name="Sentinel-1A",
        sensor_modality="sar",
        sensor_confidence="explicit_from_filename",
        roles=(
            "sar_source_imagery",
            "dem",
            "derived_water_body",
            "derived_water_level",
            "geojson",
            "presentation",
        ),
        observation_dates=("2021-04-11",),
        interval_days=(),
        acquisition_count=1,
        matched_pair_count=0,
        file_count=11,
        total_bytes=4_646_743_557,
        readiness="single_date_reference",
        direct_prediction=False,
        direct_prediction_reason_ko=(
            "수체·DEM·수위 산출 예시가 한 날짜뿐이어서 과거 시퀀스를 구성할 수 없습니다."
        ),
        cautions_ko=(
            "DEM은 고도 보조자료이지 수체 예측 프레임이 아닙니다.",
            "안동·대청 결과의 AOI와 원천영상 연결을 메타데이터로 확인해야 합니다.",
            "일부 WB JSON에는 마지막 항목 뒤 쉼표가 있어 표준 JSON으로 사용하기 전 정제가 필요합니다.",
            "다른 지역의 부산·ICEYE 라벨에 이 수위값을 결합하면 안 됩니다.",
        ),
    ),
    NasDatasetGroup(
        group_id="giheung-optical",
        display_name="기흥호수 · PAN/MUL 광학 원천",
        relative_paths=("샘플데이터_1세부제공/050258218010_기흥호수",),
        sensor_name="상용 광학 센서 · 메타데이터 확인 필요",
        sensor_modality="optical",
        sensor_confidence="unverified",
        roles=(
            "panchromatic_imagery",
            "multispectral_imagery",
            "browse_preview",
            "footprint",
            "metadata",
        ),
        observation_dates=("2022-11-11", "2023-01-25", "2023-07-30"),
        interval_days=(75, 186),
        acquisition_count=3,
        matched_pair_count=0,
        file_count=95,
        total_bytes=1_014_513_287,
        readiness="preprocess_required",
        direct_prediction=False,
        direct_prediction_reason_ko=(
            "PAN/MUL 원천영상만 있고 날짜별 수체 마스크 라벨이 없으므로 영상 보정과 "
            "수체 탐지가 선행되어야 합니다."
        ),
        cautions_ko=(
            "PAN과 MUL은 같은 관측의 서로 다른 밴드 제품이며 별도 시간 프레임이 아닙니다.",
            "세 시점만으로 시계열 모델을 학습하거나 검증하기에는 부족합니다.",
            "센서·제품 단계·좌표계는 IMD/XML 등 메타데이터에서 확정해야 합니다.",
        ),
    ),
    NasDatasetGroup(
        group_id="hoedong-optical",
        display_name="회동저수지 · PAN/MUL 광학 원천",
        relative_paths=("샘플데이터_1세부제공/050258288060_회동저수지",),
        sensor_name="상용 광학 센서 · 메타데이터 확인 필요",
        sensor_modality="optical",
        sensor_confidence="unverified",
        roles=(
            "panchromatic_imagery",
            "multispectral_imagery",
            "browse_preview",
            "footprint",
            "metadata",
        ),
        observation_dates=("2023-11-08", "2024-08-23"),
        interval_days=(289,),
        acquisition_count=3,
        matched_pair_count=0,
        file_count=92,
        total_bytes=767_021_995,
        readiness="preprocess_required",
        direct_prediction=False,
        direct_prediction_reason_ko=(
            "고유 관측일이 두 날짜뿐이고 수체 라벨이 없어 직접 시계열 예측에 "
            "사용할 수 없습니다."
        ),
        cautions_ko=(
            "2023-11-08의 두 acquisition은 촬영시각·공간 범위를 확인해야 하며 시간 프레임 두 개로 자동 집계하면 안 됩니다.",
            "PAN/MUL 정합과 수체 탐지 후에도 시간 표본이 두 날짜뿐입니다.",
            "센서명과 제품 단계는 메타데이터 검증 전 확정할 수 없습니다.",
        ),
    ),
    NasDatasetGroup(
        group_id="support-assets",
        display_name="보관 ZIP · 전달 보조자료",
        relative_paths=(
            "신규데이터_위성영상_라벨링/Planetscope_WB.zip",
            "신규데이터_위성영상_라벨링/ICEYE_WB.zip",
            "샘플데이터_1세부제공/050258218010_기흥호수-20250707T042830Z-1-001.zip",
            "샘플데이터_1세부제공/050258288060_회동저수지-20250707T042830Z-1-001.zip",
        ),
        sensor_name="해당 없음",
        sensor_modality="none",
        sensor_confidence="not_applicable",
        roles=("delivery_archive", "duplicate_candidate", "support_material"),
        observation_dates=(),
        interval_days=(),
        acquisition_count=0,
        matched_pair_count=0,
        file_count=4,
        total_bytes=9_182_012_124,
        readiness="support_only",
        direct_prediction=False,
        direct_prediction_reason_ko=(
            "압축 보관본은 영상 프레임이 아니며 추출본과 중복될 수 있어 자동 예측에서 제외합니다."
        ),
        cautions_ko=(
            "추출본과 해시를 비교하기 전 파일 수와 용량에 중복 합산하지 않아야 합니다.",
            "UI에서는 다운로드 계보나 보관 상태만 표시하고 모델 입력으로 노출하지 않습니다.",
        ),
        related_group_ids=(
            "iceye-water-labels",
            "busan-water-labels",
            "giheung-optical",
            "hoedong-optical",
        ),
    ),
)


NAS_DATASET_BY_ID: Mapping[str, NasDatasetGroup] = MappingProxyType(
    {group.group_id: group for group in NAS_DATASET_GROUPS}
)


NAS_CATALOG_SUMMARY = NasCatalogSummary(
    file_count=TOTAL_FILE_COUNT,
    total_bytes=TOTAL_BYTES,
    group_count=len(NAS_DATASET_GROUPS),
    direct_prediction_group_count=sum(
        group.direct_prediction for group in NAS_DATASET_GROUPS
    ),
    materialized_sample_count=sum(group.materialized for group in NAS_DATASET_GROUPS),
)


def list_nas_dataset_groups() -> tuple[NasDatasetGroup, ...]:
    """Return the immutable catalog in stable presentation order."""

    return NAS_DATASET_GROUPS


def get_nas_dataset_group(group_id: str) -> NasDatasetGroup:
    """Look up one group and give a useful error for unknown identifiers."""

    try:
        return NAS_DATASET_BY_ID[group_id]
    except KeyError as exc:
        available = ", ".join(NAS_DATASET_BY_ID)
        raise KeyError(f"unknown NAS dataset group {group_id!r}; available: {available}") from exc


def materialized_nas_samples() -> tuple[NasDatasetGroup, ...]:
    """Return only groups connected to local backend-ready sample assets."""

    return tuple(group for group in NAS_DATASET_GROUPS if group.materialized)
