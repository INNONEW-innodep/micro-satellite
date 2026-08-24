"""Pure, immutable EDA summaries for the classified NAS delivery data.

The Streamlit renderer intentionally does not own the calculations in this
module.  The functions below consume the static NAS catalog and the sanitized
Busan materialization manifest, return frozen data classes, and build Plotly
figures from those values.  They never connect to the NAS and never treat a
display thumbnail as a scientific raster.

There are two deliberately different analysis levels:

* catalog EDA describes all classified files, bytes, groups and observation
  dates from :mod:`ui_next.nas_catalog`;
* Busan source EDA uses the source-common-grid statistics recorded while the
  real GeoTIFF labels were read.  It keeps those values separate from the
  512 x 512 presentation/API assets.

The holdout scores are one-point area baselines.  They are useful for checking
the EDA-to-model wiring, but they are not generalization metrics and must not
be presented as a validated 7/14/30-day forecasting result.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

import plotly.graph_objects as go

try:  # package imports used by pytest and other Python callers
    from .eda import axis_range_for
except ImportError:  # direct Streamlit execution adds ui_next/ to sys.path
    from eda import axis_range_for

try:  # package imports used by pytest and other Python callers
    from .nas_catalog import NasDatasetGroup, list_nas_dataset_groups
except ImportError:  # direct Streamlit execution adds ui_next/ to sys.path
    from nas_catalog import NasDatasetGroup, list_nas_dataset_groups


GIB = 1024**3
BUSAN_SOURCE_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "assets" / "nas" / "busan_2020" / "manifest.json"
)

GateState = Literal["ready", "limited", "blocked"]

MODALITY_LABELS: Mapping[str, str] = MappingProxyType(
    {"optical": "광학", "sar": "SAR", "none": "센서 없음"}
)
READINESS_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "demo_ready": "마스크 시연 후보",
        "preprocess_required": "전처리 필요",
        "single_date_reference": "단일 시점 참고",
        "support_only": "보조·보관 자료",
    }
)
GATE_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "catalog_eda": "카탈로그 EDA",
        "source_raster_eda": "원본 수치 EDA",
        "mask_forecast_demo": "마스크 예측 시연",
        "learned_timeseries": "학습 시계열 모델",
        "water_level_forecast": "절대 수위 예측",
        "future_truth_evaluation": "미래 정답 검증",
    }
)

_MODALITY_COLORS = MappingProxyType(
    {"optical": "#22d3ee", "sar": "#a78bfa", "none": "#64748b"}
)
_GATE_COLORSCALE = (
    (0.0, "#7f1d1d"),
    (0.499, "#7f1d1d"),
    (0.5, "#92400e"),
    (0.749, "#92400e"),
    (0.75, "#0f766e"),
    (1.0, "#0f766e"),
)
_DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(2,6,23,.42)",
    "font": {"color": "#cbd5e1"},
    "margin": {"l": 18, "r": 18, "t": 52, "b": 28},
}


@dataclass(frozen=True, slots=True)
class InventoryBucket:
    """One exclusive catalog breakdown bucket."""

    key: str
    label_ko: str
    group_count: int
    file_count: int
    total_bytes: int
    share_pct: float

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB


@dataclass(frozen=True, slots=True)
class InventoryGroupEDA:
    """Presentation-safe aggregate for one classified source group."""

    group_id: str
    display_name: str
    sensor_name: str
    sensor_modality: str
    readiness: str
    file_count: int
    total_bytes: int
    size_share_pct: float
    unique_date_count: int
    acquisition_count: int
    matched_pair_count: int
    first_date: dt.date | None
    last_date: dt.date | None
    period_days: int | None
    interval_days: tuple[int, ...]
    materialized: bool
    direct_prediction_candidate: bool

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB


@dataclass(frozen=True, slots=True)
class InventoryTimelinePoint:
    """One unique observation date on a catalog timeline lane."""

    group_id: str
    display_name: str
    observed_on: dt.date
    sensor_modality: str
    readiness: str
    materialized: bool


@dataclass(frozen=True, slots=True)
class NasInventoryEDA:
    """Whole-catalog EDA without claiming per-raster statistics."""

    file_count: int
    total_bytes: int
    group_count: int
    dated_group_count: int
    timeline_point_count: int
    acquisition_count: int
    matched_pair_count: int
    materialized_group_count: int
    direct_candidate_group_count: int
    groups: tuple[InventoryGroupEDA, ...]
    modality_buckets: tuple[InventoryBucket, ...]
    readiness_buckets: tuple[InventoryBucket, ...]
    timeline: tuple[InventoryTimelinePoint, ...]
    notes_ko: tuple[str, ...]

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB


@dataclass(frozen=True, slots=True)
class BusanSourceFrameEDA:
    """One paired input/label frame measured on the source common grid."""

    observed_on: dt.date
    interval_from_previous_days: int | None
    input_relative_path: str
    label_relative_path: str
    input_width: int
    input_height: int
    label_width: int
    label_height: int
    input_size_bytes: int
    label_size_bytes: int
    pair_grid_equal: bool
    source_water_pixels: int
    source_water_fraction_pct: float
    source_water_area_km2: float
    source_area_change_pct: float | None
    preview_water_pixels: int
    preview_water_area_km2: float
    preview_area_error_pct: float


@dataclass(frozen=True, slots=True)
class BusanSourceEDA:
    """Source-derived Busan sequence profile plus preview fidelity checks."""

    schema_version: str
    dataset_id: str
    classification: str
    sensor: str
    crs: str
    source_pixel_width_m: float
    source_pixel_height_m: float
    source_pixel_area_m2: float
    preview_width: int
    preview_height: int
    preview_pixel_area_m2: float
    common_bounds_utm: tuple[float, float, float, float]
    common_grid_width: int
    common_grid_height: int
    common_grid_pixel_count: int
    frame_count: int
    first_date: dt.date
    last_date: dt.date
    period_days: int
    interval_days: tuple[int, ...]
    interval_min_days: int
    interval_median_days: float
    interval_max_days: int
    source_area_first_km2: float
    source_area_last_km2: float
    source_area_min_km2: float
    source_area_max_km2: float
    source_area_change_pct: float
    max_abs_preview_area_error_pct: float
    frames: tuple[BusanSourceFrameEDA, ...]
    processing: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_ko: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AreaHoldoutScore:
    """One baseline evaluated against exactly one held-out area observation."""

    model_id: str
    model_name_ko: str
    training_frame_count: int
    training_start_date: dt.date
    training_end_date: dt.date
    holdout_date: dt.date
    observed_area_km2: float
    predicted_area_km2: float
    signed_error_km2: float
    absolute_error_km2: float
    absolute_percentage_error_pct: float | None
    mae_km2: float
    rmse_km2: float
    evaluation_count: int = 1
    scientific_validation_allowed: bool = False
    note_ko: str = (
        "마지막 한 시점만 평가한 연결 점검용 면적 기준선이며 일반화 성능이 아닙니다."
    )


@dataclass(frozen=True, slots=True)
class ReadinessGate:
    """One explicit capability gate; unlike one coarse readiness enum."""

    key: str
    label_ko: str
    state: GateState
    explanation_ko: str


@dataclass(frozen=True, slots=True)
class NasDatasetReadiness:
    """What can and cannot currently be done with one catalog group."""

    group_id: str
    display_name: str
    can_execute_current_api: bool
    headline_ko: str
    next_action_ko: str
    gates: tuple[ReadinessGate, ...]


def _parse_catalog_date(value: str, *, field: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date, got {value!r}") from exc


def _parse_manifest_date(value: Any, *, field: str) -> dt.date:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return _parse_catalog_date(text, field=field)


def _bucket_summaries(
    groups: Sequence[NasDatasetGroup],
    *,
    attribute: str,
    labels: Mapping[str, str],
    total_bytes: int,
) -> tuple[InventoryBucket, ...]:
    values: dict[str, list[NasDatasetGroup]] = defaultdict(list)
    for group in groups:
        values[str(getattr(group, attribute))].append(group)
    order = tuple(labels) + tuple(sorted(set(values).difference(labels)))
    return tuple(
        InventoryBucket(
            key=key,
            label_ko=labels.get(key, key),
            group_count=len(values[key]),
            file_count=sum(group.file_count for group in values[key]),
            total_bytes=sum(group.total_bytes for group in values[key]),
            share_pct=(
                sum(group.total_bytes for group in values[key]) / total_bytes * 100.0
                if total_bytes
                else 0.0
            ),
        )
        for key in order
        if values.get(key)
    )


def summarize_inventory(
    groups: Sequence[NasDatasetGroup] | None = None,
) -> NasInventoryEDA:
    """Summarize the complete classified catalog into immutable chart data."""

    source_groups = tuple(groups) if groups is not None else list_nas_dataset_groups()
    group_ids = [group.group_id for group in source_groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("NAS catalog group ids must be unique")
    total_files = sum(group.file_count for group in source_groups)
    total_bytes = sum(group.total_bytes for group in source_groups)
    group_points: list[InventoryGroupEDA] = []
    timeline: list[InventoryTimelinePoint] = []
    for group in source_groups:
        dates = tuple(
            _parse_catalog_date(value, field=f"{group.group_id}.observation_dates")
            for value in group.observation_dates
        )
        if dates != tuple(sorted(dates)) or len(dates) != len(set(dates)):
            raise ValueError(
                f"{group.group_id} observation_dates must be sorted and unique"
            )
        expected_intervals = tuple(
            (right - left).days for left, right in pairwise(dates)
        )
        if expected_intervals != group.interval_days:
            raise ValueError(
                f"{group.group_id} interval_days do not match observation_dates"
            )
        first = dates[0] if dates else None
        last = dates[-1] if dates else None
        group_points.append(
            InventoryGroupEDA(
                group_id=group.group_id,
                display_name=group.display_name,
                sensor_name=group.sensor_name,
                sensor_modality=group.sensor_modality,
                readiness=group.readiness,
                file_count=group.file_count,
                total_bytes=group.total_bytes,
                size_share_pct=(
                    group.total_bytes / total_bytes * 100.0 if total_bytes else 0.0
                ),
                unique_date_count=len(dates),
                acquisition_count=group.acquisition_count,
                matched_pair_count=group.matched_pair_count,
                first_date=first,
                last_date=last,
                period_days=(last - first).days
                if first is not None and last is not None
                else None,
                interval_days=group.interval_days,
                materialized=group.materialized,
                direct_prediction_candidate=group.direct_prediction,
            )
        )
        timeline.extend(
            InventoryTimelinePoint(
                group_id=group.group_id,
                display_name=group.display_name,
                observed_on=observed_on,
                sensor_modality=group.sensor_modality,
                readiness=group.readiness,
                materialized=group.materialized,
            )
            for observed_on in dates
        )

    return NasInventoryEDA(
        file_count=total_files,
        total_bytes=total_bytes,
        group_count=len(source_groups),
        dated_group_count=sum(bool(group.observation_dates) for group in source_groups),
        timeline_point_count=len(timeline),
        acquisition_count=sum(group.acquisition_count for group in source_groups),
        matched_pair_count=sum(group.matched_pair_count for group in source_groups),
        materialized_group_count=sum(group.materialized for group in source_groups),
        direct_candidate_group_count=sum(
            group.direct_prediction for group in source_groups
        ),
        groups=tuple(group_points),
        modality_buckets=_bucket_summaries(
            source_groups,
            attribute="sensor_modality",
            labels=MODALITY_LABELS,
            total_bytes=total_bytes,
        ),
        readiness_buckets=_bucket_summaries(
            source_groups,
            attribute="readiness",
            labels=READINESS_LABELS,
            total_bytes=total_bytes,
        ),
        timeline=tuple(
            sorted(timeline, key=lambda item: (item.observed_on, item.group_id))
        ),
        notes_ko=(
            "파일·용량은 ZIP 보관본을 포함한 전달 트리의 논리 합계입니다.",
            "타임라인 점은 그룹별 고유 날짜이며 같은 촬영의 원천·라벨 관계를 전역에서 중복 제거한 값이 아닙니다.",
            "파일 확장자·밴드·NoData 통계는 현재 정적 카탈로그에 없으므로 원본 래스터 프로파일과 구분합니다.",
        ),
    )


def inventory_group_rows(summary: NasInventoryEDA) -> tuple[dict[str, Any], ...]:
    """Return fresh table rows for Streamlit/dataframe consumers."""

    return tuple(
        {
            "group_id": group.group_id,
            "데이터 그룹": group.display_name,
            "센서 유형": MODALITY_LABELS.get(
                group.sensor_modality, group.sensor_modality
            ),
            "준비 상태": READINESS_LABELS.get(group.readiness, group.readiness),
            "파일": group.file_count,
            "용량 (GiB)": round(group.total_gib, 3),
            "용량 비중 (%)": round(group.size_share_pct, 2),
            "고유 날짜": group.unique_date_count,
            "Acquisition": group.acquisition_count,
            "입력-라벨 쌍": group.matched_pair_count,
            "로컬 시연셋": group.materialized,
        }
        for group in summary.groups
    )


def build_inventory_size_figure(summary: NasInventoryEDA) -> go.Figure:
    """Show every catalog group by source bytes, with file count in hover/text."""

    groups = sorted(summary.groups, key=lambda item: item.total_bytes)
    figure = go.Figure(
        go.Bar(
            x=[group.total_gib for group in groups],
            y=[group.display_name for group in groups],
            orientation="h",
            marker={
                "color": [
                    _MODALITY_COLORS.get(group.sensor_modality, "#64748b")
                    for group in groups
                ]
            },
            text=[
                f"{group.file_count:,}개 · {group.size_share_pct:.1f}%"
                for group in groups
            ],
            textposition="auto",
            customdata=[
                [group.file_count, group.unique_date_count, group.matched_pair_count]
                for group in groups
            ],
            hovertemplate=(
                "%{y}<br>%{x:.3f} GiB<br>파일 %{customdata[0]:,}개"
                "<br>고유 날짜 %{customdata[1]}일<br>입력-라벨 %{customdata[2]}쌍<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_DARK_LAYOUT,
        height=420,
        title={"text": "NAS 그룹별 원천 용량 · 보관본 포함", "x": 0.01},
        showlegend=False,
    )
    figure.update_xaxes(
        title_text="원천 용량 (GiB)", gridcolor="#1e293b", rangemode="tozero"
    )
    figure.update_yaxes(title_text="분류 그룹", gridcolor="#1e293b")
    return figure


def build_inventory_timeline_figure(summary: NasInventoryEDA) -> go.Figure:
    """Plot group-local unique dates without pretending products are one series."""

    figure = go.Figure()
    for group in summary.groups:
        points = [item for item in summary.timeline if item.group_id == group.group_id]
        if not points:
            continue
        figure.add_trace(
            go.Scatter(
                x=[item.observed_on.isoformat() for item in points],
                y=[item.display_name for item in points],
                mode="lines+markers",
                name=group.display_name,
                line={
                    "color": _MODALITY_COLORS.get(group.sensor_modality, "#64748b"),
                    "width": 1.5,
                },
                marker={
                    "size": 10 if group.materialized else 8,
                    "symbol": "diamond" if group.materialized else "circle",
                    "line": {"color": "#e2e8f0", "width": 0.7},
                },
                customdata=[
                    [
                        MODALITY_LABELS.get(
                            group.sensor_modality, group.sensor_modality
                        ),
                        READINESS_LABELS.get(group.readiness, group.readiness),
                    ]
                    for _ in points
                ],
                hovertemplate=(
                    "%{y}<br>%{x}<br>센서 %{customdata[0]}"
                    "<br>준비 %{customdata[1]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **_DARK_LAYOUT,
        height=440,
        title={
            "text": "그룹별 관측 타임라인 · 서로 다른 센서/지역은 별도 lane",
            "x": 0.01,
        },
        showlegend=False,
        hovermode="closest",
    )
    figure.update_xaxes(title_text="관측 날짜", gridcolor="#1e293b")
    figure.update_yaxes(title_text="데이터 그룹", gridcolor="#1e293b")
    return figure


def _as_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _as_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a JSON array")
    return value


def _number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return parsed


def _integer(value: Any, *, field: str, positive: bool = False) -> int:
    parsed = _number(value, field=field, positive=positive)
    if not parsed.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(parsed)


def _relative_path(value: Any, *, field: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{field} must be a safe logical relative path")
    return path.as_posix()


def _bounds(value: Any, *, field: str) -> tuple[float, float, float, float]:
    source = _as_mapping(value, field=field)
    xmin = _number(source.get("xmin"), field=f"{field}.xmin")
    ymin = _number(source.get("ymin"), field=f"{field}.ymin")
    xmax = _number(source.get("xmax"), field=f"{field}.xmax")
    ymax = _number(source.get("ymax"), field=f"{field}.ymax")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"{field} must describe positive bounds")
    return xmin, ymin, xmax, ymax


def _source_entry(
    frame: Mapping[str, Any], key: str, *, index: int
) -> tuple[str, int, int, int, tuple[float, float, float, float]]:
    source = _as_mapping(frame.get(key), field=f"frames[{index}].{key}")
    return (
        _relative_path(
            source.get("relative_path"), field=f"frames[{index}].{key}.relative_path"
        ),
        _integer(
            source.get("width"), field=f"frames[{index}].{key}.width", positive=True
        ),
        _integer(
            source.get("height"), field=f"frames[{index}].{key}.height", positive=True
        ),
        _integer(
            source.get("size_bytes"),
            field=f"frames[{index}].{key}.size_bytes",
            positive=True,
        ),
        _bounds(source.get("bounds_utm"), field=f"frames[{index}].{key}.bounds_utm"),
    )


def summarize_busan_source_eda(
    manifest_path: str | Path = BUSAN_SOURCE_MANIFEST_PATH,
) -> BusanSourceEDA:
    """Load and validate source-grid Busan EDA from its immutable manifest."""

    path = Path(manifest_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    manifest = _as_mapping(value, field="manifest")
    raw_frames = _as_sequence(manifest.get("frames"), field="frames")
    if len(raw_frames) < 2:
        raise ValueError("Busan source EDA requires at least two frames")

    declared_dates = tuple(
        _parse_manifest_date(item, field="dates")
        for item in _as_sequence(manifest.get("dates"), field="dates")
    )
    source_size = _as_sequence(
        manifest.get("source_pixel_size_m"), field="source_pixel_size_m"
    )
    if len(source_size) != 2:
        raise ValueError("source_pixel_size_m must contain width and height")
    pixel_width = _number(source_size[0], field="source_pixel_size_m[0]", positive=True)
    pixel_height = _number(
        source_size[1], field="source_pixel_size_m[1]", positive=True
    )
    pixel_area = _number(
        manifest.get("source_pixel_area_m2"),
        field="source_pixel_area_m2",
        positive=True,
    )
    if not math.isclose(pixel_area, pixel_width * pixel_height, rel_tol=1e-9):
        raise ValueError("source_pixel_area_m2 does not match source pixel dimensions")

    preview_shape = _as_sequence(manifest.get("preview_shape"), field="preview_shape")
    if len(preview_shape) != 2:
        raise ValueError("preview_shape must contain height and width")
    preview_height = _integer(preview_shape[0], field="preview_shape[0]", positive=True)
    preview_width = _integer(preview_shape[1], field="preview_shape[1]", positive=True)
    preview_pixel_area = _number(
        manifest.get("preview_pixel_area_m2"),
        field="preview_pixel_area_m2",
        positive=True,
    )
    common_bounds = _bounds(
        manifest.get("common_bounds_utm"), field="common_bounds_utm"
    )
    xmin, ymin, xmax, ymax = common_bounds
    grid_width_raw = (xmax - xmin) / pixel_width
    grid_height_raw = (ymax - ymin) / pixel_height
    grid_width = round(grid_width_raw)
    grid_height = round(grid_height_raw)
    if not math.isclose(grid_width_raw, grid_width, abs_tol=1e-6) or not math.isclose(
        grid_height_raw, grid_height, abs_tol=1e-6
    ):
        raise ValueError("common bounds do not align to the declared source pixel size")
    grid_pixel_count = grid_width * grid_height

    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_frames):
        frame = _as_mapping(raw, field=f"frames[{index}]")
        observed_on = _parse_manifest_date(
            frame.get("date"), field=f"frames[{index}].date"
        )
        input_path, input_width, input_height, input_bytes, input_bounds = (
            _source_entry(frame, "input_source", index=index)
        )
        label_path, label_width, label_height, label_bytes, label_bounds = (
            _source_entry(frame, "label_source", index=index)
        )
        source_pixels = _integer(
            frame.get("source_common_grid_water_pixels"),
            field=f"frames[{index}].source_common_grid_water_pixels",
        )
        if not 0 <= source_pixels <= grid_pixel_count:
            raise ValueError(
                f"frames[{index}] source water pixels exceed the common grid"
            )
        source_area_m2 = _number(
            frame.get("source_common_grid_water_area_m2"),
            field=f"frames[{index}].source_common_grid_water_area_m2",
        )
        if not math.isclose(source_area_m2, source_pixels * pixel_area, rel_tol=1e-9):
            raise ValueError(
                f"frames[{index}] source water area does not match pixel count"
            )
        preview_pixels = _integer(
            frame.get("preview_water_pixels"),
            field=f"frames[{index}].preview_water_pixels",
        )
        if not 0 <= preview_pixels <= preview_width * preview_height:
            raise ValueError(
                f"frames[{index}] preview water pixels exceed preview dimensions"
            )
        preview_area_m2 = _number(
            frame.get("preview_water_area_m2"),
            field=f"frames[{index}].preview_water_area_m2",
        )
        if not math.isclose(
            preview_area_m2, preview_pixels * preview_pixel_area, rel_tol=1e-9
        ):
            raise ValueError(
                f"frames[{index}] preview water area does not match pixel count"
            )
        parsed.append(
            {
                "date": observed_on,
                "input_path": input_path,
                "label_path": label_path,
                "input_width": input_width,
                "input_height": input_height,
                "label_width": label_width,
                "label_height": label_height,
                "input_bytes": input_bytes,
                "label_bytes": label_bytes,
                "pair_grid_equal": (
                    input_width == label_width
                    and input_height == label_height
                    and input_bounds == label_bounds
                ),
                "source_pixels": source_pixels,
                "source_area_km2": source_area_m2 / 1_000_000.0,
                "preview_pixels": preview_pixels,
                "preview_area_km2": preview_area_m2 / 1_000_000.0,
            }
        )

    dates = tuple(item["date"] for item in parsed)
    if dates != declared_dates:
        raise ValueError("manifest frame dates do not match declared dates")
    if dates != tuple(sorted(dates)) or len(dates) != len(set(dates)):
        raise ValueError("manifest frame dates must be sorted and unique")
    intervals = tuple((right - left).days for left, right in pairwise(dates))
    source_areas = tuple(float(item["source_area_km2"]) for item in parsed)
    preview_errors = tuple(
        (float(item["preview_area_km2"]) - source_area) / source_area * 100.0
        if source_area
        else 0.0
        for item, source_area in zip(parsed, source_areas, strict=True)
    )
    frames: list[BusanSourceFrameEDA] = []
    for index, (item, preview_error) in enumerate(
        zip(parsed, preview_errors, strict=True)
    ):
        source_area = float(item["source_area_km2"])
        previous_area = source_areas[index - 1] if index else None
        frames.append(
            BusanSourceFrameEDA(
                observed_on=item["date"],
                interval_from_previous_days=intervals[index - 1] if index else None,
                input_relative_path=item["input_path"],
                label_relative_path=item["label_path"],
                input_width=item["input_width"],
                input_height=item["input_height"],
                label_width=item["label_width"],
                label_height=item["label_height"],
                input_size_bytes=item["input_bytes"],
                label_size_bytes=item["label_bytes"],
                pair_grid_equal=item["pair_grid_equal"],
                source_water_pixels=item["source_pixels"],
                source_water_fraction_pct=item["source_pixels"]
                / grid_pixel_count
                * 100.0,
                source_water_area_km2=source_area,
                source_area_change_pct=(
                    (source_area - previous_area) / previous_area * 100.0
                    if previous_area
                    else None
                ),
                preview_water_pixels=item["preview_pixels"],
                preview_water_area_km2=item["preview_area_km2"],
                preview_area_error_pct=preview_error,
            )
        )

    first_area = source_areas[0]
    last_area = source_areas[-1]
    processing = tuple(
        str(value)
        for value in _as_sequence(manifest.get("processing", ()), field="processing")
    )
    limitations = tuple(
        str(value)
        for value in _as_sequence(manifest.get("limitations", ()), field="limitations")
    )
    return BusanSourceEDA(
        schema_version=str(manifest.get("schema_version") or ""),
        dataset_id=str(manifest.get("dataset_id") or ""),
        classification=str(manifest.get("classification") or ""),
        sensor=str(manifest.get("sensor") or ""),
        crs=str(manifest.get("crs") or ""),
        source_pixel_width_m=pixel_width,
        source_pixel_height_m=pixel_height,
        source_pixel_area_m2=pixel_area,
        preview_width=preview_width,
        preview_height=preview_height,
        preview_pixel_area_m2=preview_pixel_area,
        common_bounds_utm=common_bounds,
        common_grid_width=grid_width,
        common_grid_height=grid_height,
        common_grid_pixel_count=grid_pixel_count,
        frame_count=len(frames),
        first_date=dates[0],
        last_date=dates[-1],
        period_days=(dates[-1] - dates[0]).days,
        interval_days=intervals,
        interval_min_days=min(intervals),
        interval_median_days=float(statistics.median(intervals)),
        interval_max_days=max(intervals),
        source_area_first_km2=first_area,
        source_area_last_km2=last_area,
        source_area_min_km2=min(source_areas),
        source_area_max_km2=max(source_areas),
        source_area_change_pct=(last_area - first_area) / first_area * 100.0,
        max_abs_preview_area_error_pct=max(abs(value) for value in preview_errors),
        frames=tuple(frames),
        processing=processing,
        limitations=limitations,
        evidence_ko=(
            "수체 픽셀 수는 원본 라벨의 네 시점 공통 교집합을 블록 단위로 읽어 1값을 정확 집계했습니다.",
            "km²는 manifest의 3 m x 3 m 픽셀 면적을 적용한 원본 공통격자 계산값입니다.",
            "512 x 512 값은 API/발표용 최근접 리샘플 자산이며 원본 수치와 별도 계열로 표시합니다.",
            "센서명과 CRS 확정 수준, NoData·밴드 품질은 별도 원본 raster profile이 추가되기 전까지 제한적입니다.",
        ),
    )


def busan_source_rows(summary: BusanSourceEDA) -> tuple[dict[str, Any], ...]:
    """Return source/presentation comparison rows for a visible EDA table."""

    return tuple(
        {
            "관측 날짜": frame.observed_on.isoformat(),
            "이전 간격 (일)": frame.interval_from_previous_days,
            "원본 수체 픽셀": frame.source_water_pixels,
            "원본 수체 비율 (%)": round(frame.source_water_fraction_pct, 4),
            "원본 공통격자 면적 (km²)": round(frame.source_water_area_km2, 6),
            "직전 대비 (%)": (
                round(frame.source_area_change_pct, 3)
                if frame.source_area_change_pct is not None
                else None
            ),
            "512 면적 (km²)": round(frame.preview_water_area_km2, 6),
            "512 면적 오차 (%)": round(frame.preview_area_error_pct, 3),
            "입력-라벨 격자 일치": frame.pair_grid_equal,
        }
        for frame in summary.frames
    )


def evaluate_busan_area_holdout(
    summary: BusanSourceEDA | None = None,
) -> tuple[AreaHoldoutScore, ...]:
    """Evaluate persistence and irregular-date linear trend on the last point."""

    source = summary or summarize_busan_source_eda()
    if len(source.frames) < 3:
        raise ValueError("area holdout requires at least three frames")
    training = source.frames[:-1]
    holdout = source.frames[-1]
    observed = holdout.source_water_area_km2

    persistence = training[-1].source_water_area_km2
    first_date = training[0].observed_on
    x = tuple((frame.observed_on - first_date).days for frame in training)
    y = tuple(frame.source_water_area_km2 for frame in training)
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("linear trend needs distinct training dates")
    slope = (
        sum((time - mean_x) * (area - mean_y) for time, area in zip(x, y, strict=True))
        / denominator
    )
    intercept = mean_y - slope * mean_x
    holdout_day = (holdout.observed_on - first_date).days
    linear = max(0.0, intercept + slope * holdout_day)

    def score(model_id: str, name: str, predicted: float) -> AreaHoldoutScore:
        signed = predicted - observed
        absolute = abs(signed)
        return AreaHoldoutScore(
            model_id=model_id,
            model_name_ko=name,
            training_frame_count=len(training),
            training_start_date=training[0].observed_on,
            training_end_date=training[-1].observed_on,
            holdout_date=holdout.observed_on,
            observed_area_km2=observed,
            predicted_area_km2=predicted,
            signed_error_km2=signed,
            absolute_error_km2=absolute,
            absolute_percentage_error_pct=absolute / observed * 100.0
            if observed
            else None,
            mae_km2=absolute,
            rmse_km2=absolute,
        )

    return (
        score("area-persistence", "직전 면적 유지", persistence),
        score("irregular-linear-area", "불규칙 날짜 선형 면적 추세", linear),
    )


def holdout_rows(scores: Sequence[AreaHoldoutScore]) -> tuple[dict[str, Any], ...]:
    """Return explicit one-point holdout rows; MAE/RMSE equality stays visible."""

    return tuple(
        {
            "기준선": score.model_name_ko,
            "학습 프레임": score.training_frame_count,
            "홀드아웃 날짜": score.holdout_date.isoformat(),
            "관측 면적 (km²)": round(score.observed_area_km2, 6),
            "예측 면적 (km²)": round(score.predicted_area_km2, 6),
            "MAE (km²)": round(score.mae_km2, 6),
            "RMSE (km²)": round(score.rmse_km2, 6),
            "절대 백분율 오차 (%)": (
                round(score.absolute_percentage_error_pct, 3)
                if score.absolute_percentage_error_pct is not None
                else None
            ),
            "평가 표본": score.evaluation_count,
            "검증 성능 주장 가능": score.scientific_validation_allowed,
        }
        for score in scores
    )


def build_busan_source_area_figure(
    summary: BusanSourceEDA,
    holdout_scores: Sequence[AreaHoldoutScore] | None = None,
) -> go.Figure:
    """Compare source-common-grid and 512-derived area without mixing them."""

    dates = [frame.observed_on.isoformat() for frame in summary.frames]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=[frame.source_water_area_km2 for frame in summary.frames],
            name="원본 공통격자 계산",
            mode="lines+markers",
            line={"color": "#22d3ee", "width": 3},
            marker={"size": 9},
            hovertemplate="%{x}<br>원본 면적 %{y:.6f} km²<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=[frame.preview_water_area_km2 for frame in summary.frames],
            name="512 파생 자산",
            mode="lines+markers",
            line={"color": "#f59e0b", "width": 2, "dash": "dot"},
            marker={"size": 7},
            hovertemplate="%{x}<br>512 면적 %{y:.6f} km²<extra></extra>",
        )
    )
    for index, score in enumerate(holdout_scores or ()):
        figure.add_trace(
            go.Scatter(
                x=[score.holdout_date.isoformat()],
                y=[score.predicted_area_km2],
                name=f"홀드아웃 · {score.model_name_ko}",
                mode="markers",
                marker={
                    "size": 13,
                    "symbol": "x" if index == 0 else "diamond-open",
                    "color": "#fb7185" if index == 0 else "#a78bfa",
                    "line": {"width": 2},
                },
                hovertemplate=(
                    "%{x}<br>예측 면적 %{y:.6f} km²"
                    f"<br>단일 홀드아웃 MAE {score.mae_km2:.6f} km²<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **_DARK_LAYOUT,
        height=390,
        title={"text": "부산 수체면적 EDA · 원본 계산과 512 파생본 분리", "x": 0.01},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.14},
    )
    figure.update_xaxes(title_text="관측 날짜", gridcolor="#1e293b")
    # 면적·수위는 0에서 시작하면 실제 변동이 선 두께에 묻힌다(축 범위 이슈).
    figure.update_yaxes(
        title_text="수체 면적 (km²)", gridcolor="#1e293b", range=axis_range_for(figure)
    )
    return figure


def build_busan_resampling_error_figure(summary: BusanSourceEDA) -> go.Figure:
    """Plot signed area error introduced by the 512 presentation grid."""

    values = [frame.preview_area_error_pct for frame in summary.frames]
    figure = go.Figure(
        go.Bar(
            x=[frame.observed_on.isoformat() for frame in summary.frames],
            y=values,
            marker={
                "color": ["#38bdf8" if value >= 0 else "#f59e0b" for value in values]
            },
            text=[f"{value:+.3f}%" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>512-원본 면적 오차 %{y:+.3f}%<extra></extra>",
        )
    )
    figure.update_layout(
        **_DARK_LAYOUT,
        height=310,
        title={"text": "512 리샘플 면적 보존 오차 · 원본 공통격자 대비", "x": 0.01},
        showlegend=False,
    )
    figure.update_xaxes(title_text="관측 날짜", gridcolor="#1e293b")
    figure.update_yaxes(title_text="면적 오차 (%)", gridcolor="#1e293b", zeroline=True)
    return figure


def _gate(key: str, state: GateState, explanation: str) -> ReadinessGate:
    return ReadinessGate(key, GATE_LABELS[key], state, explanation)


def explain_readiness(group: NasDatasetGroup) -> NasDatasetReadiness:
    """Explain capability-by-capability readiness for one classified group."""

    gates: list[ReadinessGate] = [
        _gate(
            "catalog_eda",
            "ready",
            "분류, 파일 수, 용량, 센서 유형, 날짜와 관측 간격을 확인할 수 있습니다.",
        )
    ]
    if group.group_id == "busan-water-labels" and group.materialized:
        gates.append(
            _gate(
                "source_raster_eda",
                "limited",
                "원본 공통격자 수체 픽셀·면적과 pair 크기는 확인됐지만 NoData·밴드 품질 전체 프로파일은 없습니다.",
            )
        )
        gates.append(
            _gate(
                "mask_forecast_demo",
                "ready",
                "날짜가 맞는 이진 라벨 네 장이 512 공통격자로 materialize되어 현재 API 기준선을 실행할 수 있습니다.",
            )
        )
        headline = "원본 파생 수치 EDA와 마스크 기준선 시연 가능"
        next_action = "EDA에서 날짜·면적·리샘플 오차를 확인한 뒤 이 profile을 그대로 예측 입력에 전달합니다."
    elif group.group_id == "iceye-water-labels" and group.materialized:
        gates.append(
            _gate(
                "source_raster_eda",
                "limited",
                "전체 라벨 값·NoData·날짜별 pair 정합·공통영역 면적은 확인했지만 SAR 입력의 전체 방사 품질과 유효 footprint는 별도 검수가 필요합니다.",
            )
        )
        gates.append(
            _gate(
                "mask_forecast_demo",
                "ready",
                "검수된 0/1 라벨 네 장이 512×699 공통 지도격자로 materialize되어 현재 다중시점 기준선 API를 실행할 수 있습니다.",
            )
        )
        headline = "ICEYE 원본 라벨 EDA와 공통격자 기준선 시연 가능"
        next_action = "입사각·look side·input=0 유효영역 주의를 확인하고 공통격자 라벨을 다중시점 기준선에 전달합니다."
    elif group.direct_prediction:
        gates.append(
            _gate(
                "source_raster_eda",
                "blocked",
                "동일 격자·CRS·NoData·라벨 값의 원본 raster profile이 아직 materialize되지 않았습니다.",
            )
        )
        gates.append(
            _gate(
                "mask_forecast_demo",
                "limited",
                "입력-라벨 시퀀스는 있으나 로컬 공통격자 마스크와 검수 manifest 생성이 필요합니다.",
            )
        )
        headline = "원천 구성은 후보지만 원본 프로파일·변환 대기"
        next_action = "원본을 읽기 전용으로 profile한 뒤 날짜 pair, 격자, NoData, 이진 라벨을 검수하고 경량 시연셋을 만듭니다."
    else:
        gates.append(
            _gate(
                "source_raster_eda",
                "blocked",
                "현재는 정적 메타데이터 또는 표시 전용 썸네일만 있어 원본 픽셀 품질 EDA가 없습니다.",
            )
        )
        gates.append(
            _gate(
                "mask_forecast_demo",
                "blocked",
                group.direct_prediction_reason_ko,
            )
        )
        if group.readiness == "support_only":
            headline = "보관·보조자료 · 분석 입력 제외"
            next_action = (
                "추출본과 중복 여부·계보만 관리하고 예측 입력으로 사용하지 않습니다."
            )
        elif group.readiness == "single_date_reference":
            headline = "단일 시점 산출 참고만 가능"
            next_action = (
                "같은 AOI의 추가 날짜와 일관된 수체 마스크·수위 관측을 확보합니다."
            )
        else:
            headline = "원천영상 전처리와 수체 탐지가 먼저 필요"
            next_action = "센서별 보정·정합·품질 마스크 적용 후 수체 탐지 결과를 날짜별 이진 마스크로 만듭니다."

    gates.extend(
        (
            _gate(
                "learned_timeseries",
                "blocked",
                "다중시점 면적 추세 기준선은 실행되지만 학습 모델이 아니며, 제공 관측은 최대 5시점이라 일별 일반화 학습·검증에 부족합니다.",
            ),
            _gate(
                "water_level_forecast",
                "blocked",
                "동일 AOI·날짜의 연속 실측 수위와 검교정된 면적-수위 관계가 없습니다.",
            ),
            _gate(
                "future_truth_evaluation",
                "blocked",
                "마지막 관측 이후 일별 미래 수체/수위 정답이 없어 7·14·30일 검증 RMSE를 계산할 수 없습니다.",
            ),
        )
    )
    return NasDatasetReadiness(
        group_id=group.group_id,
        display_name=group.display_name,
        can_execute_current_api=bool(group.materialized and group.direct_prediction),
        headline_ko=headline,
        next_action_ko=next_action,
        gates=tuple(gates),
    )


def summarize_readiness(
    groups: Sequence[NasDatasetGroup] | None = None,
) -> tuple[NasDatasetReadiness, ...]:
    """Return readiness matrices in stable catalog order."""

    source = tuple(groups) if groups is not None else list_nas_dataset_groups()
    return tuple(explain_readiness(group) for group in source)


def readiness_rows(
    summaries: Sequence[NasDatasetReadiness],
) -> tuple[dict[str, Any], ...]:
    """Flatten readiness gates for a visible audit table."""

    rows: list[dict[str, Any]] = []
    state_labels = {"ready": "가능", "limited": "제한/준비중", "blocked": "불가"}
    for summary in summaries:
        for gate in summary.gates:
            rows.append(
                {
                    "group_id": summary.group_id,
                    "데이터 그룹": summary.display_name,
                    "기능": gate.label_ko,
                    "상태": state_labels[gate.state],
                    "설명": gate.explanation_ko,
                }
            )
    return tuple(rows)


def build_readiness_matrix_figure(
    summaries: Sequence[NasDatasetReadiness],
) -> go.Figure:
    """Visualize distinct EDA/prediction capabilities instead of one badge."""

    items = tuple(summaries)
    gate_keys = tuple(GATE_LABELS)
    numeric = {"blocked": 0, "limited": 1, "ready": 2}
    state_labels = {"blocked": "불가", "limited": "제한", "ready": "가능"}
    z: list[list[int]] = []
    text: list[list[str]] = []
    hover: list[list[str]] = []
    for summary in items:
        by_key = {gate.key: gate for gate in summary.gates}
        z.append([numeric[by_key[key].state] for key in gate_keys])
        text.append([state_labels[by_key[key].state] for key in gate_keys])
        hover.append([by_key[key].explanation_ko for key in gate_keys])
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=[GATE_LABELS[key] for key in gate_keys],
            y=[summary.display_name for summary in items],
            zmin=0,
            zmax=2,
            colorscale=_GATE_COLORSCALE,
            showscale=False,
            text=text,
            texttemplate="%{text}",
            customdata=hover,
            hovertemplate="%{y}<br>%{x} · %{text}<br>%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        **_DARK_LAYOUT,
        height=470,
        title={"text": "데이터셋별 분석·예측 준비도 · 기능별 분리", "x": 0.01},
    )
    figure.update_xaxes(side="top", tickangle=-18)
    figure.update_yaxes(autorange="reversed")
    return figure


__all__ = [
    "BUSAN_SOURCE_MANIFEST_PATH",
    "GATE_LABELS",
    "MODALITY_LABELS",
    "READINESS_LABELS",
    "AreaHoldoutScore",
    "BusanSourceEDA",
    "BusanSourceFrameEDA",
    "InventoryBucket",
    "InventoryGroupEDA",
    "InventoryTimelinePoint",
    "NasDatasetReadiness",
    "NasInventoryEDA",
    "ReadinessGate",
    "build_busan_resampling_error_figure",
    "build_busan_source_area_figure",
    "build_inventory_size_figure",
    "build_inventory_timeline_figure",
    "build_readiness_matrix_figure",
    "busan_source_rows",
    "evaluate_busan_area_holdout",
    "explain_readiness",
    "holdout_rows",
    "inventory_group_rows",
    "readiness_rows",
    "summarize_busan_source_eda",
    "summarize_inventory",
    "summarize_readiness",
]
