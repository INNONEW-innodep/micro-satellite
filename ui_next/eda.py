"""Presentation-friendly exploratory summaries for the operator UI.

The summary functions in this module are deliberately independent from
Streamlit session state.  They accept the immutable ``SampleDataset`` objects
used by the quick-start catalog as well as ordinary mapping rows, and return
frozen data classes that are straightforward to test or reuse in an API guide.

Physical area is shown only when the caller explicitly confirms that pixel
area is known.  A placeholder pixel size must never turn a documentation or
synthetic mask into an apparently measured square-kilometre value.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from typing import Any

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageSequence, UnidentifiedImageError
from plotly.subplots import make_subplots

_DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(2,6,23,.42)",
    "font": {"color": "#cbd5e1"},
    "hovermode": "x unified",
    "margin": {"l": 18, "r": 18, "t": 42, "b": 22},
}

_RISK_ORDER = ("normal", "caution", "flood_risk", "drought_risk", "unknown")
_RISK_LABELS = {
    "normal": "정상",
    "caution": "주의",
    "flood_risk": "홍수 위험",
    "drought_risk": "가뭄 위험",
    "unknown": "미분류",
}
_RISK_COLORS = {
    "normal": "#22c55e",
    "caution": "#f97316",
    "flood_risk": "#ef4444",
    "drought_risk": "#a855f7",
    "unknown": "#64748b",
}


@dataclass(frozen=True, slots=True)
class InputEDAPoint:
    """One decoded observation used by the input EDA chart."""

    frame_index: int
    name: str
    observed_on: dt.date | None
    width: int | None
    height: int | None
    water_pixels: int | None
    water_area_km2: float | None
    change_pct: float | None
    water_level_m: float | None


@dataclass(frozen=True, slots=True)
class InputEDASummary:
    frame_count: int
    unique_date_count: int
    duplicate_date_count: int
    invalid_date_count: int
    start_date: dt.date | None
    end_date: dt.date | None
    period_days: int | None
    interval_min_days: int | None
    interval_median_days: float | None
    interval_max_days: int | None
    resolutions: tuple[tuple[int, int], ...]
    resolution_label: str
    decoded_frame_count: int
    water_pixel_first: int | None
    water_pixel_last: int | None
    water_pixel_min: int | None
    water_pixel_max: int | None
    water_pixel_change_pct: float | None
    pixel_area_known: bool
    pixel_area_m2: float | None
    water_area_first_km2: float | None
    water_area_last_km2: float | None
    points: tuple[InputEDAPoint, ...]
    caption: str
    interpretation: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WeatherEDAPoint:
    row_index: int
    observed_on: dt.date | None
    kind: str
    precipitation_mm: float | None
    temperature_c: float | None
    humidity_pct: float | None


@dataclass(frozen=True, slots=True)
class WeatherEDASummary:
    row_count: int
    dated_row_count: int
    unique_date_count: int
    duplicate_date_count: int
    invalid_date_count: int
    observed_count: int
    scenario_count: int
    unknown_kind_count: int
    start_date: dt.date | None
    end_date: dt.date | None
    period_days: int | None
    precipitation_count: int
    precipitation_sum_mm: float | None
    precipitation_mean_mm: float | None
    precipitation_max_mm: float | None
    precipitation_max_date: dt.date | None
    precipitation_missing_rate_pct: float
    feature_missing_rate_pct: float
    temperature_count: int
    humidity_count: int
    points: tuple[WeatherEDAPoint, ...]
    caption: str
    interpretation: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResultEDAPoint:
    row_index: int
    horizon: int | None
    target_date: dt.date | None
    water_area_pixels: int | None
    water_area_km2: float | None
    area_change_pct: float | None
    water_level_m: float | None
    water_level_change_m: float | None
    risk: str


@dataclass(frozen=True, slots=True)
class ResultEDASummary:
    step_count: int
    unique_horizon_count: int
    horizon_min: int | None
    horizon_max: int | None
    start_date: dt.date | None
    end_date: dt.date | None
    period_days: int | None
    area_unit: str
    area_value_first: float | None
    area_value_last: float | None
    area_value_min: float | None
    area_value_max: float | None
    area_net_change: float | None
    area_net_change_pct: float | None
    latest_area_change_from_input_pct: float | None
    water_level_count: int
    water_level_first_m: float | None
    water_level_last_m: float | None
    water_level_min_m: float | None
    water_level_max_m: float | None
    water_level_net_change_m: float | None
    risk_counts: tuple[tuple[str, int], ...]
    pixel_area_known: bool
    points: tuple[ResultEDAPoint, ...]
    caption: str
    interpretation: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EDAOverview:
    """All three summaries returned by :func:`summarize_eda`."""

    input: InputEDASummary
    weather: WeatherEDASummary
    result: ResultEDASummary


@dataclass(frozen=True, slots=True)
class _DecodedMask:
    width: int
    height: int
    water_pixels: int


def _value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _first_value(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None and value != "":
            return value
    return None


def _parse_date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            converted = value.to_pydatetime()
            return converted.date() if isinstance(converted, dt.datetime) else converted
        except (AttributeError, TypeError, ValueError):
            pass
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return None
    candidates = (text[:10], text.replace(" ", "").rstrip("."))
    for candidate in candidates:
        try:
            if "-" in candidate:
                return dt.date.fromisoformat(candidate)
            if "." in candidate:
                year, month, day = (int(part) for part in candidate.split("."))
                return dt.date(year, month, day)
            if len(candidate) == 8 and candidate.isdigit():
                return dt.date(
                    int(candidate[:4]), int(candidate[4:6]), int(candidate[6:])
                )
        except ValueError:
            continue
    return None


def _finite_float(value: Any, *, non_negative: bool = False) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or (non_negative and parsed < 0):
        return None
    return parsed


def _finite_int(value: Any, *, non_negative: bool = False) -> int | None:
    parsed = _finite_float(value, non_negative=non_negative)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _percentage_change(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None or previous == 0:
        return 0.0 if previous == 0 and current == 0 else None
    return (float(current) - float(previous)) / abs(float(previous)) * 100.0


def _date_metrics(
    dates: Sequence[dt.date | None],
) -> tuple[
    int,
    int,
    int,
    dt.date | None,
    dt.date | None,
    int | None,
    int | None,
    float | None,
    int | None,
]:
    valid = [value for value in dates if value is not None]
    unique = sorted(set(valid))
    duplicate_count = len(valid) - len(unique)
    start = unique[0] if unique else None
    end = unique[-1] if unique else None
    period = (end - start).days if start is not None and end is not None else None
    intervals = [(right - left).days for left, right in pairwise(unique)]
    return (
        len(valid),
        len(unique),
        duplicate_count,
        start,
        end,
        period,
        min(intervals) if intervals else None,
        float(statistics.median(intervals)) if intervals else None,
        max(intervals) if intervals else None,
    )


def _threshold_count(array: np.ndarray, threshold: float) -> _DecodedMask:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("mask frame is not two-dimensional")
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("mask frame is empty or non-numeric")
    numeric = values.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or float(numeric.min()) < 0:
        raise ValueError("mask frame contains negative or non-finite values")
    maximum = float(numeric.max())
    if maximum <= 1:
        cutoff = threshold
    elif np.issubdtype(values.dtype, np.integer) and maximum > 255:
        cutoff = threshold * float(np.iinfo(values.dtype).max)
    else:
        cutoff = threshold * 255.0
    return _DecodedMask(
        width=int(values.shape[1]),
        height=int(values.shape[0]),
        water_pixels=int(np.count_nonzero(numeric >= cutoff)),
    )


def _array_frames(array: np.ndarray) -> tuple[np.ndarray, ...]:
    if array.ndim == 2:
        return (array,)
    if array.ndim == 3 and array.shape[-1] == 1:
        return (array[..., 0],)
    if array.ndim == 3:
        return tuple(array[index] for index in range(array.shape[0]))
    if array.ndim == 4 and array.shape[-1] == 1:
        return tuple(array[index, ..., 0] for index in range(array.shape[0]))
    raise ValueError("mask must use [H,W], [T,H,W], or a singleton channel")


def _content_bytes(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if hasattr(value, "getvalue"):
        try:
            result = value.getvalue()
            return bytes(result) if result is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _decode_row_masks(
    row: Mapping[str, Any], threshold: float
) -> tuple[_DecodedMask, ...]:
    explicit_pixels = _finite_int(
        _first_value(row, "water_pixels", "water_area_pixels"), non_negative=True
    )
    explicit_width = _finite_int(
        _first_value(row, "width", "mask_width"), non_negative=True
    )
    explicit_height = _finite_int(
        _first_value(row, "height", "mask_height"), non_negative=True
    )
    if explicit_pixels is not None and explicit_width and explicit_height:
        return (_DecodedMask(explicit_width, explicit_height, explicit_pixels),)

    direct_mask = row.get("mask")
    if direct_mask is not None:
        return tuple(
            _threshold_count(frame, threshold)
            for frame in _array_frames(np.asarray(direct_mask))
        )

    payload = _content_bytes(_first_value(row, "content", "png_bytes", "bytes"))
    if not payload:
        if explicit_pixels is not None:
            return (
                _DecodedMask(
                    explicit_width or 0, explicit_height or 0, explicit_pixels
                ),
            )
        return ()

    name = str(row.get("name") or row.get("filename") or "").lower()
    if name.endswith(".npy") or payload.startswith(b"\x93NUMPY"):
        array = np.load(BytesIO(payload), allow_pickle=False)
        return tuple(
            _threshold_count(frame, threshold) for frame in _array_frames(array)
        )

    try:
        with Image.open(BytesIO(payload)) as image:
            return tuple(
                _threshold_count(np.asarray(frame.convert("L")), threshold)
                for frame in ImageSequence.Iterator(image)
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"image payload could not be decoded: {exc}") from exc


def _sample_rows(sample: Any) -> tuple[Mapping[str, Any], ...]:
    if sample is None:
        return ()
    factory = _value(sample, "input_rows")
    if callable(factory):
        return tuple(factory())
    frames = _value(sample, "frames", ()) or ()
    rows: list[Mapping[str, Any]] = []
    for frame in frames:
        rows.append(
            {
                "name": _value(frame, "name", "frame.png"),
                "date": _value(frame, "observed_on"),
                "water_level_m": _value(frame, "water_level_m"),
                "content": _value(frame, "png_bytes"),
            }
        )
    return tuple(rows)


def _resolve_pixel_area(
    sample: Any,
    rows: Sequence[Mapping[str, Any]],
    pixel_area_m2: float | None,
    pixel_area_known: bool | None,
) -> tuple[bool, float | None]:
    if pixel_area_known is None:
        sample_flag = _value(sample, "pixel_area_known")
        if sample_flag is not None:
            pixel_area_known = bool(sample_flag)
        else:
            pixel_area_known = any(bool(row.get("pixel_area_known")) for row in rows)
    if pixel_area_m2 is None:
        pixel_area_m2 = _finite_float(
            _value(sample, "pixel_area_m2"), non_negative=True
        )
    if pixel_area_m2 is None:
        pixel_area_m2 = next(
            (
                value
                for row in rows
                if (value := _finite_float(row.get("pixel_area_m2"), non_negative=True))
                is not None
            ),
            None,
        )
    valid = bool(pixel_area_known and pixel_area_m2 is not None and pixel_area_m2 > 0)
    return valid, pixel_area_m2 if valid else None


def summarize_input_eda(
    sample: Any = None,
    input_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    pixel_area_m2: float | None = None,
    pixel_area_known: bool | None = None,
) -> InputEDASummary:
    """Summarize dated mask observations without asserting unknown area units."""

    rows = tuple(input_rows) if input_rows is not None else _sample_rows(sample)
    threshold = _finite_float(_value(sample, "threshold")) or 0.5
    area_known, resolved_pixel_area = _resolve_pixel_area(
        sample, rows, pixel_area_m2, pixel_area_known
    )
    notes: list[str] = []
    raw_points: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        observed_on = _parse_date(
            _first_value(row, "date", "timestamp", "observed_on", "source_date")
        )
        name = str(row.get("name") or row.get("filename") or f"frame_{row_index}")
        water_level = _finite_float(row.get("water_level_m"))
        try:
            decoded = _decode_row_masks(row, threshold)
        except (TypeError, ValueError, OSError) as exc:
            decoded = ()
            notes.append(f"{name}: 마스크 통계를 읽지 못했습니다 ({exc}).")
        if not decoded:
            raw_points.append(
                {
                    "name": name,
                    "date": observed_on,
                    "width": None,
                    "height": None,
                    "pixels": None,
                    "level": water_level,
                }
            )
            continue
        if len(decoded) > 1:
            notes.append(
                f"{name}: 한 파일에서 {len(decoded)}개 프레임을 해석했습니다. 동일한 행 날짜를 공유합니다."
            )
        for decoded_index, mask in enumerate(decoded, start=1):
            raw_points.append(
                {
                    "name": name if len(decoded) == 1 else f"{name}#{decoded_index}",
                    "date": observed_on,
                    "width": mask.width or None,
                    "height": mask.height or None,
                    "pixels": mask.water_pixels,
                    "level": water_level,
                }
            )

    raw_points.sort(
        key=lambda item: (
            item["date"] is None,
            item["date"] or dt.date.max,
            item["name"],
        )
    )
    dates = [item["date"] for item in raw_points]
    (
        _dated_count,
        unique_date_count,
        duplicate_date_count,
        start_date,
        end_date,
        period_days,
        interval_min,
        interval_median,
        interval_max,
    ) = _date_metrics(dates)
    invalid_date_count = len(dates) - _dated_count

    pixels = [item["pixels"] for item in raw_points if item["pixels"] is not None]
    first_pixels = pixels[0] if pixels else None
    last_pixels = pixels[-1] if pixels else None
    overall_change = _percentage_change(first_pixels, last_pixels)
    previous_pixels: int | None = None
    points: list[InputEDAPoint] = []
    for index, item in enumerate(raw_points, start=1):
        current_pixels = item["pixels"]
        change = _percentage_change(previous_pixels, current_pixels)
        if current_pixels is not None:
            previous_pixels = current_pixels
        area = (
            current_pixels * resolved_pixel_area / 1_000_000.0
            if current_pixels is not None and resolved_pixel_area is not None
            else None
        )
        points.append(
            InputEDAPoint(
                frame_index=index,
                name=item["name"],
                observed_on=item["date"],
                width=item["width"],
                height=item["height"],
                water_pixels=current_pixels,
                water_area_km2=area,
                change_pct=change,
                water_level_m=item["level"],
            )
        )

    resolutions = tuple(
        sorted(
            {
                (point.width, point.height)
                for point in points
                if point.width is not None and point.height is not None
            }
        )
    )
    if not resolutions:
        resolution_label = "확인 불가"
    elif len(resolutions) == 1:
        width, height = resolutions[0]
        resolution_label = f"{width:,} × {height:,} px"
    else:
        resolution_label = f"혼합 {len(resolutions)}종"

    if duplicate_date_count:
        notes.append(
            f"중복 날짜 {duplicate_date_count}건이 있습니다. 관측 간격은 고유 날짜 기준입니다."
        )
    if invalid_date_count:
        notes.append(f"날짜를 해석하지 못한 프레임이 {invalid_date_count}건 있습니다.")
    if len(resolutions) > 1:
        notes.append("프레임 해상도가 서로 달라 모델 입력 전 격자 정렬이 필요합니다.")
    if not area_known:
        notes.append(
            "픽셀 면적이 확인되지 않아 km²로 환산하지 않고 수체 픽셀만 표시합니다."
        )

    interval_text = _interval_text(interval_min, interval_median, interval_max)
    caption = (
        f"관측 {len(points)}프레임 · 고유 날짜 {unique_date_count}일 · "
        f"기간 {_period_text(start_date, end_date, period_days)} · 간격 {interval_text} · "
        f"해상도 {resolution_label}"
    )
    if not points:
        interpretation = "관측 마스크가 없어 입력 변화 추이를 계산할 수 없습니다."
    elif not pixels:
        interpretation = "프레임은 등록됐지만 마스크 픽셀을 읽지 못해 수체 변화를 계산하지 않았습니다."
    elif overall_change is None:
        interpretation = (
            "첫 프레임의 수체 픽셀이 0이어서 시작 대비 변화율을 계산하지 않았습니다."
        )
    else:
        direction = (
            "증가"
            if overall_change > 0.05
            else "감소"
            if overall_change < -0.05
            else "유지"
        )
        interpretation = (
            f"첫 관측 대비 마지막 관측의 수체 픽셀은 {abs(overall_change):.2f}% {direction}했습니다. "
            "이는 입력 변화 요약이며 원인이나 예측 정확도를 뜻하지 않습니다."
        )

    return InputEDASummary(
        frame_count=len(points),
        unique_date_count=unique_date_count,
        duplicate_date_count=duplicate_date_count,
        invalid_date_count=invalid_date_count,
        start_date=start_date,
        end_date=end_date,
        period_days=period_days,
        interval_min_days=interval_min,
        interval_median_days=interval_median,
        interval_max_days=interval_max,
        resolutions=resolutions,
        resolution_label=resolution_label,
        decoded_frame_count=len(pixels),
        water_pixel_first=first_pixels,
        water_pixel_last=last_pixels,
        water_pixel_min=min(pixels) if pixels else None,
        water_pixel_max=max(pixels) if pixels else None,
        water_pixel_change_pct=overall_change,
        pixel_area_known=area_known,
        pixel_area_m2=resolved_pixel_area,
        water_area_first_km2=(
            first_pixels * resolved_pixel_area / 1_000_000.0
            if first_pixels is not None and resolved_pixel_area is not None
            else None
        ),
        water_area_last_km2=(
            last_pixels * resolved_pixel_area / 1_000_000.0
            if last_pixels is not None and resolved_pixel_area is not None
            else None
        ),
        points=tuple(points),
        caption=caption,
        interpretation=interpretation,
        notes=tuple(dict.fromkeys(notes)),
    )


def _weather_kind(row: Mapping[str, Any]) -> str:
    value = str(row.get("kind") or row.get("type") or "").strip().lower()
    if value in {"observed", "observation", "actual"}:
        return "observed"
    if value in {"scenario", "forecast", "future"}:
        return "scenario"
    return "unknown"


def summarize_weather_eda(
    weather_rows: Sequence[Mapping[str, Any]] | None,
) -> WeatherEDASummary:
    """Summarize observed/scenario weather rows and missingness."""

    rows = tuple(weather_rows or ())
    points: list[WeatherEDAPoint] = []
    for index, row in enumerate(rows, start=1):
        observed_on = _parse_date(
            _first_value(row, "date", "timestamp", "observed_on", "target_date")
        )
        precipitation = _finite_float(
            _first_value(
                row,
                "precipitation_mm",
                "rainfall_mm",
                "total_precipitation_mm",
                "sum_rn",
            ),
            non_negative=True,
        )
        temperature = _finite_float(
            _first_value(row, "temperature_c", "avg_temperature_c", "temp_c")
        )
        if temperature is None:
            minimum = _finite_float(
                _first_value(row, "min_temperature_c", "minimum_temperature_c")
            )
            maximum = _finite_float(
                _first_value(row, "max_temperature_c", "maximum_temperature_c")
            )
            if minimum is not None and maximum is not None:
                temperature = (minimum + maximum) / 2.0
        humidity = _finite_float(
            _first_value(
                row,
                "humidity_pct",
                "avg_humidity_pct",
                "avg_humidity_percent",
                "humidity",
            )
        )
        points.append(
            WeatherEDAPoint(
                row_index=index,
                observed_on=observed_on,
                kind=_weather_kind(row),
                precipitation_mm=precipitation,
                temperature_c=temperature,
                humidity_pct=humidity,
            )
        )

    points.sort(
        key=lambda point: (
            point.observed_on is None,
            point.observed_on or dt.date.max,
            point.row_index,
        )
    )
    dates = [point.observed_on for point in points]
    (
        dated_count,
        unique_date_count,
        duplicate_date_count,
        start_date,
        end_date,
        period_days,
        _interval_min,
        _interval_median,
        _interval_max,
    ) = _date_metrics(dates)
    precipitation_points = [
        point for point in points if point.precipitation_mm is not None
    ]
    precipitation_values = [point.precipitation_mm for point in precipitation_points]
    precipitation_values = [
        value for value in precipitation_values if value is not None
    ]
    maximum_point = (
        max(precipitation_points, key=lambda point: point.precipitation_mm or 0.0)
        if precipitation_points
        else None
    )
    missing_precipitation = len(points) - len(precipitation_values)
    canonical_present = sum(
        value is not None
        for point in points
        for value in (
            point.precipitation_mm,
            point.temperature_c,
            point.humidity_pct,
        )
    )
    feature_slots = len(points) * 3
    feature_missing_rate = (
        (feature_slots - canonical_present) / feature_slots * 100.0
        if feature_slots
        else 0.0
    )
    kind_counts = Counter(point.kind for point in points)
    notes: list[str] = []
    invalid_date_count = len(points) - dated_count
    if duplicate_date_count:
        notes.append(
            f"같은 날짜의 기상 행이 {duplicate_date_count}건 중복되어 있습니다."
        )
    if invalid_date_count:
        notes.append(f"날짜를 해석하지 못한 기상 행이 {invalid_date_count}건 있습니다.")
    if kind_counts["unknown"]:
        notes.append(
            f"observed/scenario 구분이 없는 행이 {kind_counts['unknown']}건 있습니다."
        )
    if missing_precipitation:
        notes.append(f"강수량 결측 행이 {missing_precipitation}건 있습니다.")

    caption = (
        f"기상 {len(points)}행 · observed {kind_counts['observed']} · "
        f"scenario {kind_counts['scenario']} · 기간 {_period_text(start_date, end_date, period_days)} · "
        f"핵심 3개 특성 결측률 {feature_missing_rate:.1f}%"
    )
    if not points:
        interpretation = "기상 행이 없어 강수·기온·습도 분포를 계산할 수 없습니다."
    elif not precipitation_values:
        interpretation = "강수량이 모두 결측이므로 강수 효과를 해석하지 않았습니다."
    else:
        max_date_text = (
            maximum_point.observed_on.isoformat()
            if maximum_point and maximum_point.observed_on
            else "날짜 미상"
        )
        interpretation = (
            f"분석 구간 누적 강수는 {sum(precipitation_values):.1f} mm, "
            f"일 최대는 {max(precipitation_values):.1f} mm({max_date_text})입니다."
        )
        if kind_counts["scenario"]:
            interpretation += (
                " scenario 행은 관측이 아닌 가정값이므로 결과 설명에서 분리해야 합니다."
            )

    return WeatherEDASummary(
        row_count=len(points),
        dated_row_count=dated_count,
        unique_date_count=unique_date_count,
        duplicate_date_count=duplicate_date_count,
        invalid_date_count=invalid_date_count,
        observed_count=kind_counts["observed"],
        scenario_count=kind_counts["scenario"],
        unknown_kind_count=kind_counts["unknown"],
        start_date=start_date,
        end_date=end_date,
        period_days=period_days,
        precipitation_count=len(precipitation_values),
        precipitation_sum_mm=sum(precipitation_values)
        if precipitation_values
        else None,
        precipitation_mean_mm=(
            statistics.fmean(precipitation_values) if precipitation_values else None
        ),
        precipitation_max_mm=max(precipitation_values)
        if precipitation_values
        else None,
        precipitation_max_date=maximum_point.observed_on if maximum_point else None,
        precipitation_missing_rate_pct=(
            missing_precipitation / len(points) * 100.0 if points else 0.0
        ),
        feature_missing_rate_pct=feature_missing_rate,
        temperature_count=sum(point.temperature_c is not None for point in points),
        humidity_count=sum(point.humidity_pct is not None for point in points),
        points=tuple(points),
        caption=caption,
        interpretation=interpretation,
        notes=tuple(notes),
    )


def _risk_value(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    parsed = str(value or "unknown").strip().lower()
    return parsed if parsed in _RISK_LABELS else "unknown"


def summarize_result_eda(
    result_steps: Sequence[Mapping[str, Any]] | None,
    *,
    sample: Any = None,
    pixel_area_known: bool | None = None,
) -> ResultEDASummary:
    """Summarize forecast horizons, physical/pixel area, levels, and risk."""

    rows = tuple(result_steps or ())
    if pixel_area_known is None:
        pixel_area_known = bool(_value(sample, "pixel_area_known", False))
    points: list[ResultEDAPoint] = []
    for index, row in enumerate(rows, start=1):
        horizon = _finite_int(
            _first_value(row, "horizon", "step", "frame"), non_negative=True
        )
        target_date = _parse_date(
            _first_value(row, "target_date", "date", "forecast_date")
        )
        points.append(
            ResultEDAPoint(
                row_index=index,
                horizon=horizon,
                target_date=target_date,
                water_area_pixels=_finite_int(
                    _first_value(row, "water_area_pixels", "area_pixels"),
                    non_negative=True,
                ),
                water_area_km2=_finite_float(
                    _first_value(row, "water_area_km2", "area_km2"),
                    non_negative=True,
                ),
                area_change_pct=_finite_float(
                    _first_value(row, "area_change_pct", "change_pct")
                ),
                water_level_m=_finite_float(row.get("water_level_m")),
                water_level_change_m=_finite_float(row.get("water_level_change_m")),
                risk=_risk_value(_first_value(row, "risk", "risk_level")),
            )
        )
    points.sort(
        key=lambda point: (
            point.horizon is None,
            point.horizon if point.horizon is not None else point.row_index,
            point.row_index,
        )
    )
    horizons = [point.horizon for point in points if point.horizon is not None]
    dates = [point.target_date for point in points]
    (
        _dated_count,
        _unique_dates,
        _duplicate_dates,
        start_date,
        end_date,
        period_days,
        _interval_min,
        _interval_median,
        _interval_max,
    ) = _date_metrics(dates)

    use_physical_area = bool(
        pixel_area_known and any(point.water_area_km2 is not None for point in points)
    )
    if use_physical_area:
        area_unit = "km²"
        area_values = [
            point.water_area_km2 for point in points if point.water_area_km2 is not None
        ]
    else:
        area_unit = "px"
        area_values = [
            float(point.water_area_pixels)
            for point in points
            if point.water_area_pixels is not None
        ]
    levels = [
        point.water_level_m for point in points if point.water_level_m is not None
    ]
    first_area = area_values[0] if area_values else None
    last_area = area_values[-1] if area_values else None
    first_level = levels[0] if levels else None
    last_level = levels[-1] if levels else None
    risk_counter = Counter(point.risk for point in points)
    risk_counts = tuple(
        (risk, risk_counter[risk]) for risk in _RISK_ORDER if risk_counter[risk]
    )
    notes: list[str] = []
    if not use_physical_area:
        notes.append(
            "검증된 픽셀 면적이 없어 결과 면적은 km² 대신 수체 픽셀로 요약합니다."
        )
    if points and not levels:
        notes.append("수위 산출값이 없어 면적 변화만 요약합니다.")
    if any(point.target_date is None for point in points):
        notes.append(
            "목표 날짜가 없는 결과 단계가 포함되어 기간 요약에서 제외했습니다."
        )

    latest_relative_change = next(
        (
            point.area_change_pct
            for point in reversed(points)
            if point.area_change_pct is not None
        ),
        None,
    )
    caption = (
        f"예측 {len(points)}단계 · horizon {_horizon_text(horizons)} · "
        f"기간 {_period_text(start_date, end_date, period_days)} · "
        f"수위 산출 {len(levels)}/{len(points)}"
    )
    if not points:
        interpretation = "예측 단계가 없어 결과 추이를 계산할 수 없습니다."
    else:
        parts: list[str] = []
        if latest_relative_change is not None:
            direction = (
                "증가"
                if latest_relative_change > 0.05
                else "감소"
                if latest_relative_change < -0.05
                else "유지"
            )
            parts.append(
                f"최종 단계 수체 면적은 입력 기준 {abs(latest_relative_change):.2f}% {direction}"
            )
        elif first_area is not None and last_area is not None:
            net_pct = _percentage_change(first_area, last_area)
            if net_pct is not None:
                parts.append(f"예측 첫 단계 대비 최종 면적 변화 {net_pct:+.2f}%")
        if first_level is not None and last_level is not None:
            parts.append(f"예측 구간 수위 변화 {last_level - first_level:+.3f} m")
        if risk_counts:
            dominant_risk, dominant_count = max(risk_counts, key=lambda item: item[1])
            parts.append(f"최종 위험도 {_RISK_LABELS[points[-1].risk]}")
            parts.append(
                f"최빈 위험도 {_RISK_LABELS[dominant_risk]} {dominant_count}단계"
            )
        interpretation = " · ".join(parts) + (
            "." if parts else "요약 가능한 수치 결과가 없습니다."
        )

    return ResultEDASummary(
        step_count=len(points),
        unique_horizon_count=len(set(horizons)),
        horizon_min=min(horizons) if horizons else None,
        horizon_max=max(horizons) if horizons else None,
        start_date=start_date,
        end_date=end_date,
        period_days=period_days,
        area_unit=area_unit,
        area_value_first=first_area,
        area_value_last=last_area,
        area_value_min=min(area_values) if area_values else None,
        area_value_max=max(area_values) if area_values else None,
        area_net_change=(
            last_area - first_area
            if first_area is not None and last_area is not None
            else None
        ),
        area_net_change_pct=_percentage_change(first_area, last_area),
        latest_area_change_from_input_pct=latest_relative_change,
        water_level_count=len(levels),
        water_level_first_m=first_level,
        water_level_last_m=last_level,
        water_level_min_m=min(levels) if levels else None,
        water_level_max_m=max(levels) if levels else None,
        water_level_net_change_m=(
            last_level - first_level
            if first_level is not None and last_level is not None
            else None
        ),
        risk_counts=risk_counts,
        pixel_area_known=bool(pixel_area_known),
        points=tuple(points),
        caption=caption,
        interpretation=interpretation,
        notes=tuple(notes),
    )


def summarize_eda(
    *,
    sample: Any = None,
    input_rows: Sequence[Mapping[str, Any]] | None = None,
    weather_rows: Sequence[Mapping[str, Any]] | None = None,
    result_steps: Sequence[Mapping[str, Any]] | None = None,
) -> EDAOverview:
    """Build all EDA sections from one UI/session snapshot."""

    if weather_rows is None and sample is not None:
        payload = _value(sample, "weather_payload")
        weather_rows = (
            payload() if callable(payload) else _value(sample, "weather_rows", ())
        )
    return EDAOverview(
        input=summarize_input_eda(sample, input_rows),
        weather=summarize_weather_eda(weather_rows),
        result=summarize_result_eda(result_steps, sample=sample),
    )


def build_input_water_figure(summary: InputEDASummary) -> go.Figure | None:
    """Return the input water-change chart, or ``None`` without pixel data."""

    points = [point for point in summary.points if point.water_pixels is not None]
    if not points:
        return None
    x = [_point_label(point.observed_on, point.frame_index) for point in points]
    if summary.pixel_area_known:
        y = [point.water_area_km2 for point in points]
        title = "관측 수체 변화 · 검증된 픽셀 면적 적용"
        y_title = "수체 면적 (km²)"
        hover = "%{x}<br>수체 면적 %{y:,.6f} km²<extra></extra>"
    else:
        y = [point.water_pixels for point in points]
        title = "관측 수체 변화 · 픽셀 기준"
        y_title = "수체 픽셀 수"
        hover = "%{x}<br>수체 픽셀 %{y:,.0f} px<extra></extra>"
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            name=y_title,
            mode="lines+markers",
            line={"color": "#22d3ee", "width": 3},
            marker={"size": 8, "color": "#38bdf8"},
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.08)",
            hovertemplate=hover,
        )
    )
    figure.update_layout(**_DARK_LAYOUT, height=330, title={"text": title, "x": 0.01})
    figure.update_xaxes(title_text="관측 날짜 / 프레임", gridcolor="#1e293b")
    # 0부터 그리면 안 된다. 수체 면적은 1만 px 근처에서 수백 px씩 움직이는데
    # 축이 0에서 시작하면 그 변화가 선 두께에 묻혀 "값이 안 변한다"로 보인다.
    figure.update_yaxes(
        title_text=y_title, gridcolor="#1e293b", range=padded_range(y)
    )
    return figure



def padded_range(
    values: Sequence[float] | Sequence[int], *, pad_ratio: float = 0.18
) -> list[float] | None:
    """데이터 범위에 여백을 붙인 축 범위. 변화가 보이도록 0에서 시작하지 않는다.

    값이 모두 같으면 축이 한 점으로 붕괴하므로 그때만 값 주변으로 넓힌다.
    """

    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    low, high = min(numbers), max(numbers)
    span = high - low
    if span <= 0:
        pad = abs(high) * pad_ratio or 1.0
        return [high - pad, high + pad]
    pad = span * pad_ratio
    return [low - pad, high + pad]


def axis_range_for(figure, *, secondary_y: bool | None = None, pad_ratio: float = 0.18):
    """figure 안 trace 들의 y 값에서 축 범위를 만든다.

    면적·수위처럼 0에서 멀리 떨어진 값이 좁은 폭으로 움직이는 계열은 0부터
    그리면 변화가 보이지 않는다. 해당 축에 실린 trace 만 골라 범위를 잡는다.
    """

    values: list[float] = []
    for trace in figure.data:
        if secondary_y is not None:
            on_secondary = getattr(trace, "yaxis", "y") == "y2"
            if on_secondary != secondary_y:
                continue
        # numpy 배열에 ``or`` 를 쓰면 truth value 예외가 난다. None 검사만 한다.
        series = getattr(trace, "y", None)
        if series is None:
            continue
        for item in series:
            if item is None:
                continue
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
    return padded_range(values, pad_ratio=pad_ratio)

def build_weather_figure(summary: WeatherEDASummary) -> go.Figure | None:
    """Return a dark rainfall + temperature/humidity chart."""

    if not summary.points:
        return None
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for kind, label, color, pattern in (
        ("observed", "관측 강수", "#22d3ee", ""),
        ("scenario", "시나리오 강수", "#f97316", "/"),
        ("unknown", "구분 미상 강수", "#64748b", "x"),
    ):
        subset = [point for point in summary.points if point.kind == kind]
        if not subset:
            continue
        figure.add_trace(
            go.Bar(
                x=[
                    _point_label(point.observed_on, point.row_index) for point in subset
                ],
                y=[point.precipitation_mm for point in subset],
                name=label,
                marker={"color": color, "pattern": {"shape": pattern}},
                opacity=0.78,
                hovertemplate="%{x}<br>강수 %{y:.1f} mm<extra></extra>",
            ),
            secondary_y=False,
        )
    if any(point.temperature_c is not None for point in summary.points):
        figure.add_trace(
            go.Scatter(
                x=[
                    _point_label(point.observed_on, point.row_index)
                    for point in summary.points
                ],
                y=[point.temperature_c for point in summary.points],
                name="평균 기온",
                mode="lines+markers",
                line={"color": "#fb7185", "width": 2},
                hovertemplate="%{x}<br>기온 %{y:.1f} °C<extra></extra>",
            ),
            secondary_y=True,
        )
    if any(point.humidity_pct is not None for point in summary.points):
        figure.add_trace(
            go.Scatter(
                x=[
                    _point_label(point.observed_on, point.row_index)
                    for point in summary.points
                ],
                y=[point.humidity_pct for point in summary.points],
                name="평균 습도",
                mode="lines+markers",
                line={"color": "#a78bfa", "width": 2, "dash": "dot"},
                hovertemplate="%{x}<br>습도 %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
    figure.update_layout(
        **_DARK_LAYOUT,
        height=350,
        title={"text": "기상 외생변수 · 관측과 시나리오 분리", "x": 0.01},
        legend={"orientation": "h", "y": 1.13},
        barmode="group",
    )
    figure.update_xaxes(title_text="날짜", gridcolor="#1e293b")
    figure.update_yaxes(
        title_text="강수량 (mm)",
        gridcolor="#1e293b",
        rangemode="tozero",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="기온 (°C) / 습도 (%)", showgrid=False, secondary_y=True
    )
    return figure


def build_result_summary_figure(summary: ResultEDASummary) -> go.Figure | None:
    """Return an optional compact guide chart; result pages may omit it."""

    if not summary.points:
        return None
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    x = [
        _point_label(point.target_date, point.horizon or point.row_index)
        for point in summary.points
    ]
    if summary.area_unit == "km²":
        area_values = [point.water_area_km2 for point in summary.points]
        area_title = "수체 면적 (km²)"
    else:
        area_values = [point.water_area_pixels for point in summary.points]
        area_title = "수체 픽셀 수"
    if any(value is not None for value in area_values):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=area_values,
                name=area_title,
                mode="lines+markers",
                line={"color": "#22d3ee", "width": 3},
                marker={
                    "size": 9,
                    "color": [_RISK_COLORS[point.risk] for point in summary.points],
                },
            ),
            secondary_y=False,
        )
    if summary.water_level_count:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[point.water_level_m for point in summary.points],
                name="수위 (m)",
                mode="lines+markers",
                line={"color": "#fbbf24", "width": 2, "dash": "dot"},
            ),
            secondary_y=True,
        )
    figure.update_layout(
        **_DARK_LAYOUT,
        height=320,
        title={"text": "결과 요약 · 위험 색상은 규칙 기반", "x": 0.01},
        legend={"orientation": "h", "y": 1.13},
    )
    figure.update_xaxes(title_text="목표 날짜 / horizon", gridcolor="#1e293b")
    figure.update_yaxes(title_text=area_title, gridcolor="#1e293b", secondary_y=False)
    figure.update_yaxes(title_text="수위 (m)", showgrid=False, secondary_y=True)
    return figure


def render_input_eda(
    sample: Any = None,
    input_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    st_module: Any = None,
) -> InputEDASummary:
    """Render and return the input EDA section."""

    st = _streamlit(st_module)
    summary = summarize_input_eda(sample, input_rows)
    st.markdown("#### 입력 데이터 요약")
    metrics = st.columns(5)
    metrics[0].metric("관측 프레임", f"{summary.frame_count:,}")
    metrics[1].metric("고유 날짜", f"{summary.unique_date_count:,}")
    metrics[2].metric(
        "관측 기간",
        f"{summary.period_days}일" if summary.period_days is not None else "-",
    )
    metrics[3].metric(
        "관측 간격",
        f"{summary.interval_median_days:g}일"
        if summary.interval_median_days is not None
        else "-",
    )
    metrics[4].metric("해상도", summary.resolution_label)
    st.caption(summary.caption)
    figure = build_input_water_figure(summary)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    else:
        st.info("표시할 수체 픽셀 통계가 없습니다.")
    st.info(summary.interpretation)
    for note in summary.notes:
        st.caption(f"• {note}")
    return summary


def render_weather_eda(
    weather_rows: Sequence[Mapping[str, Any]] | None,
    *,
    st_module: Any = None,
) -> WeatherEDASummary:
    """Render and return the weather EDA section."""

    st = _streamlit(st_module)
    summary = summarize_weather_eda(weather_rows)
    st.markdown("#### 기상 데이터 요약")
    metrics = st.columns(5)
    metrics[0].metric("기상 행", f"{summary.row_count:,}")
    metrics[1].metric(
        "관측 / 시나리오", f"{summary.observed_count} / {summary.scenario_count}"
    )
    metrics[2].metric(
        "누적 강수",
        f"{summary.precipitation_sum_mm:,.1f} mm"
        if summary.precipitation_sum_mm is not None
        else "-",
    )
    metrics[3].metric(
        "일 최대 강수",
        f"{summary.precipitation_max_mm:,.1f} mm"
        if summary.precipitation_max_mm is not None
        else "-",
    )
    metrics[4].metric("핵심 결측률", f"{summary.feature_missing_rate_pct:.1f}%")
    st.caption(summary.caption)
    figure = build_weather_figure(summary)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    else:
        st.info("표시할 기상 데이터가 없습니다.")
    st.info(summary.interpretation)
    for note in summary.notes:
        st.caption(f"• {note}")
    return summary


def render_result_eda(
    result_steps: Sequence[Mapping[str, Any]] | None,
    *,
    sample: Any = None,
    pixel_area_known: bool | None = None,
    st_module: Any = None,
    show_chart: bool = False,
) -> ResultEDASummary:
    """Render a compact result summary.

    ``show_chart`` defaults to ``False`` because the main result phase already
    owns a detailed area/level plot.  API guides and presentations can opt in.
    """

    st = _streamlit(st_module)
    summary = summarize_result_eda(
        result_steps,
        sample=sample,
        pixel_area_known=pixel_area_known,
    )
    st.markdown("#### 예측 결과 요약")
    metrics = st.columns(5)
    metrics[0].metric("예측 horizon", f"{summary.step_count:,}")
    metrics[1].metric(
        "최종 수체",
        _format_area(summary.area_value_last, summary.area_unit),
        delta=(
            f"{summary.latest_area_change_from_input_pct:+.2f}%"
            if summary.latest_area_change_from_input_pct is not None
            else None
        ),
    )
    metrics[2].metric(
        "수위 변화",
        f"{summary.water_level_net_change_m:+.3f} m"
        if summary.water_level_net_change_m is not None
        else "미산출",
    )
    metrics[3].metric("수위 산출", f"{summary.water_level_count}/{summary.step_count}")
    dominant = max(
        summary.risk_counts, key=lambda item: item[1], default=("unknown", 0)
    )
    metrics[4].metric("최빈 위험도", _RISK_LABELS[dominant[0]])
    st.caption(summary.caption)
    if show_chart:
        figure = build_result_summary_figure(summary)
        if figure is not None:
            st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.info(summary.interpretation)
    if summary.risk_counts:
        risk_text = " · ".join(
            f"{_RISK_LABELS[risk]} {count}" for risk, count in summary.risk_counts
        )
        st.caption(f"위험도 분포: {risk_text}")
    for note in summary.notes:
        st.caption(f"• {note}")
    return summary


def render_eda_overview(
    *,
    sample: Any = None,
    input_rows: Sequence[Mapping[str, Any]] | None = None,
    weather_rows: Sequence[Mapping[str, Any]] | None = None,
    result_steps: Sequence[Mapping[str, Any]] | None = None,
    st_module: Any = None,
    show_result_chart: bool = False,
) -> EDAOverview:
    """Render all available sections and return the corresponding summaries."""

    st = _streamlit(st_module)
    if weather_rows is None and sample is not None:
        payload = _value(sample, "weather_payload")
        weather_rows = (
            payload() if callable(payload) else _value(sample, "weather_rows", ())
        )
    input_summary = render_input_eda(sample, input_rows, st_module=st)
    weather_summary = render_weather_eda(weather_rows, st_module=st)
    result_summary = render_result_eda(
        result_steps,
        sample=sample,
        st_module=st,
        show_chart=show_result_chart,
    )
    return EDAOverview(
        input=input_summary, weather=weather_summary, result=result_summary
    )


def _streamlit(module: Any) -> Any:
    if module is not None:
        return module
    import streamlit as st

    return st


def _period_text(
    start_date: dt.date | None, end_date: dt.date | None, period_days: int | None
) -> str:
    if start_date is None or end_date is None or period_days is None:
        return "확인 불가"
    if start_date == end_date:
        return f"{start_date.isoformat()} (당일)"
    return f"{start_date.isoformat()}~{end_date.isoformat()} ({period_days}일 경과)"


def _interval_text(
    minimum: int | None, median: float | None, maximum: int | None
) -> str:
    if minimum is None or median is None or maximum is None:
        return "확인 불가"
    return f"min/median/max {minimum}/{median:g}/{maximum}일"


def _horizon_text(horizons: Sequence[int]) -> str:
    if not horizons:
        return "확인 불가"
    if min(horizons) == max(horizons):
        return str(min(horizons))
    return f"{min(horizons)}~{max(horizons)}"


def _point_label(value: dt.date | None, index: int) -> str:
    return value.isoformat() if value is not None else f"#{index}"


def _format_area(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "km²":
        return f"{value:,.6f} km²"
    return f"{value:,.0f} px"


__all__ = [
    "EDAOverview",
    "InputEDAPoint",
    "InputEDASummary",
    "ResultEDAPoint",
    "ResultEDASummary",
    "WeatherEDAPoint",
    "WeatherEDASummary",
    "build_input_water_figure",
    "build_result_summary_figure",
    "build_weather_figure",
    "render_eda_overview",
    "render_input_eda",
    "render_result_eda",
    "render_weather_eda",
    "summarize_eda",
    "summarize_input_eda",
    "summarize_result_eda",
    "summarize_weather_eda",
]
