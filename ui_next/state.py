"""Pure validation and session-state transition helpers.

This module intentionally has no Streamlit dependency.  The UI wraps these
functions around ``st.session_state`` and the tests can exercise the cascading
invalidation rules without starting a browser.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


FINGERPRINTS_KEY = "_config_fingerprints"

# A changed upstream configuration removes only artifacts derived from it.
DOMAIN_INVALIDATIONS: dict[str, tuple[str, ...]] = {
    "connection": (
        "connection_status",
        "models",
        "models_attempted",
        "models_error",
        "stations",
        "weather_capabilities",
        "weather_capabilities_attempted",
        "weather_capabilities_error",
        "weather_query_result",
        "weather_query_notice",
        "weather_data",
        "weather_confirmed",
        "prediction_result",
        "prediction_context",
        "artifact_cache",
        "bundle_cache",
        "openapi_spec",
        "openapi_error",
    ),
    "input": (
        "input_bundle",
        "weather_query_result",
        "weather_query_notice",
        "weather_data",
        "weather_confirmed",
        "prediction_result",
        "prediction_context",
        "artifact_cache",
        "bundle_cache",
    ),
    "weather_query": (
        "weather_query_result",
        "weather_query_notice",
        "weather_data",
        "weather_confirmed",
        "prediction_result",
        "prediction_context",
        "artifact_cache",
        "bundle_cache",
    ),
    "weather_data": (
        "prediction_result",
        "prediction_context",
        "artifact_cache",
        "bundle_cache",
    ),
    "model": (
        "prediction_result",
        "prediction_context",
        "artifact_cache",
        "bundle_cache",
    ),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}")


def fingerprint(value: Any) -> str:
    """Return a deterministic digest for JSON-like configuration values."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_configuration(
    state: Mapping[str, Any], domain: str, value: Any
) -> tuple[dict[str, Any], tuple[str, ...], bool]:
    """Apply a config fingerprint and remove stale downstream artifacts.

    The first observation establishes the baseline and does not invalidate
    anything.  The returned dictionary is a shallow copy; the input mapping is
    never mutated.
    """

    if domain not in DOMAIN_INVALIDATIONS:
        raise KeyError(f"Unknown configuration domain: {domain}")

    updated = dict(state)
    fingerprints = dict(updated.get(FINGERPRINTS_KEY, {}))
    next_fingerprint = fingerprint(value)
    previous = fingerprints.get(domain)
    changed = previous is not None and previous != next_fingerprint
    fingerprints[domain] = next_fingerprint
    updated[FINGERPRINTS_KEY] = fingerprints

    removed: list[str] = []
    if changed:
        for key in DOMAIN_INVALIDATIONS[domain]:
            if key in updated:
                updated.pop(key)
                removed.append(key)
    return updated, tuple(removed), changed


def validate_input_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate dated mask observations and return user-facing errors."""

    errors: list[str] = []
    if len(rows) < 2:
        errors.append("시계열 입력에는 서로 다른 날짜의 마스크가 최소 2개 필요합니다.")

    parsed_dates: list[dt.date] = []
    supported = {".npy", ".tif", ".tiff", ".png"}
    for index, row in enumerate(rows, start=1):
        name = str(row.get("name", "")).strip()
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix not in supported:
            errors.append(f"{index}번 파일({name or '이름 없음'}): NPY/TIF/TIFF/PNG만 지원합니다.")

        raw_date = row.get("date")
        try:
            parsed = raw_date if isinstance(raw_date, dt.date) else dt.date.fromisoformat(str(raw_date))
            parsed_dates.append(parsed)
        except (TypeError, ValueError):
            errors.append(f"{index}번 파일: 관측 날짜가 올바르지 않습니다.")

        level = row.get("water_level_m")
        if level not in (None, ""):
            try:
                if not math.isfinite(float(level)):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{index}번 파일: 수위는 유한한 숫자여야 합니다.")

    if len(parsed_dates) != len(set(parsed_dates)):
        errors.append("각 마스크에는 서로 다른 관측 날짜를 지정해야 합니다.")
    return errors


WEATHER_COLUMNS = (
    "timestamp",
    "precipitation_mm",
    "temperature_c",
    "min_temperature_c",
    "max_temperature_c",
    "humidity_pct",
    "wind_speed_mps",
    "avg_local_pressure_hpa",
    "kind",
    "source",
)


def normalize_weather_rows(
    rows: Sequence[Mapping[str, Any]], today: dt.date | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize the editable weather table and enforce observation semantics."""

    today = today or dt.date.today()
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, row in enumerate(rows, start=1):
        raw_timestamp = row.get("timestamp") or row.get("date") or row.get("observed_at")
        if raw_timestamp in (None, ""):
            # Completely empty rows from st.data_editor are ignored.
            if not any(row.get(key) not in (None, "", "nan") for key in WEATHER_COLUMNS[1:]):
                continue
            errors.append(f"기상 {index}행: 날짜가 필요합니다.")
            continue
        try:
            if isinstance(raw_timestamp, dt.datetime):
                day = raw_timestamp.date()
            elif isinstance(raw_timestamp, dt.date):
                day = raw_timestamp
            else:
                day = dt.date.fromisoformat(str(raw_timestamp)[:10])
        except (TypeError, ValueError):
            errors.append(f"기상 {index}행: 날짜 형식은 YYYY-MM-DD여야 합니다.")
            continue

        kind = str(row.get("kind") or "observed").strip().lower()
        if kind not in {"observed", "scenario"}:
            errors.append(f"기상 {index}행: kind는 observed 또는 scenario여야 합니다.")
        if day >= today and kind != "scenario":
            errors.append(
                f"기상 {index}행({day.isoformat()}): 오늘/미래 값은 ASOS 관측이 아닌 scenario로 표시해야 합니다."
            )

        values: dict[str, float | None] = {}
        aliases = {
            "precipitation_mm": ("precipitation_mm", "rainfall_mm", "total_precipitation_mm"),
            "temperature_c": ("temperature_c", "temp_c", "avg_temperature_c"),
            "min_temperature_c": ("min_temperature_c", "minimum_temperature_c", "min_temp_c"),
            "max_temperature_c": ("max_temperature_c", "maximum_temperature_c", "max_temp_c"),
            "humidity_pct": (
                "humidity_pct",
                "humidity",
                "avg_humidity_pct",
                "avg_humidity_percent",
            ),
            "wind_speed_mps": (
                "wind_speed_mps",
                "avg_wind_speed_mps",
                "avg_wind_speed_m_s",
            ),
            "avg_local_pressure_hpa": ("avg_local_pressure_hpa", "local_pressure_hpa"),
        }
        for output_key, candidates in aliases.items():
            raw = next((row.get(key) for key in candidates if row.get(key) not in (None, "")), None)
            if raw is None or str(raw).lower() == "nan":
                values[output_key] = None
                continue
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
                values[output_key] = value
            except (TypeError, ValueError):
                errors.append(f"기상 {index}행: {output_key} 값이 숫자가 아닙니다.")
                values[output_key] = None

        humidity = values.get("humidity_pct")
        if humidity is not None and not 0 <= humidity <= 100:
            errors.append(f"기상 {index}행: 습도는 0~100% 범위여야 합니다.")
        rainfall = values.get("precipitation_mm")
        if rainfall is not None and rainfall < 0:
            errors.append(f"기상 {index}행: 강수량은 음수일 수 없습니다.")

        # Preserve provider-specific KMA columns for adapters that consume more
        # than the standard rain/temperature/humidity features.
        extra = {str(key): _json_compatible(value) for key, value in row.items()}
        normalized.append(
            {
                **extra,
                "timestamp": day.isoformat(),
                **values,
                "kind": kind,
                "source": str(row.get("source") or ("user_scenario" if kind == "scenario" else "unknown")),
            }
        )

    normalized.sort(key=lambda item: item["timestamp"])
    if not normalized:
        errors.append("예측에 전달할 기상 행이 하나 이상 필요합니다.")
    return normalized, errors


def _json_compatible(value: Any) -> Any:
    """Convert common dataframe/provider scalars while retaining extra fields."""

    if value is None:
        return None
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def extract_prediction_steps(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read frame/step results from common synchronous API response shapes."""

    candidates: list[Any] = [payload.get("steps"), payload.get("frames")]
    output = payload.get("output")
    if isinstance(output, Mapping):
        candidates.extend([output.get("steps"), output.get("frames")])
    result = payload.get("result")
    if isinstance(result, Mapping):
        candidates.extend([result.get("steps"), result.get("frames")])
    for value in candidates:
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def openapi_operation_rows(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    """Flatten an OpenAPI paths object into a small endpoint table."""

    rows: list[dict[str, str]] = []
    methods = {"get", "post", "put", "patch", "delete"}
    for path, operations in spec.get("paths", {}).items():
        if not isinstance(operations, Mapping):
            continue
        for method, operation in operations.items():
            if method.lower() not in methods or not isinstance(operation, Mapping):
                continue
            rows.append(
                {
                    "method": method.upper(),
                    "path": str(path),
                    "summary": str(operation.get("summary") or operation.get("operationId") or "-"),
                }
            )
    return sorted(rows, key=lambda row: (row["path"], row["method"]))
