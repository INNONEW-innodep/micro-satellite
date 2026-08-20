"""Weather plans for quick-start samples.

Synthetic scenarios keep using their deterministic rows.  The materialized NAS
Busan sample instead requests KMA ASOS station 159 and replays the *historical*
observations after the final image date.  Those target-period rows are model
forcing for a retrospective demonstration; they are measured observations, not
an operational weather forecast.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # package imports in tests; direct imports when Streamlit executes app.py
    from .samples import NAS_BUSAN_SAMPLE_ID, NAS_ICEYE_SAMPLE_ID, SampleDataset
except ImportError:  # pragma: no cover - exercised by the Streamlit entry point
    from samples import NAS_BUSAN_SAMPLE_ID, NAS_ICEYE_SAMPLE_ID, SampleDataset


KMA_BUSAN_STATION_ID = "159"
KMA_REPLAY_SAMPLE_IDS = frozenset((NAS_BUSAN_SAMPLE_ID, NAS_ICEYE_SAMPLE_ID))


@dataclass(frozen=True, slots=True)
class SampleWeatherPlan:
    """Resolved rows and provenance for one quick-start execution."""

    rows: tuple[dict[str, Any], ...]
    source: str
    note_ko: str
    missing_dates: tuple[str, ...] = ()
    historical_replay: bool = False


ObservationLoader = Callable[..., Mapping[str, Any]]


def resolve_sample_weather(
    sample: SampleDataset,
    horizon_days: int,
    *,
    observation_loader: ObservationLoader | None = None,
) -> SampleWeatherPlan:
    """Return backend-ready weather rows for one catalog sample.

    ``observation_loader`` has the same keyword contract as
    :meth:`ui_next.api_client.APIClient.weather_observations`.  It is injected
    so this module stays testable without network access or a KMA key.
    """

    forecast_dates = sample.forecast_dates(horizon_days)
    if sample.sample_id not in KMA_REPLAY_SAMPLE_IDS:
        rows = tuple(sample.forecast_weather_payload(horizon_days))
        return SampleWeatherPlan(
            rows=rows,
            source="built_in_demo" if rows else "none",
            note_ko=(
                f"저장소 예시값을 변환한 {horizon_days}일 시나리오입니다. "
                "기상청 실관측이나 예보가 아닙니다."
                if rows
                else "이 샘플에는 연결된 기상 행이 없습니다."
            ),
        )

    if observation_loader is None:
        return SampleWeatherPlan(
            rows=(),
            source="none",
            note_ko=(
                "부산 NAS 시연은 기상청 ASOS 역사 재현을 사용하도록 설정되어 있지만 "
                "이번 실행에서는 관측 조회기를 사용할 수 없습니다."
            ),
            missing_dates=tuple(
                value.isoformat() for value in (*sample.source_dates, *forecast_dates)
            ),
            historical_replay=True,
        )

    response = observation_loader(
        source="asos",
        station_id=KMA_BUSAN_STATION_ID,
        start_date=sample.source_dates[0].isoformat(),
        end_date=forecast_dates[-1].isoformat(),
    )
    return build_kma_historical_replay(
        response,
        source_dates=sample.source_dates,
        target_dates=forecast_dates,
    )


def build_kma_historical_replay(
    response: Mapping[str, Any],
    *,
    source_dates: Sequence[dt.date],
    target_dates: Sequence[dt.date],
) -> SampleWeatherPlan:
    """Select image dates and target dates from one KMA ASOS response."""

    raw_rows = response.get("rows") or response.get("records") or response.get("data")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raw_rows = ()

    by_date: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        date_value = _date_string(raw.get("date") or raw.get("timestamp"))
        if date_value is not None:
            by_date[date_value] = raw

    source_set = {value.isoformat() for value in source_dates}
    requested_dates = tuple(
        value.isoformat() for value in (*tuple(source_dates), *tuple(target_dates))
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    station_id = str(response.get("station_id") or KMA_BUSAN_STATION_ID)
    station_name = str(response.get("station_name") or "부산")
    for date_value in requested_dates:
        raw = by_date.get(date_value)
        if raw is None:
            missing.append(date_value)
            continue
        row = dict(raw)
        provider_source = str(row.get("source") or response.get("source") or "asos")
        row.update(
            {
                "date": date_value,
                "timestamp": date_value,
                "kind": "observed" if date_value in source_set else "scenario",
                "station_id": station_id,
                "station_name": station_name,
                "source": "kma_asos_historical_replay",
                "provider_source": provider_source,
                "is_measured": True,
                "is_forecast": False,
                "historical_replay": True,
                "forcing_role": (
                    "image_date_context" if date_value in source_set else "target_period_forcing"
                ),
            }
        )
        rows.append(row)

    target_start = target_dates[0].isoformat() if target_dates else "-"
    target_end = target_dates[-1].isoformat() if target_dates else "-"
    return SampleWeatherPlan(
        rows=tuple(rows),
        source="kma_asos_historical_replay",
        note_ko=(
            f"부산 ASOS 159의 과거 실관측을 {target_start}~{target_end} 구간에 "
            "역사 재현용 forcing으로 연결했습니다. 실제 관측값이지만 미래예보가 "
            "아니며, 미래 수체 정답이 없어 모델 성능 검증에는 사용할 수 없습니다."
        ),
        missing_dates=tuple(missing),
        historical_replay=True,
    )


def _date_string(value: Any) -> str | None:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if value is None:
        return None
    text = str(value).strip()
    try:
        return dt.date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


__all__ = [
    "KMA_BUSAN_STATION_ID",
    "KMA_REPLAY_SAMPLE_IDS",
    "SampleWeatherPlan",
    "build_kma_historical_replay",
    "resolve_sample_weather",
]
