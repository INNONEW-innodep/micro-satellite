"""지역(region) 레지스트리 — 플랫폼의 데이터 입력 규약.

내장 지역은 저장소의 ``regions/<id>/`` 에 커밋되어 배포 이미지에 들어가고,
운영 중 API로 만든 지역·지점은 ``REGIONS_USER_DIR`` 에 저장된다. 같은 id가
양쪽에 있으면 사용자 디렉터리가 우선한다 — 이미지를 다시 굽지 않고도 자료를
갱신할 수 있어야 하기 때문이다.

파일 규약은 ``regions/README.md`` 에 있다.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forecasters import DailySeries

REGION_SCHEMA = "watercast.region/1"
SERIES_SCHEMA = "watercast.gauge.series/1"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class RegionError(ValueError):
    """잘못된 지역·지점 입력."""


class RegionNotFoundError(KeyError):
    pass


class StationNotFoundError(KeyError):
    pass


WEATHER_FORECAST_FILE = "weather/kma_forecast_daily.json"  # 2026-10-20 기상청 단기예보(일 집계) — DailySeries.exog_tail 로 간다
DAM_DAILY_FILE = "weather/dam_daily.json"                   # 2026-10-20 댐 운영 일자료(WAMIS mn_dtdata) — 외생 dam_out/dam_in/dam_level


@dataclass(slots=True)
class StationRecord:
    id: str
    name: str
    source: str = ""
    source_code: str = ""
    terrain: str = ""
    terrain_note: str = ""
    unit: str = "m"
    n_days: int = 0
    period: tuple[str, str] | None = None
    satellite_location: str | None = None
    qc: dict[str, Any] | None = None
    lat: float | None = None  # WGS84 십진도(WAMIS wl_obsinfo) — 모르면 None
    lon: float | None = None
    address: str | None = None
    river: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "source_code": self.source_code,
            "terrain": self.terrain,
            "terrain_note": self.terrain_note,
            "unit": self.unit,
            "n_days": self.n_days,
            "period": list(self.period) if self.period else None,
            "satellite_location": self.satellite_location,
            "qc": self.qc,
            "lat": self.lat,
            "lon": self.lon,
            "address": self.address,
            "river": self.river,
        }


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


@dataclass(slots=True)
class RegionRecord:
    id: str
    name: str
    manifest: dict[str, Any]
    path: Path
    origin: str  # "builtin" | "user"
    # 사용자 사본이 내장 지역을 가릴 때의 원본 경로. 위성 자산처럼 사본에 복사되지 않는 파일을
    # 여기서 찾는다(재학습으로 사본이 생기면 장면·마스크가 사라지던 문제, 2026-09-15).
    builtin_path: Path | None = None

    def asset_path(self, relative: str) -> Path:
        """지역 자산의 실제 경로. 사용자 사본에 없으면 내장 원본에서 찾는다."""

        candidate = self.path / relative
        if candidate.is_file() or candidate.is_dir():
            return candidate
        if self.builtin_path is not None:
            fallback = self.builtin_path / relative
            if fallback.is_file() or fallback.is_dir():
                return fallback
        return candidate

    @property
    def gauge_available(self) -> bool:
        return bool(self.manifest.get("gauge", {}).get("available")) and self.series_path.is_file()

    @property
    def satellite_available(self) -> bool:
        return bool(self.manifest.get("satellite", {}).get("available"))

    @property
    def series_path(self) -> Path:
        rel = self.manifest.get("gauge", {}).get("series_file") or "gauge/series.json"
        return self.path / rel

    @property
    def artifacts_dir(self) -> Path:
        rel = self.manifest.get("forecast", {}).get("artifacts_dir") or "models/gauge"
        return self.path / rel

    @property
    def split(self) -> dict[str, str]:
        return dict(self.manifest.get("forecast", {}).get("split") or {})

    @property
    def horizon_max(self) -> int:
        return int(self.manifest.get("forecast", {}).get("horizon_max") or 30)

    @property
    def report_horizons(self) -> list[int]:
        raw = self.manifest.get("forecast", {}).get("horizons_report") or [1, 3, 5, 7, 14, 30]
        return [int(h) for h in raw]

    def satellite_links(self) -> dict[str, str]:
        return dict(self.manifest.get("gauge", {}).get("satellite_location_links") or {})

    def satellite_locations(self) -> dict[str, dict[str, Any]]:
        """위성 수위 산출 위치(AOI) — `satellite.locations` (loc_id → 경위도 상자·연결 지점). 없으면 {}."""

        raw = (self.manifest.get("satellite") or {}).get("locations") or {}
        if not isinstance(raw, Mapping):
            return {}
        return {str(k): dict(v) for k, v in raw.items() if isinstance(v, Mapping)}

    @property
    def satellite_levels_path(self) -> Path | None:
        """명시(`satellite.levels_file`)가 없으면 규약 경로 `satellite/levels.json`을 본다."""

        rel = (self.manifest.get("satellite") or {}).get("levels_file") or "satellite/levels.json"
        return self.asset_path(rel)

    @property
    def satellite_dir(self) -> Path:
        """장면·마스크가 실제로 있는 디렉터리(사용자 사본에 없으면 내장 원본)."""
        own = self.path / "satellite"
        if own.is_dir() and any(own.iterdir()):
            return own
        return self.asset_path("satellite")

    def scenes(self) -> dict[str, Any] | None:
        """위성 장면 목록(`satellite/scenes.json`). 퀵룩·마스크 PNG 파일명을 담는다."""

        rel = (self.manifest.get("satellite") or {}).get("scenes_file") or "satellite/scenes.json"
        path = self.asset_path(rel)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, Mapping) else None

    def satellite_file(self, relative: str) -> Path:
        """`satellite/` 아래 파일을 안전하게 찾는다(경로 탈출 금지)."""

        base = self.satellite_dir.resolve()
        target = (base / relative).resolve()
        if base not in target.parents and target != base:
            raise RegionError("satellite 디렉터리 밖의 파일은 열 수 없습니다")
        if not target.is_file():
            raise FileNotFoundError(relative)
        return target

    def area_series(self) -> dict[str, Any] | None:
        """공통 격자 기준 수체 면적 시계열(`satellite/area_series.json`).

        씬마다 촬영범위가 달라 `scenes.json` 의 `water_area_km2` 는 날짜 간 비교가 안 된다.
        이 파일은 센서별 공통 격자로 다시 센 `area_km2_common_frame` 을 담는다
        (생성: `scripts/build_common_frame_area.py`). 없으면 None.
        """

        path = self.asset_path("satellite/area_series.json")
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, Mapping) else None

    def timeseries_eval(self) -> dict[str, Any] | None:
        """위성 기반 수위·면적 시계열 지표 기록(`satellite/timeseries_eval.json`)."""

        path = self.asset_path("satellite/timeseries_eval.json")
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, Mapping) else None

    def detection_eval(self) -> dict[str, Any] | None:
        """수체 탐지 정확도 기록(`satellite/detection_eval.json`). GT 라벨이 없는 지역은 None."""

        rel = (self.manifest.get("satellite") or {}).get("detection_eval_file") or "satellite/detection_eval.json"
        path = self.asset_path(rel)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, Mapping) else None

    def satellite_levels(self) -> list[dict[str, Any]]:
        """WBMS ③/④ 산출 위성 수위 기록(지역 파일). 없으면 빈 목록."""

        path = self.satellite_levels_path
        if path is None or not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        records = payload.get("records") if isinstance(payload, Mapping) else payload
        return [dict(r) for r in (records or []) if isinstance(r, Mapping)]


@dataclass(slots=True)
class GaugeSeriesSet:
    region_id: str
    stations: list[StationRecord]
    series: dict[str, DailySeries]
    raw_rows: dict[str, list[dict[str, Any]]]
    meta: dict[str, Any] = field(default_factory=dict)

    def station(self, station_id: str) -> StationRecord:
        for item in self.stations:
            if item.id == station_id:
                return item
        raise StationNotFoundError(station_id)


def _validate_id(value: str, kind: str) -> str:
    value = str(value or "").strip()
    if not _ID_PATTERN.match(value):
        raise RegionError(f"{kind} id 는 영문·숫자·'_-.' 1~64자여야 합니다: {value!r}")
    if value in {".", ".."}:
        raise RegionError(f"{kind} id 가 올바르지 않습니다")
    return value


def _safe_station_dirname(station_id: str) -> str:
    """지점 id(한글 가능)를 파일 이름에 쓸 수 있게 만든다."""

    cleaned = re.sub(r"[^\w가-힣()\-.]", "_", station_id, flags=re.UNICODE)
    return cleaned[:80] or "station"


def parse_series_rows(payload: Any) -> list[dict[str, Any]]:
    """JSON 리스트·JSON 객체·CSV 텍스트를 ``[{"date", "water_level_m"}]`` 로 정규화한다."""

    rows: Sequence[Any]
    if isinstance(payload, (bytes, str)):
        text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
        text = text.strip()
        if not text:
            raise RegionError("빈 시계열입니다")
        if text[0] in "[{":
            payload = json.loads(text)
        else:
            reader = csv.DictReader(io.StringIO(text))
            payload = list(reader)
    if isinstance(payload, Mapping):
        if "rows" in payload:
            payload = payload["rows"]
        elif "series" in payload and isinstance(payload["series"], list):
            payload = payload["series"]
    if not isinstance(payload, Sequence):
        raise RegionError("시계열은 행 목록이어야 합니다")
    rows = payload

    date_keys = ("date", "obs_date", "ymd", "day", "datetime", "timestamp")
    level_keys = ("water_level_m", "water_level", "level_m", "level", "wl", "value")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise RegionError(f"{index}행: 객체가 아닙니다")
        raw_date = next((row[k] for k in date_keys if k in row and row[k] not in (None, "")), None)
        raw_level = next((row[k] for k in level_keys if k in row and row[k] not in (None, "")), None)
        if raw_date is None:
            raise RegionError(f"{index}행: 날짜 열(date)이 없습니다")
        text = str(raw_date).strip()[:10]
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError as exc:
            raise RegionError(f"{index}행: 날짜 형식은 YYYY-MM-DD 여야 합니다 ({raw_date!r})") from exc
        if raw_level is None:
            continue
        try:
            level = float(raw_level)
        except (TypeError, ValueError) as exc:
            raise RegionError(f"{index}행: 수위가 숫자가 아닙니다 ({raw_level!r})") from exc
        if level != level or level in (float("inf"), float("-inf")):
            continue
        out.append({"date": text, "water_level_m": round(level, 4)})
    if not out:
        raise RegionError("유효한 (date, water_level_m) 행이 없습니다")
    out.sort(key=lambda item: item["date"])
    return out



def quality_control_rows(rows: Sequence[Mapping[str, Any]], *, flat_days: int = 14, flat_tol_m: float = 0.02, spike_factor: float = 5.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """관측 자료의 명백한 계기·전송 오류만 결측으로 돌린다(값을 고치지 않고 뺀다). 홍수 봉우리·기준면 변경은 남긴다.

    - **배율 오류 정체 구간**: ``flat_days`` 일 이상 값이 ``flat_tol_m`` 안에서 멈춰 있고, 그 수준이 지점 중앙값의 15% 미만이거나
      5배 초과(단위·배율 오류 의심)이며 강건 범위(중앙값 ± 4·MAD) 밖이면 뺀다(예: 낙동강하구언(내) 2020년 353일 ≈ 0.009 m, 정상 0.77 m).
      보 관리수위처럼 정상 범위 안에서 멈춘 구간이나 기준면 변경(2 m 이동)은 남긴다.
    - **물리적으로 불가능한 단발 스파이크**: 하루 값이 나머지 자료의 4번째 큰 값(작은 값)보다 ``spike_factor``×범위폭 이상 더 크고(작고),
      전날·다음날과의 차이도 그만큼 크면 뺀다(예: 하구언 2015-11-16 13.79 m, 안동대교 4167 m). 같은 규모의 홍수가 기록에 여러 번
      있으면(나주대교 7~8 m 봉우리) 스파이크가 아니다.
    돌려주는 ``flags`` 에 제거한 날짜 수와 구간을 남겨 화면·평가표에서 밝힌다.
    """

    values: list[tuple[str, float]] = []
    for row in rows:
        try:
            values.append((str(row["date"]), float(row["water_level_m"])))
        except (KeyError, TypeError, ValueError):
            continue
    empty_flags = {"removed_days": 0, "flat_runs": [], "spikes": []}
    if len(values) < 60:
        return [dict(r) for r in rows], empty_flags
    # 1) 정체 후보 구간
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index < len(values) and abs(values[index][1] - values[start][1]) <= flat_tol_m:
            continue
        if index - start >= flat_days:
            runs.append((start, index))
        start = index
    in_run: set[int] = set()
    for a, b in runs:
        in_run.update(range(a, b))
    baseline = sorted(v for i, (_, v) in enumerate(values) if i not in in_run) or sorted(v for _, v in values)
    median = baseline[len(baseline) // 2]
    mad = sorted(abs(v - median) for v in baseline)[len(baseline) // 2] * 1.4826
    robust = max(mad, 0.02)
    trimmed = [v for v in baseline if abs(v - median) <= 4.0 * robust] or baseline
    lo = trimmed[int(0.01 * (len(trimmed) - 1))]
    hi = trimmed[int(0.99 * (len(trimmed) - 1))]
    span = max(hi - lo, 0.05)
    def local_stats(a: int, b: int, window: int = 365) -> tuple[float, float]:
        """구간 앞뒤 ``window`` 일(정체 후보 제외)의 중앙값·강건 산포 — 기준면 변경으로 전체 통계가 어긋나도 그 시기의 정상 수준을 본다."""
        neighbours = [v for i, (_, v) in enumerate(values) if (a - window <= i < a or b <= i < b + window) and i not in in_run]
        if len(neighbours) < 30:
            neighbours = [v for i, (_, v) in enumerate(values) if i not in in_run] or [v for _, v in values]
        neighbours.sort()
        med = neighbours[len(neighbours) // 2]
        spread = sorted(abs(v - med) for v in neighbours)[len(neighbours) // 2] * 1.4826
        return med, max(spread, 0.02)

    def scale_error(level: float, reference: float, spread: float) -> bool:
        # 기준이 0 근처(조위 기준면 등)면 산포(2·MAD)를 기준으로 삼아 배율 판정이 흔들리지 않게 한다
        ref = max(abs(reference), 2.0 * spread, 0.05)
        return abs(level) < 0.15 * ref or abs(level) > 5.0 * ref

    drop: set[str] = set()
    flat_runs: list[dict[str, Any]] = []
    bad_levels: list[float] = []
    for a, b in runs:
        run = values[a:b]
        level = sum(v for _, v in run) / len(run)
        reference, spread = local_stats(a, b)
        if abs(level - reference) > 4.0 * spread and scale_error(level, reference, spread):
            drop.update(d for d, _ in run)
            bad_levels.append(level)
            flat_runs.append({"start": run[0][0], "end": run[-1][0], "days": len(run), "level_m": round(level, 4), "local_normal_m": round(reference, 4)})
    # 정체 구간 사이에 낀 같은 수준의 낱개 오류(예: 0.009 m 가 며칠 끊겼다 이어짐)도 함께 뺀다
    if bad_levels:
        for index, (date, value) in enumerate(values):
            if date in drop or not any(abs(value - level) <= 2 * flat_tol_m for level in bad_levels):
                continue
            reference, spread = local_stats(index, index + 1)
            if abs(value - reference) > 4.0 * spread and scale_error(value, reference, spread):
                drop.add(date)
    # 2) 물리적으로 불가능한 단발 스파이크 — 나머지 기록의 4번째 극값보다 훨씬 더 극단이어야 한다
    remaining = sorted(v for d, v in values if d not in drop)
    # 후보 자신을 뺀 '나머지 기록'의 4번째 극값 → 5번째 극값을 쓴다(후보가 최댓값일 때 정확, 아니면 보수적)
    top4 = remaining[-5] if len(remaining) >= 5 else remaining[-1]
    bottom4 = remaining[4] if len(remaining) >= 5 else remaining[0]
    band_lo, band_hi = lo - span, hi + span  # 평상시 범위(1~99 백분위 ± 폭) — 이 안이면 '평소 수위'

    def normal_day(index: int) -> bool:
        if index < 0 or index >= len(values):
            return True
        date, value = values[index]
        return date not in drop and band_lo <= value <= band_hi

    spikes: list[dict[str, Any]] = []
    for index, (date, value) in enumerate(values):
        if date in drop:
            continue
        beyond = value > top4 + spike_factor * span or value < bottom4 - spike_factor * span
        absurd = value > top4 + 50.0 * span or value < bottom4 - 50.0 * span  # 기록 극값의 수십 배 밖 — 이웃과 무관하게 오류
        if not beyond:
            continue
        # 전날·다음날이 평소 수위여야 '하루 만에 튀었다 돌아온' 계기 오류다 — 며칠에 걸친 홍수(이웃 날도 높음)는 남긴다
        if absurd or (normal_day(index - 1) and normal_day(index + 1)):
            drop.add(date)
            spikes.append({"date": date, "level_m": round(value, 4)})
    clean = [dict(row) for row in rows if str(row.get("date")) not in drop]
    flags = {"removed_days": len(drop), "flat_runs": flat_runs, "spikes": spikes, "normal_range_m": [round(lo, 4), round(hi, 4)]}
    return clean, flags


class RegionRegistry:
    def __init__(self, builtin_dir: Path, user_dir: Path) -> None:
        self.builtin_dir = Path(builtin_dir)
        self.user_dir = Path(user_dir)

    # --- 조회 --------------------------------------------------------------------

    def _scan(self, root: Path, origin: str) -> dict[str, RegionRecord]:
        out: dict[str, RegionRecord] = {}
        if not root.is_dir():
            return out
        for path in sorted(root.iterdir()):
            manifest_path = path / "region.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            region_id = str(manifest.get("id") or path.name)
            out[region_id] = RegionRecord(
                id=region_id,
                name=str(manifest.get("name") or region_id),
                manifest=manifest,
                path=path,
                origin=origin,
            )
        return out

    def list(self) -> list[RegionRecord]:
        builtin = self._scan(self.builtin_dir, "builtin")
        merged = dict(builtin)
        for key, record in self._scan(self.user_dir, "user").items():
            origin = builtin.get(key)
            if origin is not None:
                # 사용자 사본이 내장 지역을 가릴 때, 사본에 없는 자산(위성 장면·마스크 등)은
                # 내장 원본에서 찾도록 경로를 달아 준다.
                record.builtin_path = origin.path
            merged[key] = record
        return [merged[key] for key in sorted(merged)]

    def get(self, region_id: str) -> RegionRecord:
        for record in self.list():
            if record.id == region_id:
                return record
        raise RegionNotFoundError(region_id)

    def builtin_record(self, region_id: str) -> RegionRecord | None:
        """저장소 내장 정의(사용자 사본이 덮어써도 원본). 내장 지역이 아니면 None."""

        return self._scan(self.builtin_dir, "builtin").get(region_id)

    def builtin_artifacts_dir(self, region_id: str) -> Path | None:
        """내장 학습 산출물 디렉터리 — 지점 산출물(*.json)이 하나라도 있을 때만. 되돌리기(reset)의 대상."""

        builtin = self.builtin_record(region_id)
        if builtin is None:
            return None
        path = builtin.artifacts_dir
        if path.is_dir() and any(path.glob("*.json")):
            return path
        return None

    def user_artifacts_dir(self, region_id: str) -> Path:
        """사용자 디렉터리의 산출물 경로(존재 여부와 무관)."""

        record = self.get(region_id)
        rel = record.manifest.get("forecast", {}).get("artifacts_dir") or "models/gauge"
        return self.user_dir / _validate_id(record.id, "region") / rel

    def reset_user_artifacts(self, region_id: str) -> list[str]:
        """사용자 디렉터리의 학습 산출물(``models/gauge/*.json``)만 지워 내장 산출물이 다시 활성화되게 한다.

        내장 산출물이 없는 지역(사용자 생성 지역, 또는 아직 학습되지 않은 내장 지역)은 되돌릴 곳이
        없으므로 ``RegionError``. 내장 파일은 절대 건드리지 않는다 — 삭제 대상은 사용자 디렉터리 바로 아래
        ``*.json`` 뿐이고(하위 디렉터리·심볼릭 링크 제외), 경로가 내장 디렉터리와 같으면 거부한다.
        """

        builtin_dir = self.builtin_artifacts_dir(region_id)
        if builtin_dir is None:
            raise RegionError(
                f"'{region_id}' 에는 되돌릴 내장 학습 산출물이 없습니다 — 사용자 생성 지역이거나 내장 산출물이 없는 지역입니다"
            )
        user_dir = self.user_artifacts_dir(region_id)
        if not user_dir.is_dir():
            return []
        if user_dir.resolve() == builtin_dir.resolve():
            raise RegionError("사용자 산출물 경로가 내장 경로와 같아 삭제하지 않습니다")
        removed: list[str] = []
        for path in sorted(user_dir.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            path.unlink()
            removed.append(path.name)
        # 사용자 사본에 '사용자가 더한 것'이 없으면(장면·업로드 없음, 수위계 파일이 내장과 같음) 사본 전체를 걷어낸다 —
        # 그대로 두면 내장 지역이 갱신돼도(좌표·AOI 등) 사본이 계속 가린다(2026-09-09 서버에서 발생).
        builtin_record = self.builtin_record(region_id)
        user_region = self._user_region_dir(region_id)
        if builtin_record is not None and user_region.is_dir() and self._user_copy_is_pristine(builtin_record, user_region):
            shutil.rmtree(user_region, ignore_errors=True)
            removed.append("(사용자 사본 전체 제거 — 내장 지역으로 복귀)")
        return removed

    def _user_copy_is_pristine(self, builtin_record: RegionRecord, user_region: Path) -> bool:
        """사용자 사본이 내장 자료의 복사본일 뿐인지: 장면·업로드 디렉터리가 없고 수위계 파일이 내장과 byte 단위로 같다."""

        if (user_region / "satellite").exists() or (user_region / "uploads").exists():
            return False
        series_rel = builtin_record.manifest.get("gauge", {}).get("series_file") or "gauge/series.json"
        user_series, builtin_series = user_region / series_rel, builtin_record.path / series_rel
        if user_series.is_file():
            if not builtin_series.is_file():
                return False
            try:
                if user_series.read_bytes() != builtin_series.read_bytes():
                    return False
            except OSError:
                return False
        models_dir = user_region / (builtin_record.manifest.get("forecast", {}).get("artifacts_dir") or "models/gauge")
        if models_dir.is_dir() and any(models_dir.iterdir()):
            return False
        return True

    def load_weather_rows(self, region_id: str) -> list[dict[str, Any]]:
        """지역 기상 일자료(`weather/asos_daily.json`). 없으면 빈 목록."""

        record = self.get(region_id)
        rel = (record.manifest.get("weather") or {}).get("daily_file") or "weather/asos_daily.json"
        path = record.path / rel
        if not path.is_file() and record.origin == "user":
            # 사용자 사본(내장 지역을 고친 경우)에 기상 파일이 없으면 내장 원본의 기상을 쓴다
            builtin = self._scan(self.builtin_dir, "builtin").get(region_id)
            if builtin is not None and (builtin.path / rel).is_file():
                path = builtin.path / rel
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        rows = payload.get("rows") if isinstance(payload, Mapping) else payload
        return [dict(r) for r in (rows or []) if isinstance(r, Mapping)]

    def _load_aux_rows(self, region_id: str, rel: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """지역 보조 자료 파일(사용자 사본 우선, 없으면 내장 원본) → (rows, 파일 메타)."""

        record = self.get(region_id)
        path = record.path / rel
        if not path.is_file() and record.origin == "user":
            builtin = self._scan(self.builtin_dir, "builtin").get(region_id)
            if builtin is not None and (builtin.path / rel).is_file():
                path = builtin.path / rel
        if not path.is_file():
            return [], {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return [], {}
        if not isinstance(payload, Mapping):
            return [], {}
        rows = [dict(r) for r in (payload.get("rows") or []) if isinstance(r, Mapping)]
        return rows, {k: v for k, v in payload.items() if k != "rows"}

    def load_weather_forecast(self, region_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """기상청 단기예보 일 강수·기온(`weather/kma_forecast_daily.json`, 2026-10-20). 없으면 ([], {})."""

        return self._load_aux_rows(region_id, WEATHER_FORECAST_FILE)

    def load_dam_rows(self, region_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """댐 운영 일자료(`weather/dam_daily.json`: date, dam_level_m, inflow_cms, outflow_cms …, 2026-10-20). 없으면 ([], {})."""

        return self._load_aux_rows(region_id, DAM_DAILY_FILE)

    def weather_forecast_write_path(self, region_id: str) -> Path:
        record = self.get(region_id)
        return self._materialize_user_copy(record) / WEATHER_FORECAST_FILE

    def dam_write_path(self, region_id: str) -> Path:
        record = self.get(region_id)
        return self._materialize_user_copy(record) / DAM_DAILY_FILE

    def load_series(self, region_id: str) -> GaugeSeriesSet:
        record = self.get(region_id)
        if not record.series_path.is_file():
            return GaugeSeriesSet(region_id=region_id, stations=[], series={}, raw_rows={})
        payload = json.loads(record.series_path.read_text(encoding="utf-8"))
        weather_rows = self.load_weather_rows(region_id)
        forecast_rows, forecast_meta = self.load_weather_forecast(region_id)
        if weather_rows and forecast_rows:
            # 예보는 관측이 끝난 다음 날부터만 붙인다(관측이 있는 날은 관측이 우선). 달력 끝 뒤의 날은 DailySeries.exog_tail 로 간다.
            last_obs = max(str(r.get("date") or "")[:10] for r in weather_rows)
            weather_rows = weather_rows + [r for r in forecast_rows if str(r.get("date") or "")[:10] > last_obs]
        dam_rows, dam_meta = self.load_dam_rows(region_id)
        links = record.satellite_links()
        stations: list[StationRecord] = []
        series: dict[str, DailySeries] = {}
        raw_rows: dict[str, list[dict[str, Any]]] = {}
        declared = {str(item.get("id")): dict(item) for item in payload.get("stations", []) if isinstance(item, Mapping)}
        for station_id, rows in (payload.get("series") or {}).items():
            if not isinstance(rows, list) or not rows:
                continue
            rows, qc_flags = quality_control_rows(rows)
            if not rows:
                continue
            try:
                daily = DailySeries.from_rows(rows)
            except ValueError:
                continue
            if weather_rows:
                daily = daily.with_exog(weather_rows, {"precipitation_mm": "rain_mm", "avg_temperature_c": "temp_c"})
            if dam_rows:
                daily = daily.with_exog(dam_rows, {"outflow_cms": "dam_out", "inflow_cms": "dam_in", "dam_level_m": "dam_level"})
            meta = declared.get(station_id, {})
            first = str(rows[0].get("date"))
            last = str(rows[-1].get("date"))
            stations.append(
                StationRecord(
                    id=station_id,
                    name=str(meta.get("name") or station_id),
                    source=str(meta.get("source") or payload.get("source") or ""),
                    source_code=str(meta.get("source_code") or ""),
                    terrain=str(meta.get("terrain") or ""),
                    terrain_note=str(meta.get("terrain_note") or ""),
                    unit=str(meta.get("unit") or payload.get("unit") or "m"),
                    n_days=int(daily.valid.sum()),
                    period=(first, last),
                    satellite_location=links.get(station_id),
                    qc=qc_flags if qc_flags.get("removed_days") else None,
                    lat=_optional_float(meta.get("lat")),
                    lon=_optional_float(meta.get("lon")),
                    address=str(meta.get("address")) if meta.get("address") else None,
                    river=str(meta.get("river")) if meta.get("river") else None,
                )
            )
            series[station_id] = daily
            raw_rows[station_id] = [dict(row) for row in rows]
        meta = {k: v for k, v in payload.items() if k not in {"series", "stations"}}
        meta["weather_available"] = bool(weather_rows)
        meta["weather_forecast_available"] = bool(forecast_rows)
        meta["weather_forecast_issued_at"] = forecast_meta.get("issued_at") if forecast_rows else None
        meta["dam_available"] = bool(dam_rows)
        meta["dam_code"] = dam_meta.get("dam_code") if dam_rows else None
        return GaugeSeriesSet(region_id=region_id, stations=stations, series=series, raw_rows=raw_rows, meta=meta)

    # --- 쓰기 (사용자 디렉터리) ------------------------------------------------------

    def _user_region_dir(self, region_id: str) -> Path:
        return self.user_dir / _validate_id(region_id, "region")

    def _materialize_user_copy(self, record: RegionRecord) -> Path:
        """내장 지역을 고칠 때는 사용자 디렉터리로 복사한 뒤 거기서 고친다."""

        target = self._user_region_dir(record.id)
        if record.origin == "user":
            return target
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "region.json"
        if not manifest_path.is_file():
            manifest_path.write_text(
                json.dumps(record.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if record.series_path.is_file():
            dest = target / (record.manifest.get("gauge", {}).get("series_file") or "gauge/series.json")
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(record.series_path, dest)
        if record.artifacts_dir.is_dir():
            dest_models = target / (record.manifest.get("forecast", {}).get("artifacts_dir") or "models/gauge")
            if not dest_models.is_dir():
                shutil.copytree(record.artifacts_dir, dest_models)
        # 기상 외생 자료도 함께 — 없으면 사용자 사본에서 ridge_arx·앙상블 예측이 422 로 막힌다(2026-09-08 감사에서 발견)
        weather_rel = (record.manifest.get("weather") or {}).get("daily_file") or "weather/asos_daily.json"
        if (record.path / weather_rel).is_file() and not (target / weather_rel).is_file():
            (target / weather_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.path / weather_rel, target / weather_rel)
        return target

    def create_region(self, manifest: Mapping[str, Any], *, overwrite: bool = False) -> RegionRecord:
        region_id = _validate_id(str(manifest.get("id") or ""), "region")
        name = str(manifest.get("name") or region_id).strip()
        existing = {record.id for record in self.list()}
        if region_id in existing and not overwrite:
            raise RegionError(f"지역 {region_id!r} 가 이미 있습니다 (overwrite=true 로 덮어쓰기)")
        target = self._user_region_dir(region_id)
        target.mkdir(parents=True, exist_ok=True)
        document: dict[str, Any] = {
            "schema": REGION_SCHEMA,
            "id": region_id,
            "name": name,
            "description": str(manifest.get("description") or ""),
            "crs": str(manifest.get("crs") or ""),
            "bbox_wgs84": list(manifest.get("bbox_wgs84") or []),
            "weather": dict(manifest.get("weather") or {}),
            "satellite": dict(manifest.get("satellite") or {"available": False, "sensors": [], "scenes": {}}),
            "gauge": {
                "available": False,
                "series_file": "gauge/series.json",
                "satellite_location_links": dict((manifest.get("gauge") or {}).get("satellite_location_links") or {}),
            },
            "forecast": {
                "artifacts_dir": "models/gauge",
                "horizons_report": list((manifest.get("forecast") or {}).get("horizons_report") or [1, 3, 5, 7, 14, 30]),
                "horizon_max": int((manifest.get("forecast") or {}).get("horizon_max") or 30),
                "split": dict((manifest.get("forecast") or {}).get("split") or {}),
            },
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (target / "region.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.get(region_id)

    def upsert_station_series(
        self,
        region_id: str,
        station_id: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        replace: bool = False,
        station_meta: Mapping[str, Any] | None = None,
    ) -> StationRecord:
        record = self.get(region_id)
        station_id = str(station_id).strip()
        if not station_id or len(station_id) > 80 or "/" in station_id or "\\" in station_id:
            raise RegionError("지점 id 는 1~80자이고 경로 구분자를 포함할 수 없습니다")
        new_rows = parse_series_rows(list(rows))
        target = self._materialize_user_copy(record)
        series_rel = record.manifest.get("gauge", {}).get("series_file") or "gauge/series.json"
        series_path = target / series_rel
        if series_path.is_file():
            payload = json.loads(series_path.read_text(encoding="utf-8"))
        else:
            payload = {"schema": SERIES_SCHEMA, "unit": "m", "frequency": "daily", "stations": [], "series": {}}
        payload.setdefault("stations", [])
        payload.setdefault("series", {})

        merged: dict[str, float] = {}
        if not replace:
            for row in payload["series"].get(station_id, []):
                merged[str(row["date"])] = float(row["water_level_m"])
        for row in new_rows:
            merged[row["date"]] = row["water_level_m"]
        rows_out = [{"date": d, "water_level_m": merged[d]} for d in sorted(merged)]
        payload["series"][station_id] = rows_out

        previous = next((dict(s) for s in payload["stations"] if str(s.get("id")) == station_id), {})
        stations = [s for s in payload["stations"] if str(s.get("id")) != station_id]
        # 이어 붙이기(replace=False)는 기존 지점 메타(이름·출처·지형)를 지운다 — 새 메타가 없으면 이전 값을 유지
        meta = {k: v for k, v in dict(station_meta or {}).items() if v not in (None, "")}
        base = previous if not replace else {}
        stations.append(
            {
                "id": station_id,
                "name": str(meta.get("name") or base.get("name") or station_id),
                "source": str(meta.get("source") or base.get("source") or "user_upload"),
                "source_code": str(meta.get("source_code") or base.get("source_code") or ""),
                "terrain": str(meta.get("terrain") or base.get("terrain") or ""),
                "terrain_note": str(meta.get("terrain_note") or base.get("terrain_note") or ""),
                "unit": "m",
                "n_days": len(rows_out),
                "period": [rows_out[0]["date"], rows_out[-1]["date"]],
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                # 관측소 제원(WAMIS wl_obsinfo) — 새 메타가 주면 그것, 아니면 이전 값(없으면 None)
                **{key: meta.get(key, base.get(key)) for key in ("lat", "lon", "address", "river", "tm_xy")},
            }
        )
        payload["stations"] = sorted(stations, key=lambda s: str(s.get("id")))
        series_path.parent.mkdir(parents=True, exist_ok=True)
        series_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

        manifest_path = target / "region.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("gauge", {})["available"] = True
        manifest["gauge"].setdefault("series_file", series_rel)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.load_series(region_id).station(station_id)

    def weather_write_path(self, region_id: str) -> Path:
        """기상 일자료(`weather/asos_daily.json`)를 쓸 경로 — 내장 지역이면 사용자 사본을 만든 뒤 그 안.

        읽기(`load_weather_rows`)와 같은 상대경로(`weather.daily_file`)를 쓰므로, 여기 쓰면 예측기가 바로 읽는다.
        """

        record = self.get(region_id)
        target = self._materialize_user_copy(record)
        rel = (record.manifest.get("weather") or {}).get("daily_file") or "weather/asos_daily.json"
        return target / rel

    def update_manifest_section(self, region_id: str, section: str, values: Mapping[str, Any]) -> dict[str, Any]:
        """사용자 사본 `region.json` 의 한 섹션(예: ``weather``)에 값을 병합해 저장하고 전체 매니페스트를 돌려준다."""

        record = self.get(region_id)
        target = self._materialize_user_copy(record)
        path = target / "region.json"
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else dict(record.manifest)
        block = dict(manifest.get(section) or {})
        block.update({str(k): v for k, v in values.items()})
        manifest[section] = block
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def satellite_write_dir(self, region_id: str) -> Path:
        """새 장면 자산을 쓸 `satellite/` 디렉터리 — 내장 지역이면 사용자 사본(기존 자산 복사 포함)."""

        record = self.get(region_id)
        target = self._materialize_user_copy(record)
        dest = target / "satellite"
        if record.origin == "builtin" and record.satellite_dir.is_dir() and not dest.is_dir():
            shutil.copytree(record.satellite_dir, dest)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def artifacts_write_dir(self, region_id: str, *, prefer_builtin: bool = False) -> Path:
        """학습 산출물을 쓸 디렉터리. API는 사용자 디렉터리, CLI(``prefer_builtin``)는 저장소."""

        record = self.get(region_id)
        if prefer_builtin and record.origin == "builtin":
            return record.artifacts_dir
        target = self._materialize_user_copy(record)
        write_dir = target / (record.manifest.get("forecast", {}).get("artifacts_dir") or "models/gauge")
        # 되돌리기(reset) 뒤에는 사용자 디렉터리가 비어 있다. 지점 일부만 다시 학습하면 그 디렉터리가 활성이 되면서
        # 나머지 지점의 내장 산출물이 보이지 않게 되므로(/model·/forecast 404), 학습 전에 내장 산출물을 다시 채운다.
        builtin_dir = self.builtin_artifacts_dir(region_id)
        if builtin_dir is not None and not any(write_dir.glob("*.json")) and write_dir.resolve() != builtin_dir.resolve():
            write_dir.mkdir(parents=True, exist_ok=True)
            for src in builtin_dir.glob("*.json"):
                if src.is_file():
                    shutil.copyfile(src, write_dir / src.name)
        return write_dir

    @staticmethod
    def station_artifact_path(artifacts_dir: Path, station_id: str) -> Path:
        return artifacts_dir / f"{_safe_station_dirname(station_id)}.json"
