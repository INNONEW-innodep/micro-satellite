"""Presentation-safe EDA views backed by the sanitized NAS source profile.

This module never connects to the NAS.  It reads the checked-in, read-only
audit profile under ``assets/nas`` and turns the five source-data groups into
small immutable metric/table payloads and Plotly figures.  It deliberately
keeps source metadata EDA separate from model validation: every profiled group
still lacks either enough dates, a water-label sequence, or measured gauge
truth for a defensible daily forecast evaluation.

The loader rejects absolute paths, connection strings, credentials and an
unexpected schema before any value can reach the UI.  Returned table rows are
new presentation dictionaries; raw profile mappings are never exposed to a
Streamlit caller.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAS_SOURCE_PROFILE_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "assets" / "nas" / "nas_source_profile.json"
)
MAX_PROFILE_BYTES: Final[int] = 2 * 1024 * 1024

ICEYE_MAP_ASSET_DIR: Final[Path] = (
    Path(__file__).resolve().parent / "assets" / "nas" / "iceye_2020"
)
ICEYE_META_PATH: Final[Path] = ICEYE_MAP_ASSET_DIR / "iceye_meta.json"
ICEYE_MANIFEST_PATH: Final[Path] = ICEYE_MAP_ASSET_DIR / "manifest.json"
MAX_MAP_ASSET_BYTES: Final[int] = MAX_PROFILE_BYTES

# The audited ICEYE stack covers the Busan reservoirs only; every map
# coordinate must fall inside this plausibility box or the layer is dropped.
_MAP_LAT_RANGE: Final[tuple[float, float]] = (33.5, 36.5)
_MAP_LON_RANGE: Final[tuple[float, float]] = (127.5, 130.5)
_UTM52N_CRS: Final[str] = "EPSG:32652"
_WGS84_CRS: Final[str] = "EPSG:4326"
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PROFILE_KEY_BY_CATALOG_GROUP: Mapping[str, str] = MappingProxyType(
    {
        "planetscope-raw": "planetscope_raw",
        "iceye-raw": "iceye_raw",
        "university-single-date": "university_single_date",
        "giheung-optical": "giheung_worldview",
        "hoedong-optical": "hoedong_worldview",
    }
)

_REQUIRED_PROFILE_GROUPS = frozenset(PROFILE_KEY_BY_CATALOG_GROUP.values())
_SENSITIVE_KEY_NAMES = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_key",
        "username",
        "user_name",
        "hostname",
        "host",
        "server_address",
        "nas_address",
        "ip_address",
    }
)
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_CONNECTION_URI_RE = re.compile(r"\b(?:ssh|smb|nfs|file)://", re.IGNORECASE)
_USER_AT_HOST_RE = re.compile(
    r"(?<![\w.-])[A-Za-z0-9._-]+@(?:[A-Za-z0-9-]+\.)*[A-Za-z0-9-]+"
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")

_DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(2,6,23,.42)",
    "font": {"color": "#cbd5e1"},
    "hovermode": "x unified",
    "margin": {"l": 22, "r": 22, "t": 58, "b": 30},
    "legend": {"orientation": "h", "y": 1.12, "x": 0.0},
}

_NEXT_STEPS_KO: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "planetscope-raw": (
            "같은 날짜의 두 scene을 한 관측으로 묶기",
            "UDM2 unusable·구름·헤이즈 영역 제외",
            "날짜별 영상을 동일 AOI·CRS·격자로 모자이크",
            "각 날짜의 수체 마스크를 생성하거나 정답 라벨과 연결",
        ),
        "iceye-raw": (
            "방사보정과 스펙클 필터링",
            "GCP 기반 정사·지형보정 후 공통 격자로 변환",
            "입사각과 look side 차이를 정규화",
            "날짜별 수체 마스크 생성 및 라벨 검수",
        ),
        "university-single-date": (
            "동일 지점의 추가 관측일과 실측 수위 확보",
            "WB JSON trailing comma 정제",
            "원본과 DEM·WB를 geotransform 기준으로 정렬",
        ),
        "giheung-optical": (
            "Catalog ID로 PAN·MUL pair 구성",
            "WV02·WV03 방사 특성을 날짜 간 정규화",
            "MUL 8밴드에서 날짜별 수체 마스크 생성",
            "모든 날짜를 동일 AOI·CRS·격자로 정렬",
        ),
        "hoedong-optical": (
            "Catalog ID로 PAN·MUL pair 구성",
            "2023-11-08 부분영상을 모자이크하여 한 시점으로 구성",
            "MUL 8밴드에서 날짜별 수체 마스크 생성",
            "추가 관측일과 수위 정답 확보",
        ),
    }
)

_CAUTIONS_KO: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "planetscope-raw": (
            "네 시점의 관측 간격이 불규칙하며 UDM2는 수체 라벨이 아닙니다.",
            "현재 수치만으로 일별 예측 모델의 일반화 성능을 평가할 수 없습니다.",
        ),
        "iceye-raw": (
            "중심 입사각이 11.61~33.17°로 달라지고 2020-03-30은 left-looking입니다.",
            "이 관측기하 차이를 보정하지 않은 SAR 밝기 변화는 수체 변화가 아닙니다.",
        ),
        "university-single-date": (
            "관측일이 한 번뿐이며 WB JSON 두 파일은 strict JSON 형식 오류입니다.",
            "다른 지역·날짜의 영상에 안동·대청 수위값을 결합하면 안 됩니다.",
        ),
        "giheung-optical": (
            "WV02와 WV03가 섞여 있고 전달 수체 마스크·실측 수위가 없습니다.",
            "PAN과 MUL은 같은 획득의 제품이며 별도 시간 프레임이 아닙니다.",
        ),
        "hoedong-optical": (
            (
                "2023-11-08 두 획득은 11.55초 차이의 겹치는 부분영상으로 "
                "독립 일별 시점이 아닙니다."
            ),
            "모자이크 후 고유 날짜는 두 개뿐이며 수체 마스크·실측 수위가 없습니다.",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class NasSourceProfile:
    """Validated and recursively immutable source-profile payload."""

    schema_version: str
    profile_id: str
    audited_at_utc: str
    source_label: str
    limitations: tuple[str, ...]
    groups: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SourceReadiness:
    """Capability-specific explanation for one audited source group."""

    status: str
    headline_ko: str
    explanation_ko: str
    next_steps_ko: tuple[str, ...]
    cautions_ko: tuple[str, ...]
    source_eda_available: bool = True
    daily_timeseries_prediction_available: bool = False
    scientific_validation_available: bool = False


@dataclass(frozen=True, slots=True)
class NasSourceEDAView:
    """One UI-ready source EDA panel without NAS connection information."""

    group_id: str
    profile_group_key: str
    display_name_ko: str
    audited_at_utc: str
    provenance_ko: str
    metric_rows: tuple[Mapping[str, str], ...]
    table_rows: tuple[Mapping[str, Any], ...]
    figure: go.Figure
    readiness: SourceReadiness
    limitations_ko: tuple[str, ...]
    map_figure: go.Figure | None = None


def _looks_like_absolute_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith(("/", "\\\\", "//")):
        return True
    if _WINDOWS_ABSOLUTE_RE.match(text):
        return True
    try:
        return PurePosixPath(text).is_absolute()
    except (TypeError, ValueError):
        return False


def _scan_for_exposure(value: Any, *, path: str = "$") -> None:
    """Reject connection details and absolute source paths recursively."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_KEY_NAMES:
                raise ValueError(f"connection or credential key is forbidden at {path}")
            _scan_for_exposure(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _scan_for_exposure(item, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    text = value.strip()
    if _looks_like_absolute_path(text):
        raise ValueError(f"absolute path is forbidden at {path}")
    if (
        _IPV4_RE.search(text)
        or _CONNECTION_URI_RE.search(text)
        or _USER_AT_HOST_RE.search(text)
    ):
        raise ValueError(f"connection or credential value is forbidden at {path}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be an array")
    return tuple(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: Any, *, field: str, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or (non_negative and parsed < 0):
        raise ValueError(f"{field} must be finite and valid")
    return parsed


def _integer(value: Any, *, field: str, non_negative: bool = False) -> int:
    parsed = _number(value, field=field, non_negative=non_negative)
    if not parsed.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(parsed)


def _shape_text(shape: Sequence[Any], *, field: str) -> str:
    if len(shape) != 2:
        raise ValueError(f"{field} must contain height and width")
    height = _integer(shape[0], field=f"{field}.height", non_negative=True)
    width = _integer(shape[1], field=f"{field}.width", non_negative=True)
    return f"{height} × {width}"


def _validate_logical_roots(groups: Mapping[str, Any]) -> None:
    for group_key, raw_group in groups.items():
        group = _mapping(raw_group, field=f"groups.{group_key}")
        root = _text(
            group.get("logical_root"), field=f"groups.{group_key}.logical_root"
        )
        logical = PurePosixPath(root)
        if logical.is_absolute() or ".." in logical.parts:
            raise ValueError(f"groups.{group_key}.logical_root must be relative")


def load_nas_source_profile(
    path: str | Path = NAS_SOURCE_PROFILE_PATH,
) -> NasSourceProfile:
    """Load, sanitize and freeze the checked-in NAS source audit profile."""

    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ValueError(f"NAS source profile is unavailable: {source.name}") from exc
    if size <= 0 or size > MAX_PROFILE_BYTES:
        raise ValueError("NAS source profile size is outside the safe limit")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("NAS source profile must be valid UTF-8 JSON") from exc
    root = _mapping(payload, field="profile")
    if root.get("schema_version") != "1.0":
        raise ValueError("unsupported NAS source profile schema_version")
    profile_id = _text(root.get("profile_id"), field="profile.profile_id")
    audit = _mapping(root.get("audit"), field="profile.audit")
    if audit.get("source") != "NAS read-only audit":
        raise ValueError("profile audit source must identify the read-only audit")
    if audit.get("source_files_modified") is not False:
        raise ValueError("profile must confirm source_files_modified=false")
    if audit.get("credentials_included") is not False:
        raise ValueError("profile must confirm credentials_included=false")
    if audit.get("absolute_paths_included") is not False:
        raise ValueError("profile must confirm absolute_paths_included=false")
    audited_at = _text(
        audit.get("audited_at_utc"), field="profile.audit.audited_at_utc"
    )
    limitations = tuple(
        _text(item, field="profile.audit.limitations[]")
        for item in _sequence(
            audit.get("limitations"), field="profile.audit.limitations"
        )
    )
    groups = _mapping(root.get("groups"), field="profile.groups")
    missing = sorted(_REQUIRED_PROFILE_GROUPS.difference(groups))
    if missing:
        raise ValueError(f"NAS source profile is missing groups: {', '.join(missing)}")
    _validate_logical_roots(groups)
    _scan_for_exposure(root)
    frozen_groups = _freeze_json(dict(groups))
    return NasSourceProfile(
        schema_version="1.0",
        profile_id=profile_id,
        audited_at_utc=audited_at,
        source_label="NAS 읽기 전용 원본 감사",
        limitations=limitations,
        groups=frozen_groups,
    )


def available_source_group_ids() -> tuple[str, ...]:
    """Catalog group ids for which the detailed source profile has an EDA."""

    return tuple(PROFILE_KEY_BY_CATALOG_GROUP)


def _metric(label: str, value: str, help_ko: str) -> Mapping[str, str]:
    return MappingProxyType({"label": label, "value": value, "help": help_ko})


def _row(**values: Any) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _readiness(
    group_id: str,
    group: Mapping[str, Any],
) -> SourceReadiness:
    raw = _mapping(
        group.get("prediction_readiness"), field=f"{group_id}.prediction_readiness"
    )
    status = _text(raw.get("status"), field=f"{group_id}.prediction_readiness.status")
    if status not in {"preprocess_required", "single_date_reference"}:
        raise ValueError(f"unsupported readiness status for {group_id}: {status}")
    raw_steps = tuple(
        _text(item, field=f"{group_id}.required_steps[]")
        for item in _sequence(
            raw.get("required_steps", ()), field=f"{group_id}.required_steps"
        )
    )
    caution_values = tuple(
        text
        for text in (
            raw.get("critical_caution"),
            raw.get("caution"),
            raw.get("limitation"),
            raw.get("reason"),
        )
        if text is not None
    )
    raw_cautions = tuple(
        _text(item, field=f"{group_id}.readiness.caution") for item in caution_values
    )
    # Validate the source explanation above, but return operator-facing Korean
    # instructions so ``*_ko`` fields never silently mix languages in the UI.
    steps = _NEXT_STEPS_KO.get(group_id, raw_steps)
    cautions = _CAUTIONS_KO.get(group_id, raw_cautions)
    if status == "single_date_reference":
        headline = "단일시점 결과 EDA 가능 · 시계열 예측 불가"
        explanation = (
            "제공 면적·수위와 공간 산출 형식은 점검할 수 있지만 "
            "관측일이 한 번뿐입니다. "
            "이 자료만으로 일별 7·14·30일 모델을 학습하거나 RMSE를 계산할 수 없습니다."
        )
    else:
        headline = "원본 EDA 가능 · 전처리 후 수체 탐지 · 직접 예측 불가"
        explanation = (
            "현재 표와 그래프는 원본 센서·격자·품질·획득조건의 EDA입니다. "
            "필수 전처리와 날짜별 수체 마스크 생성 전에는 수체 시계열 입력이 아니며, "
            "실측 수위와 충분한 시간 표본이 없어 예측 성능을 주장할 수 없습니다."
        )
    return SourceReadiness(
        status=status,
        headline_ko=headline,
        explanation_ko=explanation,
        next_steps_ko=steps,
        cautions_ko=cautions,
    )


def _style_figure(figure: go.Figure, *, title: str) -> go.Figure:
    figure.update_layout(title=title, **_DARK_LAYOUT)
    return figure


def _planetscope_view(
    group_id: str,
    group: Mapping[str, Any],
    profile: NasSourceProfile,
) -> NasSourceEDAView:
    scenes = _sequence(group.get("scenes"), field="planetscope.scenes")
    if not scenes:
        raise ValueError("planetscope.scenes must not be empty")
    rows: list[Mapping[str, Any]] = []
    usable: list[float] = []
    clear: list[float] = []
    cloud: list[float] = []
    labels: list[str] = []
    for index, raw_scene in enumerate(scenes):
        scene = _mapping(raw_scene, field=f"planetscope.scenes[{index}]")
        udm2 = _mapping(scene.get("udm2"), field=f"planetscope.scenes[{index}].udm2")
        shape = _sequence(
            scene.get("shape"), field=f"planetscope.scenes[{index}].shape"
        )
        if len(shape) != 2:
            raise ValueError("PlanetScope scene shape must contain height and width")
        scene_id = _text(scene.get("scene_id"), field="planetscope.scene_id")
        acquired = _text(scene.get("acquired_utc"), field="planetscope.acquired_utc")
        use = _number(
            udm2.get("usable_percent"), field="udm2.usable_percent", non_negative=True
        )
        clr = _number(
            udm2.get("clear_percent_of_usable"), field="udm2.clear", non_negative=True
        )
        cld = _number(
            udm2.get("cloud_percent_of_usable"), field="udm2.cloud", non_negative=True
        )
        usable.append(use)
        clear.append(clr)
        cloud.append(cld)
        labels.append(f"{acquired[:10]} · {scene_id[-4:]}")
        shape_label = _shape_text(shape, field="planetscope.shape")
        rows.append(
            _row(
                **{
                    "관측시각 (UTC)": acquired,
                    "Scene ID": scene_id,
                    "위성 ID": _text(
                        scene.get("satellite_id"), field="planetscope.satellite_id"
                    ),
                    "크기 (H×W)": shape_label,
                    "UDM2 usable (%)": round(use, 3),
                    "usable 중 clear (%)": round(clr, 4),
                    "usable 중 cloud (%)": round(cld, 5),
                    "usable 중 snow (%)": round(
                        _number(
                            udm2.get("snow_percent_of_usable"),
                            field="udm2.snow",
                            non_negative=True,
                        ),
                        5,
                    ),
                    "usable 중 light haze (%)": round(
                        _number(
                            udm2.get("light_haze_percent_of_usable"),
                            field="udm2.light_haze",
                            non_negative=True,
                        ),
                        5,
                    ),
                    "평균 신뢰도": round(
                        _number(
                            udm2.get("mean_confidence_of_usable"),
                            field="udm2.confidence",
                            non_negative=True,
                        ),
                        3,
                    ),
                }
            )
        )

    dates = _sequence(
        group.get("observation_dates"), field="planetscope.observation_dates"
    )
    metrics = (
        _metric(
            "고유 관측일",
            f"{len(dates)}일",
            "같은 날짜의 두 scene은 한 시점으로 묶습니다.",
        ),
        _metric(
            "원본 Scene", f"{len(scenes)}개", "날짜마다 공간적으로 나뉜 두 scene입니다."
        ),
        _metric(
            "UDM2 usable",
            f"{min(usable):.1f}~{max(usable):.1f}%",
            "사각 클립 중 실제 usable 영역 비율입니다.",
        ),
        _metric(
            "유효영역 Cloud",
            f"최대 {max(cloud):.5f}%",
            "UDM2 cloud 픽셀을 usable 영역으로 나눈 값입니다.",
        ),
    )
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(x=labels, y=usable, name="usable 영역", marker_color="#0891b2"),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=clear,
            name="usable 중 clear",
            mode="lines+markers",
            line={"color": "#22c55e", "width": 2},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=cloud,
            name="usable 중 cloud",
            mode="lines+markers",
            line={"color": "#f97316", "width": 2},
        ),
        secondary_y=True,
    )
    figure.update_yaxes(
        title_text="usable / clear (%)", range=[0, 105], secondary_y=False
    )
    figure.update_yaxes(title_text="cloud (%)", rangemode="tozero", secondary_y=True)
    _style_figure(
        figure, title="PlanetScope UDM2 품질 EDA · UDM2는 수체 라벨이 아닙니다"
    )
    return NasSourceEDAView(
        group_id=group_id,
        profile_group_key="planetscope_raw",
        display_name_ko=_text(
            group.get("display_name_ko"), field="planetscope.display_name_ko"
        ),
        audited_at_utc=profile.audited_at_utc,
        provenance_ko="NAS 원본 STAC·UDM2를 읽기 전용으로 감사한 수치",
        metric_rows=metrics,
        table_rows=tuple(rows),
        figure=figure,
        readiness=_readiness(group_id, group),
        limitations_ko=profile.limitations,
    )


def _read_optional_map_json(path: Path) -> Any | None:
    """Best-effort read of an optional checked-in map asset; never raises.

    The SLC ingest metadata and materialization manifest are produced by
    background scripts, so absence, truncation or invalid JSON must degrade
    to "no map layer" instead of breaking the whole source EDA panel.
    """

    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > MAX_MAP_ASSET_BYTES:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _map_lat_lon(value: Any) -> tuple[float, float] | None:
    """Validate one ``[lat, lon]`` pair against the Busan plausibility box."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 2:
        return None
    try:
        lat = _number(value[0], field="map.lat")
        lon = _number(value[1], field="map.lon")
    except (TypeError, ValueError):
        return None
    if not _MAP_LAT_RANGE[0] <= lat <= _MAP_LAT_RANGE[1]:
        return None
    if not _MAP_LON_RANGE[0] <= lon <= _MAP_LON_RANGE[1]:
        return None
    return lat, lon


def _iceye_footprint_layer(meta_path: Path) -> Mapping[str, Any] | None:
    """Representative-scene footprint from the optional SLC ingest metadata.

    The four corners come from the single locally ingested SLC
    (SM_23467, 2020-03-02), so the polygon is labeled as one representative
    scene and never as the footprint of every acquisition.
    """

    payload = _read_optional_map_json(meta_path)
    if not isinstance(payload, Mapping):
        return None
    raw_corners = payload.get("footprint_latlon")
    if (
        not isinstance(raw_corners, Sequence)
        or isinstance(raw_corners, (str, bytes, bytearray))
        or len(raw_corners) != 4
    ):
        return None
    corners: list[tuple[float, float]] = []
    for raw_corner in raw_corners:
        pair = _map_lat_lon(raw_corner)
        if pair is None:
            return None
        corners.append(pair)
    date_value = payload.get("acquisition_date")
    if not isinstance(date_value, str) or not _ISO_DATE_RE.match(date_value.strip()):
        return None
    satellite = payload.get("satellite")
    if not isinstance(satellite, str) or not satellite.strip():
        return None
    try:
        heading = _number(payload.get("heading_deg"), field="map.heading_deg")
    except (TypeError, ValueError):
        return None
    if not 0.0 <= heading < 360.0:
        return None
    return MappingProxyType(
        {
            "corners": tuple(corners),
            "date": date_value.strip(),
            "satellite": satellite.strip(),
            "heading_deg": heading,
        }
    )


def _utm52n_bounds_to_corners(bounds: Any) -> tuple[tuple[float, float], ...] | None:
    """Convert one EPSG:32652 (left, bottom, right, top) box to lat/lon corners."""

    if (
        not isinstance(bounds, Sequence)
        or isinstance(bounds, (str, bytes, bytearray))
        or len(bounds) != 4
    ):
        return None
    try:
        left, bottom, right, top = (
            _number(value, field="map.bounds") for value in bounds
        )
    except (TypeError, ValueError):
        return None
    if not (left < right and bottom < top):
        return None
    try:
        from rasterio.warp import transform as _crs_transform
    except ImportError:
        return None
    try:
        lons, lats = _crs_transform(
            _UTM52N_CRS,
            _WGS84_CRS,
            [left, right, right, left],
            [bottom, bottom, top, top],
        )
    except Exception:  # rasterio raises library-specific CRS errors
        return None
    corners: list[tuple[float, float]] = []
    for lat, lon in zip(lats, lons, strict=True):
        pair = _map_lat_lon((lat, lon))
        if pair is None:
            return None
        corners.append(pair)
    return tuple(corners)


def _iceye_coverage_layers(manifest_path: Path) -> tuple[Mapping[str, Any], ...]:
    """Per-date label coverage rectangles from the materialization manifest.

    Prefers each item's audited native EPSG:32652 bounds; falls back to the
    single common-grid bounding box when no per-date block is usable.
    """

    payload = _read_optional_map_json(manifest_path)
    if not isinstance(payload, Mapping):
        return ()
    layers: list[Mapping[str, Any]] = []
    items = payload.get("items")
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        for raw_item in items:
            if not isinstance(raw_item, Mapping):
                continue
            date_value = raw_item.get("date")
            if not isinstance(date_value, str) or not _ISO_DATE_RE.match(
                date_value.strip()
            ):
                continue
            native = raw_item.get("native")
            block: Mapping[str, Any] | None = None
            if isinstance(native, Mapping):
                for key in ("label", "input"):
                    candidate = native.get(key)
                    if (
                        isinstance(candidate, Mapping)
                        and candidate.get("crs") == _UTM52N_CRS
                    ):
                        block = candidate
                        break
            if block is None:
                continue
            corners = _utm52n_bounds_to_corners(block.get("bounds"))
            if corners is None:
                continue
            layers.append(
                MappingProxyType({"date": date_value.strip(), "corners": corners})
            )
    if layers:
        return tuple(layers)
    common = payload.get("common_grid")
    if isinstance(common, Mapping) and common.get("crs") == _UTM52N_CRS:
        corners = _utm52n_bounds_to_corners(common.get("bounds"))
        if corners is not None:
            return (MappingProxyType({"date": "공통격자", "corners": corners}),)
    return ()


def _iceye_center_markers(group: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Acquisition center markers from the already validated profile group."""

    acquisitions = group.get("acquisitions")
    if not isinstance(acquisitions, Sequence) or isinstance(
        acquisitions, (str, bytes, bytearray)
    ):
        return ()
    common = group.get("common_acquisition")
    satellite = "ICEYE"
    if isinstance(common, Mapping):
        common_satellite = common.get("satellite")
        if isinstance(common_satellite, str) and common_satellite.strip():
            satellite = common_satellite.strip()
    markers: list[Mapping[str, Any]] = []
    for raw_item in acquisitions:
        if not isinstance(raw_item, Mapping):
            continue
        pair = _map_lat_lon(raw_item.get("center_lat_lon"))
        if pair is None:
            continue
        start = raw_item.get("start_utc")
        scene = raw_item.get("scene_id")
        if not isinstance(start, str) or len(start.strip()) < 10:
            continue
        if not isinstance(scene, str) or not scene.strip():
            continue
        side = str(raw_item.get("look_side", "")).strip().lower()
        try:
            heading: float | None = _number(
                raw_item.get("heading_deg"), field="map.heading_deg"
            )
        except (TypeError, ValueError):
            heading = None
        markers.append(
            MappingProxyType(
                {
                    "lat": pair[0],
                    "lon": pair[1],
                    "date": start.strip()[:10],
                    "scene": scene.strip(),
                    "satellite": satellite,
                    "look_side": side,
                    "heading_deg": heading,
                }
            )
        )
    return tuple(markers)


def build_iceye_footprint_map(
    group: Mapping[str, Any],
    *,
    meta_path: str | Path = ICEYE_META_PATH,
    manifest_path: str | Path = ICEYE_MANIFEST_PATH,
) -> go.Figure | None:
    """Busan acquisition-footprint map for the ICEYE source EDA panel.

    Layers degrade independently: a missing/invalid SLC metadata file drops
    only the representative footprint polygon, a missing manifest (or missing
    rasterio) drops only the label coverage rectangles, and the center markers
    come from the strict source profile. With no layer at all the function
    returns ``None`` so the caller can simply skip the map.
    """

    footprint = _iceye_footprint_layer(Path(meta_path))
    coverage = _iceye_coverage_layers(Path(manifest_path))
    markers = _iceye_center_markers(group)
    if footprint is None and not coverage and not markers:
        return None

    hover_footprint = ""
    if footprint is not None:
        hover_footprint = (
            f"대표 장면 {footprint['date']} · {footprint['satellite']}"
            f" · heading {footprint['heading_deg']:.1f}°"
        )
    marker_texts = tuple(
        f"{marker['date']} · {marker['scene']} · {marker['satellite']}"
        + (
            f" · heading {marker['heading_deg']:.1f}°"
            if marker["heading_deg"] is not None
            else ""
        )
        + (f" · look {marker['look_side']}" if marker["look_side"] else "")
        for marker in markers
    )
    coverage_texts = tuple(f"라벨 커버리지 {layer['date']}" for layer in coverage)
    try:
        _scan_for_exposure(list((hover_footprint, *marker_texts, *coverage_texts)))
    except ValueError:
        return None

    use_map_tiles = hasattr(go, "Scattermap")

    def _map_trace(
        lats: Sequence[float], lons: Sequence[float], **kwargs: Any
    ) -> Any:
        # ``go.Scattermap`` needs plotly>=5.24; older installs fall back to an
        # equirectangular ``go.Scatter`` (x=lon, y=lat) without map tiles.
        if use_map_tiles:
            return go.Scattermap(lat=list(lats), lon=list(lons), **kwargs)
        return go.Scatter(x=list(lons), y=list(lats), **kwargs)

    figure = go.Figure()
    all_lats: list[float] = []
    all_lons: list[float] = []
    for layer, hover in zip(coverage, coverage_texts, strict=True):
        ring = (*layer["corners"], layer["corners"][0])
        ring_lats = [lat for lat, _ in ring]
        ring_lons = [lon for _, lon in ring]
        all_lats.extend(ring_lats)
        all_lons.extend(ring_lons)
        figure.add_trace(
            _map_trace(
                ring_lats,
                ring_lons,
                mode="lines",
                fill="toself",
                fillcolor="rgba(245,158,11,0.07)",
                line={"color": "#f59e0b", "width": 1.5},
                name=f"라벨 커버리지 {layer['date']}",
                text=[hover] * len(ring),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if footprint is not None:
        ring = (*footprint["corners"], footprint["corners"][0])
        ring_lats = [lat for lat, _ in ring]
        ring_lons = [lon for _, lon in ring]
        all_lats.extend(ring_lats)
        all_lons.extend(ring_lons)
        figure.add_trace(
            _map_trace(
                ring_lats,
                ring_lons,
                mode="lines",
                fill="toself",
                fillcolor="rgba(34,211,238,0.14)",
                line={"color": "#22d3ee", "width": 2.5},
                name=f"대표 장면 footprint ({footprint['date']})",
                text=[hover_footprint] * len(ring),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if markers:
        marker_lats = [marker["lat"] for marker in markers]
        marker_lons = [marker["lon"] for marker in markers]
        all_lats.extend(marker_lats)
        all_lons.extend(marker_lons)
        figure.add_trace(
            _map_trace(
                marker_lats,
                marker_lons,
                mode="markers",
                marker={
                    "size": 11,
                    "color": [
                        "#f97316" if marker["look_side"] == "left" else "#a78bfa"
                        for marker in markers
                    ],
                },
                name="획득 중심 · 주황=left-looking",
                text=list(marker_texts),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    layout: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#cbd5e1"},
        "margin": {"l": 22, "r": 22, "t": 58, "b": 30},
        "legend": {"orientation": "h", "y": 1.12, "x": 0.0},
        "hovermode": "closest",
        "title": "ICEYE 촬영 footprint 지도 · 폴리곤은 대표 장면 (2020-03-02)",
    }
    if use_map_tiles:
        layout["map"] = {
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 8,
        }
    else:
        layout["plot_bgcolor"] = "rgba(2,6,23,.42)"
        layout["xaxis"] = {"title": {"text": "경도 (°E)"}}
        layout["yaxis"] = {
            "title": {"text": "위도 (°N)"},
            "scaleanchor": "x",
            "scaleratio": 1,
        }
    figure.update_layout(**layout)
    return figure


def _iceye_view(
    group_id: str,
    group: Mapping[str, Any],
    profile: NasSourceProfile,
) -> NasSourceEDAView:
    acquisitions = _sequence(group.get("acquisitions"), field="iceye.acquisitions")
    rows: list[Mapping[str, Any]] = []
    x: list[str] = []
    near: list[float] = []
    center: list[float] = []
    far: list[float] = []
    symbols: list[str] = []
    sides: list[str] = []
    for index, raw_acquisition in enumerate(acquisitions):
        item = _mapping(raw_acquisition, field=f"iceye.acquisitions[{index}]")
        incidence = _mapping(item.get("incidence_deg"), field="iceye.incidence_deg")
        grd_shape = _sequence(item.get("grd_shape"), field="iceye.grd_shape")
        slc_shape = _sequence(item.get("slc_shape"), field="iceye.slc_shape")
        quicklook_shape = _sequence(
            item.get("quicklook_shape"), field="iceye.quicklook_shape"
        )
        if not (len(grd_shape) == len(slc_shape) == len(quicklook_shape) == 2):
            raise ValueError("ICEYE shapes must contain height and width")
        start = _text(item.get("start_utc"), field="iceye.start_utc")
        side = _text(item.get("look_side"), field="iceye.look_side").lower()
        if side not in {"left", "right"}:
            raise ValueError("ICEYE look_side must be left or right")
        near_value = _number(incidence.get("near"), field="iceye.incidence.near")
        center_value = _number(incidence.get("center"), field="iceye.incidence.center")
        far_value = _number(incidence.get("far"), field="iceye.incidence.far")
        x.append(start[:10])
        near.append(near_value)
        center.append(center_value)
        far.append(far_value)
        sides.append(side)
        symbols.append("diamond" if side == "left" else "circle")
        incidence_label = f"{near_value:.2f} / {center_value:.2f} / {far_value:.2f}"
        grd_shape_label = _shape_text(grd_shape, field="iceye.grd_shape")
        slc_shape_label = _shape_text(slc_shape, field="iceye.slc_shape")
        quicklook_shape_label = _shape_text(
            quicklook_shape, field="iceye.quicklook_shape"
        )
        rows.append(
            _row(
                **{
                    "관측시작 (UTC)": start,
                    "Scene": _text(item.get("scene_id"), field="iceye.scene_id"),
                    "Look side": side,
                    "위성 look angle (°)": _number(
                        item.get("satellite_look_angle_deg"), field="iceye.look_angle"
                    ),
                    "입사각 near/center/far (°)": incidence_label,
                    "GRD 크기 (H×W)": grd_shape_label,
                    "GRD GCP": _integer(
                        item.get("grd_gcp_count"), field="iceye.gcp", non_negative=True
                    ),
                    "SLC 크기 (H×W)": slc_shape_label,
                    "Quicklook 크기": quicklook_shape_label,
                }
            )
        )
    if not center:
        raise ValueError("iceye.acquisitions must not be empty")
    unmatched = _sequence(
        _mapping(group.get("label_pairing"), field="iceye.label_pairing").get(
            "unmatched_raw_dates"
        ),
        field="iceye.unmatched_raw_dates",
    )
    metrics = (
        _metric(
            "원본 획득",
            f"{len(acquisitions)}회",
            "GRD·SLC·Quicklook은 같은 획득의 서로 다른 제품입니다.",
        ),
        _metric(
            "중심 입사각",
            f"{min(center):.2f}~{max(center):.2f}°",
            "입사각 차이는 SAR 밝기에 직접 영향을 줍니다.",
        ),
        _metric(
            "Look side",
            f"right {sides.count('right')} · left {sides.count('left')}",
            "좌·우 관측기하가 바뀌면 직접 픽셀 비교할 수 없습니다.",
        ),
        _metric(
            "라벨 없는 원본",
            f"{len(unmatched)}일",
            "현재 전달 라벨과 날짜가 맞지 않는 원본입니다.",
        ),
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=near,
            name="near",
            mode="lines+markers",
            line={"color": "#64748b", "dash": "dot"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=center,
            name="center",
            mode="lines+markers",
            marker={
                "size": 10,
                "symbol": symbols,
                "color": ["#f97316" if side == "left" else "#a78bfa" for side in sides],
            },
            line={"color": "#a78bfa", "width": 3},
            text=[f"look: {side}" for side in sides],
            hovertemplate="%{x}<br>center %{y:.2f}°<br>%{text}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=far,
            name="far",
            mode="lines+markers",
            line={"color": "#94a3b8", "dash": "dot"},
        )
    )
    figure.update_yaxes(title_text="입사각 (°)", rangemode="tozero")
    _style_figure(figure, title="ICEYE 획득기하 EDA · ◇는 left-looking 관측")
    return NasSourceEDAView(
        group_id=group_id,
        profile_group_key="iceye_raw",
        display_name_ko=_text(
            group.get("display_name_ko"), field="iceye.display_name_ko"
        ),
        audited_at_utc=profile.audited_at_utc,
        provenance_ko="NAS 원본 ICEYE XML·GeoTIFF geokey를 읽기 전용으로 감사한 수치",
        metric_rows=metrics,
        table_rows=tuple(rows),
        figure=figure,
        readiness=_readiness(group_id, group),
        limitations_ko=profile.limitations,
        map_figure=build_iceye_footprint_map(group),
    )


def _university_view(
    group_id: str,
    group: Mapping[str, Any],
    profile: NasSourceProfile,
) -> NasSourceEDAView:
    results = _sequence(
        group.get("reported_results"), field="university.reported_results"
    )
    rows: list[Mapping[str, Any]] = []
    locations: list[str] = []
    areas: list[float] = []
    levels: list[float] = []
    valid_json_count = 0
    for index, raw_result in enumerate(results):
        result = _mapping(raw_result, field=f"university.reported_results[{index}]")
        location = _text(result.get("location"), field="university.location")
        area = _number(
            result.get("water_area_km2"),
            field="university.water_area",
            non_negative=True,
        )
        level = _number(result.get("water_level_el_m"), field="university.water_level")
        wb_valid = result.get("water_body_json_valid")
        wl_valid = result.get("water_level_json_valid")
        if not isinstance(wb_valid, bool) or not isinstance(wl_valid, bool):
            raise TypeError("university JSON validity flags must be boolean")
        valid_json_count += int(wb_valid) + int(wl_valid)
        locations.append(location)
        areas.append(area)
        levels.append(level)
        roi_top = _sequence(
            result.get("roi_top_left_lat_lon"), field="university.roi_top"
        )
        roi_bottom = _sequence(
            result.get("roi_bottom_right_lat_lon"), field="university.roi_bottom"
        )
        roi_top_label = (
            f"{_number(roi_top[0], field='roi.lat'):.6f}, "
            f"{_number(roi_top[1], field='roi.lon'):.6f}"
        )
        roi_bottom_label = (
            f"{_number(roi_bottom[0], field='roi.lat'):.6f}, "
            f"{_number(roi_bottom[1], field='roi.lon'):.6f}"
        )
        rows.append(
            _row(
                **{
                    "지점": location,
                    "수체면적 (km²)": area,
                    "WB JSON": "정상" if wb_valid else "형식 오류",
                    "수위 (EL.m)": level,
                    "WL JSON": "정상" if wl_valid else "형식 오류",
                    "ROI 좌상단 (lat, lon)": roi_top_label,
                    "ROI 우하단 (lat, lon)": roi_bottom_label,
                }
            )
        )
    alignment = _mapping(
        group.get("alignment_audit"), field="university.alignment_audit"
    )
    offset = _sequence(
        alignment.get("image_to_dem_water_offset_pixels"), field="university.offset"
    )
    geojson = _mapping(group.get("geojson"), field="university.geojson")
    polygon_count = _integer(
        geojson.get("feature_count"), field="geojson.feature_count"
    )
    total_json_count = len(results) * 2
    metrics = (
        _metric("관측시점", "1회", "면적·수위 결과가 같은 한 날짜에만 존재합니다."),
        _metric("보고 지점", f"{len(results)}곳", "안동과 대청의 전달 결과입니다."),
        _metric(
            "정상 JSON",
            f"{valid_json_count}/{total_json_count}",
            "WB 두 파일은 trailing comma, WL 두 파일은 정상입니다.",
        ),
        _metric(
            "영상↔결과 정합",
            f"({float(offset[0]):+.1f}, {float(offset[1]):+.1f}) px",
            "원본과 DEM/WB tiepoint의 반 픽셀 차이입니다.",
        ),
        _metric(
            "수체 Polygon",
            f"{polygon_count:,}개",
            "광역 Pred_WB GeoJSON의 Polygon 수입니다.",
        ),
    )
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=locations,
            y=areas,
            name="보고 수체면적",
            marker_color="#22d3ee",
            text=[f"{value:.2f} km²" for value in areas],
            textposition="outside",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=locations,
            y=levels,
            name="보고 수위",
            mode="lines+markers",
            line={"color": "#a78bfa", "width": 3},
            marker={"size": 10},
        ),
        secondary_y=True,
    )
    figure.update_yaxes(
        title_text="수체면적 (km²)", rangemode="tozero", secondary_y=False
    )
    figure.update_yaxes(title_text="수위 (EL.m)", rangemode="tozero", secondary_y=True)
    _style_figure(
        figure, title="단일시점 보고 결과 · 서로 다른 단위와 지점을 분리해 해석"
    )
    return NasSourceEDAView(
        group_id=group_id,
        profile_group_key="university_single_date",
        display_name_ko=_text(
            group.get("display_name_ko"), field="university.display_name_ko"
        ),
        audited_at_utc=profile.audited_at_utc,
        provenance_ko="NAS 전달 JSON·GeoJSON·GeoTIFF를 읽기 전용으로 감사한 수치",
        metric_rows=metrics,
        table_rows=tuple(rows),
        figure=figure,
        readiness=_readiness(group_id, group),
        limitations_ko=profile.limitations
        + (_text(group.get("json_issue"), field="university.json_issue"),),
    )


def _worldview_view(
    group_id: str,
    group: Mapping[str, Any],
    profile: NasSourceProfile,
    *,
    profile_key: str,
) -> NasSourceEDAView:
    acquisitions = _sequence(
        group.get("acquisitions"), field=f"{profile_key}.acquisitions"
    )
    common = _mapping(
        group.get("common_product"), field=f"{profile_key}.common_product"
    )
    common_sensor = common.get("satellite")
    rows: list[Mapping[str, Any]] = []
    labels: list[str] = []
    mul_megapixels: list[float] = []
    pan_megapixels: list[float] = []
    sensors: list[str] = []
    dates: list[str] = []
    for index, raw_acquisition in enumerate(acquisitions):
        item = _mapping(raw_acquisition, field=f"{profile_key}.acquisitions[{index}]")
        acquired = _text(item.get("acquired_utc"), field=f"{profile_key}.acquired_utc")
        pair_id = _text(item.get("pair_id"), field=f"{profile_key}.pair_id")
        sensor = _text(
            item.get("satellite", common_sensor), field=f"{profile_key}.satellite"
        )
        mul_shape = _sequence(
            item.get("multispectral_shape"), field=f"{profile_key}.mul_shape"
        )
        pan_shape = _sequence(
            item.get("panchromatic_shape"), field=f"{profile_key}.pan_shape"
        )
        if len(mul_shape) != 2 or len(pan_shape) != 2:
            raise ValueError("WorldView shapes must contain height and width")
        mul_h = _integer(mul_shape[0], field="mul.h", non_negative=True)
        mul_w = _integer(mul_shape[1], field="mul.w", non_negative=True)
        pan_h = _integer(pan_shape[0], field="pan.h", non_negative=True)
        pan_w = _integer(pan_shape[1], field="pan.w", non_negative=True)
        gsd = _mapping(
            item.get("collected_gsd_m"), field=f"{profile_key}.collected_gsd"
        )
        labels.append(f"{acquired[:10]} · {pair_id}")
        dates.append(acquired[:10])
        sensors.append(sensor)
        mul_megapixels.append(mul_h * mul_w / 1_000_000.0)
        pan_megapixels.append(pan_h * pan_w / 1_000_000.0)
        mul_gsd = _number(gsd.get("multispectral"), field="gsd.mul")
        pan_gsd = _number(gsd.get("panchromatic"), field="gsd.pan")
        rows.append(
            _row(
                **{
                    "관측시각 (UTC)": acquired,
                    "Pair": pair_id,
                    "Catalog ID": _text(
                        item.get("catalog_id"), field=f"{profile_key}.catalog_id"
                    ),
                    "센서": sensor,
                    "MUL 크기 (H×W)": f"{mul_h} × {mul_w}",
                    "PAN 크기 (H×W)": f"{pan_h} × {pan_w}",
                    "수집 GSD MUL/PAN (m)": f"{mul_gsd:.3f} / {pan_gsd:.3f}",
                    "Cloud (%)": _number(
                        item.get("cloud_cover_percent"),
                        field="cloud",
                        non_negative=True,
                    ),
                    "Off-nadir (°)": _number(
                        item.get("off_nadir_deg"), field="off_nadir", non_negative=True
                    ),
                    "PNIIRS": _number(
                        item.get("pniirs"), field="pniirs", non_negative=True
                    ),
                }
            )
        )
    if not rows:
        raise ValueError(f"{profile_key}.acquisitions must not be empty")
    unique_dates = tuple(dict.fromkeys(dates))
    sensor_label = " · ".join(dict.fromkeys(sensors))
    metrics = (
        _metric(
            "고유 관측일",
            f"{len(unique_dates)}일",
            "같은 날짜의 부분영상을 중복 시점으로 세지 않습니다.",
        ),
        _metric(
            "PAN·MUL Pair",
            f"{len(rows)}쌍",
            "한 pair는 같은 획득의 전정색·다중분광 제품입니다.",
        ),
        _metric(
            "센서", sensor_label, "센서가 섞이면 날짜 간 방사 정규화가 필요합니다."
        ),
        _metric(
            "분석 밴드",
            "MUL 8 · PAN 1",
            "MUL NIR 계열로 수체 탐지가 가능하지만 전달 수체 라벨은 없습니다.",
        ),
    )
    figure = go.Figure()
    figure.add_trace(
        go.Bar(x=labels, y=mul_megapixels, name="MUL 픽셀", marker_color="#22d3ee")
    )
    figure.add_trace(
        go.Bar(x=labels, y=pan_megapixels, name="PAN 픽셀", marker_color="#a78bfa")
    )
    figure.update_layout(barmode="group")
    figure.update_yaxes(title_text="영상 크기 (Megapixels)", rangemode="tozero")
    _style_figure(figure, title="WorldView PAN·MUL 획득별 영상 크기 · pair는 한 시점")
    return NasSourceEDAView(
        group_id=group_id,
        profile_group_key=profile_key,
        display_name_ko=_text(
            group.get("display_name_ko"), field=f"{profile_key}.display_name_ko"
        ),
        audited_at_utc=profile.audited_at_utc,
        provenance_ko=(
            "NAS 원본 WorldView IMD·GeoTIFF geokey를 읽기 전용으로 감사한 수치"
        ),
        metric_rows=metrics,
        table_rows=tuple(rows),
        figure=figure,
        readiness=_readiness(group_id, group),
        limitations_ko=profile.limitations,
    )


def build_nas_source_eda(
    group_id: str,
    *,
    profile: NasSourceProfile | None = None,
    profile_path: str | Path = NAS_SOURCE_PROFILE_PATH,
) -> NasSourceEDAView:
    """Build metrics, table rows, readiness text and one figure for a group."""

    try:
        profile_key = PROFILE_KEY_BY_CATALOG_GROUP[group_id]
    except KeyError as exc:
        available = ", ".join(PROFILE_KEY_BY_CATALOG_GROUP)
        raise KeyError(
            f"no source EDA for {group_id!r}; available: {available}"
        ) from exc
    source_profile = profile or load_nas_source_profile(profile_path)
    group = _mapping(
        source_profile.groups.get(profile_key), field=f"groups.{profile_key}"
    )
    if profile_key == "planetscope_raw":
        view = _planetscope_view(group_id, group, source_profile)
    elif profile_key == "iceye_raw":
        view = _iceye_view(group_id, group, source_profile)
    elif profile_key == "university_single_date":
        view = _university_view(group_id, group, source_profile)
    else:
        view = _worldview_view(
            group_id,
            group,
            source_profile,
            profile_key=profile_key,
        )
    _scan_for_exposure(
        {
            "display_name": view.display_name_ko,
            "provenance": view.provenance_ko,
            "metrics": [dict(row) for row in view.metric_rows],
            "rows": [dict(row) for row in view.table_rows],
            "readiness": {
                "headline": view.readiness.headline_ko,
                "explanation": view.readiness.explanation_ko,
                "steps": list(view.readiness.next_steps_ko),
                "cautions": list(view.readiness.cautions_ko),
            },
        }
    )
    return view


def source_metric_rows(
    group_id: str,
    **kwargs: Any,
) -> tuple[Mapping[str, str], ...]:
    """Convenience wrapper for renderers that only need metric cards."""

    return build_nas_source_eda(group_id, **kwargs).metric_rows


def source_table_rows(
    group_id: str,
    **kwargs: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Convenience wrapper for renderers that only need a dataframe payload."""

    return build_nas_source_eda(group_id, **kwargs).table_rows


def build_nas_source_figure(
    group_id: str,
    **kwargs: Any,
) -> go.Figure:
    """Convenience wrapper for renderers that only need the Plotly figure."""

    return build_nas_source_eda(group_id, **kwargs).figure
