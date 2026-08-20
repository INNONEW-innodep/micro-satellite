"""Quantitative evaluation views for the delivered real-model evidence.

This module reads only local, checked-in delivery evidence: the handover
validation reports (``07_test_evidence/VALREPORT_*.json``), the local smoke
re-inference summary (``data/wb_smoke/smoke_eval.json``), optional future
batch outputs (``data/eval/*.json`` and ``data/wbms_runs/fused/*.json``),
the WAMIS gauge water-level table and the Busan AWS weather CSV from the
handover package.

Three separate metric series must never be blended into one headline number:

* ``연구 프로토콜`` — numbers quoted in the handover README (research grid).
* ``배포 프로토콜`` — VALREPORT measurements on the deployment UTM grid.
* ``스모크 재현``   — the local re-inference smoke run.

Pooled metrics and the in-sample gauge table are additionally labelled so
they cannot masquerade as representative generalization performance.

Every loader is best-effort: a missing or malformed file degrades to a typed
status (``missing`` / ``invalid`` / ``waiting``) with a Korean message and
never raises into the Streamlit caller.  Absolute filesystem paths are never
exposed: provenance strings are repo-relative or bare file names.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = _REPO_ROOT / "data"
HANDOVER_DIR: Final[Path] = DATA_DIR / "incoming" / "handover"
EVIDENCE_DIR: Final[Path] = HANDOVER_DIR / "07_test_evidence"
AUX_DIR: Final[Path] = HANDOVER_DIR / "06_aux"

VALREPORT_FILENAMES: Final[tuple[str, ...]] = (
    "VALREPORT_Busan_ICEYE.json",
    "VALREPORT_Busan_PlanetScope.json",
)
SMOKE_EVAL_PATH: Final[Path] = DATA_DIR / "wb_smoke" / "smoke_eval.json"
BATCH_EVAL_DIR: Final[Path] = DATA_DIR / "eval"
FUSED_RUN_DIR: Final[Path] = DATA_DIR / "wbms_runs" / "fused"
GAUGE_CSV_PATH: Final[Path] = AUX_DIR / "deploy_train_wamis_v3_finalwb.csv"
WEATHER_CSV_PATH: Final[Path] = AUX_DIR / "busan_aws_2019_202004.csv"

MAX_EVIDENCE_BYTES: Final[int] = 2 * 1024 * 1024

# --- metric-series discipline -------------------------------------------------
PROTOCOL_RESEARCH_KO: Final[str] = "연구 프로토콜 · 인수문서 기재"
PROTOCOL_DEPLOY_KO: Final[str] = "배포 프로토콜 · VALREPORT 실측"
PROTOCOL_SMOKE_KO: Final[str] = "스모크 재현 · 로컬 실추론"
PROTOCOL_BATCH_KO: Final[str] = "배치 산출 · 체인 러너"
PROTOCOL_FUSED_KO: Final[str] = "융합 보정 · 당일 보정치"
POOLED_CAUTION_KO: Final[str] = (
    "pooled 값은 4개 씬 픽셀 합산 지표이며 대표 성능이 아닙니다. 씬별 편차를 함께 보세요."
)
IN_SAMPLE_LABEL_KO: Final[str] = "in-sample · 융합 LSTM 학습표본과 동일"
BATCH_WAITING_MESSAGE_KO: Final[str] = (
    "배치 대기 중 · 체인 러너 산출(data/eval/*.json)이 아직 없습니다."
)
FUSED_WAITING_MESSAGE_KO: Final[str] = (
    "융합 보정 수위 산출(data/wbms_runs/fused/)이 아직 없어 실측 대조 열을 생략합니다."
)

# 시험씬(배포 프로토콜)과 연구 프로토콜 문서값.  연구 수치는 배포 UTM 그리드가
# 아니라 연구 격자·경로에서 계산된 값이므로 씬이 같아도 수치가 다른 것이 정상입니다.
DEPLOY_TEST_SCENES: Final[Mapping[str, str]] = MappingProxyType(
    {"SAR": "2020-04-16", "OPTIC": "2020-03-25"}
)
RESEARCH_PROTOCOL_IOUS: Final[Mapping[str, float]] = MappingProxyType(
    {"SAR": 0.8949, "OPTIC": 0.9487}
)
RESEARCH_PROVENANCE_KO: Final[str] = (
    "data/incoming/handover/00_README_인수문서.md 기재 · 연구 격자/경로"
)

# FUSED 증적(run 20260817T183137Z)이 참조한 AWS CSV의 sha256.  화면에서 현재
# CSV가 같은 파일인지 확인하는 근거로만 사용합니다.
FUSED_RECORDED_AWS_SHA256: Final[str] = (
    "4cccfaadc42976f5040c5cf85748e198f849ca833b21da82507a6c21f9218377"
)

# loc_id 매핑 근거: CSV의 median_value가 FUSED_Busan_<이름>.json records의
# water_level_m와 날짜·위성별로 정확히 일치합니다 (07_test_evidence 대조).
GAUGE_LOC_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {0: "jeongcheon", 1: "hupo", 2: "gimhae", 3: "gupo"}
)
SATELLITE_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {0: "PlanetScope", 1: "ICEYE"}
)

GAUGE_CSV_HEADER: Final[tuple[str, ...]] = (
    "date",
    "file",
    "median_value",
    "n_boundary_pixels",
    "out_h",
    "out_w",
    "sat_type",
    "loc_id",
    "real_water_level",
)

_SCENE_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^\d{8}$")
_WB_FILE_RE: Final[re.Pattern[str]] = re.compile(
    r"^WB_[A-Za-z]+_([A-Za-z0-9]+)_(\d{8})T\d{6}\.tif$"
)
_METRIC_KEYS: Final[tuple[str, ...]] = (
    "water_iou",
    "accuracy",
    "recall",
    "precision",
    "f1",
)

_DARK_LAYOUT: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(2,6,23,.42)",
        "font": {"color": "#cbd5e1"},
        "hovermode": "x unified",
        "margin": {"l": 22, "r": 22, "t": 58, "b": 30},
        "legend": {"orientation": "h", "y": 1.14, "x": 0.0},
    }
)


def _relative_provenance(path: Path) -> str:
    """Return a repo-relative provenance label, never an absolute path."""

    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _read_small_text(path: Path) -> str:
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError(f"evidence file too large: {path.name}")
    return path.read_text(encoding="utf-8")


def _iso_scene_date(raw: Any) -> str:
    text = str(raw)
    if not _SCENE_DATE_RE.match(text):
        raise ValueError(f"scene date must be 8 digits: {text!r}")
    return dt.datetime.strptime(text, "%Y%m%d").date().isoformat()


def _unit_metric(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"metric {key!r} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"metric {key!r} out of [0, 1]: {number}")
    return number


@dataclass(frozen=True, slots=True)
class SceneMetric:
    """One evaluated scene (or the pooled aggregate) of a validation report."""

    date: str
    water_iou: float
    accuracy: float
    recall: float
    precision: float
    f1: float


@dataclass(frozen=True, slots=True)
class ValReport:
    """One deployment-protocol validation report, or its degraded stand-in."""

    status: str  # "ok" | "missing" | "invalid"
    message_ko: str
    protocol_label_ko: str
    provenance: str
    sensor_type: str = ""
    sensor_name: str = ""
    testset_id: str = ""
    n_scenes: int = 0
    n_evaluated: int = 0
    model_name: str = ""
    model_sha256: str = ""
    gt_md5: str = ""
    threshold: float | None = None
    pooled: SceneMetric | None = None
    per_scene: tuple[SceneMetric, ...] = ()
    pooled_caution_ko: str = POOLED_CAUTION_KO


@dataclass(frozen=True, slots=True)
class SmokeEval:
    """The local re-inference smoke summary, kept as its own series."""

    status: str  # "ok" | "missing" | "invalid"
    message_ko: str
    protocol_label_ko: str
    provenance: str
    sensor_name: str = ""
    scene_date: str = ""
    metrics: SceneMetric | None = None
    valid_pixels: int = 0


@dataclass(frozen=True, slots=True)
class BatchEvalReport:
    """One future chain-runner evaluation JSON parsed best-effort."""

    name: str
    sensor_name: str
    scene_date: str
    water_iou: float
    f1: float | None
    provenance: str


@dataclass(frozen=True, slots=True)
class BatchEvalCollection:
    """All chain-runner outputs, or the explicit waiting state."""

    status: str  # "ok" | "waiting"
    message_ko: str
    protocol_label_ko: str
    provenance: str
    reports: tuple[BatchEvalReport, ...] = ()
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GaugeLevelRow:
    """One gauge observation aligned to one satellite water-mask date."""

    date: str
    satellite: str
    loc_id: int
    loc_name: str
    real_level_m: float
    boundary_median: float
    fused_level_m: float | None = None
    fused_delta_m: float | None = None
    # 아래는 FUSED 레코드에서 그대로 옮긴 값이다. offset은 위성 수위를 절대 수위
    # 체계로 옮긴 정렬량이라 모델 오차가 아니고, paired=False와 학습범위 이탈은
    # 숨기지 않고 행에 표시한다.
    satellite_level_m: float | None = None
    offset_m: float | None = None
    correction_mode: str = ""
    paired: bool | None = None
    out_of_train_range: bool | None = None


@dataclass(frozen=True, slots=True)
class GaugeLevelTable:
    """The parsed in-sample gauge table plus optional fused comparison."""

    status: str  # "ok" | "missing" | "invalid"
    message_ko: str
    provenance: str
    in_sample: bool = True
    in_sample_label_ko: str = IN_SAMPLE_LABEL_KO
    rows: tuple[GaugeLevelRow, ...] = ()
    n_samples: int = 0
    level_min_m: float = 0.0
    level_max_m: float = 0.0
    date_min: str = ""
    date_max: str = ""
    has_fused: bool = False
    fused_status: str = "missing"
    fused_message_ko: str = FUSED_WAITING_MESSAGE_KO
    fused_provenance: str = ""


@dataclass(frozen=True, slots=True)
class WeatherDailyRow:
    """One day of the Busan AWS record; missing cells stay ``None``."""

    date: str
    temperature_c: float | None
    precipitation_mm: float | None
    humidity_pct: float | None


@dataclass(frozen=True, slots=True)
class WeatherSeries:
    """The CP949 AWS daily table backing the 60-day fusion window."""

    status: str  # "ok" | "missing" | "invalid"
    message_ko: str
    provenance: str
    encoding: str = ""
    station_ids: tuple[str, ...] = ()
    rows: tuple[WeatherDailyRow, ...] = ()
    n_days: int = 0
    date_min: str = ""
    date_max: str = ""
    missing_temperature: int = 0
    missing_precipitation: int = 0
    missing_humidity: int = 0
    sha256: str = ""
    sha256_matches_fused_evidence: bool = field(default=False)


# --- validation reports -------------------------------------------------------


def _parse_scene_metric(payload: Mapping[str, Any], *, date_label: str) -> SceneMetric:
    return SceneMetric(
        date=date_label,
        water_iou=_unit_metric(payload, "water_iou"),
        accuracy=_unit_metric(payload, "accuracy"),
        recall=_unit_metric(payload, "recall"),
        precision=_unit_metric(payload, "precision"),
        f1=_unit_metric(payload, "f1"),
    )


def _load_one_valreport(path: Path) -> ValReport:
    provenance = _relative_provenance(path)
    if not path.is_file():
        return ValReport(
            status="missing",
            message_ko=f"인수 패키지 검증 리포트를 찾지 못했습니다: {path.name}",
            protocol_label_ko=PROTOCOL_DEPLOY_KO,
            provenance=provenance,
        )
    try:
        payload = json.loads(_read_small_text(path))
        if not isinstance(payload, Mapping):
            raise ValueError("VALREPORT must be a JSON object")
        if payload.get("mode") != "with_gt":
            raise ValueError("mode must be 'with_gt'")
        sensor_type = str(payload["sensor_type"])
        if sensor_type not in {"SAR", "OPTIC"}:
            raise ValueError(f"unknown sensor_type: {sensor_type!r}")
        per_scene_raw = payload["per_scene"]
        if not isinstance(per_scene_raw, list) or not per_scene_raw:
            raise ValueError("per_scene must be a non-empty list")
        n_evaluated = int(payload["n_evaluated"])
        if n_evaluated != len(per_scene_raw):
            raise ValueError("n_evaluated does not match per_scene length")
        per_scene = tuple(
            _parse_scene_metric(scene, date_label=_iso_scene_date(scene["date"]))
            for scene in per_scene_raw
        )
        pooled = _parse_scene_metric(
            payload["metrics"], date_label=f"pooled({len(per_scene)}씬)"
        )
        model = payload.get("model")
        model_path = str(model["path"]) if isinstance(model, Mapping) else ""
        model_sha = str(model.get("sha256", "")) if isinstance(model, Mapping) else ""
        threshold = float(payload["threshold"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return ValReport(
            status="invalid",
            message_ko=f"검증 리포트 형식이 계약과 다릅니다 ({path.name}): {exc}",
            protocol_label_ko=PROTOCOL_DEPLOY_KO,
            provenance=provenance,
        )
    return ValReport(
        status="ok",
        message_ko="배포 프로토콜 검증 리포트를 읽었습니다.",
        protocol_label_ko=PROTOCOL_DEPLOY_KO,
        provenance=provenance,
        sensor_type=sensor_type,
        sensor_name=str(payload.get("sensor_name", "")),
        testset_id=str(payload.get("testset_id", "")),
        n_scenes=int(payload.get("n_scenes", len(per_scene))),
        n_evaluated=n_evaluated,
        model_name=PurePosixPath(model_path).name if model_path else "",
        model_sha256=model_sha,
        gt_md5=str(payload.get("gt_md5", "")),
        threshold=threshold,
        pooled=pooled,
        per_scene=per_scene,
    )


def load_valreports(evidence_dir: Path | None = None) -> tuple[ValReport, ...]:
    """Load both deployment-protocol validation reports best-effort."""

    directory = evidence_dir if evidence_dir is not None else EVIDENCE_DIR
    return tuple(_load_one_valreport(directory / name) for name in VALREPORT_FILENAMES)


# --- smoke re-inference -------------------------------------------------------


def load_smoke_eval(path: Path | None = None) -> SmokeEval:
    """Load the local smoke re-inference summary or degrade explicitly."""

    smoke_path = path if path is not None else SMOKE_EVAL_PATH
    provenance = _relative_provenance(smoke_path)
    if not smoke_path.is_file():
        return SmokeEval(
            status="missing",
            message_ko="스모크 재현 산출(data/wb_smoke/smoke_eval.json)이 아직 없습니다.",
            protocol_label_ko=PROTOCOL_SMOKE_KO,
            provenance=provenance,
        )
    try:
        payload = json.loads(_read_small_text(smoke_path))
        if not isinstance(payload, Mapping):
            raise ValueError("smoke_eval must be a JSON object")
        metrics_raw = payload["metrics"]
        if not isinstance(metrics_raw, Mapping):
            raise ValueError("metrics must be an object")
        wb_name = PurePosixPath(str(payload.get("wb", ""))).name
        matched = _WB_FILE_RE.match(wb_name)
        if matched is None:
            raise ValueError(f"unrecognized WB file name: {wb_name!r}")
        sensor_name, scene_date_raw = matched.groups()
        counts = metrics_raw.get("counts")
        valid_pixels = (
            int(counts["valid"]) if isinstance(counts, Mapping) and "valid" in counts else 0
        )
        if valid_pixels < 0:
            raise ValueError("valid pixel count must not be negative")
        metrics = _parse_scene_metric(
            metrics_raw, date_label=_iso_scene_date(scene_date_raw)
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return SmokeEval(
            status="invalid",
            message_ko=f"스모크 산출 형식이 계약과 다릅니다: {exc}",
            protocol_label_ko=PROTOCOL_SMOKE_KO,
            provenance=provenance,
        )
    return SmokeEval(
        status="ok",
        message_ko="스모크 재현 지표를 읽었습니다.",
        protocol_label_ko=PROTOCOL_SMOKE_KO,
        provenance=provenance,
        sensor_name=sensor_name,
        scene_date=metrics.date,
        metrics=metrics,
        valid_pixels=valid_pixels,
    )


# --- future batch outputs -----------------------------------------------------


def load_batch_evals(batch_dir: Path | None = None) -> BatchEvalCollection:
    """Collect chain-runner outputs; absence is the normal waiting state."""

    directory = batch_dir if batch_dir is not None else BATCH_EVAL_DIR
    provenance = _relative_provenance(directory)
    if not directory.is_dir():
        return BatchEvalCollection(
            status="waiting",
            message_ko=BATCH_WAITING_MESSAGE_KO,
            protocol_label_ko=PROTOCOL_BATCH_KO,
            provenance=provenance,
        )
    json_paths = sorted(directory.glob("*.json"))
    if not json_paths:
        return BatchEvalCollection(
            status="waiting",
            message_ko=BATCH_WAITING_MESSAGE_KO,
            protocol_label_ko=PROTOCOL_BATCH_KO,
            provenance=provenance,
        )
    reports: list[BatchEvalReport] = []
    skipped: list[str] = []
    for json_path in json_paths:
        try:
            payload = json.loads(_read_small_text(json_path))
            if not isinstance(payload, Mapping):
                raise ValueError("batch eval must be a JSON object")
            metrics = payload["metrics"]
            if not isinstance(metrics, Mapping):
                raise ValueError("metrics must be an object")
            water_iou = _unit_metric(metrics, "water_iou")
            f1 = (
                _unit_metric(metrics, "f1") if "f1" in metrics else None
            )
            wb_name = PurePosixPath(str(payload.get("wb", ""))).name
            matched = _WB_FILE_RE.match(wb_name)
            sensor_name = matched.group(1) if matched else ""
            scene_date = _iso_scene_date(matched.group(2)) if matched else ""
        except (KeyError, TypeError, ValueError, OSError):
            skipped.append(json_path.name)
            continue
        reports.append(
            BatchEvalReport(
                name=json_path.stem,
                sensor_name=sensor_name,
                scene_date=scene_date,
                water_iou=water_iou,
                f1=f1,
                provenance=_relative_provenance(json_path),
            )
        )
    if not reports:
        return BatchEvalCollection(
            status="waiting",
            message_ko=(
                BATCH_WAITING_MESSAGE_KO
                + f" 형식이 다른 파일 {len(skipped)}건은 건너뛰었습니다."
            ),
            protocol_label_ko=PROTOCOL_BATCH_KO,
            provenance=provenance,
            skipped=tuple(skipped),
        )
    return BatchEvalCollection(
        status="ok",
        message_ko=f"배치 산출 {len(reports)}건을 읽었습니다.",
        protocol_label_ko=PROTOCOL_BATCH_KO,
        provenance=provenance,
        reports=tuple(reports),
        skipped=tuple(skipped),
    )


# --- gauge water levels -------------------------------------------------------


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_fused_corrections(
    fused_dir: Path,
) -> tuple[str, str, str, Mapping[tuple[str, str, str], Mapping[str, Any]]]:
    """Read fused-run corrections keyed by (loc, satellite, date).

    Each entry keeps the whole correction context, not just the corrected
    level: the uncorrected satellite level, the alignment offset, whether a
    cross-sensor pair existed and whether the level fell outside the training
    range. The panel has to show those, so dropping them here would hide a
    WARN the viewer is entitled to see.
    """

    provenance = _relative_provenance(fused_dir)
    if not fused_dir.is_dir():
        return "missing", FUSED_WAITING_MESSAGE_KO, provenance, MappingProxyType({})
    # 체인 러너 계약: fused/<station>/FUSED_Busan_<station>.json (지점별 하위 디렉터리)
    json_paths = sorted(fused_dir.glob("*/FUSED_*.json"))
    if not json_paths:
        return "missing", FUSED_WAITING_MESSAGE_KO, provenance, MappingProxyType({})
    corrections: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    parsed_files = 0
    for json_path in json_paths:
        try:
            payload = json.loads(_read_small_text(json_path))
            records = payload["records"] if isinstance(payload, Mapping) else None
            if not isinstance(records, list):
                raise ValueError("records must be a list")
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError("record must be an object")
                corrected = _optional_number(record.get("corrected_water_level_m"))
                if corrected is None:
                    continue
                key = (
                    str(record["loc_id"]).strip().lower(),
                    str(record["satellite"]).strip(),
                    _iso_scene_date(record["date"]),
                )
                paired = record.get("paired")
                out_of_range = record.get("wl_out_of_train_range")
                corrections[key] = MappingProxyType(
                    {
                        "corrected_water_level_m": corrected,
                        "water_level_m": _optional_number(record.get("water_level_m")),
                        "offset_m": _optional_number(record.get("offset_m")),
                        "correction_mode": str(record.get("correction_mode") or "").strip(),
                        "paired": paired if isinstance(paired, bool) else None,
                        "wl_out_of_train_range": (
                            out_of_range if isinstance(out_of_range, bool) else None
                        ),
                    }
                )
        except (KeyError, TypeError, ValueError, OSError):
            continue
        parsed_files += 1
    if not corrections:
        return (
            "missing",
            FUSED_WAITING_MESSAGE_KO + " 읽을 수 있는 보정 레코드가 없습니다.",
            provenance,
            MappingProxyType({}),
        )
    return (
        "ok",
        f"융합 보정 수위 {len(corrections)}건({parsed_files}개 파일)을 대조합니다.",
        provenance,
        MappingProxyType(corrections),
    )


def load_gauge_levels(
    csv_path: Path | None = None,
    fused_dir: Path | None = None,
) -> GaugeLevelTable:
    """Parse the in-sample WAMIS gauge table and optionally merge fused runs."""

    table_path = csv_path if csv_path is not None else GAUGE_CSV_PATH
    provenance = _relative_provenance(table_path)
    if not table_path.is_file():
        return GaugeLevelTable(
            status="missing",
            message_ko=f"게이지 실측 수위 CSV를 찾지 못했습니다: {table_path.name}",
            provenance=provenance,
        )
    try:
        text = _read_small_text(table_path)
        reader = csv.reader(text.splitlines())
        header = tuple(next(reader, ()))
        if header != GAUGE_CSV_HEADER:
            raise ValueError(f"unexpected gauge CSV header: {header}")
        raw_rows: list[GaugeLevelRow] = []
        for line_number, cells in enumerate(reader, start=2):
            if not cells or all(not cell.strip() for cell in cells):
                continue
            if len(cells) != len(GAUGE_CSV_HEADER):
                raise ValueError(f"row {line_number}: wrong column count")
            record = dict(zip(GAUGE_CSV_HEADER, cells, strict=True))
            observed_on = dt.date.fromisoformat(record["date"].strip()).isoformat()
            if _WB_FILE_RE.match(record["file"].strip()) is None:
                raise ValueError(f"row {line_number}: unrecognized WB file name")
            sat_type = int(record["sat_type"])
            if sat_type not in SATELLITE_NAMES:
                raise ValueError(f"row {line_number}: unknown sat_type {sat_type}")
            loc_id = int(record["loc_id"])
            if loc_id not in GAUGE_LOC_NAMES:
                raise ValueError(f"row {line_number}: unknown loc_id {loc_id}")
            level = float(record["real_water_level"])
            if not 0.0 < level < 50.0:
                raise ValueError(f"row {line_number}: implausible level {level}")
            raw_rows.append(
                GaugeLevelRow(
                    date=observed_on,
                    satellite=SATELLITE_NAMES[sat_type],
                    loc_id=loc_id,
                    loc_name=GAUGE_LOC_NAMES[loc_id],
                    real_level_m=level,
                    boundary_median=float(record["median_value"]),
                )
            )
        if not raw_rows:
            raise ValueError("gauge CSV has no data rows")
    except (StopIteration, TypeError, ValueError, OSError) as exc:
        return GaugeLevelTable(
            status="invalid",
            message_ko=f"게이지 CSV 형식이 계약과 다릅니다: {exc}",
            provenance=provenance,
        )

    fused_status, fused_message, fused_provenance, corrections = _load_fused_corrections(
        fused_dir if fused_dir is not None else FUSED_RUN_DIR
    )
    merged: list[GaugeLevelRow] = []
    matched = 0
    for row in sorted(raw_rows, key=lambda item: (item.date, item.satellite, item.loc_id)):
        entry = corrections.get((row.loc_name, row.satellite, row.date))
        if entry is None:
            merged.append(row)
            continue
        fused_level = float(entry["corrected_water_level_m"])
        matched += 1
        merged.append(
            GaugeLevelRow(
                date=row.date,
                satellite=row.satellite,
                loc_id=row.loc_id,
                loc_name=row.loc_name,
                real_level_m=row.real_level_m,
                boundary_median=row.boundary_median,
                fused_level_m=fused_level,
                fused_delta_m=fused_level - row.real_level_m,
                satellite_level_m=entry["water_level_m"],
                offset_m=entry["offset_m"],
                correction_mode=entry["correction_mode"],
                paired=entry["paired"],
                out_of_train_range=entry["wl_out_of_train_range"],
            )
        )
    levels = [row.real_level_m for row in merged]
    return GaugeLevelTable(
        status="ok",
        message_ko=(
            f"게이지 실측 수위 {len(merged)}표본을 읽었습니다. "
            "이 표본은 융합 LSTM 학습표본과 동일한 in-sample 값입니다."
        ),
        provenance=provenance,
        rows=tuple(merged),
        n_samples=len(merged),
        level_min_m=min(levels),
        level_max_m=max(levels),
        date_min=merged[0].date,
        date_max=merged[-1].date,
        has_fused=matched > 0,
        fused_status=fused_status if matched > 0 else "missing",
        fused_message_ko=(
            fused_message if matched > 0 else FUSED_WAITING_MESSAGE_KO
        ),
        fused_provenance=fused_provenance if matched > 0 else "",
    )


def gauge_pivot_rows(table: GaugeLevelTable) -> tuple[Mapping[str, Any], ...]:
    """Return date×satellite rows with one measured-level column per gauge."""

    if table.status != "ok":
        return ()
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in table.rows:
        key = (row.date, row.satellite)
        entry = grouped.setdefault(
            key,
            {
                "관측 날짜": row.date,
                "위성": row.satellite,
                **{
                    f"{GAUGE_LOC_NAMES[loc_id]} (m)": None
                    for loc_id in sorted(GAUGE_LOC_NAMES)
                },
            },
        )
        entry[f"{row.loc_name} (m)"] = row.real_level_m
    ordered = sorted(grouped.items(), key=lambda item: item[0])
    return tuple(MappingProxyType(entry) for _, entry in ordered)


FUSED_SECTION_HEADER_KO: Final[str] = "융합 LSTM 보정 수위 — 당일 보정"
FUSED_BADGE_KO: Final[str] = "당일 보정 · 예측 아님"
FUSED_SOLO_SENSOR_KO: Final[str] = "단독 센서"
FUSED_NOTICE_LINES_KO: Final[tuple[str, ...]] = (
    "보정 수위는 관측일의 위성 수위를 게이지 절대 수위 체계로 정렬한 당일 "
    "보정치입니다 — 미래 예측이 아닙니다. (correction_mode=absolute)",
    "offset(예: gupo +2.26 m)은 위성 수위와 절대 수위 체계 간 정렬량이며 모델 "
    "오차가 아닙니다. 실제 오차는 '실측 대비 차이' 열을 보세요.",
    "성능 표기 주의: 융합 LODO RMSE 0.0453 m는 목표(0.5 m)는 충족하지만 위성 "
    "미사용 베이스라인(0.0199 m)이 더 낮습니다 — 이 모델은 '위성 수위의 절대화 "
    "보정기'로만 소개하고 '융합으로 향상' 주장은 금지.",
)


def fused_correction_rows(
    table: GaugeLevelTable, *, satellite: str | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Rows for the 당일 보정 panel — only observations that carry a correction.

    Keeps ``paired`` and ``wl_out_of_train_range`` visible as their own cells:
    a correction produced without a cross-sensor pair, or outside the training
    range, is still shown but must never look like an ordinary row.
    """

    if table.status != "ok" or not table.has_fused:
        return ()
    rows: list[Mapping[str, Any]] = []
    for row in table.rows:
        if row.fused_level_m is None:
            continue
        if satellite is not None and row.satellite != satellite:
            continue
        flags: list[str] = []
        if row.paired is False:
            flags.append(FUSED_SOLO_SENSOR_KO)
        if row.out_of_train_range:
            flags.append("학습범위 이탈")
        rows.append(
            MappingProxyType(
                {
                    "관측일": row.date,
                    "위성": row.satellite,
                    "지점": f"{row.loc_name} (지점 {row.loc_id})",
                    "위성 수위 (m)": (
                        None
                        if row.satellite_level_m is None
                        else round(row.satellite_level_m, 4)
                    ),
                    "보정 수위 (m)": round(row.fused_level_m, 4),
                    "offset (m)": (
                        None if row.offset_m is None else round(row.offset_m, 4)
                    ),
                    "게이지 실측 (m)": row.real_level_m,
                    "실측 대비 차이 (m)": (
                        None
                        if row.fused_delta_m is None
                        else round(row.fused_delta_m, 4)
                    ),
                    "비고": " · ".join(flags),
                }
            )
        )
    return tuple(rows)


def gauge_detail_rows(
    table: GaugeLevelTable, *, satellite: str | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Return long-form rows; fused columns appear only when runs exist."""

    if table.status != "ok":
        return ()
    rows: list[Mapping[str, Any]] = []
    for row in table.rows:
        if satellite is not None and row.satellite != satellite:
            continue
        entry: dict[str, Any] = {
            "관측 날짜": row.date,
            "위성": row.satellite,
            "게이지 지점": f"{row.loc_name} (지점 {row.loc_id})",
            "게이지 실측 수위 (m)": row.real_level_m,
        }
        if table.has_fused:
            entry["융합 보정 수위 (m)"] = row.fused_level_m
            entry["보정-실측 차이 (m)"] = (
                None if row.fused_delta_m is None else round(row.fused_delta_m, 4)
            )
        rows.append(MappingProxyType(entry))
    return tuple(rows)


# --- weather ------------------------------------------------------------------


def load_weather_series(path: Path | None = None) -> WeatherSeries:
    """Parse the CP949 Busan AWS daily CSV without inventing missing values."""

    csv_path = path if path is not None else WEATHER_CSV_PATH
    provenance = _relative_provenance(csv_path)
    if not csv_path.is_file():
        return WeatherSeries(
            status="missing",
            message_ko=f"기상 CSV를 찾지 못했습니다: {csv_path.name}",
            provenance=provenance,
        )
    try:
        raw = csv_path.read_bytes()
        if len(raw) > MAX_EVIDENCE_BYTES:
            raise ValueError("weather CSV too large")
        encoding = "cp949"
        try:
            text = raw.decode("cp949")
        except UnicodeDecodeError:
            encoding = "utf-8-sig"
            text = raw.decode("utf-8-sig")
        reader = csv.reader(text.splitlines())
        header = [cell.strip() for cell in next(reader, [])]
        if len(header) < 5 or header[0] != "지점" or header[1] != "일시":
            raise ValueError(f"unexpected weather header: {header}")

        def _column(keyword: str) -> int:
            for index, name in enumerate(header):
                if keyword in name:
                    return index
            raise ValueError(f"weather header misses column containing {keyword!r}")

        temp_idx = _column("평균기온")
        precip_idx = _column("일강수량")
        humidity_idx = _column("상대습도")

        def _optional_float(cells: list[str], index: int) -> float | None:
            value = cells[index].strip() if index < len(cells) else ""
            return float(value) if value else None

        rows: list[WeatherDailyRow] = []
        stations: set[str] = set()
        for line_number, cells in enumerate(reader, start=2):
            if not cells or all(not cell.strip() for cell in cells):
                continue  # 파일 끝의 ",,,,," 빈 행은 데이터가 아닙니다.
            station = cells[0].strip()
            date_text = cells[1].strip()
            if not station or not date_text:
                raise ValueError(f"row {line_number}: station/date missing")
            observed_on = dt.datetime.strptime(date_text, "%m/%d/%Y").date()
            stations.add(station)
            rows.append(
                WeatherDailyRow(
                    date=observed_on.isoformat(),
                    temperature_c=_optional_float(cells, temp_idx),
                    precipitation_mm=_optional_float(cells, precip_idx),
                    humidity_pct=_optional_float(cells, humidity_idx),
                )
            )
        if not rows:
            raise ValueError("weather CSV has no data rows")
        rows.sort(key=lambda row: row.date)
    except (StopIteration, TypeError, ValueError, OSError) as exc:
        return WeatherSeries(
            status="invalid",
            message_ko=f"기상 CSV 형식이 계약과 다릅니다: {exc}",
            provenance=provenance,
        )
    digest = hashlib.sha256(raw).hexdigest()
    return WeatherSeries(
        status="ok",
        message_ko=f"부산 AWS 일 관측 {len(rows)}일을 읽었습니다.",
        provenance=provenance,
        encoding=encoding,
        station_ids=tuple(sorted(stations)),
        rows=tuple(rows),
        n_days=len(rows),
        date_min=rows[0].date,
        date_max=rows[-1].date,
        missing_temperature=sum(1 for row in rows if row.temperature_c is None),
        missing_precipitation=sum(1 for row in rows if row.precipitation_mm is None),
        missing_humidity=sum(1 for row in rows if row.humidity_pct is None),
        sha256=digest,
        sha256_matches_fused_evidence=digest == FUSED_RECORDED_AWS_SHA256,
    )


# --- series comparison payloads ----------------------------------------------


def _deploy_test_scene_iou(report: ValReport) -> float | None:
    if report.status != "ok":
        return None
    target = DEPLOY_TEST_SCENES.get(report.sensor_type)
    for scene in report.per_scene:
        if scene.date == target:
            return scene.water_iou
    return None


def protocol_comparison_rows(
    valreports: tuple[ValReport, ...], smoke: SmokeEval
) -> tuple[Mapping[str, Any], ...]:
    """Build the three-series test-scene comparison; series never merge."""

    by_type = {report.sensor_type: report for report in valreports if report.status == "ok"}
    sar = by_type.get("SAR")
    optic = by_type.get("OPTIC")
    rows: list[Mapping[str, Any]] = [
        MappingProxyType(
            {
                "수치 계열": PROTOCOL_RESEARCH_KO,
                "SAR(ICEYE) 시험씬 water IoU": RESEARCH_PROTOCOL_IOUS["SAR"],
                "광학(PlanetScope) 시험씬 water IoU": RESEARCH_PROTOCOL_IOUS["OPTIC"],
                "근거": RESEARCH_PROVENANCE_KO,
            }
        ),
        MappingProxyType(
            {
                "수치 계열": PROTOCOL_DEPLOY_KO,
                "SAR(ICEYE) 시험씬 water IoU": (
                    None if sar is None else _deploy_test_scene_iou(sar)
                ),
                "광학(PlanetScope) 시험씬 water IoU": (
                    None if optic is None else _deploy_test_scene_iou(optic)
                ),
                "근거": (
                    "VALREPORT per_scene · 배포 UTM 그리드 "
                    f"(SAR {DEPLOY_TEST_SCENES['SAR']} · 광학 {DEPLOY_TEST_SCENES['OPTIC']})"
                ),
            }
        ),
        MappingProxyType(
            {
                "수치 계열": PROTOCOL_SMOKE_KO,
                "SAR(ICEYE) 시험씬 water IoU": (
                    smoke.metrics.water_iou
                    if smoke.status == "ok"
                    and smoke.metrics is not None
                    and smoke.sensor_name == "ICEYE"
                    and smoke.scene_date == DEPLOY_TEST_SCENES["SAR"]
                    else None
                ),
                "광학(PlanetScope) 시험씬 water IoU": None,
                "근거": f"{smoke.provenance} · 로컬 재추론 (광학 스모크는 미실행)",
            }
        ),
    ]
    return tuple(rows)


def per_scene_rows(report: ValReport) -> tuple[Mapping[str, Any], ...]:
    """Scene rows with the series label attached to every line."""

    if report.status != "ok":
        return ()
    return tuple(
        MappingProxyType(
            {
                "수치 계열": report.protocol_label_ko,
                "센서": report.sensor_name,
                "씬 날짜": scene.date,
                "water IoU": scene.water_iou,
                "F1": scene.f1,
                "Precision": scene.precision,
                "Recall": scene.recall,
                "Accuracy": scene.accuracy,
            }
        )
        for scene in report.per_scene
    )


def pooled_rows(valreports: tuple[ValReport, ...]) -> tuple[Mapping[str, Any], ...]:
    """Pooled aggregates, explicitly labelled as non-representative."""

    rows: list[Mapping[str, Any]] = []
    for report in valreports:
        if report.status != "ok" or report.pooled is None:
            continue
        rows.append(
            MappingProxyType(
                {
                    "수치 계열": f"{report.protocol_label_ko} · {report.pooled.date}",
                    "센서": report.sensor_name,
                    "water IoU": report.pooled.water_iou,
                    "F1": report.pooled.f1,
                    "주의": POOLED_CAUTION_KO,
                }
            )
        )
    return tuple(rows)


# --- figures ------------------------------------------------------------------


def build_scene_iou_figure(
    valreports: tuple[ValReport, ...], smoke: SmokeEval
) -> go.Figure:
    """Per-scene deployment IoU bars with clearly separated overlay series."""

    figure = go.Figure()
    colors = {"SAR": "#22d3ee", "OPTIC": "#f59e0b"}
    for report in valreports:
        if report.status != "ok":
            continue
        labels = [f"{report.sensor_name} {scene.date}" for scene in report.per_scene]
        figure.add_trace(
            go.Bar(
                x=labels,
                y=[scene.water_iou for scene in report.per_scene],
                name=f"{report.protocol_label_ko} · {report.sensor_name}",
                marker_color=colors.get(report.sensor_type, "#38bdf8"),
                opacity=0.85,
                text=[f"{scene.water_iou:.4f}" for scene in report.per_scene],
                textposition="outside",
                hovertemplate="%{x}<br>water IoU %{y:.6f}<extra></extra>",
            )
        )
    research_x: list[str] = []
    research_y: list[float] = []
    by_type = {report.sensor_type: report for report in valreports if report.status == "ok"}
    for sensor_type, iou in RESEARCH_PROTOCOL_IOUS.items():
        report = by_type.get(sensor_type)
        if report is None:
            continue
        research_x.append(f"{report.sensor_name} {DEPLOY_TEST_SCENES[sensor_type]}")
        research_y.append(iou)
    if research_x:
        figure.add_trace(
            go.Scatter(
                x=research_x,
                y=research_y,
                name=PROTOCOL_RESEARCH_KO,
                mode="markers",
                marker={"symbol": "star", "size": 15, "color": "#a855f7"},
                hovertemplate="%{x}<br>연구 프로토콜 %{y:.4f}<extra></extra>",
            )
        )
    if smoke.status == "ok" and smoke.metrics is not None:
        figure.add_trace(
            go.Scatter(
                x=[f"{smoke.sensor_name} {smoke.scene_date}"],
                y=[smoke.metrics.water_iou],
                name=PROTOCOL_SMOKE_KO,
                mode="markers",
                marker={"symbol": "diamond", "size": 13, "color": "#ef4444"},
                hovertemplate="%{x}<br>스모크 재현 %{y:.6f}<extra></extra>",
            )
        )
    figure.update_layout(
        **dict(_DARK_LAYOUT),
        height=420,
        title={
            "text": "씬별 water IoU · 연구/배포/스모크 계열은 격자·경로가 달라 수치가 다릅니다",
            "x": 0.01,
        },
        barmode="group",
    )
    figure.update_xaxes(title_text="센서 · 씬 날짜", gridcolor="#1e293b")
    figure.update_yaxes(
        title_text="water IoU", gridcolor="#1e293b", range=[0.8, 1.0]
    )
    return figure


def build_gauge_level_figure(table: GaugeLevelTable) -> go.Figure | None:
    """One measured-level line per gauge; None when the table is degraded."""

    if table.status != "ok":
        return None
    figure = go.Figure()
    palette = ("#22d3ee", "#f59e0b", "#a855f7", "#34d399")
    for color, loc_id in zip(palette, sorted(GAUGE_LOC_NAMES), strict=True):
        rows = [row for row in table.rows if row.loc_id == loc_id]
        if not rows:
            continue
        figure.add_trace(
            go.Scatter(
                x=[row.date for row in rows],
                y=[row.real_level_m for row in rows],
                name=f"{GAUGE_LOC_NAMES[loc_id]} (지점 {loc_id})",
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={
                    "size": 9,
                    "symbol": [
                        "diamond" if row.satellite == "ICEYE" else "circle"
                        for row in rows
                    ],
                },
                customdata=[row.satellite for row in rows],
                hovertemplate=(
                    "%{x} · %{customdata}<br>실측 수위 %{y:.3f} m<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **dict(_DARK_LAYOUT),
        height=380,
        title={
            "text": "게이지 실측 수위 30표본 · in-sample (융합 LSTM 학습표본과 동일)",
            "x": 0.01,
        },
    )
    figure.update_xaxes(title_text="위성 관측 날짜", gridcolor="#1e293b")
    figure.update_yaxes(title_text="실측 수위 (m)", gridcolor="#1e293b")
    return figure


def build_weather_figure(series: WeatherSeries) -> go.Figure | None:
    """T·H·PP three-channel daily chart; None when the CSV is degraded."""

    if series.status != "ok":
        return None
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    precipitation = [
        (row.date, row.precipitation_mm)
        for row in series.rows
        if row.precipitation_mm is not None
    ]
    if precipitation:
        figure.add_trace(
            go.Bar(
                x=[item[0] for item in precipitation],
                y=[item[1] for item in precipitation],
                name="일강수량 (mm)",
                marker_color="#087f8c",
                opacity=0.75,
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=[row.date for row in series.rows],
            y=[row.temperature_c for row in series.rows],
            name="평균기온 (°C)",
            mode="lines",
            line={"color": "#e4572e", "width": 2},
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=[row.date for row in series.rows],
            y=[row.humidity_pct for row in series.rows],
            name="평균 상대습도 (%)",
            mode="lines",
            line={"color": "#4763d2", "width": 1.6, "dash": "dot"},
        ),
        secondary_y=True,
    )
    figure.update_layout(
        **dict(_DARK_LAYOUT),
        height=400,
        title={
            "text": f"부산 AWS 일 관측 · {series.date_min} ~ {series.date_max}",
            "x": 0.01,
        },
    )
    figure.update_yaxes(title_text="강수량 (mm)", gridcolor="#1e293b", secondary_y=False)
    figure.update_yaxes(
        title_text="기온 (°C) / 습도 (%)", gridcolor="#1e293b", secondary_y=True
    )
    figure.update_xaxes(gridcolor="#1e293b")
    return figure


__all__ = [
    "BATCH_EVAL_DIR",
    "BATCH_WAITING_MESSAGE_KO",
    "DEPLOY_TEST_SCENES",
    "FUSED_RUN_DIR",
    "FUSED_WAITING_MESSAGE_KO",
    "GAUGE_CSV_HEADER",
    "GAUGE_CSV_PATH",
    "GAUGE_LOC_NAMES",
    "IN_SAMPLE_LABEL_KO",
    "POOLED_CAUTION_KO",
    "PROTOCOL_BATCH_KO",
    "PROTOCOL_DEPLOY_KO",
    "PROTOCOL_RESEARCH_KO",
    "PROTOCOL_SMOKE_KO",
    "RESEARCH_PROTOCOL_IOUS",
    "SATELLITE_NAMES",
    "SMOKE_EVAL_PATH",
    "WEATHER_CSV_PATH",
    "BatchEvalCollection",
    "BatchEvalReport",
    "GaugeLevelRow",
    "GaugeLevelTable",
    "SceneMetric",
    "SmokeEval",
    "ValReport",
    "WeatherDailyRow",
    "WeatherSeries",
    "build_gauge_level_figure",
    "build_scene_iou_figure",
    "build_weather_figure",
    "gauge_detail_rows",
    "gauge_pivot_rows",
    "load_batch_evals",
    "load_gauge_levels",
    "load_smoke_eval",
    "load_valreports",
    "load_weather_series",
    "per_scene_rows",
    "pooled_rows",
    "protocol_comparison_rows",
]
