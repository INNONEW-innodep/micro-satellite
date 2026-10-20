#!/usr/bin/env python3
"""WAMIS 수위관측소 제원(wl_obsinfo)으로 `regions/*/gauge/series.json` 의 지점에 좌표·주소·하천명을 채운다.

    http://www.wamis.go.kr:8080/wamis/openapi/wkw/wl_obsinfo?obscd=<code>&output=json
    → list[0].lat / lon ("35-12-13" 도-분-초), addr, rivnm, tmx/tmy (TM 좌표)

- `stations[].source_code` 가 있는 지점만 조회한다. 3회 재시도(지수 백오프) 뒤에도 실패하면 필드를 null 로 두고 보고한다.
- 결과 좌표는 한국 범위(경도 124~132, 위도 33~39) 와 지역 `bbox_wgs84` 에서 50 km 이내여야 한다. 벗어나면
  null 로 두고 `[이상치]` 로 보고한다(조용히 남기지 않는다).
- 멱등: 다른 키는 그대로 두고 `lat`·`lon`·`address`·`river`·`tm_xy`(+`station_info_fetched_at`)만 갱신한다.
  이미 좌표가 있는 지점은 `--refresh` 없이는 다시 조회하지 않는다.

    .venv/bin/python scripts/fetch_wamis_station_info.py                # 전 지역
    .venv/bin/python scripts/fetch_wamis_station_info.py --region han_river --refresh
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGIONS_DIR = ROOT / "regions"
OBSINFO_URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkw/wl_obsinfo?obscd={code}&output=json"
_HEADERS = {"User-Agent": "Mozilla/5.0 (watercast station-info)", "Accept": "application/json"}
KOREA_LON = (124.0, 132.0)
KOREA_LAT = (33.0, 39.0)
MAX_BBOX_DISTANCE_KM = 50.0
FIELDS = ("lat", "lon", "address", "river", "tm_xy")


def parse_dms(text: Any) -> float | None:
    """'35-12-13' / '128-59-36.5' / '35 12 13' → 십진도. 이미 십진수면 그대로."""

    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [p for p in re.split(r"[-\s:]+", raw) if p != ""]
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return None
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    degrees, minutes = numbers[0], numbers[1]
    seconds = numbers[2] if len(numbers) > 2 else 0.0
    sign = -1.0 if raw.startswith("-") else 1.0
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def fetch_obsinfo(code: str, *, attempts: int = 3, timeout: float = 30.0) -> tuple[dict[str, Any] | None, str | None]:
    """(list[0], None) 또는 (None, 오류 메시지). 연결 리셋·부분 응답은 백오프 뒤 재시도."""

    last_error = "unknown"
    url = OBSINFO_URL.format(code=code)
    for index in range(attempts):
        try:
            request = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", "replace")
            payload = json.loads(text)
            items = payload.get("list") or []
            if not items:
                return None, f"응답에 list 가 비어 있음: {payload.get('result')}"
            return dict(items[0]), None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if index < attempts - 1:
            time.sleep(2.0 * (2**index))
    return None, last_error


def distance_to_bbox_km(lon: float, lat: float, bbox: list[float]) -> float:
    """점에서 bbox(경위도)까지의 거리(km) — bbox 안이면 0. 소규모 영역이라 등거리 근사로 충분하다."""

    minx, miny, maxx, maxy = (float(v) for v in bbox)
    dx_deg = max(minx - lon, 0.0, lon - maxx)
    dy_deg = max(miny - lat, 0.0, lat - maxy)
    dx_km = dx_deg * 111.32 * math.cos(math.radians(lat))
    dy_km = dy_deg * 110.57
    return math.hypot(dx_km, dy_km)


def validate(lon: float | None, lat: float | None, bbox: list[float] | None) -> str | None:
    """이상이면 사유 문자열, 정상이면 None."""

    if lon is None or lat is None:
        return "좌표 없음"
    if not (KOREA_LON[0] <= lon <= KOREA_LON[1] and KOREA_LAT[0] <= lat <= KOREA_LAT[1]):
        return f"한국 범위 밖 (lon={lon:.5f}, lat={lat:.5f})"
    if bbox and len(bbox) == 4:
        dist = distance_to_bbox_km(lon, lat, bbox)
        if dist > MAX_BBOX_DISTANCE_KM:
            return f"지역 bbox 에서 {dist:.1f} km 떨어짐 (lon={lon:.5f}, lat={lat:.5f})"
    return None


def enrich_station(station: dict[str, Any], info: dict[str, Any] | None, bbox: list[float] | None) -> tuple[dict[str, Any], str | None]:
    """지점 항목에 제원을 적는다. (갱신된 항목, 경고) — 이상치는 좌표만 null."""

    out = dict(station)
    if info is None:
        for key in FIELDS:
            out.setdefault(key, None)
        return out, None
    lat = parse_dms(info.get("lat"))
    lon = parse_dms(info.get("lon"))
    warning = validate(lon, lat, bbox)
    if warning:
        lat = lon = None
    out["lat"] = round(lat, 6) if lat is not None else None
    out["lon"] = round(lon, 6) if lon is not None else None
    out["address"] = str(info.get("addr") or "").strip() or None
    out["river"] = str(info.get("rivnm") or "").strip() or None
    try:
        out["tm_xy"] = [float(info["tmx"]), float(info["tmy"])] if info.get("tmx") not in (None, "") and info.get("tmy") not in (None, "") else None
    except (TypeError, ValueError):
        out["tm_xy"] = None
    out["station_info_fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out, warning


def process_region(region_dir: Path, *, refresh: bool = False, dry_run: bool = False) -> dict[str, Any]:
    region_id = region_dir.name
    series_path = region_dir / "gauge" / "series.json"
    manifest_path = region_dir / "region.json"
    report: dict[str, Any] = {"region_id": region_id, "ok": [], "failed": [], "outliers": [], "skipped": [], "changed": False}
    if not series_path.is_file():
        return report
    bbox = None
    if manifest_path.is_file():
        try:
            bbox = json.loads(manifest_path.read_text(encoding="utf-8")).get("bbox_wgs84")
        except ValueError:
            bbox = None
    payload = json.loads(series_path.read_text(encoding="utf-8"))
    stations = list(payload.get("stations") or [])
    updated: list[dict[str, Any]] = []
    for station in stations:
        code = str(station.get("source_code") or "").strip()
        station_id = str(station.get("id"))
        if not code:
            updated.append(station)
            report["skipped"].append(f"{station_id}(코드 없음)")
            continue
        if not refresh and station.get("lat") is not None and station.get("lon") is not None:
            updated.append(station)
            report["skipped"].append(f"{station_id}(이미 있음)")
            continue
        info, error = fetch_obsinfo(code)
        item, warning = enrich_station(station, info, bbox)
        updated.append(item)
        if error:
            report["failed"].append(f"{station_id}[{code}]: {error}")
        elif warning:
            report["outliers"].append(f"{station_id}[{code}]: {warning}")
        else:
            report["ok"].append(f"{station_id}[{code}] lon={item['lon']} lat={item['lat']} {item.get('river') or ''}")
        time.sleep(0.4)
    # 저장 시각만 다른 경우는 변경으로 치지 않는다
    def _strip(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in s.items() if k != "station_info_fetched_at"} for s in items]

    report["changed"] = _strip(updated) != _strip(stations)
    if report["changed"] and not dry_run:
        payload["stations"] = updated
        series_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", action="append", help="특정 지역 id 만(기본: regions/ 전부)")
    parser.add_argument("--regions-dir", type=Path, default=REGIONS_DIR)
    parser.add_argument("--refresh", action="store_true", help="이미 좌표가 있는 지점도 다시 조회")
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않는다")
    args = parser.parse_args()
    region_dirs = [args.regions_dir / r for r in args.region] if args.region else sorted(p for p in args.regions_dir.iterdir() if (p / "region.json").is_file())
    exit_code = 0
    for region_dir in region_dirs:
        report = process_region(region_dir, refresh=args.refresh, dry_run=args.dry_run)
        state = "변경 없음" if not report["changed"] else ("(dry-run) 변경 예정" if args.dry_run else "기록")
        print(f"[{report['region_id']}] 좌표 확보 {len(report['ok'])} / 실패 {len(report['failed'])} / 이상치 {len(report['outliers'])} / 건너뜀 {len(report['skipped'])} — {state}")
        for line in report["ok"]:
            print(f"  [확보] {line}")
        for line in report["outliers"]:
            print(f"  [이상치→null] {line}")
            exit_code = 2
        for line in report["failed"]:
            print(f"  [실패→null] {line}")
            exit_code = 2
        for line in report["skipped"]:
            print(f"  [건너뜀] {line}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
