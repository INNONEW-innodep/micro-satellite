#!/usr/bin/env python3
"""지역 레지스트리용 기상청 ASOS 일자료(`regions/<id>/weather/asos_daily.json`)를 만든다.

로컬 백엔드에는 KMA 인증키가 없을 수 있으므로 기본값은 배포된 API
(`https://wc-api.semyeongsoft.com`)의 `/api/v1/weather/observations` 를 연 단위로 부른다.
응답 `source` 가 `asos` 가 아니면(합성 `sample`) 저장하지 않는다 — 합성값을 실측처럼
학습에 쓰면 안 되기 때문이다.

    python3 scripts/fetch_region_weather.py --region-id yeongsan_river --station-id 156 \
        --start 2015-01-01 --end 2024-12-31 --out regions/yeongsan_river/weather/asos_daily.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "watercast.weather.daily/1"
DEFAULT_API = "https://wc-api.semyeongsoft.com"
FIELDS = ("precipitation_mm", "avg_temperature_c", "min_temperature_c", "max_temperature_c", "humidity_pct")


def post_json(url: str, body: dict[str, Any], attempts: int = 4, timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    last: Exception | None = None
    for index in range(attempts):
        try:
            request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            time.sleep(3.0 * (index + 1))
    raise RuntimeError(f"weather API 실패: {last}")


def fetch_year_rows(api: str, station_id: str, start: str, end: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = post_json(
        f"{api.rstrip('/')}/api/v1/weather/observations",
        {"station_id": station_id, "start_date": start, "end_date": end, "source": "asos"},
    )
    source = str(payload.get("source") or "")
    if source != "asos":
        raise RuntimeError(f"{start}~{end}: 응답 source={source!r} — 실측(asos)이 아니어서 저장하지 않음")
    rows = payload.get("rows") or []
    return [dict(r) for r in rows if isinstance(r, dict)], payload


def _num(value: Any, default: float | None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return default


def normalise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        date = str(row.get("date") or "")[:10]
        if len(date) != 10:
            continue
        out[date] = {
            "date": date,
            # 강수 빈칸은 '무강수'로 본다 (기존 지역 파일과 같은 규칙)
            "precipitation_mm": _num(row.get("precipitation_mm"), 0.0),
            "avg_temperature_c": _num(row.get("avg_temperature_c", row.get("temperature_c")), None),
            "min_temperature_c": _num(row.get("min_temperature_c"), None),
            "max_temperature_c": _num(row.get("max_temperature_c"), None),
            "humidity_pct": _num(row.get("humidity_pct", row.get("avg_humidity_percent")), None),
        }
    return [out[key] for key in sorted(out)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--station-id", required=True, help="ASOS 지점 번호 (예: 광주 156, 안동 136)")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--api", default=DEFAULT_API, help="weather API 베이스 URL")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    first = dt.date.fromisoformat(args.start)
    last = dt.date.fromisoformat(args.end)
    all_rows: list[dict[str, Any]] = []
    station_name = None
    for year in range(first.year, last.year + 1):
        lo = max(first, dt.date(year, 1, 1)).isoformat()
        hi = min(last, dt.date(year, 12, 31)).isoformat()
        rows, payload = fetch_year_rows(args.api, args.station_id, lo, hi)
        station_name = station_name or payload.get("station_name")
        print(f"  {year}: {len(rows)}행 (source={payload.get('source')})", flush=True)
        all_rows.extend(rows)
        time.sleep(0.5)

    rows = normalise(all_rows)
    have = {r["date"] for r in rows}
    missing = []
    day = first
    while day <= last:
        if day.isoformat() not in have:
            missing.append(day.isoformat())
        day += dt.timedelta(days=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "region_id": args.region_id,
                "provider": "KMA_ASOS",
                "station_id": str(args.station_id),
                "station_name": station_name,
                "fetched_range": [args.start, args.end],
                "fetched_at": dt.date.today().isoformat(),
                "note_ko": "기상청 ASOS 일자료(AsosDalyInfoService). 강수 빈칸은 0으로 둠. 예측기의 외생 입력(원점 이전 강수·기온)으로 쓴다.",
                "missing_dates": missing,
                "rows": rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"저장 {out} · {len(rows)}행 · 결측 {len(missing)}일 · 지점 {args.station_id} {station_name}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
