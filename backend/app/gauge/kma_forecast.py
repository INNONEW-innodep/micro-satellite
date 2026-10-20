"""기상청 단기예보(공공데이터포털 VilageFcstInfoService_2.0 getVilageFcst) → 지역 일 강수·기온 예보 행. 2026-10-20.

- 격자: 기상청 DFS(Lambert conformal) 변환으로 위경도 → (nx, ny).
- 발표: 02·05·08·11·14·17·20·23시(KST), 발표 후 약 10분 뒤 제공 → 가장 최근 발표분을 고른다(안전 여유 1시간).
- 항목: PCP(1시간 강수량, 문자열 "강수없음"·"1mm 미만"·"30.0~50.0mm"·"50.0mm 이상"·"3.2mm"), TMP(기온), POP(강수확률).
- 집계: fcstDate 별 강수 합(mm)·기온 평균·강수확률 최대. 결과 행은 지역 ``weather/kma_forecast_daily.json`` 에 저장되고
  ``RegionRegistry.load_series`` 가 관측 뒤 날짜만 ``DailySeries.exog_tail`` 로 붙인다(``ridge_arx_future``·``ridge_arx_season_future``).
- 키: 서버 환경변수 ``KMA_API_KEY``(ASOS 수집과 같은 키 — 단기예보 서비스도 활용신청이 돼 있어야 한다). 인터넷 필요.
- 과거 발표분은 이 API 로 받을 수 없다(최근 발표만). 과거 예보 재측정은 기상청 API 허브(별도 키) — 문서 참고.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

PROVIDER = "KMA_VILAGE_FCST"
SCHEMA = "watercast.weather.forecast/1"
ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
BASE_TIMES = ("0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300")
KST = timezone(timedelta(hours=9))
_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def dfs_grid(lat: float, lon: float) -> tuple[int, int]:
    """위경도(WGS84) → 기상청 동네예보 격자(nx, ny). 대전 36.37,127.37 → (67,100)."""

    re_, grid = 6371.00877, 5.0
    slat1, slat2, olon, olat = math.radians(30.0), math.radians(60.0), math.radians(126.0), math.radians(38.0)
    xo, yo = 43, 136
    r = re_ / grid
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = sf ** sn * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = r * sf / ro ** sn
    ra = math.tan(math.pi * 0.25 + math.radians(lat) * 0.5)
    ra = r * sf / ra ** sn
    theta = math.radians(lon) - olon
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi
    theta *= sn
    return int(math.floor(ra * math.sin(theta) + xo + 0.5)), int(math.floor(ro - ra * math.cos(theta) + yo + 0.5))


def latest_base(now: datetime | None = None, *, margin_minutes: int = 60) -> tuple[str, str]:
    """가장 최근 발표(base_date, base_time). 발표 뒤 ``margin_minutes`` 가 지나지 않았으면 그 전 발표."""

    now = (now or datetime.now(KST)).astimezone(KST) - timedelta(minutes=margin_minutes)
    hhmm = now.strftime("%H%M")
    candidates = [t for t in BASE_TIMES if t <= hhmm]
    if candidates:
        return now.strftime("%Y%m%d"), candidates[-1]
    prev = now - timedelta(days=1)
    return prev.strftime("%Y%m%d"), BASE_TIMES[-1]


def parse_pcp(text: str) -> float:
    """PCP 문자열 → mm. '강수없음' 0, '1mm 미만' 0.5, '30.0~50.0mm' 40(구간 중앙), '50.0mm 이상' 50, '3.2mm' 3.2."""

    raw = str(text or "").strip()
    if not raw or "없음" in raw:
        return 0.0
    nums = [float(x) for x in _NUM.findall(raw)]
    if not nums:
        return 0.0
    if "미만" in raw:
        return nums[0] / 2.0
    if "~" in raw and len(nums) >= 2:
        return (nums[0] + nums[1]) / 2.0
    return nums[0]


def fetch_items(service_key: str, nx: int, ny: int, base_date: str, base_time: str, *, fetch: Callable[[str], str] | None = None, timeout: float = 30.0) -> list[dict[str, Any]]:
    """getVilageFcst 호출 → item[] (baseDate, baseTime, category, fcstDate, fcstTime, fcstValue). 오류는 RuntimeError."""

    key = urllib.parse.unquote(service_key.strip())  # 인코딩된 키를 받아도 한 번만 인코딩되게
    query = urllib.parse.urlencode({
        "serviceKey": key, "pageNo": 1, "numOfRows": 1500, "dataType": "JSON",
        "base_date": base_date, "base_time": base_time, "nx": int(nx), "ny": int(ny),
    })
    url = f"{ENDPOINT}?{query}"
    if fetch is None:
        def fetch(u: str) -> str:  # noqa: E306
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "watercast/1.0"}), timeout=timeout) as resp:
                return resp.read().decode("utf-8", "replace")
    text = fetch(url)
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise RuntimeError(f"기상청 단기예보 응답이 JSON 이 아닙니다: {text[:160]!r}") from exc
    header = ((doc.get("response") or {}).get("header") or {})
    code = str(header.get("resultCode") or "")
    if code != "00":
        raise RuntimeError(f"기상청 단기예보 오류 {code}: {header.get('resultMsg')}")
    items = (((doc.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
    return [dict(i) for i in items if isinstance(i, dict)]


def daily_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """item[] → 날짜별 {date, precipitation_mm, avg_temperature_c, pop_max_pct, n_hours}. 날짜 오름차순."""

    by_day: dict[str, dict[str, Any]] = {}
    for it in items:
        d = str(it.get("fcstDate") or "")
        if len(d) != 8:
            continue
        key = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        row = by_day.setdefault(key, {"date": key, "precipitation_mm": 0.0, "_temps": [], "pop_max_pct": 0, "n_hours": 0})
        cat = str(it.get("category") or "")
        val = it.get("fcstValue")
        if cat == "PCP":
            row["precipitation_mm"] += parse_pcp(str(val))
            row["n_hours"] += 1
        elif cat == "TMP":
            try:
                row["_temps"].append(float(val))
            except (TypeError, ValueError):
                pass
        elif cat == "POP":
            try:
                row["pop_max_pct"] = max(row["pop_max_pct"], int(float(val)))
            except (TypeError, ValueError):
                pass
    out = []
    for key in sorted(by_day):
        row = by_day[key]
        temps = row.pop("_temps")
        row["precipitation_mm"] = round(row["precipitation_mm"], 1)
        row["avg_temperature_c"] = round(sum(temps) / len(temps), 1) if temps else None
        out.append(row)
    return out


def collect(service_key: str, lat: float, lon: float, *, now: datetime | None = None, fetch: Callable[[str], str] | None = None) -> dict[str, Any]:
    """위경도 한 점의 단기예보를 받아 저장 문서(schema, provider, issued_at, nx, ny, rows[])로 만든다."""

    nx, ny = dfs_grid(lat, lon)
    base_date, base_time = latest_base(now)
    items = fetch_items(service_key, nx, ny, base_date, base_time, fetch=fetch)
    rows = daily_rows(items)
    return {
        "schema": SCHEMA, "provider": PROVIDER, "issued_at": f"{base_date[:4]}-{base_date[4:6]}-{base_date[6:]}T{base_time[:2]}:{base_time[2:]}+09:00",
        "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny, "lat": lat, "lon": lon,
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "n_items": len(items), "rows": rows,
        "note_ko": "기상청 단기예보(동네예보 격자) 1시간 PCP 합·TMP 평균·POP 최대를 날짜별로 집계. 첫날은 발표 시각 이후만 포함. 관측이 있는 날짜는 관측이 우선한다.",
    }


__all__ = ["BASE_TIMES", "ENDPOINT", "PROVIDER", "SCHEMA", "collect", "daily_rows", "dfs_grid", "fetch_items", "latest_base", "parse_pcp"]
