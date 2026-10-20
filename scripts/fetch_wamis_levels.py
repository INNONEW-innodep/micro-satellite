#!/usr/bin/env python3
"""WAMIS 공개 API에서 일 단위 실측 수위를 받아 저장한다.

위성 관측은 지점당 6~8시점뿐이라 시계열 예측을 검증할 홀드아웃을 만들 수 없다.
같은 지점의 지상 수위계는 일 단위로 관측하므로 표본이 수십 배 늘고, 그때야
비로소 "학습 구간과 검증 구간을 날짜로 가르는" 정상적인 평가가 가능해진다.

WAMIS 개방 API는 인증키가 필요 없다. 다만 응답이 간헐적으로 잘려 오므로
재시도하고, 잘린 JSON은 정규식으로 건져 낸다.

범용 경로 (유역 → 후보 지점 → 커버리지 검사 → regions/<id>/gauge/series.json):

    # 유역 관측소 목록 (1=한강 2=낙동강 3=금강 4=섬진강 5=영산강)
    python3 scripts/fetch_wamis_levels.py --basin 5 --list --match "나주|담양"
    # 후보를 받아 2015~2024 일 커버리지 85% 이상인 지점만 남겨 지역 파일로 저장
    python3 scripts/fetch_wamis_levels.py --stations "나주=5004640,영산포=5004660" \
        --start 20150101 --end 20241231 --min-coverage 0.85 \
        --region-id yeongsan_river --out regions/yeongsan_river/gauge/series.json
    # 지형·설명까지 담은 JSON 스펙으로 (stations[{id,name,code,terrain,terrain_note}])
    python3 scripts/fetch_wamis_levels.py --spec regions/yeongsan_river/gauge/wamis_spec.json \
        --start 20150101 --end 20241231 --out regions/yeongsan_river/gauge/series.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = "http://www.wamis.go.kr:8080/wamis/openapi/wkw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gauge"
SERIES_SCHEMA = "watercast.gauge.series/1"

# WAMIS 유역 코드 (wl_dubwlobs?basin=N)
BASINS: dict[int, str] = {1: "한강", 2: "낙동강", 3: "금강", 4: "섬진강", 5: "영산강"}

# 지형(하천 물리)이 다르면 잘 맞는 모델도 다르다. 감조하천·본류·댐·상류를
# 골고루 넣어야 "지점마다 다른 모델"이라는 주장을 데이터로 확인할 수 있다.
STATION_GROUPS: dict[str, dict[str, tuple[str, str]]] = {
    # 이름: (관측소코드, 지형 설명)
    "estuary": {  # 낙동강 하구 — 조위 영향(감조)
        "낙동강하구언(내)": ("2022696", "하구언 내측·감조"),
        "구포대교": ("2022680", "하구부 본류·감조"),
    },
    "mainstem": {  # 낙동강 본류 중·하류
        "삼랑진교": ("2022610", "본류 중류"),
        "양산교": ("2022660", "본류 하류"),
        "호포대교": ("2022670", "본류 하류"),
    },
    "tributary": {  # 지류 — 유역이 작아 강우 응답이 빠름
        "정천교": ("2022685", "김해 지류"),
        "주천교": ("2020671", "김해 지류 상류"),
    },
    "dam": {  # 댐 — 인위 조절
        "충주본댐우안": ("1003664", "댐 직하류"),
        "팔당댐": ("1017690", "댐 직하류"),
    },
    "han": {  # 한강 본류 — 다른 유역, 대조군
        "한강대교": ("1018683", "한강 본류 하류"),
        "여주대교": ("1007635", "한강 본류 중류"),
        "양평교": ("1007685", "한강 본류 상류"),
    },
}

BUSAN_STATIONS: dict[str, str] = {
    name: code for group in ("estuary", "mainstem", "tributary")
    for name, (code, _) in STATION_GROUPS[group].items()
}
OTHER_STATIONS: dict[str, str] = {
    name: code for group in ("dam", "han")
    for name, (code, _) in STATION_GROUPS[group].items()
}
TERRAIN: dict[str, str] = {
    name: f"{group}·{desc}"
    for group, members in STATION_GROUPS.items()
    for name, (_, desc) in members.items()
}


_HEADERS = {
    # 서버가 헤더 전·본문 중 아무 때나 연결을 끊는다(Connection reset). 헤더와 무관하게
    # 재시도·부분 응답 회수로 대응한다.
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) watercast-fetch/1.0",
    "Accept": "application/json,*/*",
}


def _read_partial(response: Any) -> tuple[bytes, bool]:
    """본문을 조각으로 읽는다. 서버가 도중에 연결을 끊으면(reset) 그때까지 받은 조각을 돌려준다."""

    chunks: list[bytes] = []
    try:
        while True:
            piece = response.read(65536)
            if not piece:
                return b"".join(chunks), True
            chunks.append(piece)
    except (ConnectionResetError, http.client.IncompleteRead, TimeoutError, OSError):
        return b"".join(chunks), False


def fetch(url: str, attempts: int = 4, timeout: float = 60.0) -> str:
    """완전한 JSON('}'로 끝남)을 받을 때까지 재시도. 못 받으면 가장 긴 부분 응답을 돌려준다
    (호출자가 정규식으로 건진다). 연결 자체가 안 되면 '__error__ ...' 문자열."""

    best = ""
    last_error = ""
    for index in range(attempts):
        try:
            request = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw, complete = _read_partial(response)
            text = raw.decode("utf-8", "replace")
            if complete and text.rstrip().endswith("}"):
                return text
            if len(text) > len(best):
                best = text
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"__error__ {exc}"
        time.sleep(1.5 * (index + 1))
    return best or last_error


_ROW = re.compile(r'\{"ymd":"(\d{8})","wl":"([^"]*)"\}')


def parse_rows(payload: str) -> list[tuple[str, float | None]]:
    """정상 JSON이면 그대로, 잘렸으면 정규식으로 건진다."""

    try:
        data = json.loads(payload)
        rows = data.get("list") or []
        out: list[tuple[str, float | None]] = []
        for row in rows:
            raw = str(row.get("wl", "")).strip()
            try:
                out.append((str(row["ymd"]), float(raw) if raw else None))
            except (KeyError, ValueError):
                continue
        return out
    except (ValueError, AttributeError):
        return [
            (ymd, float(wl) if wl.strip() else None)
            for ymd, wl in _ROW.findall(payload)
        ]


FETCH_FAILURES: list[str] = []


def fetch_year(name: str, code: str, lo: str, hi: str, passes: int = 3) -> list[tuple[str, float | None]] | None:
    """한 지점의 한 구간(보통 1년). 서버가 연결을 끊으면 잠시 쉬었다가 다시 시도한다.
    끝내 실패하면 None (FETCH_FAILURES 에 기록) — 결측과 구분해야 하므로 빈 목록을 돌려주지 않는다."""

    url = f"{BASE}/wl_dtdata?obscd={code}&startdt={lo}&enddt={hi}&output=json"
    reason = ""
    for attempt in range(passes):
        payload = fetch(url)
        if payload.startswith("__error__"):
            reason = payload[10:200]
        else:
            got = parse_rows(payload)
            if got:
                return got
            reason = f"0행 (응답 {len(payload)}자)"
            # 운영 관측소의 1년치가 0행이면 대개 서버 쪽 문제다 — 한 번 더 확인 후 결측으로 인정
            if attempt >= 1:
                return []
        time.sleep(5.0 * (attempt + 1))
    FETCH_FAILURES.append(f"{name}({code}) {lo[:4]}: {reason}")
    print(f"  [실패] {name} {lo[:4]}: {reason[:120]}", file=sys.stderr)
    return None


def collect(stations: dict[str, str], start: str, end: str) -> dict[str, list]:
    result: dict[str, list] = {}
    for name, code in stations.items():
        # 긴 구간을 한 번에 요청하면 응답이 잘린다. 연 단위로 끊어 받는다.
        rows: list[tuple[str, float | None]] = []
        for year in range(int(start[:4]), int(end[:4]) + 1):
            lo = max(f"{year}0101", start)
            hi = min(f"{year}1231", end)
            got = fetch_year(name, code, lo, hi)
            # 수집 실패는 조용히 결측이 되면 안 된다 — 관측 결측과 구분이 안 된다.
            if got is None:
                continue
            if not got:
                print(f"  [경고] {name} {year}: 0행", file=sys.stderr)
            rows.extend(got)
            time.sleep(0.4)
        rows = sorted({d: v for d, v in rows}.items())
        valid = [(d, v) for d, v in rows if v is not None]
        result[name] = [{"date": d, "water_level_m": v, "obscd": code} for d, v in valid]
        gap = len(rows) - len(valid)
        print(
            f"  {name:12} obscd={code}  {len(valid):>4}일"
            + (f"  (결측 {gap}일 제외)" if gap else ""),
            flush=True,
        )
    return result


# --- 범용 경로: 유역 목록 → 후보 검사 → 지역 series.json ---------------------------

_STATION_ROW = re.compile(
    r'\{"bbsnnm":"([^"]*)","obscd":"(\d+)","obsnm":"([^"]*)","clsyn":"([^"]*)"'
    r'(?:,"obsknd":"([^"]*)")?(?:,"sbsncd":"([^"]*)")?(?:,"mngorg":"([^"]*)")?'
)


def list_basin_stations(basin: int, attempts: int = 12) -> list[dict[str, str]]:
    """유역 관측소 목록. 응답이 자주 잘리므로 완전한 JSON을 받을 때까지 재시도하고,
    끝내 못 받으면 정규식으로 건진 부분 목록을 (경고와 함께) 돌려준다."""

    url = f"{BASE}/wl_dubwlobs?basin={int(basin)}&output=json"
    best = ""
    for index in range(attempts):
        payload = fetch(url, attempts=1)
        if payload.startswith("__error__"):
            time.sleep(2.0 * (index + 1))
            continue
        try:
            data = json.loads(payload)
            rows = data.get("list") or []
            return [{k: str(v) for k, v in row.items()} for row in rows]
        except ValueError:
            if len(payload) > len(best):
                best = payload
            time.sleep(1.5 * (index + 1))
    found = _STATION_ROW.findall(best)
    print(f"  [경고] basin={basin} 목록이 {attempts}회 모두 잘림 — 부분 목록 {len(found)}건 사용", file=sys.stderr)
    keys = ("bbsnnm", "obscd", "obsnm", "clsyn", "obsknd", "sbsncd", "mngorg")
    return [dict(zip(keys, row)) for row in found]


def parse_station_arg(text: str) -> dict[str, str]:
    """'이름=코드,이름=코드' → {이름: 코드}."""

    out: dict[str, str] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"--stations 형식 오류: '{chunk}' (이름=코드)")
        name, code = (part.strip() for part in chunk.split("=", 1))
        if not re.fullmatch(r"\d{7}", code):
            raise SystemExit(f"--stations 관측소 코드 오류: '{code}' (7자리 숫자)")
        out[name] = code
    return out


def expected_days(start: str, end: str) -> int:
    lo = _dt.date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    hi = _dt.date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    return (hi - lo).days + 1


def coverage_report(series: dict[str, list], start: str, end: str) -> dict[str, dict[str, Any]]:
    total = expected_days(start, end)
    out: dict[str, dict[str, Any]] = {}
    for name, rows in series.items():
        n = len(rows)
        out[name] = {
            "n_days": n,
            "expected_days": total,
            "coverage": round(n / total, 4) if total else 0.0,
            "period": [rows[0]["date"], rows[-1]["date"]] if rows else None,
        }
    return out


def build_series_payload(
    region_id: str,
    specs: list[dict[str, str]],
    series: dict[str, list],
    start: str,
    end: str,
) -> dict[str, Any]:
    """regions/<id>/gauge/series.json (watercast.gauge.series/1) 본문."""

    iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}"  # noqa: E731
    stations = []
    out_series: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        rows = series.get(spec["id"]) or []
        if not rows:
            continue
        stations.append(
            {
                "id": spec["id"],
                "name": spec.get("name") or spec["id"],
                "source": "WAMIS",
                "source_code": spec["code"],
                "terrain": spec.get("terrain", ""),
                "terrain_note": spec.get("terrain_note", ""),
                "unit": "m",
                "n_days": len(rows),
                "period": [rows[0]["date"], rows[-1]["date"]],
            }
        )
        out_series[spec["id"]] = [{"date": iso(r["date"]), "water_level_m": r["water_level_m"]} for r in rows]
    return {
        "schema": SERIES_SCHEMA,
        "region_id": region_id,
        "source": "WAMIS wl_dtdata (개방 API, 무인증)",
        "fetched_range": [iso(start), iso(end)],
        "fetched_at": _dt.date.today().isoformat(),
        "unit": "m",
        "frequency": "daily",
        "stations": stations,
        "series": out_series,
    }


def run_generic(args: argparse.Namespace) -> int:
    """--basin/--stations/--spec 경로. 후보를 받아 커버리지로 거르고 지역 파일로 저장."""

    if args.basin and args.list:
        rows = list_basin_stations(args.basin)
        pattern = re.compile(args.match) if args.match else None
        print(f"basin={args.basin} {BASINS.get(args.basin, '')} · {len(rows)}개 관측소")
        for row in rows:
            if pattern and not pattern.search(row.get("obsnm", "")):
                continue
            print(f"  {row.get('obscd')}  {row.get('obsnm'):20} {row.get('clsyn', ''):4} {row.get('obsknd', ''):5} {row.get('mngorg', '')}")
        return 0

    specs: list[dict[str, str]] = []
    if args.spec:
        spec_doc = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        args.region_id = args.region_id or spec_doc.get("region_id")
        for item in spec_doc.get("stations") or []:
            specs.append(
                {
                    "id": str(item.get("id") or item.get("name")),
                    "name": str(item.get("name") or item.get("id")),
                    "code": str(item["code"]),
                    "terrain": str(item.get("terrain", "")),
                    "terrain_note": str(item.get("terrain_note", "")),
                }
            )
    if args.stations:
        for name, code in parse_station_arg(args.stations).items():
            specs.append({"id": name, "name": name, "code": code, "terrain": "", "terrain_note": ""})
    if args.basin and args.match and not specs:
        # 유역 목록에서 이름 패턴으로 후보를 뽑는다 (운영 중인 관측소만)
        pattern = re.compile(args.match)
        for row in list_basin_stations(args.basin):
            if row.get("clsyn", "운영") != "운영":
                continue
            if pattern.search(row.get("obsnm", "")):
                specs.append({"id": row["obsnm"], "name": row["obsnm"], "code": row["obscd"], "terrain": "", "terrain_note": ""})
    if not specs:
        print("후보 지점이 없습니다 — --stations, --spec, 또는 --basin N --match 패턴을 주세요.", file=sys.stderr)
        return 1

    print(f"WAMIS 일 수위 수집 · {args.start} ~ {args.end} · 후보 {len(specs)}지점")
    fetched = collect({spec["id"]: spec["code"] for spec in specs}, args.start, args.end)
    report = coverage_report(fetched, args.start, args.end)
    kept: list[dict[str, str]] = []
    for spec in specs:
        info = report[spec["id"]]
        ok = info["coverage"] >= args.min_coverage
        flag = "유지" if ok else "제외"
        print(f"  [{flag}] {spec['id']:16} obscd={spec['code']}  {info['n_days']:>5}/{info['expected_days']}일  커버리지 {info['coverage']:.1%}")
        if ok:
            kept.append(spec)
    if len(kept) < 1:
        print(f"커버리지 {args.min_coverage:.0%} 이상인 지점이 없습니다.", file=sys.stderr)
        return 1

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = build_series_payload(args.region_id or out.parent.parent.name, kept, fetched, args.start, args.end)
        payload["coverage"] = {spec["id"]: report[spec["id"]] for spec in kept}
        out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"저장 {out} · 지점 {len(kept)}개 · {sum(len(fetched[s['id']]) for s in kept):,}일")
    if FETCH_FAILURES:
        print(f"수집 실패/빈 응답 {len(FETCH_FAILURES)}건:", file=sys.stderr)
        for item in FETCH_FAILURES:
            print("  -", item, file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20201231")
    parser.add_argument(
        "--group", choices=["busan", "other", "all"], default="all",
        help="busan=위성 4지점, other=비교 지역, all=전체",
    )
    generic = parser.add_argument_group("범용 경로 (유역·후보 지점·지역 파일)")
    generic.add_argument("--basin", type=int, choices=sorted(BASINS), default=None, help="유역 코드 1=한강 2=낙동강 3=금강 4=섬진강 5=영산강")
    generic.add_argument("--list", action="store_true", help="--basin 관측소 목록만 출력")
    generic.add_argument("--match", default=None, help="관측소 이름 정규식 (목록 필터 또는 후보 선택)")
    generic.add_argument("--stations", default=None, help="'이름=코드,이름=코드' 후보 지점")
    generic.add_argument("--spec", default=None, help="JSON 스펙 {region_id, stations:[{id,name,code,terrain,terrain_note}]}")
    generic.add_argument("--min-coverage", type=float, default=0.85, help="기간 대비 유효 일수 비율 하한 (기본 0.85)")
    generic.add_argument("--region-id", default=None, help="series.json 의 region_id")
    generic.add_argument("--out", default=None, help="watercast.gauge.series/1 로 저장할 경로 (예: regions/<id>/gauge/series.json)")
    args = parser.parse_args()

    if args.basin or args.stations or args.spec:
        return run_generic(args)

    stations: dict[str, str] = {}
    if args.group in ("busan", "all"):
        stations.update(BUSAN_STATIONS)
    if args.group in ("other", "all"):
        stations.update(OTHER_STATIONS)

    print(f"WAMIS 일 수위 수집 · {args.start} ~ {args.end} · {len(stations)}지점")
    data = collect(stations, args.start, args.end)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"wamis_daily_{args.start}_{args.end}.json"
    out.write_text(
        json.dumps(
            {
                "source": "WAMIS 개방 API (wl_dtdata) · 인증키 불요",
                "fetched_range": [args.start, args.end],
                "stations": stations,
                "series": data,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    total = sum(len(v) for v in data.values())
    print(f"\n총 {total:,}일 · 저장 {out}")
    if FETCH_FAILURES:
        print(f"수집 실패/빈 응답 {len(FETCH_FAILURES)}건 — 결측이 관측 결측인지 수집 실패인지 확인하세요:", file=sys.stderr)
        for item in FETCH_FAILURES:
            print("  -", item, file=sys.stderr)
        return 2
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
