#!/usr/bin/env python3
"""WAMIS 공개 API에서 일 단위 실측 수위를 받아 저장한다.

위성 관측은 지점당 6~8시점뿐이라 시계열 예측을 검증할 홀드아웃을 만들 수 없다.
같은 지점의 지상 수위계는 일 단위로 관측하므로 표본이 수십 배 늘고, 그때야
비로소 "학습 구간과 검증 구간을 날짜로 가르는" 정상적인 평가가 가능해진다.

WAMIS 개방 API는 인증키가 필요 없다. 다만 응답이 간헐적으로 잘려 오므로
재시도하고, 잘린 JSON은 정규식으로 건져 낸다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://www.wamis.go.kr:8080/wamis/openapi/wkw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gauge"

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


def fetch(url: str, attempts: int = 4, timeout: float = 60.0) -> str:
    last = ""
    for index in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                last = response.read().decode("utf-8", "replace")
            if last.rstrip().endswith("}"):
                return last
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"__error__ {exc}"
        time.sleep(1.5 * (index + 1))
    return last


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


def collect(stations: dict[str, str], start: str, end: str) -> dict[str, list]:
    result: dict[str, list] = {}
    for name, code in stations.items():
        # 긴 구간을 한 번에 요청하면 응답이 잘린다. 연 단위로 끊어 받는다.
        rows: list[tuple[str, float | None]] = []
        for year in range(int(start[:4]), int(end[:4]) + 1):
            lo = max(f"{year}0101", start)
            hi = min(f"{year}1231", end)
            url = f"{BASE}/wl_dtdata?obscd={code}&startdt={lo}&enddt={hi}&output=json"
            rows.extend(parse_rows(fetch(url)))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20201231")
    parser.add_argument(
        "--group", choices=["busan", "other", "all"], default="all",
        help="busan=위성 4지점, other=비교 지역, all=전체",
    )
    args = parser.parse_args()

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
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
