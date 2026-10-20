"""위성 수위 산출 구역(AOI)과 지상 수위계 위치를 그리는 순수 도우미.

"위성 수위계랑 실제 수위계랑 어디가 동일한지, 위성 영상의 어느 부분이 어느 다리와 맵핑되는지" 를 보여 주는
두 그림을 만든다.

- overlay_figure: 퀵룩 PNG 위에 AOI 상자(청록)와 연결된 수위계(주황)를 그린다. 퀵룩의 EPSG:32652 범위·픽셀
  크기(백엔드가 scenes.json 에 넣어 주는 ``quicklook`` 블록)로 미터 좌표 → 픽셀 좌표를 아핀 변환한다.
- map_figure: 지도(carto-darkmatter) 위에 수위계 점·AOI 다각형·장면 발자국을 경위도로 그린다.

좌표계 주의 — 이 UI 가상환경에는 pyproj 가 없다. 그래서 투영 변환은 하지 않고, 백엔드가 미리 계산해 준
``bbox_epsg32652``(영상용)와 ``*_lonlat``(지도용)만 쓴다. 수위계 점만은 EPSG:32652 좌표가 계약에 없어서,
AOI 중심(두 좌표계 모두 제공됨)을 원점으로 한 **등장방형(equirectangular) 근사**로 미터 오프셋을 구한다:
``dx = (lon − lon0) · 111320 · cos(lat0)``, ``dy = (lat − lat0) · 110540``. 수위계는 AOI 에서 수백 m 안에
있으므로(1 km 규모) 축척 오차는 1 m 미만이다. 남는 오차는 UTM 격자 수렴각(중앙 자오선 129°E 에서 멀수록 큼 —
부산 ≈0°, 대청 ≈0.9°)에 의한 회전으로, 거리 × sin(수렴각) 만큼이다: 실측 검증(rasterio 재투영 대비)에서 부산
0.2~0.8 m, 대청 4~11 m(750 m 거리) = 퀵룩 0.4 px 미만이라 표시에는 문제없다. 이 근사는 화면 표시·거리 라벨에만
쓰고, 어떤 수치 산출에도 쓰지 않는다.

모든 함수는 순수 함수다(Streamlit·네트워크 없음) — 테스트가 픽셀 좌표를 직접 검사한다.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

AOI_COLOR = "#22d3ee"
AOI_FILL = "rgba(34,211,238,.28)"
STATION_COLOR = "#fb923c"
FOOTPRINT_COLOR = "#f472b6"
GAUGE_ONLY_COLOR = "#94a3b8"

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQUATOR = 111_320.0

Bounds = Sequence[float]  # [minx, miny, maxx, maxy]
LonLat = Sequence[float]  # [lon, lat]


# --- 좌표 변환(순수) ---------------------------------------------------------------------------


def to_pixel(x: float, y: float, bounds: Bounds, width: int, height: int) -> tuple[float, float]:
    """EPSG:32652 (x, y) → 퀵룩 픽셀 (px, py). 왼쪽 위가 (0, 0), y 는 아래로 커진다.

    bounds = [minx, miny, maxx, maxy] 가 영상의 (0,0)~(width,height) 에 정확히 대응한다는 아핀 가정.
    """

    minx, miny, maxx, maxy = (float(v) for v in bounds)
    span_x = maxx - minx
    span_y = maxy - miny
    if span_x <= 0 or span_y <= 0 or width <= 0 or height <= 0:
        raise ValueError("bounds/width/height 가 비어 있거나 뒤집혀 있습니다")
    px = (float(x) - minx) / span_x * float(width)
    py = (maxy - float(y)) / span_y * float(height)
    return px, py


def bbox_to_pixel_rect(bbox: Bounds, bounds: Bounds, width: int, height: int) -> tuple[float, float, float, float]:
    """AOI bbox [minx, miny, maxx, maxy] → 픽셀 사각형 (x0, y0, x1, y1), x0<x1·y0<y1 (y0 가 위)."""

    minx, miny, maxx, maxy = (float(v) for v in bbox)
    x0, y1 = to_pixel(minx, miny, bounds, width, height)
    x1, y0 = to_pixel(maxx, maxy, bounds, width, height)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def lonlat_offset_m(lonlat: LonLat, origin_lonlat: LonLat) -> tuple[float, float]:
    """origin 을 원점으로 한 (동쪽 m, 북쪽 m) 등장방형 근사(모듈 설명 참고)."""

    lon, lat = float(lonlat[0]), float(lonlat[1])
    lon0, lat0 = float(origin_lonlat[0]), float(origin_lonlat[1])
    dx = (lon - lon0) * M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    dy = (lat - lat0) * M_PER_DEG_LAT
    return dx, dy


def bbox_center(bbox: Bounds) -> tuple[float, float]:
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def inside(x: float, y: float, bounds: Bounds) -> bool:
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    return minx <= x <= maxx and miny <= y <= maxy


def station_lonlat(station: Mapping[str, Any] | None, location: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """수위계 경위도: WAMIS 지점 좌표(lat/lon) 우선, 없으면 AOI 가 스냅된 원점(anchor_lonlat)."""

    if station and isinstance(station.get("lon"), (int, float)) and isinstance(station.get("lat"), (int, float)):
        return float(station["lon"]), float(station["lat"])
    anchor = (location or {}).get("anchor_lonlat")
    if isinstance(anchor, Sequence) and len(anchor) == 2 and all(isinstance(v, (int, float)) for v in anchor):
        return float(anchor[0]), float(anchor[1])
    return None


def station_xy(location: Mapping[str, Any], station: Mapping[str, Any] | None = None) -> tuple[float, float] | None:
    """수위계의 EPSG:32652 근사 좌표. AOI 중심(bbox_epsg32652 ↔ center_lonlat)을 원점으로 오프셋을 더한다."""

    bbox = location.get("bbox_epsg32652")
    center_ll = location.get("center_lonlat")
    point = station_lonlat(station, location)
    if not bbox or not center_ll or point is None:
        return None
    cx, cy = bbox_center(bbox)
    dx, dy = lonlat_offset_m(point, center_ll)
    return cx + dx, cy + dy


def station_distance_m(location: Mapping[str, Any], station: Mapping[str, Any] | None = None) -> float | None:
    """수위계 ↔ AOI 중심 거리(m). 좌표가 없으면 계약의 snap_distance_m, 그것도 없으면 None."""

    xy = station_xy(location, station)
    if xy is None:
        value = location.get("snap_distance_m")
        return float(value) if isinstance(value, (int, float)) else None
    cx, cy = bbox_center(location["bbox_epsg32652"])
    return math.hypot(xy[0] - cx, xy[1] - cy)


def distance_label(meters: float | None) -> str:
    if meters is None:
        return "–"
    return f"{meters / 1000:.1f} km" if meters >= 1000 else f"{meters:.0f} m"


def short_note(text: Any, limit: int = 40) -> str:
    """구역 설정 메모를 표에 넣을 만큼만 자른다(첫 문장·limit 자)."""

    note = str(text or "").strip()
    if not note:
        return "–"
    first = note.split(" — ")[0].split(". ")[0].split("\n")[0].strip()
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def link_stations(locations: Mapping[str, Mapping[str, Any]], stations: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any] | None]:
    """loc_id → 연결된 수위계 레코드(없으면 None). 지점의 satellite_location 과 구역의 gauge_station 양쪽을 본다."""

    by_id = {str(s.get("id")): s for s in stations}
    by_name = {str(s.get("name")): s for s in stations}
    linked: dict[str, Mapping[str, Any] | None] = {}
    for loc_id, location in locations.items():
        station = next((s for s in stations if str(s.get("satellite_location") or "") == loc_id), None)
        if station is None:
            gauge = location.get("gauge_station")
            if gauge is not None:
                station = by_id.get(str(gauge)) or by_name.get(str(gauge))
        linked[loc_id] = station
    return linked


def auto_zoom(lons: Sequence[float], lats: Sequence[float], *, width_px: int = 640, height_px: int = 420, pad: float = 0.7) -> float:
    """점들이 다 보이는 지도 줌(웹 메르카토르 타일 512px 기준). 점 하나면 12."""

    if not lons or not lats:
        return 6.0
    span_lon = (max(lons) - min(lons)) * (1 + pad)
    mid_lat = (max(lats) + min(lats)) / 2.0
    span_lat = (max(lats) - min(lats)) * (1 + pad) / max(0.2, math.cos(math.radians(mid_lat)))
    if span_lon < 1e-6 and span_lat < 1e-6:
        return 12.0
    zoom_x = math.log2(360.0 * width_px / (512.0 * max(span_lon, 1e-6)))
    zoom_y = math.log2(360.0 * height_px / (512.0 * max(span_lat, 1e-6)))
    return max(3.0, min(15.0, min(zoom_x, zoom_y)))


# --- 그림 ---------------------------------------------------------------------------------------


def _spread_positions(xs: Sequence[float], ys: Sequence[float], *, near_px: float = 48.0) -> list[str]:
    """이웃한 구역(수백 m = 몇 픽셀)의 라벨이 겹치지 않게, 가까운 앞 점이 있으면 다른 방향으로 놓는다."""

    alternatives = ["top center", "top right", "top left", "bottom right", "bottom left"]
    positions: list[str] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        crowded = sum(1 for j in range(i) if abs(xs[j] - x) < near_px and abs(ys[j] - y) < near_px)
        positions.append(alternatives[min(crowded, len(alternatives) - 1)])
    return positions


def _label(loc_id: str, location: Mapping[str, Any]) -> str:
    return f"{loc_id} · {location.get('name_ko') or ''}".rstrip(" ·")


def _hover(location: Mapping[str, Any], loc_id: str, station: Mapping[str, Any] | None, distance: float | None) -> str:
    lines = [f"<b>{_label(loc_id, location)}</b>"]
    if location.get("river"):
        lines.append(f"하천 {location['river']}")
    if station is not None:
        lines.append(f"수위계 {station.get('name')} · {distance_label(distance)}")
    elif location.get("gauge_station"):
        lines.append(f"수위계 {location['gauge_station']} (좌표 없음)")
    else:
        lines.append("공개 수위계 없음")
    if location.get("note_ko"):
        lines.append(short_note(location.get("note_ko"), 60))
    return "<br>".join(lines)


def overlay_figure(
    image_bytes: bytes | None,
    quicklook: Mapping[str, Any],
    locations: Mapping[str, Mapping[str, Any]],
    stations: Sequence[Mapping[str, Any]],
    *,
    title: str = "",
    height: int = 620,
    zoom_to_locations: bool = False,
    zoom_pad_m: float = 2500.0,
    compact_labels: bool = False,
) -> go.Figure:
    """퀵룩 PNG 위에 AOI 상자·수위계·거리를 그린 Plotly Figure.

    quicklook = {"bounds_epsg32652": [...], "width": W, "height": H}. AOI 상자는 300 m 라 전체 영상에서는 몇 픽셀에
    불과하므로 상자 위에 큰 고리 마커와 라벨을 함께 찍고, zoom_to_locations 면 구역 주변만 보이게 축을 잘라 준다.
    compact_labels 면 라벨을 loc_id·지점명만으로 줄인다(전체 영상처럼 구역이 몰려 보일 때).
    """

    bounds = quicklook.get("bounds_epsg32652")
    width = int(quicklook.get("width") or 0)
    px_height = int(quicklook.get("height") or 0)
    if not bounds or width <= 0 or px_height <= 0:
        raise ValueError("quicklook 에 bounds_epsg32652/width/height 가 필요합니다")
    figure = go.Figure()
    if image_bytes:
        figure.add_layout_image(
            source="data:image/png;base64," + base64.b64encode(bytes(image_bytes)).decode("ascii"),
            xref="x", yref="y", x=0, y=0, sizex=width, sizey=px_height,
            xanchor="left", yanchor="top", sizing="stretch", layer="below", opacity=1.0,
        )
    m_per_px = (float(bounds[2]) - float(bounds[0])) / width
    linked = link_stations(locations, stations)
    ring_x: list[float] = []
    ring_y: list[float] = []
    ring_text: list[str] = []
    ring_hover: list[str] = []
    drawn: list[tuple[float, float, float, float]] = []
    for loc_id, location in locations.items():
        bbox = location.get("bbox_epsg32652")
        if not bbox:
            continue
        cx, cy = bbox_center(bbox)
        if not inside(cx, cy, bounds):
            continue
        x0, y0, x1, y1 = bbox_to_pixel_rect(bbox, bounds, width, px_height)
        figure.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line={"color": AOI_COLOR, "width": 2}, fillcolor=AOI_FILL, layer="above")
        station = linked.get(loc_id)
        distance = station_distance_m(location, station)
        pcx, pcy = to_pixel(cx, cy, bounds, width, px_height)
        ring_x.append(pcx)
        ring_y.append(pcy)
        ring_text.append(loc_id if compact_labels else _label(loc_id, location))
        ring_hover.append(_hover(location, loc_id, station, distance))
        drawn.append((x0, y0, x1, y1))
        xy = station_xy(location, station)
        if xy is not None and inside(xy[0], xy[1], bounds):
            sx, sy = to_pixel(xy[0], xy[1], bounds, width, px_height)
            figure.add_trace(
                go.Scatter(
                    x=[sx, pcx], y=[sy, pcy], mode="lines", line={"color": STATION_COLOR, "width": 1.5, "dash": "dash"},
                    hoverinfo="skip", showlegend=False,
                )
            )
            if station is not None:
                name = str(station.get("name") or location.get("gauge_station") or "수위계")
                text = name if compact_labels else f"{name} · 수위계"
                hover = (
                    f"<b>{name} · 지상 수위계</b><br>"
                    + (f"{station.get('address')}<br>" if station.get("address") else "")
                    + (f"하천 {station.get('river')}<br>" if station.get("river") else "")
                    + f"위성 산출 구역 {loc_id} 중심까지 {distance_label(distance)}"
                )
                marker = {"size": 13, "color": STATION_COLOR, "symbol": "triangle-up", "line": {"color": "#0f172a", "width": 1}}
            else:  # 공개 일수위가 없는 기준 지점(AOI 를 스냅한 WAMIS 지점) — 짝지은 수위계가 아니라는 것을 분명히 한다
                text = "기준점" if compact_labels else "기준점 · 수위계 자료 없음"
                hover = (
                    f"<b>구역 {loc_id} 의 기준 지점</b><br>WAMIS 지점 {location.get('obscd') or '?'} (공개 일수위 없음 → 대조 불가)<br>"
                    f"구역 중심까지 {distance_label(distance)}"
                )
                marker = {"size": 11, "color": "rgba(0,0,0,0)", "symbol": "triangle-up", "line": {"color": STATION_COLOR, "width": 1.5}}
            figure.add_trace(
                go.Scatter(
                    x=[sx], y=[sy], mode="markers+text", text=[text], textposition="bottom center" if station is not None else "bottom left",
                    textfont={"color": STATION_COLOR, "size": 12}, marker=marker, cliponaxis=False,
                    hovertemplate=hover + "<extra></extra>", showlegend=False,
                )
            )
            figure.add_annotation(
                x=(sx + pcx) / 2, y=(sy + pcy) / 2, text=distance_label(distance), showarrow=False,
                font={"color": STATION_COLOR, "size": 11}, bgcolor="rgba(2,6,23,.65)", xshift=8, yshift=-8,
            )
            drawn.append((min(sx, x0), min(sy, y0), max(sx, x1), max(sy, y1)))
    if ring_x:
        figure.add_trace(
            go.Scatter(
                x=ring_x, y=ring_y, mode="markers+text", text=ring_text, textposition=_spread_positions(ring_x, ring_y), cliponaxis=False,
                textfont={"color": AOI_COLOR, "size": 12},
                marker={"size": 22, "color": "rgba(0,0,0,0)", "line": {"color": AOI_COLOR, "width": 2.5}, "symbol": "circle"},
                hovertemplate="%{customdata}<extra></extra>", customdata=ring_hover, showlegend=False, name="위성 산출 구역",
            )
        )
    x_range: list[float] = [0, width]
    y_range: list[float] = [px_height, 0]
    if zoom_to_locations and drawn:
        pad_px = zoom_pad_m / m_per_px
        x_lo = max(0.0, min(d[0] for d in drawn) - pad_px)
        x_hi = min(float(width), max(d[2] for d in drawn) + pad_px)
        y_lo = max(0.0, min(d[1] for d in drawn) - pad_px)
        y_hi = min(float(px_height), max(d[3] for d in drawn) + pad_px)
        # 너무 좁은 창(구역 하나·짧은 거리)은 최소 폭을 보장해 픽셀 뭉개짐을 줄인다
        min_px = min(float(min(width, px_height)), 240.0)
        if x_hi - x_lo < min_px:
            mid = (x_lo + x_hi) / 2
            x_lo, x_hi = max(0.0, mid - min_px / 2), min(float(width), mid + min_px / 2)
        if y_hi - y_lo < min_px:
            mid = (y_lo + y_hi) / 2
            y_lo, y_hi = max(0.0, mid - min_px / 2), min(float(px_height), mid + min_px / 2)
        x_range = [x_lo, x_hi]
        y_range = [y_hi, y_lo]
    figure.update_xaxes(range=x_range, visible=False, constrain="domain")
    figure.update_yaxes(range=y_range, visible=False, scaleanchor="x", scaleratio=1, constrain="domain")
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,6,23,.6)",
        font={"color": "#cbd5e1", "size": 12}, height=height, hovermode="closest",
        margin={"l": 6, "r": 6, "t": 40 if title else 8, "b": 6},
        showlegend=False, dragmode="pan",
    )
    if title:  # title=None 을 넘기면 layout.title={} 가 되어 Streamlit 테마가 "undefined" 를 제목으로 찍는다
        figure.update_layout(title={"text": title, "x": 0.01, "font": {"size": 14}})
    scale_note = f"1 px ≈ {m_per_px:.0f} m · 청록 상자 = 위성 수위 산출 구역(AOI) · 주황 ▲ = 지상 수위계"
    figure.add_annotation(
        xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="bottom", text=scale_note, showarrow=False,
        font={"size": 10, "color": "#cbd5e1"}, bgcolor="rgba(2,6,23,.65)",
    )
    return figure


def _closed_ring(points: Sequence[LonLat]) -> tuple[list[float], list[float]]:
    lons = [float(p[0]) for p in points]
    lats = [float(p[1]) for p in points]
    if lons and (lons[0] != lons[-1] or lats[0] != lats[-1]):
        lons.append(lons[0])
        lats.append(lats[0])
    return lons, lats


def map_figure(
    region: Mapping[str, Any],
    locations: Mapping[str, Mapping[str, Any]],
    stations: Sequence[Mapping[str, Any]],
    scene_footprint: Sequence[LonLat] | None = None,
    *,
    height: int = 420,
    title: str = "",
) -> go.Figure:
    """지도 위 수위계 점·AOI 다각형·장면 발자국. 점이 하나도 없으면 지역 bbox 중심만 보여 준다."""

    figure = go.Figure()
    lons: list[float] = []
    lats: list[float] = []
    linked = link_stations(locations, stations)
    linked_ids = {id(s) for s in linked.values() if s is not None}

    footprint_lons: list[float] = []
    footprint_lats: list[float] = []
    if scene_footprint:
        footprint_lons, footprint_lats = _closed_ring(scene_footprint)
        figure.add_trace(
            go.Scattermap(
                lon=footprint_lons, lat=footprint_lats, mode="lines", line={"color": FOOTPRINT_COLOR, "width": 1.5}, fill="toself",
                fillcolor="rgba(244,114,182,.06)", name="위성 장면 범위", hoverinfo="name",
            )
        )

    for loc_id, location in locations.items():
        ring = location.get("bbox_lonlat")
        if not ring:
            continue
        b_lon, b_lat = _closed_ring(ring)
        station = linked.get(loc_id)
        hover = _hover(location, loc_id, station, station_distance_m(location, station))
        figure.add_trace(
            go.Scattermap(
                lon=b_lon, lat=b_lat, mode="lines", line={"color": AOI_COLOR, "width": 2}, fill="toself", fillcolor=AOI_FILL,
                name=_label(loc_id, location), hovertemplate=hover + "<extra></extra>", showlegend=False,
            )
        )
        center = location.get("center_lonlat") or [sum(b_lon[:-1]) / max(1, len(b_lon) - 1), sum(b_lat[:-1]) / max(1, len(b_lat) - 1)]
        figure.add_trace(
            go.Scattermap(
                lon=[float(center[0])], lat=[float(center[1])], mode="markers+text", text=[_label(loc_id, location)], textposition="top right",
                textfont={"color": AOI_COLOR, "size": 12},
                marker={"size": 16, "color": AOI_COLOR, "opacity": 0.35}, hovertemplate=hover + "<extra></extra>", showlegend=False,
            )
        )
        lons += b_lon
        lats += b_lat

    with_coords = [s for s in stations if isinstance(s.get("lon"), (int, float)) and isinstance(s.get("lat"), (int, float))]
    without = [str(s.get("name")) for s in stations if s not in with_coords]
    if with_coords:
        figure.add_trace(
            go.Scattermap(
                lon=[float(s["lon"]) for s in with_coords], lat=[float(s["lat"]) for s in with_coords], mode="markers+text",
                text=[str(s.get("name")) for s in with_coords], textposition="bottom center", textfont={"color": "#fde68a", "size": 12},
                marker={"size": [14 if id(s) in linked_ids else 11 for s in with_coords], "color": [STATION_COLOR if id(s) in linked_ids else GAUGE_ONLY_COLOR for s in with_coords]},
                customdata=[
                    [
                        str(s.get("address") or "주소 미상"), str(s.get("river") or "–"),
                        str(s.get("terrain_note") or s.get("terrain") or "–"),
                        (f"위성 산출 구역 {s.get('satellite_location')} 과 연결" if s.get("satellite_location") else "위성 연결 없음"),
                        " ~ ".join(str(v) for v in (s.get("period") or [])) or "–",
                    ]
                    for s in with_coords
                ],
                hovertemplate="<b>%{text} · 지상 수위계</b><br>%{customdata[0]}<br>하천 %{customdata[1]} · 지형 %{customdata[2]}<br>%{customdata[3]}<br>기간 %{customdata[4]}<extra></extra>",
                name="지상 수위계", showlegend=False,
            )
        )
        lons += [float(s["lon"]) for s in with_coords]
        lats += [float(s["lat"]) for s in with_coords]

    if not lons:  # 지점·구역이 없으면 장면 발자국에라도 맞춘다
        lons, lats = footprint_lons, footprint_lats
    if lons:
        # 줌은 수위계·구역(다리 이름이 읽히는 규모)에 맞춘다 — 장면 발자국(수십 km)은 테두리로만 보인다
        center = {"lon": (max(lons) + min(lons)) / 2.0, "lat": (max(lats) + min(lats)) / 2.0}
        zoom = auto_zoom(lons, lats, height_px=height)
    else:
        bbox = region.get("bbox_wgs84")
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            center = {"lon": (float(bbox[0]) + float(bbox[2])) / 2.0, "lat": (float(bbox[1]) + float(bbox[3])) / 2.0}
            zoom = auto_zoom([float(bbox[0]), float(bbox[2])], [float(bbox[1]), float(bbox[3])], height_px=height)
        else:
            center, zoom = {"lon": 127.6, "lat": 36.4}, 5.6
    figure.update_layout(
        map={"style": "carto-darkmatter", "center": center, "zoom": zoom}, height=height,
        margin={"l": 0, "r": 0, "t": 34 if title else 4, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        font={"color": "#cbd5e1"},
    )
    if title:  # 위와 같은 이유로 제목이 있을 때만 넣는다
        figure.update_layout(title={"text": title, "x": 0.01, "font": {"size": 14, "color": "#cbd5e1"}})
    if without:
        figure.add_annotation(
            xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="bottom", showarrow=False,
            text="좌표 없음: " + ", ".join(without), font={"size": 10, "color": "#cbd5e1"}, bgcolor="rgba(2,6,23,.65)",
        )
    return figure


def location_table_fields(location: Mapping[str, Any] | None, station: Mapping[str, Any] | None) -> dict[str, str]:
    """대조표에 붙는 세 열 — '위성 산출 구역', '수위계까지 거리', '비고(구역 설정 메모)'."""

    if not location:
        return {"위성 산출 구역": "–", "수위계까지 거리": "–", "구역 설정 메모": "–"}
    loc_id = str(location.get("loc_id") or location.get("id") or "")
    label = f"{loc_id} · {location.get('name_ko')}" if loc_id and location.get("name_ko") else (loc_id or str(location.get("name_ko") or "–"))
    return {
        "위성 산출 구역": label,
        "수위계까지 거리": distance_label(station_distance_m(location, station)),
        "구역 설정 메모": short_note(location.get("note_ko")),
    }
