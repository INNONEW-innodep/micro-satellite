"""Small, explicit ASOS station mapping for the project UI and API.

Numeric IDs outside this convenience map remain usable because the official
station catalogue changes over time. Names and aliases here are for resolving
the regions currently exposed by the project UI.
"""

from __future__ import annotations

import re
from types import MappingProxyType

from .errors import WeatherValidationError
from .schemas import Station


_STATIONS = (
    Station("108", "서울", ("서울특별시", "한강", "서울 (한강)")),
    Station("159", "부산", ("부산광역시", "낙동강", "낙동강 하구", "부산 (낙동강 하구)")),
    Station("143", "대구", ("대구광역시", "금호강", "대구 (금호강)")),
    Station("156", "광주", ("광주광역시", "영산강", "광주 (영산강)")),
    Station("112", "인천", ("인천광역시",)),
    Station("119", "수원", ("수원시",)),
    Station("133", "대전", ("대전광역시",)),
    Station("152", "울산", ("울산광역시",)),
    Station("184", "제주", ("제주시", "제주도", "제주특별자치도")),
)

ASOS_STATIONS = MappingProxyType({station.id: station for station in _STATIONS})


def _normalize_name(value: str) -> str:
    return re.sub(r"[\s()_-]+", "", value).casefold()


_ALIASES: dict[str, Station] = {}
for _station in _STATIONS:
    for _name in (_station.name, *_station.aliases):
        if _name:
            _ALIASES[_normalize_name(_name)] = _station


def resolve_station(value: Station | str | int) -> Station:
    """Resolve an ID, Korean region label, or existing :class:`Station`.

    An unknown numeric ID is accepted so newly added official ASOS stations do
    not require a package release. Unknown human-readable names are rejected.
    """

    if isinstance(value, Station):
        return value
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        raise WeatherValidationError("station must be a Station, ASOS id, or known region name")

    cleaned = value.strip()
    if cleaned.isdigit():
        if cleaned in ASOS_STATIONS:
            return ASOS_STATIONS[cleaned]
        return Station(cleaned, None)

    station = _ALIASES.get(_normalize_name(cleaned))
    if station is None:
        known = ", ".join(station.name or station.id for station in _STATIONS)
        raise WeatherValidationError(
            f"unknown ASOS station name {value!r}; use a numeric station id or one of: {known}"
        )
    return station


def list_stations() -> tuple[Station, ...]:
    """Return stations included in the built-in convenience mapping."""

    return _STATIONS

