"""WBMS 센서·테스트베드 카탈로그.

과거에는 센서 키·가중치 파일명·테스트베드 코드가 어댑터·스키마·체인 러너에 각각
하드코딩되어 있어 새 지역이나 센서를 붙이려면 세 곳을 고쳐야 했다. 이제 한 JSON
(``catalog.json``, ``BACKEND_WBMS_CATALOG_JSON`` 으로 교체 가능)을 모두가 읽는다.

한계: 컨테이너 이미지 안의 ``wbms_common.py`` 도 같은 키 목록을 갖고 있어, 카탈로그에
추가한 값이 실제로 돌려면 컨테이너 쪽도 맞춰야 한다. 카탈로그는 백엔드가 요청을
검증하고 경로를 조립하는 단일 근거이지 컨테이너를 대신하지 않는다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


@dataclass(frozen=True, slots=True)
class WbmsCatalog:
    sensors: dict[str, dict[str, Any]]
    testbeds: dict[str, dict[str, Any]]
    source: Path = field(default=DEFAULT_CATALOG_PATH)

    @classmethod
    def load(cls, path: Path | None = None) -> WbmsCatalog:
        raw = os.getenv("BACKEND_WBMS_CATALOG_JSON", "").strip()
        target = path or (Path(raw).resolve() if raw else DEFAULT_CATALOG_PATH)
        payload = json.loads(target.read_text(encoding="utf-8"))
        sensors = {str(k).lower(): dict(v) for k, v in (payload.get("sensors") or {}).items()}
        testbeds = {str(k).lower(): dict(v) for k, v in (payload.get("testbeds") or {}).items()}
        if not sensors or not testbeds:
            raise ValueError(f"WBMS 카탈로그에 sensors/testbeds 가 비어 있습니다: {target}")
        for key, profile in sensors.items():
            for required in ("name", "type", "weights", "pre_dirs"):
                if required not in profile:
                    raise ValueError(f"sensor {key!r} 에 {required!r} 가 없습니다")
            profile.setdefault("needs_mask", profile["type"].upper() == "SAR")
            # 컨테이너에 넘길 ``--sensor`` 값. 전용 모델이 없는 센서(umbra)는 다른 센서의 모델·프로파일을
            # 빌려 쓰며, 그때 산출물 이름·qc 의 sensor_name 은 빌린 센서 표기(ICEYE)로 나온다.
            profile["module_sensor"] = str(profile.get("module_sensor") or key).lower()
            # ① 파일명 ``Pre_<TB>_<라벨>_…`` 의 라벨. 기본은 name 과 같다(ICEYE·PlanetScope); Umbra 는 UMBRA.
            profile["file_label"] = str(profile.get("file_label") or profile["name"])
        for key, profile in sensors.items():
            if profile["module_sensor"] not in sensors:
                raise ValueError(
                    f"sensor {key!r} 의 module_sensor {profile['module_sensor']!r} 가 카탈로그에 없습니다"
                )
        for key, profile in testbeds.items():
            profile.setdefault("code", key.capitalize())
            profile.setdefault("mask_template", "Masks/Img_DEM_3class_utm/{testbed}_{image_date}_mask3_utm.tif")
        return cls(sensors=sensors, testbeds=testbeds, source=target)

    @property
    def tb_codes(self) -> dict[str, str]:
        return {key: str(profile["code"]) for key, profile in self.testbeds.items()}

    def sensor(self, key: str) -> dict[str, Any]:
        try:
            return self.sensors[str(key).lower()]
        except KeyError as exc:
            raise KeyError(f"unknown sensor {key!r}; known: {sorted(self.sensors)}") from exc

    def testbed(self, key: str) -> dict[str, Any]:
        try:
            return self.testbeds[str(key).lower()]
        except KeyError as exc:
            raise KeyError(f"unknown testbed {key!r}; known: {sorted(self.testbeds)}") from exc

    def module_sensor(self, key: str) -> str:
        """컨테이너 ``--sensor`` 에 넘길 키. 전용 모델이 있는 센서는 자기 키, umbra 는 iceye."""

        return str(self.sensor(key)["module_sensor"])

    def module_name(self, key: str) -> str:
        """컨테이너가 산출물 이름(``WB_<TB>_<표기>_…``)·qc ``params.sensor_name`` 에 적는 표기(모듈 KEY2NAME)."""

        return str(self.sensor(self.module_sensor(key))["name"])

    def file_label(self, key: str) -> str:
        """① 파일명 ``Pre_<TB>_<라벨>_…`` 에 쓰는 라벨 — 장면 API 의 ``sensors[].id`` 이기도 하다."""

        return str(self.sensor(key)["file_label"])

    def cache_sensor_names(self, key: str, testbed: str | None = None) -> list[str]:
        """캐시된 ② 산출물에서 이 센서 것으로 받아들일 표기 — 자기 이름·대문자·파일 라벨·빌린 모듈 표기 순.

        빌린 모듈 표기(umbra → ICEYE)는 이 센서가 실제로 인수된 테스트베드(``pre_dirs``)에서만 받는다 —
        umbra 를 busan 으로 조회했을 때 ICEYE 부산 산출물을 집어오지 않게 한다. ``testbed`` 가 None 이면 모두 받는다.
        """

        profile = self.sensor(key)
        names = [str(profile["name"]), str(profile["name"]).upper(), str(profile["file_label"])]
        if self.module_sensor(key) == str(key).lower() or testbed is None or self.supports(key, testbed):
            names.append(self.module_name(key))
        return list(dict.fromkeys(names))

    def resolve_key(self, text: str) -> str | None:
        """키·표시명·파일 라벨 어느 표기로 와도 카탈로그 키로(대소문자 무시). 모르면 None."""

        needle = str(text or "").strip().lower()
        if not needle:
            return None
        for key, profile in self.sensors.items():
            if needle in (key, str(profile["name"]).lower(), str(profile["file_label"]).lower()):
                return key
        return None

    def supports(self, sensor: str, testbed: str) -> bool:
        try:
            return str(testbed).lower() in self.sensor(sensor)["pre_dirs"]
        except KeyError:
            return False

    def capabilities(self) -> dict[str, Any]:
        """UI/API가 '이 플랫폼이 지금 받을 수 있는 입력'을 보여줄 때 쓰는 요약."""

        return {
            "source": str(self.source),
            "sensors": {
                key: {
                    "name": p["name"],
                    "type": p["type"],
                    "weights": p["weights"],
                    "testbeds": sorted(p["pre_dirs"]),
                    "needs_mask": bool(p.get("needs_mask")),
                    "module_sensor": p["module_sensor"],
                    "file_label": p["file_label"],
                    "description_ko": p.get("description_ko", ""),
                }
                for key, p in self.sensors.items()
            },
            "testbeds": {
                key: {
                    "code": p["code"],
                    "name_ko": p.get("name_ko", key),
                    "region_id": p.get("region_id"),
                    "crs": p.get("crs"),
                    "stations": list(p.get("stations") or []),
                }
                for key, p in self.testbeds.items()
            },
        }


_DEFAULT: WbmsCatalog | None = None


def default_catalog() -> WbmsCatalog:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = WbmsCatalog.load()
    return _DEFAULT
