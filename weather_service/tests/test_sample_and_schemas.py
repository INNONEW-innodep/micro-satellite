from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from weather_service import (
    DailyWeatherObservation,
    SampleWeatherProvider,
    WeatherValidationError,
    list_stations,
    parse_date,
    resolve_station,
)


class StationAndSchemaTests(unittest.TestCase):
    def test_ui_region_aliases_resolve_to_official_ids(self):
        expected = {
            "서울 (한강)": "108",
            "부산 (낙동강 하구)": "159",
            "대구 (금호강)": "143",
            "광주 (영산강)": "156",
        }
        for label, station_id in expected.items():
            with self.subTest(label=label):
                self.assertEqual(resolve_station(label).id, station_id)

    def test_unknown_numeric_station_is_allowed_but_unknown_name_is_not(self):
        self.assertEqual(resolve_station(999).id, "999")
        self.assertIsNone(resolve_station(999).name)
        with self.assertRaises(WeatherValidationError):
            resolve_station("존재하지 않는 지역")

    def test_mapping_includes_requested_major_stations(self):
        station_ids = {station.id for station in list_stations()}
        self.assertTrue({"108", "159", "143", "156", "112", "119", "133", "152", "184"} <= station_ids)

    def test_parse_date_accepts_api_and_iso_formats(self):
        expected = date(2025, 1, 2)
        self.assertEqual(parse_date("20250102"), expected)
        self.assertEqual(parse_date("2025-01-02"), expected)
        self.assertEqual(parse_date(datetime(2025, 1, 2, 15, 30)), expected)

    def test_missing_numeric_fields_remain_none(self):
        record = DailyWeatherObservation.from_api_item(
            {"tm": "2025-01-01", "stnId": "108", "sumRn": "", "avgTa": "bad"},
            station_name="서울",
        )
        self.assertIsNone(record.precipitation_mm)
        self.assertIsNone(record.avg_temperature_c)
        self.assertEqual(record.station_name, "서울")


class SampleProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_sample_is_deterministic_and_unambiguously_labelled(self):
        provider = SampleWeatherProvider(
            seed=7,
            today_provider=lambda: date(2025, 1, 10),
            now_provider=lambda: datetime(2025, 1, 10, tzinfo=timezone.utc),
        )
        first = await provider.get_daily_observations("20250101", "20250103", "부산")
        second = await provider.get_daily_observations("20250101", "20250103", "부산")

        self.assertEqual(first.records, second.records)
        self.assertEqual(len(first.records), 3)
        self.assertEqual(first.metadata.mode, "sample")
        self.assertEqual(first.metadata.data_category, "synthetic_sample")
        self.assertTrue(first.metadata.is_sample)
        self.assertFalse(first.metadata.is_forecast)
        self.assertIsNone(first.metadata.source_url)
        self.assertIn("합성 샘플", first.metadata.disclaimer_ko)
        self.assertTrue(all(record.raw.get("sample") is True for record in first.records))

    async def test_sample_also_rejects_future_dates(self):
        provider = SampleWeatherProvider(today_provider=lambda: date(2025, 1, 10))
        with self.assertRaisesRegex(WeatherValidationError, "not a forecast"):
            await provider.get_daily_observations("20250110", "20250111", "서울")

    async def test_series_serialization_is_api_ready(self):
        provider = SampleWeatherProvider(
            today_provider=lambda: date(2025, 1, 10),
            now_provider=lambda: datetime(2025, 1, 10, tzinfo=timezone.utc),
        )
        series = await provider.get_daily_observations("20250101", "20250101", "서울")
        payload = series.to_dict()
        self.assertEqual(payload["requested_start"], "2025-01-01")
        self.assertEqual(payload["record_count"], 1)
        self.assertEqual(payload["records"][0]["date"], "2025-01-01")
        self.assertNotIn("raw", payload["records"][0])
        self.assertIn("raw", series.to_dict(include_raw=True)["records"][0])


if __name__ == "__main__":
    unittest.main()

