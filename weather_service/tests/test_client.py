from __future__ import annotations

import json
import os
import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from weather_service import (
    AsosDailyClient,
    HttpResponse,
    WeatherApiError,
    WeatherConfigurationError,
    WeatherResponseFormatError,
    WeatherTransportError,
    WeatherValidationError,
)


FIXED_TODAY = date(2025, 1, 10)
FIXED_NOW = datetime(2025, 1, 10, 3, 0, tzinfo=timezone.utc)


def api_payload(items, *, total_count=None, code="00", message="NORMAL_SERVICE"):
    if total_count is None:
        total_count = len(items)
    return {
        "response": {
            "header": {"resultCode": code, "resultMsg": message},
            "body": {
                "dataType": "JSON",
                "items": {"item": items},
                "pageNo": 1,
                "numOfRows": 10,
                "totalCount": total_count,
            },
        }
    }


def json_response(payload, status=200):
    return HttpResponse(status, json.dumps(payload).encode("utf-8"), {})


class QueueTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class AsosDailyClientTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, transport, **kwargs):
        return AsosDailyClient(
            "test-key",
            transport=transport,
            max_retries=0,
            today_provider=lambda: FIXED_TODAY,
            now_provider=lambda: FIXED_NOW,
            **kwargs,
        )

    async def test_fetches_all_pages_and_parses_model_fields(self):
        page_one = api_payload(
            [
                {
                    "tm": "2025-01-01",
                    "stnId": "159",
                    "avgTa": "4.5",
                    "minTa": "1.0",
                    "maxTa": "8.2",
                    "avgRhm": "63",
                    "sumRn": "2.4",
                    "avgPa": "1014.1",
                    "avgWs": "2.7",
                },
                {"tm": "2025-01-02", "stnId": "159", "sumRn": ""},
            ],
            total_count=3,
        )
        page_two = api_payload(
            [{"tm": "2025-01-03", "stnId": "159", "sumRn": "0"}],
            total_count=3,
        )
        transport = QueueTransport([json_response(page_one), json_response(page_two)])
        client = self.make_client(transport, page_size=2)

        series = await client.get_daily_observations("20250101", "2025-01-03", "부산 (낙동강 하구)")

        self.assertEqual(series.station.id, "159")
        self.assertEqual(len(series.records), 3)
        self.assertEqual(series.records[0].precipitation_mm, 2.4)
        self.assertEqual(series.records[0].avg_temperature_c, 4.5)
        self.assertIsNone(series.records[1].precipitation_mm)
        self.assertEqual(series.records[2].precipitation_mm, 0.0)
        self.assertEqual(series.missing_dates, ())
        self.assertEqual(series.metadata.data_category, "historical_observation")
        self.assertFalse(series.metadata.is_forecast)
        self.assertFalse(series.metadata.is_sample)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[0]["params"]["pageNo"], "1")
        self.assertEqual(transport.calls[1]["params"]["pageNo"], "2")
        self.assertEqual(transport.calls[0]["params"]["stnIds"], "159")

    async def test_decodes_portal_encoded_key_once_before_transport(self):
        transport = QueueTransport([json_response(api_payload([], total_count=0))])
        client = AsosDailyClient(
            "abc%2B%2F%3D",
            transport=transport,
            max_retries=0,
            today_provider=lambda: FIXED_TODAY,
        )
        await client.get_daily_observations("20250101", "20250101", 108)
        self.assertEqual(transport.calls[0]["params"]["ServiceKey"], "abc+/=")

    async def test_https_is_default_and_http_fallback_is_explicit(self):
        transport = QueueTransport(
            [OSError("tls unavailable"), json_response(api_payload([], total_count=0))]
        )
        client = self.make_client(transport, allow_http_fallback=True)

        series = await client.get_daily_observations("20250101", "20250101", "서울")

        self.assertTrue(transport.calls[0]["url"].startswith("https://"))
        self.assertTrue(transport.calls[1]["url"].startswith("http://"))
        self.assertTrue(series.metadata.endpoint_url.startswith("http://"))

    async def test_does_not_fallback_when_disabled(self):
        transport = QueueTransport([OSError("tls unavailable")])
        client = self.make_client(transport)
        with self.assertRaises(WeatherTransportError):
            await client.get_daily_observations("20250101", "20250101", "서울")
        self.assertEqual(len(transport.calls), 1)

    async def test_api_error_is_typed_and_key_is_not_in_message(self):
        payload = {"response": {"header": {"resultCode": "30", "resultMsg": "SERVICE_KEY_IS_NOT_REGISTERED_ERROR"}}}
        transport = QueueTransport([json_response(payload)])
        client = self.make_client(transport)

        with self.assertRaises(WeatherApiError) as caught:
            await client.get_daily_observations("20250101", "20250101", "서울")
        self.assertEqual(caught.exception.code, "30")
        self.assertNotIn("test-key", str(caught.exception))

    async def test_xml_gateway_error_is_parsed(self):
        xml = b"""<?xml version='1.0' encoding='UTF-8'?>
        <OpenAPI_ServiceResponse><cmmMsgHeader>
          <returnReasonCode>30</returnReasonCode>
          <returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg>
        </cmmMsgHeader></OpenAPI_ServiceResponse>"""
        transport = QueueTransport([HttpResponse(200, xml, {})])
        client = self.make_client(transport)
        with self.assertRaises(WeatherApiError) as caught:
            await client.get_daily_observations("20250101", "20250101", "서울")
        self.assertEqual(caught.exception.code, "30")

    async def test_no_data_error_becomes_empty_series(self):
        payload = {"response": {"header": {"resultCode": "03", "resultMsg": "NODATA_ERROR"}}}
        transport = QueueTransport([json_response(payload)])
        client = self.make_client(transport)
        series = await client.get_daily_observations("20250101", "20250102", "서울")
        self.assertEqual(series.records, ())
        self.assertEqual(series.missing_dates, (date(2025, 1, 1), date(2025, 1, 2)))

    async def test_reports_missing_calendar_dates(self):
        payload = api_payload(
            [
                {"tm": "2025-01-01", "stnId": "108"},
                {"tm": "2025-01-03", "stnId": "108"},
            ],
            total_count=2,
        )
        client = self.make_client(QueueTransport([json_response(payload)]))
        series = await client.get_daily_observations("20250101", "20250103", "서울")
        self.assertEqual(series.missing_dates, (date(2025, 1, 2),))

    async def test_rejects_future_or_same_day_as_observation(self):
        client = self.make_client(QueueTransport([]))
        with self.assertRaisesRegex(WeatherValidationError, "not a forecast"):
            await client.get_daily_observations("20250101", "20250110", "서울")

    async def test_rejects_malformed_item(self):
        client = self.make_client(
            QueueTransport([json_response(api_payload([{"stnId": "108"}]))])
        )
        with self.assertRaises(WeatherResponseFormatError):
            await client.get_daily_observations("20250101", "20250101", "서울")

    def test_reads_key_from_environment_and_argument_takes_precedence(self):
        with patch.dict(os.environ, {"CUSTOM_KMA_KEY": "environment-key"}):
            client_from_env = AsosDailyClient(env_var="CUSTOM_KMA_KEY")
            client_from_arg = AsosDailyClient("argument-key", env_var="CUSTOM_KMA_KEY")
        self.assertEqual(client_from_env._api_key, "environment-key")
        self.assertEqual(client_from_arg._api_key, "argument-key")

    def test_missing_key_is_configuration_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(WeatherConfigurationError):
                AsosDailyClient(env_var="MISSING_KMA_KEY")


if __name__ == "__main__":
    unittest.main()

