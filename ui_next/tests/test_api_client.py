import json
import unittest

from ui_next.api_client import (
    APIClient,
    APIError,
    UploadPart,
    weather_records,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, content_type="application/json", content=b""):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class APIClientTests(unittest.TestCase):
    def test_create_prediction_multipart_contract(self):
        session = FakeSession([FakeResponse({"id": "p1", "status": "completed"})])
        client = APIClient("http://backend:8000/", session=session)
        result = client.create_prediction(
            files=[UploadPart("a.tif", b"abc", "image/tiff"), UploadPart("b.npy", b"def")],
            model_id="m1",
            horizon_steps=2,
            threshold=0.4,
            source_dates=["2026-01-01", "2026-01-02"],
            input_metadata={"sample_id": "nas-demo", "sensor": "optical"},
            weather=[{"timestamp": "2026-01-01", "kind": "observed"}],
            pixel_area_m2=9,
            target_dates=["2026-01-03", "2026-01-04"],
            historical_water_levels=[1.1, None],
            reference_water_levels=[1.2, 1.3],
            evaluation_kind="holdout",
            evaluation_truth_provenance="station test split",
            water_level_config=None,
            model_options={"alpha": 2},
            caution_pct=4,
            risk_pct=12,
        )
        self.assertEqual(result["id"], "p1")
        method, url, kwargs = session.calls[0]
        self.assertEqual((method, url), ("POST", "http://backend:8000/api/v1/predictions"))
        self.assertEqual([field for field, _ in kwargs["files"]], ["files", "files"])
        self.assertEqual(json.loads(kwargs["data"]["source_dates_json"]), ["2026-01-01", "2026-01-02"])
        self.assertEqual(json.loads(kwargs["data"]["target_dates_json"]), ["2026-01-03", "2026-01-04"])
        self.assertEqual(
            json.loads(kwargs["data"]["input_metadata_json"]),
            {"sample_id": "nas-demo", "sensor": "optical"},
        )
        self.assertEqual(json.loads(kwargs["data"]["historical_water_levels_json"]), [1.1, None])
        self.assertEqual(json.loads(kwargs["data"]["reference_water_levels_json"]), [1.2, 1.3])
        self.assertEqual(kwargs["data"]["evaluation_kind"], "holdout")
        self.assertEqual(
            kwargs["data"]["evaluation_truth_provenance"], "station test split"
        )
        self.assertEqual(json.loads(kwargs["data"]["weather_json"])[0]["date"], "2026-01-01")
        self.assertEqual(json.loads(kwargs["data"]["model_options_json"]), {"alpha": 2})
        self.assertEqual(kwargs["data"]["caution_pct"], "4")
        self.assertEqual(kwargs["data"]["risk_pct"], "12")

    def test_list_predictions_uses_pagination(self):
        session = FakeSession([FakeResponse({"items": [{"id": "p1"}], "total": 1})])
        client = APIClient("http://backend", session=session)
        result = client.list_predictions(offset=5, limit=7)
        self.assertEqual(result["total"], 1)
        method, url, kwargs = session.calls[0]
        self.assertEqual((method, url), ("GET", "http://backend/api/v1/predictions"))
        self.assertEqual(kwargs["params"], {"offset": 5, "limit": 7})

    def test_artifact_path_is_quoted(self):
        session = FakeSession([FakeResponse(content_type="image/png", content=b"png")])
        client = APIClient("http://backend", session=session)
        self.assertEqual(client.download_artifact("p/1", "mask one.png"), b"png")
        self.assertTrue(session.calls[0][1].endswith("/api/v1/predictions/p%2F1/files/mask%20one.png"))

    def test_file_list_path_is_quoted_and_normalized(self):
        session = FakeSession(
            [FakeResponse({"prediction_id": "p/1", "items": [{"name": "result.json"}]})]
        )
        client = APIClient("http://backend", session=session)
        self.assertEqual(client.list_files("p/1"), [{"name": "result.json"}])
        self.assertTrue(
            session.calls[0][1].endswith("/api/v1/predictions/p%2F1/files")
        )

    def test_error_includes_server_detail(self):
        session = FakeSession([FakeResponse({"detail": "bad input"}, status_code=422)])
        with self.assertRaises(APIError) as raised:
            APIClient("http://backend", session=session).health()
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(str(raised.exception), "bad input")

    def test_weather_records_nested_shape(self):
        self.assertEqual(weather_records({"data": {"records": [{"date": "2026-01-01"}]}})[0]["date"], "2026-01-01")
        self.assertEqual(weather_records({"rows": [{"date": "2026-01-02"}]})[0]["date"], "2026-01-02")

    def test_weather_status_and_per_request_key(self):
        session = FakeSession(
            [
                FakeResponse({"server_key_configured": False, "sample_available": True}),
                FakeResponse({"rows": []}),
            ]
        )
        client = APIClient("http://backend", session=session)
        self.assertFalse(client.weather_status()["server_key_configured"])
        client.weather_observations(
            source="asos",
            station_id="159",
            start_date="2026-08-01",
            end_date="2026-08-02",
            service_key="request-key",
        )
        self.assertEqual(session.calls[0][:2], ("GET", "http://backend/api/v1/weather/status"))
        self.assertEqual(session.calls[1][2]["json"]["service_key"], "request-key")

    def test_structured_error_prefers_korean_message(self):
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "detail": {
                            "code": "KMA_API_KEY_REQUIRED",
                            "message_ko": "기상청 인증키가 필요합니다.",
                        }
                    },
                    status_code=503,
                )
            ]
        )
        with self.assertRaisesRegex(APIError, "기상청 인증키") as raised:
            APIClient("http://backend", session=session).weather_status()
        self.assertEqual(raised.exception.status_code, 503)

    def test_bundle_download(self):
        session = FakeSession([FakeResponse(content_type="application/zip", content=b"zip")])
        client = APIClient("http://backend", session=session)
        self.assertEqual(client.download_bundle("p1"), b"zip")
        self.assertTrue(session.calls[0][1].endswith("/api/v1/predictions/p1/bundle"))


if __name__ == "__main__":
    unittest.main()
