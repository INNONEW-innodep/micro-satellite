import datetime as dt
import unittest

from ui_next.state import (
    FINGERPRINTS_KEY,
    apply_configuration,
    extract_prediction_steps,
    fingerprint,
    normalize_weather_rows,
    openapi_operation_rows,
    validate_input_rows,
)


class StateTests(unittest.TestCase):
    def test_fingerprint_ignores_mapping_order(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))

    def test_first_config_sets_baseline_without_invalidation(self):
        state, removed, changed = apply_configuration({"prediction_result": {"id": "p1"}}, "model", {"id": "a"})
        self.assertFalse(changed)
        self.assertEqual(removed, ())
        self.assertIn("prediction_result", state)
        self.assertIn("model", state[FINGERPRINTS_KEY])

    def test_input_change_cascades_but_keeps_models(self):
        initial, _, _ = apply_configuration({}, "input", {"files": ["a", "b"]})
        initial.update(
            {
                "input_bundle": [1, 2],
                "weather_data": [3],
                "prediction_result": {"id": "p1"},
                "prediction_context": {"sample_id": "demo"},
                "models": [{"id": "m1"}],
            }
        )
        changed_state, removed, changed = apply_configuration(initial, "input", {"files": ["a", "c"]})
        self.assertTrue(changed)
        self.assertIn("input_bundle", removed)
        self.assertIn("prediction_result", removed)
        self.assertIn("prediction_context", removed)
        self.assertNotIn("prediction_result", changed_state)
        self.assertNotIn("prediction_context", changed_state)
        self.assertIn("models", changed_state)

    def test_input_validation_rejects_duplicate_dates(self):
        errors = validate_input_rows(
            [
                {"name": "a.tif", "date": "2026-01-01", "water_level_m": 1.0},
                {"name": "b.npy", "date": "2026-01-01", "water_level_m": None},
            ]
        )
        self.assertTrue(any("서로 다른" in error for error in errors))

    def test_future_weather_must_be_scenario(self):
        rows, errors = normalize_weather_rows(
            [{"timestamp": "2026-08-10", "precipitation_mm": 2, "kind": "observed"}],
            today=dt.date(2026, 8, 9),
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(any("scenario" in error for error in errors))

    def test_weather_aliases_are_normalized(self):
        rows, errors = normalize_weather_rows(
            [
                {
                    "date": "2026-08-01",
                    "total_precipitation_mm": "3.5",
                    "avg_temperature_c": 22,
                    "min_temperature_c": 18,
                    "max_temperature_c": 27,
                    "avg_humidity_percent": 75,
                    "avg_wind_speed_m_s": 3.2,
                    "surface_pressure_hpa": 1009.1,
                    "kind": "observed",
                }
            ],
            today=dt.date(2026, 8, 9),
        )
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["precipitation_mm"], 3.5)
        self.assertEqual(rows[0]["temperature_c"], 22.0)
        self.assertEqual(rows[0]["wind_speed_mps"], 3.2)
        self.assertEqual(rows[0]["min_temperature_c"], 18.0)
        self.assertEqual(rows[0]["max_temperature_c"], 27.0)
        self.assertEqual(rows[0]["humidity_pct"], 75.0)
        self.assertEqual(rows[0]["surface_pressure_hpa"], 1009.1)

    def test_prediction_step_shapes(self):
        self.assertEqual(extract_prediction_steps({"output": {"frames": [{"step": 1}]}}), [{"step": 1}])

    def test_openapi_rows(self):
        rows = openapi_operation_rows(
            {"paths": {"/x": {"get": {"summary": "Read X"}, "parameters": []}}}
        )
        self.assertEqual(rows, [{"method": "GET", "path": "/x", "summary": "Read X"}])


if __name__ == "__main__":
    unittest.main()
