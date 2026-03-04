"""
test_webapp.py — End-to-End Test Suite for the Visa Estimator Web App
======================================================================
Tests three layers:
  1. predictor.py  (unit tests)
  2. Flask API     (integration tests — starts test client)
  3. Sample cases  (multiple real-world inputs)

Run:
    python test_webapp.py

Or via pytest:
    pytest test_webapp.py -v
"""

import sys
import json
import unittest
from datetime import date, datetime

# ── 1. Predictor Unit Tests ───────────────────────────────────────────────────

class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[setup] Loading VisaPredictor …")
        from predictor import VisaPredictor
        cls.predictor = VisaPredictor()

    # -- Consulate list -------------------------------------------------------
    def test_consulate_list_not_empty(self):
        consulates = self.predictor.list_consulates()
        self.assertGreater(len(consulates), 0, "Consulate list should not be empty")
        print(f"  OK: {len(consulates)} consulates loaded")

    def test_consulate_list_fields(self):
        consulates = self.predictor.list_consulates()
        for c in consulates[:5]:
            self.assertIn("code",         c)
            self.assertIn("label",        c)
            self.assertIn("mean_pt",      c)
            self.assertIn("approval_rate",c)
        print("  OK: Consulate list fields OK")

    def test_consulate_list_sorted(self):
        consulates = self.predictor.list_consulates()
        labels = [c["label"] for c in consulates]
        self.assertEqual(labels, sorted(labels), "Consulates should be sorted by label")
        print("  OK: Consulate list sorted correctly")

    # -- Basic prediction -----------------------------------------------------
    def test_predict_returns_positive_days(self):
        result = self.predictor.predict("KGL", "2024-06-15", 1000)
        self.assertGreater(result["predicted_days"], 0)
        print(f"  OK: predicted_days={result['predicted_days']}")

    def test_predict_confidence_interval(self):
        result = self.predictor.predict("KGL", "2024-06-15", 1000)
        self.assertLess(result["lower_bound"], result["predicted_days"])
        self.assertGreater(result["upper_bound"], result["predicted_days"])
        print(f"  OK: CI=[{result['lower_bound']}, {result['upper_bound']}]")

    def test_predict_category_values(self):
        result = self.predictor.predict("KGL", "2024-06-15", 1000)
        self.assertIn(result["category"], ("Fast", "Normal", "Slow"))
        print(f"  OK: category={result['category']}")

    def test_predict_date_object(self):
        result = self.predictor.predict("ADD", date(2024, 1, 15), 500)
        self.assertIsInstance(result["predicted_days"], float)
        print(f"  OK: date object input accepted: {result['predicted_days']}d")

    def test_predict_peak_season(self):
        """October–January should be peak season."""
        r_peak    = self.predictor.predict("KGL", "2024-11-01", 1000)
        r_offpeak = self.predictor.predict("KGL", "2024-06-01", 1000)
        self.assertTrue(r_peak["is_peak_season"])
        self.assertFalse(r_offpeak["is_peak_season"])
        print(f"  OK: Peak season detection correct")

    def test_predict_required_fields(self):
        result = self.predictor.predict("FRN", "2024-03-20", 2000)
        required = [
            "predicted_days", "lower_bound", "upper_bound", "predicted_months",
            "category", "consulate_code", "consulate_label", "consulate_stats",
            "submit_date", "model_name", "model_mae_days", "model_r2",
            "is_peak_season", "features_used",
        ]
        for field in required:
            self.assertIn(field, result, f"Missing field: {field}")
        print(f"  OK: All required fields present")

    def test_predict_unknown_consulate_fallback(self):
        """Unknown consulate should use global fallback stats, not crash."""
        try:
            result = self.predictor.predict("ZZZ", "2024-06-15", 1000)
            self.assertGreater(result["predicted_days"], 0)
            print(f"  OK: Unknown consulate fallback works: {result['predicted_days']}d")
        except Exception as e:
            self.fail(f"Unknown consulate should not raise exception: {e}")

    def test_predict_months_consistent(self):
        result = self.predictor.predict("SGP", "2024-09-05", 800)
        expected_months = round(result["predicted_days"] / 30.44, 1)
        self.assertAlmostEqual(result["predicted_months"], expected_months, places=1)
        print(f"  OK: predicted_months={result['predicted_months']} consistent with days")


# ── 2. Flask API Integration Tests ───────────────────────────────────────────

class TestFlaskAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[setup] Creating Flask test client …")
        from app import app
        app.testing = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model",             data)
        self.assertIn("consulates_loaded", data)
        print(f"  OK: /health → {data}")

    def test_consulates_endpoint(self):
        resp = self.client.get("/api/consulates")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("consulates", data)
        self.assertIn("count",      data)
        self.assertGreater(data["count"], 0)
        print(f"  OK: /api/consulates → {data['count']} consulates")

    def test_predict_endpoint_valid(self):
        payload = {
            "consulate":   "KGL",
            "submit_date": "2024-06-15",
            "case_number": 1234,
        }
        resp = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_days", data)
        self.assertGreater(data["predicted_days"], 0)
        print(f"  OK: /api/predict (KGL) → {data['predicted_days']}d ({data['category']})")

    def test_predict_endpoint_missing_consulate(self):
        payload = {"submit_date": "2024-06-15"}
        resp = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("errors",     data)
        self.assertIn("consulate",  data["errors"])
        print(f"  OK: /api/predict missing consulate → 400")

    def test_predict_endpoint_invalid_date(self):
        payload = {
            "consulate":   "KGL",
            "submit_date": "not-a-date",
        }
        resp = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("submit_date", data["errors"])
        print(f"  OK: /api/predict invalid date → 400")

    def test_model_info_endpoint(self):
        resp = self.client.get("/api/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("model_name",    data)
        self.assertIn("n_features",    data)
        self.assertIn("feature_names", data)
        self.assertIn("test_metrics",  data)
        print(f"  OK: /api/model-info → {data['model_name']} ({data['n_features']} features)")

    def test_sample_cases_endpoint(self):
        resp = self.client.get("/api/sample-cases")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("sample_cases", data)
        for case in data["sample_cases"]:
            self.assertIn("input",  case)
            self.assertIn("result", case)
        print(f"  OK: /api/sample-cases → {len(data['sample_cases'])} cases")


# ── 3. Multiple Sample Cases ──────────────────────────────────────────────────

class TestSampleCases(unittest.TestCase):

    SAMPLE_CASES = [
        # (consulate, submit_date, case_number, description)
        ("KGL",  "2024-06-15", 1234, "Kigali, mid-year, medium case"),
        ("ADD",  "2023-10-01",  500, "Addis Ababa, peak season start"),
        ("FRN",  "2024-01-20", 2000, "Frankfurt, winter peak, high case#"),
        ("SGP",  "2024-09-05",  800, "Singapore, off-peak"),
        ("LGS",  "2023-12-25",  300, "Lagos, Christmas day, peak"),
        ("MNL",  "2024-03-15", 1500, "Manila, spring"),
        ("CHN",  "2024-07-04", 3000, "Chennai, July 4th"),
        ("SAO",  "2024-02-14",  100, "São Paulo, Valentine's Day, early case"),
        ("MEX",  "2023-11-15",  750, "Mexico City, Nov, peak"),
        ("SHN",  "2024-08-01",  450, "Shanghai, summer"),
    ]

    @classmethod
    def setUpClass(cls):
        print("\n[setup] Loading predictor for sample case tests …")
        from predictor import get_predictor
        cls.predictor = get_predictor()

    def test_all_sample_cases(self):
        print("\n  Results for sample cases:")
        print(f"  {'Consulate':<8} {'Date':<12} {'Predicted':>12} {'Range':>22} {'Category':<10}")
        print("  " + "-"*70)
        for consulate, dt, cn, desc in self.SAMPLE_CASES:
            with self.subTest(consulate=consulate, date=dt):
                result = self.predictor.predict(consulate, dt, cn)
                self.assertGreater(result["predicted_days"], 0, f"Failed: {desc}")
                self.assertIn(result["category"], ("Fast", "Normal", "Slow"))
                print(
                    f"  {consulate:<8} {dt:<12} "
                    f"{result['predicted_days']:>8.0f}d  "
                    f"[{result['lower_bound']:.0f}–{result['upper_bound']:.0f}]  "
                    f"{result['category']:<10}  {desc}"
                )


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Visa Estimator Web App — End-to-End Test Suite")
    print("=" * 70)
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestFlaskAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestSampleCases))
    runner  = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


