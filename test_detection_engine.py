import unittest
import numpy as np
from data_generator import FEATURES
from train_models import train_models
from detection_engine import analyze_traffic, validate_traffic_record, combine_predictions


class DetectionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.models = Path(__file__).resolve().parents[1] / "models"
        train_models(model_dir=cls.models)

    def test_normal_record_shape(self):
        record = dict(zip(FEATURES, [80, 6000, 40, 0, 5, 140]))
        result = analyze_traffic(record)
        self.assertIn(result["random_forest_prediction"], {"NORMAL", "SUSPICIOUS"})
        self.assertIn(result["isolation_forest_prediction"], {"NORMAL", "ANOMALY"})
        self.assertIsInstance(result["alert_triggered"], bool)
        self.assertGreaterEqual(result["random_forest_confidence"], 0)

    def test_suspicious_record_shape_and_alert_logic(self):
        record = dict(zip(FEATURES, [900, 900000, 3, 12, 250, 250000]))
        result = analyze_traffic(record)
        flags = result["random_forest_prediction"] == "SUSPICIOUS" or result["isolation_forest_prediction"] == "ANOMALY"
        self.assertEqual(result["alert_triggered"], flags)
        if result["random_forest_prediction"] == "SUSPICIOUS" and result["isolation_forest_prediction"] == "ANOMALY":
            self.assertEqual(result["combined_status"], "HIGH_PRIORITY_REVIEW")

    def test_validation(self):
        good = dict.fromkeys(FEATURES, 1)
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_traffic_record({})
        for bad in ["nope", float("nan"), float("inf"), -1]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate_traffic_record({**good, "packets": bad})

    def test_all_decision_combinations(self):
        self.assertEqual(combine_predictions(False, False)[0], "NORMAL")
        self.assertEqual(combine_predictions(True, False)[0], "REVIEW")
        self.assertEqual(combine_predictions(False, True)[0], "REVIEW")
        self.assertEqual(combine_predictions(True, True)[0], "HIGH_PRIORITY_REVIEW")


if __name__ == "__main__":
    unittest.main()
