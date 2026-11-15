"""
Unit tests for the PotholeDetector inference class.
"""

import unittest
from src.detector import PotholeDetector, ULTRALYTICS_AVAILABLE


class TestDetector(unittest.TestCase):

    def test_detector_missing_weights(self):
        # Initializing with non-existent path should not crash
        detector = PotholeDetector(model_path="models/non_existent_weights.pt")
        self.assertFalse(detector.is_ready)

        if not ULTRALYTICS_AVAILABLE:
            self.assertIn("Ultralytics package is not installed", detector.status_message)
        else:
            self.assertIn("Model weights not found", detector.status_message)

        # Predicting without weights or dependencies should return graceful failure
        result = detector.predict("any_image.jpg")
        self.assertFalse(result.is_success)
        self.assertEqual(result.num_detections, 0)
        self.assertIsNotNone(result.error_message)


if __name__ == "__main__":
    unittest.main()
