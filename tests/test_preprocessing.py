"""
Unit tests for image preprocessing and validation utilities.
"""

from io import BytesIO
import unittest
import numpy as np
from PIL import Image

from src.preprocessing import (
    denormalize_box_coordinates,
    draw_bounding_boxes,
    letterbox_image,
    validate_image_file,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a valid in-memory test image (100x100 RGB)
        self.valid_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = BytesIO()
        self.valid_img.save(buf, format="JPEG")
        self.valid_jpeg_bytes = buf.getvalue()

    def test_validate_image_file_valid_jpeg(self):
        is_valid, msg, img = validate_image_file(self.valid_jpeg_bytes)
        self.assertTrue(is_valid)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (100, 100))

    def test_validate_image_file_invalid_bytes(self):
        corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_STREAM"
        is_valid, msg, img = validate_image_file(corrupt_bytes)
        self.assertFalse(is_valid)
        self.assertIn("Failed to decode", msg)

    def test_validate_image_file_too_small(self):
        tiny_img = Image.new("RGB", (16, 16), color=(255, 0, 0))
        buf = BytesIO()
        tiny_img.save(buf, format="PNG")
        is_valid, msg, _ = validate_image_file(buf.getvalue())
        self.assertFalse(is_valid)
        self.assertIn("too small", msg)

    def test_letterbox_image(self):
        np_img = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, ratio, (dw, dh) = letterbox_image(np_img, target_shape=(640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertGreater(ratio, 0)

    def test_draw_bounding_boxes(self):
        np_img = np.zeros((200, 200, 3), dtype=np.uint8)
        boxes = [[20, 20, 80, 80]]
        confidences = [0.88]
        classes = ["pothole"]

        annotated = draw_bounding_boxes(np_img, boxes, confidences, classes)
        self.assertEqual(annotated.shape, np_img.shape)
        # Check that canvas has been drawn on (sum > 0)
        self.assertGreater(annotated.sum(), 0)


if __name__ == "__main__":
    unittest.main()
