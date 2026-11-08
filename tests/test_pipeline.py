"""
Unit tests for data transforms, preprocessing, and classification metrics.
"""

import unittest
import numpy as np
import torch
from PIL import Image

from src.utils import preprocess_drawn_image
from src.training import compute_classification_metrics
from src.data import SyntheticDigitDataset


class TestPipeline(unittest.TestCase):

    def test_preprocess_drawn_image(self):
        # Create a blank white canvas with a black stroke
        arr = np.full((100, 100), 255, dtype=np.uint8)
        arr[30:70, 48:52] = 0  # Black vertical stroke
        pil_img = Image.fromarray(arr)

        tensor, norm_img = preprocess_drawn_image(pil_img, invert=True, center_mass=True)

        self.assertEqual(tensor.shape, (1, 1, 28, 28))
        self.assertEqual(norm_img.shape, (28, 28))
        self.assertTrue(0.0 <= np.max(norm_img) <= 1.0)
        # Stroke must be active in center
        self.assertGreater(norm_img[14, 14], 0.0)

    def test_compute_classification_metrics(self):
        y_true = np.array([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9])
        y_pred = np.array([0, 1, 2, 1, 3, 4, 5, 6, 7, 8, 9])  # One mistake at index 3

        metrics = compute_classification_metrics(y_true, y_pred, num_classes=10)

        self.assertAlmostEqual(metrics["accuracy"], 10 / 11, places=3)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["confusion_matrix"].shape, (10, 10))
        self.assertIn("macro_f1", metrics)
        self.assertIn("per_class", metrics)

    def test_synthetic_dataset(self):
        ds = SyntheticDigitDataset(size=20, num_classes=10)
        self.assertEqual(len(ds), 20)
        tensor, label = ds[0]
        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertTrue(0 <= label < 10)


if __name__ == "__main__":
    unittest.main()
