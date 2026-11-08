"""
Unit tests for Explainable AI (XAI) modules: Grad-CAM, Saliency, and Feature Maps.
"""

import unittest
import numpy as np
import torch

from src.models import ConvNetClassifier
from src.explainability import GradCAM, SaliencyMap, FeatureMapExtractor


class TestExplainability(unittest.TestCase):

    def setUp(self):
        self.model = ConvNetClassifier()
        self.model.eval()
        self.dummy_input = torch.rand(1, 1, 28, 28)

    def test_gradcam_generation(self):
        cam = GradCAM(self.model)
        heatmap, pred_class, confidence = cam.generate(self.dummy_input)

        self.assertEqual(heatmap.shape, (28, 28))
        self.assertTrue(0 <= pred_class < 10)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertTrue(np.min(heatmap) >= 0.0)
        self.assertTrue(np.max(heatmap) <= 1.0)

        # Test overlay
        raw_img = self.dummy_input.squeeze().numpy()
        overlay = cam.overlay_on_image(raw_img, heatmap)
        self.assertEqual(overlay.shape, (28, 28, 3))
        cam.remove_hooks()

    def test_saliency_generation(self):
        sal = SaliencyMap(self.model)
        saliency, pred_class, conf = sal.generate(self.dummy_input, smoothgrad=False)
        self.assertEqual(saliency.shape, (28, 28))
        self.assertTrue(np.min(saliency) >= 0.0)
        self.assertTrue(np.max(saliency) <= 1.0)

        # Test SmoothGrad
        smooth, _, _ = sal.generate(self.dummy_input, smoothgrad=True, num_samples=5)
        self.assertEqual(smooth.shape, (28, 28))

    def test_feature_map_extractor(self):
        extractor = FeatureMapExtractor(self.model)
        features = extractor.extract(self.dummy_input)
        self.assertTrue(len(features) > 0)
        first_layer = list(features.values())[0]
        self.assertEqual(len(first_layer.shape), 3)  # [C, H, W]

        # Test grid generation
        grid = extractor.create_grid(first_layer, max_channels=8, cols=4)
        self.assertEqual(len(grid.shape), 2)


if __name__ == "__main__":
    unittest.main()
