"""
Unit tests for FGSM adversarial attacks and robustness assessment.
"""

import unittest
import numpy as np
import torch

from src.models import ConvNetClassifier
from src.adversarial import FGSMAttack


class TestAdversarial(unittest.TestCase):

    def setUp(self):
        self.model = ConvNetClassifier()
        self.model.eval()
        self.attacker = FGSMAttack(self.model)
        self.dummy_input = torch.rand(1, 1, 28, 28)

    def test_fgsm_untargeted(self):
        eps = 0.2
        results = self.attacker.attack(self.dummy_input, epsilon=eps)

        self.assertIn("orig_prediction", results)
        self.assertIn("adv_prediction", results)
        self.assertIn("is_fooled", results)
        self.assertIn("l_inf_distortion", results)

        # Ensure L-infinity norm is bounded by epsilon (+ tolerance for numerical precision)
        self.assertLessEqual(results["l_inf_distortion"], eps + 1e-4)

        # Check visual image shapes
        self.assertEqual(results["adv_image_np"].shape, (28, 28))
        self.assertEqual(results["perturbation_np"].shape, (28, 28))

    def test_fgsm_targeted(self):
        target = 4
        results = self.attacker.attack(self.dummy_input, epsilon=0.3, target_class=target)
        self.assertIn("adv_prediction", results)
        self.assertTrue(0 <= results["adv_prediction"] < 10)


if __name__ == "__main__":
    unittest.main()
