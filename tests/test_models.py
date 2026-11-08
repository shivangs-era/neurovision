"""
Unit tests for model architectures: ConvNetClassifier, NeuroResNet, and ConvVAE.
"""

import unittest
import torch

from src.models import ConvNetClassifier, NeuroResNet, ConvVAE, ResidualBlock


class TestModels(unittest.TestCase):

    def test_cnn_forward(self):
        model = ConvNetClassifier(in_channels=1, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 10))

    def test_cnn_features(self):
        model = ConvNetClassifier(in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        logits, features = model.forward_with_features(x)
        self.assertEqual(logits.shape, (2, 10))
        self.assertIn("Block 1 (Edges & Contours)", features)
        self.assertIn("Block 3 (High-Level Semantics)", features)
        self.assertEqual(features["Block 1 (Edges & Contours)"].shape[0], 2)

    def test_resnet_forward(self):
        model = NeuroResNet(in_channels=1, num_classes=10)
        x = torch.randn(3, 1, 28, 28)
        logits = model(x)
        self.assertEqual(logits.shape, (3, 10))

    def test_resnet_features(self):
        model = NeuroResNet(in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        logits, features = model.forward_with_features(x)
        self.assertEqual(logits.shape, (2, 10))
        self.assertIn("Residual Stage 3", features)

    def test_vae_pipeline(self):
        vae = ConvVAE(in_channels=1, latent_dim=2)
        x = torch.rand(4, 1, 28, 28)
        recon, mu, logvar = vae(x)
        self.assertEqual(recon.shape, (4, 1, 28, 28))
        self.assertEqual(mu.shape, (4, 2))
        self.assertEqual(logvar.shape, (4, 2))

        # Test loss function
        loss_dict = vae.loss_function(recon, x, mu, logvar)
        self.assertIn("loss", loss_dict)
        self.assertIn("recon_loss", loss_dict)
        self.assertIn("kld", loss_dict)
        self.assertFalse(torch.isnan(loss_dict["loss"]))

    def test_vae_decode(self):
        vae = ConvVAE(in_channels=1, latent_dim=2)
        z = torch.randn(5, 2)
        samples = vae.decode(z)
        self.assertEqual(samples.shape, (5, 1, 28, 28))
        self.assertTrue((samples >= 0.0).all() and (samples <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
