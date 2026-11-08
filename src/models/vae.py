"""
Convolutional Variational Autoencoder (ConvVAE) for Latent Space Synthesis.
Encodes visual inputs into a continuous Gaussian latent distribution and
reconstructs novel or manipulated samples.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional VAE with 2D or arbitrary-D latent space.
    Designed for latent space manifold exploration and synthetic reconstruction.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder: 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_flatten_dim = 128 * 7 * 7
        self.fc_mu = nn.Linear(self.fc_flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_flatten_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.fc_flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 28x28
            nn.Sigmoid(),  # Pixel values normalized to [0, 1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes an image to mean and log-variance parameters."""
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector z to reconstructed image."""
        h = self.decoder_input(z)
        h = h.view(-1, 128, 7, 7)
        recon = self.decoder(h)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns reconstruction, mu, logvar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Computes VAE loss = Reconstruction Loss + Beta * KL Divergence.
        Uses BCE for normalized pixel reconstruction.
        """
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        # KL divergence = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kld": kld,
        }
