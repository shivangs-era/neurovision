"""
Convolutional Neural Network Architecture for Visual Classification.
Includes BatchNorm, Dropout, AdaptiveAvgPooling, and hook attachment points.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetClassifier(nn.Module):
    """
    Modern deep convolutional neural network for image recognition.
    Architected for high representational capacity with batch normalization,
    spatial regularization, and transparent layer access for explainability.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: Low-level features (edges, contours) -> 28x28 to 14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        # Block 2: Mid-level features (textures, parts) -> 14x14 to 7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
        )

        # Block 3: High-level semantic features -> 7x7
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global pooling and dense classifier
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning class logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass capturing intermediate feature map representations."""
        features = {}
        out1 = self.block1(x)
        features["Block 1 (Edges & Contours)"] = out1
        out2 = self.block2(out1)
        features["Block 2 (Patterns & Textures)"] = out2
        out3 = self.block3(out2)
        features["Block 3 (High-Level Semantics)"] = out3
        pooled = self.pool(out3)
        features["Adaptive Pooling"] = pooled
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits, features

    def get_target_conv_layer(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM activation mapping."""
        return self.block3[0]

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
