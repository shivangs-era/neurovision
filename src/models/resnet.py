"""
Deep Residual Convolutional Network (NeuroResNet) with Skip Connections.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard 2-conv residual block with identity/projection shortcut.
    F(x) + x mechanism enables stable gradient flow across deep layers.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class NeuroResNet(nn.Module):
    """
    Residual Network tailored for 28x28 grayscale vision tasks.
    Combines initial receptive field expansion, multi-stage residual processing,
    and global average pooling.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial stem convolution
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32 channels, 28x28
        self.layer1 = ResidualBlock(32, 32, stride=1)
        # Stage 2: 64 channels, downsampled to 14x14
        self.layer2 = ResidualBlock(32, 64, stride=2)
        # Stage 3: 128 channels, downsampled to 7x7
        self.layer3 = ResidualBlock(64, 128, stride=2)

        # Global average pool + classification head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = {}
        prep_out = self.prep(x)
        features["Stem Convolution"] = prep_out
        l1 = self.layer1(prep_out)
        features["Residual Stage 1"] = l1
        l2 = self.layer2(l1)
        features["Residual Stage 2"] = l2
        l3 = self.layer3(l2)
        features["Residual Stage 3"] = l3
        gap_out = self.gap(l3)
        flat = torch.flatten(gap_out, 1)
        logits = self.fc(self.dropout(flat))
        return logits, features

    def get_target_conv_layer(self) -> nn.Module:
        """Returns the final convolutional layer of Stage 3 for Grad-CAM."""
        return self.layer3.conv2

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
