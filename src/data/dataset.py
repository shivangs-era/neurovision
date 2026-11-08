"""
Dataset loading, augmentation, and data loader generation.
Supports offline local MNIST loading with automatic caching and synthetic fallback.
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np


class SyntheticDigitDataset(Dataset):
    """
    Fallback dataset generating synthetic mathematical digits / glyphs.
    Ensures zero failure modes even in air-gapped systems.
    """

    def __init__(self, size: int = 1000, num_classes: int = 10):
        self.size = size
        self.num_classes = num_classes
        self.data = []
        self.labels = []
        self._generate()

    def _generate(self):
        rng = np.random.RandomState(42)
        for i in range(self.size):
            label = i % self.num_classes
            # Create a 28x28 matrix with characteristic shapes per class
            img = np.zeros((28, 28), dtype=np.float32)
            # Center region
            if label == 0:  # Circle / Oval
                rr, cc = np.ogrid[:28, :28]
                mask = ((rr - 14) ** 2 / 8**2 + (cc - 14) ** 2 / 6**2 <= 1) & (
                    (rr - 14) ** 2 / 5**2 + (cc - 14) ** 2 / 3**2 >= 1
                )
                img[mask] = 1.0
            elif label == 1:  # Vertical line
                img[5:23, 13:16] = 1.0
            elif label == 2:  # Top curve and base
                img[6:9, 9:20] = 1.0
                img[9:17, 17:20] = 1.0
                img[17:20, 9:20] = 1.0
                img[20:23, 8:21] = 1.0
            elif label == 3:  # Two arcs
                img[6:9, 9:19] = 1.0
                img[13:16, 12:19] = 1.0
                img[20:23, 9:19] = 1.0
                img[6:23, 17:20] = 1.0
            elif label == 4:  # Angle and vertical stem
                img[6:16, 9:12] = 1.0
                img[14:17, 9:21] = 1.0
                img[6:23, 17:20] = 1.0
            elif label == 5:  # Top bar, left bar, curve
                img[6:9, 9:20] = 1.0
                img[9:15, 9:12] = 1.0
                img[13:16, 9:20] = 1.0
                img[15:21, 17:20] = 1.0
                img[20:23, 9:20] = 1.0
            elif label == 6:  # Loop at bottom
                img[6:22, 9:12] = 1.0
                img[14:17, 9:20] = 1.0
                img[20:23, 9:20] = 1.0
                img[14:23, 17:20] = 1.0
            elif label == 7:  # Top bar and diagonal
                img[6:9, 8:21] = 1.0
                for r in range(9, 23):
                    c = int(20 - (r - 9) * 0.7)
                    img[r, max(0, c - 1) : min(28, c + 2)] = 1.0
            elif label == 8:  # Double loop
                rr, cc = np.ogrid[:28, :28]
                top = ((rr - 10) ** 2 / 5**2 + (cc - 14) ** 2 / 5**2 <= 1) & (
                    (rr - 10) ** 2 / 2**2 + (cc - 14) ** 2 / 2**2 >= 1
                )
                bot = ((rr - 19) ** 2 / 6**2 + (cc - 14) ** 2 / 6**2 <= 1) & (
                    (rr - 19) ** 2 / 2**2 + (cc - 14) ** 2 / 2**2 >= 1
                )
                img[top | bot] = 1.0
            else:  # 9: Top loop and stem
                img[6:15, 9:12] = 1.0
                img[6:9, 9:20] = 1.0
                img[13:16, 9:20] = 1.0
                img[6:23, 17:20] = 1.0

            # Add gentle random noise and translation
            noise = rng.normal(0, 0.05, (28, 28)).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

            self.data.append(torch.tensor(img, dtype=torch.float32).unsqueeze(0))
            self.labels.append(label)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


def get_data_loaders(
    data_dir: str = "./data_cache",
    batch_size: int = 64,
    val_split: float = 0.15,
    num_workers: int = 0,
    force_synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test data loaders.
    Attempts to load local MNIST dataset; gracefully falls back to synthetic if unavailable.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [0, 1] range; keep tensors directly viewable & bounded
    ])

    use_mnist = False
    if not force_synthetic and os.path.exists(data_dir):
        try:
            full_train_dataset = torchvision.datasets.MNIST(
                root=data_dir, train=True, download=False, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root=data_dir, train=False, download=False, transform=transform
            )
            use_mnist = True
        except Exception:
            use_mnist = False

    if not use_mnist:
        train_raw = SyntheticDigitDataset(size=2000)
        test_dataset = SyntheticDigitDataset(size=400)
        val_size = int(len(train_raw) * val_split)
        train_size = len(train_raw) - val_size
        train_dataset, val_dataset = random_split(
            train_raw, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
