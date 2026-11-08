"""
Curated sample presets for instant inference, explainability, and adversarial attacks.
"""

from typing import Dict, List, Tuple
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_preset_samples(
    data_dir: str = "./data_cache", samples_per_class: int = 2
) -> Dict[int, List[np.ndarray]]:
    """
    Retrieves representative test samples for classes 0-9.
    """
    presets: Dict[int, List[np.ndarray]] = {i: [] for i in range(10)}

    if os.path.exists(data_dir):
        try:
            dataset = torchvision.datasets.MNIST(
                root=data_dir, train=False, download=False, transform=transforms.ToTensor()
            )
            for img_tensor, label in dataset:
                if len(presets[label]) < samples_per_class:
                    presets[label].append(img_tensor.squeeze(0).numpy())
                if all(len(v) >= samples_per_class for v in presets.values()):
                    break
            return presets
        except Exception:
            pass

    # Fallback synthetic presets
    from .dataset import SyntheticDigitDataset
    syn = SyntheticDigitDataset(size=50)
    for img_tensor, label in syn:
        if len(presets[label]) < samples_per_class:
            presets[label].append(img_tensor.squeeze(0).numpy())
        if all(len(v) >= samples_per_class for v in presets.values()):
            break

    return presets
