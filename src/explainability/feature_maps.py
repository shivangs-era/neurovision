"""
Feature Map Extractor and Intermediate Layer Visualizer.
Allows inspecting the internal feature representations of deep layers.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


class FeatureMapExtractor:
    """
    Extracts and organizes channel activations from convolutional layers.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def extract(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Executes forward pass and returns activations from each key stage.

        Returns:
            Dict mapping stage name to numpy array [Channels, H, W].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            if hasattr(self.model, "forward_with_features"):
                _, features = self.model.forward_with_features(input_tensor)
                result = {}
                for name, tensor in features.items():
                    result[name] = tensor.squeeze(0).cpu().numpy()
                return result
            else:
                # Fallback: manually hook conv layers
                activations = {}
                handles = []

                def make_hook(layer_name):
                    def hook(module, inp, out):
                        activations[layer_name] = out.squeeze(0).detach().cpu().numpy()
                    return hook

                conv_idx = 0
                for name, layer in self.model.named_modules():
                    if isinstance(layer, nn.Conv2d):
                        conv_idx += 1
                        h = layer.register_forward_hook(make_hook(f"Conv2d_{conv_idx} ({name})"))
                        handles.append(h)

                _ = self.model(input_tensor)

                for h in handles:
                    h.remove()

                return activations

    @staticmethod
    def create_grid(
        feature_maps: np.ndarray, max_channels: int = 16, cols: int = 4
    ) -> np.ndarray:
        """
        Arranges top feature map channels into a unified 2D grid matrix.
        """
        channels = min(feature_maps.shape[0], max_channels)
        h, w = feature_maps.shape[1], feature_maps.shape[2]
        rows = (channels + cols - 1) // cols

        grid = np.zeros((rows * (h + 2), cols * (w + 2)), dtype=np.float32)

        for idx in range(channels):
            r = idx // cols
            c = idx % cols
            channel_slice = feature_maps[idx]
            # Normalize channel to [0, 1]
            c_min, c_max = channel_slice.min(), channel_slice.max()
            if c_max - c_min > 1e-8:
                channel_norm = (channel_slice - c_min) / (c_max - c_min)
            else:
                channel_norm = np.zeros_like(channel_slice)

            y_start = r * (h + 2) + 1
            x_start = c * (w + 2) + 1
            grid[y_start : y_start + h, x_start : x_start + w] = channel_norm

        return grid
