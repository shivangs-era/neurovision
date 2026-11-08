"""
Gradient-weighted Class Activation Mapping (Grad-CAM).
Visualizes the decision regions of convolutional layers for interpretability.
Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
"""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    """
    Computes class activation heatmaps by projecting class-specific gradients
    onto the final convolutional feature maps.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_target_conv_layer"):
                self.target_layer = model.get_target_conv_layer()
            else:
                # Fallback: scan for the last Conv2d layer in children
                convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if not convs:
                    raise ValueError("Model does not contain any Conv2d layers for Grad-CAM.")
                self.target_layer = convs[-1]
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks to capture layer telemetry."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        counterfactual: bool = False,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates Grad-CAM heatmap for a single input image tensor [1, C, H, W].

        Args:
            input_tensor: 4D tensor representing the input image.
            target_class: Specific class to explain. If None, uses top predicted class.
            counterfactual: If True, inverts gradients to show what negates the class.

        Returns:
            heatmap_resized: 2D numpy array [H, W] normalized in [0, 1].
            predicted_class: Integer class index.
            confidence: Probability of the predicted class.
        """
        self.model.zero_grad()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        confidence = float(probabilities[0, target_class].item())

        # Backward pass targeting specific class logit
        score = logits[0, target_class]
        if counterfactual:
            score = -score
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to collect activations or gradients.")

        # Compute importance weights: Global Average Pooling of gradients along spatial dims
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, Channels, 1, 1]

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H_act, W_act]

        # Apply ReLU to retain features that positively correlate with the target score
        cam = F.relu(cam)

        # Upsample heatmap to match original input image spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize to [0, 1]
        cam_np = cam_upsampled.squeeze().cpu().numpy()
        cam_max = np.max(cam_np)
        cam_min = np.min(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam_np)

        return cam_normalized, target_class, confidence

    def overlay_on_image(
        self,
        image_np: np.ndarray,
        heatmap: np.ndarray,
        colormap: str = "jet",
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlays normalized heatmap on grayscale/RGB input image.

        Args:
            image_np: 2D array [H, W] in [0, 1] or 3D [H, W, 3].
            heatmap: 2D array [H, W] in [0, 1].
            colormap: Matplotlib colormap name ('jet', 'inferno', 'viridis', etc.).
            alpha: Transparency factor for heatmap overlay.

        Returns:
            RGB image numpy array [H, W, 3] in [0, 1].
        """
        # Ensure image is 3-channel RGB in [0, 1]
        if image_np.ndim == 2:
            base_rgb = np.stack([image_np] * 3, axis=-1)
        elif image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            # Channel-first format
            base_rgb = np.transpose(image_np, (1, 2, 0))
            if base_rgb.shape[2] == 1:
                base_rgb = np.repeat(base_rgb, 3, axis=2)
        else:
            base_rgb = image_np.copy()

        # Normalize base image if needed
        if base_rgb.max() > 1.0:
            base_rgb = base_rgb / 255.0

        # Apply colormap to heatmap
        cmap_func = plt.get_cmap(colormap)
        color_heatmap = cmap_func(heatmap)[:, :, :3]  # Strip alpha channel

        # Blend base image and heatmap
        blended = (1.0 - alpha) * base_rgb + alpha * color_heatmap
        blended = np.clip(blended, 0.0, 1.0)
        return blended

    def remove_hooks(self):
        """Cleanly removes registered PyTorch hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self):
        self.remove_hooks()
