"""
Input Saliency Maps and SmoothGrad Explainability.
Visualizes pixel-level input attribution via first-order loss gradients.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyMap:
    """
    Computes input pixel sensitivity gradients and SmoothGrad attribution.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        smoothgrad: bool = False,
        num_samples: int = 25,
        noise_level: float = 0.15,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes pixel sensitivity attribution.

        Args:
            input_tensor: [1, C, H, W] or [C, H, W].
            target_class: Target class index. Defaults to argmax predicted class.
            smoothgrad: If True, averages gradients over noisy samples.
            num_samples: Number of noisy samples for SmoothGrad.
            noise_level: Standard deviation of Gaussian noise relative to input range.

        Returns:
            saliency: 2D numpy array [H, W] normalized to [0, 1].
            predicted_class: Integer class index.
            confidence: Probability of class.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Baseline prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = int(torch.argmax(logits, dim=1).item())
            confidence = float(probs[0, pred_class].item())

        target = target_class if target_class is not None else pred_class

        if not smoothgrad:
            img = input_tensor.clone().detach().requires_grad_(True)
            out = self.model(img)
            score = out[0, target]
            score.backward()
            grad = img.grad.data.abs()  # [1, C, H, W]
            saliency = torch.max(grad, dim=1)[0].squeeze().cpu().numpy()
        else:
            # SmoothGrad: Average gradients of perturbed inputs
            grad_sum = torch.zeros_like(input_tensor)
            spread = (input_tensor.max() - input_tensor.min()).item() * noise_level

            for _ in range(num_samples):
                noise = torch.randn_like(input_tensor) * spread
                noisy_img = (input_tensor + noise).clone().detach().requires_grad_(True)
                out = self.model(noisy_img)
                score = out[0, target]
                score.backward()
                grad_sum += noisy_img.grad.data.abs()

            grad_avg = grad_sum / num_samples
            saliency = torch.max(grad_avg, dim=1)[0].squeeze().cpu().numpy()

        # Normalize to [0, 1]
        s_min, s_max = np.min(saliency), np.max(saliency)
        if s_max - s_min > 1e-8:
            saliency_norm = (saliency - s_min) / (s_max - s_min)
        else:
            saliency_norm = np.zeros_like(saliency)

        return saliency_norm, target, confidence
