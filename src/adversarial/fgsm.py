"""
Fast Gradient Sign Method (FGSM) Adversarial Attack and Robustness Assessment.
Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSMAttack:
    """
    Generates single-step gradient-directed adversarial perturbations.
    Supports both untargeted and targeted adversarial attacks.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def attack(
        self,
        input_tensor: torch.Tensor,
        epsilon: float = 0.15,
        target_class: Optional[int] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Executes FGSM attack on a given input image tensor.

        Args:
            input_tensor: Image tensor [1, C, H, W] or [C, H, W] in [0, 1].
            epsilon: Perturbation magnitude (0.0 to 0.5+).
            target_class: If provided, runs targeted attack to force target_class.
                          If None, runs untargeted attack to degrade predicted class.
            clip_min: Minimum valid pixel value.
            clip_max: Maximum valid pixel value.

        Returns:
            Dictionary containing original and adversarial tensors, numpy arrays,
            probabilities, predicted labels, confidence shift, and distortion metrics.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Baseline evaluation
        with torch.no_grad():
            orig_logits = self.model(input_tensor)
            orig_probs = F.softmax(orig_logits, dim=1)
            orig_pred = int(torch.argmax(orig_logits, dim=1).item())
            orig_conf = float(orig_probs[0, orig_pred].item())

        # Enable gradient tracking on input
        x_adv = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(x_adv)

        if target_class is None:
            # Untargeted: Maximize loss of the true/predicted class
            loss = F.cross_entropy(logits, torch.tensor([orig_pred], device=input_tensor.device))
            loss.backward()
            gradient_sign = x_adv.grad.data.sign()
            # Step in direction of increasing loss
            perturbed = x_adv + epsilon * gradient_sign
        else:
            # Targeted: Minimize loss of the target class
            loss = F.cross_entropy(logits, torch.tensor([target_class], device=input_tensor.device))
            loss.backward()
            gradient_sign = x_adv.grad.data.sign()
            # Step in direction of decreasing target loss
            perturbed = x_adv - epsilon * gradient_sign

        # Enforce valid visual dynamic range
        perturbed = torch.clamp(perturbed, clip_min, clip_max).detach()

        # Adversarial evaluation
        with torch.no_grad():
            adv_logits = self.model(perturbed)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_pred = int(torch.argmax(adv_logits, dim=1).item())
            adv_conf = float(adv_probs[0, adv_pred].item())
            orig_class_adv_conf = float(adv_probs[0, orig_pred].item())

        # Compute perturbation noise map and metrics
        diff = perturbed - input_tensor
        l_inf = float(torch.max(torch.abs(diff)).item())
        l_2 = float(torch.norm(diff).item())

        # Prepare numpy visuals
        orig_img_np = input_tensor.squeeze().cpu().numpy()
        adv_img_np = perturbed.squeeze().cpu().numpy()
        perturbation_np = diff.squeeze().cpu().numpy()

        # Rescale perturbation map for clear visualization
        p_max = np.max(np.abs(perturbation_np))
        if p_max > 1e-8:
            norm_perturbation = (perturbation_np / (2 * p_max)) + 0.5
        else:
            norm_perturbation = np.full_like(perturbation_np, 0.5)

        is_fooled = (adv_pred != orig_pred)

        return {
            "epsilon": epsilon,
            "orig_prediction": orig_pred,
            "orig_confidence": orig_conf,
            "adv_prediction": adv_pred,
            "adv_confidence": adv_conf,
            "orig_class_new_confidence": orig_class_adv_conf,
            "confidence_drop": orig_conf - orig_class_adv_conf,
            "is_fooled": is_fooled,
            "l_inf_distortion": l_inf,
            "l_2_distortion": l_2,
            "orig_probs": orig_probs.squeeze().cpu().numpy(),
            "adv_probs": adv_probs.squeeze().cpu().numpy(),
            "orig_image_np": orig_img_np,
            "adv_image_np": adv_img_np,
            "perturbation_np": norm_perturbation,
            "raw_perturbation_np": perturbation_np,
            "adv_tensor": perturbed,
        }
