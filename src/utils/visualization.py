"""
Visualization suite for model predictions, explainability maps, adversarial effects, and training metrics.
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless & web environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_prediction_bars(
    probabilities: np.ndarray, top_k: int = 5, class_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plots horizontal bar chart of top-K prediction probabilities.
    """
    if class_names is None:
        class_names = [f"Digit {i}" for i in range(len(probabilities))]

    sorted_indices = np.argsort(probabilities)[::-1][:top_k]
    top_probs = probabilities[sorted_indices]
    top_labels = [class_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(6, 2.8), dpi=120)
    y_pos = np.arange(len(top_labels))

    colors = ["#2563EB" if i == 0 else "#94A3B8" for i in range(len(top_labels))]
    bars = ax.barh(y_pos, top_probs * 100, color=colors, height=0.6, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=10, fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Prediction Confidence Distribution", fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, prob in zip(bars, top_probs):
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1E293B",
        )

    plt.tight_layout()
    return fig


def plot_gradcam_breakdown(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    pred_label: str,
    confidence: float,
) -> plt.Figure:
    """
    Plots a 3-panel explanation figure: Input Image, Heatmap, and Grad-CAM Overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=130)

    # Panel 1: Original
    axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Heatmap
    im1 = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Focus on {pred_label} ({confidence * 100:.1f}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def plot_saliency_breakdown(
    original_image: np.ndarray,
    vanilla_saliency: np.ndarray,
    smoothgrad_saliency: np.ndarray,
    pred_label: str,
) -> plt.Figure:
    """
    Plots a 3-panel saliency comparison: Input, Vanilla Gradient, and SmoothGrad.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=130)

    axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(vanilla_saliency, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Vanilla Saliency", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(smoothgrad_saliency, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"SmoothGrad ({pred_label})", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_adversarial_breakdown(attack_results: Dict) -> plt.Figure:
    """
    Plots side-by-side comparison of Clean Image, Perturbation, Adversarial Image, and Confidence Delta.
    """
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), dpi=130)

    # Clean
    axes[0].imshow(attack_results["orig_image_np"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"Clean Input\nClass {attack_results['orig_prediction']} ({attack_results['orig_confidence']*100:.1f}%)",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].axis("off")

    # Perturbation
    im1 = axes[1].imshow(attack_results["perturbation_np"], cmap="coolwarm", vmin=0, vmax=1)
    axes[1].set_title(
        f"Perturbation (\u03b5={attack_results['epsilon']:.2f})\nL\u221e={attack_results['l_inf_distortion']:.3f}",
        fontsize=10,
        fontweight="bold",
    )
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Adversarial
    status = "FOOLED" if attack_results["is_fooled"] else "RESISTED"
    title_color = "#DC2626" if attack_results["is_fooled"] else "#16A34A"
    axes[2].imshow(attack_results["adv_image_np"], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Adversarial [{status}]\nClass {attack_results['adv_prediction']} ({attack_results['adv_confidence']*100:.1f}%)",
        fontsize=10,
        fontweight="bold",
        color=title_color,
    )
    axes[2].axis("off")

    # Probability shift bar
    orig_c = attack_results["orig_prediction"]
    adv_c = attack_results["adv_prediction"]
    classes = [f"Class {orig_c}", f"Class {adv_c}"]
    before_probs = [
        attack_results["orig_probs"][orig_c] * 100,
        attack_results["orig_probs"][adv_c] * 100,
    ]
    after_probs = [
        attack_results["adv_probs"][orig_c] * 100,
        attack_results["adv_probs"][adv_c] * 100,
    ]

    x = np.arange(len(classes))
    width = 0.35
    axes[3].bar(x - width / 2, before_probs, width, label="Before", color="#2563EB")
    axes[3].bar(x + width / 2, after_probs, width, label="After Attack", color="#EF4444")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(classes, fontsize=9)
    axes[3].set_ylabel("Probability (%)", fontsize=9)
    axes[3].set_title("Confidence Shift", fontsize=10, fontweight="bold")
    axes[3].legend(fontsize=8)
    axes[3].grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_curves(history: Dict[str, List[float]]) -> plt.Figure:
    """
    Plots dual-axis Loss and Accuracy learning trajectories across epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", color="#2563EB", label="Train Loss", lw=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#EA580C", label="Val Loss", lw=2)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Accuracy subplot
    ax2.plot(
        epochs,
        [a * 100 for a in history["train_acc"]],
        "o-",
        color="#16A34A",
        label="Train Acc",
        lw=2,
    )
    ax2.plot(
        epochs,
        [a * 100 for a in history["val_acc"]],
        "s--",
        color="#9333EA",
        label="Val Acc",
        lw=2,
    )
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Training & Validation Accuracy", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Plots annotated confusion matrix heatmap.
    """
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Multiclass Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    return fig


def plot_vae_manifold(vae_model, device: str = "cpu", grid_size: int = 15, span: float = 2.5) -> plt.Figure:
    """
    Decodes a 2D grid across latent space coordinates z1, z2 to visualize learned manifold.
    """
    import torch

    vae_model.eval()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)

    z1 = np.linspace(-span, span, grid_size)
    z2 = np.linspace(-span, span, grid_size)
    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    with torch.no_grad():
        for i, y_val in enumerate(z2):
            for j, x_val in enumerate(z1):
                z = torch.tensor([[x_val, y_val]], dtype=torch.float32, device=device)
                sample = vae_model.decode(z).squeeze().cpu().numpy()
                canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = sample

    ax.imshow(canvas, cmap="gnuplot2")
    ax.axis("off")
    ax.set_title(f"Latent Manifold Synthesis Grid ({grid_size}x{grid_size})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig
