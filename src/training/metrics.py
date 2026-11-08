"""
Classification Evaluation Metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix.
Calculated directly via vector operations without third-party dependencies.
"""

from typing import Dict, Any, List
import numpy as np
import torch


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics.

    Args:
        y_true: 1D array of ground truth labels.
        y_pred: 1D array of predicted class indices.
        num_classes: Number of distinct classes.

    Returns:
        Dictionary containing overall accuracy, macro/weighted F1, confusion matrix,
        and per-class precision, recall, and F1 scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_samples = len(y_true)

    if n_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "confusion_matrix": np.zeros((num_classes, num_classes))}

    # Overall Accuracy
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion Matrix: row = true, col = pred
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class = {}
    f1_list = []
    support_list = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = int(np.sum(cm[c, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_list.append(f1)
        support_list.append(support)

    macro_f1 = float(np.mean(f1_list))
    total_support = sum(support_list)
    weighted_f1 = (
        float(sum(f * s for f, s in zip(f1_list, support_list)) / total_support)
        if total_support > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class": per_class,
        "total_samples": n_samples,
    }
