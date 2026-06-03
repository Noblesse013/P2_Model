"""
Comprehensive evaluation metrics for engine health grading.

Metrics reported:
  - Accuracy
  - Macro F1, Weighted F1
  - Macro Precision, Macro Recall
  - Cohen's Kappa (appropriate for ordinal/multi-class)
  - Matthews Correlation Coefficient (MCC)
  - ROC-AUC (one-vs-rest, macro)
  - Per-class precision, recall, F1
  - Confusion matrix
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ..models.ordinal_head import OrdinalHead


# ── Core metric computation ───────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    num_classes: int = 4,
) -> Dict:
    """
    Compute all evaluation metrics from arrays.

    Args:
        y_true:  true integer labels [N]
        y_pred:  predicted integer labels [N]
        y_prob:  softmax probabilities [N, K] for AUC computation
        class_names: list of class name strings
    Returns:
        dict of metric name → value
    """
    labels = list(range(num_classes))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)

    names = class_names or [str(i) for i in range(num_classes)]
    for i, name in enumerate(names):
        metrics[f"f1_{name}"] = per_class_f1[i] if i < len(per_class_f1) else 0.0
        metrics[f"precision_{name}"] = per_class_p[i] if i < len(per_class_p) else 0.0
        metrics[f"recall_{name}"] = per_class_r[i] if i < len(per_class_r) else 0.0

    # ROC-AUC (requires probability estimates)
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=labels,
            )
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ordinal: bool = True,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on a DataLoader and return metrics + raw arrays.

    Returns:
        (metrics_dict, y_true, y_pred, y_prob)
    """
    from ..models.ordinal_head import OrdinalHead
    import torch.nn.functional as F

    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)

            if ordinal:
                preds = OrdinalHead.decode(logits)
                probs = OrdinalHead.to_class_probs(logits)
            else:
                preds = logits.argmax(dim=1)
                probs = F.softmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names, num_classes)
    return metrics, y_true, y_pred, y_prob


def print_metrics_table(metrics: Dict, class_names: Optional[List[str]] = None):
    """Pretty-print a metrics summary to stdout."""
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    top_keys = ["accuracy", "macro_f1", "weighted_f1", "macro_precision",
                "macro_recall", "cohen_kappa", "mcc", "roc_auc"]
    for k in top_keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f"  {k:<22} {v:.4f}")
    print()
    if class_names:
        print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("  " + "-" * 44)
        for name in class_names:
            p = metrics.get(f"precision_{name}", 0)
            r = metrics.get(f"recall_{name}", 0)
            f = metrics.get(f"f1_{name}", 0)
            print(f"  {name:<12} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print("=" * 60 + "\n")
