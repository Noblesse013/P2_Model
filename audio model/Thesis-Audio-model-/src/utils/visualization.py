"""
Publication-quality visualizations for engine health grading results.
All figures are saved at 300 DPI with tight layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")    # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

_PALETTE = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]   # green→yellow→red→purple
CLASS_COLORS = {i: _PALETTE[i] for i in range(4)}


# ── Confusion matrix ──────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalise: bool = True,
) -> plt.Figure:
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
        fmt = ".2%"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── ROC curves ────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=_PALETTE[i % len(_PALETTE)],
                lw=1.8, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Training history ──────────────────────────────────────────

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#3498db")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#e74c3c")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#2ecc71")
    ax2.plot(epochs, history["val_f1"], label="Val Macro F1", color="#9b59b6", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Model comparison bar chart ────────────────────────────────

def plot_metric_comparison(
    model_metrics: Dict[str, Dict],
    metrics_to_plot: List[str] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    ci_data: Optional[Dict[str, Dict[str, Tuple]]] = None,
) -> plt.Figure:
    """
    Bar chart comparing multiple models across several metrics.

    Args:
        model_metrics: {model_name: {metric_name: value}}
        ci_data: optional {model_name: {metric_name: (point, lower, upper)}}
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "mcc"]

    models = list(model_metrics.keys())
    n_models = len(models)
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model_name in enumerate(models):
        vals = [model_metrics[model_name].get(m, 0) for m in metrics_to_plot]
        offsets = x + (i - n_models / 2 + 0.5) * width
        bars = ax.bar(offsets, vals, width * 0.9, label=model_name, color=colors[i], alpha=0.85)

        # Error bars from bootstrap CI
        if ci_data and model_name in ci_data:
            yerr_lo, yerr_hi = [], []
            for m in metrics_to_plot:
                if m in ci_data[model_name]:
                    pt, lo, hi = ci_data[model_name][m]
                    yerr_lo.append(pt - lo)
                    yerr_hi.append(hi - pt)
                else:
                    yerr_lo.append(0)
                    yerr_hi.append(0)
            ax.errorbar(offsets, vals, yerr=[yerr_lo, yerr_hi],
                        fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Spectrogram ───────────────────────────────────────────────

def plot_spectrogram(
    spec: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 512,
    title: str = "Log-Mel Spectrogram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    n_mels, n_frames = spec.shape
    duration = n_frames * hop_length / sample_rate

    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration, 0, n_mels],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Frequency Band")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Per-class F1 radar chart ──────────────────────────────────

def plot_f1_radar(
    model_metrics: Dict[str, Dict],
    class_names: List[str],
    title: str = "Per-Class F1 Scores",
    save_path: Optional[str] = None,
) -> plt.Figure:
    labels = class_names
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_metrics)))

    for (model_name, metrics), color in zip(model_metrics.items(), colors):
        values = [metrics.get(f"f1_{c}", 0) for c in class_names]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
