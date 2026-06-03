"""
Evaluation on the held-out test set:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix plot
  - Per-class ROC-AUC curves
  - Training history plot
"""

import os
import pickle
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

from dataset import build_loaders
from model import build_model


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def get_predictions(model, loader, device, n_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())
        all_probs.append(probs)
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix (% of true class)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(y_true, y_probs, label_names, n_classes, save_path):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "darkorange", "green", "red"]

    for i, (name, color) in enumerate(zip(label_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Per Health Grade", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pr_curves(y_true, y_probs, label_names, n_classes, save_path):
    y_bin   = label_binarize(y_true, classes=list(range(n_classes)))
    colors  = ["steelblue", "darkorange", "green", "red"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, color) in enumerate(zip(label_names, colors)):
        if i >= y_bin.shape[1]:
            continue
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Per Health Grade", fontsize=13)
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_calibration_curves(y_true, y_probs, label_names, n_classes, save_path):
    colors  = ["steelblue", "darkorange", "green", "red"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

    for i, (name, color) in enumerate(zip(label_names, colors)):
        if i >= y_probs.shape[1]:
            continue
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        frac_pos, mean_pred = calibration_curve(y_bin, y_probs[:, i], n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", color=color, lw=2, label=name)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves — Per Health Grade", fontsize=13)
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_history(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs"); ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy over Epochs"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def run(config_path="config.yaml", history=None):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = cfg["labels"]["names"]
    n_classes = cfg["model"]["n_classes"]
    out_dir = cfg["data"]["processed_dir"]

    print("=== Evaluation ===")

    # Load best model
    checkpoint = torch.load(cfg["training"]["checkpoint_path"], map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}  (val_loss={checkpoint['val_loss']:.4f})")

    # Test loader
    _, _, test_loader, _ = build_loaders(cfg["data"]["processed_dir"], batch_size=64)

    # Predictions
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device, n_classes)

    # Save predictions for baseline comparison
    np.save(os.path.join(out_dir, "cnn_test_preds.npy"), y_pred)

    # Classification report — only include classes present in test set
    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    present_names = [label_names[i] for i in present]
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=present, target_names=present_names, digits=4))

    # Overall ROC-AUC (macro) — only for classes present in test set
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        macro_auc = roc_auc_score(y_bin[:, present], y_probs[:, present],
                                  average="macro", multi_class="ovr")
        print(f"  Macro ROC-AUC: {macro_auc:.4f}")
    except Exception as e:
        print(f"  Macro ROC-AUC: could not compute ({e})")

    # Plots
    print("\n--- Saving plots ---")
    plot_confusion_matrix(
        y_true, y_pred, label_names,
        save_path=os.path.join(out_dir, "confusion_matrix.png")
    )
    plot_roc_curves(
        y_true, y_probs, present_names, len(present),
        save_path=os.path.join(out_dir, "roc_curves.png")
    )
    plot_pr_curves(
        y_true, y_probs, present_names, len(present),
        save_path=os.path.join(out_dir, "pr_curves.png")
    )
    plot_calibration_curves(
        y_true, y_probs, label_names, n_classes,
        save_path=os.path.join(out_dir, "calibration_curves.png")
    )
    if history is not None:
        plot_training_history(
            history,
            save_path=os.path.join(out_dir, "training_history.png")
        )

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)
