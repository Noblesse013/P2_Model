"""
Generate all performance metric diagrams for the thesis.

Saves PNG figures to figures/ :
  1. training_curves.png        - loss + F1 + Kappa over epochs
  2. confusion_matrix.png       - absolute counts heatmap
  3. confusion_matrix_norm.png  - row-normalised (recall) heatmap
  4. per_class_metrics.png      - precision / recall / F1 per grade
  5. overall_metrics.png        - Accuracy / Macro-F1 / Weighted-F1 / QWK / MCC bar
  6. grade_distribution.png     - test-set sample count per grade

Usage:
    python plot_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

GRADE_LABELS = ["Normal", "Warning", "Fault", "Critical"]
COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]   # green/orange/red/purple
ACCENT  = "#2c3e50"

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})

# ── Load data ─────────────────────────────────────────────────────────────────
metrics  = json.load(open("experiments/fusion_test_metrics.json"))
history  = json.load(open("experiments/fusion_training_history.json"))

cm = np.array(metrics["confusion_matrix"])
n_epochs = len(history["train_loss"])
epochs   = list(range(1, n_epochs + 1))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Fusion Model Training Curves", fontsize=14, fontweight="bold", y=1.02)

# Loss
axes[0].plot(epochs, history["train_loss"], label="Train", color="#3498db", linewidth=2)
axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#e74c3c", linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss"); axes[0].legend()
axes[0].axvline(15, color="gray", linestyle="--", linewidth=1, label="Best epoch")

# Validation F1
axes[1].plot(epochs, history["val_f1"], color="#2ecc71", linewidth=2)
axes[1].axvline(15, color="gray", linestyle="--", linewidth=1)
axes[1].axhline(max(history["val_f1"]), color="#2ecc71", linestyle=":", linewidth=1,
                label=f"Best = {max(history['val_f1']):.4f}")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
axes[1].set_title("Validation Macro F1"); axes[1].legend()
axes[1].set_ylim(0, 1.05)

# Validation Kappa
axes[2].plot(epochs, history["val_kappa"], color="#9b59b6", linewidth=2)
axes[2].axvline(15, color="gray", linestyle="--", linewidth=1)
axes[2].axhline(max(history["val_kappa"]), color="#9b59b6", linestyle=":", linewidth=1,
                label=f"Best = {max(history['val_kappa']):.4f}")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Quadratic Kappa")
axes[2].set_title("Validation Quadratic Kappa"); axes[2].legend()
axes[2].set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "training_curves.png", bbox_inches="tight")
plt.close(fig)
print("Saved: training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion Matrix (absolute)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(GRADE_LABELS, rotation=30, ha="right")
ax.set_yticklabels(GRADE_LABELS)
ax.set_xlabel("Predicted Grade"); ax.set_ylabel("True Grade")
ax.set_title("Confusion Matrix (counts)", fontweight="bold")

thresh = cm.max() / 2
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i,j]:,}",
                ha="center", va="center",
                color="white" if cm[i,j] > thresh else "black",
                fontsize=11, fontweight="bold")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close(fig)
print("Saved: confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix (row-normalised = recall per class)
# ─────────────────────────────────────────────────────────────────────────────
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Recall (row %)")

ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(GRADE_LABELS, rotation=30, ha="right")
ax.set_yticklabels(GRADE_LABELS)
ax.set_xlabel("Predicted Grade"); ax.set_ylabel("True Grade")
ax.set_title("Normalised Confusion Matrix (recall)", fontweight="bold")

for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm_norm[i,j]:.2%}",
                ha="center", va="center",
                color="white" if cm_norm[i,j] > 0.6 else "black",
                fontsize=10)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "confusion_matrix_norm.png", bbox_inches="tight")
plt.close(fig)
print("Saved: confusion_matrix_norm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-class Precision / Recall / F1
# ─────────────────────────────────────────────────────────────────────────────
pc = metrics["per_class"]
precision = [pc[g]["precision"] for g in GRADE_LABELS]
recall    = [pc[g]["recall"]    for g in GRADE_LABELS]
f1        = [pc[g]["f1-score"]  for g in GRADE_LABELS]

x = np.arange(4)
w = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w, precision, w, label="Precision", color="#3498db", alpha=0.9)
b2 = ax.bar(x,     recall,    w, label="Recall",    color="#2ecc71", alpha=0.9)
b3 = ax.bar(x + w, f1,        w, label="F1-Score",  color="#e74c3c", alpha=0.9)

ax.set_xticks(x); ax.set_xticklabels(GRADE_LABELS)
ax.set_ylabel("Score"); ax.set_ylim(0.97, 1.005)
ax.set_title("Per-Class Precision, Recall and F1 Score", fontweight="bold")
ax.legend()
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

for bar_group in [b1, b2, b3]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0002,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "per_class_metrics.png", bbox_inches="tight")
plt.close(fig)
print("Saved: per_class_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Overall Metrics Bar
# ─────────────────────────────────────────────────────────────────────────────
metric_names  = ["Accuracy", "Macro F1", "Weighted F1", "Quad. Kappa", "MCC"]
metric_values = [
    metrics["accuracy"],
    metrics["macro_f1"],
    metrics["weighted_f1"],
    metrics["quadratic_kappa"],
    metrics["mcc"],
]
bar_colors = ["#3498db", "#2ecc71", "#27ae60", "#9b59b6", "#e67e22"]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(metric_names, metric_values, color=bar_colors, alpha=0.9, width=0.55)
ax.set_ylabel("Score"); ax.set_ylim(0.99, 1.002)
ax.set_title("Fusion Model — Overall Test-Set Metrics", fontweight="bold")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.4f"))

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0001,
            f"{h:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "overall_metrics.png", bbox_inches="tight")
plt.close(fig)
print("Saved: overall_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Test-Set Grade Distribution
# ─────────────────────────────────────────────────────────────────────────────
support = [int(pc[g]["support"]) for g in GRADE_LABELS]
total   = sum(support)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(GRADE_LABELS, support, color=COLORS, alpha=0.9, width=0.55)
ax.set_ylabel("Number of Samples")
ax.set_title("Test-Set Grade Distribution", fontweight="bold")

for bar, count in zip(bars, support):
    pct = count / total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10)

ax.set_ylim(0, max(support) * 1.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "grade_distribution.png", bbox_inches="tight")
plt.close(fig)
print("Saved: grade_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Loss components over training (CORN vs Kappa component)
# ─────────────────────────────────────────────────────────────────────────────
if "corn" in history and "kappa_component" in history:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["corn"],            label="CORN loss",           color="#3498db", linewidth=2)
    ax.plot(epochs, history["kappa_component"], label="Weighted Kappa loss", color="#e74c3c", linewidth=2)
    ax.plot(epochs, history["train_loss"],      label="Total train loss",    color=ACCENT,    linewidth=2, linestyle="--")
    ax.axvline(15, color="gray", linestyle=":", linewidth=1, label="Best epoch (15)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss Components", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_components.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: loss_components.png")

print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")
