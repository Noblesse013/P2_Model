"""
Generates the overfitting / underfitting / best-fit diagnosis figure.
Saves to figures/fitting_diagnosis.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

h      = json.load(open("experiments/fusion_training_history.json"))
tl     = h["train_loss"]
vl     = h["val_loss"]
f1     = h["val_f1"]
kp     = h["val_kappa"]
epochs = list(range(1, len(tl) + 1))
gap    = [v - t for t, v in zip(tl, vl)]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fitting Diagnosis — Fusion MLP (25 epochs)", fontsize=15, fontweight="bold")

# ── Phase shading helper ──────────────────────────────────────────────────────
def shade_phases(ax):
    ax.axvspan(1,  11.5, alpha=0.06, color="orange",  label="Phase 1: Learning")
    ax.axvspan(11.5, 15.5, alpha=0.06, color="green",  label="Phase 2: Convergence")
    ax.axvspan(15.5, 25,  alpha=0.06, color="blue",   label="Phase 3: Plateau")
    ax.axvline(15, color="#27ae60", linestyle="--", linewidth=1.2, label="Best epoch (15)")

# ── 1. Train vs Val Loss ──────────────────────────────────────────────────────
ax = axes[0, 0]
shade_phases(ax)
ax.plot(epochs, tl, label="Train loss", color="#3498db", linewidth=2.2)
ax.plot(epochs, vl, label="Val loss",   color="#e74c3c", linewidth=2.2)
ax.fill_between(epochs, tl, vl, where=[v > t for t,v in zip(tl,vl)],
                alpha=0.12, color="#e74c3c", label="Generalisation gap")
ax.set_title("Train vs Validation Loss", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=9)

# annotation
ax.annotate("Gap closing\n(good sign)", xy=(20, (tl[19]+vl[19])/2),
            xytext=(16, 0.35), fontsize=9, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b"))

# ── 2. Generalisation Gap over time ─────────────────────────────────────────
ax = axes[0, 1]
shade_phases(ax)
ax.plot(epochs, gap, color="#8e44ad", linewidth=2.2, label="Val - Train loss")
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
ax.fill_between(epochs, 0, gap, where=[g > 0 for g in gap],
                alpha=0.15, color="#8e44ad")
ax.set_title("Generalisation Gap  (Val Loss − Train Loss)", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Gap")
ax.legend(fontsize=9)
ax.annotate(f"Peak gap: {max(gap):.3f}", xy=(2, max(gap)),
            xytext=(5, max(gap)+0.01), fontsize=9,
            arrowprops=dict(arrowstyle="->"))
ax.annotate(f"Final gap: {gap[-1]:.3f}", xy=(25, gap[-1]),
            xytext=(19, gap[-1]+0.02), fontsize=9,
            arrowprops=dict(arrowstyle="->"))

# ── 3. Val F1 over epochs ─────────────────────────────────────────────────────
ax = axes[1, 0]
shade_phases(ax)
ax.plot(epochs, f1, color="#2ecc71", linewidth=2.2, label="Val Macro F1")
ax.scatter([15], [max(f1)], color="#27ae60", zorder=5, s=80, label=f"Best F1={max(f1):.4f}")
ax.set_title("Validation Macro F1", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1")
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=9)

# phase transition annotation
ax.annotate("Phase\ntransition\n(epoch 11-15)",
            xy=(13, f1[12]), xytext=(4, 0.7), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))

# ── 4. Val Kappa over epochs ─────────────────────────────────────────────────
ax = axes[1, 1]
shade_phases(ax)
ax.plot(epochs, kp, color="#9b59b6", linewidth=2.2, label="Val Quadratic Kappa")
ax.scatter([f1.index(max(f1))+1], [kp[f1.index(max(f1))]],
           color="#8e44ad", zorder=5, s=80,
           label=f"At best F1 epoch: {kp[f1.index(max(f1))]:.4f}")
ax.set_title("Validation Quadratic Kappa", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Kappa")
ax.set_ylim(0.65, 1.05)
ax.legend(fontsize=9)

# ── Phase legend ─────────────────────────────────────────────────────────────
p1 = mpatches.Patch(color="orange", alpha=0.35, label="Phase 1: Learning (ep 1-11) — both losses drop, F1 stuck at 0.14")
p2 = mpatches.Patch(color="green",  alpha=0.35, label="Phase 2: Convergence (ep 12-15) — F1 jumps 0.14 -> 0.9995")
p3 = mpatches.Patch(color="blue",   alpha=0.35, label="Phase 3: Plateau (ep 16-25) — F1 stable, gap narrowing, BEST FIT")
fig.legend(handles=[p1, p2, p3], loc="lower center", ncol=1,
           fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(FIGURES_DIR / "fitting_diagnosis.png", bbox_inches="tight")
plt.close(fig)
print("Saved: figures/fitting_diagnosis.png")
