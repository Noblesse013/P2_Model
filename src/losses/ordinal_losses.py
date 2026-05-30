"""
Ordinal loss functions for 4-class health grading.

CORN loss:         Binary cross-entropy on K-1 cumulative threshold classifiers.
                   Compatible with the ordinal head already used in the audio branch.

WeightedKappa loss: MSE on the expected ordinal grade — a differentiable proxy
                    for 1 - quadratic weighted Cohen's kappa. Penalises confusing
                    Normal with Critical far more than adjacent-grade mistakes.

Combined loss:     CORN + alpha * WeightedKappa  (default alpha = 0.5)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CORN ─────────────────────────────────────────────────────────────────────

def corn_loss(logits: torch.Tensor, targets: torch.Tensor,
              n_classes: int = 4) -> torch.Tensor:
    """
    Args:
        logits:    (B, K-1) — raw threshold logits from OrdinalHead
        targets:   (B,) — integer grade labels 0…K-1
        n_classes: K (default 4)
    Returns:
        scalar loss
    """
    K = n_classes - 1
    # binary target: threshold k is 1 if grade > k
    binary = torch.zeros(targets.size(0), K, device=logits.device, dtype=logits.dtype)
    for k in range(K):
        binary[:, k] = (targets > k).float()
    return F.binary_cross_entropy_with_logits(logits, binary)


# ── Weighted Kappa (MSE proxy) ───────────────────────────────────────────────

def weighted_kappa_loss(logits: torch.Tensor, targets: torch.Tensor,
                        n_classes: int = 4) -> torch.Tensor:
    """
    Differentiable proxy for quadratic weighted kappa.
    Computes the expected grade from the cumulative probability distribution
    and minimises squared distance to the true grade.

    Args:
        logits:    (B, K-1) — threshold logits
        targets:   (B,) — integer grade labels 0…K-1
    Returns:
        scalar MSE loss on ordinal grade
    """
    probs = _ordinal_to_class_probs(logits, n_classes)          # (B, K)
    grades = torch.arange(n_classes, device=logits.device, dtype=logits.dtype)
    expected = (probs * grades).sum(dim=1)                       # (B,)
    return F.mse_loss(expected, targets.float())


def _ordinal_to_class_probs(logits: torch.Tensor,
                             n_classes: int) -> torch.Tensor:
    """Convert (B, K-1) threshold logits → (B, K) class probabilities."""
    p = torch.sigmoid(logits)
    B = p.size(0)
    ones  = torch.ones(B, 1, device=logits.device, dtype=logits.dtype)
    zeros = torch.zeros(B, 1, device=logits.device, dtype=logits.dtype)
    cumulative = torch.cat([ones, p, zeros], dim=1)             # (B, K+1)
    class_probs = cumulative[:, :-1] - cumulative[:, 1:]
    return class_probs.clamp(min=1e-8)


# ── Combined loss ────────────────────────────────────────────────────────────

class OrdinalFusionLoss(nn.Module):
    """
    Combined CORN + WeightedKappa loss for the fusion model.

    L = L_CORN + alpha * L_WeightedKappa

    alpha = 0.5 by default — found empirically to balance the two objectives.
    Both losses are on the same (B, K-1) logit tensor.
    """

    def __init__(self, n_classes: int = 4, alpha: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l_corn  = corn_loss(logits, targets, self.n_classes)
        l_kappa = weighted_kappa_loss(logits, targets, self.n_classes)
        total   = l_corn + self.alpha * l_kappa
        return total, {"corn": l_corn.item(), "kappa": l_kappa.item()}
