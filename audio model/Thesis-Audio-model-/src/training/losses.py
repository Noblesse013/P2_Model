"""
Loss functions for engine health grading.

Implemented:
  - CrossEntropyLoss (standard baseline)
  - FocalLoss (class-imbalance robust)
  - LabelSmoothingCE (over-confidence regularizer)
  - OrdinalCrossEntropyLoss (respects grade ordering)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Focal Loss ────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
    Down-weights easy examples; focuses on hard misclassified samples.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [B, C], targets: [B] integer labels."""
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


# ── Label Smoothing CE ────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing to prevent overconfident predictions."""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


# ── Ordinal Cross-Entropy ─────────────────────────────────────

class OrdinalCrossEntropyLoss(nn.Module):
    """
    Ordinal regression loss (Niu et al., CVPR 2016).

    For K classes, the model outputs K-1 logits from OrdinalHead.
    The target for grade g is a binary vector: [1]*g + [0]*(K-1-g).
    Loss = mean BCE over all K-1 binary classifiers.

    Supports optional class weighting (applied per binary threshold).
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights    # [K] per-class weights if provided

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, K-1]  — raw outputs from OrdinalHead
        targets: [B]       — integer grades in {0, ..., K-1}
        """
        K = self.num_classes
        B = logits.shape[0]

        # Build binary label matrix [B, K-1]: label[b, k] = 1 if targets[b] > k
        threshold_labels = torch.zeros(B, K - 1, device=logits.device)
        for k in range(K - 1):
            threshold_labels[:, k] = (targets > k).float()

        loss = F.binary_cross_entropy_with_logits(
            logits, threshold_labels, reduction="none"   # [B, K-1]
        )

        if self.class_weights is not None:
            # Weight each sample by its class weight
            w = self.class_weights[targets]              # [B]
            loss = loss * w.unsqueeze(1)

        return loss.mean()


# ── Mixup-aware loss ──────────────────────────────────────────

class MixupLoss(nn.Module):
    """Wraps any loss function to handle mixed (y_a, y_b, lam) targets."""

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base = base_loss

    def forward(
        self,
        logits: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        return lam * self.base(logits, y_a) + (1.0 - lam) * self.base(logits, y_b)


# ── Factory ───────────────────────────────────────────────────

def build_loss(
    loss_type: str,
    num_classes: int = 4,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smooth: float = 0.1,
) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if loss_type == "label_smooth_ce":
        return LabelSmoothingCE(num_classes, smoothing=label_smooth)
    if loss_type == "ordinal_ce":
        return OrdinalCrossEntropyLoss(num_classes, class_weights)
    raise ValueError(f"Unknown loss: '{loss_type}'")
