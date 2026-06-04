"""
Ordinal classification head based on:
  Niu et al., "Ordinal Regression with Multiple Output CNN for Age Estimation,"
  CVPR 2016.

Outputs K-1 cumulative probabilities P(grade >= k) for k = 1, ..., K-1.
During inference, the predicted grade is sum(P_k > 0.5).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OrdinalHead(nn.Module):
    """
    Binary threshold layer: one sigmoid unit per threshold.
    For K classes, outputs [B, K-1] logits.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # Shared projection → per-threshold bias
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, K-1] logits (apply sigmoid for probabilities)."""
        shared = self.fc(x)                          # [B, 1]
        logits = shared + self.bias.unsqueeze(0)     # [B, K-1]
        return logits

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        """Convert [B, K-1] logits → [B] integer grade predictions."""
        probs = torch.sigmoid(logits)
        return (probs > 0.5).long().sum(dim=1)

    @staticmethod
    def to_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, K-1] cumulative logits → [B, K] class probability distribution.
        P(y=k) = P(y>=k) - P(y>=k+1), with boundary conditions.
        """
        p = torch.sigmoid(logits)                    # [B, K-1]
        B, Km1 = p.shape
        # P(y >= 0) = 1, P(y >= K) = 0
        ones = torch.ones(B, 1, device=p.device)
        zeros = torch.zeros(B, 1, device=p.device)
        cumulative = torch.cat([ones, p, zeros], dim=1)  # [B, K+1]
        class_probs = cumulative[:, :-1] - cumulative[:, 1:]
        return class_probs.clamp(min=1e-8)
