"""
FusionModel — frozen encoder embeddings → MLP → Ordinal Head.

Architecture:
    h_obd2  (256,) ─┐
                     ├─ concat (768,) → FC(256) → GELU → Dropout
    h_audio (512,) ─┘                → FC(128) → GELU → Dropout
                                     → OrdinalHead (3 binary thresholds)
                                     → grade ∈ {0, 1, 2, 3}

Only the MLP + OrdinalHead weights are trained.
Both encoders are loaded externally and kept frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import sys
from pathlib import Path


class OrdinalHead(nn.Module):
    """K-1 binary threshold classifiers (CORN formulation)."""

    def __init__(self, in_features: int, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.fc   = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, K-1) logits."""
        return self.fc(x) + self.bias.unsqueeze(0)

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        """(B, K-1) logits → (B,) integer grades."""
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=1)

    @staticmethod
    def to_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """(B, K-1) logits → (B, K) class probabilities."""
        p = torch.sigmoid(logits)
        B = p.size(0)
        ones  = torch.ones(B, 1, device=p.device)
        zeros = torch.zeros(B, 1, device=p.device)
        cum   = torch.cat([ones, p, zeros], dim=1)
        return (cum[:, :-1] - cum[:, 1:]).clamp(min=1e-8)


class FusionMLP(nn.Module):
    """
    Two-layer MLP that fuses concatenated encoder embeddings.

    Args:
        obd2_dim:   OBD2 embedding dimension (256)
        audio_dim:  Audio embedding dimension (512)
        hidden:     First hidden layer width (256)
        n_classes:  Number of health grades (4)
        dropout:    Dropout probability
    """

    def __init__(
        self,
        obd2_dim:  int = 256,
        audio_dim: int = 512,
        hidden:    int = 256,
        n_classes: int = 4,
        dropout:   float = 0.3,
    ):
        super().__init__()
        concat_dim = obd2_dim + audio_dim  # 768

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),
        )
        self.head = OrdinalHead(hidden // 2, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_obd2: torch.Tensor,
                h_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_obd2:  (B, 256) — frozen OBD2 encoder output
            h_audio: (B, 512) — frozen audio encoder output
        Returns:
            logits:  (B, K-1) — ordinal threshold logits
        """
        z = torch.cat([h_obd2, h_audio], dim=1)   # (B, 768)
        z = self.mlp(z)                            # (B, 128)
        return self.head(z)                        # (B, K-1)

    def predict(self, h_obd2: torch.Tensor,
                h_audio: torch.Tensor) -> torch.Tensor:
        """Returns (B,) integer grade predictions."""
        return OrdinalHead.decode(self.forward(h_obd2, h_audio))

    def predict_proba(self, h_obd2: torch.Tensor,
                      h_audio: torch.Tensor) -> torch.Tensor:
        """Returns (B, K) class probability distributions."""
        return OrdinalHead.to_class_probs(self.forward(h_obd2, h_audio))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
