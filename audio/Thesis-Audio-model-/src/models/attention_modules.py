"""
Attention modules used in MSDA-Net:
  - SqueezeExcitation: channel-wise recalibration (Hu et al., CVPR 2018)
  - TemporalSelfAttention: multi-head self-attention along the time axis
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention via global average pooling + FC gating."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, F, T] → returns recalibrated [B, C, F, T]."""
        scale = x.mean(dim=[2, 3])          # [B, C]
        scale = self.fc(scale).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * scale


class FrequencyAttention(nn.Module):
    """
    Frequency-band attention: squeeze along time → weight each mel band.
    Helps the network focus on fault-discriminative frequency ranges.
    """

    def __init__(self, n_mels: int):
        super().__init__()
        mid = max(n_mels // 4, 2)
        self.attn = nn.Sequential(
            nn.Linear(n_mels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_mels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, F, T] → [B, C, F, T]."""
        energy = x.mean(dim=[1, 3])          # [B, F]
        weights = self.attn(energy).unsqueeze(1).unsqueeze(3)   # [B, 1, F, 1]
        return x * weights


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention across the time axis of a feature map.
    Input feature map [B, C, F, T] → flatten F×C → attend over T → reshape back.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, F, T] → [B, C, F, T]."""
        B, C, F, T = x.shape
        # Pool along frequency to get [B, T, C] temporal sequence
        seq = x.mean(dim=2).permute(0, 2, 1)        # [B, T, C]

        # Pad embed_dim if needed (when C != embed_dim)
        if C != self.embed_dim:
            raise ValueError(f"Channel dim {C} must equal embed_dim {self.embed_dim}")

        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + attn_out)
        seq = self.norm2(seq + self.ff(seq))         # [B, T, C]

        # Broadcast back to [B, C, F, T]
        gate = seq.permute(0, 2, 1).unsqueeze(2)     # [B, C, 1, T]
        return x * torch.sigmoid(gate)
