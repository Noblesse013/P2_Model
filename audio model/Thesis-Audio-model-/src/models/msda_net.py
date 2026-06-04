"""
MSDA-Net: Multi-Scale Dual-Attention Network for Engine Health Grading.

Novel contributions:
  1. Multi-scale temporal feature extraction (parallel branches at 3 kernel widths)
     captures both transient knock/knock patterns and long-term bearing degradation.
  2. Dual-attention: Squeeze-Excitation (channel recalibration) + temporal
     self-attention (long-range dependency modelling).
  3. Ordinal classification head that respects the natural ordering of health grades.

Reference architecture:
  Input [B, 1, 128, T]
     │
     ├── Branch s=3  → SE → [B, C, F/8, T/8]  ─╮
     ├── Branch s=7  → SE → [B, C, F/8, T/8]   ├── Concat → FusionConv → TemporalAttn
     └── Branch s=15 → SE → [B, C, F/8, T/8]  ─╯
                                                    │
                                              GAP + GMP → FC → OrdinalHead
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .attention_modules import SqueezeExcitation, FrequencyAttention, TemporalSelfAttention
from .ordinal_head import OrdinalHead


# ── Multi-scale branch ────────────────────────────────────────

class _ScaleBranch(nn.Module):
    """
    One temporal-scale processing branch.
    Uses a wide temporal kernel to capture patterns at a specific time granularity.
    """

    def __init__(self, in_ch: int, out_ch: int, temporal_kernel: int, se_reduction: int):
        super().__init__()
        pad = temporal_kernel // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, temporal_kernel),
                      padding=(1, pad), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.se = SqueezeExcitation(out_ch, se_reduction) if se_reduction > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.pool(x)


# ── Encoder stage ─────────────────────────────────────────────

class _EncoderStage(nn.Module):
    """Three parallel scale branches followed by feature fusion."""

    def __init__(self, in_ch: int, out_ch: int, scales: list[int], se_reduction: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [_ScaleBranch(in_ch, out_ch, k, se_reduction) for k in scales]
        )
        total_ch = out_ch * len(scales)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [b(x) for b in self.branches]
        fused = torch.cat(branch_outs, dim=1)
        return self.fusion(fused)


# ── MSDA-Net ──────────────────────────────────────────────────

class MSDANet(nn.Module):
    """
    Multi-Scale Dual-Attention Network for 4-class engine health grading.

    Args:
        num_classes:       number of health grades (default 4)
        in_channels:       input channels (1 for single mel spectrogram)
        base_channels:     feature map depth per stage
        scales:            temporal kernel sizes for multi-scale branches
        se_reduction:      squeeze-excitation reduction ratio
        num_attn_heads:    heads in temporal self-attention
        attn_dropout:      dropout in attention layers
        dropout:           dropout before the classification head
        ordinal:           use ordinal regression head
        use_se:            enable Squeeze-Excitation blocks (ablation flag)
        use_temporal_attn: enable temporal self-attention (ablation flag)
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        base_channels: int = 64,
        n_mels: int = 128,
        scales: list[int] = None,
        se_reduction: int = 16,
        num_attn_heads: int = 8,
        attn_dropout: float = 0.1,
        dropout: float = 0.4,
        ordinal: bool = True,
        use_se: bool = True,
        use_temporal_attn: bool = True,
    ):
        super().__init__()
        if scales is None:
            scales = [3, 7, 15]
        c = base_channels
        # 4× MaxPool2d(2,2): stem + stage1 + stage2 + stage3
        freq_dim = max(n_mels // 16, 2)

        # ── Stage 0: stem ────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
        )

        # ── Stage 1-3: multi-scale encoder ──────────────────────
        _se = se_reduction if use_se else 0   # 0 → identity in _ScaleBranch
        self.stage1 = _EncoderStage(c,     c * 2,  scales, _se)
        self.stage2 = _EncoderStage(c * 2, c * 4,  scales, _se)
        self.stage3 = _EncoderStage(c * 4, c * 8,  scales, _se)

        # ── Frequency attention after stage 3 ────────────────────
        self.freq_attn = FrequencyAttention(freq_dim) if use_se else nn.Identity()

        # ── Temporal self-attention ──────────────────────────────
        if use_temporal_attn:
            self.temporal_attn = TemporalSelfAttention(
                embed_dim=c * 8,
                num_heads=num_attn_heads,
                dropout=attn_dropout,
            )
        else:
            self.temporal_attn = nn.Identity()

        # ── Pooling & head ───────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        feat_dim = c * 8 * 2       # concat avg + max pool

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if ordinal:
            self.head = OrdinalHead(feat_dim // 2, num_classes)
        else:
            self.head = nn.Linear(feat_dim // 2, num_classes)

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                  # [B, C,   F/2, T/2]
        x = self.stage1(x)                # [B, 2C,  F/4, T/4]
        x = self.stage2(x)                # [B, 4C,  F/8, T/8]
        x = self.stage3(x)                # [B, 8C, F/16, T/16]
        x = self.freq_attn(x)             # channel-wise frequency attention
        x = self.temporal_attn(x)         # temporal self-attention

        avg = self.gap(x).flatten(1)
        mx = self.gmp(x).flatten(1)
        feats = torch.cat([avg, mx], dim=1)

        feats = self.classifier(feats)
        return self.head(feats)

    def get_cam_target_layer(self) -> nn.Module:
        """Last conv-equivalent layer for Grad-CAM."""
        return self.stage3.fusion[0]       # 1×1 conv of final fusion block
