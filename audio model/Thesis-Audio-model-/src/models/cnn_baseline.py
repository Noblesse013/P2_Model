"""
Shallow CNN baseline for engine health grading.
Four convolutional blocks operating on log-mel spectrograms.
Used as the simplest neural baseline in ablation comparisons.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .ordinal_head import OrdinalHead


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    Simple 4-block CNN on mel spectrograms.
    Architecture: 4 × ConvBlock → GAP → FC → OrdinalHead/Softmax
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.4,
        ordinal: bool = True,
    ):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, c),
            ConvBlock(c, c * 2),
            ConvBlock(c * 2, c * 4),
            ConvBlock(c * 4, c * 8, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        feat_dim = c * 8

        if ordinal:
            self.head = OrdinalHead(feat_dim, num_classes)
        else:
            self.head = nn.Linear(feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.gap(features).flatten(1)
        pooled = self.dropout(pooled)
        return self.head(pooled)

    def get_cam_target_layer(self) -> nn.Module:
        return self.encoder[-1].block[-2]   # last BN before final pool
