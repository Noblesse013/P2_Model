"""
Deep VGG-style CNN baseline — stronger than shallow CNN, weaker than MSDA-Net.
Used to show depth-only gains in the ablation study.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .ordinal_head import OrdinalHead


class VGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_convs: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_convs):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepCNN(nn.Module):
    """
    5-stage VGG-style deep CNN.
    Architecture: [64, 128, 256, 512, 512] channels with 2 convs each → GAP → FC head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.5,
        ordinal: bool = True,
    ):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            VGGBlock(in_channels, c, num_convs=2),
            VGGBlock(c, c * 2, num_convs=2),
            VGGBlock(c * 2, c * 4, num_convs=3),
            VGGBlock(c * 4, c * 8, num_convs=3),
            VGGBlock(c * 8, c * 8, num_convs=3),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(c * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        if ordinal:
            self.head = OrdinalHead(256, num_classes)
        else:
            self.head = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.classifier(x)
        return self.head(x)

    def get_cam_target_layer(self) -> nn.Module:
        return self.features[-1].block[-2]   # last BN before pool
