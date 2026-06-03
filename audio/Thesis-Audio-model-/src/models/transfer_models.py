"""
Transfer learning baselines: ResNet50 and EfficientNet-B4.
Both accept single-channel log-mel spectrograms and fine-tune
a pretrained ImageNet backbone with a custom classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from .ordinal_head import OrdinalHead


class _TransferBase(nn.Module):
    """Shared logic for all transfer learning wrappers."""

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
        dropout: float,
        ordinal: bool,
        in_channels: int = 1,
    ):
        super().__init__()
        # Adapt first conv to single-channel input by averaging pretrained weights
        if in_channels != 3:
            self._adapt_first_conv(backbone, in_channels)

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)

        if ordinal:
            self.head = OrdinalHead(feat_dim, num_classes)
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    @staticmethod
    def _adapt_first_conv(backbone: nn.Module, in_channels: int):
        """Replace the first conv so it accepts `in_channels` channels."""
        # timm places the first conv in different locations; find it
        first_conv = None
        first_name = None
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                first_name = name
                break
        if first_conv is None:
            return

        old_weight = first_conv.weight  # [out, 3, H, W]
        new_weight = old_weight.mean(dim=1, keepdim=True)   # [out, 1, H, W]
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(new_weight.repeat(1, in_channels, 1, 1) / in_channels)
        # Replace via attribute path
        parts = first_name.split(".")
        parent = backbone
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_conv)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, lr_mult: float = 0.1):
        for param in self.backbone.parameters():
            param.requires_grad = True
        # Caller should manually set lr for backbone param group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        return self.head(features)

    def get_cam_target_layer(self) -> nn.Module:
        raise NotImplementedError


class ResNet50Model(_TransferBase):
    """ResNet-50 fine-tuned on log-mel spectrograms."""

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.4,
        ordinal: bool = True,
    ):
        backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,          # remove classification head
            global_pool="avg",
        )
        feat_dim = backbone.num_features
        super().__init__(backbone, feat_dim, num_classes, dropout, ordinal, in_channels)

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1].bn2


class EfficientNetB4Model(_TransferBase):
    """EfficientNet-B4 fine-tuned on log-mel spectrograms."""

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.4,
        ordinal: bool = True,
    ):
        backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=dropout,
        )
        feat_dim = backbone.num_features
        super().__init__(backbone, feat_dim, num_classes, dropout, ordinal, in_channels)

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.blocks[-1][-1].bn3
