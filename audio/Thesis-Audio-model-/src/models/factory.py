"""Model factory: builds any model from the config."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name.lower()
    nc = cfg.model.num_classes
    ic = cfg.model.input_channels
    ordinal = cfg.model.ordinal

    if name == "cnn_baseline":
        from .cnn_baseline import CNNBaseline
        c = cfg.model.cnn_baseline
        return CNNBaseline(nc, ic, c.base_channels, c.dropout, ordinal)

    if name == "deep_cnn":
        from .deep_cnn import DeepCNN
        c = cfg.model.deep_cnn
        return DeepCNN(nc, ic, c.base_channels, c.dropout, ordinal)

    if name == "resnet50":
        from .transfer_models import ResNet50Model
        c = cfg.model.transfer
        return ResNet50Model(nc, ic, c.pretrained, dropout=0.4, ordinal=ordinal)

    if name == "efficientnet_b4":
        from .transfer_models import EfficientNetB4Model
        c = cfg.model.transfer
        return EfficientNetB4Model(nc, ic, c.pretrained, dropout=0.4, ordinal=ordinal)

    if name == "msda_net":
        from .msda_net import MSDANet
        c = cfg.model.msda_net
        return MSDANet(
            num_classes=nc,
            in_channels=ic,
            base_channels=c.base_channels,
            n_mels=int(cfg.features.n_mels),
            scales=list(c.scales),
            se_reduction=c.se_reduction,
            num_attn_heads=c.num_attn_heads,
            attn_dropout=float(c.attn_dropout),
            dropout=float(c.dropout),
            ordinal=ordinal,
        )

    raise ValueError(f"Unknown model: '{name}'. "
                     "Choose from: cnn_baseline, deep_cnn, resnet50, efficientnet_b4, msda_net")
