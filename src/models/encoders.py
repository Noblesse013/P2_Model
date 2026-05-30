"""
Frozen encoder wrappers.

Both encoders are loaded from pretrained checkpoints and kept completely
frozen. Their only job is to extract grade-discriminative embedding vectors
from their respective modalities.

OBD2Encoder  → 256-dim context vector (after attention pooling, before classifier)
AudioEncoder → 512-dim feature vector (after FC projection, before ordinal head)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


# ── OBD2 Encoder ─────────────────────────────────────────────────────────────

class OBD2Encoder(nn.Module):
    """
    Wraps the pretrained CNN+BiGRU+Attention model.
    Returns the 256-dim context vector from the attention pooling step,
    bypassing the final 4-class classifier head.
    """
    OBD2_DIM = 256

    def __init__(self, model_dir: str, checkpoint: str, config: str,
                 device: torch.device = None):
        super().__init__()
        self.model_dir = Path(model_dir)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        src_path = str(self.model_dir / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        cfg_path = self.model_dir / config
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        from model import build_model
        ckpt = torch.load(self.model_dir / checkpoint, map_location=self._device, weights_only=False)
        backbone = build_model(cfg)
        backbone.load_state_dict(ckpt["model_state_dict"])

        # Keep everything except the final classifier head
        self.cnn         = backbone.cnn
        self.cnn_dropout = backbone.cnn_dropout
        self.bigru       = backbone.bigru
        self.gru_dropout = backbone.gru_dropout
        self.attention   = backbone.attention

        self.to(self._device)
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F)  →  embedding: (B, 256)"""
        x = x.permute(0, 2, 1)                     # (B, F, T)
        x = self.cnn(x)
        x = self.cnn_dropout(x)
        x = x.permute(0, 2, 1)                     # (B, T', C)
        gru_out, _ = self.bigru(x)
        gru_out = self.gru_dropout(gru_out)
        context, _ = self.attention(gru_out)        # (B, 256)
        return context

    def load_scaler(self, scaler_path: str):
        """Optionally load the RobustScaler for raw input scaling."""
        with open(scaler_path, "rb") as f:
            return pickle.load(f)


# ── Audio Encoder ─────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Wraps the pretrained MSDA-Net.
    Returns the 512-dim feature vector after the FC projection layer,
    bypassing the final ordinal regression head.
    """
    AUDIO_DIM = 512

    def __init__(self, model_dir: str, checkpoint: str, config: str,
                 device: torch.device = None):
        super().__init__()
        self.model_dir = Path(model_dir)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import yaml as _yaml
        from omegaconf import OmegaConf

        with open(self.model_dir / config) as f:
            raw = _yaml.safe_load(f)
        cfg = OmegaConf.create(raw)

        # Both projects have src/models/. Temporarily clear our project's src.*
        # from sys.modules so the audio factory.py's relative imports resolve
        # against the audio project's src/ instead of ours.
        audio_root = str(self.model_dir)
        _src_save = {k: sys.modules.pop(k)
                     for k in list(sys.modules)
                     if k == 'src' or k.startswith('src.')}
        if audio_root in sys.path:
            sys.path.remove(audio_root)
        sys.path.insert(0, audio_root)
        try:
            from src.models.factory import build_model as _audio_build_model
            ckpt = torch.load(self.model_dir / checkpoint,
                              map_location=self._device, weights_only=False)
            backbone = _audio_build_model(cfg)
            backbone.load_state_dict(ckpt["model_state_dict"])
        finally:
            for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
                del sys.modules[k]
            sys.modules.update(_src_save)

        # Keep everything except the ordinal head
        self.stem          = backbone.stem
        self.stage1        = backbone.stage1
        self.stage2        = backbone.stage2
        self.stage3        = backbone.stage3
        self.freq_attn     = backbone.freq_attn
        self.temporal_attn = backbone.temporal_attn
        self.gap           = backbone.gap
        self.gmp           = backbone.gmp
        self.classifier    = backbone.classifier   # FC(1024→512) + GELU + Dropout

        self.to(self._device)
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T)  →  embedding: (B, 512)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.freq_attn(x)
        x = self.temporal_attn(x)
        avg  = self.gap(x).flatten(1)
        mx   = self.gmp(x).flatten(1)
        feats = torch.cat([avg, mx], dim=1)         # (B, 1024)
        feats = self.classifier(feats)              # (B, 512)
        return feats
