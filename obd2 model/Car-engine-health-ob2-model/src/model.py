"""
CNN + Bidirectional GRU + Attention classifier for 4-class engine health grading.

Architecture:
  Input (batch, timesteps, features)
    → 1D-CNN blocks  [local pattern extraction]
    → BiGRU layers   [temporal dependency modeling]
    → Attention pool [weight important timesteps]
    → MLP head       [classification]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over the time dimension.
    Returns weighted sum of GRU hidden states.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, gru_out):
        # gru_out: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.score(gru_out), dim=1)   # (batch, seq_len, 1)
        context = (weights * gru_out).sum(dim=1)              # (batch, hidden_dim)
        return context, weights.squeeze(-1)


class EngineHealthClassifier(nn.Module):
    def __init__(
        self,
        n_features: int = 5,
        n_classes: int = 4,
        cnn_channels: list = None,
        cnn_kernels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128]
        if cnn_kernels is None:
            cnn_kernels = [5, 3]

        # --- CNN blocks ---
        cnn_layers = []
        in_ch = n_features
        for out_ch, k in zip(cnn_channels, cnn_kernels):
            cnn_layers.append(ConvBlock(in_ch, out_ch, k))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_dropout = nn.Dropout(dropout)

        # --- Bidirectional GRU ---
        self.bigru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_dropout = nn.Dropout(dropout)

        # --- Attention ---
        gru_out_dim = gru_hidden * 2          # bidirectional doubles size
        self.attention = AdditiveAttention(gru_out_dim)

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, return_attention=False):
        # x: (batch, timesteps, features)

        # CNN expects (batch, features, timesteps)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)                      # (batch, cnn_channels[-1], reduced_time)
        x = self.cnn_dropout(x)
        x = x.permute(0, 2, 1)              # (batch, reduced_time, cnn_channels[-1])

        gru_out, _ = self.bigru(x)           # (batch, reduced_time, gru_hidden*2)
        gru_out = self.gru_dropout(gru_out)

        context, attn_weights = self.attention(gru_out)   # (batch, gru_hidden*2)

        logits = self.classifier(context)    # (batch, n_classes)

        if return_attention:
            return logits, attn_weights
        return logits


def build_model(cfg: dict) -> EngineHealthClassifier:
    m = cfg["model"]
    return EngineHealthClassifier(
        n_features=m["n_features"],
        n_classes=m["n_classes"],
        cnn_channels=m["cnn_channels"],
        cnn_kernels=m["cnn_kernel_sizes"],
        gru_hidden=m["gru_hidden"],
        gru_layers=m["gru_layers"],
        dropout=m["dropout"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
