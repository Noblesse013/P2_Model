"""
Ensemble model: combines predictions from multiple trained models.
Supports learned weight optimization on a validation set.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ordinal_head import OrdinalHead


class EnsembleModel(nn.Module):
    """
    Weighted soft-voting ensemble.
    Each sub-model outputs [B, K-1] ordinal logits or [B, K] class logits.
    Weights are learnable (optimized on validation set).
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int = 4,
        ordinal: bool = True,
        learn_weights: bool = True,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.ordinal = ordinal

        raw_weights = torch.ones(len(models))
        if learn_weights:
            self.log_weights = nn.Parameter(raw_weights)
        else:
            self.register_buffer("log_weights", raw_weights)

    @property
    def weights(self) -> torch.Tensor:
        return F.softmax(self.log_weights, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns averaged probability distribution [B, num_classes]."""
        probs_list = []
        for model in self.models:
            logits = model(x)
            if self.ordinal:
                probs = OrdinalHead.to_class_probs(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

        # Stack: [n_models, B, K]
        stacked = torch.stack(probs_list, dim=0)
        weights = self.weights.view(-1, 1, 1)           # [n_models, 1, 1]
        averaged = (stacked * weights).sum(dim=0)        # [B, K]
        return averaged

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted integer grade [B]."""
        probs = self.forward(x)
        return probs.argmax(dim=1)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Dict[str, str],
        model_instances: Dict[str, nn.Module],
        device: torch.device,
        num_classes: int = 4,
        ordinal: bool = True,
    ) -> "EnsembleModel":
        """Load weights from checkpoints and build ensemble."""
        loaded_models = []
        for name, ckpt_path in checkpoint_paths.items():
            model = model_instances[name]
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            loaded_models.append(model)
        return cls(loaded_models, num_classes=num_classes, ordinal=ordinal)
