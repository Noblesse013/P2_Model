"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for mel spectrograms.

Selkoe et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization," ICCV 2017.

Usage:
    cam = GradCAM(model, target_layer=model.get_cam_target_layer())
    heatmap = cam(input_tensor, target_class=2)  # [H, W] numpy array in [0,1]
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation compatible with all model architectures
    used in this project.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._hooks: list = []
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        ordinal: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: [1, C, F, T] single sample
            target_class: class index to explain; None → argmax
            ordinal: whether model uses ordinal head

        Returns:
            heatmap: [F, T] numpy array in [0, 1], same spatial size as input
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if ordinal:
            # Explain via class probability
            from ..models.ordinal_head import OrdinalHead
            class_probs = OrdinalHead.to_class_probs(logits)
            if target_class is None:
                target_class = class_probs.argmax(dim=1).item()
            score = class_probs[0, target_class]
        else:
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            score = logits[0, target_class]

        self.model.zero_grad()
        score.backward()

        # Global average pooling of gradients → channel weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        # Upsample to input spatial size
        _, _, F_in, T_in = input_tensor.shape
        cam = F.interpolate(cam, size=(F_in, T_in), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def __del__(self):
        self.remove_hooks()


# ── Batch generation helper ───────────────────────────────────

def generate_gradcam_grid(
    model: nn.Module,
    dataset,
    device: torch.device,
    class_names: List[str],
    num_samples_per_class: int = 5,
    ordinal: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Generate a grid of Grad-CAM overlays for publication figures.
    Each row = one health grade; columns = sample examples.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    target_layer = model.get_cam_target_layer()
    cam_extractor = GradCAM(model, target_layer)
    num_classes = len(class_names)

    # Group samples by class
    by_class: dict[int, list] = {c: [] for c in range(num_classes)}
    for idx in range(len(dataset)):
        spec, label = dataset[idx]
        if len(by_class[label]) < num_samples_per_class:
            by_class[label].append((spec, label))
        if all(len(v) >= num_samples_per_class for v in by_class.values()):
            break

    n_cols = num_samples_per_class
    n_rows = num_classes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    fig.suptitle("Grad-CAM: Engine Health Grade Activation Maps", fontsize=13, fontweight="bold")

    for row_idx, cls in enumerate(range(num_classes)):
        samples = by_class[cls]
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            ax.axis("off")
            if col_idx >= len(samples):
                continue

            spec, lbl = samples[col_idx]
            inp = spec.unsqueeze(0).to(device)
            cam = cam_extractor(inp, target_class=cls, ordinal=ordinal)

            # Show log-mel spectrogram
            img = spec.squeeze().numpy()
            ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                      interpolation="nearest")
            # Overlay CAM
            ax.imshow(cam, aspect="auto", origin="lower", cmap="jet",
                      alpha=0.45, interpolation="nearest")

            if col_idx == 0:
                ax.set_ylabel(class_names[cls], fontsize=10, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"Sample {col_idx + 1}", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Grad-CAM grid saved → {save_path}")
    else:
        plt.show()

    cam_extractor.remove_hooks()
