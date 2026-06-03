"""
Training loop with:
  - Cosine annealing with linear warmup
  - Automatic mixed precision (AMP)
  - Gradient clipping
  - Early stopping
  - MLflow experiment tracking
  - Mixup augmentation support
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import build_loss, MixupLoss, OrdinalCrossEntropyLoss
from ..data.augmentation import mixup_batch
from ..models.ordinal_head import OrdinalHead


class CosineWarmupScheduler(LambdaLR):
    """Linear warmup + cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Orchestrates training, validation, and checkpointing for a single model.
    """

    def __init__(self, cfg, model: nn.Module, device: torch.device):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(cfg.project.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Loss ─────────────────────────────────────────────
        self.ordinal = cfg.model.ordinal
        class_weights = self._compute_class_weights()
        self.criterion = build_loss(
            cfg.training.loss,
            num_classes=cfg.data.num_classes,
            class_weights=class_weights,
            focal_gamma=float(cfg.training.focal_gamma),
            label_smooth=float(cfg.training.label_smooth),
        )

        use_mixup = getattr(cfg.augmentation, "mixup", False)
        self.use_mixup = use_mixup and cfg.augmentation.enabled
        self.mixup_alpha = float(getattr(cfg.augmentation, "mixup_alpha", 0.4))
        if self.use_mixup:
            self.criterion = MixupLoss(self.criterion)

        # ── Optimizer ────────────────────────────────────────
        self.optimizer = self._build_optimizer()

        # ── Scheduler ────────────────────────────────────────
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=cfg.training.warmup_epochs,
            total_epochs=cfg.training.num_epochs,
        )

        # ── AMP ──────────────────────────────────────────────
        self.use_amp = cfg.training.use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── Early stopping ───────────────────────────────────
        self.early_stopper = EarlyStopping(patience=cfg.training.early_stopping_patience)

        # ── Transfer learning schedule ───────────────────────
        self.freeze_backbone_epochs = getattr(cfg.model.get("transfer", {}), "freeze_backbone_epochs", 0)

    # ── Public API ────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_name: str = "run",
    ) -> dict:
        """Train and return best validation metrics."""
        best_val_f1 = 0.0
        best_ckpt_path = self.output_dir / f"{run_name}_best.pt"
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self._flat_params())

            for epoch in range(1, self.cfg.training.num_epochs + 1):
                # Backbone unfreeze after warm-up epochs
                if epoch == self.freeze_backbone_epochs + 1:
                    if hasattr(self.model, "unfreeze_backbone"):
                        self.model.unfreeze_backbone()

                t0 = time.time()
                train_loss = self._train_epoch(train_loader)
                val_loss, val_acc, val_f1 = self._val_epoch(val_loader)
                self.scheduler.step()

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)

                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss,
                     "val_acc": val_acc, "val_f1": val_f1},
                    step=epoch,
                )

                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:3d}/{self.cfg.training.num_epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | {elapsed:.1f}s"
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self._save_checkpoint(best_ckpt_path, epoch, val_f1)

                if self.early_stopper(val_f1):
                    print(f"Early stopping at epoch {epoch}")
                    break

            mlflow.log_metric("best_val_f1", best_val_f1)
            mlflow.log_artifact(str(best_ckpt_path))

        return {"best_val_f1": best_val_f1, "history": history, "ckpt": str(best_ckpt_path)}

    # ── Train / val steps ─────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for inputs, labels in tqdm(loader, desc="  train", leave=False):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if self.use_mixup:
                inputs, y_a, y_b, lam = mixup_batch(inputs, labels, self.mixup_alpha)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(inputs)
                if self.use_mixup:
                    loss = self.criterion(logits, y_a, y_b, lam)
                else:
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * inputs.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(inputs)
                # For validation, always use plain CE or ordinal CE (no mixup)
                if isinstance(self.criterion, MixupLoss):
                    loss = self.criterion.base(logits, labels)
                else:
                    loss = self.criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)

            if self.ordinal:
                preds = OrdinalHead.decode(logits)
            else:
                preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, float(acc), float(f1)

    # ── Helpers ───────────────────────────────────────────────

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        mode = getattr(self.cfg.training, "class_weights", None)
        if mode == "balanced":
            # Will be recomputed from loader counts in fit(); placeholder
            return None
        return None

    def _build_optimizer(self):
        opt = self.cfg.training.optimizer.lower()
        lr = float(self.cfg.training.lr)
        wd = float(self.cfg.training.weight_decay)
        params = self.model.parameters()

        if opt == "adamw":
            return AdamW(params, lr=lr, weight_decay=wd)
        if opt == "adam":
            return Adam(params, lr=lr, weight_decay=wd)
        if opt == "sgd":
            return SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
        raise ValueError(f"Unknown optimizer: {opt}")

    def _save_checkpoint(self, path: Path, epoch: int, val_f1: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": val_f1,
                "cfg": dict(self.cfg),
            },
            path,
        )

    def _flat_params(self) -> dict:
        flat = {}
        for key in ["training", "model", "features", "augmentation"]:
            section = getattr(self.cfg, key, {})
            for k, v in section.items():
                flat[f"{key}.{k}"] = str(v)
        return flat
