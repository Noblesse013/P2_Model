"""
Training loop for the FusionMLP.

Handles:
  - Cosine LR schedule with linear warmup
  - Early stopping on validation macro F1
  - Per-epoch logging of CORN + WeightedKappa components
  - Best checkpoint saving
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score
from tqdm import tqdm

from src.losses.ordinal_losses import OrdinalFusionLoss
from src.models.fusion_model import OrdinalHead


class FusionTrainer:
    def __init__(
        self,
        model:          nn.Module,
        train_loader,
        val_loader,
        lr:             float = 1e-3,
        weight_decay:   float = 1e-4,
        warmup_epochs:  int   = 5,
        max_epochs:     int   = 60,
        patience:       int   = 10,
        alpha:          float = 0.5,     # WeightedKappa loss weight
        ckpt_path:      str   = "experiments/fusion_best.pt",
        device:         torch.device = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.ckpt_path    = Path(ckpt_path)
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.criterion = OrdinalFusionLoss(n_classes=4, alpha=alpha)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Cosine schedule (takes over after warmup)
        cosine_epochs = max(1, max_epochs - warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=1e-6
        )
        self.warmup_epochs = warmup_epochs
        self.base_lr       = lr

        self.history: dict = {
            "train_loss": [], "val_loss": [],
            "val_f1": [], "val_kappa": [],
            "corn": [], "kappa_component": [],
        }

    # ------------------------------------------------------------------
    def _warmup_lr(self, epoch: int):
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _train_epoch(self) -> tuple[float, dict]:
        self.model.train()
        total_loss, corn_sum, kappa_sum, n = 0.0, 0.0, 0.0, 0

        for h_obd2, h_audio, labels in self.train_loader:
            h_obd2  = h_obd2.to(self.device)
            h_audio = h_audio.to(self.device)
            labels  = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(h_obd2, h_audio)
            loss, components = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            corn_sum   += components["corn"]  * bs
            kappa_sum  += components["kappa"] * bs
            n += bs

        return total_loss / n, {"corn": corn_sum / n, "kappa": kappa_sum / n}

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_labels = [], []

        for h_obd2, h_audio, labels in self.val_loader:
            h_obd2  = h_obd2.to(self.device)
            h_audio = h_audio.to(self.device)
            labels  = labels.to(self.device)

            logits = self.model(h_obd2, h_audio)
            loss, _ = self.criterion(logits, labels)
            preds   = OrdinalHead.decode(logits)

            total_loss  += loss.item() * labels.size(0)
            n           += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        kappa    = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
        return total_loss / n, macro_f1, kappa

    # ------------------------------------------------------------------
    def train(self) -> dict:
        best_f1       = -1.0
        patience_cnt  = 0

        header = f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val F1':>7} | {'Kappa':>7} | {'LR':>8}"
        print(f"\n{header}")
        print("-" * len(header))

        for epoch in range(1, self.max_epochs + 1):
            self._warmup_lr(epoch)

            train_loss, components = self._train_epoch()
            val_loss, val_f1, val_kappa = self._val_epoch()

            if epoch > self.warmup_epochs:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["val_kappa"].append(val_kappa)
            self.history["corn"].append(components["corn"])
            self.history["kappa_component"].append(components["kappa"])

            print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
                  f"{val_f1:>6.4f} | {val_kappa:>6.4f} | {lr:>8.2e}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_cnt = 0
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_f1":           val_f1,
                    "val_kappa":        val_kappa,
                }, self.ckpt_path)
                print(f"         ^ Best model saved (macro F1={val_f1:.4f})")
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no F1 improvement for {self.patience} epochs)")
                    break

        print(f"\nBest validation macro F1: {best_f1:.4f}")
        return self.history
