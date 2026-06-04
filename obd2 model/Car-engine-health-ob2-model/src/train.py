"""
Full training loop with:
  - Class-weighted CrossEntropy loss
  - WeightedRandomSampler (balanced batches)
  - AdamW optimizer + ReduceLROnPlateau scheduler
  - Early stopping
  - Best model checkpoint
  - Live epoch logging
"""

import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from dataset import build_loaders
from model import build_model, count_parameters


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_class_weights(y_train: np.ndarray, n_classes: int, device: torch.device):
    present = np.unique(y_train)
    partial = compute_class_weight("balanced", classes=present, y=y_train)
    # Fill weights for any missing class with the mean weight so loss stays valid
    weights = np.ones(n_classes, dtype=np.float32) * partial.mean()
    for cls, w in zip(present, partial):
        weights[cls] = w
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def run(config_path="config.yaml"):
    cfg = load_config(config_path)
    t_cfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    processed_dir = cfg["data"]["processed_dir"]
    train_loader, val_loader, _, y_train = build_loaders(processed_dir, t_cfg["batch_size"])

    # Model
    model = build_model(cfg).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss with class weights
    class_weights = get_class_weights(y_train, cfg["model"]["n_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    base_lr = t_cfg["learning_rate"]
    warmup_epochs = 20  # ramp lr from base_lr/20 to base_lr over first 20 epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr / 20,            # start low; warmup loop brings it up
        weight_decay=t_cfg["weight_decay"]
    )
    cosine_epochs = t_cfg["epochs"] - warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-5
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    checkpoint_path = t_cfg["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, t_cfg["epochs"] + 1):
        # Linear LR warmup: ramp from base_lr/10 to base_lr over warmup_epochs
        if epoch <= warmup_epochs:
            warmup_lr = base_lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate_epoch(model, val_loader, criterion, device)

        # Cosine annealing takes over after warmup (no val_loss argument needed)
        if epoch > warmup_epochs:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.3%} | {val_loss:>8.4f} | {val_acc:>6.3%} | {lr:>8.2e}")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": cfg,
            }, checkpoint_path)
            print(f"         ^ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= t_cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {t_cfg['patience']} epochs)")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    # Save history to disk so the plot can be regenerated without retraining
    history_path = os.path.join(processed_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    return history


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)
