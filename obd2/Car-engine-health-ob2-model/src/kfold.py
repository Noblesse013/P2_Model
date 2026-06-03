"""
Leave-One-Driver-Out K-Fold Cross-Validation (3 folds).

For each fold one driver is held out as the test set and the
remaining two drivers are split into train (85%) and val (15%).
A fresh model is trained from scratch for every fold.

Reports mean +/- std across all 3 folds to show that performance
is consistent regardless of which driver is held out.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from model import build_model
from preprocess import load_classified_data, auto_label_classified, make_windows
from augment import add_sensor_noise, apply_smote, soften_boundaries


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_class_weights(y_train, n_classes, device):
    present = np.unique(y_train)
    partial = compute_class_weight("balanced", classes=present, y=y_train)
    weights = np.ones(n_classes, dtype=np.float32) * partial.mean()
    for cls, w in zip(present, partial):
        weights[cls] = w
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_fold(model, train_loader, val_loader, y_train_np, cfg, device, checkpoint_path):
    t_cfg = cfg["training"]
    base_lr = t_cfg["learning_rate"]
    warmup_epochs = 20

    class_weights = get_class_weights(y_train_np, cfg["model"]["n_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr / 20, weight_decay=t_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, t_cfg["epochs"] + 1):
        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * epoch / warmup_epochs

        # --- train ---
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(y_b)
            tc += (logits.argmax(1) == y_b).sum().item()
            tt += len(y_b)

        # --- val ---
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                vl += loss.item() * len(y_b)
                vc += (logits.argmax(1) == y_b).sum().item()
                vt += len(y_b)

        val_loss = vl / vt
        val_acc  = vc / vt
        train_loss = tl / tt
        lr = optimizer.param_groups[0]["lr"]

        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        print(f"    Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}"
              f" | val_acc={val_acc:.3%} | lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= t_cfg["patience"]:
                print(f"    Early stopping at epoch {epoch}")
                break

    return best_val_loss


@torch.no_grad()
def evaluate_fold(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X_b, y_b in loader:
        preds.append(model(X_b.to(device)).argmax(1).cpu().numpy())
        trues.append(y_b.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def plot_kfold(fold_results, out_dir):
    folds = [f"Fold {r['fold']}\n(Driver {r['test_driver']})" for r in fold_results]
    metrics = ["accuracy", "macro_f1", "f1_normal", "f1_warning", "f1_faulty", "f1_critical"]
    labels  = ["Accuracy", "Macro F1", "F1-Normal", "F1-Warning", "F1-Faulty", "F1-Critical"]
    colors  = ["steelblue", "darkorange", "green", "red", "purple", "brown"]

    x = np.arange(len(folds))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [r[m] * 100 for r in fold_results]
        ax.bar(x + i * width, vals, width, label=lbl, color=col, alpha=0.85, edgecolor="white")

    # Mean line
    mean_acc = np.mean([r["accuracy"] for r in fold_results]) * 100
    ax.axhline(mean_acc, color="black", lw=1.2, linestyle="--",
               label=f"Mean Accuracy = {mean_acc:.2f}%")

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(folds, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_title("Leave-One-Driver-Out K-Fold Cross-Validation", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "kfold_results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run(config_path="config.yaml"):
    cfg         = load_config(config_path)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = cfg["labels"]["names"]
    n_classes   = cfg["model"]["n_classes"]
    ds_cfg      = cfg["data"]["classified_dataset"]
    feat_cols   = ds_cfg["feature_names"]
    thresholds  = cfg["data"]["label_thresholds"]
    ws          = cfg["preprocessing"]["window_size"]
    st          = cfg["preprocessing"]["stride"]
    val_size    = cfg["preprocessing"]["val_size"]
    seed        = cfg["preprocessing"]["random_seed"]
    out_dir     = cfg["data"]["processed_dir"]
    batch_size  = cfg["training"]["batch_size"]

    print("\n" + "=" * 60)
    print("  LEAVE-ONE-DRIVER-OUT K-FOLD CROSS-VALIDATION")
    print("=" * 60)
    print(f"  Device: {device}")

    # Load data once — shared across all folds
    print("\n[Loading data...]")
    df        = load_classified_data(cfg)
    labels    = auto_label_classified(df, thresholds)
    feats_arr = df[feat_cols].values.astype(np.float64)
    drv_ids   = df["Conductor_ID"].values
    drivers   = sorted(df["Conductor_ID"].unique())
    print(f"  Drivers: {drivers}")

    fold_results = []

    for fold_idx, test_drv in enumerate(drivers):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx+1}/3  |  Test Driver: {test_drv}  |  "
              f"Train Drivers: {[d for d in drivers if d != test_drv]}")
        print(f"{'='*60}")

        idx_all      = np.arange(len(df))
        idx_test     = idx_all[drv_ids == test_drv]
        idx_trainval = idx_all[drv_ids != test_drv]

        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_size,
            random_state=seed,
            stratify=labels[idx_trainval],
        )

        print(f"  Rows  — Train: {len(idx_train):,}  Val: {len(idx_val):,}  "
              f"Test: {len(idx_test):,}")

        # Scale on train rows only
        scaler    = RobustScaler()
        f_train   = scaler.fit_transform(feats_arr[idx_train]).astype(np.float32)
        f_val     = scaler.transform(feats_arr[idx_val]).astype(np.float32)
        f_test    = scaler.transform(feats_arr[idx_test]).astype(np.float32)

        # Windows
        X_tr, y_tr = make_windows(f_train, labels[idx_train], ws, st)
        X_vl, y_vl = make_windows(f_val,   labels[idx_val],   ws, st)
        X_te, y_te = make_windows(f_test,  labels[idx_test],  ws, st)

        print(f"  Windows — Train: {len(y_tr):,}  Val: {len(y_vl):,}  "
              f"Test: {len(y_te):,}")
        for i, nm in enumerate(label_names):
            print(f"    {nm:8s}: train={np.sum(y_tr==i):4d}  "
                  f"val={np.sum(y_vl==i):3d}  test={np.sum(y_te==i):4d}")

        # Augment training data only (same as main pipeline)
        print(f"\n  Augmenting fold {fold_idx+1} training data...")
        feat_names = cfg["data"]["feature_columns"]
        X_tr = add_sensor_noise(X_tr, feat_names, seed=42)
        X_tr, y_tr = apply_smote(X_tr, y_tr, seed=42)
        X_tr, y_tr = soften_boundaries(X_tr, y_tr, seed=42)
        print(f"  After augmentation: {len(y_tr):,} windows")
        for i, nm in enumerate(label_names):
            print(f"    {nm:8s}: {np.sum(y_tr==i):4d}")

        # DataLoaders
        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=batch_size, shuffle=True
        )
        vl_loader = DataLoader(
            TensorDataset(torch.tensor(X_vl), torch.tensor(y_vl)),
            batch_size=batch_size
        )
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
            batch_size=batch_size
        )

        # Fresh model
        model = build_model(cfg).to(device)
        ckpt_path = os.path.join(out_dir, f"kfold_fold{fold_idx+1}.pt")

        print(f"\n  Training fold {fold_idx+1}...")
        train_fold(model, tr_loader, vl_loader, y_tr, cfg, device, ckpt_path)

        # Load best checkpoint and evaluate
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Best checkpoint: epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

        y_pred, y_true = evaluate_fold(model, te_loader, device)

        acc      = float((y_pred == y_true).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        pf1      = f1_score(y_true, y_pred, average=None,
                            labels=list(range(n_classes)), zero_division=0)

        print(f"\n  Classification Report (test driver={test_drv}):")
        print(classification_report(
            y_true, y_pred, target_names=label_names, digits=4, zero_division=0
        ))
        print(f"  Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")

        fold_results.append({
            "fold":         fold_idx + 1,
            "test_driver":  int(test_drv),
            "accuracy":     acc,
            "macro_f1":     macro_f1,
            "f1_normal":    float(pf1[0]),
            "f1_warning":   float(pf1[1]),
            "f1_faulty":    float(pf1[2]),
            "f1_critical":  float(pf1[3]) if len(pf1) > 3 else 0.0,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  K-FOLD SUMMARY")
    print("=" * 60)
    accs = [r["accuracy"]  for r in fold_results]
    f1s  = [r["macro_f1"]  for r in fold_results]

    header = f"  {'Fold':>6} | {'Test Driver':>11} | {'Accuracy':>9} | {'Macro F1':>9}"
    print(header)
    print("  " + "-" * 44)
    for r in fold_results:
        print(f"  {r['fold']:>6} | {r['test_driver']:>11} | "
              f"{r['accuracy']:>8.4f}  | {r['macro_f1']:>8.4f}")

    print(f"\n  Mean Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean Macro F1 : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Save JSON
    summary = {
        "folds":           fold_results,
        "mean_accuracy":   float(np.mean(accs)),
        "std_accuracy":    float(np.std(accs)),
        "mean_macro_f1":   float(np.mean(f1s)),
        "std_macro_f1":    float(np.std(f1s)),
    }
    json_path = os.path.join(out_dir, "kfold_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Plot
    print("\n  Generating plot...")
    plot_kfold(fold_results, out_dir)

    print("\n=== K-Fold cross-validation complete ===")
    return summary


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(__file__))
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(PROJECT_ROOT, "config.yaml")
    run(config_path)
