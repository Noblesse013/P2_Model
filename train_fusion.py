"""
Train the frozen-encoder fusion model.

Run AFTER extract_embeddings.py has been completed.

Usage:
    python train_fusion.py
    python train_fusion.py --config configs/config.yaml
    python train_fusion.py --lr 5e-4 --epochs 80 --batch_size 512
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.data.fusion_dataset import build_loaders
from src.models.fusion_model import FusionMLP
from src.training.trainer import FusionTrainer


def main():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--emb_dir",     default="embeddings")
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--patience",    type=int,   default=None)
    parser.add_argument("--alpha",       type=float, default=None,
                        help="Weight of WeightedKappa loss term (default 0.5)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg.get("fusion_training", {})

    # CLI overrides beat config file
    lr         = args.lr         or tcfg.get("lr",         1e-3)
    max_epochs = args.epochs     or tcfg.get("max_epochs", 60)
    batch_size = args.batch_size or tcfg.get("batch_size", 256)
    patience   = args.patience   or tcfg.get("patience",   10)
    alpha      = args.alpha      or tcfg.get("alpha",      0.5)
    warmup     =                    tcfg.get("warmup_epochs", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Embeddings : {args.emb_dir}")
    print(f"LR={lr}  epochs={max_epochs}  batch={batch_size}  patience={patience}  alpha={alpha}")

    # ── Data ─────────────────────────────────────────────────────────
    emb_dir = Path(args.emb_dir)
    if not (emb_dir / "obd2_train.npy").exists():
        print("\nEmbedding files not found. Run extract_embeddings.py first:\n"
              "  python extract_embeddings.py")
        sys.exit(1)

    train_loader, val_loader, test_loader = build_loaders(
        str(emb_dir), batch_size=batch_size
    )
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────────────
    model = FusionMLP(
        obd2_dim  = 256,
        audio_dim = 512,
        hidden    = cfg.get("fusion_model", {}).get("hidden", 256),
        n_classes = 4,
        dropout   = cfg.get("fusion_model", {}).get("dropout", 0.3),
    )
    print(f"\nFusion model parameters: {model.count_parameters():,}")

    # ── Train ────────────────────────────────────────────────────────
    trainer = FusionTrainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        lr             = lr,
        warmup_epochs  = warmup,
        max_epochs     = max_epochs,
        patience       = patience,
        alpha          = alpha,
        ckpt_path      = "experiments/fusion_best.pt",
        device         = device,
    )
    history = trainer.train()

    # Save training history
    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/fusion_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nTraining history saved -> experiments/fusion_training_history.json")

    # ── Quick test evaluation ─────────────────────────────────────────
    print("\nRunning test set evaluation ...")
    import subprocess
    subprocess.run([sys.executable, "evaluate_fusion.py",
                    "--checkpoint", "experiments/fusion_best.pt"], check=True)


if __name__ == "__main__":
    main()
