"""
Main training script for engine health grading.

Usage:
    python train.py                              # train MSDA-Net with default config
    python train.py model.name=cnn_baseline      # train a different model
    python train.py training.lr=5e-4             # override any config key
    python train.py model.name=resnet50 training.loss=focal

Supports Hydra-style dot-key overrides directly on the command line.
"""

import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.data.dataset import build_dataloaders, scan_folder
from src.data.preprocessing import AudioPreprocessor
from src.features.extractor import FeatureExtractor
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.logger import setup_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prebuild_cache(cfg, logger, max_workers: int = 4) -> None:
    """Pre-compute and cache all spectrograms before training starts."""
    cache_dir = Path(cfg.data.data_dir) / ".spec_cache"
    all_samples = scan_folder(cfg.data.data_dir)
    missing = [(p, l) for p, l in all_samples
               if not (cache_dir / Path(p).parent.name / (Path(p).stem + ".npy")).exists()]

    if not missing:
        logger.info("Spectrogram cache: all files cached, skipping pre-build.")
        return

    logger.info(f"Pre-building spectrogram cache for {len(missing)}/{len(all_samples)} files "
                f"using {max_workers} threads ...")

    fe = FeatureExtractor(
        sample_rate=int(cfg.data.sample_rate),
        n_fft=int(cfg.features.n_fft),
        hop_length=int(cfg.features.hop_length),
        n_mels=int(cfg.features.n_mels),
        f_min=float(cfg.features.f_min),
        f_max=float(cfg.features.f_max),
        n_mfcc=int(cfg.features.n_mfcc),
        use_delta=bool(cfg.features.use_delta),
        normalize=bool(cfg.features.normalize),
    )
    prep = AudioPreprocessor(
        sample_rate=int(cfg.data.sample_rate),
        segment_duration=float(cfg.data.segment_duration),
        hop_duration=float(cfg.data.hop_duration),
        normalize=True,
    )

    def _compute_one(item):
        path, _ = item
        p = Path(path)
        out = cache_dir / p.parent.name / (p.stem + ".npy")
        if out.exists():
            return True
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            segments = prep.process(path)
            waveform = segments[len(segments) // 2]
            spec = fe.extract(waveform)
            np.save(str(out), spec.numpy())
            return True
        except Exception:
            return False

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_compute_one, item): item for item in missing}
        for fut in as_completed(futs):
            if fut.result():
                done += 1
            else:
                failed += 1
            if (done + failed) % 500 == 0:
                logger.info(f"  Cache: {done + failed}/{len(missing)} done ...")

    logger.info(f"Cache pre-build complete: {done} ok, {failed} failed.")


def apply_cli_overrides(cfg, overrides: list[str]):
    """Apply CLI key=value overrides to OmegaConf config."""
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        # Try to cast to appropriate type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def main():
    # ── Load config ──────────────────────────────────────────
    cfg_path = Path("configs/config.yaml")
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    # Apply CLI overrides
    cli_overrides = [a for a in sys.argv[1:] if "=" in a]
    cfg = apply_cli_overrides(cfg, cli_overrides)

    # ── Setup ────────────────────────────────────────────────
    set_seed(cfg.project.seed)
    logger = setup_logger(
        log_file=f"{cfg.project.log_dir}/train_{cfg.model.name}.log"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model:  {cfg.model.name}")
    logger.info(f"Loss:   {cfg.training.loss}")

    # ── Spectrogram cache pre-build ──────────────────────────
    prebuild_cache(cfg, logger)

    # ── Data ─────────────────────────────────────────────────
    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(
        f"Train={len(train_loader.dataset)} | "
        f"Val={len(val_loader.dataset)} | "
        f"Test={len(test_loader.dataset)}"
    )

    # ── Model ────────────────────────────────────────────────
    logger.info("Building model...")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Freeze backbone if transfer learning
    if cfg.model.name in ("resnet50", "efficientnet_b4"):
        freeze_epochs = cfg.model.transfer.freeze_backbone_epochs
        if freeze_epochs > 0:
            model.freeze_backbone()
            logger.info(f"Backbone frozen for first {freeze_epochs} epochs")

    # ── Train ────────────────────────────────────────────────
    run_name = f"{cfg.model.name}_{cfg.training.loss}"
    trainer = Trainer(cfg, model, device)
    result = trainer.fit(train_loader, val_loader, run_name=run_name)

    logger.info(f"Training complete. Best val F1 = {result['best_val_f1']:.4f}")
    logger.info(f"Checkpoint saved → {result['ckpt']}")

    # ── Quick test evaluation ────────────────────────────────
    logger.info("Evaluating on test set...")
    from src.evaluation.metrics import evaluate_model, print_metrics_table

    ckpt = torch.load(result["ckpt"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, device,
        ordinal=cfg.model.ordinal,
        num_classes=cfg.data.num_classes,
        class_names=list(cfg.data.class_names),
    )
    print_metrics_table(metrics, list(cfg.data.class_names))

    # Save confusion matrix
    from src.utils.visualization import plot_confusion_matrix
    out_dir = Path(cfg.project.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=list(cfg.data.class_names),
        title=f"Confusion Matrix — {cfg.model.name}",
        save_path=str(out_dir / "confusion_matrix.png"),
    )
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
