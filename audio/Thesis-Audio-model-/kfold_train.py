"""
Stratified k-fold cross-validation training.
Produces per-fold and aggregate (mean ± std) results for the paper's Table.

Usage:
    python kfold_train.py
    python kfold_train.py model.name=cnn_baseline
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.data.dataset import (
    scan_folder, load_csv, stratified_split,
    EngineAudioDataset, make_weighted_sampler,
)
from src.data.preprocessing import AudioPreprocessor
from src.data.augmentation import AudioAugmentor, SpecAugment
from src.features.extractor import FeatureExtractor
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, compute_all_metrics
from src.evaluation.statistical_tests import summarise_kfold
from src.utils.logger import setup_logger
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


def main():
    logger = setup_logger("kfold")

    with open("configs/config.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    cli_overrides = [a for a in sys.argv[1:] if "=" in a]
    for ov in cli_overrides:
        k, v = ov.split("=", 1)
        try: v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError: pass
        OmegaConf.update(cfg, k, v, merge=True)

    random.seed(cfg.project.seed)
    np.random.seed(cfg.project.seed)
    torch.manual_seed(cfg.project.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = cfg.evaluation.k_folds
    out_dir = Path(cfg.project.output_dir) / f"kfold_{cfg.model.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data.get("train_csv"):
        all_samples = load_csv(cfg.data.train_csv)
    else:
        all_samples = scan_folder(cfg.data.data_dir)

    if not all_samples:
        logger.error("No audio files found.")
        return

    labels = np.array([s[1] for s in all_samples])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.project.seed)

    fold_metrics: dict = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels)):
        logger.info(f"\n{'─'*55}\n  Fold {fold_idx + 1}/{k}\n{'─'*55}")

        train_s = [all_samples[i] for i in train_idx]
        val_s   = [all_samples[i] for i in val_idx]

        # Build components
        fe = FeatureExtractor(
            sample_rate=cfg.data.sample_rate,
            n_fft=cfg.features.n_fft,
            hop_length=cfg.features.hop_length,
            n_mels=cfg.features.n_mels,
            f_min=cfg.features.f_min,
            f_max=cfg.features.f_max,
            n_mfcc=cfg.features.n_mfcc,
            use_delta=cfg.features.use_delta,
            normalize=cfg.features.normalize,
        )
        prep = AudioPreprocessor(
            sample_rate=cfg.data.sample_rate,
            segment_duration=cfg.data.segment_duration,
            hop_duration=cfg.data.hop_duration,
        )
        aug_cfg = dict(cfg.augmentation)
        aug_cfg["sample_rate"] = cfg.data.sample_rate
        aug_cfg["segment_duration"] = cfg.data.segment_duration

        train_ds = EngineAudioDataset(
            train_s, fe, prep,
            AudioAugmentor(aug_cfg) if cfg.augmentation.enabled else None,
            SpecAugment(aug_cfg) if cfg.augmentation.enabled else None,
            is_train=True,
        )
        val_ds = EngineAudioDataset(val_s, fe, prep, is_train=False)

        sampler = make_weighted_sampler(train_ds.labels, cfg.data.num_classes)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size,
            sampler=sampler, num_workers=cfg.data.num_workers,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.training.batch_size * 2,
            shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True,
        )

        model = build_model(cfg)
        trainer = Trainer(cfg, model, device)
        result = trainer.fit(train_loader, val_loader, run_name=f"{cfg.model.name}_fold{fold_idx+1}")

        # Evaluate on validation fold
        ckpt = torch.load(result["ckpt"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        metrics, _, _, _ = evaluate_model(
            model, val_loader, device,
            ordinal=cfg.model.ordinal,
            num_classes=cfg.data.num_classes,
            class_names=list(cfg.data.class_names),
        )

        for mname, val in metrics.items():
            if isinstance(val, (int, float, np.floating)):
                fold_metrics.setdefault(mname, []).append(float(val))

        logger.info(
            f"  Fold {fold_idx+1}: acc={metrics['accuracy']:.4f} "
            f"| macro_f1={metrics['macro_f1']:.4f} "
            f"| kappa={metrics['cohen_kappa']:.4f}"
        )

    # ── Summary table ─────────────────────────────────────────
    summary = summarise_kfold(fold_metrics)
    logger.info("\n" + "=" * 55)
    logger.info(f"  {k}-Fold CV Summary — {cfg.model.name}")
    logger.info("=" * 55)
    for mname, stats in summary.items():
        logger.info(f"  {mname:<22} {stats['mean']:.4f} ± {stats['std']:.4f}")
    logger.info("=" * 55)

    out_path = out_dir / "kfold_summary.json"
    with open(out_path, "w") as f:
        json.dump({"fold_metrics": fold_metrics, "summary": summary}, f, indent=2)
    logger.info(f"K-fold results → {out_path}")


if __name__ == "__main__":
    main()
