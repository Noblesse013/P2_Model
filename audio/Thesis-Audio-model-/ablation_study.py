"""
Ablation study: isolates contribution of each MSDA-Net component.

Ablation conditions:
  A) Full MSDA-Net (proposed)
  B) No multi-scale (single branch, k=7 only)
  C) No SE attention (remove channel recalibration)
  D) No temporal attention (remove TemporalSelfAttention)
  E) No ordinal head (standard softmax CE)
  F) No augmentation (train with clean data only)
  G) No SpecAugment (time-domain augmentation only)

Results are saved to experiments/results/ablation/
"""

import sys
import copy
import json
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from src.data.dataset import build_dataloaders, scan_folder, stratified_split
from src.evaluation.metrics import evaluate_model, compute_all_metrics
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.visualization import plot_metric_comparison


# ── Ablation model variants ───────────────────────────────────

def build_ablation_models(cfg) -> dict:
    """Build one MSDA-Net variant per ablation condition (A–E)."""
    from src.models.msda_net import MSDANet

    c = cfg.model.msda_net

    def make(**overrides):
        kw = dict(
            num_classes=cfg.model.num_classes,
            in_channels=cfg.model.input_channels,
            base_channels=c.base_channels,
            scales=list(c.scales),
            se_reduction=c.se_reduction,
            num_attn_heads=c.num_attn_heads,
            attn_dropout=c.attn_dropout,
            dropout=c.dropout,
            ordinal=True,
            use_se=True,
            use_temporal_attn=True,
        )
        kw.update(overrides)
        return MSDANet(**kw)

    return {
        "MSDA-Net (Full)":       make(),
        "No Multi-Scale":        make(scales=[7]),
        "No SE Attention":       make(use_se=False),
        "No Temporal Attn":      make(use_temporal_attn=False),
        "Softmax (No Ordinal)":  make(ordinal=False),
    }


# ── Training helper ───────────────────────────────────────────

def train_variant(
    variant_name: str,
    model,
    cfg,
    train_loader,
    val_loader,
    test_loader,
    device,
    out_dir: Path,
    use_ordinal: bool = True,
) -> dict:
    """Train one ablation variant, evaluate on test set."""
    # Override ordinal flag in cfg based on model
    run_cfg = OmegaConf.merge(cfg, {"model": {"ordinal": use_ordinal}})

    trainer = Trainer(run_cfg, model, device)
    run_name = variant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    result = trainer.fit(train_loader, val_loader, run_name=f"ablation_{run_name}")

    # Test evaluation
    ckpt = torch.load(result["ckpt"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, device,
        ordinal=use_ordinal,
        num_classes=cfg.data.num_classes,
        class_names=list(cfg.data.class_names),
    )

    print(f"\n  {variant_name}: acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}")
    return metrics


# ── Main ──────────────────────────────────────────────────────

def main():
    import random
    import numpy as np

    logger = setup_logger("ablation")

    with open("configs/config.yaml") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # Faster ablation: reduce epochs
    cfg.training.num_epochs = 50
    cfg.training.early_stopping_patience = 10

    random.seed(cfg.project.seed)
    np.random.seed(cfg.project.seed)
    torch.manual_seed(cfg.project.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.project.output_dir) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    variants = build_ablation_models(cfg)
    ablation_results: dict = {}

    for name, model in variants.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Ablation variant: {name}")
        logger.info(f"{'='*60}")
        is_ordinal = "Softmax" not in name
        metrics = train_variant(
            name, model, cfg,
            train_loader, val_loader, test_loader,
            device, out_dir, use_ordinal=is_ordinal,
        )
        ablation_results[name] = metrics

    # ── Augmentation ablation ──────────────────────────────────
    # F — No augmentation
    logger.info("Ablation F: No augmentation")
    no_aug_cfg = OmegaConf.merge(cfg, {"augmentation": {"enabled": False}})
    from src.models.msda_net import MSDANet
    c = cfg.model.msda_net
    model_no_aug = MSDANet(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.input_channels,
        base_channels=c.base_channels,
        scales=list(c.scales),
        se_reduction=c.se_reduction,
        num_attn_heads=c.num_attn_heads,
        attn_dropout=c.attn_dropout,
        dropout=c.dropout,
        ordinal=True,
    )
    tl_na, vl_na, test_na = build_dataloaders(no_aug_cfg)
    ablation_results["No Augmentation"] = train_variant(
        "No Augmentation", model_no_aug, no_aug_cfg,
        tl_na, vl_na, test_na, device, out_dir,
    )

    # G — No SpecAugment
    logger.info("Ablation G: No SpecAugment")
    no_spec_cfg = OmegaConf.merge(cfg, {"augmentation": {"spec_augment": False}})
    model_no_spec = MSDANet(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.input_channels,
        base_channels=c.base_channels,
        scales=list(c.scales),
        se_reduction=c.se_reduction,
        num_attn_heads=c.num_attn_heads,
        attn_dropout=c.attn_dropout,
        dropout=c.dropout,
        ordinal=True,
    )
    tl_ns, vl_ns, test_ns = build_dataloaders(no_spec_cfg)
    ablation_results["No SpecAugment"] = train_variant(
        "No SpecAugment", model_no_spec, no_spec_cfg,
        tl_ns, vl_ns, test_ns, device, out_dir,
    )

    # ── Save results ──────────────────────────────────────────
    summary = {
        name: {
            k: float(v) if isinstance(v, (float, __import__("numpy").floating)) else
               v.tolist() if hasattr(v, "tolist") else v
            for k, v in metrics.items()
        }
        for name, metrics in ablation_results.items()
    }

    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print ablation table
    logger.info("\n" + "=" * 70)
    logger.info(f"{'Variant':<28} {'Accuracy':>10} {'Macro F1':>10} {'Kappa':>10} {'MCC':>10}")
    logger.info("-" * 70)
    for name, m in ablation_results.items():
        logger.info(
            f"{name:<28} {m.get('accuracy',0):>10.4f} "
            f"{m.get('macro_f1',0):>10.4f} "
            f"{m.get('cohen_kappa',0):>10.4f} "
            f"{m.get('mcc',0):>10.4f}"
        )
    logger.info("=" * 70)

    # Comparison figure
    plot_metric_comparison(
        ablation_results,
        metrics_to_plot=["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"],
        title="Ablation Study: Component Contributions",
        save_path=str(out_dir / "ablation_comparison.png"),
    )
    logger.info(f"Ablation results → {out_dir}")


if __name__ == "__main__":
    main()
