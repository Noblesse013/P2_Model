"""
Full evaluation pipeline: loads all trained models, runs the test set,
computes all metrics, bootstrap CIs, McNemar tests, and generates all
publication figures.

Usage:
    python evaluate.py --results_dir experiments/results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import evaluate_model, print_metrics_table
from src.evaluation.statistical_tests import (
    bootstrap_all_metrics,
    mcnemar_table,
    summarise_kfold,
)
from src.models.factory import build_model
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_roc_curves,
    plot_f1_radar,
)
from src.utils.logger import setup_logger


MODEL_NAMES = ["cnn_baseline", "deep_cnn", "resnet50", "efficientnet_b4", "msda_net"]


def load_model_from_ckpt(ckpt_path: str, cfg, device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/results", type=str)
    parser.add_argument("--config", default="configs/config.yaml", type=str)
    parser.add_argument("--n_bootstrap", default=1000, type=int)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    args = parser.parse_args()

    logger = setup_logger("evaluate")
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(cfg.data.class_names)

    _, _, test_loader = build_dataloaders(cfg)
    logger.info(f"Test set: {len(test_loader.dataset)} samples")

    all_metrics: dict = {}
    all_preds: dict = {}
    all_probs: dict = {}
    y_true_ref = None
    ci_data: dict = {}

    for model_name in args.models:
        run_name = f"{model_name}_{cfg.training.loss}"
        ckpt_path = results_dir / run_name / f"{run_name}_best.pt"

        if not ckpt_path.exists():
            # Fallback: look directly in results_dir
            ckpt_path = results_dir / f"{run_name}_best.pt"

        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found for '{model_name}': {ckpt_path}")
            continue

        logger.info(f"Evaluating: {model_name}")
        override_cfg = OmegaConf.merge(cfg, {"model": {"name": model_name}})

        try:
            model = load_model_from_ckpt(str(ckpt_path), override_cfg, device)
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

        metrics, y_true, y_pred, y_prob = evaluate_model(
            model, test_loader, device,
            ordinal=cfg.model.ordinal,
            num_classes=cfg.data.num_classes,
            class_names=class_names,
        )

        print_metrics_table(metrics, class_names)
        all_metrics[model_name] = metrics
        all_preds[model_name] = y_pred
        all_probs[model_name] = y_prob
        if y_true_ref is None:
            y_true_ref = y_true

        # Bootstrap CIs
        logger.info(f"  Computing bootstrap CIs (n={args.n_bootstrap})...")
        ci_data[model_name] = bootstrap_all_metrics(
            y_true, y_pred, y_prob,
            n_bootstrap=args.n_bootstrap,
            num_classes=cfg.data.num_classes,
            class_names=class_names,
        )

        # Per-model figures
        model_fig_dir = figures_dir / model_name
        model_fig_dir.mkdir(exist_ok=True)

        plot_confusion_matrix(
            metrics["confusion_matrix"], class_names,
            title=f"Confusion Matrix — {model_name}",
            save_path=str(model_fig_dir / "confusion_matrix.png"),
        )
        if y_prob is not None:
            plot_roc_curves(
                y_true, y_prob, class_names,
                title=f"ROC Curves — {model_name}",
                save_path=str(model_fig_dir / "roc_curves.png"),
            )

    if len(all_metrics) < 2:
        logger.warning("Need at least 2 models for comparative figures.")
    else:
        # ── McNemar pairwise tests ────────────────────────────
        logger.info("Running McNemar's pairwise significance tests...")
        mcnemar_results = mcnemar_table(y_true_ref, all_preds, alpha=cfg.evaluation.mcnemar_alpha)
        print("\nMcNemar's Test Results:")
        print(f"{'Model A':<20} {'Model B':<20} {'Chi2':>8} {'p-value':>10} {'Significant':>12}")
        print("-" * 72)
        for (ma, mb), res in mcnemar_results.items():
            sig = "*" if res["significant"] else ""
            print(f"{ma:<20} {mb:<20} {res['statistic']:>8.3f} {res['p_value']:>10.4f} {sig:>12}")

        # Save McNemar results
        mc_path = results_dir / "mcnemar_tests.json"
        with open(mc_path, "w") as f:
            json.dump({f"{a}|{b}": v for (a, b), v in mcnemar_results.items()}, f, indent=2)
        logger.info(f"McNemar results → {mc_path}")

        # ── Comparative figures ───────────────────────────────
        plot_metric_comparison(
            all_metrics,
            metrics_to_plot=["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "mcc"],
            title="Model Performance Comparison",
            save_path=str(figures_dir / "model_comparison.png"),
            ci_data={m: {k: (v[0], v[1], v[2]) for k, v in ci_data[m].items()}
                     for m in ci_data},
        )

        plot_f1_radar(
            all_metrics,
            class_names,
            title="Per-Class F1 — All Models",
            save_path=str(figures_dir / "f1_radar.png"),
        )

    # ── Save all metrics to JSON ──────────────────────────────
    summary = {}
    for model_name, metrics in all_metrics.items():
        summary[model_name] = {
            k: float(v) if isinstance(v, (float, np.floating)) else
               v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        # Add CI data
        if model_name in ci_data:
            summary[model_name]["bootstrap_ci"] = {
                k: {"mean": v[0], "lower_95ci": v[1], "upper_95ci": v[2]}
                for k, v in ci_data[model_name].items()
            }

    out_path = results_dir / "all_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"All metrics saved → {out_path}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
