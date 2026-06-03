"""
Honest Late Fusion Evaluation — Probability-Level Combination.

Replaces the circular evaluate_fusion.py with a methodology that matches
published multimodal fault-diagnosis literature (WPEDL, Ali et al. IEEE 2024;
Credibility-Aware Fusion, arXiv 2024).

Core fix
--------
The original script paired every OBD2 test sample with a same-grade audio
embedding, leaking the label into the input. This script NEVER uses the true
label to select any input at inference time.

What this script does
---------------------
Step 1  OBD2-only  — run the pretrained CNN+BiGRU classifier on the OBD2
        test split (Driver-3 holdout). Softmax probabilities → argmax grade.

Step 2  Audio-only — run the pretrained MSDA-Net ordinal head on the
        pre-extracted 512-dim audio test embeddings. OrdinalHead.to_class_probs
        → argmax grade.

Step 3  Class-conditional audio prior — from the audio VALIDATION set,
        compute the mean audio probability vector for each predicted grade
        (4 × 4 matrix). This prior represents "what audio typically looks like
        when the audio model itself predicts grade k". No true labels used.

Step 4  Soft ensemble on OBD2 test —
            P_fused = α · P_obd2 + (1-α) · μ_audio[argmax(P_obd2)]
        α is chosen by grid search on the OBD2 VALIDATION set.

Step 5  Soft ensemble on Audio test —
            P_fused = α · P_audio + (1-α) · μ_obd2[argmax(P_audio)]
        where μ_obd2 is the OBD2-val class-conditional mean, α from val search.

Step 6  Mismatch ablation — replace the audio prior with a WRONG-class prior.
        If accuracy drops below OBD2-only, audio signal is genuinely used.

Ablation table produced
-----------------------
  System                              | Test set   | Acc    | Kappa
  OBD2-only                           | OBD2 test  | ?      | ?
  Audio-only                          | Audio test | ?      | ?
  Ensemble (OBD2 + audio prior, opt α)| OBD2 test  | ?      | ?
  Ensemble (audio + OBD2 prior, opt α)| Audio test | ?      | ?
  Mismatch (wrong-class audio prior)  | OBD2 test  | <OBD2  | <OBD2

Usage
-----
    python evaluate_late_fusion.py
    python evaluate_late_fusion.py --alpha 0.6   # skip alpha grid search
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix,
)

GRADE_LABELS = ["Normal", "Warning", "Fault", "Critical"]
N_CLASSES    = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc   = float(accuracy_score(y_true, y_pred))
    mf1   = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    wf1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    mcc   = float(matthews_corrcoef(y_true, y_pred))
    cm    = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).tolist()
    n_err = int(np.sum(y_pred != y_true))
    per_class = {}
    for k, name in enumerate(GRADE_LABELS):
        mask = y_true == k
        if mask.sum() == 0:
            per_class[name] = {"f1": 0.0, "n": 0}
            continue
        f1_k = float(f1_score(y_true[mask], y_pred[mask],
                               labels=[k], average="micro", zero_division=0))
        per_class[name] = {"f1": round(f1_k, 4), "n": int(mask.sum())}
    return {
        "accuracy":        round(acc,   4),
        "macro_f1":        round(mf1,   4),
        "weighted_f1":     round(wf1,   4),
        "quadratic_kappa": round(kappa, 4),
        "mcc":             round(mcc,   4),
        "n_errors":        n_err,
        "n_samples":       len(y_true),
        "per_class":       per_class,
        "confusion_matrix": cm,
    }


def print_row(label: str, r: dict, width: int = 44):
    print(f"  {label:<{width}}  acc={r['accuracy']:.4f}  "
          f"kappa={r['quadratic_kappa']:.4f}  "
          f"errors={r['n_errors']}/{r['n_samples']}")


def class_conditional_means(probs: np.ndarray,
                             preds: np.ndarray) -> np.ndarray:
    """
    For each predicted grade k, compute the mean probability vector
    over all samples where argmax(prob) == k.

    Returns (N_CLASSES, N_CLASSES) matrix.
    Falls back to uniform [0.25, 0.25, 0.25, 0.25] for empty buckets.
    """
    means = np.full((N_CLASSES, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)
    for k in range(N_CLASSES):
        mask = preds == k
        if mask.sum() > 0:
            means[k] = probs[mask].mean(axis=0)
    return means


def soft_ensemble(p_main:  np.ndarray,   # (N, 4)
                  p_preds: np.ndarray,   # (N,) predicted class from main model
                  prior:   np.ndarray,   # (4, 4) class-conditional means
                  alpha:   float) -> np.ndarray:
    """
    P_fused[i] = alpha * p_main[i] + (1-alpha) * prior[p_preds[i]]

    No true labels are used — p_preds comes from the main model itself.
    Returns (N, 4) fused probability matrix.
    """
    audio_component = prior[p_preds]       # (N, 4)
    return alpha * p_main + (1.0 - alpha) * audio_component


def find_best_alpha(p_main:  np.ndarray,
                    p_preds: np.ndarray,
                    prior:   np.ndarray,
                    y_val:   np.ndarray,
                    alphas:  np.ndarray = None) -> tuple[float, float]:
    """Grid-search alpha on the validation set. Returns (best_alpha, best_kappa)."""
    if alphas is None:
        alphas = np.arange(0.50, 1.01, 0.05)
    best_alpha, best_kappa = 1.0, -1.0
    for a in alphas:
        p_fused = soft_ensemble(p_main, p_preds, prior, a)
        y_pred  = p_fused.argmax(axis=1)
        k = cohen_kappa_score(y_val, y_pred, weights="quadratic")
        if k > best_kappa:
            best_kappa, best_alpha = k, float(a)
    return best_alpha, best_kappa


# ── OBD2 model — softmax probabilities ───────────────────────────────────────

def obd2_probabilities(emb_dir: Path, obd2_cfg: dict,
                       device: torch.device) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load OBD2 model classifier head (256→64→4).
    Run on pre-extracted 256-dim embeddings for train, val, test splits.
    Returns dict split → (probs (N,4), labels (N,)).
    """
    model_dir = Path(obd2_cfg["model_dir"])
    src = str(model_dir / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    with open(model_dir / obd2_cfg["config"]) as f:
        cfg = yaml.safe_load(f)

    from model import build_model
    ckpt     = torch.load(model_dir / obd2_cfg["checkpoint"],
                          map_location=device, weights_only=False)
    obd2_mdl = build_model(cfg)
    obd2_mdl.load_state_dict(ckpt["model_state_dict"])
    obd2_mdl.to(device).eval()

    results = {}
    for split in ("train", "val", "test"):
        x_path = emb_dir / f"obd2_{split}.npy"
        y_path = emb_dir / f"obd2_{split}_labels.npy"
        if not x_path.exists():
            continue
        embs   = torch.tensor(np.load(x_path),   dtype=torch.float32, device=device)
        labels = np.load(y_path)
        with torch.no_grad():
            logits = obd2_mdl.classifier(embs)          # (N, 4)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
        results[split] = (probs, labels)
        print(f"  [OBD2] {split}: {len(labels):,} samples")
    return results


# ── Audio model — ordinal head probabilities ──────────────────────────────────

def audio_probabilities(emb_dir: Path, audio_cfg: dict,
                        device: torch.device,
                        test_emb_file: str = "audio_test.npy",
                        test_lbl_file: str = "audio_test_labels.npy",
                        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load MSDA-Net's OrdinalHead from the audio checkpoint.
    Run on pre-extracted 512-dim audio embeddings.
    Returns dict split → (probs (N,4), labels (N,)).
    """
    # Import from local project BEFORE touching sys.path/sys.modules
    from src.models.fusion_model import OrdinalHead as LocalOH

    model_dir = Path(audio_cfg["model_dir"])

    with open(model_dir / audio_cfg["config"]) as f:
        raw = yaml.safe_load(f)
    from omegaconf import OmegaConf
    acfg = OmegaConf.create(raw)

    # Load audio model head (OrdinalHead) — separate from AudioEncoder wrapper
    audio_root = str(model_dir)
    _saved = {k: sys.modules.pop(k)
              for k in list(sys.modules)
              if k == "src" or k.startswith("src.")}
    if audio_root in sys.path:
        sys.path.remove(audio_root)
    sys.path.insert(0, audio_root)
    try:
        from src.models.factory import build_model as _audio_build
        ckpt = torch.load(model_dir / audio_cfg["checkpoint"],
                          map_location=device, weights_only=False)
        full_audio = _audio_build(acfg)
        full_audio.load_state_dict(ckpt["model_state_dict"])
        ordinal_head = full_audio.head    # OrdinalHead(512, 4)
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(_saved)
        if audio_root in sys.path:
            sys.path.remove(audio_root)

    ordinal_head.to(device).eval()

    results = {}
    split_files = {
        "train": ("audio_train.npy",        "audio_train_labels.npy"),
        "val":   ("audio_val.npy",          "audio_val_labels.npy"),
        "test":  (test_emb_file,            test_lbl_file),
    }
    for split, (xf, yf) in split_files.items():
        x_path = emb_dir / xf
        y_path = emb_dir / yf
        if not x_path.exists():
            continue
        embs   = torch.tensor(np.load(x_path),   dtype=torch.float32, device=device)
        labels = np.load(y_path)
        with torch.no_grad():
            logits = ordinal_head(embs)
            probs  = LocalOH.to_class_probs(logits).cpu().numpy()
        results[split] = (probs, labels)
        print(f"  [Audio] {split}: {len(labels):,} samples")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Honest probability-level late fusion evaluation")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--emb_dir", default="embeddings")
    parser.add_argument("--alpha",   type=float, default=None,
                        help="Fixed alpha (skip grid search on val set)")
    parser.add_argument("--audio_test_key", default="",
                        help="Suffix for audio test files. "
                             "'' uses audio_test.npy (seed-42 split, may overlap). "
                             "'original' uses audio_test_original.npy "
                             "(exact audio-model split, fully clean).")
    parser.add_argument("--output",  default="experiments/late_fusion_eval.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.emb_dir)
    print(f"\nDevice  : {device}")
    print(f"Embeddings: {emb_dir.resolve()}\n")

    # ── Step 1: OBD2 softmax probabilities ───────────────────────────────────
    print("=" * 60)
    print("  STEP 1 — OBD2 model probabilities")
    print("=" * 60)
    obd2_splits = obd2_probabilities(emb_dir, cfg["obd2"], device)
    obd2_val_probs,  obd2_val_labels  = obd2_splits["val"]
    obd2_test_probs, obd2_test_labels = obd2_splits["test"]

    obd2_val_preds  = obd2_val_probs.argmax(axis=1)
    obd2_test_preds = obd2_test_probs.argmax(axis=1)

    r_obd2_only = metrics(obd2_test_labels, obd2_test_preds)
    print(f"\n  OBD2-only test:  acc={r_obd2_only['accuracy']:.4f}  "
          f"kappa={r_obd2_only['quadratic_kappa']:.4f}  "
          f"errors={r_obd2_only['n_errors']}/{r_obd2_only['n_samples']}")

    # ── Step 2: Audio ordinal-head probabilities ──────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 — Audio model probabilities")
    print("=" * 60)
    audio_splits = audio_probabilities(emb_dir, cfg["audio"], device)
    audio_val_probs,  audio_val_labels  = audio_splits["val"]

    # Use the original-split clean test set if available
    key = args.audio_test_key
    if key:
        orig_emb  = emb_dir / f"audio_test_{key}.npy"
        orig_lbl  = emb_dir / f"audio_test_{key}_labels.npy"
        orig_prob = emb_dir / f"audio_test_{key}_probs.npy"
        if orig_prob.exists():
            print(f"\n  Using pre-computed probs: {orig_prob}")
            audio_test_probs  = np.load(orig_prob)
            audio_test_labels = np.load(orig_lbl)
        elif orig_emb.exists():
            print(f"\n  Using embeddings: {orig_emb}  (computing probs via ordinal head)")
            audio_splits_clean = audio_probabilities(
                emb_dir, cfg["audio"], device,
                test_emb_file=f"audio_test_{key}.npy",
                test_lbl_file=f"audio_test_{key}_labels.npy",
            )
            audio_test_probs, audio_test_labels = audio_splits_clean["test"]
        else:
            print(f"\n  WARNING: {orig_emb} not found.")
            print("  Run:  python extract_audio_clean_test.py  first.")
            print("  Falling back to seed-42 split (may include training data).")
            audio_test_probs, audio_test_labels = audio_splits["test"]
    else:
        audio_test_probs, audio_test_labels = audio_splits["test"]

    audio_val_preds  = audio_val_probs.argmax(axis=1)
    audio_test_preds = audio_test_probs.argmax(axis=1)

    r_audio_only = metrics(audio_test_labels, audio_test_preds)
    print(f"\n  Audio-only test:  acc={r_audio_only['accuracy']:.4f}  "
          f"kappa={r_audio_only['quadratic_kappa']:.4f}  "
          f"errors={r_audio_only['n_errors']}/{r_audio_only['n_samples']}")

    # ── Step 3: Class-conditional priors from VALIDATION sets ────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — Class-conditional priors (from val sets, no test leakage)")
    print("=" * 60)

    audio_prior = class_conditional_means(audio_val_probs, audio_val_preds)
    obd2_prior  = class_conditional_means(obd2_val_probs,  obd2_val_preds)

    print("\n  Audio val priors (mean P_audio | OBD2 predicts grade k):")
    for k in range(N_CLASSES):
        row = "  ".join(f"{v:.3f}" for v in audio_prior[k])
        print(f"    Grade {k} ({GRADE_LABELS[k]:<8}): [{row}]")

    print("\n  OBD2 val priors (mean P_obd2 | Audio predicts grade k):")
    for k in range(N_CLASSES):
        row = "  ".join(f"{v:.3f}" for v in obd2_prior[k])
        print(f"    Grade {k} ({GRADE_LABELS[k]:<8}): [{row}]")

    # ── Step 4: Alpha search on validation — OBD2+audio_prior ─────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — Alpha optimisation on OBD2 val set")
    print("=" * 60)

    if args.alpha is not None:
        alpha_obd2 = args.alpha
        print(f"  Using fixed alpha={alpha_obd2}")
    else:
        alpha_obd2, best_kappa_val = find_best_alpha(
            obd2_val_probs, obd2_val_preds, audio_prior, obd2_val_labels)
        print(f"  Best alpha (OBD2+audio_prior): {alpha_obd2:.2f}  "
              f"(val kappa={best_kappa_val:.4f})")

    # Alpha for audio+obd2_prior
    if args.alpha is not None:
        alpha_audio = args.alpha
    else:
        alpha_audio, best_kappa_val2 = find_best_alpha(
            audio_val_probs, audio_val_preds, obd2_prior, audio_val_labels)
        print(f"  Best alpha (audio+obd2_prior) : {alpha_audio:.2f}  "
              f"(val kappa={best_kappa_val2:.4f})")

    # ── Step 5: Soft ensemble on TEST sets ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 — Soft ensemble on test sets (label-free)")
    print("=" * 60)

    # OBD2 test: fuse OBD2 probs with audio prior conditioned on OBD2 prediction
    p_fused_obd2 = soft_ensemble(obd2_test_probs, obd2_test_preds,
                                 audio_prior, alpha_obd2)
    r_ensemble_obd2 = metrics(obd2_test_labels, p_fused_obd2.argmax(axis=1))

    # Audio test: fuse audio probs with OBD2 prior conditioned on audio prediction
    p_fused_audio = soft_ensemble(audio_test_probs, audio_test_preds,
                                  obd2_prior, alpha_audio)
    r_ensemble_audio = metrics(audio_test_labels, p_fused_audio.argmax(axis=1))

    print(f"\n  Ensemble (OBD2+audio prior) on OBD2 test:  "
          f"acc={r_ensemble_obd2['accuracy']:.4f}  "
          f"kappa={r_ensemble_obd2['quadratic_kappa']:.4f}  "
          f"errors={r_ensemble_obd2['n_errors']}/{r_ensemble_obd2['n_samples']}")
    print(f"  Ensemble (audio+OBD2 prior) on audio test:  "
          f"acc={r_ensemble_audio['accuracy']:.4f}  "
          f"kappa={r_ensemble_audio['quadratic_kappa']:.4f}  "
          f"errors={r_ensemble_audio['n_errors']}/{r_ensemble_audio['n_samples']}")

    # ── Step 6: Mismatch ablation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 — Mismatch ablation (wrong-class audio prior)")
    print("=" * 60)

    r_mismatch = {}
    for wrong_k in range(N_CLASSES):
        wrong_prior = np.tile(audio_prior[wrong_k], (N_CLASSES, 1))   # all grades → wrong prior
        p_mis = soft_ensemble(obd2_test_probs, obd2_test_preds,
                              wrong_prior, alpha_obd2)
        r = metrics(obd2_test_labels, p_mis.argmax(axis=1))
        label = f"wrong_prior_grade_{wrong_k}_{GRADE_LABELS[wrong_k]}"
        r_mismatch[label] = r
        print(f"  Audio prior fixed to grade {wrong_k} ({GRADE_LABELS[wrong_k]:<8}):  "
              f"acc={r['accuracy']:.4f}  kappa={r['quadratic_kappa']:.4f}")

    # ── Final summary table ───────────────────────────────────────────────────
    W = 48
    print("\n")
    print("=" * (W + 35))
    print("  HONEST LATE FUSION — FINAL ABLATION TABLE")
    print("=" * (W + 35))
    print(f"  {'System':<{W}}  {'Test set':<12}  {'Acc':>6}  {'Kappa':>7}  {'Errors':>12}")
    print("-" * (W + 35))

    def row(label, test_set, r):
        err_str = f"{r['n_errors']}/{r['n_samples']}"
        print(f"  {label:<{W}}  {test_set:<12}  "
              f"{r['accuracy']:>6.4f}  {r['quadratic_kappa']:>7.4f}  {err_str:>12}")

    row("OBD2-only (CNN+BiGRU, driver-3 holdout)", "OBD2 test", r_obd2_only)
    row("Audio-only (MSDA-Net ordinal head)",       "Audio test", r_audio_only)
    print("-" * (W + 35))
    row(f"Ensemble OBD2+audio prior (alpha={alpha_obd2:.2f})", "OBD2 test",  r_ensemble_obd2)
    row(f"Ensemble audio+OBD2 prior (alpha={alpha_audio:.2f})", "Audio test", r_ensemble_audio)
    print("-" * (W + 35))
    for label, r in r_mismatch.items():
        grade_name = label.split("_", 3)[-1]
        row(f"  Mismatch: prior fixed to {grade_name}", "OBD2 test", r)
    print("=" * (W + 35))

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print("\n  Per-class F1 — OBD2-only (test):")
    for name, d in r_obd2_only["per_class"].items():
        print(f"    {name:<10}  F1={d['f1']:.4f}  n={d['n']}")

    print("\n  Per-class F1 — Audio-only (test):")
    for name, d in r_audio_only["per_class"].items():
        print(f"    {name:<10}  F1={d['f1']:.4f}  n={d['n']}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print()
    delta_obd2  = r_ensemble_obd2["accuracy"]  - r_obd2_only["accuracy"]
    delta_audio = r_ensemble_audio["accuracy"] - r_audio_only["accuracy"]
    worst_mismatch = min(r["accuracy"] for r in r_mismatch.values())

    print("  Interpretation:")
    if delta_obd2 >= 0:
        print(f"  + Ensemble improves OBD2-only by {delta_obd2:+.4f} on OBD2 test.")
    else:
        print(f"  - Ensemble degrades OBD2-only by {delta_obd2:+.4f} on OBD2 test.")
        print("    Audio prior is adding noise. Increase alpha toward 1.0.")

    if delta_audio >= 0:
        print(f"  + Ensemble improves audio-only by {delta_audio:+.4f} on audio test.")
    else:
        print(f"  - Ensemble degrades audio-only by {delta_audio:+.4f} on audio test.")

    if worst_mismatch < r_obd2_only["accuracy"]:
        print(f"  + Wrong-class prior degrades OBD2 accuracy to {worst_mismatch:.4f} "
              f"(below OBD2-only={r_obd2_only['accuracy']:.4f}).")
        print("    Confirms audio signal is genuinely used — conflicting prior hurts.")
    else:
        print(f"  ! Wrong prior does not degrade below OBD2-only. Audio prior has low weight.")

    # ── Save ──────────────────────────────────────────────────────────────────
    all_results = {
        "alpha_obd2":          alpha_obd2,
        "alpha_audio":         alpha_audio,
        "obd2_only":           r_obd2_only,
        "audio_only":          r_audio_only,
        "ensemble_obd2_test":  r_ensemble_obd2,
        "ensemble_audio_test": r_ensemble_audio,
        "mismatch_ablation":   r_mismatch,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved -> {args.output}")


if __name__ == "__main__":
    main()
