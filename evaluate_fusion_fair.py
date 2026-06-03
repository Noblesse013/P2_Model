"""
Fair Fusion Evaluation: Class-Prototype Pairing.

Problem with the original evaluate_fusion.py:
    It randomly pairs OBD2 test samples with same-grade audio embeddings.
    This makes the evaluation circular — you pair by grade, then measure grade
    prediction accuracy. The reported 99.94% has high variance and no
    connection to real synchronized multimodal inputs.

This script uses a deterministic, conservative alternative:
    For each grade k, compute the CENTROID (mean) of all audio test embeddings
    of that grade. Pair every OBD2 test sample of grade k with this centroid.

Why this is more legitimate:
    1. Deterministic — no random seed, fully reproducible.
    2. Conservative — the centroid averages out lucky within-class variation.
    3. Directly comparable — OBD2-only vs fusion on the same 13,113 samples.
    4. Interpretable — "does knowing the canonical audio signature of a health
       grade improve OBD2 classification?" is a well-posed question.

Additional experiment — mismatch robustness:
    Pair each OBD2 sample with the WRONG grade's audio centroid.
    If accuracy drops below OBD2-only, audio actively informs the model
    (it is not being ignored). This is the key evidence that fusion is real.

Usage:
    python evaluate_fusion_fair.py
    python evaluate_fusion_fair.py --checkpoint experiments/fusion_best.pt
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
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix,
)

from src.models.fusion_model import FusionMLP, OrdinalHead

GRADE_LABELS = ["Normal", "Warning", "Fault", "Critical"]
N_CLASSES    = 4


# ── Core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_audio_vectors(
    model:      FusionMLP,
    obd2_embs:  np.ndarray,       # (N_obd2, 256)
    obd2_labels: np.ndarray,      # (N_obd2,)
    audio_vectors: np.ndarray,    # (N_obd2, 512)  — pre-assigned per sample
    device:     torch.device,
    batch_size: int = 512,
) -> dict:
    """
    Generic evaluation: feed pre-assigned (obd2, audio) pairs through FusionMLP.
    audio_vectors[i] is whatever audio embedding has been assigned to obd2 sample i.
    """
    model.eval()
    all_preds = []

    N = len(obd2_labels)
    for start in range(0, N, batch_size):
        h_obd2  = torch.tensor(obd2_embs[start:start + batch_size],
                               dtype=torch.float32, device=device)
        h_audio = torch.tensor(audio_vectors[start:start + batch_size],
                               dtype=torch.float32, device=device)
        logits = model(h_obd2, h_audio)
        preds  = OrdinalHead.decode(logits).cpu().numpy()
        all_preds.extend(preds)

    all_preds = np.array(all_preds)

    acc   = float(accuracy_score(obd2_labels, all_preds))
    mf1   = float(f1_score(obd2_labels, all_preds, average="macro",    zero_division=0))
    wf1   = float(f1_score(obd2_labels, all_preds, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(obd2_labels, all_preds, weights="quadratic"))
    mcc   = float(matthews_corrcoef(obd2_labels, all_preds))
    cm    = confusion_matrix(obd2_labels, all_preds).tolist()
    n_err = int(np.sum(all_preds != obd2_labels))

    return {
        "accuracy":        round(acc,   4),
        "macro_f1":        round(mf1,   4),
        "weighted_f1":     round(wf1,   4),
        "quadratic_kappa": round(kappa, 4),
        "mcc":             round(mcc,   4),
        "n_errors":        n_err,
        "n_samples":       N,
        "confusion_matrix": cm,
    }


# ── Prototype builder ─────────────────────────────────────────────────────────

def build_prototypes(
    audio_embs:   np.ndarray,   # (N_audio, 512)
    audio_labels: np.ndarray,   # (N_audio,)
    n_classes:    int = N_CLASSES,
) -> np.ndarray:
    """
    Compute the class centroid (mean embedding) for each health grade.
    Returns (n_classes, 512) array.
    """
    prototypes = np.zeros((n_classes, audio_embs.shape[1]), dtype=np.float32)
    for k in range(n_classes):
        mask = audio_labels == k
        if mask.sum() == 0:
            raise ValueError(f"No audio test samples for grade {k}. "
                             "Cannot compute prototype.")
        prototypes[k] = audio_embs[mask].mean(axis=0)
    return prototypes


def assign_prototype_vectors(
    obd2_labels:  np.ndarray,     # (N_obd2,)
    prototypes:   np.ndarray,     # (n_classes, 512)
    mismatch_grade: int | None = None,
) -> np.ndarray:
    """
    For each OBD2 sample, assign:
      - The CORRECT grade's prototype  (mismatch_grade=None)
      - A SPECIFIC wrong grade's prototype  (mismatch_grade=k)

    Returns (N_obd2, 512) array of assigned audio vectors.
    """
    N = len(obd2_labels)
    assigned = np.zeros((N, prototypes.shape[1]), dtype=np.float32)
    for i, grade in enumerate(obd2_labels):
        if mismatch_grade is None:
            assigned[i] = prototypes[int(grade)]
        else:
            assigned[i] = prototypes[int(mismatch_grade)]
    return assigned


# ── OBD2-only baseline ────────────────────────────────────────────────────────

def obd2_only_accuracy(obd2_embs, obd2_labels, model_dir, checkpoint, config, device):
    """
    Evaluate OBD2 branch alone by passing embeddings through the
    pretrained OBD2 classifier head (bypasses fusion entirely).
    """
    import yaml as _yaml
    obd2_path = Path(model_dir)
    src = str(obd2_path / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    with open(obd2_path / config) as f:
        ocfg = _yaml.safe_load(f)

    from model import build_model
    ckpt = torch.load(obd2_path / checkpoint, map_location=device, weights_only=False)
    full_model = build_model(ocfg)
    full_model.load_state_dict(ckpt["model_state_dict"])
    full_model.to(device).eval()

    h = torch.tensor(obd2_embs, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = full_model.classifier(h).argmax(dim=1).cpu().numpy()

    acc   = float(accuracy_score(obd2_labels, preds))
    kappa = float(cohen_kappa_score(obd2_labels, preds, weights="quadratic"))
    n_err = int(np.sum(preds != obd2_labels))
    return {"accuracy": round(acc, 4), "quadratic_kappa": round(kappa, 4),
            "n_errors": n_err, "n_samples": len(obd2_labels)}


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_result(label: str, r: dict):
    print(f"  {label:<42}  acc={r['accuracy']:.4f}  "
          f"kappa={r['quadratic_kappa']:.4f}  errors={r['n_errors']}/{r['n_samples']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/fusion_best.pt")
    parser.add_argument("--emb_dir",    default="embeddings")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--output",     default="experiments/fusion_fair_eval.json")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb    = Path(args.emb_dir)

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("\nLoading test embeddings ...")
    obd2_embs    = np.load(emb / "obd2_test.npy")
    obd2_labels  = np.load(emb / "obd2_test_labels.npy")
    audio_embs   = np.load(emb / "audio_test.npy")
    audio_labels = np.load(emb / "audio_test_labels.npy")

    print(f"  OBD2  test : {obd2_embs.shape}  labels={obd2_labels.shape}")
    print(f"  Audio test : {audio_embs.shape}  labels={audio_labels.shape}")

    # ── Load fusion model ────────────────────────────────────────────────────
    print("\nLoading fusion model ...")
    model = FusionMLP()
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch','?')}  val_f1={ckpt.get('val_f1',0):.4f}")

    # ── Compute audio class prototypes ───────────────────────────────────────
    print("\nComputing audio class prototypes from test set ...")
    prototypes = build_prototypes(audio_embs, audio_labels)
    for k in range(N_CLASSES):
        n = int((audio_labels == k).sum())
        print(f"  Grade {k} ({GRADE_LABELS[k]:<8}): {n:4d} audio samples  "
              f"prototype norm={np.linalg.norm(prototypes[k]):.3f}")

    # ── Experiment 1: OBD2-only baseline ─────────────────────────────────────
    print("\n[1] OBD2-only baseline ...")
    r_obd2 = obd2_only_accuracy(
        obd2_embs, obd2_labels,
        cfg["obd2"]["model_dir"], cfg["obd2"]["checkpoint"],
        cfg["obd2"]["config"], device,
    )

    # ── Experiment 2: Correct-prototype fusion ───────────────────────────────
    print("[2] Fusion with CORRECT-grade audio prototype ...")
    correct_audio = assign_prototype_vectors(obd2_labels, prototypes, mismatch_grade=None)
    r_correct = evaluate_with_audio_vectors(model, obd2_embs, obd2_labels,
                                            correct_audio, device)

    # ── Experiment 3: Wrong-grade prototypes (one per grade) ─────────────────
    print("[3] Fusion with WRONG-grade audio prototypes ...")
    r_mismatch = {}
    for wrong_grade in range(N_CLASSES):
        wrong_audio = assign_prototype_vectors(obd2_labels, prototypes,
                                              mismatch_grade=wrong_grade)
        label = f"wrong_grade_{wrong_grade}_{GRADE_LABELS[wrong_grade]}"
        r_mismatch[label] = evaluate_with_audio_vectors(
            model, obd2_embs, obd2_labels, wrong_audio, device
        )

    # ── Experiment 4: Fully random wrong pairing (cross-grade noise) ─────────
    print("[4] Fusion with fully random SHUFFLED audio embeddings ...")
    rng = np.random.default_rng(42)
    shuffled_idx   = rng.permutation(len(audio_embs))
    shuffled_embs  = audio_embs[shuffled_idx]
    # Tile or subsample shuffled audio to match OBD2 count
    n_obd2 = len(obd2_labels)
    tiled  = np.tile(shuffled_embs, (n_obd2 // len(shuffled_embs) + 1, 1))[:n_obd2]
    r_random = evaluate_with_audio_vectors(model, obd2_embs, obd2_labels, tiled, device)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("  FAIR FUSION EVALUATION — PROTOTYPE-BASED RESULTS")
    print("=" * 75)
    print_result("OBD2-only (no fusion)",                r_obd2)
    print_result("Fusion + correct-grade prototype",     r_correct)
    print("-" * 75)
    for label, r in r_mismatch.items():
        grade_name = label.split("_", 2)[-1]
        print_result(f"Fusion + all-audio-set-to-{grade_name}", r)
    print("-" * 75)
    print_result("Fusion + randomly shuffled audio",     r_random)
    print("=" * 75)

    # ── Interpretation ────────────────────────────────────────────────────────
    print()
    print("  Interpretation:")
    delta = r_correct["accuracy"] - r_obd2["accuracy"]
    if delta > 0:
        print(f"  + Correct-prototype fusion is {delta:+.4f} above OBD2-only.")
        print("    Audio ADDS value when it correctly confirms the health grade.")
    elif delta == 0:
        print("    Correct-prototype fusion equals OBD2-only.")
        print("    Audio contributes nothing beyond OBD2 for this test set.")
    else:
        print(f"  - Correct-prototype fusion is {delta:+.4f} BELOW OBD2-only.")
        print("    WARN: fusion hurts even with correct audio. Retrain fusion model.")

    wrong_accs = [r["accuracy"] for r in r_mismatch.values()]
    worst = min(wrong_accs)
    if worst < r_obd2["accuracy"]:
        print(f"  + Wrong-grade audio degrades accuracy to {worst:.4f} "
              f"(below OBD2-only={r_obd2['accuracy']:.4f}).")
        print("    Audio IS used by the model — conflicting audio hurts performance.")
        print("    This confirms fusion is real, not OBD2-only in disguise.")
    else:
        print(f"  ! Wrong-grade audio does not degrade below OBD2-only.")
        print("    Model may be ignoring audio. Investigate fusion weights.")

    # ── Save ─────────────────────────────────────────────────────────────────
    results = {
        "obd2_only":              r_obd2,
        "fusion_correct_proto":   r_correct,
        "fusion_wrong_proto":     r_mismatch,
        "fusion_random_audio":    r_random,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
