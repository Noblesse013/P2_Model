"""
Analyse the prepared dataset: class distribution, audio duration,
signal statistics, and class imbalance check.

Usage:
    python data_preparation/dataset_stats.py
    python data_preparation/dataset_stats.py --data_dir data/
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRADE_NAMES = ["normal", "warning", "fault", "critical"]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def analyse_dataset(data_dir: str):
    root = Path(data_dir)
    stats = defaultdict(list)   # grade_name → list of durations

    for grade_name in GRADE_NAMES:
        cls_dir = root / grade_name
        if not cls_dir.exists():
            print(f"  WARNING: '{cls_dir}' not found")
            continue
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in AUDIO_EXTS:
                try:
                    info = sf.info(str(f))
                    stats[grade_name].append(info.duration)
                except Exception:
                    pass

    # ── Summary table ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DATASET STATISTICS")
    print("=" * 65)
    print(f"  {'Grade':<12} {'Files':>6} {'Total (s)':>12} {'Mean (s)':>10} {'Std (s)':>10}")
    print("  " + "-" * 52)

    total_files = 0
    all_durations = []
    for grade_name in GRADE_NAMES:
        durations = stats[grade_name]
        n = len(durations)
        total_files += n
        all_durations.extend(durations)
        if n > 0:
            print(
                f"  {grade_name:<12} {n:>6} {sum(durations):>12.1f}"
                f" {np.mean(durations):>10.2f} {np.std(durations):>10.2f}"
            )
        else:
            print(f"  {grade_name:<12} {'0':>6} {'—':>12}")

    print("  " + "-" * 52)
    print(f"  {'TOTAL':<12} {total_files:>6} {sum(all_durations):>12.1f}")
    print("=" * 65)

    # ── Class imbalance ratio ──────────────────────────────────
    counts = [len(stats[g]) for g in GRADE_NAMES]
    if min(counts) > 0:
        ratio = max(counts) / min(counts)
        if ratio > 3.0:
            print(f"\n  ⚠  Imbalance ratio {ratio:.1f}x — WeightedRandomSampler will compensate.")
        else:
            print(f"\n  ✓  Imbalance ratio {ratio:.1f}x — acceptable.")

    # ── Bar chart ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
    ax1.bar(GRADE_NAMES, counts, color=colors, edgecolor="black", linewidth=0.7)
    ax1.set_title("Sample Count per Health Grade")
    ax1.set_ylabel("Number of Audio Files")
    for i, v in enumerate(counts):
        ax1.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=10)

    total_durations = [sum(stats[g]) / 60 for g in GRADE_NAMES]
    ax2.bar(GRADE_NAMES, total_durations, color=colors, edgecolor="black", linewidth=0.7)
    ax2.set_title("Total Duration per Health Grade")
    ax2.set_ylabel("Duration (minutes)")

    plt.tight_layout()
    out_path = Path(data_dir) / "dataset_distribution.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n  Distribution chart → {out_path}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()
    analyse_dataset(args.data_dir)


if __name__ == "__main__":
    main()
