"""
Dataset augmentation to improve realism and fairness of results.

Applies three techniques:
  1. Gaussian sensor noise  — simulates real OBD-2 sensor measurement error
  2. SMOTE oversampling     — balances Critical/Faulty minority classes
  3. Boundary softening     — adds label noise near threshold boundaries
                              to prevent the model from memorising hard cutoffs

Run after preprocess.py and before train.py.
Saves augmented X_train / y_train back to data/processed/.
"""

import os
import sys
import yaml
import numpy as np
import pickle
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: imbalanced-learn not installed. SMOTE disabled.")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Gaussian sensor noise
# ---------------------------------------------------------------------------

# Realistic noise levels per sensor type (as fraction of IQR).
# These approximate the measurement uncertainty of automotive-grade sensors.
SENSOR_NOISE_FRACTION = {
    "engine_rpm":        0.008,   # ±0.8% — crankshaft sensor (very accurate)
    "coolant_temp":      0.015,   # ±1.5% — NTC thermistor
    "engine_load":       0.020,   # ±2.0% — calculated value
    "map_pressure":      0.025,   # ±2.5% — pressure sensor
    "intake_air_temp":   0.015,   # ±1.5% — temperature sensor
    "throttle_position": 0.010,   # ±1.0% — TPS sensor
    "vehicle_speed":     0.005,   # ±0.5% — wheel speed sensor (very accurate)
    "catalyst_temp":     0.020,   # ±2.0% — thermocouple
}

DEFAULT_NOISE_FRACTION = 0.02   # fallback for unknown features


def add_sensor_noise(X_train, feature_names, seed=42):
    """
    Add per-feature Gaussian noise scaled to each feature's IQR.
    This simulates realistic sensor measurement uncertainty without
    distorting the overall feature distribution.

    X_train: (n, window_size, n_features)
    """
    rng = np.random.default_rng(seed)
    X_noisy = X_train.copy()

    for i, feat in enumerate(feature_names):
        # Compute IQR from the training data (already scaled)
        col_flat = X_train[:, :, i].ravel()
        q75, q25 = np.percentile(col_flat, [75, 25])
        iqr = max(q75 - q25, 1e-6)

        frac = SENSOR_NOISE_FRACTION.get(feat, DEFAULT_NOISE_FRACTION)
        sigma = frac * iqr

        noise = rng.normal(0, sigma, size=X_noisy[:, :, i].shape)
        X_noisy[:, :, i] += noise

    print(f"  Added Gaussian noise (per-feature, IQR-scaled)")
    return X_noisy


# ---------------------------------------------------------------------------
# 2. SMOTE oversampling on minority classes
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train, seed=42):
    """
    Oversample Critical and Faulty windows using SMOTE.
    Windows are flattened for SMOTE then reshaped back.
    Target: bring minority classes up to at least 15% of majority class count.
    """
    if not HAS_SMOTE:
        print("  SMOTE skipped (imbalanced-learn not installed)")
        return X_train, y_train

    counts = Counter(y_train)
    majority_n = max(counts.values())
    target_n = max(int(majority_n * 0.15), max(counts.values()))

    # Only oversample if minority classes are very sparse
    sampling_strategy = {}
    for cls, n in counts.items():
        if cls >= 2 and n < int(majority_n * 0.15):   # Faulty and Critical
            sampling_strategy[cls] = max(n * 3, int(majority_n * 0.12))

    if not sampling_strategy:
        print("  SMOTE skipped (class balance already acceptable)")
        return X_train, y_train

    original_shape = X_train.shape
    X_flat = X_train.reshape(len(X_train), -1)

    # SMOTE requires at least k+1 samples per class
    min_samples = min(counts[cls] for cls in sampling_strategy)
    k = min(5, min_samples - 1)
    if k < 1:
        print(f"  SMOTE skipped (too few minority samples: {min_samples})")
        return X_train, y_train

    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=seed)
    X_res, y_res = smote.fit_resample(X_flat, y_train)

    X_res = X_res.reshape(-1, original_shape[1], original_shape[2])
    new_counts = Counter(y_res)
    print(f"  SMOTE applied:")
    for cls in sorted(new_counts):
        old = counts.get(cls, 0)
        new = new_counts[cls]
        tag = f" (+{new-old})" if new > old else ""
        print(f"    Class {cls}: {old} -> {new}{tag}")

    return X_res.astype(np.float32), y_res.astype(np.int64)


# ---------------------------------------------------------------------------
# 3. Boundary softening (label noise near thresholds)
# ---------------------------------------------------------------------------

def soften_boundaries(X_train, y_train, noise_rate=0.03, seed=42):
    """
    Randomly flip a small fraction of boundary-region samples to an adjacent class.
    This prevents the model from learning hard threshold boundaries
    and forces it to model genuine uncertainty at class transitions.

    Only flips between adjacent classes (Normal↔Warning, Warning↔Faulty, Faulty↔Critical).
    noise_rate: fraction of samples to perturb (default 3%)
    """
    rng = np.random.default_rng(seed)
    y_soft = y_train.copy()
    n_flip = int(len(y_train) * noise_rate)

    if n_flip == 0:
        return X_train, y_soft

    flip_idx = rng.choice(len(y_train), n_flip, replace=False)
    for idx in flip_idx:
        cls = y_soft[idx]
        # Flip to an adjacent class only
        neighbours = [c for c in [cls - 1, cls + 1] if 0 <= c <= 3]
        if neighbours:
            y_soft[idx] = rng.choice(neighbours)

    n_changed = int((y_soft != y_train).sum())
    print(f"  Boundary softening: {n_changed} labels flipped ({noise_rate*100:.1f}% rate)")
    return X_train, y_soft


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path="config.yaml", noise=True, smote=True, soften=True):
    cfg         = load_config(config_path)
    out_dir     = cfg["data"]["processed_dir"]
    feat_names  = cfg["data"]["feature_columns"]
    label_names = cfg["labels"]["names"]

    print("=== Data Augmentation ===")

    X_train = np.load(os.path.join(out_dir, "X_train.npy"))
    y_train = np.load(os.path.join(out_dir, "y_train.npy"))

    print(f"\n  Before augmentation: {len(y_train)} windows")
    for i, name in enumerate(label_names):
        n = int((y_train == i).sum())
        print(f"    {name:8s}: {n:4d} ({100*n/len(y_train):.1f}%)")

    if noise:
        print("\n[1] Sensor noise augmentation...")
        X_train = add_sensor_noise(X_train, feat_names)

    if smote:
        print("\n[2] SMOTE minority oversampling...")
        X_train, y_train = apply_smote(X_train, y_train)

    if soften:
        print("\n[3] Boundary label softening...")
        X_train, y_train = soften_boundaries(X_train, y_train)

    print(f"\n  After augmentation: {len(y_train)} windows")
    for i, name in enumerate(label_names):
        n = int((y_train == i).sum())
        print(f"    {name:8s}: {n:4d} ({100*n/len(y_train):.1f}%)")

    # Save back — val and test are NOT augmented (must reflect real distribution)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    print("\n  Saved augmented X_train / y_train (val and test unchanged)")
    print("=== Augmentation complete ===")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(__file__))
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "config.yaml")
    run(config_path)
