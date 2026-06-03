"""
Cross-dataset validation on the KIT OBD-II Dataset (Seat Leon, Germany, 2017-2018).

The trained CNN+BiGRU+Attention model (trained on the OBD-II Classified dataset
from a different vehicle/country) is applied to the KIT dataset WITHOUT retraining.

Since all KIT recordings are from a healthy vehicle, the model should predominantly
predict Normal (class 0). The proportion of Normal predictions is the key metric.

This is a zero-shot domain-transfer test:
  - Different vehicle   : Seat Leon (turbocharged petrol) vs training vehicle
  - Different country   : Germany vs training vehicle's country
  - Different climate   : German winter/summer vs training climate
  - No retraining       : same weights, same scaler

Feature mapping:
  6 of 8 training features are available in the KIT dataset.
  2 missing features (engine_load, catalyst_temp) are imputed with their
  training-set median (= 0 in scaled space after RobustScaler).
"""

import os
import sys
import glob
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import build_model


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# KIT column name -> our standard feature name
KIT_COLUMN_MAP = {
    "Engine RPM [RPM]":                      "engine_rpm",
    "Engine Coolant Temperature [°C]":       "coolant_temp",
    "Engine Coolant Temperature [\xb0C]":    "coolant_temp",   # encoding variant
    "Intake Manifold Absolute Pressure [kPa]": "map_pressure",
    "Intake Air Temperature [°C]":           "intake_air_temp",
    "Intake Air Temperature [\xb0C]":        "intake_air_temp",
    "Absolute Throttle Position [%]":        "throttle_position",
    "Vehicle Speed Sensor [km/h]":           "vehicle_speed",
}

# Features available in KIT (subset of our 8 training features)
KIT_AVAILABLE = ["engine_rpm", "coolant_temp", "map_pressure",
                 "intake_air_temp", "throttle_position", "vehicle_speed"]

# Features missing from KIT — will be filled with training median (scaled = 0)
KIT_MISSING   = ["engine_load", "catalyst_temp"]


def load_kit_data(kit_dir, feat_cols):
    """Load and merge all KIT CSVs, map to standard feature names."""
    files = sorted(glob.glob(os.path.join(kit_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {kit_dir}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        renamed = {}
        for col in df.columns:
            std = KIT_COLUMN_MAP.get(col) or KIT_COLUMN_MAP.get(
                col.encode("latin-1").decode("utf-8", errors="replace"), None)
            if std:
                renamed[col] = std
        df = df.rename(columns=renamed)
        frames.append(df)

    kit = pd.concat(frames, ignore_index=True)

    # Keep only the available features we need
    available = [f for f in KIT_AVAILABLE if f in kit.columns]
    kit = kit[available].copy()

    # Drop rows where all available features are NaN
    kit = kit.dropna(how="all").reset_index(drop=True)

    # Fill remaining per-column NaN with column median (sensor drop-outs)
    for col in kit.columns:
        kit[col] = kit[col].fillna(kit[col].median())

    print(f"  KIT rows after cleaning: {len(kit):,}")
    print(f"  Available features: {available}")
    print(f"  Imputed with training median (scaled=0): {KIT_MISSING}")
    return kit, available


def make_windows(features, window_size, stride):
    """Simple temporal windowing — no labels needed (all KIT is Normal)."""
    X = []
    n = len(features)
    for start in range(0, n - window_size + 1, stride):
        X.append(features[start: start + window_size])
    return np.array(X, dtype=np.float32)


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    all_preds, all_probs = [], []
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i: i + batch_size]).to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs)
    return np.concatenate(all_preds), np.concatenate(all_probs)


def plot_prediction_distribution(preds, probs, label_names, out_dir):
    """Bar chart of predicted class distribution on KIT data."""
    counts = np.bincount(preds, minlength=len(label_names))
    pcts   = 100 * counts / len(preds)
    colors = ["steelblue", "darkorange", "green", "red"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: prediction distribution
    bars = ax1.bar(label_names, pcts, color=colors, edgecolor="white", width=0.5)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{pct:.1f}%\n(n={cnt:,})",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("% of windows", fontsize=12)
    ax1.set_ylim(0, max(pcts) * 1.25)
    ax1.set_title("Predicted Health Grade Distribution\n(KIT Seat Leon — healthy vehicle)",
                  fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Right: mean confidence per class
    mean_conf = probs.mean(axis=0)
    bars2 = ax2.bar(label_names, mean_conf * 100, color=colors,
                    edgecolor="white", width=0.5, alpha=0.85)
    for bar, val in zip(bars2, mean_conf):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{val*100:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Mean softmax probability (%)", fontsize=12)
    ax2.set_ylim(0, max(mean_conf) * 125)
    ax2.set_title("Mean Model Confidence per Class\n(KIT cross-dataset)",
                  fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-Dataset Validation: KIT OBD-II Dataset (Seat Leon, Germany)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "cross_dataset_kit.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run(config_path="config.yaml"):
    cfg         = load_config(config_path)
    out_dir     = cfg["data"]["processed_dir"]
    label_names = cfg["labels"]["names"]
    feat_cols   = cfg["data"]["feature_columns"]   # 8 training features
    ws          = cfg["preprocessing"]["window_size"]
    st          = cfg["preprocessing"]["stride"]
    device      = torch.device("cpu")              # small batches, CPU is fine

    kit_dir = os.path.normpath(os.path.join("data", "kit", "OBD-II-Dataset"))

    print("\n" + "=" * 60)
    print("  CROSS-DATASET VALIDATION — KIT OBD-II (Seat Leon)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load KIT data
    # ------------------------------------------------------------------
    print("\n[1] Loading KIT dataset...")
    kit_df, available = load_kit_data(kit_dir, feat_cols)

    # ------------------------------------------------------------------
    # 2. Build full 8-feature matrix
    #    Available features: 6 from KIT
    #    Missing features: filled with 0 (= training median in scaled space)
    # ------------------------------------------------------------------
    print("\n[2] Building feature matrix (imputing 2 missing features)...")
    feat_matrix = np.zeros((len(kit_df), len(feat_cols)), dtype=np.float64)
    for j, feat in enumerate(feat_cols):
        if feat in kit_df.columns:
            feat_matrix[:, j] = kit_df[feat].values
        # else: stays 0 (filled after scaling below)

    # ------------------------------------------------------------------
    # 3. Scale using TRAINING scaler (fit on Classified dataset train rows)
    # ------------------------------------------------------------------
    print("\n[3] Applying training RobustScaler...")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Scale only the available features; missing stay 0 in UNSCALED space
    # We need to transform the full matrix and then zero out the missing cols
    feat_scaled = scaler.transform(feat_matrix).astype(np.float32)

    # Zero out missing feature columns IN SCALED SPACE (= training median)
    for j, feat in enumerate(feat_cols):
        if feat in KIT_MISSING:
            feat_scaled[:, j] = 0.0

    # ------------------------------------------------------------------
    # 4. Build sliding windows
    # ------------------------------------------------------------------
    print(f"\n[4] Building windows (size={ws}, stride={st})...")
    X_kit = make_windows(feat_scaled, ws, st)
    print(f"  Total windows: {len(X_kit):,}")

    # ------------------------------------------------------------------
    # 5. Load model and predict
    # ------------------------------------------------------------------
    print("\n[5] Loading model and running inference...")
    checkpoint = torch.load(cfg["training"]["checkpoint_path"], map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Checkpoint epoch {checkpoint['epoch']}  "
          f"(val_loss={checkpoint['val_loss']:.4f})")

    preds, probs = predict(model, X_kit, device)

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print("\n[6] Results:")
    counts = np.bincount(preds, minlength=len(label_names))
    for i, name in enumerate(label_names):
        pct = 100 * counts[i] / len(preds)
        bar = "#" * int(pct / 2)
        print(f"  {name:8s}: {counts[i]:6,}  ({pct:5.1f}%)  {bar}")

    normal_pct = 100 * counts[0] / len(preds)
    print(f"\n  Normal prediction rate: {normal_pct:.1f}%")
    if normal_pct >= 80:
        print("  [PASS] Model correctly identifies the KIT vehicle as healthy.")
    elif normal_pct >= 50:
        print("  [PARTIAL] Model partially generalises — some domain shift detected.")
    else:
        print("  [NOTE] Significant domain shift — model flags KIT data as faulty.")
        print("         This is expected when feature distributions differ greatly.")

    print(f"\n  Mean confidence — Normal:   {probs[:,0].mean()*100:.1f}%")
    print(f"  Mean confidence — Warning:  {probs[:,1].mean()*100:.1f}%")
    print(f"  Mean confidence — Faulty:   {probs[:,2].mean()*100:.1f}%")
    print(f"  Mean confidence — Critical: {probs[:,3].mean()*100:.1f}%")

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    print("\n[7] Saving plot...")
    plot_prediction_distribution(preds, probs, label_names, out_dir)

    print("\n=== Cross-dataset validation complete ===")
    return preds, probs


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(__file__))
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(PROJECT_ROOT, "config.yaml")
    run(config_path)
