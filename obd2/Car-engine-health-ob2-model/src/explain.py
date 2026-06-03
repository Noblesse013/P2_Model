"""
SHAP-based explainability for the trained CNN+BiGRU+Attention model.

Produces:
  - Feature importance bar plot (mean |SHAP| across all samples)
  - SHAP beeswarm summary plot
  - Per-class SHAP values

Note: SHAP's DeepExplainer is used because the model is a PyTorch DNN.
For speed, a random subset of the test set is used as background and
explanation samples (controlled by n_background and n_explain).
"""

import os
import pickle
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import shap

from dataset import build_loaders
from model import build_model


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def run(config_path="config.yaml", n_background=100, n_explain=50):
    cfg = load_config(config_path)
    device = torch.device("cpu")       # SHAP DeepExplainer works best on CPU
    label_names = cfg["labels"]["names"]
    out_dir = cfg["data"]["processed_dir"]

    print("=== SHAP Explainability ===")

    # Load model
    checkpoint = torch.load(cfg["training"]["checkpoint_path"], map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    X_test = np.load(os.path.join(out_dir, "X_test.npy"))
    y_test = np.load(os.path.join(out_dir, "y_test.npy"))

    # Load metadata for feature names
    with open(os.path.join(out_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    feature_cols = meta["feature_columns"]
    window_size  = meta["window_size"]
    n_features   = meta["n_features"]

    # Flatten windows for SHAP: (N, window_size, n_features) → (N, window_size*n_features)
    X_flat = X_test.reshape(len(X_test), -1).astype(np.float32)

    # Random background and explanation subsets
    rng = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_flat), min(n_background, len(X_flat)), replace=False)
    exp_idx = rng.choice(len(X_flat), min(n_explain,    len(X_flat)), replace=False)

    background_flat = X_flat[bg_idx]
    explain_flat    = X_flat[exp_idx]

    # KernelExplainer: model-agnostic, works with GRU/LSTM/any architecture.
    # Takes a plain Python function (numpy in → numpy out) as input.
    def model_predict(x_flat_np):
        x_t = torch.tensor(x_flat_np.astype(np.float32))
        x_t = x_t.view(-1, window_size, n_features)
        with torch.no_grad():
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    print(f"  Computing SHAP values with KernelExplainer "
          f"(background={len(bg_idx)}, explain={len(exp_idx)}, nsamples=200)...")
    print("  (This may take a few minutes — KernelExplainer is model-agnostic but slower)")
    explainer = shap.KernelExplainer(model_predict, background_flat)
    # shap_values: list of n_classes arrays, each (n_explain, window_size*n_features)
    shap_values = explainer.shap_values(explain_flat, nsamples=200, silent=True)

    # Normalise shap_values to a consistent 3D array:
    #   (n_classes, n_explain, window_size * n_features)
    # SHAP ≥0.45 returns a list[array(n_explain, n_flat)] — one array per class.
    # SHAP ≥0.47 sometimes returns a single array(n_explain, n_flat, n_classes).
    # Handle both formats defensively.
    n_classes = len(label_names)
    n_exp     = len(exp_idx)

    if isinstance(shap_values, list):
        # List format: [class_0_array, class_1_array, ...]
        shap_3d = np.stack([np.array(sv) for sv in shap_values], axis=0)
        # → (n_classes, n_explain, n_flat_features)
    else:
        sv_arr = np.array(shap_values)
        if sv_arr.ndim == 3 and sv_arr.shape[-1] == n_classes:
            # Shape (n_explain, n_flat_features, n_classes) → transpose
            shap_3d = sv_arr.transpose(2, 0, 1)
        elif sv_arr.ndim == 2:
            # Single-output fallback: (n_explain, n_flat_features)
            shap_3d = sv_arr[np.newaxis]
        else:
            shap_3d = sv_arr  # best effort

    # shap_3d: (n_classes, n_explain, window_size * n_features)
    # Reshape time dimension and average |SHAP| over timesteps → per-feature importance
    n_flat = window_size * n_features
    shap_per_feature = np.zeros((n_classes, shap_3d.shape[1], n_features))
    for c in range(min(n_classes, shap_3d.shape[0])):
        sv = shap_3d[c].reshape(-1, window_size, n_features)   # (n_explain, ws, feats)
        shap_per_feature[c, :sv.shape[0]] = np.abs(sv).mean(axis=1)

    # Global feature importance (mean over classes and samples)
    global_importance = shap_per_feature.mean(axis=(0, 1))   # (n_features,)
    sorted_idx = np.argsort(global_importance)[::-1]

    # --- Plot 1: Global Feature Importance Bar Chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        [feature_cols[i] for i in sorted_idx],
        global_importance[sorted_idx],
        color="steelblue"
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (CNN+BiGRU+Attention)")
    ax.invert_yaxis()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "shap_feature_importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    # --- Plot 2: Per-class Feature Importance ---
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 5), sharey=True)
    for c, ax in enumerate(axes):
        class_importance = shap_per_feature[c].mean(axis=0)
        s_idx = np.argsort(class_importance)[::-1]
        ax.barh(
            [feature_cols[i] for i in s_idx],
            class_importance[s_idx],
            color=["steelblue", "darkorange", "green", "red"][c]
        )
        ax.set_title(label_names[c])
        ax.set_xlabel("Mean |SHAP|")
        ax.invert_yaxis()
    plt.suptitle("Per-Class Feature Importance", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "shap_per_class.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    print("\n=== SHAP complete ===")
    return shap_per_feature, feature_cols


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)
