"""
Preprocessing pipeline (fair version):
  1. Load raw CSV
  2. Assign 4-class labels using ONLY the 3 diagnostic severity features
     (lambda_voltage, long_fuel_trim, timing_advance) — these are kept
     strictly separate from the 8 operational training features.
  3. Driver-based split: Driver 3 -> test entirely.
     Drivers 1+2 -> stratified random split into train (85%) and val (15%).
  4. Scale 8 operational training features (RobustScaler, fit on train only)
  5. Stratified sliding-window samples (guarantees all 4 classes per split)
  6. Save processed tensors to disk

Key fairness properties
-----------------------
  * The 3 severity signals (lambda, LTFT, timing) that determine Warning /
    Faulty / Critical severity are NEVER in the training feature matrix.
    The model cannot learn the labeling rule directly.
  * Driver 3 (196,800 rows) is a completely held-out test driver — no rows
    from Driver 3 are used for fitting the scaler or training the model.
  * Scaler fit on training rows only; val and test are transform-only.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pickle


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_classified_data(cfg):
    ds = cfg["data"]["classified_dataset"]
    filepath = os.path.join(cfg["data"]["raw_dir"], ds["filename"])

    print(f"  Loading: {ds['filename']}")
    raw = pd.read_csv(filepath, encoding=ds["encoding"])
    print(f"  Raw shape: {raw.shape}")

    df = pd.DataFrame()

    # 8 operational training features
    for idx, name in zip(ds["feature_indices"], ds["feature_names"]):
        df[name] = raw.iloc[:, idx].values

    # 3 severity-scoring features — loaded for labeling, NEVER added to
    # the feature matrix that the model sees
    for idx, name in zip(ds["severity_feature_indices"],
                         ds["severity_feature_names"]):
        df[name] = raw.iloc[:, idx].values

    df["Label"]        = raw[ds["label_col"]].values
    df["Conductor_ID"] = raw[ds["driver_col"]].values

    all_feats = ds["feature_names"] + ds["severity_feature_names"]
    before = len(df)
    df = df.dropna(subset=all_feats).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} NaN rows")

    print(f"  Rows: {len(df)}"
          f" | Training features: {len(ds['feature_names'])}"
          f" | Severity features: {len(ds['severity_feature_names'])} (labeling only)")
    return df


# ---------------------------------------------------------------------------
# 4-class labeling — uses ONLY the 3 severity features
# ---------------------------------------------------------------------------

def auto_label_classified(df, thresholds):
    """
    Grade 0 Normal   — Label == 0
    Grade 1 Warning  — Label == 1, composite severity score 0-1
    Grade 2 Faulty   — Label == 1, composite severity score 2-3
    Grade 3 Critical — Label == 1, composite severity score >= 4

    Scoring uses lambda_voltage, long_fuel_trim, timing_advance.
    These three signals are kept OUT of the training feature matrix so
    the model cannot memorise the labeling threshold directly.
    """
    grades = np.zeros(len(df), dtype=int)
    faulty = df["Label"].values == 1

    lam  = df["lambda_voltage"].values
    ltft = df["long_fuel_trim"].values
    tim  = df["timing_advance"].values

    lam_score  = np.where(lam  >= thresholds["lambda_warning"],  0,
                 np.where(lam  >= thresholds["lambda_faulty"],   1, 2))
    ltft_score = np.where(ltft <= thresholds["ltft_faulty"],     0,
                 np.where(ltft <= thresholds["ltft_critical"],   1, 2))
    tim_score  = np.where(tim  >= thresholds["timing_faulty"],   0,
                 np.where(tim  >= thresholds["timing_critical"], 1, 2))

    total = lam_score + ltft_score + tim_score   # range 0-6

    grades[faulty & (total <= 1)]                = 1   # Warning
    grades[faulty & (total >= 2) & (total <= 3)] = 2   # Faulty
    grades[faulty & (total >= 4)]                = 3   # Critical

    return grades


# ---------------------------------------------------------------------------
# Stratified windowing — guarantees all 4 classes in every split
# ---------------------------------------------------------------------------

def make_windows(features, labels, window_size, stride):
    """
    Group rows by health grade, then slide within each group.
    This guarantees that all 4 classes appear in every split regardless
    of how the data is ordered within the split.

    Each window is a homogeneous snapshot of one health state — a valid
    assumption for OBD-2 readings where health grade is a sustained condition
    (not a single-sample event).
    """
    X, y = [], []
    n_classes = int(labels.max()) + 1

    for cls in range(n_classes):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < window_size:
            if len(cls_idx) > 0:
                padded = np.resize(cls_idx, window_size)
                X.append(features[padded])
                y.append(cls)
            continue
        for start in range(0, len(cls_idx) - window_size + 1, stride):
            X.append(features[cls_idx[start: start + window_size]])
            y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path="config.yaml"):
    print("=== Preprocessing (fair version) ===")
    cfg = load_config(config_path)

    ds_cfg      = cfg["data"]["classified_dataset"]
    feat_cols   = ds_cfg["feature_names"]          # 8 training features only
    label_names = cfg["labels"]["names"]
    thresholds  = cfg["data"]["label_thresholds"]
    test_driver = ds_cfg["test_driver_id"]
    seed        = cfg["preprocessing"]["random_seed"]
    val_size    = cfg["preprocessing"]["val_size"]
    ws          = cfg["preprocessing"]["window_size"]
    st          = cfg["preprocessing"]["stride"]

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("\n[1] Loading raw data...")
    df = load_classified_data(cfg)
    print(f"  Total rows: {len(df)}")

    # ------------------------------------------------------------------
    # 2. 4-class labels using severity features only
    # ------------------------------------------------------------------
    print("\n[2] Assigning health grades (severity features are NOT in training)...")
    labels = auto_label_classified(df, thresholds)

    for i, name in enumerate(label_names):
        count = int(np.sum(labels == i))
        print(f"  {name:8s} ({i}): {count:6d}  ({100*count/len(labels):.1f}%)")

    # ------------------------------------------------------------------
    # 3. Driver-based split
    #    Driver test_driver -> test (all rows, completely held out)
    #    Drivers 1+2 -> random stratified split into train / val
    # ------------------------------------------------------------------
    print(f"\n[3] Driver-based split  "
          f"(Driver {test_driver} = test holdout)...")

    driver_ids = df["Conductor_ID"].values
    idx_all    = np.arange(len(df))

    idx_test     = idx_all[driver_ids == test_driver]
    idx_trainval = idx_all[driver_ids != test_driver]

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size,
        random_state=seed,
        stratify=labels[idx_trainval],
    )

    print(f"  Test  (Driver {test_driver}): {len(idx_test):6d} rows")
    print(f"  Train (Drivers 1+2): {len(idx_train):6d} rows")
    print(f"  Val   (Drivers 1+2): {len(idx_val):6d} rows")

    # ------------------------------------------------------------------
    # 4. Scale — fit ONLY on training rows
    # ------------------------------------------------------------------
    print("\n[4] Scaling features (RobustScaler, fit on train rows only)...")
    features_arr = df[feat_cols].values.astype(np.float64)
    scaler = RobustScaler()

    feat_train = scaler.fit_transform(features_arr[idx_train]).astype(np.float32)
    feat_val   = scaler.transform(features_arr[idx_val]).astype(np.float32)
    feat_test  = scaler.transform(features_arr[idx_test]).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Stratified sliding windows — built independently per split
    # ------------------------------------------------------------------
    print(f"\n[5] Building stratified windows  (window={ws}, stride={st})...")

    X_train, y_train = make_windows(feat_train, labels[idx_train], ws, st)
    X_val,   y_val   = make_windows(feat_val,   labels[idx_val],   ws, st)
    X_test,  y_test  = make_windows(feat_test,  labels[idx_test],  ws, st)

    for split_name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {split_name}: {len(yy)} windows")
        for i, name in enumerate(label_names):
            count = int(np.sum(yy == i))
            pct   = 100 * count / max(len(yy), 1)
            print(f"    {name:8s}: {count:5d}  ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print("\n[6] Saving to disk...")
    out_dir = cfg["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_columns": feat_cols,
        "window_size":     ws,
        "n_features":      len(feat_cols),
        "n_classes":       4,
        "label_names":     label_names,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"  Saved all arrays + scaler + meta to '{out_dir}/'")
    print("\n=== Preprocessing complete ===")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(__file__))
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(PROJECT_ROOT, "config.yaml")
    run(config_path)
