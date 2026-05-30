"""
OBD2 branch predictor.

Loads the pre-trained CNN+BiGRU+Attention model and RobustScaler,
then predicts engine health grade from a window of OBD2 sensor readings.

Input formats accepted by predict():
  - dict  {feature: value}               → single timestep (replicated to window_size)
  - list  [{feature: value}, ...]         → sequence of timesteps
  - np.ndarray (window_size, n_features)  → already shaped window
  - pd.DataFrame with feature columns     → takes the last window_size rows
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml


class OBD2Predictor:
    FEATURE_NAMES = [
        "engine_rpm", "coolant_temp", "engine_load", "map_pressure",
        "intake_air_temp", "throttle_position", "vehicle_speed", "catalyst_temp",
    ]

    def __init__(self, model_dir: str, checkpoint: str, scaler: str,
                 meta: str, config: str, device: torch.device = None):
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Insert the OBD2 project's src/ into path so model.py is importable
        src_path = str(self.model_dir / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        self._load_artifacts(checkpoint, scaler, meta, config)

    # ------------------------------------------------------------------
    def _load_artifacts(self, ckpt_rel: str, scaler_rel: str,
                        meta_rel: str, cfg_rel: str) -> None:
        # Config
        cfg_path = self.model_dir / cfg_rel
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.window_size = cfg["preprocessing"]["window_size"]
        self.n_features  = cfg["model"]["n_features"]

        # Scaler
        with open(self.model_dir / scaler_rel, "rb") as f:
            self.scaler = pickle.load(f)

        # Meta (feature names, etc.)
        with open(self.model_dir / meta_rel, "rb") as f:
            meta = pickle.load(f)
        self.feature_names = meta.get("feature_columns", self.FEATURE_NAMES)

        # Model
        from model import build_model, EngineHealthClassifier  # noqa: F401
        ckpt = torch.load(self.model_dir / ckpt_rel, map_location=self.device, weights_only=False)
        self.model = build_model(cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    def _to_window(self, data) -> np.ndarray:
        """Convert any input form to a (window_size, n_features) float32 array."""
        import pandas as pd

        if isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        elif isinstance(data, pd.DataFrame):
            arr = data[self.feature_names].values.astype(np.float32)
        elif isinstance(data, dict):
            arr = np.array([[data[f] for f in self.feature_names]], dtype=np.float32)
        elif isinstance(data, list):
            if isinstance(data[0], dict):
                arr = np.array([[d[f] for f in self.feature_names] for d in data], dtype=np.float32)
            else:
                arr = np.array(data, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported OBD2 input type: {type(data)}")

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Pad or truncate to window_size
        T = arr.shape[0]
        if T < self.window_size:
            # Tile (repeat) to fill the window — keeps temporal statistics stable
            reps = int(np.ceil(self.window_size / T))
            arr = np.tile(arr, (reps, 1))
        arr = arr[-self.window_size:]   # take the most recent window_size rows
        return arr  # (window_size, n_features)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, data) -> dict:
        """
        Returns:
            grade       int  0-3
            probs       np.ndarray  (4,) class probabilities
            logits      np.ndarray  (4,) raw logits
        """
        window = self._to_window(data)   # (window_size, n_features)
        scaled = self.scaler.transform(window)   # (window_size, n_features)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        # x: (1, window_size, n_features)

        logits = self.model(x)           # (1, 4)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        grade  = int(np.argmax(probs))

        return {"grade": grade, "probs": probs, "logits": logits.squeeze(0).cpu().numpy()}
