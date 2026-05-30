"""
EngineHealthGrader — top-level class that orchestrates both modality
predictors and the fusion layer.

Usage:
    grader = EngineHealthGrader("configs/config.yaml")

    # From file paths
    result = grader.grade(obd2="readings.csv", audio="engine.wav")

    # From live readings (last 30 timesteps as a list of dicts)
    result = grader.grade(
        obd2=[{"engine_rpm": 2500, "coolant_temp": 90, ...}, ...],
        audio="engine.wav"
    )

    # Single reading (replicated to fill window)
    result = grader.grade(
        obd2={"engine_rpm": 2500, "coolant_temp": 90,
              "engine_load": 65, "map_pressure": 101,
              "intake_air_temp": 35, "throttle_position": 40,
              "vehicle_speed": 80, "catalyst_temp": 450},
        audio="engine.wav"
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml

from src.obd2_predictor import OBD2Predictor
from src.audio_predictor import AudioPredictor
from src.fusion import fuse, GRADE_INFO


class EngineHealthGrader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            # Resolve relative to the project root (parent of configs/)
            cfg_path = Path(__file__).parent.parent / config_path
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._obd2: OBD2Predictor | None = None
        self._audio: AudioPredictor | None = None

    # ------------------------------------------------------------------
    # Lazy model loading — only instantiate when first needed

    @property
    def obd2_predictor(self) -> OBD2Predictor:
        if self._obd2 is None:
            c = self.cfg["obd2"]
            self._obd2 = OBD2Predictor(
                model_dir=c["model_dir"],
                checkpoint=c["checkpoint"],
                scaler=c["scaler"],
                meta=c["meta"],
                config=c["config"],
                device=self.device,
            )
        return self._obd2

    @property
    def audio_predictor(self) -> AudioPredictor:
        if self._audio is None:
            c = self.cfg["audio"]
            self._audio = AudioPredictor(
                model_dir=c["model_dir"],
                checkpoint=c["checkpoint"],
                config=c["config"],
                device=self.device,
            )
        return self._audio

    # ------------------------------------------------------------------
    def grade(self, obd2, audio: str) -> dict:
        """
        Grade engine health using both modalities.

        Args:
            obd2:  str (CSV path) | dict (single reading) | list[dict] (sequence) | np.ndarray
            audio: str path to WAV/MP3/FLAC file

        Returns:
            Full result dict — see fusion.fuse() for keys, plus:
                obd2_n_timesteps   int
                audio_n_segments   int
                device             str
        """
        # ── OBD2 ──────────────────────────────────────────────
        if isinstance(obd2, (str, Path)):
            import pandas as pd
            df = pd.read_csv(obd2)
            obd2_data = df
        else:
            obd2_data = obd2

        obd2_result  = self.obd2_predictor.predict(obd2_data)
        audio_result = self.audio_predictor.predict(str(audio))

        # ── Fusion ────────────────────────────────────────────
        fc = self.cfg["fusion"]
        result = fuse(
            obd2_probs  = obd2_result["probs"],
            audio_probs = audio_result["probs"],
            strategy    = fc.get("strategy", "weighted_average"),
            obd2_weight = fc.get("obd2_weight", 0.4),
            audio_weight= fc.get("audio_weight", 0.6),
        )

        result["audio_n_segments"]  = audio_result["n_segments"]
        result["device"]            = str(self.device)
        return result

    # ------------------------------------------------------------------
    def grade_obd2_only(self, obd2) -> dict:
        """Grade using OBD2 branch only (audio unavailable)."""
        if isinstance(obd2, (str, Path)):
            import pandas as pd
            obd2 = pd.read_csv(obd2)
        r = self.obd2_predictor.predict(obd2)
        grade = r["grade"]
        return {
            "grade": grade,
            "grade_label": GRADE_INFO[grade]["label"],
            "confidence": float(r["probs"][grade]),
            "probs": r["probs"],
            "action": GRADE_INFO[grade]["action"],
            "modality": "obd2_only",
        }

    def grade_audio_only(self, audio: str) -> dict:
        """Grade using audio branch only (OBD2 unavailable)."""
        r = self.audio_predictor.predict(str(audio))
        grade = r["grade"]
        return {
            "grade": grade,
            "grade_label": GRADE_INFO[grade]["label"],
            "confidence": float(r["probs"][grade]),
            "probs": r["probs"],
            "action": GRADE_INFO[grade]["action"],
            "n_segments": r["n_segments"],
            "modality": "audio_only",
        }
