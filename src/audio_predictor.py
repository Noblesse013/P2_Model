"""
Audio branch predictor.

Loads the pre-trained MSDA-Net model and uses the audio project's
AudioPreprocessor + FeatureExtractor pipeline to predict engine health
grade from a WAV/MP3/FLAC audio file.

Prediction uses majority-vote over all extracted 3-second segments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


class AudioPredictor:
    def __init__(self, model_dir: str, checkpoint: str, config: str,
                 device: torch.device = None):
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make the audio project's src/ importable
        audio_src = str(self.model_dir)
        if audio_src not in sys.path:
            sys.path.insert(0, audio_src)

        self._load_artifacts(checkpoint, config)

    # ------------------------------------------------------------------
    def _load_artifacts(self, ckpt_rel: str, cfg_rel: str) -> None:
        with open(self.model_dir / cfg_rel) as f:
            raw = yaml.safe_load(f)
        self.cfg = OmegaConf.create(raw)

        audio_root = str(self.model_dir)
        _src_save = {k: sys.modules.pop(k)
                     for k in list(sys.modules)
                     if k == 'src' or k.startswith('src.')}
        if audio_root in sys.path:
            sys.path.remove(audio_root)
        sys.path.insert(0, audio_root)
        try:
            from src.data.preprocessing import AudioPreprocessor
            from src.features.extractor import FeatureExtractor
            from src.models.factory import build_model

            self.preprocessor = AudioPreprocessor(
                sample_rate=self.cfg.data.sample_rate,
                segment_duration=self.cfg.data.segment_duration,
                hop_duration=self.cfg.data.hop_duration,
            )
            self.feature_extractor = FeatureExtractor(
                sample_rate=self.cfg.data.sample_rate,
                n_fft=self.cfg.features.n_fft,
                hop_length=self.cfg.features.hop_length,
                n_mels=self.cfg.features.n_mels,
                f_min=self.cfg.features.f_min,
                f_max=self.cfg.features.f_max,
                normalize=self.cfg.features.normalize,
            )

            ckpt = torch.load(self.model_dir / ckpt_rel, map_location=self.device, weights_only=False)
            self.model = build_model(self.cfg)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device).eval()
        finally:
            for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
                del sys.modules[k]
            sys.modules.update(_src_save)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, audio_path: str) -> dict:
        """
        Segments the audio file, runs each segment through MSDA-Net,
        and returns the average class probabilities.

        Returns:
            grade       int  0-3
            probs       np.ndarray  (4,) class probabilities
            n_segments  int  number of 3-sec segments processed
        """
        audio_root = str(self.model_dir)
        _src_save = {k: sys.modules.pop(k)
                     for k in list(sys.modules)
                     if k == 'src' or k.startswith('src.')}
        if audio_root in sys.path:
            sys.path.remove(audio_root)
        sys.path.insert(0, audio_root)
        try:
            from src.models.ordinal_head import OrdinalHead
        finally:
            for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
                del sys.modules[k]
            sys.modules.update(_src_save)

        segments = self.preprocessor.process(str(audio_path))
        if not segments:
            raise ValueError(f"No audio segments extracted from: {audio_path}")

        all_probs = []
        for seg in segments:
            spec = self.feature_extractor.extract(seg).unsqueeze(0).to(self.device)
            logits = self.model(spec)
            probs = OrdinalHead.to_class_probs(logits).squeeze(0).cpu().numpy()
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        grade = int(np.argmax(avg_probs))

        return {"grade": grade, "probs": avg_probs, "n_segments": len(segments)}
