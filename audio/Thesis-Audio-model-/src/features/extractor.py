"""
Feature extraction: log-mel spectrogram (primary) + MFCC statistics (auxiliary).

Primary representation fed to CNN models: log-mel spectrogram [1, n_mels, T].
Auxiliary hand-crafted features used for ablation studies.
"""

from __future__ import annotations

import numpy as np
import torch
import librosa
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FeatureExtractor:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 8000.0
    n_mfcc: int = 40
    use_delta: bool = True
    normalize: bool = True

    def extract(self, waveform: np.ndarray) -> torch.Tensor:
        """Return log-mel spectrogram as float32 tensor [1, n_mels, T]."""
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        if self.normalize:
            mean = log_mel.mean()
            std = log_mel.std() + 1e-9
            log_mel = (log_mel - mean) / std

        return torch.from_numpy(log_mel).unsqueeze(0)  # [1, F, T]

    def extract_mfcc_stats(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """Return a dict of MFCC-based hand-crafted statistical features."""
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
        )
        features: Dict[str, np.ndarray] = {
            "mfcc_mean": mfcc.mean(axis=1),
            "mfcc_std": mfcc.std(axis=1),
        }

        if self.use_delta:
            d1 = librosa.feature.delta(mfcc, order=1)
            d2 = librosa.feature.delta(mfcc, order=2)
            features["mfcc_d1_mean"] = d1.mean(axis=1)
            features["mfcc_d1_std"] = d1.std(axis=1)
            features["mfcc_d2_mean"] = d2.mean(axis=1)
            features["mfcc_d2_std"] = d2.std(axis=1)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        contrast = librosa.feature.spectral_contrast(
            y=waveform, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=waveform, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=self.hop_length)
        rms = librosa.feature.rms(y=waveform, hop_length=self.hop_length)

        for name, feat in [
            ("centroid", centroid),
            ("bandwidth", bandwidth),
            ("rolloff", rolloff),
            ("zcr", zcr),
            ("rms", rms),
        ]:
            features[f"{name}_mean"] = feat.mean(axis=1)
            features[f"{name}_std"] = feat.std(axis=1)

        features["contrast_mean"] = contrast.mean(axis=1)
        features["contrast_std"] = contrast.std(axis=1)

        return features

    def extract_flat_vector(self, waveform: np.ndarray) -> np.ndarray:
        """Flat feature vector for traditional ML baselines."""
        stats = self.extract_mfcc_stats(waveform)
        return np.concatenate([v.ravel() for v in stats.values()])

    @property
    def flat_feature_dim(self) -> int:
        """Dimension of the flat feature vector (estimated)."""
        # 40 mfcc × (mean+std) × (1+2 deltas) + spectral
        base = self.n_mfcc * 2
        if self.use_delta:
            base += self.n_mfcc * 4
        base += (1 + 1 + 1 + 1 + 1) * 2   # centroid, bandwidth, rolloff, zcr, rms
        base += 7 * 2                        # spectral contrast (7 bands)
        return base
