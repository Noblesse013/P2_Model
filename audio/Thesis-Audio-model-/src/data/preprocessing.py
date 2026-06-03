"""Audio preprocessing: resampling, normalisation, segmentation."""

from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AudioPreprocessor:
    sample_rate: int = 22050
    segment_duration: float = 4.0   # seconds
    hop_duration: float = 2.0       # stride between segments
    normalize: bool = True

    # ── Public API ────────────────────────────────────────────

    def load(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate.

        Uses soundfile for WAV/FLAC (fast); falls back to librosa for MP3/M4A.
        """
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)  # stereo → mono
            if sr != self.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
            return data, self.sample_rate
        except Exception:
            waveform, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return waveform, sr

    def segment(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Slice waveform into fixed-length, overlapping segments."""
        seg_len = int(self.segment_duration * self.sample_rate)
        hop_len = int(self.hop_duration * self.sample_rate)

        if len(waveform) < seg_len:
            # Pad short clips with reflection
            waveform = self._pad_to_length(waveform, seg_len)

        segments: List[np.ndarray] = []
        start = 0
        while start + seg_len <= len(waveform):
            segments.append(waveform[start : start + seg_len])
            start += hop_len

        if not segments:
            segments.append(self._pad_to_length(waveform, seg_len))

        return segments

    def process(self, path: str) -> List[np.ndarray]:
        """Full pipeline: load → segment."""
        waveform, _ = self.load(path)
        if self.normalize:
            waveform = self._rms_normalize(waveform)
        return self.segment(waveform)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _rms_normalize(waveform: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        rms = np.sqrt(np.mean(waveform ** 2)) + 1e-9
        return waveform * (target_rms / rms)

    @staticmethod
    def _pad_to_length(waveform: np.ndarray, length: int) -> np.ndarray:
        if len(waveform) == 0:
            return np.zeros(length, dtype=np.float32)
        repeats = int(np.ceil(length / len(waveform)))
        tiled = np.tile(waveform, repeats)
        return tiled[:length]
