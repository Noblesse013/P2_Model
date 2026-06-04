"""Audio and spectrogram augmentation for training robustness."""

from __future__ import annotations

import random
import numpy as np
import torch
import librosa


class AudioAugmentor:
    """Time-domain waveform augmentations."""

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", True)
        self.add_noise = cfg.get("add_noise", True)
        self.noise_factor = cfg.get("noise_factor", 0.005)
        self.time_stretch = cfg.get("time_stretch", True)
        self.stretch_range = cfg.get("stretch_range", [0.85, 1.15])
        self.pitch_shift = cfg.get("pitch_shift", True)
        self.pitch_semitones = cfg.get("pitch_semitones", 2)
        self.segment_duration = cfg.get("segment_duration", 4.0)
        self.sample_rate = cfg.get("sample_rate", 22050)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return waveform

        if self.add_noise and random.random() < 0.5:
            waveform = self._add_gaussian_noise(waveform)

        if self.time_stretch and random.random() < 0.5:
            rate = random.uniform(*self.stretch_range)
            waveform = librosa.effects.time_stretch(waveform, rate=rate)

        if self.pitch_shift and random.random() < 0.5:
            steps = random.uniform(-self.pitch_semitones, self.pitch_semitones)
            waveform = librosa.effects.pitch_shift(
                waveform, sr=self.sample_rate, n_steps=steps
            )

        # Re-pad / trim to original length
        target_len = int(self.segment_duration * self.sample_rate)
        waveform = self._ensure_length(waveform, target_len)
        return waveform

    def _add_gaussian_noise(self, waveform: np.ndarray) -> np.ndarray:
        noise = np.random.randn(len(waveform)).astype(np.float32)
        return waveform + self.noise_factor * noise

    @staticmethod
    def _ensure_length(waveform: np.ndarray, length: int) -> np.ndarray:
        if len(waveform) > length:
            start = random.randint(0, len(waveform) - length)
            return waveform[start : start + length]
        if len(waveform) < length:
            pad = length - len(waveform)
            return np.pad(waveform, (0, pad), mode="reflect")
        return waveform


class SpecAugment:
    """Frequency and time masking on log-mel spectrograms (PyTorch tensor)."""

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("spec_augment", True)
        self.freq_mask_param = cfg.get("freq_mask_param", 20)
        self.time_mask_param = cfg.get("time_mask_param", 40)
        self.num_freq_masks = cfg.get("num_freq_masks", 2)
        self.num_time_masks = cfg.get("num_time_masks", 2)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: [C, F, T] or [F, T]."""
        if not self.enabled:
            return spec

        # Work on a clone to avoid in-place ops on shared storage
        spec = spec.clone()
        _, F, T = spec.shape if spec.dim() == 3 else (1, spec.shape[0], spec.shape[1])
        s = spec if spec.dim() == 3 else spec.unsqueeze(0)

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, F - f))
            s[:, f0 : f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, T - t))
            s[:, :, t0 : t0 + t] = 0.0

        return s.squeeze(0) if spec.dim() == 2 else s


def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation. Returns (mixed_inputs, y_a, y_b, lam)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[index]
    return mixed, targets, targets[index], lam
