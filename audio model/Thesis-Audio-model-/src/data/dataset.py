"""Dataset classes and DataLoader factory for engine health grading."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .preprocessing import AudioPreprocessor
from .augmentation import AudioAugmentor, SpecAugment
from ..features.extractor import FeatureExtractor

# ── Constants ─────────────────────────────────────────────────
GRADE_MAP = {"normal": 0, "warning": 1, "fault": 2, "critical": 3}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Dataset ───────────────────────────────────────────────────

class EngineAudioDataset(Dataset):
    """
    Loads engine audio files and returns (spectrogram, label) pairs.

    Supports two data layouts:
      1. Folder-per-class: data/{normal,warning,fault,critical}/*.wav
      2. CSV with columns [filepath, label] or [filepath, grade]

    If cache_dir is provided, computed spectrograms are saved as .npy on first
    access and reloaded instantly on all subsequent epochs.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        feature_extractor: FeatureExtractor,
        preprocessor: AudioPreprocessor,
        audio_augmentor: Optional[AudioAugmentor] = None,
        spec_augmentor: Optional[SpecAugment] = None,
        is_train: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.samples = samples
        self.fe = feature_extractor
        self.prep = preprocessor
        self.audio_aug = audio_augmentor
        self.spec_aug = spec_augmentor
        self.is_train = is_train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, audio_path: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        p = Path(audio_path)
        return self.cache_dir / p.parent.name / (p.stem + ".npy")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        cache_path = self._cache_path(path)

        if cache_path is not None and cache_path.exists():
            spec = torch.from_numpy(np.load(str(cache_path)))
        else:
            segments = self.prep.process(path)
            waveform = segments[len(segments) // 2]  # always use centre for cache
            spec = self.fe.extract(waveform)          # [1, F, T]
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_path), spec.numpy())

        if self.is_train and self.spec_aug is not None:
            spec = self.spec_aug(spec)

        return spec, label

    # ── Class helpers ─────────────────────────────────────────

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for _, lbl in self.samples:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts


# ── Sample builders ───────────────────────────────────────────

def scan_folder(data_dir: str, grade_map: Dict[str, int] = GRADE_MAP) -> List[Tuple[str, int]]:
    """Walk class-named subdirectories and collect (filepath, grade) pairs."""
    samples: List[Tuple[str, int]] = []
    root = Path(data_dir)
    for cls_name, grade in grade_map.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((str(p), grade))
    return samples


def load_csv(csv_path: str, grade_map: Dict[str, int] = GRADE_MAP) -> List[Tuple[str, int]]:
    """Load samples from a CSV with [filepath, label] columns."""
    df = pd.read_csv(csv_path)
    label_col = "grade" if "grade" in df.columns else "label"
    samples: List[Tuple[str, int]] = []
    for _, row in df.iterrows():
        lbl = row[label_col]
        grade = lbl if isinstance(lbl, int) else grade_map.get(str(lbl).lower(), -1)
        if grade >= 0:
            samples.append((str(row["filepath"]), grade))
    return samples


def stratified_split(
    samples: List[Tuple[str, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Stratified train/val/test split preserving class distribution."""
    rng = random.Random(seed)
    by_class: Dict[int, List] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train, val, test = [], [], []
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        n = len(cls_samples)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test.extend(cls_samples[:n_test])
        val.extend(cls_samples[n_test : n_test + n_val])
        train.extend(cls_samples[n_test + n_val :])

    return train, val, test


def make_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    """Over-sample minority classes for balanced mini-batches."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_weights = 1.0 / counts
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ── DataLoader factory ────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train/val/test DataLoaders from config.
    Returns (train_loader, val_loader, test_loader).
    """
    from omegaconf import OmegaConf

    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    # Feature extractor
    fe = FeatureExtractor(
        sample_rate=cfg.data.sample_rate,
        n_fft=cfg.features.n_fft,
        hop_length=cfg.features.hop_length,
        n_mels=cfg.features.n_mels,
        f_min=cfg.features.f_min,
        f_max=cfg.features.f_max,
        n_mfcc=cfg.features.n_mfcc,
        use_delta=cfg.features.use_delta,
        normalize=cfg.features.normalize,
    )

    prep = AudioPreprocessor(
        sample_rate=cfg.data.sample_rate,
        segment_duration=cfg.data.segment_duration,
        hop_duration=cfg.data.hop_duration,
        normalize=True,
    )

    aug_cfg = dict(cfg.augmentation)
    aug_cfg["sample_rate"] = cfg.data.sample_rate
    aug_cfg["segment_duration"] = cfg.data.segment_duration
    audio_aug = AudioAugmentor(aug_cfg) if cfg.augmentation.enabled else None
    spec_aug = SpecAugment(aug_cfg) if cfg.augmentation.enabled else None

    # Load samples
    if cfg.data.get("train_csv"):
        all_samples = load_csv(cfg.data.train_csv)
    else:
        all_samples = scan_folder(cfg.data.data_dir)

    if not all_samples:
        raise FileNotFoundError(
            f"No audio files found in '{cfg.data.data_dir}'. "
            "Organise as data/{normal,warning,fault,critical}/*.wav"
        )

    train_s, val_s, test_s = stratified_split(
        all_samples,
        val_ratio=cfg.data.val_split,
        test_ratio=cfg.data.test_split,
        seed=cfg.project.seed,
    )

    cache_dir = str(Path(cfg.data.data_dir) / ".spec_cache")
    train_ds = EngineAudioDataset(train_s, fe, prep, None, spec_aug, is_train=True,
                                  cache_dir=cache_dir)
    val_ds = EngineAudioDataset(val_s, fe, prep, is_train=False, cache_dir=cache_dir)
    test_ds = EngineAudioDataset(test_s, fe, prep, is_train=False, cache_dir=cache_dir)

    sampler = make_weighted_sampler(train_ds.labels, cfg.data.num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
