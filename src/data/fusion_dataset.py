"""
PseudoPairedDataset — training/evaluation dataset for the fusion model.

Since the OBD2 and audio datasets are from different recordings, we create
pseudo-pairs by matching same-grade samples from each modality. This is a
documented technique when synchronized multimodal data is unavailable.

At each access the dataset draws a random (obd2_emb, audio_emb) pair from
the same grade bucket, so the model learns: "when both modalities signal
grade k, what is the joint ordinal estimate?" — without relying on spurious
cross-recording correlations.

Usage:
    # Build after running extract_embeddings.py
    train_ds = PseudoPairedDataset.from_split("embeddings", split="train")
    val_ds   = PseudoPairedDataset.from_split("embeddings", split="val")
    test_ds  = PseudoPairedDataset.from_split("embeddings", split="test")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class PseudoPairedDataset(Dataset):
    """
    Pseudo-paired dataset from pre-extracted embedding files.

    Each item is a (h_obd2, h_audio, grade) triplet where both embeddings
    share the same health grade label.

    Args:
        obd2_embs:   (N_obd2, 256) float32
        obd2_labels: (N_obd2,)     int64
        audio_embs:  (N_audio, 512) float32
        audio_labels:(N_audio,)     int64
        n_classes:   4
    """

    def __init__(
        self,
        obd2_embs:    np.ndarray,
        obd2_labels:  np.ndarray,
        audio_embs:   np.ndarray,
        audio_labels: np.ndarray,
        n_classes:    int = 4,
    ):
        self.n_classes = n_classes

        # Split both modalities into per-grade buckets
        self.obd2_by_grade  = self._group_by_grade(obd2_embs,  obd2_labels)
        self.audio_by_grade = self._group_by_grade(audio_embs, audio_labels)

        # One index entry per OBD2 sample — audio pairing is random at access time
        self.obd2_embs   = torch.tensor(obd2_embs,   dtype=torch.float32)
        self.obd2_labels = torch.tensor(obd2_labels, dtype=torch.long)

        # Validate all grades present in both modalities
        for g in range(n_classes):
            assert g in self.obd2_by_grade,  f"Grade {g} missing from OBD2 embeddings"
            assert g in self.audio_by_grade, f"Grade {g} missing from audio embeddings"

    @staticmethod
    def _group_by_grade(embs: np.ndarray,
                        labels: np.ndarray) -> dict[int, torch.Tensor]:
        groups = {}
        for g in np.unique(labels):
            mask = labels == g
            groups[int(g)] = torch.tensor(embs[mask], dtype=torch.float32)
        return groups

    def __len__(self) -> int:
        return len(self.obd2_labels)

    def __getitem__(self, idx: int):
        h_obd2 = self.obd2_embs[idx]
        grade  = int(self.obd2_labels[idx].item())

        # Randomly pick one audio embedding of the same grade
        audio_pool = self.audio_by_grade[grade]
        rand_idx   = torch.randint(len(audio_pool), (1,)).item()
        h_audio    = audio_pool[rand_idx]

        return h_obd2, h_audio, torch.tensor(grade, dtype=torch.long)

    # ------------------------------------------------------------------
    # Factory constructors

    @classmethod
    def from_split(cls, emb_dir: str, split: str = "train",
                   n_classes: int = 4) -> "PseudoPairedDataset":
        """
        Load pre-extracted embedding .npy files for a given split.
        Expects:
            {emb_dir}/obd2_{split}.npy
            {emb_dir}/obd2_{split}_labels.npy
            {emb_dir}/audio_{split}.npy
            {emb_dir}/audio_{split}_labels.npy
        """
        d = Path(emb_dir)
        return cls(
            obd2_embs    = np.load(d / f"obd2_{split}.npy"),
            obd2_labels  = np.load(d / f"obd2_{split}_labels.npy"),
            audio_embs   = np.load(d / f"audio_{split}.npy"),
            audio_labels = np.load(d / f"audio_{split}_labels.npy"),
            n_classes    = n_classes,
        )


# ── DataLoader builders ──────────────────────────────────────────────────────

def build_loaders(emb_dir: str, batch_size: int = 256,
                  num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Training loader uses WeightedRandomSampler to balance the 4 health grades.
    Val and test loaders are unshuffled for reproducible evaluation.
    """
    train_ds = PseudoPairedDataset.from_split(emb_dir, "train")
    val_ds   = PseudoPairedDataset.from_split(emb_dir, "val")
    test_ds  = PseudoPairedDataset.from_split(emb_dir, "test")

    # Compute per-sample weights for balanced sampling
    labels  = train_ds.obd2_labels.numpy()
    counts  = np.bincount(labels, minlength=4)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
