"""
PyTorch Dataset wrapper for the preprocessed sliding-window tensors.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight


class EngineHealthDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, window_size, n_features)
        # y: (N,)  integer class labels
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights so WeightedRandomSampler balances classes."""
    classes = np.unique(y)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_map = {cls: w for cls, w in zip(classes, class_weights)}
    return np.array([weight_map[label] for label in y], dtype=np.float32)


def build_loaders(processed_dir: str, batch_size: int = 64):
    """
    Load preprocessed .npy files and return DataLoaders.
    Training loader uses WeightedRandomSampler to combat class imbalance.
    """
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test  = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(processed_dir, "y_test.npy"))

    train_dataset = EngineHealthDataset(X_train, y_train)
    val_dataset   = EngineHealthDataset(X_val,   y_val)
    test_dataset  = EngineHealthDataset(X_test,  y_test)

    # Weighted sampler for balanced training batches
    sample_weights = compute_sample_weights(y_train)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, y_train
