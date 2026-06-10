from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyFeatureDataset(Dataset):
    """Dataset for sequence features saved as .npy files."""

    expected_ndim: int | None = 2

    def __init__(
        self,
        rows: list[dict],
        label_to_idx: dict[str, int],
        flatten: bool = True,
        transform=None,
    ) -> None:
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.flatten = flatten
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        features = self.load_feature(Path(row["feature_path"]))
        label = self.label_to_idx[row["gloss"]]
        return (
            torch.from_numpy(features),
            features.shape[0],
            torch.tensor(label, dtype=torch.long),
        )

    def load_feature(self, path: Path) -> np.ndarray:
        features = np.load(path).astype(np.float32, copy=False)
        if self.transform is not None:
            features = self.transform(features)
        if self.flatten and features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        if self.expected_ndim is not None and features.ndim != self.expected_ndim:
            raise ValueError(
                f"Expected {self.expected_ndim}D feature array, got {features.shape}: {path}"
            )
        return features
