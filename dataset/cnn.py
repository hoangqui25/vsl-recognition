from __future__ import annotations

from .base import NumpyFeatureDataset


class CNNFeatureDataset(NumpyFeatureDataset):
    """Dataset for CNN features under feature/cnn/."""

    expected_ndim = 2
