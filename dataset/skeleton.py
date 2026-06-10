from __future__ import annotations

from .base import NumpyFeatureDataset


class SkeletonFeatureDataset(NumpyFeatureDataset):
    """Dataset for MediaPipe skeleton features under feature/skeleton/."""

    expected_ndim = 2
