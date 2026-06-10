from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureAugmentationConfig:
    enabled: bool = False
    prob: float = 0.5
    rotation_deg: float = 0.0
    shear: float = 0.0
    scale: float = 0.0
    gaussian_noise_std: float = 0.0


class SkeletonAugmentation:
    """Spatial augmentation for MediaPipe skeleton arrays.

    Expected input after loading is either (frames, 300) or (frames, 75, 4).
    Only x,y coordinates are transformed. z/visibility are kept unchanged,
    except optional Gaussian noise on x,y,z.
    """

    def __init__(self, config: FeatureAugmentationConfig) -> None:
        self.config = config

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if not self.config.enabled or np.random.random() > self.config.prob:
            return features

        original_shape = features.shape
        skeleton = features.reshape(features.shape[0], -1, 4).copy()
        valid_mask = skeleton[..., 3] > 0

        if valid_mask.any():
            center = skeleton[..., :2][valid_mask].mean(axis=0)
            transform = self._sample_affine_matrix()
            xy = skeleton[..., :2] - center
            skeleton[..., :2] = xy @ transform.T + center

        if self.config.gaussian_noise_std > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.config.gaussian_noise_std,
                size=skeleton[..., :3].shape,
            ).astype(np.float32)
            skeleton[..., :3] += noise

        return skeleton.reshape(original_shape).astype(np.float32, copy=False)

    def _sample_affine_matrix(self) -> np.ndarray:
        angle = np.deg2rad(
            np.random.uniform(-self.config.rotation_deg, self.config.rotation_deg)
        )
        shear = np.random.uniform(-self.config.shear, self.config.shear)
        scale = 1.0 + np.random.uniform(-self.config.scale, self.config.scale)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        shear_matrix = np.array([[1.0, shear], [0.0, 1.0]], dtype=np.float32)
        return (scale * rotation @ shear_matrix).astype(np.float32)


class GaussianNoiseAugmentation:
    def __init__(self, std: float, prob: float = 0.5) -> None:
        self.std = std
        self.prob = prob

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if self.std <= 0 or np.random.random() > self.prob:
            return features
        noise = np.random.normal(0.0, self.std, size=features.shape).astype(np.float32)
        return (features + noise).astype(np.float32, copy=False)
