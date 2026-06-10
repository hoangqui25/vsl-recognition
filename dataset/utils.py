from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


VIEW_NAMES = ("front_view", "left_view", "right_view")


def resolve_views(requested_views: list[str]) -> list[str]:
    if len(requested_views) == 1 and requested_views[0] == "all":
        return list(VIEW_NAMES)
    invalid = sorted(set(requested_views) - set(VIEW_NAMES))
    if invalid:
        raise ValueError(
            "Invalid view(s): "
            + ", ".join(invalid)
            + ". Expected one or more of: "
            + ", ".join(VIEW_NAMES)
            + ", or all"
        )
    return requested_views


def load_manifest_rows(feature_root: Path, splits: list[str], views: list[str] | str) -> list[dict]:
    if isinstance(views, str):
        views = [views]
    views = resolve_views(views)
    rows = []
    for split in splits:
        for view in views:
            manifest_path = feature_root / split / f"{view}.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")

            split_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in split_rows:
                feature_path = Path(row["feature_path"])
                if feature_path.is_file():
                    rows.append(row)

    if not rows:
        raise RuntimeError(f"No feature rows found for splits: {splits}")
    return rows


def build_label_map(rows: list[dict]) -> dict[str, int]:
    labels = sorted({row["gloss"] for row in rows})
    return {label: index for index, label in enumerate(labels)}


def validate_labels(rows: list[dict], label_to_idx: dict[str, int], split_name: str) -> None:
    unknown = sorted({row["gloss"] for row in rows if row["gloss"] not in label_to_idx})
    if unknown:
        preview = ", ".join(unknown[:10])
        raise ValueError(
            f"{split_name} contains {len(unknown)} labels not present in training data: {preview}"
        )


def infer_input_dim(rows: list[dict], flatten: bool = True) -> int:
    for row in rows:
        feature_shape = row.get("feature_shape")
        if feature_shape:
            if len(feature_shape) == 2:
                return int(feature_shape[1])
            if flatten and len(feature_shape) > 2:
                return int(np.prod(feature_shape[1:]))

        feature = np.load(row["feature_path"], mmap_mode="r")
        if feature.ndim == 2:
            return int(feature.shape[1])
        if flatten and feature.ndim > 2:
            return int(np.prod(feature.shape[1:]))
        raise ValueError(f"Expected sequence feature array, got {feature.shape}: {row['feature_path']}")

    raise RuntimeError("Cannot infer input_dim from empty rows")


def collate_features(batch):
    features, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(labels)

    max_len = int(lengths.max().item())
    feature_dim = features[0].shape[1]
    padded = torch.zeros(len(features), max_len, feature_dim, dtype=torch.float32)
    for index, item in enumerate(features):
        padded[index, : item.shape[0]] = item

    return padded, lengths, labels
