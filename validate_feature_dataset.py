from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


VIEW_NAMES = ("front_view", "left_view", "right_view")


@dataclass
class FeatureCheck:
    expected: int = 0
    manifest_rows: int = 0
    valid_files: int = 0
    missing_manifest: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    extra_manifest: list[str] = field(default_factory=list)
    unlisted_files: list[str] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)
    label_mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.missing_manifest,
                self.missing_files,
                self.unreadable_files,
                self.shape_mismatches,
                self.duplicate_ids,
                self.extra_manifest,
                self.unlisted_files,
                self.orphan_files,
                self.label_mismatches,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check labels and feature completeness across dataset splits."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--feature-root", type=Path, default=Path("feature/skeleton"))
    parser.add_argument(
        "--views",
        nargs="+",
        default=["front_view"],
        help="Views to check, or 'all'.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to check. Default: all split_* folders under data-root.",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=None,
        help="Require this number of frames in every feature.",
    )
    parser.add_argument(
        "--expected-feature-dim",
        type=int,
        default=None,
        help="Require this flattened feature dimension, e.g. 300.",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=10,
        help="Maximum examples printed for each error type.",
    )
    return parser.parse_args()


def resolve_views(views: list[str]) -> list[str]:
    if views == ["all"]:
        return list(VIEW_NAMES)
    invalid = sorted(set(views) - set(VIEW_NAMES))
    if invalid:
        raise ValueError(f"Invalid views: {', '.join(invalid)}")
    return views


def resolve_splits(data_root: Path, splits: list[str] | None) -> list[str]:
    if splits:
        return list(dict.fromkeys(splits))

    split_names = sorted(
        (path.name for path in data_root.glob("split_*") if path.is_dir()),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    if not split_names:
        raise RuntimeError(f"No split_* folders found under {data_root}")
    return split_names


def load_json(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must contain a JSON list: {path}")
    return rows


def rows_by_id(rows: list[dict]) -> tuple[dict[str, dict], list[str]]:
    indexed = {}
    duplicates = []
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in indexed:
            duplicates.append(video_id)
        indexed[video_id] = row
    return indexed, sorted(set(duplicates))


def feature_shape_error(
    feature_path: Path,
    row: dict,
    expected_frames: int | None,
    expected_feature_dim: int | None,
) -> str | None:
    feature = np.load(feature_path, mmap_mode="r")
    actual = tuple(feature.shape)
    manifest_shape = tuple(row.get("feature_shape", ()))

    if manifest_shape and actual != manifest_shape:
        return f"{feature_path}: actual={actual}, manifest={manifest_shape}"
    if expected_frames is not None and (not actual or actual[0] != expected_frames):
        return f"{feature_path}: shape={actual}, expected_frames={expected_frames}"
    if expected_feature_dim is not None:
        actual_dim = int(np.prod(actual[1:])) if len(actual) > 1 else 0
        if actual_dim != expected_feature_dim:
            return (
                f"{feature_path}: shape={actual}, "
                f"expected_feature_dim={expected_feature_dim}"
            )
    return None


def check_split_view(
    data_root: Path,
    feature_root: Path,
    split: str,
    view: str,
    expected_frames: int | None,
    expected_feature_dim: int | None,
) -> FeatureCheck:
    source_rows = load_json(data_root / split / f"{view}.json")
    feature_rows = load_json(feature_root / split / f"{view}.json")
    source_by_id, source_duplicates = rows_by_id(source_rows)
    feature_by_id, feature_duplicates = rows_by_id(feature_rows)

    source_ids = set(source_by_id)
    feature_ids = set(feature_by_id)
    feature_dir = feature_root / split / view
    file_ids = {path.stem for path in feature_dir.glob("*.npy")}

    result = FeatureCheck(
        expected=len(source_ids),
        manifest_rows=len(feature_rows),
        missing_manifest=sorted(source_ids - feature_ids),
        duplicate_ids=sorted(set(source_duplicates + feature_duplicates)),
        extra_manifest=sorted(feature_ids - source_ids),
        unlisted_files=sorted((file_ids & source_ids) - feature_ids),
        orphan_files=sorted(file_ids - source_ids),
    )
    for video_id in sorted(source_ids & feature_ids):
        source_label = source_by_id[video_id].get("gloss")
        feature_label = feature_by_id[video_id].get("gloss")
        if source_label != feature_label:
            result.label_mismatches.append(
                f"{video_id}: source={source_label!r}, feature={feature_label!r}"
            )

    for video_id in sorted(source_ids):
        feature_path = feature_dir / f"{video_id}.npy"
        if not feature_path.is_file():
            result.missing_files.append(video_id)
            continue
        if video_id not in feature_by_id:
            continue

        try:
            error = feature_shape_error(
                feature_path,
                feature_by_id[video_id],
                expected_frames,
                expected_feature_dim,
            )
        except Exception as exc:
            result.unreadable_files.append(f"{feature_path}: {exc}")
            continue

        if error:
            result.shape_mismatches.append(error)
        else:
            result.valid_files += 1
    return result


def collect_split_labels(
    data_root: Path,
    split: str,
    views: list[str],
) -> set[str]:
    labels = set()
    for view in views:
        rows = load_json(data_root / split / f"{view}.json")
        labels.update(row["gloss"] for row in rows)
    return labels


def infer_feature_dim(feature_root: Path, splits: list[str], views: list[str]) -> int:
    for split in splits:
        for view in views:
            feature_dir = feature_root / split / view
            for feature_path in sorted(feature_dir.glob("*.npy")):
                try:
                    shape = np.load(feature_path, mmap_mode="r").shape
                except Exception:
                    continue
                if len(shape) < 2:
                    continue
                return int(np.prod(shape[1:]))
    raise RuntimeError(f"Cannot infer feature dimension under {feature_root}")


def print_examples(name: str, values: list[str], limit: int) -> None:
    if not values:
        return
    print(f"  {name}: {len(values)}")
    for value in values[:limit]:
        print(f"    - {value}")
    if len(values) > limit:
        print(f"    ... and {len(values) - limit} more")


def print_feature_check(
    split: str,
    view: str,
    result: FeatureCheck,
    limit: int,
) -> None:
    status = "OK" if result.ok else "FAILED"
    print(
        f"[{status}] {split}/{view}: expected={result.expected} "
        f"manifest={result.manifest_rows} valid={result.valid_files}"
    )
    print_examples("missing_manifest", result.missing_manifest, limit)
    print_examples("missing_files", result.missing_files, limit)
    print_examples("unreadable_files", result.unreadable_files, limit)
    print_examples("shape_mismatches", result.shape_mismatches, limit)
    print_examples("duplicate_ids", result.duplicate_ids, limit)
    print_examples("extra_manifest", result.extra_manifest, limit)
    print_examples("unlisted_files", result.unlisted_files, limit)
    print_examples("orphan_files", result.orphan_files, limit)
    print_examples("label_mismatches", result.label_mismatches, limit)


def main() -> None:
    args = parse_args()
    views = resolve_views(args.views)
    splits = resolve_splits(args.data_root, args.splits)
    expected_feature_dim = args.expected_feature_dim
    if expected_feature_dim is None:
        expected_feature_dim = infer_feature_dim(args.feature_root, splits, views)

    all_features_ok = True
    print("Feature completeness")
    print(f"splits: {', '.join(splits)}")
    print(f"views: {', '.join(views)}")
    print(f"expected_feature_dim: {expected_feature_dim}")
    if args.expected_frames is not None:
        print(f"expected_frames: {args.expected_frames}")
    for split in splits:
        for view in views:
            result = check_split_view(
                args.data_root,
                args.feature_root,
                split,
                view,
                args.expected_frames,
                expected_feature_dim,
            )
            print_feature_check(split, view, result, args.show_limit)
            all_features_ok = all_features_ok and result.ok

    labels_by_split = {
        split: collect_split_labels(args.data_root, split, views)
        for split in splits
    }
    all_labels = set().union(*labels_by_split.values())
    labels_ok = True
    print("\nLabel coverage")
    print(f"all_label_count: {len(all_labels)}")
    for split, labels in labels_by_split.items():
        missing_labels = sorted(all_labels - labels)
        print(f"{split}: label_count={len(labels)}")
        print_examples("missing_labels", missing_labels, args.show_limit)
        labels_ok = labels_ok and not missing_labels

    if all_features_ok and labels_ok:
        print("\nOK: labels and feature files are complete")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
