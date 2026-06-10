from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether all labels in validation splits exist in training splits."
    )
    parser.add_argument("--feature-root", type=Path, default=Path("feature/skeleton"))
    parser.add_argument("--view", default="front_view")
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["split_1", "split_2", "split_3", "split_4"],
    )
    parser.add_argument("--val-splits", nargs="+", default=["split_5"])
    return parser.parse_args()


def load_labels(feature_root: Path, splits: list[str], view: str) -> set[str]:
    labels = set()
    for split in splits:
        manifest_path = feature_root / split / f"{view}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels.update(row["gloss"] for row in rows)
    return labels


def main() -> None:
    args = parse_args()

    train_labels = load_labels(args.feature_root, args.train_splits, args.view)
    val_labels = load_labels(args.feature_root, args.val_splits, args.view)

    missing = sorted(val_labels - train_labels)
    extra_train = sorted(train_labels - val_labels)

    print(f"feature_root: {args.feature_root}")
    print(f"train_splits: {', '.join(args.train_splits)}")
    print(f"val_splits: {', '.join(args.val_splits)}")
    print(f"train_label_count: {len(train_labels)}")
    print(f"val_label_count: {len(val_labels)}")

    if missing:
        print(f"missing_in_train_count: {len(missing)}")
        for label in missing:
            print(f"missing_in_train: {label}")
    else:
        print("OK: all validation labels exist in training labels")

    if extra_train:
        print(f"extra_train_label_count: {len(extra_train)}")


if __name__ == "__main__":
    main()
