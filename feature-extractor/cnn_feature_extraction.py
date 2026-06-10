#!/usr/bin/env python3
"""Extract CNN frame features for VSL split videos."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VIEW_NAMES = ("front_view", "left_view", "right_view")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class VideoItem:
    split_name: str
    view: str
    metadata: dict
    video_path: Path
    feature_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-video CNN features for split videos in "
            "data/split_*/ and save them to feature/cnn/."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=Path("feature/cnn"))
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Split names to process, e.g. split_1 split_2. Default: all split_* folders.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["front_view"],
        help="Views to process: front_view left_view right_view, or all. Default: front_view.",
    )
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large"),
        default="resnet50",
        help="Torchvision CNN backbone used as a feature extractor.",
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet pretrained weights or random weights.",
    )
    parser.add_argument(
        "--num-frames",
        "--frames-per-video",
        dest="num_frames",
        type=int,
        default=16,
        help="Uniformly sampled frames per video. Use 0 or a negative value for all frames.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Log progress every N videos while extracting.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or any torch device string.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only create view manifests from existing .npy feature files.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a video cannot be decoded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print how many videos would be processed.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def iter_split_dirs(data_root: Path, requested_splits: list[str] | None) -> list[Path]:
    if requested_splits:
        split_dirs = [data_root / split_name for split_name in requested_splits]
    else:
        split_dirs = sorted(
            (path for path in data_root.glob("split_*") if path.is_dir()),
            key=lambda path: int(path.name.split("_")[-1]),
        )

    missing = [path for path in split_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing split folders: " + ", ".join(str(path) for path in missing)
        )
    return split_dirs


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


def load_video_items(
    data_root: Path,
    out_root: Path,
    requested_splits: list[str] | None,
    views: list[str],
) -> list[VideoItem]:
    items: list[VideoItem] = []
    for split_dir in iter_split_dirs(data_root, requested_splits):
        for view in views:
            metadata_path = split_dir / f"{view}.json"
            video_dir = split_dir / view

            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
            if not video_dir.is_dir():
                raise FileNotFoundError(f"Missing video folder: {video_dir}")

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for row in metadata:
                video_id = row["video_id"]
                items.append(
                    VideoItem(
                        split_name=split_dir.name,
                        view=view,
                        metadata=row,
                        video_path=video_dir / f"{video_id}.mp4",
                        feature_path=out_root / split_dir.name / view / f"{video_id}.npy",
                    )
                )
    return items


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_feature_extractor(model_name: str, weights_name: str, device):
    import torch.nn as nn
    from torchvision import models

    use_weights = weights_name == "imagenet"

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if use_weights else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        extractor = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if use_weights else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        extractor = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_weights else None
        model = models.mobilenet_v3_small(weights=weights)
        feature_dim = model.classifier[0].in_features
        extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if use_weights else None
        model = models.mobilenet_v3_large(weights=weights)
        feature_dim = model.classifier[0].in_features
        extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    extractor.eval()
    extractor.to(device)
    return extractor, feature_dim


def uniform_indices(frame_count: int, num_frames: int) -> np.ndarray | None:
    if num_frames <= 0:
        return None
    if frame_count <= 0:
        return np.arange(num_frames, dtype=np.int64)
    if frame_count == 1:
        return np.zeros(num_frames, dtype=np.int64)
    return np.linspace(0, frame_count - 1, num_frames).round().astype(np.int64)


def read_sampled_frames(video_path: Path, num_frames: int, image_size: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = uniform_indices(frame_count, num_frames)
        frames = []

        if indices is None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(preprocess_frame(frame, image_size))
        else:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    continue
                frames.append(preprocess_frame(frame, image_size))

        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")
        return np.stack(frames, axis=0)
    finally:
        cap.release()


def preprocess_frame(frame_bgr: np.ndarray, image_size: int) -> np.ndarray:
    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = resize_short_side_and_center_crop(frame_rgb, image_size)
    frame = frame_rgb.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(frame, (2, 0, 1))


def resize_short_side_and_center_crop(frame: np.ndarray, image_size: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    scale = image_size / min(height, width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    top = max((new_height - image_size) // 2, 0)
    left = max((new_width - image_size) // 2, 0)
    return resized[top : top + image_size, left : left + image_size]


def extract_video_features(
    extractor,
    video_path: Path,
    num_frames: int,
    image_size: int,
    batch_size: int,
    device,
) -> np.ndarray:
    import torch

    frames = read_sampled_frames(video_path, num_frames, image_size)
    features = []

    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch_np = frames[start : start + batch_size]
            batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            output = extractor(batch).flatten(1)
            features.append(output.cpu().numpy())

    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def write_split_manifests(out_root: Path, items: Iterable[VideoItem]) -> None:
    by_split_view: dict[tuple[str, str], list[VideoItem]] = {}
    for item in items:
        by_split_view.setdefault((item.split_name, item.view), []).append(item)

    for (split_name, view), split_items in by_split_view.items():
        manifest = []
        for item in split_items:
            if not item.feature_path.exists():
                continue
            row = dict(item.metadata)
            row["view"] = item.view
            row["video_path"] = str(item.video_path)
            row["feature_path"] = str(item.feature_path)
            row["feature_shape"] = list(np.load(item.feature_path, mmap_mode="r").shape)
            manifest.append(row)

        manifest_path = out_root / split_name / f"{view}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_failed_videos(out_root: Path, failures: list[dict]) -> None:
    if not failures:
        return
    failed_path = out_root / "failed_videos.json"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    views = resolve_views(args.views)
    items = load_video_items(args.data_root, args.out_root, args.splits, views)
    to_process = [
        item
        for item in items
        if args.overwrite or not item.feature_path.exists()
    ]

    missing_videos = [item.video_path for item in to_process if not item.video_path.is_file()]
    if missing_videos:
        preview = ", ".join(str(path) for path in missing_videos[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} video(s). First paths: {preview}"
        )

    logging.info("Views: %s", ", ".join(views))
    logging.info("Found %d videos", len(items))
    logging.info("Need to process %d videos", len(to_process))
    logging.info("Output root: %s", args.out_root)

    if args.manifest_only:
        write_split_manifests(args.out_root, items)
        logging.info("Finished writing manifests from existing features")
        return

    if args.dry_run:
        return

    device = resolve_device(args.device)
    extractor, feature_dim = build_feature_extractor(args.model, args.weights, device)
    logging.info(
        "Using %s (%s weights) on %s, feature_dim=%d",
        args.model,
        args.weights,
        device,
        feature_dim,
    )

    failures = []
    for index, item in enumerate(to_process, start=1):
        item.feature_path.parent.mkdir(parents=True, exist_ok=True)
        should_log = (
            index == 1
            or index == len(to_process)
            or (args.progress_every > 0 and index % args.progress_every == 0)
        )
        if should_log:
            logging.info(
                "[%d/%d] %s -> %s",
                index,
                len(to_process),
                item.video_path,
                item.feature_path,
            )
        try:
            features = extract_video_features(
                extractor=extractor,
                video_path=item.video_path,
                num_frames=args.num_frames,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=device,
            )
            np.save(item.feature_path, features)
        except Exception as exc:
            failure = {
                "split": item.split_name,
                "view": item.view,
                "video_id": item.metadata.get("video_id"),
                "video_path": str(item.video_path),
                "feature_path": str(item.feature_path),
                "error": str(exc),
            }
            failures.append(failure)
            write_failed_videos(args.out_root, failures)
            logging.exception("Skipping unreadable video: %s", item.video_path)
            if args.stop_on_error:
                raise

        if should_log:
            write_split_manifests(args.out_root, items)
            write_failed_videos(args.out_root, failures)

    write_split_manifests(args.out_root, items)
    write_failed_videos(args.out_root, failures)
    if failures:
        logging.warning("Skipped %d video(s). See %s", len(failures), args.out_root / "failed_videos.json")
    logging.info("Finished CNN feature extraction")


if __name__ == "__main__":
    main()
