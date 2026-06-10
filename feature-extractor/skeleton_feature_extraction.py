#!/usr/bin/env python3
"""Extract MediaPipe skeleton features for VSL split videos."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    class _NoOpTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable, **kwargs):
        return _NoOpTqdm(iterable, **kwargs)


VIEW_NAMES = ("front_view", "left_view", "right_view")
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
TOTAL_LANDMARKS = POSE_LANDMARKS + HAND_LANDMARKS + HAND_LANDMARKS
COORD_DIM = 4


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
            "Extract MediaPipe pose+hand skeleton features for split videos "
            "in data/split_*/ and save them to feature/skeleton/."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=Path("feature/skeleton"))
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
        "--num-frames",
        "--frames-per-video",
        dest="num_frames",
        type=int,
        default=16,
        help="Uniformly sampled frames per video. Use 0 or a negative value for all frames.",
    )
    parser.add_argument(
        "--flatten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save features as (frames, 300). Disable for (frames, 75, 4).",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="MediaPipe legacy Holistic model complexity.",
    )
    parser.add_argument(
        "--api",
        choices=("auto", "tasks", "legacy"),
        default="auto",
        help="MediaPipe API to use. Newer MediaPipe versions use tasks.",
    )
    parser.add_argument(
        "--model-asset-path",
        type=Path,
        default=None,
        help="Path to holistic_landmarker.task for MediaPipe Tasks API.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Update manifests every N processed videos.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bar.",
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
        help="Stop immediately when a video cannot be decoded or processed.",
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


def uniform_indices(frame_count: int, num_frames: int) -> np.ndarray | None:
    if num_frames <= 0:
        return None
    if frame_count <= 0:
        return np.arange(num_frames, dtype=np.int64)
    if frame_count == 1:
        return np.zeros(num_frames, dtype=np.int64)
    return np.linspace(0, frame_count - 1, num_frames).round().astype(np.int64)


def read_sampled_rgb_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
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
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")
        return frames
    finally:
        cap.release()


def landmarks_to_array(landmarks, count: int, has_visibility: bool) -> np.ndarray:
    output = np.zeros((count, COORD_DIM), dtype=np.float32)
    if landmarks is None:
        return output

    landmark_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    for index, landmark in enumerate(landmark_list[:count]):
        output[index, 0] = landmark.x
        output[index, 1] = landmark.y
        output[index, 2] = landmark.z
        output[index, 3] = landmark.visibility if has_visibility else 1.0
    return output


def extract_frame_skeleton(holistic, frame_rgb: np.ndarray) -> np.ndarray:
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)
    return skeleton_from_results(results)


def extract_frame_skeleton_tasks(landmarker, frame_rgb: np.ndarray, mp_module) -> np.ndarray:
    image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=frame_rgb)
    results = landmarker.detect(image)
    return skeleton_from_results(results)


def skeleton_from_results(results) -> np.ndarray:
    pose = landmarks_to_array(results.pose_landmarks, POSE_LANDMARKS, has_visibility=True)
    left_hand = landmarks_to_array(
        results.left_hand_landmarks,
        HAND_LANDMARKS,
        has_visibility=False,
    )
    right_hand = landmarks_to_array(
        results.right_hand_landmarks,
        HAND_LANDMARKS,
        has_visibility=False,
    )
    return np.concatenate([pose, left_hand, right_hand], axis=0)


def extract_video_skeleton(
    holistic,
    video_path: Path,
    num_frames: int,
    flatten: bool,
) -> np.ndarray:
    frames = read_sampled_rgb_frames(video_path, num_frames)
    skeleton = np.stack(
        [extract_frame_skeleton(holistic, frame) for frame in frames],
        axis=0,
    ).astype(np.float32, copy=False)

    if flatten:
        return skeleton.reshape(skeleton.shape[0], TOTAL_LANDMARKS * COORD_DIM)
    return skeleton


def extract_video_skeleton_tasks(
    landmarker,
    video_path: Path,
    num_frames: int,
    flatten: bool,
    mp_module,
) -> np.ndarray:
    frames = read_sampled_rgb_frames(video_path, num_frames)
    skeleton = np.stack(
        [extract_frame_skeleton_tasks(landmarker, frame, mp_module) for frame in frames],
        axis=0,
    ).astype(np.float32, copy=False)

    if flatten:
        return skeleton.reshape(skeleton.shape[0], TOTAL_LANDMARKS * COORD_DIM)
    return skeleton


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
            row["feature_type"] = "mediapipe_pose_hands"
            row["landmarks"] = {
                "pose": POSE_LANDMARKS,
                "left_hand": HAND_LANDMARKS,
                "right_hand": HAND_LANDMARKS,
                "coord_dim": COORD_DIM,
                "coord_order": ["x", "y", "z", "visibility_or_presence"],
            }
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
        item for item in items if args.overwrite or not item.feature_path.exists()
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
    logging.info(
        "Output shape: %s",
        f"(frames, {TOTAL_LANDMARKS * COORD_DIM})"
        if args.flatten
        else f"(frames, {TOTAL_LANDMARKS}, {COORD_DIM})",
    )

    if args.manifest_only:
        write_split_manifests(args.out_root, items)
        logging.info("Finished writing manifests from existing skeleton features")
        return

    if args.dry_run:
        return

    use_tasks_api = False
    mp_holistic = None
    mp_module = None
    vision = None
    BaseOptions = None

    if args.api in {"auto", "tasks"}:
        try:
            import mediapipe as mp_module
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            use_tasks_api = True
        except ModuleNotFoundError:
            if args.api == "tasks":
                raise

    if not use_tasks_api:
        try:
            mp_holistic = importlib.import_module("mediapipe.solutions.holistic")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "No supported MediaPipe API found. For new MediaPipe, use "
                "--api tasks --model-asset-path path/to/holistic_landmarker.task. "
                "For legacy MediaPipe, install a version that provides "
                "mediapipe.solutions.holistic."
            ) from exc

    failures = []
    if use_tasks_api:
        if args.model_asset_path is None:
            raise ValueError(
                "MediaPipe Tasks API requires --model-asset-path "
                "pointing to holistic_landmarker.task."
            )
        if not args.model_asset_path.is_file():
            raise FileNotFoundError(f"Missing model asset: {args.model_asset_path}")

        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.model_asset_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_pose_landmarks_confidence=args.min_tracking_confidence,
            min_hand_landmarks_confidence=args.min_tracking_confidence,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        detector_context = vision.HolisticLandmarker.create_from_options(options)
        logging.info("Using MediaPipe Tasks HolisticLandmarker API")
    else:
        detector_context = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        logging.info("Using MediaPipe legacy solutions.holistic API")

    with detector_context as holistic:
        progress = tqdm(
            to_process,
            desc="skeleton",
            dynamic_ncols=True,
            leave=True,
            position=0,
            disable=not args.progress,
        )
        for index, item in enumerate(progress, start=1):
            item.feature_path.parent.mkdir(parents=True, exist_ok=True)
            progress.set_postfix(
                split=item.split_name,
                view=item.view,
                video=item.metadata.get("video_id"),
            )

            try:
                if use_tasks_api:
                    features = extract_video_skeleton_tasks(
                        landmarker=holistic,
                        video_path=item.video_path,
                        num_frames=args.num_frames,
                        flatten=args.flatten,
                        mp_module=mp_module,
                    )
                else:
                    features = extract_video_skeleton(
                        holistic=holistic,
                        video_path=item.video_path,
                        num_frames=args.num_frames,
                        flatten=args.flatten,
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

            should_update = (
                index == 1
                or index == len(to_process)
                or (args.progress_every > 0 and index % args.progress_every == 0)
            )
            if should_update:
                write_split_manifests(args.out_root, items)
                write_failed_videos(args.out_root, failures)

    write_split_manifests(args.out_root, items)
    write_failed_videos(args.out_root, failures)
    if failures:
        logging.warning(
            "Skipped %d video(s). See %s",
            len(failures),
            args.out_root / "failed_videos.json",
        )
    logging.info("Finished skeleton feature extraction")


if __name__ == "__main__":
    main()
