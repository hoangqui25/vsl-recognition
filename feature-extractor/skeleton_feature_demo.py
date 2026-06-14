#!/usr/bin/env python3
"""Visualize extracted skeleton features on their source video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


POSE_COUNT = 33
HAND_COUNT = 21
TOTAL_LANDMARKS = POSE_COUNT + HAND_COUNT * 2

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw extracted skeleton landmarks over sampled video frames."
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("feature/skeleton_8"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="split_1")
    parser.add_argument(
        "--view",
        choices=("front_view", "left_view", "right_view"),
        default="front_view",
    )
    parser.add_argument("--video-id", default="000000")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo/skeleton"),
        help="Directory for generated demo files. Default: demo/skeleton",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional MP4 path overriding --output-dir.",
    )
    parser.add_argument(
        "--frames-output",
        type=Path,
        default=None,
        help="Output image containing all sampled frames. Default: <output>_frames.png",
    )
    parser.add_argument(
        "--frame-columns",
        type=int,
        default=4,
        help="Number of columns in the sampled-frame image.",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=360,
        help="Width of each frame in the sampled-frame image.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--point-radius", type=int, default=3)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.1,
        help="Minimum pose visibility for drawing body landmarks.",
    )
    return parser.parse_args()


def feature_path(args: argparse.Namespace) -> Path:
    return (
        args.feature_root
        / args.split
        / args.view
        / f"{args.video_id}.npy"
    )


def video_path(args: argparse.Namespace) -> Path:
    return args.data_root / args.split / args.view / f"{args.video_id}.mp4"


def load_skeleton(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing skeleton feature: {path}")

    skeleton = np.load(path).astype(np.float32, copy=False)
    if skeleton.ndim == 2:
        expected_dim = TOTAL_LANDMARKS * 4
        if skeleton.shape[1] != expected_dim:
            raise ValueError(
                f"Expected flattened feature dim {expected_dim}, got {skeleton.shape}"
            )
        skeleton = skeleton.reshape(skeleton.shape[0], TOTAL_LANDMARKS, 4)

    if skeleton.ndim != 3 or skeleton.shape[1:] != (TOTAL_LANDMARKS, 4):
        raise ValueError(
            f"Expected feature shape (frames, {TOTAL_LANDMARKS}, 4), got {skeleton.shape}"
        )
    return skeleton


def read_uniform_frames(path: Path, frame_count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Invalid frame count for video: {path}")

        indices = np.linspace(0, total_frames - 1, frame_count).round().astype(int)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {path}")
            frames.append(frame)
        return frames
    finally:
        cap.release()


def is_valid(point: np.ndarray, is_pose: bool, min_visibility: float) -> bool:
    x, y, _, presence = point
    if presence <= (min_visibility if is_pose else 0):
        return False
    return np.isfinite(x) and np.isfinite(y) and x != 0 and y != 0


def pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(round(float(point[0]) * width))
    y = int(round(float(point[1]) * height))
    return x, y


def draw_group(
    frame: np.ndarray,
    landmarks: np.ndarray,
    connections: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    is_pose: bool,
    min_visibility: float,
    point_radius: int,
    line_thickness: int,
) -> None:
    height, width = frame.shape[:2]

    for start, end in connections:
        if (
            is_valid(landmarks[start], is_pose, min_visibility)
            and is_valid(landmarks[end], is_pose, min_visibility)
        ):
            cv2.line(
                frame,
                pixel(landmarks[start], width, height),
                pixel(landmarks[end], width, height),
                color,
                line_thickness,
                cv2.LINE_AA,
            )

    for point in landmarks:
        if is_valid(point, is_pose, min_visibility):
            cv2.circle(
                frame,
                pixel(point, width, height),
                point_radius,
                color,
                -1,
                cv2.LINE_AA,
            )


def draw_skeleton(
    frame: np.ndarray,
    skeleton: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    output = frame.copy()
    pose = skeleton[:POSE_COUNT]
    left_hand = skeleton[POSE_COUNT : POSE_COUNT + HAND_COUNT]
    right_hand = skeleton[POSE_COUNT + HAND_COUNT :]

    draw_group(
        output,
        pose,
        POSE_CONNECTIONS,
        (0, 220, 0),
        True,
        args.min_visibility,
        args.point_radius,
        args.line_thickness,
    )
    draw_group(
        output,
        left_hand,
        HAND_CONNECTIONS,
        (255, 120, 0),
        False,
        args.min_visibility,
        args.point_radius,
        args.line_thickness,
    )
    draw_group(
        output,
        right_hand,
        HAND_CONNECTIONS,
        (0, 80, 255),
        False,
        args.min_visibility,
        args.point_radius,
        args.line_thickness,
    )
    return output


def save_video(frames: list[np.ndarray], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def save_frame_grid(
    frames: list[np.ndarray],
    output: Path,
    columns: int,
    thumbnail_width: int,
) -> None:
    if not frames:
        raise ValueError("Cannot create a frame grid without frames")
    if columns < 1:
        raise ValueError("--frame-columns must be positive")
    if thumbnail_width < 1:
        raise ValueError("--thumbnail-width must be positive")

    source_height, source_width = frames[0].shape[:2]
    thumbnail_height = round(source_height * thumbnail_width / source_width)
    label_height = 34
    rows = (len(frames) + columns - 1) // columns
    canvas = np.full(
        (
            rows * (thumbnail_height + label_height),
            columns * thumbnail_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        x = column * thumbnail_width
        y = row * (thumbnail_height + label_height)
        thumbnail = cv2.resize(
            frame,
            (thumbnail_width, thumbnail_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas[y : y + thumbnail_height, x : x + thumbnail_width] = thumbnail
        label = f"Frame {index + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )
        text_x = x + (thumbnail_width - text_width) // 2
        text_y = y + thumbnail_height + (label_height + text_height) // 2
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not save sampled-frame image: {output}")


def print_statistics(skeleton: np.ndarray) -> None:
    pose = skeleton[:, :POSE_COUNT]
    left = skeleton[:, POSE_COUNT : POSE_COUNT + HAND_COUNT]
    right = skeleton[:, POSE_COUNT + HAND_COUNT :]

    stats = {
        "shape": list(skeleton.shape),
        "pose_detected_ratio": float((pose[..., 3] > 0).mean()),
        "left_hand_detected_ratio": float((left[..., 3] > 0).mean()),
        "right_hand_detected_ratio": float((right[..., 3] > 0).mean()),
    }
    print(json.dumps(stats, indent=2))


def main() -> None:
    args = parse_args()
    source_feature = feature_path(args)
    source_video = video_path(args)
    output = args.output or (
        args.output_dir
        / f"{args.split}_{args.view}_{args.video_id}.mp4"
    )
    frames_output = args.frames_output or output.with_name(
        f"{output.stem}_frames.png"
    )

    skeleton = load_skeleton(source_feature)
    frames = read_uniform_frames(source_video, len(skeleton))
    rendered = [
        draw_skeleton(frame, frame_skeleton, args)
        for frame, frame_skeleton in zip(frames, skeleton)
    ]
    save_video(rendered, output, args.fps)
    save_frame_grid(
        rendered,
        frames_output,
        columns=args.frame_columns,
        thumbnail_width=args.thumbnail_width,
    )

    print_statistics(skeleton)
    print(f"video: {source_video}")
    print(f"feature: {source_feature}")
    print(f"output: {output}")
    print(f"frames: {frames_output}")


if __name__ == "__main__":
    main()
