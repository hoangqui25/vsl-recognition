from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from evaluate import build_model, load_checkpoint, resolve_device


EXTRACTOR_PATH = Path("feature-extractor/skeleton_feature_extraction.py")
DEFAULT_MEDIAPIPE_MODEL = Path("models/mediapipe/holistic_landmarker.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a VSL gloss directly from one video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-asset-path",
        type=Path,
        default=DEFAULT_MEDIAPIPE_MODEL,
        help="MediaPipe holistic_landmarker.task file.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of uniformly sampled video frames.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference_results"),
        help=(
            "Directory for prediction JSON and extracted skeleton. "
            "Files are named from the input video. Default: inference_results"
        ),
    )
    parser.add_argument(
        "--save-feature",
        type=Path,
        default=None,
        help="Optional path for saving the extracted skeleton as .npy.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for saving prediction results.",
    )
    return parser.parse_args()


def load_skeleton_extractor() -> ModuleType:
    if not EXTRACTOR_PATH.is_file():
        raise FileNotFoundError(f"Missing extractor: {EXTRACTOR_PATH}")

    module_name = "_skeleton_feature_extraction"
    spec = importlib.util.spec_from_file_location(module_name, EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load extractor: {EXTRACTOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_landmarker(model_asset_path: Path):
    if not model_asset_path.is_file():
        raise FileNotFoundError(f"Missing MediaPipe model: {model_asset_path}")

    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_asset_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    return landmarker, mp


def extract_skeleton(
    video_path: Path,
    model_asset_path: Path,
    num_frames: int,
) -> np.ndarray:
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video: {video_path}")

    extractor = load_skeleton_extractor()
    landmarker, mp = create_landmarker(model_asset_path)
    with landmarker:
        features = extractor.extract_video_skeleton_tasks(
            landmarker=landmarker,
            video_path=video_path,
            num_frames=num_frames,
            flatten=True,
            mp_module=mp,
        )
    return features.astype(np.float32, copy=False)


def detection_statistics(features: np.ndarray) -> dict[str, float]:
    skeleton = features.reshape(features.shape[0], 75, 4)
    pose = skeleton[:, :33]
    left_hand = skeleton[:, 33:54]
    right_hand = skeleton[:, 54:75]
    return {
        "pose_detected_ratio": float((pose[..., 3] > 0).mean()),
        "left_hand_detected_ratio": float((left_hand[..., 3] > 0).mean()),
        "right_hand_detected_ratio": float((right_hand[..., 3] > 0).mean()),
    }


def index_to_label(label_to_idx: dict[str, int]) -> list[str]:
    labels = [""] * len(label_to_idx)
    for label, index in label_to_idx.items():
        labels[int(index)] = label
    if any(not label for label in labels):
        raise ValueError("Checkpoint contains an invalid label_to_idx mapping")
    return labels


def predict(
    checkpoint_path: Path,
    features: np.ndarray,
    device: torch.device,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    metadata = checkpoint["metadata"]
    if metadata.get("feature_type") != "skeleton":
        raise ValueError(
            "This script requires a checkpoint trained with skeleton features"
        )
    if features.ndim != 2:
        raise ValueError(f"Expected feature shape (frames, dim), got {features.shape}")
    if features.shape[1] != int(metadata["input_dim"]):
        raise ValueError(
            f"Feature dim {features.shape[1]} does not match checkpoint "
            f"input_dim {metadata['input_dim']}"
        )

    model = build_model(metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    inputs = torch.from_numpy(features).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(inputs, lengths)
        probabilities = torch.softmax(logits, dim=1)[0]

    labels = index_to_label(metadata["label_to_idx"])
    count = min(top_k, len(labels))
    scores, indices = probabilities.topk(count)
    predictions = [
        {
            "rank": rank,
            "label": labels[index],
            "class_index": index,
            "confidence": score,
        }
        for rank, (index, score) in enumerate(
            zip(indices.tolist(), scores.tolist()),
            start=1,
        )
    ]
    return predictions, metadata


def main() -> None:
    args = parse_args()
    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_feature is None:
        args.save_feature = args.output_dir / f"{args.video.stem}_skeleton.npy"
    if args.output_json is None:
        args.output_json = args.output_dir / f"{args.video.stem}_prediction.json"

    features = extract_skeleton(
        video_path=args.video,
        model_asset_path=args.model_asset_path,
        num_frames=args.num_frames,
    )
    predictions, metadata = predict(
        checkpoint_path=args.checkpoint,
        features=features,
        device=device,
        top_k=args.top_k,
    )
    statistics = detection_statistics(features)

    result = {
        "video": str(args.video),
        "checkpoint": str(args.checkpoint),
        "model": metadata["model"],
        "device": str(device),
        "feature_shape": list(features.shape),
        "detection": statistics,
        "prediction": predictions[0],
        "top_k": predictions,
    }

    print(f"prediction={predictions[0]['label']}")
    print(f"confidence={predictions[0]['confidence']:.4f}")
    print("\nTop predictions:")
    for item in predictions:
        print(
            f"{item['rank']:>2}. {item['label']:<30} "
            f"{item['confidence']:.4f}"
        )
    print(
        "\nDetection: "
        f"pose={statistics['pose_detected_ratio']:.3f} "
        f"left_hand={statistics['left_hand_detected_ratio']:.3f} "
        f"right_hand={statistics['right_hand_detected_ratio']:.3f}"
    )

    if args.save_feature is not None:
        args.save_feature.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_feature, features)
        print(f"feature={args.save_feature}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"result={args.output_json}")
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
