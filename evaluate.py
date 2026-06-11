from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    CNNFeatureDataset,
    SkeletonFeatureDataset,
    collate_features,
    load_manifest_rows,
    resolve_views,
    validate_labels,
)
from models import (
    DecoderTransformerClassifier,
    LSTMClassifier,
    RNNClassifier,
    TransformerClassifier,
)
from trainer import Trainer


DATASET_CLASSES = {
    "cnn": CNNFeatureDataset,
    "skeleton": SkeletonFeatureDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on selected feature splits."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["split_6", "split_7"],
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Views to evaluate. Default: views stored in checkpoint metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for a quick smoke test.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state_dict", "metadata"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {', '.join(sorted(missing))}"
        )
    return checkpoint


def build_model(metadata: dict[str, Any]) -> nn.Module:
    model_name = metadata["model"]
    common = {
        "input_dim": int(metadata["input_dim"]),
        "num_classes": int(metadata["num_classes"]),
        "hidden_dim": int(metadata["hidden_dim"]),
        "num_layers": int(metadata["num_layers"]),
        "dropout": float(metadata["dropout"]),
        "pooling": metadata["pooling"],
    }

    if model_name == "rnn":
        return RNNClassifier(
            **common,
            bidirectional=bool(metadata.get("bidirectional", True)),
            nonlinearity=metadata.get("nonlinearity", "tanh"),
        )
    if model_name == "lstm":
        return LSTMClassifier(
            **common,
            bidirectional=bool(metadata.get("bidirectional", True)),
        )
    if model_name == "transformer":
        return TransformerClassifier(
            **common,
            num_heads=int(metadata.get("num_heads", 8)),
            feedforward_dim=int(
                metadata.get("feedforward_dim", int(metadata["hidden_dim"]) * 4)
            ),
            max_len=int(metadata.get("max_len", 512)),
            position_encoding=metadata.get(
                "position_encoding",
                "sinusoidal",
            ),
        )
    if model_name == "transformer_decoder":
        return DecoderTransformerClassifier(
            input_dim=int(metadata["input_dim"]),
            num_classes=int(metadata["num_classes"]),
            hidden_dim=int(metadata["hidden_dim"]),
            num_layers=int(metadata["num_layers"]),
            decoder_layers=int(metadata.get("decoder_layers", 2)),
            num_heads=int(metadata.get("num_heads", 8)),
            feedforward_dim=int(
                metadata.get("feedforward_dim", int(metadata["hidden_dim"]) * 4)
            ),
            dropout=float(metadata["dropout"]),
            max_len=int(metadata.get("max_len", 512)),
            position_encoding=metadata.get(
                "position_encoding",
                "sinusoidal",
            ),
        )
    raise ValueError(f"Unsupported model in checkpoint: {model_name}")


def class_names(label_to_idx: dict[str, int]) -> list[str]:
    return [
        label
        for label, _ in sorted(label_to_idx.items(), key=lambda item: item[1])
    ]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    top1_correct = 0
    top5_correct = 0
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        progress = tqdm(loader, desc="test", dynamic_ncols=True, leave=True)
        for features, lengths, labels in progress:
            features = features.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(features, lengths)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total += batch_size
            top1_correct += (predictions == labels).sum().item()
            top_k = min(5, logits.size(1))
            top5_correct += (
                logits.topk(top_k, dim=1).indices == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

            progress.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{top1_correct / total:.4f}",
            )

    if total == 0:
        raise RuntimeError("Evaluation dataset is empty")
    return {
        "loss": total_loss / total,
        "top1_accuracy": top1_correct / total,
        "top5_accuracy": top5_correct / total,
        "labels": torch.cat(all_labels),
        "predictions": torch.cat(all_predictions),
    }


def confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    indices = labels.to(torch.long) * num_classes + predictions.to(torch.long)
    matrix = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return matrix.numpy()


def save_predictions(
    path: Path,
    rows: list[dict],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    names: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "video_id",
                "view",
                "true_label",
                "predicted_label",
                "correct",
                "feature_path",
            ),
        )
        writer.writeheader()
        for row, target, prediction in zip(
            rows,
            labels.tolist(),
            predictions.tolist(),
        ):
            writer.writerow(
                {
                    "video_id": row.get("video_id", ""),
                    "view": row.get("view", ""),
                    "true_label": names[target],
                    "predicted_label": names[prediction],
                    "correct": target == prediction,
                    "feature_path": row["feature_path"],
                }
            )


def print_metrics(metrics: dict[str, Any]) -> None:
    print(
        f"loss={metrics['loss']:.4f} "
        f"top1_accuracy={metrics['top1_accuracy']:.4f} "
        f"top5_accuracy={metrics['top5_accuracy']:.4f}"
    )
    print(
        "macro: "
        f"precision={metrics['macro']['precision']:.4f} "
        f"recall={metrics['macro']['recall']:.4f} "
        f"f1={metrics['macro']['f1']:.4f}"
    )
    print(
        "weighted: "
        f"precision={metrics['weighted']['precision']:.4f} "
        f"recall={metrics['weighted']['recall']:.4f} "
        f"f1={metrics['weighted']['f1']:.4f}"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    metadata = checkpoint["metadata"]

    feature_type = metadata["feature_type"]
    feature_root = args.feature_root or Path(metadata["feature_root"])
    views = resolve_views(args.views or metadata.get("views", ["front_view"]))
    label_to_idx = metadata["label_to_idx"]
    names = class_names(label_to_idx)

    rows = load_manifest_rows(feature_root, args.splits, views)
    validate_labels(rows, label_to_idx, "test")
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        rows = rows[: args.max_samples]

    dataset_class = DATASET_CLASSES[feature_type]
    dataset = dataset_class(rows, label_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_features,
    )

    model = build_model(metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    output_dir = args.output_dir or args.checkpoint.parent / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"device={device} model={metadata['model']}")
    print(f"feature_root={feature_root}")
    print(f"splits={','.join(args.splits)} views={','.join(views)}")
    print(f"samples={len(dataset)} classes={len(label_to_idx)}")

    evaluation = evaluate(model, loader, device)
    report = Trainer.classification_metrics(
        evaluation["labels"],
        evaluation["predictions"],
        len(label_to_idx),
        names,
    )
    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model": metadata["model"],
        "feature_type": feature_type,
        "feature_root": str(feature_root),
        "splits": args.splits,
        "views": views,
        "samples": len(dataset),
        "loss": evaluation["loss"],
        "top1_accuracy": evaluation["top1_accuracy"],
        "top5_accuracy": evaluation["top5_accuracy"],
        "macro": report["macro"],
        "weighted": report["weighted"],
    }

    print_metrics(metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(
        output_dir / "confusion_matrix.npy",
        confusion_matrix(
            evaluation["labels"],
            evaluation["predictions"],
            len(label_to_idx),
        ),
    )
    save_predictions(
        output_dir / "predictions.csv",
        rows,
        evaluation["labels"],
        evaluation["predictions"],
        names,
    )
    print(f"results={output_dir}")


if __name__ == "__main__":
    main()
