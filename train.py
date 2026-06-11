from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    CNNFeatureDataset,
    FeatureAugmentationConfig,
    GaussianNoiseAugmentation,
    SkeletonAugmentation,
    SkeletonFeatureDataset,
    build_label_map,
    collate_features,
    infer_input_dim,
    load_manifest_rows,
    resolve_views,
    validate_labels,
)
from models import (
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
        description="Train sequence classifiers on pre-extracted features."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--feature-type", choices=DATASET_CLASSES, default="cnn")
    data.add_argument("--feature-root", type=Path)
    data.add_argument("--views", nargs="+", default=["front_view"])
    data.add_argument(
        "--train-splits",
        nargs="+",
        default=["split_1", "split_2", "split_3", "split_4"],
    )
    data.add_argument("--val-splits", nargs="+", default=["split_5"])
    data.add_argument("--num-workers", type=int, default=2)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        choices=("rnn", "lstm", "transformer"),
        default="lstm",
    )
    model.add_argument("--hidden-dim", type=int, default=256)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.3)
    model.add_argument("--pooling", choices=("last", "mean", "max", "cls"), default="cls")
    model.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument(
        "--rnn-nonlinearity",
        choices=("tanh", "relu"),
        default="tanh",
    )
    model.add_argument("--num-heads", type=int, default=8)
    model.add_argument("--feedforward-dim", type=int, default=1024)
    model.add_argument("--max-len", type=int, default=512)
    training = parser.add_argument_group("training")
    training.add_argument("--output-dir", type=Path, default=Path("checkpoints/run"))
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument(
        "--scheduler",
        choices=("cosine", "plateau", "none"),
        default="none",
        help="Learning-rate scheduler. Default: none.",
    )
    training.add_argument("--plateau-factor", type=float, default=0.5)
    training.add_argument("--plateau-patience", type=int, default=3)
    training.add_argument("--min-lr", type=float, default=1e-6)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="auto")

    augmentation = parser.add_argument_group("augmentation")
    augmentation.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    augmentation.add_argument("--aug-prob", type=float, default=0.5)
    augmentation.add_argument("--rotation-deg", type=float, default=10.0)
    augmentation.add_argument("--shear", type=float, default=0.08)
    augmentation.add_argument("--scale", type=float, default=0.10)
    augmentation.add_argument("--gaussian-noise-std", type=float, default=0.01)

    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.feature_root = args.feature_root or Path("feature") / args.feature_type
    args.views = resolve_views(args.views)

    if args.model in {"rnn", "lstm"} and args.pooling == "cls":
        raise ValueError(f"{args.model} does not support pooling='cls'")
    if args.model == "transformer" and args.pooling == "last":
        raise ValueError(f"{args.model} does not support pooling='last'")
    if not 0.0 < args.plateau_factor < 1.0:
        raise ValueError("--plateau-factor must be between 0 and 1")
    if args.plateau_patience < 0:
        raise ValueError("--plateau-patience must be non-negative")
    if args.min_lr < 0:
        raise ValueError("--min-lr must be non-negative")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def build_transform(args: argparse.Namespace):
    if not args.augment:
        return None
    if args.feature_type == "cnn":
        return GaussianNoiseAugmentation(args.gaussian_noise_std, args.aug_prob)

    config = FeatureAugmentationConfig(
        enabled=True,
        prob=args.aug_prob,
        rotation_deg=args.rotation_deg,
        shear=args.shear,
        scale=args.scale,
        gaussian_noise_std=args.gaussian_noise_std,
    )
    return SkeletonAugmentation(config)


def build_datasets(args: argparse.Namespace):
    train_rows = load_manifest_rows(args.feature_root, args.train_splits, args.views)
    val_rows = load_manifest_rows(args.feature_root, args.val_splits, args.views)

    label_to_idx = build_label_map(train_rows)
    validate_labels(val_rows, label_to_idx, "validation")

    dataset_class = DATASET_CLASSES[args.feature_type]
    train_dataset = dataset_class(
        train_rows,
        label_to_idx,
        transform=build_transform(args),
    )
    val_dataset = dataset_class(val_rows, label_to_idx)
    return train_dataset, val_dataset, label_to_idx, infer_input_dim(train_rows)


def build_loader(dataset, args: argparse.Namespace, device: torch.device, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_features,
    )


def build_model(args: argparse.Namespace, input_dim: int, num_classes: int) -> nn.Module:
    common = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pooling": args.pooling,
    }
    if args.model == "rnn":
        return RNNClassifier(
            **common,
            bidirectional=args.bidirectional,
            nonlinearity=args.rnn_nonlinearity,
        )
    if args.model == "lstm":
        return LSTMClassifier(**common, bidirectional=args.bidirectional)
    transformer_args = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "feedforward_dim": args.feedforward_dim,
        "max_len": args.max_len,
    }
    return TransformerClassifier(
        **transformer_args,
        pooling=args.pooling,
    )


def build_metadata(
    args: argparse.Namespace,
    label_to_idx: dict[str, int],
    input_dim: int,
    train_size: int,
    val_size: int,
) -> dict:
    metadata = {
        "model": args.model,
        "feature_type": args.feature_type,
        "feature_root": str(args.feature_root),
        "views": args.views,
        "train_splits": args.train_splits,
        "val_splits": args.val_splits,
        "input_dim": input_dim,
        "num_classes": len(label_to_idx),
        "train_size": train_size,
        "val_size": val_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "augment": args.augment,
        "scheduler": args.scheduler,
        "label_to_idx": label_to_idx,
    }
    if args.scheduler == "plateau":
        metadata["scheduler_config"] = {
            "monitor": "val_loss",
            "factor": args.plateau_factor,
            "patience": args.plateau_patience,
            "min_lr": args.min_lr,
        }
    if args.model in {"rnn", "lstm"}:
        metadata["bidirectional"] = args.bidirectional
    if args.model == "rnn":
        metadata["nonlinearity"] = args.rnn_nonlinearity
    if args.model == "transformer":
        metadata.update(
            num_heads=args.num_heads,
            feedforward_dim=args.feedforward_dim,
            max_len=args.max_len,
        )
    if args.augment:
        metadata["augmentation"] = {
            "prob": args.aug_prob,
            "rotation_deg": args.rotation_deg,
            "shear": args.shear,
            "scale": args.scale,
            "gaussian_noise_std": args.gaussian_noise_std,
        }
    return metadata


def save_metadata(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("metadata.json", metadata),
        ("label_to_idx.json", metadata["label_to_idx"]),
    ):
        (output_dir / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def print_summary(args, device, train_dataset, val_dataset, input_dim, num_classes):
    print(f"device={device} model={args.model} feature_type={args.feature_type}")
    print(f"feature_root={args.feature_root} views={','.join(args.views)}")
    print(f"train_size={len(train_dataset)} val_size={len(val_dataset)}")
    print(
        f"input_dim={input_dim} num_classes={num_classes} "
        f"augment={args.augment} scheduler={args.scheduler}"
    )


def main() -> None:
    args = prepare_args(parse_args())
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_dataset, val_dataset, label_to_idx, input_dim = build_datasets(args)
    train_loader = build_loader(train_dataset, args, device, shuffle=True)
    val_loader = build_loader(val_dataset, args, device, shuffle=False)
    model = build_model(args, input_dim, len(label_to_idx)).to(device)

    metadata = build_metadata(
        args,
        label_to_idx,
        input_dim,
        len(train_dataset),
        len(val_dataset),
    )
    save_metadata(args.output_dir, metadata)
    print_summary(
        args,
        device,
        train_dataset,
        val_dataset,
        input_dim,
        len(label_to_idx),
    )

    if args.dry_run:
        features, lengths, labels = next(iter(train_loader))
        with torch.no_grad():
            logits = model(features.to(device), lengths.to(device))
        print(
            f"features={tuple(features.shape)} lengths={tuple(lengths.shape)} "
            f"labels={tuple(labels.shape)} logits={tuple(logits.shape)}"
        )
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        checkpoint_dir=args.output_dir,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
    )
    trainer.fit(train_loader, val_loader, args.epochs, metadata)


if __name__ == "__main__":
    main()
