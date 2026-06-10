from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm


class Trainer:
    """Trainer with checkpoints, classification reports, and learning curves."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: Path,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.best_val_acc = 0.0
        self.history: list[dict[str, float]] = []

        self.reports_dir = checkpoint_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, float]]:
        metadata = metadata or {}
        num_classes = int(metadata.get("num_classes", 0))
        class_names = self._class_names(metadata)

        for epoch in range(1, epochs + 1):
            train = self._run_epoch(train_loader, epoch, epochs, training=True)
            val = self._run_epoch(val_loader, epoch, epochs, training=False)

            train_report = self.classification_metrics(
                train.pop("labels"),
                train.pop("predictions"),
                num_classes,
                class_names,
            )
            val_report = self.classification_metrics(
                val.pop("labels"),
                val.pop("predictions"),
                num_classes,
                class_names,
            )

            self._step_scheduler(val["loss"])

            metrics = self._epoch_metrics(epoch, train, val, train_report, val_report)
            self.history.append(metrics)
            self._save_training_outputs(epoch, train_report, val_report)
            self._print_metrics(metrics)

            self.save_checkpoint("last.pt", epoch, metrics, metadata)
            if metrics["val_accuracy"] > self.best_val_acc:
                self.best_val_acc = metrics["val_accuracy"]
                self.save_checkpoint("best.pt", epoch, metrics, metadata)

        return self.history

    def _step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _run_epoch(
        self,
        loader,
        epoch: int,
        epochs: int,
        training: bool,
    ) -> dict[str, Any]:
        self.model.train(training)
        mode = "train" if training else "val"
        total_loss = correct = total = 0
        all_labels = []
        all_predictions = []

        progress = tqdm(
            loader,
            desc=f"{mode:<5} {epoch}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for features, lengths, labels in progress:
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(features, lengths)
                loss = self.criterion(logits, labels)

                if training:
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                predictions = logits.argmax(dim=1)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (predictions == labels).sum().item()
                total += batch_size
                all_labels.append(labels.detach().cpu())
                all_predictions.append(predictions.detach().cpu())

                progress.set_postfix(
                    loss=f"{total_loss / total:.4f}",
                    acc=f"{correct / total:.4f}",
                )

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "labels": torch.cat(all_labels),
            "predictions": torch.cat(all_predictions),
        }

    @staticmethod
    def classification_metrics(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        num_classes: int,
        class_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate and per-class precision, recall, and F1."""
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        labels = labels.to(torch.long)
        predictions = predictions.to(torch.long)
        indices = labels * num_classes + predictions
        confusion = torch.bincount(
            indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes).to(torch.float64)

        true_positive = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        precision = Trainer._safe_divide(true_positive, predicted)
        recall = Trainer._safe_divide(true_positive, support)
        f1 = Trainer._safe_divide(2 * precision * recall, precision + recall)
        total = support.sum().clamp(min=1)

        names = class_names or [str(index) for index in range(num_classes)]
        per_class = {
            names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(num_classes)
        }

        return {
            "accuracy": float(true_positive.sum() / total),
            "macro": {
                "precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1": float(f1.mean()),
            },
            "weighted": {
                "precision": float((precision * support).sum() / total),
                "recall": float((recall * support).sum() / total),
                "f1": float((f1 * support).sum() / total),
            },
            "per_class": per_class,
        }

    def get_history(self, metric: str | None = None):
        """Return all history rows or the values of one metric."""
        if metric is None:
            return self.history
        if self.history and metric not in self.history[0]:
            raise KeyError(f"Unknown metric: {metric}")
        return [row[metric] for row in self.history]

    def get_classification_report(self, epoch: int, split: str = "val") -> dict:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        path = self.reports_dir / f"epoch_{epoch:03d}_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing classification report: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_class_metric(
        self,
        epoch: int,
        class_name: str,
        metric: str = "f1",
        split: str = "val",
    ) -> float | int:
        """Return precision, recall, F1, or support for one class."""
        if metric not in {"precision", "recall", "f1", "support"}:
            raise ValueError("metric must be precision, recall, f1, or support")
        report = self.get_classification_report(epoch, split)
        try:
            return report["per_class"][class_name][metric]
        except KeyError as exc:
            raise KeyError(f"Unknown class or metric: {class_name}/{metric}") from exc

    def plot_learning_curves(self, output_path: Path | None = None) -> Path:
        """Plot train/validation loss, accuracy, and macro F1."""
        if not self.history:
            raise RuntimeError("No training history is available")

        plt = self._pyplot()

        epochs = self.get_history("epoch")
        curves = (
            ("Loss", "train_loss", "val_loss"),
            ("Accuracy", "train_accuracy", "val_accuracy"),
            ("F1", "train_f1_macro", "val_f1_macro"),
        )
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, (title, train_key, val_key) in zip(axes, curves):
            axis.plot(epochs, self.get_history(train_key), label="train")
            axis.plot(epochs, self.get_history(val_key), label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.3)
            axis.legend()

        figure.tight_layout()
        output_path = output_path or self.checkpoint_dir / "learning_curves.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return output_path

    def plot_validation_metrics(self, output_path: Path | None = None) -> Path:
        """Plot validation classification metrics for every epoch."""
        if not self.history:
            raise RuntimeError("No training history is available")

        plt = self._pyplot()
        epochs = self.get_history("epoch")
        metrics = (
            ("Accuracy", "val_accuracy"),
            ("Precision (macro)", "val_precision_macro"),
            ("Recall (macro)", "val_recall_macro"),
            ("F1 (macro)", "val_f1_macro"),
        )

        figure, axis = plt.subplots(figsize=(8, 5))
        for label, key in metrics:
            axis.plot(epochs, self.get_history(key), marker="o", label=label)

        axis.set_title("Validation Metrics")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()

        output_path = output_path or self.checkpoint_dir / "validation_metrics.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return output_path

    def save_checkpoint(
        self,
        name: str,
        epoch: int,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "metadata": metadata or {},
            },
            self.checkpoint_dir / name,
        )

    def _save_training_outputs(
        self,
        epoch: int,
        train_report: dict,
        val_report: dict,
    ) -> None:
        self._save_json(self.checkpoint_dir / "history.json", self.history)
        self._save_history_csv()
        self._save_json(
            self.reports_dir / f"epoch_{epoch:03d}_train.json",
            train_report,
        )
        self._save_json(
            self.reports_dir / f"epoch_{epoch:03d}_val.json",
            val_report,
        )
        self.plot_learning_curves()
        self.plot_validation_metrics()

    def _save_history_csv(self) -> None:
        path = self.checkpoint_dir / "history.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

    def _epoch_metrics(
        self,
        epoch: int,
        train: dict,
        val: dict,
        train_report: dict,
        val_report: dict,
    ) -> dict[str, float]:
        return {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "train_precision_macro": train_report["macro"]["precision"],
            "train_recall_macro": train_report["macro"]["recall"],
            "train_f1_macro": train_report["macro"]["f1"],
            "train_precision_weighted": train_report["weighted"]["precision"],
            "train_recall_weighted": train_report["weighted"]["recall"],
            "train_f1_weighted": train_report["weighted"]["f1"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_precision_macro": val_report["macro"]["precision"],
            "val_recall_macro": val_report["macro"]["recall"],
            "val_f1_macro": val_report["macro"]["f1"],
            "val_precision_weighted": val_report["weighted"]["precision"],
            "val_recall_weighted": val_report["weighted"]["recall"],
            "val_f1_weighted": val_report["weighted"]["f1"],
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @staticmethod
    def _class_names(metadata: dict[str, Any]) -> list[str] | None:
        label_to_idx = metadata.get("label_to_idx")
        if not label_to_idx:
            return None
        return [
            label
            for label, _ in sorted(label_to_idx.items(), key=lambda item: item[1])
        ]

    @staticmethod
    def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(numerator, dtype=torch.float64)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result

    def _pyplot(self):
        matplotlib_cache = self.checkpoint_dir / ".matplotlib"
        matplotlib_cache.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    @staticmethod
    def _save_json(path: Path, data) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _print_metrics(metrics: dict[str, float]) -> None:
        print(
            "epoch={epoch} train_loss={train_loss:.4f} "
            "train_acc={train_accuracy:.4f} train_f1={train_f1_macro:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            "val_f1={val_f1_macro:.4f} lr={lr:.6g}".format(**metrics)
        )
