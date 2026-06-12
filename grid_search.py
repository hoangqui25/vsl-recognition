from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_NAMES = ("rnn", "lstm", "transformer")
BOOLEAN_PARAMETERS = {"augment", "bidirectional"}
MINIMIZE_METRICS = {"train_loss", "val_loss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grid search by launching train.py for each configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/grid_search_config.json"),
        help="JSON file containing common and model-specific search spaces.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--feature-type", choices=("cnn", "skeleton"), default="skeleton")
    parser.add_argument("--feature-root", type=Path, default=Path("feature/skeleton_8"))
    parser.add_argument("--views", nargs="+", default=["all"])
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["split_1", "split_2", "split_3", "split_4"],
    )
    parser.add_argument("--val-splits", nargs="+", default=["split_5"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/grid_search"),
    )
    parser.add_argument(
        "--metric",
        default="val_accuracy",
        help="Metric from history.json used to rank runs.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Run only the first N valid configurations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run configurations again even when their history is complete.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print commands without starting training.",
    )
    return parser.parse_args()


def load_search_space(path: Path) -> dict[str, dict[str, list[Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Grid configuration not found: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    allowed_sections = {"common", *MODEL_NAMES}
    unknown = set(config) - allowed_sections
    if unknown:
        raise ValueError(f"Unknown grid sections: {sorted(unknown)}")

    for section, parameters in config.items():
        if not isinstance(parameters, dict):
            raise TypeError(f"Section '{section}' must be a JSON object")
        for name, values in parameters.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"Parameter '{section}.{name}' must contain a non-empty list"
                )
    return config


def parameter_combinations(
    common: dict[str, list[Any]],
    model_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    merged = {**common, **model_grid}
    names = sorted(merged)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(merged[name] for name in names))
    ]


def is_valid_configuration(model: str, parameters: dict[str, Any]) -> bool:
    pooling = parameters.get("pooling")
    if model in {"rnn", "lstm"} and pooling == "cls":
        return False
    if model == "transformer":
        if pooling == "last":
            return False
        hidden_dim = parameters.get("hidden_dim")
        num_heads = parameters.get("num_heads")
        if hidden_dim is not None and num_heads is not None:
            return hidden_dim % num_heads == 0
    return True


def configuration_id(model: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, **parameters},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{model}_{digest}"


def append_parameter(command: list[str], name: str, value: Any) -> None:
    option = f"--{name.replace('_', '-')}"
    if name in BOOLEAN_PARAMETERS:
        if not isinstance(value, bool):
            raise TypeError(f"Boolean parameter '{name}' must be true or false")
        command.append(option if value else f"--no-{name.replace('_', '-')}")
        return

    if isinstance(value, bool):
        raise TypeError(
            f"Unsupported boolean parameter '{name}'. "
            f"Add it to BOOLEAN_PARAMETERS."
        )
    if isinstance(value, list):
        command.append(option)
        command.extend(str(item) for item in value)
        return
    command.extend((option, str(value)))


def build_command(
    args: argparse.Namespace,
    model: str,
    parameters: dict[str, Any],
    run_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "--model",
        model,
        "--feature-type",
        args.feature_type,
        "--feature-root",
        str(args.feature_root),
        "--views",
        *args.views,
        "--train-splits",
        *args.train_splits,
        "--val-splits",
        *args.val_splits,
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(run_dir),
    ]
    for name, value in sorted(parameters.items()):
        append_parameter(command, name, value)
    return command


def read_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.is_file():
        return []
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise TypeError(f"Invalid history format: {path}")
    return history


def best_epoch(
    history: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    rows = [row for row in history if metric in row]
    if not rows:
        raise KeyError(f"Metric '{metric}' is missing from history")
    selector = min if metric in MINIMIZE_METRICS else max
    return selector(rows, key=lambda row: row[metric])


def collect_result(
    run_id: str,
    model: str,
    parameters: dict[str, Any],
    run_dir: Path,
    metric: str,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    result = {
        "run_id": run_id,
        "model": model,
        "status": status,
        "output_dir": str(run_dir),
        **parameters,
        "error": error,
    }
    history = read_history(run_dir)
    if history:
        epoch = best_epoch(history, metric)
        result.update(
            best_epoch=epoch["epoch"],
            score=epoch[metric],
            val_loss=epoch.get("val_loss"),
            val_accuracy=epoch.get("val_accuracy"),
            val_f1_macro=epoch.get("val_f1_macro"),
            val_f1_weighted=epoch.get("val_f1_weighted"),
        )
    return result


def save_results(
    output_dir: Path,
    results: list[dict[str, Any]],
    metric: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = [row for row in results if row.get("score") is not None]
    reverse = metric not in MINIMIZE_METRICS
    successful.sort(key=lambda row: row["score"], reverse=reverse)
    unsuccessful = [row for row in results if row.get("score") is None]
    ordered = successful + unsuccessful

    (output_dir / "results.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if ordered:
        fieldnames = list(dict.fromkeys(key for row in ordered for key in row))
        with (output_dir / "results.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)
    if successful:
        (output_dir / "best.json").write_text(
            json.dumps(successful[0], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def print_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    search_space = load_search_space(args.config)
    jobs: list[tuple[str, str, dict[str, Any], Path, list[str]]] = []
    for model in args.models:
        combinations = parameter_combinations(
            search_space.get("common", {}),
            search_space.get(model, {}),
        )
        for parameters in combinations:
            if not is_valid_configuration(model, parameters):
                continue
            run_id = configuration_id(model, parameters)
            run_dir = args.output_dir / run_id
            command = build_command(args, model, parameters, run_dir)
            jobs.append((run_id, model, parameters, run_dir, command))

    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]

    print(
        f"grid_search models={','.join(args.models)} "
        f"valid_runs={len(jobs)} metric={args.metric}",
        flush=True,
    )
    results: list[dict[str, Any]] = []

    for index, (run_id, model, parameters, run_dir, command) in enumerate(jobs, 1):
        history = read_history(run_dir)
        complete = len(history) >= args.epochs
        if complete and not args.force:
            print(f"[{index}/{len(jobs)}] skip completed {run_id}", flush=True)
            results.append(
                collect_result(
                    run_id,
                    model,
                    parameters,
                    run_dir,
                    args.metric,
                    "skipped",
                )
            )
            continue

        print(f"\n[{index}/{len(jobs)}] {run_id}", flush=True)
        print_command(command)
        if args.plan_only:
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "grid_parameters.json").write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        process = subprocess.run(command, check=False)
        status = "completed" if process.returncode == 0 else "failed"
        error = "" if process.returncode == 0 else f"exit_code={process.returncode}"
        try:
            result = collect_result(
                run_id,
                model,
                parameters,
                run_dir,
                args.metric,
                status,
                error,
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            result = {
                "run_id": run_id,
                "model": model,
                "status": "failed",
                "output_dir": str(run_dir),
                **parameters,
                "error": str(exc),
            }
        results.append(result)
        save_results(args.output_dir, results, args.metric)

    if not args.plan_only:
        save_results(args.output_dir, results, args.metric)
        successful = [row for row in results if row.get("score") is not None]
        if successful:
            reverse = args.metric not in MINIMIZE_METRICS
            best = sorted(
                successful,
                key=lambda row: row["score"],
                reverse=reverse,
            )[0]
            print(
                f"\nbest run={best['run_id']} "
                f"{args.metric}={best['score']:.6f} "
                f"epoch={best['best_epoch']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
