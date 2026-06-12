# Vietnamese Sign Language Recognition

## Overview

This project addresses isolated Vietnamese Sign Language recognition. The input
is a video containing a person performing one sign, and the output is the
corresponding word label.

The following features are extracted from each video:

- **CNN features:** frame-level visual feature vectors obtained from a CNN
  backbone after global pooling.
- **Skeleton features:** 75 landmarks consisting of 33 body pose landmarks,
  21 left-hand landmarks, and 21 right-hand landmarks. Each landmark contains
  four values `(x, y, z, visibility_or_presence)`, resulting in a
  300-dimensional feature vector per frame.

The resulting feature sequence is processed by an RNN, LSTM, or Transformer
Encoder to learn temporal relationships between frames and classify the sign.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Project Structure

```text
feature-extractor/            CNN and skeleton feature extraction
dataset/                      Datasets, collation, and augmentation
models/                       RNN, LSTM, and Transformer models
config/grid_search_config.json
train.py                      Training and validation entry point
trainer.py                    Training loop, metrics, and checkpoints
evaluate.py                   Checkpoint evaluation
grid_search.py                Hyperparameter grid search
validate_feature_dataset.py   Feature dataset validation
```

A feature dataset follows this structure:

```text
feature/skeleton_8/
  split_1/
    front_view.json
    left_view.json
    right_view.json
    front_view/*.npy
    left_view/*.npy
    right_view/*.npy
```

Each `.npy` file contains a sequence with shape `(frames, feature_dim)`.
Flattened skeleton features have shape `(frames, 300)`.

`--views all` loads the three views as independent samples. The current
pipeline does not fuse all views into a single sample.

## Feature Validation

```bash
python validate_feature_dataset.py \
  --feature-root feature/skeleton_8 \
  --views all \
  --splits split_1 split_2 split_3 split_4 split_5 \
  --expected-frames 8 \
  --expected-feature-dim 300
```

## Training

### Common Parameters

| Parameter | Description |
| --- | --- |
| `--model` | Model type: `rnn`, `lstm`, or `transformer`. |
| `--feature-type` | Feature type: `cnn` or `skeleton`. |
| `--feature-root` | Root directory of the feature dataset. |
| `--views` | One or more views; use `all` for all three views. |
| `--train-splits` | Splits used for training. |
| `--val-splits` | Splits used for validation. |
| `--hidden-dim` | Hidden representation dimension. |
| `--num-layers` | Number of model layers. |
| `--dropout` | Dropout probability. |
| `--pooling` | Sequence pooling method. |
| `--batch-size` | Number of samples per batch. |
| `--epochs` | Number of training epochs. |
| `--lr` | AdamW learning rate. |
| `--weight-decay` | AdamW weight decay. |
| `--scheduler` | Learning-rate scheduler: `none`, `cosine`, or `plateau`. |
| `--device` | `auto`, `cpu`, `cuda`, or another PyTorch device. |
| `--output-dir` | Directory for checkpoints and metrics. |
| `--dry-run` | Validate one batch without training. |

### RNN

```bash
python train.py \
  --model rnn \
  --feature-type skeleton \
  --feature-root feature/skeleton_8 \
  --views all \
  --train-splits split_1 split_2 split_3 split_4 \
  --val-splits split_5 \
  --hidden-dim 256 \
  --num-layers 2 \
  --pooling mean \
  --bidirectional \
  --rnn-nonlinearity relu \
  --dropout 0.4 \
  --augment \
  --batch-size 64 \
  --epochs 50 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --scheduler none \
  --device auto \
  --num-workers 2 \
  --output-dir checkpoints/rnn_skeleton_3views
```

Model-specific parameters:

| Parameter | Description |
| --- | --- |
| `--rnn-nonlinearity` | Recurrent activation function: `tanh` or `relu`. |
| `--bidirectional` | Use a bidirectional RNN. |
| `--no-bidirectional` | Use a unidirectional RNN. |
| `--pooling` | Available options: `last`, `mean`, and `max`. |

### LSTM

```bash
python train.py \
  --model lstm \
  --feature-type skeleton \
  --feature-root feature/skeleton_8 \
  --views all \
  --train-splits split_1 split_2 split_3 split_4 \
  --val-splits split_5 \
  --hidden-dim 128 \
  --num-layers 2 \
  --pooling mean \
  --bidirectional \
  --dropout 0.4 \
  --augment \
  --batch-size 64 \
  --epochs 50 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --scheduler none \
  --device auto \
  --num-workers 2 \
  --output-dir checkpoints/lstm_skeleton_3views
```

Model-specific parameters:

| Parameter | Description |
| --- | --- |
| `--bidirectional` | Use a bidirectional LSTM. |
| `--no-bidirectional` | Use a unidirectional LSTM. |
| `--pooling` | Available options: `last`, `mean`, and `max`. |

### Transformer

```bash
python train.py \
  --model transformer \
  --feature-type skeleton \
  --feature-root feature/skeleton_8 \
  --views all \
  --train-splits split_1 split_2 split_3 split_4 \
  --val-splits split_5 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --pooling cls \
  --dropout 0.4 \
  --augment \
  --batch-size 64 \
  --epochs 50 \
  --lr 0.001 \
  --weight-decay 0.01 \
  --scheduler none \
  --device auto \
  --num-workers 2 \
  --output-dir checkpoints/transformer_skeleton_3views
```

Model-specific parameters:

| Parameter | Description |
| --- | --- |
| `--num-heads` | Number of attention heads. `hidden_dim` must be divisible by this value. |
| `--max-len` | Maximum sequence length. The default is `512`. |
| `--pooling` | Available options: `mean`, `max`, and `cls`. |

The Transformer uses sinusoidal positional encoding.

## Augmentation

Enable training augmentation with `--augment`.

| Parameter | Default |
| --- | ---: |
| `--aug-prob` | `0.5` |
| `--rotation-deg` | `10.0` |
| `--shear` | `0.08` |
| `--scale` | `0.10` |
| `--gaussian-noise-std` | `0.01` |

Skeleton features use rotation, shear, scale, and Gaussian noise. CNN features
use Gaussian noise only.

## Learning-Rate Scheduler

Disable scheduling:

```bash
--scheduler none
```

Use cosine annealing:

```bash
--scheduler cosine
```

Use ReduceLROnPlateau:

```bash
--scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 3 \
  --min-lr 0.000001
```

## Training Outputs

```text
OUTPUT_DIR/
  best.pt
  last.pt
  metadata.json
  label_to_idx.json
  history.json
  history.csv
  learning_curves.png
  validation_metrics.png
  reports/
```

`best.pt` is selected by validation accuracy. `history.json` and `history.csv`
contain loss, accuracy, macro precision/recall/F1, and weighted
precision/recall/F1 for each epoch.

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/lstm_skeleton_3views/best.pt \
  --feature-root feature/skeleton_8 \
  --splits split_6 split_7 \
  --views all \
  --batch-size 64 \
  --device auto \
  --num-workers 2 \
  --output-dir checkpoints/lstm_skeleton_3views/test_results
```

| Parameter | Description |
| --- | --- |
| `--checkpoint` | Path to a `.pt` checkpoint. |
| `--feature-root` | Feature dataset used for evaluation. |
| `--splits` | Test splits. |
| `--views` | Views included in evaluation. |
| `--max-samples` | Optional sample limit for a quick test. |
| `--output-dir` | Directory for evaluation outputs. |

Evaluation produces:

```text
metrics.json
classification_report.json
confusion_matrix.npy
predictions.csv
```

## Grid Search

The search space is defined in `config/grid_search_config.json`. Parameters in
a model-specific section override parameters with the same name in `common`.

### Configuration Format

The configuration file is a JSON object with one shared section and one
optional section for each model:

```json
{
  "common": {
    "batch_size": [128],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "lr": [0.001],
    "scheduler": ["plateau"],
    "augment": [true]
  },
  "rnn": {
    "hidden_dim": [128, 256],
    "num_layers": [2, 4],
    "pooling": ["mean"],
    "bidirectional": [true],
    "rnn_nonlinearity": ["relu"],
    "weight_decay": [0.0001]
  },
  "lstm": {
    "hidden_dim": [128, 256],
    "num_layers": [2, 4],
    "pooling": ["mean"],
    "bidirectional": [true],
    "weight_decay": [0.0001]
  },
  "transformer": {
    "hidden_dim": [128, 256],
    "num_layers": [2, 4],
    "num_heads": [8],
    "pooling": ["cls"],
    "weight_decay": [0.01]
  }
}
```

Each parameter value must be a non-empty JSON array, even when only one value
is tested. Grid search generates the Cartesian product of all parameter arrays
for the selected model.

For example, two `hidden_dim` values, two `num_layers` values, and four
`dropout` values produce `2 * 2 * 4 = 16` configurations before combining
other parameters.

The supported top-level sections are:

| Section | Description |
| --- | --- |
| `common` | Parameters included in every selected model. |
| `rnn` | RNN-specific parameters and overrides. |
| `lstm` | LSTM-specific parameters and overrides. |
| `transformer` | Transformer-specific parameters and overrides. |

Parameter names use the Python-style underscore form of the corresponding
`train.py` option. For example, `hidden_dim` becomes `--hidden-dim`, and
`weight_decay` becomes `--weight-decay`. Boolean values such as `augment` and
`bidirectional` must use JSON booleans `true` or `false`.

Preview all configurations without training:

```bash
python grid_search.py \
  --models rnn lstm transformer \
  --config config/grid_search_config.json \
  --epochs 30 \
  --plan-only
```

Run grid search for LSTM:

```bash
python grid_search.py \
  --models lstm \
  --config config/grid_search_config.json \
  --feature-type skeleton \
  --feature-root feature/skeleton_8 \
  --views all \
  --train-splits split_1 split_2 split_3 split_4 \
  --val-splits split_5 \
  --epochs 60 \
  --metric val_f1_macro \
  --device auto \
  --num-workers 2 \
  --output-dir checkpoints/grid_search_lstm
```

| Parameter | Description |
| --- | --- |
| `--models` | One or more models to search. |
| `--config` | JSON file defining the hyperparameter search space. |
| `--metric` | Metric used to rank runs. |
| `--max-runs` | Maximum number of configurations to run. |
| `--plan-only` | Print commands without starting training. |
| `--force` | Run completed configurations again. |

Grid search runs sequentially and writes the following files to
`--output-dir`:

```text
best.json
results.csv
results.json
MODEL_<id>/
```

`best.json` contains the best hyperparameter configuration according to the
selected metric.

## Command-Line Reference

```bash
python train.py --help
python evaluate.py --help
python grid_search.py --help
python validate_feature_dataset.py --help
```
