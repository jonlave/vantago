"""CNN policy training command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vantago.training import CnnEpochResult, CnnTrainingResult

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_SEED = 0


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a processed .npz dataset artifact.",
    )
    parser.add_argument(
        "splits",
        type=Path,
        help="Path to a train/validation/test split manifest.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        required=True,
        help="Path where the best CNN checkpoint will be written.",
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=None,
        help="Path for JSON epoch history. Defaults beside the checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of CNN training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=DEFAULT_HIDDEN_CHANNELS,
        help="Hidden channel width for the CNN feature extractor.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for initialization and training order.",
    )
    parser.add_argument(
        "--mask-topk",
        action="store_true",
        help="Apply legal masking before computing validation top-k accuracy.",
    )


def train_cnn_policy_command(
    dataset: Path,
    splits: Path,
    *,
    checkpoint_out: Path,
    history_out: Path | None,
    epochs: int,
    batch_size: int,
    hidden_channels: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    mask_topk: bool,
) -> int:
    from vantago.training import (
        CnnTrainingConfig,
        CnnTrainingError,
        train_cnn_policy,
    )

    config = CnnTrainingConfig(
        checkpoint_path=checkpoint_out,
        history_path=history_out,
        epochs=epochs,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        mask_topk=mask_topk,
    )
    try:
        result = train_cnn_policy(dataset, splits, config=config)
    except CnnTrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_cnn_training_result(result), end="")
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"history: {result.history_path}")
    return 0


def format_cnn_training_result(result: CnnTrainingResult) -> str:
    headers = (
        "epoch",
        "train_loss",
        "validation_top_1",
        "validation_top_3",
        "validation_top_5",
        "validation_cross_entropy",
        "validation_illegal_move_rate",
        "best",
    )
    rows = [_epoch_row_values(epoch) for epoch in result.history]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [_format_table_row(headers, widths)]
    lines.extend(_format_table_row(row, widths) for row in rows)
    return "\n".join(lines) + "\n"


def _epoch_row_values(epoch: CnnEpochResult) -> tuple[str, ...]:
    metrics = epoch.validation_metrics
    return (
        str(epoch.epoch),
        _format_metric(epoch.train_loss),
        _format_metric(metrics.top_1),
        _format_metric(metrics.top_3),
        _format_metric(metrics.top_5),
        (
            "n/a"
            if metrics.cross_entropy is None
            else _format_metric(metrics.cross_entropy)
        ),
        _format_metric(metrics.illegal_move_rate),
        "*" if epoch.is_best else "",
    )


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _format_table_row(values: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(widths[index])
        for index, value in enumerate(values)
    ).rstrip()
