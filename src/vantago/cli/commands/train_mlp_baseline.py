"""Flattened-board MLP baseline training command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vantago.cli.commands.evaluate_baselines import (
    format_baseline_phase_rows,
    format_baseline_rows,
)

if TYPE_CHECKING:
    from vantago.baselines import (
        MlpBaselineEpochResult,
        MlpBaselineTrainingResult,
    )

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_HIDDEN_SIZE = 128
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
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of MLP training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Hidden layer width for the flattened-board MLP.",
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


def train_mlp_baseline_command(
    dataset: Path,
    splits: Path,
    *,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    mask_topk: bool,
) -> int:
    from vantago.baselines import (
        BaselineEvaluationError,
        MlpBaselineConfig,
        MlpBaselineTrainingError,
        evaluate_baselines,
        train_mlp_baseline,
    )

    config = MlpBaselineConfig(
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        mask_topk=mask_topk,
    )
    try:
        mlp_result = train_mlp_baseline(dataset, splits, config=config)
        baseline_result = evaluate_baselines(
            dataset,
            splits,
            split="validation",
            seed=seed,
            mask_topk=mask_topk,
        )
    except (BaselineEvaluationError, MlpBaselineTrainingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_mlp_baseline_training_result(mlp_result), end="")
    print()
    print(
        format_baseline_rows((*baseline_result.rows, mlp_result.validation_row)),
        end="",
    )
    print()
    print(
        format_baseline_phase_rows(
            (*baseline_result.phase_rows, *mlp_result.validation_phase_rows)
        ),
        end="",
    )
    return 0


def format_mlp_baseline_training_result(
    result: MlpBaselineTrainingResult,
) -> str:
    headers = (
        "epoch",
        "train_loss",
        "validation_top_1",
        "validation_top_3",
        "validation_top_5",
        "validation_cross_entropy",
        "validation_illegal_move_rate",
    )
    rows = [_epoch_row_values(epoch) for epoch in result.history]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [_format_table_row(headers, widths)]
    lines.extend(_format_table_row(row, widths) for row in rows)
    return "\n".join(lines) + "\n"


def _epoch_row_values(epoch: MlpBaselineEpochResult) -> tuple[str, ...]:
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
    )


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _format_table_row(values: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(widths[index])
        for index, value in enumerate(values)
    ).rstrip()
