"""Policy model comparison command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from vantago.cli.commands.evaluate_baselines import (
    format_baseline_phase_rows,
    format_baseline_rows,
)
from vantago.cli.commands.train_cnn_policy import format_cnn_training_result
from vantago.cli.commands.train_mlp_baseline import (
    format_mlp_baseline_training_result,
)

if TYPE_CHECKING:
    from vantago.comparison import (
        PolicyComparisonDelta,
        PolicyModelComparisonResult,
    )

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_CNN_HIDDEN_CHANNELS = 64
DEFAULT_MLP_HIDDEN_SIZE = 128
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
        help="Path for CNN JSON epoch history. Defaults beside the checkpoint.",
    )
    parser.add_argument(
        "--mlp-history-out",
        type=Path,
        default=None,
        help="Path for MLP JSON epoch history. Defaults beside the checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs for MLP and CNN models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--cnn-hidden-channels",
        type=int,
        default=DEFAULT_CNN_HIDDEN_CHANNELS,
        help="Hidden channel width for the CNN feature extractor.",
    )
    parser.add_argument(
        "--mlp-hidden-size",
        type=int,
        default=DEFAULT_MLP_HIDDEN_SIZE,
        help="Hidden layer width for the flattened-board MLP.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="AdamW learning rate for MLP and CNN training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay for MLP and CNN training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for training order and random baseline.",
    )
    parser.add_argument(
        "--mask-topk",
        action="store_true",
        help="Apply legal masking before computing validation top-k accuracy.",
    )


def compare_policy_models_command(
    dataset: Path,
    splits: Path,
    *,
    checkpoint_out: Path,
    history_out: Path | None,
    mlp_history_out: Path | None,
    epochs: int,
    batch_size: int,
    cnn_hidden_channels: int,
    mlp_hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    mask_topk: bool,
) -> int:
    from vantago.comparison import (
        PolicyModelComparisonConfig,
        PolicyModelComparisonError,
        compare_policy_models,
    )

    config = PolicyModelComparisonConfig(
        checkpoint_path=checkpoint_out,
        history_path=history_out,
        mlp_history_path=mlp_history_out,
        epochs=epochs,
        batch_size=batch_size,
        cnn_hidden_channels=cnn_hidden_channels,
        mlp_hidden_size=mlp_hidden_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        mask_topk=mask_topk,
    )
    try:
        result = compare_policy_models(dataset, splits, config=config)
    except PolicyModelComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_policy_model_comparison_result(result), end="")
    return 0


def format_policy_model_comparison_result(
    result: PolicyModelComparisonResult,
) -> str:
    return (
        "mlp_history\n"
        + format_mlp_baseline_training_result(result.mlp_result)
        + "\n"
        + "cnn_history\n"
        + format_cnn_training_result(result.cnn_training_result)
        + "\n"
        + "comparison\n"
        + format_baseline_rows(result.rows)
        + "\n"
        + "phase_comparison\n"
        + format_baseline_phase_rows(result.phase_rows)
        + "\n"
        + f"checkpoint: {result.cnn_training_result.checkpoint_path}\n"
        + f"history: {result.cnn_training_result.history_path}\n"
        + f"mlp_history: {result.mlp_history_path}\n"
        + format_model_selection(result.model_selection_notes)
        + f"target: {'met' if result.target.met else 'missed'}\n"
        + format_target_deltas(result.target.deltas)
        + format_next_steps(result.target.next_steps)
    )


def format_model_selection(notes: Sequence[str]) -> str:
    return "model_selection:\n" + "".join(f"- {note}\n" for note in notes)


def format_target_deltas(
    deltas: Sequence[PolicyComparisonDelta],
) -> str:
    headers = ("baseline", "baseline_top_1", "cnn_top_1", "delta")
    rows = [
        (
            delta.baseline,
            _format_metric(delta.baseline_top_1),
            _format_metric(delta.cnn_top_1),
            _format_metric(delta.delta),
        )
        for delta in deltas
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = ["target_deltas", _format_table_row(headers, widths)]
    lines.extend(_format_table_row(row, widths) for row in rows)
    return "\n".join(lines) + "\n"


def format_next_steps(next_steps: Sequence[str]) -> str:
    if not next_steps:
        return ""

    return "diagnostics:\n" + "".join(f"- {step}\n" for step in next_steps)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _format_table_row(values: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(widths[index])
        for index, value in enumerate(values)
    ).rstrip()
