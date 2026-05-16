"""CNN policy checkpoint evaluation command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vantago.cli.commands.evaluate_baselines import (
    format_baseline_phase_rows,
    format_baseline_rows,
)
from vantago.data.splits import SPLIT_NAMES

if TYPE_CHECKING:
    from vantago.training import CnnPolicyEvaluationResult

DEFAULT_BATCH_SIZE = 128


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
        "checkpoint",
        type=Path,
        help="Path to a saved CNN policy checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        default="validation",
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--mask-topk",
        action="store_true",
        help="Apply legal masking before computing top-k accuracy.",
    )


def evaluate_cnn_policy_command(
    dataset: Path,
    splits: Path,
    checkpoint: Path,
    split: str,
    batch_size: int,
    mask_topk: bool,
) -> int:
    from vantago.training import CnnTrainingError, evaluate_cnn_policy_checkpoint

    try:
        result = evaluate_cnn_policy_checkpoint(
            dataset,
            splits,
            checkpoint,
            split=split,
            batch_size=batch_size,
            mask_topk=mask_topk,
        )
    except CnnTrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_cnn_policy_evaluation_result(result), end="")
    return 0


def format_cnn_policy_evaluation_result(
    result: CnnPolicyEvaluationResult,
) -> str:
    return (
        format_baseline_rows(result.rows)
        + "\n"
        + format_baseline_phase_rows(result.phase_rows)
    )
