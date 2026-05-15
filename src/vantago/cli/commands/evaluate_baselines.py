"""Non-neural baseline evaluation command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from vantago.data.splits import SPLIT_NAMES

if TYPE_CHECKING:
    from vantago.baselines import BaselineEvaluationResult, BaselineEvaluationRow


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
        "--split",
        choices=SPLIT_NAMES,
        default="validation",
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random baseline seed.",
    )
    parser.add_argument(
        "--mask-topk",
        action="store_true",
        help="Apply legal masking before computing top-k accuracy.",
    )


def evaluate_baselines_command(
    dataset: Path,
    splits: Path,
    split: str,
    seed: int,
    mask_topk: bool,
) -> int:
    from vantago.baselines import BaselineEvaluationError, evaluate_baselines

    try:
        result = evaluate_baselines(
            dataset,
            splits,
            split=split,
            seed=seed,
            mask_topk=mask_topk,
        )
    except BaselineEvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_baseline_evaluation_result(result), end="")
    return 0


def format_baseline_evaluation_result(result: BaselineEvaluationResult) -> str:
    return format_baseline_rows(result.rows)


def format_baseline_rows(rows: Sequence[BaselineEvaluationRow]) -> str:
    headers = (
        "baseline",
        "examples",
        "top_1",
        "top_3",
        "top_5",
        "cross_entropy",
        "illegal_move_rate",
    )
    if not rows:
        msg = "at least one baseline row is required"
        raise ValueError(msg)

    row_values = [_row_values(row) for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in row_values))
        for index in range(len(headers))
    ]
    lines = [_format_table_row(headers, widths)]
    lines.extend(_format_table_row(row, widths) for row in row_values)
    return "\n".join(lines) + "\n"


def _row_values(row: BaselineEvaluationRow) -> tuple[str, ...]:
    metrics = row.metrics
    return (
        row.baseline,
        str(metrics.example_count),
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
