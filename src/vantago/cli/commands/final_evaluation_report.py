"""Final held-out evaluation report command."""

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
    from vantago.evaluation import PolicyMetricSummary
    from vantago.final_evaluation import FinalEvaluationReport

DEFAULT_BATCH_SIZE = 128
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
        "checkpoint",
        type=Path,
        help="Path to the selected CNN policy checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="CNN evaluation batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic random baseline seed.",
    )
    mask_topk_group = parser.add_mutually_exclusive_group()
    mask_topk_group.add_argument(
        "--mask-topk",
        dest="mask_topk",
        action="store_true",
        default=None,
        help="Apply legal masking before computing top-k accuracy.",
    )
    mask_topk_group.add_argument(
        "--no-mask-topk",
        dest="mask_topk",
        action="store_false",
        help="Do not apply legal masking before computing top-k accuracy.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )


def final_evaluation_report_command(
    dataset: Path,
    splits: Path,
    checkpoint: Path,
    *,
    batch_size: int,
    seed: int,
    mask_topk: bool | None,
    json_out: Path | None,
) -> int:
    from vantago.final_evaluation import (
        FinalEvaluationConfig,
        FinalEvaluationReportError,
        generate_final_evaluation_report,
        validate_final_evaluation_json_output_path,
        write_final_evaluation_report_json,
    )

    try:
        if json_out is not None:
            validate_final_evaluation_json_output_path(
                json_out,
                dataset_path=dataset,
                manifest_path=splits,
                checkpoint_path=checkpoint,
            )
        report = generate_final_evaluation_report(
            dataset,
            splits,
            checkpoint,
            config=FinalEvaluationConfig(
                batch_size=batch_size,
                seed=seed,
                mask_topk=mask_topk,
            ),
        )
        if json_out is not None:
            write_final_evaluation_report_json(json_out, report)
    except FinalEvaluationReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_final_evaluation_report(report), end="")
    return 0


def format_final_evaluation_report(report: FinalEvaluationReport) -> str:
    return (
        "selection\n"
        + format_selection(report)
        + "\n"
        + "comparison\n"
        + format_baseline_rows(report.rows)
        + "\n"
        + "phase_comparison\n"
        + format_baseline_phase_rows(report.phase_rows)
        + "\n"
        + "provenance\n"
        + format_provenance(report)
    )


def format_selection(report: FinalEvaluationReport) -> str:
    metrics = report.selection.validation_metrics
    return (
        f"model: {report.selection.model}\n"
        f"method: {report.selection.method}\n"
        f"split: {report.selection.split}\n"
        f"best_epoch: {report.selection.best_epoch}\n"
        + _format_metric_lines("validation", metrics)
    )


def format_provenance(report: FinalEvaluationReport) -> str:
    return (
        f"dataset: {report.dataset_path}\n"
        f"splits: {report.manifest_path}\n"
        f"checkpoint: {report.checkpoint_path}\n"
        f"split: {report.split}\n"
        f"batch_size: {report.config.batch_size}\n"
        f"seed: {report.config.seed}\n"
        f"mask_topk: {str(report.resolved_mask_topk).lower()}\n"
        f"train_games: {report.game_counts['train']}\n"
        f"validation_games: {report.game_counts['validation']}\n"
        f"test_games: {report.game_counts['test']}\n"
        f"train_examples: {report.record_counts['train']}\n"
        f"validation_examples: {report.record_counts['validation']}\n"
        f"test_examples: {report.record_counts['test']}\n"
    )


def _format_metric_lines(
    prefix: str,
    metrics: PolicyMetricSummary,
) -> str:
    return (
        f"{prefix}_examples: {metrics.example_count}\n"
        f"{prefix}_top_1: {_format_metric(metrics.top_1)}\n"
        f"{prefix}_top_3: {_format_metric(metrics.top_3)}\n"
        f"{prefix}_top_5: {_format_metric(metrics.top_5)}\n"
        f"{prefix}_cross_entropy: "
        f"{_format_optional_metric(metrics.cross_entropy)}\n"
        f"{prefix}_illegal_move_rate: "
        f"{_format_metric(metrics.illegal_move_rate)}\n"
    )


def _format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None else _format_metric(value)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"
