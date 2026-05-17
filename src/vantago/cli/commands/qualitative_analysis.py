"""Qualitative mistake analysis command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vantago.qualitative_constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EXAMPLES_PER_PHASE,
    DEFAULT_QUALITATIVE_SPLIT,
    DEFAULT_TOP_K,
    QUALITATIVE_ANALYSIS_SPLITS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vantago.qualitative import (
        Heatmap,
        QualitativeAnalysisReport,
        QualitativeExample,
        QualitativePrediction,
    )

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
        "--split",
        choices=QUALITATIVE_ANALYSIS_SPLITS,
        default=DEFAULT_QUALITATIVE_SPLIT,
        help="Split to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="CNN inference batch size.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of ranked predictions to show per example.",
    )
    parser.add_argument(
        "--examples-per-phase",
        type=int,
        default=DEFAULT_EXAMPLES_PER_PHASE,
        help="Maximum examples to print for each game phase.",
    )
    mask_topk_group = parser.add_mutually_exclusive_group()
    mask_topk_group.add_argument(
        "--mask-topk",
        dest="mask_topk",
        action="store_true",
        default=False,
        help="Apply legal masking before ranking top-k predictions.",
    )
    mask_topk_group.add_argument(
        "--no-mask-topk",
        dest="mask_topk",
        action="store_false",
        help="Do not apply legal masking before ranking top-k predictions.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )


def qualitative_analysis_command(
    dataset: Path,
    splits: Path,
    checkpoint: Path,
    *,
    split: str,
    batch_size: int,
    top_k: int,
    examples_per_phase: int,
    mask_topk: bool,
    json_out: Path | None,
) -> int:
    from vantago.qualitative import (
        QualitativeAnalysisConfig,
        QualitativeAnalysisError,
        generate_qualitative_analysis_report,
        validate_qualitative_analysis_json_output_path,
        write_qualitative_analysis_report_json,
    )

    try:
        if json_out is not None:
            validate_qualitative_analysis_json_output_path(
                json_out,
                dataset_path=dataset,
                manifest_path=splits,
                checkpoint_path=checkpoint,
            )
        report = generate_qualitative_analysis_report(
            dataset,
            splits,
            checkpoint,
            config=QualitativeAnalysisConfig(
                split=split,
                batch_size=batch_size,
                top_k=top_k,
                examples_per_phase=examples_per_phase,
                mask_topk=mask_topk,
            ),
        )
        if json_out is not None:
            write_qualitative_analysis_report_json(json_out, report)
    except QualitativeAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_qualitative_analysis_report(report), end="")
    return 0


def format_qualitative_analysis_report(
    report: QualitativeAnalysisReport,
) -> str:
    return (
        "qualitative_analysis\n"
        + format_provenance(report)
        + "\n"
        + "human_move_heatmap_top\n"
        + format_heatmap_top(report.human_move_heatmap)
        + "\n"
        + "model_top1_heatmap_top\n"
        + format_heatmap_top(report.model_top1_heatmap)
        + "\n"
        + "phase_examples\n"
        + format_phase_examples(report)
    )


def format_provenance(report: QualitativeAnalysisReport) -> str:
    return (
        f"dataset: {report.dataset_path}\n"
        f"splits: {report.manifest_path}\n"
        f"checkpoint: {report.checkpoint_path}\n"
        f"split: {report.split}\n"
        f"batch_size: {report.batch_size}\n"
        f"top_k: {report.top_k}\n"
        f"examples_per_phase: {report.examples_per_phase}\n"
        f"mask_topk: {str(report.mask_topk).lower()}\n"
    )


def format_heatmap_top(heatmap: Heatmap, *, limit: int = 10) -> str:
    rows = [
        (row, col, count)
        for row, counts in enumerate(heatmap)
        for col, count in enumerate(counts)
        if count > 0
    ]
    rows.sort(key=lambda item: (-item[2], item[0], item[1]))
    if not rows:
        return "row  col  count\n"

    values = [
        (str(row), str(col), str(count))
        for row, col, count in rows[:limit]
    ]
    return _format_table(("row", "col", "count"), values)


def format_phase_examples(report: QualitativeAnalysisReport) -> str:
    parts: list[str] = []
    for phase, examples in report.phase_examples.items():
        parts.append(f"phase: {phase}\n")
        if not examples:
            parts.append("no examples\n")
            continue
        for index, example in enumerate(examples, start=1):
            parts.append(format_example(example, index=index))
    return "".join(parts)


def format_example(example: QualitativeExample, *, index: int) -> str:
    return (
        f"example: {index}\n"
        f"game_id: {example.game_id}\n"
        f"source_name: {example.source_name}\n"
        f"move_number: {example.move_number}\n"
        f"category: {example.category}\n"
        f"category_basis: {example.category_basis}\n"
        "human_move: "
        f"row={example.human_row} col={example.human_col} "
        f"label={example.human_label} target_rank={example.target_rank}\n"
        "raw_top1: "
        f"row={example.raw_top1_row} col={example.raw_top1_col} "
        f"label={example.raw_top1_label} "
        f"probability={_format_probability(example.raw_top1_probability)} "
        f"legal={str(example.raw_top1_is_legal).lower()}\n"
        + "top_predictions\n"
        + format_predictions(example.top_predictions)
        + "board\n"
        + "\n".join(example.board)
        + "\n"
    )


def format_predictions(predictions: Sequence[QualitativePrediction]) -> str:
    values = [
        (
            str(prediction.rank),
            str(prediction.row),
            str(prediction.col),
            str(prediction.label),
            _format_probability(prediction.probability),
            str(prediction.is_legal).lower(),
            str(prediction.is_human_move).lower(),
        )
        for prediction in predictions
    ]
    return _format_table(
        (
            "rank",
            "row",
            "col",
            "label",
            "probability",
            "legal",
            "human",
        ),
        values,
    )


def _format_probability(value: float) -> str:
    return f"{value:.4f}"


def _format_table(
    headers: tuple[str, ...],
    rows: Sequence[tuple[str, ...]],
) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [_format_table_row(headers, widths)]
    lines.extend(_format_table_row(row, widths) for row in rows)
    return "\n".join(lines) + "\n"


def _format_table_row(values: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(widths[index])
        for index, value in enumerate(values)
    ).rstrip()
