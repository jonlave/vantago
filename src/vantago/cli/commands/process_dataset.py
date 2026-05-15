"""Processed dataset generation command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vantago.data import (
    ProcessedDatasetBuildResult,
    ProcessedDatasetError,
    write_processed_dataset,
)
from vantago.replay import ReplayDiagnostic


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "sgf_path",
        type=Path,
        help="Path to an SGF file or directory tree to encode.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the processed .npz artifact to write.",
    )


def process_dataset(sgf_path: Path, output: Path) -> int:
    try:
        result = write_processed_dataset(sgf_path, output)
    except ProcessedDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_processed_dataset_build_result(result), end="")
    if result.failed or result.records_written == 0:
        return 1
    return 0


def format_processed_dataset_build_result(
    result: ProcessedDatasetBuildResult,
) -> str:
    lines = [
        f"input: {result.input_path}",
        f"output: {result.output_path}",
        f"files_scanned: {result.files_scanned}",
        f"ok: {result.ok}",
        f"skipped: {result.skipped}",
        f"failed: {result.failed}",
        f"records_written: {result.records_written}",
        "skipped_by_reason:",
    ]
    if result.skipped_by_reason:
        lines.extend(
            f"  {skip_count.reason.value}: {skip_count.count}"
            for skip_count in result.skipped_by_reason
        )
    else:
        lines.append("  none")

    lines.append("skipped_files:")
    if result.skipped_diagnostics:
        lines.extend(
            _format_skipped_file(diagnostic)
            for diagnostic in result.skipped_diagnostics
        )
    else:
        lines.append("  none")

    lines.append("failed_files:")
    if result.failures:
        lines.extend(
            f"  {failure.source_name}: {failure.message}"
            for failure in result.failures
        )
    else:
        lines.append("  none")

    return "\n".join(lines) + "\n"


def _format_skipped_file(diagnostic: ReplayDiagnostic) -> str:
    reason = diagnostic.reason.value if diagnostic.reason is not None else "unknown"
    return (
        f"  {reason}: {diagnostic.source_name}: "
        f"{_message_without_source_prefix(diagnostic)}"
    )


def _message_without_source_prefix(diagnostic: ReplayDiagnostic) -> str:
    prefix = f"{diagnostic.source_name}: "
    if diagnostic.message.startswith(prefix):
        return diagnostic.message[len(prefix) :]
    return diagnostic.message
