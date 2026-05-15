"""Processed dataset inspection command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vantago.data import (
    ProcessedDatasetError,
    ProcessedDatasetInspection,
    inspect_processed_dataset,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a processed .npz dataset artifact.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based record index to inspect.",
    )


def inspect_dataset(path: Path, index: int) -> int:
    try:
        inspection = inspect_processed_dataset(path, index=index)
    except ProcessedDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_processed_dataset_inspection(inspection), end="")
    return 0


def format_processed_dataset_inspection(
    inspection: ProcessedDatasetInspection,
) -> str:
    x_shape = ", ".join(str(dim) for dim in inspection.x_shape)
    return (
        f"path: {inspection.path}\n"
        f"record_count: {inspection.record_count}\n"
        f"index: {inspection.index}\n"
        f"game_id: {inspection.game_id}\n"
        f"source_name: {inspection.source_name}\n"
        f"move_number: {inspection.move_number}\n"
        f"y: {inspection.y}\n"
        f"decoded_y: row={inspection.decoded_row} col={inspection.decoded_col}\n"
        f"x_shape: [{x_shape}]\n"
        f"legal_mask_count: {inspection.legal_mask_count}\n"
        f"label_is_legal: {str(inspection.label_is_legal).lower()}\n"
    )
