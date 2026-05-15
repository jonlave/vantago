"""Processed dataset split manifest command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vantago.data import (
    DatasetSplitBuildResult,
    DatasetSplitError,
    write_dataset_split_manifest,
)
from vantago.data.splits import SPLIT_NAMES


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a processed .npz dataset artifact.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the split manifest .json file to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic split seed.",
    )


def split_dataset(dataset: Path, output: Path, seed: int) -> int:
    try:
        result = write_dataset_split_manifest(dataset, output, seed=seed)
    except DatasetSplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_dataset_split_build_result(result), end="")
    return 0


def format_dataset_split_build_result(result: DatasetSplitBuildResult) -> str:
    manifest = result.manifest
    lines = [
        f"dataset: {manifest.dataset_path}",
        f"manifest: {result.output_path}",
        f"seed: {manifest.seed}",
        f"games_total: {manifest.game_counts['total']}",
        f"records_total: {manifest.record_counts['total']}",
        "game_counts:",
    ]
    lines.extend(
        f"  {name}: {manifest.game_counts[name]}"
        for name in SPLIT_NAMES
    )
    lines.append("record_counts:")
    lines.extend(
        f"  {name}: {manifest.record_counts[name]}"
        for name in SPLIT_NAMES
    )
    lines.append("overlap: none")
    return "\n".join(lines) + "\n"
