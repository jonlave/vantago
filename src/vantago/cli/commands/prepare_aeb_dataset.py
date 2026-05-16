"""Fetch AEB SGFs and prepare trainable dataset artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vantago.data import DatasetSplitError, ProcessedDatasetError
from vantago.data.aeb import (
    AEB_ARCHIVE_URL,
    DEFAULT_AEB_CACHE_DIR,
    DEFAULT_AEB_TIMEOUT_SECONDS,
    AebDatasetPrepareConfig,
    AebDatasetPrepareResult,
    AebFetchError,
    prepare_aeb_dataset,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--games",
        type=_positive_int,
        required=True,
        help="Number of replay-valid games to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic selection and split seed.",
    )
    parser.add_argument(
        "--name",
        help="Artifact name. Defaults to aeb-{games}-s{seed}.",
    )
    parser.add_argument(
        "--archive-url",
        default=AEB_ARCHIVE_URL,
        help="AEB games.tgz URL or local file URL.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_AEB_CACHE_DIR,
        help="Directory for verified archives and replay catalogs.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_AEB_TIMEOUT_SECONDS,
        help="Network timeout in seconds for AEB archive requests.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Override the generated raw SGF corpus directory.",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        help="Override the processed .npz output path.",
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        help="Override the split manifest .json output path.",
    )


def prepare_aeb_dataset_command(
    games: int,
    seed: int,
    name: str | None,
    archive_url: str,
    cache_dir: Path,
    timeout: float,
    raw_output: Path | None,
    dataset_output: Path | None,
    splits_output: Path | None,
) -> int:
    try:
        result = prepare_aeb_dataset(
            AebDatasetPrepareConfig(
                games=games,
                seed=seed,
                name=name,
                archive_url=archive_url,
                cache_dir=cache_dir,
                timeout_seconds=timeout,
                progress_callback=_print_progress,
                raw_output_dir=raw_output,
                dataset_output=dataset_output,
                splits_output=splits_output,
            )
        )
    except (AebFetchError, ProcessedDatasetError, DatasetSplitError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_aeb_dataset_prepare_result(result), end="")
    return 0


def format_aeb_dataset_prepare_result(result: AebDatasetPrepareResult) -> str:
    split_manifest = result.split_result.manifest
    lines = [
        f"name: {result.name}",
        f"raw_output: {result.fetch_result.output_dir}",
        f"dataset: {result.dataset_result.output_path}",
        f"splits: {result.split_result.output_path}",
        f"archive_sha256: {result.fetch_result.archive_metadata.archive_sha256}",
        f"games_requested: {result.fetch_result.games_requested}",
        f"seed: {result.fetch_result.seed}",
        f"selected_files: {len(result.fetch_result.selected_games)}",
        f"records_written: {result.dataset_result.records_written}",
        f"games_total: {split_manifest.game_counts['total']}",
        f"records_total: {split_manifest.record_counts['total']}",
        "game_counts:",
    ]
    lines.extend(
        f"  {name}: {split_manifest.game_counts[name]}"
        for name in ("train", "validation", "test")
    )
    lines.append("record_counts:")
    lines.extend(
        f"  {name}: {split_manifest.record_counts[name]}"
        for name in ("train", "validation", "test")
    )
    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"invalid positive integer: {value}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed < 1:
        msg = f"must be a positive integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        msg = f"invalid positive number: {value}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed <= 0:
        msg = f"must be positive, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr)
