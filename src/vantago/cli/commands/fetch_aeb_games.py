"""Fetch replay-valid SGFs from the AEB online Go game archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vantago.data.aeb import (
    AEB_ARCHIVE_URL,
    DEFAULT_AEB_CACHE_DIR,
    DEFAULT_AEB_TIMEOUT_SECONDS,
    AebFetchConfig,
    AebFetchError,
    AebFetchResult,
    fetch_aeb_games,
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
        help="Deterministic selection seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the fetched raw SGF corpus will be written.",
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


def fetch_aeb_games_command(
    games: int,
    seed: int,
    output: Path,
    archive_url: str,
    cache_dir: Path,
    timeout: float,
) -> int:
    try:
        result = fetch_aeb_games(
            AebFetchConfig(
                games=games,
                seed=seed,
                output_dir=output,
                archive_url=archive_url,
                cache_dir=cache_dir,
                timeout_seconds=timeout,
                progress_callback=_print_progress,
            )
        )
    except AebFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_aeb_fetch_result(result), end="")
    return 0


def format_aeb_fetch_result(result: AebFetchResult) -> str:
    lines = [
        f"source_url: {result.archive_metadata.source_url}",
        f"output: {result.output_dir}",
        f"manifest: {result.manifest_path}",
        f"readme: {result.readme_path}",
        f"archive_cache: {result.archive_path}",
        f"catalog_cache: {result.catalog_path}",
        f"archive_sha256: {result.archive_metadata.archive_sha256}",
        f"games_requested: {result.games_requested}",
        f"seed: {result.seed}",
        f"files_scanned: {result.files_scanned}",
        f"valid_games_available: {result.valid_games_available}",
        f"selected_files: {len(result.selected_games)}",
        f"moves_replayed: {result.moves_replayed}",
        f"skipped: {result.skipped}",
        f"failed: {result.failed}",
        "skipped_by_reason:",
    ]
    if result.skipped_by_reason:
        lines.extend(
            f"  {skip_count.reason.value}: {skip_count.count}"
            for skip_count in result.skipped_by_reason
        )
    else:
        lines.append("  none")
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
