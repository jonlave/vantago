"""Command line interface for Go policy tooling."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import NoReturn

from vantago.cli.commands import evaluate_baselines as evaluate_baselines_command
from vantago.cli.commands import inspect_dataset as inspect_dataset_command
from vantago.cli.commands import process_dataset as process_dataset_command
from vantago.cli.commands import replay_batch as replay_batch_command
from vantago.cli.commands import split_dataset as split_dataset_command
from vantago.cli.commands.inspect_sgf import configure_parser, inspect_sgf
from vantago.sgf import SgfParseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vantago",
        description="Tools for Go policy data preparation and modeling.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subcommands.add_parser(
        "inspect-sgf",
        help="Print metadata and top-left row/column moves for one SGF.",
    )
    configure_parser(inspect_parser)
    replay_batch_parser = subcommands.add_parser(
        "replay-batch",
        help="Replay a file or directory of SGFs and print summary diagnostics.",
    )
    replay_batch_command.configure_parser(replay_batch_parser)
    process_dataset_parser = subcommands.add_parser(
        "process-dataset",
        help="Encode SGFs into a processed NumPy dataset artifact.",
    )
    process_dataset_command.configure_parser(process_dataset_parser)
    inspect_dataset_parser = subcommands.add_parser(
        "inspect-dataset",
        help="Inspect one record from a processed dataset artifact.",
    )
    inspect_dataset_command.configure_parser(inspect_dataset_parser)
    split_dataset_parser = subcommands.add_parser(
        "split-dataset",
        help="Write a deterministic game-level train/validation/test manifest.",
    )
    split_dataset_command.configure_parser(split_dataset_parser)
    evaluate_baselines_parser = subcommands.add_parser(
        "evaluate-baselines",
        help="Evaluate random legal and frequency baselines on a split.",
    )
    evaluate_baselines_command.configure_parser(evaluate_baselines_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "inspect-sgf":
            return _handle_inspect_sgf(args)
        if args.command == "replay-batch":
            return _handle_replay_batch(args)
        if args.command == "process-dataset":
            return _handle_process_dataset(args)
        if args.command == "inspect-dataset":
            return _handle_inspect_dataset(args)
        if args.command == "split-dataset":
            return _handle_split_dataset(args)
        if args.command == "evaluate-baselines":
            return _handle_evaluate_baselines(args)
        parser.error(f"unknown command: {args.command}")
    except SgfParseError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


def _handle_inspect_sgf(args: argparse.Namespace) -> int:
    return inspect_sgf(args.path)


def _handle_replay_batch(args: argparse.Namespace) -> int:
    return replay_batch_command.replay_batch(args.path)


def _handle_process_dataset(args: argparse.Namespace) -> int:
    return process_dataset_command.process_dataset(args.sgf_path, args.output)


def _handle_inspect_dataset(args: argparse.Namespace) -> int:
    return inspect_dataset_command.inspect_dataset(args.path, args.index)


def _handle_split_dataset(args: argparse.Namespace) -> int:
    return split_dataset_command.split_dataset(args.dataset, args.output, args.seed)


def _handle_evaluate_baselines(args: argparse.Namespace) -> int:
    return evaluate_baselines_command.evaluate_baselines_command(
        args.dataset,
        args.splits,
        args.split,
        args.seed,
        args.mask_topk,
    )


def _main() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _main()
