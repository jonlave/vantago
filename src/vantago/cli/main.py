"""Command line interface for Go policy tooling."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import NoReturn

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "inspect-sgf":
            return _handle_inspect_sgf(args)
        parser.error(f"unknown command: {args.command}")
    except SgfParseError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


def _handle_inspect_sgf(args: argparse.Namespace) -> int:
    return inspect_sgf(args.path)


def _main() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _main()
