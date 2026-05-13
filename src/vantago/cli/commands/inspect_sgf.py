"""SGF inspection command."""

from __future__ import annotations

import argparse
from pathlib import Path

from vantago.sgf import ParsedMove, load_sgf


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path",
        type=Path,
        help="Path to an SGF file to inspect.",
    )


def format_move(move: ParsedMove) -> str:
    if move.point is None:
        return f"{move.color.upper()} PASS"
    return f"{move.color.upper()} row={move.point.row} col={move.point.col}"


def inspect_sgf(path: Path) -> int:
    game = load_sgf(path)

    print(f"source: {game.source_name}")
    print(f"board_size: {game.metadata.board_size}")
    print(f"black_player: {game.metadata.black_player or '<unknown>'}")
    print(f"white_player: {game.metadata.white_player or '<unknown>'}")
    print(f"komi: {game.metadata.komi or '<unknown>'}")
    print(f"result: {game.metadata.result or '<unknown>'}")
    print("moves:")
    for index, move in enumerate(game.moves, start=1):
        print(f"{index}: {format_move(move)}")
    return 0
