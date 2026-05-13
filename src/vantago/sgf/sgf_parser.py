"""SGF parser for supported game records.

Internal board coordinates use ML-friendly row-major indexing:
``row=0, col=0`` is the top-left board point, rows increase downward, and columns
increase rightward. SGF's text format also names the top-left point ``aa``.

The sgfmill library exposes points as ``(row, col)`` from a bottom-left origin,
so all sgfmill points are converted at this package boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sgfmill import sgf  # type: ignore[import-untyped]

MoveColor = Literal["b", "w"]


class SgfParseError(ValueError):
    """Raised when an SGF cannot be parsed by this package."""


@dataclass(frozen=True, slots=True)
class BoardPoint:
    """A 0-based top-left row/column board coordinate."""

    row: int
    col: int

    def flat_index(self, board_size: int) -> int:
        """Return the row-major board index for future policy labels."""

        if board_size <= 0:
            msg = f"board_size must be positive, got {board_size}"
            raise ValueError(msg)
        if not 0 <= self.row < board_size or not 0 <= self.col < board_size:
            msg = f"point {self} is outside board size {board_size}"
            raise ValueError(msg)
        return self.row * board_size + self.col


@dataclass(frozen=True, slots=True)
class GameMetadata:
    """Minimum metadata needed before broader replay work."""

    board_size: int
    black_player: str | None
    white_player: str | None
    komi: str | None
    result: str | None


@dataclass(frozen=True, slots=True)
class ParsedMove:
    """One parsed move from the SGF main sequence."""

    color: MoveColor
    point: BoardPoint | None


@dataclass(frozen=True, slots=True)
class ParsedGame:
    """A parsed SGF game."""

    source_name: str
    metadata: GameMetadata
    moves: tuple[ParsedMove, ...]


def load_sgf(path: Path) -> ParsedGame:
    """Load and parse an SGF file from disk."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        msg = f"unable to read SGF file {path}: {exc}"
        raise SgfParseError(msg) from exc
    return parse_sgf_bytes(content, source_name=str(path))


def parse_sgf_bytes(content: bytes, source_name: str | None = None) -> ParsedGame:
    """Parse SGF bytes into project-owned metadata and move objects."""

    resolved_source_name = source_name or "<bytes>"
    try:
        game = sgf.Sgf_game.from_bytes(content)
    except Exception as exc:  # sgfmill raises several SGF parse exception types.
        msg = f"{resolved_source_name}: unable to parse SGF: {exc}"
        raise SgfParseError(msg) from exc

    board_size = _get_board_size(game, resolved_source_name)
    sequence = game.get_main_sequence()
    _validate_game_type(game, resolved_source_name)
    _reject_variations(sequence, resolved_source_name)
    _reject_setup_stones(sequence, resolved_source_name)
    _reject_invalid_move_nodes(sequence, resolved_source_name)

    metadata = GameMetadata(
        board_size=board_size,
        black_player=_get_root_property(game, "PB"),
        white_player=_get_root_property(game, "PW"),
        komi=_get_root_property(game, "KM"),
        result=_get_root_property(game, "RE"),
    )
    moves = tuple(
        _parse_move(node=node, board_size=board_size, source_name=resolved_source_name)
        for node in sequence
        if _node_has_move(node)
    )
    return ParsedGame(
        source_name=resolved_source_name,
        metadata=metadata,
        moves=moves,
    )


def _get_board_size(game: Any, source_name: str) -> int:
    try:
        board_size = game.get_size()
    except Exception as exc:
        msg = f"{source_name}: unable to read board size: {exc}"
        raise SgfParseError(msg) from exc

    if not isinstance(board_size, int):
        msg = f"{source_name}: unsupported non-integer board size {board_size!r}"
        raise SgfParseError(msg)
    if board_size <= 0:
        msg = f"{source_name}: unsupported board size {board_size}"
        raise SgfParseError(msg)
    return board_size


def _get_root_property(game: Any, property_name: str) -> str | None:
    root = game.get_root()
    try:
        value = root.get(property_name)
    except KeyError:
        return None
    if value is None:
        return None
    return str(value)


def _validate_game_type(game: Any, source_name: str) -> None:
    game_type = _get_root_property(game, "GM")
    if game_type != "1":
        msg = f"{source_name}: unsupported SGF game type {game_type!r}; expected GM[1]"
        raise SgfParseError(msg)


def _reject_variations(main_sequence: list[Any], source_name: str) -> None:
    for node in main_sequence:
        if len(node) > 1:
            msg = f"{source_name}: unsupported variation branch in SGF"
            raise SgfParseError(msg)


def _reject_setup_stones(sequence: list[Any], source_name: str) -> None:
    setup_properties = ("AB", "AW", "AE")
    for node in sequence:
        for property_name in setup_properties:
            if _has_property(node, property_name):
                msg = (
                    f"{source_name}: unsupported setup stone property {property_name}; "
                    "setup stones are not supported by the SGF parser"
                )
                raise SgfParseError(msg)


def _reject_invalid_move_nodes(sequence: list[Any], source_name: str) -> None:
    for node in sequence:
        has_black_move = _has_property(node, "B")
        has_white_move = _has_property(node, "W")
        if has_black_move and has_white_move:
            msg = (
                f"{source_name}: unsupported move node with both B and W "
                "properties"
            )
            raise SgfParseError(msg)


def _parse_move(node: Any, board_size: int, source_name: str) -> ParsedMove:
    try:
        color, sgfmill_point = node.get_move()
    except Exception as exc:
        msg = f"{source_name}: unable to read move: {exc}"
        raise SgfParseError(msg) from exc

    if color not in {"b", "w"}:
        msg = f"{source_name}: unsupported move color {color!r}"
        raise SgfParseError(msg)

    if sgfmill_point is None:
        return ParsedMove(color=color, point=None)

    return ParsedMove(
        color=color,
        point=_from_sgfmill_point(sgfmill_point, board_size=board_size),
    )


def _from_sgfmill_point(sgfmill_point: tuple[int, int], board_size: int) -> BoardPoint:
    sgfmill_row, sgfmill_col = sgfmill_point
    board_row = board_size - 1 - sgfmill_row
    board_col = sgfmill_col
    point = BoardPoint(row=board_row, col=board_col)
    point.flat_index(board_size)
    return point


def _node_has_move(node: Any) -> bool:
    return _has_property(node, "B") or _has_property(node, "W")


def _has_property(node: Any, property_name: str) -> bool:
    try:
        node.get(property_name)
    except KeyError:
        return False
    return True
