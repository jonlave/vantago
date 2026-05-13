"""Replay parsed Go games into board states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from vantago.sgf import BoardPoint, ParsedGame, ParsedMove

StoneColor = Literal["b", "w"]


class IllegalMoveError(ValueError):
    """Raised when a move cannot be applied to a board state."""


@dataclass(frozen=True, slots=True)
class BoardState:
    """Immutable row-major Go board state."""

    board_size: int
    stones: tuple[StoneColor | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stones", tuple(self.stones))
        if self.board_size <= 0:
            msg = f"board_size must be positive, got {self.board_size}"
            raise ValueError(msg)
        expected_point_count = self.board_size * self.board_size
        if len(self.stones) != expected_point_count:
            msg = (
                f"stones must contain {expected_point_count} entries for "
                f"board size {self.board_size}, got {len(self.stones)}"
            )
            raise ValueError(msg)
        invalid_stones = set(self.stones) - {None, "b", "w"}
        if invalid_stones:
            msg = f"unsupported stone values: {invalid_stones!r}"
            raise ValueError(msg)

    @classmethod
    def empty(cls, board_size: int) -> Self:
        """Return an empty board with the requested size."""

        return cls(board_size=board_size, stones=(None,) * (board_size * board_size))

    def stone_at(self, point: BoardPoint) -> StoneColor | None:
        """Return the stone at a point, or None if it is empty."""

        return self.stones[self._index(point)]

    def apply_move(self, move: ParsedMove) -> Self:
        """Return the board state after applying one parsed move."""

        if move.point is None:
            return self

        point_index = self._move_index(move)
        if self.stones[point_index] is not None:
            msg = (
                f"{move.color.upper()} move at row={move.point.row} "
                f"col={move.point.col} is illegal: point is occupied"
            )
            raise IllegalMoveError(msg)

        updated_stones = list(self.stones)
        updated_stones[point_index] = move.color
        opponent_color = _opponent(move.color)

        for neighbor_index in self._neighbor_indices(move.point):
            if updated_stones[neighbor_index] != opponent_color:
                continue
            group = self._collect_group(neighbor_index, updated_stones)
            if not self._group_has_liberty(group, updated_stones):
                for captured_index in group:
                    updated_stones[captured_index] = None

        own_group = self._collect_group(point_index, updated_stones)
        if not self._group_has_liberty(own_group, updated_stones):
            msg = (
                f"{move.color.upper()} move at row={move.point.row} "
                f"col={move.point.col} is illegal: move has no liberties"
            )
            raise IllegalMoveError(msg)

        return self.__class__(
            board_size=self.board_size,
            stones=tuple(updated_stones),
        )

    def _index(self, point: BoardPoint) -> int:
        return point.flat_index(self.board_size)

    def _move_index(self, move: ParsedMove) -> int:
        if move.point is None:
            msg = "pass moves do not have a board point"
            raise ValueError(msg)

        try:
            return self._index(move.point)
        except ValueError as exc:
            msg = (
                f"{move.color.upper()} move at row={move.point.row} "
                f"col={move.point.col} is illegal: {exc}"
            )
            raise IllegalMoveError(msg) from exc

    def _neighbor_indices(self, point: BoardPoint) -> tuple[int, ...]:
        neighbors: list[int] = []
        if point.row > 0:
            neighbors.append(self._index(BoardPoint(row=point.row - 1, col=point.col)))
        if point.row < self.board_size - 1:
            neighbors.append(self._index(BoardPoint(row=point.row + 1, col=point.col)))
        if point.col > 0:
            neighbors.append(self._index(BoardPoint(row=point.row, col=point.col - 1)))
        if point.col < self.board_size - 1:
            neighbors.append(self._index(BoardPoint(row=point.row, col=point.col + 1)))
        return tuple(neighbors)

    def _collect_group(
        self,
        start_index: int,
        stones: list[StoneColor | None],
    ) -> set[int]:
        color = stones[start_index]
        if color is None:
            return set()

        group: set[int] = set()
        frontier = [start_index]
        while frontier:
            current_index = frontier.pop()
            if current_index in group:
                continue
            if stones[current_index] != color:
                continue
            group.add(current_index)
            current_point = self._point_from_index(current_index)
            frontier.extend(self._neighbor_indices(current_point))
        return group

    def _group_has_liberty(
        self,
        group: set[int],
        stones: list[StoneColor | None],
    ) -> bool:
        for stone_index in group:
            point = self._point_from_index(stone_index)
            if any(
                stones[neighbor] is None
                for neighbor in self._neighbor_indices(point)
            ):
                return True
        return False

    def _point_from_index(self, index: int) -> BoardPoint:
        return BoardPoint(
            row=index // self.board_size,
            col=index % self.board_size,
        )


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One replayed move with the board before and after it."""

    move_number: int
    move: ParsedMove
    board_before: BoardState
    board_after: BoardState


def replay_game(parsed_game: ParsedGame) -> tuple[ReplayStep, ...]:
    """Replay parsed SGF moves and return each before/after board state."""

    board = BoardState.empty(parsed_game.metadata.board_size)
    steps: list[ReplayStep] = []
    for move_number, move in enumerate(parsed_game.moves, start=1):
        board_before = board
        board_after = board.apply_move(move)
        steps.append(
            ReplayStep(
                move_number=move_number,
                move=move,
                board_before=board_before,
                board_after=board_after,
            )
        )
        board = board_after
    return tuple(steps)


def _opponent(color: StoneColor) -> StoneColor:
    return "w" if color == "b" else "b"
