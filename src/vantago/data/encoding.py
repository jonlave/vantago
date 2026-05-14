"""Encode replayed Go positions into supervised policy records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from vantago.replay import SUPPORTED_BOARD_SIZE, BoardState, ReplayStep
from vantago.sgf import BoardPoint, MoveColor

CHANNEL_COUNT = 5
SUPPORTED_LABEL_COUNT = SUPPORTED_BOARD_SIZE * SUPPORTED_BOARD_SIZE

Float32Array: TypeAlias = npt.NDArray[np.float32]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _GroupInfo:
    color: MoveColor
    liberties: frozenset[int]


class PositionEncodingError(ValueError):
    """Raised when a replay step cannot be encoded as a policy record."""


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """One model-ready supervised next-move example."""

    x: Float32Array
    y: int
    game_id: str
    move_number: int
    legal_mask: BoolArray


def encode_replay_steps(
    game_id: str,
    replay_steps: tuple[ReplayStep, ...],
) -> tuple[PositionRecord, ...]:
    """Encode replayed steps into current-player-perspective records."""

    records: list[PositionRecord] = []
    previous_move_point: BoardPoint | None = None
    for step in replay_steps:
        records.append(
            _encode_replay_step(
                game_id=game_id,
                step=step,
                previous_move_point=previous_move_point,
            )
        )
        previous_move_point = step.move.point
    return tuple(records)


def encode_board_tensor(
    board: BoardState,
    current_player: MoveColor,
    previous_move_point: BoardPoint | None = None,
    *,
    legal_mask: BoolArray | None = None,
) -> Float32Array:
    """Encode a board as ``[5, 19, 19]`` from the current player's view."""

    _validate_supported_board(board)
    resolved_legal_mask = (
        encode_legal_mask(board, current_player)
        if legal_mask is None
        else _validate_legal_mask(legal_mask)
    )
    x = np.zeros(
        (CHANNEL_COUNT, SUPPORTED_BOARD_SIZE, SUPPORTED_BOARD_SIZE),
        dtype=np.float32,
    )
    opponent = _opponent(current_player)

    for flat_index, stone in enumerate(board.stones):
        row, col = divmod(flat_index, SUPPORTED_BOARD_SIZE)
        if stone == current_player:
            x[0, row, col] = 1.0
        elif stone == opponent:
            x[1, row, col] = 1.0
        elif stone is None:
            x[2, row, col] = 1.0

    if previous_move_point is not None:
        previous_label = encode_label(previous_move_point)
        row, col = divmod(previous_label, SUPPORTED_BOARD_SIZE)
        x[3, row, col] = 1.0

    x[4] = resolved_legal_mask.reshape(
        SUPPORTED_BOARD_SIZE,
        SUPPORTED_BOARD_SIZE,
    )
    return x


def encode_legal_mask(board: BoardState, current_player: MoveColor) -> BoolArray:
    """Return a flat legal-point mask matching the 361-class target space."""

    _validate_supported_board(board)
    group_infos = _build_group_infos(board.stones)
    opponent = _opponent(current_player)
    legal_mask = np.zeros(SUPPORTED_LABEL_COUNT, dtype=np.bool_)
    for label, stone in enumerate(board.stones):
        if stone is not None:
            continue
        legal_mask[label] = _is_legal_empty_point(
            label=label,
            stones=board.stones,
            group_infos=group_infos,
            current_player=current_player,
            opponent=opponent,
        )
    return legal_mask


def encode_label(point: BoardPoint) -> int:
    """Encode a board point as a row-major 19x19 class label."""

    return point.flat_index(SUPPORTED_BOARD_SIZE)


def decode_label(label: int) -> BoardPoint:
    """Decode a row-major 19x19 class label back to a board point."""

    if not 0 <= label < SUPPORTED_LABEL_COUNT:
        msg = (
            f"label must be in [0, {SUPPORTED_LABEL_COUNT}), "
            f"got {label}"
        )
        raise ValueError(msg)
    return BoardPoint(
        row=label // SUPPORTED_BOARD_SIZE,
        col=label % SUPPORTED_BOARD_SIZE,
    )


def _encode_replay_step(
    *,
    game_id: str,
    step: ReplayStep,
    previous_move_point: BoardPoint | None,
) -> PositionRecord:
    _validate_supported_board(step.board_before)
    if step.move.point is None:
        msg = (
            f"{game_id} move {step.move_number}: pass moves are not in the "
            f"{SUPPORTED_LABEL_COUNT}-point target space"
        )
        raise PositionEncodingError(msg)

    y = encode_label(step.move.point)
    legal_mask = encode_legal_mask(step.board_before, step.move.color)
    if not legal_mask[y]:
        msg = (
            f"{game_id} move {step.move_number}: target label {y} is not "
            "legal for the pre-move board"
        )
        raise PositionEncodingError(msg)

    return PositionRecord(
        x=encode_board_tensor(
            board=step.board_before,
            current_player=step.move.color,
            previous_move_point=previous_move_point,
            legal_mask=legal_mask,
        ),
        y=y,
        game_id=game_id,
        move_number=step.move_number,
        legal_mask=legal_mask,
    )


def _validate_supported_board(board: BoardState) -> None:
    if board.board_size != SUPPORTED_BOARD_SIZE:
        msg = (
            f"position encoding supports {SUPPORTED_BOARD_SIZE}x"
            f"{SUPPORTED_BOARD_SIZE} boards, got {board.board_size}x"
            f"{board.board_size}"
        )
        raise PositionEncodingError(msg)


def _validate_legal_mask(legal_mask: BoolArray) -> BoolArray:
    if legal_mask.shape != (SUPPORTED_LABEL_COUNT,):
        msg = (
            f"legal_mask must have shape ({SUPPORTED_LABEL_COUNT},), "
            f"got {legal_mask.shape}"
        )
        raise PositionEncodingError(msg)
    if legal_mask.dtype != np.dtype(np.bool_):
        msg = f"legal_mask must have bool dtype, got {legal_mask.dtype}"
        raise PositionEncodingError(msg)
    return legal_mask


def _build_group_infos(
    stones: tuple[MoveColor | None, ...],
) -> list[_GroupInfo | None]:
    group_infos: list[_GroupInfo | None] = [None] * SUPPORTED_LABEL_COUNT
    visited: set[int] = set()

    for start_label, color in enumerate(stones):
        if color is None or start_label in visited:
            continue

        group: set[int] = set()
        liberties: set[int] = set()
        frontier = [start_label]
        while frontier:
            label = frontier.pop()
            if label in visited or stones[label] != color:
                continue

            visited.add(label)
            group.add(label)
            for neighbor in _NEIGHBOR_INDICES_BY_LABEL[label]:
                neighbor_stone = stones[neighbor]
                if neighbor_stone is None:
                    liberties.add(neighbor)
                elif neighbor_stone == color and neighbor not in visited:
                    frontier.append(neighbor)

        group_info = _GroupInfo(color=color, liberties=frozenset(liberties))
        for group_label in group:
            group_infos[group_label] = group_info

    return group_infos


def _is_legal_empty_point(
    *,
    label: int,
    stones: tuple[MoveColor | None, ...],
    group_infos: list[_GroupInfo | None],
    current_player: MoveColor,
    opponent: MoveColor,
) -> bool:
    own_group_has_other_liberty = False
    seen_group_ids: set[int] = set()

    for neighbor in _NEIGHBOR_INDICES_BY_LABEL[label]:
        neighbor_stone = stones[neighbor]
        if neighbor_stone is None:
            return True

        group_info = group_infos[neighbor]
        if group_info is None:
            continue

        group_id = id(group_info)
        if group_id in seen_group_ids:
            continue
        seen_group_ids.add(group_id)

        liberty_count = len(group_info.liberties)
        if group_info.color == opponent and liberty_count == 1:
            return True
        if group_info.color == current_player and liberty_count > 1:
            own_group_has_other_liberty = True

    return own_group_has_other_liberty


def _build_neighbor_indices(board_size: int) -> tuple[tuple[int, ...], ...]:
    neighbor_indices: list[tuple[int, ...]] = []
    for label in range(board_size * board_size):
        row, col = divmod(label, board_size)
        neighbors: list[int] = []
        if row > 0:
            neighbors.append((row - 1) * board_size + col)
        if row < board_size - 1:
            neighbors.append((row + 1) * board_size + col)
        if col > 0:
            neighbors.append(row * board_size + col - 1)
        if col < board_size - 1:
            neighbors.append(row * board_size + col + 1)
        neighbor_indices.append(tuple(neighbors))
    return tuple(neighbor_indices)


_NEIGHBOR_INDICES_BY_LABEL = _build_neighbor_indices(SUPPORTED_BOARD_SIZE)


def _opponent(color: MoveColor) -> MoveColor:
    return "w" if color == "b" else "b"
