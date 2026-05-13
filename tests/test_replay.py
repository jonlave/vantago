from __future__ import annotations

from typing import Any, cast

import pytest

from vantago.replay import BoardState, IllegalMoveError, replay_game
from vantago.sgf import BoardPoint, GameMetadata, ParsedGame, ParsedMove


def test_replay_game_records_before_and_after_board_states() -> None:
    game = _game(
        ParsedMove(color="b", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=0, col=0)),
        ParsedMove(color="b", point=None),
    )

    steps = replay_game(game)

    assert [step.move_number for step in steps] == [1, 2, 3]
    assert steps[0].board_before.stone_at(BoardPoint(row=1, col=1)) is None
    assert steps[0].board_after.stone_at(BoardPoint(row=1, col=1)) == "b"
    assert steps[1].board_before.stone_at(BoardPoint(row=0, col=0)) is None
    assert steps[1].board_after.stone_at(BoardPoint(row=0, col=0)) == "w"
    assert steps[2].board_before == steps[2].board_after


def test_ordinary_moves_update_board_without_captures() -> None:
    board = BoardState.empty(board_size=5)

    board = board.apply_move(ParsedMove(color="b", point=BoardPoint(row=2, col=2)))
    board = board.apply_move(ParsedMove(color="w", point=BoardPoint(row=3, col=2)))

    assert board.stone_at(BoardPoint(row=2, col=2)) == "b"
    assert board.stone_at(BoardPoint(row=3, col=2)) == "w"


def test_board_state_normalizes_mutable_stones_to_tuple() -> None:
    mutable_stones: list[Any] = [None, "b", None, None]

    board = BoardState(board_size=2, stones=cast(Any, mutable_stones))
    mutable_stones[1] = None

    assert board.stones == (None, "b", None, None)
    assert board.stone_at(BoardPoint(row=0, col=1)) == "b"


def test_occupied_point_move_is_rejected() -> None:
    board = BoardState.empty(board_size=5).apply_move(
        ParsedMove(color="b", point=BoardPoint(row=2, col=2))
    )

    with pytest.raises(IllegalMoveError, match="occupied"):
        board.apply_move(ParsedMove(color="w", point=BoardPoint(row=2, col=2)))


def test_out_of_bounds_move_raises_illegal_move_error() -> None:
    board = BoardState.empty(board_size=3)

    with pytest.raises(IllegalMoveError, match="outside board size 3"):
        board.apply_move(ParsedMove(color="b", point=BoardPoint(row=3, col=0)))


def test_single_stone_capture_removes_captured_stone() -> None:
    board = _apply_moves(
        3,
        ParsedMove(color="w", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=2)),
    )

    assert board.stone_at(BoardPoint(row=1, col=1)) is None
    assert board.stone_at(BoardPoint(row=1, col=2)) == "b"


def test_multi_stone_capture_removes_full_group() -> None:
    board = _apply_moves(
        5,
        ParsedMove(color="w", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=1, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=3)),
    )

    assert board.stone_at(BoardPoint(row=1, col=1)) is None
    assert board.stone_at(BoardPoint(row=1, col=2)) is None
    assert board.stone_at(BoardPoint(row=1, col=3)) == "b"


def test_one_move_can_capture_multiple_separate_groups() -> None:
    board = _apply_moves(
        3,
        ParsedMove(color="w", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=0)),
    )

    assert board.stone_at(BoardPoint(row=0, col=1)) is None
    assert board.stone_at(BoardPoint(row=1, col=0)) is None
    assert board.stone_at(BoardPoint(row=0, col=0)) == "b"


def test_corner_move_can_capture_two_adjacent_side_stones() -> None:
    board = _apply_moves(
        3,
        ParsedMove(color="w", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=0)),
    )

    assert board.stone_at(BoardPoint(row=0, col=1)) is None
    assert board.stone_at(BoardPoint(row=1, col=0)) is None
    assert board.stone_at(BoardPoint(row=0, col=0)) == "b"


def test_edge_move_can_capture_chain_along_board_side() -> None:
    board = _apply_moves(
        4,
        ParsedMove(color="w", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=0, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=3)),
    )

    assert board.stone_at(BoardPoint(row=0, col=1)) is None
    assert board.stone_at(BoardPoint(row=0, col=2)) is None
    assert board.stone_at(BoardPoint(row=0, col=3)) == "b"


def test_capture_removes_only_adjacent_groups_without_liberties() -> None:
    board = _apply_moves(
        4,
        ParsedMove(color="w", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="w", point=BoardPoint(row=1, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=1)),
    )

    assert board.stone_at(BoardPoint(row=1, col=0)) is None
    assert board.stone_at(BoardPoint(row=1, col=2)) == "w"
    assert board.stone_at(BoardPoint(row=1, col=3)) is None


def test_diagonal_stones_are_not_part_of_captured_group() -> None:
    board = _apply_moves(
        4,
        ParsedMove(color="w", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=2, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=2)),
    )

    assert board.stone_at(BoardPoint(row=1, col=1)) is None
    assert board.stone_at(BoardPoint(row=2, col=2)) == "w"
    assert board.stone_at(BoardPoint(row=1, col=2)) == "b"


def test_captured_stones_are_removed_before_own_liberty_check() -> None:
    board = _apply_moves(
        5,
        ParsedMove(color="w", point=BoardPoint(row=1, col=2)),
        ParsedMove(color="w", point=BoardPoint(row=2, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=2, col=3)),
        ParsedMove(color="w", point=BoardPoint(row=3, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=0, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=3)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=4)),
        ParsedMove(color="b", point=BoardPoint(row=3, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=3, col=3)),
        ParsedMove(color="b", point=BoardPoint(row=4, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=2)),
    )

    for point in (
        BoardPoint(row=1, col=2),
        BoardPoint(row=2, col=1),
        BoardPoint(row=2, col=3),
        BoardPoint(row=3, col=2),
    ):
        assert board.stone_at(point) is None
    assert board.stone_at(BoardPoint(row=2, col=2)) == "b"


def test_move_without_liberties_is_rejected() -> None:
    board = _apply_moves(
        3,
        ParsedMove(color="b", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=1)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=2)),
    )

    with pytest.raises(IllegalMoveError, match="no liberties"):
        board.apply_move(ParsedMove(color="w", point=BoardPoint(row=1, col=1)))


def _apply_moves(board_size: int, *moves: ParsedMove) -> BoardState:
    board = BoardState.empty(board_size=board_size)
    for move in moves:
        board = board.apply_move(move)
    return board


def _game(*moves: ParsedMove) -> ParsedGame:
    return ParsedGame(
        source_name="inline",
        metadata=GameMetadata(
            board_size=5,
            black_player=None,
            white_player=None,
            komi=None,
            result=None,
        ),
        moves=moves,
    )
