from __future__ import annotations

import numpy as np
import pytest

from vantago.data import (
    PositionEncodingError,
    decode_label,
    encode_label,
    encode_legal_mask,
    encode_replay_steps,
)
from vantago.replay import BoardState, replay_game
from vantago.sgf import BoardPoint, GameMetadata, ParsedGame, ParsedMove

BOARD_SIZE = 19
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


def test_encode_replay_steps_produces_model_ready_records() -> None:
    first_move = BoardPoint(row=3, col=3)
    second_move = BoardPoint(row=16, col=3)
    game = _game(
        ParsedMove(color="b", point=first_move),
        ParsedMove(color="w", point=second_move),
    )

    records = encode_replay_steps("game-1", replay_game(game))

    first_record = records[0]
    assert first_record.game_id == "game-1"
    assert first_record.move_number == 1
    assert first_record.y == encode_label(first_move)
    assert first_record.x.shape == (5, BOARD_SIZE, BOARD_SIZE)
    assert first_record.x.dtype == np.float32
    assert first_record.legal_mask.shape == (POINT_COUNT,)
    assert first_record.legal_mask.dtype == np.bool_
    assert first_record.legal_mask[first_record.y]
    assert first_record.x[0].sum() == 0
    assert first_record.x[1].sum() == 0
    assert first_record.x[2].sum() == POINT_COUNT
    assert first_record.x[3].sum() == 0
    assert first_record.x[4].sum() == POINT_COUNT
    np.testing.assert_array_equal(first_record.x[4].ravel(), first_record.legal_mask)

    second_record = records[1]
    assert second_record.y == encode_label(second_move)
    assert second_record.x[1, first_move.row, first_move.col] == 1.0
    assert second_record.x[2].sum() == POINT_COUNT - 1
    assert second_record.x[3, first_move.row, first_move.col] == 1.0
    assert second_record.x[4, first_move.row, first_move.col] == 0.0
    assert second_record.legal_mask[second_record.y]
    np.testing.assert_array_equal(second_record.x[4].ravel(), second_record.legal_mask)


def test_encoder_flips_stone_channels_to_current_player_perspective() -> None:
    black_move = BoardPoint(row=3, col=3)
    white_move = BoardPoint(row=16, col=3)
    next_black_move = BoardPoint(row=3, col=4)
    game = _game(
        ParsedMove(color="b", point=black_move),
        ParsedMove(color="w", point=white_move),
        ParsedMove(color="b", point=next_black_move),
    )

    records = encode_replay_steps("perspective", replay_game(game))

    black_to_move_record = records[2]
    assert black_to_move_record.x[0, black_move.row, black_move.col] == 1.0
    assert black_to_move_record.x[1, white_move.row, white_move.col] == 1.0
    assert black_to_move_record.x[3, white_move.row, white_move.col] == 1.0


def test_label_helpers_round_trip_row_major_points() -> None:
    point = BoardPoint(row=4, col=5)

    label = encode_label(point)

    assert label == 81
    assert decode_label(label) == point


@pytest.mark.parametrize("label", (-1, POINT_COUNT))
def test_decode_label_rejects_out_of_range_values(label: int) -> None:
    with pytest.raises(ValueError, match="label must be in"):
        decode_label(label)


def test_legal_mask_matches_replay_legality_for_occupied_and_suicide_points() -> None:
    occupied_point = BoardPoint(row=0, col=1)
    suicide_point = BoardPoint(row=1, col=1)
    ordinary_point = BoardPoint(row=5, col=5)
    board = _apply_moves(
        ParsedMove(color="b", point=occupied_point),
        ParsedMove(color="b", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="b", point=BoardPoint(row=1, col=2)),
        ParsedMove(color="b", point=BoardPoint(row=2, col=1)),
    )

    legal_mask = encode_legal_mask(board, "w")

    assert not legal_mask[encode_label(occupied_point)]
    assert not legal_mask[encode_label(suicide_point)]
    assert legal_mask[encode_label(ordinary_point)]
    assert not board.is_legal_point("w", suicide_point)
    assert board.is_legal_point("w", ordinary_point)


def test_legal_mask_matches_replay_legality_oracle_for_every_point() -> None:
    capture_point = BoardPoint(row=1, col=2)
    board = _apply_moves(
        ParsedMove(color="b", point=BoardPoint(row=1, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=0, col=1)),
        ParsedMove(color="w", point=BoardPoint(row=1, col=0)),
        ParsedMove(color="w", point=BoardPoint(row=2, col=1)),
    )

    legal_mask = encode_legal_mask(board, "w")
    expected_mask = np.array(
        [
            board.is_legal_point("w", decode_label(label))
            for label in range(POINT_COUNT)
        ],
        dtype=np.bool_,
    )

    assert legal_mask[encode_label(capture_point)]
    np.testing.assert_array_equal(legal_mask, expected_mask)


def test_encode_replay_steps_rejects_pass_targets() -> None:
    game = _game(ParsedMove(color="b", point=None))

    with pytest.raises(PositionEncodingError, match="pass moves"):
        encode_replay_steps("pass-game", replay_game(game))


def test_encode_replay_steps_rejects_non_19x19_boards() -> None:
    game = _game(
        ParsedMove(color="b", point=BoardPoint(row=0, col=0)),
        board_size=9,
    )

    with pytest.raises(PositionEncodingError, match="19x19"):
        encode_replay_steps("small-board", replay_game(game))


def _apply_moves(*moves: ParsedMove) -> BoardState:
    board = BoardState.empty(board_size=BOARD_SIZE)
    for move in moves:
        board = board.apply_move(move)
    return board


def _game(*moves: ParsedMove, board_size: int = BOARD_SIZE) -> ParsedGame:
    return ParsedGame(
        source_name="inline",
        metadata=GameMetadata(
            board_size=board_size,
            black_player=None,
            white_player=None,
            komi=None,
            result=None,
        ),
        moves=moves,
    )
