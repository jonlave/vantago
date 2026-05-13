from __future__ import annotations

from pathlib import Path

import pytest

from vantago.sgf import (
    BoardPoint,
    ParsedMove,
    SgfParseError,
    load_sgf,
    parse_sgf_bytes,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sgf" / "known_good.sgf"


def test_known_good_sgf_loads_metadata_and_moves() -> None:
    game = load_sgf(FIXTURE_PATH)

    assert game.metadata.board_size == 19
    assert game.metadata.black_player == "Black Test"
    assert game.metadata.white_player == "White Test"
    assert game.metadata.komi == "6.5"
    assert game.metadata.result == "B+R"
    assert game.moves == (
        ParsedMove(color="b", point=BoardPoint(row=0, col=0)),
        ParsedMove(color="w", point=BoardPoint(row=0, col=18)),
        ParsedMove(color="b", point=BoardPoint(row=18, col=0)),
        ParsedMove(color="w", point=BoardPoint(row=3, col=3)),
        ParsedMove(color="b", point=None),
    )


def test_flat_index_uses_row_major_board_coordinates() -> None:
    assert BoardPoint(row=3, col=3).flat_index(board_size=19) == 60


def test_pass_move_is_not_encoded_as_a_board_point() -> None:
    game = load_sgf(FIXTURE_PATH)

    assert game.moves[-1].point is None


def test_setup_stones_are_rejected_explicitly() -> None:
    sgf = b"(;FF[4]GM[1]SZ[19]AB[dd];B[pp])"

    with pytest.raises(SgfParseError, match="unsupported setup stone"):
        parse_sgf_bytes(sgf, source_name="setup.sgf")


def test_variations_are_rejected_explicitly() -> None:
    sgf = b"(;FF[4]GM[1]SZ[19];B[dd](;W[pp])(;W[qq]))"

    with pytest.raises(SgfParseError, match="unsupported variation"):
        parse_sgf_bytes(sgf, source_name="variation.sgf")


def test_non_go_game_type_is_rejected_explicitly() -> None:
    sgf = b"(;FF[4]GM[3]SZ[19];B[aa])"

    with pytest.raises(SgfParseError, match=r"expected GM\[1\]"):
        parse_sgf_bytes(sgf, source_name="chess.sgf")


def test_missing_game_type_is_rejected_explicitly() -> None:
    sgf = b"(;FF[4]SZ[19];B[aa])"

    with pytest.raises(SgfParseError, match=r"expected GM\[1\]"):
        parse_sgf_bytes(sgf, source_name="missing-gm.sgf")


def test_move_node_with_both_colors_is_rejected_explicitly() -> None:
    sgf = b"(;FF[4]GM[1]SZ[19];B[aa]W[bb])"

    with pytest.raises(SgfParseError, match="both B and W"):
        parse_sgf_bytes(sgf, source_name="both-colors.sgf")
