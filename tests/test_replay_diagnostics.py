from __future__ import annotations

from pathlib import Path

import pytest

from vantago.replay import (
    SUPPORTED_BOARD_SIZE,
    SUPPORTED_POINT_COUNT,
    ReplayDiagnostic,
    ReplayDiagnosticStatus,
    ReplaySkipReason,
    diagnose_sgf_replay_bytes,
    diagnose_sgf_replay_file,
)
from vantago.sgf import BoardPoint

SGF_FILE_FORMAT = 4
SGF_GO_GAME_TYPE = 1
SGF_UNSUPPORTED_GAME_TYPE = 3
SMALL_BOARD_SIZE = 9
HANDICAP_STONE_COUNT = 2
PARSE_SKIP_MOVE_COUNT = 0
SINGLE_MOVE_COUNT = 1
TWO_MOVE_COUNT = 2
BOARD_TOP_ROW = 0
RIGHT_EDGE_COL = SUPPORTED_BOARD_SIZE - 1


def _sgf_bytes(
    *,
    game_type: int = SGF_GO_GAME_TYPE,
    board_size: int = SUPPORTED_BOARD_SIZE,
    root_properties: str = "",
    sequence: str,
) -> bytes:
    sgf_text = (
        f"(;FF[{SGF_FILE_FORMAT}]GM[{game_type}]SZ[{board_size}]"
        f"{root_properties}{sequence})"
    )
    return sgf_text.encode()


def _assert_skipped(
    diagnostic: ReplayDiagnostic,
    *,
    reason: ReplaySkipReason,
    move_count: int,
    message_fragment: str,
) -> None:
    assert diagnostic.status == ReplayDiagnosticStatus.SKIPPED
    assert diagnostic.reason == reason
    assert diagnostic.move_count == move_count
    assert diagnostic.replay_steps == ()
    assert message_fragment in diagnostic.message


def test_valid_19x19_game_replays_successfully() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        _sgf_bytes(sequence=";B[aa];W[sa]"),
        source_name="valid.sgf",
    )

    assert diagnostic.status == ReplayDiagnosticStatus.OK
    assert diagnostic.reason is None
    assert diagnostic.source_name == "valid.sgf"
    assert diagnostic.move_count == TWO_MOVE_COUNT
    assert len(diagnostic.replay_steps) == TWO_MOVE_COUNT
    last_board = diagnostic.replay_steps[-1].board_after
    assert last_board.stone_at(BoardPoint(row=BOARD_TOP_ROW, col=RIGHT_EDGE_COL)) == "w"


def test_non_19x19_board_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        _sgf_bytes(board_size=SMALL_BOARD_SIZE, sequence=";B[aa]"),
        source_name="small-board.sgf",
    )

    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.NON_19X19_BOARD,
        move_count=SINGLE_MOVE_COUNT,
        message_fragment=f"expected {SUPPORTED_BOARD_SIZE}",
    )


def test_handicap_game_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        _sgf_bytes(
            root_properties=f"HA[{HANDICAP_STONE_COUNT}]",
            sequence=";B[dd]",
        ),
        source_name="handicap.sgf",
    )

    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.HANDICAP,
        move_count=PARSE_SKIP_MOVE_COUNT,
        message_fragment="handicap",
    )


def test_setup_stones_are_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        _sgf_bytes(root_properties="AB[dd]", sequence=";B[pp]"),
        source_name="setup.sgf",
    )

    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.SETUP_STONES,
        move_count=PARSE_SKIP_MOVE_COUNT,
        message_fragment="setup stone",
    )


def test_malformed_sgf_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"not an sgf",
        source_name="malformed.sgf",
    )

    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.MALFORMED_SGF,
        move_count=PARSE_SKIP_MOVE_COUNT,
        message_fragment="unable to parse SGF",
    )


@pytest.mark.parametrize(
    ("sgf_content", "expected_message"),
    (
        (
            _sgf_bytes(game_type=SGF_UNSUPPORTED_GAME_TYPE, sequence=";B[aa]"),
            f"expected GM[{SGF_GO_GAME_TYPE}]",
        ),
        (_sgf_bytes(sequence=";B[dd](;W[pp])(;W[qq])"), "variation"),
        (_sgf_bytes(sequence=";B[aa]W[bb]"), "both B and W"),
    ),
)
def test_unsupported_sgf_features_are_skipped_with_reason(
    sgf_content: bytes,
    expected_message: str,
) -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        sgf_content,
        source_name="unsupported.sgf",
    )

    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.UNSUPPORTED_SGF_FEATURE,
        move_count=PARSE_SKIP_MOVE_COUNT,
        message_fragment=expected_message,
    )


def test_unreadable_sgf_file_is_skipped_with_read_message(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.sgf"

    diagnostic = diagnose_sgf_replay_file(missing_path)

    assert diagnostic.source_name == str(missing_path)
    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.MALFORMED_SGF,
        move_count=PARSE_SKIP_MOVE_COUNT,
        message_fragment="unable to read SGF file",
    )


def test_illegal_move_sequence_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        _sgf_bytes(sequence=";B[aa];W[aa]"),
        source_name="occupied.sgf",
    )

    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.ILLEGAL_MOVE_SEQUENCE,
        move_count=TWO_MOVE_COUNT,
        message_fragment="occupied",
    )


def test_pass_move_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        _sgf_bytes(sequence=";B[aa];W[]"),
        source_name="pass.sgf",
    )

    _assert_skipped(
        diagnostic,
        reason=ReplaySkipReason.PASS_MOVE,
        move_count=TWO_MOVE_COUNT,
        message_fragment=f"{SUPPORTED_POINT_COUNT}-point target space",
    )

