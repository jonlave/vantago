from __future__ import annotations

from pathlib import Path

import pytest

from vantago.replay import diagnose_sgf_replay_bytes, diagnose_sgf_replay_file
from vantago.sgf import BoardPoint


def test_valid_19x19_game_replays_successfully() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"(;FF[4]GM[1]SZ[19];B[aa];W[sa])",
        source_name="valid.sgf",
    )

    assert diagnostic.status == "ok"
    assert diagnostic.reason is None
    assert diagnostic.source_name == "valid.sgf"
    assert diagnostic.move_count == 2
    assert len(diagnostic.replay_steps) == 2
    last_board = diagnostic.replay_steps[-1].board_after
    assert last_board.stone_at(BoardPoint(row=0, col=18)) == "w"


def test_non_19x19_board_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"(;FF[4]GM[1]SZ[9];B[aa])",
        source_name="small-board.sgf",
    )

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "non_19x19_board"
    assert diagnostic.move_count == 1
    assert diagnostic.replay_steps == ()
    assert "expected 19" in diagnostic.message


def test_handicap_game_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"(;FF[4]GM[1]SZ[19]HA[2];B[dd])",
        source_name="handicap.sgf",
    )

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "handicap"
    assert diagnostic.move_count == 0
    assert diagnostic.replay_steps == ()
    assert "handicap" in diagnostic.message


def test_setup_stones_are_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"(;FF[4]GM[1]SZ[19]AB[dd];B[pp])",
        source_name="setup.sgf",
    )

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "setup_stones"
    assert diagnostic.move_count == 0
    assert diagnostic.replay_steps == ()
    assert "setup stone" in diagnostic.message


def test_malformed_sgf_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"not an sgf",
        source_name="malformed.sgf",
    )

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "malformed_sgf"
    assert diagnostic.move_count == 0
    assert diagnostic.replay_steps == ()
    assert "unable to parse SGF" in diagnostic.message


@pytest.mark.parametrize(
    ("sgf_content", "expected_message"),
    (
        (b"(;FF[4]GM[3]SZ[19];B[aa])", "expected GM[1]"),
        (b"(;FF[4]GM[1]SZ[19];B[dd](;W[pp])(;W[qq]))", "variation"),
        (b"(;FF[4]GM[1]SZ[19];B[aa]W[bb])", "both B and W"),
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

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "unsupported_sgf_feature"
    assert diagnostic.move_count == 0
    assert diagnostic.replay_steps == ()
    assert expected_message in diagnostic.message


def test_unreadable_sgf_file_is_skipped_with_read_message(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.sgf"

    diagnostic = diagnose_sgf_replay_file(missing_path)

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "malformed_sgf"
    assert diagnostic.source_name == str(missing_path)
    assert diagnostic.move_count == 0
    assert diagnostic.replay_steps == ()
    assert "unable to read SGF file" in diagnostic.message


def test_illegal_move_sequence_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"(;FF[4]GM[1]SZ[19];B[aa];W[aa])",
        source_name="occupied.sgf",
    )

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "illegal_move_sequence"
    assert diagnostic.move_count == 2
    assert diagnostic.replay_steps == ()
    assert "occupied" in diagnostic.message


def test_pass_move_is_skipped_with_reason() -> None:
    diagnostic = diagnose_sgf_replay_bytes(
        b"(;FF[4]GM[1]SZ[19];B[aa];W[])",
        source_name="pass.sgf",
    )

    assert diagnostic.status == "skipped"
    assert diagnostic.reason == "pass_move"
    assert diagnostic.move_count == 2
    assert diagnostic.replay_steps == ()
    assert "361-point target space" in diagnostic.message
