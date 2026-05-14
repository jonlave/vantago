from __future__ import annotations

from pathlib import Path

from vantago.replay import diagnose_sgf_replay_file, replay_sgf_batch
from vantago.sgf import BoardPoint

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "aeb-small-100"
AEB_SMALL_CORPUS_FILE_COUNT = 100
AEB_SMALL_CORPUS_MOVE_COUNT = 20_065
AEB_CAPTURE_SPOT_CHECK = CORPUS_DIR / "games" / "AJ1st" / "01" / "1.sgf"


def test_aeb_small_corpus_replays_100_valid_games() -> None:
    result = replay_sgf_batch(CORPUS_DIR)

    assert result.files_scanned == AEB_SMALL_CORPUS_FILE_COUNT
    assert result.ok == AEB_SMALL_CORPUS_FILE_COUNT
    assert result.skipped == 0
    assert result.failed == 0
    assert result.moves_replayed == AEB_SMALL_CORPUS_MOVE_COUNT
    assert result.skipped_by_reason == ()
    assert result.skipped_diagnostics == ()
    assert result.failures == ()


def test_aeb_small_corpus_spot_checks_ordinary_move_and_capture() -> None:
    diagnostic = diagnose_sgf_replay_file(AEB_CAPTURE_SPOT_CHECK)

    first_step = diagnostic.replay_steps[0]
    assert first_step.move.point == BoardPoint(row=3, col=16)
    assert first_step.board_before.stone_at(BoardPoint(row=3, col=16)) is None
    assert first_step.board_after.stone_at(BoardPoint(row=3, col=16)) == "b"

    capture_step = diagnostic.replay_steps[78]
    captured_point = BoardPoint(row=8, col=4)
    assert capture_step.move_number == 79
    assert capture_step.move.point == BoardPoint(row=8, col=3)
    assert capture_step.board_before.stone_at(captured_point) == "w"
    assert capture_step.board_after.stone_at(captured_point) is None
