from __future__ import annotations

from pathlib import Path

import pytest

import vantago.replay.batch as batch_module
from vantago.cli.main import main
from vantago.replay import (
    ReplayBatchOkGame,
    ReplayBatchSkipCount,
    ReplayDiagnostic,
    ReplaySkipReason,
    find_sgf_files,
    replay_sgf_batch,
)

SGF_FILE_FORMAT = 4
SGF_GO_GAME_TYPE = 1
SUPPORTED_BOARD_SIZE = 19
SMALL_BOARD_SIZE = 9


def test_find_sgf_files_returns_recursive_paths_in_stable_order(
    tmp_path: Path,
) -> None:
    _write_sgf(tmp_path / "z.sgf", _sgf(sequence=";B[aa]"))
    _write_sgf(tmp_path / "nested" / "a.SGF", _sgf(sequence=";B[bb]"))
    (tmp_path / "notes.txt").write_text("not an sgf", encoding="utf-8")

    assert [
        path.relative_to(tmp_path).as_posix()
        for path in find_sgf_files(tmp_path)
    ] == ["nested/a.SGF", "z.sgf"]


def test_replay_sgf_batch_aggregates_ok_skips_and_moves(tmp_path: Path) -> None:
    _write_sgf(tmp_path / "a_valid.sgf", _sgf(sequence=";B[aa];W[sa]"))
    _write_sgf(tmp_path / "b_pass.sgf", _sgf(sequence=";B[aa];W[]"))
    _write_sgf(tmp_path / "c_empty.sgf", _sgf(sequence=""))
    _write_sgf(
        tmp_path / "d_small.sgf",
        _sgf(board_size=SMALL_BOARD_SIZE, sequence=";B[aa]"),
    )
    _write_sgf(tmp_path / "e_illegal.sgf", _sgf(sequence=";B[aa];W[aa]"))
    _write_sgf(tmp_path / "z_malformed.sgf", "not an sgf")

    result = replay_sgf_batch(tmp_path)

    assert result.files_scanned == 6
    assert result.ok == 1
    assert result.skipped == 5
    assert result.failed == 0
    assert result.moves_replayed == 2
    assert result.ok_games == (
        ReplayBatchOkGame(source_name=str(tmp_path / "a_valid.sgf"), move_count=2),
    )
    assert _skip_counts(result.skipped_by_reason) == {
        ReplaySkipReason.NON_19X19_BOARD: 1,
        ReplaySkipReason.EMPTY_GAME: 1,
        ReplaySkipReason.PASS_MOVE: 1,
        ReplaySkipReason.MALFORMED_SGF: 1,
        ReplaySkipReason.ILLEGAL_MOVE_SEQUENCE: 1,
    }
    assert [diagnostic.source_name for diagnostic in result.skipped_diagnostics] == [
        str(tmp_path / "b_pass.sgf"),
        str(tmp_path / "c_empty.sgf"),
        str(tmp_path / "d_small.sgf"),
        str(tmp_path / "e_illegal.sgf"),
        str(tmp_path / "z_malformed.sgf"),
    ]


def test_replay_sgf_batch_records_unexpected_file_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sgf_path = tmp_path / "valid.sgf"
    _write_sgf(sgf_path, _sgf(sequence=";B[aa]"))

    def fail_diagnostic(path: Path) -> ReplayDiagnostic:
        raise RuntimeError(f"boom: {path.name}")

    monkeypatch.setattr(batch_module, "diagnose_sgf_replay_file", fail_diagnostic)

    result = replay_sgf_batch(tmp_path)

    assert result.files_scanned == 1
    assert result.ok == 0
    assert result.skipped == 0
    assert result.failed == 1
    assert result.failures[0].source_name == str(sgf_path)
    assert result.failures[0].message == "boom: valid.sgf"


def test_replay_batch_cli_fails_when_directory_has_no_sgfs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "notes.txt").write_text("not an sgf", encoding="utf-8")

    exit_code = main(["replay-batch", str(tmp_path)])

    assert exit_code == 1
    assert capsys.readouterr().out == (
        "files_scanned: 0\n"
        "ok: 0\n"
        "skipped: 0\n"
        "failed: 0\n"
        "moves_replayed: 0\n"
        "skipped_by_reason:\n"
        "  none\n"
        "skipped_files:\n"
        "  none\n"
        "failed_files:\n"
        "  none\n"
    )


def test_replay_batch_cli_fails_when_every_sgf_is_skipped(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_sgf(tmp_path / "pass.sgf", _sgf(sequence=";B[aa];W[]"))

    exit_code = main(["replay-batch", str(tmp_path)])

    assert exit_code == 1
    assert "ok: 0\n" in capsys.readouterr().out


def test_replay_batch_cli_prints_summary_and_per_file_skips(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_sgf(tmp_path / "a_valid.sgf", _sgf(sequence=";B[aa];W[sa]"))
    _write_sgf(tmp_path / "b_pass.sgf", _sgf(sequence=";B[aa];W[]"))

    exit_code = main(["replay-batch", str(tmp_path)])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "files_scanned: 2\n"
        "ok: 1\n"
        "skipped: 1\n"
        "failed: 0\n"
        "moves_replayed: 2\n"
        "skipped_by_reason:\n"
        "  pass_move: 1\n"
        "skipped_files:\n"
        f"  pass_move: {tmp_path / 'b_pass.sgf'}: "
        "skipped pass move at move 2; pass is not in the "
        "361-point target space\n"
        "failed_files:\n"
        "  none\n"
    )


def test_replay_batch_cli_reports_missing_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing"

    exit_code = main(["replay-batch", str(missing_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == f"error: SGF batch path does not exist: {missing_path}\n"


def _sgf(
    *,
    board_size: int = SUPPORTED_BOARD_SIZE,
    sequence: str,
) -> str:
    return (
        f"(;FF[{SGF_FILE_FORMAT}]GM[{SGF_GO_GAME_TYPE}]"
        f"SZ[{board_size}]{sequence})"
    )


def _write_sgf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _skip_counts(
    skipped_by_reason: tuple[ReplayBatchSkipCount, ...],
) -> dict[ReplaySkipReason, int]:
    return {
        skip_count.reason: skip_count.count
        for skip_count in skipped_by_reason
    }
