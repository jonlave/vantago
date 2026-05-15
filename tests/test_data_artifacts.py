from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vantago.data.artifacts as artifacts_module
from vantago.cli.main import main
from vantago.data import (
    ProcessedDatasetError,
    ProcessedDatasetSkipCount,
    decode_label,
    inspect_processed_dataset,
    load_processed_dataset,
    write_processed_dataset,
)
from vantago.replay import (
    ReplayDiagnostic,
    ReplaySkipReason,
    diagnose_sgf_replay_file,
)
from vantago.sgf import BoardPoint

SGF_FILE_FORMAT = 4
SGF_GO_GAME_TYPE = 1
SUPPORTED_BOARD_SIZE = 19


def test_write_processed_dataset_persists_valid_npz_in_stable_order(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "sgf"
    first_game = input_dir / "a_game.sgf"
    second_game = input_dir / "nested" / "b_game.sgf"
    _write_sgf(first_game, _sgf(sequence=";B[aa];W[bb]"))
    _write_sgf(second_game, _sgf(sequence=";B[cc]"))
    output = tmp_path / "processed" / "dataset.npz"

    result = write_processed_dataset(input_dir, output)

    assert result.files_scanned == 2
    assert result.ok == 2
    assert result.skipped == 0
    assert result.failed == 0
    assert result.records_written == 3

    artifact = load_processed_dataset(output)
    assert artifact.x.shape == (3, 5, SUPPORTED_BOARD_SIZE, SUPPORTED_BOARD_SIZE)
    assert artifact.x.dtype == np.float32
    assert artifact.y.dtype == np.int64
    assert artifact.legal_mask.shape == (3, SUPPORTED_BOARD_SIZE * SUPPORTED_BOARD_SIZE)
    assert artifact.legal_mask.dtype == np.bool_
    assert artifact.game_id.tolist() == [
        "a_game.sgf",
        "a_game.sgf",
        "nested/b_game.sgf",
    ]
    assert artifact.source_name.tolist() == [
        str(first_game),
        str(first_game),
        str(second_game),
    ]
    assert artifact.move_number.tolist() == [1, 2, 1]
    assert [decode_label(int(label)) for label in artifact.y.tolist()] == [
        BoardPoint(row=0, col=0),
        BoardPoint(row=1, col=1),
        BoardPoint(row=2, col=2),
    ]
    assert np.all(artifact.legal_mask[np.arange(3), artifact.y])


def test_write_processed_dataset_is_repeatable_for_loaded_arrays(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "sgf"
    _write_sgf(input_dir / "a_game.sgf", _sgf(sequence=";B[aa];W[bb]"))
    first_output = tmp_path / "first.npz"
    second_output = tmp_path / "second.npz"

    write_processed_dataset(input_dir, first_output)
    write_processed_dataset(input_dir, second_output)

    first = load_processed_dataset(first_output)
    second = load_processed_dataset(second_output)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(first.legal_mask, second.legal_mask)
    np.testing.assert_array_equal(first.game_id, second.game_id)
    np.testing.assert_array_equal(first.move_number, second.move_number)
    np.testing.assert_array_equal(first.source_name, second.source_name)


def test_write_processed_dataset_reports_skips_while_writing_valid_records(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "sgf"
    valid_game = input_dir / "a_valid.sgf"
    pass_game = input_dir / "b_pass.sgf"
    _write_sgf(valid_game, _sgf(sequence=";B[aa]"))
    _write_sgf(pass_game, _sgf(sequence=";B[aa];W[]"))
    output = tmp_path / "dataset.npz"

    result = write_processed_dataset(input_dir, output)

    assert result.ok == 1
    assert result.skipped == 1
    assert result.failed == 0
    assert result.records_written == 1
    assert result.skipped_by_reason == (
        ProcessedDatasetSkipCount(reason=ReplaySkipReason.PASS_MOVE, count=1),
    )
    assert result.skipped_diagnostics[0].source_name == str(pass_game)
    artifact = load_processed_dataset(output)
    assert artifact.game_id.tolist() == ["a_valid.sgf"]


def test_process_dataset_cli_prints_summary_and_writes_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "sgf"
    _write_sgf(input_dir / "a_valid.sgf", _sgf(sequence=";B[aa]"))
    _write_sgf(input_dir / "b_pass.sgf", _sgf(sequence=";B[aa];W[]"))
    output = tmp_path / "dataset.npz"

    exit_code = main(["process-dataset", str(input_dir), str(output)])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f"input: {input_dir}\n"
        f"output: {output}\n"
        "files_scanned: 2\n"
        "ok: 1\n"
        "skipped: 1\n"
        "failed: 0\n"
        "records_written: 1\n"
        "skipped_by_reason:\n"
        "  pass_move: 1\n"
        "skipped_files:\n"
        f"  pass_move: {input_dir / 'b_pass.sgf'}: "
        "skipped pass move at move 2; pass is not in the "
        "361-point target space\n"
        "failed_files:\n"
        "  none\n"
    )
    assert load_processed_dataset(output).y.tolist() == [0]


def test_process_dataset_cli_reports_missing_or_non_sgf_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("not an sgf", encoding="utf-8")

    exit_code = main(["process-dataset", str(notes_path), str(tmp_path / "out.npz")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        f"error: SGF batch file does not have .sgf suffix: {notes_path}\n"
    )


def test_process_dataset_cli_rejects_output_without_npz_suffix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "sgf"
    _write_sgf(input_dir / "game.sgf", _sgf(sequence=";B[aa]"))
    output = tmp_path / "artifact"

    exit_code = main(["process-dataset", str(input_dir), str(output)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        f"error: processed dataset output path must end with .npz: {output}\n"
    )
    assert not output.exists()
    assert not output.with_suffix(".npz").exists()


def test_process_dataset_cli_reports_output_directory_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "sgf"
    _write_sgf(input_dir / "game.sgf", _sgf(sequence=";B[aa]"))
    output = tmp_path / "dataset.npz"
    output.mkdir()

    exit_code = main(["process-dataset", str(input_dir), str(output)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        f"error: processed dataset output path is a directory: {output}\n"
    )
    assert "Traceback" not in captured.err


def test_process_dataset_cli_fails_when_no_records_are_encoded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "sgf"
    _write_sgf(input_dir / "pass.sgf", _sgf(sequence=";B[aa];W[]"))
    output = tmp_path / "dataset.npz"
    output.write_text("stale", encoding="utf-8")

    exit_code = main(["process-dataset", str(input_dir), str(output)])

    assert exit_code == 1
    assert "records_written: 0\n" in capsys.readouterr().out
    assert not output.exists()


def test_write_processed_dataset_removes_stale_output_on_unexpected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "sgf"
    _write_sgf(input_dir / "a_valid.sgf", _sgf(sequence=";B[aa]"))
    _write_sgf(input_dir / "b_boom.sgf", _sgf(sequence=";B[bb]"))
    output = tmp_path / "dataset.npz"
    output.write_text("stale", encoding="utf-8")

    def diagnose(path: Path) -> ReplayDiagnostic:
        if path.name == "b_boom.sgf":
            raise RuntimeError("boom")
        return diagnose_sgf_replay_file(path)

    monkeypatch.setattr(artifacts_module, "diagnose_sgf_replay_file", diagnose)

    result = artifacts_module.write_processed_dataset(input_dir, output)

    assert result.ok == 1
    assert result.failed == 1
    assert result.records_written == 0
    assert not output.exists()


def test_inspect_dataset_cli_prints_decoded_record_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "sgf"
    sgf_path = input_dir / "game.sgf"
    _write_sgf(sgf_path, _sgf(sequence=";B[aa];W[bb]"))
    output = tmp_path / "dataset.npz"
    write_processed_dataset(input_dir, output)

    exit_code = main(["inspect-dataset", str(output), "--index", "0"])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f"path: {output}\n"
        "record_count: 2\n"
        "index: 0\n"
        "game_id: game.sgf\n"
        f"source_name: {sgf_path}\n"
        "move_number: 1\n"
        "y: 0\n"
        "decoded_y: row=0 col=0\n"
        "x_shape: [5, 19, 19]\n"
        "legal_mask_count: 361\n"
        "label_is_legal: true\n"
    )


def test_inspect_dataset_cli_reports_out_of_range_index(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "sgf"
    _write_sgf(input_dir / "game.sgf", _sgf(sequence=";B[aa]"))
    output = tmp_path / "dataset.npz"
    write_processed_dataset(input_dir, output)

    exit_code = main(["inspect-dataset", str(output), "--index", "1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: index must be in [0, 1), got 1\n"


def test_load_processed_dataset_rejects_invalid_artifact_arrays(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "bad.npz"
    np.savez(
        artifact_path,
        x=np.zeros((1, 5, 19, 19), dtype=np.float64),
        y=np.array([0], dtype=np.int64),
        legal_mask=np.ones((1, 361), dtype=np.bool_),
        game_id=np.array(["game.sgf"], dtype=np.str_),
        move_number=np.array([1], dtype=np.int64),
        source_name=np.array(["game.sgf"], dtype=np.str_),
    )

    with pytest.raises(ProcessedDatasetError, match="x must have float32 dtype"):
        load_processed_dataset(artifact_path)


def test_inspect_processed_dataset_rejects_invalid_label_index(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "bad.npz"
    np.savez(
        artifact_path,
        x=np.zeros((1, 5, 19, 19), dtype=np.float32),
        y=np.array([361], dtype=np.int64),
        legal_mask=np.ones((1, 361), dtype=np.bool_),
        game_id=np.array(["game.sgf"], dtype=np.str_),
        move_number=np.array([1], dtype=np.int64),
        source_name=np.array(["game.sgf"], dtype=np.str_),
    )

    with pytest.raises(ProcessedDatasetError, match="y values must be in"):
        inspect_processed_dataset(artifact_path, index=0)


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
