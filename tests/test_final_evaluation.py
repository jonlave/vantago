from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vantago.baselines import PHASE_NAMES
from vantago.cli.main import main
from vantago.final_evaluation import (
    MODEL_SELECTION_METHOD,
    FinalEvaluationConfig,
    generate_final_evaluation_report,
    validate_final_evaluation_json_output_path,
    write_final_evaluation_report_json,
)
from vantago.training import CnnTrainingConfig, train_cnn_policy

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True, slots=True)
class _FixtureRecord:
    game_id: str
    move_number: int
    y: int
    legal_labels: tuple[int, ...]


def test_generate_final_evaluation_report_combines_test_rows_and_selection(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(
        tmp_path,
        dataset_path,
        manifest_path,
        mask_topk=True,
    )

    report = generate_final_evaluation_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
    )

    assert report.split == "test"
    assert report.resolved_mask_topk is True
    assert tuple(row.baseline for row in report.rows) == (
        "random_legal",
        "frequency_overall",
        "frequency_by_phase",
        "cnn_policy",
    )
    assert all(row.split == "test" for row in report.rows)
    assert all(row.metrics.example_count == 3 for row in report.rows)
    assert report.selection.model == "cnn_policy"
    assert report.selection.method == MODEL_SELECTION_METHOD
    assert report.selection.split == "validation"
    assert report.selection.best_epoch == 1
    assert report.selection.validation_metrics.example_count == 1
    assert report.game_counts == {
        "total": 10,
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    assert report.record_counts == {
        "total": 12,
        "train": 8,
        "validation": 1,
        "test": 3,
    }


def test_final_evaluation_report_phase_rows_cover_models_and_phases(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(tmp_path, dataset_path, manifest_path)

    report = generate_final_evaluation_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
    )

    assert tuple(
        (row.baseline, row.phase)
        for row in report.phase_rows
    ) == tuple(
        (baseline, phase)
        for baseline in (
            "random_legal",
            "frequency_overall",
            "frequency_by_phase",
            "cnn_policy",
        )
        for phase in PHASE_NAMES
    )
    assert all(row.split == "test" for row in report.phase_rows)
    assert all(row.metrics is not None for row in report.phase_rows)


def test_final_evaluation_json_records_provenance_without_split_lists(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(
        tmp_path,
        dataset_path,
        manifest_path,
        mask_topk=True,
    )
    report = generate_final_evaluation_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
    )
    json_path = tmp_path / "reports" / "final-evaluation.json"

    write_final_evaluation_report_json(json_path, report)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["dataset_path"] == str(dataset_path)
    assert data["manifest_path"] == str(manifest_path)
    assert data["checkpoint_path"] == str(checkpoint_path)
    assert data["split"] == "test"
    assert data["config"] == {
        "batch_size": 128,
        "seed": 0,
        "requested_mask_topk": None,
        "mask_topk": True,
    }
    assert data["split_counts"] == {
        "games": {
            "total": 10,
            "train": 8,
            "validation": 1,
            "test": 1,
        },
        "records": {
            "total": 12,
            "train": 8,
            "validation": 1,
            "test": 3,
        },
    }
    assert data["split_manifest"] == {
        "seed": 0,
        "ratios": {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        },
    }
    assert data["checkpoint"]["path"] == str(checkpoint_path)
    assert data["checkpoint"]["dataset_path"] == str(dataset_path)
    assert data["checkpoint"]["manifest_path"] == str(manifest_path)
    assert data["checkpoint"]["format_version"] == 1
    assert data["checkpoint"]["model_kind"] == "cnn_policy"
    assert data["checkpoint"]["config"]["checkpoint_path"] == str(checkpoint_path)
    assert data["checkpoint"]["config"]["epochs"] == 1
    assert data["checkpoint"]["config"]["hidden_channels"] == 1
    assert data["checkpoint"]["started_at"].endswith("Z")
    assert data["checkpoint"]["finished_at"].endswith("Z")
    assert data["checkpoint"]["duration_seconds"] > 0.0
    assert "splits" not in data
    assert data["selection"]["method"] == MODEL_SELECTION_METHOD
    assert [row["baseline"] for row in data["rows"]] == [
        "random_legal",
        "frequency_overall",
        "frequency_by_phase",
        "cnn_policy",
    ]
    assert len(data["phase_rows"]) == 4 * len(PHASE_NAMES)


def test_final_evaluation_mask_topk_can_override_checkpoint_default(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(
        tmp_path,
        dataset_path,
        manifest_path,
        mask_topk=True,
    )

    report = generate_final_evaluation_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
        config=FinalEvaluationConfig(mask_topk=False),
    )

    assert report.config.mask_topk is False
    assert report.resolved_mask_topk is False


def test_final_evaluation_report_cli_prints_sections_and_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(tmp_path, dataset_path, manifest_path)
    json_path = tmp_path / "reports" / "report.json"

    exit_code = main(
        [
            "final-evaluation-report",
            str(dataset_path),
            str(manifest_path),
            str(checkpoint_path),
            "--batch-size",
            "1",
            "--seed",
            "17",
            "--mask-topk",
            "--json-out",
            str(json_path),
        ]
    )

    assert exit_code == 0
    assert json_path.is_file()
    output = capsys.readouterr()
    assert output.err == ""
    lines = output.out.splitlines()
    assert "selection" in lines
    assert "comparison" in lines
    assert "phase_comparison" in lines
    assert "provenance" in lines
    assert "method: best_validation_cross_entropy_checkpoint" in lines
    assert "split: test" in lines
    assert "batch_size: 1" in lines
    assert "seed: 17" in lines
    assert "mask_topk: true" in lines

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["config"]["batch_size"] == 1
    assert data["config"]["seed"] == 17
    assert data["config"]["requested_mask_topk"] is True
    assert data["config"]["mask_topk"] is True


def test_final_evaluation_report_cli_honors_no_mask_topk(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(
        tmp_path,
        dataset_path,
        manifest_path,
        mask_topk=True,
    )
    json_path = tmp_path / "reports" / "report.json"

    exit_code = main(
        [
            "final-evaluation-report",
            str(dataset_path),
            str(manifest_path),
            str(checkpoint_path),
            "--no-mask-topk",
            "--json-out",
            str(json_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "mask_topk: false" in output.out.splitlines()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["config"]["requested_mask_topk"] is False
    assert data["config"]["mask_topk"] is False


@pytest.mark.parametrize(
    "artifact_name",
    ("dataset", "manifest", "checkpoint"),
)
def test_final_evaluation_json_out_rejects_artifact_collisions(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(tmp_path, dataset_path, manifest_path)
    path_by_name = {
        "dataset": dataset_path,
        "manifest": manifest_path,
        "checkpoint": checkpoint_path,
    }

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_final_evaluation_json_output_path(
            path_by_name[artifact_name],
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
        )


def test_final_evaluation_json_out_rejects_non_json_path(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(tmp_path, dataset_path, manifest_path)

    with pytest.raises(ValueError, match="must end with .json"):
        validate_final_evaluation_json_output_path(
            tmp_path / "report.txt",
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
        )


def test_final_evaluation_report_cli_rejects_json_out_artifact_collision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = _train_checkpoint(tmp_path, dataset_path, manifest_path)

    exit_code = main(
        [
            "final-evaluation-report",
            str(dataset_path),
            str(manifest_path),
            str(checkpoint_path),
            "--json-out",
            str(checkpoint_path),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "json_out must not overwrite checkpoint artifact" in output.err


def test_final_evaluation_report_cli_reports_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "final-evaluation-report",
            str(tmp_path / "missing.npz"),
            str(tmp_path / "missing-splits.json"),
            str(tmp_path / "missing.pt"),
            "--batch-size",
            "0",
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "batch_size must be positive" in output.err


def _train_checkpoint(
    tmp_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    *,
    mask_topk: bool = False,
) -> Path:
    checkpoint_path = tmp_path / "runs" / "cnn.pt"
    train_cnn_policy(
        dataset_path,
        manifest_path,
        config=CnnTrainingConfig(
            checkpoint_path=checkpoint_path,
            epochs=1,
            batch_size=4,
            hidden_channels=1,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=23,
            mask_topk=mask_topk,
        ),
    )
    return checkpoint_path


def _records() -> tuple[_FixtureRecord, ...]:
    train_records = tuple(
        _FixtureRecord(
            game_id=f"game-{index:02d}.sgf",
            move_number=1,
            y=10 + index,
            legal_labels=(10 + index,),
        )
        for index in range(8)
    )
    return (
        *train_records,
        _FixtureRecord(
            game_id="game-08.sgf",
            move_number=1,
            y=5,
            legal_labels=(5,),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=1,
            y=4,
            legal_labels=(4,),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=41,
            y=5,
            legal_labels=(5,),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=151,
            y=6,
            legal_labels=(6,),
        ),
    )


def _write_fixture(
    tmp_path: Path,
    records: Sequence[_FixtureRecord],
) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    _write_processed_artifact(dataset_path, records)
    _write_manifest(manifest_path, dataset_path, records)
    return dataset_path, manifest_path


def _write_processed_artifact(
    path: Path,
    records: Sequence[_FixtureRecord],
) -> None:
    record_count = len(records)
    legal_mask = np.zeros((record_count, POINT_COUNT), dtype=np.bool_)
    x = np.zeros(
        (record_count, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    for index, record in enumerate(records):
        legal_labels = {*record.legal_labels, record.y}
        legal_mask[index, sorted(legal_labels)] = True
        x[index, 0, record.y // BOARD_SIZE, record.y % BOARD_SIZE] = 1.0
        x[index, 2] = 1.0
        x[index, 4] = legal_mask[index].reshape(BOARD_SIZE, BOARD_SIZE)

    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.array([record.y for record in records], dtype=np.int64)
    game_id = np.array([record.game_id for record in records], dtype=np.str_)
    move_number = np.array(
        [record.move_number for record in records],
        dtype=np.int64,
    )
    np.savez(
        path,
        x=x,
        y=y,
        legal_mask=legal_mask,
        game_id=game_id,
        move_number=move_number,
        source_name=game_id,
    )


def _write_manifest(
    path: Path,
    dataset_path: Path,
    records: Sequence[_FixtureRecord],
) -> None:
    splits = {
        "train": tuple(f"game-{index:02d}.sgf" for index in range(8)),
        "validation": ("game-08.sgf",),
        "test": ("game-09.sgf",),
    }
    record_counts_by_game = _record_counts_by_game(records)
    record_counts = {
        name: sum(record_counts_by_game[game_id] for game_id in split_game_ids)
        for name, split_game_ids in splits.items()
    }
    data = {
        "dataset_path": str(dataset_path),
        "seed": 0,
        "ratios": {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        },
        "game_counts": {
            "total": 10,
            "train": 8,
            "validation": 1,
            "test": 1,
        },
        "record_counts": {
            "total": sum(record_counts.values()),
            **record_counts,
        },
        "splits": {
            name: list(split_game_ids)
            for name, split_game_ids in splits.items()
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _record_counts_by_game(
    records: Sequence[_FixtureRecord],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.game_id] = counts.get(record.game_id, 0) + 1
    return counts
