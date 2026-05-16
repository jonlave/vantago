from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vantago.baselines import PHASE_NAMES, BaselineEvaluationRow, BaselineName
from vantago.cli.main import main
from vantago.comparison import (
    MISSED_TARGET_NEXT_STEPS,
    TARGET_BASELINES,
    PolicyModelComparisonConfig,
    compare_policy_models,
    summarize_policy_comparison_target,
)
from vantago.evaluation import PolicyMetricSummary

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True, slots=True)
class _FixtureRecord:
    game_id: str
    move_number: int
    y: int
    legal_labels: tuple[int, ...]


def test_compare_policy_models_combines_validation_rows_and_artifacts(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "runs" / "cnn.pt"
    history_path = tmp_path / "runs" / "history.json"
    mlp_history_path = checkpoint_path.with_suffix(".mlp-history.json")

    result = compare_policy_models(
        dataset_path,
        manifest_path,
        config=PolicyModelComparisonConfig(
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            epochs=1,
            batch_size=4,
            cnn_hidden_channels=1,
            mlp_hidden_size=8,
            learning_rate=0.01,
            weight_decay=0.0,
            seed=31,
            mask_topk=True,
        ),
    )

    assert checkpoint_path.is_file()
    assert history_path.is_file()
    assert mlp_history_path.is_file()
    assert result.mlp_history_path == mlp_history_path
    assert tuple(row.baseline for row in result.rows) == (
        "random_legal",
        "frequency_overall",
        "frequency_by_phase",
        "mlp_flattened",
        "cnn_policy",
    )
    assert len(result.phase_rows) == 5 * len(PHASE_NAMES)
    assert tuple(delta.baseline for delta in result.target.deltas) == TARGET_BASELINES
    assert isinstance(result.target.met, bool)
    assert result.cnn_training_result.history_path == history_path
    assert result.cnn_evaluation_result.checkpoint_path == checkpoint_path
    assert result.model_selection_notes == (
        "mlp_flattened: final_epoch",
        "cnn_policy: best_validation_cross_entropy_checkpoint",
    )

    history_json = json.loads(mlp_history_path.read_text(encoding="utf-8"))
    assert history_json["model"] == "mlp_flattened"
    assert history_json["selection"] == "final_epoch"
    assert len(history_json["history"]) == 1
    assert history_json["history"][0]["validation_metrics"]["example_count"] == 1


def test_compare_policy_models_cli_prints_combined_tables_and_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "cli" / "cnn.pt"
    history_path = tmp_path / "cli" / "history.json"
    mlp_history_path = tmp_path / "cli" / "mlp-history.json"

    exit_code = main(
        [
            "compare-policy-models",
            str(dataset_path),
            str(manifest_path),
            "--checkpoint-out",
            str(checkpoint_path),
            "--history-out",
            str(history_path),
            "--mlp-history-out",
            str(mlp_history_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--cnn-hidden-channels",
            "1",
            "--mlp-hidden-size",
            "8",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0.0",
            "--mask-topk",
        ]
    )

    assert exit_code == 0
    assert checkpoint_path.is_file()
    assert history_path.is_file()
    assert mlp_history_path.is_file()
    output = capsys.readouterr()
    assert output.err == ""

    lines = output.out.splitlines()
    assert "mlp_history" in lines
    assert "cnn_history" in lines
    assert "comparison" in lines
    assert "phase_comparison" in lines
    assert f"checkpoint: {checkpoint_path}" in lines
    assert f"history: {history_path}" in lines
    assert f"mlp_history: {mlp_history_path}" in lines
    assert "model_selection:" in lines
    assert "- mlp_flattened: final_epoch" in lines
    assert "- cnn_policy: best_validation_cross_entropy_checkpoint" in lines
    assert any(line in ("target: met", "target: missed") for line in lines)
    assert "target_deltas" in lines

    comparison_index = lines.index("comparison")
    phase_index = lines.index("phase_comparison")
    rows_by_name = {
        values[0]: values
        for values in (
            line.split()
            for line in lines[comparison_index + 2 : phase_index - 1]
        )
    }
    assert set(rows_by_name) == {
        "random_legal",
        "frequency_overall",
        "frequency_by_phase",
        "mlp_flattened",
        "cnn_policy",
    }


@pytest.mark.parametrize(
    ("cnn_top_1", "expected_met"),
    (
        (0.21, True),
        (0.20, False),
        (0.10, False),
    ),
)
def test_summarize_policy_comparison_target_requires_strict_top1_beat(
    cnn_top_1: float,
    expected_met: bool,
) -> None:
    target = summarize_policy_comparison_target(
        (
            _row("random_legal", top_1=0.20),
            _row("frequency_overall", top_1=0.18),
            _row("frequency_by_phase", top_1=0.19),
            _row("cnn_policy", top_1=cnn_top_1),
        )
    )

    assert target.met is expected_met
    assert target.next_steps == (() if expected_met else MISSED_TARGET_NEXT_STEPS)
    if expected_met:
        assert all(delta.delta > 0.0 for delta in target.deltas)
    else:
        assert any(delta.delta <= 0.0 for delta in target.deltas)


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


def _row(
    baseline: BaselineName,
    *,
    top_1: float,
) -> BaselineEvaluationRow:
    return BaselineEvaluationRow(
        baseline=baseline,
        split="validation",
        metrics=PolicyMetricSummary(
            example_count=10,
            top_1=top_1,
            top_3=top_1,
            top_5=top_1,
            cross_entropy=None,
            illegal_move_rate=0.0,
        ),
    )
