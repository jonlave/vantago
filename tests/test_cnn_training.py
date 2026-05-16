from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import vantago.training.cnn as cnn_training
from vantago.baselines import PHASE_NAMES, BaselinePhaseEvaluationRow
from vantago.cli.main import main
from vantago.data import load_policy_dataloaders
from vantago.data.torch_loading import PolicyBatch
from vantago.evaluation import PolicyMetricSummary
from vantago.training import (
    CnnTrainingConfig,
    CnnTrainingError,
    evaluate_cnn_policy_checkpoint,
    load_cnn_policy_checkpoint,
    train_cnn_policy,
)
from vantago.training.cnn import evaluate_cnn_policy

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True, slots=True)
class _FixtureRecord:
    game_id: str
    move_number: int
    y: int
    legal_labels: tuple[int, ...]


def test_train_cnn_policy_smoke_writes_checkpoint_and_history(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "runs" / "cnn.pt"

    result = train_cnn_policy(
        dataset_path,
        manifest_path,
        config=CnnTrainingConfig(
            checkpoint_path=checkpoint_path,
            epochs=1,
            batch_size=4,
            hidden_channels=1,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=7,
        ),
    )

    assert checkpoint_path.is_file()
    assert result.history_path == checkpoint_path.with_suffix(".history.json")
    assert result.history_path.is_file()
    assert len(result.history) == 1
    assert result.history[0].epoch == 1
    assert result.history[0].train_loss > 0.0
    assert result.history[0].validation_metrics.example_count == 1
    assert result.history[0].validation_metrics.cross_entropy is not None
    assert result.best_epoch == 1
    assert result.history[0].is_best
    assert result.best_validation_metrics == result.history[0].validation_metrics
    assert not result.model.training

    history_json = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history_json["dataset_path"] == str(dataset_path)
    assert history_json["manifest_path"] == str(manifest_path)
    assert history_json["checkpoint_path"] == str(checkpoint_path)
    assert history_json["best_epoch"] == 1
    assert history_json["config"]["hidden_channels"] == 1
    assert history_json["history"][0]["is_best"] is True


def test_load_cnn_policy_checkpoint_rebuilds_model_for_validation(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "cnn.pt"
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
            seed=11,
        ),
    )

    checkpoint = load_cnn_policy_checkpoint(checkpoint_path)
    loaders = load_policy_dataloaders(
        dataset_path,
        manifest_path,
        batch_size=2,
        splits=("validation",),
        shuffle_train=False,
    )
    metrics = evaluate_cnn_policy(checkpoint.model, loaders["validation"])

    assert checkpoint.path == checkpoint_path
    assert checkpoint.dataset_path == dataset_path
    assert checkpoint.manifest_path == manifest_path
    assert checkpoint.config.hidden_channels == 1
    assert checkpoint.best_epoch == 1
    assert checkpoint.history[0].is_best
    assert metrics.example_count == 1
    assert metrics.cross_entropy is not None


def test_evaluate_cnn_policy_checkpoint_reports_overall_and_phase_rows(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _phase_records())
    checkpoint_path = tmp_path / "cnn.pt"
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
        ),
    )

    result = evaluate_cnn_policy_checkpoint(
        dataset_path,
        manifest_path,
        checkpoint_path,
        split="validation",
        batch_size=2,
        mask_topk=True,
    )

    assert result.dataset_path == dataset_path
    assert result.manifest_path == manifest_path
    assert result.checkpoint_path == checkpoint_path
    assert result.checkpoint_dataset_path == dataset_path
    assert result.checkpoint_manifest_path == manifest_path
    assert result.split == "validation"
    assert result.batch_size == 2
    assert result.mask_topk is True
    assert len(result.rows) == 1
    assert result.rows[0].baseline == "cnn_policy"
    assert result.rows[0].split == "validation"
    assert result.rows[0].metrics.example_count == 3
    assert result.rows[0].metrics.cross_entropy is not None
    assert tuple(row.phase for row in result.phase_rows) == PHASE_NAMES
    assert all(row.baseline == "cnn_policy" for row in result.phase_rows)
    assert all(row.split == "validation" for row in result.phase_rows)
    assert all(row.metrics is not None for row in result.phase_rows)
    assert (
        sum(row.metrics.example_count for row in result.phase_rows if row.metrics)
        == 3
    )
    assert result.rows[0].metrics.top_1 == pytest.approx(
        _weighted_phase_metric(result.phase_rows, "top_1")
    )
    assert result.rows[0].metrics.top_3 == pytest.approx(
        _weighted_phase_metric(result.phase_rows, "top_3")
    )
    assert result.rows[0].metrics.top_5 == pytest.approx(
        _weighted_phase_metric(result.phase_rows, "top_5")
    )
    assert result.rows[0].metrics.illegal_move_rate == pytest.approx(
        _weighted_phase_metric(result.phase_rows, "illegal_move_rate")
    )
    assert result.rows[0].metrics.cross_entropy == pytest.approx(
        _weighted_phase_metric(result.phase_rows, "cross_entropy")
    )


def test_evaluate_cnn_policy_checkpoint_uses_requested_split(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "cnn.pt"
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
            seed=29,
        ),
    )

    result = evaluate_cnn_policy_checkpoint(
        dataset_path,
        manifest_path,
        checkpoint_path,
        split="test",
        batch_size=1,
    )

    assert result.split == "test"
    assert result.rows[0].split == "test"
    assert result.rows[0].metrics.example_count == 1
    assert result.phase_rows[0].metrics is not None
    assert result.phase_rows[0].metrics.example_count == 1
    assert result.phase_rows[1].metrics is None
    assert result.phase_rows[2].metrics is None


def test_evaluate_cnn_policy_checkpoint_rejects_mismatched_provenance(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "cnn.pt"
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
            seed=37,
        ),
    )

    with pytest.raises(CnnTrainingError, match="dataset_path does not match"):
        evaluate_cnn_policy_checkpoint(
            tmp_path / "other-dataset.npz",
            manifest_path,
            checkpoint_path,
        )

    with pytest.raises(CnnTrainingError, match="manifest_path does not match"):
        evaluate_cnn_policy_checkpoint(
            dataset_path,
            tmp_path / "other-splits.json",
            checkpoint_path,
        )


def test_evaluate_cnn_policy_mask_topk_changes_topk_not_raw_illegal_rate() -> None:
    logits = torch.full((1, POINT_COUNT), -10.0, dtype=torch.float32)
    logits[0, 0] = 10.0
    logits[0, 1] = 9.0
    legal_mask = torch.zeros((1, POINT_COUNT), dtype=torch.bool)
    legal_mask[0, 1] = True
    batch = {
        "x": torch.zeros((1, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)),
        "y": torch.tensor([1], dtype=torch.int64),
        "legal_mask": legal_mask,
        "game_id": ["game-00.sgf"],
        "move_number": torch.tensor([1], dtype=torch.int64),
        "source_name": ["game-00.sgf"],
    }
    dataloader = cast(DataLoader[PolicyBatch], [batch])
    model = _FixedPolicy(logits)

    unmasked = evaluate_cnn_policy(model, dataloader, mask_topk=False)
    masked = evaluate_cnn_policy(model, dataloader, mask_topk=True)

    assert unmasked.top_1 == pytest.approx(0.0)
    assert masked.top_1 == pytest.approx(1.0)
    assert unmasked.illegal_move_rate == pytest.approx(1.0)
    assert masked.illegal_move_rate == pytest.approx(1.0)


def test_train_cnn_policy_marks_lowest_validation_cross_entropy_as_best(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())

    result = train_cnn_policy(
        dataset_path,
        manifest_path,
        config=CnnTrainingConfig(
            checkpoint_path=tmp_path / "cnn.pt",
            epochs=2,
            batch_size=4,
            hidden_channels=1,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=13,
        ),
    )

    cross_entropies = [
        _require_cross_entropy(epoch.validation_metrics.cross_entropy)
        for epoch in result.history
    ]
    expected_best_epoch = min(
        range(1, len(cross_entropies) + 1),
        key=lambda epoch: (cross_entropies[epoch - 1], epoch),
    )

    assert result.best_epoch == expected_best_epoch
    assert [epoch.is_best for epoch in result.history].count(True) == 1
    assert result.history[result.best_epoch - 1].is_best

    checkpoint = load_cnn_policy_checkpoint(result.checkpoint_path)
    assert checkpoint.best_epoch == result.best_epoch
    assert checkpoint.best_validation_metrics == result.best_validation_metrics


def test_train_cnn_policy_rejects_checkpoint_history_path_collision(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    output_path = tmp_path / "cnn.json"

    with pytest.raises(CnnTrainingError, match="must be different"):
        train_cnn_policy(
            dataset_path,
            manifest_path,
            config=CnnTrainingConfig(
                checkpoint_path=output_path,
                history_path=output_path,
                epochs=1,
                batch_size=4,
                hidden_channels=1,
            ),
        )

    assert not output_path.exists()


def test_train_cnn_policy_persists_best_checkpoint_before_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "cnn.pt"
    evaluate_call_count = 0
    original_evaluate = cnn_training.evaluate_cnn_policy

    def failing_evaluate(
        *args: Any,
        **kwargs: Any,
    ) -> PolicyMetricSummary:
        nonlocal evaluate_call_count
        evaluate_call_count += 1
        if evaluate_call_count == 2:
            msg = "validation failed after first best checkpoint"
            raise CnnTrainingError(msg)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(cnn_training, "evaluate_cnn_policy", failing_evaluate)

    with pytest.raises(CnnTrainingError, match="validation failed"):
        train_cnn_policy(
            dataset_path,
            manifest_path,
            config=CnnTrainingConfig(
                checkpoint_path=checkpoint_path,
                epochs=2,
                batch_size=4,
                hidden_channels=1,
                learning_rate=1e-2,
                weight_decay=0.0,
                seed=19,
            ),
        )

    checkpoint = load_cnn_policy_checkpoint(checkpoint_path)
    assert checkpoint.best_epoch == 1
    assert len(checkpoint.history) == 1
    assert checkpoint.history[0].is_best


def test_train_cnn_policy_is_repeatable_with_same_seed(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())

    first = train_cnn_policy(
        dataset_path,
        manifest_path,
        config=CnnTrainingConfig(
            checkpoint_path=tmp_path / "first.pt",
            epochs=2,
            batch_size=4,
            hidden_channels=1,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=17,
        ),
    )
    second = train_cnn_policy(
        dataset_path,
        manifest_path,
        config=CnnTrainingConfig(
            checkpoint_path=tmp_path / "second.pt",
            epochs=2,
            batch_size=4,
            hidden_channels=1,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=17,
        ),
    )

    assert [epoch.train_loss for epoch in second.history] == pytest.approx(
        [epoch.train_loss for epoch in first.history]
    )
    assert [
        _require_cross_entropy(epoch.validation_metrics.cross_entropy)
        for epoch in second.history
    ] == pytest.approx(
        [
            _require_cross_entropy(epoch.validation_metrics.cross_entropy)
            for epoch in first.history
        ]
    )
    assert second.best_epoch == first.best_epoch


def test_train_cnn_policy_rejects_invalid_config(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())

    with pytest.raises(CnnTrainingError, match="epochs must be positive"):
        train_cnn_policy(
            dataset_path,
            manifest_path,
            config=CnnTrainingConfig(checkpoint_path=tmp_path / "cnn.pt", epochs=0),
        )


def test_train_cnn_policy_cli_prints_table_and_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "cli" / "cnn.pt"
    history_path = tmp_path / "cli" / "history.json"

    exit_code = main(
        [
            "train-cnn-policy",
            str(dataset_path),
            str(manifest_path),
            "--checkpoint-out",
            str(checkpoint_path),
            "--history-out",
            str(history_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-channels",
            "1",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0.0",
        ]
    )

    assert exit_code == 0
    assert checkpoint_path.is_file()
    assert history_path.is_file()
    output = capsys.readouterr()
    assert output.err == ""

    lines = output.out.splitlines()
    assert lines[0].split() == [
        "epoch",
        "train_loss",
        "validation_top_1",
        "validation_top_3",
        "validation_top_5",
        "validation_cross_entropy",
        "validation_illegal_move_rate",
        "best",
    ]
    assert lines[1].split()[0] == "1"
    assert lines[1].split()[-1] == "*"
    assert lines[2] == f"checkpoint: {checkpoint_path}"
    assert lines[3] == f"history: {history_path}"


def test_evaluate_cnn_policy_cli_prints_overall_and_phase_tables(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "cli" / "cnn.pt"
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
            seed=31,
        ),
    )

    exit_code = main(
        [
            "evaluate-cnn-policy",
            str(dataset_path),
            str(manifest_path),
            str(checkpoint_path),
            "--split",
            "test",
            "--batch-size",
            "1",
            "--mask-topk",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert output.err == ""

    lines = output.out.splitlines()
    assert lines[0].split() == [
        "baseline",
        "examples",
        "top_1",
        "top_3",
        "top_5",
        "cross_entropy",
        "illegal_move_rate",
    ]
    assert lines[1].split()[0] == "cnn_policy"
    assert lines[1].split()[1] == "1"

    phase_header_index = next(
        index
        for index, line in enumerate(lines)
        if line.split()
        == [
            "baseline",
            "phase",
            "examples",
            "top_1",
            "top_3",
            "top_5",
            "cross_entropy",
            "illegal_move_rate",
        ]
    )
    phase_rows = [
        line.split()
        for line in lines[phase_header_index + 1 :]
    ]
    assert len(phase_rows) == len(PHASE_NAMES)
    assert [values[1] for values in phase_rows] == list(PHASE_NAMES)
    assert phase_rows[0][0] == "cnn_policy"
    assert phase_rows[0][2] == "1"
    assert phase_rows[1][2:] == ["0", "n/a", "n/a", "n/a", "n/a", "n/a"]
    assert phase_rows[2][2:] == ["0", "n/a", "n/a", "n/a", "n/a", "n/a"]


def _require_cross_entropy(value: float | None) -> float:
    assert value is not None
    return value


class _FixedPolicy(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.device_anchor = nn.Parameter(torch.empty(0))
        self.register_buffer("logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = cast(torch.Tensor, self.logits)
        return logits.expand(int(x.shape[0]), -1)


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


def _phase_records() -> tuple[_FixtureRecord, ...]:
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
            game_id="game-08.sgf",
            move_number=41,
            y=6,
            legal_labels=(6,),
        ),
        _FixtureRecord(
            game_id="game-08.sgf",
            move_number=151,
            y=7,
            legal_labels=(7,),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=1,
            y=4,
            legal_labels=(4,),
        ),
    )


def _weighted_phase_metric(
    rows: Sequence[BaselinePhaseEvaluationRow],
    metric_name: str,
) -> float:
    total = 0.0
    example_count = 0
    for row in rows:
        assert row.metrics is not None
        value = getattr(row.metrics, metric_name)
        assert isinstance(value, float)
        total += value * row.metrics.example_count
        example_count += row.metrics.example_count
    return total / example_count


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
