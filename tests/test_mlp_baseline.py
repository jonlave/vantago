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

from vantago.baselines import (
    FlattenedMlpPolicy,
    MlpBaselineConfig,
    MlpBaselineTrainingError,
    evaluate_mlp_policy,
    train_mlp_baseline,
)
from vantago.cli.main import main
from vantago.data.torch_loading import (
    PolicyBatch,
)
from vantago.data.torch_loading import (
    load_policy_dataloaders as original_load_policy_dataloaders,
)

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True, slots=True)
class _FixtureRecord:
    game_id: str
    move_number: int
    y: int
    legal_labels: tuple[int, ...]


def test_flattened_mlp_policy_maps_board_tensors_to_policy_logits() -> None:
    model = FlattenedMlpPolicy(hidden_size=8)
    logits = model(torch.zeros((2, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)))

    assert logits.shape == (2, POINT_COUNT)


def test_train_mlp_baseline_smoke_records_validation_metrics(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())

    result = train_mlp_baseline(
        dataset_path,
        manifest_path,
        config=MlpBaselineConfig(
            epochs=1,
            batch_size=4,
            hidden_size=8,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=7,
        ),
    )

    assert len(result.history) == 1
    assert result.history[0].epoch == 1
    assert result.history[0].train_loss > 0.0
    assert result.validation_row.baseline == "mlp_flattened"
    assert result.validation_row.split == "validation"
    assert result.validation_row.metrics.example_count == 1
    assert result.validation_row.metrics.cross_entropy is not None
    assert result.validation_row.metrics == result.history[-1].validation_metrics


def test_train_mlp_baseline_is_repeatable_with_same_seed(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    config = MlpBaselineConfig(
        epochs=2,
        batch_size=4,
        hidden_size=8,
        learning_rate=1e-2,
        weight_decay=0.0,
        seed=11,
    )

    first = train_mlp_baseline(dataset_path, manifest_path, config=config)
    second = train_mlp_baseline(dataset_path, manifest_path, config=config)

    assert [epoch.train_loss for epoch in second.history] == pytest.approx(
        [epoch.train_loss for epoch in first.history]
    )
    assert second.validation_row.metrics.top_1 == pytest.approx(
        first.validation_row.metrics.top_1
    )
    assert second.validation_row.metrics.cross_entropy == pytest.approx(
        first.validation_row.metrics.cross_entropy
    )


def test_train_mlp_baseline_loads_only_train_and_validation_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    observed_kwargs: dict[str, object] = {}

    def recording_load_policy_dataloaders(
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, DataLoader[PolicyBatch]]:
        observed_kwargs.update(kwargs)
        return original_load_policy_dataloaders(*args, **kwargs)

    monkeypatch.setattr(
        "vantago.baselines.mlp.load_policy_dataloaders",
        recording_load_policy_dataloaders,
    )

    train_mlp_baseline(
        dataset_path,
        manifest_path,
        config=MlpBaselineConfig(epochs=1, batch_size=4, hidden_size=8, seed=5),
    )

    assert observed_kwargs["splits"] == ("train", "validation")
    assert isinstance(observed_kwargs["train_generator"], torch.Generator)


def test_evaluate_mlp_policy_mask_topk_changes_topk_not_raw_illegal_rate() -> None:
    logits = torch.full((1, POINT_COUNT), -10.0, dtype=torch.float32)
    logits[0, 0] = 10.0
    logits[0, 1] = 9.0
    legal_mask = torch.zeros((1, POINT_COUNT), dtype=torch.bool)
    legal_mask[0, 1] = True
    batch: PolicyBatch = {
        "x": torch.zeros((1, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)),
        "y": torch.tensor([1], dtype=torch.int64),
        "legal_mask": legal_mask,
        "game_id": ["game-00.sgf"],
        "move_number": torch.tensor([1], dtype=torch.int64),
        "source_name": ["game-00.sgf"],
    }
    dataloader = cast(DataLoader[PolicyBatch], [batch])
    model = _FixedPolicy(logits)

    unmasked = evaluate_mlp_policy(model, dataloader, mask_topk=False)
    masked = evaluate_mlp_policy(model, dataloader, mask_topk=True)

    assert unmasked.top_1 == pytest.approx(0.0)
    assert masked.top_1 == pytest.approx(1.0)
    assert unmasked.illegal_move_rate == pytest.approx(1.0)
    assert masked.illegal_move_rate == pytest.approx(1.0)


def test_train_mlp_baseline_rejects_invalid_config(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())

    with pytest.raises(MlpBaselineTrainingError, match="epochs must be positive"):
        train_mlp_baseline(
            dataset_path,
            manifest_path,
            config=MlpBaselineConfig(epochs=0),
        )


def test_train_mlp_baseline_cli_prints_epoch_and_comparison_tables(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())

    exit_code = main(
        [
            "train-mlp-baseline",
            str(dataset_path),
            str(manifest_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-size",
            "8",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0.0",
        ]
    )

    assert exit_code == 0
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
    ]
    assert lines[1].split()[0] == "1"

    baseline_header_index = next(
        index
        for index, line in enumerate(lines)
        if line.split()
        == [
            "baseline",
            "examples",
            "top_1",
            "top_3",
            "top_5",
            "cross_entropy",
            "illegal_move_rate",
        ]
    )
    rows_by_name = {
        values[0]: values
        for values in (line.split() for line in lines[baseline_header_index + 1 :])
    }
    assert set(rows_by_name) == {
        "random_legal",
        "frequency_overall",
        "frequency_by_phase",
        "mlp_flattened",
    }
    assert rows_by_name["mlp_flattened"][5] != "n/a"


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
