from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn

import vantago.qualitative as qualitative_module
from vantago.baselines import PHASE_NAMES
from vantago.cli.main import main
from vantago.evaluation import PolicyMetricSummary
from vantago.models import CnnPolicyNetwork
from vantago.qualitative import (
    QualitativeAnalysisConfig,
    generate_qualitative_analysis_report,
    write_qualitative_analysis_report_json,
)
from vantago.training import CnnPolicyCheckpoint, CnnTrainingConfig, train_cnn_policy

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True, slots=True)
class _FixtureRecord:
    game_id: str
    move_number: int
    y: int
    legal_labels: tuple[int, ...]


def test_generate_qualitative_analysis_report_groups_examples_and_heatmaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "runs" / "cnn.pt"
    _patch_checkpoint(monkeypatch, dataset_path, manifest_path, checkpoint_path)

    report = generate_qualitative_analysis_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
        config=QualitativeAnalysisConfig(
            split="test",
            batch_size=2,
            top_k=5,
            examples_per_phase=2,
            mask_topk=False,
        ),
    )

    assert report.split == "test"
    assert report.top_k == 5
    assert report.human_move_heatmap[0][4] == 1
    assert report.human_move_heatmap[0][5] == 1
    assert report.human_move_heatmap[0][6] == 1
    assert report.model_top1_heatmap[0][0] == 3
    assert tuple(report.phase_examples) == PHASE_NAMES

    opening = report.phase_examples["opening"][0]
    assert opening.game_id == "game-09.sgf"
    assert opening.source_name == "source/game-09.sgf"
    assert opening.move_number == 1
    assert opening.category == "occupied_or_illegal_point"
    assert opening.category_basis == "raw_top1"
    assert opening.human_label == 4
    assert opening.target_rank == 2
    assert opening.raw_top1_label == 0
    assert opening.raw_top1_is_legal is False
    assert opening.top_predictions[0].label == 0
    assert opening.top_predictions[0].is_legal is False
    assert opening.top_predictions[1].is_human_move is True
    assert any("H2" in line for line in opening.board)

    middle_game = report.phase_examples["middle_game"][0]
    assert middle_game.move_number == 41
    assert middle_game.target_rank == 3

    endgame = report.phase_examples["endgame"][0]
    assert endgame.move_number == 151
    assert endgame.target_rank == 4


def test_qualitative_analysis_mask_topk_changes_ranked_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "runs" / "cnn.pt"
    _patch_checkpoint(monkeypatch, dataset_path, manifest_path, checkpoint_path)

    unmasked = generate_qualitative_analysis_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
        config=QualitativeAnalysisConfig(mask_topk=False),
    )
    masked = generate_qualitative_analysis_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
        config=QualitativeAnalysisConfig(mask_topk=True),
    )

    assert unmasked.phase_examples["opening"][0].top_predictions[0].label == 0
    assert unmasked.model_top1_heatmap[0][0] == 3
    assert masked.phase_examples["opening"][0].top_predictions[0].label == 4
    assert masked.phase_examples["opening"][0].target_rank == 1
    assert masked.phase_examples["opening"][0].category_basis == "raw_top1"
    assert masked.phase_examples["opening"][0].raw_top1_label == 0
    assert masked.phase_examples["opening"][0].raw_top1_is_legal is False
    assert masked.model_top1_heatmap[0][0] == 0
    assert masked.model_top1_heatmap[0][4] == 3


def test_write_qualitative_analysis_json_records_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "runs" / "cnn.pt"
    json_path = tmp_path / "reports" / "qualitative.json"
    _patch_checkpoint(monkeypatch, dataset_path, manifest_path, checkpoint_path)
    report = generate_qualitative_analysis_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
        config=QualitativeAnalysisConfig(mask_topk=True),
    )

    write_qualitative_analysis_report_json(json_path, report)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["dataset_path"] == str(dataset_path)
    assert data["manifest_path"] == str(manifest_path)
    assert data["checkpoint_path"] == str(checkpoint_path)
    assert data["split"] == "test"
    assert data["top_k"] == 5
    assert data["mask_topk"] is True
    assert len(data["heatmaps"]["human_moves"]) == BOARD_SIZE
    assert data["heatmaps"]["human_moves"][0][4] == 1
    assert data["heatmaps"]["model_top1"][0][4] == 3
    opening = data["phase_examples"]["opening"][0]
    assert opening["human_move"] == {
        "label": 4,
        "row": 0,
        "col": 4,
        "target_rank": 1,
    }
    assert opening["raw_top1"]["label"] == 0
    assert opening["raw_top1"]["is_legal"] is False
    assert opening["category_basis"] == "raw_top1"
    assert opening["top_predictions"][0]["label"] == 4
    assert len(opening["board"]) == BOARD_SIZE + 1


def test_qualitative_analysis_cli_prints_sections_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
    checkpoint_path = tmp_path / "runs" / "cnn.pt"
    json_path = tmp_path / "reports" / "qualitative.json"
    _patch_checkpoint(monkeypatch, dataset_path, manifest_path, checkpoint_path)

    exit_code = main(
        [
            "qualitative-analysis",
            str(dataset_path),
            str(manifest_path),
            str(checkpoint_path),
            "--split",
            "test",
            "--batch-size",
            "2",
            "--top-k",
            "5",
            "--examples-per-phase",
            "1",
            "--mask-topk",
            "--json-out",
            str(json_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert output.err == ""
    lines = output.out.splitlines()
    assert "qualitative_analysis" in lines
    assert "human_move_heatmap_top" in lines
    assert "model_top1_heatmap_top" in lines
    assert "phase_examples" in lines
    assert "mask_topk: true" in lines
    assert "phase: opening" in lines
    assert "category: occupied_or_illegal_point" in lines
    assert "category_basis: raw_top1" in lines

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["batch_size"] == 2
    assert data["examples_per_phase"] == 1
    assert data["heatmaps"]["model_top1"][0][4] == 3


def test_qualitative_analysis_reserves_top5_near_miss_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _selection_records())
    checkpoint_path = tmp_path / "runs" / "cnn.pt"
    _patch_checkpoint(monkeypatch, dataset_path, manifest_path, checkpoint_path)

    report = generate_qualitative_analysis_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
        config=QualitativeAnalysisConfig(
            examples_per_phase=2,
            top_k=5,
            mask_topk=False,
        ),
    )

    opening_examples = report.phase_examples["opening"]
    assert len(opening_examples) == 2
    assert any(example.target_rank > 5 for example in opening_examples)
    assert any(2 <= example.target_rank <= 5 for example in opening_examples)


def test_qualitative_analysis_loads_real_trained_checkpoint(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _records())
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
            seed=31,
            mask_topk=True,
        ),
    )

    report = generate_qualitative_analysis_report(
        dataset_path,
        manifest_path,
        checkpoint_path,
        config=QualitativeAnalysisConfig(
            batch_size=2,
            top_k=3,
            examples_per_phase=1,
            mask_topk=True,
        ),
    )

    assert report.checkpoint_path == checkpoint_path
    assert sum(sum(row) for row in report.human_move_heatmap) == 3
    assert sum(sum(row) for row in report.model_top1_heatmap) == 3
    assert report.phase_examples["opening"]


def _patch_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    dataset_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
) -> None:
    logits = torch.full((1, POINT_COUNT), -20.0, dtype=torch.float32)
    for label, score in (
        (0, 10.0),
        (4, 9.0),
        (5, 8.0),
        (6, 7.0),
        (1, 6.0),
        (2, 5.0),
    ):
        logits[0, label] = score
    checkpoint = CnnPolicyCheckpoint(
        path=checkpoint_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        format_version=1,
        model_kind="cnn_policy",
        config=CnnTrainingConfig(checkpoint_path=checkpoint_path),
        model=cast(CnnPolicyNetwork, _FixedPolicy(logits)),
        history=(),
        best_epoch=1,
        best_validation_metrics=PolicyMetricSummary(
            example_count=1,
            top_1=0.0,
            top_3=0.0,
            top_5=0.0,
            cross_entropy=1.0,
            illegal_move_rate=0.0,
        ),
        started_at=None,
        finished_at=None,
        duration_seconds=None,
    )

    def fake_load_cnn_policy_checkpoint(path: Path) -> CnnPolicyCheckpoint:
        assert path == checkpoint_path
        return checkpoint

    monkeypatch.setattr(
        qualitative_module,
        "load_cnn_policy_checkpoint",
        fake_load_cnn_policy_checkpoint,
    )


class _FixedPolicy(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
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
            y=3,
            legal_labels=(1, 2, 3, 4, 5, 6),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=1,
            y=4,
            legal_labels=(1, 2, 3, 4, 5, 6),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=41,
            y=5,
            legal_labels=(1, 2, 3, 4, 5, 6),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=151,
            y=6,
            legal_labels=(1, 2, 3, 4, 5, 6),
        ),
    )


def _selection_records() -> tuple[_FixtureRecord, ...]:
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
            y=3,
            legal_labels=(1, 2, 3, 4, 5, 6),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=1,
            y=7,
            legal_labels=(1, 2, 3, 4, 5, 6, 7, 8),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=2,
            y=8,
            legal_labels=(1, 2, 3, 4, 5, 6, 7, 8),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=3,
            y=4,
            legal_labels=(1, 2, 3, 4, 5, 6, 7, 8),
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
        x[index, 2] = 1.0
        x[index, 0, 1, 1] = 1.0
        x[index, 2, 1, 1] = 0.0
        x[index, 1, 1, 2] = 1.0
        x[index, 2, 1, 2] = 0.0
        x[index, 4] = legal_mask[index].reshape(BOARD_SIZE, BOARD_SIZE)

    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.array([record.y for record in records], dtype=np.int64)
    game_id = np.array([record.game_id for record in records], dtype=np.str_)
    source_name = np.array(
        [f"source/{record.game_id}" for record in records],
        dtype=np.str_,
    )
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
        source_name=source_name,
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
