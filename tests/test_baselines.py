from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from vantago.baselines import (
    BASELINE_NAMES,
    COMPARISON_BASELINE_NAMES,
    NON_NEURAL_BASELINE_NAMES,
    PHASE_NAMES,
    BaselineEvaluationError,
    BaselineEvaluationResult,
    BaselineEvaluationRow,
    BaselinePhaseEvaluationRow,
    evaluate_baselines,
    game_phase_for_move_number,
)
from vantago.baselines.evaluation import phase_mask_for_move_numbers
from vantago.cli.main import main

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True, slots=True)
class _FixtureRecord:
    game_id: str
    move_number: int
    y: int
    legal_labels: tuple[int, ...]


def test_game_phase_for_move_number_uses_roadmap_boundaries() -> None:
    assert game_phase_for_move_number(1) == "opening"
    assert game_phase_for_move_number(40) == "opening"
    assert game_phase_for_move_number(41) == "middle_game"
    assert game_phase_for_move_number(150) == "middle_game"
    assert game_phase_for_move_number(151) == "endgame"

    with pytest.raises(BaselineEvaluationError, match="positive"):
        game_phase_for_move_number(0)


def test_phase_mask_rejects_unknown_phase() -> None:
    move_numbers = torch.tensor([1, 41, 151], dtype=torch.int64)

    with pytest.raises(BaselineEvaluationError, match="phase must be one of"):
        phase_mask_for_move_numbers(move_numbers, "middle-earth")


def test_evaluate_baselines_rejects_non_positive_move_numbers(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(
            validation_y=5,
            validation_move_number=0,
            validation_legal_labels=(5,),
        ),
    )

    with pytest.raises(BaselineEvaluationError, match="move_number.*positive"):
        evaluate_baselines(dataset_path, manifest_path)


def test_baseline_name_contract_matches_non_neural_evaluator(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(validation_y=5, validation_legal_labels=(5,)),
    )

    result = evaluate_baselines(dataset_path, manifest_path)

    assert BASELINE_NAMES == NON_NEURAL_BASELINE_NAMES
    assert (
        *NON_NEURAL_BASELINE_NAMES,
        "mlp_flattened",
        "cnn_policy",
    ) == COMPARISON_BASELINE_NAMES
    assert tuple(row.baseline for row in result.rows) == BASELINE_NAMES


def test_evaluate_baselines_reports_metrics_for_each_phase(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(tmp_path, _phase_records())

    result = evaluate_baselines(dataset_path, manifest_path, seed=123)

    assert tuple(
        (row.baseline, row.phase)
        for row in result.phase_rows
    ) == tuple(
        (baseline, phase)
        for baseline in BASELINE_NAMES
        for phase in PHASE_NAMES
    )
    for baseline in BASELINE_NAMES:
        overall = _row(result, baseline).metrics
        phase_rows = [
            row
            for row in result.phase_rows
            if row.baseline == baseline
        ]
        phase_top_1_total = 0.0
        phase_example_count = 0
        for row in phase_rows:
            assert row.metrics is not None
            assert row.metrics.example_count == 1
            phase_top_1_total += row.metrics.top_1 * row.metrics.example_count
            phase_example_count += row.metrics.example_count

        assert phase_example_count == overall.example_count
        assert overall.top_1 == pytest.approx(
            phase_top_1_total / phase_example_count
        )
        assert overall.top_3 == pytest.approx(
            _weighted_phase_metric(phase_rows, "top_3")
        )
        assert overall.top_5 == pytest.approx(
            _weighted_phase_metric(phase_rows, "top_5")
        )
        assert overall.illegal_move_rate == pytest.approx(
            _weighted_phase_metric(phase_rows, "illegal_move_rate")
        )
        assert overall.cross_entropy is None
        for row in phase_rows:
            assert row.metrics is not None
            assert row.metrics.cross_entropy is None


def test_random_legal_baseline_ranks_only_legal_points(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(validation_y=5, validation_legal_labels=(5,)),
    )

    result = evaluate_baselines(dataset_path, manifest_path, seed=123)
    row = _row(result, "random_legal")

    assert row.metrics.example_count == 1
    assert row.metrics.top_1 == pytest.approx(1.0)
    assert row.metrics.illegal_move_rate == pytest.approx(0.0)
    assert row.metrics.cross_entropy is None


def test_random_legal_baseline_is_seeded_and_repeatable(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _multi_legal_records(),
    )

    first = _row(
        evaluate_baselines(dataset_path, manifest_path, seed=1),
        "random_legal",
    )
    repeat = _row(
        evaluate_baselines(dataset_path, manifest_path, seed=1),
        "random_legal",
    )
    different = _row(
        evaluate_baselines(dataset_path, manifest_path, seed=0),
        "random_legal",
    )

    assert repeat.metrics == first.metrics
    assert different.metrics.top_1 != pytest.approx(first.metrics.top_1)


def test_overall_frequency_baseline_uses_train_labels_only(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(validation_y=3, validation_legal_labels=(3, 10)),
    )

    result = evaluate_baselines(dataset_path, manifest_path)
    row = _row(result, "frequency_overall")

    assert row.metrics.example_count == 1
    assert row.metrics.top_1 == pytest.approx(0.0)


def test_frequency_baseline_mask_topk_changes_illegal_raw_top1(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(validation_y=3, validation_legal_labels=(3,)),
    )

    unmasked = _row(
        evaluate_baselines(dataset_path, manifest_path, mask_topk=False),
        "frequency_overall",
    )
    masked = _row(
        evaluate_baselines(dataset_path, manifest_path, mask_topk=True),
        "frequency_overall",
    )

    assert unmasked.metrics.top_1 == pytest.approx(0.0)
    assert unmasked.metrics.illegal_move_rate == pytest.approx(1.0)
    assert masked.metrics.top_1 == pytest.approx(1.0)
    assert masked.metrics.illegal_move_rate == pytest.approx(1.0)


def test_phase_frequency_baseline_falls_back_to_overall_counts(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(
            validation_y=10,
            validation_move_number=151,
            validation_legal_labels=(10,),
        ),
    )

    result = evaluate_baselines(dataset_path, manifest_path)
    row = _row(result, "frequency_by_phase")

    assert row.metrics.example_count == 1
    assert row.metrics.top_1 == pytest.approx(1.0)


def test_evaluate_baselines_does_not_load_feature_array(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(validation_y=10, validation_legal_labels=(10,)),
        include_features=False,
    )

    result = evaluate_baselines(dataset_path, manifest_path)

    assert _row(result, "random_legal").metrics.example_count == 1


def test_evaluate_baselines_cli_prints_comparable_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, manifest_path = _write_fixture(
        tmp_path,
        _records(validation_y=10, validation_legal_labels=(10,)),
    )

    exit_code = main(
        [
            "evaluate-baselines",
            str(dataset_path),
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "baseline",
        "examples",
        "top_1",
        "top_3",
        "top_5",
        "cross_entropy",
        "illegal_move_rate",
    ]
    rows_by_name = {
        values[0]: values
        for values in (line.split() for line in lines[1:4])
    }
    assert set(rows_by_name) == {
        "random_legal",
        "frequency_overall",
        "frequency_by_phase",
    }
    for values in rows_by_name.values():
        assert len(values) == 7
        assert values[1] == "1"
        assert values[5] == "n/a"

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
    assert len(phase_rows) == len(BASELINE_NAMES) * len(PHASE_NAMES)
    assert [values[1] for values in phase_rows[:3]] == list(PHASE_NAMES)
    empty_phase_values = [
        values
        for values in phase_rows
        if values[1] in {"middle_game", "endgame"}
    ]
    assert empty_phase_values
    for values in empty_phase_values:
        assert values[2:] == ["0", "n/a", "n/a", "n/a", "n/a", "n/a"]


def test_cli_main_does_not_eagerly_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, vantago.cli.main; print('torch' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def _records(
    *,
    validation_y: int,
    validation_legal_labels: tuple[int, ...],
    validation_move_number: int = 1,
) -> tuple[_FixtureRecord, ...]:
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
            move_number=validation_move_number,
            y=validation_y,
            legal_labels=validation_legal_labels,
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=1,
            y=4,
            legal_labels=(4,),
        ),
    )


def _multi_legal_records() -> tuple[_FixtureRecord, ...]:
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
            legal_labels=(5, 6),
        ),
        _FixtureRecord(
            game_id="game-08.sgf",
            move_number=2,
            y=5,
            legal_labels=(5, 6),
        ),
        _FixtureRecord(
            game_id="game-08.sgf",
            move_number=3,
            y=6,
            legal_labels=(5, 6),
        ),
        _FixtureRecord(
            game_id="game-09.sgf",
            move_number=1,
            y=4,
            legal_labels=(4,),
        ),
    )


def _phase_records() -> tuple[_FixtureRecord, ...]:
    train_records = (
        _FixtureRecord(
            game_id="game-00.sgf",
            move_number=40,
            y=5,
            legal_labels=(5,),
        ),
        _FixtureRecord(
            game_id="game-01.sgf",
            move_number=41,
            y=6,
            legal_labels=(6,),
        ),
        _FixtureRecord(
            game_id="game-02.sgf",
            move_number=151,
            y=7,
            legal_labels=(7,),
        ),
        *tuple(
            _FixtureRecord(
                game_id=f"game-{index:02d}.sgf",
                move_number=1,
                y=10 + index,
                legal_labels=(10 + index,),
            )
            for index in range(3, 8)
        ),
    )
    return (
        *train_records,
        _FixtureRecord(
            game_id="game-08.sgf",
            move_number=40,
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


def _row(
    result: BaselineEvaluationResult,
    baseline: str,
) -> BaselineEvaluationRow:
    for row in result.rows:
        if row.baseline == baseline:
            return row
    raise AssertionError(f"missing baseline row: {baseline}")


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
    *,
    include_features: bool = True,
) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    _write_processed_artifact(
        dataset_path,
        records,
        include_features=include_features,
    )
    _write_manifest(manifest_path, dataset_path, records)
    return dataset_path, manifest_path


def _write_processed_artifact(
    path: Path,
    records: Sequence[_FixtureRecord],
    *,
    include_features: bool = True,
) -> None:
    record_count = len(records)
    legal_mask = np.zeros((record_count, POINT_COUNT), dtype=np.bool_)
    for index, record in enumerate(records):
        legal_labels = {*record.legal_labels, record.y}
        legal_mask[index, sorted(legal_labels)] = True

    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.array([record.y for record in records], dtype=np.int64)
    game_id = np.array([record.game_id for record in records], dtype=np.str_)
    move_number = np.array(
        [record.move_number for record in records],
        dtype=np.int64,
    )
    if include_features:
        np.savez(
            path,
            x=np.zeros(
                (record_count, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE),
                dtype=np.float32,
            ),
            y=y,
            legal_mask=legal_mask,
            game_id=game_id,
            move_number=move_number,
            source_name=game_id,
        )
        return

    np.savez(
        path,
        y=y,
        legal_mask=legal_mask,
        game_id=game_id,
        move_number=move_number,
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
