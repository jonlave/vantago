from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vantago.cli.main import main
from vantago.data import (
    DatasetSplitError,
    load_dataset_split_manifest,
    write_dataset_split_manifest,
)

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


def test_write_dataset_split_manifest_splits_games_without_overlap(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    record_counts_by_game = {
        f"game-{index:02d}.sgf": index + 1
        for index in range(10)
    }
    _write_processed_artifact(dataset_path, record_counts_by_game)

    result = write_dataset_split_manifest(dataset_path, manifest_path, seed=42)

    loaded = load_dataset_split_manifest(manifest_path)
    assert loaded == result.manifest
    assert loaded.game_counts == {
        "total": 10,
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    assert loaded.record_counts["total"] == sum(record_counts_by_game.values())
    assert loaded.record_counts == {
        "total": sum(record_counts_by_game.values()),
        "train": _record_count(record_counts_by_game, loaded.splits["train"]),
        "validation": _record_count(
            record_counts_by_game,
            loaded.splits["validation"],
        ),
        "test": _record_count(record_counts_by_game, loaded.splits["test"]),
    }
    assert _all_split_game_ids(loaded.splits) == set(record_counts_by_game)
    assert not set(loaded.splits["train"]).intersection(loaded.splits["validation"])
    assert not set(loaded.splits["train"]).intersection(loaded.splits["test"])
    assert not set(loaded.splits["validation"]).intersection(loaded.splits["test"])


def test_write_dataset_split_manifest_is_byte_repeatable_with_same_seed(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 1 for index in range(10)},
    )

    write_dataset_split_manifest(dataset_path, first_manifest, seed=7)
    write_dataset_split_manifest(dataset_path, second_manifest, seed=7)

    assert first_manifest.read_bytes() == second_manifest.read_bytes()


def test_write_dataset_split_manifest_keeps_each_game_in_one_split(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    record_counts_by_game = {
        f"game-{index:02d}.sgf": index + 2
        for index in range(10)
    }
    _write_processed_artifact(dataset_path, record_counts_by_game)

    manifest = write_dataset_split_manifest(
        dataset_path,
        manifest_path,
        seed=11,
    ).manifest

    game_to_split = {
        game_id: split_name
        for split_name, split_game_ids in manifest.splits.items()
        for game_id in split_game_ids
    }
    assert set(game_to_split) == set(record_counts_by_game)
    assert manifest.record_counts["total"] == sum(record_counts_by_game.values())
    assert sum(
        manifest.record_counts[split_name]
        for split_name in ("train", "validation", "test")
    ) == manifest.record_counts["total"]


def test_write_dataset_split_manifest_uses_largest_remainder_allocation(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 1 for index in range(14)},
    )

    manifest = write_dataset_split_manifest(
        dataset_path,
        manifest_path,
        seed=3,
    ).manifest

    assert manifest.game_counts == {
        "total": 14,
        "train": 11,
        "validation": 2,
        "test": 1,
    }
    assert manifest.splits == {
        "train": (
            "game-00.sgf",
            "game-01.sgf",
            "game-02.sgf",
            "game-04.sgf",
            "game-05.sgf",
            "game-06.sgf",
            "game-07.sgf",
            "game-10.sgf",
            "game-11.sgf",
            "game-12.sgf",
            "game-13.sgf",
        ),
        "validation": ("game-08.sgf", "game-09.sgf"),
        "test": ("game-03.sgf",),
    }


def test_write_dataset_split_manifest_rejects_too_few_unique_games(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 1 for index in range(9)},
    )

    with pytest.raises(DatasetSplitError, match="at least 10 unique games"):
        write_dataset_split_manifest(dataset_path, manifest_path)

    assert not manifest_path.exists()


def test_load_dataset_split_manifest_rejects_too_few_games(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "splits.json"
    _write_manifest_json(
        manifest_path,
        splits={
            "train": tuple(f"game-{index:02d}.sgf" for index in range(7)),
            "validation": ("game-07.sgf",),
            "test": ("game-08.sgf",),
        },
    )

    with pytest.raises(DatasetSplitError, match="at least 10 unique games"):
        load_dataset_split_manifest(manifest_path)


def test_load_dataset_split_manifest_rejects_empty_validation_or_test_split(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "splits.json"
    _write_manifest_json(
        manifest_path,
        splits={
            "train": tuple(f"game-{index:02d}.sgf" for index in range(9)),
            "validation": ("game-09.sgf",),
            "test": (),
        },
    )

    with pytest.raises(DatasetSplitError, match="validation and test splits"):
        load_dataset_split_manifest(manifest_path)


def test_load_dataset_split_manifest_rejects_wrong_fixed_allocation(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "splits.json"
    _write_manifest_json(
        manifest_path,
        splits={
            "train": tuple(f"game-{index:02d}.sgf" for index in range(12)),
            "validation": ("game-12.sgf",),
            "test": ("game-13.sgf",),
        },
    )

    with pytest.raises(DatasetSplitError, match="80/10/10 allocation"):
        load_dataset_split_manifest(manifest_path)


def test_split_dataset_cli_prints_summary_and_writes_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 1 for index in range(10)},
    )

    exit_code = main(
        [
            "split-dataset",
            str(dataset_path),
            str(manifest_path),
            "--seed",
            "5",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f"dataset: {dataset_path}\n"
        f"manifest: {manifest_path}\n"
        "seed: 5\n"
        "games_total: 10\n"
        "records_total: 10\n"
        "game_counts:\n"
        "  train: 8\n"
        "  validation: 1\n"
        "  test: 1\n"
        "record_counts:\n"
        "  train: 8\n"
        "  validation: 1\n"
        "  test: 1\n"
        "overlap: none\n"
    )
    assert load_dataset_split_manifest(manifest_path).seed == 5


def test_split_dataset_cli_rejects_output_without_json_suffix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    output_path = tmp_path / "splits.txt"
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 1 for index in range(10)},
    )

    exit_code = main(["split-dataset", str(dataset_path), str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "error: dataset split manifest output path must end with .json: "
        f"{output_path}\n"
    )
    assert not output_path.exists()


def _write_processed_artifact(
    path: Path,
    record_counts_by_game: dict[str, int],
) -> None:
    game_ids: list[str] = []
    move_numbers: list[int] = []
    source_names: list[str] = []
    for game_id, record_count in record_counts_by_game.items():
        for move_number in range(1, record_count + 1):
            game_ids.append(game_id)
            move_numbers.append(move_number)
            source_names.append(game_id)

    record_count = len(game_ids)
    y = np.arange(record_count, dtype=np.int64) % POINT_COUNT
    legal_mask = np.zeros((record_count, POINT_COUNT), dtype=np.bool_)
    legal_mask[np.arange(record_count), y] = True

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=np.zeros(
            (record_count, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        ),
        y=y,
        legal_mask=legal_mask,
        game_id=np.array(game_ids, dtype=np.str_),
        move_number=np.array(move_numbers, dtype=np.int64),
        source_name=np.array(source_names, dtype=np.str_),
    )


def _write_manifest_json(
    path: Path,
    *,
    splits: dict[str, tuple[str, ...]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_path": "dataset.npz",
                "seed": 0,
                "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "game_counts": {
                    "total": sum(len(split) for split in splits.values()),
                    "train": len(splits["train"]),
                    "validation": len(splits["validation"]),
                    "test": len(splits["test"]),
                },
                "record_counts": {
                    "total": sum(len(split) for split in splits.values()),
                    "train": len(splits["train"]),
                    "validation": len(splits["validation"]),
                    "test": len(splits["test"]),
                },
                "splits": {
                    name: list(split)
                    for name, split in splits.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _record_count(
    record_counts_by_game: dict[str, int],
    game_ids: tuple[str, ...],
) -> int:
    return sum(record_counts_by_game[game_id] for game_id in game_ids)


def _all_split_game_ids(splits: dict[str, tuple[str, ...]]) -> set[str]:
    return {
        game_id
        for split_game_ids in splits.values()
        for game_id in split_game_ids
    }
