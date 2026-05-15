from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vantago.data import (
    DatasetSplitError,
    load_dataset_split_manifest,
    load_policy_dataloaders,
    load_policy_dataset,
    load_policy_datasets,
    load_policy_metadata_datasets,
    write_dataset_split_manifest,
)

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


def test_load_policy_dataset_filters_records_by_split(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    manifest = load_dataset_split_manifest(manifest_path)

    dataset = load_policy_dataset(dataset_path, manifest_path, "validation")

    assert len(dataset) == manifest.record_counts["validation"]
    validation_game_ids = set(manifest.splits["validation"])
    observed_game_ids = {dataset[index]["game_id"] for index in range(len(dataset))}
    assert observed_game_ids <= validation_game_ids


def test_load_policy_datasets_loads_multiple_splits_from_one_artifact(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    manifest = load_dataset_split_manifest(manifest_path)

    datasets = load_policy_datasets(
        dataset_path,
        manifest_path,
        splits=("train", "test"),
    )

    assert set(datasets) == {"train", "test"}
    assert len(datasets["train"]) == manifest.record_counts["train"]
    assert len(datasets["test"]) == manifest.record_counts["test"]


def test_load_policy_metadata_datasets_does_not_require_features(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "metadata-only.npz"
    manifest_path = tmp_path / "splits.json"
    _write_metadata_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 2 for index in range(10)},
    )
    _write_fixed_split_manifest(dataset_path, manifest_path, records_per_game=2)

    datasets = load_policy_metadata_datasets(
        dataset_path,
        manifest_path,
        splits=("validation",),
    )
    validation = datasets["validation"]
    batch = validation.metadata_batch(0, len(validation))

    assert set(datasets) == {"validation"}
    assert batch["y"].dtype == torch.int64
    assert batch["legal_mask"].shape[1:] == (POINT_COUNT,)
    assert batch["legal_mask"].dtype == torch.bool
    assert batch["move_number"].dtype == torch.int64


def test_policy_dataset_returns_typed_training_item(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    dataset = load_policy_dataset(dataset_path, manifest_path, "train")

    item = dataset[0]

    assert item["x"].shape == (CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)
    assert item["x"].dtype == torch.float32
    assert item["y"].shape == torch.Size([])
    assert item["y"].dtype == torch.int64
    assert item["legal_mask"].shape == (POINT_COUNT,)
    assert item["legal_mask"].dtype == torch.bool
    assert isinstance(item["game_id"], str)
    assert isinstance(item["move_number"], int)
    assert isinstance(item["source_name"], str)


def test_load_policy_dataloaders_iterates_stable_batches(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)

    loaders = load_policy_dataloaders(
        dataset_path,
        manifest_path,
        batch_size=2,
        shuffle_train=False,
    )

    assert set(loaders) == {"train", "validation", "test"}
    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        assert batch["x"].shape[1:] == (CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)
        assert batch["x"].dtype == torch.float32
        assert batch["y"].ndim == 1
        assert batch["y"].dtype == torch.int64
        assert batch["legal_mask"].shape[1:] == (POINT_COUNT,)
        assert batch["legal_mask"].dtype == torch.bool
        assert batch["move_number"].shape == batch["y"].shape
        assert batch["move_number"].dtype == torch.int64
        assert len(batch["game_id"]) == batch["y"].shape[0]
        assert len(batch["source_name"]) == batch["y"].shape[0]
        assert batch["y"].shape[0] > 0, split_name


def test_load_policy_dataloaders_can_select_splits(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)

    loaders = load_policy_dataloaders(
        dataset_path,
        manifest_path,
        batch_size=2,
        splits=("train", "validation"),
        shuffle_train=False,
    )

    assert set(loaders) == {"train", "validation"}


def test_load_policy_dataloaders_uses_explicit_train_generator(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    first_generator = torch.Generator()
    first_generator.manual_seed(123)
    second_generator = torch.Generator()
    second_generator.manual_seed(123)

    first_loaders = load_policy_dataloaders(
        dataset_path,
        manifest_path,
        batch_size=4,
        splits=("train",),
        train_generator=first_generator,
    )
    torch.manual_seed(999)
    second_loaders = load_policy_dataloaders(
        dataset_path,
        manifest_path,
        batch_size=4,
        splits=("train",),
        train_generator=second_generator,
    )

    first_batch = next(iter(first_loaders["train"]))
    second_batch = next(iter(second_loaders["train"]))
    torch.testing.assert_close(first_batch["y"], second_batch["y"])


def test_load_policy_dataloaders_rejects_empty_splits(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)

    with pytest.raises(DatasetSplitError, match="at least one split"):
        load_policy_dataloaders(dataset_path, manifest_path, batch_size=2, splits=())


def test_load_policy_dataset_rejects_invalid_split(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)

    with pytest.raises(DatasetSplitError, match="split must be one of"):
        load_policy_dataset(dataset_path, manifest_path, "holdout")


def test_load_policy_dataset_rejects_manifest_dataset_mismatch(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    manifest = load_dataset_split_manifest(manifest_path)
    missing_game_id = manifest.splits["train"][0]
    _write_processed_artifact(
        dataset_path,
        {
            f"game-{index:02d}.sgf": 2
            for index in range(10)
            if f"game-{index:02d}.sgf" != missing_game_id
        },
    )

    with pytest.raises(DatasetSplitError, match="not present in dataset"):
        load_policy_dataset(dataset_path, manifest_path, "train")


def test_load_policy_dataset_rejects_dataset_with_extra_games(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 2 for index in range(11)},
    )

    with pytest.raises(DatasetSplitError, match="not present in split manifest"):
        load_policy_dataset(dataset_path, manifest_path, "train")


def test_load_policy_dataset_rejects_wrong_manifest_dataset_path(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    _replace_manifest_dataset_path(manifest_path, tmp_path / "other.npz")

    with pytest.raises(DatasetSplitError, match="dataset path does not match"):
        load_policy_dataset(dataset_path, manifest_path, "train")


def test_load_policy_dataset_accepts_manifest_relative_dataset_path(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path = _write_dataset_and_manifest(tmp_path)
    _replace_manifest_dataset_path(manifest_path, Path("dataset.npz"))

    dataset = load_policy_dataset(dataset_path, manifest_path, "test")

    assert len(dataset) == load_dataset_split_manifest(manifest_path).record_counts[
        "test"
    ]


def test_load_policy_dataset_accepts_manifest_cwd_relative_dataset_path(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "data" / "processed" / "dataset.npz"
    manifest_path = tmp_path / "manifests" / "splits.json"
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 2 for index in range(10)},
    )
    write_dataset_split_manifest(dataset_path, manifest_path, seed=42)
    _replace_manifest_dataset_path(
        manifest_path,
        Path("data/processed/dataset.npz"),
    )

    dataset = load_policy_dataset(dataset_path, manifest_path, "train")

    assert len(dataset) == load_dataset_split_manifest(manifest_path).record_counts[
        "train"
    ]


def test_data_package_does_not_eagerly_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, vantago.data; print('torch' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def _write_dataset_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "splits.json"
    _write_processed_artifact(
        dataset_path,
        {f"game-{index:02d}.sgf": 2 for index in range(10)},
    )
    write_dataset_split_manifest(dataset_path, manifest_path, seed=42)
    return dataset_path, manifest_path


def _write_processed_artifact(
    path: Path,
    record_counts_by_game: dict[str, int],
) -> None:
    game_ids, move_numbers, source_names = _metadata_lists(record_counts_by_game)
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


def _write_metadata_artifact(
    path: Path,
    record_counts_by_game: dict[str, int],
) -> None:
    game_ids, move_numbers, _ = _metadata_lists(record_counts_by_game)
    record_count = len(game_ids)
    y = np.arange(record_count, dtype=np.int64) % POINT_COUNT
    legal_mask = np.zeros((record_count, POINT_COUNT), dtype=np.bool_)
    legal_mask[np.arange(record_count), y] = True

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        y=y,
        legal_mask=legal_mask,
        game_id=np.array(game_ids, dtype=np.str_),
        move_number=np.array(move_numbers, dtype=np.int64),
    )


def _metadata_lists(
    record_counts_by_game: dict[str, int],
) -> tuple[list[str], list[int], list[str]]:
    game_ids: list[str] = []
    move_numbers: list[int] = []
    source_names: list[str] = []
    for game_id, record_count in record_counts_by_game.items():
        for move_number in range(1, record_count + 1):
            game_ids.append(game_id)
            move_numbers.append(move_number)
            source_names.append(game_id)
    return game_ids, move_numbers, source_names


def _write_fixed_split_manifest(
    dataset_path: Path,
    manifest_path: Path,
    *,
    records_per_game: int,
) -> None:
    splits = {
        "train": tuple(f"game-{index:02d}.sgf" for index in range(8)),
        "validation": ("game-08.sgf",),
        "test": ("game-09.sgf",),
    }
    record_counts = {
        split: len(game_ids) * records_per_game
        for split, game_ids in splits.items()
    }
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "seed": 42,
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
                    split: list(game_ids)
                    for split, game_ids in splits.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _replace_manifest_dataset_path(
    manifest_path: Path,
    dataset_path: Path,
) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["dataset_path"] = str(dataset_path)
    manifest_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
