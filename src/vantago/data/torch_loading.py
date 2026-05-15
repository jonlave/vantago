"""PyTorch datasets and loaders for processed policy artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from vantago.data.artifacts import (
    ProcessedDatasetArtifact,
    ProcessedDatasetMetadataArtifact,
    load_processed_dataset,
    load_processed_dataset_metadata,
)
from vantago.data.splits import (
    SPLIT_NAMES,
    DatasetSplitError,
    DatasetSplitManifest,
    load_dataset_split_manifest,
)

Int64Array = npt.NDArray[np.int64]
_PolicyArtifact = ProcessedDatasetArtifact | ProcessedDatasetMetadataArtifact


class PolicyDatasetItem(TypedDict):
    """One supervised policy example returned by ``ProcessedPolicyDataset``."""

    x: torch.Tensor
    y: torch.Tensor
    legal_mask: torch.Tensor
    game_id: str
    move_number: int
    source_name: str


class PolicyBatch(TypedDict):
    """One collated supervised policy batch returned by DataLoaders."""

    x: torch.Tensor
    y: torch.Tensor
    legal_mask: torch.Tensor
    game_id: list[str]
    move_number: torch.Tensor
    source_name: list[str]


class PolicyMetadataBatch(TypedDict):
    """Feature-free supervised policy metadata for a contiguous dataset slice."""

    y: torch.Tensor
    legal_mask: torch.Tensor
    move_number: torch.Tensor


class ProcessedPolicyDataset(Dataset[PolicyDatasetItem]):
    """PyTorch Dataset view over selected rows in a processed artifact."""

    def __init__(
        self,
        artifact: ProcessedDatasetArtifact,
        indices: Int64Array,
        *,
        split: str,
    ) -> None:
        if indices.ndim != 1:
            msg = f"indices must be one-dimensional, got {indices.shape}"
            raise DatasetSplitError(msg)
        self._artifact = artifact
        self._indices = indices
        self.split = split

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def metadata_batch(self, start: int, stop: int) -> PolicyMetadataBatch:
        """Return labels, legal masks, and move numbers without loading features."""

        return _metadata_batch_from_artifact(
            self._artifact,
            self._indices,
            start,
            stop,
        )

    def __getitem__(self, index: int) -> PolicyDatasetItem:
        if not 0 <= index < len(self):
            msg = f"dataset index must be in [0, {len(self)}), got {index}"
            raise IndexError(msg)

        row_index = int(self._indices[index])
        return {
            "x": torch.from_numpy(self._artifact.x[row_index]),
            "y": torch.tensor(int(self._artifact.y[row_index]), dtype=torch.int64),
            "legal_mask": torch.from_numpy(self._artifact.legal_mask[row_index]),
            "game_id": str(self._artifact.game_id[row_index]),
            "move_number": int(self._artifact.move_number[row_index]),
            "source_name": str(self._artifact.source_name[row_index]),
        }


class ProcessedPolicyMetadataDataset:
    """Feature-free Dataset-like view over selected processed artifact rows."""

    def __init__(
        self,
        artifact: ProcessedDatasetMetadataArtifact,
        indices: Int64Array,
        *,
        split: str,
    ) -> None:
        if indices.ndim != 1:
            msg = f"indices must be one-dimensional, got {indices.shape}"
            raise DatasetSplitError(msg)
        self._artifact = artifact
        self._indices = indices
        self.split = split

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def metadata_batch(self, start: int, stop: int) -> PolicyMetadataBatch:
        """Return labels, legal masks, and move numbers for a row slice."""

        return _metadata_batch_from_artifact(
            self._artifact,
            self._indices,
            start,
            stop,
        )


def load_policy_dataset(
    dataset_path: Path,
    manifest_path: Path,
    split: str,
) -> ProcessedPolicyDataset:
    """Load one train/validation/test split as a PyTorch Dataset."""

    artifact, manifest = _load_dataset_and_manifest(dataset_path, manifest_path)
    return _build_policy_dataset(artifact, manifest, split)


def load_policy_datasets(
    dataset_path: Path,
    manifest_path: Path,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, ProcessedPolicyDataset]:
    """Load multiple split datasets while reading the artifact only once."""

    if not splits:
        msg = "at least one split is required"
        raise DatasetSplitError(msg)

    artifact, manifest = _load_dataset_and_manifest(dataset_path, manifest_path)
    indices_by_split = _indices_for_splits(artifact, manifest, splits)
    return {
        split: ProcessedPolicyDataset(
            artifact,
            indices_by_split[split],
            split=split,
        )
        for split in indices_by_split
    }


def load_policy_metadata_datasets(
    dataset_path: Path,
    manifest_path: Path,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, ProcessedPolicyMetadataDataset]:
    """Load feature-free split datasets while reading only metadata arrays."""

    if not splits:
        msg = "at least one split is required"
        raise DatasetSplitError(msg)

    artifact = load_processed_dataset_metadata(dataset_path)
    manifest = load_dataset_split_manifest(manifest_path)
    _validate_manifest_dataset_path(dataset_path, manifest_path, manifest)
    _validate_manifest_game_coverage(artifact, manifest)
    indices_by_split = _indices_for_splits(artifact, manifest, splits)
    return {
        split: ProcessedPolicyMetadataDataset(
            artifact,
            indices_by_split[split],
            split=split,
        )
        for split in indices_by_split
    }


def load_policy_dataloaders(
    dataset_path: Path,
    manifest_path: Path,
    batch_size: int,
    *,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> dict[str, DataLoader[PolicyBatch]]:
    """Load train, validation, and test DataLoaders from a split manifest."""

    if batch_size < 1:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if num_workers < 0:
        msg = f"num_workers must be non-negative, got {num_workers}"
        raise ValueError(msg)

    artifact, manifest = _load_dataset_and_manifest(dataset_path, manifest_path)
    return {
        "train": _build_policy_dataloader(
            _build_policy_dataset(artifact, manifest, "train"),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        ),
        "validation": _build_policy_dataloader(
            _build_policy_dataset(artifact, manifest, "validation"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        ),
        "test": _build_policy_dataloader(
            _build_policy_dataset(artifact, manifest, "test"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        ),
    }


def _load_dataset_and_manifest(
    dataset_path: Path,
    manifest_path: Path,
) -> tuple[ProcessedDatasetArtifact, DatasetSplitManifest]:
    artifact = load_processed_dataset(dataset_path)
    manifest = load_dataset_split_manifest(manifest_path)
    _validate_manifest_dataset_path(dataset_path, manifest_path, manifest)
    _validate_manifest_game_coverage(artifact, manifest)
    return artifact, manifest


def _validate_manifest_dataset_path(
    dataset_path: Path,
    manifest_path: Path,
    manifest: DatasetSplitManifest,
) -> None:
    resolved_dataset_path = dataset_path.resolve()
    manifest_dataset_path = manifest.dataset_path
    candidate_paths = (
        (manifest_dataset_path.resolve(),)
        if manifest_dataset_path.is_absolute()
        else (
            manifest_dataset_path.resolve(),
            (manifest_path.parent / manifest_dataset_path).resolve(),
        )
    )
    if (
        resolved_dataset_path not in candidate_paths
        and not _path_has_manifest_relative_suffix(
            resolved_dataset_path,
            manifest_dataset_path,
        )
    ):
        candidates = ", ".join(str(path) for path in candidate_paths)
        msg = (
            "dataset path does not match split manifest dataset_path: "
            f"{resolved_dataset_path} not in [{candidates}]"
        )
        raise DatasetSplitError(msg)


def _path_has_manifest_relative_suffix(
    dataset_path: Path,
    manifest_dataset_path: Path,
) -> bool:
    if manifest_dataset_path.is_absolute() or len(manifest_dataset_path.parts) <= 1:
        return False
    suffix_length = len(manifest_dataset_path.parts)
    return dataset_path.parts[-suffix_length:] == manifest_dataset_path.parts


def _validate_manifest_game_coverage(
    artifact: _PolicyArtifact,
    manifest: DatasetSplitManifest,
) -> None:
    manifest_game_ids = {
        game_id
        for split_name in SPLIT_NAMES
        for game_id in manifest.splits[split_name]
    }
    artifact_game_ids = {str(game_id) for game_id in artifact.game_id}
    missing_game_ids = sorted(manifest_game_ids - artifact_game_ids)
    extra_game_ids = sorted(artifact_game_ids - manifest_game_ids)
    if not missing_game_ids and not extra_game_ids:
        return

    details: list[str] = []
    if missing_game_ids:
        details.append(
            "manifest references games not present in dataset: "
            + ", ".join(missing_game_ids)
        )
    if extra_game_ids:
        details.append(
            "dataset contains games not present in split manifest: "
            + ", ".join(extra_game_ids)
        )
    msg = "dataset game IDs do not match split manifest: " + "; ".join(details)
    raise DatasetSplitError(msg)


def _build_policy_dataset(
    artifact: ProcessedDatasetArtifact,
    manifest: DatasetSplitManifest,
    split: str,
) -> ProcessedPolicyDataset:
    split_name = _validate_split_name(split)
    indices = _indices_for_split(artifact, manifest, split_name)
    return ProcessedPolicyDataset(artifact, indices, split=split_name)


def _build_policy_metadata_dataset(
    artifact: ProcessedDatasetMetadataArtifact,
    manifest: DatasetSplitManifest,
    split: str,
) -> ProcessedPolicyMetadataDataset:
    split_name = _validate_split_name(split)
    indices = _indices_for_split(artifact, manifest, split_name)
    return ProcessedPolicyMetadataDataset(artifact, indices, split=split_name)


def _validate_split_name(split: str) -> str:
    if split not in SPLIT_NAMES:
        msg = f"split must be one of {', '.join(SPLIT_NAMES)}, got {split}"
        raise DatasetSplitError(msg)
    return split


def _indices_for_split(
    artifact: _PolicyArtifact,
    manifest: DatasetSplitManifest,
    split: str,
) -> Int64Array:
    split_game_ids = set(manifest.splits[split])
    artifact_game_ids = {str(game_id) for game_id in artifact.game_id}
    missing_game_ids = sorted(split_game_ids - artifact_game_ids)
    if missing_game_ids:
        missing = ", ".join(missing_game_ids)
        msg = f"{split} split references games not present in dataset: {missing}"
        raise DatasetSplitError(msg)

    selected = np.fromiter(
        (str(game_id) in split_game_ids for game_id in artifact.game_id),
        dtype=np.bool_,
        count=int(artifact.game_id.shape[0]),
    )
    indices = cast(Int64Array, np.flatnonzero(selected).astype(np.int64, copy=False))
    expected_count = manifest.record_counts[split]
    actual_count = int(indices.shape[0])
    if actual_count != expected_count:
        msg = (
            f"{split} split expected {expected_count} records from manifest, "
            f"found {actual_count} in dataset"
        )
        raise DatasetSplitError(msg)
    return indices


def _indices_for_splits(
    artifact: _PolicyArtifact,
    manifest: DatasetSplitManifest,
    splits: Sequence[str],
) -> dict[str, Int64Array]:
    split_names = tuple(dict.fromkeys(_validate_split_name(split) for split in splits))
    split_for_game = {
        game_id: split_name
        for split_name in SPLIT_NAMES
        for game_id in manifest.splits[split_name]
    }
    selected: dict[str, list[int]] = {split_name: [] for split_name in split_names}
    for row_index, game_id in enumerate(artifact.game_id):
        split_name = split_for_game[str(game_id)]
        if split_name in selected:
            selected[split_name].append(row_index)

    indices_by_split = {
        split_name: np.array(row_indices, dtype=np.int64)
        for split_name, row_indices in selected.items()
    }
    for split_name, indices in indices_by_split.items():
        expected_count = manifest.record_counts[split_name]
        actual_count = int(indices.shape[0])
        if actual_count != expected_count:
            msg = (
                f"{split_name} split expected {expected_count} records from "
                f"manifest, found {actual_count} in dataset"
            )
            raise DatasetSplitError(msg)
    return indices_by_split


def _metadata_batch_from_artifact(
    artifact: _PolicyArtifact,
    indices: Int64Array,
    start: int,
    stop: int,
) -> PolicyMetadataBatch:
    _validate_batch_slice(start, stop, int(indices.shape[0]))
    row_indices = indices[start:stop]
    return {
        "y": torch.from_numpy(artifact.y[row_indices]),
        "legal_mask": torch.from_numpy(artifact.legal_mask[row_indices]),
        "move_number": torch.from_numpy(artifact.move_number[row_indices]),
    }


def _validate_batch_slice(start: int, stop: int, length: int) -> None:
    if not 0 <= start <= stop <= length:
        msg = f"batch slice must satisfy 0 <= start <= stop <= {length}"
        raise IndexError(msg)


def _build_policy_dataloader(
    dataset: ProcessedPolicyDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader[PolicyBatch]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate_policy_batch,
    )
    return cast(DataLoader[PolicyBatch], dataloader)


def _collate_policy_batch(items: list[PolicyDatasetItem]) -> PolicyBatch:
    return {
        "x": torch.stack([item["x"] for item in items]),
        "y": torch.stack([item["y"] for item in items]),
        "legal_mask": torch.stack([item["legal_mask"] for item in items]),
        "game_id": [item["game_id"] for item in items],
        "move_number": torch.tensor(
            [item["move_number"] for item in items],
            dtype=torch.int64,
        ),
        "source_name": [item["source_name"] for item in items],
    }
