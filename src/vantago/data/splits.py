"""Create and validate deterministic game-level dataset split manifests."""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vantago.data.artifacts import ProcessedDatasetError, load_processed_dataset

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}
MINIMUM_SPLIT_GAME_COUNT = 10


class DatasetSplitError(ValueError):
    """Raised when a dataset split manifest cannot be built or loaded."""


@dataclass(frozen=True, slots=True)
class DatasetSplitManifest:
    """Train/validation/test game IDs and counts for a processed dataset."""

    dataset_path: Path
    seed: int
    ratios: dict[str, float]
    game_counts: dict[str, int]
    record_counts: dict[str, int]
    splits: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class DatasetSplitBuildResult:
    """Structured result for writing a dataset split manifest."""

    dataset_path: Path
    output_path: Path
    manifest: DatasetSplitManifest


def write_dataset_split_manifest(
    dataset_path: Path,
    output_path: Path,
    *,
    seed: int = 0,
) -> DatasetSplitBuildResult:
    """Write a deterministic game-level split manifest for a processed dataset."""

    _validate_output_path(output_path)
    try:
        artifact = load_processed_dataset(dataset_path)
    except ProcessedDatasetError as exc:
        raise DatasetSplitError(str(exc)) from exc

    record_counts_by_game = _record_counts_by_game(
        str(game_id) for game_id in artifact.game_id
    )
    game_ids = sorted(record_counts_by_game)
    if len(game_ids) < MINIMUM_SPLIT_GAME_COUNT:
        msg = (
            "dataset split requires at least "
            f"{MINIMUM_SPLIT_GAME_COUNT} unique games, got {len(game_ids)}"
        )
        raise DatasetSplitError(msg)

    split_counts = _allocate_split_counts(len(game_ids))
    shuffled_game_ids = game_ids.copy()
    random.Random(seed).shuffle(shuffled_game_ids)
    splits = _assign_splits(shuffled_game_ids, split_counts)
    _validate_split_coverage(game_ids, splits)

    record_counts = _record_counts_for_splits(record_counts_by_game, splits)
    manifest = DatasetSplitManifest(
        dataset_path=dataset_path,
        seed=seed,
        ratios=dict(SPLIT_RATIOS),
        game_counts={
            "total": len(game_ids),
            **{name: len(splits[name]) for name in SPLIT_NAMES},
        },
        record_counts={
            "total": int(artifact.y.shape[0]),
            **record_counts,
        },
        splits=splits,
    )
    _validate_manifest(manifest)
    _write_manifest_atomic(output_path, manifest)
    return DatasetSplitBuildResult(
        dataset_path=dataset_path,
        output_path=output_path,
        manifest=manifest,
    )


def load_dataset_split_manifest(path: Path) -> DatasetSplitManifest:
    """Load and validate a dataset split manifest JSON file."""

    try:
        data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except OSError as exc:
        msg = f"unable to load dataset split manifest {path}: {exc}"
        raise DatasetSplitError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"invalid dataset split manifest JSON {path}: {exc}"
        raise DatasetSplitError(msg) from exc

    manifest = _manifest_from_json_data(data, path)
    _validate_manifest(manifest)
    return manifest


def _record_counts_by_game(game_ids: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for game_id in game_ids:
        resolved_game_id = str(game_id)
        counts[resolved_game_id] = counts.get(resolved_game_id, 0) + 1
    return counts


def _allocate_split_counts(total_games: int) -> dict[str, int]:
    floors = {
        name: int(total_games * SPLIT_RATIOS[name])
        for name in SPLIT_NAMES
    }
    remainders = sorted(
        (
            (
                total_games * SPLIT_RATIOS[name] - floors[name],
                split_index,
                name,
            )
            for split_index, name in enumerate(SPLIT_NAMES)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    remaining_games = total_games - sum(floors.values())
    counts = floors.copy()
    for _, _, name in remainders[:remaining_games]:
        counts[name] += 1
    return counts


def _assign_splits(
    shuffled_game_ids: list[str],
    split_counts: dict[str, int],
) -> dict[str, tuple[str, ...]]:
    cursor = 0
    splits: dict[str, tuple[str, ...]] = {}
    for name in SPLIT_NAMES:
        next_cursor = cursor + split_counts[name]
        splits[name] = tuple(sorted(shuffled_game_ids[cursor:next_cursor]))
        cursor = next_cursor
    return splits


def _record_counts_for_splits(
    record_counts_by_game: dict[str, int],
    splits: dict[str, tuple[str, ...]],
) -> dict[str, int]:
    return {
        name: sum(record_counts_by_game[game_id] for game_id in splits[name])
        for name in SPLIT_NAMES
    }


def _validate_split_coverage(
    game_ids: list[str],
    splits: dict[str, tuple[str, ...]],
) -> None:
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        split_games = splits.get(name)
        if split_games is None:
            msg = f"dataset split manifest missing split: {name}"
            raise DatasetSplitError(msg)
        duplicate_games_in_split = {
            game_id
            for game_id in split_games
            if split_games.count(game_id) > 1
        }
        if duplicate_games_in_split:
            duplicate_list = ", ".join(sorted(duplicate_games_in_split))
            msg = f"dataset split manifest has duplicate games: {duplicate_list}"
            raise DatasetSplitError(msg)
        duplicate_games = seen.intersection(split_games)
        if duplicate_games:
            duplicate_list = ", ".join(sorted(duplicate_games))
            msg = f"dataset split manifest has overlapping games: {duplicate_list}"
            raise DatasetSplitError(msg)
        seen.update(split_games)

    expected = set(game_ids)
    if seen != expected:
        missing = ", ".join(sorted(expected - seen))
        extra = ", ".join(sorted(seen - expected))
        details = []
        if missing:
            details.append(f"missing: {missing}")
        if extra:
            details.append(f"extra: {extra}")
        msg = "dataset split manifest does not cover every game"
        if details:
            msg = f"{msg}: {'; '.join(details)}"
        raise DatasetSplitError(msg)


def _validate_manifest(manifest: DatasetSplitManifest) -> None:
    if manifest.ratios != SPLIT_RATIOS:
        msg = "dataset split manifest ratios must be train=0.8 validation=0.1 test=0.1"
        raise DatasetSplitError(msg)

    split_game_total = sum(len(manifest.splits[name]) for name in SPLIT_NAMES)
    if split_game_total < MINIMUM_SPLIT_GAME_COUNT:
        msg = (
            "dataset split manifest requires at least "
            f"{MINIMUM_SPLIT_GAME_COUNT} unique games, got {split_game_total}"
        )
        raise DatasetSplitError(msg)
    if not manifest.splits["validation"] or not manifest.splits["test"]:
        msg = "dataset split manifest validation and test splits must be non-empty"
        raise DatasetSplitError(msg)

    _validate_split_coverage(
        [
            game_id
            for name in SPLIT_NAMES
            for game_id in manifest.splits[name]
        ],
        manifest.splits,
    )
    expected_split_counts = _allocate_split_counts(split_game_total)
    actual_split_counts = {
        name: len(manifest.splits[name])
        for name in SPLIT_NAMES
    }
    if actual_split_counts != expected_split_counts:
        msg = (
            "dataset split manifest game counts do not match fixed "
            "80/10/10 allocation"
        )
        raise DatasetSplitError(msg)

    expected_game_counts = {
        "total": split_game_total,
        **actual_split_counts,
    }
    if manifest.game_counts != expected_game_counts:
        msg = "dataset split manifest game_counts do not match splits"
        raise DatasetSplitError(msg)

    expected_record_total = sum(manifest.record_counts[name] for name in SPLIT_NAMES)
    if manifest.record_counts.get("total") != expected_record_total:
        msg = "dataset split manifest record_counts do not sum to total"
        raise DatasetSplitError(msg)
    if any(manifest.record_counts[name] < 0 for name in ("total", *SPLIT_NAMES)):
        msg = "dataset split manifest record_counts must be non-negative"
        raise DatasetSplitError(msg)


def _validate_output_path(output_path: Path) -> None:
    if output_path.suffix != ".json":
        msg = f"dataset split manifest output path must end with .json: {output_path}"
        raise DatasetSplitError(msg)
    if output_path.exists() and output_path.is_dir():
        msg = f"dataset split manifest output path is a directory: {output_path}"
        raise DatasetSplitError(msg)


def _write_manifest_atomic(
    output_path: Path,
    manifest: DatasetSplitManifest,
) -> None:
    temp_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_output(output_path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(json.dumps(_manifest_to_json_data(manifest), indent=2))
            temp_file.write("\n")
        temp_path.replace(output_path)
    except OSError as exc:
        msg = f"unable to write dataset split manifest {output_path}: {exc}"
        raise DatasetSplitError(msg) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def _remove_existing_output(output_path: Path) -> None:
    if not output_path.exists() and not output_path.is_symlink():
        return
    if output_path.is_dir():
        msg = f"dataset split manifest output path is a directory: {output_path}"
        raise DatasetSplitError(msg)
    try:
        output_path.unlink()
    except OSError as exc:
        msg = f"unable to remove existing dataset split manifest {output_path}: {exc}"
        raise DatasetSplitError(msg) from exc


def _manifest_to_json_data(manifest: DatasetSplitManifest) -> dict[str, object]:
    return {
        "dataset_path": str(manifest.dataset_path),
        "seed": manifest.seed,
        "ratios": manifest.ratios,
        "game_counts": manifest.game_counts,
        "record_counts": manifest.record_counts,
        "splits": {
            name: list(manifest.splits[name])
            for name in SPLIT_NAMES
        },
    }


def _manifest_from_json_data(data: object, path: Path) -> DatasetSplitManifest:
    mapping = _require_mapping(data, f"{path}: manifest")
    return DatasetSplitManifest(
        dataset_path=Path(_require_string(mapping, "dataset_path", path)),
        seed=_require_int(mapping, "seed", path),
        ratios=_require_float_mapping(mapping, "ratios", path),
        game_counts=_require_int_mapping(mapping, "game_counts", path),
        record_counts=_require_int_mapping(mapping, "record_counts", path),
        splits=_require_splits(mapping, path),
    )


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{name} must be an object"
        raise DatasetSplitError(msg)
    return {
        str(key): cast(object, nested_value)
        for key, nested_value in value.items()
    }


def _require_string(mapping: dict[str, object], key: str, path: Path) -> str:
    value = _require_key(mapping, key, path)
    if not isinstance(value, str):
        msg = f"{path}: {key} must be a string"
        raise DatasetSplitError(msg)
    return value


def _require_int(mapping: dict[str, object], key: str, path: Path) -> int:
    value = _require_key(mapping, key, path)
    if not isinstance(value, int):
        msg = f"{path}: {key} must be an integer"
        raise DatasetSplitError(msg)
    return value


def _require_float_mapping(
    mapping: dict[str, object],
    key: str,
    path: Path,
) -> dict[str, float]:
    value = _require_mapping(_require_key(mapping, key, path), f"{path}: {key}")
    result: dict[str, float] = {}
    for split_name in SPLIT_NAMES:
        nested_value = _require_key(value, split_name, path)
        if not isinstance(nested_value, int | float):
            msg = f"{path}: {key}.{split_name} must be a number"
            raise DatasetSplitError(msg)
        result[split_name] = float(nested_value)
    return result


def _require_int_mapping(
    mapping: dict[str, object],
    key: str,
    path: Path,
) -> dict[str, int]:
    value = _require_mapping(_require_key(mapping, key, path), f"{path}: {key}")
    result: dict[str, int] = {}
    for count_name in ("total", *SPLIT_NAMES):
        nested_value = _require_key(value, count_name, path)
        if not isinstance(nested_value, int):
            msg = f"{path}: {key}.{count_name} must be an integer"
            raise DatasetSplitError(msg)
        result[count_name] = nested_value
    return result


def _require_splits(
    mapping: dict[str, object],
    path: Path,
) -> dict[str, tuple[str, ...]]:
    value = _require_mapping(_require_key(mapping, "splits", path), f"{path}: splits")
    result: dict[str, tuple[str, ...]] = {}
    for split_name in SPLIT_NAMES:
        split_value = _require_key(value, split_name, path)
        if not isinstance(split_value, list):
            msg = f"{path}: splits.{split_name} must be a list"
            raise DatasetSplitError(msg)
        if not all(isinstance(game_id, str) for game_id in split_value):
            msg = f"{path}: splits.{split_name} entries must be strings"
            raise DatasetSplitError(msg)
        result[split_name] = tuple(cast(list[str], split_value))
    return result


def _require_key(mapping: dict[str, object], key: str, path: Path) -> object:
    try:
        return mapping[key]
    except KeyError as exc:
        msg = f"{path}: missing required field: {key}"
        raise DatasetSplitError(msg) from exc
