"""Persist and inspect processed supervised policy datasets."""

from __future__ import annotations

import tempfile
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from vantago.data.encoding import (
    CHANNEL_COUNT,
    SUPPORTED_LABEL_COUNT,
    BoolArray,
    Float32Array,
    PositionRecord,
    decode_label,
    encode_replay_steps,
)
from vantago.replay import (
    SUPPORTED_BOARD_SIZE,
    ReplayDiagnostic,
    ReplayDiagnosticStatus,
    ReplaySkipReason,
    diagnose_sgf_replay_file,
    find_sgf_files,
)

Int64Array: TypeAlias = npt.NDArray[np.int64]
StringArray: TypeAlias = npt.NDArray[np.str_]

DATASET_KEYS = (
    "x",
    "y",
    "legal_mask",
    "game_id",
    "move_number",
    "source_name",
)


class ProcessedDatasetError(ValueError):
    """Raised when a processed dataset artifact is invalid."""


@dataclass(frozen=True, slots=True)
class ProcessedDatasetArtifact:
    """Loaded arrays from a processed dataset artifact."""

    x: Float32Array
    y: Int64Array
    legal_mask: BoolArray
    game_id: StringArray
    move_number: Int64Array
    source_name: StringArray


@dataclass(frozen=True, slots=True)
class ProcessedDatasetSkipCount:
    """Count of skipped games for one stable replay skip reason."""

    reason: ReplaySkipReason
    count: int


@dataclass(frozen=True, slots=True)
class ProcessedDatasetFailure:
    """Unexpected failure while encoding one SGF file."""

    source_name: str
    message: str


@dataclass(frozen=True, slots=True)
class ProcessedDatasetBuildResult:
    """Structured result for building a processed dataset artifact."""

    input_path: Path
    output_path: Path
    files_scanned: int
    ok: int
    skipped: int
    failed: int
    records_written: int
    skipped_by_reason: tuple[ProcessedDatasetSkipCount, ...]
    skipped_diagnostics: tuple[ReplayDiagnostic, ...]
    failures: tuple[ProcessedDatasetFailure, ...]


@dataclass(frozen=True, slots=True)
class ProcessedDatasetInspection:
    """Human-readable facts for one processed dataset record."""

    path: Path
    index: int
    record_count: int
    game_id: str
    source_name: str
    move_number: int
    y: int
    decoded_row: int
    decoded_col: int
    x_shape: tuple[int, ...]
    legal_mask_count: int
    label_is_legal: bool


def write_processed_dataset(
    input_path: Path,
    output_path: Path,
) -> ProcessedDatasetBuildResult:
    """Encode SGFs under ``input_path`` and write a processed ``.npz`` artifact."""

    sgf_files = find_sgf_files(input_path)
    _validate_output_path(output_path)
    records: list[PositionRecord] = []
    source_names: list[str] = []
    skipped_diagnostics: list[ReplayDiagnostic] = []
    failures: list[ProcessedDatasetFailure] = []
    skip_counts: Counter[ReplaySkipReason] = Counter()
    ok = 0

    for sgf_file in sgf_files:
        try:
            diagnostic = diagnose_sgf_replay_file(sgf_file)
        except Exception as exc:
            failures.append(
                ProcessedDatasetFailure(source_name=str(sgf_file), message=str(exc))
            )
            continue

        if diagnostic.status != ReplayDiagnosticStatus.OK:
            if diagnostic.reason is None:
                failures.append(
                    ProcessedDatasetFailure(
                        source_name=diagnostic.source_name,
                        message=(
                            "skipped diagnostic missing reason: "
                            f"{diagnostic.message}"
                        ),
                    )
                )
                continue
            skipped_diagnostics.append(diagnostic)
            skip_counts[diagnostic.reason] += 1
            continue

        game_id = _game_id_for_path(input_path, sgf_file)
        try:
            game_records = encode_replay_steps(game_id, diagnostic.replay_steps)
        except Exception as exc:
            failures.append(
                ProcessedDatasetFailure(
                    source_name=diagnostic.source_name,
                    message=str(exc),
                )
            )
            continue

        ok += 1
        records.extend(game_records)
        source_names.extend([diagnostic.source_name] * len(game_records))

    records_written = 0
    if failures or not records:
        _remove_existing_output(output_path)
    else:
        artifact = _artifact_from_records(records, source_names)
        _validate_artifact(artifact)
        _write_artifact_atomic(output_path, artifact)
        records_written = len(records)

    return ProcessedDatasetBuildResult(
        input_path=input_path,
        output_path=output_path,
        files_scanned=len(sgf_files),
        ok=ok,
        skipped=len(skipped_diagnostics),
        failed=len(failures),
        records_written=records_written,
        skipped_by_reason=_skip_counts_by_reason(skip_counts),
        skipped_diagnostics=tuple(skipped_diagnostics),
        failures=tuple(failures),
    )


def load_processed_dataset(path: Path) -> ProcessedDatasetArtifact:
    """Load and validate a processed dataset artifact."""

    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing_keys = sorted(set(DATASET_KEYS) - set(loaded.files))
            if missing_keys:
                msg = f"{path}: missing required arrays: {', '.join(missing_keys)}"
                raise ProcessedDatasetError(msg)

            artifact = ProcessedDatasetArtifact(
                x=cast(Float32Array, np.asarray(loaded["x"])),
                y=cast(Int64Array, np.asarray(loaded["y"])),
                legal_mask=cast(BoolArray, np.asarray(loaded["legal_mask"])),
                game_id=cast(StringArray, np.asarray(loaded["game_id"])),
                move_number=cast(Int64Array, np.asarray(loaded["move_number"])),
                source_name=cast(StringArray, np.asarray(loaded["source_name"])),
            )
    except ProcessedDatasetError:
        raise
    except (OSError, ValueError) as exc:
        msg = f"unable to load processed dataset {path}: {exc}"
        raise ProcessedDatasetError(msg) from exc

    _validate_artifact(artifact)
    return artifact


def inspect_processed_dataset(
    path: Path,
    *,
    index: int = 0,
) -> ProcessedDatasetInspection:
    """Return inspectable metadata for one processed dataset record."""

    artifact = load_processed_dataset(path)
    record_count = int(artifact.y.shape[0])
    if not 0 <= index < record_count:
        msg = f"index must be in [0, {record_count}), got {index}"
        raise ProcessedDatasetError(msg)

    y = int(artifact.y[index])
    decoded = decode_label(y)
    return ProcessedDatasetInspection(
        path=path,
        index=index,
        record_count=record_count,
        game_id=str(artifact.game_id[index]),
        source_name=str(artifact.source_name[index]),
        move_number=int(artifact.move_number[index]),
        y=y,
        decoded_row=decoded.row,
        decoded_col=decoded.col,
        x_shape=tuple(int(dim) for dim in artifact.x[index].shape),
        legal_mask_count=int(artifact.legal_mask[index].sum()),
        label_is_legal=bool(artifact.legal_mask[index, y]),
    )


def _artifact_from_records(
    records: list[PositionRecord],
    source_names: list[str],
) -> ProcessedDatasetArtifact:
    if len(records) != len(source_names):
        msg = "source_names must contain one entry per position record"
        raise ProcessedDatasetError(msg)

    return ProcessedDatasetArtifact(
        x=np.stack([record.x for record in records], axis=0).astype(
            np.float32,
            copy=False,
        ),
        y=np.array([record.y for record in records], dtype=np.int64),
        legal_mask=np.stack([record.legal_mask for record in records], axis=0).astype(
            np.bool_,
            copy=False,
        ),
        game_id=np.array([record.game_id for record in records], dtype=np.str_),
        move_number=np.array(
            [record.move_number for record in records],
            dtype=np.int64,
        ),
        source_name=np.array(source_names, dtype=np.str_),
    )


def _validate_output_path(output_path: Path) -> None:
    if output_path.suffix != ".npz":
        msg = f"processed dataset output path must end with .npz: {output_path}"
        raise ProcessedDatasetError(msg)
    if output_path.exists() and output_path.is_dir():
        msg = f"processed dataset output path is a directory: {output_path}"
        raise ProcessedDatasetError(msg)


def _write_artifact_atomic(
    output_path: Path,
    artifact: ProcessedDatasetArtifact,
) -> None:
    temp_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_output(output_path)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            np.savez_compressed(
                temp_file,
                x=artifact.x,
                y=artifact.y,
                legal_mask=artifact.legal_mask,
                game_id=artifact.game_id,
                move_number=artifact.move_number,
                source_name=artifact.source_name,
            )
        temp_path.replace(output_path)
    except OSError as exc:
        msg = f"unable to write processed dataset {output_path}: {exc}"
        raise ProcessedDatasetError(msg) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def _remove_existing_output(output_path: Path) -> None:
    if not output_path.exists() and not output_path.is_symlink():
        return
    if output_path.is_dir():
        msg = f"processed dataset output path is a directory: {output_path}"
        raise ProcessedDatasetError(msg)
    try:
        output_path.unlink()
    except OSError as exc:
        msg = f"unable to remove existing processed dataset {output_path}: {exc}"
        raise ProcessedDatasetError(msg) from exc


def _validate_artifact(artifact: ProcessedDatasetArtifact) -> None:
    record_count = _validate_x(artifact.x)
    _validate_y(artifact.y, record_count)
    _validate_legal_mask(artifact.legal_mask, record_count)
    _validate_string_array("game_id", artifact.game_id, record_count)
    _validate_move_number(artifact.move_number, record_count)
    _validate_string_array("source_name", artifact.source_name, record_count)

    row_indices = np.arange(record_count)
    if not np.all(artifact.legal_mask[row_indices, artifact.y]):
        msg = "legal_mask must mark every target label as legal"
        raise ProcessedDatasetError(msg)


def _validate_x(x: Float32Array) -> int:
    expected_suffix = (CHANNEL_COUNT, SUPPORTED_BOARD_SIZE, SUPPORTED_BOARD_SIZE)
    if x.ndim != 4 or x.shape[1:] != expected_suffix:
        msg = f"x must have shape [N, 5, 19, 19], got {x.shape}"
        raise ProcessedDatasetError(msg)
    if x.dtype != np.dtype(np.float32):
        msg = f"x must have float32 dtype, got {x.dtype}"
        raise ProcessedDatasetError(msg)
    return int(x.shape[0])


def _validate_y(y: Int64Array, record_count: int) -> None:
    if y.shape != (record_count,):
        msg = f"y must have shape ({record_count},), got {y.shape}"
        raise ProcessedDatasetError(msg)
    if y.dtype != np.dtype(np.int64):
        msg = f"y must have int64 dtype, got {y.dtype}"
        raise ProcessedDatasetError(msg)
    if np.any((y < 0) | (y >= SUPPORTED_LABEL_COUNT)):
        msg = f"y values must be in [0, {SUPPORTED_LABEL_COUNT})"
        raise ProcessedDatasetError(msg)


def _validate_legal_mask(legal_mask: BoolArray, record_count: int) -> None:
    expected_shape = (record_count, SUPPORTED_LABEL_COUNT)
    if legal_mask.shape != expected_shape:
        msg = f"legal_mask must have shape {expected_shape}, got {legal_mask.shape}"
        raise ProcessedDatasetError(msg)
    if legal_mask.dtype != np.dtype(np.bool_):
        msg = f"legal_mask must have bool dtype, got {legal_mask.dtype}"
        raise ProcessedDatasetError(msg)


def _validate_move_number(move_number: Int64Array, record_count: int) -> None:
    if move_number.shape != (record_count,):
        msg = (
            f"move_number must have shape ({record_count},), "
            f"got {move_number.shape}"
        )
        raise ProcessedDatasetError(msg)
    if move_number.dtype != np.dtype(np.int64):
        msg = f"move_number must have int64 dtype, got {move_number.dtype}"
        raise ProcessedDatasetError(msg)
    if np.any(move_number < 1):
        msg = "move_number values must be positive"
        raise ProcessedDatasetError(msg)


def _validate_string_array(name: str, array: StringArray, record_count: int) -> None:
    if array.shape != (record_count,):
        msg = f"{name} must have shape ({record_count},), got {array.shape}"
        raise ProcessedDatasetError(msg)
    if array.dtype.kind != "U":
        msg = f"{name} must have string dtype, got {array.dtype}"
        raise ProcessedDatasetError(msg)


def _skip_counts_by_reason(
    skip_counts: Counter[ReplaySkipReason],
) -> tuple[ProcessedDatasetSkipCount, ...]:
    return tuple(
        ProcessedDatasetSkipCount(reason=reason, count=skip_counts[reason])
        for reason in ReplaySkipReason
        if skip_counts[reason] > 0
    )


def _game_id_for_path(input_path: Path, sgf_file: Path) -> str:
    if input_path.is_file():
        return sgf_file.name
    return sgf_file.relative_to(input_path).as_posix()
