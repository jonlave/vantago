"""Final held-out evaluation report workflow."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from vantago.baselines import (
    BaselineEvaluationError,
    BaselineEvaluationRow,
    BaselineName,
    BaselinePhaseEvaluationRow,
    evaluate_baselines,
)
from vantago.data.splits import (
    SPLIT_NAMES,
    DatasetSplitError,
    load_dataset_split_manifest,
)
from vantago.evaluation import PolicyMetricSummary
from vantago.training import (
    CnnTrainingConfig,
    CnnTrainingError,
    evaluate_cnn_policy_checkpoint,
    load_cnn_policy_checkpoint,
)

FINAL_EVALUATION_SPLIT = "test"
MODEL_SELECTION_METHOD = "best_validation_cross_entropy_checkpoint"
MODEL_SELECTION_SPLIT = "validation"


class FinalEvaluationReportError(ValueError):
    """Raised when the final held-out evaluation report cannot be generated."""


@dataclass(frozen=True, slots=True)
class FinalEvaluationConfig:
    """Runtime options for final held-out evaluation."""

    batch_size: int = 128
    seed: int = 0
    mask_topk: bool | None = None


@dataclass(frozen=True, slots=True)
class FinalEvaluationSelection:
    """Validation-only model selection details for the final report."""

    model: BaselineName
    method: str
    split: str
    best_epoch: int
    validation_metrics: PolicyMetricSummary


@dataclass(frozen=True, slots=True)
class FinalEvaluationCheckpointMetadata:
    """Checkpoint provenance copied into the final report."""

    path: Path
    dataset_path: Path
    manifest_path: Path
    format_version: int
    model_kind: str
    config: CnnTrainingConfig
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class FinalEvaluationReport:
    """Held-out evaluation report for selected policy models."""

    dataset_path: Path
    manifest_path: Path
    checkpoint_path: Path
    split: str
    config: FinalEvaluationConfig
    resolved_mask_topk: bool
    split_seed: int
    split_ratios: dict[str, float]
    game_counts: dict[str, int]
    record_counts: dict[str, int]
    checkpoint_metadata: FinalEvaluationCheckpointMetadata
    selection: FinalEvaluationSelection
    rows: tuple[BaselineEvaluationRow, ...]
    phase_rows: tuple[BaselinePhaseEvaluationRow, ...]


def generate_final_evaluation_report(
    dataset_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    config: FinalEvaluationConfig | None = None,
) -> FinalEvaluationReport:
    """Evaluate selected models once on the held-out test split."""

    resolved_config = FinalEvaluationConfig() if config is None else config
    _validate_config(resolved_config)

    try:
        manifest = load_dataset_split_manifest(manifest_path)
        checkpoint = load_cnn_policy_checkpoint(checkpoint_path)
    except (CnnTrainingError, DatasetSplitError) as exc:
        raise FinalEvaluationReportError(str(exc)) from exc

    resolved_mask_topk = (
        checkpoint.config.mask_topk
        if resolved_config.mask_topk is None
        else resolved_config.mask_topk
    )

    try:
        baseline_result = evaluate_baselines(
            dataset_path,
            manifest_path,
            split=FINAL_EVALUATION_SPLIT,
            seed=resolved_config.seed,
            mask_topk=resolved_mask_topk,
        )
        cnn_result = evaluate_cnn_policy_checkpoint(
            dataset_path,
            manifest_path,
            checkpoint_path,
            split=FINAL_EVALUATION_SPLIT,
            batch_size=resolved_config.batch_size,
            mask_topk=resolved_mask_topk,
        )
    except (BaselineEvaluationError, CnnTrainingError) as exc:
        raise FinalEvaluationReportError(str(exc)) from exc

    return FinalEvaluationReport(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        split=FINAL_EVALUATION_SPLIT,
        config=resolved_config,
        resolved_mask_topk=resolved_mask_topk,
        split_seed=manifest.seed,
        split_ratios=dict(manifest.ratios),
        game_counts=_split_counts(manifest.game_counts),
        record_counts=_split_counts(manifest.record_counts),
        checkpoint_metadata=FinalEvaluationCheckpointMetadata(
            path=checkpoint.path,
            dataset_path=checkpoint.dataset_path,
            manifest_path=checkpoint.manifest_path,
            format_version=checkpoint.format_version,
            model_kind=checkpoint.model_kind,
            config=checkpoint.config,
            started_at=checkpoint.started_at,
            finished_at=checkpoint.finished_at,
            duration_seconds=checkpoint.duration_seconds,
        ),
        selection=FinalEvaluationSelection(
            model="cnn_policy",
            method=MODEL_SELECTION_METHOD,
            split=MODEL_SELECTION_SPLIT,
            best_epoch=checkpoint.best_epoch,
            validation_metrics=checkpoint.best_validation_metrics,
        ),
        rows=(*baseline_result.rows, *cnn_result.rows),
        phase_rows=(*baseline_result.phase_rows, *cnn_result.phase_rows),
    )


def final_evaluation_report_to_json_data(
    report: FinalEvaluationReport,
) -> dict[str, object]:
    """Return deterministic JSON-ready data for a final evaluation report."""

    return {
        "dataset_path": str(report.dataset_path),
        "manifest_path": str(report.manifest_path),
        "checkpoint_path": str(report.checkpoint_path),
        "split": report.split,
        "config": {
            "batch_size": report.config.batch_size,
            "seed": report.config.seed,
            "requested_mask_topk": report.config.mask_topk,
            "mask_topk": report.resolved_mask_topk,
        },
        "split_counts": {
            "games": report.game_counts,
            "records": report.record_counts,
        },
        "split_manifest": {
            "seed": report.split_seed,
            "ratios": report.split_ratios,
        },
        "checkpoint": {
            "path": str(report.checkpoint_metadata.path),
            "dataset_path": str(report.checkpoint_metadata.dataset_path),
            "manifest_path": str(report.checkpoint_metadata.manifest_path),
            "format_version": report.checkpoint_metadata.format_version,
            "model_kind": report.checkpoint_metadata.model_kind,
            "config": _cnn_training_config_to_json_data(
                report.checkpoint_metadata.config,
            ),
            "started_at": report.checkpoint_metadata.started_at,
            "finished_at": report.checkpoint_metadata.finished_at,
            "duration_seconds": report.checkpoint_metadata.duration_seconds,
        },
        "selection": {
            "model": report.selection.model,
            "method": report.selection.method,
            "split": report.selection.split,
            "best_epoch": report.selection.best_epoch,
            "validation_metrics": _metric_summary_to_json_data(
                report.selection.validation_metrics,
            ),
        },
        "rows": [_row_to_json_data(row) for row in report.rows],
        "phase_rows": [
            _phase_row_to_json_data(row)
            for row in report.phase_rows
        ],
    }


def write_final_evaluation_report_json(
    path: Path,
    report: FinalEvaluationReport,
) -> None:
    """Write deterministic JSON for a final evaluation report."""

    validate_final_evaluation_json_output_path(
        path,
        dataset_path=report.dataset_path,
        manifest_path=report.manifest_path,
        checkpoint_path=report.checkpoint_path,
    )
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                final_evaluation_report_to_json_data(report),
                temp_file,
                indent=2,
            )
            temp_file.write("\n")
        temp_path.replace(path)
    except OSError as exc:
        msg = f"could not write final evaluation report JSON {path}: {exc}"
        raise FinalEvaluationReportError(msg) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def validate_final_evaluation_json_output_path(
    path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
) -> None:
    """Reject report output paths that would clobber input artifacts."""

    resolved_path = _resolved_path(path)
    protected_paths = (
        ("dataset", dataset_path),
        ("splits", manifest_path),
        ("checkpoint", checkpoint_path),
    )
    for name, protected_path in protected_paths:
        if resolved_path == _resolved_path(protected_path):
            msg = f"json_out must not overwrite {name} artifact: {path}"
            raise FinalEvaluationReportError(msg)

    if path.suffix != ".json":
        msg = f"json_out must end with .json: {path}"
        raise FinalEvaluationReportError(msg)
    if path.exists() and path.is_dir():
        msg = f"json_out path is a directory: {path}"
        raise FinalEvaluationReportError(msg)


def _validate_config(config: FinalEvaluationConfig) -> None:
    if config.batch_size < 1:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise FinalEvaluationReportError(msg)


def _split_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {
        name: counts[name]
        for name in ("total", *SPLIT_NAMES)
    }


def _row_to_json_data(row: BaselineEvaluationRow) -> dict[str, object]:
    return {
        "baseline": row.baseline,
        "split": row.split,
        "metrics": _metric_summary_to_json_data(row.metrics),
    }


def _phase_row_to_json_data(
    row: BaselinePhaseEvaluationRow,
) -> dict[str, object]:
    return {
        "baseline": row.baseline,
        "split": row.split,
        "phase": row.phase,
        "metrics": (
            None
            if row.metrics is None
            else _metric_summary_to_json_data(row.metrics)
        ),
    }


def _metric_summary_to_json_data(
    metrics: PolicyMetricSummary,
) -> dict[str, float | int | None]:
    return {
        "example_count": metrics.example_count,
        "top_1": metrics.top_1,
        "top_3": metrics.top_3,
        "top_5": metrics.top_5,
        "cross_entropy": metrics.cross_entropy,
        "illegal_move_rate": metrics.illegal_move_rate,
    }


def _cnn_training_config_to_json_data(
    config: CnnTrainingConfig,
) -> dict[str, object]:
    return {
        "checkpoint_path": str(config.checkpoint_path),
        "history_path": (
            None if config.history_path is None else str(config.history_path)
        ),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "hidden_channels": config.hidden_channels,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "mask_topk": config.mask_topk,
    }


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()
