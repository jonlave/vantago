"""Model comparison workflow for first policy experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vantago.baselines import (
    BaselineEvaluationError,
    BaselineEvaluationResult,
    BaselineEvaluationRow,
    BaselinePhaseEvaluationRow,
    MlpBaselineConfig,
    MlpBaselineTrainingError,
    MlpBaselineTrainingResult,
    NonNeuralBaselineName,
    evaluate_baselines,
    train_mlp_baseline,
)
from vantago.evaluation import PolicyMetricSummary
from vantago.training import (
    CnnPolicyEvaluationResult,
    CnnTrainingConfig,
    CnnTrainingError,
    CnnTrainingResult,
    evaluate_cnn_policy_checkpoint,
    train_cnn_policy,
)

TARGET_BASELINES: tuple[NonNeuralBaselineName, ...] = (
    "random_legal",
    "frequency_overall",
    "frequency_by_phase",
)

MISSED_TARGET_NEXT_STEPS: tuple[str, ...] = (
    "Inspect training history for loss divergence or stalled validation metrics.",
    "Review illegal_move_rate and phase rows for masking or data-quality issues.",
    "Rerun with more games or epochs before changing architecture.",
)

MODEL_SELECTION_NOTES: tuple[str, ...] = (
    "mlp_flattened: final_epoch",
    "cnn_policy: best_validation_cross_entropy_checkpoint",
)


class PolicyModelComparisonError(ValueError):
    """Raised when the policy model comparison workflow cannot complete."""


@dataclass(frozen=True, slots=True)
class PolicyModelComparisonConfig:
    """Hyperparameters and output paths for comparing policy models."""

    checkpoint_path: Path
    history_path: Path | None = None
    mlp_history_path: Path | None = None
    epochs: int = 5
    batch_size: int = 128
    cnn_hidden_channels: int = 64
    mlp_hidden_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 0
    mask_topk: bool = False


@dataclass(frozen=True, slots=True)
class PolicyComparisonDelta:
    """CNN top-1 delta against one non-neural baseline."""

    baseline: NonNeuralBaselineName
    baseline_top_1: float
    cnn_top_1: float
    delta: float


@dataclass(frozen=True, slots=True)
class PolicyComparisonTarget:
    """Summary of whether the CNN cleared the initial comparison target."""

    met: bool
    deltas: tuple[PolicyComparisonDelta, ...]
    next_steps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyModelComparisonResult:
    """Completed first policy comparison workflow."""

    dataset_path: Path
    manifest_path: Path
    config: PolicyModelComparisonConfig
    baseline_result: BaselineEvaluationResult
    mlp_result: MlpBaselineTrainingResult
    cnn_training_result: CnnTrainingResult
    cnn_evaluation_result: CnnPolicyEvaluationResult
    mlp_history_path: Path
    rows: tuple[BaselineEvaluationRow, ...]
    phase_rows: tuple[BaselinePhaseEvaluationRow, ...]
    target: PolicyComparisonTarget
    model_selection_notes: tuple[str, ...]


def compare_policy_models(
    dataset_path: Path,
    manifest_path: Path,
    *,
    config: PolicyModelComparisonConfig,
) -> PolicyModelComparisonResult:
    """Train and compare non-neural, MLP, and CNN policy models."""

    mlp_history_path = _resolve_mlp_history_path(config)
    _validate_history_paths(config, mlp_history_path)

    try:
        mlp_result = train_mlp_baseline(
            dataset_path,
            manifest_path,
            config=MlpBaselineConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                hidden_size=config.mlp_hidden_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                seed=config.seed,
                mask_topk=config.mask_topk,
            ),
        )
        _write_mlp_history(
            mlp_history_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            config=config,
            mlp_result=mlp_result,
        )
        cnn_training_result = train_cnn_policy(
            dataset_path,
            manifest_path,
            config=CnnTrainingConfig(
                checkpoint_path=config.checkpoint_path,
                history_path=config.history_path,
                epochs=config.epochs,
                batch_size=config.batch_size,
                hidden_channels=config.cnn_hidden_channels,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                seed=config.seed,
                mask_topk=config.mask_topk,
            ),
        )
        baseline_result = evaluate_baselines(
            dataset_path,
            manifest_path,
            split="validation",
            seed=config.seed,
            mask_topk=config.mask_topk,
        )
        cnn_evaluation_result = evaluate_cnn_policy_checkpoint(
            dataset_path,
            manifest_path,
            cnn_training_result.checkpoint_path,
            split="validation",
            batch_size=config.batch_size,
            mask_topk=config.mask_topk,
        )
    except (
        BaselineEvaluationError,
        CnnTrainingError,
        MlpBaselineTrainingError,
    ) as exc:
        raise PolicyModelComparisonError(str(exc)) from exc

    rows = (
        *baseline_result.rows,
        mlp_result.validation_row,
        *cnn_evaluation_result.rows,
    )
    phase_rows = (
        *baseline_result.phase_rows,
        *mlp_result.validation_phase_rows,
        *cnn_evaluation_result.phase_rows,
    )

    return PolicyModelComparisonResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        config=config,
        baseline_result=baseline_result,
        mlp_result=mlp_result,
        cnn_training_result=cnn_training_result,
        cnn_evaluation_result=cnn_evaluation_result,
        mlp_history_path=mlp_history_path,
        rows=rows,
        phase_rows=phase_rows,
        target=summarize_policy_comparison_target(rows),
        model_selection_notes=MODEL_SELECTION_NOTES,
    )


def summarize_policy_comparison_target(
    rows: Sequence[BaselineEvaluationRow],
) -> PolicyComparisonTarget:
    """Return whether CNN top-1 strictly beats all non-neural baselines."""

    cnn_row = _require_row(rows, "cnn_policy")
    cnn_top_1 = cnn_row.metrics.top_1
    deltas = tuple(
        PolicyComparisonDelta(
            baseline=baseline,
            baseline_top_1=baseline_row.metrics.top_1,
            cnn_top_1=cnn_top_1,
            delta=cnn_top_1 - baseline_row.metrics.top_1,
        )
        for baseline in TARGET_BASELINES
        for baseline_row in (_require_row(rows, baseline),)
    )
    met = all(delta.delta > 0.0 for delta in deltas)
    return PolicyComparisonTarget(
        met=met,
        deltas=deltas,
        next_steps=() if met else MISSED_TARGET_NEXT_STEPS,
    )


def _require_row(
    rows: Sequence[BaselineEvaluationRow],
    baseline: str,
) -> BaselineEvaluationRow:
    for row in rows:
        if row.baseline == baseline:
            return row

    msg = f"comparison rows are missing {baseline!r}"
    raise PolicyModelComparisonError(msg)


def _resolve_mlp_history_path(config: PolicyModelComparisonConfig) -> Path:
    if config.mlp_history_path is not None:
        return config.mlp_history_path
    return config.checkpoint_path.with_suffix(".mlp-history.json")


def _resolve_cnn_history_path(config: PolicyModelComparisonConfig) -> Path:
    if config.history_path is not None:
        return config.history_path
    return config.checkpoint_path.with_suffix(".history.json")


def _validate_history_paths(
    config: PolicyModelComparisonConfig,
    mlp_history_path: Path,
) -> None:
    if mlp_history_path.suffix != ".json":
        msg = f"mlp_history_path must end with .json: {mlp_history_path}"
        raise PolicyModelComparisonError(msg)

    resolved_mlp_history_path = mlp_history_path.resolve()
    resolved_checkpoint_path = config.checkpoint_path.resolve()
    resolved_cnn_history_path = _resolve_cnn_history_path(config).resolve()
    if resolved_mlp_history_path == resolved_checkpoint_path:
        msg = (
            "mlp_history_path and checkpoint_path must be different: "
            f"{mlp_history_path}"
        )
        raise PolicyModelComparisonError(msg)
    if resolved_mlp_history_path == resolved_cnn_history_path:
        msg = (
            "mlp_history_path and history_path must be different: "
            f"{mlp_history_path}"
        )
        raise PolicyModelComparisonError(msg)


def _write_mlp_history(
    path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    config: PolicyModelComparisonConfig,
    mlp_result: MlpBaselineTrainingResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "model": "mlp_flattened",
        "selection": "final_epoch",
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "hidden_size": config.mlp_hidden_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "mask_topk": config.mask_topk,
        },
        "history": [
            {
                "epoch": epoch.epoch,
                "train_loss": epoch.train_loss,
                "validation_metrics": _metric_summary_to_json_mapping(
                    epoch.validation_metrics,
                ),
            }
            for epoch in mlp_result.history
        ],
        "validation_row": {
            "baseline": mlp_result.validation_row.baseline,
            "split": mlp_result.validation_row.split,
            "metrics": _metric_summary_to_json_mapping(
                mlp_result.validation_row.metrics,
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _metric_summary_to_json_mapping(
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
