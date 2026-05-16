"""Evaluate simple policy baselines on processed split datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from vantago.data.artifacts import ProcessedDatasetError
from vantago.data.encoding import SUPPORTED_LABEL_COUNT
from vantago.data.splits import SPLIT_NAMES, DatasetSplitError
from vantago.data.torch_loading import (
    ProcessedPolicyMetadataDataset,
    load_policy_metadata_datasets,
)
from vantago.evaluation import (
    PolicyMetricAccumulator,
    PolicyMetricError,
    PolicyMetricSummary,
)

GamePhase: TypeAlias = Literal["opening", "middle_game", "endgame"]
NonNeuralBaselineName: TypeAlias = Literal[
    "random_legal",
    "frequency_overall",
    "frequency_by_phase",
]
BaselineName: TypeAlias = Literal[
    "random_legal",
    "frequency_overall",
    "frequency_by_phase",
    "mlp_flattened",
    "cnn_policy",
]

PHASE_NAMES: tuple[GamePhase, ...] = ("opening", "middle_game", "endgame")
NON_NEURAL_BASELINE_NAMES: tuple[NonNeuralBaselineName, ...] = (
    "random_legal",
    "frequency_overall",
    "frequency_by_phase",
)
BASELINE_NAMES: tuple[NonNeuralBaselineName, ...] = NON_NEURAL_BASELINE_NAMES
COMPARISON_BASELINE_NAMES: tuple[BaselineName, ...] = (
    *NON_NEURAL_BASELINE_NAMES,
    "mlp_flattened",
    "cnn_policy",
)
DEFAULT_EVALUATION_BATCH_SIZE = 4096


class BaselineEvaluationError(ValueError):
    """Raised when baseline evaluation cannot be completed."""


@dataclass(frozen=True, slots=True)
class BaselineEvaluationRow:
    """Metric row for one evaluated baseline."""

    baseline: BaselineName
    split: str
    metrics: PolicyMetricSummary


@dataclass(frozen=True, slots=True)
class BaselinePhaseEvaluationRow:
    """Metric row for one baseline in one game phase."""

    baseline: BaselineName
    split: str
    phase: GamePhase
    metrics: PolicyMetricSummary | None


@dataclass(frozen=True, slots=True)
class BaselineEvaluationResult:
    """Structured result for non-neural baseline evaluation."""

    dataset_path: Path
    manifest_path: Path
    split: str
    seed: int
    mask_topk: bool
    rows: tuple[BaselineEvaluationRow, ...]
    phase_rows: tuple[BaselinePhaseEvaluationRow, ...]


@dataclass(frozen=True, slots=True)
class _BaselineBatch:
    labels: torch.Tensor
    legal_mask: torch.Tensor
    move_numbers: torch.Tensor


def game_phase_for_move_number(move_number: int) -> GamePhase:
    """Return the roadmap game phase bucket for a one-based move number."""

    if move_number < 1:
        msg = f"move_number must be positive, got {move_number}"
        raise BaselineEvaluationError(msg)
    if move_number <= 40:
        return "opening"
    if move_number <= 150:
        return "middle_game"
    return "endgame"


def phase_mask_for_move_numbers(
    move_numbers: torch.Tensor,
    phase: str,
) -> torch.Tensor:
    """Return a boolean mask for one roadmap phase over move numbers."""

    _validate_move_numbers(move_numbers)
    if phase == "opening":
        return move_numbers <= 40
    if phase == "middle_game":
        return (move_numbers >= 41) & (move_numbers <= 150)
    if phase == "endgame":
        return move_numbers >= 151

    msg = f"phase must be one of {', '.join(PHASE_NAMES)}, got {phase}"
    raise BaselineEvaluationError(msg)


def evaluate_baselines(
    dataset_path: Path,
    manifest_path: Path,
    *,
    split: str = "validation",
    seed: int = 0,
    mask_topk: bool = False,
) -> BaselineEvaluationResult:
    """Evaluate random legal and frequency baselines on one split."""

    split_name = _validate_split(split)
    datasets = _load_split_datasets(
        dataset_path,
        manifest_path,
        splits=("train",) if split_name == "train" else ("train", split_name),
    )
    train_dataset = datasets["train"]
    eval_dataset = datasets[split_name]

    overall_counts, phase_counts = _fit_frequency_counts(train_dataset)
    rows, phase_rows = _evaluate_all_baselines(
        eval_dataset=eval_dataset,
        overall_counts=overall_counts,
        phase_counts=phase_counts,
        split=split_name,
        seed=seed,
        mask_topk=mask_topk,
    )
    return BaselineEvaluationResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        split=split_name,
        seed=seed,
        mask_topk=mask_topk,
        rows=rows,
        phase_rows=phase_rows,
    )


def _validate_split(split: str) -> str:
    if split not in SPLIT_NAMES:
        msg = f"split must be one of {', '.join(SPLIT_NAMES)}, got {split}"
        raise BaselineEvaluationError(msg)
    return split


def _load_split_datasets(
    dataset_path: Path,
    manifest_path: Path,
    splits: Sequence[str],
) -> dict[str, ProcessedPolicyMetadataDataset]:
    try:
        return load_policy_metadata_datasets(
            dataset_path,
            manifest_path,
            splits=splits,
        )
    except (DatasetSplitError, ProcessedDatasetError) as exc:
        raise BaselineEvaluationError(str(exc)) from exc


def _ensure_nonempty_dataset(dataset: ProcessedPolicyMetadataDataset) -> None:
    if len(dataset) == 0:
        msg = f"{dataset.split} split contains no examples"
        raise BaselineEvaluationError(msg)


def _fit_frequency_counts(
    train_dataset: ProcessedPolicyMetadataDataset,
) -> tuple[torch.Tensor, dict[GamePhase, torch.Tensor]]:
    _ensure_nonempty_dataset(train_dataset)
    overall_counts = torch.zeros(SUPPORTED_LABEL_COUNT, dtype=torch.float32)
    phase_counts = {
        phase: torch.zeros(SUPPORTED_LABEL_COUNT, dtype=torch.float32)
        for phase in PHASE_NAMES
    }

    for batch in _iter_dataset_batches(train_dataset):
        overall_counts += torch.bincount(
            batch.labels,
            minlength=SUPPORTED_LABEL_COUNT,
        ).to(dtype=torch.float32)
        for phase in PHASE_NAMES:
            phase_mask = _phase_mask(batch.move_numbers, phase)
            if bool(phase_mask.any().item()):
                phase_counts[phase] += torch.bincount(
                    batch.labels[phase_mask],
                    minlength=SUPPORTED_LABEL_COUNT,
                ).to(dtype=torch.float32)

    return overall_counts, phase_counts


def _evaluate_all_baselines(
    *,
    eval_dataset: ProcessedPolicyMetadataDataset,
    overall_counts: torch.Tensor,
    phase_counts: dict[GamePhase, torch.Tensor],
    split: str,
    seed: int,
    mask_topk: bool,
) -> tuple[tuple[BaselineEvaluationRow, ...], tuple[BaselinePhaseEvaluationRow, ...]]:
    _ensure_nonempty_dataset(eval_dataset)
    random_accumulator = PolicyMetricAccumulator()
    overall_accumulator = PolicyMetricAccumulator()
    phase_accumulator = PolicyMetricAccumulator()
    random_phase_accumulators = _phase_accumulators()
    overall_phase_accumulators = _phase_accumulators()
    phase_phase_accumulators = _phase_accumulators()
    generator = torch.Generator()
    generator.manual_seed(seed)

    for batch in _iter_dataset_batches(eval_dataset):
        random_scores = _random_legal_scores(batch.legal_mask, generator=generator)
        overall_scores = _frequency_scores(overall_counts, len(batch.labels))
        phase_scores = _phase_frequency_scores(
            overall_counts=overall_counts,
            phase_counts=phase_counts,
            move_numbers=batch.move_numbers,
        )
        _update_metric_accumulator(
            random_accumulator,
            scores=random_scores,
            batch=batch,
            mask_topk=mask_topk,
        )
        _update_phase_metric_accumulators(
            random_phase_accumulators,
            scores=random_scores,
            batch=batch,
            mask_topk=mask_topk,
        )
        _update_metric_accumulator(
            overall_accumulator,
            scores=overall_scores,
            batch=batch,
            mask_topk=mask_topk,
        )
        _update_phase_metric_accumulators(
            overall_phase_accumulators,
            scores=overall_scores,
            batch=batch,
            mask_topk=mask_topk,
        )
        _update_metric_accumulator(
            phase_accumulator,
            scores=phase_scores,
            batch=batch,
            mask_topk=mask_topk,
        )
        _update_phase_metric_accumulators(
            phase_phase_accumulators,
            scores=phase_scores,
            batch=batch,
            mask_topk=mask_topk,
        )

    rows = (
        _baseline_row("random_legal", split, random_accumulator),
        _baseline_row("frequency_overall", split, overall_accumulator),
        _baseline_row("frequency_by_phase", split, phase_accumulator),
    )
    phase_rows = (
        *_baseline_phase_rows(
            "random_legal",
            split,
            random_phase_accumulators,
        ),
        *_baseline_phase_rows(
            "frequency_overall",
            split,
            overall_phase_accumulators,
        ),
        *_baseline_phase_rows(
            "frequency_by_phase",
            split,
            phase_phase_accumulators,
        ),
    )
    return rows, phase_rows


def _iter_dataset_batches(
    dataset: ProcessedPolicyMetadataDataset,
    *,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
) -> Iterator[_BaselineBatch]:
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        yield _dataset_batch(dataset, start, stop)


def _dataset_batch(
    dataset: ProcessedPolicyMetadataDataset,
    start: int,
    stop: int,
) -> _BaselineBatch:
    batch = dataset.metadata_batch(start, stop)
    move_numbers = batch["move_number"].to(dtype=torch.int64)
    _validate_move_numbers(move_numbers)

    return _BaselineBatch(
        labels=batch["y"].to(dtype=torch.int64),
        legal_mask=batch["legal_mask"].to(dtype=torch.bool),
        move_numbers=move_numbers,
    )


def _random_legal_scores(
    legal_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    scores = torch.rand(
        legal_mask.shape,
        dtype=torch.float32,
        generator=generator,
    )
    return scores.masked_fill(~legal_mask, float("-inf"))


def _frequency_scores(counts: torch.Tensor, example_count: int) -> torch.Tensor:
    return counts.unsqueeze(0).expand(example_count, -1)


def _phase_frequency_scores(
    *,
    overall_counts: torch.Tensor,
    phase_counts: dict[GamePhase, torch.Tensor],
    move_numbers: torch.Tensor,
) -> torch.Tensor:
    scores = _frequency_scores(overall_counts, int(move_numbers.shape[0])).clone()
    for phase in PHASE_NAMES:
        counts = phase_counts[phase]
        if not bool(counts.any().item()):
            continue

        phase_mask = _phase_mask(move_numbers, phase)
        if bool(phase_mask.any().item()):
            scores[phase_mask] = counts
    return scores


def _phase_mask(move_numbers: torch.Tensor, phase: GamePhase) -> torch.Tensor:
    return phase_mask_for_move_numbers(move_numbers, phase)


def _validate_move_numbers(move_numbers: torch.Tensor) -> None:
    if move_numbers.ndim != 1:
        msg = f"move_numbers must have shape [N], got {tuple(move_numbers.shape)}"
        raise BaselineEvaluationError(msg)
    if move_numbers.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        msg = f"move_numbers must be an integer tensor, got {move_numbers.dtype}"
        raise BaselineEvaluationError(msg)
    if bool((move_numbers < 1).any().item()):
        msg = "move_numbers must be positive"
        raise BaselineEvaluationError(msg)


def _update_metric_accumulator(
    accumulator: PolicyMetricAccumulator,
    *,
    scores: torch.Tensor,
    batch: _BaselineBatch,
    mask_topk: bool,
) -> None:
    try:
        accumulator.update(
            scores=scores,
            labels=batch.labels,
            legal_mask=batch.legal_mask,
            apply_legal_mask_before_topk=mask_topk,
        )
    except PolicyMetricError as exc:
        raise BaselineEvaluationError(str(exc)) from exc


def _phase_accumulators() -> dict[GamePhase, PolicyMetricAccumulator]:
    return {phase: PolicyMetricAccumulator() for phase in PHASE_NAMES}


def _update_phase_metric_accumulators(
    accumulators: dict[GamePhase, PolicyMetricAccumulator],
    *,
    scores: torch.Tensor,
    batch: _BaselineBatch,
    mask_topk: bool,
) -> None:
    for phase in PHASE_NAMES:
        example_mask = _phase_mask(batch.move_numbers, phase)
        if not bool(example_mask.any().item()):
            continue

        _update_metric_accumulator(
            accumulators[phase],
            scores=scores[example_mask],
            batch=_filter_batch(batch, example_mask),
            mask_topk=mask_topk,
        )


def _filter_batch(
    batch: _BaselineBatch,
    example_mask: torch.Tensor,
) -> _BaselineBatch:
    return _BaselineBatch(
        labels=batch.labels[example_mask],
        legal_mask=batch.legal_mask[example_mask],
        move_numbers=batch.move_numbers[example_mask],
    )


def _baseline_row(
    baseline: NonNeuralBaselineName,
    split: str,
    accumulator: PolicyMetricAccumulator,
) -> BaselineEvaluationRow:
    try:
        metrics = accumulator.summary()
    except PolicyMetricError as exc:
        raise BaselineEvaluationError(str(exc)) from exc
    return BaselineEvaluationRow(
        baseline=baseline,
        split=split,
        metrics=metrics,
    )


def _baseline_phase_rows(
    baseline: BaselineName,
    split: str,
    accumulators: dict[GamePhase, PolicyMetricAccumulator],
) -> tuple[BaselinePhaseEvaluationRow, ...]:
    return tuple(
        BaselinePhaseEvaluationRow(
            baseline=baseline,
            split=split,
            phase=phase,
            metrics=(
                accumulators[phase].summary()
                if accumulators[phase].example_count > 0
                else None
            ),
        )
        for phase in PHASE_NAMES
    )
