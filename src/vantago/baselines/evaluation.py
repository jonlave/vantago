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
class BaselineEvaluationResult:
    """Structured result for non-neural baseline evaluation."""

    dataset_path: Path
    manifest_path: Path
    split: str
    seed: int
    mask_topk: bool
    rows: tuple[BaselineEvaluationRow, ...]


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
    rows = _evaluate_all_baselines(
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
) -> tuple[BaselineEvaluationRow, ...]:
    _ensure_nonempty_dataset(eval_dataset)
    random_accumulator = PolicyMetricAccumulator()
    overall_accumulator = PolicyMetricAccumulator()
    phase_accumulator = PolicyMetricAccumulator()
    generator = torch.Generator()
    generator.manual_seed(seed)

    for batch in _iter_dataset_batches(eval_dataset):
        _update_metric_accumulator(
            random_accumulator,
            scores=_random_legal_scores(batch.legal_mask, generator=generator),
            batch=batch,
            mask_topk=mask_topk,
        )
        _update_metric_accumulator(
            overall_accumulator,
            scores=_frequency_scores(overall_counts, len(batch.labels)),
            batch=batch,
            mask_topk=mask_topk,
        )
        _update_metric_accumulator(
            phase_accumulator,
            scores=_phase_frequency_scores(
                overall_counts=overall_counts,
                phase_counts=phase_counts,
                move_numbers=batch.move_numbers,
            ),
            batch=batch,
            mask_topk=mask_topk,
        )

    return (
        _baseline_row("random_legal", split, random_accumulator),
        _baseline_row("frequency_overall", split, overall_accumulator),
        _baseline_row("frequency_by_phase", split, phase_accumulator),
    )


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

    return _BaselineBatch(
        labels=batch["y"].to(dtype=torch.int64),
        legal_mask=batch["legal_mask"].to(dtype=torch.bool),
        move_numbers=batch["move_number"].to(dtype=torch.int64),
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
    if phase == "opening":
        return move_numbers <= 40
    if phase == "middle_game":
        return (move_numbers >= 41) & (move_numbers <= 150)
    return move_numbers >= 151


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
