"""Flattened-board MLP baseline training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vantago.baselines.evaluation import BaselineEvaluationRow
from vantago.data.artifacts import ProcessedDatasetError
from vantago.data.encoding import CHANNEL_COUNT, SUPPORTED_LABEL_COUNT
from vantago.data.splits import DatasetSplitError
from vantago.data.torch_loading import PolicyBatch, load_policy_dataloaders
from vantago.evaluation import (
    PolicyMetricAccumulator,
    PolicyMetricError,
    PolicyMetricSummary,
)
from vantago.replay import SUPPORTED_BOARD_SIZE


class MlpBaselineTrainingError(ValueError):
    """Raised when the flattened-board MLP baseline cannot train or evaluate."""


@dataclass(frozen=True, slots=True)
class MlpBaselineConfig:
    """Hyperparameters for the flattened-board MLP baseline."""

    epochs: int = 5
    batch_size: int = 128
    hidden_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 0
    mask_topk: bool = False


@dataclass(frozen=True, slots=True)
class MlpBaselineEpochResult:
    """Training and validation metrics for one MLP baseline epoch."""

    epoch: int
    train_loss: float
    validation_metrics: PolicyMetricSummary


@dataclass(frozen=True, slots=True)
class MlpBaselineTrainingResult:
    """Final in-memory MLP baseline and its validation history."""

    dataset_path: Path
    manifest_path: Path
    config: MlpBaselineConfig
    model: FlattenedMlpPolicy
    history: tuple[MlpBaselineEpochResult, ...]
    validation_row: BaselineEvaluationRow


class FlattenedMlpPolicy(nn.Module):
    """A one-hidden-layer policy baseline over flattened board tensors."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int] = (
            CHANNEL_COUNT,
            SUPPORTED_BOARD_SIZE,
            SUPPORTED_BOARD_SIZE,
        ),
        hidden_size: int = 128,
        output_size: int = SUPPORTED_LABEL_COUNT,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            msg = f"hidden_size must be positive, got {hidden_size}"
            raise MlpBaselineTrainingError(msg)
        if output_size < 1:
            msg = f"output_size must be positive, got {output_size}"
            raise MlpBaselineTrainingError(msg)

        input_size = math.prod(input_shape)
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(x))


def train_mlp_baseline(
    dataset_path: Path,
    manifest_path: Path,
    *,
    config: MlpBaselineConfig | None = None,
) -> MlpBaselineTrainingResult:
    """Train and validate the flattened-board MLP baseline."""

    resolved_config = MlpBaselineConfig() if config is None else config
    _validate_config(resolved_config)
    torch.manual_seed(resolved_config.seed)

    try:
        train_generator = torch.Generator()
        train_generator.manual_seed(resolved_config.seed)
        dataloaders = load_policy_dataloaders(
            dataset_path,
            manifest_path,
            resolved_config.batch_size,
            splits=("train", "validation"),
            shuffle_train=True,
            train_generator=train_generator,
        )
    except (DatasetSplitError, ProcessedDatasetError, ValueError) as exc:
        raise MlpBaselineTrainingError(str(exc)) from exc

    model = FlattenedMlpPolicy(hidden_size=resolved_config.hidden_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )
    train_loader = dataloaders["train"]
    validation_loader = dataloaders["validation"]
    history: list[MlpBaselineEpochResult] = []

    for epoch in range(1, resolved_config.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer)
        validation_metrics = evaluate_mlp_policy(
            model,
            validation_loader,
            mask_topk=resolved_config.mask_topk,
        )
        history.append(
            MlpBaselineEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation_metrics=validation_metrics,
            )
        )

    final_metrics = history[-1].validation_metrics
    return MlpBaselineTrainingResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        config=resolved_config,
        model=model,
        history=tuple(history),
        validation_row=BaselineEvaluationRow(
            baseline="mlp_flattened",
            split="validation",
            metrics=final_metrics,
        ),
    )


def evaluate_mlp_policy(
    model: nn.Module,
    dataloader: DataLoader[PolicyBatch],
    *,
    mask_topk: bool = False,
) -> PolicyMetricSummary:
    """Evaluate an MLP-like policy model with shared policy metrics."""

    accumulator = PolicyMetricAccumulator()
    model_was_training = model.training
    model.eval()
    device = _model_device(model)
    try:
        with torch.no_grad():
            for batch in dataloader:
                x, labels, legal_mask = _batch_tensors(batch, device=device)
                logits = model(x)
                accumulator.update(
                    scores=logits,
                    labels=labels,
                    legal_mask=legal_mask,
                    apply_legal_mask_before_topk=mask_topk,
                    logits_for_cross_entropy=logits,
                )
        return accumulator.summary()
    except PolicyMetricError as exc:
        raise MlpBaselineTrainingError(str(exc)) from exc
    finally:
        if model_was_training:
            model.train()


def _validate_config(config: MlpBaselineConfig) -> None:
    if config.epochs < 1:
        msg = f"epochs must be positive, got {config.epochs}"
        raise MlpBaselineTrainingError(msg)
    if config.batch_size < 1:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise MlpBaselineTrainingError(msg)
    if config.hidden_size < 1:
        msg = f"hidden_size must be positive, got {config.hidden_size}"
        raise MlpBaselineTrainingError(msg)
    if config.learning_rate <= 0.0:
        msg = f"learning_rate must be positive, got {config.learning_rate}"
        raise MlpBaselineTrainingError(msg)
    if config.weight_decay < 0.0:
        msg = f"weight_decay must be non-negative, got {config.weight_decay}"
        raise MlpBaselineTrainingError(msg)


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[PolicyBatch],
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    device = _model_device(model)
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        x, labels, _legal_mask = _batch_tensors(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        batch_examples = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_examples
        total_examples += batch_examples

    if total_examples == 0:
        msg = "train split contains no examples"
        raise MlpBaselineTrainingError(msg)
    return total_loss / total_examples


def _batch_tensors(
    batch: PolicyBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["x"].to(device=device, dtype=torch.float32),
        batch["y"].to(device=device, dtype=torch.int64),
        batch["legal_mask"].to(device=device, dtype=torch.bool),
    )


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
