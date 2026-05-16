"""CNN policy training and checkpointing."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vantago.data.artifacts import ProcessedDatasetError
from vantago.data.splits import DatasetSplitError
from vantago.data.torch_loading import PolicyBatch, load_policy_dataloaders
from vantago.evaluation import (
    PolicyMetricAccumulator,
    PolicyMetricError,
    PolicyMetricSummary,
)
from vantago.models import CnnPolicyNetwork, CnnPolicyNetworkError

CHECKPOINT_FORMAT_VERSION = 1
MODEL_KIND = "cnn_policy"


class CnnTrainingError(ValueError):
    """Raised when CNN policy training, checkpointing, or loading fails."""


@dataclass(frozen=True, slots=True)
class CnnTrainingConfig:
    """Hyperparameters and output paths for CNN policy training."""

    checkpoint_path: Path
    history_path: Path | None = None
    epochs: int = 5
    batch_size: int = 128
    hidden_channels: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 0
    mask_topk: bool = False


@dataclass(frozen=True, slots=True)
class CnnEpochResult:
    """Training and validation metrics for one CNN epoch."""

    epoch: int
    train_loss: float
    validation_metrics: PolicyMetricSummary
    is_best: bool


@dataclass(frozen=True, slots=True)
class CnnTrainingResult:
    """Completed CNN training run and artifact paths."""

    dataset_path: Path
    manifest_path: Path
    config: CnnTrainingConfig
    model: CnnPolicyNetwork
    history: tuple[CnnEpochResult, ...]
    best_epoch: int
    best_validation_metrics: PolicyMetricSummary
    checkpoint_path: Path
    history_path: Path


@dataclass(frozen=True, slots=True)
class CnnPolicyCheckpoint:
    """Reloaded CNN policy checkpoint."""

    path: Path
    format_version: int
    model_kind: str
    config: CnnTrainingConfig
    model: CnnPolicyNetwork
    history: tuple[CnnEpochResult, ...]
    best_epoch: int
    best_validation_metrics: PolicyMetricSummary


def train_cnn_policy(
    dataset_path: Path,
    manifest_path: Path,
    *,
    config: CnnTrainingConfig,
) -> CnnTrainingResult:
    """Train a CNN policy model, save the best checkpoint, and write history."""

    _validate_config(config)
    history_path = _resolve_history_path(config)
    torch.manual_seed(config.seed)

    try:
        train_generator = torch.Generator()
        train_generator.manual_seed(config.seed)
        dataloaders = load_policy_dataloaders(
            dataset_path,
            manifest_path,
            config.batch_size,
            splits=("train", "validation"),
            shuffle_train=True,
            train_generator=train_generator,
        )
    except (DatasetSplitError, ProcessedDatasetError, ValueError) as exc:
        raise CnnTrainingError(str(exc)) from exc

    try:
        model = CnnPolicyNetwork(hidden_channels=config.hidden_channels)
    except CnnPolicyNetworkError as exc:
        raise CnnTrainingError(str(exc)) from exc

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = dataloaders["train"]
    validation_loader = dataloaders["validation"]
    history: list[CnnEpochResult] = []
    best_metrics: PolicyMetricSummary | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: object | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer)
        validation_metrics = evaluate_cnn_policy(
            model,
            validation_loader,
            mask_topk=config.mask_topk,
        )
        is_best = _is_best_validation(validation_metrics, best_metrics)
        if is_best:
            best_metrics = validation_metrics
            best_epoch = epoch
            best_state = _clone_state_dict(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
        history.append(
            CnnEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation_metrics=validation_metrics,
                is_best=is_best,
            )
        )
        if is_best:
            if best_state is None:
                msg = "best model state was not captured"
                raise CnnTrainingError(msg)
            _write_checkpoint(
                config.checkpoint_path,
                dataset_path=dataset_path,
                manifest_path=manifest_path,
                config=config,
                history=_mark_best_epoch(tuple(history), best_epoch),
                best_epoch=best_epoch,
                model_state_dict=best_state,
                optimizer_state_dict=best_optimizer_state,
            )

    if best_metrics is None or best_state is None:
        msg = "training finished without validation metrics"
        raise CnnTrainingError(msg)

    best_model = CnnPolicyNetwork(hidden_channels=config.hidden_channels)
    best_model.load_state_dict(best_state)
    best_model.eval()
    resolved_history = _mark_best_epoch(tuple(history), best_epoch)
    _write_checkpoint(
        config.checkpoint_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        config=config,
        history=resolved_history,
        best_epoch=best_epoch,
        model_state_dict=best_state,
        optimizer_state_dict=best_optimizer_state,
    )
    _write_history(
        history_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        checkpoint_path=config.checkpoint_path,
        config=config,
        history=resolved_history,
        best_epoch=best_epoch,
    )

    return CnnTrainingResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        config=config,
        model=best_model,
        history=resolved_history,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        checkpoint_path=config.checkpoint_path,
        history_path=history_path,
    )


def load_cnn_policy_checkpoint(path: Path) -> CnnPolicyCheckpoint:
    """Load a CNN policy checkpoint on CPU and rebuild its model."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        msg = f"could not load CNN policy checkpoint {path}: {exc}"
        raise CnnTrainingError(msg) from exc

    mapping = _require_mapping(payload, "checkpoint")
    format_version = _require_int(mapping, "format_version")
    if format_version != CHECKPOINT_FORMAT_VERSION:
        msg = (
            "unsupported CNN policy checkpoint format version: "
            f"{format_version}"
        )
        raise CnnTrainingError(msg)
    model_kind = _require_string(mapping, "model_kind")
    if model_kind != MODEL_KIND:
        msg = f"checkpoint model_kind must be {MODEL_KIND!r}, got {model_kind!r}"
        raise CnnTrainingError(msg)

    config = _config_from_json_mapping(_require_child_mapping(mapping, "config"))
    history = tuple(
        _epoch_from_json_mapping(item)
        for item in _require_sequence(mapping, "history")
    )
    best_epoch = _require_int(mapping, "best_epoch")
    best_metrics = _metric_summary_from_json_mapping(
        _require_child_mapping(mapping, "best_validation_metrics")
    )
    model = CnnPolicyNetwork(hidden_channels=config.hidden_channels)
    state_dict = cast(
        dict[str, torch.Tensor],
        _require_child_mapping(mapping, "model_state_dict"),
    )
    model.load_state_dict(state_dict)
    model.eval()

    return CnnPolicyCheckpoint(
        path=path,
        format_version=format_version,
        model_kind=model_kind,
        config=config,
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
    )


def evaluate_cnn_policy(
    model: nn.Module,
    dataloader: DataLoader[PolicyBatch],
    *,
    mask_topk: bool = False,
) -> PolicyMetricSummary:
    """Evaluate a CNN-like policy model with shared policy metrics."""

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
        raise CnnTrainingError(str(exc)) from exc
    finally:
        if model_was_training:
            model.train()


def _validate_config(config: CnnTrainingConfig) -> None:
    if config.epochs < 1:
        msg = f"epochs must be positive, got {config.epochs}"
        raise CnnTrainingError(msg)
    if config.batch_size < 1:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise CnnTrainingError(msg)
    if config.hidden_channels < 1:
        msg = f"hidden_channels must be positive, got {config.hidden_channels}"
        raise CnnTrainingError(msg)
    if config.learning_rate <= 0.0:
        msg = f"learning_rate must be positive, got {config.learning_rate}"
        raise CnnTrainingError(msg)
    if config.weight_decay < 0.0:
        msg = f"weight_decay must be non-negative, got {config.weight_decay}"
        raise CnnTrainingError(msg)
    if config.history_path is not None and config.history_path.suffix != ".json":
        msg = f"history_path must end with .json: {config.history_path}"
        raise CnnTrainingError(msg)
    history_path = _resolve_history_path(config)
    if config.checkpoint_path.resolve() == history_path.resolve():
        msg = (
            "checkpoint_path and history_path must be different: "
            f"{config.checkpoint_path}"
        )
        raise CnnTrainingError(msg)


def _resolve_history_path(config: CnnTrainingConfig) -> Path:
    if config.history_path is not None:
        return config.history_path
    return config.checkpoint_path.with_suffix(".history.json")


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
        raise CnnTrainingError(msg)
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


def _is_best_validation(
    metrics: PolicyMetricSummary,
    best_metrics: PolicyMetricSummary | None,
) -> bool:
    if metrics.cross_entropy is None:
        msg = "validation cross_entropy is required for checkpoint selection"
        raise CnnTrainingError(msg)
    if best_metrics is None:
        return True
    if best_metrics.cross_entropy is None:
        return True
    return metrics.cross_entropy < best_metrics.cross_entropy


def _mark_best_epoch(
    history: tuple[CnnEpochResult, ...],
    best_epoch: int,
) -> tuple[CnnEpochResult, ...]:
    return tuple(
        CnnEpochResult(
            epoch=epoch.epoch,
            train_loss=epoch.train_loss,
            validation_metrics=epoch.validation_metrics,
            is_best=epoch.epoch == best_epoch,
        )
        for epoch in history
    )


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _write_checkpoint(
    path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    config: CnnTrainingConfig,
    history: tuple[CnnEpochResult, ...],
    best_epoch: int,
    model_state_dict: dict[str, torch.Tensor],
    optimizer_state_dict: object,
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_kind": MODEL_KIND,
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "config": _config_to_json_data(config),
        "best_epoch": best_epoch,
        "best_validation_metrics": _metric_summary_to_json_data(
            history[best_epoch - 1].validation_metrics
        ),
        "history": [_epoch_to_json_data(epoch) for epoch in history],
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    except OSError as exc:
        msg = f"could not write CNN policy checkpoint {path}: {exc}"
        raise CnnTrainingError(msg) from exc


def _write_history(
    path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    config: CnnTrainingConfig,
    history: tuple[CnnEpochResult, ...],
    best_epoch: int,
) -> None:
    data = {
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "config": _config_to_json_data(config),
        "history": [_epoch_to_json_data(epoch) for epoch in history],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        msg = f"could not write CNN training history {path}: {exc}"
        raise CnnTrainingError(msg) from exc


def _config_to_json_data(config: CnnTrainingConfig) -> dict[str, object]:
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


def _epoch_to_json_data(epoch: CnnEpochResult) -> dict[str, object]:
    return {
        "epoch": epoch.epoch,
        "train_loss": epoch.train_loss,
        "validation_metrics": _metric_summary_to_json_data(
            epoch.validation_metrics
        ),
        "is_best": epoch.is_best,
    }


def _metric_summary_to_json_data(
    metrics: PolicyMetricSummary,
) -> dict[str, object]:
    return {
        "example_count": metrics.example_count,
        "top_1": metrics.top_1,
        "top_3": metrics.top_3,
        "top_5": metrics.top_5,
        "cross_entropy": metrics.cross_entropy,
        "illegal_move_rate": metrics.illegal_move_rate,
    }


def _config_from_json_mapping(mapping: dict[str, object]) -> CnnTrainingConfig:
    history_path = mapping.get("history_path")
    if history_path is not None and not isinstance(history_path, str):
        msg = "config.history_path must be a string or null"
        raise CnnTrainingError(msg)
    return CnnTrainingConfig(
        checkpoint_path=Path(_require_string(mapping, "checkpoint_path")),
        history_path=None if history_path is None else Path(history_path),
        epochs=_require_int(mapping, "epochs"),
        batch_size=_require_int(mapping, "batch_size"),
        hidden_channels=_require_int(mapping, "hidden_channels"),
        learning_rate=_require_float(mapping, "learning_rate"),
        weight_decay=_require_float(mapping, "weight_decay"),
        seed=_require_int(mapping, "seed"),
        mask_topk=_require_bool(mapping, "mask_topk"),
    )


def _epoch_from_json_mapping(item: object) -> CnnEpochResult:
    mapping = _require_mapping(item, "history item")
    return CnnEpochResult(
        epoch=_require_int(mapping, "epoch"),
        train_loss=_require_float(mapping, "train_loss"),
        validation_metrics=_metric_summary_from_json_mapping(
            _require_child_mapping(mapping, "validation_metrics")
        ),
        is_best=_require_bool(mapping, "is_best"),
    )


def _metric_summary_from_json_mapping(
    mapping: dict[str, object],
) -> PolicyMetricSummary:
    cross_entropy = mapping.get("cross_entropy")
    if cross_entropy is not None and not isinstance(cross_entropy, int | float):
        msg = "cross_entropy must be numeric or null"
        raise CnnTrainingError(msg)
    return PolicyMetricSummary(
        example_count=_require_int(mapping, "example_count"),
        top_1=_require_float(mapping, "top_1"),
        top_3=_require_float(mapping, "top_3"),
        top_5=_require_float(mapping, "top_5"),
        cross_entropy=None if cross_entropy is None else float(cross_entropy),
        illegal_move_rate=_require_float(mapping, "illegal_move_rate"),
    )


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{name} must be an object"
        raise CnnTrainingError(msg)
    return cast(dict[str, object], value)


def _require_child_mapping(
    mapping: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        msg = f"{key} must be an object"
        raise CnnTrainingError(msg)
    return cast(dict[str, object], value)


def _require_sequence(mapping: dict[str, object], key: str) -> list[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        msg = f"{key} must be a list"
        raise CnnTrainingError(msg)
    return value


def _require_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise CnnTrainingError(msg)
    return value


def _require_int(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise CnnTrainingError(msg)
    return value


def _require_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"{key} must be numeric"
        raise CnnTrainingError(msg)
    return float(value)


def _require_bool(mapping: dict[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise CnnTrainingError(msg)
    return value


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
