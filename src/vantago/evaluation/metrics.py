"""Reusable metrics for supervised policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class PolicyMetricError(ValueError):
    """Raised when policy metric inputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class PolicyMetricSummary:
    """Averaged policy metrics over one or more evaluated batches."""

    example_count: int
    top_1: float
    top_3: float
    top_5: float
    cross_entropy: float | None
    illegal_move_rate: float


@dataclass(frozen=True, slots=True)
class _PolicyMetricTotals:
    example_count: int
    top_1_correct: int
    top_3_correct: int
    top_5_correct: int
    illegal_top_1: int
    cross_entropy_sum: float | None
    cross_entropy_count: int


class PolicyMetricAccumulator:
    """Accumulate policy metrics across multiple batches."""

    def __init__(self) -> None:
        self._example_count = 0
        self._top_1_correct = 0
        self._top_3_correct = 0
        self._top_5_correct = 0
        self._illegal_top_1 = 0
        self._cross_entropy_sum = 0.0
        self._cross_entropy_count = 0
        self._has_cross_entropy: bool | None = None

    def update(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        legal_mask: torch.Tensor,
        *,
        apply_legal_mask_before_topk: bool = False,
        logits_for_cross_entropy: torch.Tensor | None = None,
    ) -> None:
        """Add one batch of policy scores to the running totals."""

        has_cross_entropy = logits_for_cross_entropy is not None
        self._validate_cross_entropy_mode(has_cross_entropy)
        totals = _compute_policy_metric_totals(
            scores=scores,
            labels=labels,
            legal_mask=legal_mask,
            apply_legal_mask_before_topk=apply_legal_mask_before_topk,
            logits_for_cross_entropy=logits_for_cross_entropy,
        )
        if self._has_cross_entropy is None:
            self._has_cross_entropy = has_cross_entropy

        self._example_count += totals.example_count
        self._top_1_correct += totals.top_1_correct
        self._top_3_correct += totals.top_3_correct
        self._top_5_correct += totals.top_5_correct
        self._illegal_top_1 += totals.illegal_top_1
        if totals.cross_entropy_sum is not None:
            self._cross_entropy_sum += totals.cross_entropy_sum
            self._cross_entropy_count += totals.cross_entropy_count

    def summary(self) -> PolicyMetricSummary:
        """Return averaged metrics for all accumulated examples."""

        if self._example_count == 0:
            msg = "cannot summarize policy metrics without examples"
            raise PolicyMetricError(msg)

        return _summarize_totals(
            _PolicyMetricTotals(
                example_count=self._example_count,
                top_1_correct=self._top_1_correct,
                top_3_correct=self._top_3_correct,
                top_5_correct=self._top_5_correct,
                illegal_top_1=self._illegal_top_1,
                cross_entropy_sum=(
                    self._cross_entropy_sum
                    if self._cross_entropy_count > 0
                    else None
                ),
                cross_entropy_count=self._cross_entropy_count,
            )
        )

    def _validate_cross_entropy_mode(self, has_cross_entropy: bool) -> None:
        if (
            self._has_cross_entropy is not None
            and self._has_cross_entropy != has_cross_entropy
        ):
            msg = (
                "cannot mix accumulator updates with and without "
                "cross-entropy logits"
            )
            raise PolicyMetricError(msg)


def apply_legal_mask(
    scores: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Return scores with illegal move locations set to negative infinity."""

    _validate_scores(scores, name="scores")
    resolved_legal_mask = _validate_legal_mask(
        legal_mask=legal_mask,
        expected_shape=scores.shape,
        device=scores.device,
    )
    _validate_each_row_has_legal_move(resolved_legal_mask)
    return scores.masked_fill(~resolved_legal_mask, float("-inf"))


def compute_policy_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    apply_legal_mask_before_topk: bool = False,
    logits_for_cross_entropy: torch.Tensor | None = None,
) -> PolicyMetricSummary:
    """Compute top-k, optional cross-entropy, and raw illegal top-1 metrics."""

    totals = _compute_policy_metric_totals(
        scores=scores,
        labels=labels,
        legal_mask=legal_mask,
        apply_legal_mask_before_topk=apply_legal_mask_before_topk,
        logits_for_cross_entropy=logits_for_cross_entropy,
    )
    return _summarize_totals(totals)


def compute_policy_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    apply_legal_mask_before_topk: bool = False,
) -> PolicyMetricSummary:
    """Compute policy metrics when the ranking scores are model logits."""

    return compute_policy_metrics(
        logits,
        labels,
        legal_mask,
        apply_legal_mask_before_topk=apply_legal_mask_before_topk,
        logits_for_cross_entropy=logits,
    )


def _compute_policy_metric_totals(
    *,
    scores: torch.Tensor,
    labels: torch.Tensor,
    legal_mask: torch.Tensor,
    apply_legal_mask_before_topk: bool,
    logits_for_cross_entropy: torch.Tensor | None,
) -> _PolicyMetricTotals:
    _validate_scores(scores, name="scores")
    resolved_labels = _validate_labels(
        labels=labels,
        example_count=int(scores.shape[0]),
        class_count=int(scores.shape[1]),
        device=scores.device,
    )
    resolved_legal_mask = _validate_legal_mask(
        legal_mask=legal_mask,
        expected_shape=scores.shape,
        device=scores.device,
    )
    _validate_each_row_has_legal_move(resolved_legal_mask)
    _validate_target_labels_are_legal(resolved_labels, resolved_legal_mask)

    ranking_scores = (
        apply_legal_mask(scores, resolved_legal_mask)
        if apply_legal_mask_before_topk
        else scores
    )
    with torch.no_grad():
        top_k = min(5, int(ranking_scores.shape[1]))
        top_k_indices = _stable_topk_indices(ranking_scores, top_k)
        label_matches = top_k_indices.eq(resolved_labels.unsqueeze(1))
        top_1_correct = _count_matches(label_matches, 1)
        top_3_correct = _count_matches(label_matches, min(3, top_k))
        top_5_correct = _count_matches(label_matches, top_k)
        raw_top_1_predictions = _stable_topk_indices(scores, 1).squeeze(1)
        predicted_legal = resolved_legal_mask.gather(
            dim=1,
            index=raw_top_1_predictions.unsqueeze(1),
        ).squeeze(1)
        illegal_top_1 = int((~predicted_legal).sum().item())

        cross_entropy_sum: float | None = None
        cross_entropy_count = 0
        if logits_for_cross_entropy is not None:
            _validate_scores(
                logits_for_cross_entropy,
                name="logits_for_cross_entropy",
            )
            if logits_for_cross_entropy.shape != scores.shape:
                msg = (
                    "logits_for_cross_entropy shape must match scores shape, "
                    f"got {logits_for_cross_entropy.shape} and {scores.shape}"
                )
                raise PolicyMetricError(msg)
            resolved_logits = logits_for_cross_entropy.to(device=scores.device)
            cross_entropy_sum = float(
                F.cross_entropy(
                    resolved_logits,
                    resolved_labels,
                    reduction="sum",
                ).item()
            )
            cross_entropy_count = int(scores.shape[0])

    return _PolicyMetricTotals(
        example_count=int(scores.shape[0]),
        top_1_correct=top_1_correct,
        top_3_correct=top_3_correct,
        top_5_correct=top_5_correct,
        illegal_top_1=illegal_top_1,
        cross_entropy_sum=cross_entropy_sum,
        cross_entropy_count=cross_entropy_count,
    )


def _summarize_totals(totals: _PolicyMetricTotals) -> PolicyMetricSummary:
    if totals.example_count == 0:
        msg = "cannot compute policy metrics without examples"
        raise PolicyMetricError(msg)

    cross_entropy = (
        totals.cross_entropy_sum / totals.cross_entropy_count
        if totals.cross_entropy_sum is not None and totals.cross_entropy_count > 0
        else None
    )
    return PolicyMetricSummary(
        example_count=totals.example_count,
        top_1=totals.top_1_correct / totals.example_count,
        top_3=totals.top_3_correct / totals.example_count,
        top_5=totals.top_5_correct / totals.example_count,
        cross_entropy=cross_entropy,
        illegal_move_rate=totals.illegal_top_1 / totals.example_count,
    )


def _count_matches(label_matches: torch.Tensor, top_k: int) -> int:
    return int(label_matches[:, :top_k].any(dim=1).sum().item())


def _stable_topk_indices(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]


def _validate_scores(scores: torch.Tensor, *, name: str) -> None:
    if scores.ndim != 2:
        msg = f"{name} must have shape [N, C], got {tuple(scores.shape)}"
        raise PolicyMetricError(msg)
    if int(scores.shape[0]) == 0:
        msg = f"{name} must contain at least one example"
        raise PolicyMetricError(msg)
    if int(scores.shape[1]) == 0:
        msg = f"{name} must contain at least one class"
        raise PolicyMetricError(msg)
    if not torch.is_floating_point(scores):
        msg = f"{name} must be a floating point tensor, got {scores.dtype}"
        raise PolicyMetricError(msg)


def _validate_labels(
    *,
    labels: torch.Tensor,
    example_count: int,
    class_count: int,
    device: torch.device,
) -> torch.Tensor:
    if labels.shape != (example_count,):
        msg = f"labels must have shape ({example_count},), got {tuple(labels.shape)}"
        raise PolicyMetricError(msg)
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        msg = f"labels must be an integer tensor, got {labels.dtype}"
        raise PolicyMetricError(msg)

    resolved_labels = labels.to(device=device, dtype=torch.int64)
    if bool(((resolved_labels < 0) | (resolved_labels >= class_count)).any().item()):
        msg = f"labels must be in [0, {class_count})"
        raise PolicyMetricError(msg)
    return resolved_labels


def _validate_legal_mask(
    *,
    legal_mask: torch.Tensor,
    expected_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if legal_mask.shape != expected_shape:
        msg = (
            f"legal_mask must have shape {tuple(expected_shape)}, "
            f"got {tuple(legal_mask.shape)}"
        )
        raise PolicyMetricError(msg)
    if legal_mask.dtype != torch.bool:
        msg = f"legal_mask must be a bool tensor, got {legal_mask.dtype}"
        raise PolicyMetricError(msg)
    return legal_mask.to(device=device)


def _validate_each_row_has_legal_move(legal_mask: torch.Tensor) -> None:
    if not bool(legal_mask.any(dim=1).all().item()):
        msg = "legal_mask must contain at least one legal move per example"
        raise PolicyMetricError(msg)


def _validate_target_labels_are_legal(
    labels: torch.Tensor,
    legal_mask: torch.Tensor,
) -> None:
    target_legal = legal_mask.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    if not bool(target_legal.all().item()):
        msg = "legal_mask must mark every target label as legal"
        raise PolicyMetricError(msg)
