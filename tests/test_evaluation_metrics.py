from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vantago.evaluation import (
    PolicyMetricAccumulator,
    PolicyMetricError,
    apply_legal_mask,
    compute_policy_metrics,
    compute_policy_metrics_from_logits,
)


def test_compute_policy_metrics_from_logits_reports_topk_and_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    legal_mask = torch.ones_like(logits, dtype=torch.bool)

    summary = compute_policy_metrics_from_logits(logits, labels, legal_mask)

    assert summary.example_count == 4
    assert summary.top_1 == pytest.approx(0.25)
    assert summary.top_3 == pytest.approx(0.50)
    assert summary.top_5 == pytest.approx(0.75)
    assert summary.cross_entropy == pytest.approx(
        float(F.cross_entropy(logits, labels).item())
    )
    assert summary.illegal_move_rate == pytest.approx(0.0)


def test_apply_legal_mask_changes_ranked_predictions_when_requested() -> None:
    scores = torch.tensor([[10.0, 9.0, 8.0, 7.0]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    legal_mask = torch.tensor([[False, True, True, True]], dtype=torch.bool)

    masked_scores = apply_legal_mask(scores, legal_mask)
    unmasked = compute_policy_metrics(scores, labels, legal_mask)
    masked = compute_policy_metrics(
        scores,
        labels,
        legal_mask,
        apply_legal_mask_before_topk=True,
    )

    assert torch.isneginf(masked_scores[0, 0])
    assert masked_scores[0, 1].item() == pytest.approx(9.0)
    assert unmasked.top_1 == pytest.approx(0.0)
    assert unmasked.illegal_move_rate == pytest.approx(1.0)
    assert masked.top_1 == pytest.approx(1.0)
    assert masked.illegal_move_rate == pytest.approx(1.0)


def test_illegal_move_rate_catches_unmasked_illegal_top1_predictions() -> None:
    scores = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    legal_mask = torch.tensor(
        [
            [False, True, True],
            [True, False, True],
        ],
        dtype=torch.bool,
    )

    summary = compute_policy_metrics(scores, labels, legal_mask)

    assert summary.example_count == 2
    assert summary.illegal_move_rate == pytest.approx(1.0)


def test_policy_metric_accumulator_matches_concatenated_batches() -> None:
    first_scores = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 5.0, 1.0],
        ],
        dtype=torch.float32,
    )
    first_labels = torch.tensor([0, 2], dtype=torch.int64)
    first_legal_mask = torch.ones_like(first_scores, dtype=torch.bool)
    second_scores = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    second_labels = torch.tensor([2], dtype=torch.int64)
    second_legal_mask = torch.ones_like(second_scores, dtype=torch.bool)

    accumulator = PolicyMetricAccumulator()
    accumulator.update(
        first_scores,
        first_labels,
        first_legal_mask,
        logits_for_cross_entropy=first_scores,
    )
    accumulator.update(
        second_scores,
        second_labels,
        second_legal_mask,
        logits_for_cross_entropy=second_scores,
    )

    combined = compute_policy_metrics(
        torch.cat([first_scores, second_scores]),
        torch.cat([first_labels, second_labels]),
        torch.cat([first_legal_mask, second_legal_mask]),
        logits_for_cross_entropy=torch.cat([first_scores, second_scores]),
    )

    accumulated = accumulator.summary()
    assert accumulated.example_count == combined.example_count
    assert accumulated.top_1 == pytest.approx(combined.top_1)
    assert accumulated.top_3 == pytest.approx(combined.top_3)
    assert accumulated.top_5 == pytest.approx(combined.top_5)
    assert accumulated.cross_entropy == pytest.approx(combined.cross_entropy)
    assert accumulated.illegal_move_rate == pytest.approx(combined.illegal_move_rate)


def test_policy_metric_accumulator_rejects_mixed_cross_entropy_updates() -> None:
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    legal_mask = torch.ones_like(scores, dtype=torch.bool)
    accumulator = PolicyMetricAccumulator()
    accumulator.update(scores, labels, legal_mask, logits_for_cross_entropy=scores)

    with pytest.raises(PolicyMetricError, match="cannot mix"):
        accumulator.update(scores, labels, legal_mask)


def test_policy_metrics_rank_tied_scores_by_lower_label_first() -> None:
    scores = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.int64)
    legal_mask = torch.ones_like(scores, dtype=torch.bool)

    summary = compute_policy_metrics(scores, labels, legal_mask)

    assert summary.top_1 == pytest.approx(0.5)
    assert summary.top_3 == pytest.approx(1.0)


def test_policy_metrics_reject_target_labels_marked_illegal() -> None:
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    legal_mask = torch.tensor([[True, False]], dtype=torch.bool)

    with pytest.raises(PolicyMetricError, match="target label"):
        compute_policy_metrics(scores, labels, legal_mask)


def test_policy_metrics_reject_empty_batches() -> None:
    scores = torch.empty((0, 3), dtype=torch.float32)
    labels = torch.empty((0,), dtype=torch.int64)
    legal_mask = torch.empty((0, 3), dtype=torch.bool)

    with pytest.raises(PolicyMetricError, match="at least one example"):
        compute_policy_metrics(scores, labels, legal_mask)


def test_policy_metrics_reject_all_false_legal_mask_rows() -> None:
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    legal_mask = torch.tensor([[False, False]], dtype=torch.bool)

    with pytest.raises(PolicyMetricError, match="at least one legal move"):
        compute_policy_metrics(scores, labels, legal_mask)


def test_policy_metrics_reject_bad_shapes_and_dtypes() -> None:
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    legal_mask = torch.ones_like(scores, dtype=torch.bool)

    with pytest.raises(PolicyMetricError, match=r"shape \(1,\)"):
        compute_policy_metrics(scores, labels.reshape(1, 1), legal_mask)

    with pytest.raises(PolicyMetricError, match="integer tensor"):
        compute_policy_metrics(scores, labels.to(dtype=torch.float32), legal_mask)

    with pytest.raises(PolicyMetricError, match="bool tensor"):
        compute_policy_metrics(scores, labels, legal_mask.to(dtype=torch.float32))


def test_policy_metrics_reject_out_of_range_labels() -> None:
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([2], dtype=torch.int64)
    legal_mask = torch.ones_like(scores, dtype=torch.bool)

    with pytest.raises(PolicyMetricError, match=r"\[0, 2\)"):
        compute_policy_metrics(scores, labels, legal_mask)
