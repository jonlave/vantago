"""Qualitative mistake analysis for saved policy checkpoints."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import torch
import torch.nn as nn

from vantago.baselines import (
    PHASE_NAMES,
    GamePhase,
    game_phase_for_move_number,
)
from vantago.data.artifacts import ProcessedDatasetError
from vantago.data.encoding import SUPPORTED_LABEL_COUNT
from vantago.data.splits import DatasetSplitError
from vantago.data.torch_loading import PolicyBatch, load_policy_dataloaders
from vantago.evaluation import PolicyMetricError, apply_legal_mask
from vantago.qualitative_constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EXAMPLES_PER_PHASE,
    DEFAULT_QUALITATIVE_SPLIT,
    DEFAULT_TOP_K,
    QUALITATIVE_ANALYSIS_SPLITS,
)
from vantago.replay import SUPPORTED_BOARD_SIZE
from vantago.training import (
    CnnPolicyCheckpoint,
    CnnTrainingError,
    load_cnn_policy_checkpoint,
)

Heatmap: TypeAlias = tuple[tuple[int, ...], ...]


class QualitativeAnalysisError(ValueError):
    """Raised when qualitative analysis cannot be generated."""


@dataclass(frozen=True, slots=True)
class QualitativeAnalysisConfig:
    """Runtime options for qualitative mistake analysis."""

    split: str = DEFAULT_QUALITATIVE_SPLIT
    batch_size: int = DEFAULT_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    examples_per_phase: int = DEFAULT_EXAMPLES_PER_PHASE
    mask_topk: bool = False


@dataclass(frozen=True, slots=True)
class QualitativePrediction:
    """One ranked model prediction for a position."""

    rank: int
    label: int
    row: int
    col: int
    probability: float
    is_legal: bool
    is_human_move: bool


@dataclass(frozen=True, slots=True)
class QualitativeExample:
    """Inspectable per-position evidence for one qualitative example."""

    game_id: str
    source_name: str
    move_number: int
    phase: GamePhase
    human_label: int
    human_row: int
    human_col: int
    target_rank: int
    top_predictions: tuple[QualitativePrediction, ...]
    raw_top1_label: int
    raw_top1_row: int
    raw_top1_col: int
    raw_top1_probability: float
    raw_top1_is_legal: bool
    category: str
    category_basis: str
    board: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualitativeAnalysisReport:
    """Qualitative analysis report for one checkpoint and split."""

    dataset_path: Path
    manifest_path: Path
    checkpoint_path: Path
    checkpoint_dataset_path: Path
    checkpoint_manifest_path: Path
    split: str
    batch_size: int
    top_k: int
    examples_per_phase: int
    mask_topk: bool
    human_move_heatmap: Heatmap
    model_top1_heatmap: Heatmap
    phase_examples: dict[GamePhase, tuple[QualitativeExample, ...]]


@dataclass(frozen=True, slots=True)
class _QualitativeCandidate:
    game_id: str
    source_name: str
    move_number: int
    phase: GamePhase
    human_label: int
    human_row: int
    human_col: int
    target_rank: int
    top_predictions: tuple[QualitativePrediction, ...]
    raw_top1_label: int
    raw_top1_row: int
    raw_top1_col: int
    raw_top1_probability: float
    raw_top1_is_legal: bool
    category: str
    category_basis: str
    board_tensor: torch.Tensor


@dataclass(slots=True)
class _PhaseCandidatePools:
    high_confidence_misses: list[_QualitativeCandidate]
    top5_near_misses: list[_QualitativeCandidate]
    fallback: list[_QualitativeCandidate]


def generate_qualitative_analysis_report(
    dataset_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    config: QualitativeAnalysisConfig | None = None,
) -> QualitativeAnalysisReport:
    """Evaluate a checkpoint and collect qualitative mistake-analysis views."""

    resolved_config = QualitativeAnalysisConfig() if config is None else config
    _validate_config(resolved_config)

    try:
        checkpoint = load_cnn_policy_checkpoint(checkpoint_path)
        _validate_checkpoint_provenance(checkpoint, dataset_path, manifest_path)
        dataloaders = load_policy_dataloaders(
            dataset_path,
            manifest_path,
            resolved_config.batch_size,
            splits=(resolved_config.split,),
            shuffle_train=False,
        )
        human_heatmap, model_heatmap, phase_examples = _collect_examples(
            checkpoint.model,
            dataloaders[resolved_config.split],
            config=resolved_config,
        )
    except (
        CnnTrainingError,
        DatasetSplitError,
        ProcessedDatasetError,
        PolicyMetricError,
        ValueError,
    ) as exc:
        raise QualitativeAnalysisError(str(exc)) from exc

    return QualitativeAnalysisReport(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        checkpoint_dataset_path=checkpoint.dataset_path,
        checkpoint_manifest_path=checkpoint.manifest_path,
        split=resolved_config.split,
        batch_size=resolved_config.batch_size,
        top_k=resolved_config.top_k,
        examples_per_phase=resolved_config.examples_per_phase,
        mask_topk=resolved_config.mask_topk,
        human_move_heatmap=_freeze_heatmap(human_heatmap),
        model_top1_heatmap=_freeze_heatmap(model_heatmap),
        phase_examples=phase_examples,
    )


def qualitative_analysis_report_to_json_data(
    report: QualitativeAnalysisReport,
) -> dict[str, object]:
    """Return deterministic JSON-ready data for a qualitative analysis report."""

    return {
        "dataset_path": str(report.dataset_path),
        "manifest_path": str(report.manifest_path),
        "checkpoint_path": str(report.checkpoint_path),
        "checkpoint": {
            "dataset_path": str(report.checkpoint_dataset_path),
            "manifest_path": str(report.checkpoint_manifest_path),
        },
        "split": report.split,
        "batch_size": report.batch_size,
        "top_k": report.top_k,
        "examples_per_phase": report.examples_per_phase,
        "mask_topk": report.mask_topk,
        "heatmaps": {
            "human_moves": _heatmap_to_json_data(report.human_move_heatmap),
            "model_top1": _heatmap_to_json_data(report.model_top1_heatmap),
        },
        "phase_examples": {
            phase: [
                _example_to_json_data(example)
                for example in report.phase_examples[phase]
            ]
            for phase in PHASE_NAMES
        },
    }


def write_qualitative_analysis_report_json(
    path: Path,
    report: QualitativeAnalysisReport,
) -> None:
    """Write deterministic JSON for a qualitative analysis report."""

    validate_qualitative_analysis_json_output_path(
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
                qualitative_analysis_report_to_json_data(report),
                temp_file,
                indent=2,
            )
            temp_file.write("\n")
        temp_path.replace(path)
    except OSError as exc:
        msg = f"could not write qualitative analysis JSON {path}: {exc}"
        raise QualitativeAnalysisError(msg) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def validate_qualitative_analysis_json_output_path(
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
            raise QualitativeAnalysisError(msg)

    if path.suffix != ".json":
        msg = f"json_out must end with .json: {path}"
        raise QualitativeAnalysisError(msg)
    if path.exists() and path.is_dir():
        msg = f"json_out path is a directory: {path}"
        raise QualitativeAnalysisError(msg)


def _collect_examples(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader[PolicyBatch],
    *,
    config: QualitativeAnalysisConfig,
) -> tuple[
    list[list[int]],
    list[list[int]],
    dict[GamePhase, tuple[QualitativeExample, ...]],
]:
    human_heatmap = _empty_heatmap()
    model_heatmap = _empty_heatmap()
    candidate_pools = _candidate_pools()
    candidate_capacity = _candidate_pool_capacity(config.examples_per_phase)
    example_count = 0
    model_was_training = model.training
    device = _model_device(model)
    model.eval()

    try:
        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(device=device, dtype=torch.float32)
                labels = batch["y"].to(device=device, dtype=torch.int64)
                legal_mask = batch["legal_mask"].to(device=device, dtype=torch.bool)
                logits = model(x)
                _validate_logits_shape(logits, example_count=int(labels.shape[0]))

                probabilities = torch.softmax(logits, dim=1)
                ranking_scores = (
                    apply_legal_mask(logits, legal_mask)
                    if config.mask_topk
                    else logits
                )
                ranked_labels = torch.argsort(
                    ranking_scores,
                    dim=1,
                    descending=True,
                    stable=True,
                )
                raw_top1_labels = torch.argsort(
                    logits,
                    dim=1,
                    descending=True,
                    stable=True,
                )[:, 0]

                batch_candidates = _batch_candidates(
                    batch,
                    x=x.detach().cpu(),
                    labels=labels,
                    legal_mask=legal_mask,
                    probabilities=probabilities,
                    ranked_labels=ranked_labels,
                    raw_top1_labels=raw_top1_labels,
                    top_k=config.top_k,
                )
                for candidate in batch_candidates:
                    example_count += 1
                    _increment_heatmap(human_heatmap, candidate.human_label)
                    _increment_heatmap(
                        model_heatmap,
                        candidate.top_predictions[0].label,
                    )
                    _add_candidate(
                        candidate_pools[candidate.phase],
                        candidate,
                        capacity=candidate_capacity,
                    )
    finally:
        if model_was_training:
            model.train()

    if example_count == 0:
        msg = f"{config.split} split contains no examples"
        raise QualitativeAnalysisError(msg)
    phase_examples = {
        phase: _select_examples_for_phase(
            candidate_pools[phase],
            limit=config.examples_per_phase,
        )
        for phase in PHASE_NAMES
    }
    return human_heatmap, model_heatmap, phase_examples


def _batch_candidates(
    batch: PolicyBatch,
    *,
    x: torch.Tensor,
    labels: torch.Tensor,
    legal_mask: torch.Tensor,
    probabilities: torch.Tensor,
    ranked_labels: torch.Tensor,
    raw_top1_labels: torch.Tensor,
    top_k: int,
) -> tuple[_QualitativeCandidate, ...]:
    candidates: list[_QualitativeCandidate] = []
    resolved_top_k = min(top_k, int(ranked_labels.shape[1]))
    for index, game_id in enumerate(batch["game_id"]):
        human_label = int(labels[index].item())
        human_row, human_col = _label_coordinates(human_label)
        raw_top1_label = int(raw_top1_labels[index].item())
        raw_top1_row, raw_top1_col = _label_coordinates(raw_top1_label)
        raw_top1_is_legal = bool(legal_mask[index, raw_top1_label].item())
        target_rank = _target_rank(ranked_labels[index], human_label)
        move_number = int(batch["move_number"][index].item())
        phase = game_phase_for_move_number(move_number)
        top_predictions = _top_predictions(
            ranked_labels[index, :resolved_top_k],
            probabilities[index],
            legal_mask[index],
            human_label=human_label,
        )
        category, category_basis = _categorize_example(
            phase=phase,
            target_rank=target_rank,
            raw_top1_is_legal=raw_top1_is_legal,
            top_k=resolved_top_k,
        )
        candidates.append(
            _QualitativeCandidate(
                game_id=game_id,
                source_name=batch["source_name"][index],
                move_number=move_number,
                phase=phase,
                human_label=human_label,
                human_row=human_row,
                human_col=human_col,
                target_rank=target_rank,
                top_predictions=top_predictions,
                raw_top1_label=raw_top1_label,
                raw_top1_row=raw_top1_row,
                raw_top1_col=raw_top1_col,
                raw_top1_probability=float(probabilities[index, raw_top1_label].item()),
                raw_top1_is_legal=raw_top1_is_legal,
                category=category,
                category_basis=category_basis,
                board_tensor=x[index].clone(),
            )
        )
    return tuple(candidates)


def _top_predictions(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    human_label: int,
) -> tuple[QualitativePrediction, ...]:
    predictions: list[QualitativePrediction] = []
    for rank, label_tensor in enumerate(labels, start=1):
        label = int(label_tensor.item())
        row, col = _label_coordinates(label)
        predictions.append(
            QualitativePrediction(
                rank=rank,
                label=label,
                row=row,
                col=col,
                probability=float(probabilities[label].item()),
                is_legal=bool(legal_mask[label].item()),
                is_human_move=label == human_label,
            )
        )
    return tuple(predictions)


def _select_examples_for_phase(
    pools: _PhaseCandidatePools,
    *,
    limit: int,
) -> tuple[QualitativeExample, ...]:
    selected: list[_QualitativeCandidate] = []
    seen: set[tuple[str, int, int]] = set()

    near_misses = sorted(pools.top5_near_misses, key=_near_miss_sort_key)
    high_confidence_limit = (
        max(0, limit - 1)
        if near_misses and limit > 1
        else limit
    )
    _add_selected_candidates(
        selected,
        seen,
        sorted(pools.high_confidence_misses, key=_high_confidence_sort_key),
        limit=high_confidence_limit,
    )
    if len(selected) < limit:
        _add_selected_candidates(selected, seen, near_misses, limit=len(selected) + 1)
    _add_selected_candidates(
        selected,
        seen,
        sorted(pools.high_confidence_misses, key=_high_confidence_sort_key),
        limit=limit,
    )
    _add_selected_candidates(
        selected,
        seen,
        near_misses,
        limit=limit,
    )
    _add_selected_candidates(
        selected,
        seen,
        sorted(pools.fallback, key=_fallback_sort_key),
        limit=limit,
    )
    return tuple(_candidate_to_example(candidate) for candidate in selected)


def _add_selected_candidates(
    selected: list[_QualitativeCandidate],
    seen: set[tuple[str, int, int]],
    candidates: list[_QualitativeCandidate],
    *,
    limit: int,
) -> None:
    for candidate in candidates:
        if len(selected) >= limit:
            return
        key = _candidate_key(candidate)
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)


def _candidate_to_example(candidate: _QualitativeCandidate) -> QualitativeExample:
    return QualitativeExample(
        game_id=candidate.game_id,
        source_name=candidate.source_name,
        move_number=candidate.move_number,
        phase=candidate.phase,
        human_label=candidate.human_label,
        human_row=candidate.human_row,
        human_col=candidate.human_col,
        target_rank=candidate.target_rank,
        top_predictions=candidate.top_predictions,
        raw_top1_label=candidate.raw_top1_label,
        raw_top1_row=candidate.raw_top1_row,
        raw_top1_col=candidate.raw_top1_col,
        raw_top1_probability=candidate.raw_top1_probability,
        raw_top1_is_legal=candidate.raw_top1_is_legal,
        category=candidate.category,
        category_basis=candidate.category_basis,
        board=_render_board(
            candidate.board_tensor,
            human_label=candidate.human_label,
            predictions=candidate.top_predictions,
        ),
    )


def _candidate_pools() -> dict[GamePhase, _PhaseCandidatePools]:
    return {
        phase: _PhaseCandidatePools(
            high_confidence_misses=[],
            top5_near_misses=[],
            fallback=[],
        )
        for phase in PHASE_NAMES
    }


def _candidate_pool_capacity(examples_per_phase: int) -> int:
    return max(8, examples_per_phase * 4)


def _add_candidate(
    pools: _PhaseCandidatePools,
    candidate: _QualitativeCandidate,
    *,
    capacity: int,
) -> None:
    if candidate.target_rank != 1:
        _add_bounded_candidate(
            pools.high_confidence_misses,
            candidate,
            capacity=capacity,
            key=_high_confidence_sort_key,
        )
    if 2 <= candidate.target_rank <= min(5, len(candidate.top_predictions)):
        _add_bounded_candidate(
            pools.top5_near_misses,
            candidate,
            capacity=capacity,
            key=_near_miss_sort_key,
        )
    _add_bounded_candidate(
        pools.fallback,
        candidate,
        capacity=capacity,
        key=_fallback_sort_key,
    )


def _add_bounded_candidate(
    candidates: list[_QualitativeCandidate],
    candidate: _QualitativeCandidate,
    *,
    capacity: int,
    key: Callable[[_QualitativeCandidate], Any],
) -> None:
    candidates.append(candidate)
    candidates.sort(key=key)
    del candidates[capacity:]


def _candidate_key(candidate: _QualitativeCandidate) -> tuple[str, int, int]:
    return (candidate.game_id, candidate.move_number, candidate.human_label)


def _high_confidence_sort_key(
    example: _QualitativeCandidate,
) -> tuple[float, str, int, int]:
    return (
        -example.top_predictions[0].probability,
        example.game_id,
        example.move_number,
        example.human_label,
    )


def _near_miss_sort_key(
    example: _QualitativeCandidate,
) -> tuple[int, float, str, int, int]:
    return (
        example.target_rank,
        -_human_probability(example),
        example.game_id,
        example.move_number,
        example.human_label,
    )


def _fallback_sort_key(
    example: _QualitativeCandidate,
) -> tuple[float, str, int, int]:
    return (
        -example.top_predictions[0].probability,
        example.game_id,
        example.move_number,
        example.human_label,
    )


def _human_probability(example: _QualitativeCandidate) -> float:
    for prediction in example.top_predictions:
        if prediction.is_human_move:
            return prediction.probability
    return 0.0


def _categorize_example(
    *,
    phase: GamePhase,
    target_rank: int,
    raw_top1_is_legal: bool,
    top_k: int,
) -> tuple[str, str]:
    if not raw_top1_is_legal:
        return "occupied_or_illegal_point", "raw_top1"
    if target_rank == 1:
        return "top_1_match", "ranked_topk"
    if target_rank <= top_k:
        return "reasonable_alternate_move", "ranked_topk"
    if phase == "opening":
        return "common_opening_pattern_confused", "ranked_topk"
    if phase == "endgame":
        return "endgame_precision_issue", "ranked_topk"
    return "global_direction_missed", "ranked_topk"


def _render_board(
    x: torch.Tensor,
    *,
    human_label: int,
    predictions: tuple[QualitativePrediction, ...],
) -> tuple[str, ...]:
    prediction_ranks = {
        prediction.label: prediction.rank
        for prediction in predictions
    }
    lines = [
        "    " + " ".join(f"{col:02d}" for col in range(SUPPORTED_BOARD_SIZE))
    ]
    for row in range(SUPPORTED_BOARD_SIZE):
        tokens: list[str] = []
        for col in range(SUPPORTED_BOARD_SIZE):
            label = row * SUPPORTED_BOARD_SIZE + col
            rank = prediction_ranks.get(label)
            rank_marker = "" if rank is None else _rank_marker(rank)
            if label == human_label:
                tokens.append(f"H{rank_marker}" if rank_marker else "H ")
            elif rank_marker:
                tokens.append(f"{rank_marker} ")
            else:
                tokens.append(f"{_board_marker(x, row, col)} ")
        lines.append(f"{row:02d}  " + " ".join(tokens).rstrip())
    return tuple(lines)


def _board_marker(x: torch.Tensor, row: int, col: int) -> str:
    if float(x[0, row, col].item()) > 0.5:
        return "X"
    if float(x[1, row, col].item()) > 0.5:
        return "O"
    return "."


def _rank_marker(rank: int) -> str:
    if 1 <= rank <= 9:
        return str(rank)
    if 10 <= rank <= 35:
        return chr(ord("A") + rank - 10)
    return "*"


def _validate_config(config: QualitativeAnalysisConfig) -> None:
    if config.split not in QUALITATIVE_ANALYSIS_SPLITS:
        msg = (
            "split must be one of "
            f"{', '.join(QUALITATIVE_ANALYSIS_SPLITS)}, got {config.split}"
        )
        raise QualitativeAnalysisError(msg)
    if config.batch_size < 1:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise QualitativeAnalysisError(msg)
    if not 1 <= config.top_k <= SUPPORTED_LABEL_COUNT:
        msg = (
            f"top_k must be in [1, {SUPPORTED_LABEL_COUNT}], "
            f"got {config.top_k}"
        )
        raise QualitativeAnalysisError(msg)
    if config.examples_per_phase < 1:
        msg = (
            "examples_per_phase must be positive, "
            f"got {config.examples_per_phase}"
        )
        raise QualitativeAnalysisError(msg)


def _validate_checkpoint_provenance(
    checkpoint: CnnPolicyCheckpoint,
    dataset_path: Path,
    manifest_path: Path,
) -> None:
    _validate_checkpoint_path_match(
        "dataset_path",
        checkpoint.dataset_path,
        dataset_path,
    )
    _validate_checkpoint_path_match(
        "manifest_path",
        checkpoint.manifest_path,
        manifest_path,
    )


def _validate_checkpoint_path_match(
    name: str,
    checkpoint_path: Path,
    evaluation_path: Path,
) -> None:
    if _resolved_path(checkpoint_path) == _resolved_path(evaluation_path):
        return

    msg = (
        f"{name} does not match checkpoint provenance: checkpoint has "
        f"{checkpoint_path}, analysis requested {evaluation_path}"
    )
    raise QualitativeAnalysisError(msg)


def _validate_logits_shape(logits: torch.Tensor, *, example_count: int) -> None:
    if logits.shape != (example_count, SUPPORTED_LABEL_COUNT):
        msg = (
            "model logits must have shape "
            f"({example_count}, {SUPPORTED_LABEL_COUNT}), got {tuple(logits.shape)}"
        )
        raise QualitativeAnalysisError(msg)
    if not torch.is_floating_point(logits):
        msg = f"model logits must be a floating point tensor, got {logits.dtype}"
        raise QualitativeAnalysisError(msg)


def _target_rank(ranked_labels: torch.Tensor, label: int) -> int:
    matches = (ranked_labels == label).nonzero(as_tuple=False)
    if int(matches.shape[0]) != 1:
        msg = f"could not locate target label {label} in ranked predictions"
        raise QualitativeAnalysisError(msg)
    return int(matches[0, 0].item()) + 1


def _empty_heatmap() -> list[list[int]]:
    return [
        [0 for _ in range(SUPPORTED_BOARD_SIZE)]
        for _ in range(SUPPORTED_BOARD_SIZE)
    ]


def _increment_heatmap(heatmap: list[list[int]], label: int) -> None:
    row, col = _label_coordinates(label)
    heatmap[row][col] += 1


def _freeze_heatmap(heatmap: list[list[int]]) -> Heatmap:
    return tuple(tuple(row) for row in heatmap)


def _label_coordinates(label: int) -> tuple[int, int]:
    if not 0 <= label < SUPPORTED_LABEL_COUNT:
        msg = f"label must be in [0, {SUPPORTED_LABEL_COUNT}), got {label}"
        raise QualitativeAnalysisError(msg)
    return divmod(label, SUPPORTED_BOARD_SIZE)


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _heatmap_to_json_data(heatmap: Heatmap) -> list[list[int]]:
    return [list(row) for row in heatmap]


def _example_to_json_data(example: QualitativeExample) -> dict[str, object]:
    return {
        "game_id": example.game_id,
        "source_name": example.source_name,
        "move_number": example.move_number,
        "phase": example.phase,
        "human_move": {
            "label": example.human_label,
            "row": example.human_row,
            "col": example.human_col,
            "target_rank": example.target_rank,
        },
        "raw_top1": {
            "label": example.raw_top1_label,
            "row": example.raw_top1_row,
            "col": example.raw_top1_col,
            "probability": example.raw_top1_probability,
            "is_legal": example.raw_top1_is_legal,
        },
        "top_predictions": [
            _prediction_to_json_data(prediction)
            for prediction in example.top_predictions
        ],
        "category": example.category,
        "category_basis": example.category_basis,
        "board": list(example.board),
    }


def _prediction_to_json_data(
    prediction: QualitativePrediction,
) -> dict[str, object]:
    return {
        "rank": prediction.rank,
        "label": prediction.label,
        "row": prediction.row,
        "col": prediction.col,
        "probability": prediction.probability,
        "is_legal": prediction.is_legal,
        "is_human_move": prediction.is_human_move,
    }


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()
