"""Baseline evaluation and training APIs."""

from vantago.baselines.evaluation import (
    BASELINE_NAMES,
    COMPARISON_BASELINE_NAMES,
    NON_NEURAL_BASELINE_NAMES,
    PHASE_NAMES,
    BaselineEvaluationError,
    BaselineEvaluationResult,
    BaselineEvaluationRow,
    BaselineName,
    GamePhase,
    NonNeuralBaselineName,
    evaluate_baselines,
    game_phase_for_move_number,
)
from vantago.baselines.mlp import (
    FlattenedMlpPolicy,
    MlpBaselineConfig,
    MlpBaselineEpochResult,
    MlpBaselineTrainingError,
    MlpBaselineTrainingResult,
    evaluate_mlp_policy,
    train_mlp_baseline,
)

__all__ = [
    "BASELINE_NAMES",
    "COMPARISON_BASELINE_NAMES",
    "NON_NEURAL_BASELINE_NAMES",
    "PHASE_NAMES",
    "BaselineEvaluationError",
    "BaselineEvaluationResult",
    "BaselineEvaluationRow",
    "BaselineName",
    "FlattenedMlpPolicy",
    "GamePhase",
    "MlpBaselineConfig",
    "MlpBaselineEpochResult",
    "MlpBaselineTrainingError",
    "MlpBaselineTrainingResult",
    "NonNeuralBaselineName",
    "evaluate_baselines",
    "evaluate_mlp_policy",
    "game_phase_for_move_number",
    "train_mlp_baseline",
]
