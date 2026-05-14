"""Go board replay APIs."""

from vantago.replay.batch import (
    ReplayBatchFailure,
    ReplayBatchOkGame,
    ReplayBatchResult,
    ReplayBatchSkipCount,
    find_sgf_files,
    replay_sgf_batch,
)
from vantago.replay.diagnostics import (
    SUPPORTED_BOARD_SIZE,
    SUPPORTED_POINT_COUNT,
    ReplayDiagnostic,
    ReplayDiagnosticStatus,
    ReplaySkipReason,
    diagnose_parsed_game_replay,
    diagnose_sgf_replay_bytes,
    diagnose_sgf_replay_file,
)
from vantago.replay.replay import (
    BoardState,
    IllegalMoveError,
    ReplayStep,
    replay_game,
)

__all__ = [
    "BoardState",
    "IllegalMoveError",
    "ReplayBatchFailure",
    "ReplayBatchOkGame",
    "ReplayBatchResult",
    "ReplayBatchSkipCount",
    "ReplayDiagnostic",
    "ReplayDiagnosticStatus",
    "ReplaySkipReason",
    "ReplayStep",
    "SUPPORTED_BOARD_SIZE",
    "SUPPORTED_POINT_COUNT",
    "diagnose_parsed_game_replay",
    "diagnose_sgf_replay_bytes",
    "diagnose_sgf_replay_file",
    "find_sgf_files",
    "replay_game",
    "replay_sgf_batch",
]
