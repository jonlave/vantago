"""Replay suitability diagnostics for constrained SGF game records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vantago.replay.replay import IllegalMoveError, ReplayStep, replay_game
from vantago.sgf import (
    MalformedSgfError,
    ParsedGame,
    SgfParseError,
    SgfReadError,
    UnsupportedHandicapError,
    UnsupportedSetupStonesError,
    UnsupportedSgfFeatureError,
    load_sgf,
    parse_sgf_bytes,
)


class ReplayDiagnosticStatus(StrEnum):
    """Stable status values returned by replay diagnostics."""

    OK = "ok"
    SKIPPED = "skipped"


class ReplaySkipReason(StrEnum):
    """Stable skip reason values returned by replay diagnostics."""

    NON_19X19_BOARD = "non_19x19_board"
    HANDICAP = "handicap"
    SETUP_STONES = "setup_stones"
    EMPTY_GAME = "empty_game"
    PASS_MOVE = "pass_move"
    MALFORMED_SGF = "malformed_sgf"
    ILLEGAL_MOVE_SEQUENCE = "illegal_move_sequence"
    UNSUPPORTED_SGF_FEATURE = "unsupported_sgf_feature"


STATUS_OK = ReplayDiagnosticStatus.OK
STATUS_SKIPPED = ReplayDiagnosticStatus.SKIPPED
REASON_NON_19X19_BOARD = ReplaySkipReason.NON_19X19_BOARD
REASON_HANDICAP = ReplaySkipReason.HANDICAP
REASON_SETUP_STONES = ReplaySkipReason.SETUP_STONES
REASON_EMPTY_GAME = ReplaySkipReason.EMPTY_GAME
REASON_PASS_MOVE = ReplaySkipReason.PASS_MOVE
REASON_MALFORMED_SGF = ReplaySkipReason.MALFORMED_SGF
REASON_ILLEGAL_MOVE_SEQUENCE = ReplaySkipReason.ILLEGAL_MOVE_SEQUENCE
REASON_UNSUPPORTED_SGF_FEATURE = ReplaySkipReason.UNSUPPORTED_SGF_FEATURE
SUPPORTED_BOARD_SIZE = 19
SUPPORTED_POINT_COUNT = SUPPORTED_BOARD_SIZE * SUPPORTED_BOARD_SIZE


@dataclass(frozen=True, slots=True)
class ReplayDiagnostic:
    """Structured result for one SGF replay suitability check."""

    status: ReplayDiagnosticStatus
    reason: ReplaySkipReason | None
    source_name: str
    message: str
    move_count: int
    replay_steps: tuple[ReplayStep, ...]


def diagnose_sgf_replay_bytes(
    content: bytes,
    source_name: str | None = None,
) -> ReplayDiagnostic:
    """Parse SGF bytes and return replay diagnostics for the supported subset."""

    resolved_source_name = source_name or "<bytes>"
    try:
        parsed_game = parse_sgf_bytes(content, source_name=resolved_source_name)
    except SgfParseError as exc:
        return _skipped_from_parse_error(resolved_source_name, exc)
    return diagnose_parsed_game_replay(parsed_game)


def diagnose_sgf_replay_file(path: Path) -> ReplayDiagnostic:
    """Load an SGF file and return replay diagnostics for the supported subset."""

    try:
        parsed_game = load_sgf(path)
    except SgfParseError as exc:
        return _skipped_from_parse_error(str(path), exc)
    return diagnose_parsed_game_replay(parsed_game)


def diagnose_parsed_game_replay(parsed_game: ParsedGame) -> ReplayDiagnostic:
    """Return replay diagnostics for an already parsed SGF game."""

    if parsed_game.metadata.board_size != SUPPORTED_BOARD_SIZE:
        return _skipped(
            source_name=parsed_game.source_name,
            reason=REASON_NON_19X19_BOARD,
            message=(
                f"{parsed_game.source_name}: skipped board size "
                f"{parsed_game.metadata.board_size}; expected "
                f"{SUPPORTED_BOARD_SIZE}"
            ),
            move_count=len(parsed_game.moves),
        )

    move_count = len(parsed_game.moves)
    if move_count == 0:
        return _skipped(
            source_name=parsed_game.source_name,
            reason=REASON_EMPTY_GAME,
            message=f"{parsed_game.source_name}: skipped empty game with no moves",
            move_count=move_count,
        )

    pass_move_number = _first_pass_move_number(parsed_game)
    if pass_move_number is not None:
        return _skipped(
            source_name=parsed_game.source_name,
            reason=REASON_PASS_MOVE,
            message=(
                f"{parsed_game.source_name}: skipped pass move at move "
                f"{pass_move_number}; pass is not in the "
                f"{SUPPORTED_POINT_COUNT}-point target space"
            ),
            move_count=len(parsed_game.moves),
        )

    try:
        replay_steps = replay_game(parsed_game)
    except IllegalMoveError as exc:
        return _skipped(
            source_name=parsed_game.source_name,
            reason=REASON_ILLEGAL_MOVE_SEQUENCE,
            message=f"{parsed_game.source_name}: illegal move sequence: {exc}",
            move_count=len(parsed_game.moves),
        )

    return ReplayDiagnostic(
        status=STATUS_OK,
        reason=None,
        source_name=parsed_game.source_name,
        message=f"{parsed_game.source_name}: replayed {move_count} moves",
        move_count=move_count,
        replay_steps=replay_steps,
    )


def _skipped_from_parse_error(
    source_name: str,
    exc: SgfParseError,
) -> ReplayDiagnostic:
    return _skipped(
        source_name=source_name,
        reason=_reason_for_parse_error(exc),
        message=str(exc),
        move_count=0,
    )


def _reason_for_parse_error(exc: SgfParseError) -> ReplaySkipReason:
    if isinstance(exc, UnsupportedHandicapError):
        return REASON_HANDICAP
    if isinstance(exc, UnsupportedSetupStonesError):
        return REASON_SETUP_STONES
    if isinstance(exc, MalformedSgfError):
        return REASON_MALFORMED_SGF
    if isinstance(exc, SgfReadError):
        return REASON_MALFORMED_SGF
    if isinstance(exc, UnsupportedSgfFeatureError):
        return REASON_UNSUPPORTED_SGF_FEATURE
    return REASON_MALFORMED_SGF


def _first_pass_move_number(parsed_game: ParsedGame) -> int | None:
    for move_number, move in enumerate(parsed_game.moves, start=1):
        if move.point is None:
            return move_number
    return None


def _skipped(
    *,
    source_name: str,
    reason: ReplaySkipReason,
    message: str,
    move_count: int,
) -> ReplayDiagnostic:
    return ReplayDiagnostic(
        status=STATUS_SKIPPED,
        reason=reason,
        source_name=source_name,
        message=message,
        move_count=move_count,
        replay_steps=(),
    )
