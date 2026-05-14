"""Batch replay helpers for SGF corpus checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vantago.replay.diagnostics import (
    ReplayDiagnostic,
    ReplayDiagnosticStatus,
    ReplaySkipReason,
    diagnose_sgf_replay_file,
)


@dataclass(frozen=True, slots=True)
class ReplayBatchOkGame:
    """Summary for one game that replayed successfully."""

    source_name: str
    move_count: int


@dataclass(frozen=True, slots=True)
class ReplayBatchSkipCount:
    """Count of skipped games for one stable replay skip reason."""

    reason: ReplaySkipReason
    count: int


@dataclass(frozen=True, slots=True)
class ReplayBatchFailure:
    """Unexpected failure while replaying one SGF file."""

    source_name: str
    message: str


@dataclass(frozen=True, slots=True)
class ReplayBatchResult:
    """Structured result for a recursive SGF batch replay check."""

    files_scanned: int
    ok: int
    skipped: int
    failed: int
    moves_replayed: int
    skipped_by_reason: tuple[ReplayBatchSkipCount, ...]
    ok_games: tuple[ReplayBatchOkGame, ...]
    skipped_diagnostics: tuple[ReplayDiagnostic, ...]
    failures: tuple[ReplayBatchFailure, ...]


def find_sgf_files(path: Path) -> tuple[Path, ...]:
    """Return SGF files under a path in deterministic order."""

    if not path.exists():
        msg = f"SGF batch path does not exist: {path}"
        raise ValueError(msg)
    if path.is_file():
        if path.suffix.lower() != ".sgf":
            msg = f"SGF batch file does not have .sgf suffix: {path}"
            raise ValueError(msg)
        return (path,)
    if not path.is_dir():
        msg = f"SGF batch path is not a file or directory: {path}"
        raise ValueError(msg)

    return tuple(
        sorted(
            (
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() == ".sgf"
            ),
            key=lambda child: child.relative_to(path).as_posix(),
        )
    )


def replay_sgf_batch(path: Path) -> ReplayBatchResult:
    """Replay all SGF files under a path and aggregate diagnostics."""

    ok_games: list[ReplayBatchOkGame] = []
    skipped_diagnostics: list[ReplayDiagnostic] = []
    failures: list[ReplayBatchFailure] = []
    skip_counts: Counter[ReplaySkipReason] = Counter()
    moves_replayed = 0

    sgf_files = find_sgf_files(path)
    for sgf_file in sgf_files:
        try:
            diagnostic = diagnose_sgf_replay_file(sgf_file)
        except Exception as exc:
            failures.append(
                ReplayBatchFailure(source_name=str(sgf_file), message=str(exc))
            )
            continue

        if diagnostic.status == ReplayDiagnosticStatus.OK:
            ok_games.append(
                ReplayBatchOkGame(
                    source_name=diagnostic.source_name,
                    move_count=diagnostic.move_count,
                )
            )
            moves_replayed += diagnostic.move_count
            continue

        if diagnostic.reason is None:
            failures.append(
                ReplayBatchFailure(
                    source_name=diagnostic.source_name,
                    message=f"skipped diagnostic missing reason: {diagnostic.message}",
                )
            )
            continue

        skipped_diagnostics.append(diagnostic)
        skip_counts[diagnostic.reason] += 1

    skipped_by_reason = tuple(
        ReplayBatchSkipCount(reason=reason, count=skip_counts[reason])
        for reason in ReplaySkipReason
        if skip_counts[reason] > 0
    )
    return ReplayBatchResult(
        files_scanned=len(sgf_files),
        ok=len(ok_games),
        skipped=len(skipped_diagnostics),
        failed=len(failures),
        moves_replayed=moves_replayed,
        skipped_by_reason=skipped_by_reason,
        ok_games=tuple(ok_games),
        skipped_diagnostics=tuple(skipped_diagnostics),
        failures=tuple(failures),
    )
