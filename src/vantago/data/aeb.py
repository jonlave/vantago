"""Fetch SGF corpora from Andries Brouwer's Database of Go Games."""

from __future__ import annotations

import hashlib
import json
import random
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Protocol, cast

from vantago.data.artifacts import (
    ProcessedDatasetBuildResult,
    write_processed_dataset,
)
from vantago.data.splits import (
    MINIMUM_SPLIT_GAME_COUNT,
    DatasetSplitBuildResult,
    write_dataset_split_manifest,
)
from vantago.replay import (
    ReplayDiagnosticStatus,
    ReplaySkipReason,
    diagnose_sgf_replay_bytes,
)

AEB_ARCHIVE_URL = "https://homepages.cwi.nl/~aeb/go/games/games.tgz"
DEFAULT_AEB_CACHE_DIR = Path("data/raw/.cache/aeb")
DEFAULT_AEB_FETCH_ROOT = Path("data/raw/aeb-fetches")
DEFAULT_AEB_PROCESSED_DIR = Path("data/processed")
DEFAULT_AEB_PREPARED_NAME_TEMPLATE = "aeb-{games}-s{seed}"
DEFAULT_AEB_TIMEOUT_SECONDS = 30.0
USER_AGENT = "vantago/0.1"
AebProgressCallback = Callable[[str], None]


class _Hash(Protocol):
    def update(self, data: bytes, /) -> object:
        """Update the digest with raw bytes."""


class AebFetchError(ValueError):
    """Raised when an AEB corpus cannot be fetched, cataloged, or written."""


@dataclass(frozen=True, slots=True)
class AebArchiveMetadata:
    """Observed metadata for one cached AEB archive."""

    source_url: str
    observed_at: str
    last_modified: str | None
    etag: str | None
    content_length: int | None
    archive_sha256: str
    cache_key: str


@dataclass(frozen=True, slots=True)
class AebCatalogEntry:
    """One replay-valid trainable SGF entry from the AEB archive."""

    source_path: str
    move_count: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AebSkipCount:
    """Count of skipped archive SGFs for one replay skip reason."""

    reason: ReplaySkipReason
    count: int


@dataclass(frozen=True, slots=True)
class AebCatalog:
    """Replay catalog for one verified AEB archive."""

    archive_sha256: str
    files_scanned: int
    valid_games: int
    skipped: int
    failed: int
    moves_replayed: int
    skipped_by_reason: tuple[AebSkipCount, ...]
    entries: tuple[AebCatalogEntry, ...]
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AebFetchConfig:
    """Configuration for selecting raw SGFs from the AEB archive."""

    games: int
    seed: int
    output_dir: Path
    archive_url: str = AEB_ARCHIVE_URL
    cache_dir: Path = DEFAULT_AEB_CACHE_DIR
    timeout_seconds: float = DEFAULT_AEB_TIMEOUT_SECONDS
    progress_callback: AebProgressCallback | None = None
    progress_interval: int = 5_000


@dataclass(frozen=True, slots=True)
class AebFetchResult:
    """Completed AEB raw corpus fetch result."""

    output_dir: Path
    manifest_path: Path
    readme_path: Path
    archive_path: Path
    catalog_path: Path
    archive_metadata: AebArchiveMetadata
    games_requested: int
    seed: int
    files_scanned: int
    valid_games_available: int
    skipped: int
    failed: int
    selected_games: tuple[AebCatalogEntry, ...]
    skipped_by_reason: tuple[AebSkipCount, ...]
    moves_replayed: int


@dataclass(frozen=True, slots=True)
class AebDatasetPrepareConfig:
    """Configuration for fetching AEB SGFs and preparing trainable artifacts."""

    games: int
    seed: int
    name: str | None = None
    archive_url: str = AEB_ARCHIVE_URL
    cache_dir: Path = DEFAULT_AEB_CACHE_DIR
    timeout_seconds: float = DEFAULT_AEB_TIMEOUT_SECONDS
    progress_callback: AebProgressCallback | None = None
    progress_interval: int = 5_000
    raw_output_dir: Path | None = None
    dataset_output: Path | None = None
    splits_output: Path | None = None


@dataclass(frozen=True, slots=True)
class AebDatasetPrepareResult:
    """Completed AEB fetch, process, and split workflow result."""

    name: str
    fetch_result: AebFetchResult
    dataset_result: ProcessedDatasetBuildResult
    split_result: DatasetSplitBuildResult


@dataclass(frozen=True, slots=True)
class _ArchiveRef:
    path: Path
    metadata_path: Path
    metadata: AebArchiveMetadata


@dataclass(frozen=True, slots=True)
class _DownloadedArchive:
    sha256: str
    last_modified: str | None
    etag: str | None
    content_length: int
    cache_key: str


def fetch_aeb_games(config: AebFetchConfig) -> AebFetchResult:
    """Fetch a deterministic sample of replay-valid AEB SGFs into a raw corpus."""

    _validate_positive_game_count(config.games)
    _validate_timeout(config.timeout_seconds)
    _validate_progress_interval(config.progress_interval)
    _validate_output_dir(config.output_dir)

    archive_ref = _fetch_archive(
        config.archive_url,
        config.cache_dir,
        timeout_seconds=config.timeout_seconds,
    )
    catalog_path, catalog = _load_or_build_catalog(
        archive_ref.path,
        archive_ref.metadata,
        config.cache_dir,
        progress_callback=config.progress_callback,
        progress_interval=config.progress_interval,
    )
    if config.games > catalog.valid_games:
        msg = (
            f"requested {config.games} games, but only "
            f"{catalog.valid_games} replay-valid games are available"
        )
        raise AebFetchError(msg)

    selected = tuple(
        sorted(
            random.Random(config.seed).sample(list(catalog.entries), config.games),
            key=lambda entry: entry.source_path,
        )
    )
    selected_bytes = _read_selected_archive_entries(
        archive_ref.path,
        (entry.source_path for entry in selected),
    )
    _write_selected_corpus(
        output_dir=config.output_dir,
        selected=selected,
        selected_bytes=selected_bytes,
        archive_metadata=archive_ref.metadata,
        catalog=catalog,
        seed=config.seed,
    )

    manifest_path = config.output_dir / "MANIFEST.txt"
    readme_path = config.output_dir / "README.md"
    return AebFetchResult(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        readme_path=readme_path,
        archive_path=archive_ref.path,
        catalog_path=catalog_path,
        archive_metadata=archive_ref.metadata,
        games_requested=config.games,
        seed=config.seed,
        files_scanned=catalog.files_scanned,
        valid_games_available=catalog.valid_games,
        skipped=catalog.skipped,
        failed=catalog.failed,
        selected_games=selected,
        skipped_by_reason=catalog.skipped_by_reason,
        moves_replayed=sum(entry.move_count for entry in selected),
    )


def prepare_aeb_dataset(
    config: AebDatasetPrepareConfig,
) -> AebDatasetPrepareResult:
    """Fetch AEB SGFs, then write the processed dataset and split manifest."""

    _validate_positive_game_count(config.games)
    _validate_timeout(config.timeout_seconds)
    _validate_progress_interval(config.progress_interval)
    if config.games < MINIMUM_SPLIT_GAME_COUNT:
        msg = (
            "prepare-aeb-dataset requires at least "
            f"{MINIMUM_SPLIT_GAME_COUNT} games so train/validation/test splits "
            f"can be created, got {config.games}"
        )
        raise AebFetchError(msg)

    name = _prepared_name(config)
    raw_output_dir = config.raw_output_dir or DEFAULT_AEB_FETCH_ROOT / name
    dataset_output = config.dataset_output or DEFAULT_AEB_PROCESSED_DIR / f"{name}.npz"
    splits_output = config.splits_output or DEFAULT_AEB_PROCESSED_DIR / (
        f"{name}-splits.json"
    )
    _validate_prepare_outputs(raw_output_dir, dataset_output, splits_output)

    fetch_result = fetch_aeb_games(
        AebFetchConfig(
            games=config.games,
            seed=config.seed,
            output_dir=raw_output_dir,
            archive_url=config.archive_url,
            cache_dir=config.cache_dir,
            timeout_seconds=config.timeout_seconds,
            progress_callback=config.progress_callback,
            progress_interval=config.progress_interval,
        )
    )
    dataset_result = write_processed_dataset(raw_output_dir, dataset_output)
    split_result = write_dataset_split_manifest(
        dataset_output,
        splits_output,
        seed=config.seed,
    )
    return AebDatasetPrepareResult(
        name=name,
        fetch_result=fetch_result,
        dataset_result=dataset_result,
        split_result=split_result,
    )


def build_aeb_catalog(
    archive_path: Path,
    archive_sha256: str | None = None,
    *,
    progress_callback: AebProgressCallback | None = None,
    progress_interval: int = 5_000,
) -> AebCatalog:
    """Build a replay-valid trainable catalog from a local tar.gz archive."""

    _validate_progress_interval(progress_interval)
    resolved_archive_sha256 = archive_sha256 or _sha256_file(archive_path)
    entries: list[AebCatalogEntry] = []
    failures: list[str] = []
    skip_counts: Counter[ReplaySkipReason] = Counter()
    files_scanned = 0
    skipped = 0
    moves_replayed = 0

    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not _is_sgf_member(member):
                    continue
                files_scanned += 1
                source_path = _normalize_archive_path(member.name)
                try:
                    content = _read_tar_member_bytes(archive, member)
                    diagnostic = diagnose_sgf_replay_bytes(
                        content,
                        source_name=source_path,
                    )
                except Exception as exc:
                    failures.append(f"{source_path}: {exc}")
                    _report_catalog_progress(
                        progress_callback=progress_callback,
                        progress_interval=progress_interval,
                        files_scanned=files_scanned,
                        valid_games=len(entries),
                        skipped=skipped,
                        failed=len(failures),
                    )
                    continue

                if diagnostic.status != ReplayDiagnosticStatus.OK:
                    skipped += 1
                    if diagnostic.reason is None:
                        failures.append(
                            f"{source_path}: skipped diagnostic missing reason"
                        )
                    else:
                        skip_counts[diagnostic.reason] += 1
                    _report_catalog_progress(
                        progress_callback=progress_callback,
                        progress_interval=progress_interval,
                        files_scanned=files_scanned,
                        valid_games=len(entries),
                        skipped=skipped,
                        failed=len(failures),
                    )
                    continue

                if diagnostic.move_count < 1:
                    skipped += 1
                    skip_counts[ReplaySkipReason.EMPTY_GAME] += 1
                    _report_catalog_progress(
                        progress_callback=progress_callback,
                        progress_interval=progress_interval,
                        files_scanned=files_scanned,
                        valid_games=len(entries),
                        skipped=skipped,
                        failed=len(failures),
                    )
                    continue

                moves_replayed += diagnostic.move_count
                entries.append(
                    AebCatalogEntry(
                        source_path=source_path,
                        move_count=diagnostic.move_count,
                        byte_count=len(content),
                        sha256=_sha256_bytes(content),
                    )
                )
                _report_catalog_progress(
                    progress_callback=progress_callback,
                    progress_interval=progress_interval,
                    files_scanned=files_scanned,
                    valid_games=len(entries),
                    skipped=skipped,
                    failed=len(failures),
                )
    except (OSError, tarfile.TarError) as exc:
        msg = f"unable to read AEB archive {archive_path}: {exc}"
        raise AebFetchError(msg) from exc

    sorted_entries = tuple(sorted(entries, key=lambda entry: entry.source_path))
    return AebCatalog(
        archive_sha256=resolved_archive_sha256,
        files_scanned=files_scanned,
        valid_games=len(sorted_entries),
        skipped=skipped,
        failed=len(failures),
        moves_replayed=moves_replayed,
        skipped_by_reason=_skip_counts_by_reason(skip_counts),
        entries=sorted_entries,
        failures=tuple(failures),
    )


def _fetch_archive(
    archive_url: str,
    cache_dir: Path,
    *,
    timeout_seconds: float,
) -> _ArchiveRef:
    cache_dir.mkdir(parents=True, exist_ok=True)
    observed = _observe_archive(archive_url, timeout_seconds=timeout_seconds)
    cache_path = cache_dir / f"archive-{observed.cache_key}.tgz"
    metadata_path = cache_dir / f"archive-{observed.cache_key}.json"

    if cache_path.exists() and metadata_path.exists():
        cached_metadata = _load_archive_metadata(metadata_path)
        if cached_metadata is not None and _verified_cached_archive(
            cache_path,
            cached_metadata,
        ):
            return _ArchiveRef(
                path=cache_path,
                metadata_path=metadata_path,
                metadata=cached_metadata,
            )
        _remove_corrupt_cache_files(cache_path, metadata_path)

    downloaded = _download_archive_to_cache(
        archive_url,
        cache_path,
        timeout_seconds=timeout_seconds,
    )
    final_cache_path = cache_dir / f"archive-{downloaded.cache_key}.tgz"
    final_metadata_path = cache_dir / f"archive-{downloaded.cache_key}.json"
    if final_cache_path != cache_path:
        try:
            cache_path.replace(final_cache_path)
        except OSError as exc:
            msg = f"unable to finalize AEB archive cache {final_cache_path}: {exc}"
            raise AebFetchError(msg) from exc

    metadata = AebArchiveMetadata(
        source_url=archive_url,
        observed_at=observed.observed_at,
        last_modified=downloaded.last_modified,
        etag=downloaded.etag,
        content_length=downloaded.content_length,
        archive_sha256=downloaded.sha256,
        cache_key=downloaded.cache_key,
    )
    _write_json_atomic(final_metadata_path, _archive_metadata_to_json(metadata))
    return _ArchiveRef(
        path=final_cache_path,
        metadata_path=final_metadata_path,
        metadata=metadata,
    )


def _load_or_build_catalog(
    archive_path: Path,
    archive_metadata: AebArchiveMetadata,
    cache_dir: Path,
    *,
    progress_callback: AebProgressCallback | None,
    progress_interval: int,
) -> tuple[Path, AebCatalog]:
    catalog_path = cache_dir / f"catalog-{archive_metadata.archive_sha256}.json"
    if catalog_path.exists():
        try:
            catalog = _load_catalog(catalog_path)
        except AebFetchError:
            with suppress(OSError):
                catalog_path.unlink(missing_ok=True)
        else:
            if catalog.archive_sha256 == archive_metadata.archive_sha256:
                return catalog_path, catalog

    catalog = build_aeb_catalog(
        archive_path,
        archive_sha256=archive_metadata.archive_sha256,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    _write_json_atomic(catalog_path, _catalog_to_json(catalog))
    return catalog_path, catalog


@dataclass(frozen=True, slots=True)
class _ObservedArchive:
    source_url: str
    observed_at: str
    last_modified: str | None
    etag: str | None
    content_length: int | None
    cache_key: str


def _observe_archive(archive_url: str, *, timeout_seconds: float) -> _ObservedArchive:
    parsed = urllib.parse.urlparse(archive_url)
    observed_at = _utc_now()
    last_modified: str | None
    content_length: int | None
    if parsed.scheme in {"", "file"}:
        path = _path_from_file_url_or_path(archive_url)
        try:
            stat = path.stat()
        except OSError as exc:
            msg = f"unable to observe AEB archive {archive_url}: {exc}"
            raise AebFetchError(msg) from exc
        last_modified = _timestamp_to_utc_iso(stat.st_mtime)
        content_length = stat.st_size
        return _ObservedArchive(
            source_url=archive_url,
            observed_at=observed_at,
            last_modified=last_modified,
            etag=None,
            content_length=content_length,
            cache_key=_cache_key(
                archive_url,
                last_modified=last_modified,
                etag=None,
                content_length=content_length,
            ),
        )

    request = urllib.request.Request(
        archive_url,
        headers={"User-Agent": USER_AGENT},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            headers = response.headers
            last_modified = headers.get("Last-Modified")
            etag = headers.get("ETag")
            content_length = _parse_optional_int(headers.get("Content-Length"))
    except OSError as exc:
        msg = f"unable to observe AEB archive {archive_url}: {exc}"
        raise AebFetchError(msg) from exc

    return _ObservedArchive(
        source_url=archive_url,
        observed_at=observed_at,
        last_modified=last_modified,
        etag=etag,
        content_length=content_length,
        cache_key=_cache_key(
            archive_url,
            last_modified=last_modified,
            etag=etag,
            content_length=content_length,
        ),
    )


def _download_archive_to_cache(
    archive_url: str,
    cache_path: Path,
    *,
    timeout_seconds: float,
) -> _DownloadedArchive:
    temp_path: Path | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            downloaded = _copy_url_to_file(
                archive_url,
                temp_file,
                timeout_seconds=timeout_seconds,
            )
        temp_path.replace(cache_path)
        return downloaded
    except OSError as exc:
        msg = f"unable to cache AEB archive {archive_url}: {exc}"
        raise AebFetchError(msg) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def _copy_url_to_file(
    archive_url: str,
    output: IO[bytes],
    *,
    timeout_seconds: float,
) -> _DownloadedArchive:
    parsed = urllib.parse.urlparse(archive_url)
    digest = hashlib.sha256()
    if parsed.scheme in {"", "file"}:
        path = _path_from_file_url_or_path(archive_url)
        try:
            with path.open("rb") as source:
                bytes_written = _copy_stream(source, output, digest)
            stat = path.stat()
        except OSError as exc:
            msg = f"unable to read AEB archive {archive_url}: {exc}"
            raise AebFetchError(msg) from exc
        if bytes_written != stat.st_size:
            msg = (
                f"AEB archive changed while reading {archive_url}: "
                f"copied {bytes_written} bytes but source size is {stat.st_size}"
            )
            raise AebFetchError(msg)
        last_modified = _timestamp_to_utc_iso(stat.st_mtime)
        return _DownloadedArchive(
            sha256=digest.hexdigest(),
            last_modified=last_modified,
            etag=None,
            content_length=bytes_written,
            cache_key=_cache_key(
                archive_url,
                last_modified=last_modified,
                etag=None,
                content_length=bytes_written,
            ),
        )

    request = urllib.request.Request(
        archive_url,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            headers = response.headers
            last_modified = headers.get("Last-Modified")
            etag = headers.get("ETag")
            expected_length = _parse_optional_int(headers.get("Content-Length"))
            bytes_written = _copy_stream(cast(IO[bytes], response), output, digest)
    except OSError as exc:
        msg = f"unable to download AEB archive {archive_url}: {exc}"
        raise AebFetchError(msg) from exc
    if expected_length is not None and bytes_written != expected_length:
        msg = (
            f"AEB archive download length mismatch for {archive_url}: "
            f"expected {expected_length} bytes, got {bytes_written}"
        )
        raise AebFetchError(msg)
    return _DownloadedArchive(
        sha256=digest.hexdigest(),
        last_modified=last_modified,
        etag=etag,
        content_length=bytes_written,
        cache_key=_cache_key(
            archive_url,
            last_modified=last_modified,
            etag=etag,
            content_length=bytes_written,
        ),
    )


def _copy_stream(source: IO[bytes], output: IO[bytes], digest: _Hash) -> int:
    bytes_written = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return bytes_written
        digest.update(chunk)
        output.write(chunk)
        bytes_written += len(chunk)


def _verified_cached_archive(
    cache_path: Path,
    metadata: AebArchiveMetadata,
) -> bool:
    try:
        return _sha256_file(cache_path) == metadata.archive_sha256
    except AebFetchError:
        return False


def _remove_corrupt_cache_files(cache_path: Path, metadata_path: Path) -> None:
    for path in (cache_path, metadata_path):
        with suppress(OSError):
            path.unlink(missing_ok=True)


def _read_selected_archive_entries(
    archive_path: Path,
    source_paths: Iterable[str],
) -> dict[str, bytes]:
    requested = set(source_paths)
    found: dict[str, bytes] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not _is_sgf_member(member):
                    continue
                source_path = _normalize_archive_path(member.name)
                if source_path not in requested or source_path in found:
                    continue
                found[source_path] = _read_tar_member_bytes(archive, member)
                if len(found) == len(requested):
                    break
    except (OSError, tarfile.TarError) as exc:
        msg = f"unable to read selected AEB SGFs from {archive_path}: {exc}"
        raise AebFetchError(msg) from exc

    missing = sorted(requested - set(found))
    if missing:
        msg = "selected AEB archive members were not found: " + ", ".join(missing)
        raise AebFetchError(msg)
    return found


def _write_selected_corpus(
    *,
    output_dir: Path,
    selected: tuple[AebCatalogEntry, ...],
    selected_bytes: dict[str, bytes],
    archive_metadata: AebArchiveMetadata,
    catalog: AebCatalog,
    seed: int,
) -> None:
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        width = max(4, len(str(len(selected))))
        with tempfile.TemporaryDirectory(
            dir=parent,
            prefix=f".{output_dir.name}.",
        ) as temp_name:
            temp_dir = Path(temp_name)
            sgf_dir = temp_dir / "sgf"
            sgf_dir.mkdir(parents=True)
            rows: list[_ManifestRow] = []
            for index, entry in enumerate(selected, start=1):
                local_id = f"aeb-fetch-{index:0{width}d}"
                local_file = Path("sgf") / f"{local_id}.sgf"
                content = selected_bytes[entry.source_path]
                if _sha256_bytes(content) != entry.sha256:
                    msg = f"selected SGF changed while writing: {entry.source_path}"
                    raise AebFetchError(msg)
                (temp_dir / local_file).write_bytes(content)
                rows.append(
                    _ManifestRow(
                        local_id=local_id,
                        local_file=local_file.as_posix(),
                        source_path=entry.source_path,
                        move_count=entry.move_count,
                        byte_count=entry.byte_count,
                        sha256=entry.sha256,
                    )
                )

            (temp_dir / "MANIFEST.txt").write_text(
                _format_manifest(
                    archive_metadata=archive_metadata,
                    catalog=catalog,
                    seed=seed,
                    rows=tuple(rows),
                ),
                encoding="utf-8",
            )
            (temp_dir / "README.md").write_text(
                _format_readme(
                    output_name=output_dir.name,
                    archive_metadata=archive_metadata,
                    catalog=catalog,
                    seed=seed,
                    rows=tuple(rows),
                ),
                encoding="utf-8",
            )
            temp_dir.replace(output_dir)
    except OSError as exc:
        msg = f"unable to write AEB corpus {output_dir}: {exc}"
        raise AebFetchError(msg) from exc


@dataclass(frozen=True, slots=True)
class _ManifestRow:
    local_id: str
    local_file: str
    source_path: str
    move_count: int
    byte_count: int
    sha256: str


def _format_manifest(
    *,
    archive_metadata: AebArchiveMetadata,
    catalog: AebCatalog,
    seed: int,
    rows: tuple[_ManifestRow, ...],
) -> str:
    total_moves = sum(row.move_count for row in rows)
    lines = [
        "# AEB Fetched SGF Manifest",
        "",
        "Selection rule: build the replay-valid catalog for the observed "
        "games.tgz archive, sample the requested number of valid games with "
        "random.Random(seed).sample, sort selected archive paths, then copy "
        "SGF bytes unchanged under stable local filenames.",
        "",
        f"source_url: {archive_metadata.source_url}",
        f"observed_at: {archive_metadata.observed_at}",
        f"archive_last_modified: {_format_optional(archive_metadata.last_modified)}",
        f"archive_etag: {_format_optional(archive_metadata.etag)}",
        "archive_content_length: "
        f"{_format_optional_int(archive_metadata.content_length)}",
        f"archive_sha256: {archive_metadata.archive_sha256}",
        f"selection_seed: {seed}",
        f"games_requested: {len(rows)}",
        f"valid_games_available: {catalog.valid_games}",
        f"archive_sgf_files_scanned: {catalog.files_scanned}",
        f"archive_sgf_files_skipped: {catalog.skipped}",
        f"archive_sgf_files_failed: {catalog.failed}",
        f"selected_files: {len(rows)}",
        f"total_moves_replayed: {total_moves}",
        "skipped_during_catalog:",
    ]
    if catalog.skipped_by_reason:
        lines.extend(
            f"  {skip_count.reason.value}: {skip_count.count}"
            for skip_count in catalog.skipped_by_reason
        )
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "files:",
            "id\tfile\tsource_path\tmoves\tbytes\tsha256",
        ]
    )
    lines.extend(
        "\t".join(
            (
                row.local_id,
                row.local_file,
                row.source_path,
                str(row.move_count),
                str(row.byte_count),
                row.sha256,
            )
        )
        for row in rows
    )
    return "\n".join(lines) + "\n"


def _format_readme(
    *,
    output_name: str,
    archive_metadata: AebArchiveMetadata,
    catalog: AebCatalog,
    seed: int,
    rows: tuple[_ManifestRow, ...],
) -> str:
    return (
        f"# {output_name} AEB SGF Corpus\n\n"
        "This directory contains replay-valid SGF files selected on demand from "
        "Andries Brouwer's Database of Go Games. SGF bytes are copied unchanged "
        "from the source archive.\n\n"
        f"Source archive: {archive_metadata.source_url}\n"
        f"Archive observed at: {archive_metadata.observed_at}\n"
        f"Archive Last-Modified: {_format_optional(archive_metadata.last_modified)}\n"
        f"Archive ETag: {_format_optional(archive_metadata.etag)}\n"
        "Archive Content-Length: "
        f"{_format_optional_int(archive_metadata.content_length)}\n"
        f"Archive SHA-256: {archive_metadata.archive_sha256}\n\n"
        f"Selection seed: {seed}\n"
        f"Selected games: {len(rows)}\n"
        f"Replay-valid games available: {catalog.valid_games}\n"
        f"Total moves replayed: {sum(row.move_count for row in rows)}\n\n"
        "See `MANIFEST.txt` for source paths, local filenames, move counts, "
        "byte counts, per-file SHA-256 hashes, and catalog skip counts.\n"
    )


def _validate_positive_game_count(games: int) -> None:
    if games < 1:
        msg = f"games must be a positive integer, got {games}"
        raise AebFetchError(msg)


def _validate_timeout(timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {timeout_seconds}"
        raise AebFetchError(msg)


def _validate_progress_interval(progress_interval: int) -> None:
    if progress_interval < 1:
        msg = f"progress_interval must be positive, got {progress_interval}"
        raise AebFetchError(msg)


def _validate_output_dir(output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        msg = f"AEB output directory already exists: {output_dir}"
        raise AebFetchError(msg)


def _validate_prepare_outputs(
    raw_output_dir: Path,
    dataset_output: Path,
    splits_output: Path,
) -> None:
    _validate_output_dir(raw_output_dir)
    if dataset_output.suffix != ".npz":
        msg = f"processed dataset output path must end with .npz: {dataset_output}"
        raise AebFetchError(msg)
    if dataset_output.exists() and dataset_output.is_dir():
        msg = f"processed dataset output path is a directory: {dataset_output}"
        raise AebFetchError(msg)
    if splits_output.suffix != ".json":
        msg = f"dataset split manifest output path must end with .json: {splits_output}"
        raise AebFetchError(msg)
    if splits_output.exists() and splits_output.is_dir():
        msg = f"dataset split manifest output path is a directory: {splits_output}"
        raise AebFetchError(msg)


def _prepared_name(config: AebDatasetPrepareConfig) -> str:
    if config.name is not None and config.name:
        return config.name
    return DEFAULT_AEB_PREPARED_NAME_TEMPLATE.format(
        games=config.games,
        seed=config.seed,
    )


def _is_sgf_member(member: tarfile.TarInfo) -> bool:
    return member.isfile() and member.name.lower().endswith(".sgf")


def _report_catalog_progress(
    *,
    progress_callback: AebProgressCallback | None,
    progress_interval: int,
    files_scanned: int,
    valid_games: int,
    skipped: int,
    failed: int,
) -> None:
    if progress_callback is None or files_scanned % progress_interval != 0:
        return
    progress_callback(
        "cataloged "
        f"{files_scanned} SGF files "
        f"({valid_games} valid, {skipped} skipped, {failed} failed)"
    )


def _normalize_archive_path(path: str) -> str:
    return path.lstrip("./")


def _read_tar_member_bytes(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        msg = f"archive member cannot be read: {member.name}"
        raise AebFetchError(msg)
    try:
        return extracted.read()
    except OSError as exc:
        msg = f"unable to read archive member {member.name}: {exc}"
        raise AebFetchError(msg) from exc


def _skip_counts_by_reason(
    skip_counts: Counter[ReplaySkipReason],
) -> tuple[AebSkipCount, ...]:
    return tuple(
        AebSkipCount(reason=reason, count=skip_counts[reason])
        for reason in ReplaySkipReason
        if skip_counts[reason] > 0
    )


def _cache_key(
    source_url: str,
    *,
    last_modified: str | None,
    etag: str | None,
    content_length: int | None,
) -> str:
    content = "\n".join(
        (
            source_url,
            last_modified or "",
            etag or "",
            "" if content_length is None else str(content_length),
        )
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _path_from_file_url_or_path(archive_url: str) -> Path:
    parsed = urllib.parse.urlparse(archive_url)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path))
    return Path(archive_url)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    return digest.hexdigest()
                digest.update(chunk)
    except OSError as exc:
        msg = f"unable to hash {path}: {exc}"
        raise AebFetchError(msg) from exc


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_to_utc_iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _format_optional(value: str | None) -> str:
    return value if value is not None else "<unknown>"


def _format_optional_int(value: int | None) -> str:
    return str(value) if value is not None else "<unknown>"


def _archive_metadata_to_json(metadata: AebArchiveMetadata) -> dict[str, object]:
    return {
        "source_url": metadata.source_url,
        "observed_at": metadata.observed_at,
        "last_modified": metadata.last_modified,
        "etag": metadata.etag,
        "content_length": metadata.content_length,
        "archive_sha256": metadata.archive_sha256,
        "cache_key": metadata.cache_key,
    }


def _archive_metadata_from_json(data: object, path: Path) -> AebArchiveMetadata:
    mapping = _require_mapping(data, path)
    return AebArchiveMetadata(
        source_url=_require_string(mapping, "source_url", path),
        observed_at=_require_string(mapping, "observed_at", path),
        last_modified=_optional_string(mapping, "last_modified", path),
        etag=_optional_string(mapping, "etag", path),
        content_length=_optional_int(mapping, "content_length", path),
        archive_sha256=_require_string(mapping, "archive_sha256", path),
        cache_key=_require_string(mapping, "cache_key", path),
    )


def _load_archive_metadata(path: Path) -> AebArchiveMetadata | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _archive_metadata_from_json(cast(object, data), path)
    except (AebFetchError, OSError, json.JSONDecodeError):
        return None


def _catalog_to_json(catalog: AebCatalog) -> dict[str, object]:
    return {
        "archive_sha256": catalog.archive_sha256,
        "files_scanned": catalog.files_scanned,
        "valid_games": catalog.valid_games,
        "skipped": catalog.skipped,
        "failed": catalog.failed,
        "moves_replayed": catalog.moves_replayed,
        "skipped_by_reason": [
            {"reason": skip_count.reason.value, "count": skip_count.count}
            for skip_count in catalog.skipped_by_reason
        ],
        "entries": [
            {
                "source_path": entry.source_path,
                "move_count": entry.move_count,
                "byte_count": entry.byte_count,
                "sha256": entry.sha256,
            }
            for entry in catalog.entries
        ],
        "failures": list(catalog.failures),
    }


def _catalog_from_json(data: object, path: Path) -> AebCatalog:
    mapping = _require_mapping(data, path)
    skipped_by_reason = tuple(
        _skip_count_from_json(item, path)
        for item in _require_list(mapping, "skipped_by_reason", path)
    )
    entries = tuple(
        _catalog_entry_from_json(item, path)
        for item in _require_list(mapping, "entries", path)
    )
    return AebCatalog(
        archive_sha256=_require_string(mapping, "archive_sha256", path),
        files_scanned=_require_int(mapping, "files_scanned", path),
        valid_games=_require_int(mapping, "valid_games", path),
        skipped=_require_int(mapping, "skipped", path),
        failed=_require_int(mapping, "failed", path),
        moves_replayed=_require_int(mapping, "moves_replayed", path),
        skipped_by_reason=skipped_by_reason,
        entries=entries,
        failures=tuple(
            _require_string_item(item, "failures", path)
            for item in _require_list(mapping, "failures", path)
        ),
    )


def _skip_count_from_json(data: object, path: Path) -> AebSkipCount:
    mapping = _require_mapping(data, path)
    reason_value = _require_string(mapping, "reason", path)
    try:
        reason = ReplaySkipReason(reason_value)
    except ValueError as exc:
        msg = f"{path}: unsupported skip reason {reason_value!r}"
        raise AebFetchError(msg) from exc
    return AebSkipCount(
        reason=reason,
        count=_require_int(mapping, "count", path),
    )


def _catalog_entry_from_json(data: object, path: Path) -> AebCatalogEntry:
    mapping = _require_mapping(data, path)
    source_path = _require_string(mapping, "source_path", path)
    move_count = _require_int(mapping, "move_count", path)
    if move_count < 1:
        msg = f"{path}: catalog entry {source_path!r} has no trainable moves"
        raise AebFetchError(msg)
    return AebCatalogEntry(
        source_path=source_path,
        move_count=move_count,
        byte_count=_require_int(mapping, "byte_count", path),
        sha256=_require_string(mapping, "sha256", path),
    )


def _load_catalog(path: Path) -> AebCatalog:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"unable to load AEB catalog {path}: {exc}"
        raise AebFetchError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"invalid AEB catalog JSON {path}: {exc}"
        raise AebFetchError(msg) from exc
    return _catalog_from_json(cast(object, data), path)


def _write_json_atomic(path: Path, data: dict[str, object]) -> None:
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
            temp_file.write(json.dumps(data, indent=2, sort_keys=True))
            temp_file.write("\n")
        temp_path.replace(path)
    except OSError as exc:
        msg = f"unable to write {path}: {exc}"
        raise AebFetchError(msg) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def _require_mapping(data: object, path: Path) -> dict[str, object]:
    if not isinstance(data, dict):
        msg = f"{path}: expected object"
        raise AebFetchError(msg)
    return {str(key): cast(object, value) for key, value in data.items()}


def _require_string(mapping: dict[str, object], key: str, path: Path) -> str:
    value = _require_key(mapping, key, path)
    if not isinstance(value, str):
        msg = f"{path}: {key} must be a string"
        raise AebFetchError(msg)
    return value


def _require_string_item(value: object, key: str, path: Path) -> str:
    if not isinstance(value, str):
        msg = f"{path}: {key} entries must be strings"
        raise AebFetchError(msg)
    return value


def _optional_string(mapping: dict[str, object], key: str, path: Path) -> str | None:
    value = _require_key(mapping, key, path)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{path}: {key} must be a string or null"
        raise AebFetchError(msg)
    return value


def _optional_int(mapping: dict[str, object], key: str, path: Path) -> int | None:
    value = _require_key(mapping, key, path)
    if value is None:
        return None
    if not isinstance(value, int):
        msg = f"{path}: {key} must be an integer or null"
        raise AebFetchError(msg)
    return value


def _require_int(mapping: dict[str, object], key: str, path: Path) -> int:
    value = _require_key(mapping, key, path)
    if not isinstance(value, int):
        msg = f"{path}: {key} must be an integer"
        raise AebFetchError(msg)
    return value


def _require_list(mapping: dict[str, object], key: str, path: Path) -> list[object]:
    value = _require_key(mapping, key, path)
    if not isinstance(value, list):
        msg = f"{path}: {key} must be a list"
        raise AebFetchError(msg)
    return cast(list[object], value)


def _require_key(mapping: dict[str, object], key: str, path: Path) -> object:
    try:
        return mapping[key]
    except KeyError as exc:
        msg = f"{path}: missing required field: {key}"
        raise AebFetchError(msg) from exc
