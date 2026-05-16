from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from vantago.cli.main import main
from vantago.data import load_dataset_split_manifest, load_processed_dataset
from vantago.data.aeb import (
    AebFetchConfig,
    AebFetchError,
    AebSkipCount,
    build_aeb_catalog,
    fetch_aeb_games,
)
from vantago.replay import ReplaySkipReason

SGF_FILE_FORMAT = 4
SGF_GO_GAME_TYPE = 1
SUPPORTED_BOARD_SIZE = 19
SMALL_BOARD_SIZE = 9


def test_build_aeb_catalog_records_valid_games_and_skip_reasons(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "games.tgz"
    _write_tgz(
        archive_path,
        {
            "./games/valid/1.sgf": _sgf(sequence=";B[aa];W[sa]"),
            "games/pass/1.sgf": _sgf(sequence=";B[aa];W[]"),
            "games/empty/1.sgf": _sgf(sequence=""),
            "games/small/1.sgf": _sgf(
                board_size=SMALL_BOARD_SIZE,
                sequence=";B[aa]",
            ),
            "games/illegal/1.sgf": _sgf(sequence=";B[aa];W[aa]"),
            "games/malformed/1.sgf": "not an sgf",
            "games/notes.txt": "ignore me",
        },
    )

    catalog = build_aeb_catalog(archive_path)

    assert catalog.files_scanned == 6
    assert catalog.valid_games == 1
    assert catalog.skipped == 5
    assert catalog.failed == 0
    assert catalog.moves_replayed == 2
    assert catalog.entries[0].source_path == "games/valid/1.sgf"
    assert catalog.entries[0].move_count == 2
    assert _skip_counts(catalog.skipped_by_reason) == {
        ReplaySkipReason.PASS_MOVE: 1,
        ReplaySkipReason.EMPTY_GAME: 1,
        ReplaySkipReason.NON_19X19_BOARD: 1,
        ReplaySkipReason.ILLEGAL_MOVE_SEQUENCE: 1,
        ReplaySkipReason.MALFORMED_SGF: 1,
    }


def test_build_aeb_catalog_reports_periodic_progress(tmp_path: Path) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=5)
    messages: list[str] = []

    catalog = build_aeb_catalog(
        archive_path,
        progress_callback=messages.append,
        progress_interval=2,
    )

    assert catalog.valid_games == 5
    assert messages == [
        "cataloged 2 SGF files (2 valid, 0 skipped, 0 failed)",
        "cataloged 4 SGF files (4 valid, 0 skipped, 0 failed)",
    ]


def test_fetch_aeb_games_samples_deterministically_and_writes_manifest(
    tmp_path: Path,
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=12)
    cache_dir = tmp_path / "cache"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = fetch_aeb_games(
        AebFetchConfig(
            games=4,
            seed=7,
            output_dir=first_output,
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )
    second = fetch_aeb_games(
        AebFetchConfig(
            games=4,
            seed=7,
            output_dir=second_output,
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )

    first_sources = [entry.source_path for entry in first.selected_games]
    second_sources = [entry.source_path for entry in second.selected_games]
    assert first_sources == second_sources
    assert first_sources == sorted(first_sources)
    assert first.valid_games_available == 12
    assert first.skipped_by_reason == ()

    sgf_files = sorted(path.name for path in (first_output / "sgf").glob("*.sgf"))
    assert sgf_files == [
        "aeb-fetch-0001.sgf",
        "aeb-fetch-0002.sgf",
        "aeb-fetch-0003.sgf",
        "aeb-fetch-0004.sgf",
    ]
    manifest = (first_output / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "games_requested: 4\n" in manifest
    assert "valid_games_available: 12\n" in manifest
    assert "selection_seed: 7\n" in manifest
    assert "id\tfile\tsource_path\tmoves\tbytes\tsha256\n" in manifest
    assert first_sources[0] in manifest
    assert (first_output / "README.md").exists()
    assert first.archive_path == second.archive_path
    assert first.catalog_path == second.catalog_path


def test_fetch_aeb_games_rejects_existing_output_and_too_many_games(
    tmp_path: Path,
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=2)
    cache_dir = tmp_path / "cache"
    existing_output = tmp_path / "existing"
    existing_output.mkdir()

    with pytest.raises(AebFetchError, match="already exists"):
        fetch_aeb_games(
            AebFetchConfig(
                games=1,
                seed=0,
                output_dir=existing_output,
                archive_url=archive_path.as_uri(),
                cache_dir=cache_dir,
            )
        )

    with pytest.raises(AebFetchError, match="only 2 replay-valid games"):
        fetch_aeb_games(
            AebFetchConfig(
                games=3,
                seed=0,
                output_dir=tmp_path / "too-many",
                archive_url=archive_path.as_uri(),
                cache_dir=cache_dir,
            )
        )


def test_fetch_aeb_games_redownloads_corrupt_cached_archive(
    tmp_path: Path,
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=3)
    cache_dir = tmp_path / "cache"

    first = fetch_aeb_games(
        AebFetchConfig(
            games=2,
            seed=1,
            output_dir=tmp_path / "first",
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )
    first.archive_path.write_bytes(b"corrupt")

    second = fetch_aeb_games(
        AebFetchConfig(
            games=2,
            seed=1,
            output_dir=tmp_path / "second",
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )

    assert (
        second.archive_metadata.archive_sha256
        == first.archive_metadata.archive_sha256
    )
    assert second.archive_path.read_bytes() == archive_path.read_bytes()


def test_fetch_aeb_games_rebuilds_corrupt_cached_catalog(
    tmp_path: Path,
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=4)
    cache_dir = tmp_path / "cache"

    first = fetch_aeb_games(
        AebFetchConfig(
            games=2,
            seed=1,
            output_dir=tmp_path / "first",
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )
    first.catalog_path.write_text("not json", encoding="utf-8")

    second = fetch_aeb_games(
        AebFetchConfig(
            games=2,
            seed=1,
            output_dir=tmp_path / "second",
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )

    assert second.catalog_path == first.catalog_path
    assert second.valid_games_available == 4
    assert "not json" not in second.catalog_path.read_text(encoding="utf-8")


def test_fetch_aeb_games_rebuilds_cached_catalog_with_empty_entries(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "games.tgz"
    _write_tgz(
        archive_path,
        {
            "games/empty.sgf": _sgf(sequence=""),
            "games/valid.sgf": _sgf(sequence=";B[aa]"),
        },
    )
    cache_dir = tmp_path / "cache"

    first = fetch_aeb_games(
        AebFetchConfig(
            games=1,
            seed=0,
            output_dir=tmp_path / "first",
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )
    first.catalog_path.write_text(
        json.dumps(
            {
                "archive_sha256": first.archive_metadata.archive_sha256,
                "files_scanned": 2,
                "valid_games": 2,
                "skipped": 0,
                "failed": 0,
                "moves_replayed": 1,
                "skipped_by_reason": [],
                "entries": [
                    {
                        "source_path": "games/empty.sgf",
                        "move_count": 0,
                        "byte_count": 1,
                        "sha256": "stale",
                    },
                    {
                        "source_path": "games/valid.sgf",
                        "move_count": 1,
                        "byte_count": 1,
                        "sha256": "stale",
                    },
                ],
                "failures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    second = fetch_aeb_games(
        AebFetchConfig(
            games=1,
            seed=0,
            output_dir=tmp_path / "second",
            archive_url=archive_path.as_uri(),
            cache_dir=cache_dir,
        )
    )

    assert second.catalog_path == first.catalog_path
    assert second.valid_games_available == 1
    assert second.selected_games[0].source_path == "games/valid.sgf"
    assert _skip_counts(second.skipped_by_reason) == {
        ReplaySkipReason.EMPTY_GAME: 1,
    }
    assert '"move_count": 0' not in second.catalog_path.read_text(encoding="utf-8")


def test_fetch_aeb_games_cli_prints_summary_and_writes_corpus(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=3)
    output = tmp_path / "raw"

    exit_code = main(
        [
            "fetch-aeb-games",
            "--games",
            "2",
            "--seed",
            "0",
            "--output",
            str(output),
            "--archive-url",
            archive_path.as_uri(),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert f"source_url: {archive_path.as_uri()}\n" in captured.out
    assert f"output: {output}\n" in captured.out
    assert "games_requested: 2\n" in captured.out
    assert "selected_files: 2\n" in captured.out
    assert "skipped_by_reason:\n  none\n" in captured.out
    assert len(list((output / "sgf").glob("*.sgf"))) == 2
    assert (output / "MANIFEST.txt").exists()


def test_prepare_aeb_dataset_cli_rejects_too_few_games_before_fetching(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=9)
    raw_output = tmp_path / "raw"
    dataset_output = tmp_path / "processed" / "tiny.npz"
    splits_output = tmp_path / "processed" / "tiny-splits.json"

    exit_code = main(
        [
            "prepare-aeb-dataset",
            "--games",
            "9",
            "--archive-url",
            archive_path.as_uri(),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--raw-output",
            str(raw_output),
            "--dataset-output",
            str(dataset_output),
            "--splits-output",
            str(splits_output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "requires at least 10 games" in captured.err
    assert not raw_output.exists()
    assert not dataset_output.exists()
    assert not splits_output.exists()
    assert not (tmp_path / "cache").exists()


def test_prepare_aeb_dataset_cli_validates_outputs_before_fetching(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=10)
    raw_output = tmp_path / "raw"
    dataset_output = tmp_path / "processed" / "tiny"
    splits_output = tmp_path / "processed" / "tiny-splits.json"

    exit_code = main(
        [
            "prepare-aeb-dataset",
            "--games",
            "10",
            "--archive-url",
            archive_path.as_uri(),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--raw-output",
            str(raw_output),
            "--dataset-output",
            str(dataset_output),
            "--splits-output",
            str(splits_output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "processed dataset output path must end with .npz" in captured.err
    assert not raw_output.exists()
    assert not dataset_output.exists()
    assert not splits_output.exists()
    assert not (tmp_path / "cache").exists()


def test_prepare_aeb_dataset_cli_fetches_processes_and_splits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_path = _write_valid_archive(tmp_path / "games.tgz", games=10)
    raw_output = tmp_path / "raw"
    dataset_output = tmp_path / "processed" / "tiny.npz"
    splits_output = tmp_path / "processed" / "tiny-splits.json"

    exit_code = main(
        [
            "prepare-aeb-dataset",
            "--games",
            "10",
            "--seed",
            "0",
            "--name",
            "tiny",
            "--archive-url",
            archive_path.as_uri(),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--raw-output",
            str(raw_output),
            "--dataset-output",
            str(dataset_output),
            "--splits-output",
            str(splits_output),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert f"raw_output: {raw_output}\n" in captured.out
    assert f"dataset: {dataset_output}\n" in captured.out
    assert f"splits: {splits_output}\n" in captured.out
    assert "records_written: 10\n" in captured.out
    assert "games_total: 10\n" in captured.out

    artifact = load_processed_dataset(dataset_output)
    manifest = load_dataset_split_manifest(splits_output)
    assert artifact.y.shape == (10,)
    assert manifest.game_counts["total"] == 10
    assert manifest.record_counts["total"] == 10


def _write_valid_archive(path: Path, *, games: int) -> Path:
    _write_tgz(
        path,
        {
            f"games/Event/{index:02d}.sgf": _sgf(sequence=";B[aa]")
            for index in range(games)
        },
    )
    return path


def _write_tgz(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, content in sorted(members.items()):
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))


def _sgf(
    *,
    board_size: int = SUPPORTED_BOARD_SIZE,
    sequence: str,
) -> str:
    return (
        f"(;FF[{SGF_FILE_FORMAT}]GM[{SGF_GO_GAME_TYPE}]"
        f"SZ[{board_size}]{sequence})"
    )


def _skip_counts(
    skipped_by_reason: tuple[AebSkipCount, ...],
) -> dict[ReplaySkipReason, int]:
    return {
        skip_count.reason: skip_count.count
        for skip_count in skipped_by_reason
    }
