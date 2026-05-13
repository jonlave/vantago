from __future__ import annotations

from pathlib import Path

import pytest

from vantago.cli.main import main

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sgf" / "known_good.sgf"


def test_inspect_cli_prints_stable_metadata_and_moves(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["inspect-sgf", str(FIXTURE_PATH)])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f"source: {FIXTURE_PATH}\n"
        "board_size: 19\n"
        "black_player: Black Test\n"
        "white_player: White Test\n"
        "komi: 6.5\n"
        "result: B+R\n"
        "moves:\n"
        "1: B row=0 col=0\n"
        "2: W row=0 col=18\n"
        "3: B row=18 col=0\n"
        "4: W row=3 col=3\n"
        "5: B PASS\n"
    )
