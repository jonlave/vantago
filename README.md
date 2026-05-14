# VantaGo

ML project for training a supervised policy model that predicts human Go moves
from board positions.

See [ROADMAP.md](ROADMAP.md) for the project scope, milestones, evaluation
plan, and known risks.

## Development

This repo uses `uv` with a `src/` Python package layout. To install dependencies
and run the first SGF parser checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
```

Inspect the committed known-good SGF fixture with the top-level CLI:

```bash
uv run vantago inspect-sgf tests/fixtures/sgf/known_good.sgf
```

Replay the committed 100-game public-domain SGF corpus and print diagnostics:

```bash
uv run vantago replay-batch data/raw/aeb-small-100
```

Internal board coordinates are 0-based top-left row/column coordinates. For
example, SGF `aa` is `row=0 col=0`, and future flat labels can use
`row * board_size + col`.
