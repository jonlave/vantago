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

## CLI usage

Inspect the committed known-good SGF fixture:

```bash
uv run vantago inspect-sgf tests/fixtures/sgf/known_good.sgf
```

Replay the committed 100-game public-domain SGF corpus and print diagnostics:

```bash
uv run vantago replay-batch data/raw/aeb-small-100
```

Encode replayed positions into a processed NumPy dataset artifact:

```bash
uv run vantago process-dataset data/raw/aeb-small-100 data/processed/aeb-small-100.npz
```

Inspect one processed record and verify its decoded label, tensor shape, and
legal-mask status:

```bash
uv run vantago inspect-dataset data/processed/aeb-small-100.npz --index 0
```

Create a deterministic game-level train/validation/test split manifest:

```bash
uv run vantago split-dataset data/processed/aeb-small-100.npz data/processed/aeb-small-100-splits.json --seed 0
```

`split-dataset` reads unique `game_id` values from the processed artifact,
shuffles them with the provided seed, and writes fixed 80/10/10 split metadata
without splitting positions from the same game across train, validation, and
test.

Internal board coordinates are 0-based top-left row/column coordinates. For
example, SGF `aa` is `row=0 col=0`, and flat labels use
`row * board_size + col`.

Evaluate random legal, overall frequency, and phase-aware frequency baselines
on a split:

```bash
uv run vantago evaluate-baselines data/processed/aeb-small-100.npz data/processed/aeb-small-100-splits.json --split validation --seed 0
```
