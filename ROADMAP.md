# Go Move Predictor Roadmap

This project builds a supervised policy model for Go. Given a 19x19 board
position, the model predicts the next move a human player is likely to make.

The project is intentionally scoped as policy imitation rather than full game
playing. It uses historical SGF games as the source of training data, replays
each game into intermediate board states, and trains models to predict the move
that was actually played from each state.

The roadmap is organized around working milestones. The intent is to keep the
data pipeline, evaluation, and baselines trustworthy before spending much time
on larger model architectures.

## Project Scope

The core task is supervised next-move prediction:

```text
input: board position before move n
target: human move played at move n
```

For an initial 19x19 board setup, the target space is 361 board intersections.
Pass can be added as a 362nd class later, but the first implementation should
keep that decision explicit rather than hidden in the data pipeline.

This is not intended to be a Go engine. The model is not optimizing for game
outcome, and it will not use search, self-play, or reinforcement learning in
the first pass.

## Success Criteria

A complete first version should include:

- SGF parsing for a constrained set of valid games
- board-state replay with capture and legality handling
- tensor encoding for board positions
- game-level train, validation, and test splits
- random, frequency, and MLP baselines
- a CNN policy model
- top-1, top-3, and top-5 accuracy
- illegal-move tracking
- evaluation broken down by game phase
- qualitative examples showing model predictions on real boards

This is enough to make the project a complete ML workflow. ResNets,
transformers, augmentation, and a richer demo are useful follow-up work, but
they should build on a validated pipeline rather than replace it.

## Non-Goals

The following are deliberately out of scope for the first version:

- reinforcement learning
- self-play
- Monte Carlo Tree Search
- value networks
- tsumego solving
- game outcome prediction
- online play against humans
- superhuman move search

Keeping these out of scope keeps the project focused on supervised learning,
dataset quality, and evaluation.

## Technical Framing

The learning problem is multiclass classification over board intersections.
The model receives a board tensor and outputs one logit per legal or potentially
legal move location.

The first useful representation should be compact but not toy-level:

| Channel | Meaning |
| ---: | --- |
| 0 | Current player's stones |
| 1 | Opponent's stones |
| 2 | Empty points |
| 3 | Last move location |
| 4 | Legal move mask |

Representing the board from the current player's perspective should simplify
the learning problem because the model does not need separate black and white
versions of the same local pattern.

Initial labels can be encoded as a flat index:

```text
label = row * 19 + column
```

Legal move masking should be tracked from the start, even if training initially
uses a plain cross-entropy objective. During evaluation, illegal locations can
be masked before computing top-k predictions.

## Phase 1: Get One Game Working

The first milestone is a small parser spike. The goal is to load one known-good
SGF file, inspect its metadata, and verify the coordinate mapping.

Build:

- SGF loading experiment
- move parser
- coordinate conversion between SGF and board indices
- basic handling for board size, player color, and pass moves

Done when:

- one known-good SGF file loads consistently
- moves can be printed in board coordinates
- the coordinate convention is documented in code or tests

## Phase 2: Replay Games Correctly

After parsing moves, the project needs a reliable board replay layer. This is
the first part where Go-specific correctness matters.

Build:

- board representation
- move application
- capture detection
- legal move checks
- logging for bad or unsupported games

Initial game filters:

- 19x19 boards only
- no handicap stones
- no setup stones
- legal move sequence
- pass moves handled explicitly or skipped intentionally
- malformed SGF files skipped with diagnostics

Done when:

- around 100 games can be replayed without illegal move errors
- invalid games are skipped with useful logs
- spot checks show plausible board states after captures and ordinary moves

## Phase 3: Turn Positions Into Training Data

Each move in a replayed game becomes one supervised training example. The board
before the move is the input, and the human move is the label.

Build:

- board tensor encoder
- move label encoder
- legal move mask generation
- processed dataset writer
- sanity checks for tensor shapes, labels, and legal masks

Initial processed record:

| Field | Purpose |
| --- | --- |
| `x` | Board tensor with shape `[channels, 19, 19]` |
| `y` | Move label as a flat board index |
| `game_id` | Source game identifier |
| `move_number` | Move index inside the game |
| `legal_mask` | Legal board points for the position |

Storage can start with `.npz` files or PyTorch tensors. If dataset size becomes
an issue, the loader can move to chunked storage or lazy reads later.

Done when:

- a small processed dataset can be generated repeatably
- labels can be decoded back into board coordinates
- sample tensors can be inspected against the original SGF replay

## Phase 4: Split the Dataset Without Leaking Games

The split must happen by game, not by individual position. Adjacent positions
from the same game are highly correlated, so a random position-level split would
make validation and test results too optimistic.

Initial split:

| Split | Percentage of games |
| --- | ---: |
| Train | 80 percent |
| Validation | 10 percent |
| Test | 10 percent |

Build:

- deterministic game-level split
- PyTorch Dataset class
- DataLoader setup
- optional lazy loading path for larger datasets

Done when:

- no game appears in more than one split
- the split can be reproduced with a fixed seed
- model training can run directly from the DataLoader

## Phase 5: Build Baselines Before the CNN

Baselines establish whether later neural models are learning useful spatial
structure or only reproducing simple move frequencies.

Baselines:

- random legal move
- most common moves overall
- most common moves by game phase
- MLP over a flattened board tensor

Game phase buckets:

| Phase | Move numbers |
| --- | --- |
| Opening | 1-40 |
| Middle game | 41-150 |
| Endgame | 151+ |

Done when:

- top-k metrics work for all baselines
- random, frequency, and MLP results are in one table
- the CNN has a clear baseline target to beat

## Phase 6: Train the First CNN

The first main model should be a straightforward CNN policy network. The goal
is to prove the end-to-end supervised pipeline before adding architectural
complexity.

Starting architecture:

```text
input: [5, 19, 19]
3x3 convolution layers with padding
batch normalization
ReLU activations
1x1 policy head
361 output logits
```

Training setup:

- cross-entropy loss
- AdamW optimizer
- batch size around 128 to start
- validation loss and top-k metrics per epoch
- checkpointing for the best validation run
- legal move masking during evaluation

Done when:

- the CNN trains without shape or label bugs
- training and validation curves are recorded
- the CNN beats at least the random and frequency baselines
- checkpoints can be reloaded for evaluation

## Phase 7: Make Evaluation More Honest

Exact next-move prediction is a strict target because more than one move can be
reasonable in a Go position. Top-k accuracy is a better fit than top-1 alone.

Primary metrics:

| Metric | Why it matters |
| --- | --- |
| Top-1 accuracy | Direct next-move match |
| Top-3 accuracy | Captures close alternatives |
| Top-5 accuracy | Better reflects plausible candidate moves |
| Cross-entropy loss | Tracks confidence and calibration during training |
| Illegal move rate | Catches model or masking failures |

Evaluation should also be reported by game phase:

- opening
- middle game
- endgame

Done when:

- model selection uses validation data only
- final metrics are reported on held-out test games
- each model is compared with the same metric set

## Phase 8: Look at the Mistakes

Qualitative error analysis is important because aggregate accuracy can hide
whether the model is making plausible Go moves.

Analysis views:

- heatmap of human moves
- heatmap of model predictions
- high-confidence incorrect predictions
- examples where the human move was in the top 5
- examples grouped by opening, middle game, and endgame

Error categories:

- reasonable alternate move
- occupied or illegal point
- common opening pattern confused
- local tactic missed
- global direction missed
- endgame precision issue

Done when:

- several representative board examples are included in the report
- common failure modes are described with examples
- the results section goes beyond a metrics table

## Phase 9: Try One Stronger Model

After the CNN and evaluation path are stable, the next model should be a small
ResNet. This keeps the experiment close to the CNN baseline while testing
whether deeper spatial feature extraction helps.

Small ResNet shape:

```text
input convolution
4 residual blocks
policy head
```

A transformer can remain a stretch experiment. If it is added, it should be
small and treated as a comparison model rather than a replacement for the CNN
pipeline.

Done when:

- one stronger model is trained on the same split
- results are compared against the same baselines and CNN metrics
- the added complexity is justified or rejected based on results

## Phase 10: Demo and Final Polish

The final project should include a simple way to inspect predictions. A
notebook is enough for the first version; a CLI adds a useful reproducible path.

Demo capabilities:

- load an SGF file
- select a move number
- show the board before that move
- mark the human move
- mark the model's top-k predictions
- print predicted probabilities

Example CLI:

```bash
python src/predict.py --sgf data/raw/example.sgf --move-number 87
```

Done when:

- the README has reproducible setup, preprocessing, training, and evaluation
  commands
- the report includes results, figures, limitations, and next steps
- the demo can show several interesting positions from held-out games

## Milestone Data Scale

The dataset should grow in stages so parser, replay, and encoding bugs stay
visible.

| Stage | Games | Rough position count | Purpose |
| --- | ---: | ---: | --- |
| Parser test | 1-10 | Hundreds | Find obvious SGF and replay bugs |
| Small prototype | 50-100 | 7,500-25,000 | Test encoding, splits, and loaders |
| First real run | About 1,000 | 150,000-250,000 | Train baselines and first CNN |
| Bigger run | 5,000-10,000 | 750,000-2,000,000 | Improve CNN or ResNet performance |
| Stretch scale | 50,000+ | 7,500,000+ | Scaling experiment after the pipeline is stable |

The project should not depend on the largest dataset. Smaller runs are useful
milestones because they make correctness and performance issues easier to
debug.

## Data Augmentation

Go boards have eight geometric symmetries: four rotations and four reflected
versions. These can be used to expand the training set once the first CNN is
working.

Augmentation rules:

- split by game before augmenting
- augment only the training split
- apply the same transform to the board tensor and the move label
- verify transformed labels by decoding them back to board coordinates

Augmentation should wait until the base pipeline is stable. It improves sample
efficiency, but it also makes data bugs harder to inspect.

## Expected Artifacts

By the end of the first complete pass, the repository should contain:

| Artifact | Purpose |
| --- | --- |
| SGF parser and replay code | Converts raw games into board states |
| Processed dataset builder | Produces tensors, labels, metadata, and masks |
| Dataset split manifest | Records train, validation, and test game IDs |
| PyTorch Dataset/DataLoader | Provides repeatable model input |
| Baseline implementations | Establishes comparison points |
| CNN policy model | Main supervised model |
| Evaluation script | Computes shared metrics for every model |
| Visualization notebook or CLI | Shows predictions on real board positions |
| Results table | Summarizes model performance |

## Main Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| SGF parsing is messier than expected | Replay failures or inconsistent examples | Start with one known-good source, filter aggressively, and log skipped games |
| Board replay has subtle rule bugs | Invalid training labels or misleading metrics | Add small replay tests, inspect captures manually, and keep legality checks explicit |
| Dataset size grows too quickly | Slow preprocessing or memory pressure | Start around 1,000 games, save processed tensors, and add lazy loading only when needed |
| Model predicts illegal moves | Bad qualitative output and inflated mistakes | Include legal masks, track illegal move rate, and mask logits during evaluation |
| Evaluation is too optimistic | Results do not reflect generalization | Split by game, keep augmented positions out of validation/test, and reserve the test set for final evaluation |
| Advanced models distract from the core project | Time spent on architecture before the data is trustworthy | Finish parser, replay, baselines, CNN, and error analysis before trying transformers |
| Accuracy looks low | Results may appear weak despite useful learning | Report top-k metrics, compare against baselines, and include qualitative examples |

## Working Order

1. Parse one SGF game and inspect the moves.
2. Replay a small batch of SGF games.
3. Build board tensors, move labels, and legal masks.
4. Save a small processed dataset.
5. Split by game and build the PyTorch loader.
6. Implement top-k metrics.
7. Run random and frequency baselines.
8. Train the MLP baseline.
9. Train the first CNN.
10. Evaluate by game phase.
11. Add visualizations for predictions and mistakes.
12. Try a small ResNet if the CNN pipeline is stable.
13. Add augmentation once the basic model is working.
14. Polish the README, report, and demo.

The project should earn each step with working code. Once parsing, replay,
splits, and baselines are solid, the later model experiments will be much
easier to trust.
