# rl-chess

Learning-first reinforcement learning loops for chess, implemented by hand around `python-chess`.

## What is here

- `rl_chess.env`: tiny `python-chess.Board` wrapper plus Unicode board diagram conversion helpers.
- `rl_chess.nn_model.PolicyValueNet`: small policy/value network over 12 piece planes plus side-to-move.
- `rl_chess.puct_mcts.PUCTMCTS`: hand-written neural-net-guided PUCT search over legal UCI moves.
- `rl_chess.self_play.play_self_game`: one AlphaZero-style self-play game that records visit-count policy targets and terminal value targets.
- `rl_chess.train.train`: replay-buffered policy/value training loop with optional checkpointing.
- `rl_chess.validation`: model-vs-Stockfish evaluation helpers.
- `rl_chess.endgame_validation`: narrow KQK value-head overfit/greedy-conversion diagnostic.
- `rl_chess.modal_app`: Modal entrypoints that call the same local training and validation code remotely.

## Direction

The AlphaGo lesson for this project is concise:

> Search improves the model; the model improves search.

We keep the board inspectable with Unicode chess diagrams instead of exposing compact chess notation to the RL loop. The next milestone is MCTS self-play training: run search at each board, store the visit-count policy as a better per-move target, then train a small policy/value learner from those examples. Local training remains the source of truth; Modal only scales the same loop remotely.

See `docs/alphago-from-scratch-lessons.md` for the Dwarkesh/Eric Jang AlphaGo-from-scratch notes, and `docs/plans/2026-05-25-rl-mcts-self-play-modal.md` for the implementation plan.

## Local training

Smoke run:

```bash
uv run rl-chess --iterations 1 --max-plies 1 --mcts-iterations 2 --seed 123
```

`--max-plies` is a safety cap, not a training truncation mechanism. If a game reaches the cap while non-terminal, the run raises instead of converting the unfinished game into a draw target. Omit the flag, or pass `0`, for uncapped self-play that runs until `python-chess` reports a terminal result.

Meaningful first run:

```bash
uv run rl-chess \
  --first-meaningful-run \
  --checkpoint-dir runs/first-meaningful/checkpoints \
  --stockfish-path ~/.local/bin/stockfish \
  --seed 123
```

The first-run preset is deliberately still small enough for local iteration but no longer a one-batch smoke test: 3 iterations, 2 self-play games per iteration, 32 PUCT simulations per move, 4 training updates per iteration, replay capacity 5,000, checkpoint after every iteration, and 4 validation games against the Stockfish UCI Elo floor.

## Modal training

```bash
uv run modal run src/rl_chess/modal_app.py --episodes 1000 --seed 123
```

The Modal app runs the same `train()` loop remotely, so local and remote execution share one core implementation.

## Endgame value validation

The endgame value-validation loop is a narrow, deterministic check that the model can learn terminal-backed value targets before we scale self-play. It builds a small dataset from ten KQK forced-mate positions, trains only the value head, and then checks whether a one-ply value-greedy player can convert the positions within the ply cap.

Local smoke:

```bash
uv run pytest tests/test_core.py::test_endgame_value_validation_can_overfit_tiny_model_smoke -q
```

Remote Modal validation:

```bash
uv run modal run src/rl_chess/modal_app.py --validate-endgames --endgame-steps 800 --seed 123
```

## Tests

```bash
uv run pytest -q
```

## Research log

### 2026-05-27 18:02:05 UTC — README and agent research-log convention

- Added `AGENTS.md` with the standing repo instruction that all changes, validation runs, training experiments, and design decisions must be recorded in this research log with date/time, commands, metrics, and artifact paths.
- Updated README to document the endgame value-validation loop and how to run it locally/remotely.

### 2026-05-27 18:02:05 UTC — Narrow and broad learning validation status

- Narrow validation: added the endgame value-validation loop over ten KQK forced-mate positions. The loop verifies that the model can reduce value MSE on terminal-backed targets and that a value-greedy policy can be evaluated from those learned values.
- Broad validation: the existing NN-guided PUCT self-play loop, replay buffer, policy/value training path, checkpointing, and weak-Stockfish validation path are covered by the test suite and first-meaningful-run preset.
- Latest full unit/integration suite before this documentation update: `uv run pytest -q` completed with `24 passed, 1 warning`.

### 2026-05-27 18:03:54 UTC — Documentation update verification

- Command: `uv run pytest -q`
- Result: passed (`24 passed, 1 warning in 61.67s`).
- Warning: Modal local-entrypoint smoke warns that local execution does not access mounted remote volume data; expected for the local test path.

### 2026-05-27 18:07:00 UTC — First full remote training/validation run

- Command: `uv run modal run src/rl_chess/modal_app.py --first-meaningful-run --seed 123`
- Modal run: https://modal.com/apps/koogle-frick/main/ap-5c0RM2iYhmle35dMOe3ICn
- Result: completed successfully as an execution run, but failed the Stockfish validation gate.
- Training: `iterations=3`, `games=6`, `examples=653`, `terminal_games=2`, `truncated_games=4`, `replay_size=653`.
- Final losses: total `3.581167`, policy `3.308196`, value `0.272971`.
- Checkpoints: `/checkpoints/first-meaningful-run/iteration-0001.pt`, `/checkpoints/first-meaningful-run/iteration-0002.pt`, `/checkpoints/first-meaningful-run/iteration-0003.pt`.
- Stockfish validation: Elo `1320`, `validation_games=4`, `wins=0`, `losses=4`, `draws=0`, `score=0.000`, `validation_passed=False`.
- Interpretation: the loop runs end-to-end and produces checkpoints, but this first meaningful preset is still too weak to score against Stockfish's supported Elo floor.

### 2026-05-27 18:25:30 UTC — Removed self-play truncation as a learning target

- Correction: reverted the mistaken side-to-move input-plane removal. The model again receives the side-to-move plane.
- Change: removed truncation from self-play/training metrics and checkpoint summaries. Self-play now either reaches a terminal `python-chess` result or raises if an optional safety `max_plies` cap is hit while non-terminal.
- Change: first-meaningful training preset now uses uncapped self-play (`max_plies=None`). CLI/Modal `--max-plies` remains only as a safety cap; `0`/omitted means no cap.
- Change: added temporary compact starting-position support for deterministic smoke tests without relying on artificial truncation. This was later removed because diagnostic positions should use ASCII board diagrams.
- TDD red command: `uv run pytest tests/test_core.py::test_self_play_rejects_safety_cap_instead_of_truncating_game tests/test_core.py::test_training_metrics_do_not_report_truncation tests/test_core.py::test_first_meaningful_run_is_bigger_than_smoke_but_bounded -q`
- Red result: failed as expected because capped non-terminal self-play still returned a truncated game, `train()` did not accept `starting_board`, and `FIRST_MEANINGFUL_RUN.max_plies` was still `120`.
- Targeted green command: `uv run pytest tests/test_core.py::test_self_play_rejects_safety_cap_instead_of_truncating_game tests/test_core.py::test_training_metrics_do_not_report_truncation tests/test_core.py::test_first_meaningful_run_is_bigger_than_smoke_but_bounded tests/test_core.py::test_cli_smoke tests/test_core.py::test_modal_remote_training_entrypoint_can_run_tiny_local_smoke -q`
- Targeted green result: passed (`5 passed, 1 warning in 2.64s`).
- Full verification command: `uv run pytest -q`
- Full verification result: passed (`25 passed, 1 warning in 61.70s`).

### 2026-05-27 18:40:45 UTC — Replaced diagnostic compact chess notation plumbing with ASCII boards

- Correction: removed the public compact starting-position CLI flag, Modal parameter, and endgame compact fixture strings. Diagnostic starting positions now use `board_to_ascii()` diagrams plus an explicit side-to-move.
- Change: added `ascii_to_board()` as the inverse of the inspectable Unicode board format so tests/diagnostics can still construct exact `python-chess.Board` states without exposing compact chess notation at public or RL-facing boundaries.
- Change: endgame validation fixtures are now `EndgamePosition(board_ascii, turn)` values, and validation game reports include starting/final ASCII boards rather than compact chess notation fields.
- TDD red command: targeted tests for ASCII board parsing, ASCII-board CLI input, and removal of the previous compact starting-position flag.
- Red result: failed as expected because `DEFAULT_ENDGAME_POSITIONS` and `ascii_to_board()` did not exist and the old compact chess notation-based CLI flag was still present.
- Targeted green command: targeted tests for ASCII board parsing, ASCII-board CLI input, and removal of the previous compact starting-position flag.
- Targeted green result: passed (`3 passed in 1.27s`).
- Full verification command: `uv run pytest -q`
- Full verification result: passed (`28 passed, 1 warning in 63.07s`).

### 2026-05-27 18:44:17 UTC — Removed remaining compact position references

- Correction: removed remaining compact position references from tests, docs, code comments, and historical log text so the repo consistently describes ASCII board diagrams as the diagnostic/state representation.
- Targeted verification command: targeted tests for ASCII board parsing, ASCII-board CLI input, and removal of the previous compact starting-position flag.
- Targeted verification result: passed (`3 passed in 1.32s`).
- Full verification command: `uv run pytest -q`
- Full verification result: passed (`28 passed, 1 warning in 64.25s`).

### 2026-05-27 18:53:33 UTC — Removed over-broad repository text guard

- Correction: removed the repository-wide text guard test. The focused public-surface regression test remains: the CLI parser help must expose only ASCII starting-board flags.
- Targeted verification command: targeted tests for ASCII board parsing, ASCII-board CLI input, and removal of the previous compact starting-position flag.
- Targeted verification result: passed (`3 passed in 1.32s`).
- Full verification command: `uv run pytest -q`
- Full verification result: passed (`28 passed, 1 warning in 64.25s`).

### 2026-05-27 20:23:46 UTC — Cleaned stale repo docs and metadata

- Cleanup: updated the README component list, Modal training wording, and historical plan doc to match the current NN-guided PUCT implementation (`env`, `nn_model`, `puct_mcts`, `self_play`, `train`, validation, and thin Modal wrappers).
- Cleanup: replaced the placeholder package description in `pyproject.toml`, exported `ascii_to_board` from the package root, and removed an unused local variable in endgame validation.
- Cleanup: renamed the focused CLI public-surface regression test so it describes the current ASCII starting-board interface rather than old terminology.
- Search verification: swept tracked files for stale module names, previous public starting-position terms, placeholder metadata, and work-marker comments; result was no matches outside intentional historical metrics/tests.
- Lint command: `uvx ruff check .`
- Lint result: passed (`All checks passed!`).
- Targeted verification command: `uv run pytest tests/test_core.py::test_public_cli_exposes_only_ascii_starting_board_flags tests/test_core.py::test_ascii_board_parser_reconstructs_python_chess_position tests/test_core.py::test_training_metrics_do_not_report_truncation -q`
- Targeted verification result: passed (`3 passed in 2.25s`).
- Full verification command: `uv run pytest -q`
- Full verification result: passed (`28 passed, 1 warning in 63.92s`).

### 2026-05-27 20:33:07 UTC — Investigated requirements for low-Elo progress runs

- Investigation: inspected the current training loop, first-meaningful preset, Modal runner, and Stockfish validation wrapper to identify what is missing for a meaningful progress run.
- Stockfish capability check: local Stockfish resolves to `/home/hermes/.local/bin/stockfish`; `UCI_Elo` supports `1320–3190`, and requested Elo values below `1320` map to `Skill Level: 0` through the current wrapper.
- Runtime diagnostic command: ran tiny local `train()` probes with `hidden_channels=64`, `residual_blocks=4`, one game, one train step, and `simulations in {8,16,32}`.
- Runtime diagnostic result: examples/seconds were approximately `190/7.52s`, `368/21.45s`, and `129/17.80s`, showing full-start self-play game length dominates runtime and is stochastic.
- Low-Elo diagnostic command: trained a small local model with `iterations=2`, `games_per_iteration=2`, `simulations=8`, `train_steps=4`, then evaluated 4 games against requested Elo `500` / Stockfish `Skill Level 0`.
- Low-Elo diagnostic result: training took `27.48s` for `4` games and `983` examples; final losses were `[3.583, 3.624, 3.588, 3.583, 3.044, 3.256, 3.250, 3.268]`; validation took `2.80s` and scored `0/4` (`wins=0`, `losses=4`, `draws=0`, `score=0.000`).
- Interpretation: the current code can execute the loop and evaluate a weak engine, but a meaningful run needs checkpoint-by-checkpoint evaluation, a lower/noisier baseline ladder before Stockfish, and larger remote self-play scale than the current tiny preset.
