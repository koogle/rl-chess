# rl-chess

Learning-first reinforcement learning loops for chess, implemented by hand around `python-chess`.

## What is here

- `rl_chess.env.ChessEnv`: tiny environment wrapper over `python-chess.Board`.
- `rl_chess.state`: state helpers for Unicode board diagrams, legal UCI moves, and simple 12-plane board tensors.
- `rl_chess.self_play.play_episode`: alternates policies through one game.
- `rl_chess.replay.ReplayBuffer`: bounded replay storage.
- `rl_chess.agents.TabularMoveValueAgent`: minimal ε-greedy tabular move-value learner.
- `rl_chess.train.train_self_play`: the core training loop.
- `rl_chess.mcts.MCTS`: hand-written UCT Monte Carlo Tree Search using selection, expansion, random rollout evaluation, and backpropagation.
- `rl_chess.modal_app`: Modal entrypoint for remote training.

## Direction

The AlphaGo lesson for this project is concise:

> Search improves the model; the model improves search.

We keep the board inspectable with Unicode chess diagrams instead of exposing FEN to the RL loop. The next milestone is MCTS self-play training: run search at each board, store the visit-count policy as a better per-move target, then train a small policy/value learner from those examples. Local training remains the source of truth; Modal only scales the same loop remotely.

See `docs/alphago-from-scratch-lessons.md` for the Dwarkesh/Eric Jang AlphaGo-from-scratch notes, and `docs/plans/2026-05-25-rl-mcts-self-play-modal.md` for the implementation plan.

## Local training

Smoke run:

```bash
uv run rl-chess --iterations 1 --max-plies 20 --mcts-iterations 8 --seed 123
```

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
uv run modal run src/rl_chess/modal_app.py --episodes 1000 --max-plies 200 --seed 123
```

The Modal app runs the same `train_self_play` function remotely, so local and remote execution share one core loop.

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
