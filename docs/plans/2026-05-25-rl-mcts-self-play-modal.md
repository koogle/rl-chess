# RL MCTS Self-Play on Modal Plan

> **Status:** Implemented baseline. This document now records the current architecture and the next cleanup/scaling direction rather than the older task-by-task bootstrap plan.

## Goal

Build an inspectable AlphaGo-style chess loop: self-play uses PUCT search to improve each move, then training distills those search-improved targets into a small policy/value network.

## Architecture

- `python-chess` remains the rule engine for legal moves, state transitions, terminal detection, and result accounting.
- RL-facing and diagnostic state is a Unicode board diagram (`board_ascii`) plus explicit side-to-move where reconstruction is needed.
- `PolicyValueNet` consumes 12 piece planes plus a side-to-move plane.
- `PUCTMCTS` queries the model for legal-move priors and a value estimate, then returns a normalized visit-count policy over legal UCI moves.
- `play_self_game()` records one AlphaZero-style game as `TrainingExample(state_ascii, turn, policy_target, value_target)` values.
- `train()` owns replay buffering, gradient updates, checkpointing, and optional diagnostic starting boards.
- Modal entrypoints in `modal_app.py` are thin remote wrappers over the same local code.

## Current implementation files

- `src/rl_chess/env.py` — board wrapper, Unicode board serialization/parsing, reward conversion.
- `src/rl_chess/nn_model.py` — policy/value model, board tensor encoding, policy/value loss.
- `src/rl_chess/puct_mcts.py` — hand-written neural PUCT search.
- `src/rl_chess/self_play.py` — MCTS-guided game collection.
- `src/rl_chess/train.py` — replay-buffered training loop and checkpoint IO.
- `src/rl_chess/validation.py` — model-vs-Stockfish validation.
- `src/rl_chess/endgame_validation.py` — narrow KQK value-head validation.
- `src/rl_chess/cli.py` and `src/rl_chess/modal_app.py` — local and remote entrypoints.

## Verification commands

```bash
uv run pytest -q
uv run rl-chess --iterations 1 --max-plies 1 --mcts-iterations 2 --seed 123
uv run modal run src/rl_chess/modal_app.py --first-meaningful-run --seed 123
```

`--max-plies` is only a self-play safety cap. Uncapped self-play should run until `python-chess` reports a terminal result; cap hits in training are errors, not draw targets.

## Next work

1. Add an evaluation ladder: random player, simple heuristic player, then Stockfish at the supported UCI Elo floor.
2. Run a larger diagnostic training job now that self-play no longer converts safety caps into draw labels.
3. Improve replay accounting so checkpoint age/source is visible in metrics.
4. Add a temperature schedule for early-game exploration and late-game determinism.
5. Keep every experiment and meaningful design decision in the README research log.
