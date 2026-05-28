# RL MCTS Self-Play on Modal Plan

> **Status:** Implemented baseline. This document records the current Modal-first architecture and the next cleanup/scaling direction.

## Goal

Build an inspectable AlphaGo-style chess loop: self-play uses PUCT search to improve each move, then training distills those search-improved targets into a small policy/value network.

## Architecture

- `python-chess` remains the rule engine for legal moves, state transitions, terminal detection, and result accounting.
- RL-facing and diagnostic state is a Unicode board diagram (`board_ascii`) plus explicit side-to-move where reconstruction is needed.
- `PolicyValueNet` consumes 12 piece planes plus a side-to-move plane.
- `PUCTMCTS` queries the model for legal-move priors and a value estimate, then returns a normalized visit-count policy over legal UCI moves.
- `play_self_game()` records one AlphaZero-style game as `TrainingExample(state_ascii, turn, policy_target, value_target)` values.
- `train()` owns the core replay buffering, gradient updates, and checkpoint IO used by Modal.
- `modal_app.py` is the supported execution surface for real train/eval runs.

## Current implementation files

- `src/rl_chess/env.py` — board wrapper, Unicode board serialization/parsing, reward conversion.
- `src/rl_chess/nn_model.py` — policy/value model, board tensor encoding, policy/value loss.
- `src/rl_chess/puct_mcts.py` — hand-written neural PUCT search.
- `src/rl_chess/self_play.py` — MCTS-guided game collection.
- `src/rl_chess/train.py` — replay-buffered training loop and checkpoint IO.
- `src/rl_chess/validation.py` — model-vs-baseline validation.
- `src/rl_chess/modal_app.py` — Modal training/evaluation entrypoint.

## Verification commands

```bash
uv run pytest -q
uv run modal run src/rl_chess/modal_app.py --iterations 1 --max-plies 1 --simulations 2 --seed 123
```

`--max-plies` is only a self-play safety cap. Uncapped self-play should run until `python-chess` reports a terminal result; cap hits in training are errors, not draw targets.

## Next work

1. Add checkpoint-sized remote update blocks: many self-play games, bounded train steps, one checkpoint, then evaluation.
2. Add random and `weakest_stockfish` checkpoint evaluation; skip material-greedy baselines.
3. Parallelize self-play games across Modal workers inside each checkpoint block.
4. Improve replay accounting so checkpoint age/source is visible in metrics.
5. Add a temperature schedule for early-game exploration and late-game determinism.
6. Keep every experiment and meaningful design decision in the README research log.
