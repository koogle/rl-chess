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

```bash
uv run rl-chess --episodes 10 --max-plies 80 --seed 123
```

MCTS self-play smoke run:

```bash
uv run rl-chess --policy mcts --episodes 1 --max-plies 20 --mcts-iterations 50 --seed 123
```

## Modal training

```bash
uv run modal run src/rl_chess/modal_app.py --episodes 1000 --max-plies 200 --seed 123
```

The Modal app runs the same `train_self_play` function remotely, so local and remote execution share one core loop.

## Tests

```bash
uv run pytest -q
```
