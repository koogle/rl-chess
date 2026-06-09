# rl-chess

Learning-first reinforcement learning loops for chess, implemented by hand around `python-chess`.

## What is here

- `rl_chess.env`: Unicode board diagram conversion helpers plus result-to-reward accounting.
- `rl_chess.nn_model.PolicyValueNet`: small policy/value network over piece, side-to-move, castling, en-passant, halfmove-clock, and claimable-draw state planes.
- `rl_chess.puct_mcts.PUCTMCTS`: hand-written neural-net-guided PUCT search over legal UCI moves.
- `rl_chess.self_play.play_self_game`: one AlphaZero-style self-play game that records visit-count policy targets and terminal value targets.
- `rl_chess.train.train`: fresh-batch policy/value training loop with legal color-flip augmentation and optional checkpointing; each iteration generates new self-play from the latest model snapshot, then updates only on that batch.
- `rl_chess.validation`: model-vs-baseline evaluation helpers, including per-checkpoint random and weakest Stockfish / supported UCI Elo baselines.
- `rl_chess.modal_app`: the supported training/evaluation entrypoint for real runs on Modal.

## Direction

The AlphaGo lesson for this project is concise:

> Search improves the model; the model improves search.

We keep the board inspectable with Unicode chess diagrams instead of exposing compact chess notation to the RL loop. The target architecture is Modal-first: local code owns the inspectable core RL/search behavior and local files only receive checkpoints/artifacts, while training and evaluation runs execute through Modal. The next scale step is checkpoint-sized update blocks: many parallel self-play games, a bounded number of gradient updates, one checkpoint, then evaluation of that checkpoint.

See `docs/alphago-from-scratch-lessons.md` for the Dwarkesh/Eric Jang AlphaGo-from-scratch notes, and `docs/plans/2026-05-25-rl-mcts-self-play-modal.md` for the implementation plan.

## Modal training

Tiny smoke run:

Quick smoke runs can wait for their result by passing `--wait`:

```bash
uv run modal run src/rl_chess/modal_app.py::main --iterations 1 --max-plies 1 --simulations 2 --seed 123 --wait
```

Real/non-smoke Modal training runs use the default spawn-based handoff. The local entrypoint calls `train_remote.spawn(...)`, prints JSON containing `function_call_id`, `dashboard_url`, and `checkpoint_dir`, then exits while the Modal function continues independently:

```bash
uv run modal run src/rl_chess/modal_app.py::main <training flags>
```

Fetch a spawned run's final JSON result later with:

```bash
uv run modal run src/rl_chess/modal_app.py::result --function-call-id <fc-...>
```

When `--checkpoint-dir` is set, the remote function also writes `/checkpoints/<run-id>/summary.json` after final validation and commits it to the Modal volume. Always retrieve and report those final evaluation fields back to the user when a run finishes.

Pilot checkpoint block sizing target:

```text
parallel_workers = 8
fresh_batch_games = 100
train_updates_per_batch = 1
mcts_simulations = 8
batch_size = 4096
eval_after_run = true
```

`--max-plies` is a safety cap, not a training truncation mechanism. If a game reaches the cap while non-terminal, the run raises instead of converting the unfinished game into a draw target. Omit the flag for uncapped self-play that runs until `python-chess` reports a terminal result.

## Tests

```bash
uv run pytest -q
```

## Research log

### 2026-05-27 18:02:05 UTC — Research-log convention and first validation baseline

- Added `AGENTS.md` so research work keeps a dated, factual log of structural changes, methodology decisions, meaningful experiments, metrics, and artifacts.
- Baseline status: the endgame value-validation loop over ten KQK forced-mate positions showed the model could reduce value MSE on terminal-backed targets; the broader NN-guided PUCT self-play/checkpoint/weak-Stockfish path existed but was only covered at tiny scale.

### 2026-05-27 18:07:00 UTC — First full remote training/validation run

- Command: `uv run modal run src/rl_chess/modal_app.py --first-meaningful-run --seed 123`
- Modal run: https://modal.com/apps/koogle-frick/main/ap-5c0RM2iYhmle35dMOe3ICn
- Result: completed as an execution run but failed the Stockfish validation gate.
- Training: `iterations=3`, `games=6`, `examples=653`, `terminal_games=2`, `truncated_games=4`, `replay_size=653`; final loss `3.581167`.
- Checkpoints: `/checkpoints/first-meaningful-run/iteration-0001.pt`, `/checkpoints/first-meaningful-run/iteration-0002.pt`, `/checkpoints/first-meaningful-run/iteration-0003.pt`.
- Stockfish validation: Elo `1320`, `validation_games=4`, `wins=0`, `losses=4`, `draws=0`, `score=0.000`, `validation_passed=False`.
- Interpretation: the end-to-end loop and checkpointing worked, but this preset was too weak and too truncated to measure learning against Stockfish's supported Elo floor.

### 2026-05-27 18:25:30 UTC — Removed self-play truncation as a learning target

- Correction: restored the side-to-move input plane; the model again receives side-to-move information.
- Methodology change: self-play now either reaches a terminal `python-chess` result or raises if an optional safety `max_plies` cap is hit while non-terminal. Truncated non-terminal games are no longer converted into draw targets.
- Architecture change: first-meaningful training moved to uncapped self-play (`max_plies=None`); `--max-plies` remains only a safety cap.
- Temporary compact starting-position support was added for deterministic smoke tests, then removed when diagnostics moved to inspectable board diagrams.

### 2026-05-27 18:40:45 UTC — Replaced diagnostic compact chess notation with ASCII boards

- Methodology change: diagnostic starting positions now use `board_to_ascii()` diagrams plus explicit side-to-move, keeping public/RL-facing boundaries inspectable.
- Architecture change: added `ascii_to_board()` as the inverse of the inspectable board format so tests and diagnostics can construct exact `python-chess.Board` states without exposing compact chess notation at public boundaries.
- Validation reports now use starting/final ASCII boards rather than compact position fields.

### 2026-05-27 20:23:46 UTC — Aligned repo docs and metadata with the current architecture

- Updated the README component list, Modal training wording, and implementation plan to match the current NN-guided PUCT system: `env`, `nn_model`, `puct_mcts`, `self_play`, `train`, validation helpers, and thin Modal wrappers.
- Replaced placeholder package metadata and exported `ascii_to_board` from the package root.
- Removed stale comments, obsolete public starting-position terminology, and leftover work-marker text so future work starts from the current architecture rather than old scaffolding.

### 2026-05-27 20:33:07 UTC — Investigated requirements for low-Elo progress runs

- Stockfish capability check: local Stockfish resolves to `/home/hermes/.local/bin/stockfish`; `UCI_Elo` supports `1320–3190`, and requested Elo values below `1320` map to `Skill Level: 0` through the current wrapper.
- Runtime diagnostic: tiny local full-start probes with `hidden_channels=64`, `residual_blocks=4`, one game, one train step, and `simulations in {8,16,32}` produced roughly `190/7.52s`, `368/21.45s`, and `129/17.80s` examples/seconds, showing game length dominates runtime and is stochastic.
- Low-engine diagnostic: a small local `iterations=2`, `games_per_iteration=2`, `simulations=8`, `train_steps=4` model scored `0/4` against requested Elo `500` / Stockfish `Skill Level 0`; training took `27.48s` for `4` games and `983` examples.
- Interpretation: meaningful progress measurement needs checkpoint-by-checkpoint evaluation, a lower/noisier baseline ladder before supported UCI Elo Stockfish, and larger remote self-play scale than the tiny preset.

### 2026-05-28 05:42:14 UTC — Removed legacy local/diagnostic entrypoints

- Architecture change: removed the local `rl-chess` console training entrypoint, the old first-meaningful-run preset module, and the KQK endgame value-validation diagnostic.
- Supported execution surface is now `src/rl_chess/modal_app.py` plus shared core training/evaluation modules.
- Interpretation: this reduces accidental local unscaled runs and focuses maintenance on the Modal-first path.

### 2026-05-28 06:41:03 UTC — Three-checkpoint Modal structural smoke loop

- Command: `uv run modal run src/rl_chess/modal_app.py --iterations 3 --games-per-iteration 1 --max-plies 1 --simulations 1 --train-steps 1 --batch-size 1 --hidden-channels 8 --residual-blocks 0 --checkpoint-dir /checkpoints/three-checkpoint-smoke-20260528-064014 --starting-board-ascii <KQK_BLACK_TO_MOVE_ASCII> --starting-turn black --seed 123`
- Modal run: https://modal.com/apps/koogle-frick/main/ap-pWAcIQbOVIxtwaQAS6xdGF
- Result: completed with `iterations=3`, `games=3`, `examples=3`, `terminal_games=3`, `replay_size=3`, and three checkpoint files in `/checkpoints/three-checkpoint-smoke-20260528-064014/`.
- Interpretation: Modal can produce and persist three checkpoints, but this was only a structural checkpointing smoke test using a one-move terminal diagnostic position.

### 2026-05-28 07:38:15 UTC — Real three-checkpoint Modal training run with progress

- Methodology change before run: added per-checkpoint progress printing from the Modal training function so long runs expose `iteration`, cumulative games/examples, optimizer updates, latest loss, and checkpoint path as each checkpoint completes.
- Training command: `uv run modal run src/rl_chess/modal_app.py --iterations 3 --games-per-iteration 16 --simulations 8 --train-steps 32 --batch-size 512 --replay-capacity 50000 --learning-rate 0.001 --temperature 1.0 --hidden-channels 64 --residual-blocks 4 --checkpoint-dir /checkpoints/real-3ckpt-progress-20260528-071500 --seed 20260528`
- Modal run: https://modal.com/apps/koogle-frick/main/ap-x8DsYKzqekTYYbH6ws76o6
- Progress: checkpoint 1 at `games=16`, `examples=3516`, `updates=32`, `latest_loss=3.314674139022827`; checkpoint 2 at `games=32`, `examples=7343`, `updates=64`, `latest_loss=3.2679388523101807`; checkpoint 3 at `games=48`, `examples=10664`, `updates=96`, `latest_loss=3.403989791870117`.
- Final result: `iterations=3`, `games=48`, `examples=10664`, `terminal_games=48`, `replay_size=10664`; checkpoints in `/checkpoints/real-3ckpt-progress-20260528-071500/`.
- Interpretation: this was real full-start self-play and the loop produced persistent checkpoints. It is still pilot-sized at 32 updates/checkpoint, below the target ~125 updates/checkpoint; the next real run should use 128 games/checkpoint and ~125 updates/checkpoint after parallel self-play or a longer Modal worker plan.

### 2026-05-28 07:41:50 UTC — Curated the research log scope

- Methodology change: refined `AGENTS.md` so the research log is reserved for structural architecture changes, methodology progress, meaningful training/evaluation experiments, and design decisions.
- Cleanup: removed routine RED/GREEN/full-suite test output, standalone documentation verification, transient aborted-run detail, and mechanical cleanup notes that did not change architecture or methodology.

### 2026-05-28 14:54:10 UTC — Added and ran random-baseline validation

- Methodology change: added a random legal-move baseline to the validation ladder so early models can be checked against a weaker signal before Stockfish.
- Command: `uv run modal run src/rl_chess/modal_app.py --iterations 1 --games-per-iteration 16 --simulations 8 --train-steps 64 --batch-size 512 --replay-capacity 50000 --learning-rate 0.001 --temperature 1.0 --hidden-channels 64 --residual-blocks 4 --checkpoint-dir /checkpoints/random-eval-16train-12games-20260528-143441 --validate-random --validation-games 12 --validation-max-plies 200 --seed 20260528`
- Modal run: https://modal.com/apps/koogle-frick/main/ap-MiAxPCrqmNXO2avDAnrnvA
- Training result: `games=16`, `terminal_games=16`, `examples=5348`, `updates=64`, latest loss `2.9693610668182373`; checkpoint `/checkpoints/random-eval-16train-12games-20260528-143441/iteration-0001.pt`.
- Random validation: `games=12`, `wins=1`, `losses=0`, `draws=11`, `score=0.5416666666666666`, `passed=True`.
- Interpretation: this clears the first weak baseline by avoiding losses and scoring just above 0.5, but most games are still draws; next methodology step should evaluate each checkpoint against random and improve decisiveness before treating Stockfish as the main signal.

### 2026-05-28 22:58:04 UTC — Removed replay buffer and launched 10,000-game fresh-batch run

- Methodology change: removed replay capacity/replay-size accounting from the training loop. Each iteration now freezes the latest model snapshot, generates a fresh parallel self-play batch from that snapshot, trains only on that batch, checkpoints, and then uses the updated model for the next batch.
- Parallelism: added `--self-play-workers`; Modal training now requests `cpu=8` and uses worker-local model copies for self-play generation.
- Verification: `uv run pytest -q` passed with `25 passed, 1 warning`.
- Command: `uv run modal run src/rl_chess/modal_app.py --iterations 100 --games-per-iteration 100 --simulations 8 --train-steps 1 --batch-size 4096 --learning-rate 0.001 --temperature 1.0 --hidden-channels 64 --residual-blocks 4 --self-play-workers 8 --checkpoint-dir /checkpoints/fresh-batch-10000games-20260528-224952 --validate-random --validation-games 32 --validation-max-plies 200 --seed 20260528`
- Modal run: https://modal.com/apps/koogle-frick/main/ap-C1zsRlZoKEZaWWggcZm9tR
- Intended scale: `100` fresh batches × `100` games = `10,000` terminal self-play games, with one optimizer update per batch and final 32-game random-baseline validation.
- Completion checked 2026-06-01 00:49:45 UTC: Modal showed zero active tasks and the checkpoint volume contained all 100 checkpoints through `/checkpoints/fresh-batch-10000games-20260528-224952/iteration-0100.pt`.
- Final checkpoint metrics: `iterations=100`, `games=10000`, `terminal_games=10000`, `examples=366527`, `iteration_examples=2120`, `updates=100`; loss moved from `3.0997283458709717` to `0.20128056406974792`.
- Final 32-game random validation rerun from downloaded `iteration-0100.pt`: `wins=0`, `losses=3`, `draws=29`, `score=0.453125`, `passed=False`.
- Interpretation: the 10,000-game fresh-batch run completed and fit its self-play targets strongly, but the final checkpoint regressed below the random baseline. The next fix should not be “more games” blindly; investigate why fresh-batch training is converging toward draw/loss behavior, likely by adding checkpoint-by-checkpoint random validation and improving update frequency/target quality.

### 2026-06-01 00:57:49 UTC — Detached Modal launch convention for real runs

- Operational change: real/non-smoke Modal training runs should use `uv run modal run --detach ...` so the app keeps running if the local client disconnects; tiny smoke runs can remain attached for quick feedback.
- Reason: the 10,000-game run completed and persisted checkpoints, but the attached local client later emitted repeated log-continuity warnings and was killed locally after completion.

### 2026-06-08 13:21:17 PDT — Added chess-state fidelity, color-flip augmentation, and checkpoint validation

- Architecture change: PUCT and validation now preserve `python-chess` move history when copying boards, so repetition and fifty-move claim detection are not dropped inside search/evaluation.
- Model-boundary change: the policy/value net now receives legal-state planes for castling rights, en-passant square, halfmove clock, claimable threefold repetition, and claimable fifty-move draw in addition to piece and side-to-move planes.
- Methodology change: training can augment fresh self-play examples with the legal color-swap/rank-mirror equivalent; metrics now distinguish raw self-play `examples` from augmented `training_examples`.
- Evaluation change: Modal summaries can include per-checkpoint validation entries when `--validate-random` or `--validate-stockfish` is used with checkpoints, so final-checkpoint regressions are easier to localize.
- Verification: `uv run pytest -q` passed with `32 passed, 2 warnings`.

### 2026-06-09 12:07:40 PDT — Local 1,000-game checkpoint random-validation curve

- Modal launch was blocked by missing local Modal credentials, so this used the shared local training/evaluation core rather than the remote runner.
- Command:

```bash
.venv/bin/python -c '
from pathlib import Path
import json
from rl_chess.nn_model import PolicyValueNet
from rl_chess.train import train, load_checkpoint_model
from rl_chess.validation import validate_model_against_random
checkpoint_dir = Path("/private/tmp/rl-chess-local-1000-20260609")
model = PolicyValueNet(hidden_channels=16, residual_blocks=1)
metrics = train(
    model=model,
    iterations=5,
    games_per_iteration=2,
    simulations=2,
    train_steps=2,
    batch_size=128,
    learning_rate=0.001,
    temperature=1.0,
    seed=20260609,
    checkpoint_dir=checkpoint_dir,
    self_play_workers=1,
)
results = []
for idx, path in enumerate(metrics.checkpoint_paths, start=1):
    ckpt_model = load_checkpoint_model(path)
    validation = validate_model_against_random(
        ckpt_model,
        games=200,
        max_plies=100,
        simulations=2,
        seed=20260609 + idx,
    )
    item = {
        "iteration": idx,
        "checkpoint_path": str(path),
        "wins": validation.wins,
        "losses": validation.losses,
        "draws": validation.draws,
        "score": validation.score,
        "passed": validation.passed,
    }
    print("checkpoint_validation " + json.dumps(item, sort_keys=True), flush=True)
    results.append(item)
summary = {
    "training": {
        "iterations": metrics.iterations,
        "games": metrics.games,
        "examples": metrics.examples,
        "training_examples": metrics.training_examples,
        "terminal_games": metrics.terminal_games,
        "loss_curve": metrics.loss_curve,
        "policy_loss_curve": metrics.policy_loss_curve,
        "value_loss_curve": metrics.value_loss_curve,
        "checkpoint_paths": [str(path) for path in metrics.checkpoint_paths],
    },
    "validation_games_total": sum(item["wins"] + item["losses"] + item["draws"] for item in results),
    "checkpoint_validations": results,
}
print("FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
'
```
- Training: `iterations=5`, `games=10`, `examples=3653`, `training_examples=7306`, `terminal_games=10`; checkpoints written to `/private/tmp/rl-chess-local-1000-20260609/iteration-0001.pt` through `iteration-0005.pt`.
- Loss moved from `3.0041658878326416` to `2.8489530086517334`; value loss moved from `0.06095225363969803` to `0.052564822137355804`.
- Checkpoint random validation, 200 games/checkpoint and 1,000 games total: checkpoint 1 `5W/6L/189D`, score `0.4975`; checkpoint 2 `4W/5L/191D`, score `0.4975`; checkpoint 3 `1W/8L/191D`, score `0.4825`; checkpoint 4 `0W/3L/197D`, score `0.4925`; checkpoint 5 `3W/3L/194D`, score `0.5000`.
- Interpretation: this small local run does not show learning against random. The curve is flat-to-worse and dominated by capped draws; decreasing training loss should not be read as strength improvement.
