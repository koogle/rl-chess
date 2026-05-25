# RL MCTS Self-Play on Modal Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build an inspectable AlphaGo-style chess loop: self-play uses MCTS to improve each move, then training distills those search-improved targets.

**Architecture:** `python-chess` remains the rule engine. The RL-facing state is the Unicode board diagram (`board_ascii`), not FEN. Local training is the source of truth; Modal is only a remote runner for the same code.

**Tech Stack:** Python, uv, python-chess, pytest, Modal, hand-written MCTS/RL loops.

---

## Concise lesson from the lecture

AlphaGo was powerful because it turned sparse game outcomes into dense per-move supervision.

- Self-play creates endless data.
- MCTS improves the current policy at each board state.
- The model trains on MCTS visit distributions, not just final wins.
- The value head learns who should win from a board.
- The policy/value model then makes future MCTS cheaper and stronger.

In short: **search improves the model; the model improves search.**

---

## Initial plan

1. Keep `ChessEnv` simple: reset, observe, step, reward.
2. Use `board_ascii` as the model/replay state.
3. Extend MCTS to return visit-count policies per move.
4. Store self-play examples: `board_ascii`, legal moves, MCTS policy target, final value target.
5. Add a tiny policy/value learner from scratch before neural nets.
6. Train locally with deterministic seeds and tiny tests.
7. Add Modal as a thin wrapper over the same training function.
8. Scale self-play on Modal after the local loop is correct.

---

## Self-critique

The initial plan is directionally right but too vague in three places:

- It says “tiny learner” without defining the first minimal target format.
- It jumps to Modal before proving the MCTS training data is correct.
- It does not separate search, data collection, training, and evaluation enough.

---

## Revised implementation plan

### Task 1: Define MCTS training examples

**Objective:** Add a replay record for search-improved supervision.

**Files:**
- Modify: `src/rl_chess/replay.py`
- Test: `tests/test_core_loops.py`

**Add:**

```python
@dataclass(frozen=True)
class SearchTrainingExample:
    state_ascii: str
    legal_moves: tuple[str, ...]
    policy_target: dict[str, float]
    value_target: float | None = None
```

**Verify:**

```bash
uv run pytest -q
```

Expected: all tests pass.

---

### Task 2: Make MCTS expose visit policies

**Objective:** MCTS should return a normalized move distribution, not only a chosen move.

**Files:**
- Modify: `src/rl_chess/mcts.py`
- Test: `tests/test_mcts.py`

**Behavior:**

```python
policy = mcts.search_policy(board, iterations=25, seed=123)
assert abs(sum(policy.values()) - 1.0) < 1e-6
assert set(policy).issubset({move.uci() for move in board.legal_moves})
```

**Verify:**

```bash
uv run pytest tests/test_mcts.py -q
```

---

### Task 3: Collect MCTS self-play examples

**Objective:** During self-play, run MCTS at each board state and store the improved policy target.

**Files:**
- Create: `src/rl_chess/search_self_play.py`
- Test: `tests/test_mcts_training.py`

**Function:**

```python
def collect_search_episode(env, mcts, max_plies, iterations, seed=None) -> list[SearchTrainingExample]:
    ...
```

**Rules:**

- Store `board_ascii` before the move.
- Store legal UCI moves.
- Store normalized MCTS visits as `policy_target`.
- Fill `value_target` after the game ends, from the actor/player perspective.

**Verify:**

```bash
uv run pytest tests/test_mcts_training.py -q
```

---

### Task 4: Add a minimal tabular policy distiller

**Objective:** Learn from MCTS policy targets before adding neural nets.

**Files:**
- Modify: `src/rl_chess/agents.py`
- Test: `tests/test_mcts_training.py`

**Behavior:**

- Key by `(state_ascii, action_uci)`.
- Update toward `policy_target[action_uci]`.
- Select legal moves by learned probability with deterministic tie-breaking.

**Verify:**

```bash
uv run pytest -q
```

---

### Task 5: Add local MCTS training loop

**Objective:** Compose search self-play + policy distillation into one local training function.

**Files:**
- Modify: `src/rl_chess/train.py`
- Test: `tests/test_mcts_training.py`

**Function:**

```python
def train_mcts_self_play(...):
    ...
```

**Return metrics:**

- episodes
- total_plies
- examples_collected
- policy_entries
- results

**Verify:**

```bash
uv run pytest -q
uv run rl-chess --policy mcts-train --episodes 2 --max-plies 4 --mcts-iterations 4 --seed 123
```

---

### Task 6: Add CLI path for MCTS training

**Objective:** Make the new loop runnable from the command line.

**Files:**
- Modify: `src/rl_chess/cli.py`
- Test: `tests/test_core_loops.py`

**Command:**

```bash
uv run rl-chess --policy mcts-train --episodes 2 --max-plies 6 --mcts-iterations 8 --seed 123
```

**Expected output includes:**

- `policy=mcts-train`
- `examples_collected=`
- `policy_entries=`

---

### Task 7: Wire Modal to the same training function

**Objective:** Run the exact local MCTS training loop remotely.

**Files:**
- Modify: `src/rl_chess/modal_app.py`
- Test: `tests/test_modal_app.py`

**Rule:** Modal must call `train_mcts_self_play`; it must not duplicate training logic.

**Verify locally:**

```bash
uv run pytest tests/test_modal_app.py -q
```

**Smoke remote:**

```bash
uv run modal run src/rl_chess/modal_app.py --policy mcts-train --episodes 2 --max-plies 6 --mcts-iterations 8 --seed 123
```

---

### Task 8: Add README examples

**Objective:** Document the conceptual loop and exact commands.

**Files:**
- Modify: `README.md`

**Include:**

```text
search improves model; model improves search
```

and local/Modal commands.

**Verify:**

```bash
uv run pytest -q
```

---

## Definition of done

- `uv run pytest -q` passes.
- MCTS returns normalized per-move policies.
- Self-play stores `board_ascii` training examples.
- Local MCTS training runs from CLI.
- Modal calls the same local training function.
- README explains the loop in concise language.
