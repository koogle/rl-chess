# Lessons from Dwarkesh/Eric Jang: Building AlphaGo from scratch

Source: Dwarkesh Patel, **"Eric Jang – Building AlphaGo from scratch"**, published 2026-05-15 at <https://www.dwarkesh.com/p/eric-jang>; matching YouTube video: <https://youtu.be/X_ZVSPcZhtw>. Direct YouTube transcript retrieval was blocked from this environment, so this summary is grounded in the official Dwarkesh episode page and its embedded transcript.

## High-level takeaway

The episode frames AlphaGo as the cleanest worked example of a learning system where:

```text
search improves the policy target
policy/value learning improves future search
self-play supplies unlimited fresh data
```

The important lesson is not that Go is special. It is that MCTS turns sparse win/loss feedback into a better per-move learning target. Naive policy-gradient RL has to infer which action in a long trajectory caused the final reward. AlphaGo-style search instead asks, at every state, "after lookahead, which legal moves look better than the raw policy guessed?" That improved visit-count distribution becomes the policy label.

## Core algorithm described in the episode

1. **Represent a perfect-information board state.**
   - Go is deterministic and fully observable, so a search tree can faithfully represent future states.
   - The action can be inferred from the child state, though implementations often store it explicitly.

2. **Use MCTS to search from the current state.**
   - Each node tracks visit counts and mean action value `Q(s, a)`.
   - Search alternates between selection, expansion/evaluation, and backup.
   - The selection rule is PUCT: choose moves with high value, high policy prior, and low visit count.

3. **Use a neural net as both policy and value intuition.**
   - The policy head predicts a prior distribution over legal moves.
   - The value head predicts win/loss from the current state.
   - The architecture is less important than setting up the right target loop; Eric notes that ResNets are still strong in small-budget board-game regimes because convolution gives useful local spatial bias.

4. **Turn search into training data.**
   - The policy target is not the raw move played by the network; it is the improved MCTS visit distribution.
   - The value target is the final game outcome from that player's perspective.
   - This is why AlphaGo-style MCTS helps credit assignment: every visited state gets a stronger local target than the terminal reward alone.

5. **Self-play closes the loop.**
   - The current model plays itself using search.
   - New games become replay data.
   - Training improves the policy/value net.
   - The improved net makes the next round of search cheaper and stronger.

6. **A replay buffer is acceptable but imperfectly off-policy.**
   - Older games may contain states the current policy would not visit.
   - This wastes some capacity but is usually tolerable because MCTS relabels each stored state with an improved local action distribution.
   - Collapse can happen if the replay distribution loses coverage of important late-game states.

7. **Search is not guaranteed to improve the policy.**
   - MCTS depends on the value function and on enough simulations.
   - If the value head is wrong, or if games resign before late positions are resolved, backups can poison the visit distribution.
   - A practical mitigation is to sometimes force games to play out to a real terminal result instead of always resigning early.

8. **Modern implementation lesson: get to a strong opponent quickly.**
   - Eric emphasizes that many details matter less than reaching a useful training signal quickly.
   - KataGo-style improvements greatly reduced compute versus the original DeepMind systems.
   - Small-board warm starts, better architecture/hyperparameter search, and LLM-assisted experimentation can reduce wall-clock and dollar cost.

## How this repository is similar

This repo is already an intentionally small AlphaZero-like loop for chess:

- `rl_chess.puct_mcts.PUCTMCTS` implements neural-net-guided PUCT search.
- `rl_chess.nn_model.PolicyValueNet` has policy and value heads.
- `rl_chess.self_play.play_self_game` generates self-play games.
- Each self-play position stores:
  - visual board state
  - MCTS visit-count policy target
  - final outcome value target
- `rl_chess.train.train` trains the model from those examples.
- `rl_chess.validation` benchmarks against a weak supported Stockfish baseline.

So the repo captures the central lesson:

```text
model -> PUCT search -> improved policy/value examples -> model
```

## How this repository differs from the episode's AlphaGo-from-scratch system

### 1. Chess instead of Go

The episode focuses on Go. This repo uses chess through `python-chess`.

Important differences:

- Chess has draws, checkmate, stalemate, repetition, the fifty-move rule, castling, en passant, and promotion.
- Chess has a much more tactically sharp branching structure.
- Chess has strong existing engines, so validation can use Stockfish instead of only self-play Elo.
- Go has large spatial regularity; chess still has spatial structure, but piece identity and tactical forcing lines matter more.

### 2. Minimal AlphaZero-style loop, not a full AlphaGo reproduction

The repo is deliberately small. It does not yet include:

- distributed self-play actors
- checkpoint leagues
- best-model gating before replacing the self-play generator
- large replay windows or prioritized replay
- resignation/adjudication policy
- symmetry augmentation
- opening randomization/curriculum
- supervised human/expert pretraining
- small-board warm starts
- large residual towers
- serious hyperparameter sweeps

That is intentional for now: the code is optimized for inspectability and fast iteration, not strength.

### 3. No rollout policy

The original AlphaGo Lee system mixed policy/value nets with rollout-style components. The repo is closer to AlphaGo Zero / AlphaZero: no handcrafted rollout evaluator, just neural policy/value plus tree search.

This keeps the implementation cleaner and matches the episode's broader point that search-generated policy labels are the core reusable idea.

### 4. The value target is very sparse

Current self-play value targets come from the final game result. Training self-play should therefore run until a terminal `python-chess` result; `max_plies` is only a safety guard and raises if a non-terminal game reaches it.

That is clean, but sparse:

- many early positions get the same delayed target
- artificial capped draws can teach false values if they leak back into training data
- weak early models may generate low-quality late-game data

The episode's warning applies directly: if the value head is poor, MCTS can amplify bad evaluation. The repo should therefore treat loss curves as plumbing checks, not strength proof.

### 5. Root exploration is present but simple

The repo already adds Dirichlet-style root noise during self-play. It does not yet have a mature exploration schedule, temperature decay by move number, or adaptive simulation budget.

### 6. Validation is external rather than only self-play relative

The episode emphasizes self-play improvement. This repo adds an external floor: weak Stockfish validation. That is useful because chess has a strong, cheap reference engine. The current supported Stockfish UCI Elo floor is `1320`, so that is the first honest baseline.

## What to borrow next

The next improvements should preserve the repo's simple shape while making the training signal less toy-like:

1. **Checkpoint evaluation gate**
   - Keep a previous best model.
   - Only promote a candidate if it beats the previous best over a small match.
   - This prevents self-play collapse.

2. **Real replay buffer**
   - Keep examples across iterations instead of only the latest tiny batch.
   - Track the age/source checkpoint of examples so off-policy drift is visible.

3. **Temperature schedule**
   - Sample early moves from the visit distribution.
   - Choose argmax later in the game.
   - This improves opening diversity without making every tactical position noisy.

4. **Avoid misleading truncation**
   - Let training self-play reach terminal outcomes instead of converting cap hits into draws.
   - Keep validation caps explicit, and treat capped validation draws as a measurement limit rather than proof of strength.

5. **Exploit chess symmetries only where legal**
   - Board flips by color are useful.
   - Unlike Go, arbitrary rotations/reflections are not generally valid because pawn direction and castling semantics matter.

6. **Use Stockfish as a curriculum, not a crutch**
   - First goal: survive or score against Stockfish at the supported floor (`1320`).
   - Later: increase Stockfish strength and reduce MCTS simulations required at evaluation time.
   - Avoid training directly to mimic Stockfish until the self-play loop is understood.

7. **Add small, measurable experiments**
   - Vary simulations, replay size, hidden channels, and temperature.
   - Measure against the same Stockfish baseline.
   - Prefer experiments that answer one question at a time.

## Repo north star after the episode

The platonic implementation should remain:

```text
python-chess board
 -> visual state encoding
 -> policy/value net
 -> PUCT search
 -> self-play examples
 -> train
 -> Stockfish/checkpoint validation
```

Do not add generic RL machinery unless it improves that loop. The lesson from the episode is that the power comes from the tight search-learning-self-play feedback cycle, not from a pile of helper abstractions.
