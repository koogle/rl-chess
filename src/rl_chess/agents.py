from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

import chess

from rl_chess.env import board_to_ascii
from rl_chess.replay import SearchTrainingExample, Transition


class Policy(Protocol):
    def select_move(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        ...


class RandomPolicy:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_move(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        chooser = rng if rng is not None else self.rng
        return chooser.choice(list(board.legal_moves))


@dataclass
class TabularMoveValueAgent:
    """Minimal hand-written ε-greedy tabular move-value learner."""

    learning_rate: float = 0.1
    epsilon: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.rng = random.Random(self.seed)
        self.q: dict[tuple[str, str], float] = {}

    def key(self, board: chess.Board, move: chess.Move) -> tuple[str, str]:
        return board_to_ascii(board), move.uci()

    def value(self, board: chess.Board, move: chess.Move) -> float:
        return self.q.get(self.key(board, move), 0.0)

    def select_move(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("cannot select a move from a terminal board")

        chooser = rng if rng is not None else self.rng
        if chooser.random() < self.epsilon:
            return chooser.choice(legal_moves)

        # Deterministic tie-breaker for learning/debuggability.
        return max(legal_moves, key=lambda move: (self.value(board, move), move.uci()))

    def learn(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            if transition.return_ is None:
                continue
            key = (transition.state_ascii, transition.action_uci)
            old_value = self.q.get(key, 0.0)
            self.q[key] = old_value + self.learning_rate * (transition.return_ - old_value)


@dataclass
class TabularPolicyDistiller:
    """Tiny policy/value learner that distills MCTS targets into tables."""

    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.policy: dict[tuple[str, str], float] = {}
        self.values: dict[str, float] = {}

    def policy_probability(self, state_ascii: str, action_uci: str) -> float:
        return self.policy.get((state_ascii, action_uci), 0.0)

    def value(self, state_ascii: str) -> float:
        return self.values.get(state_ascii, 0.0)

    @property
    def policy_entries(self) -> int:
        return len(self.policy)

    def learn(self, examples: list[SearchTrainingExample]) -> float:
        if not examples:
            return 0.0

        total_loss = 0.0
        target_count = 0
        for example in examples:
            for action_uci in example.legal_moves:
                target = example.policy_target.get(action_uci, 0.0)
                key = (example.state_ascii, action_uci)
                old = self.policy.get(key, 0.0)
                total_loss += (target - old) ** 2
                target_count += 1
                self.policy[key] = old + self.learning_rate * (target - old)

            if example.value_target is not None:
                old_value = self.values.get(example.state_ascii, 0.0)
                total_loss += (example.value_target - old_value) ** 2
                target_count += 1
                self.values[example.state_ascii] = old_value + self.learning_rate * (example.value_target - old_value)

        return total_loss / max(target_count, 1)
