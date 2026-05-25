from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

import chess

from rl_chess.env import board_to_ascii
from rl_chess.replay import Transition


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
