from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol

import chess

from rl_chess.env import result_to_white_reward


class PolicyValueEvaluator(Protocol):
    def evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Return legal UCI move priors and value from White's perspective."""
        ...


@dataclass
class PUCTNode:
    board: chess.Board
    player: bool
    move: chess.Move | None = None
    parent: PUCTNode | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: list[PUCTNode] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def expand(self, priors: dict[str, float]) -> None:
        if self.children:
            return
        for move in self.board.legal_moves:
            child_board = self.board.copy(stack=False)
            child_board.push(move)
            self.children.append(
                PUCTNode(
                    board=child_board,
                    player=child_board.turn,
                    move=move,
                    parent=self,
                    prior=max(float(priors.get(move.uci(), 0.0)), 0.0),
                )
            )

    def backup(self, white_value: float) -> None:
        node: PUCTNode | None = self
        while node is not None:
            node.visits += 1
            node.value_sum += white_value if node.player == chess.WHITE else -white_value
            node = node.parent


@dataclass
class PUCTMCTS:
    evaluator: PolicyValueEvaluator
    iterations: int = 64
    c_puct: float = 1.5
    root_noise_alpha: float = 0.3
    root_noise_fraction: float = 0.25
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        self.rng = random.Random(self.seed)
        self.last_root: PUCTNode | None = None

    def search_policy(self, board: chess.Board, add_root_noise: bool = False) -> dict[str, float]:
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot search from a terminal board")

        root = PUCTNode(board=board.copy(stack=False), player=board.turn)
        priors, _ = self.evaluator.evaluate(board)
        root.expand(self._with_root_noise(priors) if add_root_noise else priors)

        for _ in range(self.iterations):
            leaf = root
            while leaf.children:
                leaf = max(leaf.children, key=lambda child: self._selection_score(leaf, child))
            if leaf.board.is_game_over(claim_draw=True):
                white_value = result_to_white_reward(leaf.board.result(claim_draw=True))
            else:
                child_priors, white_value = self.evaluator.evaluate(leaf.board)
                leaf.expand(child_priors)
            leaf.backup(white_value)

        self.last_root = root
        total = sum(child.visits for child in root.children)
        if total == 0:
            raise ValueError("search produced zero visits")
        return {child.move.uci(): child.visits / total for child in root.children if child.move is not None}

    def select_move(self, board: chess.Board, add_root_noise: bool = False) -> chess.Move:
        policy = self.search_policy(board, add_root_noise=add_root_noise)
        return chess.Move.from_uci(max(policy, key=policy.__getitem__))

    def _selection_score(self, parent: PUCTNode, child: PUCTNode) -> float:
        return -child.mean_value + self.c_puct * child.prior * math.sqrt(max(parent.visits, 1)) / (1 + child.visits)

    def _with_root_noise(self, priors: dict[str, float]) -> dict[str, float]:
        if not priors:
            return priors
        noise = [self.rng.gammavariate(self.root_noise_alpha, 1.0) for _ in priors]
        total_noise = sum(noise)
        if total_noise <= 0:
            return priors
        return {
            move: (1 - self.root_noise_fraction) * prior + self.root_noise_fraction * noise_i / total_noise
            for (move, prior), noise_i in zip(priors.items(), noise)
        }
