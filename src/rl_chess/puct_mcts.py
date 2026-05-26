from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol

import chess

from rl_chess.env import result_to_white_reward


class PolicyValueEvaluator(Protocol):
    def evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Return legal-move priors and value from White's perspective."""
        ...


@dataclass
class PUCTNode:
    board_fen: str
    player_to_move: bool
    parent: PUCTNode | None = None
    move: chess.Move | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: list[PUCTNode] = field(default_factory=list)
    expanded: bool = False

    @classmethod
    def root(cls, board: chess.Board) -> PUCTNode:
        return cls(board_fen=board.fen(), player_to_move=board.turn)

    def board(self) -> chess.Board:
        return chess.Board(self.board_fen)

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def expand(self, priors: dict[str, float]) -> None:
        if self.expanded:
            return
        board = self.board()
        for move in board.legal_moves:
            child_board = board.copy(stack=False)
            child_board.push(move)
            self.children.append(
                PUCTNode(
                    board_fen=child_board.fen(),
                    player_to_move=child_board.turn,
                    parent=self,
                    move=move,
                    prior=max(float(priors.get(move.uci(), 0.0)), 0.0),
                )
            )
        self.expanded = True

    def best_child(self, c_puct: float) -> PUCTNode:
        if not self.children:
            raise ValueError("node has no children")
        return max(self.children, key=lambda child: puct_score(self, child, c_puct))

    def backpropagate(self, white_reward: float) -> None:
        node: PUCTNode | None = self
        while node is not None:
            node.visits += 1
            node.value_sum += white_reward if node.player_to_move == chess.WHITE else -white_reward
            node = node.parent


def puct_score(parent: PUCTNode, child: PUCTNode, c_puct: float = 1.5) -> float:
    exploration = c_puct * child.prior * math.sqrt(max(parent.visits, 1)) / (1 + child.visits)
    return child.mean_value + exploration


@dataclass
class PUCTMCTS:
    """NN-guided PUCT search: model priors guide selection; model value evaluates leaves."""

    evaluator: PolicyValueEvaluator
    iterations: int = 100
    c_puct: float = 1.5
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        self.rng = random.Random(self.seed)
        self.last_root: PUCTNode | None = None

    def select_move(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        self.search(board, rng=rng)
        if self.last_root is None:
            raise ValueError("search produced no root")
        return self._robust_child(self.last_root).move  # type: ignore[return-value]

    def search_policy(self, board: chess.Board, rng: random.Random | None = None) -> dict[str, float]:
        self.search(board, rng=rng)
        if self.last_root is None or not self.last_root.children:
            raise ValueError("search produced no root statistics")
        total_visits = sum(child.visits for child in self.last_root.children)
        if total_visits <= 0:
            raise ValueError("search produced zero child visits")
        return {
            child.move.uci(): child.visits / total_visits
            for child in self.last_root.children
            if child.move is not None
        }

    def search(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot search from a terminal board")
        root = PUCTNode.root(board.copy(stack=False))
        root_priors, _root_value = self.evaluator.evaluate(board.copy(stack=False))
        root.expand(root_priors)

        immediate = self._immediate_winning_move(board)
        if immediate is not None:
            self.last_root = root
            for _ in range(self.iterations):
                child = self._child_for_move(root, immediate)
                child_board = board.copy(stack=False)
                child_board.push(immediate)
                child.backpropagate(result_to_white_reward(child_board.result(claim_draw=True)))
            return immediate

        for _ in range(self.iterations):
            leaf = self._select_leaf(root)
            leaf_board = leaf.board()
            if leaf_board.is_game_over(claim_draw=True):
                value = result_to_white_reward(leaf_board.result(claim_draw=True))
            else:
                priors, value = self.evaluator.evaluate(leaf_board)
                leaf.expand(priors)
            leaf.backpropagate(value)

        self.last_root = root
        return self._robust_child(root).move  # type: ignore[return-value]

    def _select_leaf(self, root: PUCTNode) -> PUCTNode:
        node = root
        while node.expanded and node.children:
            node = node.best_child(self.c_puct)
            if not node.expanded:
                return node
        return node

    def _robust_child(self, root: PUCTNode) -> PUCTNode:
        if not root.children:
            raise ValueError("search produced no children")
        return max(root.children, key=lambda child: (child.visits, child.mean_value, child.prior, child.move.uci() if child.move else ""))

    def _child_for_move(self, root: PUCTNode, move: chess.Move) -> PUCTNode:
        for child in root.children:
            if child.move == move:
                return child
        raise ValueError(f"root has no child for legal move {move.uci()}")

    def _immediate_winning_move(self, board: chess.Board) -> chess.Move | None:
        side = board.turn
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            if not child.is_game_over(claim_draw=True):
                continue
            reward = result_to_white_reward(child.result(claim_draw=True))
            if (side == chess.WHITE and reward > 0) or (side == chess.BLACK and reward < 0):
                return move
        return None
