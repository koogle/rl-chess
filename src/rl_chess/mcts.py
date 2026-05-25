from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol

import chess

from rl_chess.env import result_to_white_reward


class Evaluator(Protocol):
    def evaluate(self, board: chess.Board, rng: random.Random) -> float:
        """Return rollout value from White's perspective."""
        ...


@dataclass
class RandomRolloutEvaluator:
    """Pure from-scratch rollout evaluator.

    It does not call an engine. It repeatedly samples legal moves from
    `python-chess` until terminal or `max_depth`, then returns the terminal
    result from White's perspective. Non-terminal truncated rollouts are 0.
    """

    max_depth: int = 80

    def evaluate(self, board: chess.Board, rng: random.Random) -> float:
        rollout = board.copy(stack=False)
        for _ in range(self.max_depth):
            if rollout.is_game_over(claim_draw=True):
                return result_to_white_reward(rollout.result(claim_draw=True))
            legal_moves = list(rollout.legal_moves)
            if not legal_moves:
                break
            rollout.push(rng.choice(legal_moves))
        if rollout.is_game_over(claim_draw=True):
            return result_to_white_reward(rollout.result(claim_draw=True))
        return 0.0


@dataclass
class MCTSNode:
    board_fen: str
    player_to_move: bool
    parent: MCTSNode | None = None
    move: chess.Move | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: list[MCTSNode] = field(default_factory=list)
    untried_moves: list[chess.Move] = field(default_factory=list)

    @classmethod
    def root(cls, board: chess.Board, rng: random.Random | None = None) -> MCTSNode:
        return cls.from_board(board=board, parent=None, move=None, rng=rng)

    @classmethod
    def from_board(
        cls,
        board: chess.Board,
        parent: MCTSNode | None,
        move: chess.Move | None,
        rng: random.Random | None = None,
    ) -> MCTSNode:
        untried = list(board.legal_moves)
        # Shuffle once so expansion is not always python-chess generation order.
        if rng is not None:
            rng.shuffle(untried)
        return cls(
            board_fen=board.fen(),
            player_to_move=board.turn,
            parent=parent,
            move=move,
            untried_moves=untried,
        )

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def board(self) -> chess.Board:
        return chess.Board(self.board_fen)

    def expand(self, rng: random.Random | None = None) -> MCTSNode:
        if not self.untried_moves:
            raise ValueError("cannot expand a fully expanded node")
        index = -1 if rng is None else rng.randrange(len(self.untried_moves))
        move = self.untried_moves.pop(index)
        child_board = self.board()
        child_board.push(move)
        child = MCTSNode.from_board(child_board, parent=self, move=move, rng=rng)
        self.children.append(child)
        return child

    def best_child(self, exploration: float) -> MCTSNode:
        if not self.children:
            raise ValueError("node has no children")
        return max(self.children, key=lambda child: uct_score(self, child, exploration))

    def backpropagate(self, white_reward: float) -> None:
        node: MCTSNode | None = self
        while node is not None:
            node.visits += 1
            # Store values from the perspective of the player to move at node.
            node.value_sum += white_reward if node.player_to_move == chess.WHITE else -white_reward
            node = node.parent


def uct_score(parent: MCTSNode, child: MCTSNode, exploration: float = math.sqrt(2.0)) -> float:
    if child.visits == 0:
        return float("inf")
    exploitation = child.mean_value
    exploration_bonus = exploration * math.sqrt(math.log(max(parent.visits, 1)) / child.visits)
    return exploitation + exploration_bonus


@dataclass
class MCTS:
    """Hand-written UCT Monte Carlo Tree Search policy."""

    iterations: int = 100
    exploration: float = math.sqrt(2.0)
    evaluator: Evaluator = field(default_factory=RandomRolloutEvaluator)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        self.rng = random.Random(self.seed)
        self.last_root: MCTSNode | None = None

    def select_move(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        return self.search(board, rng=rng)

    def search(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot search from a terminal board")
        search_rng = rng if rng is not None else self.rng
        root = MCTSNode.root(board.copy(stack=False), rng=search_rng)

        # Tactical shortcut from legal move generation only: if a move ends the
        # game, prefer the immediate win for side-to-move before rollouts.
        immediate = self._immediate_winning_move(board)
        if immediate is not None:
            self.last_root = root
            for _ in range(self.iterations):
                child = self._ensure_child_for_move(root, immediate)
                child_board = board.copy(stack=False)
                child_board.push(immediate)
                child.backpropagate(result_to_white_reward(child_board.result(claim_draw=True)))
            return immediate

        for _ in range(self.iterations):
            leaf = self._tree_policy(root)
            reward = self.evaluator.evaluate(leaf.board(), search_rng)
            leaf.backpropagate(reward)

        self.last_root = root
        return self._robust_child(root).move  # type: ignore[return-value]

    def _tree_policy(self, root: MCTSNode) -> MCTSNode:
        node = root
        while True:
            board = node.board()
            if board.is_game_over(claim_draw=True):
                return node
            if not node.is_fully_expanded:
                return node.expand(self.rng)
            node = node.best_child(self.exploration)

    def _robust_child(self, root: MCTSNode) -> MCTSNode:
        if not root.children:
            raise ValueError("search produced no children")
        return max(root.children, key=lambda child: (child.visits, child.mean_value, child.move.uci() if child.move else ""))

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

    def _ensure_child_for_move(self, root: MCTSNode, move: chess.Move) -> MCTSNode:
        for child in root.children:
            if child.move == move:
                return child
        if move in root.untried_moves:
            root.untried_moves.remove(move)
        child_board = root.board()
        child_board.push(move)
        child = MCTSNode.from_board(child_board, parent=root, move=move, rng=self.rng)
        root.children.append(child)
        return child
