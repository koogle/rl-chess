from __future__ import annotations

from dataclasses import dataclass
import random

import chess

from rl_chess.env import ascii_to_board, board_to_ascii, result_to_white_reward
from rl_chess.puct_mcts import PUCTMCTS, PolicyValueEvaluator


@dataclass(frozen=True)
class TrainingExample:
    """One AlphaZero-style example: visual state, improved policy, final outcome."""

    state_ascii: str
    turn: bool
    policy_target: dict[str, float]
    value_target: float
    castling_rights: int = 0
    ep_square: int | None = None
    halfmove_clock: int = 0
    can_claim_threefold: bool = False
    can_claim_fifty_moves: bool = False

    def __post_init__(self) -> None:
        if not self.policy_target:
            raise ValueError("policy_target must not be empty")
        total = 0.0
        for move, weight in self.policy_target.items():
            try:
                chess.Move.from_uci(move)
            except ValueError as exc:
                raise ValueError(f"invalid UCI move in policy_target: {move!r}") from exc
            if weight < 0:
                raise ValueError("policy_target weights must be non-negative")
            total += weight
        if total <= 0:
            raise ValueError("policy_target must have positive total weight")
        if not -1.0 <= self.value_target <= 1.0:
            raise ValueError("value_target must be in [-1, 1]")
        if self.halfmove_clock < 0:
            raise ValueError("halfmove_clock must be non-negative")

    @classmethod
    def from_board(cls, board: chess.Board, policy_target: dict[str, float], value_target: float) -> TrainingExample:
        return cls(
            state_ascii=board_to_ascii(board),
            turn=board.turn,
            policy_target=policy_target,
            value_target=value_target,
            castling_rights=board.castling_rights,
            ep_square=board.ep_square,
            halfmove_clock=board.halfmove_clock,
            can_claim_threefold=board.can_claim_threefold_repetition(),
            can_claim_fifty_moves=board.can_claim_fifty_moves(),
        )

    def color_flipped(self) -> TrainingExample:
        """Return the legal color-swap/rank-mirror equivalent example."""

        board = ascii_to_board(self.state_ascii, self.turn)
        board.castling_rights = self.castling_rights
        board.ep_square = self.ep_square
        board.halfmove_clock = self.halfmove_clock
        mirrored = board.mirror()
        return TrainingExample(
            state_ascii=board_to_ascii(mirrored),
            turn=mirrored.turn,
            policy_target={mirror_move_uci(move): weight for move, weight in self.policy_target.items()},
            value_target=self.value_target,
            castling_rights=mirrored.castling_rights,
            ep_square=mirrored.ep_square,
            halfmove_clock=mirrored.halfmove_clock,
            can_claim_threefold=self.can_claim_threefold,
            can_claim_fifty_moves=self.can_claim_fifty_moves,
        )


@dataclass(frozen=True)
class GameStats:
    plies: int
    result: str


@dataclass(frozen=True)
class SelfPlayGame:
    examples: list[TrainingExample]
    stats: GameStats


def play_self_game(
    model_evaluator: PolicyValueEvaluator,
    simulations: int = 64,
    max_plies: int | None = None,
    temperature: float = 1.0,
    seed: int | None = None,
    starting_board: chess.Board | None = None,
) -> SelfPlayGame:
    """Generate one NN-guided PUCT self-play game.

    Games always play until python-chess says the position is terminal. A
    `max_plies` value is only a safety guard: reaching it on a non-terminal game
    raises instead of turning an incomplete game into a draw target.
    """

    if max_plies is not None and max_plies <= 0:
        raise ValueError("max_plies must be positive or None")

    board = starting_board.copy(stack=True) if starting_board is not None else chess.Board()
    rng = random.Random(seed)
    pending: list[tuple[chess.Board, dict[str, float]]] = []
    mcts = PUCTMCTS(evaluator=model_evaluator, iterations=simulations, seed=seed)

    plies = 0
    while True:
        if board.is_game_over(claim_draw=True):
            break
        if max_plies is not None and plies >= max_plies:
            raise RuntimeError("non-terminal self-play game reached safety cap")
        policy = mcts.search_policy(board, add_root_noise=True)
        pending.append((board.copy(stack=True), policy))
        board.push(chess.Move.from_uci(sample_policy(policy, temperature, rng)))
        plies += 1

    result = board.result(claim_draw=True)
    white_reward = result_to_white_reward(result)
    examples = [
        TrainingExample.from_board(
            board=example_board,
            policy_target=policy,
            value_target=white_reward if example_board.turn == chess.WHITE else -white_reward,
        )
        for example_board, policy in pending
    ]
    return SelfPlayGame(examples=examples, stats=GameStats(len(pending), result))


def mirror_move_uci(uci: str) -> str:
    move = chess.Move.from_uci(uci)
    mirrored = chess.Move(
        from_square=chess.square_mirror(move.from_square),
        to_square=chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )
    return mirrored.uci()


def augment_examples_color_flip(examples: list[TrainingExample]) -> list[TrainingExample]:
    augmented: list[TrainingExample] = []
    for example in examples:
        augmented.append(example)
        augmented.append(example.color_flipped())
    return augmented


def sample_policy(policy: dict[str, float], temperature: float, rng: random.Random) -> str:
    if not policy:
        raise ValueError("cannot sample from empty policy")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return max(policy, key=policy.__getitem__)

    moves = tuple(policy)
    weights = [max(policy[move], 0.0) ** (1.0 / temperature) for move in moves]
    total = sum(weights)
    if total <= 0:
        return rng.choice(moves)

    threshold = rng.random() * total
    cumulative = 0.0
    for move, weight in zip(moves, weights):
        cumulative += weight
        if cumulative >= threshold:
            return move
    return moves[-1]
