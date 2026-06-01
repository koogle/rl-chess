from __future__ import annotations

from dataclasses import dataclass
import random

import chess

from rl_chess.env import board_to_ascii, result_to_white_reward
from rl_chess.puct_mcts import PUCTMCTS, PolicyValueEvaluator


@dataclass(frozen=True)
class TrainingExample:
    """One AlphaZero-style example: visual state, improved policy, final outcome."""

    state_ascii: str
    turn: bool
    policy_target: dict[str, float]
    value_target: float

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


@dataclass(frozen=True)
class GameStats:
    plies: int
    result: str


@dataclass(frozen=True)
class SelfPlayGame:
    examples: list[TrainingExample]
    stats: GameStats


def validate_draw_value(draw_value: float) -> float:
    """Validate the actor-perspective value target used for drawn self-play games."""

    if not -1.0 <= draw_value <= 1.0:
        raise ValueError("draw_value must be in [-1, 1]")
    return draw_value


def play_self_game(
    model_evaluator: PolicyValueEvaluator,
    simulations: int = 64,
    max_plies: int | None = None,
    temperature: float = 1.0,
    seed: int | None = None,
    starting_board: chess.Board | None = None,
    draw_value: float = 0.0,
) -> SelfPlayGame:
    """Generate one NN-guided PUCT self-play game.

    Games always play until python-chess says the position is terminal. A
    `max_plies` value is only a safety guard: reaching it on a non-terminal game
    raises instead of turning an incomplete game into a draw target.

    `draw_value` is optional training-target shaping for self-play draws. The
    default `0.0` preserves normal chess result targets; experimental negative
    values (for example `-0.05`) make every actor in a drawn game receive that
    small actor-perspective value target.
    """

    if max_plies is not None and max_plies <= 0:
        raise ValueError("max_plies must be positive or None")
    draw_value = validate_draw_value(draw_value)

    board = starting_board.copy(stack=False) if starting_board is not None else chess.Board()
    rng = random.Random(seed)
    pending: list[tuple[str, bool, dict[str, float]]] = []
    mcts = PUCTMCTS(evaluator=model_evaluator, iterations=simulations, seed=seed)

    plies = 0
    while True:
        if board.is_game_over(claim_draw=True):
            break
        if max_plies is not None and plies >= max_plies:
            raise RuntimeError("non-terminal self-play game reached safety cap")
        policy = mcts.search_policy(board, add_root_noise=True)
        pending.append((board_to_ascii(board), board.turn, policy))
        board.push(chess.Move.from_uci(sample_policy(policy, temperature, rng)))
        plies += 1

    result = board.result(claim_draw=True)
    white_reward = result_to_white_reward(result)

    def actor_value(turn: bool) -> float:
        if result == "1/2-1/2":
            return draw_value
        return white_reward if turn == chess.WHITE else -white_reward

    examples = [
        TrainingExample(
            state_ascii=state_ascii,
            turn=turn,
            policy_target=policy,
            value_target=actor_value(turn),
        )
        for state_ascii, turn, policy in pending
    ]
    return SelfPlayGame(examples=examples, stats=GameStats(len(pending), result))


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
