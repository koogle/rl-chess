from __future__ import annotations

from dataclasses import dataclass
import random

import chess

from rl_chess.agents import Policy
from rl_chess.env import ChessEnv
from rl_chess.replay import Transition


@dataclass(frozen=True)
class Episode:
    transitions: list[Transition]
    result: str
    winner_reward: float


def play_episode(
    env: ChessEnv,
    white_policy: Policy,
    black_policy: Policy,
    max_plies: int = 200,
    seed: int | None = None,
    assign_returns: bool = False,
) -> Episode:
    """Run one self-play game by alternating policies until terminal/max plies."""

    if max_plies <= 0:
        raise ValueError("max_plies must be positive")

    rng = random.Random(seed)
    obs = env.reset()
    transitions: list[Transition] = []
    final_reward = 0.0
    result = "*"

    for _ply in range(max_plies):
        board_before = env.board.copy(stack=False)
        player = board_before.turn
        policy = white_policy if player == chess.WHITE else black_policy
        move = policy.select_move(board_before, rng=rng)

        next_obs, reward, done, info = env.step(move)
        transitions.append(
            Transition(
                state_ascii=obs.board_ascii,
                action_uci=move.uci(),
                player=player,
                reward=reward if done else 0.0,
                done=done,
                next_state_ascii=next_obs.board_ascii,
                result=info["result"],
            )
        )
        obs = next_obs

        if done:
            final_reward = reward
            result = info["result"]
            break

    if assign_returns:
        transitions = assign_episode_returns(transitions, final_reward)

    return Episode(transitions=transitions, result=result, winner_reward=final_reward)


def assign_episode_returns(transitions: list[Transition], white_reward: float) -> list[Transition]:
    """Attach terminal outcome as returns from each acting player's perspective."""

    assigned: list[Transition] = []
    for transition in transitions:
        actor_return = white_reward if transition.player == chess.WHITE else -white_reward
        assigned.append(transition.with_return(actor_return))
    return assigned
