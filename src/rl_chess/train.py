from __future__ import annotations

from dataclasses import dataclass, field

from rl_chess.agents import TabularMoveValueAgent
from rl_chess.env import ChessEnv
from rl_chess.replay import ReplayBuffer
from rl_chess.self_play import play_episode


@dataclass(frozen=True)
class TrainMetrics:
    episodes: int
    total_plies: int
    replay_size: int
    results: list[str] = field(default_factory=list)


def train_self_play(
    agent: TabularMoveValueAgent,
    episodes: int,
    max_plies: int = 200,
    replay_capacity: int = 10_000,
    seed: int | None = None,
) -> TrainMetrics:
    """Minimal self-play training loop.

    The same agent plays both sides. Each episode is converted to Monte Carlo
    returns from the actor's perspective and used for an incremental tabular
    update. This is intentionally simple so the loop is inspectable.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    env = ChessEnv()
    replay = ReplayBuffer(capacity=replay_capacity)
    total_plies = 0
    results: list[str] = []

    for episode_idx in range(episodes):
        episode_seed = None if seed is None else seed + episode_idx
        episode = play_episode(
            env=env,
            white_policy=agent,
            black_policy=agent,
            max_plies=max_plies,
            seed=episode_seed,
            assign_returns=True,
        )
        replay.extend(episode.transitions)
        agent.learn(episode.transitions)
        total_plies += len(episode.transitions)
        results.append(episode.result)

    return TrainMetrics(
        episodes=episodes,
        total_plies=total_plies,
        replay_size=len(replay),
        results=results,
    )
