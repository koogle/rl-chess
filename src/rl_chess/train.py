from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rl_chess.agents import TabularMoveValueAgent, TabularPolicyDistiller
from rl_chess.env import ChessEnv
from rl_chess.mcts import MCTS, RandomRolloutEvaluator
from rl_chess.nn_model import ChessPolicyValueNet, NeuralPolicyValueTrainer
from rl_chess.replay import ReplayBuffer
from rl_chess.search_self_play import collect_search_episode
from rl_chess.self_play import play_episode


@dataclass(frozen=True)
class TrainMetrics:
    episodes: int
    total_plies: int
    replay_size: int
    results: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MCTSTrainMetrics:
    episodes: int
    total_plies: int
    examples_collected: int
    policy_entries: int
    loss_curve: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class NeuralMCTSTrainMetrics:
    episodes: int
    total_plies: int
    examples_collected: int
    loss_curve: list[float] = field(default_factory=list)
    policy_loss_curve: list[float] = field(default_factory=list)
    value_loss_curve: list[float] = field(default_factory=list)


def train_self_play(
    agent: TabularMoveValueAgent,
    episodes: int,
    max_plies: int | None = 200,
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


def train_mcts_self_play(
    learner: TabularPolicyDistiller,
    episodes: int,
    max_plies: int | None = 200,
    mcts_iterations: int = 50,
    rollout_depth: int = 20,
    seed: int | None = None,
) -> MCTSTrainMetrics:
    """First AlphaGo-style loop: MCTS self-play then tabular distillation."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    total_plies = 0
    examples_collected = 0
    loss_curve: list[float] = []

    for episode_idx in range(episodes):
        episode_seed = None if seed is None else seed + episode_idx
        mcts = MCTS(
            iterations=mcts_iterations,
            evaluator=RandomRolloutEvaluator(max_depth=rollout_depth),
            seed=episode_seed,
        )
        examples = collect_search_episode(
            env=ChessEnv(),
            mcts=mcts,
            max_plies=max_plies,
            seed=episode_seed,
        )
        loss = learner.learn(examples)
        loss_curve.append(loss)
        total_plies += len(examples)
        examples_collected += len(examples)

    return MCTSTrainMetrics(
        episodes=episodes,
        total_plies=total_plies,
        examples_collected=examples_collected,
        policy_entries=learner.policy_entries,
        loss_curve=loss_curve,
    )


def train_neural_mcts_self_play(
    model: ChessPolicyValueNet,
    episodes: int,
    max_plies: int | None = 200,
    mcts_iterations: int = 50,
    rollout_depth: int = 20,
    learning_rate: float = 1e-3,
    seed: int | None = None,
) -> NeuralMCTSTrainMetrics:
    """AlphaGo-style loop: MCTS self-play then neural policy/value update."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    if seed is not None:
        torch.manual_seed(seed)

    trainer = NeuralPolicyValueTrainer(model=model, learning_rate=learning_rate)
    total_plies = 0
    examples_collected = 0
    loss_curve: list[float] = []
    policy_loss_curve: list[float] = []
    value_loss_curve: list[float] = []

    for episode_idx in range(episodes):
        episode_seed = None if seed is None else seed + episode_idx
        mcts = MCTS(
            iterations=mcts_iterations,
            evaluator=RandomRolloutEvaluator(max_depth=rollout_depth),
            seed=episode_seed,
        )
        examples = collect_search_episode(
            env=ChessEnv(),
            mcts=mcts,
            max_plies=max_plies,
            seed=episode_seed,
        )
        stats = trainer.train_batch(examples)
        loss_curve.append(stats.total_loss)
        policy_loss_curve.append(stats.policy_loss)
        value_loss_curve.append(stats.value_loss)
        total_plies += len(examples)
        examples_collected += len(examples)

    return NeuralMCTSTrainMetrics(
        episodes=episodes,
        total_plies=total_plies,
        examples_collected=examples_collected,
        loss_curve=loss_curve,
        policy_loss_curve=policy_loss_curve,
        value_loss_curve=value_loss_curve,
    )
