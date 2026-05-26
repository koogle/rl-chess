from __future__ import annotations

from dataclasses import dataclass, field
import random

import torch

from rl_chess.nn_model import NeuralPolicyValueEvaluator, PolicyValueNet, PolicyValueTrainer
from rl_chess.self_play import TrainingExample, play_self_game


@dataclass(frozen=True)
class TrainMetrics:
    iterations: int
    games: int
    examples: int
    terminal_games: int
    truncated_games: int
    replay_size: int
    loss_curve: list[float] = field(default_factory=list)
    policy_loss_curve: list[float] = field(default_factory=list)
    value_loss_curve: list[float] = field(default_factory=list)


def train(
    model: PolicyValueNet,
    iterations: int,
    games_per_iteration: int = 1,
    simulations: int = 64,
    max_plies: int | None = 200,
    train_steps: int = 1,
    batch_size: int = 64,
    replay_capacity: int = 10_000,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    seed: int | None = None,
) -> TrainMetrics:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if games_per_iteration <= 0:
        raise ValueError("games_per_iteration must be positive")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if max_plies is not None and max_plies <= 0:
        raise ValueError("max_plies must be positive or None")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)
    trainer = PolicyValueTrainer(model, learning_rate=learning_rate)
    replay: list[TrainingExample] = []
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    examples = 0
    terminal_games = 0
    truncated_games = 0

    for iteration in range(iterations):
        for game_idx in range(games_per_iteration):
            game_seed = None if seed is None else seed + iteration * games_per_iteration + game_idx
            game = play_self_game(
                NeuralPolicyValueEvaluator(model),
                simulations=simulations,
                max_plies=max_plies,
                temperature=temperature,
                seed=game_seed,
            )
            replay.extend(game.examples)
            del replay[:-replay_capacity]
            examples += len(game.examples)
            terminal_games += int(not game.stats.truncated)
            truncated_games += int(game.stats.truncated)

        for _ in range(train_steps):
            if not replay:
                continue
            batch = rng.sample(replay, k=min(batch_size, len(replay)))
            stats = trainer.train_batch(batch)
            losses.append(stats.total_loss)
            policy_losses.append(stats.policy_loss)
            value_losses.append(stats.value_loss)

    return TrainMetrics(
        iterations=iterations,
        games=iterations * games_per_iteration,
        examples=examples,
        terminal_games=terminal_games,
        truncated_games=truncated_games,
        replay_size=len(replay),
        loss_curve=losses,
        policy_loss_curve=policy_losses,
        value_loss_curve=value_losses,
    )
