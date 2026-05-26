from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingRunConfig:
    iterations: int
    games_per_iteration: int
    simulations: int
    max_plies: int
    train_steps: int
    batch_size: int
    replay_capacity: int
    learning_rate: float
    temperature: float
    hidden_channels: int
    validation_games: int
    validation_max_plies: int


FIRST_MEANINGFUL_RUN = TrainingRunConfig(
    iterations=3,
    games_per_iteration=2,
    simulations=32,
    max_plies=120,
    train_steps=4,
    batch_size=128,
    replay_capacity=5_000,
    learning_rate=1e-3,
    temperature=1.0,
    hidden_channels=32,
    validation_games=4,
    validation_max_plies=160,
)
