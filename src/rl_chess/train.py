from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from collections.abc import Callable
from typing import Any

import torch

from rl_chess.nn_model import PolicyValueNet, train_batch
from rl_chess.self_play import TrainingExample, play_self_game


@dataclass(frozen=True)
class TrainMetrics:
    iterations: int
    games: int
    examples: int
    terminal_games: int
    replay_size: int
    loss_curve: list[float] = field(default_factory=list)
    policy_loss_curve: list[float] = field(default_factory=list)
    value_loss_curve: list[float] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)


def save_checkpoint(model: PolicyValueNet, path: str | Path, metrics: TrainMetrics | None = None) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_channels": model.hidden_channels,
            "residual_blocks": model.residual_blocks,
            "model_state_dict": model.state_dict(),
            "metrics": None if metrics is None else checkpoint_metrics(metrics),
        },
        checkpoint_path,
    )
    return checkpoint_path


def checkpoint_metrics(metrics: TrainMetrics) -> dict[str, object]:
    return {
        "iterations": metrics.iterations,
        "games": metrics.games,
        "examples": metrics.examples,
        "terminal_games": metrics.terminal_games,
        "replay_size": metrics.replay_size,
        "loss_curve": list(metrics.loss_curve),
        "policy_loss_curve": list(metrics.policy_loss_curve),
        "value_loss_curve": list(metrics.value_loss_curve),
        "checkpoint_paths": [str(path) for path in metrics.checkpoint_paths],
    }


def load_checkpoint_model(path: str | Path) -> PolicyValueNet:
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = PolicyValueNet(
        hidden_channels=int(checkpoint["hidden_channels"]),
        residual_blocks=int(checkpoint.get("residual_blocks", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def train(
    model: PolicyValueNet,
    iterations: int,
    games_per_iteration: int = 1,
    simulations: int = 64,
    max_plies: int | None = None,
    train_steps: int = 1,
    batch_size: int = 64,
    replay_capacity: int = 10_000,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
    starting_board: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay: list[TrainingExample] = []
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    checkpoint_paths: list[Path] = []
    examples = 0
    terminal_games = 0

    for iteration in range(iterations):
        for game_idx in range(games_per_iteration):
            game_seed = None if seed is None else seed + iteration * games_per_iteration + game_idx
            game = play_self_game(
                model,
                simulations=simulations,
                max_plies=max_plies,
                temperature=temperature,
                seed=game_seed,
                starting_board=starting_board,
            )
            replay.extend(game.examples)
            del replay[:-replay_capacity]
            examples += len(game.examples)
            terminal_games += 1

        for _ in range(train_steps):
            if not replay:
                continue
            batch = rng.sample(replay, k=min(batch_size, len(replay)))
            stats = train_batch(model, optimizer, batch)
            losses.append(stats.total_loss)
            policy_losses.append(stats.policy_loss)
            value_losses.append(stats.value_loss)

        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir) / f"iteration-{iteration + 1:04d}.pt"
            snapshot = TrainMetrics(
                iterations=iteration + 1,
                games=(iteration + 1) * games_per_iteration,
                examples=examples,
                terminal_games=terminal_games,
                replay_size=len(replay),
                loss_curve=list(losses),
                policy_loss_curve=list(policy_losses),
                value_loss_curve=list(value_losses),
                checkpoint_paths=[*checkpoint_paths, checkpoint_path],
            )
            checkpoint_paths.append(save_checkpoint(model, checkpoint_path, snapshot))
            if progress_callback is not None:
                progress_callback(
                    {
                        "iteration": iteration + 1,
                        "games": (iteration + 1) * games_per_iteration,
                        "examples": examples,
                        "terminal_games": terminal_games,
                        "replay_size": len(replay),
                        "updates": len(losses),
                        "latest_loss": losses[-1] if losses else None,
                        "latest_policy_loss": policy_losses[-1] if policy_losses else None,
                        "latest_value_loss": value_losses[-1] if value_losses else None,
                        "checkpoint_path": checkpoint_path,
                    }
                )

    return TrainMetrics(
        iterations=iterations,
        games=iterations * games_per_iteration,
        examples=examples,
        terminal_games=terminal_games,
        replay_size=len(replay),
        loss_curve=losses,
        policy_loss_curve=policy_losses,
        value_loss_curve=value_losses,
        checkpoint_paths=checkpoint_paths,
    )
