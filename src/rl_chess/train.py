from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

import chess
import torch

from rl_chess.nn_model import PolicyValueNet, TrainStats, train_batch
from rl_chess.self_play import GameStats, SelfPlayGame, TrainingExample, play_self_game


@dataclass(frozen=True)
class TrainMetrics:
    iterations: int
    games: int
    examples: int
    terminal_games: int
    iteration_examples: int
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
        "iteration_examples": metrics.iteration_examples,
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


def _model_snapshot(model: PolicyValueNet) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _play_chunk(
    state_dict: dict[str, torch.Tensor],
    hidden_channels: int,
    residual_blocks: int,
    seeds: Sequence[int | None],
    simulations: int,
    max_plies: int | None,
    temperature: float,
    starting_board: chess.Board | None,
) -> list[SelfPlayGame]:
    worker_model = PolicyValueNet(hidden_channels=hidden_channels, residual_blocks=residual_blocks)
    worker_model.load_state_dict(state_dict)
    worker_model.eval()
    return [
        play_self_game(
            worker_model,
            simulations=simulations,
            max_plies=max_plies,
            temperature=temperature,
            seed=game_seed,
            starting_board=starting_board,
        )
        for game_seed in seeds
    ]


def generate_self_play_batch(
    model: PolicyValueNet,
    games: int,
    simulations: int,
    max_plies: int | None,
    temperature: float,
    seed_offset: int | None,
    starting_board: chess.Board | None = None,
    self_play_workers: int = 1,
) -> list[SelfPlayGame]:
    """Generate one fresh self-play batch from a frozen snapshot of the latest model."""

    if games <= 0:
        raise ValueError("games must be positive")
    if self_play_workers <= 0:
        raise ValueError("self_play_workers must be positive")

    seeds = [None if seed_offset is None else seed_offset + game_idx for game_idx in range(games)]
    if self_play_workers == 1 or games == 1:
        return [
            play_self_game(
                model,
                simulations=simulations,
                max_plies=max_plies,
                temperature=temperature,
                seed=game_seed,
                starting_board=starting_board,
            )
            for game_seed in seeds
        ]

    workers = min(self_play_workers, games)
    chunks = [seeds[index::workers] for index in range(workers)]
    state_dict = _model_snapshot(model)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunk_results = executor.map(
            lambda chunk: _play_chunk(
                state_dict=state_dict,
                hidden_channels=model.hidden_channels,
                residual_blocks=model.residual_blocks,
                seeds=chunk,
                simulations=simulations,
                max_plies=max_plies,
                temperature=temperature,
                starting_board=starting_board,
            ),
            chunks,
        )
    games_by_chunk = list(chunk_results)
    ordered: list[SelfPlayGame] = []
    for game_index in range(games):
        chunk_index = game_index % workers
        position_in_chunk = game_index // workers
        ordered.append(games_by_chunk[chunk_index][position_in_chunk])
    return ordered


def _fresh_epoch_minibatches(
    fresh_examples: Sequence[TrainingExample],
    batch_size: int,
    fresh_batch_epochs: int,
    train_steps: int,
    rng: random.Random,
) -> list[list[TrainingExample]]:
    """Return shuffled minibatches drawn only from the current fresh batch.

    ``fresh_batch_epochs`` controls coherent passes over the whole fresh batch: each
    example appears exactly once per epoch, split into minibatches of at most
    ``batch_size``. ``train_steps`` is retained as a minimum optimizer-update count;
    if it exceeds the requested epoch minibatches, extra shuffled fresh-batch passes
    are appended and truncated at ``train_steps`` updates. No examples from prior
    iterations are retained or sampled.
    """

    if not fresh_examples:
        return []

    batches: list[list[TrainingExample]] = []

    def append_epoch() -> None:
        epoch_examples = list(fresh_examples)
        rng.shuffle(epoch_examples)
        for start in range(0, len(epoch_examples), batch_size):
            batches.append(epoch_examples[start : start + batch_size])

    for _ in range(fresh_batch_epochs):
        append_epoch()

    target_updates = max(train_steps, len(batches))
    while len(batches) < target_updates:
        append_epoch()

    return batches[:target_updates]


def train(
    model: PolicyValueNet,
    iterations: int,
    games_per_iteration: int = 1,
    simulations: int = 64,
    max_plies: int | None = None,
    train_steps: int = 1,
    fresh_batch_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
    starting_board: Any | None = None,
    self_play_workers: int = 1,
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
    if fresh_batch_epochs <= 0:
        raise ValueError("fresh_batch_epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if self_play_workers <= 0:
        raise ValueError("self_play_workers must be positive")

    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    checkpoint_paths: list[Path] = []
    examples = 0
    terminal_games = 0
    latest_iteration_examples = 0

    for iteration in range(iterations):
        seed_offset = None if seed is None else seed + iteration * games_per_iteration
        games = generate_self_play_batch(
            model=model,
            games=games_per_iteration,
            simulations=simulations,
            max_plies=max_plies,
            temperature=temperature,
            seed_offset=seed_offset,
            starting_board=starting_board,
            self_play_workers=self_play_workers,
        )
        fresh_examples = [example for game in games for example in game.examples]
        latest_iteration_examples = len(fresh_examples)
        examples += latest_iteration_examples
        terminal_games += len(games)

        for batch in _fresh_epoch_minibatches(
            fresh_examples=fresh_examples,
            batch_size=batch_size,
            fresh_batch_epochs=fresh_batch_epochs,
            train_steps=train_steps,
            rng=rng,
        ):
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
                iteration_examples=latest_iteration_examples,
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
                        "iteration_examples": latest_iteration_examples,
                        "terminal_games": terminal_games,
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
        iteration_examples=latest_iteration_examples,
        loss_curve=losses,
        policy_loss_curve=policy_losses,
        value_loss_curve=value_losses,
        checkpoint_paths=checkpoint_paths,
    )
