from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path
import random
import time
from typing import Any

import chess
import torch

from rl_chess.nn_model import PolicyValueNet, TrainStats, train_batch
from rl_chess.self_play import GameStats, SelfPlayGame, TrainingExample, augment_examples_color_flip, play_self_game


@dataclass(frozen=True)
class TrainMetrics:
    iterations: int
    games: int
    examples: int
    training_examples: int
    terminal_games: int
    iteration_examples: int
    iteration_training_examples: int
    result_counts: dict[str, int] = field(default_factory=dict)
    iteration_result_counts: dict[str, int] = field(default_factory=dict)
    average_plies: float = 0.0
    iteration_average_plies: float = 0.0
    loss_curve: list[float] = field(default_factory=list)
    policy_loss_curve: list[float] = field(default_factory=list)
    value_loss_curve: list[float] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)
    training_device: str = "cpu"


def save_checkpoint(model: PolicyValueNet, path: str | Path, metrics: TrainMetrics | None = None) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_channels": model.hidden_channels,
            "residual_blocks": model.residual_blocks,
            "model_state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
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
        "training_examples": metrics.training_examples,
        "terminal_games": metrics.terminal_games,
        "iteration_examples": metrics.iteration_examples,
        "iteration_training_examples": metrics.iteration_training_examples,
        "result_counts": dict(metrics.result_counts),
        "iteration_result_counts": dict(metrics.iteration_result_counts),
        "average_plies": metrics.average_plies,
        "iteration_average_plies": metrics.iteration_average_plies,
        "loss_curve": list(metrics.loss_curve),
        "policy_loss_curve": list(metrics.policy_loss_curve),
        "value_loss_curve": list(metrics.value_loss_curve),
        "checkpoint_paths": [str(path) for path in metrics.checkpoint_paths],
        "training_device": metrics.training_device,
    }


def load_checkpoint_model(path: str | Path) -> PolicyValueNet:
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = PolicyValueNet(
        hidden_channels=int(checkpoint["hidden_channels"]),
        residual_blocks=int(checkpoint.get("residual_blocks", 0)),
    )
    model.load_state_dict(_migrate_state_dict(checkpoint["model_state_dict"], model.state_dict()))
    return model


def _migrate_state_dict(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    migrated = dict(checkpoint_state)
    trunk_weight_key = "trunk.0.weight"
    if trunk_weight_key in migrated and migrated[trunk_weight_key].shape != model_state[trunk_weight_key].shape:
        old_weight = migrated[trunk_weight_key]
        new_weight = model_state[trunk_weight_key].clone()
        channels = min(old_weight.shape[1], new_weight.shape[1])
        new_weight[:, :channels] = old_weight[:, :channels]
        migrated[trunk_weight_key] = new_weight
    return migrated


def _model_snapshot(model: PolicyValueNet) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _self_play_process_context(start_method: str | None = None) -> multiprocessing.context.BaseContext:
    methods = multiprocessing.get_all_start_methods()
    if start_method is not None:
        if start_method not in methods:
            raise ValueError(f"unsupported multiprocessing start method: {start_method}")
        return multiprocessing.get_context(start_method)
    return multiprocessing.get_context("fork" if "fork" in methods else methods[0])


def _resolve_training_device(training_device: str | torch.device) -> torch.device:
    if str(training_device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(training_device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid training_device: {training_device}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda training_device requested but CUDA is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("mps training_device requested but MPS is not available")
    return resolved


def _play_chunk(
    state_dict: dict[str, torch.Tensor],
    hidden_channels: int,
    residual_blocks: int,
    seeds: Sequence[int | None],
    simulations: int,
    max_plies: int | None,
    temperature: float,
    final_temperature: float | None,
    temperature_drop_plies: int | None,
    starting_board: chess.Board | None,
) -> list[SelfPlayGame]:
    torch.set_num_threads(1)
    worker_model = PolicyValueNet(hidden_channels=hidden_channels, residual_blocks=residual_blocks)
    worker_model.load_state_dict(state_dict)
    worker_model.eval()
    return [
        play_self_game(
            worker_model,
            simulations=simulations,
            max_plies=max_plies,
            temperature=temperature,
            final_temperature=final_temperature,
            temperature_drop_plies=temperature_drop_plies,
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
    final_temperature: float | None = None,
    temperature_drop_plies: int | None = None,
    starting_board: chess.Board | None = None,
    self_play_workers: int = 1,
    self_play_start_method: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[SelfPlayGame]:
    """Generate one fresh self-play batch from a frozen snapshot of the latest model."""

    if games <= 0:
        raise ValueError("games must be positive")
    if self_play_workers <= 0:
        raise ValueError("self_play_workers must be positive")

    seeds = [None if seed_offset is None else seed_offset + game_idx for game_idx in range(games)]
    if self_play_workers == 1 or games == 1:
        completed: list[SelfPlayGame] = []
        started_at = time.perf_counter()
        for game_index, game_seed in enumerate(seeds):
            game = play_self_game(
                model,
                simulations=simulations,
                max_plies=max_plies,
                temperature=temperature,
                final_temperature=final_temperature,
                temperature_drop_plies=temperature_drop_plies,
                seed=game_seed,
                starting_board=starting_board,
            )
            completed.append(game)
            if progress_callback is not None:
                progress_callback(
                    {
                        "game_index": game_index,
                        "completed_games": game_index + 1,
                        "total_games": games,
                        "plies": game.stats.plies,
                        "result": game.stats.result,
                        "elapsed_seconds": time.perf_counter() - started_at,
                    }
                )
        return completed

    workers = min(self_play_workers, games)
    state_dict = _model_snapshot(model)
    started_at = time.perf_counter()
    ordered: list[SelfPlayGame | None] = [None] * games
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=_self_play_process_context(self_play_start_method),
    ) as executor:
        futures = {
            executor.submit(
                _play_chunk,
                state_dict=state_dict,
                hidden_channels=model.hidden_channels,
                residual_blocks=model.residual_blocks,
                seeds=[game_seed],
                simulations=simulations,
                max_plies=max_plies,
                temperature=temperature,
                final_temperature=final_temperature,
                temperature_drop_plies=temperature_drop_plies,
                starting_board=starting_board,
            ): game_index
            for game_index, game_seed in enumerate(seeds)
        }
        completed_games = 0
        for future in as_completed(futures):
            game_index = futures[future]
            game = future.result()[0]
            ordered[game_index] = game
            completed_games += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "game_index": game_index,
                        "completed_games": completed_games,
                        "total_games": games,
                        "plies": game.stats.plies,
                        "result": game.stats.result,
                        "elapsed_seconds": time.perf_counter() - started_at,
                    }
                )
    if any(game is None for game in ordered):
        raise RuntimeError("self-play batch completed with missing games")
    return [game for game in ordered if game is not None]


def _select_training_games(
    games: list[SelfPlayGame],
    rng: random.Random,
    draw_training_weight: float,
    min_draw_games_for_training: int,
) -> list[SelfPlayGame]:
    selected = [
        game
        for game in games
        if game.stats.result != "1/2-1/2" or rng.random() < draw_training_weight
    ]
    if selected:
        return selected

    draw_games = [game for game in games if game.stats.result == "1/2-1/2"]
    if not draw_games or min_draw_games_for_training == 0:
        return selected

    return rng.sample(draw_games, k=min(min_draw_games_for_training, len(draw_games)))


def train(
    model: PolicyValueNet,
    iterations: int,
    games_per_iteration: int = 1,
    simulations: int = 64,
    max_plies: int | None = None,
    train_steps: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    final_temperature: float | None = None,
    temperature_drop_plies: int | None = None,
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
    starting_board: Any | None = None,
    self_play_workers: int = 1,
    augment_color_flip: bool = True,
    draw_training_weight: float = 1.0,
    min_draw_games_for_training: int = 0,
    training_device: str | torch.device = "cpu",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
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
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if final_temperature is not None and final_temperature < 0:
        raise ValueError("final_temperature must be non-negative or None")
    if temperature_drop_plies is not None and temperature_drop_plies < 0:
        raise ValueError("temperature_drop_plies must be non-negative or None")
    if self_play_workers <= 0:
        raise ValueError("self_play_workers must be positive")
    if not 0.0 <= draw_training_weight <= 1.0:
        raise ValueError("draw_training_weight must be in [0, 1]")
    if min_draw_games_for_training < 0:
        raise ValueError("min_draw_games_for_training must be non-negative")

    resolved_training_device = _resolve_training_device(training_device)
    resolved_training_device_name = str(resolved_training_device)
    model.to(resolved_training_device)
    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    checkpoint_paths: list[Path] = []
    examples = 0
    training_examples = 0
    terminal_games = 0
    latest_iteration_examples = 0
    latest_iteration_training_examples = 0
    result_counts: Counter[str] = Counter()
    total_plies = 0
    latest_iteration_result_counts: dict[str, int] = {}
    latest_iteration_average_plies = 0.0

    for iteration in range(iterations):
        seed_offset = None if seed is None else seed + iteration * games_per_iteration
        iteration_start = time.perf_counter()
        if event_callback is not None:
            event_callback(
                {
                    "phase": "self_play_start",
                    "iteration": iteration + 1,
                    "games_per_iteration": games_per_iteration,
                    "simulations": simulations,
                    "self_play_workers": self_play_workers,
                    "training_device": resolved_training_device_name,
                }
            )
        self_play_start_method = "spawn" if resolved_training_device.type == "cuda" else None

        def report_self_play_game(progress: dict[str, Any]) -> None:
            if event_callback is None:
                return
            event_callback({"phase": "self_play_game_complete", "iteration": iteration + 1, **progress})

        games = generate_self_play_batch(
            model=model,
            games=games_per_iteration,
            simulations=simulations,
            max_plies=max_plies,
            temperature=temperature,
            final_temperature=final_temperature,
            temperature_drop_plies=temperature_drop_plies,
            seed_offset=seed_offset,
            starting_board=starting_board,
            self_play_workers=self_play_workers,
            self_play_start_method=self_play_start_method,
            progress_callback=report_self_play_game if event_callback is not None else None,
        )
        self_play_elapsed_seconds = time.perf_counter() - iteration_start
        raw_examples = [example for game in games for example in game.examples]
        selected_training_games = _select_training_games(
            games=games,
            rng=rng,
            draw_training_weight=draw_training_weight,
            min_draw_games_for_training=min_draw_games_for_training,
        )
        train_source_examples = [example for game in selected_training_games for example in game.examples]
        training_fresh_examples = (
            augment_examples_color_flip(train_source_examples) if augment_color_flip else train_source_examples
        )
        latest_iteration_examples = len(raw_examples)
        latest_iteration_training_examples = len(training_fresh_examples)
        examples += latest_iteration_examples
        training_examples += latest_iteration_training_examples
        terminal_games += len(games)
        iteration_result_counter = Counter(game.stats.result for game in games)
        latest_iteration_result_counts = dict(iteration_result_counter)
        latest_iteration_plies = sum(game.stats.plies for game in games)
        latest_iteration_average_plies = latest_iteration_plies / len(games)
        result_counts.update(iteration_result_counter)
        total_plies += latest_iteration_plies
        if event_callback is not None:
            event_callback(
                {
                    "phase": "self_play_complete",
                    "iteration": iteration + 1,
                    "iteration_examples": latest_iteration_examples,
                    "iteration_training_examples": latest_iteration_training_examples,
                    "iteration_result_counts": latest_iteration_result_counts,
                    "iteration_average_plies": latest_iteration_average_plies,
                    "elapsed_seconds": self_play_elapsed_seconds,
                }
            )

        training_start = time.perf_counter()
        updates_before = len(losses)
        for _ in range(train_steps):
            if not training_fresh_examples:
                continue
            batch = rng.sample(training_fresh_examples, k=min(batch_size, len(training_fresh_examples)))
            stats = train_batch(model, optimizer, batch)
            losses.append(stats.total_loss)
            policy_losses.append(stats.policy_loss)
            value_losses.append(stats.value_loss)
        if event_callback is not None:
            event_callback(
                {
                    "phase": "train_updates_complete",
                    "iteration": iteration + 1,
                    "updates": len(losses) - updates_before,
                    "elapsed_seconds": time.perf_counter() - training_start,
                    "latest_loss": losses[-1] if losses else None,
                    "latest_policy_loss": policy_losses[-1] if policy_losses else None,
                    "latest_value_loss": value_losses[-1] if value_losses else None,
                }
            )

        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir) / f"iteration-{iteration + 1:04d}.pt"
            snapshot = TrainMetrics(
                iterations=iteration + 1,
                games=(iteration + 1) * games_per_iteration,
                examples=examples,
                training_examples=training_examples,
                terminal_games=terminal_games,
                iteration_examples=latest_iteration_examples,
                iteration_training_examples=latest_iteration_training_examples,
                result_counts=dict(result_counts),
                iteration_result_counts=latest_iteration_result_counts,
                average_plies=total_plies / ((iteration + 1) * games_per_iteration),
                iteration_average_plies=latest_iteration_average_plies,
                loss_curve=list(losses),
                policy_loss_curve=list(policy_losses),
                value_loss_curve=list(value_losses),
                checkpoint_paths=[*checkpoint_paths, checkpoint_path],
                training_device=resolved_training_device_name,
            )
            checkpoint_paths.append(save_checkpoint(model, checkpoint_path, snapshot))
            if progress_callback is not None:
                progress_callback(
                    {
                        "iteration": iteration + 1,
                        "games": (iteration + 1) * games_per_iteration,
                        "examples": examples,
                        "training_examples": training_examples,
                        "iteration_examples": latest_iteration_examples,
                        "iteration_training_examples": latest_iteration_training_examples,
                        "terminal_games": terminal_games,
                        "result_counts": dict(result_counts),
                        "iteration_result_counts": latest_iteration_result_counts,
                        "average_plies": total_plies / ((iteration + 1) * games_per_iteration),
                        "iteration_average_plies": latest_iteration_average_plies,
                        "updates": len(losses),
                        "latest_loss": losses[-1] if losses else None,
                        "latest_policy_loss": policy_losses[-1] if policy_losses else None,
                        "latest_value_loss": value_losses[-1] if value_losses else None,
                        "checkpoint_path": checkpoint_path,
                        "training_device": resolved_training_device_name,
                    }
                )

    return TrainMetrics(
        iterations=iterations,
        games=iterations * games_per_iteration,
        examples=examples,
        training_examples=training_examples,
        terminal_games=terminal_games,
        iteration_examples=latest_iteration_examples,
        iteration_training_examples=latest_iteration_training_examples,
        result_counts=dict(result_counts),
        iteration_result_counts=latest_iteration_result_counts,
        average_plies=0.0 if iterations * games_per_iteration == 0 else total_plies / (iterations * games_per_iteration),
        iteration_average_plies=latest_iteration_average_plies,
        loss_curve=losses,
        policy_loss_curve=policy_losses,
        value_loss_curve=value_losses,
        checkpoint_paths=checkpoint_paths,
        training_device=resolved_training_device_name,
    )
