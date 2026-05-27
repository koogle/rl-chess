from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

app = modal.App("rl-chess-training")
checkpoint_volume = modal.Volume.from_name("rl-chess-checkpoints", create_if_missing=True)
CHECKPOINT_ROOT = Path("/checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .pip_install("python-chess>=1.999", "torch>=2.12.0", "numpy>=2.4.6")
    .add_local_python_source("rl_chess")
)


def _jsonable_metrics(metrics: Any) -> dict[str, object]:
    return {
        "loop": "nn-puct",
        "iterations": metrics.iterations,
        "games": metrics.games,
        "examples": metrics.examples,
        "terminal_games": metrics.terminal_games,
        "replay_size": metrics.replay_size,
        "loss_curve": metrics.loss_curve,
        "policy_loss_curve": metrics.policy_loss_curve,
        "value_loss_curve": metrics.value_loss_curve,
        "checkpoint_paths": [str(path) for path in metrics.checkpoint_paths],
    }


@app.function(image=image, timeout=60 * 60, volumes={str(CHECKPOINT_ROOT): checkpoint_volume})
def train_remote(
    iterations: int = 10,
    games_per_iteration: int = 1,
    max_plies: int | None = None,
    simulations: int = 64,
    train_steps: int = 1,
    batch_size: int = 64,
    replay_capacity: int = 10_000,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    hidden_channels: int = 64,
    residual_blocks: int = 4,
    checkpoint_dir: str | None = None,
    first_meaningful_run: bool = False,
    validate_stockfish: bool = False,
    stockfish_elo: int = 1320,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
    starting_board_ascii: str | None = None,
    starting_turn: str = "white",
) -> dict[str, object]:
    from rl_chess.env import ascii_to_board
    from rl_chess.nn_model import PolicyValueNet
    from rl_chess.run_presets import FIRST_MEANINGFUL_RUN
    from rl_chess.train import train
    from rl_chess.validation import validate_model_against_stockfish

    if first_meaningful_run:
        preset = FIRST_MEANINGFUL_RUN
        iterations = preset.iterations
        games_per_iteration = preset.games_per_iteration
        max_plies = preset.max_plies
        simulations = preset.simulations
        train_steps = preset.train_steps
        batch_size = preset.batch_size
        replay_capacity = preset.replay_capacity
        learning_rate = preset.learning_rate
        temperature = preset.temperature
        hidden_channels = preset.hidden_channels
        residual_blocks = preset.residual_blocks
        validation_games = preset.validation_games
        validation_max_plies = preset.validation_max_plies
        validate_stockfish = True
        checkpoint_dir = checkpoint_dir or str(CHECKPOINT_ROOT / "first-meaningful-run")

    model = PolicyValueNet(hidden_channels=hidden_channels, residual_blocks=residual_blocks)
    metrics = train(
        model=model,
        iterations=iterations,
        games_per_iteration=games_per_iteration,
        simulations=simulations,
        max_plies=max_plies,
        train_steps=train_steps,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        learning_rate=learning_rate,
        temperature=temperature,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        starting_board=None if starting_board_ascii is None else ascii_to_board(starting_board_ascii, starting_turn == "white"),
    )
    summary = _jsonable_metrics(metrics)
    summary.update(
        {
            "hidden_channels": hidden_channels,
            "residual_blocks": residual_blocks,
            "checkpoint_dir": checkpoint_dir,
        }
    )
    if checkpoint_dir is not None:
        checkpoint_volume.commit()
    if validate_stockfish:
        validation = validate_model_against_stockfish(
            model=model,
            elo=stockfish_elo,
            games=validation_games,
            max_plies=validation_max_plies,
            simulations=simulations,
            stockfish_movetime=stockfish_movetime,
            seed=seed,
        )
        summary.update(
            {
                "stockfish_elo": stockfish_elo,
                "validation_games": validation.games,
                "validation_wins": validation.wins,
                "validation_losses": validation.losses,
                "validation_draws": validation.draws,
                "validation_score": validation.score,
                "validation_passed": validation.passed,
            }
        )
    return summary


@app.function(image=image, timeout=60 * 60)
def validate_endgames_remote(
    depth: int = 5,
    hidden_channels: int = 64,
    residual_blocks: int = 4,
    steps: int = 400,
    learning_rate: float = 0.001,
    seed: int = 1,
    max_plies: int = 5,
    batch_size: int = 64,
) -> dict[str, object]:
    from rl_chess.endgame_validation import run_endgame_value_validation

    return run_endgame_value_validation(
        depth=depth,
        hidden_channels=hidden_channels,
        residual_blocks=residual_blocks,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        max_plies=max_plies,
        batch_size=batch_size,
    )


@app.local_entrypoint()
def main(
    iterations: int = 10,
    games_per_iteration: int = 1,
    max_plies: int | None = None,
    simulations: int = 64,
    train_steps: int = 1,
    batch_size: int = 64,
    replay_capacity: int = 10_000,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    hidden_channels: int = 64,
    residual_blocks: int = 4,
    checkpoint_dir: str | None = None,
    first_meaningful_run: bool = False,
    validate_stockfish: bool = False,
    stockfish_elo: int = 1320,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
    starting_board_ascii: str | None = None,
    starting_turn: str = "white",
    validate_endgames: bool = False,
    endgame_depth: int = 5,
    endgame_steps: int = 800,
    endgame_max_plies: int = 5,
    endgame_batch_size: int = 64,
) -> None:
    if validate_endgames:
        print(
            validate_endgames_remote.remote(
                depth=endgame_depth,
                hidden_channels=hidden_channels,
                residual_blocks=residual_blocks,
                steps=endgame_steps,
                learning_rate=learning_rate,
                seed=1 if seed is None else seed,
                max_plies=endgame_max_plies,
                batch_size=endgame_batch_size,
            )
        )
        return

    print(
        train_remote.remote(
            iterations=iterations,
            games_per_iteration=games_per_iteration,
            max_plies=max_plies,
            simulations=simulations,
            train_steps=train_steps,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            learning_rate=learning_rate,
            temperature=temperature,
            hidden_channels=hidden_channels,
            residual_blocks=residual_blocks,
            checkpoint_dir=checkpoint_dir,
            first_meaningful_run=first_meaningful_run,
            validate_stockfish=validate_stockfish,
            stockfish_elo=stockfish_elo,
            validation_games=validation_games,
            validation_max_plies=validation_max_plies,
            stockfish_movetime=stockfish_movetime,
            seed=seed,
            starting_board_ascii=starting_board_ascii,
            starting_turn=starting_turn,
        )
    )
