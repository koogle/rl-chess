from __future__ import annotations

import modal

app = modal.App("rl-chess-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .pip_install("python-chess>=1.999", "torch>=2.12.0")
    .add_local_python_source("rl_chess")
)


@app.function(image=image, timeout=60 * 60)
def train_remote(
    iterations: int = 10,
    games_per_iteration: int = 1,
    max_plies: int | None = 200,
    simulations: int = 64,
    train_steps: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    hidden_channels: int = 32,
    validate_stockfish: bool = False,
    stockfish_elo: int = 500,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
) -> dict[str, object]:
    from rl_chess.nn_model import PolicyValueNet
    from rl_chess.train import train
    from rl_chess.validation import validate_model_against_stockfish

    model = PolicyValueNet(hidden_channels=hidden_channels)
    metrics = train(
        model=model,
        iterations=iterations,
        games_per_iteration=games_per_iteration,
        simulations=simulations,
        max_plies=max_plies,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        seed=seed,
    )
    summary: dict[str, object] = {
        "loop": "nn-puct",
        "iterations": metrics.iterations,
        "games": metrics.games,
        "examples": metrics.examples,
        "terminal_games": metrics.terminal_games,
        "truncated_games": metrics.truncated_games,
        "replay_size": metrics.replay_size,
        "loss_curve": metrics.loss_curve,
        "policy_loss_curve": metrics.policy_loss_curve,
        "value_loss_curve": metrics.value_loss_curve,
    }
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


@app.local_entrypoint()
def main(
    iterations: int = 10,
    games_per_iteration: int = 1,
    max_plies: int = 200,
    simulations: int = 64,
    train_steps: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    hidden_channels: int = 32,
    validate_stockfish: bool = False,
    stockfish_elo: int = 500,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
) -> None:
    print(
        train_remote.remote(
            iterations=iterations,
            games_per_iteration=games_per_iteration,
            max_plies=None if max_plies == 0 else max_plies,
            simulations=simulations,
            train_steps=train_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            temperature=temperature,
            hidden_channels=hidden_channels,
            validate_stockfish=validate_stockfish,
            stockfish_elo=stockfish_elo,
            validation_games=validation_games,
            validation_max_plies=validation_max_plies,
            stockfish_movetime=stockfish_movetime,
            seed=seed,
        )
    )
