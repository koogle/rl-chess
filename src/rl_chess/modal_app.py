from __future__ import annotations

import json
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
        "training_examples": metrics.training_examples,
        "terminal_games": metrics.terminal_games,
        "iteration_examples": metrics.iteration_examples,
        "iteration_training_examples": metrics.iteration_training_examples,
        "result_counts": metrics.result_counts,
        "iteration_result_counts": metrics.iteration_result_counts,
        "average_plies": metrics.average_plies,
        "iteration_average_plies": metrics.iteration_average_plies,
        "loss_curve": metrics.loss_curve,
        "policy_loss_curve": metrics.policy_loss_curve,
        "value_loss_curve": metrics.value_loss_curve,
        "checkpoint_paths": [str(path) for path in metrics.checkpoint_paths],
    }


@app.function(image=image, timeout=24 * 60 * 60, cpu=8, volumes={str(CHECKPOINT_ROOT): checkpoint_volume})
def train_remote(
    iterations: int = 10,
    games_per_iteration: int = 1,
    max_plies: int | None = None,
    simulations: int = 64,
    train_steps: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    hidden_channels: int = 64,
    residual_blocks: int = 4,
    checkpoint_dir: str | None = None,
    validate_stockfish: bool = False,
    validate_random: bool = False,
    stockfish_elo: int = 1320,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
    starting_board_ascii: str | None = None,
    starting_turn: str = "white",
    self_play_workers: int = 8,
    augment_color_flip: bool = True,
    validate_each_checkpoint: bool = True,
) -> dict[str, object]:
    from rl_chess.env import ascii_to_board
    from rl_chess.nn_model import PolicyValueNet
    from rl_chess.train import load_checkpoint_model, train
    from rl_chess.validation import validate_model_against_random, validate_model_against_stockfish

    model = PolicyValueNet(hidden_channels=hidden_channels, residual_blocks=residual_blocks)

    def report_progress(progress: dict[str, object]) -> None:
        print(
            "checkpoint_progress "
            + " ".join(
                [
                    f"iteration={progress['iteration']}",
                    f"games={progress['games']}",
                    f"examples={progress['examples']}",
                    f"training_examples={progress['training_examples']}",
                    f"iteration_examples={progress['iteration_examples']}",
                    f"iteration_training_examples={progress['iteration_training_examples']}",
                    f"result_counts={progress['result_counts']}",
                    f"iteration_average_plies={progress['iteration_average_plies']}",
                    f"updates={progress['updates']}",
                    f"latest_loss={progress['latest_loss']}",
                    f"checkpoint_path={progress['checkpoint_path']}",
                ]
            ),
            flush=True,
        )

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
        checkpoint_dir=checkpoint_dir,
        starting_board=None if starting_board_ascii is None else ascii_to_board(starting_board_ascii, starting_turn == "white"),
        self_play_workers=self_play_workers,
        augment_color_flip=augment_color_flip,
        progress_callback=report_progress if checkpoint_dir is not None else None,
    )
    summary = _jsonable_metrics(metrics)
    summary.update(
        {
            "hidden_channels": hidden_channels,
            "residual_blocks": residual_blocks,
            "checkpoint_dir": checkpoint_dir,
            "self_play_workers": self_play_workers,
            "augment_color_flip": augment_color_flip,
            "validate_each_checkpoint": validate_each_checkpoint,
        }
    )
    if validate_each_checkpoint and metrics.checkpoint_paths and (validate_stockfish or validate_random):
        summary["checkpoint_validations"] = [
            _validate_checkpoint(
                path=checkpoint_path,
                validate_stockfish=validate_stockfish,
                validate_random=validate_random,
                stockfish_elo=stockfish_elo,
                validation_games=validation_games,
                validation_max_plies=validation_max_plies,
                simulations=simulations,
                stockfish_movetime=stockfish_movetime,
                seed=None if seed is None else seed + checkpoint_index,
                load_checkpoint_model=load_checkpoint_model,
                validate_model_against_random=validate_model_against_random,
                validate_model_against_stockfish=validate_model_against_stockfish,
            )
            for checkpoint_index, checkpoint_path in enumerate(metrics.checkpoint_paths, start=1)
        ]
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
                "validation_capped_draws": validation.capped_draws,
                "validation_score": validation.score,
                "validation_passed": validation.passed,
            }
        )
    if validate_random:
        validation = validate_model_against_random(
            model=model,
            games=validation_games,
            max_plies=validation_max_plies,
            simulations=simulations,
            seed=seed,
        )
        summary.update(
            {
                "random_validation_games": validation.games,
                "random_validation_wins": validation.wins,
                "random_validation_losses": validation.losses,
                "random_validation_draws": validation.draws,
                "random_validation_capped_draws": validation.capped_draws,
                "random_validation_score": validation.score,
                "random_validation_passed": validation.passed,
            }
        )
    _persist_summary(checkpoint_dir, summary)
    return summary


def _validation_summary(prefix: str, validation: Any) -> dict[str, object]:
    return {
        f"{prefix}_games": validation.games,
        f"{prefix}_wins": validation.wins,
        f"{prefix}_losses": validation.losses,
        f"{prefix}_draws": validation.draws,
        f"{prefix}_capped_draws": validation.capped_draws,
        f"{prefix}_score": validation.score,
        f"{prefix}_passed": validation.passed,
    }


def _validate_checkpoint(
    path: Path,
    validate_stockfish: bool,
    validate_random: bool,
    stockfish_elo: int,
    validation_games: int,
    validation_max_plies: int,
    simulations: int,
    stockfish_movetime: float,
    seed: int | None,
    load_checkpoint_model: Any,
    validate_model_against_random: Any,
    validate_model_against_stockfish: Any,
) -> dict[str, object]:
    model = load_checkpoint_model(path)
    result: dict[str, object] = {"checkpoint_path": str(path)}
    if validate_random:
        random_validation = validate_model_against_random(
            model=model,
            games=validation_games,
            max_plies=validation_max_plies,
            simulations=simulations,
            seed=seed,
        )
        result.update(_validation_summary("random", random_validation))
    if validate_stockfish:
        stockfish_validation = validate_model_against_stockfish(
            model=model,
            elo=stockfish_elo,
            games=validation_games,
            max_plies=validation_max_plies,
            simulations=simulations,
            stockfish_movetime=stockfish_movetime,
            seed=seed,
        )
        result["stockfish_elo"] = stockfish_elo
        result.update(_validation_summary("stockfish", stockfish_validation))
    return result


def _persist_summary(checkpoint_dir: str | None, summary: dict[str, object]) -> None:
    if checkpoint_dir is None:
        return
    summary_path = Path(checkpoint_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    checkpoint_volume.commit()


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _spawn_train(**kwargs: object) -> None:
    function_call = train_remote.spawn(**kwargs)
    _print_json(
        {
            "status": "spawned",
            "function_call_id": function_call.object_id,
            "dashboard_url": function_call.get_dashboard_url(),
            "checkpoint_dir": kwargs.get("checkpoint_dir"),
        }
    )


def _wait_for_train(function_call_id: str) -> None:
    function_call = modal.FunctionCall.from_id(function_call_id)
    _print_json(function_call.get())


@app.local_entrypoint()
def main(
    iterations: int = 10,
    games_per_iteration: int = 1,
    max_plies: int | None = None,
    simulations: int = 64,
    train_steps: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    hidden_channels: int = 64,
    residual_blocks: int = 4,
    checkpoint_dir: str | None = None,
    validate_stockfish: bool = False,
    validate_random: bool = False,
    stockfish_elo: int = 1320,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
    starting_board_ascii: str | None = None,
    starting_turn: str = "white",
    self_play_workers: int = 8,
    augment_color_flip: bool = True,
    validate_each_checkpoint: bool = True,
    wait: bool = False,
) -> None:
    kwargs: dict[str, object] = dict(
            iterations=iterations,
            games_per_iteration=games_per_iteration,
            max_plies=max_plies,
            simulations=simulations,
            train_steps=train_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            temperature=temperature,
            hidden_channels=hidden_channels,
            residual_blocks=residual_blocks,
            checkpoint_dir=checkpoint_dir,
            validate_stockfish=validate_stockfish,
            validate_random=validate_random,
            stockfish_elo=stockfish_elo,
            validation_games=validation_games,
            validation_max_plies=validation_max_plies,
            stockfish_movetime=stockfish_movetime,
            seed=seed,
            starting_board_ascii=starting_board_ascii,
            starting_turn=starting_turn,
            self_play_workers=self_play_workers,
            augment_color_flip=augment_color_flip,
            validate_each_checkpoint=validate_each_checkpoint,
    )
    if wait:
        function_call = train_remote.spawn(**kwargs)
        _print_json(
            {
                "status": "spawned",
                "function_call_id": function_call.object_id,
                "dashboard_url": function_call.get_dashboard_url(),
                "checkpoint_dir": checkpoint_dir,
            }
        )
        _print_json(function_call.get())
    else:
        _spawn_train(**kwargs)


@app.local_entrypoint()
def result(function_call_id: str) -> None:
    """Fetch and print the JSON result for a spawned Modal training call."""
    _wait_for_train(function_call_id)
