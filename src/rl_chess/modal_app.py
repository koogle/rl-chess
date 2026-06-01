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
        "terminal_games": metrics.terminal_games,
        "iteration_examples": metrics.iteration_examples,
        "loss_curve": metrics.loss_curve,
        "policy_loss_curve": metrics.policy_loss_curve,
        "value_loss_curve": metrics.value_loss_curve,
        "checkpoint_paths": [str(path) for path in metrics.checkpoint_paths],
    }


def _random_checkpoint_validation_summary(best_validation: Any | None) -> dict[str, object]:
    if best_validation is None:
        return {
            "best_random_checkpoint": None,
            "best_random_score": None,
            "best_random_iteration": None,
        }
    return {
        "best_random_checkpoint": str(best_validation.checkpoint_path),
        "best_random_score": best_validation.score,
        "best_random_iteration": best_validation.iteration,
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
    validate_random_each_checkpoint: bool = False,
    stockfish_elo: int = 1320,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    checkpoint_validation_games: int | None = None,
    checkpoint_validation_max_plies: int | None = None,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
    starting_board_ascii: str | None = None,
    starting_turn: str = "white",
    self_play_workers: int = 8,
) -> dict[str, object]:
    from rl_chess.env import ascii_to_board
    from rl_chess.nn_model import PolicyValueNet
    from rl_chess.train import train
    from rl_chess.validation import (
        CheckpointRandomValidation,
        append_checkpoint_random_validation,
        best_checkpoint_random_validation,
        validate_model_against_random,
        validate_model_against_stockfish,
    )

    if validate_random_each_checkpoint and checkpoint_dir is None:
        raise ValueError("validate_random_each_checkpoint requires checkpoint_dir")
    per_checkpoint_games = checkpoint_validation_games or validation_games
    per_checkpoint_max_plies = checkpoint_validation_max_plies or validation_max_plies

    model = PolicyValueNet(hidden_channels=hidden_channels, residual_blocks=residual_blocks)
    best_random_validation: CheckpointRandomValidation | None = None
    random_validation_jsonl = None if checkpoint_dir is None else Path(checkpoint_dir) / "random-validation.jsonl"

    def report_progress(progress: dict[str, object]) -> None:
        nonlocal best_random_validation
        checkpoint_path = Path(str(progress["checkpoint_path"]))
        iteration = int(str(progress["iteration"]))
        print(
            "checkpoint_progress "
            + " ".join(
                [
                    f"iteration={progress['iteration']}",
                    f"games={progress['games']}",
                    f"examples={progress['examples']}",
                    f"iteration_examples={progress['iteration_examples']}",
                    f"updates={progress['updates']}",
                    f"latest_loss={progress['latest_loss']}",
                    f"checkpoint_path={progress['checkpoint_path']}",
                ]
            ),
            flush=True,
        )
        if validate_random_each_checkpoint:
            validation = validate_model_against_random(
                model=model,
                games=per_checkpoint_games,
                max_plies=per_checkpoint_max_plies,
                simulations=simulations,
                seed=None if seed is None else seed + 1_000_000 + iteration,
            )
            checkpoint_validation = CheckpointRandomValidation.from_result(
                iteration=iteration,
                checkpoint_path=checkpoint_path,
                result=validation,
            )
            if random_validation_jsonl is None:
                raise RuntimeError("random validation JSONL path was not initialized")
            append_checkpoint_random_validation(random_validation_jsonl, checkpoint_validation)
            best_random_validation = best_checkpoint_random_validation(best_random_validation, checkpoint_validation)
            print(
                "checkpoint_random_validation "
                + " ".join(
                    [
                        f"iteration={checkpoint_validation.iteration}",
                        f"wins={checkpoint_validation.wins}",
                        f"losses={checkpoint_validation.losses}",
                        f"draws={checkpoint_validation.draws}",
                        f"score={checkpoint_validation.score}",
                        f"passed={checkpoint_validation.passed}",
                        f"best_iteration={best_random_validation.iteration}",
                        f"best_score={best_random_validation.score}",
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
        progress_callback=report_progress if checkpoint_dir is not None else None,
    )
    summary = _jsonable_metrics(metrics)
    summary.update(
        {
            "hidden_channels": hidden_channels,
            "residual_blocks": residual_blocks,
            "checkpoint_dir": checkpoint_dir,
            "self_play_workers": self_play_workers,
            "validate_random_each_checkpoint": validate_random_each_checkpoint,
            "checkpoint_validation_games": per_checkpoint_games if validate_random_each_checkpoint else None,
            "checkpoint_validation_max_plies": per_checkpoint_max_plies if validate_random_each_checkpoint else None,
            "random_validation_jsonl": str(random_validation_jsonl) if validate_random_each_checkpoint else None,
        }
    )
    summary.update(_random_checkpoint_validation_summary(best_random_validation))
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
                "random_validation_score": validation.score,
                "random_validation_passed": validation.passed,
            }
        )
    _persist_summary(checkpoint_dir, summary)
    return summary


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
    validate_random_each_checkpoint: bool = False,
    stockfish_elo: int = 1320,
    validation_games: int = 2,
    validation_max_plies: int = 200,
    checkpoint_validation_games: int | None = None,
    checkpoint_validation_max_plies: int | None = None,
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
    starting_board_ascii: str | None = None,
    starting_turn: str = "white",
    self_play_workers: int = 8,
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
            validate_random_each_checkpoint=validate_random_each_checkpoint,
            stockfish_elo=stockfish_elo,
            validation_games=validation_games,
            validation_max_plies=validation_max_plies,
            checkpoint_validation_games=checkpoint_validation_games,
            checkpoint_validation_max_plies=checkpoint_validation_max_plies,
            stockfish_movetime=stockfish_movetime,
            seed=seed,
            starting_board_ascii=starting_board_ascii,
            starting_turn=starting_turn,
            self_play_workers=self_play_workers,
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
