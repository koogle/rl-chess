from __future__ import annotations

import argparse
from pathlib import Path

from rl_chess.nn_model import PolicyValueNet
from rl_chess.run_presets import FIRST_MEANINGFUL_RUN
from rl_chess.train import load_checkpoint_model, train
from rl_chess.validation import STOCKFISH_ELO_FLOOR, validate_model_against_stockfish


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the minimal NN-guided PUCT chess loop.")
    parser.add_argument(
        "--first-meaningful-run",
        action="store_true",
        help="Use the first non-smoke training preset: multiple games, replay updates, checkpoints, and Stockfish eval.",
    )
    parser.add_argument("--iterations", "--episodes", dest="iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies per game; 0 means no cap.")
    parser.add_argument("--mcts-iterations", "--simulations", dest="simulations", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory for iteration checkpoints.")
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Start from a saved model checkpoint.")
    parser.add_argument("--validate-stockfish", action="store_true", help="After training, play the model against a weak Stockfish baseline.")
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=STOCKFISH_ELO_FLOOR,
        help="Stockfish UCI_Elo baseline for validation; defaults to the supported floor.",
    )
    parser.add_argument("--stockfish-path", default="stockfish", help="Path to the Stockfish executable.")
    parser.add_argument("--stockfish-movetime", type=float, default=0.05, help="Seconds per Stockfish move.")
    parser.add_argument("--validation-games", type=int, default=2, help="Validation games, alternating colors.")
    parser.add_argument("--validation-max-plies", type=int, default=200, help="Validation ply cap; capped games count as draws.")
    return parser


def apply_first_run_preset(args: argparse.Namespace) -> None:
    if not args.first_meaningful_run:
        return
    preset = FIRST_MEANINGFUL_RUN
    args.iterations = preset.iterations
    args.games_per_iteration = preset.games_per_iteration
    args.max_plies = preset.max_plies
    args.simulations = preset.simulations
    args.train_steps = preset.train_steps
    args.batch_size = preset.batch_size
    args.replay_capacity = preset.replay_capacity
    args.learning_rate = preset.learning_rate
    args.temperature = preset.temperature
    args.hidden_channels = preset.hidden_channels
    args.validation_games = preset.validation_games
    args.validation_max_plies = preset.validation_max_plies
    args.validate_stockfish = True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_first_run_preset(args)

    model = load_checkpoint_model(args.load_checkpoint) if args.load_checkpoint else PolicyValueNet(hidden_channels=args.hidden_channels)
    metrics = train(
        model=model,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        simulations=args.simulations,
        max_plies=None if args.max_plies == 0 else args.max_plies,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    summary = [
        "loop=nn-puct",
        f"iterations={metrics.iterations}",
        f"games={metrics.games}",
        f"examples={metrics.examples}",
        f"terminal_games={metrics.terminal_games}",
        f"truncated_games={metrics.truncated_games}",
        f"replay_size={metrics.replay_size}",
        "loss_curve=" + ",".join(f"{loss:.6f}" for loss in metrics.loss_curve),
        "policy_loss_curve=" + ",".join(f"{loss:.6f}" for loss in metrics.policy_loss_curve),
        "value_loss_curve=" + ",".join(f"{loss:.6f}" for loss in metrics.value_loss_curve),
    ]
    if metrics.checkpoint_paths:
        summary.append("checkpoint_paths=" + ",".join(str(path) for path in metrics.checkpoint_paths))
    if args.validate_stockfish:
        validation = validate_model_against_stockfish(
            model=model,
            elo=args.stockfish_elo,
            games=args.validation_games,
            max_plies=args.validation_max_plies,
            simulations=args.simulations,
            stockfish_path=args.stockfish_path,
            stockfish_movetime=args.stockfish_movetime,
            seed=args.seed,
        )
        summary.extend(
            [
                f"stockfish_elo={args.stockfish_elo}",
                f"validation_games={validation.games}",
                f"validation_wins={validation.wins}",
                f"validation_losses={validation.losses}",
                f"validation_draws={validation.draws}",
                f"validation_score={validation.score:.3f}",
                f"validation_passed={validation.passed}",
            ]
        )
    print(" ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
