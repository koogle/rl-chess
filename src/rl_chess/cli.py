from __future__ import annotations

import argparse

from rl_chess.nn_model import PolicyValueNet
from rl_chess.train import train
from rl_chess.validation import validate_model_against_stockfish


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the minimal NN-guided PUCT chess loop.")
    parser.add_argument("--iterations", "--episodes", dest="iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies per game; 0 means no cap.")
    parser.add_argument("--mcts-iterations", "--simulations", dest="simulations", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validate-stockfish", action="store_true", help="After training, play the model against a weak Stockfish baseline.")
    parser.add_argument("--stockfish-elo", type=int, default=500, help="Stockfish UCI_Elo baseline for validation.")
    parser.add_argument("--stockfish-path", default="stockfish", help="Path to the Stockfish executable.")
    parser.add_argument("--stockfish-movetime", type=float, default=0.05, help="Seconds per Stockfish move.")
    parser.add_argument("--validation-games", type=int, default=2, help="Validation games, alternating colors.")
    parser.add_argument("--validation-max-plies", type=int, default=200, help="Validation ply cap; capped games count as draws.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = PolicyValueNet(hidden_channels=args.hidden_channels)
    metrics = train(
        model=model,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        simulations=args.simulations,
        max_plies=None if args.max_plies == 0 else args.max_plies,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
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
