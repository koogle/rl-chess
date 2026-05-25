from __future__ import annotations

import argparse

from rl_chess.agents import TabularMoveValueAgent, TabularPolicyDistiller
from rl_chess.mcts import MCTS, RandomRolloutEvaluator
from rl_chess.self_play import play_episode
from rl_chess.train import train_mcts_self_play, train_self_play


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny chess RL self-play training loop.")
    parser.add_argument("--policy", choices=["tabular", "mcts", "mcts-train"], default="tabular", help="Policy loop to run.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of self-play games.")
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies per game.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Tabular update rate.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="ε-greedy exploration rate.")
    parser.add_argument("--mcts-iterations", type=int, default=100, help="MCTS simulations per move.")
    parser.add_argument("--rollout-depth", type=int, default=80, help="Random rollout depth for MCTS leaf evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.policy == "mcts-train":
        learner = TabularPolicyDistiller(learning_rate=args.learning_rate)
        metrics = train_mcts_self_play(
            learner=learner,
            episodes=args.episodes,
            max_plies=args.max_plies,
            mcts_iterations=args.mcts_iterations,
            rollout_depth=args.rollout_depth,
            seed=args.seed,
        )
        print(
            " ".join(
                [
                    "policy=mcts-train",
                    f"episodes={metrics.episodes}",
                    f"total_plies={metrics.total_plies}",
                    f"examples_collected={metrics.examples_collected}",
                    f"policy_entries={metrics.policy_entries}",
                    "loss_curve=" + ",".join(f"{loss:.6f}" for loss in metrics.loss_curve),
                ]
            )
        )
        return 0

    if args.policy == "mcts":
        policy = MCTS(
            iterations=args.mcts_iterations,
            evaluator=RandomRolloutEvaluator(max_depth=args.rollout_depth),
            seed=args.seed,
        )
        total_plies = 0
        results: list[str] = []
        for episode_idx in range(args.episodes):
            from rl_chess.env import ChessEnv

            episode = play_episode(
                env=ChessEnv(),
                white_policy=policy,
                black_policy=policy,
                max_plies=args.max_plies,
                seed=None if args.seed is None else args.seed + episode_idx,
            )
            total_plies += len(episode.transitions)
            results.append(episode.result)
        print(
            " ".join(
                [
                    "policy=mcts",
                    f"episodes={args.episodes}",
                    f"total_plies={total_plies}",
                    f"mcts_iterations={args.mcts_iterations}",
                    f"results={','.join(results)}",
                ]
            )
        )
        return 0

    agent = TabularMoveValueAgent(
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    metrics = train_self_play(
        agent=agent,
        episodes=args.episodes,
        max_plies=args.max_plies,
        seed=args.seed,
    )
    print(
        " ".join(
            [
                "policy=tabular",
                f"episodes={metrics.episodes}",
                f"total_plies={metrics.total_plies}",
                f"replay_size={metrics.replay_size}",
                f"q_entries={len(agent.q)}",
                f"results={','.join(metrics.results)}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
