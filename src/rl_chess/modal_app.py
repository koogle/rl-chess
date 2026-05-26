from __future__ import annotations

import modal

app = modal.App("rl-chess-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("python-chess>=1.999", "numpy>=2.4.6", "torch>=2.12.0")
    .add_local_python_source("rl_chess")
)


@app.function(image=image, timeout=60 * 60)
def train_remote(
    policy: str = "tabular",
    episodes: int = 100,
    max_plies: int = 200,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    mcts_iterations: int = 50,
    rollout_depth: int = 20,
    hidden_channels: int = 32,
    seed: int | None = None,
) -> dict[str, object]:
    """Run the exact same hand-written training loops on Modal."""

    if policy == "mcts-train":
        from rl_chess.agents import TabularPolicyDistiller
        from rl_chess.train import train_mcts_self_play

        learner = TabularPolicyDistiller(learning_rate=learning_rate)
        metrics = train_mcts_self_play(
            learner=learner,
            episodes=episodes,
            max_plies=max_plies,
            mcts_iterations=mcts_iterations,
            rollout_depth=rollout_depth,
            seed=seed,
        )
        return {
            "policy": "mcts-train",
            "episodes": metrics.episodes,
            "total_plies": metrics.total_plies,
            "examples_collected": metrics.examples_collected,
            "policy_entries": metrics.policy_entries,
            "loss_curve": metrics.loss_curve,
        }

    if policy == "nn-train":
        from rl_chess.nn_model import ChessPolicyValueNet
        from rl_chess.train import train_neural_mcts_self_play

        model = ChessPolicyValueNet(hidden_channels=hidden_channels)
        metrics = train_neural_mcts_self_play(
            model=model,
            episodes=episodes,
            max_plies=max_plies,
            mcts_iterations=mcts_iterations,
            rollout_depth=rollout_depth,
            learning_rate=learning_rate,
            seed=seed,
        )
        return {
            "policy": "nn-train",
            "episodes": metrics.episodes,
            "total_plies": metrics.total_plies,
            "examples_collected": metrics.examples_collected,
            "loss_curve": metrics.loss_curve,
            "policy_loss_curve": metrics.policy_loss_curve,
            "value_loss_curve": metrics.value_loss_curve,
        }

    if policy != "tabular":
        raise ValueError(f"unsupported remote training policy {policy!r}")

    from rl_chess.agents import TabularMoveValueAgent
    from rl_chess.train import train_self_play

    agent = TabularMoveValueAgent(
        learning_rate=learning_rate,
        epsilon=epsilon,
        seed=seed,
    )
    metrics = train_self_play(
        agent=agent,
        episodes=episodes,
        max_plies=max_plies,
        seed=seed,
    )
    return {
        "policy": "tabular",
        "episodes": metrics.episodes,
        "total_plies": metrics.total_plies,
        "replay_size": metrics.replay_size,
        "results": metrics.results,
        "q_entries": len(agent.q),
    }


@app.local_entrypoint()
def main(
    policy: str = "tabular",
    episodes: int = 100,
    max_plies: int = 200,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    mcts_iterations: int = 50,
    rollout_depth: int = 20,
    hidden_channels: int = 32,
    seed: int | None = None,
) -> None:
    """CLI entrypoint: modal run src/rl_chess/modal_app.py --episodes 1000"""

    summary = train_remote.remote(
        policy=policy,
        episodes=episodes,
        max_plies=max_plies,
        learning_rate=learning_rate,
        epsilon=epsilon,
        mcts_iterations=mcts_iterations,
        rollout_depth=rollout_depth,
        hidden_channels=hidden_channels,
        seed=seed,
    )
    print(summary)
