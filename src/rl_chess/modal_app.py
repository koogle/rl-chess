from __future__ import annotations

import modal

app = modal.App("rl-chess-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("python-chess>=1.999", "numpy>=2.4.6")
    .add_local_python_source("rl_chess")
)


@app.function(image=image, timeout=60 * 60)
def train_remote(
    episodes: int = 100,
    max_plies: int = 200,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict[str, object]:
    """Run the exact same hand-written training loop on Modal."""

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
        "episodes": metrics.episodes,
        "total_plies": metrics.total_plies,
        "replay_size": metrics.replay_size,
        "results": metrics.results,
        "q_entries": len(agent.q),
    }


@app.local_entrypoint()
def main(
    episodes: int = 100,
    max_plies: int = 200,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> None:
    """CLI entrypoint: modal run src/rl_chess/modal_app.py --episodes 1000"""

    summary = train_remote.remote(
        episodes=episodes,
        max_plies=max_plies,
        learning_rate=learning_rate,
        epsilon=epsilon,
        seed=seed,
    )
    print(summary)
