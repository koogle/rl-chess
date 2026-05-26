from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from rl_chess.nn_model import ChessPolicyValueNet
from rl_chess.train import train_neural_mcts_self_play

OUT_DIR = Path("artifacts/puct_nn_training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = ChessPolicyValueNet(hidden_channels=16)
metrics = train_neural_mcts_self_play(
    model=model,
    episodes=10,
    max_plies=4,
    mcts_iterations=4,
    rollout_depth=2,
    learning_rate=0.001,
    neural_search=True,
    seed=2026,
)

xs = list(range(1, metrics.episodes + 1))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(xs, metrics.loss_curve, marker="o", label="total")
ax.plot(xs, metrics.policy_loss_curve, marker="o", label="policy CE")
ax.plot(xs, metrics.value_loss_curve, marker="o", label="value MSE")
ax.set_title("NN-guided PUCT self-play training loss")
ax.set_xlabel("episode")
ax.set_ylabel("loss")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plot_path = OUT_DIR / "loss_curve.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

report_path = OUT_DIR / "training_report.txt"
report_path.write_text(
    "\n".join(
        [
            "NN-guided PUCT training report",
            f"search={metrics.search_kind}",
            f"episodes={metrics.episodes}",
            f"total_plies={metrics.total_plies}",
            f"examples_collected={metrics.examples_collected}",
            "loss_curve=" + ",".join(f"{x:.6f}" for x in metrics.loss_curve),
            "policy_loss_curve=" + ",".join(f"{x:.6f}" for x in metrics.policy_loss_curve),
            "value_loss_curve=" + ",".join(f"{x:.6f}" for x in metrics.value_loss_curve),
        ]
    )
    + "\n"
)

print(report_path)
print(plot_path)
print("loss_curve=" + ",".join(f"{x:.6f}" for x in metrics.loss_curve))
print("policy_loss_curve=" + ",".join(f"{x:.6f}" for x in metrics.policy_loss_curve))
print("value_loss_curve=" + ",".join(f"{x:.6f}" for x in metrics.value_loss_curve))
