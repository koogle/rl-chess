from __future__ import annotations

from pathlib import Path

import chess
import matplotlib.pyplot as plt

from rl_chess.agents import TabularPolicyDistiller
from rl_chess.env import ChessEnv, board_to_ascii
from rl_chess.mcts import MCTS, RandomRolloutEvaluator
from rl_chess.train import train_mcts_self_play

OUT_DIR = Path("artifacts/latest_model_game")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = 25
MAX_PLIES_TRAIN = 8
MCTS_ITERATIONS = 8
ROLLOUT_DEPTH = 4
LEARNING_RATE = 0.3
SEED = 2026
GAME_MAX_PLIES = 20

learner = TabularPolicyDistiller(learning_rate=LEARNING_RATE)
metrics = train_mcts_self_play(
    learner=learner,
    episodes=EPISODES,
    max_plies=MAX_PLIES_TRAIN,
    mcts_iterations=MCTS_ITERATIONS,
    rollout_depth=ROLLOUT_DEPTH,
    seed=SEED,
)

# Plot loss curve with matplotlib.
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
episodes = list(range(1, len(metrics.loss_curve) + 1))
ax.plot(episodes, metrics.loss_curve, marker="o", linewidth=2, markersize=4)
ax.set_title("rl-chess: first MCTS self-play distillation loss")
ax.set_xlabel("episode")
ax.set_ylabel("mean squared tabular distillation loss")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_path = OUT_DIR / "loss_curve.png"
fig.savefig(plot_path)
plt.close(fig)

# Play one game from the trained tabular policy. If an unseen position has no
# learned legal move, fall back to the MCTS teacher so the game stays legal.
env = ChessEnv()
obs = env.reset()
move_rows: list[tuple[int, str, str, float, str]] = []
board_snapshots = ["Initial\n" + obs.board_ascii]
fallback_count = 0

for ply in range(1, GAME_MAX_PLIES + 1):
    board = env.board.copy(stack=False)
    legal_moves = list(board.legal_moves)
    if not legal_moves or board.is_game_over(claim_draw=True):
        break

    scored = [
        (learner.policy_probability(obs.board_ascii, move.uci()), move)
        for move in legal_moves
    ]
    best_score, move = max(scored, key=lambda item: (item[0], item[1].uci()))
    source = "model"

    if best_score <= 0.0:
        fallback_count += 1
        teacher = MCTS(
            iterations=MCTS_ITERATIONS,
            evaluator=RandomRolloutEvaluator(max_depth=ROLLOUT_DEPTH),
            seed=SEED + 1000 + ply,
        )
        policy = teacher.search_policy(board)
        move = chess.Move.from_uci(max(policy, key=lambda uci: (policy[uci], uci)))
        best_score = policy[move.uci()]
        source = "mcts-fallback"

    san = board.san(move)
    obs, reward, done, info = env.step(move)
    move_rows.append((ply, san, move.uci(), best_score, source))
    board_snapshots.append(f"After ply {ply}: {san} ({move.uci()}, {source}, p={best_score:.3f})\n{obs.board_ascii}")
    if done:
        break

# Write a compact game report.
result = env.board.result(claim_draw=True) if env.board.is_game_over(claim_draw=True) else "*"
report_lines = [
    "rl-chess latest-model sample game",
    "",
    f"training: episodes={metrics.episodes} max_plies={MAX_PLIES_TRAIN} mcts_iterations={MCTS_ITERATIONS} rollout_depth={ROLLOUT_DEPTH} learning_rate={LEARNING_RATE} seed={SEED}",
    f"training examples={metrics.examples_collected} policy_entries={metrics.policy_entries}",
    f"loss_first={metrics.loss_curve[0]:.6f} loss_last={metrics.loss_curve[-1]:.6f} loss_min={min(metrics.loss_curve):.6f}",
    f"sample_game_plies={len(move_rows)} result={result} fallback_moves={fallback_count}",
    "",
    "moves:",
]
for ply, san, uci, score, source in move_rows:
    report_lines.append(f"{ply:02d}. {san:<8} {uci:<5} p={score:.3f} {source}")
report_lines.extend(["", "boards:", ""])
report_lines.extend(board_snapshots)
report_path = OUT_DIR / "sample_game.txt"
report_path.write_text("\n\n".join(report_lines), encoding="utf-8")

# Also make a visual final board + move list image.
fig, (ax_board, ax_moves) = plt.subplots(1, 2, figsize=(11, 6), dpi=160, gridspec_kw={"width_ratios": [1.1, 1]})
ax_board.axis("off")
ax_moves.axis("off")
ax_board.set_title(f"Final board after {len(move_rows)} plies")
ax_board.text(0.0, 1.0, board_to_ascii(env.board), family="DejaVu Sans Mono", fontsize=13, va="top")
move_text = "\n".join(f"{ply:02d}. {san} ({uci}) [{source}]" for ply, san, uci, score, source in move_rows)
ax_moves.set_title("Sample game moves")
ax_moves.text(0.0, 1.0, move_text or "no moves", family="DejaVu Sans Mono", fontsize=9, va="top")
fig.tight_layout()
game_png_path = OUT_DIR / "sample_game.png"
fig.savefig(game_png_path)
plt.close(fig)

print(plot_path.resolve())
print(game_png_path.resolve())
print(report_path.resolve())
print(f"loss_curve={','.join(f'{x:.6f}' for x in metrics.loss_curve)}")
print(f"moves={' '.join(row[1] for row in move_rows)}")
print(f"result={result} fallback_moves={fallback_count}")
