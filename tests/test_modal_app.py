from rl_chess.modal_app import train_remote


def test_modal_training_function_uses_same_core_loop_locally():
    summary = train_remote.local(episodes=2, max_plies=4, seed=11)

    assert summary["episodes"] == 2
    assert summary["total_plies"] > 0
    assert summary["q_entries"] > 0
    assert len(summary["results"]) == 2


def test_modal_training_function_can_run_mcts_training_locally():
    summary = train_remote.local(
        policy="mcts-train",
        episodes=2,
        max_plies=2,
        mcts_iterations=2,
        rollout_depth=1,
        seed=11,
    )

    assert summary["policy"] == "mcts-train"
    assert summary["episodes"] == 2
    assert summary["examples_collected"] == 4
    assert len(summary["loss_curve"]) == 2
