import chess

from rl_chess.agents import TabularPolicyDistiller
from rl_chess.env import ChessEnv, board_to_ascii
from rl_chess.mcts import MCTS, RandomRolloutEvaluator
from rl_chess.replay import SearchTrainingExample
from rl_chess.search_self_play import collect_search_episode
from rl_chess.train import train_mcts_self_play


def test_mcts_returns_normalized_visit_policy_over_legal_moves():
    board = chess.Board()
    mcts = MCTS(iterations=8, seed=123)

    policy = mcts.search_policy(board)

    assert abs(sum(policy.values()) - 1.0) < 1e-6
    assert set(policy).issubset({move.uci() for move in board.legal_moves})
    assert all(prob > 0.0 for prob in policy.values())


def test_collect_search_episode_stores_ascii_policy_and_value_targets():
    env = ChessEnv()
    mcts = MCTS(iterations=4, evaluator=RandomRolloutEvaluator(max_depth=2), seed=7)

    examples = collect_search_episode(env=env, mcts=mcts, max_plies=3, seed=7)

    assert len(examples) == 3
    first = examples[0]
    assert isinstance(first, SearchTrainingExample)
    assert first.state_ascii == board_to_ascii(chess.Board())
    assert "e2e4" in first.legal_moves
    assert abs(sum(first.policy_target.values()) - 1.0) < 1e-6
    assert first.value_target in {-1.0, 0.0, 1.0}


def test_tabular_policy_distiller_learns_toward_search_targets():
    board = chess.Board()
    state_ascii = board_to_ascii(board)
    learner = TabularPolicyDistiller(learning_rate=0.5)
    example = SearchTrainingExample(
        state_ascii=state_ascii,
        legal_moves=("e2e4", "d2d4"),
        policy_target={"e2e4": 0.8, "d2d4": 0.2},
        value_target=1.0,
    )

    loss = learner.learn([example])

    assert loss > 0.0
    assert learner.policy_probability(state_ascii, "e2e4") == 0.4
    assert learner.policy_probability(state_ascii, "d2d4") == 0.1
    assert learner.value(state_ascii) == 0.5


def test_train_mcts_self_play_returns_loss_curve():
    learner = TabularPolicyDistiller(learning_rate=0.5)

    metrics = train_mcts_self_play(
        learner=learner,
        episodes=3,
        max_plies=3,
        mcts_iterations=4,
        rollout_depth=2,
        seed=11,
    )

    assert metrics.episodes == 3
    assert metrics.examples_collected == 9
    assert len(metrics.loss_curve) == 3
    assert all(loss >= 0.0 for loss in metrics.loss_curve)
    assert metrics.policy_entries > 0
