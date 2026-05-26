import chess

from rl_chess.env import board_to_ascii
from rl_chess.nn_model import ChessPolicyValueNet, NeuralPolicyValueEvaluator
from rl_chess.puct_mcts import PolicyValueEvaluator, PUCTMCTS


class BiasedEvaluator(PolicyValueEvaluator):
    def evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        priors = {move.uci(): 1.0 for move in board.legal_moves}
        if "e2e4" in priors:
            priors["e2e4"] = 100.0
        if "e7e5" in priors:
            priors["e7e5"] = 100.0
        total = sum(priors.values())
        return {uci: weight / total for uci, weight in priors.items()}, 0.25


def test_neural_policy_value_evaluator_returns_legal_priors_and_white_value():
    model = ChessPolicyValueNet(hidden_channels=8)
    evaluator = NeuralPolicyValueEvaluator(model)
    board = chess.Board()

    priors, white_value = evaluator.evaluate(board)

    assert set(priors) == {move.uci() for move in board.legal_moves}
    assert abs(sum(priors.values()) - 1.0) < 1e-6
    assert -1.0 <= white_value <= 1.0


def test_puct_mcts_uses_policy_priors_to_shape_root_visit_policy():
    board = chess.Board()
    mcts = PUCTMCTS(evaluator=BiasedEvaluator(), iterations=16, seed=7)

    policy = mcts.search_policy(board)

    assert policy["e2e4"] == max(policy.values())
    assert policy["e2e4"] > policy["g1f3"]


def test_puct_mcts_policy_targets_can_feed_neural_training_examples():
    board = chess.Board()
    mcts = PUCTMCTS(evaluator=BiasedEvaluator(), iterations=4, seed=7)

    policy = mcts.search_policy(board)

    assert policy
    assert set(policy).issubset({move.uci() for move in board.legal_moves})
    assert board_to_ascii(board).startswith("  a b c d e f g h")
