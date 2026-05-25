import chess

from rl_chess.mcts import MCTS, MCTSNode, RandomRolloutEvaluator, uct_score


def test_mcts_node_expands_one_untried_legal_move():
    board = chess.Board()
    node = MCTSNode.root(board)

    child = node.expand()

    assert child.parent is node
    assert child.move in board.legal_moves
    assert child.board_fen != board.fen()
    assert len(node.children) == 1
    assert child.move not in node.untried_moves


def test_uct_score_prefers_unvisited_children():
    parent = MCTSNode.root(chess.Board())
    child = parent.expand()
    parent.visits = 10

    assert uct_score(parent, child, exploration=1.4) == float("inf")


def test_backpropagates_result_from_each_node_player_perspective():
    board = chess.Board()
    root = MCTSNode.root(board)
    child = root.expand()

    child.backpropagate(white_reward=1.0)

    assert root.visits == 1
    assert child.visits == 1
    assert root.value_sum == 1.0
    expected_child_value = 1.0 if child.player_to_move == chess.WHITE else -1.0
    assert child.value_sum == expected_child_value


def test_search_finds_mate_in_one_without_engine():
    # White to move; several queen moves are immediate checkmate.
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    mcts = MCTS(iterations=25, exploration=1.4, evaluator=RandomRolloutEvaluator(max_depth=4), seed=7)

    move = mcts.search(board)

    assert board.copy(stack=False).san(move).endswith("#")


def test_mcts_policy_selects_legal_move_and_records_root_stats():
    board = chess.Board()
    mcts = MCTS(iterations=8, seed=123)

    move = mcts.select_move(board)

    assert move in board.legal_moves
    assert mcts.last_root is not None
    assert mcts.last_root.visits == 8
    assert len(mcts.last_root.children) > 0
