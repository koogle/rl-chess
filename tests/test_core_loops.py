import chess

from rl_chess.agents import TabularMoveValueAgent
from rl_chess.cli import main
from rl_chess.env import ChessEnv, board_to_ascii
from rl_chess.replay import ReplayBuffer, Transition
from rl_chess.self_play import play_episode
from rl_chess.state import encode_board_planes, legal_move_uci, state_from_board
from rl_chess.train import train_self_play


class FirstLegalPolicy:
    def select_move(self, board, rng=None):
        return next(iter(board.legal_moves))


def test_env_step_applies_legal_move_and_rejects_illegal_move():
    env = ChessEnv()
    obs = env.reset()

    assert not hasattr(obs, "fen")
    assert obs.legal_moves

    illegal = chess.Move.from_uci("e2e5")
    try:
        env.step(illegal)
    except ValueError as exc:
        assert "Illegal move" in str(exc)
    else:
        raise AssertionError("illegal moves must raise")

    obs, reward, done, info = env.step(chess.Move.from_uci("e2e4"))

    assert env.board.piece_at(chess.E4).piece_type == chess.PAWN
    assert reward == 0.0
    assert done is False
    assert info["result"] is None
    assert obs.turn == chess.BLACK


def test_env_observation_includes_ascii_board_diagram():
    env = ChessEnv()
    obs = env.reset()

    assert obs.board_ascii == board_to_ascii(env.board)
    assert obs.board_ascii.splitlines() == [
        "  a b c d e f g h",
        "8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8",
        "7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟ 7",
        "6 . . . . . . . . 6",
        "5 . . . . . . . . 5",
        "4 . . . . . . . . 4",
        "3 . . . . . . . . 3",
        "2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ 2",
        "1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1",
        "  a b c d e f g h",
    ]

    obs, _, _, _ = env.step("e2e4")

    assert "4 . . . . ♙ . . . 4" in obs.board_ascii
    assert "2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2" in obs.board_ascii


def test_state_helpers_use_python_chess_for_ascii_legal_moves_and_planes():
    board = chess.Board()
    state = state_from_board(board)

    assert state.board_ascii == board_to_ascii(board)
    assert "e2e4" in state.legal_moves
    assert legal_move_uci(board) == state.legal_moves

    planes = encode_board_planes(board)
    assert planes.shape == (12, 8, 8)
    assert planes.sum() == 32
    assert planes[0, 6, 4] == 1  # white pawn on e2
    assert planes[11, 0, 4] == 1  # black king on e8


def test_env_scores_checkmate_from_white_perspective():
    env = ChessEnv()
    env.reset()

    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        obs, reward, done, info = env.step(chess.Move.from_uci(uci))

    assert done is True
    assert reward == -1.0
    assert info["result"] == "0-1"
    assert obs.board_ascii == board_to_ascii(env.board)


def test_play_episode_records_every_ply_with_actor_returns():
    env = ChessEnv()
    episode = play_episode(
        env=env,
        white_policy=FirstLegalPolicy(),
        black_policy=FirstLegalPolicy(),
        max_plies=6,
    )

    assert len(episode.transitions) == 6
    assert episode.result == "*"
    assert episode.winner_reward == 0.0
    first = episode.transitions[0]
    assert first.state_ascii == board_to_ascii(chess.Board())
    assert isinstance(first.action_uci, str)
    assert first.player == chess.WHITE
    assert all(t.return_ is None for t in episode.transitions)


def test_replay_buffer_is_bounded_and_samples_deterministically():
    buffer = ReplayBuffer(capacity=3)
    for idx in range(5):
        buffer.add(
            Transition(
                state_ascii=f"board-{idx}",
                action_uci="e2e4",
                player=chess.WHITE,
                reward=0.0,
                done=False,
            )
        )

    assert [t.state_ascii for t in buffer] == ["board-2", "board-3", "board-4"]
    sample = buffer.sample(batch_size=2, seed=7)
    assert [t.state_ascii for t in sample] == ["board-3", "board-2"]


def test_tabular_agent_updates_move_values_from_returns():
    agent = TabularMoveValueAgent(learning_rate=0.5, epsilon=0.0)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    transition = Transition(
        state_ascii=board_to_ascii(board),
        action_uci=move.uci(),
        player=chess.WHITE,
        reward=0.0,
        done=True,
        return_=1.0,
    )

    agent.learn([transition])

    assert agent.value(board, move) == 0.5
    assert agent.select_move(board).uci() == "e2e4"


def test_train_self_play_runs_core_loop_and_returns_metrics():
    agent = TabularMoveValueAgent(learning_rate=0.2, epsilon=0.1, seed=123)
    metrics = train_self_play(agent=agent, episodes=3, max_plies=8, replay_capacity=100, seed=123)

    assert metrics.episodes == 3
    assert metrics.total_plies > 0
    assert metrics.replay_size == metrics.total_plies
    assert set(metrics.results).issubset({"1-0", "0-1", "1/2-1/2", "*"})
    assert len(agent.q) > 0


def test_cli_main_prints_training_summary(capsys):
    exit_code = main(["--episodes", "2", "--max-plies", "4", "--seed", "99"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "episodes=2" in captured.out
    assert "total_plies=" in captured.out
    assert "q_entries=" in captured.out


def test_cli_can_run_mcts_self_play_smoke(capsys):
    exit_code = main([
        "--policy",
        "mcts",
        "--episodes",
        "1",
        "--max-plies",
        "2",
        "--mcts-iterations",
        "2",
        "--seed",
        "99",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "policy=mcts" in captured.out
    assert "episodes=1" in captured.out


def test_cli_can_run_mcts_training_smoke(capsys):
    exit_code = main([
        "--policy",
        "mcts-train",
        "--episodes",
        "2",
        "--max-plies",
        "2",
        "--mcts-iterations",
        "2",
        "--rollout-depth",
        "1",
        "--seed",
        "99",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "policy=mcts-train" in captured.out
    assert "loss_curve=" in captured.out
    assert "policy_entries=" in captured.out
