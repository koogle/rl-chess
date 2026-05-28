import chess
import pytest
import torch

from rl_chess.env import ChessEnv, ascii_to_board, board_to_ascii, result_to_white_reward
from rl_chess.nn_model import PolicyValueNet, train_batch
from rl_chess.puct_mcts import PUCTMCTS, PolicyValueEvaluator
from rl_chess.self_play import TrainingExample, play_self_game, sample_policy
from rl_chess.train import load_checkpoint_model, train
from rl_chess.validation import (
    FixedMovePlayer,
    FirstLegalPlayer,
    ValidationResult,
    resolve_stockfish_path,
    stockfish_strength_config,
    play_validation_game,
)

KQK_BLACK_TO_MOVE = """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 . . . . . . . . 2
1 ♔ . ♚ ♕ . . . . 1
  a b c d e f g h"""

MATE_IN_ONE = """  a b c d e f g h
8 . . . . . . . ♚ 8
7 . . . . . . . ♕ 7
6 . . . . . . ♔ . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 . . . . . . . . 2
1 . . . . . . . . 1
  a b c d e f g h"""



class E4Evaluator(PolicyValueEvaluator):
    def evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        priors = {move.uci(): 1.0 for move in board.legal_moves}
        if "e2e4" in priors:
            priors["e2e4"] = 100.0
        total = sum(priors.values())
        return {move: weight / total for move, weight in priors.items()}, 0.0


def test_env_uses_unicode_board_and_python_chess_legality():
    env = ChessEnv()
    obs = env.reset()
    assert "♔" in obs.board_ascii
    next_obs, reward, done, info = env.step("e2e4")
    assert reward == 0.0
    assert done is False
    assert info["result"] is None
    assert "e7e5" in next_obs.legal_moves


def test_board_encoder_preserves_visual_state_and_side_to_move():
    board = chess.Board()
    encoded = PolicyValueNet.encode_board_ascii(board_to_ascii(board), board.turn)
    assert encoded.shape == (13, 8, 8)
    assert encoded[:12].sum().item() == 32
    assert encoded[12].sum().item() == 64


def test_puct_uses_priors_for_visit_policy():
    policy = PUCTMCTS(E4Evaluator(), iterations=16, seed=1).search_policy(chess.Board())
    assert policy["e2e4"] == max(policy.values())


def test_ascii_board_parser_reconstructs_python_chess_position():
    board = ascii_to_board(KQK_BLACK_TO_MOVE, turn=chess.BLACK)
    assert board.turn == chess.BLACK
    assert board_to_ascii(board) == KQK_BLACK_TO_MOVE
    assert {move.uci() for move in board.legal_moves} == {"c1d1"}


def test_package_has_no_local_console_training_entrypoint():
    import tomllib
    from pathlib import Path

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "scripts" not in project.get("project", {})


def test_modal_app_exposes_training_entrypoint_without_endgame_diagnostic():
    from rl_chess import modal_app

    assert hasattr(modal_app, "train_remote")
    assert not hasattr(modal_app, "validate_endgames_remote")


def test_modal_remote_training_accepts_ascii_starting_board():
    from rl_chess.modal_app import train_remote

    summary = train_remote.local(
        iterations=1,
        games_per_iteration=1,
        max_plies=1,
        simulations=1,
        train_steps=1,
        batch_size=1,
        hidden_channels=8,
        residual_blocks=0,
        starting_board_ascii=KQK_BLACK_TO_MOVE,
        starting_turn="black",
        seed=1,
    )
    assert summary["loop"] == "nn-puct"
    assert summary["games"] == 1
    assert summary["hidden_channels"] == 8
    assert summary["residual_blocks"] == 0


def test_policy_value_trainer_reduces_loss_on_repeated_target():
    torch.manual_seed(1)
    board = chess.Board()
    example = TrainingExample(
        state_ascii=board_to_ascii(board),
        turn=board.turn,
        policy_target={"e2e4": 1.0},
        value_target=0.5,
    )
    model = PolicyValueNet(hidden_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    first = train_batch(model, optimizer, [example]).total_loss
    last = first
    for _ in range(12):
        last = train_batch(model, optimizer, [example]).total_loss
    assert last < first


def test_model_is_the_puct_evaluator():
    model = PolicyValueNet(hidden_channels=8)
    priors, white_value = model.evaluate(chess.Board())
    assert "e2e4" in priors
    assert abs(sum(priors.values()) - 1.0) < 1e-6
    assert -1.0 <= white_value <= 1.0


def test_model_uses_a_deeper_residual_tower():
    model = PolicyValueNet(hidden_channels=8, residual_blocks=3)
    assert model.residual_blocks == 3
    logits, values = model(torch.zeros((2, 13, 8, 8)))
    assert logits.shape == (2, 64 * 64 * 5)
    assert values.shape == (2,)


def test_self_play_can_be_uncapped_until_terminal_from_mate_in_one():
    board = ascii_to_board(MATE_IN_ONE, turn=chess.WHITE)
    env = ChessEnv(starting_board=board)
    # Direct env proof: no environment turn cap exists.
    _obs, reward, done, info = env.step("h7g7")
    assert done is True
    assert result_to_white_reward(info["result"]) == reward == 1.0


def test_self_play_rejects_safety_cap_instead_of_truncating_game():
    with pytest.raises(RuntimeError, match="non-terminal self-play game reached safety cap"):
        play_self_game(E4Evaluator(), simulations=1, max_plies=1, seed=3)


def test_training_metrics_do_not_report_truncation():
    starting_board = ascii_to_board(KQK_BLACK_TO_MOVE, turn=chess.BLACK)
    metrics = train(
        model=PolicyValueNet(hidden_channels=8),
        iterations=2,
        games_per_iteration=1,
        simulations=2,
        max_plies=1,
        train_steps=1,
        starting_board=starting_board,
        seed=3,
    )
    assert metrics.games == 2
    assert metrics.examples == 2
    assert metrics.terminal_games == 2
    assert not hasattr(metrics, "truncated_games")
    assert len(metrics.loss_curve) == 2


def test_training_writes_iteration_checkpoints(tmp_path):
    model = PolicyValueNet(hidden_channels=8)
    metrics = train(
        model=model,
        iterations=2,
        games_per_iteration=1,
        simulations=2,
        max_plies=1,
        train_steps=1,
        starting_board=ascii_to_board(KQK_BLACK_TO_MOVE, turn=chess.BLACK),
        seed=3,
        checkpoint_dir=tmp_path,
    )

    assert [path.name for path in metrics.checkpoint_paths] == ["iteration-0001.pt", "iteration-0002.pt"]
    checkpoint = torch.load(metrics.checkpoint_paths[-1], map_location="cpu", weights_only=True)
    assert checkpoint["metrics"]["checkpoint_paths"][-1].endswith("iteration-0002.pt")
    reloaded = load_checkpoint_model(metrics.checkpoint_paths[-1])
    assert isinstance(reloaded, PolicyValueNet)


def test_training_rejects_invalid_public_knobs():
    model = PolicyValueNet(hidden_channels=8)
    bad_configs = [
        {"iterations": 0},
        {"games_per_iteration": 0},
        {"simulations": 0},
        {"max_plies": 0},
        {"train_steps": 0},
        {"batch_size": 0},
        {"replay_capacity": 0},
        {"learning_rate": 0.0},
        {"temperature": -1.0},
    ]
    for kwargs in bad_configs:
        params = {"iterations": 1, **kwargs}
        with pytest.raises(ValueError):
            train(model=model, **params)


def test_training_example_rejects_invalid_policy_targets():
    board = chess.Board()
    base = {"state_ascii": board_to_ascii(board), "turn": board.turn, "value_target": 0.0}
    for policy_target in [{}, {"e2e4": 0.0}, {"e2e4": -1.0}, {"not-uci": 1.0}]:
        with pytest.raises(ValueError):
            TrainingExample(policy_target=policy_target, **base)


def test_sample_policy_rejects_negative_temperature():
    with pytest.raises(ValueError):
        sample_policy({"e2e4": 1.0}, temperature=-0.1, rng=__import__("random").Random(1))


def test_stockfish_path_resolver_handles_debian_usr_games_layout(tmp_path):
    stockfish = tmp_path / "stockfish"
    stockfish.write_text("#!/bin/sh\n", encoding="utf-8")
    assert resolve_stockfish_path(str(stockfish)) == str(stockfish)


def test_stockfish_supported_elo_floor_uses_uci_limit_strength():
    class Option:
        def __init__(self, min_value, max_value):
            self.min = min_value
            self.max = max_value

    assert stockfish_strength_config(
        1320,
        {
            "UCI_LimitStrength": Option(None, None),
            "UCI_Elo": Option(1320, 3190),
            "Skill Level": Option(0, 20),
        },
    ) == {"UCI_LimitStrength": True, "UCI_Elo": 1320}


def test_validation_game_scores_candidate_result_from_candidate_perspective():
    board = ascii_to_board(MATE_IN_ONE, turn=chess.WHITE)
    result = play_validation_game(
        candidate=FixedMovePlayer(["h7g7"]),
        baseline=FirstLegalPlayer(),
        candidate_color=chess.WHITE,
        starting_board=board,
    )
    assert result == ValidationResult(wins=1, losses=0, draws=0)
    assert result.score == 1.0
    assert result.passed is True


def test_validation_game_can_score_candidate_loss_as_black():
    board = ascii_to_board(MATE_IN_ONE, turn=chess.WHITE)
    result = play_validation_game(
        candidate=FirstLegalPlayer(),
        baseline=FixedMovePlayer(["h7g7"]),
        candidate_color=chess.BLACK,
        starting_board=board,
    )
    assert result == ValidationResult(wins=0, losses=1, draws=0)
    assert result.score == 0.0
    assert result.passed is False
