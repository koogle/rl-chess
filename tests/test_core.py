import chess
import pytest
import torch

from rl_chess.env import ChessEnv, board_to_ascii, result_to_white_reward
from rl_chess.nn_model import PolicyValueNet, PolicyValueTrainer, encode_board_ascii
from rl_chess.puct_mcts import PUCTMCTS, PolicyValueEvaluator
from rl_chess.self_play import TrainingExample, sample_policy
from rl_chess.train import train
from rl_chess.validation import (
    FixedMovePlayer,
    FirstLegalPlayer,
    ValidationResult,
    stockfish_strength_config,
    play_validation_game,
)


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
    encoded = encode_board_ascii(board_to_ascii(board), board.turn)
    assert encoded.shape == (13, 8, 8)
    assert encoded[:12].sum().item() == 32
    assert encoded[12].sum().item() == 64


def test_puct_uses_priors_for_visit_policy():
    policy = PUCTMCTS(E4Evaluator(), iterations=16, seed=1).search_policy(chess.Board())
    assert policy["e2e4"] == max(policy.values())


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
    trainer = PolicyValueTrainer(model, learning_rate=0.01)
    first = trainer.train_batch([example]).total_loss
    last = first
    for _ in range(12):
        last = trainer.train_batch([example]).total_loss
    assert last < first


def test_self_play_can_be_uncapped_until_terminal_from_mate_in_one():
    board = chess.Board("7k/7Q/6K1/8/8/8/8/8 w - - 0 1")
    env = ChessEnv(starting_board=board)
    # Direct env proof: no environment turn cap exists.
    _obs, reward, done, info = env.step("h7g7")
    assert done is True
    assert result_to_white_reward(info["result"]) == reward == 1.0


def test_training_reports_truncation_instead_of_hiding_it_as_learning():
    metrics = train(
        model=PolicyValueNet(hidden_channels=8),
        iterations=2,
        games_per_iteration=1,
        simulations=2,
        max_plies=2,
        train_steps=1,
        seed=3,
    )
    assert metrics.games == 2
    assert metrics.examples == 4
    assert metrics.truncated_games == 2
    assert len(metrics.loss_curve) == 2


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


def test_stockfish_elo_500_uses_weakest_skill_when_uci_elo_floor_is_higher():
    class Option:
        def __init__(self, min_value, max_value):
            self.min = min_value
            self.max = max_value

    assert stockfish_strength_config(
        500,
        {
            "UCI_LimitStrength": Option(None, None),
            "UCI_Elo": Option(1320, 3190),
            "Skill Level": Option(0, 20),
        },
    ) == {"Skill Level": 0}


def test_validation_game_scores_candidate_result_from_candidate_perspective():
    board = chess.Board("7k/7Q/6K1/8/8/8/8/8 w - - 0 1")
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
    board = chess.Board("7k/7Q/6K1/8/8/8/8/8 w - - 0 1")
    result = play_validation_game(
        candidate=FirstLegalPlayer(),
        baseline=FixedMovePlayer(["h7g7"]),
        candidate_color=chess.BLACK,
        starting_board=board,
    )
    assert result == ValidationResult(wins=0, losses=1, draws=0)
    assert result.score == 0.0
    assert result.passed is False


def test_cli_can_run_stockfish_validation_with_injected_runner(monkeypatch, capsys):
    from rl_chess import cli

    def fake_validate_model_against_stockfish(**kwargs):
        assert kwargs["elo"] == 500
        assert kwargs["games"] == 1
        return ValidationResult(wins=1)

    monkeypatch.setattr(cli, "validate_model_against_stockfish", fake_validate_model_against_stockfish)
    exit_code = cli.main([
        "--iterations",
        "1",
        "--max-plies",
        "2",
        "--mcts-iterations",
        "2",
        "--hidden-channels",
        "8",
        "--validate-stockfish",
        "--validation-games",
        "1",
        "--seed",
        "1",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stockfish_elo=500" in captured.out
    assert "validation_passed=True" in captured.out


def test_cli_smoke(capsys):
    from rl_chess.cli import main

    exit_code = main(["--iterations", "1", "--max-plies", "2", "--mcts-iterations", "2", "--hidden-channels", "8", "--seed", "1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "loop=nn-puct" in captured.out
    assert "truncated_games=" in captured.out
