import chess
import torch

from rl_chess.env import board_to_ascii
from rl_chess.nn_model import (
    ACTION_SIZE,
    ChessPolicyValueNet,
    NeuralPolicyValueTrainer,
    action_index,
    encode_board_ascii,
)
from rl_chess.replay import SearchTrainingExample
from rl_chess.train import train_neural_mcts_self_play


def test_encode_board_ascii_preserves_piece_locations_and_turn_plane():
    board = chess.Board()

    encoded = encode_board_ascii(board_to_ascii(board), turn=chess.WHITE)

    assert encoded.shape == (13, 8, 8)
    assert encoded[0, 6, 0] == 1.0  # white pawn at a2
    assert encoded[5, 7, 4] == 1.0  # white king at e1
    assert encoded[6, 1, 0] == 1.0  # black pawn at a7
    assert encoded[11, 0, 4] == 1.0  # black king at e8
    assert encoded[12].sum() == 64.0


def test_action_index_maps_normal_and_promotion_moves_into_fixed_space():
    assert 0 <= action_index("e2e4") < ACTION_SIZE
    assert 0 <= action_index("a7a8q") < ACTION_SIZE
    assert action_index("a7a8q") != action_index("a7a8n")


def test_policy_value_network_outputs_policy_logits_and_value():
    model = ChessPolicyValueNet(hidden_channels=16)
    x = torch.zeros((2, 13, 8, 8), dtype=torch.float32)

    logits, values = model(x)

    assert logits.shape == (2, ACTION_SIZE)
    assert values.shape == (2,)
    assert torch.all(values <= 1.0)
    assert torch.all(values >= -1.0)


def test_neural_trainer_updates_from_mcts_policy_and_value_targets():
    torch.manual_seed(0)
    model = ChessPolicyValueNet(hidden_channels=16)
    trainer = NeuralPolicyValueTrainer(model=model, learning_rate=0.01)
    board = chess.Board()
    example = SearchTrainingExample(
        state_ascii=board_to_ascii(board),
        turn=chess.WHITE,
        legal_moves=("e2e4", "d2d4"),
        policy_target={"e2e4": 0.75, "d2d4": 0.25},
        value_target=1.0,
    )

    before = [param.detach().clone() for param in model.parameters()]
    stats = trainer.train_batch([example])

    assert stats.total_loss > 0.0
    assert stats.policy_loss > 0.0
    assert stats.value_loss > 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_train_neural_mcts_self_play_returns_policy_and_value_loss_curves():
    torch.manual_seed(0)
    model = ChessPolicyValueNet(hidden_channels=8)

    metrics = train_neural_mcts_self_play(
        model=model,
        episodes=2,
        max_plies=2,
        mcts_iterations=2,
        rollout_depth=1,
        learning_rate=0.01,
        seed=7,
    )

    assert metrics.episodes == 2
    assert metrics.examples_collected == 4
    assert len(metrics.loss_curve) == 2
    assert len(metrics.policy_loss_curve) == 2
    assert len(metrics.value_loss_curve) == 2
    assert all(loss > 0.0 for loss in metrics.loss_curve)
