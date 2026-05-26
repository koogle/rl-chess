from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from torch import nn
import torch.nn.functional as F

from rl_chess.replay import SearchTrainingExample

PIECE_TO_PLANE = {
    "♙": 0,
    "♘": 1,
    "♗": 2,
    "♖": 3,
    "♕": 4,
    "♔": 5,
    "♟": 6,
    "♞": 7,
    "♝": 8,
    "♜": 9,
    "♛": 10,
    "♚": 11,
}
PROMOTION_TO_OFFSET = {"q": 0, "r": 1, "b": 2, "n": 3}
BASE_MOVE_SIZE = 64 * 64
ACTION_SIZE = BASE_MOVE_SIZE * 5


def action_index(uci: str) -> int:
    """Map a UCI move into a fixed policy head index."""

    move = chess.Move.from_uci(uci)
    base = move.from_square * 64 + move.to_square
    if move.promotion is None:
        return base
    promotion_symbol = chess.piece_symbol(move.promotion)
    return BASE_MOVE_SIZE + base * 4 + PROMOTION_TO_OFFSET[promotion_symbol]


def encode_board_ascii(board_ascii: str, turn: bool) -> torch.Tensor:
    """Encode the visual Unicode board into 12 piece planes + side-to-move."""

    tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
    rank_lines = [line for line in board_ascii.splitlines() if line and line[0] in "12345678"]
    if len(rank_lines) != 8:
        raise ValueError("board_ascii must contain 8 rank lines")

    for row, line in enumerate(rank_lines):
        parts = line.split()
        squares = parts[1:9]
        if len(squares) != 8:
            raise ValueError(f"rank line must contain 8 squares: {line!r}")
        for col, symbol in enumerate(squares):
            if symbol == ".":
                continue
            try:
                plane = PIECE_TO_PLANE[symbol]
            except KeyError as exc:
                raise ValueError(f"unsupported board symbol {symbol!r}") from exc
            tensor[plane, row, col] = 1.0

    if turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    return tensor


class ChessPolicyValueNet(nn.Module):
    """Small policy/value network for AlphaZero-style distillation."""

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(13, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


@dataclass(frozen=True)
class NeuralTrainStats:
    total_loss: float
    policy_loss: float
    value_loss: float


@dataclass
class NeuralPolicyValueTrainer:
    """Hand-written supervised RL update for MCTS policy/value targets."""

    model: ChessPolicyValueNet
    learning_rate: float = 1e-3
    value_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_batch(self, examples: list[SearchTrainingExample]) -> NeuralTrainStats:
        if not examples:
            return NeuralTrainStats(total_loss=0.0, policy_loss=0.0, value_loss=0.0)

        self.model.train()
        states = torch.stack([encode_board_ascii(ex.state_ascii, ex.turn) for ex in examples])
        policy_targets = torch.zeros((len(examples), ACTION_SIZE), dtype=torch.float32)
        value_targets = torch.zeros((len(examples),), dtype=torch.float32)

        for row, example in enumerate(examples):
            total_prob = sum(example.policy_target.values())
            if total_prob <= 0.0:
                raise ValueError("policy_target must have positive mass")
            for uci, prob in example.policy_target.items():
                policy_targets[row, action_index(uci)] = prob / total_prob
            value_targets[row] = 0.0 if example.value_target is None else example.value_target

        logits, values = self.model(states)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)
        total_loss = policy_loss + self.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return NeuralTrainStats(
            total_loss=float(total_loss.detach()),
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
        )

    @torch.no_grad()
    def policy(self, state_ascii: str, turn: bool, legal_moves: tuple[str, ...]) -> dict[str, float]:
        self.model.eval()
        state = encode_board_ascii(state_ascii, turn).unsqueeze(0)
        logits, _values = self.model(state)
        legal_indices = torch.tensor([action_index(move) for move in legal_moves], dtype=torch.long)
        legal_probs = torch.softmax(logits[0, legal_indices], dim=0)
        return {move: float(prob) for move, prob in zip(legal_moves, legal_probs)}


@dataclass
class NeuralPolicyValueEvaluator:
    """Use the neural policy/value net as a PUCT evaluator.

    The model's value head is trained from the side-to-move perspective. MCTS
    backpropagation expects values from White's perspective, so black-to-move
    values are negated before being returned.
    """

    model: ChessPolicyValueNet

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        self.model.eval()
        legal_moves = tuple(move.uci() for move in board.legal_moves)
        if not legal_moves:
            return {}, 0.0

        state_ascii = _board_to_ascii_without_env_import(board)
        state = encode_board_ascii(state_ascii, board.turn).unsqueeze(0)
        logits, values = self.model(state)
        legal_indices = torch.tensor([action_index(move) for move in legal_moves], dtype=torch.long)
        legal_probs = torch.softmax(logits[0, legal_indices], dim=0)
        side_to_move_value = float(values[0])
        white_value = side_to_move_value if board.turn == chess.WHITE else -side_to_move_value
        return {move: float(prob) for move, prob in zip(legal_moves, legal_probs)}, white_value


def _board_to_ascii_without_env_import(board: chess.Board) -> str:
    files = "a b c d e f g h"
    lines = [f"  {files}"]
    white_symbols = {
        chess.PAWN: "♙",
        chess.KNIGHT: "♘",
        chess.BISHOP: "♗",
        chess.ROOK: "♖",
        chess.QUEEN: "♕",
        chess.KING: "♔",
    }
    black_symbols = str.maketrans("♙♘♗♖♕♔", "♟♞♝♜♛♚")
    for rank in range(7, -1, -1):
        row: list[str] = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                row.append(".")
                continue
            symbol = white_symbols[piece.piece_type]
            row.append(symbol if piece.color == chess.WHITE else symbol.translate(black_symbols))
        rank_label = str(rank + 1)
        lines.append(f"{rank_label} {' '.join(row)} {rank_label}")
    lines.append(f"  {files}")
    return "\n".join(lines)
