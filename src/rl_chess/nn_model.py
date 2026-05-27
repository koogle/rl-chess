from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from torch import nn
import torch.nn.functional as F

from rl_chess.env import board_to_ascii
from rl_chess.self_play import TrainingExample

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
BASE_MOVE_SIZE = 64 * 64
ACTION_SIZE = BASE_MOVE_SIZE * 5
PROMOTION_TO_OFFSET = {"q": 0, "r": 1, "b": 2, "n": 3}


@dataclass(frozen=True)
class TrainStats:
    total_loss: float
    policy_loss: float
    value_loss: float


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return F.relu(states + self.layers(states))


class PolicyValueNet(nn.Module):
    """Small inspectable chess policy/value model.

    This class owns the neural inference boundary:
    - visual board encoding
    - UCI move indexing
    - policy/value forward pass
    - PUCT evaluator interface
    """

    def __init__(self, hidden_channels: int = 64, residual_blocks: int = 4) -> None:
        super().__init__()
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be non-negative")
        self.hidden_channels = hidden_channels
        self.residual_blocks = residual_blocks
        self.trunk = nn.Sequential(
            nn.Conv2d(13, hidden_channels, 3, padding=1),
            nn.ReLU(),
            *(ResidualBlock(hidden_channels) for _ in range(residual_blocks)),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(states)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Return legal UCI move priors and value from White's perspective for PUCT."""

        was_training = self.training
        self.eval()
        legal_moves = tuple(move.uci() for move in board.legal_moves)
        if not legal_moves:
            return {}, 0.0

        device = next(self.parameters()).device
        state = self.encode_board_ascii(board_to_ascii(board), board.turn).unsqueeze(0).to(device)
        logits, values = self(state)
        indices = torch.tensor([self.action_index(move) for move in legal_moves], dtype=torch.long, device=device)
        probs = torch.softmax(logits[0, indices], dim=0)
        side_value = float(values[0])
        white_value = side_value if board.turn == chess.WHITE else -side_value
        if was_training:
            self.train()
        return {move: float(prob) for move, prob in zip(legal_moves, probs)}, white_value

    @staticmethod
    def encode_board_ascii(board_ascii: str, turn: bool) -> torch.Tensor:
        """Encode the project's visual Unicode board: 12 piece planes + side to move."""

        tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
        rank_lines = [line for line in board_ascii.splitlines() if line and line[0] in "12345678"]
        if len(rank_lines) != 8:
            raise ValueError("board_ascii must contain 8 rank lines")

        for row, line in enumerate(rank_lines):
            squares = line.split()[1:9]
            if len(squares) != 8:
                raise ValueError(f"rank line must contain 8 squares: {line!r}")
            for col, symbol in enumerate(squares):
                if symbol != ".":
                    tensor[PIECE_TO_PLANE[symbol], row, col] = 1.0

        if turn == chess.WHITE:
            tensor[12].fill_(1.0)
        return tensor

    @staticmethod
    def action_index(uci: str) -> int:
        move = chess.Move.from_uci(uci)
        base = move.from_square * 64 + move.to_square
        if move.promotion is None:
            return base
        return BASE_MOVE_SIZE + base * 4 + PROMOTION_TO_OFFSET[chess.piece_symbol(move.promotion)]

    @classmethod
    def policy_loss(cls, logits: torch.Tensor, examples: list[TrainingExample]) -> torch.Tensor:
        """Cross entropy over legal/search moves only, matching PUCT inference."""

        losses = []
        for row, example in enumerate(examples):
            moves = tuple(example.policy_target)
            indices = torch.tensor([cls.action_index(move) for move in moves], dtype=torch.long, device=logits.device)
            target = torch.tensor([example.policy_target[move] for move in moves], dtype=torch.float32, device=logits.device)
            target = target / target.sum()
            losses.append(-(target * F.log_softmax(logits[row, indices], dim=0)).sum())
        return torch.stack(losses).mean()


def train_batch(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    examples: list[TrainingExample],
    value_loss_weight: float = 1.0,
) -> TrainStats:
    if value_loss_weight < 0:
        raise ValueError("value_loss_weight must be non-negative")
    if not examples:
        return TrainStats(0.0, 0.0, 0.0)

    model.train()
    device = next(model.parameters()).device
    states = torch.stack([model.encode_board_ascii(ex.state_ascii, ex.turn) for ex in examples]).to(device)
    logits, values = model(states)
    policy_loss = model.policy_loss(logits, examples)
    value_targets = torch.tensor([ex.value_target for ex in examples], dtype=torch.float32, device=device)
    value_loss = F.mse_loss(values, value_targets)
    total_loss = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return TrainStats(float(total_loss.detach()), float(policy_loss.detach()), float(value_loss.detach()))
