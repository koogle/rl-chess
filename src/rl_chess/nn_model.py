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
INPUT_CHANNELS = 21
SIDE_TO_MOVE_PLANE = 12
WHITE_KINGSIDE_CASTLING_PLANE = 13
WHITE_QUEENSIDE_CASTLING_PLANE = 14
BLACK_KINGSIDE_CASTLING_PLANE = 15
BLACK_QUEENSIDE_CASTLING_PLANE = 16
EN_PASSANT_PLANE = 17
HALFMOVE_CLOCK_PLANE = 18
CAN_CLAIM_THREEFOLD_PLANE = 19
CAN_CLAIM_FIFTY_MOVES_PLANE = 20
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
            nn.Conv2d(INPUT_CHANNELS, hidden_channels, 3, padding=1),
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
        state = self.encode_board(board).unsqueeze(0).to(device)
        logits, values = self(state)
        indices = torch.tensor([self.action_index(move) for move in legal_moves], dtype=torch.long, device=device)
        probs = torch.softmax(logits[0, indices], dim=0)
        side_value = float(values[0])
        white_value = side_value if board.turn == chess.WHITE else -side_value
        if was_training:
            self.train()
        return {move: float(prob) for move, prob in zip(legal_moves, probs)}, white_value

    @classmethod
    def encode_board(cls, board: chess.Board) -> torch.Tensor:
        return cls.encode_board_ascii(
            board_to_ascii(board),
            board.turn,
            castling_rights=board.castling_rights,
            ep_square=board.ep_square,
            halfmove_clock=board.halfmove_clock,
            can_claim_threefold=board.can_claim_threefold_repetition(),
            can_claim_fifty_moves=board.can_claim_fifty_moves(),
        )

    @classmethod
    def encode_training_example(cls, example: TrainingExample) -> torch.Tensor:
        return cls.encode_board_ascii(
            example.state_ascii,
            example.turn,
            castling_rights=example.castling_rights,
            ep_square=example.ep_square,
            halfmove_clock=example.halfmove_clock,
            can_claim_threefold=example.can_claim_threefold,
            can_claim_fifty_moves=example.can_claim_fifty_moves,
        )

    @staticmethod
    def encode_board_ascii(
        board_ascii: str,
        turn: bool,
        castling_rights: int = 0,
        ep_square: int | None = None,
        halfmove_clock: int = 0,
        can_claim_threefold: bool = False,
        can_claim_fifty_moves: bool = False,
    ) -> torch.Tensor:
        """Encode the visual board plus legal-state planes."""

        tensor = torch.zeros((INPUT_CHANNELS, 8, 8), dtype=torch.float32)
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
            tensor[SIDE_TO_MOVE_PLANE].fill_(1.0)
        if castling_rights & chess.BB_H1:
            tensor[WHITE_KINGSIDE_CASTLING_PLANE].fill_(1.0)
        if castling_rights & chess.BB_A1:
            tensor[WHITE_QUEENSIDE_CASTLING_PLANE].fill_(1.0)
        if castling_rights & chess.BB_H8:
            tensor[BLACK_KINGSIDE_CASTLING_PLANE].fill_(1.0)
        if castling_rights & chess.BB_A8:
            tensor[BLACK_QUEENSIDE_CASTLING_PLANE].fill_(1.0)
        if ep_square is not None:
            tensor[EN_PASSANT_PLANE, 7 - chess.square_rank(ep_square), chess.square_file(ep_square)] = 1.0
        tensor[HALFMOVE_CLOCK_PLANE].fill_(min(float(halfmove_clock), 100.0) / 100.0)
        if can_claim_threefold:
            tensor[CAN_CLAIM_THREEFOLD_PLANE].fill_(1.0)
        if can_claim_fifty_moves:
            tensor[CAN_CLAIM_FIFTY_MOVES_PLANE].fill_(1.0)
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
    states = torch.stack([model.encode_training_example(ex) for ex in examples]).to(device)
    logits, values = model(states)
    policy_loss = model.policy_loss(logits, examples)
    value_targets = torch.tensor([ex.value_target for ex in examples], dtype=torch.float32, device=device)
    value_loss = F.mse_loss(values, value_targets)
    total_loss = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return TrainStats(float(total_loss.detach()), float(policy_loss.detach()), float(value_loss.detach()))
