from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import chess
import torch
import torch.nn.functional as F

from rl_chess.env import ascii_to_board, board_to_ascii, result_to_white_reward
from rl_chess.nn_model import PolicyValueNet
from rl_chess.self_play import TrainingExample


@dataclass(frozen=True)
class EndgamePosition:
    board_ascii: str
    turn: bool

    def board(self) -> chess.Board:
        return ascii_to_board(self.board_ascii, self.turn)


DEFAULT_ENDGAME_POSITIONS = [
    # Ten KQK positions where White is not mating in one, but can force mate
    # within five plies. They keep validation focused on value propagation from
    # terminal results rather than opening theory or Stockfish imitation.
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . ♚ 3
2 . . ♕ . . . . . 2
1 . . . . ♔ . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 ♕ ♔ . . . . . . 4
3 . . . . . . . . 3
2 . . . . . . . . 2
1 . . ♚ . . . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . ♔ . . . . 6
5 . . . . . . . . 5
4 ♚ . . . . . . . 4
3 . . . . . . . ♕ 3
2 . . . . . . . . 2
1 . . . . . . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . ♚ . 8
7 . . . . . . . . 7
6 . . . . . . ♔ . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 . . . . . ♕ . . 2
1 . . . . . . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . ♕ . 6
5 . . . . . . . . 5
4 . . . ♔ . . . . 4
3 . . . . . . . . 3
2 . . . . . . . . 2
1 . . . ♚ . . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . ♕ ♔ . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 ♚ . . . . . . . 4
3 . . . . . . . . 3
2 . . . . . . . . 2
1 . . . . . . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . ♔ . . 3
2 . . . . . . ♕ . 2
1 . . . . . . . ♚ 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . ♕ . . . 4
3 . . . . . ♔ . . 3
2 . . . . . . . . 2
1 . . . . . . ♚ . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . ♚ . . 8
7 . . . . . . . . 7
6 . . . ♔ . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . ♕ . 3
2 . . . . . . . . 2
1 . . . . . . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
    EndgamePosition(
        """  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . ♔ . . 4
3 . . . . . . . ♕ 3
2 . . . . . . . . 2
1 . . . . ♚ . . . 1
  a b c d e f g h""",
        chess.WHITE,
    ),
]


@dataclass(frozen=True)
class ValueExample:
    board: chess.Board
    target: float


def board_state_key(board: chess.Board) -> tuple[str, bool]:
    return (board_to_ascii(board), board.turn)


def forced_white_value(board: chess.Board, depth: int, cache: dict[tuple[tuple[str, bool], int], float] | None = None) -> float:
    cache = {} if cache is None else cache
    key = (board_state_key(board), depth)
    if key in cache:
        return cache[key]
    if board.is_game_over(claim_draw=True):
        value = result_to_white_reward(board.result(claim_draw=True))
    elif depth == 0:
        value = 0.0
    else:
        child_values = []
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            child_values.append(forced_white_value(child, depth - 1, cache))
        value = max(child_values) if board.turn == chess.WHITE else min(child_values)
    cache[key] = value
    return value


def collect_principal_value_line(start: chess.Board, depth: int) -> list[ValueExample]:
    """Collect terminal-backed values needed to choose along one winning line.

    At each state on the principal line, include the state itself and every legal
    child. That is the smallest dataset that lets a one-ply value-greedy player
    compare candidate moves without training an enormous full-width tree.
    """

    board = start.copy(stack=False)
    line: dict[tuple[str, bool], ValueExample] = {}
    cache: dict[tuple[tuple[str, bool], int], float] = {}
    for remaining_depth in range(depth, -1, -1):
        if board.is_game_over(claim_draw=True):
            break
        value = forced_white_value(board, remaining_depth, cache)
        if value == 0.0:
            break
        line[board_state_key(board)] = ValueExample(board.copy(stack=False), value)
        if remaining_depth == 0:
            break
        scored_children: list[tuple[float, chess.Move, chess.Board]] = []
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            child_value = forced_white_value(child, remaining_depth - 1, cache)
            if child.legal_moves.count() > 0:
                line[board_state_key(child)] = ValueExample(child.copy(stack=False), child_value)
            scored_children.append((child_value, move, child))
        best = max(scored_children, key=lambda item: item[0]) if board.turn == chess.WHITE else min(scored_children, key=lambda item: item[0])
        board = best[2]
    return list(line.values())


def build_value_dataset(positions: list[EndgamePosition], depth: int) -> list[TrainingExample]:
    by_position: dict[tuple[str, bool], TrainingExample] = {}
    for position in positions:
        for item in collect_principal_value_line(position.board(), depth):
            board = item.board
            legal_moves = tuple(move.uci() for move in board.legal_moves)
            actor_value = item.target if board.turn == chess.WHITE else -item.target
            by_position[board_state_key(board)] = TrainingExample(
                state_ascii=board_to_ascii(board),
                turn=board.turn,
                policy_target={move: 1.0 for move in legal_moves},
                value_target=actor_value,
            )
    return list(by_position.values())


def train_value_batch(model: PolicyValueNet, optimizer: torch.optim.Optimizer, examples: list[TrainingExample]) -> float:
    model.train()
    device = next(model.parameters()).device
    states = torch.stack([model.encode_board_ascii(example.state_ascii, example.turn) for example in examples]).to(device)
    targets = torch.tensor([example.value_target for example in examples], dtype=torch.float32, device=device)
    _logits, values = model(states)
    loss = F.mse_loss(values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate_values(model: PolicyValueNet, examples: list[TrainingExample]) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    states = torch.stack([model.encode_board_ascii(example.state_ascii, example.turn) for example in examples]).to(device)
    targets = torch.tensor([example.value_target for example in examples], dtype=torch.float32, device=device)
    _logits, values = model(states)
    sign_hits = ((values * targets) > 0).float().mean().item()
    return {
        "mse": float(F.mse_loss(values, targets).detach()),
        "sign_accuracy": float(sign_hits),
        "min_actor_value": float(values.min().detach()),
        "max_actor_value": float(values.max().detach()),
    }


@torch.no_grad()
def child_actor_value(model: PolicyValueNet, child: chess.Board, parent_turn: bool) -> float:
    _priors, white_value = model.evaluate(child)
    return white_value if parent_turn == chess.WHITE else -white_value


def value_greedy_move(model: PolicyValueNet, board: chess.Board) -> chess.Move:
    scored: list[tuple[float, str, chess.Move]] = []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_game_over(claim_draw=True):
            white_value = result_to_white_reward(child.result(claim_draw=True))
            actor_value = white_value if board.turn == chess.WHITE else -white_value
        else:
            actor_value = child_actor_value(model, child, board.turn)
        scored.append((actor_value, move.uci(), move))
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def play_value_greedy(model: PolicyValueNet, position: EndgamePosition, max_plies: int) -> dict[str, Any]:
    board = position.board()
    root_turn = board.turn
    moves: list[str] = []
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        move = value_greedy_move(model, board)
        moves.append(move.uci())
        board.push(move)
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None
    root_score = 0.0
    if result is not None:
        white_reward = result_to_white_reward(result)
        root_score = white_reward if root_turn == chess.WHITE else -white_reward
    return {
        "start_board_ascii": position.board_ascii,
        "start_turn": "white" if position.turn == chess.WHITE else "black",
        "moves": moves,
        "plies": len(moves),
        "result": result,
        "terminal": result is not None,
        "root_score": root_score,
        "won": root_score == 1.0,
        "final_board_ascii": board_to_ascii(board),
        "final_turn": "white" if board.turn == chess.WHITE else "black",
    }


def run_endgame_value_validation(
    positions: list[EndgamePosition] | None = None,
    depth: int = 5,
    hidden_channels: int = 64,
    residual_blocks: int = 4,
    steps: int = 400,
    learning_rate: float = 0.001,
    seed: int = 1,
    max_plies: int = 5,
    report_every: int = 50,
    batch_size: int = 64,
) -> dict[str, Any]:
    positions = DEFAULT_ENDGAME_POSITIONS if positions is None else positions
    random.seed(seed)
    torch.manual_seed(seed)
    examples = build_value_dataset(positions, depth)
    if not examples:
        raise ValueError("no value examples produced")

    model = PolicyValueNet(hidden_channels=hidden_channels, residual_blocks=residual_blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    before = evaluate_values(model, examples)
    loss_curve: list[dict[str, float | int]] = []
    for step in range(steps):
        batch = random.sample(examples, k=min(batch_size, len(examples)))
        loss = train_value_batch(model, optimizer, batch)
        if step == 0 or (step + 1) % report_every == 0 or step + 1 == steps:
            loss_curve.append({"step": step + 1, "value_mse": evaluate_values(model, examples)["mse"]})
    after = evaluate_values(model, examples)
    games = [play_value_greedy(model, position, max_plies=max_plies) for position in positions]
    wins = sum(1 for game in games if game["won"])
    return {
        "loop": "endgame-value-validation",
        "positions": len(positions),
        "examples": len(examples),
        "depth": depth,
        "model": {"hidden_channels": hidden_channels, "residual_blocks": residual_blocks},
        "training": {"steps": steps, "learning_rate": learning_rate, "seed": seed, "batch_size": batch_size},
        "before": before,
        "after": after,
        "loss_curve": loss_curve,
        "validation": {"wins": wins, "positions": len(positions), "passed": wins == len(positions)},
        "games": games,
    }
