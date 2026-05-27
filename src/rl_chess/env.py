from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess


@dataclass(frozen=True)
class Observation:
    """A lightweight observation of the current chess state."""

    board_ascii: str
    turn: bool
    legal_moves: tuple[str, ...]


class ChessEnv:
    """Tiny chess environment wrapper.

    Rewards are from White's perspective:
    - White win: +1
    - Black win: -1
    - Draw/non-terminal: 0
    """

    def __init__(self, starting_board: chess.Board | None = None) -> None:
        self.starting_board = starting_board.copy(stack=False) if starting_board is not None else chess.Board()
        self.board = self.starting_board.copy(stack=False)

    def reset(self) -> Observation:
        self.board = self.starting_board.copy(stack=False)
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            board_ascii=board_to_ascii(self.board),
            turn=self.board.turn,
            legal_moves=tuple(move.uci() for move in self.board.legal_moves),
        )

    def step(self, move: chess.Move | str) -> tuple[Observation, float, bool, dict[str, Any]]:
        move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move

        if move_obj not in self.board.legal_moves:
            raise ValueError(f"Illegal move {move_obj.uci()} for board:\n{board_to_ascii(self.board)}")

        self.board.push(move_obj)
        done = self.board.is_game_over(claim_draw=True)
        result = self.board.result(claim_draw=True) if done else None
        reward = result_to_white_reward(result)
        return self.observe(), reward, done, {"result": result}


def result_to_white_reward(result: str | None) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def board_to_ascii(board: chess.Board) -> str:
    """Render a board as an inspectable text diagram with chess symbols.

    White pieces use ♙♘♗♖♕♔, black pieces use ♟♞♝♜♛♚, and empty squares are
    dots. The diagram is shown from White's perspective, with rank 8 at the top
    and rank 1 at the bottom.
    """

    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row: list[str] = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            row.append(piece.unicode_symbol() if piece is not None else ".")
        rank_label = str(rank + 1)
        lines.append(f"{rank_label} " + " ".join(row) + f" {rank_label}")
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


UNICODE_TO_PIECE = {
    "♙": chess.Piece(chess.PAWN, chess.WHITE),
    "♘": chess.Piece(chess.KNIGHT, chess.WHITE),
    "♗": chess.Piece(chess.BISHOP, chess.WHITE),
    "♖": chess.Piece(chess.ROOK, chess.WHITE),
    "♕": chess.Piece(chess.QUEEN, chess.WHITE),
    "♔": chess.Piece(chess.KING, chess.WHITE),
    "♟": chess.Piece(chess.PAWN, chess.BLACK),
    "♞": chess.Piece(chess.KNIGHT, chess.BLACK),
    "♝": chess.Piece(chess.BISHOP, chess.BLACK),
    "♜": chess.Piece(chess.ROOK, chess.BLACK),
    "♛": chess.Piece(chess.QUEEN, chess.BLACK),
    "♚": chess.Piece(chess.KING, chess.BLACK),
}


def ascii_to_board(board_ascii: str, turn: bool) -> chess.Board:
    """Parse the inspectable Unicode board diagram back into python-chess.

    The text format intentionally mirrors board_to_ascii(), so fixtures and
    diagnostics can stay human-readable instead of passing FEN around public
    training/validation boundaries.
    """

    lines = [line.rstrip() for line in board_ascii.strip("\n").splitlines() if line.strip()]
    if len(lines) != 10 or lines[0].strip() != "a b c d e f g h" or lines[-1].strip() != "a b c d e f g h":
        raise ValueError("board_ascii must use board_to_ascii() format")

    board = chess.Board(None)
    board.turn = turn
    board.clear_stack()
    board.castling_rights = 0
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1

    for row_index, line in enumerate(lines[1:-1]):
        tokens = line.split()
        expected_rank = 8 - row_index
        if len(tokens) != 10 or tokens[0] != str(expected_rank) or tokens[-1] != str(expected_rank):
            raise ValueError("board_ascii rank rows must include matching rank labels")
        for file, token in enumerate(tokens[1:9]):
            if token == ".":
                continue
            piece = UNICODE_TO_PIECE.get(token)
            if piece is None:
                raise ValueError(f"unsupported board token: {token}")
            board.set_piece_at(chess.square(file, expected_rank - 1), piece)

    return board
