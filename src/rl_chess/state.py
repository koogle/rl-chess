from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from rl_chess.env import board_to_ascii


@dataclass(frozen=True)
class BoardState:
    """Serializable state extracted from a python-chess board."""

    board_ascii: str
    turn: bool
    legal_moves: tuple[str, ...]
    fullmove_number: int
    halfmove_clock: int
    is_check: bool


def legal_move_uci(board: chess.Board) -> tuple[str, ...]:
    """Return legal moves in UCI format using python-chess as source of truth."""

    return tuple(move.uci() for move in board.legal_moves)


def state_from_board(board: chess.Board) -> BoardState:
    """Create a small immutable snapshot suitable for replay/training logs."""

    return BoardState(
        board_ascii=board_to_ascii(board),
        turn=board.turn,
        legal_moves=legal_move_uci(board),
        fullmove_number=board.fullmove_number,
        halfmove_clock=board.halfmove_clock,
        is_check=board.is_check(),
    )


PIECE_TO_PLANE: dict[tuple[bool, int], int] = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def encode_board_planes(board: chess.Board) -> np.ndarray:
    """Encode pieces into 12 planes shaped (piece_plane, rank, file).

    Rank index 0 is chess rank 8 and rank index 7 is chess rank 1, matching
    normal diagram orientation. This is intentionally simple and inspectable;
    policy/value features like side-to-move or castling rights can be added as
    extra planes later.
    """

    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        rank_index = 7 - chess.square_rank(square)
        file_index = chess.square_file(square)
        planes[plane, rank_index, file_index] = 1.0
    return planes
