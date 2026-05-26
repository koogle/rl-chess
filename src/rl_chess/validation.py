from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chess
import chess.engine

from rl_chess.nn_model import NeuralPolicyValueEvaluator, PolicyValueNet
from rl_chess.puct_mcts import PUCTMCTS


def _option_bound(option: object, name: str) -> int | None:
    value = getattr(option, name, None)
    return None if value is None else int(value)


def stockfish_strength_config(elo: int, options: dict[str, object]) -> dict[str, int | bool]:
    """Return the weakest honest Stockfish config for a requested validation Elo.

    Stockfish's `UCI_Elo` floor is commonly higher than 500. For those engines,
    an Elo-500 baseline maps to the weakest available `Skill Level` instead of
    pretending the engine accepted an unsupported UCI_Elo value.
    """

    if elo <= 0:
        raise ValueError("elo must be positive")

    uci_elo = options.get("UCI_Elo")
    if "UCI_LimitStrength" in options and uci_elo is not None:
        min_elo = _option_bound(uci_elo, "min")
        max_elo = _option_bound(uci_elo, "max")
        if (min_elo is None or elo >= min_elo) and (max_elo is None or elo <= max_elo):
            return {"UCI_LimitStrength": True, "UCI_Elo": elo}

    skill = options.get("Skill Level")
    if skill is not None:
        min_skill = _option_bound(skill, "min") or 0
        max_skill = _option_bound(skill, "max") or 20
        if elo <= 500:
            level = min_skill
        else:
            level = round(min_skill + (max_skill - min_skill) * min((elo - 500) / 1000, 1.0))
        return {"Skill Level": level}

    raise ValueError("stockfish engine does not expose UCI_Elo or Skill Level strength controls")


class Player(Protocol):
    def select_move(self, board: chess.Board) -> chess.Move:
        ...


@dataclass(frozen=True)
class ValidationResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        return 0.0 if self.games == 0 else (self.wins + 0.5 * self.draws) / self.games

    @property
    def passed(self) -> bool:
        return self.score > 0.5

    def plus(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(
            wins=self.wins + other.wins,
            losses=self.losses + other.losses,
            draws=self.draws + other.draws,
        )


@dataclass
class FirstLegalPlayer:
    def select_move(self, board: chess.Board) -> chess.Move:
        return next(iter(board.legal_moves))


@dataclass
class FixedMovePlayer:
    moves: list[str]

    def select_move(self, board: chess.Board) -> chess.Move:
        if not self.moves:
            return next(iter(board.legal_moves))
        move = chess.Move.from_uci(self.moves.pop(0))
        if move not in board.legal_moves:
            raise ValueError(f"fixed move is illegal in this position: {move.uci()}")
        return move


@dataclass
class PUCTModelPlayer:
    model: PolicyValueNet
    simulations: int = 64
    seed: int | None = None

    def select_move(self, board: chess.Board) -> chess.Move:
        return PUCTMCTS(
            evaluator=NeuralPolicyValueEvaluator(self.model),
            iterations=self.simulations,
            seed=self.seed,
        ).select_move(board)


@dataclass
class StockfishPlayer:
    """UCI Stockfish player configured as a weak fixed-Elo validation baseline."""

    path: str = "stockfish"
    elo: int = 500
    movetime: float = 0.05

    def __post_init__(self) -> None:
        if self.elo <= 0:
            raise ValueError("elo must be positive")
        if self.movetime <= 0:
            raise ValueError("movetime must be positive")
        self.engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> StockfishPlayer:
        engine = chess.engine.SimpleEngine.popen_uci(self.path)
        engine.configure(stockfish_strength_config(self.elo, engine.options))
        self.engine = engine
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def select_move(self, board: chess.Board) -> chess.Move:
        if self.engine is None:
            raise RuntimeError("StockfishPlayer must be used as a context manager")
        result = self.engine.play(board, chess.engine.Limit(time=self.movetime))
        if result.move is None:
            raise ValueError("stockfish returned no move")
        return result.move


def play_validation_game(
    candidate: Player,
    baseline: Player,
    candidate_color: bool,
    starting_board: chess.Board | None = None,
    max_plies: int = 200,
) -> ValidationResult:
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")

    board = (starting_board or chess.Board()).copy(stack=False)
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        player = candidate if board.turn == candidate_color else baseline
        board.push(player.select_move(board.copy(stack=False)))

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    if result == "1/2-1/2":
        return ValidationResult(draws=1)
    candidate_won = (result == "1-0" and candidate_color == chess.WHITE) or (result == "0-1" and candidate_color == chess.BLACK)
    return ValidationResult(wins=1) if candidate_won else ValidationResult(losses=1)


def play_validation_match(
    candidate: Player,
    baseline: Player,
    games: int = 2,
    max_plies: int = 200,
) -> ValidationResult:
    if games <= 0:
        raise ValueError("games must be positive")
    total = ValidationResult()
    for game_idx in range(games):
        total = total.plus(
            play_validation_game(
                candidate=candidate,
                baseline=baseline,
                candidate_color=chess.WHITE if game_idx % 2 == 0 else chess.BLACK,
                max_plies=max_plies,
            )
        )
    return total


def validate_model_against_stockfish(
    model: PolicyValueNet,
    elo: int = 500,
    games: int = 2,
    max_plies: int = 200,
    simulations: int = 64,
    stockfish_path: str = "stockfish",
    stockfish_movetime: float = 0.05,
    seed: int | None = None,
) -> ValidationResult:
    candidate = PUCTModelPlayer(model=model, simulations=simulations, seed=seed)
    with StockfishPlayer(path=stockfish_path, elo=elo, movetime=stockfish_movetime) as stockfish:
        return play_validation_match(candidate=candidate, baseline=stockfish, games=games, max_plies=max_plies)
