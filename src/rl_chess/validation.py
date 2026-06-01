from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Protocol

import chess
import chess.engine

from rl_chess.nn_model import PolicyValueNet
from rl_chess.puct_mcts import PUCTMCTS


def _option_bound(option: object, name: str) -> int | None:
    value = getattr(option, name, None)
    return None if value is None else int(value)


STOCKFISH_ELO_FLOOR = 1320


def stockfish_strength_config(elo: int, options: dict[str, object]) -> dict[str, int | bool]:
    """Return an honest Stockfish strength config for a requested validation Elo."""

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
        if elo <= STOCKFISH_ELO_FLOOR:
            level = min_skill
        else:
            level = round(min_skill + (max_skill - min_skill) * min((elo - STOCKFISH_ELO_FLOOR) / 1000, 1.0))
        return {"Skill Level": level}

    raise ValueError("stockfish engine does not expose UCI_Elo or Skill Level strength controls")


class Player(Protocol):
    def select_move(self, board: chess.Board) -> chess.Move:
        ...


def resolve_stockfish_path(path: str = "stockfish") -> str:
    """Find Stockfish across local Linux and Debian package layouts."""

    if Path(path).is_file():
        return path
    resolved = shutil.which(path)
    if resolved is not None:
        return resolved
    for candidate in ("/usr/games/stockfish", "/usr/local/bin/stockfish"):
        if Path(candidate).is_file():
            return candidate
    return path


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


@dataclass(frozen=True)
class CheckpointRandomValidation:
    iteration: int
    checkpoint_path: Path
    wins: int
    losses: int
    draws: int
    score: float
    passed: bool

    @classmethod
    def from_result(
        cls,
        *,
        iteration: int,
        checkpoint_path: str | Path,
        result: ValidationResult,
    ) -> CheckpointRandomValidation:
        if iteration <= 0:
            raise ValueError("iteration must be positive")
        return cls(
            iteration=iteration,
            checkpoint_path=Path(checkpoint_path),
            wins=result.wins,
            losses=result.losses,
            draws=result.draws,
            score=result.score,
            passed=result.passed,
        )

    def to_jsonable(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "checkpoint_path": str(self.checkpoint_path),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "score": self.score,
            "passed": self.passed,
        }


def append_checkpoint_random_validation(path: str | Path, validation: CheckpointRandomValidation) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(validation.to_jsonable(), sort_keys=True) + "\n")


def best_checkpoint_random_validation(
    current: CheckpointRandomValidation | None,
    candidate: CheckpointRandomValidation,
) -> CheckpointRandomValidation:
    if current is None or candidate.score > current.score:
        return candidate
    return current


@dataclass
class FirstLegalPlayer:

    def select_move(self, board: chess.Board) -> chess.Move:
        return next(iter(board.legal_moves))


@dataclass
class RandomPlayer:
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


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
            evaluator=self.model,
            iterations=self.simulations,
            seed=self.seed,
        ).select_move(board)


@dataclass
class StockfishPlayer:
    """UCI Stockfish player configured as a weak fixed-Elo validation baseline."""

    path: str = "stockfish"
    elo: int = STOCKFISH_ELO_FLOOR
    movetime: float = 0.05

    def __post_init__(self) -> None:
        if self.elo <= 0:
            raise ValueError("elo must be positive")
        if self.movetime <= 0:
            raise ValueError("movetime must be positive")
        self.engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> StockfishPlayer:
        engine = chess.engine.SimpleEngine.popen_uci(resolve_stockfish_path(self.path))
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
    starting_board: chess.Board | None = None,
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
                starting_board=starting_board,
                max_plies=max_plies,
            )
        )
    return total


def validate_model_against_stockfish(
    model: PolicyValueNet,
    elo: int = STOCKFISH_ELO_FLOOR,
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


def validate_model_against_random(
    model: PolicyValueNet,
    games: int = 12,
    max_plies: int = 200,
    simulations: int = 64,
    seed: int | None = None,
) -> ValidationResult:
    return play_validation_match(
        candidate=PUCTModelPlayer(model=model, simulations=simulations, seed=seed),
        baseline=RandomPlayer(seed=seed),
        games=games,
        max_plies=max_plies,
    )
