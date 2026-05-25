from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import random
from typing import Deque, Iterator, Sequence


@dataclass(frozen=True)
class Transition:
    state_ascii: str
    action_uci: str
    player: bool
    reward: float
    done: bool
    next_state_ascii: str | None = None
    result: str | None = None
    return_: float | None = None

    def with_return(self, value: float) -> "Transition":
        return replace(self, return_=value)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def extend(self, transitions: Sequence[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int, seed: int | None = None) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rng = random.Random(seed)
        return rng.sample(list(self._items), k=min(batch_size, len(self._items)))

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)
