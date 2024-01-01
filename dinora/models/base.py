from abc import ABC, abstractmethod

import chess

Priors = dict[chess.Move, float]
StateValue = float
Evaluation = tuple[Priors, StateValue]


class BaseModel(ABC):
    @abstractmethod
    def evaluate(self, state: chess.Board) -> Evaluation:
        pass

    @abstractmethod
    def reset(self) -> None:
        """Delete caches"""
