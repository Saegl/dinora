from abc import ABC, abstractmethod
import chess

IsTerminal = bool
Priors = dict[chess.Move, float]
StateValue = float


class BaseModel(ABC):
    @abstractmethod
    def evaluate(self, state: chess.Board) -> tuple[IsTerminal, Priors, StateValue]:
        pass

    @abstractmethod
    def reset(self):
        """Delete caches"""
