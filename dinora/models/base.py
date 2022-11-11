from abc import ABC, abstractmethod
import chess

Priors = dict[chess.Move, float]
StateValue = float


class BaseModel(ABC):
    @abstractmethod
    def evaluate(self, state: chess.Board) -> tuple[Priors, StateValue]:
        pass