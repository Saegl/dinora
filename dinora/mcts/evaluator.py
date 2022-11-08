from abc import ABC, abstractmethod
import chess

Priors = dict[chess.Move, float]
StateValue = float


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, state: chess.Board) -> tuple[Priors, StateValue]:
        pass
