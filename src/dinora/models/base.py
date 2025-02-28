from abc import ABC, abstractmethod

import chess

Priors = dict[chess.Move, float]
StateValue = float
Evaluation = tuple[Priors, StateValue]


class BaseModel(ABC):
    def name(self) -> str:
        """Return model name for logs/debug_info"""
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self, board: chess.Board) -> Evaluation:
        """
        Evaluate board by returning policy and value

        sum(policy.values()) == 1.0
        -1.0 <= value <= 1.0
        value = 1.0 => Current side (board.turn) is winning
        value = -1.0 => Current side (board.turn) is losing
        value = 0.0 => Draw
        """

    def evaluate_batch(self, boards: list[chess.Board]) -> list[Evaluation]:
        """
        Same as `evaluate` but for batch of boards

        Example of slow fallback
        ```
        return [self.evaluate(board) for board in boards]
        ```
        use faster methods of your neural networks library
        """
        raise Exception(f"Batcn evaluation is not implemented on {self.name()}")

    def reset(self) -> None:  # noqa: B027
        """
        Delete caches
        Useful to call between games in benchmarks/evaluators
        """
