import chess

from dinora.models.base import BaseModel, Evaluation, Priors
from dinora.models.registry import ModelConfig


def uniform_policy(board: chess.Board) -> Priors:
    legal_moves = list(board.legal_moves)
    uniform_prob = 1.0 / len(legal_moves)
    return dict.fromkeys(legal_moves, uniform_prob)


class EmptyModel(BaseModel):
    def evaluate(self, board: chess.Board) -> Evaluation:
        return uniform_policy(board), 0.0

    def evaluate_batch(self, boards: list[chess.Board]) -> list[Evaluation]:
        return [self.evaluate(board) for board in boards]


def load(config: ModelConfig) -> BaseModel:
    return EmptyModel()
