import chess

from dinora.models import BaseModel
from dinora.search.base import BaseSearcher, ConfigType, DefaultValue
from dinora.search.stoppers import Stopper


class OneMove(BaseSearcher):
    def __init__(self) -> None:
        pass

    def config_schema(self) -> dict[str, tuple[ConfigType, DefaultValue]]:
        return {}

    def set_config_param(self, k: str, v: str) -> None:
        return None

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        priors, _ = evaluator.evaluate(board)
        maxval = 0.0
        bestmove = None
        for move, prob in priors.items():
            if prob > maxval:
                maxval = prob
                bestmove = move

        if not bestmove:
            raise Exception("Cannot find any move")

        return bestmove
