import chess
from badgyal import BGNet

from dinora.models import Priors, StateValue, BaseModel


class BadgyalModel(BaseModel):
    def __init__(self, softmax_temp: float = 1.61) -> None:
        self.softmax_temp: float = softmax_temp
        self.badg = BGNet(False, False)

    def raw_eval(self, board: chess.Board) -> tuple[Priors, StateValue]:
        return self.badg.eval(board)  # type: ignore

    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        result = board.result(claim_draw=True)
        if result == "*":
            # Game is not ended
            # evaluate by using ANN
            priors, value_estimate = self.badg.eval(board, self.softmax_temp)
        elif result == "1/2-1/2":
            # It's already draw
            # or we can claim draw, anyway `value_estimate` is 0.0
            priors = {}
            value_estimate = 0.0
        else:
            # result == '1-0' or result == '0-1'
            # we are checkmated because it's our turn to move
            # so the `value_estimate` is -1.0
            priors = {}  # no moves after checkmate
            value_estimate = -1.0
        priors = {chess.Move.from_uci(m): val for m, val in priors.items()}
        return priors, value_estimate
