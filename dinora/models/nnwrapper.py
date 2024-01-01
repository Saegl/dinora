import numpy as np
import chess
from abc import ABC, abstractmethod

from dinora.models import BaseModel, Priors, StateValue
from dinora.encoders.policy import extract_prob_from_policy


def softmax(x, tau=1.0):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


class NNWrapper(BaseModel, ABC):
    """
    This class converts raw NN outputs in numpy into
    `BaseModel` compatible interface
    """

    @abstractmethod
    def raw_outputs(self, board: chess.Board):
        """
        NN outputs as numpy arrays
        priors numpy of shape (1880,)
        state value of shape (1,)
        """

    def nn_evaluate(self, board: chess.Board):
        get_prob = extract_prob_from_policy

        raw_policy, raw_value = self.raw_outputs(board)

        moves = list(board.legal_moves)
        move_logits = [get_prob(raw_policy, move, not board.turn) for move in moves]

        move_priors = softmax(np.array(move_logits))
        priors = dict(zip(moves, move_priors))

        return priors, float(raw_value[0])

    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        return self.nn_evaluate(board)

    def reset(self):
        pass
