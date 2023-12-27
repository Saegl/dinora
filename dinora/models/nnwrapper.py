import numpy as np
import chess
from abc import ABC, abstractmethod

from dinora.models import BaseModel, IsTerminal, Priors, StateValue
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

    def evaluate(self, board: chess.Board) -> tuple[IsTerminal, Priors, StateValue]:
        outcome = board.outcome()

        if outcome is not None and outcome.winner:
            # There is a winner, but is is our turn to move
            # means we lost
            return True, {}, -1.0

        elif (
            outcome
            is not None  # We know there is no winner by `if` above, so it's draw
            or board.is_repetition(3)  # Can claim draw means it's draw
            or board.can_claim_fifty_moves()
        ):
            # There is some subtle difference between
            # board.can_claim_threefold_repetition() and board.is_repetition(3)
            # I believe it is right to use second one, but I am not 100% sure
            # Gives +-1 ply error on this one for example https://lichess.org/EJ67iHS1/black#94

            priors, _ = self.nn_evaluate(board)
            return True, priors, 0.0
        else:
            assert outcome is None
            # Use ANN evaluation if there is no outcome yet
            priors, val = self.nn_evaluate(board)
            return False, priors, val

    def reset(self):
        pass
