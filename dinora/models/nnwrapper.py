from abc import ABC, abstractmethod

import chess
import numpy as np
import numpy.typing as npt

from dinora.encoders.policy import extract_prob_from_policy
from dinora.models.base import BaseModel, Evaluation

npf32 = npt.NDArray[np.float32]


def softmax(x: npf32, tau: float = 1.0) -> npf32:
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()  # type: ignore


class NNWrapper(BaseModel, ABC):
    """
    This class converts raw NN outputs in numpy into
    `BaseModel` compatible interface
    """

    @abstractmethod
    def raw_outputs(self, board: chess.Board) -> tuple[npf32, npf32]:
        """
        NN outputs as numpy arrays
        priors numpy of shape (1880,)
        state value of shape (1,)
        """

    @abstractmethod
    def raw_outputs_batch(self, boards: list[chess.Board]) -> tuple[npf32, npf32]:
        pass

    def nn_evaluate(self, board: chess.Board) -> Evaluation:
        get_prob = extract_prob_from_policy

        raw_policy, raw_value = self.raw_outputs(board)

        moves = list(board.legal_moves)
        move_logits = [get_prob(raw_policy, move, not board.turn) for move in moves]

        move_priors = softmax(np.array(move_logits))
        priors = dict(zip(moves, move_priors))

        return priors, float(raw_value[0])

    def evaluate(self, board: chess.Board) -> Evaluation:
        return self.nn_evaluate(board)

    def evaluate_batch(self, boards: list[chess.Board]) -> list[Evaluation]:
        get_prob = extract_prob_from_policy

        raw_policies, raw_values = self.raw_outputs_batch(boards)

        ans = []

        for i in range(len(raw_policies)):
            board = boards[i]
            moves = list(board.legal_moves)
            move_logits = [get_prob(raw_policies[i], move, not board.turn) for move in moves]

            move_priors = softmax(np.array(move_logits))
            priors = dict(zip(moves, move_priors))

            ans.append((priors, float(raw_values[i][0])))

        return ans

    def evaluate_batch_naive(self, boards: list[chess.Board]) -> list[Evaluation]:
        return [self.evaluate(board) for board in boards]


    def reset(self) -> None:
        pass
